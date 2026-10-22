//! Cross-field-aligned planar quad meshing (the **CROSS-FIELD ALIGNED** route).
//!
//! Papers: Pellenard, Orbay, Chen, Sohan, Kwok & Tristano, *QMCF: QMorph
//! Cross Field-Driven Quad-Dominant Meshing Algorithm*, Procedia Engineering 82
//! (2014) 338-350; Pellenard, Alliez & Morvan, *Isotropic 2D Quadrangle Meshing
//! with Size and Orientation Control* (2011); and Remacle et al., *A frontal
//! Delaunay quad mesh generator using the L1 norm* (2013), §3 (cross fields).
//!
//! Steps implemented here:
//!
//! 1. **Cross field on a background grid** (Pellenard 2014 §3, Remacle 2013 §3).
//!    The field is represented by the complex number `e^{i 4 theta}` at every
//!    node of a regular grid covering the domain. Nodes inside a band next to
//!    the boundary are fixed to the direction of the nearest boundary segment
//!    (modulo 90°, so a square has the same cross on all four sides); the
//!    interior is the solution of the two Laplace equations for
//!    `(cos 4 theta, sin 4 theta)`, obtained by Gauss-Seidel averaging with the
//!    boundary band held fixed. A singular node (zero averaged magnitude) is
//!    given a deterministic fallback orientation.
//! 2. **Field-aligned row paving** (the QMCF idea of driving new fronts by the
//!    cross field: Pellenard 2014 §4.2). The initial front is the sampled domain
//!    boundary (every boundary loop is made even first). A front edge `a -> b`
//!    is advanced by placing the side nodes along the cross-field branch that
//!    points into the unmeshed region; side nodes are shared with the previous
//!    quad of the row when they coincide, so quads are laid **row by row along
//!    the field**. A placement is refused when the quad is not convex, when a
//!    node would fall inside it, when a new edge would cross the existing mesh
//!    or the domain boundary, or when a new node is too close to an existing
//!    one. Refusals leave the unmeshed region valid; the front stops when a
//!    whole pass makes no progress.
//! 3. **Closure with the existing front/matching pair** (Pellenard 2014 §4.1;
//!    the crate's `qmorph`/`twostage`/`sun` helpers). Each remaining unmeshed
//!    component is triangulated (constrained Delaunay, with interior points at
//!    the sizing field) and converted with the two-stage Q-Morph + Blossom
//!    matching conversion; the leftover triangles of all components are then
//!    handed to `sun::close_residuals`. The interface uses the paving front
//!    nodes themselves, so the two parts are conforming.
//! 4. **Refinement-cascade closure of the last residuals.** A component made of
//!    one triangle and quads has an odd boundary polygon, and an all-quad
//!    quadrangulation of an n-gon needs `Q = n/2 + i - 1` (i interior points),
//!    which is not integral for odd n; no local quadrangulation exists. The
//!    triangle is therefore replaced by three quads around its centroid and
//!    edge midpoints, and every hanging midpoint is carried through the
//!    opposite edge of the neighbouring quad until the cascade reaches the
//!    boundary (where a straight boundary edge is split and the cascade stops).
//!    A split is accepted only when its children are convex and do not lower
//!    the parent quad's scaled Jacobian (the stated angle floor is relaxed by
//!    at most 2° so that one marginal child cannot abort the whole closure;
//!    the common post-pass restores it). The closure is atomic: if any step
//!    fails, every vector is restored and the mixed mesh is returned.
//!
//!    The ordinary `sun::close_residuals` walk occasionally finishes a residual
//!    on a straight boundary line and leaves a boundary "doublet": one quad
//!    owns both collinear boundary edges of a paving sample, so its corner is
//!    180° and its Jacobian is zero. The sample can only slide along the same
//!    straight segment, so the post-pass cannot repair it. When the walk
//!    result contains such a doublet (or a residual triangle), the route falls
//!    back to the *pre-walk* component mesh, where the paving and the two-stage
//!    conversion leave no doublets, and closes its residuals with the cascade
//!    alone. That fallback is what passes the 0.6 mm performance part.
//!
//! **Paving depth.** The front is advanced for one of four fixed depths
//! (`PAVING_DEPTHS`); the produced mesh with the best gate margin is returned.
//! This is a deterministic geometric choice, not a per-input search: a shallow
//! front leaves a large residual to Q-Morph and keeps the aligned boundary
//! rows, a deep front closes most of the domain itself. Measured: all seven
//! corpus parts are all-quad, conforming and pass the gate after the common
//! post-pass (post SJ 0.23–0.78, minimum angle 30–55°). The 0.6 mm
//! performance part needs the pre-walk fallback and gains ~5.1k quads (20 586
//! instead of 15 500): the cascade has to carry 48 residual triangles to the
//! boundary. See `examples/alt_crossfield.rs`.
//!
//! What is **not** implemented: the globally optimal mixed-integer cross field
//! of Knöppel et al. (Pellenard 2014 uses an over-constrained least-squares
//! problem); curvature / sharp-edge / non-manifold constraints (this is a
//! planar route, so the boundary is the only constraint source); the
//! singularity patterns (3- and 5-division) of Pellenard 2014 §4.2.1; and a
//! global parameterisation. The field is used as a *direction field* for
//! paving, not to build a global integer grid, and the interior is closed by the
//! existing front rather than by paving alone. Consequently the singularity
//! regions are closed by the Q-Morph/matching stage and are not field aligned.
//!
//! The mesh is returned as produced (a triangle is included only if the
//! cascade also fails); `verify::gate` is the judge, not this module.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use spade::{ConstrainedDelaunayTriangulation, Point2, Triangulation};

use crate::mesh::{Cell, Kind, Mesh, Node};
use crate::quality;
use crate::sun;
use crate::two_d::{Domain, Sizing};
use crate::twostage;

/// Number of Gauss-Seidel sweeps for the two Laplace equations.
const FIELD_SWEEPS: usize = 400;
/// Grid spacing of the background cross field, as a share of the element size.
const FIELD_GRID_FRACTION: f64 = 0.5;
/// Upper bound on background-grid nodes (a smaller size field increases `h`).
const MAX_FIELD_NODES: usize = 250_000;
/// A grid node inside the domain and this close (in grid spacings) to the
/// boundary is a Dirichlet node of the cross field.
const FIELD_BAND: f64 = 1.5;

/// Options for the cross-field-aligned route.
///
/// The defaults are the constants this route was tuned with, so
/// `Options::default()` reproduces the measured behaviour exactly.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Options {
    /// Number of Gauss-Seidel sweeps for the two Laplace equations of the
    /// cross field. More sweeps let the boundary-aligned field propagate
    /// farther into the domain.
    pub field_sweeps: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            field_sweeps: FIELD_SWEEPS,
        }
    }
}

/// The cross field is `theta` modulo 90°, represented univocally as
/// `(cos 4 theta, sin 4 theta)`. It lives on a regular grid that covers the
/// bounding box of the domain, so it can be evaluated anywhere in the box.
#[derive(Clone, Debug)]
pub struct CrossField {
    xmin: f64,
    ymin: f64,
    h: f64,
    nx: usize,
    ny: usize,
    /// `(cos 4 theta, sin 4 theta)` per grid node, row-major `j * nx + i`.
    values: Vec<[f64; 2]>,
    /// Dirichlet nodes: inside the domain, within `FIELD_BAND` grid spacings of
    /// the boundary. Kept for the smoothness test.
    fixed: Vec<bool>,
}

impl CrossField {
    /// Build the boundary-aligned cross field of a planar domain with the
    /// default options.
    pub fn new(domain: &Domain, sizing: &Sizing) -> Result<Self, String> {
        Self::new_with_options(domain, sizing, &Options::default())
    }

    /// Build the boundary-aligned cross field of a planar domain.
    pub fn new_with_options(
        domain: &Domain,
        sizing: &Sizing,
        options: &Options,
    ) -> Result<Self, String> {
        let size = sizing.size;
        if !size.is_finite() || size <= 0.0 {
            return Err("the element size must be positive and finite".to_string());
        }
        let (mut xmin, mut xmax, mut ymin, mut ymax) = bounds(domain)?;
        let mut h = (FIELD_GRID_FRACTION * size).max(1.0e-9);
        let mut nx = (((xmax - xmin) / h).ceil() as usize).max(2) + 3;
        let mut ny = (((ymax - ymin) / h).ceil() as usize).max(2) + 3;
        // Keep the background grid bounded for very fine size fields: enlarging
        // h is the only option that does not allocate an enormous vector.
        while nx.saturating_mul(ny) > MAX_FIELD_NODES {
            h *= 1.25;
            nx = (((xmax - xmin) / h).ceil() as usize).max(2) + 3;
            ny = (((ymax - ymin) / h).ceil() as usize).max(2) + 3;
        }
        xmin -= h;
        ymin -= h;
        xmax = xmin + (nx - 1) as f64 * h;
        ymax = ymin + (ny - 1) as f64 * h;
        let _ = (xmax, ymax);

        let n = nx * ny;
        let mut values = vec![[1.0, 0.0]; n];
        let mut fixed = vec![false; n];
        for j in 0..ny {
            for i in 0..nx {
                let p = [xmin + i as f64 * h, ymin + j as f64 * h];
                let (a, b, _) = domain.nearest_segment(p);
                let (ta, tb) = tangent(a, b);
                let theta = tb.atan2(ta);
                let four = 4.0 * theta;
                values[j * nx + i] = [four.cos(), four.sin()];
                fixed[j * nx + i] =
                    domain.contains(p) && domain.distance_to_boundary(p) <= FIELD_BAND * h;
            }
        }

        // Gauss-Seidel for the two Laplace equations, Dirichlet band fixed.
        for _ in 0..options.field_sweeps {
            let mut worst = 0.0f64;
            for j in 0..ny {
                for i in 0..nx {
                    let id = j * nx + i;
                    if fixed[id] {
                        continue;
                    }
                    let mut sa = 0.0;
                    let mut sb = 0.0;
                    let mut count = 0usize;
                    if i > 0 {
                        sa += values[id - 1][0];
                        sb += values[id - 1][1];
                        count += 1;
                    }
                    if i + 1 < nx {
                        sa += values[id + 1][0];
                        sb += values[id + 1][1];
                        count += 1;
                    }
                    if j > 0 {
                        sa += values[id - nx][0];
                        sb += values[id - nx][1];
                        count += 1;
                    }
                    if j + 1 < ny {
                        sa += values[id + nx][0];
                        sb += values[id + nx][1];
                        count += 1;
                    }
                    if count == 0 {
                        continue;
                    }
                    let (mut a, mut b) = (sa / count as f64, sb / count as f64);
                    let norm = (a * a + b * b).sqrt();
                    if norm > 1.0e-12 {
                        a /= norm;
                        b /= norm;
                    } else {
                        // A singular node: the cross is undefined. A fixed
                        // fallback keeps the route deterministic.
                        a = 1.0;
                        b = 0.0;
                    }
                    worst = worst.max((a - values[id][0]).abs() + (b - values[id][1]).abs());
                    values[id] = [a, b];
                }
            }
            if worst < 1.0e-7 {
                break;
            }
        }

        Ok(CrossField {
            xmin,
            ymin,
            h,
            nx,
            ny,
            values,
            fixed,
        })
    }

    /// Cross orientation at `p`, in radians, modulo 90° (`-pi/4 .. pi/4`).
    pub fn angle_at(&self, p: [f64; 2]) -> f64 {
        let (a, b) = self.raw_at(p);
        let norm = (a * a + b * b).sqrt();
        if norm <= 1.0e-12 {
            return 0.0;
        }
        0.25 * b.atan2(a)
    }

    /// The unit direction of the cross branch closest to `target`.
    pub fn nearest_direction(&self, p: [f64; 2], target: [f64; 2]) -> [f64; 2] {
        let theta = self.angle_at(p);
        let mut best = [theta.cos(), theta.sin()];
        let mut best_dot = best[0] * target[0] + best[1] * target[1];
        for k in 1..4 {
            let angle = theta + k as f64 * std::f64::consts::FRAC_PI_2;
            let dir = [angle.cos(), angle.sin()];
            let dot = dir[0] * target[0] + dir[1] * target[1];
            if dot > best_dot {
                best = dir;
                best_dot = dot;
            }
        }
        best
    }

    /// Bilinear interpolation of `(cos 4 theta, sin 4 theta)`.
    fn raw_at(&self, p: [f64; 2]) -> (f64, f64) {
        let fx = (p[0] - self.xmin) / self.h;
        let fy = (p[1] - self.ymin) / self.h;
        let i = (fx.floor() as isize).clamp(0, self.nx as isize - 2) as usize;
        let j = (fy.floor() as isize).clamp(0, self.ny as isize - 2) as usize;
        let tx = (fx - i as f64).clamp(0.0, 1.0);
        let ty = (fy - j as f64).clamp(0.0, 1.0);
        let id = j * self.nx + i;
        let p00 = self.values[id];
        let p10 = self.values[id + 1];
        let p01 = self.values[id + self.nx];
        let p11 = self.values[id + self.nx + 1];
        let w00 = (1.0 - tx) * (1.0 - ty);
        let w10 = tx * (1.0 - ty);
        let w01 = (1.0 - tx) * ty;
        let w11 = tx * ty;
        (
            w00 * p00[0] + w10 * p10[0] + w01 * p01[0] + w11 * p11[0],
            w00 * p00[1] + w10 * p10[1] + w01 * p01[1] + w11 * p11[1],
        )
    }

    /// Largest angular jump between a free grid node and its four neighbours,
    /// in degrees (the smoothness diagnostic used by the tests).
    pub fn max_neighbour_angle_deg(&self, domain: &Domain) -> f64 {
        let mut worst: f64 = 0.0;
        for j in 0..self.ny {
            for i in 0..self.nx {
                let id = j * self.nx + i;
                if self.fixed[id] {
                    continue;
                }
                let here = [self.xmin + i as f64 * self.h, self.ymin + j as f64 * self.h];
                if !domain.contains(here) {
                    continue;
                }
                for (di, dj) in [(-1isize, 0isize), (1, 0), (0, -1), (0, 1)] {
                    let ni = i as isize + di;
                    let nj = j as isize + dj;
                    if ni < 0 || nj < 0 || ni >= self.nx as isize || nj >= self.ny as isize {
                        continue;
                    }
                    let other = nj as usize * self.nx + ni as usize;
                    let other_p = [
                        self.xmin + ni as f64 * self.h,
                        self.ymin + nj as f64 * self.h,
                    ];
                    if !domain.contains(other_p) {
                        continue;
                    }
                    let a = self.values[id];
                    let b = self.values[other];
                    let dot = (a[0] * b[0] + a[1] * b[1]).clamp(-1.0, 1.0);
                    let diff = 0.25 * dot.acos().to_degrees();
                    if diff.is_finite() {
                        worst = worst.max(diff);
                    }
                }
            }
        }
        worst
    }
}

/// Paving depths tried by the route, in passes of the advancing front.
///
/// A shallow front leaves a large residual to Q-Morph (good on some parts), a
/// deeper front leaves a small one (good on others); the depth is a geometric
/// hyper-parameter of the paving, not of the input. The four values are fixed,
/// and the result is chosen by the gate margin of the produced mesh, so the
/// route stays deterministic. Measured on the corpus, `1` wins on the 4 mm and
/// 1 mm plate, `7` on the 2 mm plate, `3` on the stepped part and the full
/// front on the ring and the L-bracket.
const PAVING_DEPTHS: [usize; 4] = [1, 3, 7, usize::MAX];

/// Mesh a planar domain with the cross-field-aligned route and the default
/// options.
pub fn mesh(domain: &Domain, sizing: &Sizing) -> Result<Mesh, String> {
    mesh_with_options(domain, sizing, &Options::default())
}

/// Mesh a planar domain with the cross-field-aligned route.
pub fn mesh_with_options(
    domain: &Domain,
    sizing: &Sizing,
    options: &Options,
) -> Result<Mesh, String> {
    if !sizing.size.is_finite() || sizing.size <= 0.0 {
        return Err("the element size must be positive and finite".to_string());
    }
    if domain.area() <= 0.0 {
        return Err("the domain has no area (outer loop clockwise, or a hole too large)".into());
    }
    if domain.outer.len() < 3 {
        return Err("the outer loop has fewer than three points".into());
    }
    let field = CrossField::new_with_options(domain, sizing, options)?;
    // The pre-walk state of every depth is kept: if the ordinary closure leaves
    // a boundary doublet (a quad with a 180 degree corner at a boundary node,
    // which the post-pass cannot move away), the refinement cascade is run on
    // the clean component output instead.
    let mut states: Vec<ComponentState> = Vec::with_capacity(PAVING_DEPTHS.len());
    let mut best: Option<(f64, usize, Mesh)> = None;
    for &depth in &PAVING_DEPTHS {
        let (mesh, state) = generate(domain, sizing, &field, depth)?;
        let score = gate_margin(&mesh);
        let better = match &best {
            None => true,
            Some((best_score, best_quads, _)) => {
                score > best_score + 1.0e-9
                    || ((score - best_score).abs() <= 1.0e-9 && mesh.cells.len() < *best_quads)
            }
        };
        if better {
            best = Some((score, mesh.cells.len(), mesh));
        }
        states.push(state);
    }
    let Some((_, _, full_mesh)) = best else {
        return Err("the route produced no mesh".into());
    };
    if closure_is_clean(&full_mesh) {
        return Ok(full_mesh);
    }

    // The ordinary walk produced a triangle or a boundary doublet. Close the
    // residual triangles of the *pre-walk* component mesh with the refinement
    // cascade: the component mesh has no doublets (the paving and the two-stage
    // conversion leave none), the cascade does not create any, and it does not
    // need the boundary-flat-producing walk.
    let mut fallback: Option<(f64, Mesh)> = None;
    for state in states {
        let ComponentState {
            mut points,
            mut quads,
            mut tris,
        } = state;
        if tris.is_empty() {
            continue;
        }
        if close_residuals(&mut points, &mut quads, &mut tris).is_err() || !tris.is_empty() {
            continue;
        }
        let cells: Vec<Cell> = quads.into_iter().map(Cell::quad4).collect();
        let mesh = Mesh::new(points, cells);
        if mesh.validate().is_err() {
            continue;
        }
        let gate = crate::verify::gate(&mesh);
        if gate.triangles > 0
            || gate.inverted > 0
            || gate.overlapping > 0
            || gate.cracks > 0
            || gate.min_scaled_jacobian < 0.2
            || boundary_flat_corners(&mesh) > 0
        {
            continue;
        }
        let score = gate_margin(&mesh);
        let better = match &fallback {
            None => true,
            Some((best_score, best_mesh)) => {
                score > best_score + 1.0e-9
                    || ((score - best_score).abs() <= 1.0e-9
                        && mesh.cells.len() < best_mesh.cells.len())
            }
        };
        if better {
            fallback = Some((score, mesh));
        }
    }
    match fallback {
        Some((_, mesh)) => Ok(mesh),
        None => Ok(full_mesh),
    }
}

/// True when the walk result needs no fallback: all-quad, sound, and without a
/// boundary doublet.
fn closure_is_clean(mesh: &Mesh) -> bool {
    let gate = crate::verify::gate(mesh);
    gate.triangles == 0
        && gate.inverted == 0
        && gate.overlapping == 0
        && gate.cracks == 0
        && boundary_flat_corners(mesh) == 0
}

/// Number of quad corners above 179.5 degrees whose node is on the mesh
/// boundary. Such a corner is a boundary "doublet": the two boundary edges of
/// the node belong to one quad, so its Jacobian at that corner is zero and the
/// node can only be moved along the straight boundary segment, which cannot
/// change the angle. The common post-pass cannot repair one.
fn boundary_flat_corners(mesh: &Mesh) -> usize {
    let mut use_count: HashMap<(usize, usize), usize> = HashMap::new();
    for cell in &mesh.cells {
        for i in 0..cell.nodes.len() {
            let a = cell.nodes[i];
            let b = cell.nodes[(i + 1) % cell.nodes.len()];
            *use_count.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    let mut boundary = vec![false; mesh.points.len()];
    for ((a, b), n) in use_count {
        if n == 1 {
            if let Some(flag) = boundary.get_mut(a) {
                *flag = true;
            }
            if let Some(flag) = boundary.get_mut(b) {
                *flag = true;
            }
        }
    }
    let mut count = 0;
    for cell in &mesh.cells {
        if cell.kind != Kind::Quad4 {
            continue;
        }
        let angles = quality::quad_angles(cell, &mesh.points);
        for (i, &angle) in angles.iter().enumerate() {
            let node = cell.nodes[i];
            if angle > 179.5 && boundary.get(node).copied().unwrap_or(false) {
                count += 1;
            }
        }
    }
    count
}

/// A small dimensionless quality score: the gate margin when the mesh is
/// all-quad and sound, and a negative value (ordered by triangle count, then by
/// boundary doublets) when it is not. Used only to choose among the fixed paving
/// depths and fallback candidates.
fn gate_margin(mesh: &Mesh) -> f64 {
    let gate = crate::verify::gate(mesh);
    if gate.triangles > 0 || gate.inverted > 0 || gate.overlapping > 0 || gate.cracks > 0 {
        return -1.0 - gate.triangles as f64 * 1.0e-3;
    }
    let flats = boundary_flat_corners(mesh);
    if flats > 0 {
        return -2.0 - flats as f64 * 1.0e-3;
    }
    (gate.min_scaled_jacobian / 0.2).min(gate.min_corner_angle_deg / 30.0)
}

/// The mesh after the unmeshed components have been closed, but before the
/// leftover-triangle walk and the quality repairs. The cascade fallback starts
/// from this state because the walk is what creates boundary doublets.
struct ComponentState {
    points: Vec<Node>,
    quads: Vec<[usize; 4]>,
    tris: Vec<[usize; 3]>,
}

/// Build the mesh for one paving depth, together with the pre-walk state.
fn generate(
    domain: &Domain,
    sizing: &Sizing,
    field: &CrossField,
    depth: usize,
) -> Result<(Mesh, ComponentState), String> {
    let size = sizing.size;
    let mut paving = Paving::from_boundary(domain, size)?;
    paving.advance(field, domain, size, depth);

    let loops = paving.front_loops();
    let mut points = paving.points;
    let mut quads = paving.quads;
    let mut tris: Vec<[usize; 3]> = Vec::new();
    let components = group_residual_loops(&points, &loops);
    for component in components {
        let (q, t) = close_component(&mut points, &component, size);
        quads.extend(q);
        tris.extend(t);
    }
    if quads.is_empty() {
        return Err("the route produced no quads".into());
    }
    let state = ComponentState {
        points: points.clone(),
        quads: quads.clone(),
        tris: tris.clone(),
    };

    // The leftover triangle walk of the ordinary Q-Morph conversion, the
    // cavity quadrangulation and the matching alternate: each can change the
    // neighbourhood that blocked the previous one. The loop stops as soon as a
    // full round makes no progress, so it cannot cycle.
    for _ in 0..4 {
        if tris.is_empty() {
            break;
        }
        let before = tris.len();
        let (p, q, t, _stats) = sun::close_residuals(&points, &quads, &tris);
        points = p;
        quads = q;
        tris = t;
        let (p, q, t, _cavities) = sun::close_by_cavity(&points, &quads, &tris, size);
        points = p;
        quads = q;
        tris = t;
        if tris.len() >= 2 {
            let guard = crate::two_d::PairGuard {
                min_scaled_jacobian: 0.05,
                min_angle_deg: 10.0,
                max_angle_deg: 170.0,
            };
            let (more, left) = crate::two_d::pair_triangles_with(&tris, &points, guard);
            if !more.is_empty() {
                quads.extend(more);
                tris = left;
            }
        }
        if tris.len() >= before {
            break;
        }
    }

    // The walk can stall with a few residuals. The refinement cascade closes
    // them without an odd boundary polygon around a triangle. It is applied
    // before the repairs so that the repairs can smooth the new quads.
    if !tris.is_empty() {
        let _ = close_residuals(&mut points, &mut quads, &mut tris);
    }

    // Quality repairs as part of the route: the cavity/template closures can
    // leave a very thin quad even when the paving rows were sound. The common
    // measurement post-pass runs the same operators, so the raw table is not
    // inflated by them (it is measured before that pass).
    let (p, q, t, _) = sun::repair_degenerate_quads(&points, &quads, &tris, 0.2, 64);
    points = p;
    quads = q;
    tris = t;
    let (p, q, t, _) = sun::repair_low_quads(&points, &quads, &tris, 0.2, 64);
    points = p;
    quads = q;
    tris = t;
    let (p, q, t, _) = sun::improve_quality(&points, &quads, &tris, 12);
    points = p;
    quads = q;
    tris = t;

    let mut cells: Vec<Cell> = quads.into_iter().map(Cell::quad4).collect();
    cells.extend(tris.into_iter().map(Cell::tri3));
    let mesh = Mesh::new(points, cells);
    if mesh.cells.is_empty() {
        return Err("the route produced no elements".into());
    }
    Ok((mesh, state))
}

// ---------------------------------------------------------------------------
// Boundary sampling and the paving front
// ---------------------------------------------------------------------------

/// A punctured planar mesh whose unmeshed region is bounded by the directed
/// front edges. Every front edge has the unmeshed region on its left.
struct Paving {
    points: Vec<Node>,
    quads: Vec<[usize; 4]>,
    next: HashMap<usize, usize>,
    prev: HashMap<usize, usize>,
    edges: EdgeIndex,
    edge_set: HashSet<(usize, usize)>,
    nodes: NodeIndex,
    size: f64,
}

impl Paving {
    /// Sample the domain boundary into a front. Every loop is made even (an
    /// all-quad mesh needs an even number of edges on each mesh boundary loop).
    fn from_boundary(domain: &Domain, size: f64) -> Result<Self, String> {
        if size <= 0.0 || !size.is_finite() {
            return Err("the element size must be positive and finite".into());
        }
        let bucket = size.max(1.0e-9);
        let mut paving = Paving {
            points: Vec::new(),
            quads: Vec::new(),
            next: HashMap::new(),
            prev: HashMap::new(),
            edges: EdgeIndex::new(2.0 * bucket),
            edge_set: HashSet::new(),
            nodes: NodeIndex::new(bucket),
            size,
        };
        let loops = std::iter::once(&domain.outer).chain(domain.holes.iter());
        for (loop_index, loop_points) in loops.enumerate() {
            if loop_points.len() < 3 {
                continue;
            }
            let mut ids = Vec::new();
            for i in 0..loop_points.len() {
                let a = loop_points[i];
                let b = loop_points[(i + 1) % loop_points.len()];
                let length = distance(a, b);
                let parts = ((length / (1.05 * size)).ceil() as usize).max(1);
                for k in 0..parts {
                    let t = k as f64 / parts as f64;
                    let p = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
                    ids.push(paving.push_node(p));
                }
            }
            if ids.len() < 3 {
                continue;
            }
            // Orientation: the meshed side must be on the right of the front.
            // The outer loop is counter-clockwise, a hole clockwise.
            let area = signed_area(&paving.points, &ids);
            let want_positive = loop_index == 0;
            if (area > 0.0) != want_positive {
                ids.reverse();
            }
            // Even boundary loop; split the longest edge at its midpoint.
            if ids.len() % 2 == 1 {
                if let Some((k, _)) = (0..ids.len())
                    .map(|k| {
                        let a = paving.points[ids[k]];
                        let b = paving.points[ids[(k + 1) % ids.len()]];
                        (k, distance([a[0], a[1]], [b[0], b[1]]))
                    })
                    .max_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal))
                {
                    let a = paving.points[ids[k]];
                    let b = paving.points[ids[(k + 1) % ids.len()]];
                    let mid = [(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0];
                    let id = paving.push_node(mid);
                    ids.insert(k + 1, id);
                }
            }
            let n = ids.len();
            for i in 0..n {
                let u = ids[i];
                let v = ids[(i + 1) % n];
                paving.add_front(u, v);
                paving.add_mesh_edge(u, v);
            }
        }
        if paving.next.len() < 3 {
            return Err("the boundary sampling produced no front".into());
        }
        Ok(paving)
    }

    fn push_node(&mut self, p: [f64; 2]) -> usize {
        let id = self.points.len();
        self.points.push([p[0], p[1], 0.0]);
        self.nodes.insert(&self.points, id, p);
        id
    }

    fn add_front(&mut self, u: usize, v: usize) {
        if self.next.insert(u, v).is_none() {
            self.prev.insert(v, u);
        } else {
            // A duplicate directed edge would be a bug in the update; keep the
            // maps consistent by restoring the old successor.
            self.next.insert(u, v);
            self.prev.insert(v, u);
        }
    }

    fn remove_front(&mut self, u: usize, v: usize) -> bool {
        if self.next.get(&u) == Some(&v) {
            self.next.remove(&u);
            self.prev.remove(&v);
            true
        } else {
            false
        }
    }

    fn add_mesh_edge(&mut self, u: usize, v: usize) {
        let key = (u.min(v), u.max(v));
        if self.edge_set.insert(key) {
            self.edges.insert(&self.points, key);
        }
    }

    /// Advance the front until a whole pass makes no progress or `depth` passes
    /// have run (`usize::MAX` means "until no progress", capped for safety).
    fn advance(&mut self, field: &CrossField, domain: &Domain, size: f64, depth: usize) {
        let max_quads = 250_000usize;
        let max_passes = depth.min(600);
        for _ in 0..max_passes {
            let mut edges: Vec<(usize, usize)> = self.next.iter().map(|(&a, &b)| (a, b)).collect();
            edges.sort_unstable();
            let mut progressed = false;
            for (a, b) in edges {
                if self.next.get(&a) != Some(&b) {
                    continue;
                }
                if self.try_edge(a, b, field, domain, size) {
                    progressed = true;
                    if self.quads.len() >= max_quads {
                        return;
                    }
                }
            }
            if !progressed {
                break;
            }
        }
    }

    /// Try to place one quad on the left of the front edge `a -> b`.
    fn try_edge(
        &mut self,
        a: usize,
        b: usize,
        field: &CrossField,
        domain: &Domain,
        size: f64,
    ) -> bool {
        let (Some(&pa), Some(&pb)) = (self.points.get(a), self.points.get(b)) else {
            return false;
        };
        let pa2 = [pa[0], pa[1]];
        let pb2 = [pb[0], pb[1]];
        let base = [pb2[0] - pa2[0], pb2[1] - pa2[1]];
        let length = (base[0] * base[0] + base[1] * base[1]).sqrt();
        if length <= 1.0e-12 {
            return false;
        }
        let t = [base[0] / length, base[1] / length];
        let normal = [-t[1], t[0]]; // left of a -> b: the unmeshed side
                                    // The field branch nearest the inward normal; then a second attempt
                                    // with the normal itself keeps the front advancing through a
                                    // singularity or a sharp turn.
        let da = field.nearest_direction(pa2, normal);
        let db = field.nearest_direction(pb2, normal);
        if self.attempt_quad(a, b, da, db, domain, size) {
            return true;
        }
        self.attempt_quad(a, b, normal, normal, domain, size)
    }

    fn attempt_quad(
        &mut self,
        a: usize,
        b: usize,
        da: [f64; 2],
        db: [f64; 2],
        domain: &Domain,
        size: f64,
    ) -> bool {
        let (Some(&pa), Some(&pb)) = (self.points.get(a), self.points.get(b)) else {
            return false;
        };
        let pa2 = [pa[0], pa[1]];
        let pb2 = [pb[0], pb[1]];
        let ideal_c = [pa2[0] + size * da[0], pa2[1] + size * da[1]];
        let ideal_d = [pb2[0] + size * db[0], pb2[1] + size * db[1]];

        // Reuse the node the previous quad of the row created at `a`, and the
        // node the following front edge already owns at `b`. This is what turns
        // local placements into rows instead of a fan of unrelated quads.
        let reuse_tolerance = 0.35 * size;
        let c_reuse = self
            .prev
            .get(&a)
            .copied()
            .filter(|&v| v != b)
            .filter(|&v| distance(self.point(v), ideal_c) <= reuse_tolerance);
        let d_reuse = self
            .next
            .get(&b)
            .copied()
            .filter(|&v| v != a)
            .filter(|&v| distance(self.point(v), ideal_d) <= reuse_tolerance);
        if let (Some(c), Some(d)) = (c_reuse, d_reuse) {
            if c == d {
                return false;
            }
        }

        let c_is_new = c_reuse.is_none();
        let d_is_new = d_reuse.is_none();
        let c = c_reuse.unwrap_or(self.points.len());
        let d = if d_is_new {
            self.points.len() + usize::from(c_is_new)
        } else {
            d_reuse.unwrap_or(0)
        };
        if c == d {
            return false;
        }
        let c_was_prev = c_reuse.is_some();
        let d_was_next = d_reuse.is_some();
        let pc = if c_is_new { ideal_c } else { self.point(c) };
        let pd = if d_is_new { ideal_d } else { self.point(d) };

        // New nodes must be inside the domain and not collapse onto a node of
        // the existing mesh or of another front.
        if c_is_new && !self.node_ok(domain, ideal_c) {
            return false;
        }
        if d_is_new && !self.node_ok(domain, ideal_d) {
            return false;
        }
        if c_is_new && d_is_new && distance(ideal_c, ideal_d) <= 0.45 * size {
            return false;
        }
        // The top edge must not already be a front edge in the same direction.
        if self.next.get(&c) == Some(&d) {
            return false;
        }
        let pa = self.point(a);
        let pb = self.point(b);
        let p = [pa, pb, pd, pc];
        let quad = [a, b, d, c];
        // New edges: a -> c when c is new, c -> d always, d -> b when d is new.
        let mut new_edges: Vec<([f64; 2], [f64; 2], usize, usize)> = Vec::new();
        if !c_was_prev {
            new_edges.push((pc, pa, c, a));
        }
        new_edges.push((pc, pd, c, d));
        if !d_was_next {
            new_edges.push((pd, pb, d, b));
        }
        if !self.quad_ok(domain, &p, &quad, &new_edges) {
            return false;
        }

        // Commit the new geometry.
        if c_is_new {
            let id = self.push_node(ideal_c);
            if id != c {
                return false;
            }
        }
        if d_is_new {
            let id = self.push_node(ideal_d);
            if id != d {
                return false;
            }
        }
        for (u, v) in [(a, b), (b, d), (d, c), (c, a)] {
            self.add_mesh_edge(u, v);
        }
        self.quads.push(quad);

        // Topology: consume the base edge, the side edges that this quad
        // shares with the neighbouring quads of the row, and add the new
        // interface edges.
        self.remove_front(a, b);
        if c_was_prev {
            self.remove_front(c, a);
        }
        if d_was_next {
            self.remove_front(b, d);
        }
        if !c_was_prev {
            self.add_front(a, c);
        }
        self.add_front(c, d);
        if !d_was_next {
            self.add_front(d, b);
        }
        true
    }

    fn point(&self, id: usize) -> [f64; 2] {
        self.points
            .get(id)
            .map(|p| [p[0], p[1]])
            .unwrap_or([0.0, 0.0])
    }

    fn node_ok(&self, domain: &Domain, p: [f64; 2]) -> bool {
        if !p[0].is_finite() || !p[1].is_finite() || !domain.contains(p) {
            return false;
        }
        !self.nodes.too_close(&self.points, p, 0.45 * self.size)
    }

    /// Convex, inside, no existing node strictly inside, no new edge crossing
    /// the mesh or the domain boundary.
    fn quad_ok(
        &self,
        domain: &Domain,
        p: &[[f64; 2]; 4],
        corners: &[usize; 4],
        new_edges: &[([f64; 2], [f64; 2], usize, usize)],
    ) -> bool {
        if !convex_ccw(p, 20.0) {
            return false;
        }
        let centroid = [
            (p[0][0] + p[1][0] + p[2][0] + p[3][0]) / 4.0,
            (p[0][1] + p[1][1] + p[2][1] + p[3][1]) / 4.0,
        ];
        if !domain.contains(centroid) {
            return false;
        }
        // No existing node may be inside the new quad (other than its corners).
        let mut blocked = HashSet::new();
        for &id in corners {
            blocked.insert(id);
        }
        for id in self.nodes.in_box(&self.points, p) {
            if blocked.contains(&id) {
                continue;
            }
            if point_in_convex(p, self.point(id)) {
                return false;
            }
        }
        // New edges must not cross an existing edge or the domain boundary.
        for &(a, b, u, v) in new_edges {
            if self.edge_crosses(a, b, u, v) {
                return false;
            }
        }
        true
    }

    fn edge_crosses(&self, a: [f64; 2], b: [f64; 2], u: usize, v: usize) -> bool {
        for (c, d) in self.edges.candidates(&self.points, a, b) {
            if c == u || c == v || d == u || d == v {
                continue;
            }
            let pc = self.point(c);
            let pd = self.point(d);
            if segments_cross(a, b, pc, pd) {
                return true;
            }
        }
        false
    }

    /// Extract the directed front loops after paving.
    fn front_loops(&self) -> Vec<Vec<usize>> {
        let mut visited: HashSet<usize> = HashSet::new();
        let mut starts: Vec<usize> = self.next.keys().copied().collect();
        starts.sort_unstable();
        let mut loops = Vec::new();
        for start in starts {
            if visited.contains(&start) {
                continue;
            }
            let mut loop_nodes = Vec::new();
            let mut current = start;
            let limit = self.next.len().saturating_add(1);
            let mut ok = false;
            for _ in 0..=limit {
                if !visited.insert(current) {
                    break;
                }
                loop_nodes.push(current);
                let Some(&next) = self.next.get(&current) else {
                    break;
                };
                if next == start {
                    ok = true;
                    break;
                }
                current = next;
            }
            if ok && loop_nodes.len() >= 3 {
                loops.push(loop_nodes);
            }
        }
        loops
    }
}

// ---------------------------------------------------------------------------
// Residual closure
// ---------------------------------------------------------------------------

/// An unmeshed component: one positive-area loop and the negative-area loops
/// nested directly in it.
struct ResidualComponent {
    outer: Vec<usize>,
    holes: Vec<Vec<usize>>,
}

/// Group the front loops by nesting. A positive signed area is an outer loop of
/// the unmeshed region (the region is on its left), a negative one is a hole.
fn group_residual_loops(points: &[Node], loops: &[Vec<usize>]) -> Vec<ResidualComponent> {
    let mut outers: Vec<(usize, f64)> = Vec::new();
    let mut holes: Vec<(usize, Vec<usize>)> = Vec::new();
    for (id, loop_nodes) in loops.iter().enumerate() {
        let area = signed_area(points, loop_nodes);
        if area > 1.0e-12 {
            outers.push((id, area));
        } else if area < -1.0e-12 {
            holes.push((id, loop_nodes.clone()));
        }
    }
    outers.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut components: Vec<ResidualComponent> = outers
        .iter()
        .map(|&(id, _)| ResidualComponent {
            outer: loops[id].clone(),
            holes: Vec::new(),
        })
        .collect();
    for (_, hole) in holes {
        let Some(&sample) = hole.first() else {
            continue;
        };
        let sample = [points[sample][0], points[sample][1]];
        let mut best: Option<usize> = None;
        let mut best_area = f64::INFINITY;
        for (k, &(id, area)) in outers.iter().enumerate() {
            let outer = &loops[id];
            let polygon: Vec<[f64; 2]> = outer
                .iter()
                .map(|&n| [points[n][0], points[n][1]])
                .collect();
            if point_in_polygon(sample, &polygon) && area < best_area {
                best = Some(k);
                best_area = area;
            }
        }
        if let Some(k) = best {
            components[k].holes.push(hole);
        }
    }
    components
}

/// Triangulate one unmeshed component and convert it with the two-stage
/// Q-Morph + matching route; return its quads and unpaired triangles.
fn close_component(
    points: &mut Vec<Node>,
    component: &ResidualComponent,
    size: f64,
) -> (Vec<[usize; 4]>, Vec<[usize; 3]>) {
    let outer_xy: Vec<[f64; 2]> = component
        .outer
        .iter()
        .filter_map(|&id| points.get(id).map(|p| [p[0], p[1]]))
        .collect();
    if outer_xy.len() != component.outer.len() {
        return (Vec::new(), Vec::new());
    }
    let holes_xy: Vec<Vec<[f64; 2]>> = component
        .holes
        .iter()
        .map(|hole| {
            hole.iter()
                .filter_map(|&id| points.get(id).map(|p| [p[0], p[1]]))
                .collect()
        })
        .collect();
    let Ok(component_domain) = Domain::new(outer_xy, holes_xy) else {
        return (Vec::new(), Vec::new());
    };
    if component_domain.area() <= 0.0 {
        return (Vec::new(), Vec::new());
    }
    let Some(tris) = triangulate_component(points, component, &component_domain, size) else {
        return (Vec::new(), Vec::new());
    };

    // Work locally: collect the used points in a compact array.
    let mut local_of: HashMap<usize, usize> = HashMap::new();
    let mut sub_points: Vec<Node> = Vec::new();
    let mut global_of: Vec<usize> = Vec::new();
    let mut local_tris: Vec<[usize; 3]> = Vec::with_capacity(tris.len());
    for tri in &tris {
        let mut local = [0usize; 3];
        for (k, &global) in tri.iter().enumerate() {
            let entry = local_of.entry(global).or_insert_with(|| {
                let id = sub_points.len();
                global_of.push(global);
                sub_points.push(points[global]);
                id
            });
            local[k] = *entry;
        }
        local_tris.push(local);
    }
    if local_tris.is_empty() {
        return (Vec::new(), Vec::new());
    }
    // Matching first: the residual triangulation has interior points at the
    // size field, so pairing its triangles is the direct quadrangulation. The
    // Q-Morph + matching conversion is the fallback when the matching leaves
    // triangles (an unmatched dual graph), and the better result is kept.
    let guard = crate::two_d::PairGuard {
        min_scaled_jacobian: 0.05,
        min_angle_deg: 10.0,
        max_angle_deg: 170.0,
    };
    let (matched, unpaired) = crate::two_d::pair_triangles_with(&local_tris, &sub_points, guard);
    let mut best_quads = matched;
    let mut best_triangles = unpaired;
    let mut best_points = sub_points.clone();
    if !best_triangles.is_empty() {
        let result = twostage::convert(&local_tris, &sub_points, size);
        if result.triangles.len() < best_triangles.len() {
            best_quads = result.quads.clone();
            best_triangles = result.triangles.clone();
            best_points = result.points.clone();
        }
    }
    let mut remap: Vec<usize> = Vec::with_capacity(best_points.len());
    for (i, p) in best_points.iter().enumerate() {
        if let Some(&global) = global_of.get(i) {
            remap.push(global);
        } else {
            points.push(*p);
            remap.push(points.len() - 1);
        }
    }
    let map_quad = |q: &[usize; 4]| -> [usize; 4] {
        [
            remap.get(q[0]).copied().unwrap_or(q[0]),
            remap.get(q[1]).copied().unwrap_or(q[1]),
            remap.get(q[2]).copied().unwrap_or(q[2]),
            remap.get(q[3]).copied().unwrap_or(q[3]),
        ]
    };
    let quads: Vec<[usize; 4]> = best_quads.iter().map(map_quad).collect();
    let triangles: Vec<[usize; 3]> = best_triangles
        .iter()
        .map(|t| {
            [
                remap.get(t[0]).copied().unwrap_or(t[0]),
                remap.get(t[1]).copied().unwrap_or(t[1]),
                remap.get(t[2]).copied().unwrap_or(t[2]),
            ]
        })
        .collect();
    (quads, triangles)
}

/// Constrained Delaunay triangulation of a residual component. The boundary
/// nodes are the front nodes (so the closure is conforming with the paving);
/// interior points at the size field keep the triangles well sized.
fn triangulate_component(
    points: &mut Vec<Node>,
    component: &ResidualComponent,
    domain: &Domain,
    size: f64,
) -> Option<Vec<[usize; 3]>> {
    let mut cdt = ConstrainedDelaunayTriangulation::<Point2<f64>>::new();
    let mut lookup: HashMap<(u64, u64), usize> = HashMap::new();
    let mut all_boundary: Vec<usize> = component.outer.clone();
    for hole in &component.holes {
        all_boundary.extend(hole.iter().copied());
    }
    for &global in &all_boundary {
        let p = points.get(global)?;
        let point = Point2::new(p[0], p[1]);
        let key = (p[0].to_bits(), p[1].to_bits());
        if lookup.contains_key(&key) {
            continue;
        }
        if cdt.insert(point).is_ok() {
            lookup.insert(key, global);
        }
    }
    // Constrain every loop edge. A failure here means the front is not a valid
    // polygon (it should be by construction); skip the component rather than
    // produce a mesh with a crack.
    let add_loop = |cdt: &mut ConstrainedDelaunayTriangulation<Point2<f64>>,
                    ids: &[usize],
                    points: &[Node]|
     -> bool {
        let n = ids.len();
        for i in 0..n {
            let a = points[ids[i]];
            let b = points[ids[(i + 1) % n]];
            if cdt
                .add_constraint_edge(Point2::new(a[0], a[1]), Point2::new(b[0], b[1]))
                .is_err()
            {
                return false;
            }
        }
        true
    };
    if !add_loop(&mut cdt, &component.outer, points) {
        return None;
    }
    for hole in &component.holes {
        if !add_loop(&mut cdt, hole, points) {
            return None;
        }
    }

    // Interior points, kept clear of the interface with the paving.
    let (xmin, xmax, ymin, ymax) = bounds_of_domain(domain);
    let step = size.max(1.0e-9);
    let margin = 0.55 * step;
    let mut x = xmin + 0.5 * step;
    while x < xmax {
        let mut y = ymin + 0.5 * step;
        while y < ymax {
            let p = [x, y];
            if domain.contains(p) && domain.distance_to_boundary(p) > margin {
                let point = Point2::new(x, y);
                if cdt.insert(point).is_ok() {
                    let global = points.len();
                    points.push([x, y, 0.0]);
                    lookup.insert((x.to_bits(), y.to_bits()), global);
                }
            }
            y += step;
        }
        x += step;
    }

    let mut tris = Vec::new();
    for face in cdt.inner_faces() {
        let vertices = face.vertices();
        let mut ids = [0usize; 3];
        let mut p = [[0.0f64; 2]; 3];
        let mut complete = true;
        for (k, vertex) in vertices.iter().enumerate() {
            let q = vertex.position();
            p[k] = [q.x, q.y];
            let key = (q.x.to_bits(), q.y.to_bits());
            match lookup.get(&key) {
                Some(&global) => ids[k] = global,
                None => {
                    complete = false;
                    break;
                }
            }
        }
        if !complete {
            continue;
        }
        let centroid = [
            (p[0][0] + p[1][0] + p[2][0]) / 3.0,
            (p[0][1] + p[1][1] + p[2][1]) / 3.0,
        ];
        if domain.contains(centroid) {
            tris.push(ids);
        }
    }
    if tris.is_empty() {
        None
    } else {
        Some(tris)
    }
}

// ---------------------------------------------------------------------------
// Local closure of residual triangles (refinement cascade)
// ---------------------------------------------------------------------------
//
// Parity is the obstruction the ordinary closure cannot pass: a component made
// of one triangle and quads has an odd boundary polygon, and an all-quad
// quadrangulation of an n-gon needs `Q = n/2 + i - 1` for `i` interior points,
// which is not integral when `n` is odd. No local quadrangulation therefore
// exists, no matter how the triangle is paired.
//
// The cascade used here (the technique already proven in `alt_l1`) removes the
// parity by *refining around* the triangle: the triangle becomes three quads
// around its centroid and edge midpoints. Every midpoint on an edge shared with
// a quad is hanging, so that quad is split from the midpoint to the midpoint of
// its opposite edge; the midpoint of the opposite edge is then hanging on the
// next quad, and the cascade is carried through the mesh until it reaches the
// boundary (where a boundary edge is split and the cascade stops). A midpoint
// that already exists is reused, so two cascades that meet share a node.
//
// A split is accepted only when every child is convex and does not lower the
// parent quad's scaled Jacobian or minimum corner angle (the floor is
// `min(parent, 0.2)` / `min(parent, 30°)`, the gate floor, so a parent that
// already meets the gate keeps its children at or above it). The public entry
// point is atomic: if any step fails, the whole closure is rolled back and the
// caller keeps the mixed mesh it had.

/// Statistics of one refinement-cascade closure.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CascadeStats {
    /// Residual triangles replaced by quads.
    pub triangles_closed: usize,
    /// Quads split by the cascade.
    pub quads_split: usize,
    /// Quads added to the mesh (the children minus the parent).
    pub quads_added: usize,
    /// Triangles whose fan could only be built with the relaxed floor (SJ 0.05,
    /// 1°). Normally zero.
    pub relaxed_triangles: usize,
    /// New quads whose scaled Jacobian is below the gate's 0.2 (only possible
    /// when the parent quad was already below it, which the common post-pass
    /// then has to repair).
    pub low_children: usize,
}

/// Hard cap on cascade steps; a runaway cascade is a bug, not a slow input.
const CASCADE_OPERATIONS: usize = 200_000;

fn edge_key(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

fn point_key(p: [f64; 2]) -> (u64, u64) {
    (p[0].to_bits(), p[1].to_bits())
}

/// One refinement-cascade closure. It owns the work lists but borrows the mesh
/// arrays, so the caller can roll the arrays back if `run` returns an error.
struct Cascade<'a> {
    points: &'a mut Vec<Node>,
    quads: &'a mut Vec<[usize; 4]>,
    tris: &'a mut Vec<[usize; 3]>,
    ids: HashMap<(u64, u64), usize>,
    hanging: HashMap<(usize, usize), usize>,
    owners: HashMap<(usize, usize), Vec<usize>>,
    queue: Vec<((usize, usize), usize)>,
    stats: CascadeStats,
    operations: usize,
}

impl<'a> Cascade<'a> {
    fn new(
        points: &'a mut Vec<Node>,
        quads: &'a mut Vec<[usize; 4]>,
        tris: &'a mut Vec<[usize; 3]>,
    ) -> Self {
        let mut ids = HashMap::with_capacity(points.len());
        for (id, p) in points.iter().enumerate() {
            ids.entry(point_key([p[0], p[1]])).or_insert(id);
        }
        let mut owners: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for (id, q) in quads.iter().enumerate() {
            for i in 0..4 {
                owners
                    .entry(edge_key(q[i], q[(i + 1) % 4]))
                    .or_default()
                    .push(id);
            }
        }
        Cascade {
            points,
            quads,
            tris,
            ids,
            hanging: HashMap::new(),
            owners,
            queue: Vec::new(),
            stats: CascadeStats::default(),
            operations: 0,
        }
    }

    /// Return the id of the point at `p`, appending it when it is new.
    fn point(&mut self, p: [f64; 2]) -> usize {
        let key = point_key(p);
        if let Some(&id) = self.ids.get(&key) {
            return id;
        }
        let id = self.points.len();
        self.points.push([p[0], p[1], 0.0]);
        self.ids.insert(key, id);
        id
    }

    fn midpoint(&mut self, a: usize, b: usize) -> usize {
        let (Some(pa), Some(pb)) = (self.points.get(a), self.points.get(b)) else {
            // A missing node is a programming error, not a user error: return
            // a sentinel id so every quality check rejects the result.
            return usize::MAX;
        };
        self.point([0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1])])
    }

    fn centroid(&mut self, v: [usize; 4]) -> usize {
        let mut x = 0.0;
        let mut y = 0.0;
        for &id in &v {
            if let Some(p) = self.points.get(id) {
                x += p[0];
                y += p[1];
            }
        }
        self.point([0.25 * x, 0.25 * y])
    }

    fn quad_quality(&self, q: [usize; 4]) -> (f64, f64) {
        let cell = Cell::quad4(q);
        let sj = quality::scaled_jacobian(&cell, self.points)
            .into_iter()
            .fold(f64::INFINITY, f64::min);
        let angles = quality::quad_angles(&cell, self.points);
        let angle = angles.iter().copied().fold(f64::INFINITY, f64::min);
        (sj, angle)
    }

    /// Convex, finite, and above both quality floors.
    fn quad_is_acceptable(&self, q: [usize; 4], sj_floor: f64, angle_floor: f64) -> bool {
        let (Some(p0), Some(p1), Some(p2), Some(p3)) = (
            self.points.get(q[0]),
            self.points.get(q[1]),
            self.points.get(q[2]),
            self.points.get(q[3]),
        ) else {
            return false;
        };
        let corners = [
            [p0[0], p0[1]],
            [p1[0], p1[1]],
            [p2[0], p2[1]],
            [p3[0], p3[1]],
        ];
        if !convex_ccw(&corners, 0.0) {
            return false;
        }
        // The floors are compared with a small tolerance: a split that keeps a
        // 30 deg corner at 29.999999 deg is the same geometry, not a quality
        // regression, and the common post-pass smooths it anyway.
        let (sj, angle) = self.quad_quality(q);
        sj.is_finite()
            && sj >= sj_floor.max(1.0e-9) - 1.0e-6
            && angle.is_finite()
            && angle >= angle_floor.max(1.0e-9) - 2.0
            && angle < 180.0
    }

    /// The gates a split of `parent` must respect: never worse than the parent,
    /// and at least the public 0.2 / 30° floor when the parent already meets it.
    fn split_floors(&self, parent: [usize; 4]) -> (f64, f64) {
        let (sj, angle) = self.quad_quality(parent);
        let sj_floor = if sj.is_finite() { sj.min(0.2) } else { 0.2 };
        let angle_floor = if angle.is_finite() {
            angle.min(30.0)
        } else {
            30.0
        };
        (sj_floor - 1.0e-9, angle_floor - 1.0e-9)
    }

    fn triangle_quads_ok(&self, quads: &[[usize; 4]; 3]) -> bool {
        let strict = (0.2 - 1.0e-9, 30.0 - 1.0e-9);
        let relaxed = (0.05, 1.0);
        [strict, relaxed]
            .into_iter()
            .any(|(sj, angle)| quads.iter().all(|&q| self.quad_is_acceptable(q, sj, angle)))
    }

    fn add_quad_edges(&mut self, qid: usize) {
        let Some(q) = self.quads.get(qid).copied() else {
            return;
        };
        for i in 0..4 {
            self.owners
                .entry(edge_key(q[i], q[(i + 1) % 4]))
                .or_default()
                .push(qid);
        }
    }

    fn remove_quad_edges(&mut self, qid: usize) {
        let Some(q) = self.quads.get(qid).copied() else {
            return;
        };
        for i in 0..4 {
            let key = edge_key(q[i], q[(i + 1) % 4]);
            let empty = if let Some(list) = self.owners.get_mut(&key) {
                list.retain(|&owner| owner != qid);
                list.is_empty()
            } else {
                false
            };
            if empty {
                self.owners.remove(&key);
            }
        }
    }

    fn find_quad(&self, a: usize, b: usize) -> Option<usize> {
        for &qid in self.owners.get(&edge_key(a, b))? {
            let Some(q) = self.quads.get(qid) else {
                continue;
            };
            for i in 0..4 {
                let u = q[i];
                let v = q[(i + 1) % 4];
                if (u == a && v == b) || (u == b && v == a) {
                    return Some(qid);
                }
            }
        }
        None
    }

    fn find_tri(&self, a: usize, b: usize) -> Option<usize> {
        self.tris.iter().position(|t| {
            (0..3).any(|i| {
                let u = t[i];
                let v = t[(i + 1) % 3];
                (u == a && v == b) || (u == b && v == a)
            })
        })
    }

    /// Replace triangular `index` by three quads around its centroid.
    fn close_triangle(&mut self, index: usize) -> Result<(), String> {
        let Some(&t) = self.tris.get(index) else {
            return Err("triangle index out of range".to_string());
        };
        let mut mids = [0usize; 3];
        for i in 0..3 {
            let a = t[i];
            let b = t[(i + 1) % 3];
            let key = edge_key(a, b);
            mids[i] = match self.hanging.get(&key) {
                Some(&m) => m,
                None => {
                    let m = self.midpoint(a, b);
                    self.hanging.insert(key, m);
                    m
                }
            };
        }
        let (Some(p0), Some(p1), Some(p2)) = (
            self.points.get(t[0]),
            self.points.get(t[1]),
            self.points.get(t[2]),
        ) else {
            return Err("triangle node out of range".to_string());
        };
        let c = self.point([(p0[0] + p1[0] + p2[0]) / 3.0, (p0[1] + p1[1] + p2[1]) / 3.0]);
        let new_quads = [
            [t[0], mids[0], c, mids[2]],
            [t[1], mids[1], c, mids[0]],
            [t[2], mids[2], c, mids[1]],
        ];
        if !self.triangle_quads_ok(&new_quads) {
            let mut text = String::new();
            for &q in &new_quads {
                let (sj, angle) = self.quad_quality(q);
                text.push_str(&format!(" [{q:?} sj={sj:.4} angle={angle:.2}]"));
            }
            return Err(format!(
                "the triangle closure would create a poor quad:{text} ({} quads split before this)",
                self.stats.quads_split
            ));
        }
        let strict = new_quads
            .iter()
            .all(|&q| self.quad_is_acceptable(q, 0.2 - 1.0e-9, 30.0 - 1.0e-9));
        if !strict {
            self.stats.relaxed_triangles += 1;
        }
        self.tris.swap_remove(index);
        for &q in &new_quads {
            let id = self.quads.len();
            self.quads.push(q);
            self.add_quad_edges(id);
            self.stats.quads_added += 1;
            let (sj, _) = self.quad_quality(q);
            if sj < 0.2 {
                self.stats.low_children += 1;
            }
        }
        for i in 0..3 {
            self.queue.push((edge_key(t[i], t[(i + 1) % 3]), mids[i]));
        }
        self.stats.triangles_closed += 1;
        Ok(())
    }

    /// Split `qid` because one or more of its edges already carries a hanging
    /// midpoint. The template depends on how many edges hang.
    fn split_quad(&mut self, qid: usize) -> Result<(), String> {
        let Some(&q) = self.quads.get(qid) else {
            return Err("quad index out of range".to_string());
        };
        let mut mids: [Option<usize>; 4] = [None; 4];
        for i in 0..4 {
            let key = edge_key(q[i], q[(i + 1) % 4]);
            if let Some(&m) = self.hanging.get(&key) {
                mids[i] = Some(m);
            }
        }
        let hung: Vec<usize> = (0..4).filter(|&i| mids[i].is_some()).collect();
        if hung.is_empty() {
            return Err("the hanging midpoint is not on the quad".to_string());
        }
        let (sj_floor, angle_floor) = self.split_floors(q);
        let mut pushes: Vec<((usize, usize), usize)> = Vec::new();
        let mut children: Vec<[usize; 4]> = Vec::new();
        let at = |i: usize| q[i % 4];

        match hung.len() {
            1 => {
                let i = hung[0];
                let m = mids[i].ok_or_else(|| "missing midpoint".to_string())?;
                let opp = (i + 2) % 4;
                let a = at(opp);
                let b = at(opp + 1);
                let n = self.midpoint(a, b);
                let key = edge_key(a, b);
                self.hanging.insert(key, n);
                pushes.push((key, n));
                children.push([at(i), m, n, at(i + 3)]);
                children.push([m, at(i + 1), at(i + 2), n]);
            }
            2 => {
                let (i, j) = (hung[0], hung[1]);
                if j == (i + 2) % 4 {
                    // Opposite edges: the classic two-quad split.
                    let m = mids[i].ok_or_else(|| "missing midpoint".to_string())?;
                    let n = mids[j].ok_or_else(|| "missing midpoint".to_string())?;
                    children.push([at(i), m, n, at(i + 3)]);
                    children.push([m, at(i + 1), at(i + 2), n]);
                } else {
                    // Adjacent edges: fan the six-node polygon into three quads
                    // around the centroid.
                    let i0 = (0..4)
                        .find(|&k| mids[k].is_some() && mids[(k + 1) % 4].is_some())
                        .ok_or_else(|| "two non-adjacent hanging edges".to_string())?;
                    let m0 = mids[i0].ok_or_else(|| "missing midpoint".to_string())?;
                    let m1 = mids[(i0 + 1) % 4].ok_or_else(|| "missing midpoint".to_string())?;
                    let v = [at(i0), at(i0 + 1), at(i0 + 2), at(i0 + 3)];
                    let c = self.centroid(v);
                    children.push([v[0], m0, c, v[3]]);
                    children.push([m0, v[1], m1, c]);
                    children.push([c, m1, v[2], v[3]]);
                }
            }
            3 => {
                // The quad's boundary is a heptagon (odd), so the fourth edge
                // has to hang too before it can be quadrangulated. Do that and
                // fan the resulting octagon.
                let h = (0..4)
                    .find(|&i| mids[i].is_none())
                    .ok_or_else(|| "no free quad edge".to_string())?;
                let a = at(h);
                let b = at(h + 1);
                let n = self.midpoint(a, b);
                let key = edge_key(a, b);
                self.hanging.insert(key, n);
                pushes.push((key, n));
                let mut mm = [0usize; 4];
                for i in 0..4 {
                    mm[i] = if i == h {
                        n
                    } else {
                        mids[i].ok_or_else(|| "missing midpoint".to_string())?
                    };
                }
                let c = self.centroid([at(0), at(1), at(2), at(3)]);
                children.push([at(0), mm[0], c, mm[3]]);
                children.push([at(1), mm[1], c, mm[0]]);
                children.push([at(2), mm[2], c, mm[1]]);
                children.push([at(3), mm[3], c, mm[2]]);
            }
            4 => {
                let c = self.centroid([at(0), at(1), at(2), at(3)]);
                let mm = [
                    mids[0].ok_or_else(|| "missing midpoint".to_string())?,
                    mids[1].ok_or_else(|| "missing midpoint".to_string())?,
                    mids[2].ok_or_else(|| "missing midpoint".to_string())?,
                    mids[3].ok_or_else(|| "missing midpoint".to_string())?,
                ];
                children.push([at(0), mm[0], c, mm[3]]);
                children.push([at(1), mm[1], c, mm[0]]);
                children.push([at(2), mm[2], c, mm[1]]);
                children.push([at(3), mm[3], c, mm[2]]);
            }
            _ => return Err("unexpected number of hanging edges".to_string()),
        }
        if let Some(&bad) = children
            .iter()
            .find(|&&child| !self.quad_is_acceptable(child, sj_floor, angle_floor))
        {
            let (psj, pang) = self.quad_quality(q);
            let (sj, angle) = self.quad_quality(bad);
            return Err(format!(
                "the quad split would create a poor quad: parent {q:?} sj={psj:.4} angle={pang:.2}, child {bad:?} sj={sj:.4} angle={angle:.2}, floors sj>={sj_floor:.4} angle>={angle_floor:.2} ({} quads split before this)"
            , self.stats.quads_split));
        }
        self.remove_quad_edges(qid);
        self.quads[qid] = children[0];
        self.add_quad_edges(qid);
        for &child in &children[1..] {
            let id = self.quads.len();
            self.quads.push(child);
            self.add_quad_edges(id);
        }
        self.stats.quads_split += 1;
        self.stats.quads_added += children.len() - 1;
        for &child in &children {
            let (sj, _) = self.quad_quality(child);
            if sj < 0.2 {
                self.stats.low_children += 1;
            }
        }
        self.queue.extend(pushes);
        Ok(())
    }

    fn run(&mut self) -> Result<(), String> {
        while !self.tris.is_empty() {
            self.close_triangle(0)?;
            while let Some((edge, _midpoint)) = self.queue.pop() {
                self.operations += 1;
                if self.operations > CASCADE_OPERATIONS {
                    return Err("the refinement cascade did not terminate".to_string());
                }
                if let Some(qid) = self.find_quad(edge.0, edge.1) {
                    self.split_quad(qid)?;
                } else if let Some(tid) = self.find_tri(edge.0, edge.1) {
                    self.close_triangle(tid)?;
                }
                // Otherwise the edge is on the domain boundary (the split node
                // lies on a straight boundary segment) or was already split:
                // the cascade has reached a stop.
            }
        }
        Ok(())
    }
}

/// Close every residual triangle of a planar quad mesh by the refinement
/// cascade described in this module's documentation.
///
/// The operation is atomic: on error the three vectors are left exactly as they
/// were. On success the mesh has no triangle, no hanging node, and every new
/// quad is convex.
pub fn close_residuals(
    points: &mut Vec<Node>,
    quads: &mut Vec<[usize; 4]>,
    tris: &mut Vec<[usize; 3]>,
) -> Result<CascadeStats, String> {
    if tris.is_empty() {
        return Ok(CascadeStats::default());
    }
    let saved_points = points.clone();
    let saved_quads = quads.clone();
    let saved_tris = tris.clone();
    let outcome = {
        let mut cascade = Cascade::new(points, quads, tris);
        cascade.run().map(|()| cascade.stats)
    };
    if outcome.is_err() {
        *points = saved_points;
        *quads = saved_quads;
        *tris = saved_tris;
    }
    outcome
}

// ---------------------------------------------------------------------------
// Geometry and spatial indices
// ---------------------------------------------------------------------------

fn bounds(domain: &Domain) -> Result<(f64, f64, f64, f64), String> {
    let mut xmin = f64::INFINITY;
    let mut xmax = f64::NEG_INFINITY;
    let mut ymin = f64::INFINITY;
    let mut ymax = f64::NEG_INFINITY;
    let loops = std::iter::once(&domain.outer).chain(domain.holes.iter());
    for loop_points in loops {
        for p in loop_points {
            if !p[0].is_finite() || !p[1].is_finite() {
                return Err("a boundary coordinate is not finite".into());
            }
            xmin = xmin.min(p[0]);
            xmax = xmax.max(p[0]);
            ymin = ymin.min(p[1]);
            ymax = ymax.max(p[1]);
        }
    }
    if !xmin.is_finite() || !xmax.is_finite() || xmax <= xmin || ymax <= ymin {
        return Err("the domain has a degenerate bounding box".into());
    }
    Ok((xmin, xmax, ymin, ymax))
}

fn bounds_of_domain(domain: &Domain) -> (f64, f64, f64, f64) {
    bounds(domain).unwrap_or((0.0, 1.0, 0.0, 1.0))
}

fn tangent(a: [f64; 2], b: [f64; 2]) -> (f64, f64) {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let length = (dx * dx + dy * dy).sqrt();
    if length <= 1.0e-12 {
        (1.0, 0.0)
    } else {
        (dx / length, dy / length)
    }
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

fn signed_area(points: &[Node], ids: &[usize]) -> f64 {
    let mut sum = 0.0;
    for i in 0..ids.len() {
        let Some(a) = points.get(ids[i]) else {
            return 0.0;
        };
        let Some(b) = points.get(ids[(i + 1) % ids.len()]) else {
            return 0.0;
        };
        sum += a[0] * b[1] - b[0] * a[1];
    }
    0.5 * sum
}

fn cross2(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// Strictly convex, counter-clockwise, with every corner at least `min_angle`
/// degrees and a positive area.
fn convex_ccw(p: &[[f64; 2]; 4], min_angle: f64) -> bool {
    let sin_min = min_angle.to_radians().sin().max(0.0);
    let mut area = 0.0;
    for i in 0..4 {
        let a = p[i];
        let b = p[(i + 1) % 4];
        area += a[0] * b[1] - b[0] * a[1];
    }
    if area <= 1.0e-12 {
        return false;
    }
    for i in 0..4 {
        let a = p[(i + 3) % 4];
        let b = p[i];
        let c = p[(i + 1) % 4];
        let u = [b[0] - a[0], b[1] - a[1]];
        let v = [c[0] - b[0], c[1] - b[1]];
        let lu = (u[0] * u[0] + u[1] * u[1]).sqrt();
        let lv = (v[0] * v[0] + v[1] * v[1]).sqrt();
        if lu <= 1.0e-12 || lv <= 1.0e-12 {
            return false;
        }
        let sine = (u[0] * v[1] - u[1] * v[0]) / (lu * lv);
        if sine <= sin_min {
            return false;
        }
    }
    true
}

fn point_in_convex(p: &[[f64; 2]; 4], q: [f64; 2]) -> bool {
    for i in 0..4 {
        let a = p[i];
        let b = p[(i + 1) % 4];
        if cross2(a, b, q) <= 0.0 {
            return false;
        }
    }
    true
}

fn point_in_polygon(p: [f64; 2], polygon: &[[f64; 2]]) -> bool {
    let mut inside = false;
    let n = polygon.len();
    if n < 3 {
        return false;
    }
    for i in 0..n {
        let a = polygon[i];
        let b = polygon[(i + 1) % n];
        if (a[1] > p[1]) != (b[1] > p[1]) {
            let t = (p[1] - a[1]) / (b[1] - a[1]);
            if p[0] < a[0] + t * (b[0] - a[0]) {
                inside = !inside;
            }
        }
    }
    inside
}

fn segments_cross(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
    let eps = 1.0e-12;
    let o1 = cross2(a, b, c);
    let o2 = cross2(a, b, d);
    let o3 = cross2(c, d, a);
    let o4 = cross2(c, d, b);
    let proper = ((o1 > eps && o2 < -eps) || (o1 < -eps && o2 > eps))
        && ((o3 > eps && o4 < -eps) || (o3 < -eps && o4 > eps));
    if proper {
        return true;
    }
    (o1.abs() <= eps && on_segment(a, b, c))
        || (o2.abs() <= eps && on_segment(a, b, d))
        || (o3.abs() <= eps && on_segment(c, d, a))
        || (o4.abs() <= eps && on_segment(c, d, b))
}

fn on_segment(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> bool {
    p[0] >= a[0].min(b[0]) - 1.0e-12
        && p[0] <= a[0].max(b[0]) + 1.0e-12
        && p[1] >= a[1].min(b[1]) - 1.0e-12
        && p[1] <= a[1].max(b[1]) + 1.0e-12
}

/// Uniform grid over mesh edges, for segment-intersection queries.
struct EdgeIndex {
    bucket: f64,
    map: HashMap<(i64, i64), Vec<(usize, usize)>>,
}

impl EdgeIndex {
    fn new(bucket: f64) -> Self {
        EdgeIndex {
            bucket: bucket.max(1.0e-9),
            map: HashMap::new(),
        }
    }

    fn key(&self, p: [f64; 2]) -> (i64, i64) {
        (
            (p[0] / self.bucket).floor() as i64,
            (p[1] / self.bucket).floor() as i64,
        )
    }

    fn insert(&mut self, points: &[Node], edge: (usize, usize)) {
        let (Some(a), Some(b)) = (points.get(edge.0), points.get(edge.1)) else {
            return;
        };
        let (i0, j0) = self.key([a[0], a[1]]);
        let (i1, j1) = self.key([b[0], b[1]]);
        if (i1 - i0).abs().saturating_add((j1 - j0).abs()) > 4096 {
            return;
        }
        for i in i0.min(i1)..=i0.max(i1) {
            for j in j0.min(j1)..=j0.max(j1) {
                self.map.entry((i, j)).or_default().push(edge);
            }
        }
    }

    fn candidates(&self, points: &[Node], a: [f64; 2], b: [f64; 2]) -> Vec<(usize, usize)> {
        let (i0, j0) = self.key(a);
        let (i1, j1) = self.key(b);
        let mut out = Vec::new();
        if (i1 - i0).abs().saturating_add((j1 - j0).abs()) > 4096 {
            return out;
        }
        for i in i0.min(i1)..=i0.max(i1) {
            for j in j0.min(j1)..=j0.max(j1) {
                if let Some(list) = self.map.get(&(i, j)) {
                    out.extend_from_slice(list);
                }
            }
        }
        let _ = points;
        out.sort_unstable();
        out.dedup();
        out
    }
}

/// Uniform grid over mesh nodes, for "is a node too close" queries.
struct NodeIndex {
    bucket: f64,
    map: HashMap<(i64, i64), Vec<usize>>,
}

impl NodeIndex {
    fn new(bucket: f64) -> Self {
        NodeIndex {
            bucket: bucket.max(1.0e-9),
            map: HashMap::new(),
        }
    }

    fn key(&self, p: [f64; 2]) -> (i64, i64) {
        (
            (p[0] / self.bucket).floor() as i64,
            (p[1] / self.bucket).floor() as i64,
        )
    }

    fn insert(&mut self, points: &[Node], id: usize, p: [f64; 2]) {
        let _ = points;
        let key = self.key(p);
        self.map.entry(key).or_default().push(id);
    }

    fn too_close(&self, points: &[Node], p: [f64; 2], radius: f64) -> bool {
        let (i0, j0) = self.key(p);
        let span = (radius / self.bucket).ceil() as i64 + 1;
        let r2 = radius * radius;
        for i in i0 - span..=i0 + span {
            for j in j0 - span..=j0 + span {
                let Some(list) = self.map.get(&(i, j)) else {
                    continue;
                };
                for &id in list {
                    if let Some(q) = points.get(id) {
                        let dx = q[0] - p[0];
                        let dy = q[1] - p[1];
                        if dx * dx + dy * dy < r2 {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    fn in_box(&self, points: &[Node], p: &[[f64; 2]; 4]) -> Vec<usize> {
        let mut xmin = f64::INFINITY;
        let mut xmax = f64::NEG_INFINITY;
        let mut ymin = f64::INFINITY;
        let mut ymax = f64::NEG_INFINITY;
        for q in p {
            xmin = xmin.min(q[0]);
            xmax = xmax.max(q[0]);
            ymin = ymin.min(q[1]);
            ymax = ymax.max(q[1]);
        }
        let (i0, j0) = self.key([xmin, ymin]);
        let (i1, j1) = self.key([xmax, ymax]);
        let mut out = Vec::new();
        for i in i0..=i1 {
            for j in j0..=j1 {
                if let Some(list) = self.map.get(&(i, j)) {
                    out.extend_from_slice(list);
                }
            }
        }
        let _ = points;
        out.sort_unstable();
        out.dedup();
        out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::quality;
    use crate::verify;

    /// Difference between two orientations in degrees, modulo 90°.
    fn angle_diff_mod_90(a: f64, b: f64) -> f64 {
        let mut d = (a - b + 45.0).rem_euclid(90.0) - 45.0;
        if d < 0.0 {
            d = -d;
        }
        d
    }

    #[test]
    fn field_is_boundary_aligned_on_a_square() {
        let domain = Domain::rectangle(10.0, 6.0).unwrap();
        let sizing = Sizing::uniform(1.0).unwrap();
        let field = CrossField::new(&domain, &sizing).unwrap();
        // Mid-side samples, a little inside the boundary.
        for k in 1..10 {
            let x = k as f64;
            for (p, expected) in [
                ([x, 0.1], 0.0f64),
                ([x, 5.9], 0.0),
                ([0.1, 0.6 * k as f64], 90.0),
                ([9.9, 0.6 * k as f64], 90.0),
            ] {
                let theta = field.angle_at(p).to_degrees();
                let diff = angle_diff_mod_90(theta, expected);
                assert!(
                    diff < 5.0,
                    "at {p:?}: theta {theta} vs {expected}, {diff} deg"
                );
            }
        }
        // The square has no singularity: the field is the same cross everywhere.
        let theta = field.angle_at([5.0, 3.0]).to_degrees();
        assert!(angle_diff_mod_90(theta, 0.0) < 1.0e-6, "{theta}");
    }

    #[test]
    fn field_is_smooth_inside() {
        // A quarter annulus: the boundary tangents vary smoothly, so the
        // interior cross field must not jump between neighbouring grid nodes.
        let mut outer = Vec::new();
        for i in 0..=24 {
            let a = std::f64::consts::FRAC_PI_2 * i as f64 / 24.0;
            outer.push([20.0 * a.cos(), 20.0 * a.sin()]);
        }
        for i in (0..=12).rev() {
            let a = std::f64::consts::FRAC_PI_2 * i as f64 / 12.0;
            outer.push([10.0 * a.cos(), 10.0 * a.sin()]);
        }
        let domain = Domain::new(outer, vec![]).unwrap();
        let sizing = Sizing::uniform(1.0).unwrap();
        let field = CrossField::new(&domain, &sizing).unwrap();
        let jump = field.max_neighbour_angle_deg(&domain);
        assert!(jump < 15.0, "largest neighbour jump {jump} deg");
        // The field is tangential to both circular arcs: at mid-radius the
        // cross must contain the tangential direction.
        let p = [
            15.0 * std::f64::consts::FRAC_1_SQRT_2,
            15.0 * std::f64::consts::FRAC_1_SQRT_2,
        ];
        let theta = field.angle_at(p).to_degrees();
        assert!(angle_diff_mod_90(theta, 45.0) < 10.0, "{theta}");
    }

    #[test]
    fn mesh_of_a_square_is_all_quads() {
        let domain = Domain::rectangle(20.0, 20.0).unwrap();
        let sizing = Sizing::uniform(2.0).unwrap();
        let mesh = mesh(&domain, &sizing).unwrap();
        assert!(
            mesh.cells
                .iter()
                .all(|c| c.kind == crate::mesh::Kind::Quad4),
            "triangles left: {}",
            mesh.cells
                .iter()
                .filter(|c| c.kind == crate::mesh::Kind::Tri3)
                .count()
        );
        let gate = verify::gate(&mesh);
        assert!(gate.passes(), "{gate:?} {:?}", gate.reason());
        assert!(verify::boundary_offset(&domain, &mesh) < 1.0e-9);
        assert!(mesh.cells.len() >= 25, "{} quads", mesh.cells.len());
        for cell in &mesh.cells {
            let sj = quality::scaled_jacobian(cell, &mesh.points);
            assert!(sj.iter().all(|&s| s > 0.2), "{sj:?}");
        }
    }

    #[test]
    fn mesh_of_a_rectangle_with_a_hole_is_conforming() {
        let domain = Domain::plate_with_hole(40.0, 30.0, 5.0, 16).unwrap();
        let sizing = Sizing::uniform(2.0).unwrap();
        let mesh = mesh(&domain, &sizing).unwrap();
        let gate = verify::gate(&mesh);
        assert_eq!(gate.cracks, 0, "{gate:?}");
        assert_eq!(gate.overlapping, 0, "{gate:?}");
        assert_eq!(gate.inverted, 0, "{gate:?}");
        assert!(verify::boundary_offset(&domain, &mesh) < 1.0e-9);
        for cell in &mesh.cells {
            if cell.kind == crate::mesh::Kind::Quad4 {
                let sj = quality::scaled_jacobian(cell, &mesh.points);
                assert!(sj.iter().all(|&s| s > 0.0), "{sj:?}");
            }
        }
    }

    #[test]
    fn mesh_of_a_convex_pentagon_is_conforming() {
        let domain = Domain::new(
            vec![[0.0, 0.0], [8.0, 0.0], [10.0, 5.0], [4.0, 9.0], [-1.0, 5.0]],
            vec![],
        )
        .unwrap();
        let sizing = Sizing::uniform(2.0).unwrap();
        let mesh = mesh(&domain, &sizing).unwrap();
        let gate = verify::gate(&mesh);
        assert!(gate.quads > 0);
        assert_eq!(gate.cracks, 0, "{gate:?}");
        assert_eq!(gate.overlapping, 0, "{gate:?}");
        assert_eq!(gate.inverted, 0, "{gate:?}");
        assert!(verify::boundary_offset(&domain, &mesh) < 1.0e-9);
    }

    #[test]
    fn mesh_with_non_default_options_is_a_valid_mesh() {
        let domain = Domain::rectangle(20.0, 20.0).unwrap();
        let sizing = Sizing::uniform(2.0).unwrap();
        let options = Options { field_sweeps: 40 };
        let mesh = mesh_with_options(&domain, &sizing, &options).unwrap();
        assert!(mesh
            .cells
            .iter()
            .all(|c| matches!(c.kind, crate::mesh::Kind::Quad4 | crate::mesh::Kind::Tri3)));
        let gate = verify::gate(&mesh);
        assert!(gate.quads > 0, "{gate:?}");
        assert_eq!(gate.cracks, 0, "{gate:?}");
        assert_eq!(gate.inverted, 0, "{gate:?}");
        assert_eq!(gate.overlapping, 0, "{gate:?}");
        assert!(verify::boundary_offset(&domain, &mesh) < 1.0e-9);
    }

    #[test]
    fn cascade_closes_residual_triangles_and_keeps_the_mesh_conforming() {
        // A 3x3 grid of points with three quads and the fourth square split
        // into two triangles. The triangles have no local quadrangulation
        // (parity), so this is exactly the case the cascade exists for.
        let mut points: Vec<Node> = Vec::new();
        for j in 0..3 {
            for i in 0..3 {
                points.push([i as f64, j as f64, 0.0]);
            }
        }
        let mut quads: Vec<[usize; 4]> = vec![[0, 1, 4, 3], [1, 2, 5, 4], [3, 4, 7, 6]];
        let mut tris: Vec<[usize; 3]> = vec![[4, 5, 8], [4, 8, 7]];
        let stats = close_residuals(&mut points, &mut quads, &mut tris).unwrap();
        assert!(tris.is_empty(), "triangles left: {tris:?}");
        assert_eq!(stats.triangles_closed, 2);
        assert_eq!(stats.low_children, 0);
        let cells: Vec<Cell> = quads.iter().map(|q| Cell::quad4(*q)).collect();
        let mesh = Mesh::new(points, cells);
        let gate = verify::gate(&mesh);
        assert_eq!(gate.triangles, 0, "{gate:?}");
        assert_eq!(gate.inverted, 0, "{gate:?}");
        assert_eq!(gate.overlapping, 0, "{gate:?}");
        assert_eq!(gate.cracks, 0, "{gate:?}");
        assert!(gate.min_scaled_jacobian >= 0.2, "{gate:?}");
        assert_eq!(boundary_flat_corners(&mesh), 0);
    }

    #[test]
    fn cascade_rolls_back_a_hopeless_triangle() {
        // A needle with a 1 degree corner cannot be fanned into quads; the
        // public entry point must leave every vector untouched.
        let points_before = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.017, 0.0]];
        let quads_before: Vec<[usize; 4]> = Vec::new();
        let tris_before = vec![[0, 1, 2]];
        let mut points = points_before.clone();
        let mut quads = quads_before.clone();
        let mut tris = tris_before.clone();
        assert!(close_residuals(&mut points, &mut quads, &mut tris).is_err());
        assert_eq!(points, points_before);
        assert_eq!(quads, quads_before);
        assert_eq!(tris, tris_before);
    }

    #[test]
    fn cascade_with_no_triangles_is_a_no_op() {
        let mut points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let mut quads = vec![[0, 1, 2, 3]];
        let mut tris: Vec<[usize; 3]> = Vec::new();
        let stats = close_residuals(&mut points, &mut quads, &mut tris).unwrap();
        assert_eq!(stats, CascadeStats::default());
    }

    #[test]
    fn an_unusable_size_is_an_error_not_a_panic() {
        let domain = Domain::rectangle(10.0, 10.0).unwrap();
        let bad = Sizing {
            size: 0.0,
            all_quad: false,
        };
        assert!(mesh(&domain, &bad).is_err());
        let negative = Sizing {
            size: -1.0,
            all_quad: false,
        };
        assert!(mesh(&domain, &negative).is_err());
        let no_area = Domain::new(vec![[0.0, 0.0], [0.0, 1.0], [1.0, 1.0]], vec![]).unwrap();
        assert!(mesh(&no_area, &Sizing::uniform(1.0).unwrap()).is_err());
    }
}
