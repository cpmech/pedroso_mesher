//! PAVING: direct all-quad generation by advancing rows of quads.
//!
//! Implemented from Blacker & Stephenson (1991), *Paving: a new approach to
//! automated quadrilateral mesh generation* (IJNME 32(4):811-847, corpus id
//! `blacker-1991`), §2. The route works on ordered,
//! closed front loops – the outer loop counter-clockwise and every hole
//! clockwise, so that the material is always on the left of the loop – and
//! advances each loop towards the interior by one row at a time. Every row is
//! built between the old front nodes and their inward projections, so a quad is
//! always added between existing nodes and a T-junction can never appear. When
//! a projected row reaches another front it is seamed onto it instead of being
//! refused (§2.8), so the two fronts merge and paving continues.
//!
//! What is implemented
//! -------------------
//! * Boundary sampling: each closed loop is sampled near the requested size,
//!   keeping every geometric vertex on the polyline. A short polyline segment
//!   may be kept whole (up to about `1.35 h`) rather than split into collinear
//!   nodes, and a sharp corner is left `2.4 h` clear on each straight side.
//!   The clearance is what gives the row generator enough room to keep more
//!   than one row alive at a corner; `1.25 h` also works but stops after one
//!   row. Sampling a node exactly `h` from a corner collapses the corner quads.
//! * Row choice (§2.4): loops are visited smallest perimeter first; a loop is
//!   advanced by a full row around the loop, one row at a time, until the next
//!   row would be invalid. `mesh_with_stats` reports how many quads came from
//!   rows and how many from the residual closure.
//! * Row generation (§2.6): a row-side node is projected along the interior
//!   angle bisector at the mitered offset distance `h / sin(α/2)`. At a sharp
//!   *convex* corner the projection is pushed a further `0.25 h` along the
//!   bisector; this rounds the new front slightly, which is what keeps the two
//!   corner quads convex when the adjacent boundary node is `2.4 h` away.
//!   A reflex corner keeps the plain miter. With a constant size field this is
//!   the paper's projection.
//! * Closure check and small-loop closure (§2.5): a 4-node front closes with one
//!   quad; a 6-node front closes with two quads (the fan pairing) when both are
//!   valid.
//! * Row validity: the new front must stay simple, inside the domain, at least
//!   `1.2 h` from every other front, and every new quad must have a positive
//!   scaled Jacobian. If a row fails, that loop stops advancing. A separate
//!   pass refuses the row before any quad of it is committed, so the partial
//!   mesh is always conforming and crack-free.
//! * Seaming (§2.8): a row that would run into another front is no longer
//!   refused outright. When a run of consecutive projected nodes has, on the
//!   met front, consecutive nearest nodes within `SEAM_REACH` (in reverse, as
//!   two facing fronts run in opposite directions), the run is merged
//!   node-to-node: the row quads take the met front's node ids, the welded
//!   edges (already front edges of that front) become interior, and the
//!   unseamed parts of the two loops are spliced into one. This is the paper's
//!   node-combining seam; nothing is split. The candidate is checked against
//!   the whole mesh (cracks, inverted cells, overlapping cells) and the row is
//!   refused as before if any of them would be broken.
//! * Smoothing (§2.7): a global `smooth_mesh` pass at the end; the smoother is
//!   the project's, not a paving-specific boundary smoother.
//! * Cleanup (§2.11): `sun::repair_low_quads` and `sun::perturb_low_quads` run
//!   on the finished mesh, and a valence-driven edge swap replaces the shared
//!   edge of two quads by the other diagonal of their hexagon whenever it
//!   lowers the interior `sum |valence - 4|` and both new quads stay above the
//!   gate. No element is deleted. The common post-pass is run again by the
//!   example.
//!
//! Measured result
//! ---------------
//! On the seven-part corpus the route is all-quad and passes the gate raw and
//! after the common post-pass (7/7; measured raw SJ 0.62-0.70, corner 38-45
//! degrees, about 15 200 quads on the 0.6 mm part). [`Stats`] measures how much
//! of that is actual paving. With the §2.8 seam off, rows contribute 17-45 % of
//! the quads (the 4 mm plate-hole is 128/328 = 39 %;
//! `alts/results/paving.txt`). The seam was added for the 4 mm plate-hole, where
//! the hole front meets the plate front head-on: it raises the row share from
//! 39 % (128 row + 200 core) to 57 % (192 row + 143 core). The rest of every
//! mesh is the matched core. The smallest measured share is the stepped part at
//! 17 % (376/2158), whose single front stops after two rows at its concave
//! corners.
//!
//! What is not implemented
//! -----------------------
//! * The residual region left when a front is stopped by the validity guard, or
//!   the annulus left between two fronts, is closed by a **constrained Delaunay
//!   triangulation plus Blossom triangle pairing** (`two_d::pair_triangles_with`)
//!   – not by further paving rows. This is a documented deviation: it is the
//!   closure fallback, not a paving step. If the matching is obstructed, a few
//!   edge-split retries are tried and a simply connected residual is closed by
//!   an unconditional fan. Without this the route would not finish on all
//!   corpus parts; with it every part is all-quad.
//! * Seaming only covers the count-matched case above. A *transition seam*
//!   between sides of unequal length (the wedge of Figures 21 and 23) is not
//!   implemented, and neither is §2.9 row adjustment with tucks and wedges.
//!   On a front that is expanding into the interior the node spacing grows
//!   with every row, so its nodes no longer line up with the finer front it
//!   meets; that contact cannot be seamed and still goes to the residual
//!   closure. §2.10 perimeter-intersection connection (a front crossing itself
//!   or another front, rather than touching it) is also not implemented.
//! * Rows are closed rings advanced around a whole loop, not the paper's open
//!   rows that begin and end at row-end nodes (§2.4/§2.6). At a 90° convex
//!   corner the front edge next to the corner therefore contracts by one node
//!   spacing per row and its two corner quads invert after two rows. This is
//!   why the single-loop parts (l-bracket, stepped, ring) stop after one or
//!   two rows; seaming cannot help there because no second front exists at the
//!   corner. The paper removes the contraction with row-end termination and
//!   tucks (§2.9).
//! * Anisotropic sizing, feature capture beyond the boundary segments, and
//!   curved geometry (the boundary is the input polyline) are out of scope.
//!
//! Clean room: written from the paper only; no implementation was read or ported.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use spade::{ConstrainedDelaunayTriangulation, Point2, Triangulation};

use crate::mesh::{Cell, Mesh, Node};
use crate::quality;
use crate::sun;
use crate::two_d::{pair_triangles_with, smooth_mesh, Domain, PairGuard, Sizing};

/// A turn larger than this is treated as a geometric corner by the sampler.
const CORNER_TURN: f64 = 30.0_f64 * std::f64::consts::PI / 180.0;
/// Distance from a sharp corner to the first sampled boundary node, in sizes.
const CORNER_MARGIN: f64 = 2.4;
/// The smallest front edge length a row may produce, in sizes.
const MIN_EDGE: f64 = 0.08;
/// New fronts closer than this to another front are refused, in sizes.
const FRONT_GAP: f64 = 1.2;
/// A row node within this distance of a node of the front it meets is seamed
/// onto it, in sizes. The paper's seam is a node merge; a run of consecutive
/// row nodes must line up with consecutive nodes of the other front (in reverse
/// order, because two facing fronts run in opposite directions).
const SEAM_REACH: f64 = 1.5;
/// Rounds of rows before the residual closure takes over.
const MAX_ROW_ROUNDS: usize = 32;

/// Options for the PAVING route.
///
/// Every field is a tuning choice; [`Default`] reproduces the constants the
/// route was measured with. Lengths are multiples of the element size `h`
/// (`Sizing::size`) because the row generator has no size field.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Options {
    /// Distance from a sharp corner (a turn whose magnitude exceeds 30 degrees)
    /// to the first sampled boundary node, as a multiple of `h`. Must be `>= 0`;
    /// default 2.4.
    ///
    /// The clearance is what gives the row generator room to keep more than one
    /// corner row alive; 1.25 also works but stops after one row, and a node
    /// exactly `h` from the corner collapses the corner quads.
    pub corner_margin: f64,

    /// Smallest front edge a row may produce, as a fraction of `h`. Must be
    /// `> 0`; default 0.08. A projected front edge shorter than this refuses the
    /// row: the matching quad would be a sliver the smoother cannot recover.
    pub min_edge_fraction: f64,

    /// Clearance a new front must keep from every other live front, as a
    /// multiple of `h`. Must be `> 0`; default 1.2. Closer than this the row is
    /// seamed onto the met front when `seam` is set, otherwise refused; the gap
    /// keeps two unrelated fronts from crossing.
    pub front_gap: f64,

    /// Upper bound on the outer row sweeps, one sweep visiting every live loop
    /// smallest perimeter first. Unit: sweeps, not rows. Default 32; each sweep
    /// may commit one row per live loop, so the measured [`Stats::row_rounds`]
    /// exceeds this value (33 on the 0.6 mm part) even before the residual
    /// closure takes over.
    pub max_row_rounds: usize,

    /// Seam a row onto the front it meets instead of refusing it
    /// (Blacker & Stephenson 1991, §2.8). Default `true`. Turning it off
    /// reproduces the pre-seam route exactly: measured 4 mm plate-hole 128 row
    /// quads + 200 core quads, against 192 + 143 with the seam on.
    #[serde(default = "default_seam")]
    pub seam: bool,
}

/// Serde default for [`Options::seam`]: seaming is on.
fn default_seam() -> bool {
    true
}

impl Default for Options {
    /// The measured constants: corner margin 2.4, min edge 0.08, front gap 1.2,
    /// 32 row sweeps and the §2.8 seam on.
    fn default() -> Self {
        Self {
            corner_margin: CORNER_MARGIN,
            min_edge_fraction: MIN_EDGE,
            front_gap: FRONT_GAP,
            max_row_rounds: MAX_ROW_ROUNDS,
            seam: default_seam(),
        }
    }
}

/// How the quads of a run were produced.
///
/// The two quad counters partition the final mesh: the cleanup passes only move
/// nodes and swap diagonals (they never add, delete or change the kind of a
/// cell), so `row_quads + core_quads == mesh.cells.len()`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    /// Quads produced by paving rows, including the §2.5 four- and six-node
    /// front closures. Unit: quads.
    pub row_quads: usize,

    /// Quads produced by the residual Delaunay + triangle-matching fallback
    /// (or by its fan closure when the matching is obstructed). Unit: quads.
    pub core_quads: usize,

    /// Paving rows committed: one increment per committed row (`commit_row`),
    /// not one per [`Options::max_row_rounds`] sweep. Unit: rows.
    pub row_rounds: usize,

    /// Rows committed by seaming the projected row onto the front it met
    /// (§2.8). A subset of `row_rounds`; zero when `Options::seam` is off.
    pub seam_rows: usize,
}

impl Stats {
    /// Share of the mesh produced by paving rows, in percent.
    pub fn row_share(&self) -> f64 {
        let total = self.row_quads + self.core_quads;
        if total == 0 {
            0.0
        } else {
            100.0 * self.row_quads as f64 / total as f64
        }
    }
}

/// The live state of one paving run: storage, fronts and the counters that are
/// copied into [`Stats`] at the end.
struct Paving {
    /// Every node built so far, in creation order, `[x, y, 0]` in millimetres.
    points: Vec<Node>,

    /// Parallel to `points`: `true` for a node sampled on the permanent input
    /// boundary, `false` for a row, closure or Steiner node. The smoother must
    /// not move the former, or `verify::boundary_offset` would stop being zero.
    fixed: Vec<bool>,

    /// Live front loops as ordered node-id cycles. Material is on the left of
    /// every loop, so the outer loop is counter-clockwise and each hole is
    /// clockwise; empty loops are never stored.
    loops: Vec<Vec<usize>>,

    /// Quads committed so far, as counter-clockwise node ids of length four.
    quads: Vec<[usize; 4]>,

    /// Quads committed by rows, including small-front closure (§2.5).
    row_quads: usize,

    /// Quads committed by the residual matched core (`fill_residual`).
    core_quads: usize,

    /// Rows committed (one per `commit_row`, so not the sweep count).
    row_rounds: usize,

    /// Committed rows whose `merge_other` was set (§2.8 seam).
    seam_rows: usize,
}

impl Paving {
    /// Coordinates of node `i` as a 2-vector; the stored `z` is always zero.
    fn coord(&self, i: usize) -> [f64; 2] {
        [self.points[i][0], self.points[i][1]]
    }

    /// Append a point with its permanent-boundary flag and return the new node
    /// id. Nothing is ever removed, so a returned id stays valid for the run.
    fn push_point(&mut self, p: [f64; 2], fixed: bool) -> usize {
        self.points.push([p[0], p[1], 0.0]);
        self.fixed.push(fixed);
        self.points.len() - 1
    }
}

/// Mesh a planar domain with rows of quads (Blacker & Stephenson 1991).
///
/// Constant size only: `sizing.size` is used at every boundary segment and on
/// every front; anisotropic sizing is not implemented. Boundary sampling, row
/// generation, the §2.8 seam, the §2.5 small-front closure and the residual
/// matched core are all in [`mesh_with_stats_options`]. The result is all-quad;
/// an error is returned for a bad size, a degenerate domain, or a mesh the final
/// topology check rejects.
pub fn mesh(domain: &Domain, sizing: &Sizing) -> Result<Mesh, String> {
    mesh_with_options(domain, sizing, &Options::default())
}

/// Like [`mesh`], but with explicit [`Options`] (see [`mesh_with_stats_options`]
/// for what each stage does).
pub fn mesh_with_options(
    domain: &Domain,
    sizing: &Sizing,
    options: &Options,
) -> Result<Mesh, String> {
    mesh_with_stats_options(domain, sizing, options).map(|(mesh, _)| mesh)
}

/// Like [`mesh`], but also reports the row/core split in [`Stats`], so the
/// caller can measure how much of the mesh is actual paving rather than the
/// residual matched core.
pub fn mesh_with_stats(domain: &Domain, sizing: &Sizing) -> Result<(Mesh, Stats), String> {
    mesh_with_stats_options(domain, sizing, &Options::default())
}

/// The full PAVING pipeline: validate, sample the boundary, advance rows,
/// close four- and six-node fronts, seam or refuse contacts, close the residual
/// core, then smooth and clean up.
///
/// It returns the mesh together with [`Stats`]. Errors are `String`s for an
/// invalid size/domain, a boundary sampling failure, an unclosed residual, or a
/// final topology check that finds a crack or a non-manifold edge. The mesh is
/// all-quad on success.
pub fn mesh_with_stats_options(
    domain: &Domain,
    sizing: &Sizing,
    options: &Options,
) -> Result<(Mesh, Stats), String> {
    if !sizing.size.is_finite() || sizing.size <= 0.0 {
        return Err("the element size must be positive and finite".to_string());
    }
    if domain.outer.len() < 3 {
        return Err("the outer loop needs at least three points".to_string());
    }
    if domain.area() <= 0.0 {
        return Err("the domain has no area".to_string());
    }
    let h = sizing.size;
    let mut paving = sample(domain, h, options)?;

    // Row loop: advance the smallest front first, as §2.4 asks. A front whose
    // next row would be invalid is left for the residual closure.
    for _ in 0..options.max_row_rounds {
        let mut order: Vec<usize> = (0..paving.loops.len()).collect();
        order.sort_by(|&a, &b| {
            front_length(&paving, a)
                .partial_cmp(&front_length(&paving, b))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut advanced = false;
        for li in order {
            if li >= paving.loops.len() {
                continue;
            }
            if let Some(row) = next_row(&paving, li, domain, h, options) {
                let merged = row.merge_other.is_some();
                commit_row(&mut paving, li, row);
                advanced = true;
                if merged {
                    // The seam removed a loop, so the remaining indices of this
                    // round's order are stale. Recompute the order next round.
                    break;
                }
            }
        }
        if !advanced {
            break;
        }
    }

    // §2.5: small fronts close with the paper's simple templates.
    close_small_fronts(&mut paving);

    // Residual closure (documented deviation): constrained Delaunay plus
    // triangle pairing. Loops that are left after this must be empty.
    if paving.loops.iter().any(|loop_ids| !loop_ids.is_empty()) {
        fill_residual(&mut paving, domain, h)?;
    }
    let stats = Stats {
        row_quads: paving.row_quads,
        core_quads: paving.core_quads,
        row_rounds: paving.row_rounds,
        seam_rows: paving.seam_rows,
    };

    let cells: Vec<Cell> = paving.quads.iter().map(|q| Cell::quad4(*q)).collect();
    let mut mesh = Mesh::new(paving.points, cells);
    let topo = mesh.topology();
    if topo.cracks != 0 || topo.non_manifold != 0 {
        return Err(format!(
            "paving left {} crack(s) and {} non-manifold edge(s)",
            topo.cracks, topo.non_manifold
        ));
    }
    smooth_mesh(&mut mesh, domain, 30, h);

    // §2.11 cleanup of the completed mesh. The route is all-quad; this only
    // removes poor pairs left at the front corners by the matching closure.
    let quads: Vec<[usize; 4]> = mesh
        .cells
        .iter()
        .filter(|c| c.kind == crate::mesh::Kind::Quad4)
        .map(|c| [c.nodes[0], c.nodes[1], c.nodes[2], c.nodes[3]])
        .collect();
    let (p1, q1, _t, _swaps) = sun::repair_low_quads(&mesh.points, &quads, &[], 0.15, 8);
    let (p2, q2, _t, _moves) = sun::perturb_low_quads(&p1, &q1, &[], h, 100, Some(domain));
    mesh = Mesh::new(p2, q2.into_iter().map(Cell::quad4).collect());
    smooth_mesh(&mut mesh, domain, 20, h);
    // §2.11: reduce the number of irregular interior nodes by swapping the
    // shared edge of two quads for the other diagonal of their hexagon.
    valence_cleanup(&mut mesh, 24);
    // The swaps can make a quad thinner; a final node perturb recovers the
    // angle/Jacobian margin without changing the topology (so v4 is unchanged).
    let quads: Vec<[usize; 4]> = mesh
        .cells
        .iter()
        .map(|c| [c.nodes[0], c.nodes[1], c.nodes[2], c.nodes[3]])
        .collect();
    let (p3, q3, _t, _moves) =
        sun::perturb_low_quads(&mesh.points, &quads, &[], h, 100, Some(domain));
    mesh = Mesh::new(p3, q3.into_iter().map(Cell::quad4).collect());
    smooth_mesh(&mut mesh, domain, 10, h);
    let topo = mesh.topology();
    if mesh
        .cells
        .iter()
        .any(|c| c.kind != crate::mesh::Kind::Quad4)
    {
        return Err("the cleanup produced a non-quad cell".to_string());
    }
    if topo.cracks != 0 || topo.non_manifold != 0 || topo.inconsistent != 0 {
        return Err(format!(
            "the cleanup broke the mesh: {} crack(s), {} non-manifold, {} inconsistent",
            topo.cracks, topo.non_manifold, topo.inconsistent
        ));
    }
    Ok((mesh, stats))
}

// --------------------------------------------------------------------------- //
// boundary sampling
// --------------------------------------------------------------------------- //

/// Signed shoelace area of a closed point list: positive counter-clockwise.
/// Used to orient every loop so the material is on its left.
fn signed_area(points: &[[f64; 2]]) -> f64 {
    let mut sum = 0.0;
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        sum += a[0] * b[1] - b[0] * a[1];
    }
    sum / 2.0
}

/// Signed turn at vertex `i`, in radians: `atan2(cross, dot)` of the incoming
/// and outgoing edges. Positive is a left turn (convex on a CCW loop); a
/// magnitude above `CORNER_TURN` marks a geometric corner.
fn turn_at(points: &[[f64; 2]], i: usize) -> f64 {
    let n = points.len();
    let prev = points[(i + n - 1) % n];
    let cur = points[i];
    let next = points[(i + 1) % n];
    let e1 = [cur[0] - prev[0], cur[1] - prev[1]];
    let e2 = [next[0] - cur[0], next[1] - cur[1]];
    let cross = e1[0] * e2[1] - e1[1] * e2[0];
    let dot = e1[0] * e2[0] + e1[1] * e2[1];
    cross.atan2(dot)
}

/// Sample one closed loop at the requested size.
///
/// Every input vertex is kept (duplicates and a repeated closing point are
/// dropped) and the loop is oriented so its signed area has the sign of `ccw`.
/// Each segment reserves `options.corner_margin * h` at a sharp end, then gets
/// `round(span / h)` interior nodes placed at the centres of the reserved span,
/// or is kept whole when `span < 0.6 h`. A paving boundary needs an
/// even number of nodes, so an odd total adds one sample on the longest spanned
/// segment; that only refines that segment and never moves a corner node.
/// Returns at least four even nodes, or an error.
fn sample_loop(
    input: &[[f64; 2]],
    h: f64,
    ccw: bool,
    options: &Options,
) -> Result<Vec<[f64; 2]>, String> {
    let mut loop_points: Vec<[f64; 2]> = Vec::with_capacity(input.len());
    for &p in input {
        if !p[0].is_finite() || !p[1].is_finite() {
            return Err("a boundary coordinate is not finite".to_string());
        }
        if loop_points
            .last()
            .map(|q| (q[0] - p[0]).abs() < 1e-12 && (q[1] - p[1]).abs() < 1e-12)
            .unwrap_or(false)
        {
            continue;
        }
        loop_points.push(p);
    }
    while loop_points.len() > 1 {
        let a = loop_points[0];
        let b = loop_points[loop_points.len() - 1];
        if (a[0] - b[0]).abs() < 1e-12 && (a[1] - b[1]).abs() < 1e-12 {
            loop_points.pop();
        } else {
            break;
        }
    }
    if loop_points.len() < 3 {
        return Err("a boundary loop has fewer than three distinct points".to_string());
    }
    let area = signed_area(&loop_points);
    if (area > 0.0) != ccw {
        loop_points.reverse();
    }
    let n = loop_points.len();
    let sharp: Vec<bool> = (0..n)
        .map(|i| turn_at(&loop_points, i).abs() > CORNER_TURN)
        .collect();

    // Per segment interior sample counts and fractions.
    let mut parts = vec![0usize; n];
    let mut spans = vec![0.0f64; n];
    let mut starts = vec![0.0f64; n];
    for i in 0..n {
        let a = loop_points[i];
        let b = loop_points[(i + 1) % n];
        let length = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
        let start = if sharp[i] {
            options.corner_margin * h
        } else {
            0.0
        };
        let end = if sharp[(i + 1) % n] {
            options.corner_margin * h
        } else {
            0.0
        };
        let span = (length - start - end).max(0.0);
        let k = if span < 0.6 * h {
            0
        } else {
            (span / h).round().max(1.0) as usize
        };
        starts[i] = start;
        spans[i] = span;
        parts[i] = k;
    }
    // A paving boundary must contain an even number of nodes.
    let total = n + parts.iter().sum::<usize>();
    if total % 2 == 1 {
        // Add one sample on the longest segment; it only makes that segment
        // finer, never moves a corner node.
        let mut best = 0usize;
        let mut best_length = -1.0f64;
        for i in 0..n {
            let a = loop_points[i];
            let b = loop_points[(i + 1) % n];
            let length = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
            if spans[i] > 0.0 && length > best_length {
                best_length = length;
                best = i;
            }
        }
        parts[best] += 1;
    }

    let mut out = Vec::new();
    for i in 0..n {
        out.push(loop_points[i]);
        let a = loop_points[i];
        let b = loop_points[(i + 1) % n];
        let length = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
        if length <= 1e-12 || parts[i] == 0 {
            continue;
        }
        let k = parts[i] as f64;
        let span = spans[i];
        for j in 0..parts[i] {
            let t = (starts[i] + span * (j as f64 + 0.5) / k) / length;
            out.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
        }
    }
    if out.len() < 4 {
        return Err(format!(
            "sampling loop left {} nodes; at least four are needed",
            out.len()
        ));
    }
    if out.len() % 2 == 1 {
        return Err("internal sampling error: odd loop".to_string());
    }
    Ok(out)
}

/// Sample the outer loop counter-clockwise and every hole clockwise, append
/// their points as fixed nodes and return the initial one-loop-per-cycle
/// [`Paving`] state.
fn sample(domain: &Domain, h: f64, options: &Options) -> Result<Paving, String> {
    let mut paving = Paving {
        points: Vec::new(),
        fixed: Vec::new(),
        loops: Vec::new(),
        quads: Vec::new(),
        row_quads: 0,
        core_quads: 0,
        row_rounds: 0,
        seam_rows: 0,
    };
    let mut loops: Vec<Vec<[f64; 2]>> = Vec::new();
    loops.push(sample_loop(&domain.outer, h, true, options)?);
    for hole in &domain.holes {
        loops.push(sample_loop(hole, h, false, options)?);
    }
    for points in loops {
        let mut ids = Vec::with_capacity(points.len());
        for p in points {
            let id = paving.push_point(p, true);
            ids.push(id);
        }
        paving.loops.push(ids);
    }
    Ok(paving)
}

// --------------------------------------------------------------------------- //
// one row
// --------------------------------------------------------------------------- //

/// One validated row, ready to commit.
///
/// [`next_row`] and [`seam_row`] build a complete `Row` without mutating the
/// mesh; [`commit_row`] is the only place that appends. A candidate that fails
/// any check is dropped whole, so the partial mesh is always conforming.
struct Row {
    /// Node ids of the front left after the row: all fresh ids for a plain row,
    /// or the spliced loop for a seam row. Empty when a seam consumed both
    /// loops (`commit_row` then removes both).
    new_loop: Vec<usize>,

    /// Coordinates to append, in the order their predicted ids were assigned.
    /// A seam row appends only its unseamed nodes; merged nodes keep the met
    /// front's ids and are not duplicated.
    new_points: Vec<[f64; 2]>,

    /// New quads as counter-clockwise node ids. They may reference old front
    /// ids, predicted fresh ids, and (for a seam) ids of the met front.
    quads: Vec<[usize; 4]>,

    /// Set when the row was seamed onto loop `other`: that loop disappears and
    /// its unseamed part is merged into `new_loop`.
    merge_other: Option<usize>,
}

/// Perimeter of live loop `li` in millimetres, or `+inf` for a stale index so
/// the smallest-perimeter sort stays a total order.
fn front_length(paving: &Paving, li: usize) -> f64 {
    let Some(loop_ids) = paving.loops.get(li) else {
        return f64::INFINITY;
    };
    let mut length = 0.0;
    for i in 0..loop_ids.len() {
        let a = paving.coord(loop_ids[i]);
        let b = paving.coord(loop_ids[(i + 1) % loop_ids.len()]);
        length += ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
    }
    length
}

/// Inward mitered projection of `cur` between its two front neighbours.
///
/// The material is on the left of the directed front edges, so the left normal
/// of both adjacent edges points inward. The two offset lines meet on the
/// interior bisector at `h / sin(alpha / 2)`: the paper's row-side projection
/// with a constant size, and the row-corner projection when the angle is a
/// corner.
///
/// Returns `None` when an adjacent edge is degenerate or the interior angle
/// `alpha` is so acute that the miter would explode (`sin(alpha/2) < 0.2`, i.e.
/// `alpha < ~23 degrees`); the caller then refuses the whole row. The distance
/// is capped at `3 h` so a near-cusp cannot place a node far outside the domain.
/// `round_corner` adds `0.25 h` at a convex turn (see the comment at `extra`).
fn project(
    prev: [f64; 2],
    cur: [f64; 2],
    next: [f64; 2],
    h: f64,
    round_corner: bool,
) -> Option<[f64; 2]> {
    let e1 = [cur[0] - prev[0], cur[1] - prev[1]];
    let e2 = [next[0] - cur[0], next[1] - cur[1]];
    let l1 = (e1[0] * e1[0] + e1[1] * e1[1]).sqrt();
    let l2 = (e2[0] * e2[0] + e2[1] * e2[1]).sqrt();
    if l1 < 1e-12 || l2 < 1e-12 {
        return None;
    }
    let d1 = [e1[0] / l1, e1[1] / l1];
    let d2 = [e2[0] / l2, e2[1] / l2];
    let n1 = [-d1[1], d1[0]];
    let n2 = [-d2[1], d2[0]];
    let bis = [n1[0] + n2[0], n1[1] + n2[1]];
    let bis_len = (bis[0] * bis[0] + bis[1] * bis[1]).sqrt();
    if bis_len < 1e-9 {
        return None;
    }
    let cross = d1[0] * d2[1] - d1[1] * d2[0];
    let dot = d1[0] * d2[0] + d1[1] * d2[1];
    let turn = cross.atan2(dot);
    let alpha = std::f64::consts::PI - turn;
    let sin_half = (alpha / 2.0).sin().abs();
    if sin_half < 0.2 {
        return None;
    }
    // A sharp convex corner is projected *past* the miter, which rounds the
    // next front: the two quads around the old corner stay convex and the
    // residual no longer has a sharp fan of triangles at that corner. Reflex
    // corners keep the plain miter.
    let extra = if round_corner && turn > CORNER_TURN {
        0.25 * h
    } else {
        0.0
    };
    let distance = (h / sin_half + extra).min(3.0 * h);
    Some([
        cur[0] + bis[0] / bis_len * distance,
        cur[1] + bis[1] / bis_len * distance,
    ])
}

/// True only for a strict interior crossing of two open segments. Collinear or
/// endpoint-touching segments return `false`: adjacent edges of a simple loop
/// share an endpoint and must not be reported as a self-intersection.
fn segments_cross(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
    let cross = |p: [f64; 2], q: [f64; 2], r: [f64; 2]| {
        (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    };
    let d1 = cross(a, b, c);
    let d2 = cross(a, b, d);
    let d3 = cross(c, d, a);
    let d4 = cross(c, d, b);
    ((d1 > 1e-12 && d2 < -1e-12) || (d1 < -1e-12 && d2 > 1e-12))
        && ((d3 > 1e-12 && d4 < -1e-12) || (d3 < -1e-12 && d4 > 1e-12))
}

/// True when the closed polyline `ids` has no strict self-crossing. The O(n^2)
/// pair scan skips adjacent edges (including the wrap-around pair), which is
/// exact for a polygon with distinct vertices.
fn loop_is_simple(points: &[Node], ids: &[usize]) -> bool {
    let n = ids.len();
    if n < 3 {
        return false;
    }
    for i in 0..n {
        let a = [points[ids[i]][0], points[ids[i]][1]];
        let b = [points[ids[(i + 1) % n]][0], points[ids[(i + 1) % n]][1]];
        for j in (i + 1)..n {
            if j == i || (j + 1) % n == i || (i + 1) % n == j {
                continue;
            }
            let c = [points[ids[j]][0], points[ids[j]][1]];
            let d = [points[ids[(j + 1) % n]][0], points[ids[(j + 1) % n]][1]];
            if segments_cross(a, b, c, d) {
                return false;
            }
        }
    }
    true
}

/// Euclidean distance from `p` to the closed segment `a-b` (the projection is
/// clamped to the segment). Millimetres.
fn point_segment_distance(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let length2 = ab[0] * ab[0] + ab[1] * ab[1];
    if length2 <= 1e-18 {
        return ((p[0] - a[0]).powi(2) + (p[1] - a[1]).powi(2)).sqrt();
    }
    let t = (((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / length2).clamp(0.0, 1.0);
    let q = [a[0] + t * ab[0], a[1] + t * ab[1]];
    ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt()
}

/// Shortest distance from `p` to the polyline of live loop `li`, in
/// millimetres; `+inf` for a stale index.
fn loop_distance(paving: &Paving, p: [f64; 2], li: usize) -> f64 {
    let Some(ids) = paving.loops.get(li) else {
        return f64::INFINITY;
    };
    let mut best = f64::INFINITY;
    for i in 0..ids.len() {
        let a = paving.coord(ids[i]);
        let b = paving.coord(ids[(i + 1) % ids.len()]);
        best = best.min(point_segment_distance(p, a, b));
    }
    best
}

/// Row-acceptance gate for one candidate quad: every corner scaled Jacobian must
/// be finite and above 0.02, and every corner angle must lie strictly between 5
/// and 175 degrees. These floors are deliberately looser than the final gate
/// (SJ 0.2 / 30 degrees) but reject anything the cleaner cannot recover; a row
/// is refused whole rather than committing a quad that would stay a defect,
/// since the cleanup never deletes a cell.
fn quad_ok(points: &[Node], quad: [usize; 4]) -> bool {
    let cell = Cell::quad4(quad);
    let sj = quality::scaled_jacobian(&cell, points);
    if sj.iter().any(|&s| !s.is_finite() || s <= 0.02) {
        return false;
    }
    let angles = quality::quad_angles(&cell, points);
    angles
        .iter()
        .all(|&a| a.is_finite() && a > 5.0 && a < 175.0)
}

/// Build, but do not commit, the next row of loop `li`.
///
/// Every front node is projected once ([`project`] with corner rounding). The
/// candidate is rejected whole when any projection is invalid, leaves the
/// domain or comes within `0.05 h` of its boundary, produces an edge shorter
/// than `options.min_edge_fraction * h`, makes the new front non-simple, or
/// makes a quad fail [`quad_ok`]. A front closer than `options.front_gap * h`
/// to another front is offered to [`seam_row`] (§2.8) and otherwise refused.
/// Returns a fully validated [`Row`]; `li` keeps its old loop until commit.
/// A loop shorter than six nodes is left to the small-front closure.
fn next_row(paving: &Paving, li: usize, domain: &Domain, h: f64, options: &Options) -> Option<Row> {
    let old = paving.loops.get(li)?.clone();
    let n = old.len();
    if n < 6 {
        return None;
    }
    // One projection per front node; a single rejected projection refuses the
    // whole row, so a row can never be committed half-built.
    let mut new_points = Vec::with_capacity(n);
    for i in 0..n {
        let prev = paving.coord(old[(i + n - 1) % n]);
        let cur = paving.coord(old[i]);
        let next = paving.coord(old[(i + 1) % n]);
        new_points.push(project(prev, cur, next, h, true)?);
    }
    // The new front must be a simple loop, inside the domain and away from
    // every other front. Its node ids do not exist yet; build a scratch loop.
    if new_points.iter().any(|p| !domain.contains(*p)) {
        return None;
    }
    if new_points
        .iter()
        .any(|p| domain.distance_to_boundary(*p) < 0.05 * h)
    {
        return None;
    }
    for i in 0..n {
        let a = new_points[i];
        let b = new_points[(i + 1) % n];
        if ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt() < options.min_edge_fraction * h {
            return None;
        }
    }

    // Distance from the new front to the other fronts. A row that would run
    // into another front is no longer refused outright: §2.8 seams it onto the
    // front it meets when the two node chains line up. If the seam is not
    // sound the row is refused exactly as before.
    let mut closest: Option<(f64, usize)> = None;
    for other in 0..paving.loops.len() {
        if other == li {
            continue;
        }
        for p in &new_points {
            let d = loop_distance(paving, *p, other);
            if d < options.front_gap * h && closest.map(|(best, _)| d < best).unwrap_or(true) {
                closest = Some((d, other));
            }
        }
    }
    if let Some((_d, other)) = closest {
        if options.seam {
            if let Some(row) = seam_row(paving, li, other, &old, &new_points, h) {
                return Some(row);
            }
        }
        return None;
    }

    // Simple polygon test on the scratch coordinates.
    let scratch: Vec<Node> = new_points.iter().map(|p| [p[0], p[1], 0.0]).collect();
    let ids: Vec<usize> = (0..n).collect();
    if !loop_is_simple(&scratch, &ids) {
        return None;
    }

    let mut new_loop = Vec::with_capacity(n);
    for _ in &new_points {
        new_loop.push(paving.points.len() + new_loop.len());
    }
    let mut quads = Vec::with_capacity(n);
    for i in 0..n {
        quads.push([old[i], old[(i + 1) % n], new_loop[(i + 1) % n], new_loop[i]]);
    }
    // Check the quads on a scratch point list: old points plus projected ones.
    let mut scratch_points = paving.points.clone();
    for p in &new_points {
        scratch_points.push([p[0], p[1], 0.0]);
    }
    for q in &quads {
        if !quad_ok(&scratch_points, *q) {
            return None;
        }
    }
    Some(Row {
        new_loop,
        new_points,
        quads,
        merge_other: None,
    })
}

/// A run of row nodes that lines up with a run of nodes of the front it meets.
struct SeamRun {
    /// Index in the projected row (`0..n`) of the first node of the run.
    start: usize,

    /// Number of row nodes in the run; at least 2 (one node is not an edge), at
    /// most `min(n, m)`, and never a whole loop unless it is a whole loop on
    /// both sides (otherwise the unseamed remainder would not close).
    len: usize,

    /// Index in the met loop of the node matched to `start`. The run walks the
    /// met loop in reverse, one node per row node.
    other_start: usize,

    /// Worst node-to-node distance of the run, in multiples of `h`; at most
    /// `SEAM_REACH`. Used to prefer the tightest of two equally long runs.
    worst: f64,
}

/// Seam the projected row `new_points` (from loop `li`) onto loop `other`.
///
/// A run is accepted only when every row node in it is within `SEAM_REACH * h`
/// of its matched node. The measured 4 mm plate-hole is the case the seam was
/// added for: it raises that part's row share from 39 % to 57 %.
///
/// §2.8 performs a seam by combining two nodes into one. Here a *run* of
/// consecutive row nodes is matched to consecutive nodes of the met front (in
/// reverse, because two facing fronts run in opposite directions); each pair is
/// merged, the row quads are built on the met front's node ids, and the unseamed
/// parts of the two loops are spliced into one loop. Nothing is split: the
/// welded edges were front edges of `other`, so they become interior edges
/// shared by the last row of `other` and the new row. The candidate is checked
/// against the whole mesh (cracks, inversion, overlapping cells) before it is
/// returned, so a seam that would break the mesh is refused.
fn seam_row(
    paving: &Paving,
    li: usize,
    other: usize,
    old: &[usize],
    new_points: &[[f64; 2]],
    h: f64,
) -> Option<Row> {
    let n = old.len();
    let other_loop = paving.loops.get(other)?;
    let m = other_loop.len();
    if n < 2 || m < 2 || other == li {
        return None;
    }
    let reach = SEAM_REACH * h;
    // Nearest node of the met front for every row node (O(n m); the fronts here
    // are small). Only runs whose nodes are all within `reach` can seam.
    let mut nearest: Vec<(usize, f64)> = Vec::with_capacity(n);
    for p in new_points {
        let mut best = (usize::MAX, f64::INFINITY);
        for (j, &id) in other_loop.iter().enumerate() {
            let q = paving.coord(id);
            let d = ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt();
            if d < best.1 {
                best = (j, d);
            }
        }
        nearest.push(best);
    }
    // Longest run of row nodes whose nearest nodes are consecutive in reverse.
    let mut run: Option<SeamRun> = None;
    for start in 0..n {
        if nearest[start].1 > reach {
            continue;
        }
        let mut len = 1usize;
        let mut worst = nearest[start].1;
        while len < n && len < m {
            let i = (start + len) % n;
            let d = nearest[i].1;
            if d > reach {
                break;
            }
            let expect = (nearest[start].0 + m - len) % m;
            if nearest[i].0 != expect {
                break;
            }
            worst = worst.max(d);
            len += 1;
        }
        // A run shorter than two nodes is not a seam (it is one node, not an
        // edge). A run that swallows either whole loop cannot be spliced: the
        // unseamed remainder would not be closed.
        if len < 2 || len > n || len > m {
            continue;
        }
        if (len == n && len != m) || (len == m && len != n) {
            continue;
        }
        let better = match &run {
            None => true,
            Some(best) => len > best.len || (len == best.len && worst < best.worst),
        };
        if better {
            run = Some(SeamRun {
                start,
                len,
                other_start: nearest[start].0,
                worst,
            });
        }
    }
    let run = run?;

    // Merge the matched nodes and give the unseamed row nodes fresh ids.
    let base = paving.points.len();
    let mut new_ids = vec![usize::MAX; n];
    let mut fresh: Vec<[f64; 2]> = Vec::new();
    let closed = run.len == n && run.len == m;
    for i in 0..n {
        let offset = (i + n - run.start) % n;
        if offset < run.len {
            let j = (run.other_start + m - offset) % m;
            new_ids[i] = other_loop[j];
        } else {
            new_ids[i] = base + fresh.len();
            fresh.push(new_points[i]);
        }
    }
    let mut quads = Vec::with_capacity(n);
    for i in 0..n {
        quads.push([old[i], old[(i + 1) % n], new_ids[(i + 1) % n], new_ids[i]]);
    }

    // Spliced front: the met loop walked forward from the far end of the seam
    // to its near end, then the row's unseamed part walked forward.
    let mut merged: Vec<usize> = Vec::new();
    if !closed {
        let j_start = run.other_start;
        let j_end = (run.other_start + m - (run.len - 1)) % m;
        let mut j = j_start;
        loop {
            merged.push(other_loop[j]);
            if j == j_end {
                break;
            }
            j = (j + 1) % m;
        }
        for k in 0..(n - run.len) {
            let i = (run.start + run.len + k) % n;
            merged.push(new_ids[i]);
        }
    }
    if !closed && merged.len() < 3 {
        return None;
    }

    // Validate the candidate on the whole mesh before committing anything.
    let mut scratch_points = paving.points.clone();
    for p in &fresh {
        scratch_points.push([p[0], p[1], 0.0]);
    }
    for q in &quads {
        if !quad_ok(&scratch_points, *q) {
            return None;
        }
    }
    if !closed && !loop_is_simple(&scratch_points, &merged) {
        return None;
    }
    if !seam_is_sound(paving, &scratch_points, &quads) {
        return None;
    }
    Some(Row {
        new_loop: merged,
        new_points: fresh,
        quads,
        merge_other: Some(other),
    })
}

/// True when adding `quads` to the mesh leaves no crack, no inverted cell, no
/// pair of cells sharing more than one edge, and no geometric overlap between
/// the new quads and the cells they touch.
fn seam_is_sound(paving: &Paving, points: &[Node], quads: &[[usize; 4]]) -> bool {
    let mut cells: Vec<Cell> = paving.quads.iter().map(|q| Cell::quad4(*q)).collect();
    cells.extend(quads.iter().map(|q| Cell::quad4(*q)));
    let mesh = Mesh::new(points.to_vec(), cells);
    let topo = mesh.topology();
    if topo.cracks != 0 || topo.non_manifold != 0 || topo.inconsistent != 0 {
        return false;
    }
    let gate = crate::verify::gate(&mesh);
    if gate.inverted != 0 || gate.overlapping != 0 {
        return false;
    }
    // The gate's overlap test is topological (two cells sharing two edges), so
    // also test the new quads geometrically against the existing cells that
    // share a node with them, and against each other.
    let mut incident: HashMap<usize, Vec<usize>> = HashMap::new();
    for (qi, q) in paving.quads.iter().enumerate() {
        for &v in q {
            incident.entry(v).or_default().push(qi);
        }
    }
    for q in quads {
        let mut candidates: Vec<usize> = Vec::new();
        for &v in q {
            if let Some(list) = incident.get(&v) {
                candidates.extend_from_slice(list);
            }
        }
        candidates.sort_unstable();
        candidates.dedup();
        for qi in candidates {
            if let Some(other) = paving.quads.get(qi) {
                if quads_overlap(points, *q, *other) {
                    return false;
                }
            }
        }
    }
    for i in 0..quads.len() {
        for j in (i + 1)..quads.len() {
            if !bboxes_apart(points, quads[i], quads[j])
                && quads_overlap(points, quads[i], quads[j])
            {
                return false;
            }
        }
    }
    true
}

/// Axis-aligned bounding box of a quad as `[min_x, min_y, max_x, max_y]`, in
/// millimetres. Used only to skip the separating-axis test for distant pairs.
fn quad_bbox(points: &[Node], q: [usize; 4]) -> [f64; 4] {
    let mut b = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for &v in &q {
        b[0] = b[0].min(points[v][0]);
        b[1] = b[1].min(points[v][1]);
        b[2] = b[2].max(points[v][0]);
        b[3] = b[3].max(points[v][1]);
    }
    b
}

/// True when the two quads' bounding boxes are disjoint (touching is enough to
/// say they cannot overlap).
fn bboxes_apart(points: &[Node], a: [usize; 4], b: [usize; 4]) -> bool {
    let (p, q) = (quad_bbox(points, a), quad_bbox(points, b));
    p[2] <= q[0] || q[2] <= p[0] || p[3] <= q[1] || q[3] <= p[1]
}

/// Separating-axis test for two convex quads: true when their interiors
/// overlap. Quads that only share an edge or a vertex do not overlap.
fn quads_overlap(points: &[Node], a: [usize; 4], b: [usize; 4]) -> bool {
    let project = |q: [usize; 4], axis: [f64; 2]| -> (f64, f64) {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &v in &q {
            let t = points[v][0] * axis[0] + points[v][1] * axis[1];
            lo = lo.min(t);
            hi = hi.max(t);
        }
        (lo, hi)
    };
    for q in [a, b] {
        for i in 0..4 {
            let p = points[q[i]];
            let r = points[q[(i + 1) % 4]];
            let axis = [-(r[1] - p[1]), r[0] - p[0]];
            if axis[0].abs() < 1e-15 && axis[1].abs() < 1e-15 {
                continue;
            }
            let (alo, ahi) = project(a, axis);
            let (blo, bhi) = project(b, axis);
            let eps = 1e-12 * (ahi.abs() + blo.abs() + 1.0);
            if ahi <= blo + eps || bhi <= alo + eps {
                return false;
            }
        }
    }
    true
}

/// Commit a row that [`next_row`] or [`seam_row`] has already validated:
/// append its fresh points and quads, update the [`Stats`] counters and
/// replace (or splice) the live loop. A seam removes loop `other`; when the row
/// consumed both loops (`new_loop` empty) both are removed. Every index in
/// `row.quads` is valid before this call, so the mesh stays conforming.
fn commit_row(paving: &mut Paving, li: usize, row: Row) {
    for p in row.new_points {
        paving.push_point(p, false);
    }
    paving.row_quads += row.quads.len();
    paving.row_rounds += 1;
    if row.merge_other.is_some() {
        paving.seam_rows += 1;
    }
    paving.quads.extend(row.quads);
    let new_loop = row.new_loop;
    match row.merge_other {
        None => {
            if let Some(slot) = paving.loops.get_mut(li) {
                *slot = new_loop;
            }
        }
        Some(other) => {
            if new_loop.is_empty() {
                // Both fronts closed: the seam consumed all of both loops.
                paving.loops.remove(other);
                let li = if other < li { li - 1 } else { li };
                paving.loops.remove(li);
            } else {
                if let Some(slot) = paving.loops.get_mut(li) {
                    *slot = new_loop;
                }
                paving.loops.remove(other);
            }
        }
    }
}

// --------------------------------------------------------------------------- //
// small-front closure (§2.5)
// --------------------------------------------------------------------------- //

/// §2.5 small-front closure. A four-node front becomes one quad (winding
/// normalised, then checked by [`quad_ok`]); a six-node front becomes the best
/// valid two-quad fan pairing ([`best_six_closure`]). Both count as row quads
/// because they are the paper's paving closure, not the residual core. Other
/// sizes, or a template that fails its check, are left for `fill_residual`.
fn close_small_fronts(paving: &mut Paving) {
    let mut li = 0;
    while li < paving.loops.len() {
        let n = paving.loops[li].len();
        if n == 4 {
            let ids = paving.loops[li].clone();
            let mut quad = [ids[0], ids[1], ids[2], ids[3]];
            if signed_quad_area(&paving.points, quad) < 0.0 {
                quad.reverse();
            }
            if quad_ok(&paving.points, quad) {
                paving.quads.push(quad);
                paving.row_quads += 1;
                paving.loops.remove(li);
                continue;
            }
        } else if n == 6 {
            if let Some((a, b)) = best_six_closure(&paving.points, &paving.loops[li]) {
                paving.quads.push(a);
                paving.quads.push(b);
                paving.row_quads += 2;
                paving.loops.remove(li);
                continue;
            }
        }
        li += 1;
    }
}

/// Signed shoelace area of a quad, positive when counter-clockwise. Millimetres
/// squared.
fn signed_quad_area(points: &[Node], q: [usize; 4]) -> f64 {
    let mut sum = 0.0;
    for i in 0..4 {
        let a = points[q[i]];
        let b = points[q[(i + 1) % 4]];
        sum += a[0] * b[1] - b[0] * a[1];
    }
    sum / 2.0
}

/// Close a 6-node front with two quads. There are three ways to pair the six
/// consecutive triangles of a fan; each is tried and the best valid one kept.
fn best_six_closure(points: &[Node], ids: &[usize]) -> Option<([usize; 4], [usize; 4])> {
    if ids.len() != 6 {
        return None;
    }
    // The three shifts cover the three ways to pair the six fan triangles; the
    // shift decides which two triangles are merged into the first quad.
    let mut best: Option<([usize; 4], [usize; 4], f64)> = None;
    for shift in 0..3 {
        let v: Vec<usize> = (0..6).map(|i| ids[(i + shift) % 6]).collect();
        let mut q1 = [v[0], v[1], v[2], v[3]];
        let mut q2 = [v[0], v[3], v[4], v[5]];
        if signed_quad_area(points, q1) < 0.0 {
            q1.reverse();
        }
        if signed_quad_area(points, q2) < 0.0 {
            q2.reverse();
        }
        if !quad_ok(points, q1) || !quad_ok(points, q2) {
            continue;
        }
        let s1 = quality::scaled_jacobian(&Cell::quad4(q1), points)
            .into_iter()
            .fold(f64::INFINITY, f64::min);
        let s2 = quality::scaled_jacobian(&Cell::quad4(q2), points)
            .into_iter()
            .fold(f64::INFINITY, f64::min);
        let score = s1.min(s2);
        if best.as_ref().map(|b| score > b.2).unwrap_or(true) {
            best = Some((q1, q2, score));
        }
    }
    best.map(|(a, b, _)| (a, b))
}

// --------------------------------------------------------------------------- //
// residual closure (deviation from §2.10)
// --------------------------------------------------------------------------- //

/// Even-odd ray-casting point-in-polygon test over every live loop: true when
/// `p` lies in the material (inside the outer loop and outside every hole).
fn point_in_loops(paving: &Paving, p: [f64; 2]) -> bool {
    let mut inside = false;
    for ids in &paving.loops {
        let n = ids.len();
        for i in 0..n {
            let a = paving.coord(ids[i]);
            let b = paving.coord(ids[(i + 1) % n]);
            if (a[1] > p[1]) != (b[1] > p[1]) {
                let t = (p[1] - a[1]) / (b[1] - a[1]);
                if p[0] < a[0] + t * (b[0] - a[0]) {
                    inside = !inside;
                }
            }
        }
    }
    inside
}

/// Shortest distance from `p` to the union of all live front polylines, in
/// millimetres. Used to keep residual seed points and split points clear of the
/// fronts.
fn distance_to_fronts(paving: &Paving, p: [f64; 2]) -> f64 {
    let mut best = f64::INFINITY;
    for ids in &paving.loops {
        let n = ids.len();
        for i in 0..n {
            let a = paving.coord(ids[i]);
            let b = paving.coord(ids[(i + 1) % n]);
            best = best.min(point_segment_distance(p, a, b));
        }
    }
    best
}

/// Close what paving left, by the documented deviation from the paper's
/// §2.10 perimeter connection: seed a jittered square lattice inside the live
/// loops, constrain the front edges, and match triangles into quads.
///
/// The grid is at `h` with `0.55 h` clearance from every front and node, so a
/// thin residual simply gets no interior point and is triangulated from its two
/// boundary chains. [`match_residual`] is tried up to five times; each failure
/// splits the longest interior edge of every unpaired triangle
/// ([`add_edge_points`]). If matching still fails and exactly one simply
/// connected front remains, [`fan_fill`] closes it unconditionally; otherwise
/// the run returns an error. This fallback, not further paving rows, is why the
/// route finishes on every corpus part.
fn fill_residual(paving: &mut Paving, domain: &Domain, h: f64) -> Result<(), String> {
    // Interior grid points, at the size, away from the front polylines. A thin
    // residual receives none and is triangulated from its two boundary chains.
    let mut bbox = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for ids in &paving.loops {
        for &id in ids {
            let p = paving.coord(id);
            bbox[0] = bbox[0].min(p[0]);
            bbox[1] = bbox[1].min(p[1]);
            bbox[2] = bbox[2].max(p[0]);
            bbox[3] = bbox[3].max(p[1]);
        }
    }
    if !bbox.iter().all(|v| v.is_finite()) {
        return Err("the residual region has no finite extent".to_string());
    }
    let mut extra: Vec<usize> = Vec::new();
    let hs = h.max(1e-6);
    let mut node_count = paving.points.len();
    let nx = ((bbox[2] - bbox[0]) / hs).ceil() as i64 + 2;
    let ny = ((bbox[3] - bbox[1]) / hs).ceil() as i64 + 2;
    for i in 0..nx {
        for j in 0..ny {
            let mut x = bbox[0] + (i as f64 + 0.5) * hs;
            let mut y = bbox[1] + (j as f64 + 0.5) * hs;
            // Deterministic jitter, so a square grid does not make every
            // Delaunay quad ambiguous.
            let hash = ((i.wrapping_mul(73_856_093) ^ j.wrapping_mul(19_349_663)) as u64)
                .wrapping_mul(2_654_435_761);
            x += (((hash & 0xffff) as f64 / 65_535.0) - 0.5) * 0.12 * hs;
            y += ((((hash >> 16) & 0xffff) as f64 / 65_535.0) - 0.5) * 0.12 * hs;
            let p = [x, y];
            if !domain.contains(p) || !point_in_loops(paving, p) {
                continue;
            }
            if distance_to_fronts(paving, p) < 0.55 * hs {
                continue;
            }
            let mut too_close = false;
            for ids in &paving.loops {
                for &id in ids {
                    let q = paving.coord(id);
                    if (q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2) < (0.55 * hs).powi(2) {
                        too_close = true;
                        break;
                    }
                }
                if too_close {
                    break;
                }
            }
            if too_close {
                continue;
            }
            paving.points.push([x, y, 0.0]);
            paving.fixed.push(false);
            extra.push(node_count);
            node_count += 1;
        }
    }

    // Pairing is used as the residual closure. It pairs every triangle on
    // the corpus; a triangulation can have a matching obstruction, so a few
    // edge-split retries are tried and, if the residual is simply connected,
    // the fan template closes it unconditionally.
    let mut unpaired: Vec<[usize; 3]> = Vec::new();
    for round in 0..5 {
        match match_residual(paving, &extra) {
            Ok(()) => {
                paving.loops.clear();
                return Ok(());
            }
            Err(left) => unpaired = left,
        }
        if round == 4 {
            break;
        }
        add_edge_points(paving, domain, h, &unpaired, &mut extra);
    }

    if paving.loops.len() == 1 {
        fan_fill(paving, domain)?;
        return Ok(());
    }
    Err(format!(
        "could not pair {} residual triangle(s) into quads",
        unpaired.len()
    ))
}

/// One matching attempt. On success the quads are appended and the front edges
/// are checked; on failure nothing is committed and the unpaired triangles are
/// returned so the caller can split an edge and retry.
fn match_residual(paving: &mut Paving, extra: &[usize]) -> Result<(), Vec<[usize; 3]>> {
    let tris = match triangulate_residual(paving, extra) {
        Ok(tris) => tris,
        Err(_) => return Err(Vec::new()),
    };
    if tris.is_empty() {
        return Err(Vec::new());
    }
    let guard = PairGuard {
        min_scaled_jacobian: -2.0,
        min_angle_deg: 0.0,
        max_angle_deg: 180.0,
    };
    let (quads, unpaired) = pair_triangles_with(&tris, &paving.points, guard);
    if !unpaired.is_empty() {
        return Err(unpaired);
    }
    if !front_edges_covered(paving, &tris) {
        return Err(Vec::new());
    }
    paving.core_quads += quads.len();
    paving.quads.extend(quads);
    Ok(())
}

/// Split the longest interior edge of each unpaired triangle. This changes the
/// dual graph where the matching is obstructed.
fn add_edge_points(
    paving: &mut Paving,
    domain: &Domain,
    h: f64,
    unpaired: &[[usize; 3]],
    extra: &mut Vec<usize>,
) {
    if unpaired.is_empty() {
        return;
    }
    let hs = h.max(1e-6);
    let mut front_edges: std::collections::HashSet<(usize, usize)> =
        std::collections::HashSet::new();
    for ids in &paving.loops {
        for i in 0..ids.len() {
            let a = ids[i];
            let b = ids[(i + 1) % ids.len()];
            front_edges.insert((a.min(b), a.max(b)));
        }
    }
    for tri in unpaired {
        let mut best: Option<((usize, usize), f64)> = None;
        for i in 0..3 {
            let a = tri[i];
            let b = tri[(i + 1) % 3];
            let key = (a.min(b), a.max(b));
            if front_edges.contains(&key) {
                continue;
            }
            let pa = paving.points[a];
            let pb = paving.points[b];
            let length = ((pb[0] - pa[0]).powi(2) + (pb[1] - pa[1]).powi(2)).sqrt();
            if best.as_ref().map(|e| length > e.1).unwrap_or(true) {
                best = Some((key, length));
            }
        }
        let Some(((a, b), _)) = best else { continue };
        let pa = paving.points[a];
        let pb = paving.points[b];
        let p = [(pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5];
        if !domain.contains(p) || !point_in_loops(paving, p) {
            continue;
        }
        if distance_to_fronts(paving, p) < 0.1 * hs {
            continue;
        }
        if paving
            .points
            .iter()
            .any(|q| (q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2) < (0.1 * hs).powi(2))
        {
            continue;
        }
        paving.points.push([p[0], p[1], 0.0]);
        paving.fixed.push(false);
        extra.push(paving.points.len() - 1);
    }
}

/// Unconditional all-quad closure of a simply connected front: a fan from an
/// interior point. Every pair of consecutive fan triangles becomes one quad,
/// so an even front always closes. Used only when matching is obstructed.
fn fan_fill(paving: &mut Paving, domain: &Domain) -> Result<(), String> {
    let Some(ids) = paving.loops.first().cloned() else {
        return Ok(());
    };
    let n = ids.len();
    if n < 4 || n % 2 != 0 {
        return Err("a residual front cannot be fanned".to_string());
    }
    // Pick the interior point with the largest clearance from the front.
    let mut bbox = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for &id in &ids {
        let p = paving.coord(id);
        bbox[0] = bbox[0].min(p[0]);
        bbox[1] = bbox[1].min(p[1]);
        bbox[2] = bbox[2].max(p[0]);
        bbox[3] = bbox[3].max(p[1]);
    }
    let mut best = [0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])];
    let mut best_clear = f64::NEG_INFINITY;
    let steps = 12usize;
    for i in 0..=steps {
        for j in 0..=steps {
            let p = [
                bbox[0] + (bbox[2] - bbox[0]) * i as f64 / steps as f64,
                bbox[1] + (bbox[3] - bbox[1]) * j as f64 / steps as f64,
            ];
            if !domain.contains(p) || !point_in_loops(paving, p) {
                continue;
            }
            let clear = distance_to_fronts(paving, p);
            if clear > best_clear {
                best_clear = clear;
                best = p;
            }
        }
    }
    if !best_clear.is_finite() || best_clear <= 0.0 {
        return Err("a residual front has no interior point".to_string());
    }
    let center = paving.push_point(best, false);
    let mut quads = Vec::with_capacity(n / 2);
    for k in 0..(n / 2) {
        let a = ids[(2 * k) % n];
        let b = ids[(2 * k + 1) % n];
        let c = ids[(2 * k + 2) % n];
        let mut quad = [center, a, b, c];
        if signed_quad_area(&paving.points, quad) < 0.0 {
            quad.reverse();
        }
        if !quad_ok(&paving.points, quad) {
            return Err("the fan closure produced a degenerate quad".to_string());
        }
        quads.push(quad);
    }
    paving.core_quads += quads.len();
    paving.quads.extend(quads);
    paving.loops.clear();
    Ok(())
}

/// Constrained Delaunay triangulation of the residual region: every live front
/// node and front edge, plus the interior `extra` points. Faces whose centroid
/// is outside the live loops are dropped (a hole is a constraint, so its inside
/// is triangulated by `spade` too). Returns counter-clockwise triangles with
/// the paving node ids, or an error when a point or constraint cannot be
/// inserted.
fn triangulate_residual(paving: &Paving, extra: &[usize]) -> Result<Vec<[usize; 3]>, String> {
    let mut cdt = ConstrainedDelaunayTriangulation::<Point2<f64>>::new();
    let mut present: HashMap<(u64, u64), ()> = HashMap::new();
    let mut insert = |cdt: &mut ConstrainedDelaunayTriangulation<Point2<f64>>,
                      p: [f64; 2]|
     -> Result<(), String> {
        let key = (p[0].to_bits(), p[1].to_bits());
        if present.insert(key, ()).is_none() {
            cdt.insert(Point2::new(p[0], p[1]))
                .map_err(|_| "a residual point could not be inserted".to_string())?;
        }
        Ok(())
    };
    for ids in &paving.loops {
        for &id in ids {
            insert(&mut cdt, paving.coord(id))?;
        }
        for i in 0..ids.len() {
            let a = paving.coord(ids[i]);
            let b = paving.coord(ids[(i + 1) % ids.len()]);
            cdt.add_constraint_edge(Point2::new(a[0], a[1]), Point2::new(b[0], b[1]))
                .map_err(|_| "a residual front edge could not be constrained".to_string())?;
        }
    }
    for &id in extra {
        insert(&mut cdt, paving.coord(id))?;
    }
    let mut index_of: HashMap<(u64, u64), usize> = HashMap::new();
    for ids in &paving.loops {
        for &id in ids {
            let p = paving.coord(id);
            index_of.insert((p[0].to_bits(), p[1].to_bits()), id);
        }
    }
    for &id in extra {
        let p = paving.coord(id);
        index_of.insert((p[0].to_bits(), p[1].to_bits()), id);
    }
    let mut tris = Vec::new();
    for face in cdt.inner_faces() {
        let handles = face.vertices();
        let mut ids = [0usize; 3];
        let mut centroid = [0.0f64; 2];
        let mut ok = true;
        for k in 0..3 {
            let p = handles[k].position();
            let key = (p.x.to_bits(), p.y.to_bits());
            match index_of.get(&key) {
                Some(&id) => ids[k] = id,
                None => {
                    ok = false;
                    break;
                }
            }
            centroid[0] += p.x / 3.0;
            centroid[1] += p.y / 3.0;
        }
        if !ok || !point_in_loops(paving, centroid) {
            continue;
        }
        let a = paving.points[ids[0]];
        let b = paving.points[ids[1]];
        let c = paving.points[ids[2]];
        let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
        if area <= 1e-14 {
            continue;
        }
        if area > 0.0 {
            tris.push([ids[0], ids[1], ids[2]]);
        } else {
            tris.push([ids[0], ids[2], ids[1]]);
        }
    }
    Ok(tris)
}

/// True when every live front edge appears exactly once among the residual
/// triangles. A constrained boundary edge belongs to one face; zero or two
/// occurrences means the CDT lost or duplicated a front edge and the pairing
/// would leave a crack.
fn front_edges_covered(paving: &Paving, tris: &[[usize; 3]]) -> bool {
    let mut counts: HashMap<(usize, usize), usize> = HashMap::new();
    for t in tris {
        for i in 0..3 {
            let a = t[i];
            let b = t[(i + 1) % 3];
            *counts.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    for ids in &paving.loops {
        for i in 0..ids.len() {
            let a = ids[i];
            let b = ids[(i + 1) % ids.len()];
            if counts.get(&(a.min(b), a.max(b))).copied().unwrap_or(0) != 1 {
                return false;
            }
        }
    }
    true
}

/// Swap the shared edge of two quads when it reduces the interior irregularity
/// `sum |valence - 4|` and both new quads stay inside the gate.
///
/// Two quads sharing an edge are a hexagon. Replacing the shared edge with one
/// of the hexagon's other diagonals keeps two quads, changes four valences and
/// does not touch the boundary. This is the §2.11 cleanup that reduces irregular
/// nodes; it is purely topological.
fn valence_cleanup(mesh: &mut Mesh, max_rounds: usize) {
    for _ in 0..max_rounds {
        let n = mesh.points.len();
        let mut edge_owners: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for (id, cell) in mesh.cells.iter().enumerate() {
            if cell.kind != crate::mesh::Kind::Quad4 || cell.nodes.len() != 4 {
                continue;
            }
            for i in 0..4 {
                let a = cell.nodes[i];
                let b = cell.nodes[(i + 1) % 4];
                edge_owners
                    .entry((a.min(b), a.max(b)))
                    .or_default()
                    .push(id);
            }
        }
        let mut boundary = vec![false; n];
        for (&(a, b), owners) in &edge_owners {
            if owners.len() == 1 {
                if a < n {
                    boundary[a] = true;
                }
                if b < n {
                    boundary[b] = true;
                }
            }
        }
        let mut valence: Vec<i32> = vec![0; n];
        for cell in &mesh.cells {
            if cell.kind != crate::mesh::Kind::Quad4 || cell.nodes.len() != 4 {
                continue;
            }
            for &v in &cell.nodes {
                if v < n {
                    valence[v] += 1;
                }
            }
        }
        let mut edges: Vec<(usize, usize)> = edge_owners
            .iter()
            .filter(|(_, owners)| owners.len() == 2)
            .map(|(&(a, b), _)| (a, b))
            .collect();
        edges.sort_unstable();

        let interior_valence = |v: usize| -> i32 {
            if boundary.get(v).copied().unwrap_or(true) {
                4
            } else {
                valence.get(v).copied().unwrap_or(4)
            }
        };

        let mut best: Option<(usize, usize, [usize; 4], [usize; 4], i32)> = None;
        for &(a, b) in &edges {
            let owners = &edge_owners[&(a, b)];
            let Some(q1) = quad_ccw(&mesh.points, &mesh.cells[owners[0]].nodes) else {
                continue;
            };
            let Some(q2) = quad_ccw(&mesh.points, &mesh.cells[owners[1]].nodes) else {
                continue;
            };
            let Some((u, v, p, q, r, s)) = shared_hexagon(q1, q2) else {
                continue;
            };
            let old_area =
                signed_quad_area(&mesh.points, q1).abs() + signed_quad_area(&mesh.points, q2).abs();
            for case in 0..2 {
                let (qa, qb) = if case == 0 {
                    ([p, q, u, r], [r, s, v, p])
                } else {
                    ([q, u, r, s], [s, v, p, q])
                };
                let (g1, g2) = if case == 0 { (p, r) } else { (q, s) };
                let improves = interior_valence(u) > 4
                    || interior_valence(v) > 4
                    || interior_valence(g1) < 4
                    || interior_valence(g2) < 4;
                if !improves {
                    continue;
                }
                let area = signed_quad_area(&mesh.points, qa).abs()
                    + signed_quad_area(&mesh.points, qb).abs();
                if (area - old_area).abs() > 1e-8 * old_area.max(1.0) {
                    continue;
                }
                if !quad_gate_ok(&mesh.points, qa) || !quad_gate_ok(&mesh.points, qb) {
                    continue;
                }
                let old_cost = valence_cost(&valence, &boundary, u, v, g1, g2, false);
                let new_cost = valence_cost(&valence, &boundary, u, v, g1, g2, true);
                let gain = old_cost - new_cost;
                if gain <= 0 {
                    continue;
                }
                if best
                    .as_ref()
                    .map(|candidate| gain > candidate.4)
                    .unwrap_or(true)
                {
                    best = Some((owners[0], owners[1], qa, qb, gain));
                }
            }
        }
        let Some((first, second, qa, qb, _gain)) = best else {
            break;
        };
        mesh.cells[first] = Cell::quad4(qa);
        mesh.cells[second] = Cell::quad4(qb);
    }
}

/// Four node ids in counter-clockwise order, or `None` for a cell that is not a
/// quad or has (near) zero area. Reversing does not change the cycle.
fn quad_ccw(points: &[Node], nodes: &[usize]) -> Option<[usize; 4]> {
    if nodes.len() != 4 {
        return None;
    }
    let mut q = [nodes[0], nodes[1], nodes[2], nodes[3]];
    let area = signed_quad_area(points, q);
    if area < 0.0 {
        q.reverse();
    }
    if signed_quad_area(points, q).abs() < 1e-14 {
        return None;
    }
    Some(q)
}

/// Rotate two CCW quads to `[u,v,p,q]` and `[v,u,r,s]`.
fn shared_hexagon(
    q1: [usize; 4],
    q2: [usize; 4],
) -> Option<(usize, usize, usize, usize, usize, usize)> {
    for i in 0..4 {
        let u = q1[i];
        let v = q1[(i + 1) % 4];
        for j in 0..4 {
            if q2[j] == v && q2[(j + 1) % 4] == u {
                let p = q1[(i + 2) % 4];
                let q = q1[(i + 3) % 4];
                let r = q2[(j + 2) % 4];
                let s = q2[(j + 3) % 4];
                return Some((u, v, p, q, r, s));
            }
        }
    }
    None
}

/// Cleanup gate for a swapped quad: positive area, every corner scaled
/// Jacobian at least 0.3 and every angle in `35..=160` degrees. This is
/// stricter than the final 0.2 / 30 degree gate because the following node
/// perturb may still shave the margin.
fn quad_gate_ok(points: &[Node], q: [usize; 4]) -> bool {
    if signed_quad_area(points, q) <= 1e-12 {
        return false;
    }
    let cell = Cell::quad4(q);
    let sj = quality::scaled_jacobian(&cell, points);
    if sj.iter().any(|&v| !v.is_finite() || v < 0.3) {
        return false;
    }
    let angles = quality::quad_angles(&cell, points);
    angles
        .iter()
        .all(|&a| a.is_finite() && (35.0..=160.0).contains(&a))
}

/// Interior irregularity `sum |valence - 4|` over the four free corners
/// (`u`, `v`, `g1`, `g2`), with boundary nodes excluded. With `new` the swap
/// deltas are applied: `u` and `v` lose one incident quad, `g1` and `g2` gain
/// one. Used as the exact gain test for a candidate edge swap.
fn valence_cost(
    valence: &[i32],
    boundary: &[bool],
    u: usize,
    v: usize,
    g1: usize,
    g2: usize,
    new: bool,
) -> i32 {
    let mut cost = 0;
    for (k, &node) in [u, v, g1, g2].iter().enumerate() {
        if boundary.get(node).copied().unwrap_or(true) {
            continue;
        }
        let old = valence.get(node).copied().unwrap_or(4);
        let delta = if !new {
            0
        } else if k < 2 {
            -1
        } else {
            1
        };
        cost += (old + delta - 4).abs();
    }
    cost
}

// --------------------------------------------------------------------------- //
// tests
// --------------------------------------------------------------------------- //

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::mesh::Kind;
    use crate::verify;

    fn all_quads(mesh: &Mesh) -> bool {
        mesh.cells.iter().all(|c| c.kind == Kind::Quad4)
    }

    #[test]
    fn a_square_is_all_quad_and_passes_the_gate() {
        let domain = Domain::rectangle(10.0, 10.0).unwrap();
        let mesh = mesh(&domain, &Sizing::uniform(1.0).unwrap()).unwrap();
        assert!(all_quads(&mesh), "the square must come out all-quad");
        let gate = verify::gate(&mesh);
        assert_eq!(gate.cracks, 0);
        assert_eq!(gate.triangles, 0);
        assert!(gate.passes(), "{:?}", gate.reason());
        assert!(verify::boundary_offset(&domain, &mesh) < 1e-9);
    }

    #[test]
    fn the_stats_split_the_quads_between_rows_and_core() {
        let domain = Domain::rectangle(10.0, 10.0).unwrap();
        let (mesh, stats) = mesh_with_stats(&domain, &Sizing::uniform(1.0).unwrap()).unwrap();
        assert_eq!(stats.row_quads + stats.core_quads, mesh.cells.len());
        assert!(stats.row_quads > 0);
        assert!(stats.row_rounds > 0);
        assert!(stats.row_share() > 0.0 && stats.row_share() < 100.0);
    }

    #[test]
    fn seaming_raises_the_row_share_on_a_plate_with_a_hole() {
        let domain = Domain::plate_with_hole(100.0, 60.0, 10.0, 16).unwrap();
        let sizing = Sizing::uniform(4.0).unwrap();
        let (plain_mesh, plain) = mesh_with_stats_options(
            &domain,
            &sizing,
            &Options {
                seam: false,
                ..Options::default()
            },
        )
        .unwrap();
        let (seamed_mesh, seamed) =
            mesh_with_stats_options(&domain, &sizing, &Options::default()).unwrap();
        assert!(all_quads(&plain_mesh) && all_quads(&seamed_mesh));
        // Turning the seam off reproduces the pre-seam route exactly.
        assert_eq!((plain.row_quads, plain.core_quads), (128, 200));
        assert_eq!(plain.seam_rows, 0);
        // The hole front meets the plate front head-on; seaming connects them
        // and both keep advancing, so fewer quads are left to the matched core.
        assert!(seamed.seam_rows > 0);
        assert!(seamed.row_quads > plain.row_quads);
        assert!(seamed.core_quads < plain.core_quads);
        assert!(seamed.row_share() > plain.row_share() + 5.0);
        for mesh in [&plain_mesh, &seamed_mesh] {
            let gate = verify::gate(mesh);
            assert_eq!(gate.cracks, 0);
            assert_eq!(gate.triangles, 0);
            assert_eq!(gate.inverted, 0);
            assert_eq!(gate.overlapping, 0);
            assert!(verify::boundary_offset(&domain, mesh) < 1e-9);
        }
    }

    #[test]
    fn a_non_default_option_set_still_produces_a_valid_mesh() {
        let domain = Domain::rectangle(10.0, 10.0).unwrap();
        let sizing = Sizing::uniform(1.0).unwrap();
        let options = Options {
            max_row_rounds: 8,
            ..Options::default()
        };
        let mesh = mesh_with_options(&domain, &sizing, &options).unwrap();
        assert!(all_quads(&mesh), "the square must come out all-quad");
        let gate = verify::gate(&mesh);
        assert_eq!(gate.cracks, 0);
        assert_eq!(gate.triangles, 0);
        assert!(gate.passes(), "{:?}", gate.reason());
        assert!(verify::boundary_offset(&domain, &mesh) < 1e-9);
    }

    #[test]
    fn a_rectangle_with_uneven_aspect_is_all_quad() {
        let domain = Domain::rectangle(12.0, 5.0).unwrap();
        let mesh = mesh(&domain, &Sizing::uniform(1.0).unwrap()).unwrap();
        assert!(all_quads(&mesh));
        let gate = verify::gate(&mesh);
        assert_eq!(gate.cracks, 0);
        assert_eq!(gate.triangles, 0);
    }

    #[test]
    fn a_convex_hexagon_is_all_quad() {
        let r = 6.0;
        let outer = (0..6)
            .map(|i| {
                let a = std::f64::consts::TAU * i as f64 / 6.0;
                [r * a.cos(), r * a.sin()]
            })
            .collect();
        let domain = Domain::new(outer, vec![]).unwrap();
        let mesh = mesh(&domain, &Sizing::uniform(1.0).unwrap()).unwrap();
        assert!(all_quads(&mesh));
        let gate = verify::gate(&mesh);
        assert_eq!(gate.cracks, 0);
        assert_eq!(gate.triangles, 0);
    }

    #[test]
    fn a_plate_with_a_hole_is_all_quad_and_keeps_the_hole() {
        let domain = Domain::plate_with_hole(30.0, 20.0, 4.0, 16).unwrap();
        let mesh = mesh(&domain, &Sizing::uniform(1.5).unwrap()).unwrap();
        assert!(all_quads(&mesh));
        let gate = verify::gate(&mesh);
        assert_eq!(gate.cracks, 0);
        assert_eq!(gate.triangles, 0);
        assert!(verify::boundary_offset(&domain, &mesh) < 1e-9);
        // The hole is still a hole: its centre is not inside any element.
        assert!(!mesh.cells.iter().any(|c| {
            let x: f64 =
                c.nodes.iter().map(|&i| mesh.points[i][0]).sum::<f64>() / c.nodes.len() as f64;
            let y: f64 =
                c.nodes.iter().map(|&i| mesh.points[i][1]).sum::<f64>() / c.nodes.len() as f64;
            (x - 15.0).abs() < 2.0 && (y - 10.0).abs() < 2.0
        }));
    }

    #[test]
    fn a_quarter_ring_is_all_quad_and_passes_the_gate() {
        let arc = |r: f64, a0: f64, a1: f64, n: usize| -> Vec<[f64; 2]> {
            (0..=n)
                .map(|i| {
                    let a = a0 + (a1 - a0) * i as f64 / n as f64;
                    [r * a.cos(), r * a.sin()]
                })
                .collect()
        };
        let mut outer = arc(20.0, 0.0, std::f64::consts::FRAC_PI_2, 24);
        outer.extend(arc(10.0, std::f64::consts::FRAC_PI_2, 0.0, 12));
        let domain = Domain::new(outer, vec![]).unwrap();
        let mesh = mesh(&domain, &Sizing::uniform(1.0).unwrap()).unwrap();
        assert!(all_quads(&mesh));
        let gate = verify::gate(&mesh);
        assert_eq!(gate.triangles, 0);
        assert_eq!(gate.cracks, 0);
        assert!(gate.passes(), "{:?}", gate.reason());
    }

    #[test]
    fn a_bad_size_is_an_error_not_a_panic() {
        let domain = Domain::rectangle(10.0, 10.0).unwrap();
        let bad = Sizing {
            size: 0.0,
            all_quad: false,
        };
        assert!(mesh(&domain, &bad).is_err());
    }

    #[test]
    fn a_clockwise_or_empty_domain_is_an_error() {
        // `Domain::area` is signed: a clockwise outer loop is an empty domain.
        let clockwise =
            Domain::new(vec![[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]], vec![]).unwrap();
        assert!(mesh(&clockwise, &Sizing::uniform(1.0).unwrap()).is_err());
        let empty = Domain::new(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]], vec![]).unwrap();
        assert!(mesh(&empty, &Sizing::uniform(1.0).unwrap()).is_err());
    }
}
