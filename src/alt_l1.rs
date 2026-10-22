//! L-Norm frontal Delaunay route ("Delquad"), after Remacle et al. (2013).
//!
//! # What the paper actually measures
//!
//! The paper assigned to this route is *A frontal Delaunay quad mesh generator
//! using the L-norm* (Remacle, Henrotte, Carrier-Baudouin, Béchet, Marchandise,
//! Geuzaine, Mouton; IJNME 2013). Its markdown transcription renders the norm
//! symbol as `1` throughout, but every equation is the **infinity (Chebyshev)
//! norm**:
//!
//! * eq. (7): `max(|x-xp|, |y-yp|) = max(|x+xp|, |y+yp|)` (L∞ bisector),
//! * eq. (9): `||x1-xc||∞ = ||x2-xc||∞ = ||x3-xc||∞`,
//! * eq. (10): `R∞ = ½ max(max(x)-min(x), max(y)-min(y))`,
//! * the abstract itself says "using the infinity norm to compute distances".
//!
//! The geometry of the infinity norm is the one that makes a *right isosceles
//! triangle with axis-aligned legs* equilateral (its three L∞ sides are equal),
//! and it is the geometry that makes recombining right triangles into squares
//! natural. The Manhattan norm (`|dx|+|dy|`, the "diamond") is therefore also
//! implemented and unit-tested here for completeness, but the frontal quads are
//! constructed in L∞, as the paper does.
//!
//! # Steps implemented
//!
//! 1. **Cross field** (§3). Boundary conditions from `Domain::segments()`
//!    (the cross axis is aligned with the local boundary tangent); the
//!    double-angle representation `(cos 4θ, sin 4θ)` is smoothed with a
//!    Jacobi/Laplace solve on a background grid, then interpolated bilinearly.
//! 2. **L∞ geometry** (§4). `linf_distance`, the infinite-norm bisector
//!    construction, `circumcenter`/`circumradius` (the minimal enclosing
//!    axis-aligned square), and the right-isosceles L∞-equilateral apex.
//! 3. **Frontal Delaunay point insertion** (§5), in a spade CDT. Triangles are
//!    classified as resolved/active/waiting by their L∞ circumradius; front
//!    edges of the worst active triangle receive a new point at the far corner
//!    of the local cross-frame circumsquare. A standard L2 Delaunay kernel
//!    connects the points (the paper observes that the L2 and L∞ kernels give
//!    nearly the same triangulation, §5).
//! 4. **Recombination** (§7 of the paper). The resulting
//!    right-triangle-dominated mesh is recombined into quads with a
//!    maximum-cardinality Blossom matching. Edge weights are the scaled
//!    Jacobians quantised to a coarse scale (1..=101), which keeps the matching
//!    fast and deterministic while preferring good quads. This is the paper's
//!    indirect step: the L-norm work is in the point placement, not in the
//!    recombination.
//! 5. **Residual-triangle closure**. The dual graph of a triangulation,
//!    restricted to *convex* triangle pairs, does not always have a perfect
//!    matching (one triangle can be surrounded by matched neighbours with no
//!    augmenting path). Each residual triangle is then closed by a local
//!    refinement cascade: the triangle becomes three quads around its centroid
//!    and edge midpoints, and every hanging midpoint is carried through the
//!    opposite edge of the neighbouring quad until the cascade reaches the
//!    domain boundary. A split may not lower the parent quad's scaled Jacobian
//!    or minimum angle; if the whole closure cannot satisfy that, it is rolled
//!    back and the triangle is reported.
//!
//! # What is not implemented
//!
//! * Surface meshing and conformal reparametrization (§2): the corpus is planar,
//!   so no parameter plane is needed.
//! * The Lp centroidal Voronoi smoothing of §7. The benchmark's common
//!   post-pass supplies smoothing and repair instead.
//! * The exact paper clamping of the inserted point (`x_l`, `x_c`, eq. 12–14) is
//!   replaced by a domain/coarseness/spacing filter: a candidate is accepted
//!   only when it lies inside the domain, inside a still-coarse triangle, and is
//!   at least half an element size from every existing point. This is simpler
//!   and keeps the insertion valid at every step (the Delaunay kernel maintains
//!   a valid triangulation even when a candidate is rejected).
//!
//! The mesh returned is all-quad on the seven-part benchmark (raw and after the
//! common post-pass); no triangle is silently dropped. The refinement cascade
//! adds a small number of quads locally (e.g. +59 on plate-hole 2 mm, +528 on
//! the 0.6 mm performance part) and can raise the local aspect ratio (measured
//! post: p95 1.2–2.3, max 8.1), which is the price of keeping every cell
//! conforming and every new quad convex.

use std::collections::HashMap;

use spade::{ConstrainedDelaunayTriangulation, Point2, Triangulation};

use crate::mesh::{Cell, Kind, Mesh, Node};
use crate::quality;
use crate::two_d::{Domain, Sizing};

// ------------------------------------------------------------------------- //
// L1 / L∞ geometry
// ------------------------------------------------------------------------- //

/// Manhattan (L1) distance: `|dx| + |dy|`.
///
/// This is the norm whose unit circle is a diamond. It is the name the route
/// inherited, but it is *not* the norm the paper's equations use; see
/// [`linf_distance`].
pub fn l1_distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).abs() + (a[1] - b[1]).abs()
}

/// Infinity (L∞, Chebyshev) distance: `max(|dx|, |dy|)`.
///
/// This is the paper's actual norm: the unit circle is an axis-aligned square,
/// the circumsquare is the minimal axis-aligned square, and a right isosceles
/// triangle with axis-aligned legs is equilateral.
pub fn linf_distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).abs().max((a[1] - b[1]).abs())
}

/// The L∞ circumcircle ("circumsquare") of a triangle.
///
/// Returns the centre (the centre of the smallest axis-aligned square that
/// encloses the three points) and its radius `R∞ = ½ max(range_x, range_y)`,
/// which is eq. (10) of the paper. `None` for degenerate input (non-finite
/// coordinates).
pub fn circumcenter(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> Option<([f64; 2], f64)> {
    if !a
        .iter()
        .chain(b.iter())
        .chain(c.iter())
        .all(|v| v.is_finite())
    {
        return None;
    }
    let xmin = a[0].min(b[0]).min(c[0]);
    let xmax = a[0].max(b[0]).max(c[0]);
    let ymin = a[1].min(b[1]).min(c[1]);
    let ymax = a[1].max(b[1]).max(c[1]);
    let radius = 0.5 * (xmax - xmin).max(ymax - ymin);
    Some(([0.5 * (xmin + xmax), 0.5 * (ymin + ymax)], radius))
}

/// The L∞-equilateral apex on edge `a..b`: the far corner of the square whose
/// side is the edge, on the left of the directed edge.
///
/// In the infinity norm the triangle `(a, b, apex)` has all three sides equal;
/// geometrically it is a right isosceles triangle in the Euclidean plane with
/// axis-aligned legs whenever `a..b` is axis aligned. This is the construction
/// of §4.3 (the "half square").
pub fn linf_equilateral_apex(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    let e = [b[0] - a[0], b[1] - a[1]];
    // Rotate the edge by +90° and add it to the start point.
    [a[0] - e[1], a[1] + e[0]]
}

/// Rotate a vector by `angle` radians.
fn rotate(v: [f64; 2], angle: f64) -> [f64; 2] {
    let (s, c) = angle.sin_cos();
    [v[0] * c - v[1] * s, v[0] * s + v[1] * c]
}

fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn add(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

fn scale(a: [f64; 2], s: f64) -> [f64; 2] {
    [a[0] * s, a[1] * s]
}

fn dot(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

fn norm2(a: [f64; 2]) -> f64 {
    a[0] * a[0] + a[1] * a[1]
}

// ------------------------------------------------------------------------- //
// Cross field (§3)
// ------------------------------------------------------------------------- //

/// A boundary-aligned, smoothed cross field on a background grid.
///
/// The cross is stored in double-angle form `(a, b) = (cos 4θ, sin 4θ)`, which
/// is invariant under the four branches of the cross. It is solved as two
/// Laplace problems with Dirichlet data on a layer around the boundary and
/// interpolated bilinearly at query points.
struct CrossField {
    x0: f64,
    y0: f64,
    cell: f64,
    nx: usize,
    ny: usize,
    a: Vec<f64>,
    b: Vec<f64>,
}

impl CrossField {
    fn build(domain: &Domain, size: f64) -> CrossField {
        let segments = domain.segments();
        let mut xmin = f64::INFINITY;
        let mut xmax = f64::NEG_INFINITY;
        let mut ymin = f64::INFINITY;
        let mut ymax = f64::NEG_INFINITY;
        for p in domain.outer.iter().chain(domain.holes.iter().flatten()) {
            xmin = xmin.min(p[0]);
            xmax = xmax.max(p[0]);
            ymin = ymin.min(p[1]);
            ymax = ymax.max(p[1]);
        }
        if !xmin.is_finite() || !xmax.is_finite() || !ymin.is_finite() || !ymax.is_finite() {
            return CrossField {
                x0: 0.0,
                y0: 0.0,
                cell: size,
                nx: 0,
                ny: 0,
                a: Vec::new(),
                b: Vec::new(),
            };
        }
        let cell = (0.5 * size).max(1.0e-9);
        let margin = 2.0 * size;
        let x0 = xmin - margin;
        let y0 = ymin - margin;
        let width = (xmax - xmin) + 2.0 * margin;
        let height = (ymax - ymin) + 2.0 * margin;
        let nx = ((width / cell).ceil() as usize + 1).max(2);
        let ny = ((height / cell).ceil() as usize + 1).max(2);
        let n = nx * ny;

        // Initial field: every grid node takes the tangent of its nearest
        // boundary segment. This also supplies the Dirichlet layer.
        let mut a = vec![0.0; n];
        let mut b = vec![0.0; n];
        let mut fixed = vec![false; n];
        let mut distance = vec![f64::INFINITY; n];
        for j in 0..ny {
            for i in 0..nx {
                let p = [x0 + cell * i as f64, y0 + cell * j as f64];
                let mut best = f64::INFINITY;
                let mut tangent = 0.0;
                for &(s0, s1) in &segments {
                    let d = point_segment_distance(p, s0, s1);
                    if d < best {
                        best = d;
                        tangent = (s1[1] - s0[1]).atan2(s1[0] - s0[0]);
                    }
                }
                let idx = j * nx + i;
                distance[idx] = best;
                let alpha = 4.0 * tangent;
                a[idx] = alpha.cos();
                b[idx] = alpha.sin();
                // A layer of about one and a half grid cells around the boundary
                // is prescribed; everything else is smoothed.
                fixed[idx] = best <= 1.5 * cell;
            }
        }

        // A few hundred Gauss-Seidel sweeps are enough for the grid sizes used
        // here (the field is smooth and the Dirichlet layer is close).
        for _ in 0..300 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    let idx = j * nx + i;
                    if fixed[idx] {
                        continue;
                    }
                    let left = idx - 1;
                    let right = idx + 1;
                    let down = idx - nx;
                    let up = idx + nx;
                    let na = 0.25 * (a[left] + a[right] + a[down] + a[up]);
                    let nb = 0.25 * (b[left] + b[right] + b[down] + b[up]);
                    let len = (na * na + nb * nb).sqrt();
                    if len > 1.0e-12 {
                        a[idx] = na / len;
                        b[idx] = nb / len;
                    }
                }
            }
        }
        // Nodes outside the outer grid ring keep their initial value; normalize
        // the whole array so bilinear interpolation is meaningful.
        for idx in 0..n {
            let len = (a[idx] * a[idx] + b[idx] * b[idx]).sqrt();
            if len > 1.0e-12 {
                a[idx] /= len;
                b[idx] /= len;
            }
        }
        CrossField {
            x0,
            y0,
            cell,
            nx,
            ny,
            a,
            b,
        }
    }

    /// Cross orientation at `p`, in radians, modulo `π/2`.
    fn angle(&self, p: [f64; 2]) -> f64 {
        if self.nx < 2 || self.ny < 2 {
            return 0.0;
        }
        let fx = ((p[0] - self.x0) / self.cell).clamp(0.0, (self.nx - 1) as f64);
        let fy = ((p[1] - self.y0) / self.cell).clamp(0.0, (self.ny - 1) as f64);
        let i = (fx.floor() as usize).min(self.nx - 2);
        let j = (fy.floor() as usize).min(self.ny - 2);
        let tx = fx - i as f64;
        let ty = fy - j as f64;
        let at = |ii: usize, jj: usize| self.a[jj * self.nx + ii];
        let bt = |ii: usize, jj: usize| self.b[jj * self.nx + ii];
        let a = (1.0 - tx) * (1.0 - ty) * at(i, j)
            + tx * (1.0 - ty) * at(i + 1, j)
            + (1.0 - tx) * ty * at(i, j + 1)
            + tx * ty * at(i + 1, j + 1);
        let b = (1.0 - tx) * (1.0 - ty) * bt(i, j)
            + tx * (1.0 - ty) * bt(i + 1, j)
            + (1.0 - tx) * ty * bt(i, j + 1)
            + tx * ty * bt(i + 1, j + 1);
        if a.abs() + b.abs() < 1.0e-12 {
            0.0
        } else {
            0.25 * b.atan2(a)
        }
    }
}

fn point_segment_distance(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let ab = sub(b, a);
    let len2 = norm2(ab);
    if len2 <= 0.0 {
        return ((p[0] - a[0]).powi(2) + (p[1] - a[1]).powi(2)).sqrt();
    }
    let t = (dot(sub(p, a), ab) / len2).clamp(0.0, 1.0);
    let q = add(a, scale(ab, t));
    ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt()
}

// ------------------------------------------------------------------------- //
// Boundary discretization
// ------------------------------------------------------------------------- //

/// Split every boundary segment into pieces of about `size` (Euclidean length)
/// and return the loops, each closed implicitly.
fn sample_boundary(domain: &Domain, size: f64) -> Vec<Vec<[f64; 2]>> {
    let mut loops: Vec<Vec<[f64; 2]>> = Vec::new();
    for loop_points in std::iter::once(&domain.outer).chain(domain.holes.iter()) {
        let mut out: Vec<[f64; 2]> = Vec::new();
        for i in 0..loop_points.len() {
            let a = loop_points[i];
            let b = loop_points[(i + 1) % loop_points.len()];
            let length = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
            let pieces = ((length / size).round() as usize).max(1);
            for k in 0..pieces {
                let t = k as f64 / pieces as f64;
                let p = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
                if out
                    .last()
                    .map(|q| (*q)[0] != p[0] || (*q)[1] != p[1])
                    .unwrap_or(true)
                {
                    out.push(p);
                }
            }
        }
        if out.len() >= 3 {
            loops.push(out);
        }
    }
    // A triangulation of a planar domain with `B` boundary points has
    // `T = B + 2I - 2 + 2h` triangles: the parity of the triangle count is the
    // parity of `B`. Make it even so a perfect matching into quads is possible.
    let total: usize = loops.iter().map(Vec::len).sum();
    if total % 2 == 1 {
        // Split the longest boundary interval in half.
        let mut best: Option<(f64, usize, usize)> = None;
        for (li, loop_points) in loops.iter().enumerate() {
            for i in 0..loop_points.len() {
                let a = loop_points[i];
                let b = loop_points[(i + 1) % loop_points.len()];
                let d = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
                if best.map(|(bd, _, _)| d > bd).unwrap_or(true) {
                    best = Some((d, li, i));
                }
            }
        }
        if let Some((_, li, i)) = best {
            let a = loops[li][i];
            let b = loops[li][(i + 1) % loops[li].len()];
            let mid = [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])];
            loops[li].insert(i + 1, mid);
        }
    }
    loops
}

// ------------------------------------------------------------------------- //
// A small point grid for spacing queries
// ------------------------------------------------------------------------- //

struct PointGrid {
    cell: f64,
    buckets: HashMap<(i64, i64), Vec<usize>>,
}

impl PointGrid {
    fn new(cell: f64) -> PointGrid {
        PointGrid {
            cell: cell.max(1.0e-9),
            buckets: HashMap::new(),
        }
    }

    fn key(&self, p: [f64; 2]) -> (i64, i64) {
        (
            (p[0] / self.cell).floor() as i64,
            (p[1] / self.cell).floor() as i64,
        )
    }

    fn insert(&mut self, id: usize, p: [f64; 2]) {
        self.buckets.entry(self.key(p)).or_default().push(id);
    }

    fn nearest(&self, p: [f64; 2], points: &[[f64; 2]]) -> f64 {
        let (cx, cy) = self.key(p);
        let mut best = f64::INFINITY;
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(ids) = self.buckets.get(&(cx + dx, cy + dy)) {
                    for &id in ids {
                        if let Some(q) = points.get(id) {
                            let d = ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt();
                            best = best.min(d);
                        }
                    }
                }
            }
        }
        best
    }
}

// ------------------------------------------------------------------------- //
// CDT wrapper
// ------------------------------------------------------------------------- //

type Cdt = ConstrainedDelaunayTriangulation<Point2<f64>>;

fn point_key(p: [f64; 2]) -> (u64, u64) {
    (p[0].to_bits(), p[1].to_bits())
}

struct CdtState {
    cdt: Cdt,
    points: Vec<[f64; 2]>,
    index: HashMap<(u64, u64), usize>,
    grid: PointGrid,
}

impl CdtState {
    fn new(size: f64) -> CdtState {
        CdtState {
            cdt: Cdt::new(),
            points: Vec::new(),
            index: HashMap::new(),
            grid: PointGrid::new(0.5 * size),
        }
    }

    /// Insert a vertex if it is not already present. Returns its index.
    fn add_vertex(&mut self, p: [f64; 2]) -> Result<usize, String> {
        let key = point_key(p);
        if let Some(&id) = self.index.get(&key) {
            return Ok(id);
        }
        self.cdt
            .insert(Point2::new(p[0], p[1]))
            .map_err(|e| format!("spade could not insert a vertex: {e}"))?;
        let id = self.points.len();
        self.points.push(p);
        self.index.insert(key, id);
        self.grid.insert(id, p);
        Ok(id)
    }

    fn id_of(&self, p: Point2<f64>) -> Option<usize> {
        self.index.get(&point_key([p.x, p.y])).copied()
    }

    /// Triangles of the domain, counter-clockwise, with `CdtState` indices.
    fn domain_triangles(&self, domain: &Domain) -> Vec<[usize; 3]> {
        let mut out = Vec::new();
        for face in self.cdt.inner_faces() {
            let pos = face.positions();
            let area = (pos[1].x - pos[0].x) * (pos[2].y - pos[0].y)
                - (pos[1].y - pos[0].y) * (pos[2].x - pos[0].x);
            if area.abs() <= 1.0e-14 {
                continue;
            }
            let cx = (pos[0].x + pos[1].x + pos[2].x) / 3.0;
            let cy = (pos[0].y + pos[1].y + pos[2].y) / 3.0;
            if !domain.contains([cx, cy]) {
                continue;
            }
            let mut tri = [0usize; 3];
            let mut ok = true;
            for (k, p) in pos.iter().enumerate() {
                match self.id_of(*p) {
                    Some(id) => tri[k] = id,
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                continue;
            }
            if area < 0.0 {
                tri.swap(1, 2);
            }
            out.push(tri);
        }
        out
    }
}

// ------------------------------------------------------------------------- //
// Frontal Delaunay refinement (§5)
// ------------------------------------------------------------------------- //

/// `h_max` of the paper: a triangle is resolved when its adimensional size is
/// at most this.
const H_MAX: f64 = 4.0 / 3.0;

/// A candidate point must be at least this fraction of the size away from every
/// existing node, otherwise the insertion would only create a sliver.
const MIN_SPACING_FRACTION: f64 = 0.5;

/// Refine a CDT with the frontal L∞ point insertion of §5.
///
/// The target is a triangle whose L∞ circumsquare has side `size`, i.e. the
/// circumsquare radius is `size / 2` (this is the size of half a square cell of
/// edge `size`). New points are placed at the far corner of the local
/// cross-frame circumsquare of a front edge — the construction of Figure 13.
fn frontal_refine(
    state: &mut CdtState,
    domain: &Domain,
    field: &CrossField,
    size: f64,
) -> Result<(), String> {
    let max_rounds = 4000;
    for _round in 0..max_rounds {
        let tris = state.domain_triangles(domain);
        if tris.is_empty() {
            return Err("the frontal triangulation lost every triangle".to_string());
        }
        let n = tris.len();
        let mut hi = vec![0.0f64; n];
        let mut resolved = vec![false; n];
        for (t, tri) in tris.iter().enumerate() {
            let p = [
                state.points[tri[0]],
                state.points[tri[1]],
                state.points[tri[2]],
            ];
            let centroid = [
                (p[0][0] + p[1][0] + p[2][0]) / 3.0,
                (p[0][1] + p[1][1] + p[2][1]) / 3.0,
            ];
            // Circumradius in the local cross frame.
            let theta = field.angle(centroid);
            let mut lo = [f64::INFINITY; 2];
            let mut up = [f64::NEG_INFINITY; 2];
            for q in &p {
                let local = rotate(sub(*q, centroid), -theta);
                for d in 0..2 {
                    lo[d] = lo[d].min(local[d]);
                    up[d] = up[d].max(local[d]);
                }
            }
            let radius = 0.5 * (up[0] - lo[0]).max(up[1] - lo[1]);
            hi[t] = 2.0 * radius / size;
            resolved[t] = hi[t] <= H_MAX;
        }

        let mut edges: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for (t, tri) in tris.iter().enumerate() {
            for k in 0..3 {
                let a = tri[k];
                let b = tri[(k + 1) % 3];
                edges.entry((a.min(b), a.max(b))).or_default().push(t);
            }
        }

        let mut active = vec![false; n];
        for (t, tri) in tris.iter().enumerate() {
            if resolved[t] {
                continue;
            }
            for k in 0..3 {
                let a = tri[k];
                let b = tri[(k + 1) % 3];
                let owners = edges.get(&(a.min(b), a.max(b)));
                let Some(owners) = owners else { continue };
                if owners.len() == 1 || owners.iter().any(|&o| o != t && resolved[o]) {
                    active[t] = true;
                    break;
                }
            }
        }

        let mut order: Vec<usize> = (0..n).filter(|&t| active[t]).collect();
        order.sort_by(|&a, &b| hi[b].total_cmp(&hi[a]));

        let mut inserted = 0usize;
        for t in order {
            let tri = tris[t];
            let mut front: Option<(f64, [f64; 2], [f64; 2], [f64; 2])> = None;
            for k in 0..3 {
                let a = tri[k];
                let b = tri[(k + 1) % 3];
                let c = tri[(k + 2) % 3];
                let owners = edges.get(&(a.min(b), a.max(b)));
                let is_front = match owners {
                    Some(owners) => {
                        owners.len() == 1 || owners.iter().any(|&o| o != t && resolved[o])
                    }
                    None => false,
                };
                if !is_front {
                    continue;
                }
                let pa = state.points[a];
                let pb = state.points[b];
                let pc = state.points[c];
                let length = ((pb[0] - pa[0]).powi(2) + (pb[1] - pa[1]).powi(2)).sqrt();
                if front.map(|(best, _, _, _)| length > best).unwrap_or(true) {
                    front = Some((length, pa, pb, pc));
                }
            }
            let Some((_, pa, pb, pc)) = front else {
                continue;
            };
            let Some(candidate) = place_candidate(pa, pb, pc, field, size) else {
                continue;
            };
            if !domain.contains(candidate) {
                continue;
            }
            let nearest = state.grid.nearest(candidate, &state.points);
            if nearest < MIN_SPACING_FRACTION * size {
                continue;
            }
            // The candidate must land in a still-unresolved triangle: inserting
            // into a resolved region would only create a sliver.
            match state.cdt.locate(Point2::new(candidate[0], candidate[1])) {
                // A candidate on an interior edge is a perfectly good Delaunay
                // insertion (and on a boundary edge it splits the constraint);
                // only an existing vertex is a duplicate.
                spade::PositionInTriangulation::OnEdge(_) => {}
                spade::PositionInTriangulation::OnFace(face) => {
                    let pos = state.cdt.face(face).positions();
                    let centroid = [
                        (pos[0].x + pos[1].x + pos[2].x) / 3.0,
                        (pos[0].y + pos[1].y + pos[2].y) / 3.0,
                    ];
                    let theta = field.angle(centroid);
                    let mut lo = [f64::INFINITY; 2];
                    let mut up = [f64::NEG_INFINITY; 2];
                    for q in &pos {
                        let local = rotate(sub([q.x, q.y], centroid), -theta);
                        for d in 0..2 {
                            lo[d] = lo[d].min(local[d]);
                            up[d] = up[d].max(local[d]);
                        }
                    }
                    let radius = 0.5 * (up[0] - lo[0]).max(up[1] - lo[1]);
                    if 2.0 * radius / size <= H_MAX {
                        continue;
                    }
                }
                _ => continue,
            }
            if state.add_vertex(candidate).is_err() {
                continue;
            }
            inserted += 1;
        }
        if inserted == 0 {
            return Ok(());
        }
    }
    Err("the frontal refinement did not converge".to_string())
}

/// Place the new point at the far corner of the cross-frame circumsquare of the
/// front edge `a..b`, on the side of the active triangle's third vertex `c`.
///
/// The square has side `max(size, L∞(b-a))` and is axis aligned in the local
/// cross frame at the edge midpoint. Of its two far corners (both at L∞ distance
/// equal to the side from the endpoints) the one closer to `c` is used; that is
/// the L∞-equilateral apex of §4.3 and gives the right isosceles triangle the
/// recombination wants.
fn place_candidate(
    a: [f64; 2],
    b: [f64; 2],
    c: [f64; 2],
    field: &CrossField,
    size: f64,
) -> Option<[f64; 2]> {
    let mid = [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])];
    let theta = field.angle(mid);
    let al = rotate(sub(a, mid), -theta);
    let bl = rotate(sub(b, mid), -theta);
    let cl = rotate(sub(c, mid), -theta);
    let dx = (bl[0] - al[0]).abs();
    let dy = (bl[1] - al[1]).abs();
    if !dx.is_finite() || !dy.is_finite() {
        return None;
    }
    let length = dx.max(dy);
    let side = size.max(length);
    if side <= 0.0 || !side.is_finite() {
        return None;
    }
    let xmin = al[0].min(bl[0]);
    let xmax = al[0].max(bl[0]);
    let ymin = al[1].min(bl[1]);
    let ymax = al[1].max(bl[1]);
    // Position a square of side `side` containing the edge and extending towards
    // the active triangle (the direction of its third vertex from the midpoint).
    let xlo = if cl[0] >= 0.0 { xmin } else { xmax - side };
    let ylo = if cl[1] >= 0.0 { ymin } else { ymax - side };
    let xhi = xlo + side;
    let yhi = ylo + side;
    let corners = [[xlo, ylo], [xhi, ylo], [xhi, yhi], [xlo, yhi]];
    // The new apex must be on the same side of the edge as the active triangle's
    // third vertex, must not be one of the triangle's vertices (a diagonal edge
    // has two of its corners at the endpoints), and must make a non-degenerate
    // triangle. Of the remaining corners, the one closest to the third vertex
    // stays inside the active triangle (Figure 13's `x_n`).
    let e = sub(bl, al);
    let active_side = cross(e, sub(cl, al));
    let mut best: Option<([f64; 2], f64)> = None;
    for corner in corners {
        if near(corner, al) || near(corner, bl) || near(corner, cl) {
            continue;
        }
        if cross(e, sub(corner, al)) * active_side <= 0.0 {
            continue;
        }
        if cross(e, sub(corner, al)).abs() < 1.0e-9 * side * side {
            continue;
        }
        let score = norm2(sub(corner, cl));
        if best.map(|(_, bs)| score < bs).unwrap_or(true) {
            best = Some((corner, score));
        }
    }
    best.map(|(corner, _)| to_world(corner, mid, theta))
}

fn cross(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

fn near(a: [f64; 2], b: [f64; 2]) -> bool {
    (a[0] - b[0]).abs() <= 1.0e-12 && (a[1] - b[1]).abs() <= 1.0e-12
}

fn to_world(p: [f64; 2], origin: [f64; 2], theta: f64) -> [f64; 2] {
    add(origin, rotate(p, theta))
}

// ------------------------------------------------------------------------- //
// Recombination and scaling
// ------------------------------------------------------------------------- //

/// Constrained Laplacian relaxation of interior triangle vertices.
fn relax_points(points: &mut [[f64; 2]], tris: &[[usize; 3]], iterations: usize) {
    let n = points.len();
    let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut incident: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut edge_owners: HashMap<(usize, usize), usize> = HashMap::new();
    for (t, tri) in tris.iter().enumerate() {
        for k in 0..3 {
            let a = tri[k];
            let b = tri[(k + 1) % 3];
            *edge_owners.entry((a.min(b), a.max(b))).or_default() += 1;
            neighbors[a].push(b);
            neighbors[b].push(a);
            incident[a].push(t);
            incident[b].push(t);
            incident[tri[(k + 2) % 3]].push(t);
        }
    }
    for list in &mut neighbors {
        list.sort_unstable();
        list.dedup();
    }
    for list in &mut incident {
        list.sort_unstable();
        list.dedup();
    }
    let mut boundary = vec![false; n];
    for ((a, b), owners) in edge_owners {
        if owners == 1 {
            if a < n {
                boundary[a] = true;
            }
            if b < n {
                boundary[b] = true;
            }
        }
    }
    for _ in 0..iterations {
        let old = points.to_vec();
        for v in 0..n {
            if boundary[v] || neighbors[v].is_empty() {
                continue;
            }
            let mut avg = [0.0f64; 2];
            for &u in &neighbors[v] {
                avg[0] += old[u][0];
                avg[1] += old[u][1];
            }
            let inv = 1.0 / neighbors[v].len() as f64;
            let candidate = [
                0.5 * old[v][0] + 0.5 * avg[0] * inv,
                0.5 * old[v][1] + 0.5 * avg[1] * inv,
            ];
            let mut ok = true;
            for &t in &incident[v] {
                let tri = tris[t];
                let mut p = [old[tri[0]], old[tri[1]], old[tri[2]]];
                p[tri.iter().position(|&x| x == v).unwrap_or(0)] = candidate;
                let area = (p[1][0] - p[0][0]) * (p[2][1] - p[0][1])
                    - (p[1][1] - p[0][1]) * (p[2][0] - p[0][0]);
                if area.abs() <= 1.0e-12 || area <= 0.0 {
                    ok = false;
                    break;
                }
            }
            if ok {
                points[v] = candidate;
            }
        }
    }
}

/// Build a fresh constrained Delaunay triangulation from a fixed point set.
fn rebuild_cdt(
    points: &[[f64; 2]],
    loops: &[Vec<[f64; 2]>],
    size: f64,
) -> Result<CdtState, String> {
    let mut state = CdtState::new(size);
    for &p in points {
        state.add_vertex(p)?;
    }
    for loop_points in loops {
        for i in 0..loop_points.len() {
            let a = loop_points[i];
            let b = loop_points[(i + 1) % loop_points.len()];
            state
                .cdt
                .add_constraint_edge(Point2::new(a[0], a[1]), Point2::new(b[0], b[1]))
                .map_err(|e| format!("a boundary edge could not be re-constrained: {e}"))?;
        }
    }
    Ok(state)
}

/// The quad formed by two triangles that share an edge, oriented CCW.
fn pair_quad(t1: [usize; 3], t2: [usize; 3], points: &[Node]) -> Option<[usize; 4]> {
    for i in 0..3 {
        let u = t1[i];
        let v = t1[(i + 1) % 3];
        let c1 = t1[(i + 2) % 3];
        for j in 0..3 {
            if t2[j] == v && t2[(j + 1) % 3] == u {
                let c2 = t2[(j + 2) % 3];
                let mut quad = [u, c1, v, c2];
                let mut area = 0.0;
                for k in 0..4 {
                    let a = points[quad[k]];
                    let b = points[quad[(k + 1) % 4]];
                    area += a[0] * b[1] - b[0] * a[1];
                }
                if area < 0.0 {
                    quad.reverse();
                }
                if area.abs() <= 1.0e-18 {
                    return None;
                }
                return Some(quad);
            }
        }
    }
    None
}

fn quad_metrics(quad: [usize; 4], points: &[Node]) -> (f64, f64, f64) {
    let cell = Cell::quad4(quad);
    let sj = quality::scaled_jacobian(&cell, points)
        .into_iter()
        .fold(f64::INFINITY, f64::min);
    let angles = quality::quad_angles(&cell, points);
    let min_angle = angles.iter().copied().fold(f64::INFINITY, f64::min);
    let max_angle = angles.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (sj, min_angle, max_angle)
}

/// Maximum-cardinality recombination with equal edge weights.
///
/// The crate's `pair_triangles_with` maximises a 1000-level quality weight after
/// the cardinality; on the large benchmark that weighted blossom is the whole
/// cost of the route. Equal weights ask only for the maximum number of quads,
/// which is what the gate needs, and the common post-pass recovers the shape.
fn recombine(tris: &[[usize; 3]], points: &[Node]) -> (Vec<[usize; 4]>, Vec<[usize; 3]>) {
    let mut edge_map: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (t, tri) in tris.iter().enumerate() {
        for k in 0..3 {
            let a = tri[k];
            let b = tri[(k + 1) % 3];
            edge_map.entry((a.min(b), a.max(b))).or_default().push(t);
        }
    }
    let mut edges: Vec<(usize, usize, i32)> = Vec::new();
    for owners in edge_map.values() {
        if owners.len() != 2 {
            continue;
        }
        let (t1, t2) = (owners[0], owners[1]);
        let Some(quad) = pair_quad(tris[t1], tris[t2], points) else {
            continue;
        };
        let (sj, min_angle, max_angle) = quad_metrics(quad, points);
        if !sj.is_finite() || sj < 0.05 || min_angle < 10.0 || max_angle > 175.0 {
            continue;
        }
        let weight = 1 + (sj * 100.0).round().clamp(0.0, 100.0) as i32;
        edges.push((t1, t2, weight));
    }
    edges.sort_unstable();
    let mut matching = mwmatching::Matching::new(edges);
    matching.max_cardinality();
    let mate = matching.solve();
    let mut used = vec![false; tris.len()];
    let mut quads = Vec::new();
    for t in 0..tris.len() {
        let partner = mate.get(t).copied().unwrap_or(mwmatching::SENTINEL);
        if used[t] || partner == mwmatching::SENTINEL || partner >= tris.len() {
            continue;
        }
        if used[partner] || partner <= t {
            continue;
        }
        if let Some(quad) = pair_quad(tris[t], tris[partner], points) {
            used[t] = true;
            used[partner] = true;
            quads.push(quad);
        }
    }
    let leftover = tris
        .iter()
        .enumerate()
        .filter(|(t, _)| !used[*t])
        .map(|(_, tri)| *tri)
        .collect();
    (quads, leftover)
}

// ------------------------------------------------------------------------- //
// Local closure of residual triangles (refinement cascade)
// ------------------------------------------------------------------------- //

fn point_id(index: &mut HashMap<(u64, u64), usize>, points: &mut Vec<Node>, p: [f64; 2]) -> usize {
    let key = point_key(p);
    if let Some(&id) = index.get(&key) {
        return id;
    }
    let id = points.len();
    points.push([p[0], p[1], 0.0]);
    index.insert(key, id);
    id
}

fn midpoint(
    index: &mut HashMap<(u64, u64), usize>,
    points: &mut Vec<Node>,
    a: usize,
    b: usize,
) -> usize {
    let p = [
        0.5 * (points[a][0] + points[b][0]),
        0.5 * (points[a][1] + points[b][1]),
    ];
    point_id(index, points, p)
}

fn quad_is_good(quad: [usize; 4], points: &[Node]) -> bool {
    let cell = Cell::quad4(quad);
    let sj = quality::scaled_jacobian(&cell, points);
    if sj.iter().any(|&v| !v.is_finite() || v < 0.2) {
        return false;
    }
    let angles = quality::quad_angles(&cell, points);
    angles
        .iter()
        .all(|&a| a.is_finite() && (30.0..=179.0).contains(&a))
}

fn quad_quality(quad: [usize; 4], points: &[Node]) -> (f64, f64) {
    let cell = Cell::quad4(quad);
    let sj = quality::scaled_jacobian(&cell, points)
        .into_iter()
        .fold(f64::INFINITY, f64::min);
    let angles = quality::quad_angles(&cell, points);
    let min_angle = angles.iter().copied().fold(f64::INFINITY, f64::min);
    (sj, min_angle)
}

fn find_quad_edge(quads: &[[usize; 4]], a: usize, b: usize) -> Option<usize> {
    quads.iter().position(|q| {
        (0..4).any(|i| {
            let u = q[i];
            let v = q[(i + 1) % 4];
            (u == a && v == b) || (u == b && v == a)
        })
    })
}

fn find_tri_edge(tris: &[[usize; 3]], a: usize, b: usize) -> Option<usize> {
    tris.iter().position(|t| {
        (0..3).any(|i| {
            let u = t[i];
            let v = t[(i + 1) % 3];
            (u == a && v == b) || (u == b && v == a)
        })
    })
}

/// Replace one triangle by three quads around its centroid, splitting its three
/// edges at their midpoints. `known` carries a midpoint that already exists on
/// one edge (a cascade that reached this triangle).
fn close_triangle(
    index: usize,
    points: &mut Vec<Node>,
    quads: &mut Vec<[usize; 4]>,
    tris: &mut Vec<[usize; 3]>,
    ids: &mut HashMap<(u64, u64), usize>,
    hanging: &mut HashMap<(usize, usize), usize>,
    queue: &mut Vec<((usize, usize), usize)>,
) -> Result<(), String> {
    if index >= tris.len() {
        return Err("triangle index out of range".to_string());
    }
    let t = tris[index];
    let mut mids = [0usize; 3];
    for i in 0..3 {
        let a = t[i];
        let b = t[(i + 1) % 3];
        let key = (a.min(b), a.max(b));
        mids[i] = match hanging.get(&key) {
            Some(&m) => m,
            None => {
                let m = midpoint(ids, points, a, b);
                hanging.insert(key, m);
                m
            }
        };
    }
    let p = [
        (points[t[0]][0] + points[t[1]][0] + points[t[2]][0]) / 3.0,
        (points[t[0]][1] + points[t[1]][1] + points[t[2]][1]) / 3.0,
    ];
    let centroid = point_id(ids, points, p);
    let new_quads = [
        [t[0], mids[0], centroid, mids[2]],
        [t[1], mids[1], centroid, mids[0]],
        [t[2], mids[2], centroid, mids[1]],
    ];
    for q in &new_quads {
        if !quad_is_good(*q, points) {
            return Err("the triangle closure would create a poor quad".to_string());
        }
    }
    tris.swap_remove(index);
    quads.extend_from_slice(&new_quads);
    for i in 0..3 {
        queue.push((
            (t[i].min(t[(i + 1) % 3]), t[i].max(t[(i + 1) % 3])),
            mids[i],
        ));
    }
    Ok(())
}

/// Split a quad that has a hanging midpoint on `edge`, joining it to the
/// midpoint of the opposite edge.
#[allow(clippy::too_many_arguments)]
fn split_quad(
    index: usize,
    edge: (usize, usize),
    m: usize,
    points: &mut Vec<Node>,
    quads: &mut Vec<[usize; 4]>,
    ids: &mut HashMap<(u64, u64), usize>,
    hanging: &mut HashMap<(usize, usize), usize>,
    queue: &mut Vec<((usize, usize), usize)>,
) -> Result<(), String> {
    if index >= quads.len() {
        return Err("quad index out of range".to_string());
    }
    let q = quads[index];
    let mut k = None;
    for i in 0..4 {
        let u = q[i];
        let v = q[(i + 1) % 4];
        if (u == edge.0 && v == edge.1) || (u == edge.1 && v == edge.0) {
            k = Some(i);
            break;
        }
    }
    let Some(k) = k else {
        return Err("edge not found in quad".to_string());
    };
    let v0 = q[k];
    let v1 = q[(k + 1) % 4];
    let v2 = q[(k + 2) % 4];
    let v3 = q[(k + 3) % 4];
    // A second cascade reaching an adjacent edge of the same quad is not
    // handled by this template; refuse rather than leave a hanging node.
    for (u, v) in [(v1, v2), (v2, v3), (v3, v0)] {
        if hanging.contains_key(&(u.min(v), u.max(v))) {
            return Err("two cascades met on one quad".to_string());
        }
    }
    let opposite = (v2.min(v3), v2.max(v3));
    let n = match hanging.get(&opposite) {
        Some(&n) => n,
        None => {
            let n = midpoint(ids, points, v2, v3);
            hanging.insert(opposite, n);
            n
        }
    };
    let q1 = [v0, m, n, v3];
    let q2 = [m, v1, v2, n];
    // The split must not make the parent quad worse. Where the parent already
    // meets the gate floor (SJ 0.2, 30 degrees) the children must too; where it
    // does not (the raw perf mesh has a few such transition quads, repaired by
    // the common post-pass), the children must at least keep the parent quality.
    let (parent_sj, parent_angle) = quad_quality(q, points);
    let sj_floor = parent_sj.min(0.2) - 1.0e-9;
    let angle_floor = parent_angle.min(30.0) - 1.0e-9;
    for child in [q1, q2] {
        let (sj, angle) = quad_quality(child, points);
        if !sj.is_finite() || sj < sj_floor || !angle.is_finite() || angle < angle_floor {
            return Err("the quad split would create a poor quad".to_string());
        }
    }
    quads[index] = q1;
    quads.push(q2);
    queue.push((opposite, n));
    Ok(())
}

/// Close every residual triangle by the refinement cascade.
fn close_leftovers(
    points: &mut Vec<Node>,
    quads: &mut Vec<[usize; 4]>,
    tris: &mut Vec<[usize; 3]>,
) -> Result<(), String> {
    let mut ids: HashMap<(u64, u64), usize> = HashMap::new();
    for (i, p) in points.iter().enumerate() {
        ids.insert(point_key([p[0], p[1]]), i);
    }
    let mut hanging: HashMap<(usize, usize), usize> = HashMap::new();
    let mut operations = 0usize;
    while !tris.is_empty() {
        let mut queue: Vec<((usize, usize), usize)> = Vec::new();
        close_triangle(0, points, quads, tris, &mut ids, &mut hanging, &mut queue)?;
        while let Some((edge, m)) = queue.pop() {
            operations += 1;
            if operations > 20_000 {
                return Err("the refinement cascade did not terminate".to_string());
            }
            if let Some(qid) = find_quad_edge(quads, edge.0, edge.1) {
                split_quad(
                    qid,
                    edge,
                    m,
                    points,
                    quads,
                    &mut ids,
                    &mut hanging,
                    &mut queue,
                )?;
            } else if let Some(tid) = find_tri_edge(tris, edge.0, edge.1) {
                close_triangle(tid, points, quads, tris, &mut ids, &mut hanging, &mut queue)?;
            }
        }
    }
    Ok(())
}

// ------------------------------------------------------------------------- //
// Entry point
// ------------------------------------------------------------------------- //

/// Mesh `domain` with the L-norm frontal Delaunay route (see the module docs).
pub fn mesh(domain: &Domain, sizing: &Sizing) -> Result<Mesh, String> {
    let size = sizing.size;
    if !size.is_finite() || size <= 0.0 {
        return Err("the element size must be positive and finite".to_string());
    }
    if domain.outer.len() < 3 {
        return Err("the outer loop has fewer than three points".to_string());
    }

    let field = CrossField::build(domain, size);
    let loops = sample_boundary(domain, size);
    if loops.is_empty() {
        return Err("the boundary could not be discretized".to_string());
    }

    let mut state = CdtState::new(size);
    for loop_points in &loops {
        for &p in loop_points {
            state.add_vertex(p)?;
        }
    }
    for loop_points in &loops {
        for i in 0..loop_points.len() {
            let a = loop_points[i];
            let b = loop_points[(i + 1) % loop_points.len()];
            let pa = Point2::new(a[0], a[1]);
            let pb = Point2::new(b[0], b[1]);
            state
                .cdt
                .add_constraint_edge(pa, pb)
                .map_err(|e| format!("a boundary edge could not be constrained: {e}"))?;
        }
    }

    frontal_refine(&mut state, domain, &field, size)?;

    let base_tris = state.domain_triangles(domain);
    if base_tris.is_empty() {
        return Err("the frontal triangulation produced no triangle".to_string());
    }
    // The frontal fronts meet at a handful of places where three points are
    // almost collinear; the resulting flat triangles cannot be paired into a
    // convex quad. A constrained Laplacian sweep on the interior nodes (boundary
    // nodes fixed, moves that would invert a triangle refused) removes those
    // degeneracies.
    relax_points(&mut state.points, &base_tris, 4);
    let rebuilt = rebuild_cdt(&state.points, &loops, size)?;
    let final_tris = rebuilt.domain_triangles(domain);
    if final_tris.is_empty() {
        return Err("the relaxed triangulation produced no triangle".to_string());
    }
    let points: Vec<Node> = rebuilt.points.iter().map(|p| [p[0], p[1], 0.0]).collect();
    let (mut quads, mut leftover) = recombine(&final_tris, &points);
    let mut points = points;
    if !leftover.is_empty() {
        // A maximum-cardinality matching over convex pairs can still leave a
        // triangle when the dual graph has no perfect matching. Close it with a
        // local refinement cascade; every new quad is validated, and the whole
        // closure is rolled back if it would produce a bad cell.
        let saved_points = points.clone();
        let saved_quads = quads.clone();
        let saved_tris = leftover.clone();
        if close_leftovers(&mut points, &mut quads, &mut leftover).is_err() {
            points = saved_points;
            quads = saved_quads;
            leftover = saved_tris;
        }
    }
    let mut cells: Vec<Cell> = quads.iter().map(|q| Cell::quad4(*q)).collect();
    for t in &leftover {
        cells.push(Cell::tri3(*t));
    }
    let mesh = Mesh::new(points, cells);
    mesh.validate().map_err(|e| e.to_string())?;
    Ok(mesh)
}

/// Number of quads and triangles in a mesh (small helper for the example).
pub fn counts(mesh: &Mesh) -> (usize, usize) {
    let quads = mesh.cells.iter().filter(|c| c.kind == Kind::Quad4).count();
    let tris = mesh.cells.iter().filter(|c| c.kind == Kind::Tri3).count();
    (quads, tris)
}

// ------------------------------------------------------------------------- //
// Tests
// ------------------------------------------------------------------------- //

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::two_d::Sizing;
    use crate::verify;

    #[test]
    fn distances_are_the_two_norms() {
        let a = [0.0, 0.0];
        let b = [3.0, -4.0];
        assert!((l1_distance(a, b) - 7.0).abs() < 1.0e-12);
        assert!((linf_distance(a, b) - 4.0).abs() < 1.0e-12);
        // A 45-degree unit vector: L∞ length 1, Manhattan length √2.
        let c = [
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
        ];
        assert!((linf_distance(a, c) - std::f64::consts::FRAC_1_SQRT_2).abs() < 1.0e-12);
        assert!((l1_distance(a, c) - std::f64::consts::SQRT_2).abs() < 1.0e-12);
    }

    #[test]
    fn the_circumsquare_of_a_right_isosceles_triangle() {
        // Legs along the axes: (-1,-1), (1,-1), (-1,1).
        let a = [-1.0, -1.0];
        let b = [1.0, -1.0];
        let c = [-1.0, 1.0];
        let (centre, radius) = circumcenter(a, b, c).unwrap();
        // The minimal enclosing square is [-1,1]^2.
        assert!((centre[0] - 0.0).abs() < 1.0e-12, "{centre:?}");
        assert!((centre[1] - 0.0).abs() < 1.0e-12, "{centre:?}");
        assert!((radius - 1.0).abs() < 1.0e-12, "{radius}");
        for p in [a, b, c] {
            assert!((linf_distance(p, centre) - radius).abs() < 1.0e-12);
        }
        // The circumcenter coincides with the midpoint of the hypotenuse.
        let hyp_mid = [0.0, 0.0];
        assert!((centre[0] - hyp_mid[0]).abs() < 1.0e-12);
        assert!((centre[1] - hyp_mid[1]).abs() < 1.0e-12);
    }

    #[test]
    fn the_linf_equilateral_triangle_has_three_equal_sides() {
        // A right isosceles triangle with axis-aligned legs is equilateral in L∞.
        let a = [0.0, 0.0];
        let b = [2.0, 0.0];
        let apex = linf_equilateral_apex(a, b);
        assert!((apex[0] - 0.0).abs() < 1.0e-12);
        assert!((apex[1] - 2.0).abs() < 1.0e-12);
        let l1 = linf_distance(a, b);
        let l2 = linf_distance(a, apex);
        let l3 = linf_distance(b, apex);
        assert!((l1 - 2.0).abs() < 1.0e-12);
        assert!((l2 - l1).abs() < 1.0e-12);
        assert!((l3 - l1).abs() < 1.0e-12);
    }

    #[test]
    fn the_cross_field_of_a_rectangle_is_axis_aligned() {
        let domain = Domain::rectangle(10.0, 6.0).unwrap();
        let field = CrossField::build(&domain, 1.0);
        // Any multiple of π/2 is the same cross.
        for p in [[2.0, 2.0], [5.0, 3.0], [8.0, 5.0]] {
            let angle = field.angle(p);
            let quarter = (angle / std::f64::consts::FRAC_PI_2).round();
            let residual = (angle - quarter * std::f64::consts::FRAC_PI_2).abs();
            assert!(residual < 1.0e-6, "angle {angle} at {p:?}");
        }
    }

    #[test]
    fn a_rectangle_becomes_all_quads_and_passes_the_gate() {
        let domain = Domain::rectangle(12.0, 8.0).unwrap();
        let sizing = Sizing::uniform(2.0).unwrap();
        let mesh = mesh(&domain, &sizing).unwrap();
        let (quads, tris) = counts(&mesh);
        assert!(quads > 0, "no quad");
        assert_eq!(tris, 0, "{tris} triangles left");
        let gate = verify::gate(&mesh);
        assert!(gate.passes(), "{gate:?} {:?}", gate.reason());
        assert!(verify::boundary_offset(&domain, &mesh) < 1.0e-9);
    }

    #[test]
    fn the_plate_with_a_hole_is_quad_dominant() {
        let domain = Domain::plate_with_hole(20.0, 12.0, 2.0, 8).unwrap();
        let sizing = Sizing::uniform(1.0).unwrap();
        let mesh = mesh(&domain, &sizing).unwrap();
        let (quads, tris) = counts(&mesh);
        assert!(quads > 0);
        assert_eq!(tris, 0, "{tris} triangles left");
        let gate = verify::gate(&mesh);
        assert!(gate.passes(), "{gate:?} {:?}", gate.reason());
        assert!(verify::boundary_offset(&domain, &mesh) < 1.0e-9);
    }

    #[test]
    fn a_bad_size_is_refused() {
        let domain = Domain::rectangle(1.0, 1.0).unwrap();
        let sizing = Sizing::uniform(1.0).unwrap();
        let mut bad = sizing;
        bad.size = 0.0;
        assert!(mesh(&domain, &bad).is_err());
    }
}
