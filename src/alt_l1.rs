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
//! 3. **Frontal Delaunay point insertion** (§5), in the clean-room CDT. Triangles are
//!    classified as resolved/active/waiting by their L∞ circumradius; front
//!    edges of the worst active triangle receive a new point at the far corner
//!    of the local cross-frame circumsquare. A standard L2 Delaunay kernel
//!    connects the points (the paper observes that the L2 and L∞ kernels give
//!    nearly the same triangulation, §5).
//! 4. **Recombination** (§7 of the paper). The resulting
//!    right-triangle-dominated mesh is recombined into quads with a
//!    maximum-cardinality matching of the triangle dual graph; edge weights are
//!    the scaled Jacobians quantised to a coarse scale (1..=101). The matching
//!    has two stages on large meshes, because the exact weighted Blossom
//!    (`mwmatching`) costs O(n^3) and needs ~55 s on the 30k-triangle
//!    performance part: the exact weighted Blossom runs on spatial blocks small
//!    enough to be cheap, then a quality-first greedy matching plus an
//!    unweighted Edmonds Blossom handle the thin interface band and restore
//!    maximum cardinality. Meshes up to `EXACT_MATCHING_TRIANGLES` use the
//!    exact weighted Blossom unchanged, so their measured table is reproduced.
//!    This is the paper's indirect step: the L-norm work is in the point
//!    placement, not in the recombination.
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
//! # Measured result and cost
//!
//! The measured table (`examples/alt_l1.rs`) is all-quad 7/7 raw and 7/7 gate
//! PASS after the common post-pass. Post-pass quality is the best of the
//! shipped routes: min scaled Jacobian 0.66–0.98, min corner angle 41.6–78.7
//! degrees and v4 77–100 %. Two raw rows still fail on the minimum scaled
//! Jacobian (plate-hole 1 mm 0.102; the 0.6 mm performance part 0.091); the
//! common post-pass lifts both. No triangle is silently dropped: the
//! refinement cascade adds a small number of quads locally (e.g. +59 on
//! plate-hole 2 mm, +269 on the 0.6 mm performance part) and can raise the
//! local aspect ratio (measured post: p95 1.2–2.3, max ~8), which is the price
//! of keeping every cell conforming and every new quad convex.
//!
//! **The 17x speedup.** The whole-graph weighted Blossom dominated the 0.6 mm
//! raw row: 56.96 s. The blocked matching used above
//! `EXACT_MATCHING_TRIANGLES` takes 3.32 s on the same part (17x) with the
//! same 7/7 post result. Only that ~31k-triangle part crosses the threshold, so
//! the six smaller parts still use the exact Blossom and reproduce their
//! measured rows unchanged. The block solve is a maximum-weight, not
//! maximum-cardinality, solve; the in-house Edmonds pass restores maximum
//! cardinality globally (see `blocked_matching`).
//!
//! [`Options`] exposes the two paper knobs, `h_max` (adimensional size
//! `2 R_inf / size`) and `min_spacing_fraction` (fraction of `size`); their
//! defaults reproduce the measured table. Neither is validated: a non-positive
//! `h_max` leaves every front unresolved until the round cap, and a negative
//! spacing fraction disables the spacing filter. The insertion filter is the
//! project's own approximation (see above), not the paper's eq. 12–14 clamping.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::delaunay::{CdtAdapter, CdtLocation};
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

/// Vector difference `a - b` (length units).
fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

/// Vector sum `a + b` (length units).
fn add(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

/// Scalar-vector product `s * a`.
fn scale(a: [f64; 2], s: f64) -> [f64; 2] {
    [a[0] * s, a[1] * s]
}

/// Euclidean dot product.
fn dot(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

/// Squared Euclidean length `a . a` (avoids a square root in comparisons).
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
    /// World x coordinate of grid node `(0, 0)` (length units). The grid extends
    /// `2 * size` beyond the domain box so bilinear interpolation is defined at
    /// every query point inside the domain.
    x0: f64,

    /// World y coordinate of grid node `(0, 0)` (length units).
    y0: f64,

    /// Grid spacing (length units), `max(0.5 * size, 1e-9)`.
    cell: f64,

    /// Number of grid nodes in x; at least 2. The `a`/`b` arrays have
    /// `nx * ny` entries.
    nx: usize,

    /// Number of grid nodes in y; at least 2.
    ny: usize,

    /// `cos(4 * theta)` at each grid node, row-major (`j * nx + i`). Together
    /// with `b` this is the double-angle cross: a unit vector invariant under
    /// the four branches `theta -> theta + k * pi/2`.
    a: Vec<f64>,

    /// `sin(4 * theta)` at each grid node; second component of the double-angle
    /// cross. Length `nx * ny`.
    b: Vec<f64>,
}

impl CrossField {
    /// Build the cross field.
    ///
    /// Boundary conditions: every grid node starts with the tangent of its
    /// nearest boundary segment, and a layer of about 1.5 grid cells keeps that
    /// value fixed. The rest of the grid is smoothed by 300 Gauss-Seidel
    /// Laplace sweeps on `(cos 4θ, sin 4θ)`; the double-angle form makes the
    /// four branches of the cross the same value. A degenerate domain (no
    /// finite point) yields an empty field whose [`CrossField::angle`] is 0.
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

/// Euclidean distance from `p` to the closed segment `a..b` (length units);
/// a zero-length segment reduces to the point distance. Used only to pick the
/// nearest boundary tangent for the cross field.
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

/// Uniform bucket grid for the "distance to the nearest existing node" query
/// used by the spacing filter.
struct PointGrid {
    /// Bucket side (length units), `max(cell, 1e-9)`; the insertion query uses
    /// `0.5 * size`, so with the default spacing fraction (0.5) a node closer
    /// than the threshold always falls in one of the 3x3 neighbouring buckets.
    /// A larger `min_spacing_fraction` is not guaranteed (see
    /// [`PointGrid::nearest`]).
    cell: f64,

    /// Bucket coordinates `(floor(x / cell), floor(y / cell))` -> point indices.
    /// Every inserted point appears in exactly one bucket; buckets are created
    /// lazily. The indices refer to the caller's `points` array.
    buckets: HashMap<(i64, i64), Vec<usize>>,
}

impl PointGrid {
    /// A bucket grid with side `cell` (length units); a non-positive or
    /// non-finite value is clamped to `1e-9` so [`PointGrid::key`] never
    /// divides by zero.
    fn new(cell: f64) -> PointGrid {
        PointGrid {
            cell: cell.max(1.0e-9),
            buckets: HashMap::new(),
        }
    }

    /// Integer bucket coordinate `(floor(x / cell), floor(y / cell))`; `i64`
    /// because the grid also holds points outside the domain box.
    fn key(&self, p: [f64; 2]) -> (i64, i64) {
        (
            (p[0] / self.cell).floor() as i64,
            (p[1] / self.cell).floor() as i64,
        )
    }

    /// Insert the point index `id` at `p`. The caller must not insert the same
    /// id twice (a duplicate would only repeat the distance in a query).
    fn insert(&mut self, id: usize, p: [f64; 2]) {
        self.buckets.entry(self.key(p)).or_default().push(id);
    }

    /// Distance from `p` to the nearest inserted point (length units), or
    /// `+inf` when the 3x3 bucket neighbourhood is empty.
    ///
    /// The 3x3 neighbourhood is sufficient for the **default** spacing filter:
    /// its threshold is `0.5 * size`, exactly one bucket side, and a point
    /// closer than that cannot be two buckets away in either axis. With
    /// `Options::min_spacing_fraction > 0.5` the true nearest point can lie
    /// outside the searched buckets, so the returned value is only an upper
    /// bound on the true nearest distance and the requested spacing is not
    /// enforced (noted in the report; the measured table uses the default).
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

/// The clean-room constrained Delaunay triangulation used by the route.
type Cdt = CdtAdapter;

/// Exact bit-pattern key of a point, used for vertex deduplication: two points
/// are the same only when both coordinates have identical bits.
fn point_key(p: [f64; 2]) -> (u64, u64) {
    (p[0].to_bits(), p[1].to_bits())
}

/// The triangulation plus the bookkeeping the frontal loop needs: a stable
/// point vector, a duplicate map and the spacing grid.
struct CdtState {
    /// The constrained Delaunay triangulation; its vertices are the same points
    /// as `points`, in insertion order.
    cdt: Cdt,

    /// Index -> coordinates (length units, z = 0 not stored). A point is
    /// appended once and never removed, so an index is stable for the run.
    points: Vec<[f64; 2]>,

    /// Exact bit-pattern key -> index into `points`. Exact keys (not a
    /// tolerance) mean "duplicate" is the same coordinate, which is what the
    /// adapter also treats as a duplicate vertex.
    index: HashMap<(u64, u64), usize>,

    /// Uniform grid over `points` for the minimum-spacing query (bucket side
    /// `0.5 * size`).
    grid: PointGrid,
}

impl CdtState {
    fn new(
        size: f64,
        outer: Vec<[f64; 2]>,
        holes: Vec<Vec<[f64; 2]>>,
        extra_points: Vec<[f64; 2]>,
    ) -> Result<CdtState, String> {
        let cdt = Cdt::from_loops_and_points(outer, holes, extra_points)
            .map_err(|error| format!("the boundary could not be triangulated: {error}"))?;
        let points = cdt.points().to_vec();
        let index = points
            .iter()
            .enumerate()
            .map(|(id, &point)| (point_key(point), id))
            .collect();
        let mut grid = PointGrid::new(0.5 * size);
        for (id, &point) in points.iter().enumerate() {
            grid.insert(id, point);
        }
        Ok(CdtState {
            cdt,
            points,
            index,
            grid,
        })
    }

    /// Insert a vertex if it is not already present. Returns its index.
    fn add_vertex(&mut self, p: [f64; 2]) -> Result<usize, String> {
        let key = point_key(p);
        if let Some(&id) = self.index.get(&key) {
            return Ok(id);
        }
        self.cdt
            .insert_point(p)
            .map_err(|error| format!("the CDT could not insert a vertex: {error}"))?;
        let id = self.points.len();
        self.points.push(p);
        self.index.insert(key, id);
        self.grid.insert(id, p);
        Ok(id)
    }

    /// Index of the `CdtState` point at a CDT position, or `None` when the
    /// position is not one of the inserted vertices.
    fn id_of(&self, p: [f64; 2]) -> Option<usize> {
        self.index.get(&point_key(p)).copied()
    }

    /// Triangles of the domain, counter-clockwise, with `CdtState` indices.
    /// The adapter removes exterior and hole faces; the centroid check retains
    /// the route's defensive domain postcondition.
    fn domain_triangles(&self, domain: &Domain) -> Vec<[usize; 3]> {
        let mut out = Vec::new();
        for triangle in self.cdt.triangles() {
            let pos = triangle.vertices.map(|vertex| self.cdt.points()[vertex]);
            let area = (pos[1][0] - pos[0][0]) * (pos[2][1] - pos[0][1])
                - (pos[1][1] - pos[0][1]) * (pos[2][0] - pos[0][0]);
            if area.abs() <= 1.0e-14 {
                continue;
            }
            let cx = (pos[0][0] + pos[1][0] + pos[2][0]) / 3.0;
            let cy = (pos[0][1] + pos[1][1] + pos[2][1]) / 3.0;
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

/// Options for the L∞ frontal Delaunay route.
///
/// The defaults reproduce the measured benchmark table exactly.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Options {
    /// `h_max` of the paper (default `H_MAX` = `4/3`, dimensionless): a
    /// triangle is resolved when its adimensional size `2 R_inf / size` is at
    /// most this. Lower values refine more (roughly quadrupling the triangle
    /// count when halved); values near 1 leave almost nothing resolved, and the
    /// insertion loop then stops at its round cap. Invariant: positive and
    /// finite; the route does not validate it.
    pub h_max: f64,

    /// Minimum distance from a candidate point to every existing node, as a
    /// fraction of the target `size` (default `MIN_SPACING_FRACTION` = 0.5,
    /// dimensionless). It prevents an insertion that would only create a
    /// sliver. `0` disables the filter; larger values make the front sparser.
    /// Invariant: finite and non-negative; the route does not validate it.
    /// Values above 0.5 are only partially enforced: the spacing query searches
    /// the 3x3 bucket neighbourhood around the candidate (see
    /// `PointGrid::nearest`), which covers one bucket side = `0.5 * size`.
    pub min_spacing_fraction: f64,
}

impl Default for Options {
    /// The measured defaults (`H_MAX`, `MIN_SPACING_FRACTION`), which
    /// reproduce the benchmark table.
    fn default() -> Self {
        Self {
            h_max: H_MAX,
            min_spacing_fraction: MIN_SPACING_FRACTION,
        }
    }
}

/// Refine a CDT with the frontal L∞ point insertion of §5.
///
/// The target is a triangle whose L∞ circumsquare has side `size`, i.e. the
/// circumsquare radius is `size / 2` (this is the size of half a square cell of
/// edge `size`). The paper's `h_max = 4/3` accepts a triangle up to
/// `R_inf = (2/3) * size`, so the target is the ideal, not the acceptance
/// bound. New points are placed at the far corner of the local cross-frame
/// circumsquare of a front edge — the construction of Figure 13.
fn frontal_refine(
    state: &mut CdtState,
    domain: &Domain,
    field: &CrossField,
    size: f64,
    options: &Options,
) -> Result<(), String> {
    // A safety cap, not a target: the loop exits as soon as a round inserts
    // nothing. 4000 is well above the measured worst case on the corpus; a run
    // that reaches it returns an error rather than looping forever.
    let max_rounds = 4000;
    for _round in 0..max_rounds {
        let tris = state.domain_triangles(domain);
        if tris.is_empty() {
            return Err("the frontal triangulation lost every triangle".to_string());
        }
        // Classify every triangle by its adimensional L-infinity size
        // `hi = 2 R_inf / size`; `hi <= h_max` means resolved.
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
            resolved[t] = hi[t] <= options.h_max;
        }

        let mut edges: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for (t, tri) in tris.iter().enumerate() {
            for k in 0..3 {
                let a = tri[k];
                let b = tri[(k + 1) % 3];
                edges.entry((a.min(b), a.max(b))).or_default().push(t);
            }
        }

        // An unresolved triangle is active only if one of its edges is a front
        // edge: a boundary edge (one owner) or the edge of an already resolved
        // triangle. A triangle surrounded by unresolved neighbours is waiting,
        // not active, so its front will move when a neighbour is refined.
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

        // Worst first: refining the largest active triangle keeps the front
        // from being starved by a cluster of slightly-too-large neighbours.
        let mut order: Vec<usize> = (0..n).filter(|&t| active[t]).collect();
        order.sort_by(|&a, &b| hi[b].total_cmp(&hi[a]));

        let mut inserted = 0usize;
        for t in order {
            let tri = tris[t];
            // Of the triangle's front edges, use the longest one: it is the one
            // whose circumsquare is furthest from the target size.
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
            // Figure 13's construction, in the local cross frame of the edge.
            let Some(candidate) = place_candidate(pa, pb, pc, field, size) else {
                continue;
            };
            if !domain.contains(candidate) {
                continue;
            }
            // Spacing filter: a candidate closer than the fraction to any
            // existing node would only create a sliver or a duplicate.
            let nearest = state.grid.nearest(candidate, &state.points);
            if nearest < options.min_spacing_fraction * size {
                continue;
            }
            // The candidate must land in a still-unresolved triangle: inserting
            // into a resolved region would only create a sliver.
            match state.cdt.locate(candidate) {
                // A candidate on an interior edge is a perfectly good Delaunay
                // insertion (and on a boundary edge it splits the constraint);
                // only an existing vertex is a duplicate.
                CdtLocation::Edge(_) => {}
                CdtLocation::Triangle(face) => {
                    let Some(triangle) = state.cdt.triangles().get(face) else {
                        continue;
                    };
                    let pos = triangle.vertices.map(|vertex| state.cdt.points()[vertex]);
                    let centroid = [
                        (pos[0][0] + pos[1][0] + pos[2][0]) / 3.0,
                        (pos[0][1] + pos[1][1] + pos[2][1]) / 3.0,
                    ];
                    let theta = field.angle(centroid);
                    let mut lo = [f64::INFINITY; 2];
                    let mut up = [f64::NEG_INFINITY; 2];
                    for q in &pos {
                        let local = rotate(sub(*q, centroid), -theta);
                        for d in 0..2 {
                            lo[d] = lo[d].min(local[d]);
                            up[d] = up[d].max(local[d]);
                        }
                    }
                    let radius = 0.5 * (up[0] - lo[0]).max(up[1] - lo[1]);
                    if 2.0 * radius / size <= options.h_max {
                        continue;
                    }
                }
                CdtLocation::Vertex(_) | CdtLocation::Outside => continue,
            }
            if state.add_vertex(candidate).is_err() {
                continue;
            }
            inserted += 1;
        }
        // No insertion means every active front is now resolved or blocked by
        // the domain/spacing filters: the refinement has converged.
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

/// 2D cross product, positive when `b` is counter-clockwise from `a`; used for
/// the side-of-edge tests of [`place_candidate`].
fn cross(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

/// Exact-coordinate proximity with a `1e-12` tolerance (length units); used to
/// avoid choosing a square corner that is already a triangle vertex.
fn near(a: [f64; 2], b: [f64; 2]) -> bool {
    (a[0] - b[0]).abs() <= 1.0e-12 && (a[1] - b[1]).abs() <= 1.0e-12
}

/// Map a point from the local cross frame back to world coordinates: rotate by
/// `theta` and translate by `origin`.
fn to_world(p: [f64; 2], origin: [f64; 2], theta: f64) -> [f64; 2] {
    add(origin, rotate(p, theta))
}

// ------------------------------------------------------------------------- //
// Recombination and scaling
// ------------------------------------------------------------------------- //

/// Constrained Laplacian relaxation of interior triangle vertices.
///
/// Each interior node moves halfway towards the average of its neighbours, and
/// a move is refused when it would make any incident triangle degenerate or
/// clockwise. Boundary nodes (endpoints of an edge used by one triangle) are
/// fixed, so the domain shape and the front's boundary points do not move. This
/// is a project choice (the paper's §7 smoothing is centroidal Voronoi in Lp);
/// it is used here only to remove the few flat triangles where the fronts meet.
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
    // A boundary edge is used by exactly one triangle; both endpoints are
    // fixed. Interior edges have two owners.
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
    // Jacobi-style sweeps: every move in a sweep is evaluated against the
    // positions at the start of the sweep, so the result does not depend on
    // node order.
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
///
/// The points are re-inserted in their existing order (so indices stay stable)
/// and every boundary edge is re-applied as a constraint; the relaxation can
/// otherwise lose the constrained-Delaunay property at the boundary.
fn rebuild_cdt(
    points: &[[f64; 2]],
    loops: &[Vec<[f64; 2]>],
    size: f64,
) -> Result<CdtState, String> {
    let Some((outer, holes)) = loops.split_first() else {
        return Err("the boundary could not be re-constrained".to_string());
    };
    CdtState::new(size, outer.clone(), holes.to_vec(), points.to_vec())
}

/// The quad formed by two triangles that share an edge, oriented CCW.
///
/// The shared edge must be traversed in opposite directions by the two
/// triangles (`u -> v` in `t1`, `v -> u` in `t2`), which is the conforming
/// neighbour relation. The four boundary nodes in order are the two shared
/// endpoints and the two opposite corners. A zero-area pair yields `None`.
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
                // Normalize to CCW. The area is twice the polygon area; a
                // numerically zero pair is not a quad and is refused.
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

/// `(min corner scaled Jacobian, min corner angle in degrees, max corner angle
/// in degrees)` of a quad. The recombination floor uses the first two; the max
/// angle rejects a folded pair whose concave corner is near 180 degrees.
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

// ------------------------------------------------------------------------- //
// Recombination: maximum-cardinality matching of adjacent triangles
// ------------------------------------------------------------------------- //

/// Sentinel of the triangle matching (a free triangle).
const UNMATCHED: usize = usize::MAX;

/// Up to this many triangles the exact weighted Blossom ([`weighted_matching`])
/// is affordable and is used unchanged, so the route's measured table is
/// reproduced where its cost is small. Above it the mesh is matched block by
/// block (`blocked_matching`).
const EXACT_MATCHING_TRIANGLES: usize = 12_000;

/// Target number of triangles per block of the large-mesh matching. The exact
/// weighted Blossom is cubic-ish, so a block of ~12k triangles costs a fraction
/// of the whole mesh while still giving most of the mesh the weighted matching.
const BLOCK_TRIANGLES: usize = 12_000;

/// Half-width of the block-interface band, in mean triangle edge lengths.
const BAND_EDGES: f64 = 2.0;

/// Candidate recombination edges: an interior edge shared by exactly two
/// triangles whose union is a convex quad passing the quality floor, returned as
/// `(triangle_a, triangle_b, weight)` with `a < b`.
///
/// The floor is deliberately loose (SJ 0.05, 10°): it removes only the pairs
/// that are not a quad at all, because every pairing that is achievable is one
/// more quad in the final mesh, and the common post-pass repairs shape. The
/// integer weight encodes the scaled Jacobian and is the matching's quality
/// tie-break.
fn recombination_edges(tris: &[[usize; 3]], points: &[Node]) -> Vec<(usize, usize, i32)> {
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
        let (ta, tb) = (owners[0], owners[1]);
        let Some(quad) = pair_quad(tris[ta], tris[tb], points) else {
            continue;
        };
        let (sj, min_angle, max_angle) = quad_metrics(quad, points);
        if !sj.is_finite() || sj < 0.05 || min_angle < 10.0 || max_angle > 175.0 {
            continue;
        }
        let weight = 1 + (sj * 100.0).round().clamp(0.0, 100.0) as i32;
        edges.push((ta, tb, weight));
    }
    edges.sort_unstable();
    edges
}

/// A deterministic, quality-first maximal matching: the heaviest edge is taken
/// first, ties broken by index. It is the starting point of the fast strategy
/// and already fixes the shape of almost every quad.
fn greedy_matching(n: usize, edges: &[(usize, usize, i32)]) -> Vec<usize> {
    let mut mate = vec![UNMATCHED; n];
    let mut order: Vec<usize> = (0..edges.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        edges[b]
            .2
            .cmp(&edges[a].2)
            .then_with(|| edges[a].0.cmp(&edges[b].0))
            .then_with(|| edges[a].1.cmp(&edges[b].1))
    });
    for &i in &order {
        let (a, b, _) = edges[i];
        if mate[a] == UNMATCHED && mate[b] == UNMATCHED {
            mate[a] = b;
            mate[b] = a;
        }
    }
    mate
}

/// Mean edge length of the triangulation (the interface-band width unit).
fn edge_scale(tris: &[[usize; 3]], points: &[Node]) -> f64 {
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for tri in tris {
        for k in 0..3 {
            let a = tri[k];
            let b = tri[(k + 1) % 3];
            sum += ((points[a][0] - points[b][0]).powi(2) + (points[a][1] - points[b][1]).powi(2))
                .sqrt();
            count += 1;
        }
    }
    if count == 0 {
        1.0
    } else {
        sum / count as f64
    }
}

/// An ordering key for the free vertices of the augmenting search: the smaller
/// the value, the worse the triangle would be closed by the refinement cascade,
/// and the earlier its search runs. A triangle that closes well may stay free;
/// one whose centroid split would make a poor quad should be matched instead.
/// This is what keeps the residual-triangle closure succeeding.
fn free_vertex_scores(tris: &[[usize; 3]], points: &[Node]) -> Vec<f64> {
    tris.iter()
        .map(|&t| {
            let cell = Cell::tri3(t);
            let angles = quality::corner_angles(&cell, points);
            let min = angles.iter().copied().fold(f64::INFINITY, f64::min);
            if min.is_finite() {
                min
            } else {
                f64::NEG_INFINITY
            }
        })
        .collect()
}

/// Maximum-cardinality matching in an unweighted general graph (Edmonds'
/// blossom: alternating-tree BFS with odd-cycle contraction).
///
/// The starting matching comes from [`greedy_matching`], so the search only
/// augments the few vertices greediness left free, and the quads it breaks are
/// only those on a genuine augmenting path. Repeating full passes until a pass
/// augments nothing makes the result maximum: a matching is maximum exactly
/// when no augmenting path exists, and every augmenting path starts at a free
/// vertex.
///
/// Edmonds' blossom state for the unweighted maximum-cardinality repair of the
/// greedy matching.
///
/// The implementation is the classic alternating-tree BFS with odd-cycle
/// contraction; it is in-house so the route has no external general-graph
/// dependency for the hot path. All indices are triangle indices.
struct Blossom<'a> {
    /// Adjacency of the triangle dual graph (only recombination candidates);
    /// `adj[t]` lists the partners of triangle `t`.
    adj: &'a [Vec<usize>],

    /// `mate[t]` is the partner triangle of `t`, or [`UNMATCHED`]. The array is
    /// symmetric (`mate[mate[t]] == t`) after every successful augmentation.
    mate: Vec<usize>,

    /// Parent in the alternating tree, [`UNMATCHED`] for a root or an unvisited
    /// vertex. The parent of an outer vertex is an inner vertex and vice versa.
    parent: Vec<usize>,

    /// Base vertex of the blossom a vertex currently belongs to (the
    /// representative after odd-cycle contraction); `base[v] == v` outside a
    /// blossom.
    base: Vec<usize>,

    /// Colour of a vertex in the current alternating tree: outer (true) or
    /// inner/unvisited (false).
    outer: Vec<bool>,

    /// Scratch marks for [`Blossom::lca`], with a generation `stamp` so the
    /// array does not need an O(n) clear per search.
    seen: Vec<u32>,

    /// Current generation for `seen`; wrapping, with a full clear on wrap.
    stamp: u32,

    /// Scratch flags of the vertices on the blossom path in [`mark_path`].
    blossom: Vec<bool>,

    /// BFS queue of outer vertices; a moving head index avoids O(n) pops.
    queue: Vec<usize>,
}

impl<'a> Blossom<'a> {
    /// A blossom state for `n` triangles with the given adjacency and starting
    /// matching; the working arrays are allocated once and reused by
    /// [`Blossom::run`].
    fn new(n: usize, adj: &'a [Vec<usize>], mate: Vec<usize>) -> Self {
        Self {
            adj,
            mate,
            parent: vec![UNMATCHED; n],
            base: (0..n).collect(),
            outer: vec![false; n],
            seen: vec![0; n],
            stamp: 0,
            blossom: vec![false; n],
            queue: Vec::with_capacity(n),
        }
    }

    /// Augment until no augmenting path is left. Free vertices are searched
    /// worst-first (`scores` ascending): a vertex that cannot stay free in any
    /// maximum matching is matched by its own search, so the vertices that
    /// remain free are the most closeable ones.
    fn run(mut self, scores: &[f64]) -> Vec<usize> {
        let n = self.base.len();
        loop {
            let mut roots: Vec<usize> = (0..n).filter(|&v| self.mate[v] == UNMATCHED).collect();
            roots
                .sort_unstable_by(|&a, &b| scores[a].total_cmp(&scores[b]).then_with(|| a.cmp(&b)));
            let mut augmented = false;
            for root in roots {
                if self.mate[root] == UNMATCHED && self.find_path(root) {
                    augmented = true;
                }
            }
            if !augmented {
                return self.mate;
            }
        }
    }

    /// Lowest common ancestor of two outer vertices in the alternating tree.
    fn lca(&mut self, a: usize, b: usize) -> Option<usize> {
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.seen.iter_mut().for_each(|s| *s = 0);
            self.stamp = 1;
        }
        let stamp = self.stamp;
        let mut a = a;
        loop {
            a = self.base[a];
            self.seen[a] = stamp;
            let partner = self.mate[a];
            if partner == UNMATCHED {
                break;
            }
            let up = self.parent[partner];
            if up == UNMATCHED {
                break;
            }
            a = up;
        }
        let mut b = b;
        loop {
            b = self.base[b];
            if self.seen[b] == stamp {
                return Some(b);
            }
            let partner = self.mate[b];
            if partner == UNMATCHED {
                return None;
            }
            let up = self.parent[partner];
            if up == UNMATCHED {
                return None;
            }
            b = up;
        }
    }

    /// Mark the alternating-tree path from `v` up to base `b` as one blossom,
    /// re-parenting it towards `child`.
    fn mark_path(&mut self, mut v: usize, b: usize, mut child: usize) {
        let mut guard = 0usize;
        while self.base[v] != b {
            guard += 1;
            if guard > self.base.len() {
                return;
            }
            let partner = self.mate[v];
            if partner == UNMATCHED {
                return;
            }
            self.blossom[self.base[v]] = true;
            self.blossom[self.base[partner]] = true;
            self.parent[v] = child;
            child = partner;
            let up = self.parent[partner];
            if up == UNMATCHED {
                return;
            }
            v = up;
        }
    }

    /// One alternating-tree search from the free vertex `root`; augments the
    /// matching and returns `true` when an augmenting path was found.
    fn find_path(&mut self, root: usize) -> bool {
        let n = self.base.len();
        for i in 0..n {
            self.parent[i] = UNMATCHED;
            self.base[i] = i;
            self.outer[i] = false;
        }
        self.outer[root] = true;
        self.queue.clear();
        self.queue.push(root);
        let mut head = 0usize;
        while head < self.queue.len() {
            let v = self.queue[head];
            head += 1;
            for i in 0..self.adj[v].len() {
                let u = self.adj[v][i];
                if self.base[v] == self.base[u] || self.mate[v] == u {
                    continue;
                }
                let partner = self.mate[u];
                let u_is_outer =
                    u == root || (partner != UNMATCHED && self.parent[partner] != UNMATCHED);
                if u_is_outer {
                    let Some(curbase) = self.lca(v, u) else {
                        continue;
                    };
                    self.blossom.iter_mut().for_each(|b| *b = false);
                    self.mark_path(v, curbase, u);
                    self.mark_path(u, curbase, v);
                    for w in 0..n {
                        if self.blossom[self.base[w]] {
                            self.base[w] = curbase;
                            if !self.outer[w] {
                                self.outer[w] = true;
                                self.queue.push(w);
                            }
                        }
                    }
                } else if self.parent[u] == UNMATCHED {
                    self.parent[u] = v;
                    if partner == UNMATCHED {
                        // Collect the alternating path first: a malformed tree
                        // (which should not happen) must not leave a half-flipped
                        // matching behind. The path is `inner - outer - inner - …`
                        // from the free vertex `u` up to the root.
                        let mut steps: Vec<(usize, usize)> = Vec::new();
                        let mut cur = u;
                        let mut seen = vec![false; n];
                        loop {
                            let up = self.parent[cur];
                            if up == UNMATCHED || steps.len() > n || seen[cur] || seen[up] {
                                return false;
                            }
                            seen[cur] = true;
                            seen[up] = true;
                            steps.push((cur, up));
                            let next = self.mate[up];
                            if next == UNMATCHED {
                                break;
                            }
                            cur = next;
                        }
                        for &(inner, outer) in &steps {
                            self.mate[inner] = outer;
                            self.mate[outer] = inner;
                        }
                        return true;
                    }
                    self.outer[partner] = true;
                    self.queue.push(partner);
                }
            }
        }
        false
    }
}

/// Rebuild the quads from a `mate` array (one entry per triangle, `UNMATCHED`
/// when free), re-deriving each quad from the two triangles.
fn quads_from_mate(
    tris: &[[usize; 3]],
    points: &[Node],
    mate: &[usize],
) -> (Vec<[usize; 4]>, Vec<[usize; 3]>) {
    let mut used = vec![false; tris.len()];
    let mut quads = Vec::new();
    // Handle each pair once (`partner > t`); a pair whose `pair_quad` fails
    // (zero area, non-convex) leaves both triangles free for the cascade.
    for t in 0..tris.len() {
        let partner = mate.get(t).copied().unwrap_or(UNMATCHED);
        if used[t] || partner == UNMATCHED || partner >= tris.len() || used[partner] || partner <= t
        {
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

/// The reference weighted maximum-cardinality Blossom from `mwmatching`.
///
/// This is the matching the route was measured with, and it is the route's
/// quality reference: it maximises the number of quads and, among those
/// matchings, the total scaled Jacobian. It is kept for meshes small enough that
/// its O(n^3) cost is invisible, which is exactly where it can be used without
/// slowing the route.
fn weighted_matching(n: usize, edges: &[(usize, usize, i32)]) -> Vec<usize> {
    if edges.is_empty() {
        return vec![UNMATCHED; n];
    }
    // `mwmatching` maximises cardinality first and total weight among those
    // matchings; the weight is the quantized scaled Jacobian from
    // [`recombination_edges`], so the result prefers the better-shaped quads.
    let mut matching = mwmatching::Matching::new(edges.to_vec());
    matching.max_cardinality();
    let mate = matching.solve();
    let mut out = vec![UNMATCHED; n];
    for (t, slot) in out.iter_mut().enumerate() {
        *slot = mate.get(t).copied().unwrap_or(UNMATCHED);
    }
    out
}

/// Number of spatial blocks for the large-mesh matching.
fn block_count(n: usize) -> usize {
    n.div_ceil(BLOCK_TRIANGLES).max(1)
}

/// Match a large mesh block by block with the weighted Blossom, then fix the
/// thin interface band with the fast greedy + unweighted Blossom and restore
/// global maximum cardinality.
///
/// The weighted Blossom is super-linear (measured 56.96 s on the 30k-triangle
/// 0.6 mm performance part), so the mesh is cut into spatial blocks whose
/// interior is small enough to match exactly; only the band around the cuts,
/// plus whatever the blocks leave free, is matched by the fast strategy. The
/// blocked path measured 3.32 s on that part (17x).
///
/// The block solve is deliberately a **maximum-weight** matching, not a
/// maximum-cardinality one: forcing every block to pair all of its triangles
/// would lock in low-quality pairs along the block boundary and prevent the
/// band pass from finding better cross-cut quads. The final unweighted Blossom
/// restores maximum cardinality over the whole graph, which is what the
/// all-quad closure requires.
#[allow(clippy::too_many_arguments)]
fn blocked_matching(
    tris: &[[usize; 3]],
    points: &[Node],
    edges: &[(usize, usize, i32)],
    adj: &[Vec<usize>],
    scores: &[f64],
    band_width: f64,
    blocks: usize,
) -> Vec<usize> {
    let n = tris.len();
    let mut mate = vec![UNMATCHED; n];
    if blocks <= 1 {
        return weighted_matching(n, edges);
    }
    // The mesh is cut along x only. The measured parts are much wider than
    // tall (100 x 60 plate), so x cuts give balanced blocks; a 2D kd-tree would
    // divide unusual domains more evenly but is not needed for this corpus.
    let centroids: Vec<f64> = tris
        .iter()
        .map(|t| (points[t[0]][0] + points[t[1]][0] + points[t[2]][0]) / 3.0)
        .collect();
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for &x in &centroids {
        lo = lo.min(x);
        hi = hi.max(x);
    }
    if !hi.is_finite() || !lo.is_finite() || hi <= lo {
        let greedy = greedy_matching(n, edges);
        return Blossom::new(n, adj, greedy).run(scores);
    }
    // Equal-width blocks; the cut positions themselves are fixed, and a
    // triangle whose centroid is within `band_width` of any cut is left to the
    // fast band pass.
    let width = (hi - lo) / blocks as f64;
    let cuts: Vec<f64> = (1..blocks).map(|i| lo + width * i as f64).collect();
    let in_band: Vec<bool> = centroids
        .iter()
        .map(|&x| cuts.iter().any(|&c| (x - c).abs() <= band_width))
        .collect();
    let block_of = |x: f64| -> usize {
        let mut b = 0usize;
        for (i, &c) in cuts.iter().enumerate() {
            if x > c {
                b = i + 1;
            }
        }
        b
    };
    let mut local_of = vec![UNMATCHED; n];
    let mut block_vertices: Vec<Vec<usize>> = vec![Vec::new(); blocks];
    for t in 0..n {
        if in_band[t] {
            continue;
        }
        let b = block_of(centroids[t]);
        local_of[t] = block_vertices[b].len();
        block_vertices[b].push(t);
    }
    // Each block's interior is matched by the weighted Blossom on local
    // indices; cross-block edges are ignored here (the band pass handles them).
    for b in 0..blocks {
        if block_vertices[b].is_empty() {
            continue;
        }
        let mut local_edges: Vec<(usize, usize, i32)> = Vec::new();
        for &(a, c, w) in edges {
            if in_band[a] || in_band[c] {
                continue;
            }
            if block_of(centroids[a]) != b || block_of(centroids[c]) != b {
                continue;
            }
            local_edges.push((local_of[a], local_of[c], w));
        }
        if local_edges.is_empty() {
            continue;
        }
        let mut matching = mwmatching::Matching::new(local_edges);
        let local_mate = matching.solve();
        for (local, &t) in block_vertices[b].iter().enumerate() {
            let p = local_mate.get(local).copied().unwrap_or(UNMATCHED);
            if p != UNMATCHED && p < block_vertices[b].len() {
                mate[t] = block_vertices[b][p];
            }
        }
    }
    // Greedy over the band and the block-free vertices, best quality first.
    let mut order: Vec<usize> = (0..edges.len())
        .filter(|&i| {
            let (a, b, _) = edges[i];
            mate[a] == UNMATCHED && mate[b] == UNMATCHED
        })
        .collect();
    order.sort_unstable_by(|&a, &b| {
        edges[b]
            .2
            .cmp(&edges[a].2)
            .then_with(|| edges[a].0.cmp(&edges[b].0))
            .then_with(|| edges[a].1.cmp(&edges[b].1))
    });
    for &i in &order {
        let (a, b, _) = edges[i];
        if mate[a] == UNMATCHED && mate[b] == UNMATCHED {
            mate[a] = b;
            mate[b] = a;
        }
    }
    // Global maximum cardinality: the block solves are maximum-weight, not
    // maximum-cardinality, so the band and the free vertices are repaired by
    // the in-house Edmonds pass. Only the band's free vertices can remain
    // unmatched; the all-quad closure needs none at all.
    Blossom::new(n, adj, mate).run(scores)
}

/// How the triangle dual graph is matched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MatchPlan {
    /// The reference weighted Blossom (`mwmatching`) on the whole graph; used
    /// up to `EXACT_MATCHING_TRIANGLES`, where it is both cheap and the best
    /// quality, so the measured table is reproduced exactly.
    Exact,

    /// Weighted Blossom on spatial block interiors (~[`BLOCK_TRIANGLES`] each),
    /// then greedy + in-house Edmonds on the interface band and the block-free
    /// vertices; used above `EXACT_MATCHING_TRIANGLES` and measured 17x
    /// faster than `Exact` on the 0.6 mm part.
    Blocked,

    /// Quality-first greedy plus unweighted Blossom augmentation over the whole
    /// graph; the fallback plan when the cascade cannot close a residual
    /// triangle produced by a better plan.
    Fast,
}

/// Match the triangles with one plan and rebuild the quads.
fn recombine_with(
    tris: &[[usize; 3]],
    points: &[Node],
    plan: MatchPlan,
) -> (Vec<[usize; 4]>, Vec<[usize; 3]>) {
    let edges = recombination_edges(tris, points);
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); tris.len()];
    // The dual graph is undirected; every recombination edge is one candidate
    // pairing.
    for &(a, b, _) in &edges {
        adj[a].push(b);
        adj[b].push(a);
    }
    // The plan decides only the matching; the cascade closes whatever triangles
    // remain afterwards.
    let mate = match plan {
        MatchPlan::Exact => weighted_matching(tris.len(), &edges),
        MatchPlan::Blocked => {
            // Wide enough for the interface band to contain every edge that
            // crosses a block cut, in mean edge lengths.
            let scores = free_vertex_scores(tris, points);
            let band_width = BAND_EDGES * edge_scale(tris, points);
            let blocks = block_count(tris.len());
            blocked_matching(tris, points, &edges, &adj, &scores, band_width, blocks)
        }
        MatchPlan::Fast => {
            let greedy = greedy_matching(tris.len(), &edges);
            let scores = free_vertex_scores(tris, points);
            Blossom::new(tris.len(), &adj, greedy).run(&scores)
        }
    };
    quads_from_mate(tris, points, &mate)
}

/// Recombine and close residual triangles, trying the plans in order.
///
/// The matching only decides *where* the few residual triangles sit; the
/// refinement cascade can fail on an unlucky pair (its first split may make a
/// poor quad). Trying the fast plan as a fallback makes the closure robust
/// without giving up the weighted matching when it works. If every plan fails,
/// the first result is kept with its triangles: an honest quad-dominant mesh is
/// better than a half-modified one.
fn recombine_and_close(
    tris: &[[usize; 3]],
    points: &[Node],
    plans: &[MatchPlan],
) -> (Vec<Node>, Vec<[usize; 4]>, Vec<[usize; 3]>) {
    let mut first: Option<(Vec<Node>, Vec<[usize; 4]>, Vec<[usize; 3]>)> = None;
    for &plan in plans {
        let (quads, leftover) = recombine_with(tris, points, plan);
        if leftover.is_empty() {
            return (points.to_vec(), quads, leftover);
        }
        let mut closed_points = points.to_vec();
        let mut closed_quads = quads.clone();
        let mut closed_left = leftover.clone();
        if close_leftovers(&mut closed_points, &mut closed_quads, &mut closed_left).is_ok() {
            return (closed_points, closed_quads, closed_left);
        }
        if first.is_none() {
            first = Some((points.to_vec(), quads, leftover));
        }
    }
    match first {
        Some(result) => result,
        None => (points.to_vec(), Vec::new(), tris.to_vec()),
    }
}

// ------------------------------------------------------------------------- //
// Local closure of residual triangles (refinement cascade)
// ------------------------------------------------------------------------- //

/// Index of the point at `p`, appending it to `points` when it is new; the
/// exact-coordinate key makes a shared midpoint one node.
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

/// Index of the midpoint of nodes `a` and `b`, inserted on first use; the
/// cascade relies on the same midpoint being reused when two closures meet.
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

/// Is a quad acceptable for the cascade? Every corner scaled Jacobian must be
/// finite and `>= 0.2`, and every corner angle in `[30, 179]` degrees. This is
/// the gate's floor, so a triangle closure can never create a failing quad.
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

/// `(min corner scaled Jacobian, min corner angle in degrees)` of a quad; the
/// split guard compares children with their parent through these two values.
fn quad_quality(quad: [usize; 4], points: &[Node]) -> (f64, f64) {
    let cell = Cell::quad4(quad);
    let sj = quality::scaled_jacobian(&cell, points)
        .into_iter()
        .fold(f64::INFINITY, f64::min);
    let angles = quality::quad_angles(&cell, points);
    let min_angle = angles.iter().copied().fold(f64::INFINITY, f64::min);
    (sj, min_angle)
}

/// Index of the first quad that has `{a, b}` as an edge (either direction), or
/// `None`; used by the cascade to carry a hanging midpoint across an edge.
fn find_quad_edge(quads: &[[usize; 4]], a: usize, b: usize) -> Option<usize> {
    quads.iter().position(|q| {
        (0..4).any(|i| {
            let u = q[i];
            let v = q[(i + 1) % 4];
            (u == a && v == b) || (u == b && v == a)
        })
    })
}

/// Index of the first triangle that has `{a, b}` as an edge (either
/// direction), or `None`; used when a cascade reaches another residual
/// triangle.
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
/// edges at their midpoints.
///
/// `hanging` maps an edge to the midpoint that already exists on it (a cascade
/// that reached this triangle), so a split never creates two nodes at the same
/// place. Every new quad is checked by [`quad_is_good`] **before** the triangle
/// is removed; on failure the function returns an error and the caller discards
/// its per-plan copy (the shared `points`/`quads` of another plan are never
/// touched). The three split edges are pushed to `queue` so the cascade can
/// carry their midpoints across the neighbouring cells.
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
    // Three quads, each with one triangle vertex, two edge midpoints and the
    // centroid. This is the "triangle -> three quads" template of the cascade.
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
///
/// This is the propagation step of the cascade: the hanging node enters one
/// side, the opposite midpoint is created (or reused), and the opposite edge is
/// queued for the next cell. The split is refused when a second cascade has
/// already reached an adjacent edge of the same quad (this one template cannot
/// handle two hanging edges) and when either child would be below the parent's
/// quality floors.
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
    // Two children by the segment from the hanging midpoint `m` to the
    // opposite midpoint `n`; `m` is already a node of the parent, so the quad
    // count grows by one.
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
///
/// Each residual triangle is expanded to three quads, and every hanging
/// midpoint is propagated across the neighbouring quad (or residual triangle)
/// until the cascade reaches the boundary. `operations` is a safety cap: a
/// non-terminating cascade (which should not happen) returns an error instead of
/// looping.
///
/// On error the caller discards `points`, `quads` and `tris`; this function
/// does not roll back a partial cascade.
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
    // `tris` is consumed one triangle at a time; the inner queue carries the
    // hanging midpoints of the current cascade only.
    while !tris.is_empty() {
        let mut queue: Vec<((usize, usize), usize)> = Vec::new();
        close_triangle(0, points, quads, tris, &mut ids, &mut hanging, &mut queue)?;
        while let Some((edge, m)) = queue.pop() {
            operations += 1;
            if operations > 20_000 {
                return Err("the refinement cascade did not terminate".to_string());
            }
            // A midpoint on a quad edge splits that quad; on a residual
            // triangle edge it closes that triangle; an edge not found means
            // the cascade has reached nothing it can refine, and it is dropped.
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

/// Mesh `domain` with the L-norm frontal Delaunay route (see the module docs),
/// using the default [`Options`].
///
/// This is the public entry point: `mesh(domain, sizing)` is equivalent to
/// [`mesh_with_options`] with `Options::default()`.
pub fn mesh(domain: &Domain, sizing: &Sizing) -> Result<Mesh, String> {
    mesh_with_options(domain, sizing, &Options::default())
}

/// Mesh `domain` with the L-norm frontal Delaunay route and explicit
/// [`Options`].
///
/// Stages: validate the size and the outer loop, build the cross field
/// (`CrossField::build`), sample and constrain the boundary
/// (`sample_boundary`), refine the front (`frontal_refine`), relax the
/// interior to remove flat triangles (`relax_points`), rebuild the
/// constrained triangulation (`rebuild_cdt`), then recombine and close with
/// the size-dependent plan list (`recombine_and_close`). The returned mesh is
/// validated with [`Mesh::validate`] and is all-quad unless the cascade could
/// not close a residual triangle on every plan; in that case the residual
/// triangles are kept rather than silently dropped.
pub fn mesh_with_options(
    domain: &Domain,
    sizing: &Sizing,
    options: &Options,
) -> Result<Mesh, String> {
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

    // Build the CDT directly from the sampled boundary loops.
    let Some((outer, holes)) = loops.split_first() else {
        return Err("the boundary could not be discretized".to_string());
    };
    let mut state = CdtState::new(size, outer.clone(), holes.to_vec(), Vec::new())?;

    // The frontal loop inserts points until every front is resolved or blocked;
    // the triangulation is a standard L2 Delaunay kernel (the paper observes
    // that the L2 and L-infinity kernels give nearly the same triangulation).
    frontal_refine(&mut state, domain, &field, size, options)?;

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
    // A maximum-cardinality matching over convex pairs can still leave a triangle
    // when the dual graph has no perfect matching. The plans below are tried in
    // order: the matching decides where the residual triangles sit, and the
    // refinement cascade closes them; an unlucky pair can make the cascade's
    // first split poor, so the fast matching is kept as a fallback.
    // The exact weighted Blossom is affordable only below the threshold; above
    // it the blocked plan is the fast path. Both are retried with the fully
    // greedy `Fast` plan if the cascade cannot close the first plan's
    // leftovers.
    let plans: &[MatchPlan] = if final_tris.len() <= EXACT_MATCHING_TRIANGLES {
        &[MatchPlan::Exact, MatchPlan::Fast]
    } else {
        &[MatchPlan::Blocked, MatchPlan::Fast]
    };
    let (points, quads, leftover) = recombine_and_close(&final_tris, &points, plans);
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

    /// The two implemented norms differ where it matters: Manhattan gives 7
    /// and L-infinity 4 for the same vector.
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

    /// Eq. (10): the minimal axis-aligned square of a right isosceles triangle
    /// with axis-aligned legs is the one with the hypotenuse as a diagonal, and
    /// the circumcentre is the hypotenuse midpoint.
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

    /// §4.3: in the infinity norm the right isosceles triangle with
    /// axis-aligned legs is equilateral, which is why the apex construction
    /// produces the square the recombination wants.
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

    /// On a rectangle every boundary tangent is axis aligned, so the smoothed
    /// cross must be a multiple of `pi / 2` everywhere inside.
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

    /// End-to-end on a rectangle: no triangle survives and the gate passes.
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

    /// The two-loop case: the inner boundary must be constrained as well, and
    /// the cascade must still close every residual triangle.
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

    /// The in-house blossom must repair the classic greedy failure on a
    /// 4-vertex path (greedy takes the middle edge, leaving both ends free).
    #[test]
    fn the_greedy_matching_is_augmented_to_maximum_cardinality() {
        // A 4-vertex path whose heaviest edge is the middle one: greedy takes
        // the middle edge and leaves both ends free, so the matching is not
        // maximum until one augmenting path replaces it.
        let edges = vec![(0usize, 1usize, 1i32), (1, 2, 100), (2, 3, 1)];
        let greedy = greedy_matching(4, &edges);
        assert_eq!(greedy.iter().filter(|&&m| m != UNMATCHED).count(), 2);
        let adj = vec![vec![1usize], vec![0, 2], vec![1, 3], vec![2]];
        let scores = vec![0.0f64; 4];
        let mate = Blossom::new(4, &adj, greedy).run(&scores);
        let pairs = (0..4)
            .filter(|&t| mate[t] != UNMATCHED && mate[t] > t)
            .count();
        assert_eq!(pairs, 2, "not a maximum matching: {mate:?}");
        for t in 0..4 {
            if mate[t] != UNMATCHED {
                assert_eq!(mate[mate[t]], t);
            }
        }
    }

    /// The triangle dual graph of a `cells` x `cells` square grid (two right
    /// triangles per square, so every square is a candidate quad).
    fn triangle_grid(cells: usize) -> (Vec<Node>, Vec<[usize; 3]>) {
        let side = cells + 1;
        let mut points = Vec::with_capacity(side * side);
        for j in 0..side {
            for i in 0..side {
                points.push([i as f64, j as f64, 0.0]);
            }
        }
        let id = |i: usize, j: usize| j * side + i;
        let mut tris = Vec::with_capacity(cells * cells * 2);
        for j in 0..cells {
            for i in 0..cells {
                tris.push([id(i, j), id(i + 1, j), id(i + 1, j + 1)]);
                tris.push([id(i, j), id(i + 1, j + 1), id(i, j + 1)]);
            }
        }
        (points, tris)
    }

    /// The block + interface-band path must still yield a perfect matching on
    /// a regular triangle grid, where a perfect matching exists.
    #[test]
    fn the_blocked_matching_is_a_perfect_matching_on_a_triangle_grid() {
        let (points, tris) = triangle_grid(24);
        // Force two blocks: the grid is far below the production threshold but
        // this exercises the block + interface-band path.
        let edges = recombination_edges(&tris, &points);
        let mut adj = vec![Vec::new(); tris.len()];
        for &(a, b, _) in &edges {
            adj[a].push(b);
            adj[b].push(a);
        }
        let scores = free_vertex_scores(&tris, &points);
        let band = BAND_EDGES * edge_scale(&tris, &points);
        let mate = blocked_matching(&tris, &points, &edges, &adj, &scores, band, 2);
        for t in 0..tris.len() {
            assert_ne!(mate[t], UNMATCHED, "triangle {t} left free");
            assert_eq!(mate[mate[t]], t);
        }
        let (quads, leftover) = quads_from_mate(&tris, &points, &mate);
        assert_eq!(quads.len(), tris.len() / 2);
        assert!(leftover.is_empty());
    }

    /// The exact plan must pair every triangle of a regular grid and pass the
    /// gate without invoking the cascade.
    #[test]
    fn the_recombination_makes_all_quads_on_a_triangle_grid() {
        let (points, tris) = triangle_grid(12);
        let (points, quads, leftover) = recombine_and_close(&tris, &points, &[MatchPlan::Exact]);
        assert_eq!(quads.len(), tris.len() / 2);
        assert!(leftover.is_empty(), "{} triangles left", leftover.len());
        let cells: Vec<Cell> = quads.iter().map(|q| Cell::quad4(*q)).collect();
        let mesh = Mesh::new(points, cells);
        let gate = verify::gate(&mesh);
        assert!(gate.passes(), "{gate:?} {:?}", gate.reason());
    }

    /// A non-positive size is refused before any triangulation work.
    #[test]
    fn a_bad_size_is_refused() {
        let domain = Domain::rectangle(1.0, 1.0).unwrap();
        let sizing = Sizing::uniform(1.0).unwrap();
        let mut bad = sizing;
        bad.size = 0.0;
        assert!(mesh(&domain, &bad).is_err());
    }

    /// Non-default [`Options`] refine differently but must still produce an
    /// all-quad, gate-passing mesh on the rectangle; the test also checks that
    /// the explicit values differ from the defaults.
    #[test]
    fn non_default_options_still_produce_a_valid_mesh() {
        let domain = Domain::rectangle(12.0, 8.0).unwrap();
        let sizing = Sizing::uniform(2.0).unwrap();
        let options = Options {
            h_max: 1.6,
            min_spacing_fraction: 0.35,
        };
        assert_ne!(options.h_max, Options::default().h_max);
        let mesh = mesh_with_options(&domain, &sizing, &options).unwrap();
        let (quads, tris) = counts(&mesh);
        assert!(quads > 0, "no quad");
        assert_eq!(tris, 0, "{tris} triangles left");
        let gate = verify::gate(&mesh);
        assert!(gate.passes(), "{gate:?} {:?}", gate.reason());
        assert_eq!(gate.cracks, 0);
        assert!(verify::boundary_offset(&domain, &mesh) < 1.0e-9);
    }
}
