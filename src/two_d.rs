//! P1: planar all-quad meshing.
//!
//! The pipeline is:
//!
//! 1. **Boundary sampling.** Every input segment is split so no piece is longer
//!    than ~the size field (`sample_loops`); a long plate edge left as one
//!    constrained edge would make every incident triangle a sliver.
//! 2. **Constrained Delaunay triangulation** (the clean-room exact-predicate
//!    kernel), with every sampled boundary segment inserted as a constraint.
//! 3. **Ruppert/Delaunay refinement** (corpus
//!    `shewchuk-2002-delaunay-refinement`): insert a triangle's circumcenter
//!    when it is inside the domain, clear of the boundary and of existing
//!    nodes; otherwise split the nearest boundary edge at its midpoint. The
//!    implementation is the simple circumcenter version, not Ruppert's full
//!    encroachment proof.
//! 4. **Blossom-Quad pairing** (corpus `remacle-2012`): the dual graph of the
//!    triangles is matched with a **maximum-cardinality, maximum-weight**
//!    matching — Edmonds' blossom algorithm, via the `mwmatching` crate. The
//!    cardinality pass is what makes the result all-quad; the weight is the
//!    candidate quad's scaled Jacobian. Quality floors reject needles before
//!    they reach the matching ([`PairGuard`]).
//! 5. **Optional all-quad closure**, then valence clean-up (`cleanup`) and
//!    smoothing ([`smooth_mesh`]).
//!
//! The triangulation is the substrate, not the product: a triangle that cannot
//! be paired is returned separately, never silently dropped. `TwoDResult` is
//! therefore quad-dominant by contract unless `Sizing::all_quad` is set and the
//! cascade succeeds.
//!
//! **Approximated or missing.** The size field is uniform: [`Sizing::size_at`]
//! ignores the point (no curvature, feature or anisotropic field). The
//! refinement uses a fixed boundary margin plus a point-grid spacing rule
//! instead of Ruppert's encroachment test, and the matching floors are
//! empirical (see the measured notes on the constants). Clean-up is edge swaps
//! only; it has no face open/close of the kind Q-Morph's clean-up uses.

use mwmatching::{Matching, SENTINEL};

use crate::delaunay::CdtAdapter;
use crate::mesh::{Cell, Kind, Mesh, Node};
use crate::quality;

/// A planar domain: one outer loop and any number of holes, all closed
/// polygons in the XY plane (millimetres).
///
/// Invariants enforced by [`Domain::new`]: every loop has at least three points
/// and every coordinate is finite. The outer loop is expected counter-clockwise
/// (a clockwise one gives a negative [`Domain::area`] and is rejected by
/// [`mesh_2d`]); holes may be given either way because [`Domain::contains`] is
/// even-odd.
#[derive(Clone, Debug, Default)]
pub struct Domain {
    /// Outer boundary loop, in order: `[x, y]` pairs in millimetres.
    pub outer: Vec<[f64; 2]>,

    /// Hole loops, each ordered around its own boundary; a hole is excluded
    /// from the domain. Empty when the domain is simply connected.
    pub holes: Vec<Vec<[f64; 2]>>,
}

/// What can go wrong building or meshing a domain.
///
/// Every variant is a rejected request, never a partial mesh: the public
/// functions return `Ok` only when they produced a mesh, so a caller can treat
/// an error as "nothing was built". The messages are user-facing (the CLI
/// prints them verbatim).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TwoDError {
    /// A loop has fewer than three points.
    LoopTooShort(usize),
    /// A coordinate is not finite.
    NonFinite,
    /// The mesh request is unusable (size ≤ 0, or the domain has no area).
    BadRequest(&'static str),
    /// The triangulation failed.
    Triangulation(&'static str),
}

impl std::fmt::Display for TwoDError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TwoDError::LoopTooShort(n) => write!(f, "a loop has {n} points; at least 3 are needed"),
            TwoDError::NonFinite => write!(f, "a boundary coordinate is not finite"),
            TwoDError::BadRequest(message) => write!(f, "{message}"),
            TwoDError::Triangulation(message) => write!(f, "triangulation failed: {message}"),
        }
    }
}

impl std::error::Error for TwoDError {}

impl Domain {
    /// A domain with an outer loop and optional holes.
    ///
    /// The loops are stored as given. Validation rejects a loop with fewer than
    /// three points and any non-finite coordinate; it does **not** reject a
    /// self-intersecting polygon, a hole outside the outer loop, or a clockwise
    /// outer loop (the latter is caught by the area check in [`mesh_2d`]).
    pub fn new(outer: Vec<[f64; 2]>, holes: Vec<Vec<[f64; 2]>>) -> Result<Self, TwoDError> {
        let domain = Domain { outer, holes };
        domain.validate()?;
        Ok(domain)
    }

    /// An axis-aligned rectangle, counter-clockwise, with its lower-left corner
    /// at the origin and dimensions in millimetres.
    pub fn rectangle(width: f64, height: f64) -> Result<Self, TwoDError> {
        Domain::new(
            vec![[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
            Vec::new(),
        )
    }

    /// A rectangle with a circular hole at its centre.
    ///
    /// `segments` points discretise the circle (the caller's sampling choice);
    /// the hole is emitted clockwise, the conventional opposite orientation to
    /// the counter-clockwise outer loop. A plate with a hole is the corpus
    /// stress-concentration part used by the quality checks.
    pub fn plate_with_hole(
        width: f64,
        height: f64,
        radius: f64,
        segments: usize,
    ) -> Result<Self, TwoDError> {
        let circle = (0..segments)
            .map(|i| {
                let a = std::f64::consts::TAU * i as f64 / segments as f64;
                // clockwise, so the hole is traversed opposite to the outer loop
                [
                    width / 2.0 + radius * a.cos(),
                    height / 2.0 - radius * a.sin(),
                ]
            })
            .collect();
        Domain::new(
            vec![[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
            vec![circle],
        )
    }

    /// Check the invariants of [`Domain`]: at least three points per loop and
    /// all coordinates finite.
    fn validate(&self) -> Result<(), TwoDError> {
        let loops = std::iter::once(&self.outer).chain(self.holes.iter());
        for loop_points in loops {
            if loop_points.len() < 3 {
                return Err(TwoDError::LoopTooShort(loop_points.len()));
            }
            if loop_points
                .iter()
                .any(|p| !p[0].is_finite() || !p[1].is_finite())
            {
                return Err(TwoDError::NonFinite);
            }
        }
        Ok(())
    }

    /// Signed area in square millimetres: positive when the outer loop is
    /// counter-clockwise.
    ///
    /// The hole terms are subtracted by absolute value, so a hole given
    /// clockwise (the convention) and one given counter-clockwise give the same
    /// result. Returns a negative value for a clockwise outer loop, which
    /// [`mesh_2d`] rejects.
    pub fn area(&self) -> f64 {
        let ring = |points: &[[f64; 2]]| -> f64 {
            let mut sum = 0.0;
            for i in 0..points.len() {
                let a = points[i];
                let b = points[(i + 1) % points.len()];
                sum += a[0] * b[1] - b[0] * a[1];
            }
            sum / 2.0
        };
        ring(&self.outer) - self.holes.iter().map(|h| ring(h).abs()).sum::<f64>()
    }

    /// Vertices where the boundary turns by more than 30 degrees.
    ///
    /// A circular boundary discretised as a 24-gon turns 15 degrees per vertex:
    /// those are sampling points, not sharp features. A square corner turns 90
    /// degrees and must be a mesh node. This is the set `corner_offset` measures;
    /// exact polygon coverage separately requires preserving shallow vertices too.
    ///
    /// The test is on the angle between the two incident edge vectors, so it is
    /// independent of loop orientation. Degenerate (zero-length) edges are
    /// skipped. Measured reason for the 150-degree threshold: on the
    /// plate-with-hole a 24-gon hole's shallow vertices (15 degrees of turn)
    /// were being slid along the arc, which moved the geometry; a 90-degree
    /// corner that was not protected ended up 1.0 mm from the nearest node.
    pub fn sharp_corners(&self) -> Vec<[f64; 2]> {
        let mut out = Vec::new();
        for loop_pts in std::iter::once(&self.outer).chain(self.holes.iter()) {
            let n = loop_pts.len();
            if n < 3 {
                continue;
            }
            for i in 0..n {
                let p = loop_pts[(i + n - 1) % n];
                let v = loop_pts[i];
                let q = loop_pts[(i + 1) % n];
                let a = [p[0] - v[0], p[1] - v[1]];
                let b = [q[0] - v[0], q[1] - v[1]];
                let la = (a[0] * a[0] + a[1] * a[1]).sqrt();
                let lb = (b[0] * b[0] + b[1] * b[1]).sqrt();
                if la < 1.0e-12 || lb < 1.0e-12 {
                    continue;
                }
                let cos = ((a[0] * b[0] + a[1] * b[1]) / (la * lb)).clamp(-1.0, 1.0);
                if cos.acos().to_degrees() < 150.0 {
                    out.push(v);
                }
            }
        }
        out
    }

    /// All boundary segments, outer loop first and then each hole, as
    /// `(start, end)` pairs in the loops' stored order.
    ///
    /// The closing segment of every loop is included. Used by
    /// [`Domain::nearest_segment`] and [`Domain::distance_to_boundary`]; it
    /// allocates on every call, so hot paths should cache the result.
    pub fn segments(&self) -> Vec<([f64; 2], [f64; 2])> {
        let mut out = Vec::new();
        for loop_points in std::iter::once(&self.outer).chain(self.holes.iter()) {
            for i in 0..loop_points.len() {
                out.push((loop_points[i], loop_points[(i + 1) % loop_points.len()]));
            }
        }
        out
    }

    /// Even-odd test: inside the outer loop and outside every hole.
    ///
    /// A point exactly on an edge or vertex has no defined side here; callers
    /// that care use [`Domain::distance_to_boundary`] with a tolerance instead.
    pub fn contains(&self, p: [f64; 2]) -> bool {
        if !point_in_polygon(p, &self.outer) {
            return false;
        }
        !self.holes.iter().any(|h| point_in_polygon(p, h))
    }

    /// Euclidean distance, in millimetres, from `p` to the nearest boundary
    /// segment (zero on the boundary).
    pub fn distance_to_boundary(&self, p: [f64; 2]) -> f64 {
        self.nearest_segment(p).2
    }

    /// The boundary segment nearest to `p`: its endpoints and the Euclidean
    /// distance to it, in millimetres.
    ///
    /// Ties are broken by iteration order (outer loop before holes, segments in
    /// stored order). A domain with no segments returns a dummy
    /// `([0, 0], [0, 0], +inf)`, which callers compare against as "not near the
    /// boundary".
    pub fn nearest_segment(&self, p: [f64; 2]) -> ([f64; 2], [f64; 2], f64) {
        let mut best = ([0.0; 2], [0.0; 2], f64::INFINITY);
        for (a, b) in self.segments() {
            let d = point_segment_distance(p, a, b);
            if d < best.2 {
                best = (a, b, d);
            }
        }
        best
    }
}

/// Classical even-odd ray casting: count the polygon edges crossed by a
/// horizontal ray from `p` to the right.
///
/// The `(a[1] > p[1]) != (b[1] > p[1])` test is the half-open rule that makes a
/// vertex on the ray count once, not twice. Boundary points are unspecified.
fn point_in_polygon(p: [f64; 2], polygon: &[[f64; 2]]) -> bool {
    let mut inside = false;
    let n = polygon.len();
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

/// Euclidean distance from `p` to the closed segment `a`-`b`.
///
/// The projection parameter is clamped to `[0, 1]`, so the distance to a
/// degenerate (zero-length) segment is the distance to its point.
fn point_segment_distance(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let length2 = ab[0] * ab[0] + ab[1] * ab[1];
    if length2 <= 0.0 {
        return ((p[0] - a[0]).powi(2) + (p[1] - a[1]).powi(2)).sqrt();
    }
    let t = (((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / length2).clamp(0.0, 1.0);
    let q = [a[0] + t * ab[0], a[1] + t * ab[1]];
    ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt()
}

/// A uniform element size, in millimetres. Curvature and feature fields come
/// later; the P1 gate only needs this.
///
/// The struct is public and has public fields, so it can be built without
/// [`Sizing::uniform`]; the meshing entry points re-check `size` and return
/// [`TwoDError::BadRequest`] rather than trusting it.
#[derive(Clone, Copy, Debug)]
pub struct Sizing {
    /// Target element size in millimetres: the refinement aims for circumradius
    /// and edge lengths around this value. Must be finite and positive.
    pub size: f64,

    /// Close every leftover triangle by a conforming refinement cascade.
    ///
    /// Off by default: the cascade is exact but expensive — the split of a
    /// triangle's edge has to propagate through the quads until it reaches a
    /// boundary, so the chain grows with the mesh. Measured on the corpus: +12 %
    /// elements at 4 mm, +219 % at 1 mm, with the thinnest quads roughly halved
    /// in quality. The triangles it removes cost accuracy only if a sliver lands
    /// on the stress path (see the FEM check in the README).
    pub all_quad: bool,
}

impl Sizing {
    /// A uniform size field with `all_quad` off.
    ///
    /// Errors with [`TwoDError::BadRequest`] when `size` is not finite or not
    /// positive. The check is repeated in [`triangulate`] because the fields are
    /// public and a caller can bypass this constructor.
    pub fn uniform(size: f64) -> Result<Self, TwoDError> {
        if !size.is_finite() || size <= 0.0 {
            return Err(TwoDError::BadRequest(
                "the element size must be positive and finite",
            ));
        }
        Ok(Sizing {
            size,
            all_quad: false,
        })
    }

    /// Ask for a mesh with no triangles, at the cost of local refinement.
    ///
    /// This is a builder: it consumes and returns the value. The cascade is
    /// best-effort — a triangle it cannot close is still returned in
    /// `TwoDResult::unpaired`.
    pub fn all_quad(mut self) -> Self {
        self.all_quad = true;
        self
    }

    /// Target size at `p`, in millimetres.
    ///
    /// Currently constant: `p` is ignored. This is the hook where a curvature
    /// or feature-size field would be evaluated; keeping the call sites on this
    /// method means such a field can be added without touching them.
    pub fn size_at(&self, _p: [f64; 2]) -> f64 {
        self.size
    }
}

/// Statistics of one planar meshing run.
///
/// Counts are taken after the all-quad closure and before the final clean-up /
/// smoothing (those change cells but not these fields). `points` counts the
/// final node set, including any node the closure cascade inserted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TwoDStats {
    /// Nodes in the returned mesh.
    pub points: usize,

    /// Triangles produced by the triangulation (before matching), not the
    /// number left in the output.
    pub triangles: usize,

    /// Quads in the returned mesh.
    pub quads: usize,

    /// Triangles the matching could not pair. They are kept in the mesh, so the
    /// output is quad-dominant rather than all-quad.
    pub unpaired_triangles: usize,

    /// Reserved for the closure pass; the field is populated but its current
    /// meaning is the number of extra quads the cascade created, not a count of
    /// absorbed triangles.
    pub absorbed_triangles: usize,

    /// Edge swaps taken by the valence-driven clean-up.
    pub cleanup_swaps: usize,
}

impl TwoDStats {
    /// Share of planar elements that are quads.
    pub fn quad_ratio(&self) -> f64 {
        let total = self.quads + self.unpaired_triangles;
        if total == 0 {
            0.0
        } else {
            self.quads as f64 / total as f64
        }
    }
}

/// A planar quad mesh, plus any triangle the matching could not pair.
///
/// `mesh` contains both quads and the leftover triangles, so it tiles the
/// domain; `unpaired` is the same leftover list, repeated for callers that want
/// to post-process the triangles without filtering the cell list.
#[derive(Clone, Debug)]
pub struct TwoDResult {
    /// The output mesh: `Cell::Quad4` cells plus one `Cell::Tri3` per entry in
    /// `unpaired`, wound counter-clockwise.
    pub mesh: Mesh,

    /// Node triples of the triangles that could not be paired, in the order the
    /// matching reported them (deterministic).
    pub unpaired: Vec<[usize; 3]>,

    /// Counts for this run; see [`TwoDStats`].
    pub stats: TwoDStats,
}

/// Passes of the valence-driven clean-up.
const CLEANUP_ROUNDS: usize = 20;

/// Longest cascade allowed for one triangle. Beyond this the triangle is kept:
/// a chain that crosses the whole mesh costs more than the triangle it removes.
const MAX_CASCADE_STEPS: usize = 64;

/// Smoothing passes after closure.
const SMOOTHING_ITERATIONS: usize = 80;

/// A candidate point closer than this fraction of the local size to an existing
/// point is rejected: Ruppert's spacing rule. Without it, circumcenters land a
/// few hundredths of a millimetre from a neighbour and the quad containing that
/// edge has an aspect ratio in the thousands.
const MIN_SPACING_FRACTION: f64 = 0.45;

/// A uniform grid of points, for "is this candidate too close to anything?".
///
/// Used by [`refine`] to enforce the spacing rule without a nearest-neighbour
/// structure: a query only has to look in the 3x3 block of buckets around it,
/// so the check is O(1) in the number of inserted points.
struct PointGrid {
    /// Bucket side length; at least `1e-9` so the division in [`PointGrid::key`]
    /// cannot be by zero. The refinement sets it to the target element size.
    cell: f64,

    /// Points bucketed by integer cell coordinates. A point belongs to exactly
    /// one bucket; `nearest` searches the 3x3 neighbourhood, which is enough
    /// because a query farther than one cell cannot find a closer point.
    buckets: std::collections::HashMap<(i64, i64), Vec<[f64; 2]>>,
}

impl PointGrid {
    /// A grid with the given cell size, clamped to a positive value.
    fn new(cell: f64) -> Self {
        PointGrid {
            cell: cell.max(1e-9),
            buckets: std::collections::HashMap::new(),
        }
    }

    /// Integer bucket coordinate of `p` (floor, so negative coordinates work).
    fn key(&self, p: [f64; 2]) -> (i64, i64) {
        (
            (p[0] / self.cell).floor() as i64,
            (p[1] / self.cell).floor() as i64,
        )
    }

    /// Store `p`. Duplicates are not rejected here; the caller's spacing check
    /// is what keeps them out.
    fn insert(&mut self, p: [f64; 2]) {
        self.buckets.entry(self.key(p)).or_default().push(p);
    }

    /// Distance to the nearest stored point, or infinity when the grid is empty.
    fn nearest(&self, p: [f64; 2]) -> f64 {
        let (cx, cy) = self.key(p);
        let mut best = f64::INFINITY;
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(points) = self.buckets.get(&(cx + dx, cy + dy)) {
                    for q in points {
                        let d = ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt();
                        best = best.min(d);
                    }
                }
            }
        }
        best
    }
}

/// A pair whose quad would be worse than this is not offered to the matching.
/// Without a floor the matching pairs two thin triangles into a needle, because
/// pairing is worth more than leaving them unpaired.
const MIN_PAIR_SCALED_JACOBIAN: f64 = 0.05;
/// Likewise for the smallest corner angle of the candidate quad.
///
/// 10° rather than 20°: the FEM accuracy check (plate with a hole, Kt) showed
/// that degrading quads from 32° to 8.7° moves the stress concentration by 4 %,
/// the same order as the reference's own uncertainty. The floor that matters is
/// the *maximum* angle below — needle pairs — not this one.
const MIN_PAIR_ANGLE_DEG: f64 = 10.0;
/// And for the largest: a corner near 180° means the pair is a needle, not a
/// quad, however good its scaled Jacobian looks.
const MAX_PAIR_ANGLE_DEG: f64 = 160.0;

/// Quality guards for [`pair_triangles_with`].
///
/// The shipped matching path uses `PairGuard::default`, i.e. the constants
/// above; a caller that pairs a **region** of an existing mesh (the two-stage
/// conversion) can ask for a stricter floor without changing that path.
#[derive(Clone, Copy, Debug)]
pub struct PairGuard {
    /// Smallest corner scaled Jacobian a candidate quad may have; a value in
    /// `[0, 1]`, where 1 is a square. Candidates below it are not offered to
    /// the matching.
    pub min_scaled_jacobian: f64,

    /// Smallest corner angle, in degrees, a candidate quad may have.
    pub min_angle_deg: f64,

    /// Largest corner angle, in degrees, a candidate quad may have. This is the
    /// guard that rejects needle pairs (a corner near 180 degrees).
    pub max_angle_deg: f64,
}

impl Default for PairGuard {
    fn default() -> Self {
        PairGuard {
            min_scaled_jacobian: MIN_PAIR_SCALED_JACOBIAN,
            min_angle_deg: MIN_PAIR_ANGLE_DEG,
            max_angle_deg: MAX_PAIR_ANGLE_DEG,
        }
    }
}

/// Refinement stops after this many passes even if the size field is unmet.
const MAX_PASSES: usize = 200;
/// A circumcenter closer than this to the boundary is not inserted: it would
/// create a sliver along the edge instead of improving the element.
const BOUNDARY_MARGIN: f64 = 0.55;

/// Mesh a planar domain with all-quad elements.
///
/// Runs the whole pipeline: sample, triangulate, refine, pair, optionally close
/// the leftovers, clean up and smooth. `domain` is in millimetres and must have
/// a positive [`Domain::area`]; `sizing.size` must be finite and positive.
///
/// # Errors
///
/// [`TwoDError::BadRequest`] for a non-positive area or size,
/// [`TwoDError::LoopTooShort`] / [`TwoDError::NonFinite`] from
/// [`Domain::new`] (if the struct was built by hand, `validate` is not re-run
/// here), and [`TwoDError::Triangulation`] when the constrained triangulation
/// cannot insert a point or a constraint. Any error means no mesh was built.
///
/// # Postconditions
///
/// The returned mesh tiles the domain: quads plus `unpaired` triangles, with no
/// cell dropped. All cells are wound counter-clockwise. Geometry is in the XY
/// plane (`z = 0`). Topology is validated only for the all-quad closure path;
/// the quad-dominant path relies on the triangulation being conforming.
pub fn mesh_2d(domain: &Domain, sizing: &Sizing) -> Result<TwoDResult, TwoDError> {
    if domain.area() <= 0.0 {
        return Err(TwoDError::BadRequest(
            "the domain has no area (outer loop clockwise, or a hole larger than it)",
        ));
    }
    let (points, tris) = triangulate(domain, sizing)?;

    let (quads, unpaired) = pair_triangles(&tris, &points);
    let base_quads = quads.len();
    // Close the leftovers by refinement when asked. A triangle that cannot be
    // closed is kept as a triangle rather than half-modified.
    let (quads, points, leftover) = if sizing.all_quad {
        let (q, p, left) = close_by_refinement(&quads, &unpaired, &points, domain);
        // The cascade can leave a T-junction where a midpoint lands on the domain
        // boundary. Verify before trusting it: an invalid "all-quad" mesh is
        // worse than an honest quad-dominant one.
        let candidate = build_mesh(&points, &q, &left);
        let topo = candidate.topology();
        if topo.cracks == 0 && topo.inconsistent == 0 && topo.non_manifold == 0 {
            (q, p, left)
        } else {
            (quads, points.clone(), unpaired.clone())
        }
    } else {
        (quads, points.clone(), unpaired.clone())
    };
    let mut stats = TwoDStats {
        points: points.len(),
        triangles: tris.len(),
        quads: quads.len(),
        unpaired_triangles: leftover.len(),
        absorbed_triangles: quads.len().saturating_sub(base_quads),
        cleanup_swaps: 0,
    };
    // Leftover triangles stay in the mesh as triangles: dropping them would
    // leave a hole, which is worse than a quad-dominant result.
    let mut cells: Vec<Cell> = quads.into_iter().map(Cell::quad4).collect();
    cells.extend(leftover.iter().map(|t| Cell::tri3(*t)));
    let mut mesh = Mesh::new(points, cells);
    stats.cleanup_swaps = cleanup(&mut mesh, CLEANUP_ROUNDS);
    smooth(&mut mesh, domain, SMOOTHING_ITERATIONS, sizing.size);
    Ok(TwoDResult {
        mesh,
        unpaired: leftover.clone(),
        stats,
    })
}

/// Assemble a mesh from quads plus leftover triangles.
///
/// The cell order is quads first, then triangles in `leftover` order; used to
/// run the topology check on the closure candidate before committing to it.
fn build_mesh(points: &[Node], quads: &[[usize; 4]], leftover: &[[usize; 3]]) -> Mesh {
    let mut cells: Vec<Cell> = quads.iter().map(|q| Cell::quad4(*q)).collect();
    cells.extend(leftover.iter().map(|t| Cell::tri3(*t)));
    Mesh::new(points.to_vec(), cells)
}

/// Smooth nodes to raise the smallest corner angle without losing Jacobian.
///
/// Interior nodes move to the centroid of their ring. Boundary sampling nodes
/// slide **along** the boundary segment they sit on, which is where the worst
/// angles are. Original domain vertices stay fixed: removing even a shallow
/// vertex replaces two boundary edges by a chord and changes the polygon. Each
/// move minimizes the sum of its incident corner objectives. As a final safety
/// guard, a gate-passing input is restored if the completed pass fails the gate.
///
/// This is a nodally based quality objective in the spirit of Knupp 2000
/// (corpus `knupp-2000-part1`), not the Laplacian smoother the name might
/// suggest. `size` is the local element size in millimetres and sets the search
/// radii; `iterations` is an upper bound (the loop stops early when no node
/// moves). The mesh's connectivity is unchanged — only `Mesh::points` moves —
/// and boundary nodes stay on their own segment.
///
/// The caller is [`mesh_2d`]; the public form exists so a caller can re-smooth a
/// mesh after its own post-processing.
pub fn smooth_mesh(mesh: &mut Mesh, domain: &Domain, iterations: usize, size: f64) {
    if !crate::verify::gate(mesh).passes() {
        smooth(mesh, domain, iterations, size);
        return;
    }
    let original = mesh.points.clone();
    smooth(mesh, domain, iterations, size);
    if !crate::verify::gate(mesh).passes() {
        mesh.points = original;
    }
}

/// Implementation of [`smooth_mesh`], also used internally by [`mesh_2d`].
fn smooth(mesh: &mut Mesh, domain: &Domain, iterations: usize, size: f64) {
    let n = mesh.points.len();
    let vertices: Vec<[f64; 2]> = domain
        .outer
        .iter()
        .chain(domain.holes.iter().flatten())
        .copied()
        .collect();
    let mut incident: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (c, cell) in mesh.cells.iter().enumerate() {
        for &v in &cell.nodes {
            incident[v].push(c);
        }
    }

    for _ in 0..iterations {
        let mut moved = 0usize;
        for v in 0..n {
            if incident[v].is_empty() {
                continue;
            }
            let current = [mesh.points[v][0], mesh.points[v][1]];
            let on_boundary = domain.distance_to_boundary(current) < 1e-9;
            if on_boundary
                && vertices
                    .iter()
                    .any(|c| (c[0] - current[0]).powi(2) + (c[1] - current[1]).powi(2) < 1.0e-18)
            {
                continue;
            }

            let mut ring = Vec::new();
            for &c in &incident[v] {
                for &u in &mesh.cells[c].nodes {
                    if u != v && !ring.contains(&u) {
                        ring.push(u);
                    }
                }
            }
            if ring.len() < 3 {
                continue;
            }
            let centroid = [
                ring.iter().map(|&u| mesh.points[u][0]).sum::<f64>() / ring.len() as f64,
                ring.iter().map(|&u| mesh.points[u][1]).sum::<f64>() / ring.len() as f64,
            ];

            // Candidate positions: the ring centroid, and points on two rings
            // around the current position. Boundary nodes may only slide along
            // their segment, which is where the worst angles live.
            let mut candidates: Vec<[f64; 2]> = Vec::new();
            if on_boundary {
                let (a, b, _) = domain.nearest_segment(current);
                for t in [0.1, 0.25, 0.4, 0.6, 0.75, 0.9] {
                    candidates.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
                }
            } else {
                candidates.push(centroid);
                for (radius, steps) in [(0.3 * size, 8), (0.1 * size, 8)] {
                    for k in 0..steps {
                        let angle = std::f64::consts::TAU * k as f64 / steps as f64;
                        candidates.push([
                            current[0] + radius * angle.cos(),
                            current[1] + radius * angle.sin(),
                        ]);
                    }
                }
            }

            let mut best = current;
            let mut best_f = node_objective(&incident[v], mesh);
            for candidate in candidates {
                if !spacing_ok(&incident[v], mesh, candidate, v, size) {
                    continue;
                }
                // Discrete descent: try the candidate in place, keep it only if
                // the objective strictly improves. The comparison is against the
                // current best (not the original position), so several good
                // candidates can compound within one visit; a candidate that
                // inverts a cell scores +inf and can never win.
                mesh.points[v] = [candidate[0], candidate[1], 0.0];
                let f = node_objective(&incident[v], mesh);
                if f < best_f - 1e-9 {
                    best = candidate;
                    best_f = f;
                }
            }
            mesh.points[v] = [best[0], best[1], 0.0];
            if best != current {
                moved += 1;
            }
        }
        if moved == 0 {
            break;
        }
    }
}

/// Topological clean-up of a quad mesh, driven by node valence.
///
/// Two quads sharing an edge form a hexagon. Replacing the shared edge with one
/// of the hexagon's other diagonals keeps two quads and changes four valences:
/// the ends of the old edge lose one, the new corners gain one. A swap is taken
/// only when it strictly lowers the irregularity `sum |valence - 4|` over the
/// interior nodes it touches and both new quads stay convex with a usable
/// Jacobian. This is the cheap half of Q-Morph's clean-up (Owen et al. 1999,
/// corpus `owen-1999`) — edge swaps only, no face opens or closes.
///
/// `max_rounds` bounds the passes, not the swaps; the loop stops early when a
/// pass changes nothing. Returns the number of swaps actually taken. Nodes on
/// the mesh boundary are excluded from the valence cost because their valence is
/// fixed by the boundary discretisation.
fn cleanup(mesh: &mut Mesh, max_rounds: usize) -> usize {
    let mut swaps = 0usize;
    for _ in 0..max_rounds {
        let mut changed = 0usize;
        let quad_ids: Vec<usize> = (0..mesh.cells.len())
            .filter(|&c| mesh.cells[c].kind == Kind::Quad4)
            .collect();
        let mut edges: std::collections::HashMap<(usize, usize), Vec<usize>> =
            std::collections::HashMap::new();
        for &c in &quad_ids {
            let q = &mesh.cells[c].nodes;
            for i in 0..4 {
                let (a, b) = (q[i], q[(i + 1) % 4]);
                edges.entry((a.min(b), a.max(b))).or_default().push(c);
            }
        }
        let boundary: std::collections::HashSet<usize> = edges
            .iter()
            .filter(|(_, v)| v.len() == 1)
            .flat_map(|(&(a, b), _)| [a, b])
            .collect();
        let mut valence: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for &c in &quad_ids {
            for &v in &mesh.cells[c].nodes {
                *valence.entry(v).or_insert(0) += 1;
            }
        }
        let cost = |nodes: &[usize], valence: &std::collections::HashMap<usize, usize>| -> i64 {
            nodes
                .iter()
                .filter(|v| !boundary.contains(v))
                .map(|v| (valence.get(v).copied().unwrap_or(0) as i64 - 4).abs())
                .sum()
        };

        // Deterministic: `HashMap` iteration order is randomised per process, and
        // each swap mutates the mesh the next candidate is measured on. Visiting
        // the keys in ascending order makes the same binary produce the same mesh;
        // it changes only the order equal-cost swaps are considered in.
        let mut keys: Vec<(usize, usize)> = edges.keys().copied().collect();
        keys.sort_unstable();
        for (a, b) in keys {
            let Some(owners) = edges.get(&(a, b)).cloned() else {
                continue;
            };
            if owners.len() != 2 {
                continue;
            }
            let (c1, c2) = (owners[0], owners[1]);
            let q1 = mesh.cells[c1].nodes.clone();
            let q2 = mesh.cells[c2].nodes.clone();
            let Some(i1) = (0..4)
                .find(|&i| (q1[i].min(q1[(i + 1) % 4]), q1[i].max(q1[(i + 1) % 4])) == (a, b))
            else {
                continue;
            };
            // q1 in the direction a -> b
            let (x, y) = (q1[i1], q1[(i1 + 1) % 4]);
            let p = q1[(i1 + 2) % 4];
            let q = q1[(i1 + 3) % 4];
            // q2 must traverse the same edge the other way
            let Some(i2) = (0..4).find(|&i| q2[i] == y && q2[(i + 1) % 4] == x) else {
                continue;
            };
            let r = q2[(i2 + 2) % 4];
            let s = q2[(i2 + 3) % 4];

            let options: [[[usize; 4]; 2]; 2] =
                [[[p, q, x, r], [p, r, s, y]], [[q, x, r, s], [q, s, y, p]]];
            let before = cost(&[x, y, p, r, q, s], &valence);
            let old_quality = quad_min_sj(&coords_of(&q1, &mesh.points))
                .min(quad_min_sj(&coords_of(&q2, &mesh.points)));
            let mut best: Option<([[usize; 4]; 2], i64)> = None;
            for pair in options {
                let valid = pair.iter().all(|quad| {
                    let coords = [
                        [mesh.points[quad[0]][0], mesh.points[quad[0]][1]],
                        [mesh.points[quad[1]][0], mesh.points[quad[1]][1]],
                        [mesh.points[quad[2]][0], mesh.points[quad[2]][1]],
                        [mesh.points[quad[3]][0], mesh.points[quad[3]][1]],
                    ];
                    convex_ccw(&coords) && quad_min_sj(&coords) > 0.05
                });
                if !valid {
                    continue;
                }
                let new_quality = pair
                    .iter()
                    .map(|quad| quad_min_sj(&coords_of(quad, &mesh.points)))
                    .fold(f64::INFINITY, f64::min);
                // A swap that buys regularity by making the elements worse is a
                // bad trade: the smoother has to live with the result.
                if new_quality < old_quality - 1e-6 {
                    continue;
                }
                let mut v = valence.clone();
                *v.entry(x).or_insert(0) -= 1;
                *v.entry(y).or_insert(0) -= 1;
                *v.entry(p).or_insert(0) += 1;
                *v.entry(r).or_insert(0) += 1;
                let after = cost(&[x, y, p, r, q, s], &v);
                if after < before && best.as_ref().is_none_or(|(_, c)| after < *c) {
                    best = Some((pair, after));
                }
            }
            if let Some((pair, _)) = best {
                mesh.cells[c1] = Cell::quad4(pair[0]);
                mesh.cells[c2] = Cell::quad4(pair[1]);
                *valence.entry(x).or_insert(0) -= 1;
                *valence.entry(y).or_insert(0) -= 1;
                *valence.entry(p).or_insert(0) += 1;
                *valence.entry(r).or_insert(0) += 1;
                changed += 1;
                swaps += 1;
            }
        }
        if changed == 0 {
            break;
        }
    }
    swaps
}

/// The XY coordinates of a quad's four nodes, in stored order.
fn coords_of(quad: &[usize], points: &[Node]) -> [[f64; 2]; 4] {
    [
        [points[quad[0]][0], points[quad[0]][1]],
        [points[quad[1]][0], points[quad[1]][1]],
        [points[quad[2]][0], points[quad[2]][1]],
        [points[quad[3]][0], points[quad[3]][1]],
    ]
}

/// Nodally based quality objective, in the spirit of Knupp (2000, corpus
/// `knupp-2000-part1`).
///
/// For every corner of every incident quad: `(1 - sin θ)²`, which is the
/// deviation of the scaled Jacobian from 1, plus a logarithmic barrier on
/// `sin θ` that grows without bound as an element collapses. The barrier is what
/// makes the minimisation refused to invert an element, instead of needing the
/// boolean guard the previous smoother used.
fn node_objective(incident: &[usize], mesh: &Mesh) -> f64 {
    let mut f = 0.0;
    for &c in incident {
        // A discrete search can step over the barrier: an inverted element must
        // be infinitely bad, not merely poor.
        if mesh.cells[c].kind == Kind::Quad4 {
            let nodes: [usize; 4] = match mesh.cells[c].nodes.as_slice().try_into() {
                Ok(n) => n,
                Err(_) => return f64::INFINITY,
            };
            let coords = [
                [mesh.points[nodes[0]][0], mesh.points[nodes[0]][1]],
                [mesh.points[nodes[1]][0], mesh.points[nodes[1]][1]],
                [mesh.points[nodes[2]][0], mesh.points[nodes[2]][1]],
                [mesh.points[nodes[3]][0], mesh.points[nodes[3]][1]],
            ];
            if !convex_ccw(&coords) {
                return f64::INFINITY;
            }
        }
        for angle in quality::corner_angles(&mesh.cells[c], &mesh.points) {
            let sine = angle.to_radians().sin().max(0.02);
            f += (1.0 - sine) * (1.0 - sine) + 0.05 * (-sine.ln());
        }
    }
    f
}

/// True when `p` keeps every edge of the incident cells at a usable length.
///
/// Only edges that touch the moving node `v` are checked. The 0.15 x size floor
/// is a guard against a move that collapses an edge (and with it a quad corner);
/// it is a hard reject, unlike the objective's soft barrier.
fn spacing_ok(incident: &[usize], mesh: &Mesh, p: [f64; 2], v: usize, size: f64) -> bool {
    for &c in incident {
        for (i, &u) in mesh.cells[c].nodes.iter().enumerate() {
            let next = mesh.cells[c].nodes[(i + 1) % mesh.cells[c].nodes.len()];
            if u != v && next != v {
                continue;
            }
            let a = if u == v {
                p
            } else {
                [mesh.points[u][0], mesh.points[u][1]]
            };
            let b = if next == v {
                p
            } else {
                [mesh.points[next][0], mesh.points[next][1]]
            };
            let length = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
            if length < 0.15 * size {
                return false;
            }
        }
    }
    true
}

/// Triangulate the domain and refine it to the size field.
///
/// Pipeline: sample the loops (`sample_loops`), constrain every sampled segment
/// in the clean-room CDT, refine with circumcenters (the private `refine`), and
/// keep the triangles whose centroid is inside the domain.
///
/// # Preconditions
///
/// `domain` must be a valid [`Domain`] and `sizing.size` finite and positive.
///
/// # Postconditions
///
/// Returns `(points, triangles)` with `points` in the XY plane and every index
/// valid. Boundary edges of the input are present in the triangulation (up to
/// the sampling); triangles outside the domain (in a hole or outside the outer
/// loop) are filtered out. The same point can appear in several triangles but
/// is emitted once, keyed by its bit pattern.
///
/// # Approximations
///
/// The refinement is Ruppert's circumcenter rule (corpus
/// `shewchuk-2002-delaunay-refinement`) with a fixed boundary margin and a
/// uniform point grid instead of the full encroachment machinery; see
/// `refine` and `BOUNDARY_MARGIN`.
pub fn triangulate(
    domain: &Domain,
    sizing: &Sizing,
) -> Result<(Vec<Node>, Vec<[usize; 3]>), TwoDError> {
    // The struct is public, so `size` can be zero or negative even though
    // `Sizing::uniform` rejects it. Refinement with a zero size would subdivide
    // for ever (measured: the test process is OOM-killed), so refuse it here.
    if !sizing.size.is_finite() || sizing.size <= 0.0 {
        return Err(TwoDError::BadRequest(
            "the size must be positive and finite",
        ));
    }
    // Sample every boundary segment down to the size field before constraining
    // it. A plate edge 100 mm long with a 1 mm size would otherwise be a single
    // constrained edge, and every triangle touching it would be a sliver.
    let loops = sample_loops(domain, sizing);
    let Some((outer, holes)) = loops.split_first() else {
        return Err(TwoDError::Triangulation("the domain has no outer loop"));
    };
    let mut cdt = CdtAdapter::from_loops(outer.clone(), holes.to_vec())
        .map_err(|_| TwoDError::Triangulation("the boundary could not be constrained"))?;

    // Ruppert refinement on the constrained triangulation; it only inserts
    // points, never moves a boundary vertex, so the constraints stay valid.
    refine(&mut cdt, domain, sizing);

    let triangles = collect_triangles(&cdt, domain);
    if triangles.is_empty() {
        return Err(TwoDError::BadRequest("the domain triangulated to nothing"));
    }

    let mut points: Vec<Node> = Vec::new();
    let mut index_of: std::collections::HashMap<(u64, u64), usize> =
        std::collections::HashMap::new();
    let mut tris: Vec<[usize; 3]> = Vec::with_capacity(triangles.len());
    for tri in &triangles {
        let mut ids = [0usize; 3];
        for (slot, p) in tri.iter().enumerate() {
            let key = (p[0].to_bits(), p[1].to_bits());
            let id = *index_of.entry(key).or_insert_with(|| {
                points.push([p[0], p[1], 0.0]);
                points.len() - 1
            });
            ids[slot] = id;
        }
        tris.push(ids);
    }
    Ok((points, tris))
}

/// Boundary loops with each segment split so none is longer than the target.
///
/// Each input segment is divided into `ceil(length / (1.1 * target))` equal
/// pieces. The 1.1 factor leaves room for the refinement's own spacing rule: a
/// piece exactly at the target would be rejected by the point grid's spacing
/// test and could never be improved.
///
/// Returns one loop per input loop, outer first, with the end point of each
/// segment omitted (the next segment starts there); the closing segment is
/// therefore included implicitly.
fn sample_loops(domain: &Domain, sizing: &Sizing) -> Vec<Vec<[f64; 2]>> {
    let mut out = Vec::new();
    for loop_points in std::iter::once(&domain.outer).chain(domain.holes.iter()) {
        let mut sampled = Vec::new();
        for i in 0..loop_points.len() {
            let a = loop_points[i];
            let b = loop_points[(i + 1) % loop_points.len()];
            let length = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
            let target = sizing.size_at(a).max(1e-9);
            let parts = (length / (1.1 * target)).ceil().max(1.0) as usize;
            for k in 0..parts {
                let t = k as f64 / parts as f64;
                sampled.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
            }
        }
        out.push(sampled);
    }
    out
}

/// Insert circumcenters until no triangle is larger than the sizing field.
///
/// Two insertion rules, both Ruppert's (corpus
/// `shewchuk-2002-delaunay-refinement`): a triangle whose circumcenter is inside
/// and far enough from the boundary is split there; when the circumcenter is
/// outside the domain or closer to the boundary than the margin, the nearest
/// boundary edge is split at its midpoint instead. Without the second rule a
/// square stalls immediately — its triangles are right-angled, so every
/// circumcenter lands *on* an edge.
///
/// Only the insertion radius is controlled, not the full encroachment proof:
/// `BOUNDARY_MARGIN` and `MIN_SPACING_FRACTION` replace it. The loop is bounded
/// by `MAX_PASSES` and `MAX_POINTS`; on either bound it returns with the current
/// (still valid) triangulation rather than reporting an error.
fn refine(cdt: &mut CdtAdapter, domain: &Domain, sizing: &Sizing) {
    let mut grid = PointGrid::new(sizing.size);
    for &point in cdt.points() {
        grid.insert(point);
    }
    let mut inserted_total = 0usize;
    for _ in 0..MAX_PASSES {
        // Pass 1: collect one candidate per over-sized triangle. The candidate
        // is either the circumcenter (the Ruppert split) or, when the
        // circumcenter is unusable, the midpoint of the nearest boundary edge
        // (the boundary-recovery split).
        let mut candidates: Vec<[f64; 2]> = Vec::new();
        for triangle in cdt.triangles() {
            let p = triangle.vertices.map(|vertex| cdt.points()[vertex]);
            if !domain.contains(centroid(&p)) {
                continue;
            }
            let Some((center, radius)) = circumcenter(&p) else {
                continue;
            };
            let target = sizing.size_at(center);
            if radius <= target {
                continue;
            }
            let margin = domain.distance_to_boundary(center);
            let spacing_ok = grid.nearest(center) >= MIN_SPACING_FRACTION * target;
            if domain.contains(center) && margin >= BOUNDARY_MARGIN * target && spacing_ok {
                candidates.push(center);
            } else {
                let (a, b, _) = domain.nearest_segment(center);
                let length = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
                if length > 1.2 * target {
                    candidates.push([(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]);
                }
            }
        }
        if candidates.is_empty() {
            return;
        }
        // deduplicate: several triangles share a circumcenter on a symmetric
        // configuration, and re-inserting a point is a no-op that would never
        // terminate
        candidates.sort_by(|a, b| {
            (a[0], a[1])
                .partial_cmp(&(b[0], b[1]))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.dedup_by(|a, b| a == b);
        // Insert every candidate this pass. The earlier cap of 64 per pass
        // multiplied the number of O(faces) scans by ~8 for no benefit: the
        // Delaunay kernel is incremental, so the order does not matter and a
        // stale candidate is simply rejected by the spacing check.
        let mut inserted = 0;
        for &center in &candidates {
            if grid.nearest(center) < MIN_SPACING_FRACTION * sizing.size_at(center) {
                continue;
            }
            if cdt.insert_point(center).is_ok() {
                grid.insert(center);
                inserted += 1;
            }
        }
        if inserted == 0 {
            return;
        }
        inserted_total += inserted;
        if inserted_total > MAX_POINTS {
            return;
        }
    }
}

/// Hard cap on inserted points, so a bad size field cannot run away.
const MAX_POINTS: usize = 200_000;

/// Triangles whose centroid is inside the domain, as positions. The adapter
/// has already removed exterior and hole faces; the route retains its centroid
/// check as a defensive postcondition.
fn collect_triangles(cdt: &CdtAdapter, domain: &Domain) -> Vec<[[f64; 2]; 3]> {
    let mut out = Vec::new();
    for triangle in cdt.triangles() {
        let tri = triangle.vertices.map(|vertex| cdt.points()[vertex]);
        if domain.contains(centroid(&tri)) {
            out.push(tri);
        }
    }
    out
}

/// Arithmetic mean of a polygon's vertices; used as the inside test point.
fn centroid(p: &[[f64; 2]]) -> [f64; 2] {
    let n = p.len() as f64;
    [
        p.iter().map(|q| q[0]).sum::<f64>() / n,
        p.iter().map(|q| q[1]).sum::<f64>() / n,
    ]
}

/// Circumcenter and circumradius of a triangle, or `None` if it is degenerate.
///
/// A triangle whose doubled signed area (the denominator `d`) is below `1e-14`
/// is treated as degenerate: its circumcenter is at infinity, and inserting it
/// would be meaningless.
fn circumcenter(p: &[[f64; 2]]) -> Option<([f64; 2], f64)> {
    let (a, b, c) = (p[0], p[1], p[2]);
    let d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
    if d.abs() < 1e-14 {
        return None;
    }
    let a2 = a[0] * a[0] + a[1] * a[1];
    let b2 = b[0] * b[0] + b[1] * b[1];
    let c2 = c[0] * c[0] + c[1] * c[1];
    let ux = (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / d;
    let uy = (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / d;
    let center = [ux, uy];
    Some((center, ((a[0] - ux).powi(2) + (a[1] - uy).powi(2)).sqrt()))
}

/// Pair adjacent triangles into quads: maximum cardinality, then maximum
/// quality, via a general weighted matching (Edmonds' blossom algorithm).
///
/// This is [`pair_triangles_with`] with the shipped [`PairGuard::default`];
/// see there for the construction and the paper reference.
fn pair_triangles(tris: &[[usize; 3]], points: &[Node]) -> (Vec<[usize; 4]>, Vec<[usize; 3]>) {
    pair_triangles_with(tris, points, PairGuard::default())
}

/// Pair adjacent triangles into quads with explicit quality guards.
///
/// This is the matching the shipped path uses, exposed so a region of an
/// existing mesh can be paired under its own floor (the two-stage Q-Morph +
/// matching conversion). Every candidate quad must be convex, have scaled
/// Jacobian ≥ `guard.min_scaled_jacobian` and corner angles within
/// `[guard.min_angle_deg, guard.max_angle_deg]`; the rest is maximum-cardinality
/// matching, weighted by scaled Jacobian.
///
/// The construction is Blossom-Quad (Remacle et al. 2012, corpus
/// `remacle-2012`): a vertex per triangle, an edge for every pair that shares a
/// triangle edge and passes the guards, weight `round(1000 * min SJ)` clamped to
/// `[1, 1000]`. `mwmatching` solves it with Edmonds' blossom algorithm; the
/// cardinality pass runs before the weights, so the number of paired triangles
/// is maximised first and quality breaks ties. Ties are made deterministic by
/// sorting the edge list before the matching.
///
/// # Postconditions
///
/// `quads` and `unpaired` partition the input triangles: every input appears in
/// exactly one of them, no triangle is used twice, and a quad's four nodes keep
/// a positive (counter-clockwise) winding. `points` is not modified.
pub fn pair_triangles_with(
    tris: &[[usize; 3]],
    points: &[Node],
    guard: PairGuard,
) -> (Vec<[usize; 4]>, Vec<[usize; 3]>) {
    // Build the dual graph: edge key -> the (up to two) triangles that use it.
    // Only an edge with exactly two owners can become a quad; a boundary edge
    // (one owner) or a non-manifold edge (three or more) cannot.
    let mut edge_to_tris: std::collections::HashMap<(usize, usize), Vec<usize>> =
        std::collections::HashMap::new();
    for (t, tri) in tris.iter().enumerate() {
        for i in 0..3 {
            let (a, b) = (tri[i], tri[(i + 1) % 3]);
            let key = (a.min(b), a.max(b));
            edge_to_tris.entry(key).or_default().push(t);
        }
    }

    // Deterministic: the blossom matching takes the edges in insertion order, and
    // the result among equal-weight candidates depends on it. Collecting the keys
    // and sorting them makes the edge list independent of `HashMap` order; the
    // matching rule itself is unchanged.
    let mut keys: Vec<(usize, usize)> = edge_to_tris.keys().copied().collect();
    keys.sort_unstable();
    let mut edges: Vec<(usize, usize, i32)> = Vec::new();
    for key in keys {
        let Some(owners) = edge_to_tris.get(&key) else {
            continue;
        };
        if owners.len() != 2 {
            continue;
        }
        let (t1, t2) = (owners[0], owners[1]);
        let Some(quad) = quad_from_pair(&tris[t1], &tris[t2], points) else {
            continue;
        };
        let cell = Cell::quad4(quad);
        let sj = quality::scaled_jacobian(&cell, points);
        let worst = sj.iter().copied().fold(f64::INFINITY, f64::min);
        if !worst.is_finite() || worst < guard.min_scaled_jacobian {
            continue;
        }
        let angles = quality::quad_angles(&cell, points);
        let min_angle = angles.iter().copied().fold(f64::INFINITY, f64::min);
        let max_angle = angles.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !min_angle.is_finite() || min_angle < guard.min_angle_deg {
            continue;
        }
        if !max_angle.is_finite() || max_angle > guard.max_angle_deg {
            continue;
        }
        // weight: quality in [0, 1] scaled to an integer; higher is better.
        // Clamped to at least 1 so a quad that passed the guards is never
        // mistaken for "no edge"; the matching is maximising, not minimising.
        let weight = (1000.0 * worst).round().clamp(1.0, 1000.0) as i32;
        edges.push((t1, t2, weight));
    }

    // Blossom-Quad: first maximise the number of matched triangles (the
    // cardinality stage), then maximise the total weight among those matchings.
    // The cardinality stage is what makes the result all-quad; without it the
    // weighted stage would leave a high-quality triangle unpaired to keep a
    // slightly better quad elsewhere.
    let mut matching = Matching::new(edges);
    matching.max_cardinality();
    let mate = matching.solve();

    // Rebuild the cells from the matching. Each mating edge is visited once
    // (the `other <= t` guard), and `used` protects against a malformed mate
    // array; a triangle absent from `mate` is unpaired, not an error.
    let mut quads: Vec<[usize; 4]> = Vec::new();
    let mut used = vec![false; tris.len()];
    for t in 0..tris.len() {
        // `mate` is sized by the vertices that appear in an edge: a triangle
        // with no pairable neighbour is simply absent from it.
        let partner = mate.get(t).copied().unwrap_or(SENTINEL);
        if used[t] || partner == SENTINEL {
            continue;
        }
        let other = partner;
        if other >= tris.len() || used[other] || other <= t {
            continue;
        }
        let Some(quad) = quad_from_pair(&tris[t], &tris[other], points) else {
            continue;
        };
        used[t] = true;
        used[other] = true;
        quads.push(quad);
    }

    let unpaired = tris
        .iter()
        .enumerate()
        .filter(|(t, _)| !used[*t])
        .map(|(_, tri)| *tri)
        .collect();
    (quads, unpaired)
}

/// Strictly convex, counter-clockwise and non-degenerate.
///
/// `1e-12` is on the doubled signed area, i.e. about `5e-13` in area units; a
/// quad below it is treated as a sliver. The four cross products then require
/// every corner to turn left by a positive amount, so a concave quad and a
/// collinear corner both fail.
fn convex_ccw(q: &[[f64; 2]; 4]) -> bool {
    let mut area = 0.0;
    for i in 0..4 {
        let a = q[i];
        let b = q[(i + 1) % 4];
        area += a[0] * b[1] - b[0] * a[1];
    }
    if area <= 1e-12 {
        return false;
    }
    (0..4).all(|i| {
        let a = q[(i + 3) % 4];
        let b = q[i];
        let c = q[(i + 1) % 4];
        let u = [b[0] - a[0], b[1] - a[1]];
        let v = [c[0] - b[0], c[1] - b[1]];
        u[0] * v[1] - u[1] * v[0] > 0.0
    })
}

/// Scaled Jacobian of a quad given by coordinates, for callers outside this
/// module.
///
/// Returns the minimum over the four corners, in `[-1, 1]` (1 is a square, 0 a
/// degenerate corner, negative an inverted one). The coordinates must be in
/// counter-clockwise order for the sign to mean anything.
pub fn quad_min_sj_public(q: &[[f64; 2]; 4]) -> f64 {
    quad_min_sj(q)
}

/// The quality contract's scaled Jacobian, for a quad given as coordinates.
///
/// Thin wrapper over `quality::scaled_jacobian` on a temporary `Cell::quad4`;
/// the node ids are irrelevant to the metric, so `[0, 1, 2, 3]` is used. Returns
/// `+inf` for an empty iterator (not reachable with four corners) and the
/// minimum corner value otherwise.
fn quad_min_sj(q: &[[f64; 2]; 4]) -> f64 {
    let pts: Vec<Node> = q.iter().map(|p| [p[0], p[1], 0.0]).collect();
    let cell = Cell::quad4([0, 1, 2, 3]);
    quality::scaled_jacobian(&cell, &pts)
        .into_iter()
        .fold(f64::INFINITY, f64::min)
}

/// Close leftover triangles by a conforming refinement cascade.
///
/// A triangle is an odd polygon: it cannot become quads without adding a vertex
/// on its boundary. Splitting its three edges at their midpoints and joining
/// them to the centroid gives three quads, but each midpoint is then a *hanging
/// node* on the quad across that edge. That quad is split by joining the hanging
/// node to the midpoint of its opposite edge, which creates a new hanging node
/// further out, and the split propagates until it reaches the domain boundary,
/// where a midpoint is simply a new boundary point.
///
/// Each triangle is attempted on its own copy: one that cannot be closed (a
/// cycle, a split that would degenerate, a quad collecting three hanging nodes)
/// is left as a triangle and reported, instead of sinking the whole mesh.
///
/// This is a local construction of this mesher, not a paper template: Sun et al.
/// (corpus `sun-2015`, `sun-2016`) close residuals without refining, and that
/// path is [`crate::sun::close_residuals`]. The cascade is exact and conforming
/// but grows the element count; see the measured note on [`Sizing::all_quad`].
///
/// # Postconditions
///
/// `quads`, `points` and `leftover` tile exactly the input: every input quad is
/// still present (possibly split into more quads), and every input triangle is
/// either replaced by a valid cascade of quads or returned in `leftover`
/// unchanged. A component that shares an edge with a triangle that is not being
/// closed is never split, so no crack is introduced.
pub(crate) fn close_by_refinement(
    quads: &[[usize; 4]],
    unpaired: &[[usize; 3]],
    points: &[Node],
    domain: &Domain,
) -> (Vec<[usize; 4]>, Vec<Node>, Vec<[usize; 3]>) {
    let mut out_quads = quads.to_vec();
    let mut out_points = points.to_vec();
    let mut leftover = Vec::new();

    // Leftover triangles that share an edge must be closed together or not at
    // all: closing one splits the shared edge, which would leave the other with
    // an edge nothing else has — a crack.
    let components = leftover_components(unpaired);
    let mut remaining: Vec<[usize; 3]> = unpaired.to_vec();
    let mut edges_of: Vec<std::collections::HashSet<(usize, usize)>> = Vec::new();
    for component in &components {
        let mut set = std::collections::HashSet::new();
        for tri in component {
            for k in 0..3 {
                let (a, b) = (tri[k], tri[(k + 1) % 3]);
                set.insert((a.min(b), a.max(b)));
            }
        }
        edges_of.push(set);
    }
    for (index, component) in components.into_iter().enumerate() {
        // edges of the triangles we are *not* closing: splitting one would leave
        // that triangle with an edge nothing else has
        let mut protected: std::collections::HashSet<(usize, usize)> =
            std::collections::HashSet::new();
        for (other, set) in edges_of.iter().enumerate() {
            if other != index {
                protected.extend(set.iter().copied());
            }
        }
        let mut quads = out_quads.clone();
        let mut pts = out_points.clone();
        let mut ok = true;
        for tri in &component {
            match close_one(&quads, &pts, tri, &protected) {
                Some((q, p)) => {
                    quads = q;
                    pts = p;
                }
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if ok && no_dangling_edges(&quads, &remaining, &component, &pts, domain) {
            out_quads = quads;
            out_points = pts;
            remaining.retain(|t| !component.contains(t));
        } else {
            leftover.extend(component);
        }
    }
    (out_quads, out_points, leftover)
}

/// True when no edge of the mesh is used exactly once off the domain boundary.
///
/// A cascade that splits an edge only one of the two cells sees leaves a
/// T-junction: the edge is used once inside the domain, which is what this
/// checks for. Boundary edges are legitimately used once.
fn no_dangling_edges(
    quads: &[[usize; 4]],
    remaining: &[[usize; 3]],
    component: &[[usize; 3]],
    points: &[Node],
    domain: &Domain,
) -> bool {
    let mut count: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
    for q in quads {
        for i in 0..4 {
            let (a, b) = (q[i], q[(i + 1) % 4]);
            *count.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    for t in remaining.iter().filter(|t| !component.contains(t)) {
        for i in 0..3 {
            let (a, b) = (t[i], t[(i + 1) % 3]);
            *count.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    for (&(a, b), &n) in &count {
        if n == 2 {
            continue;
        }
        let on_boundary = |v: usize| {
            let p = [points[v][0], points[v][1]];
            domain.distance_to_boundary(p) < 1e-9
        };
        if n == 1 && on_boundary(a) && on_boundary(b) {
            continue;
        }
        return false;
    }
    true
}

/// Groups of leftover triangles connected through shared edges.
fn leftover_components(unpaired: &[[usize; 3]]) -> Vec<Vec<[usize; 3]>> {
    let mut edge_owner: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
    let mut parent: Vec<usize> = (0..unpaired.len()).collect();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }
    for (i, tri) in unpaired.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (tri[k], tri[(k + 1) % 3]);
            let key = (a.min(b), a.max(b));
            if let Some(&other) = edge_owner.get(&key) {
                let (ri, ro) = (find(&mut parent, i), find(&mut parent, other));
                if ri != ro {
                    parent[ri] = ro;
                }
            } else {
                edge_owner.insert(key, i);
            }
        }
    }
    let mut groups: std::collections::HashMap<usize, Vec<[usize; 3]>> =
        std::collections::HashMap::new();
    for i in 0..unpaired.len() {
        let root = find(&mut parent, i);
        groups.entry(root).or_default().push(unpaired[i]);
    }
    // Deterministic: components are closed in this order and appended to the
    // mesh, so `HashMap` order would reorder the cells. Lowest component root
    // (the smallest triangle index) first.
    let mut groups: Vec<(usize, Vec<[usize; 3]>)> = groups.into_iter().collect();
    groups.sort_unstable_by_key(|(root, _)| *root);
    groups.into_iter().map(|(_, group)| group).collect()
}

/// The cascade for one triangle, or `None` if it cannot be completed.
///
/// It is driven by *edges*: split an edge, and every quad that owns it gains a
/// hanging node and must be split across its opposite edge, which enqueues that
/// edge in turn. Each edge splits once and each quad splits once, so it ends.
fn close_one(
    quads: &[[usize; 4]],
    points: &[Node],
    tri: &[usize; 3],
    protected: &std::collections::HashSet<(usize, usize)>,
) -> Option<(Vec<[usize; 4]>, Vec<Node>)> {
    let mut pts = points.to_vec();
    let mut cells: Vec<Vec<usize>> = quads.iter().map(|q| q.to_vec()).collect();
    let tri_cell = cells.len();
    cells.push(tri.to_vec());

    let mut mids: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
    let mut point_ids: std::collections::HashMap<(u64, u64), usize> = pts
        .iter()
        .enumerate()
        .map(|(id, p)| ((p[0].to_bits(), p[1].to_bits()), id))
        .collect();
    let mut split: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut queue: Vec<(usize, usize)> = (0..3).map(|i| (tri[i], tri[(i + 1) % 3])).collect();
    let mut steps = 0usize;

    while let Some((a, b)) = queue.pop() {
        steps += 1;
        if steps > MAX_CASCADE_STEPS {
            return None; // too expensive: keep the triangle
        }
        if protected.contains(&(a.min(b), a.max(b))) {
            return None; // belongs to a triangle we are not closing
        }
        let m = midpoint(&mut pts, &mut mids, &mut point_ids, a, b);
        let owners: Vec<usize> = (0..cells.len())
            .filter(|&i| cells[i].len() == 4 && has_edge(&cells[i], a, b))
            .collect();
        for c in owners {
            if !split.insert(c) {
                continue;
            }
            let q = cells[c].clone();
            let position = (0..4).find(|&i| {
                let (x, y) = (q[i], q[(i + 1) % 4]);
                (x.min(y), x.max(y)) == (a.min(b), a.max(b))
            })?;
            let (p0, p1, p2, p3) = (
                q[position],
                q[(position + 1) % 4],
                q[(position + 2) % 4],
                q[(position + 3) % 4],
            );
            if protected.contains(&(p2.min(p3), p2.max(p3))) {
                return None;
            }
            let m2 = midpoint(&mut pts, &mut mids, &mut point_ids, p2, p3);
            let two = [[p0, m, m2, p3], [m, p1, p2, m2]];
            if !two.iter().all(|x| valid_quad(x, &pts)) {
                return None;
            }
            cells[c] = two[0].to_vec();
            cells.push(two[1].to_vec());
            // the opposite edge is now split, so its neighbour must follow
            queue.push((p2, p3));
        }
    }

    // now the triangle: three quads around its centroid, using the midpoints the
    // cascade put on its edges
    let mut m = [0usize; 3];
    for i in 0..3 {
        let key = (tri[i].min(tri[(i + 1) % 3]), tri[i].max(tri[(i + 1) % 3]));
        m[i] = *mids.get(&key)?;
    }
    let c = [
        (pts[tri[0]][0] + pts[tri[1]][0] + pts[tri[2]][0]) / 3.0,
        (pts[tri[0]][1] + pts[tri[1]][1] + pts[tri[2]][1]) / 3.0,
    ];
    pts.push([c[0], c[1], 0.0]);
    let centre = pts.len() - 1;
    let fan = [
        [tri[0], m[0], centre, m[2]],
        [tri[1], m[1], centre, m[0]],
        [tri[2], m[2], centre, m[1]],
    ];
    if !fan.iter().all(|q| valid_quad(q, &pts)) {
        return None;
    }
    cells[tri_cell] = fan[0].to_vec();
    cells.push(fan[1].to_vec());
    cells.push(fan[2].to_vec());

    if cells.iter().any(|c| c.len() != 4) {
        return None;
    }
    Some((
        cells.iter().map(|c| [c[0], c[1], c[2], c[3]]).collect(),
        pts,
    ))
}

/// True when the cell's boundary contains the undirected edge `a`-`b`.
fn has_edge(cell: &[usize], a: usize, b: usize) -> bool {
    (0..cell.len()).any(|i| {
        let (x, y) = (cell[i], cell[(i + 1) % cell.len()]);
        (x.min(y), x.max(y)) == (a.min(b), a.max(b))
    })
}

/// Return the midpoint of `a`-`b`, inserting it once.
///
/// The midpoint node is memoised by the *sorted* edge key, because a cascade can
/// ask for the same edge from both of its cells. Exact coordinates also reuse a
/// midpoint created while closing an adjacent residual triangle; returning a
/// fresh node twice would create two coincident points and a crack. The new node
/// has `z = 0`.
fn midpoint(
    pts: &mut Vec<Node>,
    mids: &mut std::collections::HashMap<(usize, usize), usize>,
    point_ids: &mut std::collections::HashMap<(u64, u64), usize>,
    a: usize,
    b: usize,
) -> usize {
    let key = (a.min(b), a.max(b));
    if let Some(&m) = mids.get(&key) {
        return m;
    }
    let p = [
        0.5 * (pts[a][0] + pts[b][0]),
        0.5 * (pts[a][1] + pts[b][1]),
        0.0,
    ];
    let coordinates = (p[0].to_bits(), p[1].to_bits());
    if let Some(&id) = point_ids.get(&coordinates) {
        mids.insert(key, id);
        return id;
    }
    pts.push(p);
    let id = pts.len() - 1;
    mids.insert(key, id);
    point_ids.insert(coordinates, id);
    id
}

/// A convex, counter-clockwise quad with a usable scaled Jacobian (`> 0.02`).
///
/// The cascade uses this on every quad it is about to create; a cascade step
/// that would produce a worse quad is refused and the triangle is left alone.
fn valid_quad(q: &[usize; 4], pts: &[Node]) -> bool {
    let coords = [
        [pts[q[0]][0], pts[q[0]][1]],
        [pts[q[1]][0], pts[q[1]][1]],
        [pts[q[2]][0], pts[q[2]][1]],
        [pts[q[3]][0], pts[q[3]][1]],
    ];
    convex_ccw(&coords) && quad_min_sj(&coords) > 0.02
}

/// Build the quad from two triangles that share an edge.
///
/// The shared edge must be taken in each triangle's own winding: triangle `a`
/// traverses it one way, triangle `b` the other. Walking the *sorted* edge
/// instead produces a bowtie for half the pairs — which is exactly what made the
/// first version of this function refuse every mesh.
fn quad_from_pair(a: &[usize; 3], b: &[usize; 3], points: &[Node]) -> Option<[usize; 4]> {
    for i in 0..3 {
        let u = a[i];
        let v = a[(i + 1) % 3];
        let c1 = a[(i + 2) % 3];
        for j in 0..3 {
            if b[j] == v && b[(j + 1) % 3] == u {
                let c2 = b[(j + 2) % 3];
                let mut quad = [u, c1, v, c2];
                if polygon_signed_area(&quad, points) < 0.0 {
                    quad.reverse();
                }
                if polygon_signed_area(&quad, points).abs() < 1e-18 {
                    return None;
                }
                return Some(quad);
            }
        }
    }
    None
}

/// Signed area of a quad from the coordinates; positive when counter-clockwise.
fn polygon_signed_area(quad: &[usize; 4], points: &[Node]) -> f64 {
    let mut sum = 0.0;
    for i in 0..quad.len() {
        let a = points[quad[i]];
        let b = points[quad[(i + 1) % quad.len()]];
        sum += a[0] * b[1] - b[0] * a[1];
    }
    sum / 2.0
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::mesh::Kind;

    /// Strictly convex, counter-clockwise and non-degenerate.
    fn is_convex_ccw(quad: &[usize; 4], points: &[Node]) -> bool {
        if polygon_signed_area(quad, points) <= 1e-12 {
            return false;
        }
        (0..4).all(|i| {
            let a = points[quad[(i + 3) % 4]];
            let b = points[quad[i]];
            let c = points[quad[(i + 1) % 4]];
            let u = [b[0] - a[0], b[1] - a[1]];
            let v = [c[0] - b[0], c[1] - b[1]];
            u[0] * v[1] - u[1] * v[0] > 0.0
        })
    }

    #[test]
    fn a_rectangle_has_the_expected_area() {
        let d = Domain::rectangle(100.0, 60.0).unwrap();
        assert!((d.area() - 6000.0).abs() < 1e-9);
    }

    #[test]
    fn a_plate_with_a_hole_subtracts_it() {
        let d = Domain::plate_with_hole(100.0, 60.0, 10.0, 64).unwrap();
        let expected = 6000.0 - std::f64::consts::PI * 100.0;
        assert!(
            (d.area() - expected).abs() < 2.0,
            "{} vs {}",
            d.area(),
            expected
        );
    }

    #[test]
    fn contains_respects_the_hole() {
        let d = Domain::plate_with_hole(100.0, 60.0, 10.0, 64).unwrap();
        assert!(d.contains([10.0, 10.0]));
        assert!(!d.contains([50.0, 30.0])); // the hole
        assert!(!d.contains([-1.0, 10.0]));
    }

    #[test]
    fn a_square_is_quad_dominant() {
        let d = Domain::rectangle(10.0, 10.0).unwrap();
        let r = mesh_2d(&d, &Sizing::uniform(2.0).unwrap()).unwrap();
        assert!(
            r.stats.quad_ratio() >= 0.75,
            "quad ratio {:.2}, {:?}",
            r.stats.quad_ratio(),
            r.stats
        );
        assert_eq!(
            r.mesh.cells.len(),
            r.stats.quads + r.stats.unpaired_triangles
        );
    }

    #[test]
    fn every_quad_is_convex_and_positively_oriented() {
        let d = Domain::plate_with_hole(100.0, 60.0, 10.0, 64).unwrap();
        let r = mesh_2d(&d, &Sizing::uniform(4.0).unwrap()).unwrap();
        for cell in r.mesh.cells.iter().filter(|c| c.kind == Kind::Quad4) {
            let nodes: [usize; 4] = cell.nodes.as_slice().try_into().unwrap();
            assert!(is_convex_ccw(&nodes, &r.mesh.points), "{nodes:?}");
        }
    }

    #[test]
    fn all_quad_mode_leaves_no_triangles_where_it_can_afford_it() {
        // The 4 mm plate is small enough that every cascade finishes: the result
        // has no triangles at all, and is still watertight.
        let d = Domain::plate_with_hole(100.0, 60.0, 10.0, 16).unwrap();
        let r = mesh_2d(&d, &Sizing::uniform(4.0).unwrap().all_quad()).unwrap();
        assert_eq!(r.stats.unpaired_triangles, 0, "{:?}", r.stats);
        assert!(r.mesh.cells.iter().all(|c| c.kind == Kind::Quad4));
        let topo = r.mesh.topology();
        assert_eq!(topo.cracks, 0, "the cascade left a hole");
        assert_eq!(topo.inconsistent, 0);
        assert_eq!(topo.non_manifold, 0);
    }

    #[test]
    fn all_quad_mode_is_watertight_on_a_fine_plate_too() {
        // Fewer triangles close here (chains hit the step cap), but whatever is
        // left is kept and the mesh must stay watertight.
        let d = Domain::plate_with_hole(100.0, 60.0, 10.0, 24).unwrap();
        let r = mesh_2d(&d, &Sizing::uniform(2.0).unwrap().all_quad()).unwrap();
        let topo = r.mesh.topology();
        assert_eq!(topo.cracks, 0);
        let tris = r.mesh.cells.iter().filter(|c| c.kind == Kind::Tri3).count();
        assert_eq!(tris, r.stats.unpaired_triangles);
    }

    #[test]
    fn the_mesh_is_watertight_even_with_leftover_triangles() {
        let d = Domain::plate_with_hole(100.0, 60.0, 10.0, 64).unwrap();
        let r = mesh_2d(&d, &Sizing::uniform(4.0).unwrap()).unwrap();
        let topo = r.mesh.topology();
        assert_eq!(
            topo.cracks, 0,
            "leftover triangles were dropped, leaving a hole"
        );
        assert_eq!(topo.inconsistent, 0);
        assert_eq!(topo.non_manifold, 0);
        // every leftover triangle is in the mesh, not lost
        let tris = r.mesh.cells.iter().filter(|c| c.kind == Kind::Tri3).count();
        assert_eq!(tris, r.stats.unpaired_triangles);
    }

    #[test]
    fn the_square_mesh_is_valid_and_has_no_inverted_quad() {
        let d = Domain::rectangle(10.0, 10.0).unwrap();
        let r = mesh_2d(&d, &Sizing::uniform(2.0).unwrap()).unwrap();
        assert_eq!(r.mesh.validate(), Ok(()));
        let report = quality::report(&r.mesh);
        assert_eq!(report.inverted, 0, "{}", report.summary());
    }

    #[test]
    fn the_plate_with_a_hole_meshes_at_three_sizes() {
        for size in [4.0, 2.0, 1.0] {
            let d = Domain::plate_with_hole(100.0, 60.0, 10.0, 64).unwrap();
            let r = mesh_2d(&d, &Sizing::uniform(size).unwrap()).unwrap();
            assert!(!r.mesh.cells.is_empty(), "size {size}");
            assert_eq!(r.mesh.validate(), Ok(()), "size {size}");
            let report = quality::report(&r.mesh);
            assert_eq!(report.inverted, 0, "size {size}: {}", report.summary());
        }
    }

    #[test]
    fn a_bad_size_is_rejected() {
        assert!(Sizing::uniform(0.0).is_err());
        assert!(Sizing::uniform(-1.0).is_err());
        assert!(Sizing::uniform(f64::NAN).is_err());
    }
}
