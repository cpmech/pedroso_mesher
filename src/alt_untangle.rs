//! Untangling and singularity reduction as a post-pass over the product quad
//! route.
//!
//! **Papers.** Sastry, Shontz & Vavasis (2014), *A log-barrier method for mesh
//! quality improvement and untangling*: the objective is the log-barrier
//! `sum log(s_j)` over the corner scaled Jacobians, maximised by moving the
//! interior nodes while the boundary nodes stay on the boundary; an inverted
//! element contributes its signed Jacobian instead of a logarithm, so the same
//! objective pulls it back to the feasible side. Verma & Suresh (2017, 2018):
//! quadrilateral singularities (valence != 4) are reduced by local patch
//! re-quadrangulation; here only the atomic `O(1)` operation is implemented —
//! an **interior edge swap** between two quads (the other diagonal of their
//! hexagon) accepted when it lowers `sum |valence - 4|` on interior nodes and
//! keeps both new quads above the gate. That is the combinatorial kernel of the
//! MST templates, not the templates themselves.
//!
//! **Steps implemented**
//! 1. Run the product pipeline ([`crate::quad::mesh_quads`]); on
//!    `Err(QuadError::Unsupported)` keep the carried mesh (the failing mesh is
//!    exactly what this pass is for).
//! 2. Log-barrier untangling: a deterministic coordinate search over the nodes
//!    incident to low-quality quads. Boundary nodes are held fixed: the paper
//!    also holds them fixed on sharp-feature domains, and sliding a node by
//!    projection onto the nearest segment lets a domain corner wander along an
//!    adjacent edge (measured: (0,0) ended at (0,1.08), still "on the
//!    boundary" while cutting the corner off). A move is kept only when every
//!    incident cell stays valid and no incident triangle gets thinner; the
//!    objective is the sum of the corner-Jacobian log barriers, with the signed
//!    Jacobian of a non-positive quad as the hybrid untangling term. A
//!    deterministic evaluation budget bounds the cost on the dense parts.
//! 3. Singularity reduction: greedy safe interior edge swaps. A swap must not
//!    touch a boundary edge, must produce two simple positive quads with all
//!    corner Jacobians >= 0.2 and corner angles >= 30 deg, must not recreate an
//!    existing edge, and must lower `sum |valence - 4|`. It may lower a high
//!    Jacobian/angle as long as the gate thresholds stay met, because the
//!    topological quality is the target once the gate passes.
//! 4. A second, shorter untangling pass polishes the quads created by the swaps.
//!
//! **Not implemented / limits.** The full MST patch search and replacement
//! (BFS patch construction, template mapping through mean-value coordinates):
//! the swaps are local and cannot remove a singularity that needs a patch.
//! The plate-hole 1 mm zero Jacobian is a **boundary corner doublet** — a
//! boundary node whose two collinear boundary edges belong to one quad, so its
//! corner angle is 180 deg and no interior-node motion can change it. An
//! interior swap can split that vertex across two quads, but it then makes the
//! next boundary vertex the doublet: the defect walks along the boundary and,
//! on a closed smooth loop, never disappears (measured: 250 reported new
//! cells, min SJ still 0.000, v4 43.6 -> 49.3 %). Removing it needs a boundary
//! collapse or an MST patch, both outside this pass. No boundary-node insertion
//! or collapse. The pass cannot remove leftover triangles: the ring and the
//! 0.6 mm performance part stay failing for that reason (their quads are
//! improved, their triangle count is not).
//!
//! **Doublet / boundary-flat guards.** The pass protects against creating a
//! flat boundary corner and against mistaking one for a repairable corner:
//! `untangle` holds every boundary node fixed (a boundary node may only slide
//! along a straight segment, which cannot change a 180 deg corner);
//! `reduce_singularities` considers only edges shared by exactly two quads,
//! so a boundary edge is never swapped away; `quad_is_simple` and
//! `quad_gate_ok` reject a new quad with a collapsed or inverted corner; and
//! the area-preservation check rejects a swap whose new pair does not account
//! for the old pair's area (a proxy against folding a new quad over a
//! neighbour), and the whole-mesh gate still rejects any overlap that appears.
//! The degenerate-corner repair path in `best_swap` is the only operation aimed
//! at a zero Jacobian, and it can only split the doublet, never remove it.
//!
//! **Measured result (7-part benchmark).** 4/7 gate PASS after this pass and
//! after the common post-pass: plate-hole 4 mm, plate-hole 2 mm, l-bracket and
//! stepped pass; plate-hole 1 mm (boundary doublet, min SJ 0.000), ring (2
//! triangles left) and the 0.6 mm performance part (5 triangles left) stay
//! failing. No part is lifted from FAIL to PASS, because the pass can neither
//! create nor remove a cell. It improves the topology of every part: the
//! interior valence-4 share rises by 1-6 points (measured 42.6 -> 47.1 % on
//! plate-hole 4 mm, 43.6 -> 49.3 % on 1 mm, 40.1 -> 45.3 % on the 0.6 mm part),
//! and the pass is cheap on the passing parts (0.00-0.11 s; 5.4 s on the
//! performance part). The printed `swaps` counter is `count_changes`'s count
//! of new cells, not the number of accepted swaps: one swap replaces two quads,
//! so the printed 6-615 new cells on the corpus are roughly 3-308 swaps,
//! against `max_swaps` = 600 accepted swaps.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::mesh::{Cell, Kind, Mesh, Node};
use crate::quad::{self, QuadError};
use crate::two_d::{Domain, Sizing};
use crate::verify;

/// The gate's scaled-Jacobian floor (`0.2`, dimensionless); a swap may not
/// lower a value that is already below it.
const SJ_FLOOR: f64 = 0.2;
/// The gate's minimum corner-angle floor (`30` degrees, relaxed by the user on
/// 2026-10-22); a swap may not lower a value that is already below it.
const ANGLE_FLOOR: f64 = 30.0;
/// A corner Jacobian at or below this is treated as a degeneracy by the
/// untangler (a floating-point zero, not "a bad but valid element").
const DEGENERATE_SJ: f64 = 1.0e-6;
/// A quad whose worst corner Jacobian is below this (dimensionless) makes its
/// nodes active in the untangler.
const ACTIVE_SJ: f64 = 0.4;
/// A quad whose smallest corner angle is below this (degrees) makes its nodes
/// active in the untangler.
const ACTIVE_ANGLE: f64 = 33.0;
/// Default rounds of the coordinate search in one untangling pass.
const UNTANGLE_ROUNDS: usize = 16;
/// Candidate directions per radius (8 = one every 45 degrees).
const DIRECTIONS: usize = 8;
/// Candidate step radii as fractions of the local element size; the four
/// values are the measured stencil (2 %, 5 %, 10 % and 20 % of `size`).
const RADII: [f64; 4] = [0.02, 0.05, 0.1, 0.2];
/// Upper bound on accepted swaps in one singularity pass (one swap per loop
/// iteration, not per untangling round).
const MAX_SWAPS: usize = 600;
/// Deterministic cap on candidate evaluations in **each** untangling pass. The
/// 0.6 mm performance part has thousands of low quads, and the log barrier is
/// not worth a quadratic time budget on a mesh the triangles already fail.
const EVAL_BUDGET: usize = 1_500_000;

/// Options for the untangle route.
///
/// The defaults are the measured constants of the route; changing a field only
/// changes a loop bound or an evaluation cap, never the algorithm.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Options {
    /// Rounds of the coordinate search in **each** log-barrier pass (default
    /// `UNTANGLE_ROUNDS` = 16). One round visits every active interior node
    /// once; the loop stops early when a round moves nothing. Units: sweeps;
    /// `0` disables the two untangling passes (the swaps still run).
    pub untangle_rounds: usize,

    /// Upper bound on accepted interior edge swaps in the singularity pass
    /// (default `MAX_SWAPS` = 600). The loop also stops at the first candidate
    /// the whole-mesh gate rejects and when `best_swap` finds no improving
    /// swap. Units: swaps (unitless); it is a cap, not a target.
    pub max_swaps: usize,

    /// Deterministic cap on candidate evaluations in **each** call of
    /// `untangle` (default `EVAL_BUDGET` = 1 500 000). Counts point states
    /// tried, not nodes or moves. The two passes have separate counters, so one
    /// route run can spend at most `2 * eval_budget`; the cap is what keeps the
    /// 0.6 mm part (thousands of low quads) from an unbounded cost.
    pub eval_budget: usize,
}

impl Default for Options {
    /// The measured defaults (`UNTANGLE_ROUNDS`, `MAX_SWAPS`,
    /// `EVAL_BUDGET`); the benchmark table was produced with these.
    fn default() -> Self {
        Self {
            untangle_rounds: UNTANGLE_ROUNDS,
            max_swaps: MAX_SWAPS,
            eval_budget: EVAL_BUDGET,
        }
    }
}

/// The measured result of one run: the product mesh and the improved mesh.
///
/// The two meshes share the point numbering of the product pipeline; the pass
/// never inserts or removes a node or a cell.
pub struct Outcome {
    /// The mesh the product pipeline produced, before this pass (the raw row):
    /// exactly what `quad::mesh_quads` returned, or the mesh carried by
    /// `QuadError::Unsupported` when that pipeline refused the configuration.
    pub raw: Mesh,

    /// The best mesh this pass produced: one of `raw`, the first untangling,
    /// the swap result or the second untangling, whichever wins the gate order
    /// in `mesh_key`. It always has the same cells count and node count as
    /// `raw`.
    pub improved: Mesh,

    /// Wall-clock seconds spent in the product pipeline (units: seconds,
    /// measured with `std::time::Instant`).
    pub base_secs: f64,

    /// Wall-clock seconds spent in this post-pass (both untangle passes and the
    /// swap loop; units: seconds).
    pub improve_secs: f64,

    /// `count_changes`'s count of improved cells that are not present in
    /// `raw`, not the number of accepted swaps: one swap contributes up to two
    /// new quads, so this is roughly twice the swap count (units: cells). The
    /// example prints it as `swaps`, which is why it can exceed
    /// [`Options::max_swaps`].
    pub swaps: usize,

    /// `count_changes`'s count of nodes whose final coordinates differ from
    /// `raw`; a node moved in both untangling passes is counted once (units:
    /// nodes, not moves).
    pub moves: usize,
}

/// Mesh a planar domain with the product pipeline, then untangle and reduce the
/// singularities of the result.
pub fn mesh(domain: &Domain, sizing: &Sizing) -> Result<Mesh, String> {
    mesh_with_options(domain, sizing, &Options::default())
}

/// [`mesh`] with explicit route options.
pub fn mesh_with_options(
    domain: &Domain,
    sizing: &Sizing,
    options: &Options,
) -> Result<Mesh, String> {
    Ok(mesh_with_outcome_options(domain, sizing, options)?.improved)
}

/// [`mesh`], also returning the product mesh it started from and the counters.
///
/// The example uses this to print the raw (product) and post (improved) rows
/// from one run of the product pipeline.
pub fn mesh_with_outcome(domain: &Domain, sizing: &Sizing) -> Result<Outcome, String> {
    mesh_with_outcome_options(domain, sizing, &Options::default())
}

/// [`mesh_with_outcome`] with explicit route options.
pub fn mesh_with_outcome_options(
    domain: &Domain,
    sizing: &Sizing,
    options: &Options,
) -> Result<Outcome, String> {
    let t0 = std::time::Instant::now();
    let raw = product_mesh(domain, sizing)?;
    let base_secs = t0.elapsed().as_secs_f64();

    let t1 = std::time::Instant::now();
    let size = sizing.size;
    let mut best = raw.clone();
    let mut best_key = mesh_key(&best);

    // 1. Log-barrier untangling.
    let first = untangle(domain, &best, size, options);
    consider(&mut best, &mut best_key, first);
    // 2. Valence-driven edge swaps plus the degenerate-corner repair.
    let swapped = reduce_singularities(&best, options);
    consider(&mut best, &mut best_key, swapped);
    // 3. Polish the quads the swaps created (a shorter second barrier pass).
    let polish = untangle(domain, &best, size, options);
    consider(&mut best, &mut best_key, polish);

    let improve_secs = t1.elapsed().as_secs_f64();
    let (swaps, moves) = count_changes(&raw, &best);
    Ok(Outcome {
        raw,
        improved: best,
        base_secs,
        improve_secs,
        swaps,
        moves,
    })
}

/// Run the product pipeline, keeping the mesh a failing run carries for
/// diagnosis (that mesh is the input of this post-pass).
fn product_mesh(domain: &Domain, sizing: &Sizing) -> Result<Mesh, String> {
    match quad::mesh_quads(domain, sizing) {
        Ok(q) => Ok(q.mesh),
        Err(QuadError::Unsupported(u)) => Ok(u.mesh),
        Err(e) => Err(e.to_string()),
    }
}

// --------------------------------------------------------------------------- //
// gate ordering
// --------------------------------------------------------------------------- //

/// A total order on a float for best-state comparison (the gate values are
/// finite; a NaN is sorted as zero rather than being compared with `unwrap`).
#[derive(Clone, Copy, PartialEq)]
struct OrderedF64(f64);

impl Eq for OrderedF64 {}

impl PartialOrd for OrderedF64 {
    /// Total order delegation to [`Ord`]; always `Some` (unlike a raw float,
    /// which is only partially ordered and cannot be used in a key).
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF64 {
    /// A NaN compares Equal to everything, which keeps a gate comparison total
    /// without an `unwrap`; the gate values are finite in practice.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// The choice between two meshes.
///
/// A **passing** mesh is judged by its topology first (v4, then the total
/// valence defect) and only then by its quality margin: the gate is a
/// threshold, and a swap that lowers a 0.63 Jacobian to 0.35 while raising v4
/// is an improvement, not a regression. A **failing** mesh is judged by its
/// distance to the gate first (the worst of `sj/0.2` and `angle/30`), so the
/// untangler still pushes a failing part towards the threshold.
type MeshKey = (
    u8,                            // 1 when the gate fails
    usize,                         // triangles
    usize,                         // inverted
    usize,                         // overlapping
    usize,                         // cracks
    OrderedF64,                    // -gate margin (a failing mesh only)
    std::cmp::Reverse<usize>,      // regular interior nodes
    usize,                         // sum |valence - 4| (interior)
    std::cmp::Reverse<OrderedF64>, // min SJ
    std::cmp::Reverse<OrderedF64>, // min angle
);

/// The lexicographic gate key of a mesh; see [`MeshKey`] for the slot order.
///
/// A **passing** mesh is ordered by triangles, inverted, overlapping and cracks
/// first (all should be 0), then by topology (more regular interior nodes is
/// better, smaller valence defect is better) and only then by its quality
/// margin. A **failing** mesh puts `margin_key = -margin` before topology, so
/// the pass first pushes the worst gate ratio towards its threshold. The
/// `Reverse` wrappers make larger values sort as better.
fn mesh_key(mesh: &Mesh) -> MeshKey {
    let g = verify::gate(mesh);
    let (regular, defect) = valence_quality(mesh);
    let margin = (g.min_scaled_jacobian / SJ_FLOOR).min(g.min_corner_angle_deg / ANGLE_FLOOR);
    let margin = if margin.is_finite() { margin } else { -1.0 };
    let margin_key = if g.passes() { 0.0 } else { -margin };
    (
        u8::from(!g.passes()),
        g.triangles,
        g.inverted,
        g.overlapping,
        g.cracks,
        OrderedF64(margin_key),
        std::cmp::Reverse(regular),
        defect,
        std::cmp::Reverse(OrderedF64(g.min_scaled_jacobian)),
        std::cmp::Reverse(OrderedF64(g.min_corner_angle_deg)),
    )
}

/// Keep `candidate` only when it is strictly better under the gate order.
fn consider(best: &mut Mesh, best_key: &mut MeshKey, candidate: Mesh) {
    let key = mesh_key(&candidate);
    if key < *best_key {
        *best = candidate;
        *best_key = key;
    }
}

/// `(regular interior nodes, sum |valence - 4|)` over interior nodes.
fn valence_quality(mesh: &Mesh) -> (usize, usize) {
    let (quads, tris) = cells(mesh);
    let n = mesh.points.len();
    let boundary = boundary_nodes(&quads, &tris, n);
    let mut incidence = vec![0usize; n];
    for q in &quads {
        for &v in q {
            if v < n {
                incidence[v] += 1;
            }
        }
    }
    let mut regular = 0usize;
    let mut defect = 0usize;
    for v in 0..n {
        if boundary.get(v).copied().unwrap_or(true) {
            continue;
        }
        defect += incidence[v].abs_diff(4);
        if incidence[v] == 4 {
            regular += 1;
        }
    }
    (regular, defect)
}

/// How many swaps and node moves separate two meshes (for the report).
fn count_changes(raw: &Mesh, improved: &Mesh) -> (usize, usize) {
    // Cell keys are sorted node lists, so a quad counts as "new" whenever its
    // node set differs from every raw cell; a swap replaces two quads, hence
    // the factor two against the accepted-swap count.
    let raw_cells: HashSet<Vec<usize>> = raw
        .cells
        .iter()
        .map(|c| {
            let mut k = c.nodes.clone();
            k.sort_unstable();
            k
        })
        .collect();
    let swaps = improved
        .cells
        .iter()
        .filter(|c| {
            let mut k = c.nodes.clone();
            k.sort_unstable();
            !raw_cells.contains(&k)
        })
        .count();
    // `zip` is safe because this pass never adds or removes a node: the two
    // point arrays have the same length by construction.
    let moves = raw
        .points
        .iter()
        .zip(improved.points.iter())
        .filter(|(a, b)| a != b)
        .count();
    (swaps, moves)
}

// --------------------------------------------------------------------------- //
// geometry helpers
// --------------------------------------------------------------------------- //

/// Split a mesh into its quad and triangle index arrays; hexahedra are ignored
/// because this module is planar.
///
/// The arrays preserve the order of `mesh.cells`, so a returned index is stable
/// only while that mesh is unchanged.
fn cells(mesh: &Mesh) -> (Vec<[usize; 4]>, Vec<[usize; 3]>) {
    let mut quads = Vec::new();
    let mut tris = Vec::new();
    for c in &mesh.cells {
        match c.kind {
            Kind::Quad4 => quads.push([c.nodes[0], c.nodes[1], c.nodes[2], c.nodes[3]]),
            Kind::Tri3 => tris.push([c.nodes[0], c.nodes[1], c.nodes[2]]),
            Kind::Hex8 => {}
        }
    }
    (quads, tris)
}

/// Inverse of [`cells`]: build `Quad4` cells first, then `Tri3` cells. The
/// point vector is supplied separately by the caller.
fn cells_of(quads: &[[usize; 4]], tris: &[[usize; 3]]) -> Vec<Cell> {
    let mut out: Vec<Cell> = quads.iter().map(|q| Cell::quad4(*q)).collect();
    out.extend(tris.iter().map(|t| Cell::tri3(*t)));
    // Quads first, triangles second: the order is not part of the contract,
    // but it keeps the rebuilt mesh stable for the gate and the reports.
    out
}

/// Drop the z coordinate of a mesh node (the route is planar, so z is always 0
/// but is kept in the [`Node`] type for compatibility with the 3D crate).
fn xy(p: Node) -> [f64; 2] {
    [p[0], p[1]]
}

/// Vector difference `a - b`, in world length units.
fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

/// Euclidean dot product of two 2D vectors.
fn dot(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

/// Euclidean length of a 2D vector.
fn norm(a: [f64; 2]) -> f64 {
    dot(a, a).sqrt()
}

/// 2D cross product `a x b`; positive when `b` is counter-clockwise from `a`.
fn cross(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

/// The four corner scaled Jacobians of a quad, in node order.
///
/// At corner `i` the two Jacobian columns are the edge vectors to the next and
/// the previous node, so `s_j = sin(angle)` with the orientation sign. This is
/// the same value [`quality::scaled_jacobian`] reports for a quad (tested
/// below), computed without allocating a [`Cell`].
fn quad_corner_sj(points: &[Node], q: &[usize; 4]) -> [f64; 4] {
    let mut out = [0.0f64; 4];
    for i in 0..4 {
        let Some(&a) = q.get(i) else { continue };
        let Some(&b) = q.get((i + 1) % 4) else {
            continue;
        };
        let Some(&c) = q.get((i + 3) % 4) else {
            continue;
        };
        let Some(pa) = points.get(a) else { continue };
        let Some(pb) = points.get(b) else { continue };
        let Some(pc) = points.get(c) else { continue };
        let here = xy(*pa);
        let next = sub(xy(*pb), here);
        let prev = sub(xy(*pc), here);
        let denom = norm(next) * norm(prev);
        out[i] = if denom > 0.0 {
            cross(next, prev) / denom
        } else {
            f64::NAN
        };
    }
    out
}

/// Worst of the four [`quad_corner_sj`] values. A NaN corner (a zero-length
/// edge) is mapped to `-inf`, so a malformed quad always loses; a quad whose
/// corners cannot be read also yields `-inf`.
fn quad_min_sj(points: &[Node], q: &[usize; 4]) -> f64 {
    quad_corner_sj(points, q)
        .iter()
        .fold(f64::INFINITY, |m, &s| {
            m.min(if s.is_nan() { f64::NEG_INFINITY } else { s })
        })
}

/// Smallest corner angle of a quad, in degrees. A zero-length corner edge
/// returns 0 and an unreadable quad returns 0, so the call never yields +inf.
fn quad_min_angle(points: &[Node], q: &[usize; 4]) -> f64 {
    let mut min = f64::INFINITY;
    for i in 0..4 {
        let Some(&a) = q.get(i) else { continue };
        let Some(&b) = q.get((i + 1) % 4) else {
            continue;
        };
        let Some(&c) = q.get((i + 3) % 4) else {
            continue;
        };
        let (Some(pa), Some(pb), Some(pc)) = (points.get(a), points.get(b), points.get(c)) else {
            continue;
        };
        let u = sub(xy(*pb), xy(*pa));
        let v = sub(xy(*pc), xy(*pa));
        let denom = norm(u) * norm(v);
        if denom <= 0.0 {
            return 0.0;
        }
        let angle = (dot(u, v) / denom).clamp(-1.0, 1.0).acos().to_degrees();
        min = min.min(angle);
    }
    if min.is_finite() {
        min
    } else {
        0.0
    }
}

/// Signed **twice** the area of a triangle: positive for a counter-clockwise
/// `(a, b, c)` (the orientation the product pipeline emits), negative for an
/// inverted one, 0 for a degenerate one. The factor 2 is kept because every
/// comparison in this module is relative.
fn tri_area2(points: &[Node], t: &[usize; 3]) -> f64 {
    let (Some(a), Some(b), Some(c)) = (points.get(t[0]), points.get(t[1]), points.get(t[2])) else {
        return 0.0;
    };
    cross(sub(xy(*b), xy(*a)), sub(xy(*c), xy(*a)))
}

/// Smallest corner angle of a triangle, in degrees; 0 for a zero-length edge or
/// an unreadable triangle. This is the quantity the gate reports when a
/// leftover triangle caps the min corner angle.
fn tri_min_angle(points: &[Node], t: &[usize; 3]) -> f64 {
    let mut min = f64::INFINITY;
    for i in 0..3 {
        let Some(&a) = t.get(i) else { continue };
        let Some(&b) = t.get((i + 1) % 3) else {
            continue;
        };
        let Some(&c) = t.get((i + 2) % 3) else {
            continue;
        };
        let (Some(pa), Some(pb), Some(pc)) = (points.get(a), points.get(b), points.get(c)) else {
            continue;
        };
        let u = sub(xy(*pb), xy(*pa));
        let v = sub(xy(*pc), xy(*pa));
        let denom = norm(u) * norm(v);
        if denom <= 0.0 {
            return 0.0;
        }
        let angle = (dot(u, v) / denom).clamp(-1.0, 1.0).acos().to_degrees();
        min = min.min(angle);
    }
    if min.is_finite() {
        min
    } else {
        0.0
    }
}

/// Nodes on the mesh boundary: endpoints of an edge used by exactly one cell.
fn boundary_nodes(quads: &[[usize; 4]], tris: &[[usize; 3]], n: usize) -> Vec<bool> {
    let mut count: HashMap<(usize, usize), usize> = HashMap::new();
    for q in quads {
        for i in 0..4 {
            let (a, b) = (q[i], q[(i + 1) % 4]);
            *count.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    for t in tris {
        for i in 0..3 {
            let (a, b) = (t[i], t[(i + 1) % 3]);
            *count.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    let mut boundary = vec![false; n];
    // A boundary edge is used by exactly one cell; its two endpoints are
    // boundary nodes. Interior edges have two owners, non-manifold more.
    for ((a, b), uses) in count {
        if uses == 1 {
            if let Some(flag) = boundary.get_mut(a) {
                *flag = true;
            }
            if let Some(flag) = boundary.get_mut(b) {
                *flag = true;
            }
        }
    }
    boundary
}

/// Do the closed segments `a..b` and `c..d` intersect?
///
/// A proper crossing is detected by the signs of the four orientations; a
/// collinear/touching case is also a self-intersection for a quad with distinct
/// non-adjacent nodes, so it is tested explicitly with a `1e-12` tolerance to
/// absorb round-off in the cross products.
fn segments_cross(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
    let o1 = cross(sub(b, a), sub(c, a));
    let o2 = cross(sub(b, a), sub(d, a));
    let o3 = cross(sub(d, c), sub(a, c));
    let o4 = cross(sub(d, c), sub(b, c));
    let opposite = |x: f64, y: f64| (x > 0.0 && y < 0.0) || (x < 0.0 && y > 0.0);
    if opposite(o1, o2) && opposite(o3, o4) {
        return true;
    }
    // Collinear/touching cases are self-intersections too when the nodes are
    // distinct and non-adjacent; only the degenerate cross product needs care.
    let on = |p: [f64; 2], q: [f64; 2], r: [f64; 2]| {
        cross(sub(q, p), sub(r, p)).abs() <= 1.0e-12
            && r[0] >= p[0].min(q[0]) - 1.0e-12
            && r[0] <= p[0].max(q[0]) + 1.0e-12
            && r[1] >= p[1].min(q[1]) - 1.0e-12
            && r[1] <= p[1].max(q[1]) + 1.0e-12
    };
    on(a, b, c) || on(a, b, d) || on(c, d, a) || on(c, d, b)
}

/// A simple (non-self-intersecting), counter-clockwise quad.
fn quad_is_simple(points: &[Node], q: &[usize; 4]) -> bool {
    if q[0] == q[1] || q[0] == q[2] || q[0] == q[3] {
        return false;
    }
    if q[1] == q[2] || q[1] == q[3] || q[2] == q[3] {
        return false;
    }
    // A repeated node is not a quad at all; test it before touching `points`.
    let (Some(p0), Some(p1), Some(p2), Some(p3)) = (
        points.get(q[0]),
        points.get(q[1]),
        points.get(q[2]),
        points.get(q[3]),
    ) else {
        return false;
    };
    let p = [xy(*p0), xy(*p1), xy(*p2), xy(*p3)];
    // Positive (twice) area is the CCW test; a clockwise or zero-area quad is
    // not a valid replacement.
    let area2 = (p[0][0] * p[1][1] - p[1][0] * p[0][1])
        + (p[1][0] * p[2][1] - p[2][0] * p[1][1])
        + (p[2][0] * p[3][1] - p[3][0] * p[2][1])
        + (p[3][0] * p[0][1] - p[0][0] * p[3][1]);
    if area2 <= 0.0 {
        return false;
    }
    !segments_cross(p[0], p[1], p[2], p[3]) && !segments_cross(p[1], p[2], p[3], p[0])
}

// --------------------------------------------------------------------------- //
// 1. log-barrier untangling
// --------------------------------------------------------------------------- //

/// The hybrid barrier value of one node position, over the cells incident to
/// that node only.
///
/// Seeing only the incident cells is what makes one evaluation O(1) in the mesh
/// size, and it is also the limit of the metric: two cells that overlap without
/// sharing the node are invisible here (the whole-mesh [`regressed`] guard
/// catches that case). The fields are compared only between candidate positions
/// of the same node; none is a physical quantity.
#[derive(Clone, Copy)]
struct Local {
    /// Sum over incident quads of `ln(s_j)` for every healthy corner, or
    /// `100 * min_sj` for a degenerate/inverted quad (Sastry et al. 2014,
    /// §6.2.3 hybrid metric), plus a `0.1 * ln(sin(min_angle))` term per
    /// incident triangle. Dimensionless; larger is better, NaN is refused.
    objective: f64,

    /// `true` when every incident quad has all corner Jacobians >
    /// [`DEGENERATE_SJ`] and every incident triangle has positive signed area.
    /// A valid state accepts only strict objective ascent; while invalid the
    /// search first makes progress on [`Local::worst`].
    valid: bool,

    /// Worst incident quad corner scaled Jacobian, i.e. the smallest
    /// `sin(corner angle)` with the orientation sign (dimensionless, in
    /// `[-1, 1]`). The sentinel `1.0` means the node has no incident quad.
    worst: f64,

    /// `true` when every incident triangle is positively oriented and above the
    /// area floor. Triangles are checked separately because the gate's minimum
    /// corner angle can regress through them while the quad barrier improves.
    tri_valid: bool,

    /// Smallest corner angle of the incident triangles, in degrees; `+inf` when
    /// the node has no incident triangle.
    tri_worst: f64,
}

/// Evaluate the hybrid log barrier over the cells incident to one node.
///
/// `incident_quads` and `incident_tris` are indices into `quads` and `tris`;
/// out-of-range indices are skipped rather than panicking. The returned
/// [`Local`] is complete: `valid`, `worst`, `tri_valid` and `tri_worst` are
/// derived from the same corner values that build `objective`.
fn eval_local(
    points: &[Node],
    quads: &[[usize; 4]],
    tris: &[[usize; 3]],
    incident_quads: &[usize],
    incident_tris: &[usize],
) -> Local {
    let mut objective = 0.0;
    let mut valid = true;
    let mut worst = f64::INFINITY;
    let mut tri_valid = true;
    let mut tri_worst = f64::INFINITY;
    for &id in incident_quads {
        let Some(q) = quads.get(id) else { continue };
        let sj = quad_corner_sj(points, q);
        let min_sj = sj.iter().fold(f64::INFINITY, |m, &s| {
            m.min(if s.is_nan() { f64::NEG_INFINITY } else { s })
        });
        worst = worst.min(min_sj);
        if min_sj <= DEGENERATE_SJ || min_sj.is_nan() {
            valid = false;
            // The hybrid metric of Sastry et al. §6.2.3: a signed quantity for
            // the inverted/degenerate element, scaled so its gradient competes
            // with the logarithmic terms of the healthy ones.
            objective += 100.0 * min_sj;
        } else {
            for s in sj {
                if s > DEGENERATE_SJ {
                    objective += s.ln();
                }
            }
        }
    }
    for &id in incident_tris {
        let Some(t) = tris.get(id) else { continue };
        let area = tri_area2(points, t);
        if area <= 1.0e-12 || area.is_nan() {
            valid = false;
            tri_valid = false;
            objective += 100.0 * area;
        } else {
            // The gate sees the corner angles of a triangle; a log-area term
            // alone would let a move flatten one while keeping its area.
            let angle = tri_min_angle(points, t);
            tri_worst = tri_worst.min(angle);
            objective += 0.1 * angle.to_radians().sin().max(1.0e-6).ln();
        }
    }
    Local {
        objective,
        valid,
        worst: if worst.is_finite() { worst } else { 1.0 },
        tri_valid,
        tri_worst,
    }
}

/// Is the candidate state at least as good under the hybrid barrier?
///
/// While the patch is tangled the acceptance uses progress of the worst
/// Jacobian (a move that fixes a degeneracy is allowed even if the log terms
/// are not yet meaningful); once valid it is the plain barrier ascent.
fn accept_local(before: Local, after: Local) -> bool {
    if !after.objective.is_finite() {
        return false;
    }
    // A move may never make an incident triangle thinner. This keeps the gate's
    // min-corner-angle column from regressing through the triangles, which the
    // quad barrier cannot see.
    let tri_ok = if before.tri_valid {
        after.tri_valid && after.tri_worst + 1.0e-9 >= before.tri_worst
    } else {
        after.tri_valid || after.tri_worst > before.tri_worst
    };
    if !tri_ok {
        return false;
    }
    if after.valid {
        if before.valid {
            after.objective > before.objective + 1.0e-12
        } else {
            true
        }
    } else if before.valid {
        false
    } else {
        after.worst > before.worst + 1.0e-10 && after.objective > before.objective - 1.0e-9
    }
}

/// Better of two valid objective values (validity dominates the barrier).
fn candidate_key(local: Local) -> (u8, OrderedF64) {
    (u8::from(local.valid), OrderedF64(local.objective))
}

/// One log-barrier untangling pass over the active interior nodes.
///
/// Active nodes are the nodes of quads whose worst corner Jacobian is below
/// [`ACTIVE_SJ`] or whose smallest corner angle is below [`ACTIVE_ANGLE`]; a
/// mesh with no active interior node is returned unchanged (tested). Boundary
/// nodes are never in the list. Each round visits the nodes in index order and
/// each node keeps the best position of a fixed candidate stencil (four radii
/// times eight directions, plus long steps and the opposite-corner centroid
/// when an incident quad is degenerate). The pass is deterministic for a given
/// mesh and [`Options`].
///
/// Safety: a candidate is accepted only by [`accept_local`], which refuses to
/// make an incident triangle thinner; the whole pass is reverted when
/// [`regressed`] fires or when the boundary offset is above `1e-9` (a numerical
/// surprise, because boundary nodes never move). Returns the input mesh in
/// those cases.
pub(crate) fn untangle(domain: &Domain, mesh: &Mesh, size: f64, options: &Options) -> Mesh {
    let (quads, tris) = cells(mesh);
    let n = mesh.points.len();
    if quads.is_empty() || n == 0 {
        return mesh.clone();
    }
    let boundary = boundary_nodes(&quads, &tris, n);
    let mut incident_quads: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (id, q) in quads.iter().enumerate() {
        for &v in q {
            if let Some(list) = incident_quads.get_mut(v) {
                list.push(id);
            }
        }
    }
    let mut incident_tris: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (id, t) in tris.iter().enumerate() {
        for &v in t {
            if let Some(list) = incident_tris.get_mut(v) {
                list.push(id);
            }
        }
    }
    // The inner loop only needs the cells around one node; building the
    // incidence lists once turns each evaluation from a mesh scan into O(1).

    // Active nodes: the nodes of quads that are below the gate margin or
    // badly shaped. A good regular mesh has no active node and is returned
    // unchanged (tested).
    let mut active = vec![false; n];
    for q in &quads {
        if quad_min_sj(mesh.points.as_slice(), q) < ACTIVE_SJ
            || quad_min_angle(mesh.points.as_slice(), q) < ACTIVE_ANGLE
        {
            for &v in q {
                if let Some(flag) = active.get_mut(v) {
                    *flag = true;
                }
            }
        }
    }
    // Boundary nodes are held fixed. Sliding a boundary node by projecting it
    // onto the nearest segment lets a domain corner wander along an adjacent
    // edge (measured on a 3x3 patch: the corner (0,0) ended at (0,1.08), which
    // still has boundary offset 0 but cuts the domain corner off). Fixed
    // boundary nodes are the paper's sharp-feature setting and keep conformity
    // exact; the common post-pass may slide them afterwards.
    let nodes: Vec<usize> = (0..n)
        .filter(|&v| active[v] && !boundary.get(v).copied().unwrap_or(true))
        .collect();
    if nodes.is_empty() {
        return mesh.clone();
    }

    let mut points = mesh.points.clone();
    let mut evaluations = 0usize;
    let mut exhausted = false;
    for _round in 0..options.untangle_rounds {
        let mut moved = false;
        for &v in &nodes {
            let Some(list) = incident_quads.get(v) else {
                continue;
            };
            let before = eval_local(
                &points,
                &quads,
                &tris,
                list,
                incident_tris.get(v).map(Vec::as_slice).unwrap_or(&[]),
            );
            let Some(here_node) = points.get(v).copied() else {
                continue;
            };
            let here = xy(here_node);
            // A short coordinate-search neighbourhood, plus two untangling
            // fallbacks when the node belongs to a non-positive quad: a long
            // step (a tangled corner may need to cross a neighbour before any
            // Jacobian changes sign) and the centroid of the other corners of
            // that quad (which makes the quad convex again).
            let mut candidates: Vec<[f64; 2]> = Vec::with_capacity(64);
            for &fraction in &RADII {
                let radius = fraction * size;
                for k in 0..DIRECTIONS {
                    let angle = std::f64::consts::TAU * k as f64 / DIRECTIONS as f64;
                    candidates.push([
                        here[0] + radius * angle.cos(),
                        here[1] + radius * angle.sin(),
                    ]);
                }
            }
            if !before.valid {
                for &fraction in &[0.5, 1.0] {
                    let radius = fraction * size;
                    for k in 0..DIRECTIONS {
                        let angle = std::f64::consts::TAU * k as f64 / DIRECTIONS as f64;
                        candidates.push([
                            here[0] + radius * angle.cos(),
                            here[1] + radius * angle.sin(),
                        ]);
                    }
                }
                for &id in list {
                    let Some(q) = quads.get(id) else { continue };
                    if quad_min_sj(&points, q) > DEGENERATE_SJ {
                        continue;
                    }
                    let mut sum = [0.0f64; 2];
                    let mut count = 0usize;
                    for &other in q {
                        if other == v {
                            continue;
                        }
                        if let Some(p) = points.get(other) {
                            sum[0] += p[0];
                            sum[1] += p[1];
                            count += 1;
                        }
                    }
                    if count > 0 {
                        candidates.push([sum[0] / count as f64, sum[1] / count as f64]);
                    }
                }
            }
            // Score the whole stencil against the same `before`; a candidate
            // is applied only after it wins, so the enumeration order inside
            // the stencil cannot bias the result (ties go to the first one).
            let mut best: Option<((u8, OrderedF64), [f64; 2])> = None;
            for candidate in candidates {
                if evaluations >= options.eval_budget {
                    exhausted = true;
                    break;
                }
                evaluations += 1;
                // Evaluate on a scratch position and restore it afterwards:
                // `eval_local` reads the shared `points` array, so the node
                // must be put back before the next candidate.
                points[v] = [candidate[0], candidate[1], here_node[2]];
                let after = eval_local(
                    &points,
                    &quads,
                    &tris,
                    list,
                    incident_tris.get(v).map(Vec::as_slice).unwrap_or(&[]),
                );
                points[v] = here_node;
                if !accept_local(before, after) {
                    continue;
                }
                let key = candidate_key(after);
                if best.as_ref().is_none_or(|(bk, _)| key > *bk) {
                    best = Some((key, candidate));
                }
            }
            // Apply the winning position once per node. Re-evaluating the
            // whole neighbourhood after each accepted move would make the
            // result order-dependent and much more expensive.
            if let Some((_, p)) = best {
                points[v] = [p[0], p[1], here_node[2]];
                moved = true;
            }
            if exhausted {
                // The budget is global to this pass: no later node can be
                // scored, so stop at the node that exhausted it.
                break;
            }
        }
        // Stop at the first round with no accepted move, or as soon as the
        // deterministic budget is spent (the remaining nodes cannot be scored).
        if !moved || exhausted {
            break;
        }
    }

    let out = Mesh::new(points, cells_of(&quads, &tris));
    // Boundary nodes did not move, so this only guards against a numerical
    // surprise; a larger offset means the pass must be reverted.
    let offset = verify::boundary_offset(domain, &out);
    // A move is local by construction; this is the safety net against a rare
    // overlap the local Jacobian cannot see.
    if offset > 1.0e-9 || regressed(mesh, &out) {
        mesh.clone()
    } else {
        out
    }
}

/// True when `after` is worse than `before` under the gate order. Used to
/// reject a whole pass rather than leave the caller with a regression.
///
/// A hard guard also rejects a result that halves the min corner angle: on the
/// ring the untangler can remove the single inverted quad while collapsing a
/// different corner to ~3 deg, which the gate key alone would accept because
/// the inverted count improves. A near-degenerate corner is not a trade worth
/// making on a part that the leftover triangles already fail.
fn regressed(before: &Mesh, after: &Mesh) -> bool {
    let gb = verify::gate(before);
    let ga = verify::gate(after);
    let angle_collapsed =
        gb.min_corner_angle_deg > 5.0 && ga.min_corner_angle_deg < 0.5 * gb.min_corner_angle_deg;
    angle_collapsed || mesh_key(after) > mesh_key(before)
}

// --------------------------------------------------------------------------- //
// 2. singularity reduction (interior quad-edge swaps)
// --------------------------------------------------------------------------- //

/// One candidate interior edge swap: replace the two quads that share an edge by
/// the two quads of the other diagonal of their six-node patch.
#[derive(Clone)]
struct Swap {
    /// Index into the current `quads` vector of the first quad of the pair.
    q1: usize,

    /// Index of the second quad of the pair; `q2 != q1`, and the two quads share
    /// exactly one edge (the edge is interior, used by exactly two cells).
    q2: usize,

    /// Replacement for `q1`, counter-clockwise, using the new diagonal; it must
    /// be simple and pass `quad_gate_ok`.
    new1: [usize; 4],

    /// Replacement for `q2`, counter-clockwise; it shares the new diagonal with
    /// `new1` in the opposite direction, so the pair stays conforming.
    new2: [usize; 4],

    /// Ordering key, lower is better: `(0, delta, Reverse(new_min))` where the
    /// first slot is 0 for a degenerate-corner repair and 1 otherwise, `delta`
    /// is the change in `sum |valence - 4|` (negative is an improvement) and
    /// `new_min` is the worst corner Jacobian of the two new quads. The edge key
    /// in `best_swap` breaks ties deterministically.
    rank: (u8, i64, std::cmp::Reverse<OrderedF64>),
}

/// Reduce `sum |valence - 4|` with safe interior edge swaps, and repair
/// zero-Jacobian boundary corners.
///
/// Only an edge shared by exactly two quads is considered; its two endpoints
/// are replaced by the other diagonal of the six-node patch. Boundary edges
/// (used once) are never touched, so the boundary is unchanged. Every swap is
/// checked against the gate before it is kept: new quads must be simple,
/// positively oriented, at least 0.2 in corner Jacobian and 30 deg in corner
/// angle, and must not recreate an existing edge. The gate is recomputed after
/// each swap and a regression is rolled back.
fn reduce_singularities(mesh: &Mesh, options: &Options) -> Mesh {
    let (mut quads, tris) = cells(mesh);
    let n = mesh.points.len();
    if quads.is_empty() || n == 0 {
        return mesh.clone();
    }
    let points = &mesh.points;
    let mut swaps = 0usize;
    let mut current = Mesh::new(points.clone(), cells_of(&quads, &tris));
    let mut current_gate = verify::gate(&current);
    let baseline_gate = current_gate;
    // The repair flag is frozen on the baseline gate: a mesh that started
    // failing keeps the degenerate-corner path enabled even if an early swap
    // happens to lift the global min SJ above the floor.
    let mut defect = valence_defect(mesh);
    let gate_floor = SJ_FLOOR - 1.0e-9;

    for _ in 0..options.max_swaps {
        let swap = match best_swap(points, &quads, &tris, &baseline_gate) {
            Some(s) => s,
            None => break,
        };
        let old1 = quads.get(swap.q1).copied();
        let old2 = quads.get(swap.q2).copied();
        if old1.is_none() || old2.is_none() {
            break;
        }
        if let Some(slot) = quads.get_mut(swap.q1) {
            *slot = swap.new1;
        }
        if let Some(slot) = quads.get_mut(swap.q2) {
            *slot = swap.new2;
        }
        let candidate = Mesh::new(points.clone(), cells_of(&quads, &tris));
        let g = verify::gate(&candidate);
        let new_defect = valence_defect(&candidate);
        // A swap may lower a *high* Jacobian or angle as long as the gate
        // thresholds stay met; it may not lower a failing value, break the
        // boundary/conformity, or increase the valence defect.
        let sj_ok = if current_gate.min_scaled_jacobian >= gate_floor {
            g.min_scaled_jacobian >= gate_floor
        } else {
            g.min_scaled_jacobian + 1.0e-9 >= current_gate.min_scaled_jacobian
        };
        let angle_ok = if current_gate.min_corner_angle_deg >= ANGLE_FLOOR - 1.0e-9 {
            g.min_corner_angle_deg >= ANGLE_FLOOR - 1.0e-9
        } else {
            g.min_corner_angle_deg + 1.0e-9 >= current_gate.min_corner_angle_deg
        };
        let worse = g.inverted > current_gate.inverted
            || g.overlapping > current_gate.overlapping
            || g.cracks > current_gate.cracks
            || g.triangles > current_gate.triangles
            || !sj_ok
            || !angle_ok
            || new_defect >= defect;
        if worse {
            // Roll the swap back.
            if let (Some(slot), Some(old)) = (quads.get_mut(swap.q1), old1) {
                *slot = old;
            }
            if let (Some(slot), Some(old)) = (quads.get_mut(swap.q2), old2) {
                *slot = old;
            }
            // Do not retry: `best_swap` ranks the same local candidate again,
            // so the loop would spin. The pass ends with the last kept state.
            break;
        }
        current = candidate;
        current_gate = g;
        defect = new_defect;
        swaps += 1;
    }
    // `swaps` is kept only for the local count; `Outcome::swaps` is recomputed
    // from the cell difference in `count_changes`, not from here.
    let _ = swaps;
    current
}

/// `sum |valence - 4|` over interior nodes of a whole mesh.
fn valence_defect(mesh: &Mesh) -> usize {
    valence_quality(mesh).1
}

/// Find the best legal interior edge swap on the current mesh, or `None`.
///
/// `baseline` is the gate of the mesh **before** the first swap of the pass; it
/// decides whether the degenerate-corner repair path is enabled, so the flag
/// does not flicker as individual swaps change the current gate. Every edge
/// shared by exactly two quads is considered, both diagonals of the six-node
/// patch are tried, and a candidate must be simple, pass `quad_gate_ok` on
/// both new quads, not recreate an existing edge and preserve the pair's total
/// (twice) area to `1e-6` relative. It must either repair a corner below
/// [`SJ_FLOOR`] or strictly lower the interior valence defect. The ranking is
/// `rank` in [`Swap`], with the sorted edge list making ties deterministic.
fn best_swap(
    points: &[Node],
    quads: &[[usize; 4]],
    tris: &[[usize; 3]],
    baseline: &verify::Gate,
) -> Option<Swap> {
    let n = points.len();
    // Boundary flags are needed only by `valence_delta`: a boundary node's
    // valence is fixed by the domain edge, so its defect is not counted.
    let boundary = boundary_nodes(quads, tris, n);
    // `owners` answers "which quads use this edge"; an edge with exactly two
    // owners is interior and can be re-diagonalised. `all_edges` (quads and
    // triangles) is the set a new diagonal must not recreate.
    let mut owners: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    let mut all_edges: HashSet<(usize, usize)> = HashSet::new();
    for (id, q) in quads.iter().enumerate() {
        for i in 0..4 {
            let (a, b) = (q[i], q[(i + 1) % 4]);
            let key = (a.min(b), a.max(b));
            owners.entry(key).or_default().push(id);
            all_edges.insert(key);
        }
    }
    for t in tris {
        for i in 0..3 {
            let (a, b) = (t[i], t[(i + 1) % 3]);
            all_edges.insert((a.min(b), a.max(b)));
        }
    }
    // Current quad valence per node (the defect is `|valence - 4|` on interior
    // nodes; triangles do not count).
    let mut valence = vec![0usize; n];
    for q in quads {
        for &v in q {
            if v < n {
                valence[v] += 1;
            }
        }
    }
    // The repair path is enabled for the whole pass when either the baseline
    // gate or the current pair-level scan still has a corner below the floor.
    let global_min = quad_min_sj_global(points, quads);
    let repairing =
        baseline.min_scaled_jacobian < SJ_FLOOR - 1.0e-9 || global_min < SJ_FLOOR - 1.0e-9;

    // Fixing the edge order makes the "best" choice deterministic when several
    // swaps have the same rank; it also gives the caller a reproducible mesh.
    let mut edges: Vec<(usize, usize)> = owners.keys().copied().collect();
    edges.sort_unstable();
    let mut best: Option<Swap> = None;
    for edge in edges {
        let Some(list) = owners.get(&edge) else {
            continue;
        };
        if list.len() != 2 {
            // A boundary edge (one owner) or a non-manifold edge (three or
            // more) cannot be swapped without breaking conformity.
            continue;
        }
        let (q1_id, q2_id) = (list[0].min(list[1]), list[0].max(list[1]));
        let (Some(q1), Some(q2)) = (quads.get(q1_id), quads.get(q2_id)) else {
            continue;
        };
        for i in 0..4 {
            let x = q1[i];
            let y = q1[(i + 1) % 4];
            let p = q1[(i + 2) % 4];
            let q = q1[(i + 3) % 4];
            // The other quad must traverse the shared edge in reverse.
            let Some(j) = (0..4).find(|&j| q2[j] == y && q2[(j + 1) % 4] == x) else {
                continue;
            };
            let r = q2[(j + 2) % 4];
            let s = q2[(j + 3) % 4];
            // Candidate 1: new shared edge (p, r). Candidate 2: (q, s).
            let candidates = [
                ([p, q, x, r], [p, r, s, y], (p, r)),
                ([q, x, r, s], [q, s, y, p], (q, s)),
            ];
            for (new1, new2, diagonal) in candidates {
                // Recreating an existing edge would produce a duplicate or a
                // non-manifold edge, so it is refused before any geometry is
                // evaluated.
                if diagonal.0 == diagonal.1
                    || all_edges.contains(&(diagonal.0.min(diagonal.1), diagonal.0.max(diagonal.1)))
                {
                    continue;
                }
                // Both new quads must be simple, CCW and above the gate floors
                // on their own, so a swap can never create the next failure.
                if !quad_is_simple(points, &new1)
                    || !quad_is_simple(points, &new2)
                    || !quad_gate_ok(points, &new1)
                    || !quad_gate_ok(points, &new2)
                {
                    continue;
                }
                // Area preservation rejects a swap that would overlap one of
                // the two quads with its neighbours.
                let old_area = quad_area2(points, q1) + quad_area2(points, q2);
                let new_area = quad_area2(points, &new1) + quad_area2(points, &new2);
                if (new_area - old_area).abs() > 1.0e-6 * old_area.abs().max(1.0) {
                    continue;
                }
                let new_min = quad_min_sj(points, &new1).min(quad_min_sj(points, &new2));
                let old_pair_min = quad_min_sj(points, q1).min(quad_min_sj(points, q2));
                // Only the two old diagonal endpoints lose a quad and the two
                // new diagonal endpoints gain one; all other valences are
                // unchanged.
                let delta = valence_delta(&valence, &boundary, x, y, diagonal.0, diagonal.1);
                let fixes = repairing
                    && old_pair_min < SJ_FLOOR - 1.0e-9
                    && new_min > old_pair_min + 1.0e-6
                    && new_min >= SJ_FLOOR;
                let improves_valence = delta < 0 && new_min >= SJ_FLOOR;
                if !fixes && !improves_valence {
                    continue;
                }
                // Repair swaps first; then the largest valence drop; then the
                // better new Jacobian. The edge key makes ties deterministic.
                let rank = (
                    u8::from(!fixes),
                    delta,
                    std::cmp::Reverse(OrderedF64(new_min)),
                );
                let better = match &best {
                    None => true,
                    Some(b) => rank < b.rank,
                };
                if better {
                    best = Some(Swap {
                        q1: q1_id,
                        q2: q2_id,
                        new1,
                        new2,
                        rank,
                    });
                }
            }
        }
    }
    best
}

/// The change in `sum |valence - 4|` over interior nodes for a swap whose
/// shared-edge endpoints are `x, y` and whose new diagonal is `a, b`.
fn valence_delta(
    valence: &[usize],
    boundary: &[bool],
    x: usize,
    y: usize,
    a: usize,
    b: usize,
) -> i64 {
    let mut delta = 0i64;
    for (v, delta_v) in [(x, -1i64), (y, -1), (a, 1), (b, 1)] {
        if boundary.get(v).copied().unwrap_or(true) {
            continue;
        }
        let Some(&count) = valence.get(v) else {
            continue;
        };
        let old = (count as i64 - 4).abs();
        let new = ((count as i64 + delta_v) - 4).abs();
        delta += new - old;
    }
    delta
}

/// Both new quads must pass the gate's own thresholds on their own, so a swap
/// can never create the next failure.
fn quad_gate_ok(points: &[Node], q: &[usize; 4]) -> bool {
    quad_min_sj(points, q) >= SJ_FLOOR && quad_min_angle(points, q) >= ANGLE_FLOOR - 1.0e-9
}

/// Signed **twice** the area of a quad, by the shoelace sum; positive for the
/// counter-clockwise node order the route uses.
fn quad_area2(points: &[Node], q: &[usize; 4]) -> f64 {
    let mut sum = 0.0;
    for i in 0..4 {
        let (Some(a), Some(b)) = (points.get(q[i]), points.get(q[(i + 1) % 4])) else {
            continue;
        };
        sum += a[0] * b[1] - b[0] * a[1];
    }
    sum
}

/// Worst corner Jacobian over every quad of the mesh; the sentinel `1.0` means
/// the mesh has no quad at all.
fn quad_min_sj_global(points: &[Node], quads: &[[usize; 4]]) -> f64 {
    let mut min = f64::INFINITY;
    for q in quads {
        min = min.min(quad_min_sj(points, q));
    }
    if min.is_finite() {
        min
    } else {
        1.0
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::quality;
    use crate::two_d::Domain;

    /// A rectangle `width` x `height` (test helper).
    fn rectangle(width: f64, height: f64) -> Domain {
        Domain::rectangle(width, height).unwrap()
    }

    /// A `(nx + 1) x (ny + 1)` lattice of nodes with spacing `h`; node
    /// `(i, j)` is `j * (nx + 1) + i` in row-major order.
    fn points_grid(nx: usize, ny: usize, h: f64) -> Vec<Node> {
        let mut points = Vec::with_capacity((nx + 1) * (ny + 1));
        for j in 0..=ny {
            for i in 0..=nx {
                points.push([i as f64 * h, j as f64 * h, 0.0]);
            }
        }
        points
    }

    /// The `nx * ny` quads of [`points_grid`], counter-clockwise, one per
    /// lattice cell.
    fn grid_quads(nx: usize, ny: usize) -> Vec<[usize; 4]> {
        let mut quads = Vec::new();
        let stride = nx + 1;
        for j in 0..ny {
            for i in 0..nx {
                let a = j * stride + i;
                quads.push([a, a + 1, a + stride + 1, a + stride]);
            }
        }
        quads
    }

    /// The corner formula must agree with the crate's quality metric.
    #[test]
    fn corner_jacobians_match_the_quality_metric() {
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.3, 0.0, 0.0],
            [1.1, 0.8, 0.0],
            [-0.2, 1.1, 0.0],
        ];
        let q = [0, 1, 2, 3];
        let ours = quad_corner_sj(&points, &q);
        let theirs = quality::scaled_jacobian(&Cell::quad4(q), &points);
        for (a, b) in ours.iter().zip(theirs.iter()) {
            assert!((a - b).abs() < 1.0e-12, "{ours:?} vs {theirs:?}");
        }
    }

    /// An inverted patch must be pulled back to positive orientation by node
    /// motion alone (the test moves only the interior centre).
    #[test]
    fn an_inverted_quad_is_untangled() {
        // A 3x3 quad grid with the central node pulled far to the right, which
        // inverts its four quads. The untangler may move only interior nodes.
        let mut points = points_grid(3, 3, 1.0);
        let quads = grid_quads(3, 3);
        let centre = 5; // node (1, 1) of a 4x4 node grid
        points[centre] = [2.8, 1.0, 0.0];
        let mesh = Mesh::new(points.clone(), cells_of(&quads, &[]));
        assert!(
            verify::gate(&mesh).inverted > 0,
            "the test patch must start inverted"
        );
        let out = untangle(&rectangle(3.0, 3.0), &mesh, 1.0, &Options::default());
        let g = verify::gate(&out);
        assert_eq!(g.inverted, 0, "{:?}", g.reason());
        assert_eq!(g.cracks, 0);
        assert!(g.min_scaled_jacobian > 0.0);
    }

    /// A regular mesh has no active node, so both passes must return it
    /// bit-for-bit unchanged.
    #[test]
    fn a_good_regular_mesh_is_unchanged() {
        let points = points_grid(3, 3, 1.0);
        let quads = grid_quads(3, 3);
        let mesh = Mesh::new(points.clone(), cells_of(&quads, &[]));
        let domain = rectangle(3.0, 3.0);
        let after = untangle(&domain, &mesh, 1.0, &Options::default());
        assert_eq!(after.points, points, "a regular mesh has no active node");
        let after_swaps = reduce_singularities(&mesh, &Options::default());
        assert_eq!(after_swaps.points, points);
        assert_eq!(after_swaps.cells.len(), mesh.cells.len());
        assert!(verify::gate(&after_swaps).passes());
    }

    /// Boundary nodes never move, so the boundary offset must stay below the
    /// numerical tolerance even when the interior is perturbed.
    #[test]
    fn boundary_nodes_stay_on_the_boundary() {
        // Perturb interior nodes of a coarse plate; the boundary may slide but
        // must remain on the hole/outer polyline.
        let domain = Domain::plate_with_hole(20.0, 12.0, 2.0, 16).unwrap();
        let (mut points, tris) =
            crate::two_d::triangulate(&domain, &Sizing::uniform(2.0).unwrap()).unwrap();
        // Move every interior node a little so the untangler has work to do.
        for p in points.iter_mut() {
            if domain.distance_to_boundary([p[0], p[1]]) > 0.3 {
                p[0] += 0.17;
                p[1] -= 0.11;
            }
        }
        let cells: Vec<Cell> = tris.iter().map(|t| Cell::tri3(*t)).collect();
        let mesh = Mesh::new(points, cells);
        let out = untangle(&domain, &mesh, 2.0, &Options::default());
        let offset = verify::boundary_offset(&domain, &out);
        assert!(
            offset < 1.0e-9,
            "boundary node moved off the domain: {offset:.3e}"
        );
    }

    /// The atomic Verma operation: one swap lowers the centre's valence defect
    /// from 1 to 0 without breaking the gate.
    #[test]
    fn an_interior_edge_swap_reduces_the_valence_defect() {
        // A regular decagon with five fan quads around the centre: the centre
        // has valence 5 (defect 1). One interior edge swap between two adjacent
        // fan quads lowers it to 4, and both new quads stay above the gate.
        let mut points = vec![[0.0, 0.0, 0.0]];
        let mut boundary = Vec::new();
        for k in 0..10 {
            let a = std::f64::consts::TAU * k as f64 / 10.0;
            points.push([a.cos(), a.sin(), 0.0]);
            boundary.push(k + 1);
        }
        let mut quads = Vec::new();
        for i in 0..5 {
            quads.push([
                0,
                boundary[2 * i],
                boundary[(2 * i + 1) % 10],
                boundary[(2 * i + 2) % 10],
            ]);
        }
        let mesh = Mesh::new(points, cells_of(&quads, &[]));
        assert!(
            verify::gate(&mesh).passes(),
            "{:?}",
            verify::gate(&mesh).reason()
        );
        let (regular_before, defect_before) = valence_quality(&mesh);
        assert_eq!(regular_before, 0);
        assert_eq!(defect_before, 1);
        let out = reduce_singularities(&mesh, &Options::default());
        let (regular_after, defect_after) = valence_quality(&out);
        assert_eq!(defect_after, 0);
        assert_eq!(regular_after, 1);
        let g = verify::gate(&out);
        assert!(g.passes(), "{:?}", g.reason());
        assert_eq!(g.cracks, 0);
        assert_eq!(g.inverted, 0);
    }

    /// End-to-end: a simple rectangle must come out all-quad, passing, and on
    /// the boundary after the product pipeline plus this pass.
    #[test]
    fn a_simple_domain_is_all_quad_and_passes_the_gate() {
        let domain = rectangle(8.0, 8.0);
        let sizing = Sizing::uniform(1.0).unwrap();
        let mesh = mesh(&domain, &sizing).unwrap();
        assert!(mesh.cells.iter().all(|c| c.kind == Kind::Quad4));
        let g = verify::gate(&mesh);
        assert!(g.passes(), "{:?}", g.reason());
        assert!(verify::boundary_offset(&domain, &mesh) < 1.0e-9);
    }

    /// Non-default [`Options`] values only change loop bounds; the result must
    /// still be a valid, crack-free mesh on the boundary. The test also checks
    /// that the explicit values differ from the defaults.
    #[test]
    fn non_default_options_still_produce_a_valid_mesh() {
        let domain = rectangle(8.0, 8.0);
        let sizing = Sizing::uniform(1.0).unwrap();
        let options = Options {
            untangle_rounds: 4,
            max_swaps: 50,
            eval_budget: 250_000,
        };
        let mesh = mesh_with_options(&domain, &sizing, &options).unwrap();
        assert!(mesh
            .cells
            .iter()
            .all(|c| matches!(c.kind, Kind::Quad4 | Kind::Tri3)));
        let g = verify::gate(&mesh);
        assert_eq!(g.cracks, 0, "{:?}", g.reason());
        assert!(verify::boundary_offset(&domain, &mesh) < 1.0e-9);
    }
}
