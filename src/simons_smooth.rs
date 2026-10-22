//! Angle and angle+stretch smoothing of an all-quad mesh (Simons 2024, §2.2.10).
//!
//! Simons' thesis smooths a quad mesh as a post-processing step. Instead of
//! Laplacian smoothing it repositions each *Steiner* (interior) vertex so that
//! the angles of every incident quad move towards 90°. The thesis uses the
//! Nelder–Mead simplex algorithm (Nelder & Mead 1965; Brown 2001) because the
//! objective — a deviation of angles, or the maximum of several metrics — is
//! not differentiable; it names Mesquite and SciPy's implementation. The
//! angle-only objective is the same deviation the patch ranking uses, and the
//! thesis reports that it "produces arbitrarily skinny rectangular elements,
//! creating visible pinches": a rectangle has four right angles no matter how
//! thin it is, so nothing in the angle objective forbids a long thin cell.
//! The thesis therefore also uses a combined objective of **angle deviation and
//! stretch**, where stretch is "the scaled ratio between the longest diagonal
//! of the quad and the shortest edge" (Fig. 2.9). That is the objective pair
//! implemented here.
//!
//! # Faithful to the thesis
//!
//! * §2.2.10: the angle objective is the sum, over the quads incident to a
//!   node, of the squared deviation of every corner angle from 90°. The thesis
//!   ranks patches by the *maximum* deviation; the smoother minimises a sum
//!   (the thesis only says "minimizes the deviation of all dependent angles"),
//!   which is the standard smooth objective and the one a simplex handles
//!   well. Both use the same per-angle quantity, so a ranking function can be
//!   swapped in as the thesis describes.
//! * §2.2.10: positions are found with an in-file 2D Nelder–Mead simplex
//!   (`simplex = true`) with reflection, expansion, contraction and shrink, or
//!   with a coordinate/pattern search (`simplex = false`). No external solver.
//! * §2.2.10 and Fig. 2.9: the second objective adds
//!   `stretch_weight * sum((stretch - 1)^2)` with
//!   `stretch = max(diagonal lengths) / (min(edge lengths) * sqrt(2))`, i.e.
//!   the longest-diagonal-over-shortest-edge ratio scaled so that a square has
//!   `stretch = 1` (its unscaled ratio is `sqrt(2)`). The weight is the user's
//!   `Options::stretch_weight`; `0` recovers angle-only smoothing exactly.
//! * §2.2.10: only interior nodes move by default; boundary nodes are held
//!   exactly in place. `Options::allow_boundary` opts into moving them.
//!
//! # Guards (contract)
//!
//! * A candidate position is accepted only if every incident quad stays
//!   strictly convex and positively oriented (all four corner Jacobians
//!   positive). Because a node move touches only its incident quads, this is
//!   enough to make the whole mesh inversion-free; the output is checked
//!   globally as well.
//! * A candidate must lower the local objective by a relative tolerance, so a
//!   perfect (structured) grid is returned bit-for-bit unchanged.
//! * Topology never changes: cells and node indices are untouched, only `x, y`
//!   of free nodes move (z stays as it was). Nodes of non-manifold or boundary
//!   edges are treated as boundary and fixed unless `allow_boundary` is set.
//! * `Result<_, String>` everywhere; bad input (non-quad cell, inverted or
//!   non-convex cell, non-finite geometry, `size <= 0`, negative stretch
//!   weight) is an error, never a panic.
//!
//! # Simplified / not implemented
//!
//! * The thesis uses SciPy's implementation and reports 30 smoothing passes;
//!   this is an independent 2D implementation with its own convergence
//!   tolerances. It is a local, Gauss–Seidel sweep (node 0, 1, …), not a
//!   global solve.
//! * During smoothing, boundary nodes are never slid along the boundary
//!   polyline. With `allow_boundary = true` a boundary node may move, but only
//!   inside (or on) the domain and by a limited step; the domain geometry is
//!   then no longer preserved exactly. Sharp corners are not specially
//!   detected beyond the fact that they are boundary nodes.
//! * No topological improvement (edge swaps, local remeshing) is attempted;
//!   §2.2.10 notes that the templates may no longer be ideal after smoothing
//!   and leaves that as future work.
//! * The objective is evaluated on the `xy` projection. A planar mesh is the
//!   only supported input.
//!
//! # Public API
//!
//! ```text
//! simons_smooth::smooth(&domain, &mesh, size, &options) -> Result<Mesh, String>
//! simons_smooth::smooth_with_stats(...)                -> Result<Outcome, String>
//! simons_smooth::measure(&mesh)                        -> Result<Stats, String>
//! simons_smooth::demo()                                -> Result<DemoNumbers, String>
//! ```
//!
//! [`Options`] is `Debug + Clone + Copy + Default + PartialEq` and
//! serde-serialisable. [`Stats`] carries the reported metrics.

use std::collections::HashMap;

use crate::mesh::{Cell, Kind, Mesh};
use crate::quality;
use crate::two_d::Domain;

/// Ideal corner angle in radians.
const RIGHT_ANGLE: f64 = std::f64::consts::FRAC_PI_2;
/// Objective value used for an invalid (inverted or out-of-domain) candidate.
const INVALID: f64 = 1.0e12;
/// Smallest corner Jacobian a candidate quad may have.
const MIN_CORNER_JACOBIAN: f64 = 1.0e-8;
/// Relative objective decrease needed to accept a node move.
const ACCEPT_TOLERANCE: f64 = 1.0e-12;

/// Options for [`smooth`].
///
/// `Options::default()` is 30 passes of the simplex with a stretch weight of
/// `0.5`, interior nodes only. Set `stretch_weight = 0.0` for the angle-only
/// objective of the thesis (the one that produces skinny rectangular
/// elements), or `simplex = false` for the coordinate search.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Options {
    /// Gauss–Seidel sweeps over the free nodes. Each sweep visits every free
    /// node once; the run stops early when a sweep moves nothing. `0` returns
    /// the input unchanged.
    pub iterations: usize,

    /// Weight of the stretch term. `0` is angle-only smoothing (the thesis'
    /// Fig. 2.9b, which pinches); larger values penalise a long diagonal more
    /// strongly. Must be finite and non-negative.
    pub stretch_weight: f64,

    /// Let boundary nodes move too. Off by default: by default every boundary
    /// node is held exactly where it is, so the domain geometry is untouched.
    pub allow_boundary: bool,

    /// Use the 2D Nelder–Mead simplex (`true`, the thesis' choice) or a
    /// coordinate/pattern search (`false`).
    pub simplex: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            iterations: 30,
            stretch_weight: 0.5,
            allow_boundary: false,
            simplex: true,
        }
    }
}

/// The measured quality of an all-quad mesh.
///
/// Angles are in degrees; `aspect_p95`/`aspect_max` are longest-edge over
/// shortest-edge; `max_stretch` is the longest-diagonal-over-shortest-edge
/// ratio scaled so that a square is `1.0` (the quantity in the objective).
/// `min_scaled_jacobian` is the smallest corner scaled Jacobian over all
/// quads, so an inverted cell is negative and a degenerate one is zero.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Stats {
    /// Number of quad cells measured.
    pub cells: usize,
    /// Smallest corner angle over all quads (90° is ideal, 0° degenerate).
    pub min_angle_deg: f64,
    /// Mean absolute deviation of a corner angle from 90°, in degrees.
    pub mean_angle_deviation_deg: f64,
    /// 95th percentile of the quad aspect ratios (nearest rank).
    pub aspect_p95: f64,
    /// Largest quad aspect ratio.
    pub aspect_max: f64,
    /// Smallest corner scaled Jacobian (1 ideal, > 0 valid, < 0 inverted).
    pub min_scaled_jacobian: f64,
    /// Largest scaled longest-diagonal/shortest-edge ratio (1 ideal).
    pub max_stretch: f64,
}

/// The result of a smoothing run, with the measurements on both sides.
#[derive(Clone, Debug)]
pub struct Outcome {
    /// The smoothed mesh (a clone with only free-node `x, y` possibly changed).
    pub mesh: Mesh,
    /// Statistics of the input mesh.
    pub before: Stats,
    /// Statistics of [`Self::mesh`].
    pub after: Stats,
    /// Number of sweeps actually executed (may stop early).
    pub sweeps: usize,
    /// Number of distinct nodes that were moved at least once.
    pub moved_nodes: usize,
}

/// The numbers produced by [`demo`], ready to assert on or print.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DemoNumbers {
    /// The deterministic 6×6 grid after the perturbation, before smoothing.
    pub before: Stats,
    /// Smoothing with the angle-only objective (`stretch_weight = 0`).
    pub angle_only: Stats,
    /// Smoothing with the angle + stretch objective.
    pub angle_stretch: Stats,
}

/// Smooth an all-quad mesh with the thesis' angle or angle+stretch objective.
///
/// Only `x, y` of the free nodes are moved; cells and connectivity are kept.
/// Interior nodes are free unless `options.allow_boundary` is set, in which
/// case boundary nodes may also move (inside the domain and by at most
/// `0.75 * size` per sweep step). `size` is the target element size and sets
/// the simplex/pattern step.
///
/// Returns `Err` for a non-quad mesh, an inverted or non-convex quad, a
/// non-finite coordinate, a non-positive/non-finite `size`, or a negative
/// stretch weight. The function never panics and never returns an inverted
/// cell for a valid input.
pub fn smooth(domain: &Domain, mesh: &Mesh, size: f64, options: &Options) -> Result<Mesh, String> {
    Ok(smooth_with_stats(domain, mesh, size, options)?.mesh)
}

/// [`smooth`], also returning the before/after statistics and the work done.
pub fn smooth_with_stats(
    domain: &Domain,
    mesh: &Mesh,
    size: f64,
    options: &Options,
) -> Result<Outcome, String> {
    check_options(size, options)?;
    check_input(mesh)?;

    let before = measure(mesh)?;
    if options.iterations == 0 {
        return Ok(Outcome {
            mesh: mesh.clone(),
            before,
            after: before,
            sweeps: 0,
            moved_nodes: 0,
        });
    }

    let weight = options.stretch_weight;
    let mut points = mesh.points.clone();
    let node_cells = build_node_cells(mesh);
    let boundary = boundary_nodes(mesh);
    let mut moved = vec![false; points.len()];
    let mut sweeps = 0;

    for _ in 0..options.iterations {
        let mut moved_this_sweep = 0usize;
        for node in 0..points.len() {
            let incident = &node_cells[node];
            if incident.is_empty() {
                continue;
            }
            if boundary[node] && !options.allow_boundary {
                continue;
            }
            let start = [points[node][0], points[node][1]];
            // The candidate must stay local: a move larger than 0.75 size is
            // rejected, which keeps a node inside its star and prevents a
            // long jump across unrelated cells.
            let max_step = 0.75 * size;
            let boundary_node = boundary[node];
            let evaluate = |p: [f64; 2]| -> f64 {
                let d = (p[0] - start[0]).hypot(p[1] - start[1]);
                if !d.is_finite() || d > max_step {
                    return INVALID;
                }
                if boundary_node && !inside_or_on(domain, p, size) {
                    return INVALID;
                }
                local_objective(&points, incident, &mesh.cells, node, p, weight)
            };

            let f_start = evaluate(start);
            if !f_start.is_finite() || f_start >= INVALID {
                continue;
            }
            let mut f = evaluate;
            let best = if options.simplex {
                nelder_mead(start, 0.25 * size, 80, &mut f)
            } else {
                coordinate_search(start, 0.25 * size, 12, &mut f)
            };
            // Re-check the winner: the search may return a vertex with the
            // penalty value (e.g. when every vertex is invalid), and a strict
            // decrease is required so that a perfect grid never moves.
            let f_best = f(best);
            if f_best < f_start - ACCEPT_TOLERANCE * (1.0 + f_start.abs())
                && f_best < INVALID
                && quad_star_is_valid(&points, incident, &mesh.cells, node, best)
            {
                points[node][0] = best[0];
                points[node][1] = best[1];
                moved[node] = true;
                moved_this_sweep += 1;
            }
        }
        sweeps += 1;
        if moved_this_sweep == 0 {
            break;
        }
    }

    // The local guards are complete, but a final global check costs almost
    // nothing on a smoothing pass and turns any unforeseen defect into an
    // error instead of an inverted mesh.
    let out = Mesh::new(points, mesh.cells.clone());
    check_output(&out, mesh)?;
    let after = measure(&out)?;
    let moved_nodes = moved.iter().filter(|&&m| m).count();
    Ok(Outcome {
        mesh: out,
        before,
        after,
        sweeps,
        moved_nodes,
    })
}

/// Measure the statistics reported for a smoothing run.
///
/// `Err` for an empty mesh, a non-quad cell, or a NaN/∞ metric (a degenerate
/// quad with a zero-length edge).
pub fn measure(mesh: &Mesh) -> Result<Stats, String> {
    if mesh.cells.is_empty() {
        return Err("cannot measure a mesh with no cells".to_string());
    }
    let mut min_angle = f64::INFINITY;
    let mut deviation_sum = 0.0;
    let mut aspect = Vec::with_capacity(mesh.cells.len());
    let mut min_sj = f64::INFINITY;
    let mut max_stretch = 0.0f64;
    let mut corners = 0usize;

    for (index, cell) in mesh.cells.iter().enumerate() {
        if cell.kind != Kind::Quad4 {
            return Err(format!(
                "cell {index} is {}, but smoothing needs an all-quad mesh",
                cell.kind.name()
            ));
        }
        let quad = quad_points(cell, &mesh.points);
        let angles = quality::quad_angles(cell, &mesh.points);
        if angles.len() != 4 || angles.iter().any(|a| !a.is_finite()) {
            return Err(format!("cell {index} has a non-finite corner angle"));
        }
        for &angle in &angles {
            min_angle = min_angle.min(angle);
            deviation_sum += (angle - 90.0).abs();
            corners += 1;
        }
        let ratio = quality::aspect_ratio(cell, &mesh.points);
        if !ratio.is_finite() {
            return Err(format!("cell {index} has a zero-length edge"));
        }
        aspect.push(ratio);
        let jacobians = quality::scaled_jacobian(cell, &mesh.points);
        for j in jacobians {
            if !j.is_finite() {
                return Err(format!("cell {index} has a non-finite Jacobian"));
            }
            min_sj = min_sj.min(j);
        }
        let stretch = raw_stretch(&quad);
        if !stretch.is_finite() {
            return Err(format!("cell {index} has a zero-length edge"));
        }
        max_stretch = max_stretch.max(stretch);
    }

    let n = aspect.len();
    aspect.sort_by(f64::total_cmp);
    // Nearest-rank p95: with 36 cells this is the 35th smallest value.
    let rank = ((0.95 * n as f64).ceil() as usize).clamp(1, n);
    let aspect_p95 = aspect[rank - 1];
    let aspect_max = aspect[n - 1];
    let mean_angle_deviation_deg = if corners > 0 {
        deviation_sum / corners as f64
    } else {
        0.0
    };
    Ok(Stats {
        cells: n,
        min_angle_deg: min_angle,
        mean_angle_deviation_deg,
        aspect_p95,
        aspect_max,
        min_scaled_jacobian: min_sj,
        max_stretch,
    })
}

/// Run the deterministic demonstration and return its measured numbers.
///
/// A 6×6 grid of unit squares on a `6 × 6` square is perturbed deterministically:
/// the interior nodes move in both coordinates (up to `0.1 h`), and the boundary
/// nodes slide along their side (up to `0.25 h`, opposite sides sharing the
/// pattern, corners fixed). The boundary irregularity makes a perfect grid
/// impossible, so the angle-only objective has to trade shape for angle — the
/// §2.2.10 pinch. Both runs use 60 sweeps: (a) angle-only and (b) angle +
/// stretch with `stretch_weight = 2`. The returned [`DemoNumbers`] are what the
/// `demo_stretch_keeps_aspect_bounded` test asserts on.
pub fn demo() -> Result<DemoNumbers, String> {
    let n = 6usize;
    let h = 1.0;
    let domain = Domain::rectangle(n as f64 * h, n as f64 * h).map_err(|e| e.to_string())?;
    let perturbed = perturbed_grid(n, h, 0.25, 0.1)?;
    let before = measure(&perturbed)?;

    let angle_options = Options {
        iterations: 60,
        stretch_weight: 0.0,
        allow_boundary: false,
        simplex: true,
    };
    let angle_mesh = smooth(&domain, &perturbed, h, &angle_options)?;
    let angle_only = measure(&angle_mesh)?;

    let stretch_options = Options {
        iterations: 60,
        stretch_weight: 2.0,
        allow_boundary: false,
        simplex: true,
    };
    let stretch_mesh = smooth(&domain, &perturbed, h, &stretch_options)?;
    let angle_stretch = measure(&stretch_mesh)?;

    Ok(DemoNumbers {
        before,
        angle_only,
        angle_stretch,
    })
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn check_options(size: f64, options: &Options) -> Result<(), String> {
    if !size.is_finite() || size <= 0.0 {
        return Err(format!("size must be finite and positive, got {size}"));
    }
    if !options.stretch_weight.is_finite() || options.stretch_weight < 0.0 {
        return Err(format!(
            "stretch_weight must be finite and non-negative, got {}",
            options.stretch_weight
        ));
    }
    Ok(())
}

fn check_input(mesh: &Mesh) -> Result<(), String> {
    mesh.validate().map_err(|e| format!("invalid mesh: {e}"))?;
    if mesh.cells.is_empty() {
        return Err("cannot smooth a mesh with no cells".to_string());
    }
    for (index, cell) in mesh.cells.iter().enumerate() {
        if cell.kind != Kind::Quad4 {
            return Err(format!(
                "cell {index} is {}, but smoothing needs an all-quad mesh",
                cell.kind.name()
            ));
        }
        let quad = quad_points(cell, &mesh.points);
        if !quad_is_valid(&quad) {
            return Err(format!(
                "cell {index} is inverted, degenerate or non-convex"
            ));
        }
    }
    Ok(())
}

fn check_output(out: &Mesh, input: &Mesh) -> Result<(), String> {
    if out.points.len() != input.points.len() || out.cells.len() != input.cells.len() {
        return Err("internal error: smoothing changed the mesh size".to_string());
    }
    for (index, cell) in out.cells.iter().enumerate() {
        let quad = quad_points(cell, &out.points);
        if !quad_is_valid(&quad) {
            return Err(format!("smoothing inverted cell {index}"));
        }
        // Topology is untouched, but verify the node loop is the same one.
        if cell.nodes != input.cells[index].nodes {
            return Err("internal error: smoothing changed the connectivity".to_string());
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

fn quad_points(cell: &Cell, points: &[[f64; 3]]) -> [[f64; 2]; 4] {
    [
        [points[cell.nodes[0]][0], points[cell.nodes[0]][1]],
        [points[cell.nodes[1]][0], points[cell.nodes[1]][1]],
        [points[cell.nodes[2]][0], points[cell.nodes[2]][1]],
        [points[cell.nodes[3]][0], points[cell.nodes[3]][1]],
    ]
}

/// Signed area of a quad by the shoelace formula, in the `xy` plane.
fn quad_area(q: &[[f64; 2]; 4]) -> f64 {
    let mut sum = 0.0;
    for i in 0..4 {
        let j = (i + 1) % 4;
        sum += q[i][0] * q[j][1] - q[j][0] * q[i][1];
    }
    0.5 * sum
}

/// Corner Jacobian at corner `i`: the sine of the turn from the incoming edge
/// to the outgoing edge. Positive for a strictly convex counter-clockwise
/// corner, zero for a straight one, negative for a reflex (concave) one.
fn corner_jacobian(q: &[[f64; 2]; 4], i: usize) -> f64 {
    let prev = q[(i + 3) % 4];
    let next = q[(i + 1) % 4];
    let ax = q[i][0] - prev[0];
    let ay = q[i][1] - prev[1];
    let bx = next[0] - q[i][0];
    let by = next[1] - q[i][1];
    let denom = ax.hypot(ay) * bx.hypot(by);
    if denom > 0.0 {
        (ax * by - ay * bx) / denom
    } else {
        f64::NEG_INFINITY
    }
}

fn quad_is_valid(q: &[[f64; 2]; 4]) -> bool {
    if quad_area(q) <= 0.0 {
        return false;
    }
    (0..4).all(|i| corner_jacobian(q, i) > MIN_CORNER_JACOBIAN)
}

/// Interior angle at corner `i` of a counter-clockwise quad, in radians.
fn corner_angle(q: &[[f64; 2]; 4], i: usize) -> f64 {
    let prev = q[(i + 3) % 4];
    let next = q[(i + 1) % 4];
    let ax = prev[0] - q[i][0];
    let ay = prev[1] - q[i][1];
    let bx = next[0] - q[i][0];
    let by = next[1] - q[i][1];
    let cross = ax * by - ay * bx;
    let dot = ax * bx + ay * by;
    cross.abs().atan2(dot)
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// The thesis' stretch: longest diagonal divided by shortest edge, scaled so a
/// square is 1 (the unscaled ratio of a square is `sqrt(2)`).
fn raw_stretch(q: &[[f64; 2]; 4]) -> f64 {
    let d0 = distance(q[0], q[2]);
    let d1 = distance(q[1], q[3]);
    let mut shortest = f64::INFINITY;
    for i in 0..4 {
        shortest = shortest.min(distance(q[i], q[(i + 1) % 4]));
    }
    if shortest <= 0.0 {
        return f64::INFINITY;
    }
    let ratio = d0.max(d1) / shortest;
    ratio / std::f64::consts::SQRT_2
}

fn stretch_penalty(q: &[[f64; 2]; 4]) -> f64 {
    let stretch = raw_stretch(q);
    if !stretch.is_finite() {
        return INVALID;
    }
    (stretch - 1.0).powi(2)
}

/// Local objective for moving `node` to `candidate`: the sum of squared corner
/// angle deviations over every incident quad, plus `weight` times the sum of
/// squared stretch excesses. Invalid incident quads return [`INVALID`].
fn local_objective(
    points: &[[f64; 3]],
    incident: &[usize],
    cells: &[Cell],
    node: usize,
    candidate: [f64; 2],
    weight: f64,
) -> f64 {
    let mut total = 0.0;
    for &index in incident {
        let cell = &cells[index];
        let mut quad = quad_points(cell, points);
        // Replace the moving node's corner. `node` is guaranteed to be in the
        // cell because the incidence list was built from the cells.
        let mut found = false;
        for k in 0..4 {
            if cell.nodes[k] == node {
                quad[k] = candidate;
                found = true;
            }
        }
        if !found || !quad_is_valid(&quad) {
            return INVALID;
        }
        for i in 0..4 {
            let deviation = corner_angle(&quad, i) - RIGHT_ANGLE;
            total += deviation * deviation;
        }
        if weight > 0.0 {
            total += weight * stretch_penalty(&quad);
        }
    }
    total
}

fn quad_star_is_valid(
    points: &[[f64; 3]],
    incident: &[usize],
    cells: &[Cell],
    node: usize,
    candidate: [f64; 2],
) -> bool {
    for &index in incident {
        let cell = &cells[index];
        let mut quad = quad_points(cell, points);
        let mut found = false;
        for k in 0..4 {
            if cell.nodes[k] == node {
                quad[k] = candidate;
                found = true;
            }
        }
        if !found || !quad_is_valid(&quad) {
            return false;
        }
    }
    true
}

fn inside_or_on(domain: &Domain, p: [f64; 2], size: f64) -> bool {
    domain.contains(p) || domain.distance_to_boundary(p) <= 1.0e-10 * size.max(1.0)
}

// ---------------------------------------------------------------------------
// Mesh incidence
// ---------------------------------------------------------------------------

fn build_node_cells(mesh: &Mesh) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new(); mesh.points.len()];
    for (index, cell) in mesh.cells.iter().enumerate() {
        for &node in &cell.nodes {
            out[node].push(index);
        }
    }
    out
}

/// A node is a boundary node when one of its edges is not shared by exactly
/// two quads: an edge with one incident cell is the mesh boundary, and an edge
/// with more than two is non-manifold and is held for safety.
fn boundary_nodes(mesh: &Mesh) -> Vec<bool> {
    let mut edges: HashMap<(usize, usize), u32> = HashMap::new();
    for cell in &mesh.cells {
        for &(a, b) in Kind::Quad4.edges() {
            let u = cell.nodes[a];
            let v = cell.nodes[b];
            let key = if u < v { (u, v) } else { (v, u) };
            *edges.entry(key).or_insert(0) += 1;
        }
    }
    let mut boundary = vec![false; mesh.points.len()];
    for ((u, v), count) in edges {
        if count != 2 {
            boundary[u] = true;
            boundary[v] = true;
        }
    }
    boundary
}

// ---------------------------------------------------------------------------
// Derivative-free optimisers
// ---------------------------------------------------------------------------

/// 2D Nelder–Mead simplex (Nelder & Mead 1965) with the standard coefficients
/// `alpha = 1`, `gamma = 2`, `rho = 0.5`, `sigma = 0.5`.
///
/// The initial simplex is the right triangle `{p, p + (h,0), p + (0,h)}`; it
/// is enough in 2D and avoids an orientation-dependent starting direction. The
/// function returns the best vertex after at most `max_iterations` iterations
/// and never panics (the vertex buffers have fixed length three).
fn nelder_mead<F>(start: [f64; 2], h: f64, max_iterations: usize, objective: &mut F) -> [f64; 2]
where
    F: FnMut([f64; 2]) -> f64,
{
    let mut vertices = [start, [start[0] + h, start[1]], [start[0], start[1] + h]];
    let mut values = [
        objective(vertices[0]),
        objective(vertices[1]),
        objective(vertices[2]),
    ];
    for _ in 0..max_iterations {
        let mut order = [0usize, 1, 2];
        order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
        let best = order[0];
        let middle = order[1];
        let worst = order[2];

        let spread = values[worst] - values[best];
        let vertex_size = (0..2)
            .map(|k| (vertices[worst][k] - vertices[best][k]).abs())
            .fold(0.0f64, f64::max);
        if spread <= 1.0e-14 * (1.0 + values[best].abs()) && vertex_size <= 1.0e-10 {
            break;
        }

        let centroid = [
            0.5 * (vertices[best][0] + vertices[middle][0]),
            0.5 * (vertices[best][1] + vertices[middle][1]),
        ];
        let reflected = [
            centroid[0] + (centroid[0] - vertices[worst][0]),
            centroid[1] + (centroid[1] - vertices[worst][1]),
        ];
        let f_reflected = objective(reflected);

        if f_reflected < values[best] {
            let expanded = [
                centroid[0] + 2.0 * (centroid[0] - vertices[worst][0]),
                centroid[1] + 2.0 * (centroid[1] - vertices[worst][1]),
            ];
            let f_expanded = objective(expanded);
            if f_expanded < f_reflected {
                vertices[worst] = expanded;
                values[worst] = f_expanded;
            } else {
                vertices[worst] = reflected;
                values[worst] = f_reflected;
            }
        } else if f_reflected < values[middle] {
            vertices[worst] = reflected;
            values[worst] = f_reflected;
        } else {
            // Outside contraction when the reflection is better than the
            // worst point, inside contraction otherwise.
            let contracted = if f_reflected < values[worst] {
                [
                    centroid[0] + 0.5 * (reflected[0] - centroid[0]),
                    centroid[1] + 0.5 * (reflected[1] - centroid[1]),
                ]
            } else {
                [
                    centroid[0] + 0.5 * (vertices[worst][0] - centroid[0]),
                    centroid[1] + 0.5 * (vertices[worst][1] - centroid[1]),
                ]
            };
            let f_contracted = objective(contracted);
            let threshold = values[worst].min(f_reflected);
            if f_contracted < threshold {
                vertices[worst] = contracted;
                values[worst] = f_contracted;
            } else {
                // Shrink towards the best vertex.
                for k in [order[1], order[2]] {
                    vertices[k] = [
                        vertices[best][0] + 0.5 * (vertices[k][0] - vertices[best][0]),
                        vertices[best][1] + 0.5 * (vertices[k][1] - vertices[best][1]),
                    ];
                    values[k] = objective(vertices[k]);
                }
            }
        }
    }

    let mut best = 0usize;
    for k in 1..3 {
        if values[k] < values[best] {
            best = k;
        }
    }
    vertices[best]
}

/// Coordinate/pattern search in 2D: probe the eight neighbours at the current
/// step, move to the best improving one, and halve the step when none helps.
/// Cheaper than the simplex and monotone by construction.
fn coordinate_search<F>(
    start: [f64; 2],
    step: f64,
    max_rounds: usize,
    objective: &mut F,
) -> [f64; 2]
where
    F: FnMut([f64; 2]) -> f64,
{
    const DIAGONAL: f64 = std::f64::consts::FRAC_1_SQRT_2;
    const DIRECTIONS: [[f64; 2]; 8] = [
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, 1.0],
        [0.0, -1.0],
        [DIAGONAL, DIAGONAL],
        [DIAGONAL, -DIAGONAL],
        [-DIAGONAL, DIAGONAL],
        [-DIAGONAL, -DIAGONAL],
    ];
    let mut best = start;
    let mut f_best = objective(best);
    let mut h = step;
    for _ in 0..max_rounds {
        let mut improved = false;
        for direction in DIRECTIONS {
            let candidate = [best[0] + h * direction[0], best[1] + h * direction[1]];
            let f = objective(candidate);
            if f < f_best {
                best = candidate;
                f_best = f;
                improved = true;
            }
        }
        if !improved {
            h *= 0.5;
            if h <= step * 1.0e-4 {
                break;
            }
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Demo construction
// ---------------------------------------------------------------------------

/// A structured `n × n` grid of square cells on `[0, n*h] × [0, n*h]`.
fn grid(n: usize, h: f64) -> Mesh {
    let side = n + 1;
    let mut points = Vec::with_capacity(side * side);
    for j in 0..side {
        for i in 0..side {
            points.push([i as f64 * h, j as f64 * h, 0.0]);
        }
    }
    let mut cells = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            let n0 = j * side + i;
            let n1 = n0 + 1;
            let n3 = (j + 1) * side + i;
            let n2 = n3 + 1;
            cells.push(Cell::quad4([n0, n1, n2, n3]));
        }
    }
    Mesh::new(points, cells)
}

/// Deterministic pseudo-random perturbation of a structured grid.
///
/// Interior nodes move in both coordinates by up to `interior * h`. Boundary
/// nodes **slide along their side** of the square by up to `boundary * h`
/// (their coordinate across the side stays exact); the four corners are not
/// moved.
///
/// The offsets use a small integer hash (no dependency, no randomness seed), so
/// the demonstration is identical on every run and platform. Opposite sides
/// share the same offset pattern: the bottom and top sides use the same `x`
/// offsets, the left and right sides the same `y` offsets. A rectangular grid
/// with the resulting irregular column widths and row heights is therefore
/// still *feasible*, and it is exactly the trap of the angle-only objective:
/// all its angles are 90° no matter how slender the columns are. The
/// angle+stretch objective has to give up a little angle to avoid that grid.
///
/// With an exact, uniformly spaced square boundary the unperturbed lattice is
/// a zero of the angle objective, so an angle-only smoother would simply
/// recover it and no pinch would be visible; that is why the fixed boundary is
/// made irregular as well as the interior. The interior perturbation is what
/// gives the angle-only smoother something to fix; the boundary offset is what
/// makes a perfect grid impossible.
fn perturbed_grid(
    n: usize,
    h: f64,
    boundary_step: f64,
    interior_step: f64,
) -> Result<Mesh, String> {
    if n < 2 {
        return Err("the demo grid needs at least two cells per side".to_string());
    }
    let mut mesh = grid(n, h);
    let side = n + 1;
    for j in 0..=n {
        for i in 0..=n {
            let node = j * side + i;
            let interior = i > 0 && i < n && j > 0 && j < n;
            let on_horizontal = j == 0 || j == n;
            let on_vertical = i == 0 || i == n;
            if interior {
                let fx = unit_hash(i, j, 1) - 0.5;
                let fy = unit_hash(i, j, 2) - 0.5;
                mesh.points[node][0] += 2.0 * interior_step * h * fx;
                mesh.points[node][1] += 2.0 * interior_step * h * fy;
            } else if on_horizontal && !on_vertical {
                // Bottom and top share the pattern, so columns can be
                // straight and the whole grid can be rectangular.
                let fx = unit_hash(i, 0, 3) - 0.5;
                mesh.points[node][0] += 2.0 * boundary_step * h * fx;
            } else if on_vertical && !on_horizontal {
                // Left and right share the pattern, so rows can be straight.
                let fy = unit_hash(0, j, 4) - 0.5;
                mesh.points[node][1] += 2.0 * boundary_step * h * fy;
            }
            // The four corners stay put.
        }
    }
    check_input(&mesh)?;
    Ok(mesh)
}

/// A deterministic hash in `[0, 1)`, used only by the demonstration.
fn unit_hash(i: usize, j: usize, salt: u64) -> f64 {
    let mut h = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (j as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ salt.wrapping_mul(0xD6E8_FEB8_6659_FD93);
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 33;
    (h & 0x00FF_FFFF) as f64 / 16_777_216.0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::assertions_on_constants)]
mod tests {
    use super::*;

    fn domain_6() -> Domain {
        match Domain::rectangle(6.0, 6.0) {
            Ok(domain) => domain,
            Err(error) => {
                assert!(false, "test domain invalid: {error}");
                Domain::default()
            }
        }
    }

    fn options_stretch() -> Options {
        Options {
            iterations: 40,
            stretch_weight: 1.0,
            allow_boundary: false,
            simplex: true,
        }
    }

    fn smooth_ok(domain: &Domain, mesh: &Mesh, options: &Options) -> Mesh {
        match smooth(domain, mesh, 1.0, options) {
            Ok(mesh) => mesh,
            Err(error) => {
                assert!(false, "smoothing failed: {error}");
                Mesh::default()
            }
        }
    }

    fn measure_ok(mesh: &Mesh) -> Stats {
        match measure(mesh) {
            Ok(stats) => stats,
            Err(error) => {
                assert!(false, "measure failed: {error}");
                Stats {
                    cells: 0,
                    min_angle_deg: 0.0,
                    mean_angle_deviation_deg: 0.0,
                    aspect_p95: 0.0,
                    aspect_max: 0.0,
                    min_scaled_jacobian: 0.0,
                    max_stretch: 0.0,
                }
            }
        }
    }

    fn grid_ok(n: usize, h: f64) -> Mesh {
        grid(n, h)
    }

    fn perturbed_ok(n: usize, h: f64, boundary: f64, interior: f64) -> Mesh {
        match perturbed_grid(n, h, boundary, interior) {
            Ok(mesh) => mesh,
            Err(error) => {
                assert!(false, "perturbation failed: {error}");
                Mesh::default()
            }
        }
    }

    #[test]
    fn demo_stretch_keeps_aspect_bounded() {
        let numbers = match demo() {
            Ok(numbers) => numbers,
            Err(error) => {
                assert!(false, "demo failed: {error}");
                return;
            }
        };
        println!(
            "before:        min angle {:.3} deg, mean dev {:.3} deg, aspect p95 {:.3}, max {:.3}, min SJ {:.4}, stretch {:.3}",
            numbers.before.min_angle_deg,
            numbers.before.mean_angle_deviation_deg,
            numbers.before.aspect_p95,
            numbers.before.aspect_max,
            numbers.before.min_scaled_jacobian,
            numbers.before.max_stretch
        );
        println!(
            "angle-only:    min angle {:.3} deg, mean dev {:.3} deg, aspect p95 {:.3}, max {:.3}, min SJ {:.4}, stretch {:.3}",
            numbers.angle_only.min_angle_deg,
            numbers.angle_only.mean_angle_deviation_deg,
            numbers.angle_only.aspect_p95,
            numbers.angle_only.aspect_max,
            numbers.angle_only.min_scaled_jacobian,
            numbers.angle_only.max_stretch
        );
        println!(
            "angle+stretch: min angle {:.3} deg, mean dev {:.3} deg, aspect p95 {:.3}, max {:.3}, min SJ {:.4}, stretch {:.3}",
            numbers.angle_stretch.min_angle_deg,
            numbers.angle_stretch.mean_angle_deviation_deg,
            numbers.angle_stretch.aspect_p95,
            numbers.angle_stretch.aspect_max,
            numbers.angle_stretch.min_scaled_jacobian,
            numbers.angle_stretch.max_stretch
        );

        // Both objectives improve the angles of the perturbed grid.
        assert!(
            numbers.angle_only.min_angle_deg > numbers.before.min_angle_deg + 1.0,
            "angle-only should raise the minimum angle: {:.3} -> {:.3}",
            numbers.before.min_angle_deg,
            numbers.angle_only.min_angle_deg
        );
        assert!(
            numbers.angle_stretch.min_angle_deg > numbers.before.min_angle_deg,
            "angle+stretch should raise the minimum angle: {:.3} -> {:.3}",
            numbers.before.min_angle_deg,
            numbers.angle_stretch.min_angle_deg
        );
        assert!(
            numbers.angle_only.mean_angle_deviation_deg < numbers.before.mean_angle_deviation_deg,
            "angle-only should lower the mean angle deviation"
        );
        assert!(
            numbers.angle_stretch.mean_angle_deviation_deg
                < numbers.before.mean_angle_deviation_deg,
            "angle+stretch should lower the mean angle deviation"
        );

        // The qualitative difference the thesis reports: the angle-only run
        // buys angles by stretching cells, so it makes the aspect tail worse
        // than the input, while the stretch term keeps that tail bounded.
        assert!(
            numbers.angle_only.aspect_max > 1.1 * numbers.before.aspect_max,
            "angle-only should increase the max aspect: {:.3} -> {:.3}",
            numbers.before.aspect_max,
            numbers.angle_only.aspect_max
        );
        assert!(
            numbers.angle_only.aspect_p95 > numbers.before.aspect_p95,
            "angle-only should increase the aspect p95: {:.3} -> {:.3}",
            numbers.before.aspect_p95,
            numbers.angle_only.aspect_p95
        );
        assert!(
            numbers.angle_only.aspect_max > numbers.angle_stretch.aspect_max,
            "the angle-only run should have the worse aspect tail"
        );
        assert!(
            numbers.angle_stretch.aspect_max <= numbers.before.aspect_max,
            "the stretch term should keep max aspect at or below the input: {:.3} vs {:.3}",
            numbers.angle_stretch.aspect_max,
            numbers.before.aspect_max
        );
        assert!(
            numbers.angle_stretch.aspect_p95 <= numbers.before.aspect_p95,
            "the stretch term should keep p95 aspect at or below the input"
        );
        assert!(
            numbers.angle_stretch.aspect_max < 0.85 * numbers.angle_only.aspect_max,
            "the stretch term should bound max aspect by a clear margin: {:.3} vs {:.3}",
            numbers.angle_stretch.aspect_max,
            numbers.angle_only.aspect_max
        );
        assert!(
            numbers.angle_stretch.max_stretch < numbers.angle_only.max_stretch,
            "the stretch term should lower max stretch"
        );

        // No inversion and no boundary movement by default.
        assert!(numbers.angle_only.min_scaled_jacobian > 0.0);
        assert!(numbers.angle_stretch.min_scaled_jacobian > 0.0);
        assert_eq!(numbers.angle_stretch.cells, 36);
    }

    #[test]
    fn boundary_nodes_are_fixed_by_default() {
        let domain = domain_6();
        let mesh = perturbed_ok(6, 1.0, 0.25, 0.1);
        let smoothed = smooth_ok(&domain, &mesh, &options_stretch());
        let side = 7usize;
        for j in 0..=6 {
            for i in 0..=6 {
                if i != 0 && i != 6 && j != 0 && j != 6 {
                    continue;
                }
                let node = j * side + i;
                assert_eq!(
                    mesh.points[node], smoothed.points[node],
                    "boundary node {node} (i={i}, j={j}) moved"
                );
            }
        }
        let topology = smoothed.topology();
        assert_eq!(topology.cracks, 0);
        assert_eq!(topology.non_manifold, 0);
        assert_eq!(smoothed.cells.len(), mesh.cells.len());
        let after = measure_ok(&smoothed);
        assert!(after.min_scaled_jacobian > 0.0);
    }

    #[test]
    fn badly_perturbed_grid_never_inverts() {
        let domain = domain_6();
        // 0.48 h per node is close to the 0.5 h that would make two nodes
        // swap; the input is still valid but has very thin cells.
        let mesh = perturbed_ok(6, 1.0, 0.4, 0.15);
        let before = measure_ok(&mesh);
        assert!(
            before.min_scaled_jacobian > 0.0,
            "input is already inverted"
        );
        for simplex in [true, false] {
            let options = Options {
                iterations: 60,
                stretch_weight: 1.0,
                allow_boundary: false,
                simplex,
            };
            let smoothed = smooth_ok(&domain, &mesh, &options);
            let after = measure_ok(&smoothed);
            assert!(
                after.min_scaled_jacobian > 0.0,
                "inverted output with simplex = {simplex}: SJ {}",
                after.min_scaled_jacobian
            );
            assert_eq!(after.cells, before.cells);
            assert_eq!(smoothed.topology().cracks, 0);
        }
    }

    #[test]
    fn structured_grid_is_unchanged() {
        let domain = domain_6();
        let mesh = grid_ok(6, 1.0);
        let before = measure_ok(&mesh);
        for simplex in [true, false] {
            let options = Options {
                iterations: 10,
                stretch_weight: 1.0,
                allow_boundary: false,
                simplex,
            };
            let smoothed = smooth_ok(&domain, &mesh, &options);
            for (a, b) in mesh.points.iter().zip(smoothed.points.iter()) {
                assert!((a[0] - b[0]).abs() < 1.0e-12, "x changed on a perfect grid");
                assert!((a[1] - b[1]).abs() < 1.0e-12, "y changed on a perfect grid");
            }
            let after = measure_ok(&smoothed);
            assert!((after.min_angle_deg - 90.0).abs() < 1.0e-9);
            assert!((after.aspect_max - before.aspect_max).abs() < 1.0e-12);
        }
    }

    #[test]
    fn coordinate_search_also_improves_angles() {
        let domain = domain_6();
        let mesh = perturbed_ok(6, 1.0, 0.2, 0.1);
        let before = measure_ok(&mesh);
        let options = Options {
            iterations: 40,
            stretch_weight: 0.5,
            allow_boundary: false,
            simplex: false,
        };
        let smoothed = smooth_ok(&domain, &mesh, &options);
        let after = measure_ok(&smoothed);
        assert!(
            after.mean_angle_deviation_deg < before.mean_angle_deviation_deg,
            "pattern search should lower the mean angle deviation"
        );
        assert!(after.min_scaled_jacobian > 0.0);
    }

    #[test]
    fn bad_input_is_an_error() {
        let domain = domain_6();
        let options = Options::default();

        // A triangle is not an all-quad mesh.
        let triangle = Mesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![Cell::tri3([0, 1, 2])],
        );
        assert!(smooth(&domain, &triangle, 1.0, &options).is_err());

        // A clockwise quad is inverted.
        let inverted = Mesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            vec![Cell::quad4([0, 1, 2, 3])],
        );
        assert!(smooth(&domain, &inverted, 1.0, &options).is_err());

        // Bad size and bad weight.
        let good = grid_ok(2, 1.0);
        assert!(smooth(&domain, &good, 0.0, &options).is_err());
        assert!(smooth(&domain, &good, f64::NAN, &options).is_err());
        let negative = Options {
            stretch_weight: -1.0,
            ..options
        };
        assert!(smooth(&domain, &good, 1.0, &negative).is_err());

        // Empty mesh.
        let empty = Mesh::default();
        assert!(smooth(&domain, &empty, 1.0, &options).is_err());
    }

    #[test]
    fn allow_boundary_keeps_mesh_valid() {
        let domain = domain_6();
        let mesh = perturbed_ok(6, 1.0, 0.2, 0.1);
        let options = Options {
            iterations: 10,
            stretch_weight: 1.0,
            allow_boundary: true,
            simplex: true,
        };
        let smoothed = smooth_ok(&domain, &mesh, &options);
        let after = measure_ok(&smoothed);
        assert!(after.min_scaled_jacobian > 0.0);
        assert_eq!(smoothed.topology().cracks, 0);
        // Every node stays inside the domain (the outer loop is the square).
        for p in &smoothed.points {
            assert!(p[0] >= -1.0e-8 && p[0] <= 6.0 + 1.0e-8);
            assert!(p[1] >= -1.0e-8 && p[1] <= 6.0 + 1.0e-8);
        }
    }
}
