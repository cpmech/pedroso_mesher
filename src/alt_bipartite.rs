//! Right-angled triangle mesh + bipartite labelling, the indirect all-quad route
//! of Georgiadis, Reberol and Remacle, *Indirect all-quadrilateral meshing based
//! on bipartite topological labeling*, Engineering with Computers (2022).
//!
//! The route starts from the plane triangulation the crate already produces
//! (`two_d::triangulate`) and converts it to quads by **pairing triangles along
//! a set of edges picked by a 2-colouring of the mesh vertices**:
//!
//! * §5: the vertices are labelled with two colours. Every triangle has three
//!   vertices and two colours, so it carries either one monochromatic edge (two
//!   equal labels) or three (a *monochromatic triangle*, all labels equal). If
//!   the boundary cycles are even they can be coloured alternately — so no
//!   boundary edge is monochromatic — and, once no triangle is monochromatic,
//!   every triangle owns exactly one monochromatic edge. Those edges are
//!   interior and form a perfect matching of the triangle adjacency: removing
//!   them leaves an all-quad mesh by construction.
//! * §3.1: each feature curve must carry an even number of edges. The boundary
//!   loops are made even with `parity::even_boundary_loops` (one split of the
//!   longest edge of an odd loop) before labelling.
//! * §4: a right-angled triangle mesh pairs into squares. We do the tractable
//!   core of the local modifications: interior edge flips that strictly increase
//!   the number of near-right angles without spoiling the smallest corner angle,
//!   the convexity of the quadrilateral formed by the two triangles, or the
//!   element area.
//! * §3 (approximation): the paper's frontal point generation guided by a
//!   cross-field is replaced by a **field-aligned CDT**: the boundary is sampled
//!   at 1.1 size, one row of points is placed 0.8 size inside the boundary
//!   following its normals, and the interior is seeded with a lattice at the
//!   size spacing. The constrained Delaunay triangulation then only fills a
//!   one-cell transition, so most interior vertices sit on a square lattice and
//!   the triangles are already right-angled (the §4 flip pass measures almost
//!   nothing to do). On the corpus this lifts the interior valence-4 share from
//!   ~60 % to 74-97 %.
//! * §5.1/§5.2: after the boundary alternation the interior labels are chosen
//!   greedily and improved by label smoothing. A label is scored by:
//!
//!   - the scaled Jacobian of the quad its monochromatic edge would create,
//!     with a large penalty for a non-convex (folded) quad;
//!   - a bonus when the monochromatic edge is the triangle's longest edge:
//!     two triangles that both pick their longest edge always form a convex
//!     quad, so this is the geometric analogue of the cross-field pairing;
//!   - the valence-4 target of the interior vertices, since a vertex of
//!     triangle valence `k` ends with quad valence `k` minus its number of
//!     monochromatic edges;
//!   - a penalty for a quad that would carry two boundary edges (three boundary
//!     vertices), which on a curved boundary is flat at the middle vertex.
//!
//!   Monochromatic triangles are repaired by splitting an edge (the common edge
//!   of a monochromatic pair, otherwise the longest edge) and giving the new
//!   Steiner point the opposite label (§5.1, Fig. 9).
//! * §5.2/§5.3: the quads are extracted and the interior vertices are moved to
//!   their stencil centres. The few invalid configurations the pairing can leave
//!   are corrected locally: zero-area input triangles are dropped, pairs of
//!   quads sharing two edges are merged into one quad (§5.3 doublet), free
//!   reflex vertices are unfolded, the crate's two-quad re-split is applied, and
//!   a transactional label-flip retry is tried when a fold or a flat quad
//!   remains.
//!
//! Not implemented: the paper's frontal point generation and cross-field (§3 and
//! the cross-field part of §5.2), the full set of local modifications of §4
//! (splits, collapses and the front-driven point replacement), the boundary
//! flat-quad correction that inserts vertices, and the topological optimisation
//! of Ref. [38]. If the labelling cannot eliminate every monochromatic triangle
//! within the repair budget, the offending triangles are returned as `Tri3`
//! cells and the paired fraction is reported in [`Stats`]; no all-quad result is
//! claimed for them.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use crate::mesh::{Cell, Mesh, Node};
use crate::parity;
use crate::two_d::{Domain, Sizing};

/// Undirected edge key, smaller node first.
type Edge = (usize, usize);

/// Sentinel for "no label yet"; a genuine label is 0 or 1.
const FREE: u8 = u8::MAX;

/// Score of a monochromatic triangle. It must dwarf the scaled-Jacobian range
/// (which is at most 1 per triangle) so smoothing never buys quad quality by
/// creating a monochromatic triangle.
const MONO_PENALTY: f64 = 1.0e3;

/// Extra penalty of a monochromatic edge that pairs into a non-convex quad.
/// A pair is scored from both of its triangles, so this is charged twice: a
/// flip that removes one folded quad wins over the small Jacobian differences
/// of its whole neighbourhood.
const NONCONVEX_PENALTY: f64 = 1.0e3;

/// Bonus for picking the longest edge as the monochromatic one. Two triangles
/// that both pick their longest edge always form a convex quad (the two corner
/// angles at each end of the shared edge are both < 90°), so this is what makes
/// the labelling produce valid elements rather than folded ones.
const LONGEST_BONUS: f64 = 1.0e3;

/// An angle within this band of 90° counts as a right angle.
const RIGHT_ANGLE_TOLERANCE_DEG: f64 = 10.0;

/// Upper bound on the right-angle flip sweeps.
const FLIP_ROUNDS: usize = 60;

/// A right-angle flip may reduce the smallest corner angle by this much.
const FLIP_ANGLE_DROP_DEG: f64 = 3.0;

/// ... but never below this floor.
const FLIP_MIN_CORNER_DEG: f64 = 20.0;

/// Upper bound on label-smoothing sweeps per call.
const LABEL_PASSES: usize = 80;

/// Upper bound on monochromatic-triangle repair rounds.
const REPAIR_ROUNDS: usize = 200;

/// Upper bound on Steiner points, as a fraction of the input triangle count.
const STEINER_FRACTION: f64 = 0.25;

/// Repositioning sweeps of the final vertex stencils.
const REPOSITION_ROUNDS: usize = 8;

/// Measurements of one [`mesh_with_stats`] run.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    /// Triangles in the input triangulation, before any preprocessing.
    pub input_triangles: usize,
    /// Zero-area triangles dropped from the input.
    pub degenerate_triangles: usize,
    /// Boundary edges split to make an odd boundary loop even.
    pub boundary_loop_splits: usize,
    /// Interior edge flips taken by the right-angle pass.
    pub right_angle_flips: usize,
    /// Near-right angles over all triangles before/after the flip pass.
    pub right_angles_before: usize,
    pub right_angles_after: usize,
    /// Monochromatic triangles left after the initial labelling and smoothing.
    pub mono_initial: usize,
    /// Label changes accepted by the smoothing (initial and post-repair).
    pub label_flips: usize,
    /// Steiner points inserted to break monochromatic triangles.
    pub steiner_points: usize,
    /// Monochromatic triangles that could not be eliminated.
    pub mono_left: usize,
    /// Triangles after every split, i.e. the triangles the quads consume.
    pub triangles_final: usize,
    pub quads: usize,
    /// Triangles left over (only non-zero when `mono_left > 0`).
    pub leftover_triangles: usize,
    /// Folded (non-convex) quads after the unfolder pass.
    pub inverted_left: usize,
    /// Unfolder sweeps used.
    pub unfold_rounds: usize,
    /// Local two-quad swaps taken to remove folded quads (§5.3 correction).
    pub quad_swaps: usize,
    /// Pairs of quads sharing two edges merged into one quad (§5.3 doublet fix).
    pub merged_quads: usize,
}

impl Stats {
    /// Share of the final triangles that ended in a quad. A doublet merge folds
    /// two quads into one, so this is one minus the leftover share.
    pub fn paired_fraction(&self) -> f64 {
        if self.triangles_final == 0 {
            0.0
        } else {
            1.0 - self.leftover_triangles as f64 / self.triangles_final as f64
        }
    }
}

/// Mesh the domain with the right-angled + bipartite-labelling route.
pub fn mesh(domain: &Domain, sizing: &Sizing) -> Result<Mesh, String> {
    mesh_with_stats(domain, sizing).map(|(mesh, _)| mesh)
}

/// Same as [`mesh`], returning the run's measurements as well.
pub fn mesh_with_stats(domain: &Domain, sizing: &Sizing) -> Result<(Mesh, Stats), String> {
    let mut stats = Stats::default();
    // §3 approximation: the paper's frontal, cross-field-guided point placement
    // is not implemented, so the interior is seeded with a field-aligned grid
    // (rows at the size spacing, along the domain's principal directions) and
    // the CDT fills the boundary band. Falling back to the crate's Ruppert
    // triangulation keeps the route total.
    let (mut points, mut tris) = match field_aligned_triangulation(domain, sizing) {
        Ok(result) => result,
        Err(_) => crate::two_d::triangulate(domain, sizing).map_err(|error| error.to_string())?,
    };
    stats.input_triangles = tris.len();
    // A zero-area triangle is not an element: the constrained triangulation can
    // leave collinear boundary samples behind (the ring's straight edge is a
    // measured case). Keeping them makes a degenerate boundary quad. Removing
    // them leaves the domain area unchanged and the same boundary polyline.
    let area_floor = 1.0e-10 * sizing.size * sizing.size;
    let before = tris.len();
    tris.retain(|t| tri_area2(&points, t).abs() > area_floor);
    stats.degenerate_triangles = before - tris.len();
    if tris.is_empty() {
        return Err("the triangulation is entirely degenerate".to_string());
    }

    // §3.1: an all-quad mesh needs an even number of edges on every boundary
    // cycle. `even_boundary_loops` splits the longest edge of each odd loop.
    let (even_points, even_tris, boundary_loop_splits) =
        parity::even_boundary_loops(&points, &tris);
    points = even_points;
    tris = even_tris;
    stats.boundary_loop_splits = boundary_loop_splits;
    normalise_winding(&points, &mut tris);

    // §4: push the triangulation towards right angles.
    stats.right_angles_before = count_right_angles(&points, &tris);
    stats.right_angle_flips = right_angle_flips(&points, &mut tris, FLIP_ROUNDS, sizing.size);
    stats.right_angles_after = count_right_angles(&points, &tris);

    let Some(loops) = boundary_loops(&tris) else {
        return Err("the triangulation has an open or non-manifold boundary".to_string());
    };
    if let Some(loop_lengths) = loops.iter().map(Vec::len).find(|&n| n % 2 == 1) {
        return Err(format!(
            "a boundary loop has {loop_lengths} edges (odd); an all-quad mesh cannot exist"
        ));
    }
    let owners = edge_owners(&tris);

    // §5.1: alternate the labels along the boundary, then choose the interior.
    let (mut colour, mut fixed) = label(&points, &tris, &loops, &owners);
    stats.label_flips = smooth_labels(&mut colour, &fixed, &points, &tris, &owners);
    stats.mono_initial = count_mono(&tris, &colour);

    // Build the quads, then try to repair the labelling locally if the pairing
    // leaves a topologically invalid cell (a fold or a two-edge overlap).
    let mut assembled = assemble(
        &mut points,
        &mut tris,
        &mut colour,
        &mut fixed,
        sizing.size,
        domain,
    );
    let mut retry_flips = 0usize;
    if assembled.inverted > 0 || assembled.overlaps > 0 {
        let candidates = fold_neighbourhood(&points, &assembled.quads);
        // Evaluate every single and pair flip around the folded quads and keep
        // the one whose worst quad is best, provided it clears every inversion
        // and overlap. Requiring the first improving attempt would stop at a
        // merely flat boundary quad.
        let mut best: Option<(f64, usize, Vec<Node>, Vec<[usize; 3]>, Vec<u8>, Assembled)> = None;
        let mut try_flip = |flips: &[usize]| -> Option<()> {
            let mut trial_points = points.clone();
            let mut trial_tris = tris.clone();
            let mut trial_colour = colour.clone();
            let mut trial_fixed = fixed.clone();
            for &v in flips {
                trial_colour[v] = 1 - trial_colour[v];
            }
            let attempt = assemble(
                &mut trial_points,
                &mut trial_tris,
                &mut trial_colour,
                &mut trial_fixed,
                sizing.size,
                domain,
            );
            if attempt.inverted == 0 && attempt.overlaps == 0 {
                let better = best
                    .as_ref()
                    .map(|(sj, _, _, _, _, _)| attempt.min_sj > *sj + 1.0e-9)
                    .unwrap_or(true);
                if better {
                    best = Some((
                        attempt.min_sj,
                        flips.len(),
                        trial_points,
                        trial_tris,
                        trial_colour,
                        attempt,
                    ));
                }
            }
            Some(())
        };
        for &v in &candidates {
            if fixed.get(v).copied().unwrap_or(true) {
                continue;
            }
            let _ = try_flip(&[v]);
        }
        for i in 0..candidates.len() {
            for j in (i + 1)..candidates.len() {
                let (v, w) = (candidates[i], candidates[j]);
                if fixed.get(v).copied().unwrap_or(true) || fixed.get(w).copied().unwrap_or(true) {
                    continue;
                }
                let _ = try_flip(&[v, w]);
            }
        }
        if let Some((_, flips, trial_points, trial_tris, trial_colour, attempt)) = best {
            points = trial_points;
            tris = trial_tris;
            colour = trial_colour;
            assembled = attempt;
            retry_flips = flips;
        }
    }
    stats.label_flips += assembled.repair_flips + retry_flips;
    stats.steiner_points = assembled.steiner;
    stats.triangles_final = tris.len();
    stats.mono_left = count_mono(&tris, &colour);
    stats.merged_quads = assembled.merged;
    stats.quad_swaps = assembled.swaps;
    stats.unfold_rounds = assembled.unfold_rounds;
    stats.inverted_left = assembled.inverted;
    stats.leftover_triangles = assembled.leftover.len();
    stats.quads = assembled.quads.len();

    let mut quads = assembled.quads;
    let mut leftover = assembled.leftover;
    let points = compact(points, &mut quads, &mut leftover);
    stats.quads = quads.len();
    stats.leftover_triangles = leftover.len();

    let mut cells: Vec<Cell> = quads.iter().map(|&q| Cell::quad4(q)).collect();
    cells.extend(leftover.iter().map(|&t| Cell::tri3(t)));
    let mesh = Mesh::new(points, cells);
    mesh.validate()
        .map_err(|error| format!("internal mesh error: {error}"))?;
    Ok((mesh, stats))
}

/// Result of the post-labelling half of the route.
struct Assembled {
    quads: Vec<[usize; 4]>,
    leftover: Vec<[usize; 3]>,
    steiner: usize,
    repair_flips: usize,
    merged: usize,
    swaps: usize,
    unfold_rounds: usize,
    inverted: usize,
    overlaps: usize,
    min_sj: f64,
}

/// Repair the monochromatic triangles, pair, and apply the local corrections.
fn assemble(
    points: &mut Vec<Node>,
    tris: &mut Vec<[usize; 3]>,
    colour: &mut Vec<u8>,
    fixed: &mut Vec<bool>,
    size: f64,
    domain: &Domain,
) -> Assembled {
    let (steiner, repair_flips) = repair_monochromatic(points, tris, colour, fixed);
    let owners = edge_owners(tris);
    let (mut quads, mut leftover, _) = pair_triangles(points, tris, colour, &owners);
    let merged = merge_doublets(points, &mut quads);
    let (_, mut unfold_rounds) = unfold(points, &quads, &leftover, fixed, size, Some(domain));
    let (p, q, t, swaps) =
        crate::sun::repair_degenerate_quads(points, &quads, &leftover, 0.05, 256);
    *points = p;
    quads = q;
    leftover = t;
    let lowest = quads
        .iter()
        .map(|quad| quad_min_sj(points, quad))
        .fold(f64::INFINITY, f64::min);
    if lowest < 0.2 {
        // §5.3: a targeted topological pass for the few flat quads left on the
        // boundary. `improve_quality` only accepts a re-split when both new
        // quads keep SJ >= 0.2.
        let (p, q, t, _) = crate::sun::improve_quality(points, &quads, &leftover, 8);
        *points = p;
        quads = q;
        leftover = t;
    }
    reposition(
        points,
        &quads,
        &leftover,
        fixed,
        REPOSITION_ROUNDS,
        size,
        Some(domain),
    );
    let (inverted, more_rounds) = unfold(points, &quads, &leftover, fixed, size, Some(domain));
    unfold_rounds += more_rounds;
    reposition(
        points,
        &quads,
        &leftover,
        fixed,
        REPOSITION_ROUNDS,
        size,
        Some(domain),
    );
    let overlaps = overlap_count(&quads);
    let min_sj = quads
        .iter()
        .map(|q| quad_min_sj(points, q))
        .fold(f64::INFINITY, f64::min);
    Assembled {
        quads,
        leftover,
        steiner,
        repair_flips,
        merged,
        swaps,
        unfold_rounds,
        inverted,
        overlaps,
        min_sj,
    }
}

/// Number of pairs of quads sharing more than one edge.
fn overlap_count(quads: &[[usize; 4]]) -> usize {
    let mut owners: HashMap<Edge, Vec<usize>> = HashMap::new();
    for (id, q) in quads.iter().enumerate() {
        for i in 0..4 {
            owners
                .entry(edge_key(q[i], q[(i + 1) % 4]))
                .or_default()
                .push(id);
        }
    }
    let mut shared: HashMap<(usize, usize), usize> = HashMap::new();
    for ids in owners.values() {
        for i in 0..ids.len() {
            for j in (i + 1)..ids.len() {
                let pair = (ids[i].min(ids[j]), ids[i].max(ids[j]));
                *shared.entry(pair).or_default() += 1;
            }
        }
    }
    shared.values().filter(|&&count| count > 1).count()
}

/// Free vertices of the folded quads and their quad-mesh neighbours, the
/// candidates for a corrective label flip.
fn fold_neighbourhood(points: &[Node], quads: &[[usize; 4]]) -> Vec<usize> {
    // A vertex is on the boundary when it belongs to an edge used once.
    let mut edge_uses: HashMap<Edge, usize> = HashMap::new();
    for q in quads {
        for i in 0..4 {
            *edge_uses.entry(edge_key(q[i], q[(i + 1) % 4])).or_insert(0) += 1;
        }
    }
    let mut on_boundary: HashSet<usize> = HashSet::new();
    for (&(a, b), &uses) in &edge_uses {
        if uses == 1 {
            on_boundary.insert(a);
            on_boundary.insert(b);
        }
    }
    let mut candidates: Vec<usize> = Vec::new();
    for q in quads {
        if quad_min_sj(points, q) > 0.0 {
            continue;
        }
        for &v in q {
            if !on_boundary.contains(&v) && !candidates.contains(&v) {
                candidates.push(v);
            }
        }
    }
    let base = candidates.clone();
    for v in base {
        for q in quads {
            if !q.contains(&v) {
                continue;
            }
            for &u in q {
                if u != v && !on_boundary.contains(&u) && !candidates.contains(&u) {
                    candidates.push(u);
                }
            }
        }
    }
    candidates
}

/// Drop points no cell uses and renumber the cells.
fn compact(points: Vec<Node>, quads: &mut [[usize; 4]], leftover: &mut [[usize; 3]]) -> Vec<Node> {
    let mut used = vec![false; points.len()];
    for q in quads.iter() {
        for &v in q {
            used[v] = true;
        }
    }
    for t in leftover.iter() {
        for &v in t {
            used[v] = true;
        }
    }
    let mut remap = vec![usize::MAX; points.len()];
    let mut out = Vec::new();
    for (i, &keep) in used.iter().enumerate() {
        if keep {
            remap[i] = out.len();
            out.push(points[i]);
        }
    }
    for q in quads.iter_mut() {
        for v in q.iter_mut() {
            *v = remap[*v];
        }
    }
    for t in leftover.iter_mut() {
        for v in t.iter_mut() {
            *v = remap[*v];
        }
    }
    out
}

// --------------------------------------------------------------------------- //
// field-aligned input triangulation (§3 approximation)
// --------------------------------------------------------------------------- //

/// Build a CDT whose interior is seeded with an axis-aligned grid of points at
/// the requested size.
///
/// The paper gets a grid-like right-angled triangulation from its frontal
/// point generation guided by a cross-field. Without the cross-field we use the
/// domain's axes as the field: points on a `size` lattice, at least
/// `0.65 * size` from the boundary, plus the boundary samples themselves. The
/// constrained Delaunay triangulation then only has to fill a one-cell boundary
/// band, which keeps most interior vertices on a square lattice — where the
/// labelling can pair the triangles into squares.
fn field_aligned_triangulation(
    domain: &Domain,
    sizing: &Sizing,
) -> Result<(Vec<Node>, Vec<[usize; 3]>), String> {
    use spade::{ConstrainedDelaunayTriangulation, Point2, Triangulation};

    let size = sizing.size;
    if !size.is_finite() || size <= 0.0 {
        return Err("bad size".to_string());
    }
    let mut cdt = ConstrainedDelaunayTriangulation::<Point2<f64>>::new();

    // Boundary samples, at most 1.1 * size apart, then their constraints.
    let mut loops: Vec<Vec<[f64; 2]>> = Vec::new();
    for loop_points in std::iter::once(&domain.outer).chain(domain.holes.iter()) {
        let mut sampled = Vec::new();
        for i in 0..loop_points.len() {
            let a = loop_points[i];
            let b = loop_points[(i + 1) % loop_points.len()];
            let length = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
            let parts = (length / (1.1 * size)).ceil().max(1.0) as usize;
            for k in 0..parts {
                let t = k as f64 / parts as f64;
                sampled.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
            }
        }
        loops.push(sampled);
    }
    for cycle in &loops {
        for &[x, y] in cycle {
            cdt.insert(Point2::new(x, y))
                .map_err(|_| "a boundary point could not be inserted".to_string())?;
        }
        for i in 0..cycle.len() {
            let a = cycle[i];
            let b = cycle[(i + 1) % cycle.len()];
            cdt.add_constraint_edge(Point2::new(a[0], a[1]), Point2::new(b[0], b[1]))
                .map_err(|_| "a boundary edge could not be constrained".to_string())?;
        }
    }

    // One row following the boundary at ~0.8 size inside it. This is the piece
    // of the paper's frontal boundary layer that keeps boundary quads from
    // carrying two boundary edges (which makes them flat on a curved boundary).
    let mut ring_points: Vec<[f64; 2]> = Vec::new();
    let mut all_ring: Vec<[f64; 2]> = loops.iter().flatten().copied().collect();
    let offset = 0.8 * size;
    let probe = 0.5 * size;
    for cycle in &loops {
        let n = cycle.len();
        if n < 3 {
            continue;
        }
        for i in 0..n {
            let a = cycle[(i + n - 1) % n];
            let b = cycle[i];
            let c = cycle[(i + 1) % n];
            let tangent = [c[0] - a[0], c[1] - a[1]];
            let length = (tangent[0] * tangent[0] + tangent[1] * tangent[1]).sqrt();
            if length <= 0.0 {
                continue;
            }
            let mut normal = [-tangent[1] / length, tangent[0] / length];
            let plus = [b[0] + probe * normal[0], b[1] + probe * normal[1]];
            if !domain.contains(plus) {
                normal = [-normal[0], -normal[1]];
            }
            let p = [b[0] + offset * normal[0], b[1] + offset * normal[1]];
            if !domain.contains(p) || domain.distance_to_boundary(p) < 0.45 * size {
                continue;
            }
            let too_close = all_ring.iter().any(|q| {
                (q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2) < (0.5 * size) * (0.5 * size)
            });
            if too_close {
                continue;
            }
            if cdt.insert(Point2::new(p[0], p[1])).is_ok() {
                ring_points.push(p);
                all_ring.push(p);
            }
        }
    }

    // Axis-aligned lattice in the interior.
    let (mut min_x, mut max_x, mut min_y, mut max_y) = (
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    );
    for p in &domain.outer {
        min_x = min_x.min(p[0]);
        max_x = max_x.max(p[0]);
        min_y = min_y.min(p[1]);
        max_y = max_y.max(p[1]);
    }
    let boundary_points: Vec<[f64; 2]> = all_ring;
    let clearance = 1.7 * size;
    let exclusion = (0.5 * size) * (0.5 * size);
    let mut x = (min_x / size).ceil() * size;
    while x < max_x {
        let mut y = (min_y / size).ceil() * size;
        while y < max_y {
            let p = [x, y];
            if domain.contains(p) && domain.distance_to_boundary(p) > clearance {
                let near = boundary_points
                    .iter()
                    .any(|q| (q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2) < exclusion);
                if !near {
                    let _ = cdt.insert(Point2::new(x, y));
                }
            }
            y += size;
        }
        x += size;
    }

    // Keep the inner faces whose centroid is inside the domain (the hole loops
    // are constraints, so faces inside a hole must be discarded).
    let mut points: Vec<Node> = Vec::new();
    let mut index_of: HashMap<(u64, u64), usize> = HashMap::new();
    let mut tris: Vec<[usize; 3]> = Vec::new();
    for face in cdt.inner_faces() {
        let vertices = face.vertices();
        let mut tri = [[0.0f64; 2]; 3];
        for (slot, vertex) in vertices.iter().enumerate() {
            let p = vertex.position();
            tri[slot] = [p.x, p.y];
        }
        let centroid = [
            (tri[0][0] + tri[1][0] + tri[2][0]) / 3.0,
            (tri[0][1] + tri[1][1] + tri[2][1]) / 3.0,
        ];
        if !domain.contains(centroid) {
            continue;
        }
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
    if tris.is_empty() {
        return Err("the field-aligned triangulation is empty".to_string());
    }
    Ok((points, tris))
}

// --------------------------------------------------------------------------- //
// small geometry helpers
// --------------------------------------------------------------------------- //

#[inline]
fn edge_key(a: usize, b: usize) -> Edge {
    (a.min(b), a.max(b))
}

fn tri_area2(points: &[Node], t: &[usize; 3]) -> f64 {
    let a = points[t[0]];
    let b = points[t[1]];
    let c = points[t[2]];
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// Make every triangle counter-clockwise, so "left of an edge" is uniform.
fn normalise_winding(points: &[Node], tris: &mut [[usize; 3]]) {
    for t in tris.iter_mut() {
        if tri_area2(points, t) < 0.0 {
            t.swap(1, 2);
        }
    }
}

/// Corner angles of a triangle, in degrees, in the order of its vertices.
fn triangle_angles(points: &[Node], t: &[usize; 3]) -> [f64; 3] {
    let mut angles = [0.0f64; 3];
    for i in 0..3 {
        let cur = points[t[i]];
        let next = points[t[(i + 1) % 3]];
        let prev = points[t[(i + 2) % 3]];
        let u = [next[0] - cur[0], next[1] - cur[1]];
        let v = [prev[0] - cur[0], prev[1] - cur[1]];
        let denom = (u[0] * u[0] + u[1] * u[1]).sqrt() * (v[0] * v[0] + v[1] * v[1]).sqrt();
        angles[i] = if denom > 0.0 {
            ((u[0] * v[0] + u[1] * v[1]) / denom)
                .clamp(-1.0, 1.0)
                .acos()
                .to_degrees()
        } else {
            0.0
        };
    }
    angles
}

fn min_corner_angle(points: &[Node], t: &[usize; 3]) -> f64 {
    triangle_angles(points, t)
        .into_iter()
        .fold(f64::INFINITY, f64::min)
}

fn count_right(angles: &[f64; 3]) -> usize {
    angles
        .iter()
        .filter(|&&a| (a - 90.0).abs() <= RIGHT_ANGLE_TOLERANCE_DEG)
        .count()
}

fn count_right_angles(points: &[Node], tris: &[[usize; 3]]) -> usize {
    tris.iter()
        .map(|t| count_right(&triangle_angles(points, t)))
        .sum()
}

/// Smallest corner Jacobian of a planar quadrilateral, the same quantity as
/// `quality::scaled_jacobian` for a `Quad4` (negative when non-convex).
fn quad_min_sj(points: &[Node], q: &[usize; 4]) -> f64 {
    let mut worst = f64::INFINITY;
    for i in 0..4 {
        let cur = points[q[i]];
        let next = points[q[(i + 1) % 4]];
        let prev = points[q[(i + 3) % 4]];
        let u = [next[0] - cur[0], next[1] - cur[1]];
        let v = [prev[0] - cur[0], prev[1] - cur[1]];
        let scale = (u[0] * u[0] + u[1] * u[1]).sqrt() * (v[0] * v[0] + v[1] * v[1]).sqrt();
        if scale <= 0.0 || !scale.is_finite() {
            return f64::NEG_INFINITY;
        }
        let sj = (u[0] * v[1] - u[1] * v[0]) / scale;
        worst = worst.min(sj);
    }
    worst
}

fn quad_area2(points: &[Node], q: &[usize; 4]) -> f64 {
    let mut area = 0.0;
    for i in 0..4 {
        let a = points[q[i]];
        let b = points[q[(i + 1) % 4]];
        area += a[0] * b[1] - b[0] * a[1];
    }
    area * 0.5
}

/// True when the four points form a strictly convex, positively oriented quad.
fn strictly_convex(points: &[Node], q: &[usize; 4]) -> bool {
    if quad_area2(points, q) <= 0.0 {
        return false;
    }
    for i in 0..4 {
        let a = points[q[i]];
        let b = points[q[(i + 1) % 4]];
        let c = points[q[(i + 2) % 4]];
        let cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
        if cross <= 1.0e-12 {
            return false;
        }
    }
    true
}

fn longest_edge(points: &[Node], t: &[usize; 3]) -> Edge {
    let mut best = edge_key(t[0], t[1]);
    let mut best_length = squared_length(points, best);
    for i in 1..3 {
        let edge = edge_key(t[i], t[(i + 1) % 3]);
        let length = squared_length(points, edge);
        if length > best_length {
            best = edge;
            best_length = length;
        }
    }
    best
}

fn squared_length(points: &[Node], edge: Edge) -> f64 {
    let a = points[edge.0];
    let b = points[edge.1];
    (b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)
}

// --------------------------------------------------------------------------- //
// connectivity
// --------------------------------------------------------------------------- //

fn edge_owners(tris: &[[usize; 3]]) -> HashMap<Edge, Vec<usize>> {
    let mut map: HashMap<Edge, Vec<usize>> = HashMap::new();
    for (id, t) in tris.iter().enumerate() {
        for i in 0..3 {
            map.entry(edge_key(t[i], t[(i + 1) % 3]))
                .or_default()
                .push(id);
        }
    }
    map
}

fn triangles_by_vertex(n: usize, tris: &[[usize; 3]]) -> Vec<Vec<usize>> {
    let mut map: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (id, t) in tris.iter().enumerate() {
        for &v in t {
            map[v].push(id);
        }
    }
    map
}

/// The boundary cycles as vertex loops, or `None` when the boundary is not a
/// union of simple 2-regular cycles.
fn boundary_loops(tris: &[[usize; 3]]) -> Option<Vec<Vec<usize>>> {
    let mut adjacency: HashMap<usize, Vec<usize>> = HashMap::new();
    for (edge, ids) in edge_owners(tris) {
        if ids.len() != 1 {
            continue;
        }
        adjacency.entry(edge.0).or_default().push(edge.1);
        adjacency.entry(edge.1).or_default().push(edge.0);
    }
    if adjacency.values().any(|neighbours| neighbours.len() != 2) {
        return None;
    }
    let mut used: HashSet<usize> = HashSet::new();
    let mut starts: Vec<usize> = adjacency.keys().copied().collect();
    starts.sort_unstable();
    let mut loops = Vec::new();
    for start in starts {
        if used.contains(&start) {
            continue;
        }
        let mut cycle = Vec::new();
        let mut previous = usize::MAX;
        let mut current = start;
        loop {
            used.insert(current);
            cycle.push(current);
            let neighbours = adjacency.get(&current)?;
            let next = if neighbours[0] == previous {
                neighbours[1]
            } else {
                neighbours[0]
            };
            previous = current;
            current = next;
            if current == start {
                break;
            }
            if cycle.len() > adjacency.len() {
                return None;
            }
        }
        loops.push(cycle);
    }
    Some(loops)
}

// --------------------------------------------------------------------------- //
// labelling
// --------------------------------------------------------------------------- //

/// Penalty for a monochromatic edge that pairs two boundary triangles into a
/// quad carrying two boundary edges (three boundary vertices). On a curved
/// boundary such a quad has a corner of ~180° at the middle vertex: it is flat
/// by construction, so no vertex move can raise its scaled Jacobian.
const BOUNDARY_FLAT_PENALTY: f64 = 1.0e2;

/// Weight of the valence objective. The quad valence of an interior vertex `u`
/// is `deg(u) - same_degree(u)` (every monochromatic edge at `u` merges two of
/// its triangles into one quad), so `same_degree = deg - 4` is exactly a
/// valence-4 node. Rewarding that target directly is what lifts the share of
/// regular interior vertices; it is kept small next to the non-convex penalty.
const VALENCE_WEIGHT: f64 = 1.5;

/// The unique monochromatic edge of a non-monochromatic triangle, in the
/// triangle's winding order, with its opposite vertex.
fn mono_edge(t: &[usize; 3], colour: &[u8]) -> Option<(usize, usize, usize)> {
    let labels = [colour[t[0]], colour[t[1]], colour[t[2]]];
    let mut mono = None;
    for i in 0..3 {
        let j = (i + 1) % 3;
        if labels[i] == labels[j] {
            mono = Some((t[i], t[j], t[(i + 2) % 3]));
        }
    }
    mono
}

/// Connectivity plus the labelling score. Everything the greedy labelling and
/// the smoothing need is precomputed once, so a label flip is a handful of
/// neighbour lookups.
struct Labeller<'a> {
    points: &'a [Node],
    tris: &'a [[usize; 3]],
    owners: &'a HashMap<Edge, Vec<usize>>,
    fixed: Vec<bool>,
    by_vertex: Vec<Vec<usize>>,
    neighbours: Vec<Vec<usize>>,
}

impl<'a> Labeller<'a> {
    fn boundary_edge(&self, a: usize, b: usize) -> bool {
        self.owners
            .get(&edge_key(a, b))
            .map(|ids| ids.len() == 1)
            .unwrap_or(false)
    }

    fn new(
        points: &'a [Node],
        tris: &'a [[usize; 3]],
        owners: &'a HashMap<Edge, Vec<usize>>,
        fixed: &[bool],
    ) -> Self {
        let n = points.len();
        let by_vertex = triangles_by_vertex(n, tris);
        let mut neighbours: Vec<Vec<usize>> = vec![Vec::new(); n];
        for t in tris {
            for i in 0..3 {
                let (a, b) = (t[i], t[(i + 1) % 3]);
                neighbours[a].push(b);
                neighbours[b].push(a);
            }
        }
        for list in neighbours.iter_mut() {
            list.sort_unstable();
            list.dedup();
        }
        let mut fixed = fixed.to_vec();
        fixed.resize(n, false);
        Labeller {
            points,
            tris,
            owners,
            fixed,
            by_vertex,
            neighbours,
        }
    }

    /// Score of one triangle under the current labelling: heavily negative when
    /// monochromatic, otherwise the scaled Jacobian of the quad formed across its
    /// unique monochromatic edge, plus a bonus for pairing along the longest edge
    /// (the edge that guarantees a convex quad when both triangles pick it).
    fn triangle_score(&self, tid: usize, colour: &[u8]) -> f64 {
        let t = self.tris[tid];
        if colour[t[0]] == FREE || colour[t[1]] == FREE || colour[t[2]] == FREE {
            return 0.0;
        }
        if colour[t[0]] == colour[t[1]] && colour[t[1]] == colour[t[2]] {
            return -MONO_PENALTY;
        }
        let Some((u, v, w)) = mono_edge(&t, colour) else {
            return 0.0;
        };
        let Some(ids) = self.owners.get(&edge_key(u, v)) else {
            return -10.0;
        };
        let mut other = None;
        for &id in ids {
            if id != tid {
                other = Some(id);
            }
        }
        let Some(other) = other else {
            return -10.0;
        };
        let neighbour = self.tris[other];
        let mut opposite = None;
        for &x in &neighbour {
            if x != u && x != v {
                opposite = Some(x);
            }
        }
        let Some(d) = opposite else {
            return -10.0;
        };
        let quad = [w, u, d, v];
        let sj = quad_min_sj(self.points, &quad);
        let mut score = sj;
        if sj <= 0.0 {
            score -= NONCONVEX_PENALTY;
        }
        if edge_key(u, v) == longest_edge(self.points, &t) {
            score += LONGEST_BONUS;
        }
        let triangle_boundary = self.boundary_edge(w, u) || self.boundary_edge(w, v);
        let neighbour_boundary = self.boundary_edge(d, v) || self.boundary_edge(d, u);
        let mono_on_boundary_vertex = self.fixed.get(u).copied().unwrap_or(false)
            || self.fixed.get(v).copied().unwrap_or(false);
        if triangle_boundary && neighbour_boundary && mono_on_boundary_vertex {
            score -= BOUNDARY_FLAT_PENALTY;
        }
        score
    }

    /// Penalty for deviating from the valence-4 target of an interior vertex.
    fn valence_term(&self, u: usize, colour: &[u8]) -> f64 {
        if self.fixed.get(u).copied().unwrap_or(true) || colour[u] == FREE {
            return 0.0;
        }
        let mut same = 0usize;
        for &w in &self.neighbours[u] {
            if colour[w] == colour[u] {
                same += 1;
            }
        }
        let target = self.neighbours[u].len().saturating_sub(4);
        -VALENCE_WEIGHT * (same as f64 - target as f64).abs()
    }

    /// Score affected by the label of `v`: its incident triangles and the valence
    /// terms of `v` and its neighbours (the only ones a flip of `v` changes).
    fn local_score(&self, v: usize, colour: &[u8]) -> f64 {
        let mut score: f64 = self.by_vertex[v]
            .iter()
            .map(|&tid| self.triangle_score(tid, colour))
            .sum();
        score += self.valence_term(v, colour);
        for &u in &self.neighbours[v] {
            score += self.valence_term(u, colour);
        }
        score
    }
}

/// Colour the boundary alternately and then grow the labels inwards, choosing at
/// each vertex the label that scores better against the already coloured ring.
fn label(
    points: &[Node],
    tris: &[[usize; 3]],
    loops: &[Vec<usize>],
    owners: &HashMap<Edge, Vec<usize>>,
) -> (Vec<u8>, Vec<bool>) {
    let n = points.len();
    let mut colour = vec![FREE; n];
    let mut fixed = vec![false; n];
    for cycle in loops {
        for (i, &v) in cycle.iter().enumerate() {
            colour[v] = (i % 2) as u8;
            fixed[v] = true;
        }
    }
    let labeller = Labeller::new(points, tris, owners, &fixed);
    let mut queue: VecDeque<usize> = (0..n).filter(|&v| colour[v] != FREE).collect();
    while let Some(u) = queue.pop_front() {
        for w in labeller.neighbours[u].clone() {
            if colour[w] != FREE {
                continue;
            }
            let mut best = 0u8;
            let mut best_score = f64::NEG_INFINITY;
            for candidate in [0u8, 1u8] {
                colour[w] = candidate;
                let score = labeller.local_score(w, &colour);
                if score > best_score + 1.0e-12 {
                    best_score = score;
                    best = candidate;
                }
            }
            colour[w] = best;
            queue.push_back(w);
        }
    }
    for label in colour.iter_mut() {
        if *label == FREE {
            *label = 0;
        }
    }
    (colour, fixed)
}

/// Greedy label smoothing: flip an interior label when that strictly raises the
/// local score. Boundary labels never move.
fn smooth_labels(
    colour: &mut [u8],
    fixed: &[bool],
    points: &[Node],
    tris: &[[usize; 3]],
    owners: &HashMap<Edge, Vec<usize>>,
) -> usize {
    let labeller = Labeller::new(points, tris, owners, fixed);
    let n = colour.len();
    let mut flips = 0usize;
    for _ in 0..LABEL_PASSES {
        let mut changed = 0usize;
        for v in 0..n {
            if labeller.fixed.get(v).copied().unwrap_or(false) {
                continue;
            }
            let old = colour[v];
            let current = labeller.local_score(v, colour);
            colour[v] = 1 - old;
            let candidate = labeller.local_score(v, colour);
            if candidate > current + 1.0e-9 {
                changed += 1;
                flips += 1;
            } else {
                colour[v] = old;
            }
        }
        if changed == 0 {
            break;
        }
    }
    flips
}

fn count_mono(tris: &[[usize; 3]], colour: &[u8]) -> usize {
    tris.iter()
        .filter(|t| {
            let first = colour[t[0]];
            first != FREE && first == colour[t[1]] && colour[t[1]] == colour[t[2]]
        })
        .count()
}

// --------------------------------------------------------------------------- //
// right-angle flips (§4, tractable core)
// --------------------------------------------------------------------------- //

/// Flip interior edges while that strictly raises the number of near-right
/// angles, without making the quadrilateral of the two triangles non-convex and
/// without dropping the smallest corner angle by more than `FLIP_ANGLE_DROP_DEG`.
fn right_angle_flips(points: &[Node], tris: &mut [[usize; 3]], rounds: usize, size: f64) -> usize {
    let mut total = 0usize;
    for _ in 0..rounds {
        let owners = edge_owners(tris);
        let mut edges: Vec<Edge> = owners
            .iter()
            .filter(|(_, ids)| ids.len() == 2)
            .map(|(&edge, _)| edge)
            .collect();
        edges.sort_unstable();
        let mut flipped = 0usize;
        for edge in edges {
            let Some(ids) = owners.get(&edge) else {
                continue;
            };
            if ids.len() != 2 {
                continue;
            }
            let (t1, t2) = (ids[0], ids[1]);
            if t1 == t2 {
                continue;
            }
            let (first, second) = (tris[t1], tris[t2]);
            // The edge must still be present in both triangles: an earlier flip
            // of this sweep may have replaced one of them.
            let Some((u, v, c)) = edge_with_opposite(&first, edge) else {
                continue;
            };
            let Some((_, _, d)) = edge_with_opposite(&second, edge) else {
                continue;
            };
            if c == d {
                continue;
            }
            let quad = [c, u, d, v];
            if !strictly_convex(points, &quad) {
                continue;
            }
            let repaired = [[c, u, d], [c, d, v]];
            // Never trade a right angle for a sliver: both new triangles must
            // keep a usable share of the size field's area.
            let area_floor = 1.0e-3 * size * size;
            if tri_area2(points, &repaired[0]).abs() < area_floor
                || tri_area2(points, &repaired[1]).abs() < area_floor
            {
                continue;
            }
            let old_angles = [
                triangle_angles(points, &first),
                triangle_angles(points, &second),
            ];
            let new_angles = [
                triangle_angles(points, &repaired[0]),
                triangle_angles(points, &repaired[1]),
            ];
            let old_right = count_right(&old_angles[0]) + count_right(&old_angles[1]);
            let new_right = count_right(&new_angles[0]) + count_right(&new_angles[1]);
            let new_floor = new_angles[0]
                .into_iter()
                .fold(f64::INFINITY, f64::min)
                .min(new_angles[1].into_iter().fold(f64::INFINITY, f64::min));
            let old_sum = old_angles[0].into_iter().fold(f64::INFINITY, f64::min)
                + old_angles[1].into_iter().fold(f64::INFINITY, f64::min);
            let new_sum = new_angles[0].into_iter().fold(f64::INFINITY, f64::min)
                + new_angles[1].into_iter().fold(f64::INFINITY, f64::min);
            let accept = if new_right > old_right {
                new_floor >= FLIP_MIN_CORNER_DEG && new_sum >= old_sum - FLIP_ANGLE_DROP_DEG
            } else {
                new_right == old_right && new_sum > old_sum + 1.0e-9
            };
            if !accept {
                continue;
            }
            tris[t1] = repaired[0];
            tris[t2] = repaired[1];
            flipped += 1;
            total += 1;
            if total > 200_000 {
                return total;
            }
        }
        if flipped == 0 {
            break;
        }
    }
    total
}

/// The edge of `t` matching `edge`, in the triangle's winding order, plus the
/// opposite vertex.
fn edge_with_opposite(t: &[usize; 3], edge: Edge) -> Option<(usize, usize, usize)> {
    for i in 0..3 {
        let a = t[i];
        let b = t[(i + 1) % 3];
        if edge_key(a, b) == edge {
            return Some((a, b, t[(i + 2) % 3]));
        }
    }
    None
}

// --------------------------------------------------------------------------- //
// monochromatic-triangle repair (§5.1, Fig. 9)
// --------------------------------------------------------------------------- //

/// Split one edge of every remaining monochromatic triangle and give the
/// midpoint the opposite label. A pair of adjacent monochromatic triangles gets
/// its common edge split, an isolated one its longest edge. The new triangles
/// each carry exactly one monochromatic edge, so the construction continues.
fn repair_monochromatic(
    points: &mut Vec<Node>,
    tris: &mut Vec<[usize; 3]>,
    colour: &mut Vec<u8>,
    fixed: &mut Vec<bool>,
) -> (usize, usize) {
    let split_cap = 32 + (tris.len() as f64 * STEINER_FRACTION) as usize;
    let mut steiner = 0usize;
    let mut label_flips = 0usize;
    for _ in 0..REPAIR_ROUNDS {
        let owners = edge_owners(tris);
        let mono: Vec<usize> = (0..tris.len())
            .filter(|&id| {
                let t = tris[id];
                colour[t[0]] != FREE && colour[t[0]] == colour[t[1]] && colour[t[1]] == colour[t[2]]
            })
            .collect();
        if mono.is_empty() {
            break;
        }
        let mono_set: HashSet<usize> = mono.iter().copied().collect();
        let mut chosen: BTreeMap<Edge, usize> = BTreeMap::new();
        let mut added = 0usize;
        for &tid in &mono {
            if steiner + added >= split_cap {
                break;
            }
            let t = tris[tid];
            let mut shared_mono = None;
            for i in 0..3 {
                let edge = edge_key(t[i], t[(i + 1) % 3]);
                let touches_mono = owners
                    .get(&edge)
                    .map(|ids| {
                        ids.iter()
                            .any(|&other| other != tid && mono_set.contains(&other))
                    })
                    .unwrap_or(false);
                if touches_mono {
                    shared_mono = Some(edge);
                    break;
                }
            }
            let edge = shared_mono.unwrap_or_else(|| longest_edge(points, &t));
            if chosen.contains_key(&edge) {
                continue;
            }
            let (a, b) = edge;
            if colour[a] == FREE || colour[a] != colour[b] {
                continue;
            }
            let midpoint = [
                (points[a][0] + points[b][0]) * 0.5,
                (points[a][1] + points[b][1]) * 0.5,
                0.0,
            ];
            points.push(midpoint);
            colour.push(1 - colour[a]);
            fixed.push(false);
            chosen.insert(edge, points.len() - 1);
            added += 1;
        }
        if chosen.is_empty() {
            break;
        }
        steiner += added;
        let mut rebuilt: Vec<[usize; 3]> = Vec::with_capacity(tris.len() + 2 * chosen.len());
        for &t in tris.iter() {
            split_triangle(t, &chosen, &mut rebuilt);
        }
        *tris = rebuilt;
        let owners = edge_owners(tris);
        label_flips += smooth_labels(colour, fixed, points, tris, &owners);
    }
    (steiner, label_flips)
}

/// Replace `t` by the pieces obtained by splitting any of its edges found in
/// `split`, always preserving the winding.
fn split_triangle(t: [usize; 3], split: &BTreeMap<Edge, usize>, out: &mut Vec<[usize; 3]>) {
    let mut stack = vec![t];
    while let Some(current) = stack.pop() {
        let mut did_split = false;
        for i in 0..3 {
            let a = current[i];
            let b = current[(i + 1) % 3];
            let c = current[(i + 2) % 3];
            if let Some(&mid) = split.get(&edge_key(a, b)) {
                stack.push([a, mid, c]);
                stack.push([mid, b, c]);
                did_split = true;
                break;
            }
        }
        if !did_split {
            out.push(current);
        }
    }
}

// --------------------------------------------------------------------------- //
// quad extraction (§5.2)
// --------------------------------------------------------------------------- //

/// Remove the monochromatic edges: every triangle pairs across its unique
/// monochromatic edge. Monochromatic triangles (only when the repair budget ran
/// out) and their partners are returned as leftover triangles.
fn pair_triangles(
    points: &[Node],
    tris: &[[usize; 3]],
    colour: &[u8],
    owners: &HashMap<Edge, Vec<usize>>,
) -> (Vec<[usize; 4]>, Vec<[usize; 3]>, usize) {
    let mono: Vec<bool> = tris
        .iter()
        .map(|t| {
            let first = colour[t[0]];
            first != FREE && first == colour[t[1]] && colour[t[1]] == colour[t[2]]
        })
        .collect();
    let mono_count = mono.iter().filter(|&&flag| flag).count();
    let mut used = vec![false; tris.len()];
    let mut quads: Vec<[usize; 4]> = Vec::new();
    for tid in 0..tris.len() {
        if used[tid] || mono[tid] {
            continue;
        }
        let t = tris[tid];
        let labels = [colour[t[0]], colour[t[1]], colour[t[2]]];
        let mut mono_edge = None;
        for i in 0..3 {
            let j = (i + 1) % 3;
            if labels[i] == labels[j] {
                mono_edge = Some((t[i], t[j], t[(i + 2) % 3]));
            }
        }
        let Some((u, v, w)) = mono_edge else {
            continue;
        };
        let Some(ids) = owners.get(&edge_key(u, v)) else {
            continue;
        };
        let mut other = None;
        for &id in ids {
            if id != tid {
                other = Some(id);
            }
        }
        let Some(other) = other else {
            continue;
        };
        if used[other] || mono[other] {
            continue;
        }
        let neighbour = tris[other];
        let mut opposite = None;
        for &x in &neighbour {
            if x != u && x != v {
                opposite = Some(x);
            }
        }
        let Some(d) = opposite else {
            continue;
        };
        // The union of the two triangles is simple; keep the orientation that
        // makes the quad counter-clockwise. A non-convex quad is kept (and will
        // be reported as inverted by the gate) instead of silently turning the
        // pair back into triangles: the route's claim is the all-quad topology.
        let mut quad = [w, u, d, v];
        let area = quad_area2(points, &quad);
        if area < 0.0 {
            quad = [w, v, d, u];
        }
        quads.push(quad);
        used[tid] = true;
        used[other] = true;
    }
    let leftover: Vec<[usize; 3]> = tris
        .iter()
        .enumerate()
        .filter(|(id, _)| !used[*id])
        .map(|(_, t)| *t)
        .collect();
    (quads, leftover, mono_count)
}

// --------------------------------------------------------------------------- //
// vertex repositioning (§5.2)
// --------------------------------------------------------------------------- //

/// Move interior vertices to the centroid of their stencil while that raises the
/// smallest scaled Jacobian of the incident elements.
fn reposition(
    points: &mut [Node],
    quads: &[[usize; 4]],
    tris: &[[usize; 3]],
    fixed: &[bool],
    rounds: usize,
    size: f64,
    domain: Option<&Domain>,
) {
    let n = points.len();
    let mut incident: Vec<Vec<(bool, usize)>> = vec![Vec::new(); n];
    for (id, q) in quads.iter().enumerate() {
        for &v in q {
            incident[v].push((true, id));
        }
    }
    for (id, t) in tris.iter().enumerate() {
        for &v in t {
            incident[v].push((false, id));
        }
    }
    for _ in 0..rounds {
        let mut moved = 0usize;
        for v in 0..n {
            if fixed.get(v).copied().unwrap_or(false) || incident[v].is_empty() {
                continue;
            }
            let mut ring: Vec<usize> = Vec::new();
            for &(is_quad, id) in &incident[v] {
                let nodes: &[usize] = if is_quad { &quads[id] } else { &tris[id] };
                for &u in nodes {
                    if u != v && !ring.contains(&u) {
                        ring.push(u);
                    }
                }
            }
            if ring.len() < 3 {
                continue;
            }
            let centroid = [
                ring.iter().map(|&u| points[u][0]).sum::<f64>() / ring.len() as f64,
                ring.iter().map(|&u| points[u][1]).sum::<f64>() / ring.len() as f64,
            ];
            let current = points[v];
            let old = stencil_quality(&incident[v], quads, tris, points);
            if !old.is_finite() {
                continue;
            }
            // An interior node must not be pushed onto the boundary: on the ring
            // that produced a zero-area boundary quad.
            if let Some(domain) = domain {
                if !domain.contains(centroid)
                    || domain.distance_to_boundary(centroid) < 1.0e-3 * size
                {
                    continue;
                }
            }
            points[v] = [centroid[0], centroid[1], 0.0];
            let new = stencil_quality(&incident[v], quads, tris, points);
            let distinct = ring.iter().all(|&u| {
                (points[v][0] - points[u][0]).abs() + (points[v][1] - points[u][1]).abs() > 1.0e-12
            });
            if new > old + 1.0e-9 && new > 0.05 && distinct {
                moved += 1;
            } else {
                points[v] = current;
            }
        }
        if moved == 0 {
            break;
        }
    }
}

fn stencil_quality(
    incident: &[(bool, usize)],
    quads: &[[usize; 4]],
    tris: &[[usize; 3]],
    points: &[Node],
) -> f64 {
    let mut worst = f64::INFINITY;
    for &(is_quad, id) in incident {
        let value = if is_quad {
            quad_min_sj(points, &quads[id])
        } else {
            min_corner_angle(points, &tris[id]) / 90.0
        };
        worst = worst.min(value);
    }
    worst
}

/// Merge pairs of quads that share a path of two edges (§5.3 doublet fix).
///
/// Such a pair is a 4-triangle fan around a vertex `B`: the two quads
/// `[P, B, N, D]` and `[N, B, P, X]` share `P-B` and `B-N`. Their union is the
/// single quad `[P, X, N, D]`; merging them removes both folded quads and the
/// two-edge sharing, and leaves the surrounding quads untouched. The merge is
/// taken only when `B` belongs to no other quad (so it becomes an unused point,
/// compacted away later) and the merged quad is convex with a usable Jacobian.
fn merge_doublets(points: &[Node], quads: &mut Vec<[usize; 4]>) -> usize {
    let mut merged = 0usize;
    for _ in 0..64 {
        let mut owners: HashMap<Edge, Vec<usize>> = HashMap::new();
        let mut valence: HashMap<usize, usize> = HashMap::new();
        for (id, q) in quads.iter().enumerate() {
            for &v in q {
                *valence.entry(v).or_insert(0) += 1;
            }
            for i in 0..4 {
                owners
                    .entry(edge_key(q[i], q[(i + 1) % 4]))
                    .or_default()
                    .push(id);
            }
        }
        // pairs sharing two edges
        let mut shared: HashMap<(usize, usize), Vec<Edge>> = HashMap::new();
        for (&edge, ids) in &owners {
            if ids.len() != 2 {
                continue;
            }
            let pair = (ids[0].min(ids[1]), ids[0].max(ids[1]));
            shared.entry(pair).or_default().push(edge);
        }
        let mut done = false;
        let mut keys: Vec<(usize, usize)> = shared.keys().copied().collect();
        keys.sort_unstable();
        for (qa, qb) in keys {
            let Some(edges) = shared.get(&(qa, qb)) else {
                continue;
            };
            if edges.len() != 2 {
                continue;
            }
            let (e1, e2) = (edges[0], edges[1]);
            // common vertex B of the two shared edges, and the two outer ends
            let mut common = None;
            for candidate in [e1.0, e1.1] {
                if candidate == e2.0 || candidate == e2.1 {
                    common = Some(candidate);
                }
            }
            let Some(b) = common else {
                continue;
            };
            let p = if e1.0 == b { e1.1 } else { e1.0 };
            let n = if e2.0 == b { e2.1 } else { e2.0 };
            if b == p || b == n || p == n {
                continue;
            }
            if valence.get(&b).copied().unwrap_or(0) != 2 {
                continue;
            }
            let qa_quad = quads[qa];
            let Some(bi) = qa_quad.iter().position(|&v| v == b) else {
                continue;
            };
            // make sure the two shared edges are B's edges in qa
            let prev = qa_quad[(bi + 3) % 4];
            let next = qa_quad[(bi + 1) % 4];
            if !((prev == p && next == n) || (prev == n && next == p)) {
                continue;
            }
            let d = qa_quad[(bi + 2) % 4];
            let qb_quad = quads[qb];
            let Some(bj) = qb_quad.iter().position(|&v| v == b) else {
                continue;
            };
            let x = qb_quad[(bj + 2) % 4];
            if x == p || x == n || x == b || x == d {
                continue;
            }
            let merged_quad = [prev, x, next, d];
            if quad_area2(points, &merged_quad) <= 0.0 || quad_min_sj(points, &merged_quad) <= 0.05
            {
                continue;
            }
            quads[qa] = merged_quad;
            quads.swap_remove(qb);
            merged += 1;
            done = true;
            break;
        }
        if !done {
            break;
        }
    }
    merged
}

/// Number of non-convex quads.
fn count_folds(quads: &[[usize; 4]], points: &[Node]) -> usize {
    quads
        .iter()
        .filter(|q| quad_min_sj(points, q) <= 0.0)
        .count()
}

/// State of a vertex stencil: `(non-convex cells, non-convex corners, smallest
/// Jacobian, valid)`. The corner count lets the unfolder keep working when a
/// move fixes one fold but leaves a neighbouring cell just as folded.
fn stencil_state(
    incident: &[(bool, usize)],
    quads: &[[usize; 4]],
    tris: &[[usize; 3]],
    points: &[Node],
) -> (usize, usize, f64, bool) {
    let mut bad_cells = 0usize;
    let mut bad_corners = 0usize;
    let mut worst = f64::INFINITY;
    let mut valid = true;
    for &(is_quad, id) in incident {
        if is_quad {
            let area = quad_area2(points, &quads[id]);
            if area <= 1.0e-14 || !area.is_finite() {
                valid = false;
            }
            let sj = quad_min_sj(points, &quads[id]);
            if sj <= 0.0 {
                bad_cells += 1;
            }
            bad_corners += corners_of(points, &quads[id])
                .iter()
                .filter(|&&s| s <= 0.0)
                .count();
            worst = worst.min(sj);
        } else {
            let tri_area = tri_area2(points, &tris[id]);
            if tri_area <= 1.0e-14 || !tri_area.is_finite() {
                valid = false;
            }
            worst = worst.min(min_corner_angle(points, &tris[id]) / 90.0);
        }
    }
    (bad_cells, bad_corners, worst, valid)
}

/// The four corner Jacobians of a quad, matching `quad_min_sj`.
fn corners_of(points: &[Node], q: &[usize; 4]) -> [f64; 4] {
    let mut out = [0.0f64; 4];
    for i in 0..4 {
        let cur = points[q[i]];
        let next = points[q[(i + 1) % 4]];
        let prev = points[q[(i + 3) % 4]];
        let u = [next[0] - cur[0], next[1] - cur[1]];
        let v = [prev[0] - cur[0], prev[1] - cur[1]];
        let scale = (u[0] * u[0] + u[1] * u[1]).sqrt() * (v[0] * v[0] + v[1] * v[1]).sqrt();
        out[i] = if scale > 0.0 {
            (u[0] * v[1] - u[1] * v[0]) / scale
        } else {
            f64::NEG_INFINITY
        };
    }
    out
}

/// Mirror `x` across the line through `a` and `b`.
fn reflect_across(x: Node, a: Node, b: Node) -> [f64; 2] {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let length2 = ab[0] * ab[0] + ab[1] * ab[1];
    if length2 <= 0.0 {
        return [x[0], x[1]];
    }
    let t = ((x[0] - a[0]) * ab[0] + (x[1] - a[1]) * ab[1]) / length2;
    let foot = [a[0] + t * ab[0], a[1] + t * ab[1]];
    [2.0 * foot[0] - x[0], 2.0 * foot[1] - x[1]]
}

/// Candidate positions around an interior vertex: the ring centroid and a few
/// rings of points. The radii are fractions of the size field.
fn candidates_for(current: Node, centroid: [f64; 2], size: f64) -> Vec<[f64; 2]> {
    let mut out = vec![centroid];
    for radius in [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.35] {
        let r = radius * size;
        for k in 0..16 {
            let angle = std::f64::consts::TAU * k as f64 / 16.0;
            out.push([current[0] + r * angle.cos(), current[1] + r * angle.sin()]);
        }
    }
    out
}

/// True when moving `v` to `p` keeps every incident edge usable.
fn spacing_ok_local(
    v: usize,
    p: [f64; 2],
    incident: &[(bool, usize)],
    quads: &[[usize; 4]],
    tris: &[[usize; 3]],
    points: &[Node],
    size: f64,
) -> bool {
    let floor = 0.05 * size;
    for &(is_quad, id) in incident {
        let nodes: &[usize] = if is_quad { &quads[id] } else { &tris[id] };
        for i in 0..nodes.len() {
            let a = nodes[i];
            let b = nodes[(i + 1) % nodes.len()];
            if a != v && b != v {
                continue;
            }
            let pa = if a == v {
                p
            } else {
                [points[a][0], points[a][1]]
            };
            let pb = if b == v {
                p
            } else {
                [points[b][0], points[b][1]]
            };
            let length = ((pb[0] - pa[0]).powi(2) + (pb[1] - pa[1]).powi(2)).sqrt();
            if length < floor {
                return false;
            }
        }
    }
    true
}

/// Unfold the few non-convex quads by moving their free vertices (§5.3 treats
/// these defects locally; the paper repositions vertices for the same reason).
///
/// The objective is lexicographic: fewer non-convex cells first, then a larger
/// smallest scaled Jacobian. A move is only taken when it strictly improves that
/// state, so the pass terminates.
fn unfold(
    points: &mut [Node],
    quads: &[[usize; 4]],
    tris: &[[usize; 3]],
    fixed: &[bool],
    size: f64,
    domain: Option<&Domain>,
) -> (usize, usize) {
    let n = points.len();
    let mut incident: Vec<Vec<(bool, usize)>> = vec![Vec::new(); n];
    for (id, q) in quads.iter().enumerate() {
        for &v in q {
            incident[v].push((true, id));
        }
    }
    for (id, t) in tris.iter().enumerate() {
        for &v in t {
            incident[v].push((false, id));
        }
    }
    let mut folds = count_folds(quads, points);
    let mut rounds = 0usize;
    for _ in 0..60 {
        if folds == 0 {
            break;
        }
        let mut vertices: Vec<usize> = Vec::new();
        // Reflecting the reflex vertex across the chord of its two neighbours in
        // the folded quad is the direct way to unfold it; the circular search
        // alone only reaches a nearly flat local optimum.
        let mut extra: HashMap<usize, Vec<[f64; 2]>> = HashMap::new();
        for q in quads.iter() {
            if quad_min_sj(points, q) > 0.0 {
                continue;
            }
            let corners = corners_of(points, q);
            for i in 0..4 {
                let v = q[i];
                if fixed.get(v).copied().unwrap_or(true) {
                    continue;
                }
                if !vertices.contains(&v) {
                    vertices.push(v);
                }
                if corners[i] <= 0.0 {
                    let a = points[q[(i + 3) % 4]];
                    let b = points[q[(i + 1) % 4]];
                    let entry = extra.entry(v).or_default();
                    entry.push(reflect_across(points[v], a, b));
                    // A reflex corner is often almost on the chord of its two
                    // neighbours, where the mirror is the point itself. Small
                    // moves across that chord are what unfolds it.
                    let chord = [b[0] - a[0], b[1] - a[1]];
                    let length = (chord[0] * chord[0] + chord[1] * chord[1]).sqrt();
                    if length > 0.0 {
                        let normal = [-chord[1] / length, chord[0] / length];
                        for fraction in [0.005, 0.02, 0.05] {
                            let step = fraction * size;
                            entry.push([
                                points[v][0] + step * normal[0],
                                points[v][1] + step * normal[1],
                            ]);
                            entry.push([
                                points[v][0] - step * normal[0],
                                points[v][1] - step * normal[1],
                            ]);
                        }
                    }
                }
            }
        }
        if vertices.is_empty() {
            break;
        }
        vertices.sort_unstable();
        let mut moved = 0usize;
        for v in vertices {
            if incident[v].is_empty() {
                continue;
            }
            let mut ring: Vec<usize> = Vec::new();
            for &(is_quad, id) in &incident[v] {
                let nodes: &[usize] = if is_quad { &quads[id] } else { &tris[id] };
                for &u in nodes {
                    if u != v && !ring.contains(&u) {
                        ring.push(u);
                    }
                }
            }
            if ring.len() < 3 {
                continue;
            }
            let centroid = [
                ring.iter().map(|&u| points[u][0]).sum::<f64>() / ring.len() as f64,
                ring.iter().map(|&u| points[u][1]).sum::<f64>() / ring.len() as f64,
            ];
            let current = points[v];
            let (old_bad, old_corners, old_worst, old_valid) =
                stencil_state(&incident[v], quads, tris, points);
            if !old_valid {
                continue;
            }
            let mut best: Option<([f64; 2], usize, usize, f64)> = None;
            let circle = candidates_for(current, centroid, size);
            let reflected = extra.get(&v).cloned().unwrap_or_default();
            for candidate in circle.into_iter().chain(reflected) {
                if let Some(domain) = domain {
                    if !domain.contains(candidate)
                        || domain.distance_to_boundary(candidate) < 1.0e-3 * size
                    {
                        continue;
                    }
                }
                if !spacing_ok_local(v, candidate, &incident[v], quads, tris, points, size) {
                    continue;
                }
                points[v] = [candidate[0], candidate[1], 0.0];
                let (bad, corners, worst, valid) = stencil_state(&incident[v], quads, tris, points);
                points[v] = current;
                if !valid {
                    continue;
                }
                let better = (bad, corners) < (old_bad, old_corners)
                    || ((bad, corners) == (old_bad, old_corners) && worst > old_worst + 1.0e-9);
                if !better {
                    continue;
                }
                let replace = match best {
                    None => true,
                    Some((_, best_bad, best_corners, best_worst)) => {
                        if bad != best_bad {
                            bad < best_bad
                        } else if corners != best_corners {
                            corners < best_corners
                        } else {
                            worst > best_worst + 1.0e-9
                        }
                    }
                };
                if replace {
                    best = Some((candidate, bad, corners, worst));
                }
            }
            if let Some((candidate, _, _, _)) = best {
                points[v] = [candidate[0], candidate[1], 0.0];
                moved += 1;
            }
        }
        rounds += 1;
        folds = count_folds(quads, points);
        if moved == 0 {
            break;
        }
    }
    (folds, rounds)
}

// --------------------------------------------------------------------------- //
// tests
// --------------------------------------------------------------------------- //
#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::quality;
    use crate::verify;

    fn square() -> Domain {
        Domain::rectangle(10.0, 10.0).unwrap()
    }

    fn all_quads(mesh: &Mesh) -> bool {
        mesh.cells
            .iter()
            .all(|cell| cell.kind == crate::mesh::Kind::Quad4)
    }

    #[test]
    fn quad_min_sj_matches_the_quality_contract() {
        // A unit square, a 60/120 rhombus and a non-convex quad.
        let square = [
            [0.0f64, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let cell = Cell::quad4([0, 1, 2, 3]);
        let expected = quality::scaled_jacobian(&cell, &square)
            .into_iter()
            .fold(f64::INFINITY, f64::min);
        let got = quad_min_sj(&square, &[0, 1, 2, 3]);
        assert!((got - expected).abs() < 1.0e-12, "{got} vs {expected}");

        let rhombus = [
            [0.0f64, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.5, 0.8660254037844386, 0.0],
            [0.5, 0.8660254037844386, 0.0],
        ];
        let expected = quality::scaled_jacobian(&cell, &rhombus)
            .into_iter()
            .fold(f64::INFINITY, f64::min);
        let got = quad_min_sj(&rhombus, &[0, 1, 2, 3]);
        assert!((got - expected).abs() < 1.0e-12, "{got} vs {expected}");

        let bent = [
            [0.0f64, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.6, 0.4, 0.0],
        ];
        let got = quad_min_sj(&bent, &[0, 1, 2, 3]);
        assert!(
            got < 0.0,
            "a non-convex quad must have a negative SJ: {got}"
        );
    }

    #[test]
    fn a_square_becomes_one_or_more_quads() {
        let mesh = mesh(&square(), &Sizing::uniform(5.0).unwrap()).unwrap();
        assert!(!mesh.cells.is_empty());
        assert!(all_quads(&mesh), "{} triangle(s) left", mesh.cells.len());
        assert!(!mesh.cells.is_empty());
        let gate = verify::gate(&mesh);
        assert_eq!(gate.triangles, 0);
        assert_eq!(gate.cracks, 0, "{gate:?}");
        assert!(
            verify::boundary_offset(&square(), &mesh) < 1.0e-9,
            "boundary moved"
        );
    }

    #[test]
    fn a_plate_with_a_hole_becomes_only_quads() {
        let domain = Domain::plate_with_hole(30.0, 18.0, 4.0, 16).unwrap();
        let mesh = mesh(&domain, &Sizing::uniform(2.5).unwrap()).unwrap();
        assert!(all_quads(&mesh), "leftover triangles");
        let gate = verify::gate(&mesh);
        assert_eq!(gate.triangles, 0);
        assert_eq!(gate.cracks, 0, "{gate:?}");
        assert!(verify::boundary_offset(&domain, &mesh) < 1.0e-9);
    }

    #[test]
    fn an_l_bracket_becomes_only_quads_and_passes_the_gate() {
        let domain = Domain::new(
            vec![
                [0.0, 0.0],
                [8.0, 0.0],
                [8.0, 2.0],
                [2.0, 2.0],
                [2.0, 6.0],
                [0.0, 6.0],
            ],
            vec![],
        )
        .unwrap();
        let mesh = mesh(&domain, &Sizing::uniform(1.0).unwrap()).unwrap();
        assert!(all_quads(&mesh), "leftover triangles");
        let gate = verify::gate(&mesh);
        assert_eq!(gate.triangles, 0);
        assert!(gate.passes(), "{gate:?} {:?}", gate.reason());
        assert!(verify::boundary_offset(&domain, &mesh) < 1.0e-9);
    }

    #[test]
    fn a_bad_size_is_an_error_not_a_panic() {
        let sizing = Sizing {
            size: 0.0,
            all_quad: false,
        };
        assert!(mesh(&square(), &sizing).is_err());
    }

    #[test]
    fn stats_add_up() {
        let domain = Domain::plate_with_hole(20.0, 12.0, 3.0, 16).unwrap();
        let (mesh, stats) = mesh_with_stats(&domain, &Sizing::uniform(2.0).unwrap()).unwrap();
        assert!(all_quads(&mesh));
        // Every final triangle is paired; the doublet merge then folds pairs of
        // quads into one, so `quads * 2` can be smaller than the triangle count.
        assert_eq!(stats.leftover_triangles, 0);
        assert!(stats.quads * 2 <= stats.triangles_final);
        assert!((stats.paired_fraction() - 1.0).abs() < 1.0e-12);
        assert_eq!(stats.mono_left, 0);
        assert_eq!(stats.inverted_left, 0);
    }
}
