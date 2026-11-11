//! Two-stage conversion: Q-Morph rows, then Blossom matching on the remainder.
//!
//! Stage 1 is the Q-Morph front of Owen et al. 1999 (corpus `owen-1999`), run
//! with an **even-region guard**; stage 2 is the Blossom-Quad matching of
//! Remacle et al. 2012 (corpus `remacle-2012`), reused from
//! [`crate::two_d::pair_triangles_with`] with its own quality floor. This is an
//! experimental conversion route; the shipped one is
//! [`crate::two_d::mesh_2d`], which pairs the full triangulation.
//!
//! The hypothesis this module tests is in
//! `Pedroso-Mesher-Handover-Next`: the last triangles left by the ordinary
//! conversion survive because Q-Morph locks quads around a triangle before the
//! remaining region can be paired. So stage 1 runs the front under a guard that
//! forbids leaving an unconverted component of **odd** triangle count, and
//! stage 2 pairs the remainder with the same Blossom matching the shipped path
//! uses (`two_d::pair_triangles_with`), under its own quality guards.
//!
//! Why odd-size components and not "one connected region": measured on the six
//! parts, a connected-region guard stalls the front at 3 % coverage — any front
//! that splits a simply-connected region into two loops is refused, and on the
//! plate a row of quads cuts the annulus. The front *legitimately* splits into
//! several regions. What matching cannot survive is an **odd** region: a
//! triangle is an odd polygon, and one triangle out of every odd region must be
//! left over. So the guard is parity of each remaining component, which is also
//! what the gate ultimately measures (all-quad ⟺ every region pairs).
//!
//! The matching path is untouched: this is a conversion route, opt-in, and the
//! shipped `two_d::mesh_2d` still calls `pair_triangles` with its own guards.
//!
//! **Measured, 2026-10-22: this route does not beat the shipped one.** The guard
//! moves coverage by −3 % to +6 % (it is not monotone), and the remainder it
//! protects is a set of thin sliver components whose two-triangle unions are
//! non-convex, so the matcher pairs only 20–35 % of them. See the record for the
//! table.

use std::collections::{HashMap, HashSet};

use crate::mesh::Node;
use crate::qmorph::{self, FrontStop, Refusals};
use crate::two_d::{pair_triangles_with, PairGuard};

/// How many matching rounds are attempted on the remainder. A round either
/// pairs, or flips edges around the unpaired triangles and retries.
const MAX_MATCH_ROUNDS: usize = 8;

/// Statistics of a two-stage conversion.
///
/// Counts are snapshots of the stages, not a conservation law: `front_quads` and
/// `core_triangles` describe the state after stage 1, `matched_quads` and
/// `unpaired` the state after stage 2. When `rejected_merge` is set, the matched
/// quads are discarded and the stats still reflect the attempted merge, so
/// `matched_quads` is reset to zero and `core_triangles` is restored.
#[derive(Debug, Default, Clone)]
pub struct TwoStageStats {
    /// Quads placed by the Q-Morph front.
    pub front_quads: usize,

    /// Triangles left for the matching stage.
    pub core_triangles: usize,

    /// Components of the remainder before matching (all even).
    pub core_components: usize,

    /// Quads produced by matching.
    pub matched_quads: usize,

    /// Interior edge flips tried to unblock a matching round.
    pub flips: usize,

    /// Triangles that matching could not pair.
    pub unpaired: usize,

    /// Matching rounds that ran.
    pub rounds: usize,

    /// Set when the merge failed the manifold/area validation and the front-only
    /// state was returned instead.
    pub rejected_merge: bool,

    /// Remainder triangles whose stored winding was reversed. The front's
    /// recovery flip writes both windings (see `qmorph::CellMesh::flip`), and a
    /// matcher needs one; they are normalised before pairing, not dropped.
    pub inverted_core: usize,
}

/// The result of a two-stage conversion.
///
/// `quads` and `triangles` are the final cells; `points` is the front's point
/// set (matching never inserts a node), so every index in `quads`/`triangles`
/// is valid for `points`.
pub struct TwoStageResult {
    /// The front's node set, in the XY plane; includes any placement insertion.
    pub points: Vec<Node>,

    /// Front quads plus the quads the matcher committed, in that order.
    pub quads: Vec<[usize; 4]>,

    /// Triangles the matcher could not pair (empty on a complete conversion).
    pub triangles: Vec<[usize; 3]>,

    /// Stage counters; see [`TwoStageStats`].
    pub stats: TwoStageStats,

    /// Q-Morph refusal counters from stage 1.
    pub refusals: Refusals,
}

/// Convert a triangle mesh into quads: Q-Morph front, then matching.
///
/// `size` is the target edge length in millimetres. The front runs with
/// [`FrontStop::EvenComponents`], so every unconverted component it leaves has
/// an even triangle count; the matcher then pairs the remainder under the guard
/// `min SJ 0.05`, angles `[10, 160]` degrees. At most `MAX_MATCH_ROUNDS` rounds
/// are attempted; a round either commits its pairs (when every triangle was
/// paired) or flips edges around the unpaired ones and retries on the whole
/// remainder.
///
/// # Postconditions
///
/// `points`, `quads` and `triangles` tile the input domain. When the merge
/// validation fails, the function falls back to the front-only state
/// (`stats.rejected_merge` is set, `stats.matched_quads` is zero and
/// `triangles` is the front's remainder). The input slices are not modified.
pub fn convert(tris: &[[usize; 3]], points: &[Node], size: f64) -> TwoStageResult {
    let front = qmorph::run_front(tris, points, size, FrontStop::EvenComponents);
    let mut stats = TwoStageStats {
        front_quads: front.quads.len(),
        core_triangles: front.triangles.len(),
        core_components: front.region_components.len(),
        ..TwoStageStats::default()
    };
    let mut quads = front.quads.clone();
    let mut region = front.triangles.clone();
    let points = front.points.clone();
    // The front's recovery flip can leave a remainder triangle clockwise; the
    // matcher (and `merge_is_valid`) assumes counter-clockwise. Normalise before
    // pairing and count it — this is not a dropped triangle.
    stats.inverted_core = normalize_winding(&mut region, &points);

    // Pair the remainder greedily by rounds; a round that leaves triangles
    // flips edges around them and retries on the whole remaining region.
    let guard = PairGuard {
        min_scaled_jacobian: 0.05,
        min_angle_deg: 10.0,
        max_angle_deg: 160.0,
    };
    let mut matched: Vec<[usize; 4]> = Vec::new();
    for round in 0..MAX_MATCH_ROUNDS {
        if region.is_empty() {
            break;
        }
        stats.rounds = round + 1;
        let (pairs, left) = pair_triangles_with(&region, &points, guard);
        if left.is_empty() {
            // Perfect pairing: commit and stop. This is the all-quad exit.
            stats.matched_quads += pairs.len();
            matched.extend(pairs);
            region.clear();
            break;
        }
        // A round that leaves triangles flips edges around them and retries on
        // the whole region; the pairs of this round are then stale and are
        // **not** committed — committing them would double-cover the triangles
        // that the flips rewrote.
        let flips = flip_around(&mut region, &left, &points);
        stats.flips += flips;
        if flips == 0 {
            stats.matched_quads += pairs.len();
            matched.extend(pairs);
            region = left;
            break;
        }
    }
    stats.unpaired = region.len();

    // Merge and validate: a matched quad must not share more than one edge with
    // any other cell, and the area must be unchanged (matching only merges
    // triangles, so it is exactly unchanged unless the region was malformed).
    let candidate_quads: Vec<[usize; 4]> = quads.iter().chain(matched.iter()).copied().collect();
    let area_before = quads.iter().map(|q| quad_area(q, &points)).sum::<f64>()
        + triangle_area(&front.triangles, &points);
    if merge_is_valid(&candidate_quads, &region, &points, area_before) {
        quads = candidate_quads;
    } else {
        stats.rejected_merge = true;
        stats.matched_quads = 0;
        region = front.triangles.clone();
    }

    TwoStageResult {
        points,
        quads,
        triangles: region,
        stats,
        refusals: front.refusals,
    }
}

/// Put every triangle in counter-clockwise order; returns how many were
/// reversed. Winding does not change the geometry, only the stored order.
fn normalize_winding(region: &mut [[usize; 3]], points: &[Node]) -> usize {
    let mut fixed = 0usize;
    for t in region.iter_mut() {
        let (a, b, c) = (points[t[0]], points[t[1]], points[t[2]]);
        if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) < 0.0 {
            *t = [t[0], t[2], t[1]];
            fixed += 1;
        }
    }
    fixed
}

/// Total absolute area of a triangle list, in square millimetres.
fn triangle_area(tris: &[[usize; 3]], points: &[Node]) -> f64 {
    tris.iter()
        .map(|t| {
            let a = points[t[0]];
            let b = points[t[1]];
            let c = points[t[2]];
            ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])).abs() / 2.0
        })
        .sum()
}

/// Absolute area of a quad, in square millimetres (shoelace, halved).
fn quad_area(quad: &[usize; 4], points: &[Node]) -> f64 {
    let mut sum = 0.0;
    for i in 0..4 {
        let a = points[quad[i]];
        let b = points[quad[(i + 1) % 4]];
        sum += a[0] * b[1] - b[0] * a[1];
    }
    sum.abs() / 2.0
}

/// A quad and a triangle cell list must tile the same area they came from, and
/// no two cells may share more than one edge.
///
/// The area tolerance is relative (`1e-6 * max(area, 1)`), so a large mesh is
/// not rejected by accumulated rounding. The edge test detects two cells that
/// overlap by sharing two edges; it deliberately does not require every interior
/// edge to have two owners (the matcher's output is assumed to come from a
/// triangulation that already tiles the region).
fn merge_is_valid(
    quads: &[[usize; 4]],
    triangles: &[[usize; 3]],
    points: &[Node],
    area_before: f64,
) -> bool {
    let area_after: f64 =
        quads.iter().map(|q| quad_area(q, points)).sum::<f64>() + triangle_area(triangles, points);
    if (area_after - area_before).abs() > 1e-6 * area_before.max(1.0) {
        return false;
    }
    let mut edges: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    let mut push = |nodes: &[usize], id: usize| {
        for i in 0..nodes.len() {
            let (a, b) = (nodes[i], nodes[(i + 1) % nodes.len()]);
            edges.entry((a.min(b), a.max(b))).or_default().push(id);
        }
    };
    for (id, q) in quads.iter().enumerate() {
        push(q, id);
    }
    for (i, t) in triangles.iter().enumerate() {
        push(t, quads.len() + i);
    }
    let mut shared: HashSet<(usize, usize)> = HashSet::new();
    for owners in edges.values() {
        if owners.len() < 2 {
            continue;
        }
        for i in 0..owners.len() {
            for j in (i + 1)..owners.len() {
                let pair = (owners[i].min(owners[j]), owners[i].max(owners[j]));
                if !shared.insert(pair) {
                    return false;
                }
            }
        }
    }
    true
}

/// Flip interior edges around unpaired triangles, then let the next matching
/// round see the new adjacency. Returns how many flips were applied.
///
/// Only edges between two triangles **of the region** are flipped: those are
/// interior to the unconverted region, so the quads around it are untouched and
/// the mesh stays conforming. A flip is taken only when the two triangles form a
/// convex quadrilateral (the other diagonal exists).
///
/// The two convexity calls check the same quadrilateral in opposite windings
/// (`u-w-v-x` and its reverse `u-x-v-w`); a symmetric test of the two possible
/// diagonals would need one of them reversed. At most one edge per triangle is
/// flipped per call, and a triangle already flipped is skipped so the changes
/// cannot conflict. Returns the number of flips applied.
fn flip_around(region: &mut [[usize; 3]], unpaired: &[[usize; 3]], points: &[Node]) -> usize {
    let mut edge_to_tris: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (id, t) in region.iter().enumerate() {
        for i in 0..3 {
            let (a, b) = (t[i], t[(i + 1) % 3]);
            edge_to_tris
                .entry((a.min(b), a.max(b)))
                .or_default()
                .push(id);
        }
    }
    let unpaired_set: HashSet<[usize; 3]> = unpaired.iter().copied().collect();
    let mut flips = 0usize;
    let mut flipped: HashSet<usize> = HashSet::new();
    let mut changes: Vec<(usize, [usize; 3])> = Vec::new();
    for id in 0..region.len() {
        let t = region[id];
        if !unpaired_set.contains(&t) || flipped.contains(&id) {
            continue;
        }
        for i in 0..3 {
            let (u, v) = (t[i], t[(i + 1) % 3]);
            let key = (u.min(v), u.max(v));
            let Some(owners) = edge_to_tris.get(&key) else {
                continue;
            };
            if owners.len() != 2 {
                continue;
            }
            let other = if owners[0] == id {
                owners[1]
            } else {
                owners[0]
            };
            if flipped.contains(&other) {
                continue;
            }
            let w = t[(i + 2) % 3];
            let other_tri = region[other];
            let Some(x) = other_tri.iter().find(|&&n| n != u && n != v).copied() else {
                continue;
            };
            if w == x {
                continue;
            }
            // Flip only when the quadrilateral u-w-v-x is strictly convex; then
            // the other diagonal w-x is inside it.
            if !convex_quad(points, u, w, v, x) || !convex_quad(points, u, x, v, w) {
                continue;
            }
            changes.push((id, [w, x, u]));
            changes.push((other, [x, w, v]));
            flipped.insert(id);
            flipped.insert(other);
            flips += 1;
            break;
        }
    }
    for (id, cell) in changes {
        region[id] = cell;
    }
    flips
}

/// Strictly convex, positive orientation, for the quad `a-b-c-d` in order.
fn convex_quad(points: &[Node], a: usize, b: usize, c: usize, d: usize) -> bool {
    let p = [points[a], points[b], points[c], points[d]];
    let mut area = 0.0;
    for i in 0..4 {
        let q = p[(i + 1) % 4];
        area += p[i][0] * q[1] - q[0] * p[i][1];
    }
    if area <= 1e-12 {
        return false;
    }
    for i in 0..4 {
        let a = p[(i + 3) % 4];
        let b = p[i];
        let c = p[(i + 1) % 4];
        let u = [b[0] - a[0], b[1] - a[1]];
        let v = [c[0] - b[0], c[1] - b[1]];
        if u[0] * v[1] - u[1] * v[0] <= 0.0 {
            return false;
        }
    }
    true
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn square_points() -> Vec<Node> {
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    }

    #[test]
    fn a_square_converts_to_one_quad() {
        let points = square_points();
        let tris = [[0, 1, 2], [0, 2, 3]];
        let r = convert(&tris, &points, 1.0);
        assert_eq!(r.quads.len(), 1);
        assert!(r.triangles.is_empty(), "{:?}", r.triangles);
        assert!(!r.stats.rejected_merge);
    }

    #[test]
    fn a_reversed_winding_is_normalised_without_losing_area() {
        let points = square_points();
        let mut region = vec![[0, 2, 1], [0, 3, 2]];
        let before = triangle_area(&region, &points);
        assert_eq!(normalize_winding(&mut region, &points), 2);
        assert!((triangle_area(&region, &points) - before).abs() < 1e-12);
    }

    #[test]
    fn two_cells_sharing_two_edges_are_rejected() {
        let points = square_points();
        let quads = [[0, 1, 2, 3], [0, 1, 2, 3]];
        let area = quads.iter().map(|q| quad_area(q, &points)).sum();
        assert!(!merge_is_valid(&quads, &[], &points, area));
    }

    #[test]
    fn a_valid_tiling_passes_the_merge_check() {
        let points = square_points();
        let quads = [[0, 1, 2, 3]];
        let area = quad_area(&quads[0], &points);
        assert!(merge_is_valid(&quads, &[], &points, area));
    }

    #[test]
    fn a_square_of_two_triangles_pairs() {
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let tris = [[0, 1, 2], [0, 2, 3]];
        let (q, left) = pair_triangles_with(&tris, &points, PairGuard::default());
        assert_eq!(q.len(), 1, "{left:?}");
        assert!(left.is_empty());
    }
}
