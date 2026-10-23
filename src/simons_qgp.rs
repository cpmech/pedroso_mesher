//! Quantized Global Parameterization (QGP): integer subdivision counts and Coons
//! patch meshing, after Simons (2024), Chapter 5, Sections 5.0-5.1.3.
//!
//! # What is implemented
//!
//! 1. **Integer subdivision counts (5.1.1-5.1.2).** A small separatrix /
//!    motorcycle graph is given as [`Region`]s, each a quadrilateral whose four
//!    sides are sequences of edge ids. A side of a tile may be split into several
//!    edges by a T-junction, so the constraint that a tile can be meshed without
//!    an extra singularity is eq. (5.1),
//!
//!    ```text
//!    sum_{e in side0} s_e = sum_{e in side2} s_e
//!    sum_{e in side1} s_e = sum_{e in side3} s_e
//!    ```
//!
//!    Every edge also has an ideal real subdivision count `s_hat = integral
//!    dx / sigma` supplied by the caller. The solver minimises eq. (5.1)'s
//!    energy
//!
//!    ```text
//!    E(s) = sum_e (s_e / s_hat_e - 1)^2
//!    ```
//!
//!    starting from `s_e = 0` for every edge. A *unit augmentation* changes one
//!    edge by `+1` and, in every region-side constraint it touches, balances
//!    that change by either `+1` on an edge of the opposite side (a chord
//!    crossing the tile) or `-1` on another edge of the same side (a chord
//!    bouncing within the tile). Propagating the change through the tile/edge
//!    incidence graph gives an augmenting path that ends either at a perimeter
//!    edge (a boundary-to-boundary path) or back at the starting edge (a loop).
//!    Every edge and every start sign is tried; the path with the most negative
//!    energy change is applied and the search repeats until no unit augmentation
//!    lowers the energy, as described in Section 5.1.2. A unit augmentation
//!    changes one edge, i.e. one side's sum by one; a constraint that has already
//!    been balanced is never touched again, so a side may carry any number of
//!    edges without changing the bookkeeping.
//!
//! 2. **Coons patch meshing (5.1.3).** [`coons`] / [`coons_curves`] take four
//!    boundary polylines with counter-clockwise winding and the subdivision
//!    counts `n_u` (bottom/top) and `n_v` (left/right), build the bilinear Coons
//!    patch of eq. (5.2),
//!
//!    ```text
//!    P(u,v) = lerp(v, C0(u), C2(1-u))
//!           + lerp(u, C3(1-v), C1(v))
//!           - lerp(v, lerp(u, C0(0), C0(1)), lerp(u, C2(1), C2(0)))
//!    ```
//!
//!    and emit the `n_u * n_v` quads. The quad edges are the **logical grid
//!    lines** of the parameterisation, `f(u) = (u, lerp(u, v0, v1))`, not field
//!    lines traced through the cross field. [`coons_from_parameters`] exposes the
//!    parameter positions of the four sides, so a caller that distributes the
//!    integer counts per separatrix edge (the source of chord drift, Section
//!    5.1.5.1) can pass the resulting non-uniform `v0 != v1` seams. The plain
//!    [`coons`] / [`coons_curves`] API uses uniformly spaced parameter seams,
//!    which is the "aligned" special case of the same construction.
//!
//! # Faithful / simplified / not implemented
//!
//! * Faithful: eq. (5.1), the quadratic energy, the `s_i >= 0` condition, the
//!   augmenting-path iteration (loops and perimeter-to-perimeter paths), eq.
//!   (5.2), and the logical-grid-line quad construction.
//! * Simplified: the augmenting path is found by an exhaustive depth-first
//!   search with a fixed node budget ([`MAX_SEARCH_NODES`]) over simple paths,
//!   not by the specialised shortest-path procedure of Campen et al. It is
//!   exact for the small graphs this module targets. An edge may have at most
//!   two region-side incidences, which is the manifold motorcycle-graph case
//!   (a perimeter edge has one, an interior edge two). `coons` uses uniform
//!   parameter seams; per-edge seam distributions have to be supplied through
//!   [`coons_from_parameters`].
//! * Not implemented: extra singularity pairs and the nautilus relaxation of
//!   Section 5.1.4 (self-referential sides are rejected), singularity merging
//!   (5.1.5.2), degenerate tile collapse (5.1.6), the size function itself
//!   (`s_hat` is an input), cross-field construction (Chapter 3), and stitching
//!   the interior mesh to the domain boundary (Chapter 2).
//!
//! The module has no external dependencies, does not panic in non-test code,
//! and is `wasm32-unknown-unknown` compatible.

use crate::mesh::{Cell, Mesh};

/// A unit augmentation must lower the energy by more than this.
const EPS_IMPROVEMENT: f64 = 1e-12;
/// Safety bound on the number of applied augmentations.
const MAX_AUGMENTATIONS: usize = 200_000;
/// Safety bound on the depth-first search nodes per (edge, sign) start.
const MAX_SEARCH_NODES: usize = 100_000;

// ---------------------------------------------------------------------------
// Region / constraint model
// ---------------------------------------------------------------------------

/// One logically rectangular tile (a simple tile of the motorcycle graph).
///
/// The four sides are in counter-clockwise order and each side is a sequence of
/// edge ids. Eq. (5.1) couples `sides[0]` with `sides[2]` and `sides[1]` with
/// `sides[3]`. A side may be empty (the tile is bounded by a single collapsed
/// point there), and an edge may be shared by several regions.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Region {
    /// Four sides; side `i` is the list of motorcycle-graph edge ids on it.
    pub sides: [Vec<usize>; 4],
}

impl Region {
    /// Build a tile from its four sides.
    pub fn new(sides: [Vec<usize>; 4]) -> Region {
        Region { sides }
    }
}

/// One equality constraint `sum(side_a) == sum(side_b)` of eq. (5.1).
#[derive(Clone, Debug)]
struct Constraint {
    a: Vec<usize>,
    b: Vec<usize>,
}

/// The edge/tile incidence graph used by the augmenting-path search.
#[derive(Clone, Debug)]
struct Graph {
    constraints: Vec<Constraint>,
    /// Constraint ids containing each edge.
    incidence: Vec<Vec<usize>>,
}

/// Build the constraint graph, validating the region description.
fn build_graph(regions: &[Region], n_edges: usize) -> Result<Graph, String> {
    let mut constraints: Vec<Constraint> = Vec::with_capacity(regions.len() * 2);
    for (ri, region) in regions.iter().enumerate() {
        for (ia, ib) in [(0usize, 2usize), (1usize, 3usize)] {
            let a = match region.sides.get(ia) {
                Some(side) => side,
                None => return Err(format!("region {ri}: missing side {ia}")),
            };
            let b = match region.sides.get(ib) {
                Some(side) => side,
                None => return Err(format!("region {ri}: missing side {ib}")),
            };

            // Reject duplicate ids within one side: a side is a sequence of
            // distinct separatrix edges, and a repeated id would make the
            // constraint sum ambiguous.
            for (si, side) in [(ia, a), (ib, b)] {
                let mut seen: Vec<usize> = Vec::with_capacity(side.len());
                for &e in side {
                    if e >= n_edges {
                        return Err(format!(
                            "region {ri} side {si}: edge {e} is out of range \
                             (only {n_edges} ideal counts were supplied)"
                        ));
                    }
                    if seen.contains(&e) {
                        return Err(format!(
                            "region {ri} side {si}: edge {e} is listed more than once"
                        ));
                    }
                    seen.push(e);
                }
            }

            // An edge on both sides of the same constraint gives s_e = s_e + ...
            // (the nautilus case of Section 5.1.4). It is not part of this
            // implementation and is reported instead of silently mis-solved.
            if let Some(e) = a.iter().find(|e| b.contains(e)) {
                return Err(format!(
                    "region {ri}: edge {e} lies on both sides of constraint \
                     ({ia},{ib}); self-referential sides (nautilus, Section \
                     5.1.4) are not supported yet"
                ));
            }

            if a.is_empty() && b.is_empty() {
                continue;
            }
            constraints.push(Constraint {
                a: a.clone(),
                b: b.clone(),
            });
        }
    }

    let mut incidence: Vec<Vec<usize>> = vec![Vec::new(); n_edges];
    for (ci, constraint) in constraints.iter().enumerate() {
        for &e in constraint.a.iter().chain(constraint.b.iter()) {
            match incidence.get_mut(e) {
                Some(list) => list.push(ci),
                None => {
                    return Err(format!("internal error: edge {e} outside the graph"));
                }
            }
        }
    }
    for (e, list) in incidence.iter().enumerate() {
        if list.len() > 2 {
            return Err(format!(
                "edge {e} is incident to {} region-side constraints; this solver \
                 supports the manifold case (one perimeter incidence or two \
                 interior incidences) only",
                list.len()
            ));
        }
    }

    Ok(Graph {
        constraints,
        incidence,
    })
}

/// Sum the integer counts on one side, returning `None` for an out-of-range id.
fn side_sum(counts: &[usize], side: &[usize]) -> Option<u128> {
    let mut total = 0u128;
    for &e in side {
        total += *counts.get(e)? as u128;
    }
    Some(total)
}

/// Check eq. (5.1) on every region for the given integer counts.
///
/// Returns `false` for an out-of-range edge id. This is the test the solver runs
/// on its own result before returning it.
pub fn constraints_hold(regions: &[Region], counts: &[usize]) -> bool {
    for region in regions {
        for (ia, ib) in [(0usize, 2usize), (1usize, 3usize)] {
            let a = match region.sides.get(ia) {
                Some(side) => side,
                None => return false,
            };
            let b = match region.sides.get(ib) {
                Some(side) => side,
                None => return false,
            };
            let (sa, sb) = match (side_sum(counts, a), side_sum(counts, b)) {
                (Some(sa), Some(sb)) => (sa, sb),
                _ => return false,
            };
            if sa != sb {
                return false;
            }
        }
    }
    true
}

/// The QGP energy `sum_e (s_e / s_hat_e - 1)^2` of eq. (5.1).
///
/// `counts.len()` must equal `edge_ideal.len()`, and every ideal count must be
/// finite and positive.
pub fn energy(counts: &[usize], edge_ideal: &[f64]) -> Result<f64, String> {
    if counts.len() != edge_ideal.len() {
        return Err(format!(
            "energy: {} counts but {} ideal values",
            counts.len(),
            edge_ideal.len()
        ));
    }
    let mut total = 0.0;
    for (i, (&s, &ideal)) in counts.iter().zip(edge_ideal.iter()).enumerate() {
        if !ideal.is_finite() || ideal <= 0.0 {
            return Err(format!(
                "edge {i}: ideal subdivision count must be finite and positive \
                 (got {ideal})"
            ));
        }
        let d = s as f64 / ideal - 1.0;
        total += d * d;
    }
    Ok(total)
}

/// Change of the energy when edge `e` moves from `s` by `delta` (`+1` or `-1`).
fn edge_delta(s: usize, ideal: f64, delta: i32) -> f64 {
    let old = (s as f64 / ideal - 1.0).powi(2);
    let new = ((s as f64 + delta as f64) / ideal - 1.0).powi(2);
    new - old
}

// ---------------------------------------------------------------------------
// Augmenting-path search (Section 5.1.2)
// ---------------------------------------------------------------------------

/// Depth-first search for the best simple augmenting path from one start edge.
struct AugmentSearch<'a> {
    graph: &'a Graph,
    counts: &'a [usize],
    ideal: &'a [f64],
    start: usize,
    start_delta: i32,
    visited: Vec<bool>,
    used: Vec<bool>,
    changes: Vec<(usize, i32)>,
    best_cost: f64,
    best_changes: Option<Vec<(usize, i32)>>,
    expansions: usize,
    max_expansions: usize,
}

impl<'a> AugmentSearch<'a> {
    fn new(
        graph: &'a Graph,
        counts: &'a [usize],
        ideal: &'a [f64],
        start: usize,
        start_delta: i32,
    ) -> AugmentSearch<'a> {
        AugmentSearch {
            graph,
            counts,
            ideal,
            start,
            start_delta,
            visited: vec![false; counts.len()],
            used: vec![false; graph.constraints.len()],
            changes: Vec::new(),
            best_cost: f64::INFINITY,
            best_changes: None,
            expansions: 0,
            max_expansions: MAX_SEARCH_NODES,
        }
    }

    /// Run the search and return `(energy change, per-edge signed changes)`.
    fn run(mut self) -> Option<(f64, Vec<(usize, i32)>)> {
        let s = *self.counts.get(self.start)?;
        let ideal = *self.ideal.get(self.start)?;
        if self.start_delta < 0 && s == 0 {
            return None;
        }
        let cost = edge_delta(s, ideal, self.start_delta);
        *self.visited.get_mut(self.start)? = true;
        self.changes.push((self.start, self.start_delta));
        self.dfs(self.start, self.start_delta, cost);
        self.best_changes.map(|changes| (self.best_cost, changes))
    }

    /// Does the closing constraint balance the start's change and `delta`?
    fn balances_with_start(&self, current: usize, delta: i32, a: &[usize]) -> bool {
        let current_side = if a.contains(&current) { delta } else { -delta };
        let start_side = if a.contains(&self.start) {
            self.start_delta
        } else {
            -self.start_delta
        };
        current_side + start_side == 0
    }

    /// Every constraint incident to every changed edge must have been processed.
    fn path_is_complete(&self) -> bool {
        for &(e, _) in &self.changes {
            let incidences = match self.graph.incidence.get(e) {
                Some(list) => list,
                None => return false,
            };
            for &ci in incidences {
                if self.used.get(ci).copied() != Some(true) {
                    return false;
                }
            }
        }
        true
    }

    fn record(&mut self, cost: f64) {
        if cost < self.best_cost - EPS_IMPROVEMENT && self.path_is_complete() {
            self.best_cost = cost;
            self.best_changes = Some(self.changes.clone());
        }
    }

    fn try_step(&mut self, y: usize, sign: i32, cost: f64, current_constraint: usize) {
        if self.visited.get(y).copied().unwrap_or(true) {
            return;
        }
        // A constraint that has already been balanced must never be touched
        // again: changing `y` would alter one of its side sums after the fact.
        // This matters as soon as a tile side carries more than one edge,
        // because another edge of the same side can otherwise be reached
        // through a different constraint and silently unbalance this one.
        if let Some(incidences) = self.graph.incidence.get(y) {
            for &ci in incidences {
                if ci != current_constraint && self.used.get(ci).copied() == Some(true) {
                    return;
                }
            }
        }
        let s = match self.counts.get(y) {
            Some(&v) => v,
            None => return,
        };
        if sign < 0 && s == 0 {
            return;
        }
        let ideal = match self.ideal.get(y) {
            Some(&v) => v,
            None => return,
        };
        let delta = edge_delta(s, ideal, sign);
        match self.visited.get_mut(y) {
            Some(v) => *v = true,
            None => return,
        }
        self.changes.push((y, sign));
        self.dfs(y, sign, cost + delta);
        self.changes.pop();
        if let Some(v) = self.visited.get_mut(y) {
            *v = false;
        }
    }

    fn dfs(&mut self, current: usize, delta: i32, cost: f64) {
        if self.expansions >= self.max_expansions {
            return;
        }
        self.expansions += 1;

        let unused: Vec<usize> = match self.graph.incidence.get(current) {
            Some(list) => list
                .iter()
                .copied()
                .filter(|&ci| self.used.get(ci).copied() != Some(true))
                .collect(),
            None => return,
        };

        if unused.is_empty() {
            // A perimeter edge: the unit change has been balanced at every
            // incident tile and the outside absorbs it. A loop closes through
            // `balances_with_start` below.
            self.record(cost);
            return;
        }

        for ci in unused {
            if self.used.get(ci).copied().unwrap_or(true) {
                continue;
            }
            let (a, b) = match self.graph.constraints.get(ci) {
                Some(c) => (c.a.clone(), c.b.clone()),
                None => continue,
            };

            // A changed edge other than `current` in this constraint is only
            // legal when it is the start edge and this is the closing loop
            // constraint. Anything else would make the path non-simple and
            // would leave the constraint unbalanced.
            let mut other: Option<usize> = None;
            let mut multiple = false;
            for &(e, _) in &self.changes {
                if e != current && (a.contains(&e) || b.contains(&e)) {
                    if other.is_some() {
                        multiple = true;
                        break;
                    }
                    other = Some(e);
                }
            }
            if multiple {
                continue;
            }
            if let Some(other) = other {
                if other == self.start && self.balances_with_start(current, delta, &a) {
                    self.used[ci] = true;
                    self.record(cost);
                    self.used[ci] = false;
                }
                continue;
            }

            self.used[ci] = true;
            let current_in_a = a.contains(&current);
            let (same, opposite) = if current_in_a { (&a, &b) } else { (&b, &a) };

            // The constraint is an equality of side sums, not of edge ids.
            // Changing one edge changes its side's sum by one, so exactly one
            // edge of each side is touched per unit augmentation; the number
            // of edges a side carries does not enter the bookkeeping.
            // Crossing the tile: an edge on the opposite side changes with the
            // same sign.
            let opposite_partners: Vec<usize> = opposite.clone();
            for y in opposite_partners {
                self.try_step(y, delta, cost, ci);
            }

            // Redirecting the chord: another edge on the same side changes
            // with the opposite sign.
            let same_partners: Vec<usize> =
                same.iter().copied().filter(|&y| y != current).collect();
            for y in same_partners {
                self.try_step(y, -delta, cost, ci);
            }

            self.used[ci] = false;
        }
    }
}

/// Find the most beneficial unit augmentation over all edges and both signs.
fn find_augmentation(
    graph: &Graph,
    counts: &[usize],
    ideal: &[f64],
) -> Option<(f64, Vec<(usize, i32)>)> {
    let mut best: Option<(f64, Vec<(usize, i32)>)> = None;
    for e in 0..counts.len() {
        for delta in [1i32, -1i32] {
            if delta < 0 && counts.get(e).copied().unwrap_or(0) == 0 {
                continue;
            }
            let search = AugmentSearch::new(graph, counts, ideal, e, delta);
            let (cost, changes) = match search.run() {
                Some(found) => found,
                None => continue,
            };
            if cost >= -EPS_IMPROVEMENT {
                continue;
            }
            let replace = match &best {
                None => true,
                Some((best_cost, _)) => cost < *best_cost - EPS_IMPROVEMENT,
            };
            if replace {
                best = Some((cost, changes));
            }
        }
    }
    best
}

/// Apply a signed unit augmentation, checking the `s_i >= 0` condition.
fn apply_changes(counts: &mut [usize], changes: &[(usize, i32)]) -> Result<(), String> {
    for &(e, delta) in changes {
        let value = counts
            .get_mut(e)
            .ok_or_else(|| format!("internal error: augmentation edge {e} out of range"))?;
        if delta > 0 {
            *value = value
                .checked_add(1)
                .ok_or_else(|| format!("edge {e}: subdivision count overflow"))?;
        } else if delta < 0 {
            *value = value
                .checked_sub(1)
                .ok_or_else(|| format!("edge {e}: subdivision count became negative"))?;
        } else {
            return Err(format!("internal error: zero augmentation on edge {e}"));
        }
    }
    Ok(())
}

/// Solve for the integer subdivision counts of every edge.
///
/// `regions` is the small separatrix/motorcycle graph (each tile is a
/// [`Region`]); `edge_ideal[i]` is `s_hat_i = integral dx / sigma` on edge `i`.
/// All edge ids referenced by the regions must be smaller than
/// `edge_ideal.len()`, and every ideal count must be finite and positive.
///
/// The solver starts with all counts at zero, applies the best augmenting path
/// (loop or perimeter-to-perimeter) while one strictly lowers the energy, and
/// returns the resulting non-negative integer counts. Eq. (5.1) is re-checked
/// before returning. An error is returned for an invalid region description, a
/// non-positive ideal count, or if the safety limit on augmentations is reached.
pub fn solve_counts(regions: &[Region], edge_ideal: &[f64]) -> Result<Vec<usize>, String> {
    for (i, &ideal) in edge_ideal.iter().enumerate() {
        if !ideal.is_finite() || ideal <= 0.0 {
            return Err(format!(
                "edge {i}: ideal subdivision count must be finite and positive \
                 (got {ideal})"
            ));
        }
    }
    let graph = build_graph(regions, edge_ideal.len())?;
    let mut counts = vec![0usize; edge_ideal.len()];
    let mut steps = 0usize;

    while let Some((_cost, changes)) = find_augmentation(&graph, &counts, edge_ideal) {
        if steps >= MAX_AUGMENTATIONS {
            return Err(format!(
                "augmentation limit ({MAX_AUGMENTATIONS}) reached before the \
                 energy stopped improving"
            ));
        }
        apply_changes(&mut counts, &changes)?;
        steps += 1;
    }

    for (ci, constraint) in graph.constraints.iter().enumerate() {
        let (sa, sb) = match (
            side_sum(&counts, &constraint.a),
            side_sum(&counts, &constraint.b),
        ) {
            (Some(sa), Some(sb)) => (sa, sb),
            _ => return Err(format!("internal error: constraint {ci} has a bad edge id")),
        };
        if sa != sb {
            return Err(format!(
                "internal error: constraint {ci} is not balanced after the solve \
                 ({sa} != {sb})"
            ));
        }
    }

    Ok(counts)
}

// ---------------------------------------------------------------------------
// Coons patches (Section 5.1.3)
// ---------------------------------------------------------------------------

/// The four boundary curves of a Coons patch, all in counter-clockwise winding.
///
/// * [`bottom`](BoundaryCurves::bottom) (`C0`) runs left to right, so
///   `C0(u)` is the point at arc-length fraction `u` from the bottom-left
///   corner.
/// * [`right`](BoundaryCurves::right) (`C1`) runs bottom to top.
/// * [`top`](BoundaryCurves::top) (`C2`) runs right to left; this is why eq.
///   (5.2) samples it at `1-u`.
/// * [`left`](BoundaryCurves::left) (`C3`) runs top to bottom; eq. (5.2)
///   samples it at `1-v`.
///
/// Consecutive curves must share their end points. The corner polygon
/// bottom-left -> bottom-right -> top-right -> top-left must have positive
/// signed area (counter-clockwise).
#[derive(Clone, Debug, Default)]
pub struct BoundaryCurves {
    /// `C0`: bottom curve, left to right.
    pub bottom: Vec<[f64; 2]>,
    /// `C1`: right curve, bottom to top.
    pub right: Vec<[f64; 2]>,
    /// `C2`: top curve, right to left.
    pub top: Vec<[f64; 2]>,
    /// `C3`: left curve, top to bottom.
    pub left: Vec<[f64; 2]>,
}

impl BoundaryCurves {
    /// Straight-edged patch from its four corners, in counter-clockwise order:
    /// bottom-left, bottom-right, top-right, top-left.
    pub fn from_corners(corners: [[f64; 2]; 4]) -> BoundaryCurves {
        let [bl, br, tr, tl] = corners;
        BoundaryCurves {
            bottom: vec![bl, br],
            right: vec![br, tr],
            top: vec![tr, tl],
            left: vec![tl, bl],
        }
    }

    /// Split a closed counter-clockwise boundary loop into four equal-length
    /// chunks (bottom, right, top, left).
    ///
    /// A loop of exactly four points is read as the four corners. Longer loops
    /// may repeat the first point at the end; the duplicate is removed. The
    /// remaining points must be divisible by four, with at least two points per
    /// curve. For curves with different sampling densities, build the
    /// [`BoundaryCurves`] fields directly.
    pub fn from_loop(points: &[[f64; 2]]) -> Result<BoundaryCurves, String> {
        if points.len() == 4 {
            return Ok(BoundaryCurves::from_corners([
                points[0], points[1], points[2], points[3],
            ]));
        }

        let mut loop_points = points;
        if points.len() >= 5 {
            let first = points.first().copied();
            let last = points.last().copied();
            if let (Some(first), Some(last)) = (first, last) {
                if first == last {
                    loop_points = &points[..points.len() - 1];
                }
            }
        }
        if loop_points.len() == 4 {
            return Ok(BoundaryCurves::from_corners([
                loop_points[0],
                loop_points[1],
                loop_points[2],
                loop_points[3],
            ]));
        }
        if loop_points.len() < 8 || !loop_points.len().is_multiple_of(4) {
            return Err(
                "coons: a closed boundary loop must have 4 points, or a multiple \
                 of 4 points with at least 2 per curve"
                    .to_string(),
            );
        }
        let k = loop_points.len() / 4;
        let mut split = Vec::with_capacity(4);
        for chunk in loop_points.chunks(k) {
            split.push(chunk.to_vec());
        }
        if split.len() != 4 {
            return Err("coons: could not split the boundary loop into four curves".to_string());
        }
        let mut chunks = loop_points.chunks_exact(k);
        let bottom = chunks
            .next()
            .ok_or_else(|| "coons: missing bottom curve".to_string())?
            .to_vec();
        let right = chunks
            .next()
            .ok_or_else(|| "coons: missing right curve".to_string())?
            .to_vec();
        let top = chunks
            .next()
            .ok_or_else(|| "coons: missing top curve".to_string())?
            .to_vec();
        let left = chunks
            .next()
            .ok_or_else(|| "coons: missing left curve".to_string())?
            .to_vec();
        Ok(BoundaryCurves {
            bottom,
            right,
            top,
            left,
        })
    }
}

/// A polyline sampled by arc length, as eq. (5.2)'s `C_i(t)`.
#[derive(Clone, Debug)]
struct ArcCurve {
    points: Vec<[f64; 2]>,
    cumulative: Vec<f64>,
    total: f64,
}

impl ArcCurve {
    fn new(points: &[[f64; 2]]) -> Result<ArcCurve, String> {
        if points.len() < 2 {
            return Err("coons: a boundary curve needs at least two points".to_string());
        }
        let mut cumulative = Vec::with_capacity(points.len());
        cumulative.push(0.0);
        let mut total = 0.0;
        for pair in points.windows(2) {
            let p = pair[0];
            let q = pair[1];
            if !p[0].is_finite() || !p[1].is_finite() || !q[0].is_finite() || !q[1].is_finite() {
                return Err("coons: boundary curve has a non-finite point".to_string());
            }
            let dx = q[0] - p[0];
            let dy = q[1] - p[1];
            total += (dx * dx + dy * dy).sqrt();
            cumulative.push(total);
        }
        if !total.is_finite() || total <= 0.0 {
            return Err("coons: boundary curve has zero length".to_string());
        }
        Ok(ArcCurve {
            points: points.to_vec(),
            cumulative,
            total,
        })
    }

    /// Point at arc-length fraction `t` (clamped to `[0, 1]`).
    fn at(&self, t: f64) -> Result<[f64; 2], String> {
        if !t.is_finite() {
            return Err("coons: non-finite curve parameter".to_string());
        }
        let t = t.clamp(0.0, 1.0);
        let target = t * self.total;
        let n = self.points.len();
        if n < 2 {
            return Err("coons: degenerate boundary curve".to_string());
        }
        let mut lo = 0usize;
        let mut hi = n - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            let cm = *self
                .cumulative
                .get(mid)
                .ok_or_else(|| "coons: bad cumulative table".to_string())?;
            if cm <= target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let p0 = *self
            .points
            .get(lo)
            .ok_or_else(|| "coons: curve index out of range".to_string())?;
        let p1 = *self
            .points
            .get(lo + 1)
            .ok_or_else(|| "coons: curve index out of range".to_string())?;
        let c0 = *self
            .cumulative
            .get(lo)
            .ok_or_else(|| "coons: cumulative index out of range".to_string())?;
        let c1 = *self
            .cumulative
            .get(lo + 1)
            .ok_or_else(|| "coons: cumulative index out of range".to_string())?;
        let span = c1 - c0;
        let fraction = if span > 0.0 {
            (target - c0) / span
        } else {
            0.0
        };
        Ok([
            p0[0] + (p1[0] - p0[0]) * fraction,
            p0[1] + (p1[1] - p0[1]) * fraction,
        ])
    }
}

fn lerp2(t: f64, a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [(1.0 - t) * a[0] + t * b[0], (1.0 - t) * a[1] + t * b[1]]
}

/// Validate the four curves and return their arc-length parameterisations.
fn checked_curves(curves: &BoundaryCurves) -> Result<[ArcCurve; 4], String> {
    let c0 = ArcCurve::new(&curves.bottom)?;
    let c1 = ArcCurve::new(&curves.right)?;
    let c2 = ArcCurve::new(&curves.top)?;
    let c3 = ArcCurve::new(&curves.left)?;

    let bl = c0.at(0.0)?;
    let br = c0.at(1.0)?;
    let tr = c1.at(1.0)?;
    let tl = c3.at(0.0)?;

    let scale = {
        let xs = [bl[0], br[0], tr[0], tl[0]];
        let ys = [bl[1], br[1], tr[1], tl[1]];
        let mut min_x = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for &x in &xs {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
        }
        for &y in &ys {
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
        (max_x - min_x).max(max_y - min_y).max(1.0)
    };
    let tolerance = 1e-9 * scale;

    let corners = [
        (c1.at(0.0)?, br),
        (c2.at(0.0)?, tr),
        (c2.at(1.0)?, tl),
        (c3.at(0.0)?, tl),
        (c3.at(1.0)?, bl),
    ];
    for (index, (got, expected)) in corners.iter().enumerate() {
        let dx = got[0] - expected[0];
        let dy = got[1] - expected[1];
        if dx.abs() > tolerance || dy.abs() > tolerance {
            return Err(format!(
                "coons: boundary curves are not connected at corner {index} \
                 (got {got:?}, expected {expected:?})"
            ));
        }
    }

    let corner_polygon = [bl, br, tr, tl];
    if signed_area(&corner_polygon) <= 0.0 {
        return Err(
            "coons: boundary curves must wind counter-clockwise (the corner \
             polygon has non-positive signed area)"
                .to_string(),
        );
    }

    Ok([c0, c1, c2, c3])
}

/// Eq. (5.2): one point of the bilinear Coons patch.
fn coons_point(curves: &[ArcCurve; 4], u: f64, v: f64) -> Result<[f64; 2], String> {
    let c0u = curves[0].at(u)?;
    let c2u = curves[2].at(1.0 - u)?;
    let c3v = curves[3].at(1.0 - v)?;
    let c1v = curves[1].at(v)?;

    let c00 = curves[0].at(0.0)?;
    let c01 = curves[0].at(1.0)?;
    let c20 = curves[2].at(0.0)?;
    let c21 = curves[2].at(1.0)?;

    let ruled_v = lerp2(v, c0u, c2u);
    let ruled_u = lerp2(u, c3v, c1v);
    let bilinear = lerp2(v, lerp2(u, c00, c01), lerp2(u, c21, c20));

    Ok([
        ruled_v[0] + ruled_u[0] - bilinear[0],
        ruled_v[1] + ruled_u[1] - bilinear[1],
    ])
}

/// Intersection of one logical horizontal line and one logical vertical line.
///
/// The horizontal line runs from `(0, a)` to `(1, b)`, i.e.
/// `v = lerp(u, a, b)`; the vertical line runs from `(c, 0)` to `(d, 1)`, i.e.
/// `u = lerp(v, c, d)`. This is the `f(u) = (u, lerp(u, v0, v1))` construction
/// of Section 5.1.3: with `a != b` the line is *not* a straight `v = const`
/// grid line.
fn logical_intersection(a: f64, b: f64, c: f64, d: f64) -> Result<(f64, f64), String> {
    let da = b - a;
    let dc = d - c;
    let denominator = 1.0 - da * dc;
    if denominator.abs() < 1e-12 {
        return Err(
            "coons: logical grid lines are parallel or coincident; the parameter \
             seams are degenerate"
                .to_string(),
        );
    }
    let v = (a + c * da) / denominator;
    let u = c + v * dc;
    let tolerance = 1e-9;
    if !u.is_finite()
        || !v.is_finite()
        || u < -tolerance
        || u > 1.0 + tolerance
        || v < -tolerance
        || v > 1.0 + tolerance
    {
        return Err(
            "coons: logical grid lines cross outside the unit square; the \
             parameter seams are not monotone"
                .to_string(),
        );
    }
    Ok((u.clamp(0.0, 1.0), v.clamp(0.0, 1.0)))
}

fn validate_parameters(name: &str, parameters: &[f64]) -> Result<(), String> {
    if parameters.len() < 2 {
        return Err(format!(
            "coons: the {name} parameter seam needs at least two values"
        ));
    }
    for (i, &value) in parameters.iter().enumerate() {
        if !value.is_finite() || !(-1e-9..=1.0 + 1e-9).contains(&value) {
            return Err(format!(
                "coons: {name} parameter {i} ({value}) is outside [0, 1]"
            ));
        }
        if i > 0 {
            let previous = parameters[i - 1];
            if value + 1e-12 < previous {
                return Err(format!(
                    "coons: {name} parameters are not non-decreasing at index {i}"
                ));
            }
        }
    }
    let first = parameters[0];
    let last = parameters[parameters.len() - 1];
    if first.abs() > 1e-9 || (last - 1.0).abs() > 1e-9 {
        return Err(format!(
            "coons: {name} parameters must start at 0 and end at 1"
        ));
    }
    Ok(())
}

/// Build the logical grid and validate that no quad is inverted.
fn coons_grid(
    curves: &[ArcCurve; 4],
    bottom_u: &[f64],
    top_u: &[f64],
    left_v: &[f64],
    right_v: &[f64],
) -> Result<(Vec<[f64; 2]>, usize, usize), String> {
    validate_parameters("bottom", bottom_u)?;
    validate_parameters("top", top_u)?;
    validate_parameters("left", left_v)?;
    validate_parameters("right", right_v)?;
    if bottom_u.len() != top_u.len() {
        return Err(format!(
            "coons: bottom has {} parameters but top has {}",
            bottom_u.len(),
            top_u.len()
        ));
    }
    if left_v.len() != right_v.len() {
        return Err(format!(
            "coons: left has {} parameters but right has {}",
            left_v.len(),
            right_v.len()
        ));
    }
    let n_u = bottom_u.len() - 1;
    let n_v = left_v.len() - 1;
    if n_u == 0 || n_v == 0 {
        return Err("coons: n_u and n_v must both be at least one".to_string());
    }

    let mut grid = Vec::with_capacity((n_u + 1) * (n_v + 1));
    for j in 0..=n_v {
        let a = *left_v
            .get(j)
            .ok_or_else(|| "coons: left parameter index out of range".to_string())?;
        let b = *right_v
            .get(j)
            .ok_or_else(|| "coons: right parameter index out of range".to_string())?;
        for i in 0..=n_u {
            let c = *bottom_u
                .get(i)
                .ok_or_else(|| "coons: bottom parameter index out of range".to_string())?;
            let d = *top_u
                .get(i)
                .ok_or_else(|| "coons: top parameter index out of range".to_string())?;
            let (u, v) = logical_intersection(a, b, c, d)?;
            grid.push(coons_point(curves, u, v)?);
        }
    }

    for j in 0..n_v {
        for i in 0..n_u {
            let quad = grid_quad(&grid, n_u, i, j)?;
            let area = signed_area(&quad);
            if !area.is_finite() || area <= 0.0 {
                return Err(format!(
                    "coons: quad ({i},{j}) is inverted or degenerate \
                     (signed area {area:.6e})"
                ));
            }
        }
    }

    Ok((grid, n_u, n_v))
}

fn grid_index(n_u: usize, i: usize, j: usize) -> usize {
    j * (n_u + 1) + i
}

fn grid_quad(grid: &[[f64; 2]], n_u: usize, i: usize, j: usize) -> Result<[[f64; 2]; 4], String> {
    let p00 = *grid
        .get(grid_index(n_u, i, j))
        .ok_or_else(|| "coons: grid index out of range".to_string())?;
    let p10 = *grid
        .get(grid_index(n_u, i + 1, j))
        .ok_or_else(|| "coons: grid index out of range".to_string())?;
    let p11 = *grid
        .get(grid_index(n_u, i + 1, j + 1))
        .ok_or_else(|| "coons: grid index out of range".to_string())?;
    let p01 = *grid
        .get(grid_index(n_u, i, j + 1))
        .ok_or_else(|| "coons: grid index out of range".to_string())?;
    Ok([p00, p10, p11, p01])
}

/// Signed area of a quad, positive for counter-clockwise winding.
pub fn signed_area(quad: &[[f64; 2]; 4]) -> f64 {
    let mut twice = 0.0;
    for i in 0..4 {
        let p = quad[i];
        let q = quad[(i + 1) % 4];
        twice += p[0] * q[1] - q[0] * p[1];
    }
    0.5 * twice
}

fn uniform_parameters(n: usize) -> Vec<f64> {
    let mut parameters = Vec::with_capacity(n + 1);
    for i in 0..=n {
        parameters.push(i as f64 / n as f64);
    }
    parameters
}

/// Build the Coons patch quads with caller-supplied parameter seams.
///
/// `bottom_u[i]` and `top_u[i]` are the `u` parameters at which the `i`-th
/// vertical logical line meets the bottom and top curves; `left_v[j]` and
/// `right_v[j]` are the `v` parameters at which the `j`-th horizontal logical
/// line meets the left and right curves. Each array must start at 0, end at 1,
/// and be non-decreasing. When the two arrays of a pair differ, the corresponding
/// logical lines are slanted (`v0 != v1` in Section 5.1.3), which is how chord
/// drift is represented. This is the general form of the `n_u x n_v` mesh; use
/// [`coons_curves`] for uniformly spaced seams.
pub fn coons_from_parameters(
    curves: &BoundaryCurves,
    bottom_u: &[f64],
    top_u: &[f64],
    left_v: &[f64],
    right_v: &[f64],
) -> Result<Vec<[[f64; 2]; 4]>, String> {
    let arcs = checked_curves(curves)?;
    let (grid, n_u, n_v) = coons_grid(&arcs, bottom_u, top_u, left_v, right_v)?;
    let mut quads = Vec::with_capacity(n_u * n_v);
    for j in 0..n_v {
        for i in 0..n_u {
            quads.push(grid_quad(&grid, n_u, i, j)?);
        }
    }
    Ok(quads)
}

/// Mesh a tile with uniformly spaced parameter seams.
///
/// This is [`coons_from_parameters`] with
/// `bottom_u = top_u = [0, 1/n_u, ..., 1]` and
/// `left_v = right_v = [0, 1/n_v, ..., 1]`.
pub fn coons_curves(
    curves: &BoundaryCurves,
    n_u: usize,
    n_v: usize,
) -> Result<Vec<[[f64; 2]; 4]>, String> {
    if n_u == 0 || n_v == 0 {
        return Err("coons: n_u and n_v must both be at least one".to_string());
    }
    let bottom_u = uniform_parameters(n_u);
    let left_v = uniform_parameters(n_v);
    coons_from_parameters(curves, &bottom_u, &bottom_u, &left_v, &left_v)
}

/// Checked form of [`coons`]: parse a closed boundary loop and emit the quads.
pub fn coons_checked(
    points: &[[f64; 2]],
    n_u: usize,
    n_v: usize,
) -> Result<Vec<[[f64; 2]; 4]>, String> {
    let curves = BoundaryCurves::from_loop(points)?;
    coons_curves(&curves, n_u, n_v)
}

/// Emit the `n_u * n_v` Coons-patch quads of a tile.
///
/// `points` is a closed counter-clockwise boundary loop encoded as four
/// equal-length polyline chunks (bottom, right, top, left), or exactly the four
/// corners (bottom-left, bottom-right, top-right, top-left). See
/// [`BoundaryCurves::from_loop`]. Use [`coons_checked`] when the error message
/// matters, or [`coons_curves`] to pass four independently sampled curves.
///
/// Invalid input yields an empty vector; this function never panics.
pub fn coons(points: &[[f64; 2]], n_u: usize, n_v: usize) -> Vec<[[f64; 2]; 4]> {
    coons_checked(points, n_u, n_v).unwrap_or_default()
}

// ---------------------------------------------------------------------------
// Mesh output
// ---------------------------------------------------------------------------

/// Build a [`Mesh`] from a validated Coons grid, sharing the grid nodes.
fn mesh_from_grid(grid: &[[f64; 2]], n_u: usize, n_v: usize) -> Result<Mesh, String> {
    let mut points: Vec<[f64; 3]> = Vec::with_capacity(grid.len());
    for point in grid {
        points.push([point[0], point[1], 0.0]);
    }
    let mut cells: Vec<Cell> = Vec::with_capacity(n_u * n_v);
    for j in 0..n_v {
        for i in 0..n_u {
            let a = grid_index(n_u, i, j);
            let b = grid_index(n_u, i + 1, j);
            let c = grid_index(n_u, i + 1, j + 1);
            let d = grid_index(n_u, i, j + 1);
            cells.push(Cell::quad4([a, b, c, d]));
        }
    }
    let mesh = Mesh::new(points, cells);
    mesh.validate().map_err(|e| format!("coons: {e}"))?;
    Ok(mesh)
}

/// Mesh a tile into a [`Mesh`] of `quad4` cells with a shared grid.
///
/// `curves` are the four counter-clockwise boundary polylines and `n_u`, `n_v`
/// are the number of quads along the bottom/top and left/right directions, so
/// the mesh has `n_u * n_v` cells and `(n_u + 1) * (n_v + 1)` nodes.
pub fn mesh_tile_curves(curves: &BoundaryCurves, n_u: usize, n_v: usize) -> Result<Mesh, String> {
    if n_u == 0 || n_v == 0 {
        return Err("coons: n_u and n_v must both be at least one".to_string());
    }
    let arcs = checked_curves(curves)?;
    let bottom_u = uniform_parameters(n_u);
    let left_v = uniform_parameters(n_v);
    let (grid, n_u, n_v) = coons_grid(&arcs, &bottom_u, &bottom_u, &left_v, &left_v)?;
    mesh_from_grid(&grid, n_u, n_v)
}

/// Mesh a tile from a closed boundary loop; see [`coons`] for the encoding.
///
/// Returns an error for invalid curves or subdivision counts.
pub fn mesh_tile(points: &[[f64; 2]], n_u: usize, n_v: usize) -> Result<Mesh, String> {
    let curves = BoundaryCurves::from_loop(points)?;
    mesh_tile_curves(&curves, n_u, n_v)
}

// ---------------------------------------------------------------------------
// Demo numbers (asserted by the tests and reported by the caller)
// ---------------------------------------------------------------------------

/// Numbers measured by [`demo`].
#[derive(Clone, Debug, Default)]
pub struct DemoReport {
    /// Number of quads of the 3 x 3 square tile.
    pub square_quads: usize,
    /// Smallest signed quad area of the square tile (should be 1/9).
    pub square_min_area: f64,
    /// Integer counts of the two-region shared-side graph.
    pub shared_counts: Vec<usize>,
    /// Residual energy of the shared-side solution (should be 0).
    pub shared_energy: f64,
    /// Integer counts of the opposite-ideal mismatch tile.
    pub mismatch_counts: Vec<usize>,
    /// Energy of the mismatch tile before the solve (all counts zero).
    pub mismatch_energy_before: f64,
    /// Energy of the mismatch tile after the solve.
    pub mismatch_energy_after: f64,
    /// Number of quads of the 4 x 2 bent annular tile.
    pub bent_quads: usize,
    /// Smallest signed quad area of the bent tile.
    pub bent_min_area: f64,
}

fn min_signed_area(quads: &[[[f64; 2]; 4]]) -> Option<f64> {
    quads.iter().map(signed_area).reduce(f64::min)
}

fn two_region_graph() -> [Region; 2] {
    [
        Region::new([vec![0], vec![1], vec![2], vec![3]]),
        Region::new([vec![4], vec![5], vec![6], vec![1]]),
    ]
}

fn bent_curves() -> BoundaryCurves {
    const SAMPLES: usize = 16;
    let bottom = vec![[1.0, 0.0], [2.0, 0.0]];
    let right: Vec<[f64; 2]> = (0..=SAMPLES)
        .map(|k| {
            let angle = std::f64::consts::FRAC_PI_2 * (k as f64) / (SAMPLES as f64);
            [2.0 * angle.cos(), 2.0 * angle.sin()]
        })
        .collect();
    let top = vec![[0.0, 2.0], [0.0, 1.0]];
    let left: Vec<[f64; 2]> = (0..=SAMPLES)
        .map(|k| {
            let angle = std::f64::consts::FRAC_PI_2 * (1.0 - (k as f64) / (SAMPLES as f64));
            [angle.cos(), angle.sin()]
        })
        .collect();
    BoundaryCurves {
        bottom,
        right,
        top,
        left,
    }
}

/// Run the four reference cases and return their measured numbers.
///
/// The cases are: a 3 x 3 square tile; two adjacent regions sharing one side;
/// a tile whose opposite ideal counts disagree (5 on one side, 1 on the other);
/// and a bent annular tile meshed 4 x 2. The tests assert these numbers and the
/// module report quotes them.
pub fn demo() -> DemoReport {
    let mut report = DemoReport::default();

    let square = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    let square_quads = coons(&square, 3, 3);
    if let Some(min_area) = min_signed_area(&square_quads) {
        report.square_quads = square_quads.len();
        report.square_min_area = min_area;
    }

    let shared_ideal = [4.0, 2.0, 4.0, 2.0, 2.0, 2.0, 2.0];
    if let Ok(counts) = solve_counts(&two_region_graph(), &shared_ideal) {
        if let Ok(value) = energy(&counts, &shared_ideal) {
            report.shared_counts = counts;
            report.shared_energy = value;
        }
    }

    let mismatch_ideal = [5.0, 3.0, 1.0, 3.0];
    let mismatch_regions = [Region::new([vec![0], vec![1], vec![2], vec![3]])];
    let zero = [0usize; 4];
    if let Ok(before) = energy(&zero, &mismatch_ideal) {
        report.mismatch_energy_before = before;
        if let Ok(counts) = solve_counts(&mismatch_regions, &mismatch_ideal) {
            if let Ok(after) = energy(&counts, &mismatch_ideal) {
                report.mismatch_counts = counts;
                report.mismatch_energy_after = after;
            }
        }
    }

    if let Ok(quads) = coons_curves(&bent_curves(), 4, 2) {
        if let Some(min_area) = min_signed_area(&quads) {
            report.bent_quads = quads.len();
            report.bent_min_area = min_area;
        }
    }

    report
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_positive(quads: &[[[f64; 2]; 4]]) {
        for (i, quad) in quads.iter().enumerate() {
            let area = signed_area(quad);
            assert!(area > 0.0, "quad {i} has non-positive area {area}");
        }
    }

    #[test]
    fn square_tile_three_by_three() -> Result<(), String> {
        let corners = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let quads = coons(&corners, 3, 3);
        assert_eq!(quads.len(), 9, "a 3 x 3 tile must emit 9 quads");
        assert_positive(&quads);

        let min_area = min_signed_area(&quads).ok_or("no quads emitted")?;
        assert!(
            (min_area - 1.0 / 9.0).abs() < 1e-12,
            "square quad area {min_area} should be 1/9"
        );

        let mesh = mesh_tile(&corners, 3, 3)?;
        assert_eq!(mesh.cells.len(), 9);
        assert_eq!(mesh.points.len(), 16);
        mesh.validate().map_err(|e| e.to_string())?;
        Ok(())
    }

    #[test]
    fn two_regions_share_a_side() -> Result<(), String> {
        let regions = two_region_graph();
        let ideal = [4.0, 2.0, 4.0, 2.0, 2.0, 2.0, 2.0];
        let counts = solve_counts(&regions, &ideal)?;

        assert!(
            constraints_hold(&regions, &counts),
            "eq. (5.1) must hold on both regions"
        );
        assert_eq!(counts, vec![4, 2, 4, 2, 2, 2, 2]);
        // Edge 1 is the shared side: region B's left sum (edge 5's count) and
        // region A's right sum (edge 1's count) are the same variable.
        assert_eq!(
            counts[1], counts[5],
            "the shared side must have one consistent count"
        );
        let residual = energy(&counts, &ideal)?;
        assert!(residual < 1e-24, "residual energy {residual}");
        Ok(())
    }

    #[test]
    fn augmenting_path_balances_opposite_sides() -> Result<(), String> {
        let regions = [Region::new([vec![0], vec![1], vec![2], vec![3]])];
        let ideal = [5.0, 3.0, 1.0, 3.0];
        let zero = [0usize; 4];
        let before = energy(&zero, &ideal)?;
        let counts = solve_counts(&regions, &ideal)?;
        let after = energy(&counts, &ideal)?;

        // The constraint forces s0 == s2, so the 5 and the 1 cannot both be
        // honoured; the crossing path moves *both* sides to 1.
        assert_eq!(counts, vec![1, 3, 1, 3]);
        assert!(counts[0] > 0 && counts[2] > 0);
        assert!(counts[0] == counts[2]);
        assert!(after < before, "energy must drop: {before} -> {after}");
        assert!(
            (after - 0.64).abs() < 1e-12,
            "mismatch energy {after} should be 0.64"
        );
        Ok(())
    }

    #[test]
    fn loop_augmentation_changes_a_cycle() -> Result<(), String> {
        // One tile with e0 on sides 0 and 1 and e1 on sides 2 and 3: the two
        // constraints are e0 == e1, and both edges have degree two. No open
        // path exists, so only a loop augmentation can move the counts.
        let regions = [Region::new([vec![0], vec![0], vec![1], vec![1]])];
        let counts = solve_counts(&regions, &[2.0, 2.0])?;
        assert_eq!(counts, vec![2, 2]);
        assert!(constraints_hold(&regions, &counts));
        Ok(())
    }

    #[test]
    fn bent_tile_coons_quads_are_valid() -> Result<(), String> {
        let curves = bent_curves();
        let quads = coons_curves(&curves, 4, 2)?;
        assert_eq!(quads.len(), 8);
        assert_positive(&quads);
        let min_area = min_signed_area(&quads).ok_or("no quads emitted")?;
        assert!(min_area > 0.19, "bent-tile minimum area {min_area}");

        let mesh = mesh_tile_curves(&curves, 4, 2)?;
        assert_eq!(mesh.cells.len(), 8);
        assert_eq!(mesh.points.len(), 15);
        mesh.validate().map_err(|e| e.to_string())?;
        Ok(())
    }

    #[test]
    fn drifter_seams_follow_logical_lines() -> Result<(), String> {
        // A unit square with a shifted interior seam: the logical grid line is
        // f(u) = (u, lerp(u, v0, v1)), not a constant-v line.
        let curves = BoundaryCurves::from_corners([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]);
        let quads = coons_from_parameters(
            &curves,
            &[0.0, 0.3, 1.0],
            &[0.0, 0.7, 1.0],
            &[0.0, 1.0],
            &[0.0, 1.0],
        )?;
        assert_eq!(quads.len(), 2);
        assert_positive(&quads);
        // The middle vertical line starts at x=0.3 on the bottom and ends at
        // x=0.7 on the top, so the quad corner between the two columns sits at
        // x = 0.3 (bottom) and x = 0.7 (top).
        let middle_bottom = quads[0][1];
        let middle_top = quads[0][2];
        assert!((middle_bottom[0] - 0.3).abs() < 1e-9);
        assert!((middle_top[0] - 0.7).abs() < 1e-9);
        Ok(())
    }

    #[test]
    fn invalid_inputs_are_errors_not_panics() {
        // Non-positive ideal count.
        let regions = [Region::new([vec![0], vec![1], vec![2], vec![3]])];
        assert!(solve_counts(&regions, &[0.0, 1.0, 1.0, 1.0]).is_err());

        // Edge id out of range.
        assert!(solve_counts(&regions, &[1.0, 1.0, 1.0]).is_err());

        // Self-referential (nautilus) side.
        let nautilus = [Region::new([vec![0], vec![1], vec![0, 1], vec![2]])];
        assert!(solve_counts(&nautilus, &[1.0, 1.0, 1.0]).is_err());

        // Clockwise boundary is rejected by the checked constructor.
        let clockwise = [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]];
        assert!(coons_checked(&clockwise, 2, 2).is_err());
        assert!(coons(&clockwise, 2, 2).is_empty());
    }

    #[test]
    fn demo_numbers() -> Result<(), String> {
        let report = demo();
        assert_eq!(report.square_quads, 9);
        assert!((report.square_min_area - 1.0 / 9.0).abs() < 1e-12);
        assert_eq!(report.shared_counts, vec![4, 2, 4, 2, 2, 2, 2]);
        assert!(report.shared_energy < 1e-24);
        assert_eq!(report.mismatch_counts, vec![1, 3, 1, 3]);
        assert!((report.mismatch_energy_before - 4.0).abs() < 1e-12);
        assert!((report.mismatch_energy_after - 0.64).abs() < 1e-12);
        assert_eq!(report.bent_quads, 8);
        assert!(report.bent_min_area > 0.19);
        println!(
            "QGP demo: square {} quads (min area {:.9}), shared counts {:?} \
             (energy {:.3e}), mismatch {:?} ({:.3} -> {:.3}), bent {} quads \
             (min area {:.6})",
            report.square_quads,
            report.square_min_area,
            report.shared_counts,
            report.shared_energy,
            report.mismatch_counts,
            report.mismatch_energy_before,
            report.mismatch_energy_after,
            report.bent_quads,
            report.bent_min_area
        );
        Ok(())
    }

    #[test]
    fn multi_edge_sides_are_summed_not_single_ids() -> Result<(), String> {
        // Reproducer: region A's shared right side is split into two edges
        // e0,e1, and region B's left side is the same two edges. Before the
        // used-constraint guard this ended with the final error
        // "constraint 3 is not balanced after the solve (2 != 1)".
        let split = [
            Region::new([vec![2], vec![0, 1], vec![3], vec![4]]),
            Region::new([vec![6], vec![5], vec![7], vec![0, 1]]),
        ];
        let split_ideal = [2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0];
        let split_counts = solve_counts(&split, &split_ideal)?;
        assert!(constraints_hold(&split, &split_counts));
        let shared_split = split_counts[0] + split_counts[1];
        assert_eq!(shared_split, split_counts[4], "A: e0+e1 = e4");
        assert_eq!(shared_split, split_counts[5], "B: e5 = e0+e1");

        // Collapse that two-edge side to a single edge whose ideal is the sum
        // of the two segment ideals. One edge and two edges must produce the
        // same side sums and a balanced result.
        let coarse = [
            Region::new([vec![1], vec![0], vec![2], vec![3]]),
            Region::new([vec![5], vec![4], vec![6], vec![0]]),
        ];
        let coarse_ideal = [4.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0];
        let coarse_counts = solve_counts(&coarse, &coarse_ideal)?;
        assert!(constraints_hold(&coarse, &coarse_counts));
        assert_eq!(shared_split, coarse_counts[0], "shared side sum");
        assert_eq!(split_counts[2], coarse_counts[1], "A bottom");
        assert_eq!(split_counts[3], coarse_counts[2], "A top");
        assert_eq!(split_counts[4], coarse_counts[3], "A left");
        assert_eq!(split_counts[5], coarse_counts[4], "B right");
        assert_eq!(split_counts[6], coarse_counts[5], "B bottom");
        assert_eq!(split_counts[7], coarse_counts[6], "B top");

        assert_eq!(split_counts, vec![1, 0, 1, 1, 1, 1, 1, 1]);
        assert_eq!(coarse_counts, vec![1, 1, 1, 1, 1, 1, 1]);
        assert!(energy(&split_counts, &split_ideal)?.is_finite());
        Ok(())
    }

    #[test]
    fn multi_edge_solves_always_balance() -> Result<(), String> {
        // Regression sweep over the split-side graph: every solve must return
        // non-negative counts that satisfy both side-sum constraints.
        let regions = [
            Region::new([vec![2], vec![0, 1], vec![3], vec![4]]),
            Region::new([vec![6], vec![5], vec![7], vec![0, 1]]),
        ];
        for i0 in 1..=6 {
            for i1 in 1..=6 {
                for i4 in 1..=6 {
                    for i5 in 1..=6 {
                        let ideal = [
                            i0 as f64, i1 as f64, 1.0, 1.0, i4 as f64, i5 as f64, 1.0, 1.0,
                        ];
                        let counts = solve_counts(&regions, &ideal)?;
                        assert!(
                            constraints_hold(&regions, &counts),
                            "unbalanced solve for ideals {ideal:?}: {counts:?}"
                        );
                    }
                }
            }
        }
        Ok(())
    }
}
