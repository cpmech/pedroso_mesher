//! Hybrid fast indirect quad meshing: Blossom-Quad on the boundary band and
//! greedy triangle merging in the interior.
//!
//! This implements the hybrid scheme of Zint and Grosso (2021), *A hybrid
//! approach to fast indirect quadrilateral mesh generation*, using the shipped
//! Blossom-Quad matching of Remacle et al. (2012) for the boundary band. The
//! steps, in the order the route runs them:
//!
//! 1. [`triangulate`] (constrained Delaunay + Ruppert) gives the triangle
//!    substrate. Its size is the one the caller asks for.
//! 2. **Segmentation** (paper Sect. 3.1): every boundary vertex starts a
//!    breadth-first search over the triangle edge graph; a triangle is in the
//!    boundary band when all three of its vertices are at most two edge hops
//!    from the domain boundary. That is the paper's "two layers of triangles".
//!    All boundaries are merged into one band, and the band is made even-sized
//!    by taking the first adjacent interior triangle, as the paper does, so the
//!    matching below has an even number of nodes.
//! 3. **Blossom on the band** (paper Sect. 3.1): [`pair_triangles_with`] with a
//!    stricter guard than the shipped default, because a boundary quad is seen
//!    by the eye and by the stress field.
//! 4. **Greedy merging of the interior** (paper Sect. 3.2): a binary heap of
//!    adjacent triangle pairs, ordered by the minimum scaled Jacobian (then the
//!    minimum corner angle) of the quad each pair would form, with a quality
//!    floor. The best pair is merged, which marks both triangles used, and the
//!    paper's local continuation then tries the edges around the new quad before
//!    jumping to the best edge anywhere. The scores of the other candidates do
//!    not change because merging moves no vertex, so the heaps are genuine
//!    priority queues and the pass is `O(N log N)`. A greedy matching can leave
//!    isolated triangles; a bounded augmenting-path repair (alternating paths of
//!    at most eight dual-graph hops) recovers most of them without a Blossom
//!    solve.
//! 5. **Leftovers** (paper Sect. 3.2/4): the band triangles Blossom could not
//!    pair and the interior triangles the floor rejected are offered to each
//!    other under a looser floor and merged greedily.
//! 6. **Triangle walk** (paper Sect. 3.3): the remaining triangles are isolated
//!    from each other, so one of each pair is walked through adjacent quads with
//!    the merge/split operation of Sect. 3.3 until it touches another residual
//!    triangle, then the two are merged. The walk is a bounded backtracking
//!    search over all quads around the triangle; every intermediate quad keeps
//!    scaled Jacobian >= 0.2 and every move preserves the covered area exactly.
//!    A triangle that still cannot reach a partner under those guards stays a
//!    triangle: this route is quad-dominant, not guaranteed all-quad.
//! 7. **Topological clean-up** (Remacle et al. Sect. 6, paper Sect. 4): a
//!    doublet collapse removes an interior valence-two vertex by merging its two
//!    quads, and valence-driven edge swaps replace the shared edge of two quads
//!    when that lowers the irregularity. Both operations only reuse existing
//!    edges and are guarded by convexity and the gate's scaled-Jacobian floor,
//!    so the mesh stays conforming and watertight.
//!
//! **Not implemented**: vertex splits, extra boundary graph edges with vertex
//! duplication, and discrete mesh optimisation (DMO) smoothing. The common
//! post-pass in `examples/alt_hybrid.rs` does the smoothing for every route it
//! is compared with.
//!
//! Everything here is geometry-preserving: no point is moved and no mesh edge is
//! split. The walk only re-partitions the pentagon covered by one triangle and
//! one quad, so every boundary node stays on the domain boundary, the area is
//! conserved and [`verify::boundary_offset`] measures ~0.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::mesh::{Cell, Mesh, Node};
use crate::quality;
use crate::two_d::{pair_triangles_with, triangulate, Domain, PairGuard, Sizing};

/// Options for the hybrid route.
///
/// The defaults are the constants the route was tuned with; passing
/// [`Options::default`] reproduces the published measurements exactly.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Options {
    /// A triangle belongs to the boundary band when all of its vertices are at
    /// most this many edge hops from the domain boundary.
    pub band_layers: u8,
    /// Depth limit of the augmenting-path search, in dual-graph hops.
    pub augment_depth: usize,
    /// Depth of the backtracking walk, in triangle moves.
    pub walk_depth: usize,
    /// Search states tried per residual triangle before giving up on it.
    pub walk_budget: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            band_layers: BAND_LAYERS,
            augment_depth: AUGMENT_DEPTH,
            walk_depth: WALK_DEPTH,
            walk_budget: WALK_BUDGET,
        }
    }
}

/// A triangle belongs to the boundary band when all of its vertices are at most
/// this many edge hops from the domain boundary. Two hops give the paper's two
/// layers of boundary triangles.
const BAND_LAYERS: u8 = 2;

/// Guard for the boundary-band matching. Stricter than [`PairGuard::default`]
/// (0.05, 10°, 160°): the band is where Blossom-Quad earns its keep, and the
/// paper reports the minimum quality of both routes as coming from the boundary.
const BAND_GUARD: PairGuard = PairGuard {
    min_scaled_jacobian: 0.10,
    min_angle_deg: 15.0,
    max_angle_deg: 160.0,
};

/// A quality floor for candidate quads.
#[derive(Clone, Copy, Debug)]
struct Floor {
    min_scaled_jacobian: f64,
    min_angle_deg: f64,
    max_angle_deg: f64,
}

/// Floor for the interior greedy merge. Chosen to drop a needle (a pair whose
/// union is a sliver) without rejecting an ordinary acceptable quad: the paper's
/// extension keeps merging while quality is reasonable and lets clean-up handle
/// the rest.
const GREEDY_FLOOR: Floor = Floor {
    min_scaled_jacobian: 0.05,
    min_angle_deg: 8.0,
    max_angle_deg: 170.0,
};

/// Floor for merging the triangles left over after the band and interior
/// passes. Looser because a left-over triangle is worse for the product than a
/// mediocre quad, but still refuses a self-intersecting or inverted union.
const LEFTOVER_FLOOR: Floor = Floor {
    min_scaled_jacobian: 0.08,
    min_angle_deg: 8.0,
    max_angle_deg: 170.0,
};

/// A candidate pair of triangles and the quad they would form.
#[derive(Clone, Copy, Debug)]
struct Candidate {
    /// Minimum scaled Jacobian of the quad: the primary key.
    score: f64,
    /// Minimum corner angle of the quad: the tie-break, and a needle guard.
    angle: f64,
    /// Triangle ids and the ready-made quad.
    a: usize,
    b: usize,
    quad: [usize; 4],
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.a == other.a
            && self.b == other.b
            && self.score.to_bits() == other.score.to_bits()
            && self.angle.to_bits() == other.angle.to_bits()
    }
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap: higher score and angle first. Ties fall back
        // to the triangle ids (reversed, so the smaller id pops first) and make
        // the queue a total order, which keeps the route deterministic.
        self.score
            .partial_cmp(&other.score)
            .unwrap_or(Ordering::Equal)
            .then(
                self.angle
                    .partial_cmp(&other.angle)
                    .unwrap_or(Ordering::Equal),
            )
            .then(other.a.cmp(&self.a))
            .then(other.b.cmp(&self.b))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Counters for one hybrid run, for the tests and for diagnosing a run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HybridStats {
    /// Triangles produced by the triangulation.
    pub triangles: usize,
    /// Triangles in the boundary band (after the parity adjustment).
    pub band_triangles: usize,
    /// Quads the band matching produced.
    pub band_quads: usize,
    /// Quads the interior greedy merge produced.
    pub greedy_quads: usize,
    /// Quads the leftover merge produced.
    pub leftover_quads: usize,
    /// Quads removed by a doublet collapse.
    pub doublet_collapses: usize,
    /// Quad edge swaps taken by the valence clean-up.
    pub cleanup_swaps: usize,
    /// Pairs of residual triangles joined by the dual-graph triangle walk.
    pub walk_merges: usize,
    /// Triangles that stayed triangles.
    pub leftover_triangles: usize,
}

impl HybridStats {
    /// Share of planar elements that are quads.
    pub fn quad_ratio(&self) -> f64 {
        let total = self.triangles;
        if total == 0 {
            0.0
        } else {
            (total - self.leftover_triangles) as f64 / total as f64
        }
    }
}

/// Mesh a planar domain with the hybrid route using the default options.
pub fn mesh(domain: &Domain, sizing: &Sizing) -> Result<Mesh, String> {
    mesh_with_options(domain, sizing, &Options::default())
}

/// Mesh a planar domain with the hybrid route and explicit options.
pub fn mesh_with_options(
    domain: &Domain,
    sizing: &Sizing,
    options: &Options,
) -> Result<Mesh, String> {
    mesh_with_options_with_stats(domain, sizing, options).map(|(mesh, _)| mesh)
}

/// Mesh a planar domain with the default options and return the stage counters.
pub fn mesh_with_stats(domain: &Domain, sizing: &Sizing) -> Result<(Mesh, HybridStats), String> {
    mesh_with_options_with_stats(domain, sizing, &Options::default())
}

/// Mesh a planar domain with explicit options and return the stage counters.
pub fn mesh_with_options_with_stats(
    domain: &Domain,
    sizing: &Sizing,
    options: &Options,
) -> Result<(Mesh, HybridStats), String> {
    if domain.area() <= 0.0 {
        return Err(
            "the domain has no area (outer loop clockwise, or a hole larger than it)".into(),
        );
    }

    let (points, tris) = triangulate(domain, sizing)
        .map_err(|error| format!("the hybrid route could not triangulate the domain: {error}"))?;
    if tris.is_empty() {
        return Err("the domain triangulated to nothing".into());
    }

    let mut stats = HybridStats {
        triangles: tris.len(),
        ..HybridStats::default()
    };

    // ---------------------------------------------------------------- band --
    let boundary = boundary_vertices(&tris, points.len());
    let hop = boundary_hops(&tris, &boundary, options);
    let mut in_band = vec![false; tris.len()];
    for (t, tri) in tris.iter().enumerate() {
        if tri.iter().all(|&v| hop[v] <= options.band_layers) {
            in_band[t] = true;
        }
    }
    // The paper wants an even number of triangles per boundary region. One band
    // is used here, so one adjacent interior triangle is taken when the band is
    // odd. It is deterministic: the first band triangle and then the first
    // interior neighbour in index order.
    if in_band.iter().filter(|&&b| b).count() % 2 == 1 {
        if let Some(t) = first_adjacent_interior(&tris, &in_band) {
            in_band[t] = true;
        }
    }
    let band_ids: Vec<usize> = (0..tris.len()).filter(|&t| in_band[t]).collect();
    stats.band_triangles = band_ids.len();

    let band_tris: Vec<[usize; 3]> = band_ids.iter().map(|&t| tris[t]).collect();
    let (mut quads, band_left) = pair_triangles_with(&band_tris, &points, BAND_GUARD);
    stats.band_quads = quads.len();
    let band_left_ids = map_leftovers(&tris, &band_ids, &band_left);

    // ------------------------------------------------------------ interior --
    let interior_ids: Vec<usize> = (0..tris.len()).filter(|&t| !in_band[t]).collect();
    let (interior_quads, interior_left) =
        greedy_merge(&points, &tris, &interior_ids, GREEDY_FLOOR, options);
    stats.greedy_quads = interior_quads.len();
    quads.extend(interior_quads);

    // ------------------------------------------------------------ leftovers --
    let mut leftover_ids = band_left_ids;
    leftover_ids.extend(interior_left);
    leftover_ids.sort_unstable();
    let (leftover_quads, still_left) =
        greedy_merge(&points, &tris, &leftover_ids, LEFTOVER_FLOOR, options);
    stats.leftover_quads = leftover_quads.len();
    quads.extend(leftover_quads);
    stats.leftover_triangles = still_left.len();
    let leftover_tris: Vec<[usize; 3]> = still_left.iter().map(|&t| tris[t]).collect();

    // ------------------------------------------------------------ residuals --
    let mut slot_quads: Vec<Option<[usize; 4]>> = quads.into_iter().map(Some).collect();
    let mut tri_slots: Vec<Option<[usize; 3]>> = leftover_tris.iter().copied().map(Some).collect();
    stats.walk_merges = merge_residuals(&points, &mut slot_quads, &mut tri_slots, options);
    let leftover_tris: Vec<[usize; 3]> = tri_slots.iter().flatten().copied().collect();
    stats.leftover_triangles = leftover_tris.len();

    // -------------------------------------------------------------- cleanup --
    let boundary = boundary_vertices_all(&slot_quads, &leftover_tris);
    let doublet_collapses = collapse_doublets(
        &points,
        &mut slot_quads,
        &leftover_tris,
        &boundary,
        sizing.size,
    );
    let cleanup_swaps = swap_edges(&points, &mut slot_quads, &leftover_tris, &boundary);
    // A swap can leave a vertex at valence two; one more collapse round is cheap.
    let doublet_collapses = doublet_collapses
        + collapse_doublets(
            &points,
            &mut slot_quads,
            &leftover_tris,
            &boundary,
            sizing.size,
        );
    stats.doublet_collapses = doublet_collapses;
    stats.cleanup_swaps = cleanup_swaps;

    // ---------------------------------------------------------------- output --
    let mut cells: Vec<Cell> = slot_quads
        .iter()
        .flatten()
        .map(|quad| Cell::quad4(*quad))
        .collect();
    cells.extend(leftover_tris.iter().map(|tri| Cell::tri3(*tri)));
    Ok((Mesh::new(points, cells), stats))
}

/// Vertex ids used by a boundary edge (an edge owned by exactly one triangle).
fn boundary_vertices(tris: &[[usize; 3]], npoints: usize) -> Vec<bool> {
    let mut count: HashMap<(usize, usize), usize> = HashMap::new();
    for tri in tris {
        for (a, b) in triangle_edges(tri) {
            *count.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    let mut boundary = vec![false; npoints];
    for ((a, b), uses) in count {
        if uses == 1 {
            if a < npoints {
                boundary[a] = true;
            }
            if b < npoints {
                boundary[b] = true;
            }
        }
    }
    boundary
}

/// Boundary vertices of a mesh made of the given quads and triangles.
fn boundary_vertices_all(quads: &[Option<[usize; 4]>], tris: &[[usize; 3]]) -> Vec<bool> {
    let mut count: HashMap<(usize, usize), usize> = HashMap::new();
    for quad in quads.iter().flatten() {
        for i in 0..4 {
            let (a, b) = (quad[i], quad[(i + 1) % 4]);
            *count.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    for tri in tris {
        for (a, b) in triangle_edges(tri) {
            *count.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    let npoints = quads
        .iter()
        .flatten()
        .flat_map(|q| q.iter().copied())
        .chain(tris.iter().flat_map(|t| t.iter().copied()))
        .max()
        .map_or(0, |m| m + 1);
    let mut boundary = vec![false; npoints];
    for ((a, b), uses) in count {
        if uses == 1 {
            if a < npoints {
                boundary[a] = true;
            }
            if b < npoints {
                boundary[b] = true;
            }
        }
    }
    boundary
}

/// Number of edge hops from the nearest boundary vertex, for every vertex.
fn boundary_hops(tris: &[[usize; 3]], boundary: &[bool], options: &Options) -> Vec<u8> {
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); boundary.len()];
    for tri in tris {
        for i in 0..3 {
            let (a, b) = (tri[i], tri[(i + 1) % 3]);
            adjacency[a].push(b);
            adjacency[b].push(a);
        }
    }
    for ring in adjacency.iter_mut() {
        ring.sort_unstable();
        ring.dedup();
    }
    let unreached = u8::MAX;
    let mut hop = vec![unreached; boundary.len()];
    let mut queue = VecDeque::new();
    for (v, &on_boundary) in boundary.iter().enumerate() {
        if on_boundary {
            hop[v] = 0;
            queue.push_back(v);
        }
    }
    while let Some(v) = queue.pop_front() {
        let next = hop[v].saturating_add(1);
        if hop[v] >= options.band_layers {
            continue;
        }
        for &u in &adjacency[v] {
            if hop[u] == unreached {
                hop[u] = next;
                queue.push_back(u);
            }
        }
    }
    hop
}

/// First interior triangle adjacent (through a shared edge) to the band.
fn first_adjacent_interior(tris: &[[usize; 3]], in_band: &[bool]) -> Option<usize> {
    let mut edge_owners: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (t, tri) in tris.iter().enumerate() {
        for (a, b) in triangle_edges(tri) {
            edge_owners.entry((a.min(b), a.max(b))).or_default().push(t);
        }
    }
    for (t, tri) in tris.iter().enumerate() {
        if !in_band[t] {
            continue;
        }
        for (a, b) in triangle_edges(tri) {
            if let Some(owners) = edge_owners.get(&(a.min(b), a.max(b))) {
                for &other in owners {
                    if other != t && !in_band[other] {
                        return Some(other);
                    }
                }
            }
        }
    }
    None
}

/// Map the triangles a pairing pass returned as unpaired back to triangle ids.
fn map_leftovers(tris: &[[usize; 3]], ids: &[usize], leftover: &[[usize; 3]]) -> Vec<usize> {
    let keys: HashSet<[usize; 3]> = leftover.iter().map(sorted_key).collect();
    ids.iter()
        .copied()
        .filter(|&t| keys.contains(&sorted_key(&tris[t])))
        .collect()
}

fn sorted_key(tri: &[usize; 3]) -> [usize; 3] {
    let mut key = *tri;
    key.sort_unstable();
    key
}

/// Merge adjacent triangles from `ids` greedily, best quad first.
///
/// Only pairs that clear `floor` enter the graph. The heap is a genuine priority
/// queue: merging a pair uses its triangles but does not change the geometry of
/// any other candidate, so no score has to be recomputed. After the greedy pass
/// a bounded augmenting-path search (the classic greedy-matching repair: an
/// unmatched triangle, a matched pair, an unmatched triangle, flipped) raises
/// the cardinality with only local searches, never with a global Blossom solve.
/// Returns the quads formed and the ids that stayed triangles.
fn greedy_merge(
    points: &[Node],
    tris: &[[usize; 3]],
    ids: &[usize],
    floor: Floor,
    options: &Options,
) -> (Vec<[usize; 4]>, Vec<usize>) {
    let mut edge_owners: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for &t in ids {
        for (a, b) in triangle_edges(&tris[t]) {
            edge_owners.entry((a.min(b), a.max(b))).or_default().push(t);
        }
    }
    let mut candidates: HashMap<(usize, usize), Candidate> = HashMap::new();
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); tris.len()];
    for owners in edge_owners.values() {
        if owners.len() != 2 {
            continue;
        }
        if let Some(candidate) = candidate_for(points, tris, owners[0], owners[1], floor) {
            candidates.insert((candidate.a, candidate.b), candidate);
            adjacency[candidate.a].push(candidate.b);
            adjacency[candidate.b].push(candidate.a);
        }
    }
    for ring in adjacency.iter_mut() {
        ring.sort_unstable();
        ring.dedup();
    }

    let mut mate = vec![usize::MAX; tris.len()];
    let mut local: BinaryHeap<Candidate> = BinaryHeap::new();
    let mut global: BinaryHeap<Candidate> = candidates.values().copied().collect();
    loop {
        // Prefer the wave around the quad just created; when it is empty, jump
        // to the best candidate left anywhere. Both heaps are lazy: a stale
        // candidate is simply skipped when it comes out.
        let mut next: Option<Candidate> = None;
        while let Some(candidate) = local.pop() {
            if mate[candidate.a] == usize::MAX && mate[candidate.b] == usize::MAX {
                next = Some(candidate);
                break;
            }
        }
        if next.is_none() {
            while let Some(candidate) = global.pop() {
                if mate[candidate.a] == usize::MAX && mate[candidate.b] == usize::MAX {
                    next = Some(candidate);
                    break;
                }
            }
        }
        let Some(candidate) = next else {
            break;
        };
        mate[candidate.a] = candidate.b;
        mate[candidate.b] = candidate.a;

        // The paper's local continuation: the triangles just outside the new
        // quad, and their other edges, are the next candidates to try.
        let mut outside: Vec<usize> = Vec::new();
        for inside in [candidate.a, candidate.b] {
            for &u in &adjacency[inside] {
                if mate[u] == usize::MAX
                    && u != candidate.a
                    && u != candidate.b
                    && !outside.contains(&u)
                {
                    outside.push(u);
                }
            }
        }
        outside.sort_unstable();
        for u in outside {
            for &w in &adjacency[u] {
                if mate[w] == usize::MAX {
                    if let Some(next) = candidates.get(&(u.min(w), u.max(w))) {
                        local.push(*next);
                    }
                }
            }
        }
    }

    // Repair the greedy matching with bounded augmenting paths. This is where
    // most of the left-over triangles are recovered without a Blossom solve.
    let mut augmenter = Augmenter {
        adjacency: &adjacency,
        mate: &mut mate,
        visited: vec![0u32; tris.len()],
        stamp: 0,
        depth_limit: options.augment_depth,
    };
    for _ in 0..AUGMENT_ROUNDS {
        let mut improved = false;
        let starts: Vec<usize> = ids
            .iter()
            .copied()
            .filter(|&t| augmenter.mate[t] == usize::MAX)
            .collect();
        for start in starts {
            if augmenter.mate[start] != usize::MAX {
                continue;
            }
            if augmenter.augment_from(start) {
                improved = true;
            }
        }
        if !improved {
            break;
        }
    }
    drop(augmenter);

    let mut quads = Vec::new();
    for &t in ids {
        let partner = mate[t];
        if partner == usize::MAX || partner < t {
            continue;
        }
        if let Some(candidate) = candidates.get(&(t, partner)) {
            quads.push(candidate.quad);
        }
    }
    let leftover = ids
        .iter()
        .copied()
        .filter(|&t| mate[t] == usize::MAX)
        .collect();
    (quads, leftover)
}

/// Depth limit of the augmenting-path search, in dual-graph hops. Local repair
/// is enough; a long search would start to look like the Blossom cost the hybrid
/// route is trying to avoid.
const AUGMENT_DEPTH: usize = 8;
/// Number of full augmenting passes over the unmatched triangles.
const AUGMENT_ROUNDS: usize = 6;

/// Bounded augmenting-path repair for the greedy matching.
struct Augmenter<'a> {
    adjacency: &'a [Vec<usize>],
    mate: &'a mut [usize],
    visited: Vec<u32>,
    stamp: u32,
    depth_limit: usize,
}

impl Augmenter<'_> {
    /// Find an augmenting path from `start` and, if one exists, flip the
    /// matching along it: the two previously unmatched endpoints become matched
    /// and each matched pair on the path is re-paired with its neighbours.
    fn augment_from(&mut self, start: usize) -> bool {
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.visited.fill(0);
            self.stamp = 1;
        }
        self.visited[start] = self.stamp;
        let mut path: Vec<usize> = Vec::new();
        if !self.find(start, start, &mut path, 0) {
            return false;
        }
        let mut chain: Vec<usize> = Vec::with_capacity(path.len() + 1);
        chain.push(start);
        chain.extend(path);
        let mut index = 0;
        while index + 1 < chain.len() {
            let (a, b) = (chain[index], chain[index + 1]);
            self.mate[a] = b;
            self.mate[b] = a;
            index += 2;
        }
        true
    }

    /// Depth-first alternating search. `path` receives
    /// `v1, mate(v1), v2, mate(v2), ..., vk` with `vk` unmatched.
    fn find(&mut self, current: usize, start: usize, path: &mut Vec<usize>, depth: usize) -> bool {
        if depth > self.depth_limit {
            return false;
        }
        for k in 0..self.adjacency[current].len() {
            let v = self.adjacency[current][k];
            if v == self.mate[current] || self.visited[v] == self.stamp {
                continue;
            }
            if self.mate[v] == usize::MAX {
                if v == start {
                    continue;
                }
                path.push(v);
                return true;
            }
            let next = self.mate[v];
            if next == usize::MAX || next == current || self.visited[next] == self.stamp {
                continue;
            }
            // Mark the inner triangle and the outer triangle it leads to:
            // without this the search can enter the same matched pair twice and
            // return a path with a repeated vertex, corrupting the matching.
            self.visited[v] = self.stamp;
            self.visited[next] = self.stamp;
            path.push(v);
            path.push(next);
            if self.find(next, start, path, depth + 1) {
                return true;
            }
            path.pop();
            path.pop();
        }
        false
    }
}

/// The quad two triangles would form, if it clears `floor`.
fn candidate_for(
    points: &[Node],
    tris: &[[usize; 3]],
    t1: usize,
    t2: usize,
    floor: Floor,
) -> Option<Candidate> {
    let quad = quad_from_pair(&tris[t1], &tris[t2], points)?;
    let coords = quad_coords(&quad, points);
    if !convex_ccw(&coords) {
        return None;
    }
    // The quad must cover exactly the two triangles: a diagonal that leaves the
    // union would overlap a neighbour.
    let union_area = triangle_area(&tris[t1], points) + triangle_area(&tris[t2], points);
    let quad_area = polygon_signed_area(&quad, points);
    if (quad_area - union_area).abs() > 1e-9 * (1.0 + union_area.abs()) {
        return None;
    }
    let cell = Cell::quad4(quad);
    let sj = quality::scaled_jacobian(&cell, points)
        .into_iter()
        .fold(f64::INFINITY, f64::min);
    if !sj.is_finite() || sj < floor.min_scaled_jacobian {
        return None;
    }
    let angles = quality::quad_angles(&cell, points);
    let min_angle = angles.iter().copied().fold(f64::INFINITY, f64::min);
    let max_angle = angles.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !min_angle.is_finite() || min_angle < floor.min_angle_deg || max_angle > floor.max_angle_deg
    {
        return None;
    }
    Some(Candidate {
        score: sj,
        angle: min_angle,
        a: t1.min(t2),
        b: t1.max(t2),
        quad,
    })
}

fn triangle_edges(tri: &[usize; 3]) -> [(usize, usize); 3] {
    [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]
}

/// Build the quad that is the union of two triangles sharing an edge.
///
/// The shared edge is traversed `u -> v` by the first triangle and `v -> u` by
/// the second, so with `ca` and `cb` the two opposite vertices the union is the
/// cycle `u -> cb -> v -> ca`. This is the actual union polygon (it can be
/// concave); the caller rejects it unless it is convex and its area is exactly
/// the sum of the two triangle areas, which is what keeps the route conforming.
fn quad_from_pair(a: &[usize; 3], b: &[usize; 3], points: &[Node]) -> Option<[usize; 4]> {
    for i in 0..3 {
        let u = a[i];
        let v = a[(i + 1) % 3];
        let ca = a[(i + 2) % 3];
        for j in 0..3 {
            if b[j] != v || b[(j + 1) % 3] != u {
                continue;
            }
            let cb = b[(j + 2) % 3];
            let mut quad = [u, cb, v, ca];
            if polygon_signed_area(&quad, points) < 0.0 {
                quad.reverse();
            }
            if polygon_signed_area(&quad, points).abs() < 1e-18 {
                return None;
            }
            return Some(quad);
        }
    }
    None
}

fn polygon_signed_area(quad: &[usize; 4], points: &[Node]) -> f64 {
    let mut sum = 0.0;
    for i in 0..4 {
        let a = points[quad[i]];
        let b = points[quad[(i + 1) % 4]];
        sum += a[0] * b[1] - b[0] * a[1];
    }
    sum / 2.0
}

fn triangle_area(tri: &[usize; 3], points: &[Node]) -> f64 {
    let a = points[tri[0]];
    let b = points[tri[1]];
    let c = points[tri[2]];
    ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2.0
}

fn quad_coords(quad: &[usize; 4], points: &[Node]) -> [[f64; 2]; 4] {
    [
        [points[quad[0]][0], points[quad[0]][1]],
        [points[quad[1]][0], points[quad[1]][1]],
        [points[quad[2]][0], points[quad[2]][1]],
        [points[quad[3]][0], points[quad[3]][1]],
    ]
}

/// Strictly convex and counter-clockwise.
fn convex_ccw(quad: &[[f64; 2]; 4]) -> bool {
    convex_ccw_poly(quad)
}

fn quad_min_sj(quad: &[usize; 4], points: &[Node]) -> f64 {
    quality::scaled_jacobian(&Cell::quad4(*quad), points)
        .into_iter()
        .fold(f64::INFINITY, f64::min)
}

fn quad_min_angle(quad: &[usize; 4], points: &[Node]) -> f64 {
    quality::quad_angles(&Cell::quad4(*quad), points)
        .into_iter()
        .fold(f64::INFINITY, f64::min)
}

fn edge_key(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

/// Remove interior valence-two vertices by merging their two incident quads.
///
/// Such a vertex is a *doublet*: the two quads share both of its incident edges,
/// so their union is one quad whose four edges already exist in the mesh. The
/// operation therefore removes a vertex and an element without adding an edge or
/// moving a point, and the mesh stays conforming. It is accepted only when the
/// merged quad is convex, keeps the gate's Jacobian floor, does not lose much of
/// the worst quality and does not become longer than the local size allows.
fn collapse_doublets(
    points: &[Node],
    quads: &mut Vec<Option<[usize; 4]>>,
    tris: &[[usize; 3]],
    boundary: &[bool],
    size: f64,
) -> usize {
    let mut collapses = 0usize;
    for _ in 0..8 {
        let mut incident: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut cell_count: HashMap<usize, usize> = HashMap::new();
        for (id, slot) in quads.iter().enumerate() {
            if let Some(quad) = slot {
                for &v in quad {
                    incident.entry(v).or_default().push(id);
                    *cell_count.entry(v).or_insert(0) += 1;
                }
            }
        }
        for tri in tris {
            for &v in tri {
                *cell_count.entry(v).or_insert(0) += 1;
            }
        }
        let mut changed = false;
        for v in 0..points.len() {
            if boundary.get(v).copied().unwrap_or(false) {
                continue;
            }
            if cell_count.get(&v).copied().unwrap_or(0) != 2 {
                continue;
            }
            let Some(owners) = incident.get(&v) else {
                continue;
            };
            if owners.len() != 2 {
                continue;
            }
            let Some(q1) = quads[owners[0]] else {
                continue;
            };
            let Some(q2) = quads[owners[1]] else {
                continue;
            };
            let mut verts: Vec<usize> = Vec::with_capacity(4);
            for &u in q1.iter().chain(q2.iter()) {
                if u != v && !verts.contains(&u) {
                    verts.push(u);
                }
            }
            if verts.len() != 4 {
                continue;
            }
            let Some(merged) = order_around_centroid(&verts, points) else {
                continue;
            };
            if !convex_ccw(&quad_coords(&merged, points)) {
                continue;
            }
            let old_area =
                polygon_signed_area(&q1, points).abs() + polygon_signed_area(&q2, points).abs();
            let new_area = polygon_signed_area(&merged, points);
            if (new_area - old_area).abs() > 1e-9 * (1.0 + old_area) {
                continue;
            }
            let new_sj = quad_min_sj(&merged, points);
            let old_sj = quad_min_sj(&q1, points).min(quad_min_sj(&q2, points));
            if !new_sj.is_finite() || new_sj < 0.2 || new_sj + 0.15 < old_sj {
                continue;
            }
            if quad_min_angle(&merged, points) < 20.0 {
                continue;
            }
            if longest_edge(&merged, points) > 2.5 * size {
                continue;
            }
            quads[owners[0]] = None;
            quads[owners[1]] = None;
            quads.push(Some(merged));
            collapses += 1;
            changed = true;
            break;
        }
        if !changed {
            break;
        }
    }
    collapses
}

/// Order a set of points counter-clockwise around their centroid.
fn order_around_centroid(verts: &[usize], points: &[Node]) -> Option<[usize; 4]> {
    if verts.len() != 4 {
        return None;
    }
    let mut cx = 0.0;
    let mut cy = 0.0;
    for &v in verts {
        cx += points[v][0];
        cy += points[v][1];
    }
    cx /= 4.0;
    cy /= 4.0;
    let mut ordered = [verts[0], verts[1], verts[2], verts[3]];
    ordered.sort_by(|&i, &j| {
        let ai = (points[i][1] - cy).atan2(points[i][0] - cx);
        let aj = (points[j][1] - cy).atan2(points[j][0] - cx);
        ai.partial_cmp(&aj)
            .unwrap_or(Ordering::Equal)
            .then(i.cmp(&j))
    });
    Some(ordered)
}

fn longest_edge(quad: &[usize; 4], points: &[Node]) -> f64 {
    let mut longest: f64 = 0.0;
    for i in 0..4 {
        let a = points[quad[i]];
        let b = points[quad[(i + 1) % 4]];
        let d = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
        longest = longest.max(d);
    }
    longest
}

/// Valence-driven quad edge swaps.
///
/// Two quads sharing an edge form a hexagon; the shared edge is replaced by one
/// of the other two long diagonals. A swap is taken only when it strictly lowers
/// the irregularity `sum |valence - 4|` over the six vertices, both new quads are
/// convex and keep the gate's Jacobian floor, the pair's worst Jacobian does not
/// drop, and the new diagonal does not already exist (which would make the mesh
/// non-manifold).
fn swap_edges(
    points: &[Node],
    quads: &mut [Option<[usize; 4]>],
    tris: &[[usize; 3]],
    boundary: &[bool],
) -> usize {
    let mut swaps = 0usize;
    for _ in 0..32 {
        let mut edge_owners: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        let mut edge_set: HashSet<(usize, usize)> = HashSet::new();
        let mut valence: HashMap<usize, usize> = HashMap::new();
        for (id, slot) in quads.iter().enumerate() {
            let Some(quad) = slot else {
                continue;
            };
            for i in 0..4 {
                let (a, b) = (quad[i], quad[(i + 1) % 4]);
                let key = edge_key(a, b);
                edge_owners.entry(key).or_default().push(id);
                edge_set.insert(key);
                *valence.entry(a).or_insert(0) += 1;
            }
        }
        for tri in tris {
            for (a, b) in triangle_edges(tri) {
                edge_set.insert(edge_key(a, b));
            }
        }
        let mut keys: Vec<(usize, usize)> = edge_owners
            .iter()
            .filter(|(_, owners)| owners.len() == 2)
            .map(|(key, _)| *key)
            .collect();
        keys.sort_unstable();
        let mut changed = 0usize;
        for (a, b) in keys {
            let Some(owners) = edge_owners.get(&(a, b)) else {
                continue;
            };
            if owners.len() != 2 {
                continue;
            }
            let (c1, c2) = (owners[0], owners[1]);
            let Some(q1) = quads[c1] else {
                continue;
            };
            let Some(q2) = quads[c2] else {
                continue;
            };
            let Some(i1) = (0..4).find(|&i| edge_key(q1[i], q1[(i + 1) % 4]) == (a, b)) else {
                continue;
            };
            // q1 traverses the edge x -> y; q2 must traverse it y -> x.
            let (x, y) = (q1[i1], q1[(i1 + 1) % 4]);
            let p = q1[(i1 + 2) % 4];
            let q = q1[(i1 + 3) % 4];
            let Some(i2) = (0..4).find(|&i| q2[i] == y && q2[(i + 1) % 4] == x) else {
                continue;
            };
            let r = q2[(i2 + 2) % 4];
            let s = q2[(i2 + 3) % 4];

            let old_sj = quad_min_sj(&q1, points).min(quad_min_sj(&q2, points));
            let before = irregularity(&[x, y, p, r, q, s], &valence, boundary);
            let options: [([[usize; 4]; 2], [(usize, isize); 4]); 2] = [
                // new diagonal p-r
                (
                    [[p, q, x, r], [p, r, s, y]],
                    [(x, -1), (y, -1), (p, 1), (r, 1)],
                ),
                // new diagonal q-s
                (
                    [[q, x, r, s], [q, s, y, p]],
                    [(x, -1), (y, -1), (q, 1), (s, 1)],
                ),
            ];
            let mut best: Option<(i64, [usize; 4], [usize; 4], [(usize, isize); 4])> = None;
            for (pair, delta) in options {
                if !pair
                    .iter()
                    .all(|quad| convex_ccw(&quad_coords(quad, points)))
                {
                    continue;
                }
                let old_area =
                    polygon_signed_area(&q1, points).abs() + polygon_signed_area(&q2, points).abs();
                let new_area: f64 = pair
                    .iter()
                    .map(|quad| polygon_signed_area(quad, points).abs())
                    .sum();
                if (new_area - old_area).abs() > 1e-9 * (1.0 + old_area) {
                    continue;
                }
                if pair.iter().any(|quad| {
                    let sj = quad_min_sj(quad, points);
                    !sj.is_finite() || sj < 0.2
                }) {
                    continue;
                }
                let new_sj = pair
                    .iter()
                    .map(|quad| quad_min_sj(quad, points))
                    .fold(f64::INFINITY, f64::min);
                if new_sj + 1e-9 < old_sj {
                    continue;
                }
                let diagonal = edge_key(pair[0][0], pair[0][3]);
                if edge_set.contains(&diagonal) {
                    continue;
                }
                let mut after_valence = valence.clone();
                for (vertex, step) in delta {
                    let entry = after_valence.entry(vertex).or_insert(0);
                    if step < 0 {
                        *entry = entry.saturating_sub(1);
                    } else {
                        *entry += 1;
                    }
                }
                let after = irregularity(&[x, y, p, r, q, s], &after_valence, boundary);
                if after < before && best.as_ref().is_none_or(|(cost, _, _, _)| after < *cost) {
                    best = Some((after, pair[0], pair[1], delta));
                }
            }
            if let Some((_, left, right, delta)) = best {
                quads[c1] = Some(left);
                quads[c2] = Some(right);
                for (vertex, step) in delta {
                    let entry = valence.entry(vertex).or_insert(0);
                    if step < 0 {
                        *entry = entry.saturating_sub(1);
                    } else {
                        *entry += 1;
                    }
                }
                // Keep the edge set current so a later swap in this round cannot
                // recreate an edge that has just been introduced.
                edge_set.remove(&(a, b));
                edge_set.insert(edge_key(left[0], left[3]));
                edge_set.insert(edge_key(right[0], right[3]));
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

fn irregularity(vertices: &[usize], valence: &HashMap<usize, usize>, boundary: &[bool]) -> i64 {
    vertices
        .iter()
        .filter(|v| !boundary.get(**v).copied().unwrap_or(false))
        .map(|v| {
            let degree = valence.get(v).copied().unwrap_or(0) as i64;
            (degree - 4).abs()
        })
        .sum()
}

/// Passes of the residual triangle walk. One pass pairs what it can with the
/// current topology; a second pass sees the quads the first pass created.
const MAX_WALK_PASSES: usize = 4;
/// Depth of the backtracking walk, in triangle moves.
const WALK_DEPTH: usize = 5;
/// Search states tried per residual triangle before giving up on it.
const WALK_BUDGET: usize = 400;

/// Join residual triangles that are separated by quads.
///
/// This is the dual-graph walk of Zint & Grosso, Sect. 3.3, with the merge and
/// split steps made explicit. A triangle and an adjacent quad cover a pentagon.
/// Splitting that pentagon on a different short diagonal replaces the pair with
/// another triangle and quad covering the same area: the triangle has moved
/// through the quad without moving a node or changing the area. A backtracking
/// search walks the triangle through adjacent quads (all quads around it, not a
/// fixed path) until it is next to another residual triangle, then merges the
/// two into a quad.
///
/// Every step is guarded: the new quad must be convex with scaled Jacobian at
/// least 0.2, the new triangle non-degenerate, the area exactly conserved, and
/// the new diagonal must not already exist in the mesh (otherwise the mesh would
/// become non-manifold). The final merge uses the same guards. A residual that
/// cannot reach a partner under those guards is left as a triangle.
fn merge_residuals(
    points: &[Node],
    quads: &mut Vec<Option<[usize; 4]>>,
    tris: &mut [Option<[usize; 3]>],
    options: &Options,
) -> usize {
    let mut merges = 0usize;
    for _ in 0..MAX_WALK_PASSES {
        let mut nq = quads.len();
        let mut base = build_mesh_edges(quads, tris, nq);
        let tri_ids: Vec<usize> = tris
            .iter()
            .enumerate()
            .filter_map(|(tid, tri)| tri.map(|_| tid))
            .collect();
        if tri_ids.len() < 2 {
            break;
        }
        let mut pass_merges = 0usize;
        for &tid in &tri_ids {
            let Some(current) = tris[tid] else {
                continue;
            };
            let merged = {
                let mut walk = Walk::new(points, quads, tris, &base, nq, options.walk_budget);
                walk.search(tid, current, options.walk_depth)
            };
            if merged {
                merges += 1;
                pass_merges += 1;
                nq = quads.len();
                base = build_mesh_edges(quads, tris, nq);
            }
        }
        if pass_merges == 0 {
            break;
        }
    }
    merges
}

/// Edge key -> the cells that use it. Quad cells keep their id; triangle cell
/// ids are `nq + tid`.
fn build_mesh_edges(
    quads: &[Option<[usize; 4]>],
    tris: &[Option<[usize; 3]>],
    nq: usize,
) -> HashMap<(usize, usize), Vec<usize>> {
    let mut owners: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (qid, slot) in quads.iter().enumerate() {
        if let Some(quad) = slot {
            for i in 0..4 {
                let key = edge_key(quad[i], quad[(i + 1) % 4]);
                owners.entry(key).or_default().push(qid);
            }
        }
    }
    for (tid, slot) in tris.iter().enumerate() {
        if let Some(tri) = slot {
            for i in 0..3 {
                let key = edge_key(tri[i], tri[(i + 1) % 3]);
                owners.entry(key).or_default().push(nq + tid);
            }
        }
    }
    owners
}

/// One pentagon split that moves the triangle through a quad.
struct SplitOption {
    tri: [usize; 3],
    quad: [usize; 4],
    quad_sj: f64,
    tri_angle: f64,
}

struct Walk<'a> {
    points: &'a [Node],
    quads: &'a mut Vec<Option<[usize; 4]>>,
    tris: &'a mut [Option<[usize; 3]>],
    base: &'a HashMap<(usize, usize), Vec<usize>>,
    nq: usize,
    /// Edge owners changed by the moves applied on the current branch.
    overrides: HashMap<(usize, usize), Vec<usize>>,
    /// Undo log for `overrides`: `(edge, previous override)`.
    log: Vec<((usize, usize), Option<Vec<usize>>)>,
    budget: usize,
    visited: HashSet<[usize; 3]>,
}

impl<'a> Walk<'a> {
    fn new(
        points: &'a [Node],
        quads: &'a mut Vec<Option<[usize; 4]>>,
        tris: &'a mut [Option<[usize; 3]>],
        base: &'a HashMap<(usize, usize), Vec<usize>>,
        nq: usize,
        walk_budget: usize,
    ) -> Self {
        Walk {
            points,
            quads,
            tris,
            base,
            nq,
            overrides: HashMap::new(),
            log: Vec::new(),
            budget: walk_budget,
            visited: HashSet::new(),
        }
    }

    /// Cells using `edge` in the mesh as modified by the current branch.
    fn owners(&self, edge: (usize, usize)) -> Vec<usize> {
        match self.overrides.get(&edge) {
            Some(list) => list.clone(),
            None => self.base.get(&edge).cloned().unwrap_or_default(),
        }
    }

    fn edge_present(&self, edge: (usize, usize)) -> bool {
        match self.overrides.get(&edge) {
            Some(list) => !list.is_empty(),
            None => self.base.contains_key(&edge),
        }
    }

    /// Merge the current triangle with an adjacent residual triangle, if the
    /// union is a convex quad with scaled Jacobian at least 0.2.
    fn merge_adjacent(&mut self, tid: usize, current: &[usize; 3]) -> bool {
        let mut targets: Vec<usize> = Vec::new();
        for i in 0..3 {
            let edge = edge_key(current[i], current[(i + 1) % 3]);
            for owner in self.owners(edge) {
                if owner >= self.nq && owner - self.nq != tid {
                    targets.push(owner - self.nq);
                }
            }
        }
        targets.sort_unstable();
        targets.dedup();
        for target_tid in targets {
            let Some(target) = self.tris.get(target_tid).copied().flatten() else {
                continue;
            };
            let Some(quad) = merge_pair(self.points, current, &target) else {
                continue;
            };
            self.tris[tid] = None;
            self.tris[target_tid] = None;
            self.quads.push(Some(quad));
            return true;
        }
        false
    }

    /// Try to merge here, then move the triangle through one adjacent quad and
    /// recurse. Moves are undone on backtracking, so a failed exploration leaves
    /// the mesh exactly as it was.
    fn search(&mut self, tid: usize, current: [usize; 3], depth: usize) -> bool {
        if self.merge_adjacent(tid, &current) {
            return true;
        }
        if depth == 0 || self.budget == 0 {
            return false;
        }
        self.budget -= 1;
        let key = sorted_key(&current);
        if !self.visited.insert(key) {
            return false;
        }

        let mut near: Vec<usize> = Vec::new();
        for i in 0..3 {
            let edge = edge_key(current[i], current[(i + 1) % 3]);
            for owner in self.owners(edge) {
                if owner < self.nq {
                    near.push(owner);
                }
            }
        }
        near.sort_unstable();
        near.dedup();
        for qid in near {
            let Some(quad) = self.quads.get(qid).copied().flatten() else {
                continue;
            };
            let splits = move_splits(self.points, &current, &quad, self);
            for split in splits {
                let mark = self.log.len();
                let old_tri = self.tris[tid];
                let old_quad = self.quads[qid];
                self.apply_split(tid, qid, &current, &quad, &split);
                if self.search(tid, split.tri, depth - 1) {
                    return true;
                }
                self.rollback(mark);
                self.tris[tid] = old_tri;
                self.quads[qid] = old_quad;
            }
        }
        self.visited.remove(&key);
        false
    }

    /// Replace `(old_tri, old_quad)` by `(split.tri, split.quad)` and update the
    /// edge-owner overrides.
    fn apply_split(
        &mut self,
        tid: usize,
        qid: usize,
        old_tri: &[usize; 3],
        old_quad: &[usize; 4],
        split: &SplitOption,
    ) {
        let tri_cell = self.nq + tid;
        let mut affected: Vec<(usize, usize)> = Vec::new();
        for i in 0..3 {
            affected.push(edge_key(old_tri[i], old_tri[(i + 1) % 3]));
            affected.push(edge_key(split.tri[i], split.tri[(i + 1) % 3]));
        }
        for i in 0..4 {
            affected.push(edge_key(old_quad[i], old_quad[(i + 1) % 4]));
            affected.push(edge_key(split.quad[i], split.quad[(i + 1) % 4]));
        }
        affected.sort_unstable();
        affected.dedup();
        for edge in affected {
            let mut owners = self.owners(edge);
            owners.retain(|&cell| cell != tri_cell && cell != qid);
            if cell_has_edge(&split.tri, edge) {
                owners.push(tri_cell);
            }
            if cell_has_edge(&split.quad, edge) {
                owners.push(qid);
            }
            owners.sort_unstable();
            owners.dedup();
            let previous = self.overrides.insert(edge, owners);
            self.log.push((edge, previous));
        }
        self.tris[tid] = Some(split.tri);
        self.quads[qid] = Some(split.quad);
    }

    fn rollback(&mut self, mark: usize) {
        while self.log.len() > mark {
            let Some((edge, previous)) = self.log.pop() else {
                break;
            };
            match previous {
                Some(list) => {
                    self.overrides.insert(edge, list);
                }
                None => {
                    self.overrides.remove(&edge);
                }
            }
        }
    }
}

fn cell_has_edge(nodes: &[usize], edge: (usize, usize)) -> bool {
    (0..nodes.len()).any(|i| edge_key(nodes[i], nodes[(i + 1) % nodes.len()]) == edge)
}

/// The pentagon `p, s, t, q, r` covered by triangle `(p,q,r)` and quad
/// `(q,p,s,t)`, or `None` when they do not share exactly one edge.
fn pentagon_from(tri: &[usize; 3], quad: &[usize; 4]) -> Option<[usize; 5]> {
    for i in 0..3 {
        let p = tri[i];
        let q = tri[(i + 1) % 3];
        let r = tri[(i + 2) % 3];
        for j in 0..4 {
            if quad[j] == q && quad[(j + 1) % 4] == p {
                let s = quad[(j + 2) % 4];
                let t = quad[(j + 3) % 4];
                return Some([p, s, t, q, r]);
            }
        }
    }
    None
}

/// Splits of the pentagon that move the triangle to another corner, best first.
/// The original partition (the one equal to the current triangle) is skipped.
fn move_splits(
    points: &[Node],
    tri: &[usize; 3],
    quad: &[usize; 4],
    walk: &Walk<'_>,
) -> Vec<SplitOption> {
    let Some(p) = pentagon_from(tri, quad) else {
        return Vec::new();
    };
    let coords: [[f64; 2]; 5] = [
        [points[p[0]][0], points[p[0]][1]],
        [points[p[1]][0], points[p[1]][1]],
        [points[p[2]][0], points[p[2]][1]],
        [points[p[3]][0], points[p[3]][1]],
        [points[p[4]][0], points[p[4]][1]],
    ];
    let old_area = triangle_area(tri, points).abs() + polygon_signed_area(quad, points).abs();
    let pentagon_area = coords_area(&coords);
    if (pentagon_area - old_area).abs() > 1e-9 * (1.0 + old_area.abs()) {
        return Vec::new();
    }
    let old_key = sorted_key(tri);

    let mut out: Vec<SplitOption> = Vec::new();
    for i in 0..5 {
        let new_tri = [p[i], p[(i + 1) % 5], p[(i + 2) % 5]];
        if sorted_key(&new_tri) == old_key {
            continue;
        }
        let new_quad = [p[i], p[(i + 2) % 5], p[(i + 3) % 5], p[(i + 4) % 5]];
        let tri_area = triangle_area(&new_tri, points);
        if tri_area <= 1e-14 {
            continue;
        }
        if !convex_ccw(&quad_coords(&new_quad, points)) {
            continue;
        }
        let quad_area = polygon_signed_area(&new_quad, points);
        if (tri_area + quad_area - pentagon_area).abs() > 1e-9 * (1.0 + pentagon_area.abs()) {
            continue;
        }
        let quad_sj = quad_min_sj(&new_quad, points);
        if !quad_sj.is_finite() || quad_sj < 0.2 {
            continue;
        }
        let tri_angle = triangle_min_angle(&new_tri, points);
        if !tri_angle.is_finite() || tri_angle < 2.0 {
            continue;
        }
        let diagonal = edge_key(p[i], p[(i + 2) % 5]);
        if walk.edge_present(diagonal) {
            continue;
        }
        out.push(SplitOption {
            tri: new_tri,
            quad: new_quad,
            quad_sj,
            tri_angle,
        });
    }
    out.sort_by(|a, b| {
        b.quad_sj
            .partial_cmp(&a.quad_sj)
            .unwrap_or(Ordering::Equal)
            .then(
                b.tri_angle
                    .partial_cmp(&a.tri_angle)
                    .unwrap_or(Ordering::Equal),
            )
    });
    out
}

fn triangle_min_angle(tri: &[usize; 3], points: &[Node]) -> f64 {
    quality::corner_angles(&Cell::tri3(*tri), points)
        .into_iter()
        .fold(f64::INFINITY, f64::min)
}

/// Merge two triangles sharing an edge, if the union is a convex quad with a
/// scaled Jacobian of at least 0.2 and exactly the same area.
fn merge_pair(points: &[Node], a: &[usize; 3], b: &[usize; 3]) -> Option<[usize; 4]> {
    let quad = quad_from_pair(a, b, points)?;
    if !convex_ccw(&quad_coords(&quad, points)) {
        return None;
    }
    let union = triangle_area(a, points).abs() + triangle_area(b, points).abs();
    if (polygon_signed_area(&quad, points) - union).abs() > 1e-9 * (1.0 + union.abs()) {
        return None;
    }
    let sj = quad_min_sj(&quad, points);
    if !sj.is_finite() || sj < 0.2 {
        return None;
    }
    Some(quad)
}

fn coords_area(coords: &[[f64; 2]]) -> f64 {
    let mut sum = 0.0;
    for i in 0..coords.len() {
        let a = coords[i];
        let b = coords[(i + 1) % coords.len()];
        sum += a[0] * b[1] - b[0] * a[1];
    }
    sum / 2.0
}

fn convex_ccw_poly(coords: &[[f64; 2]]) -> bool {
    if coords.len() < 3 {
        return false;
    }
    if coords_area(coords) <= 1e-12 {
        return false;
    }
    (0..coords.len()).all(|i| {
        let a = coords[(i + coords.len() - 1) % coords.len()];
        let b = coords[i];
        let c = coords[(i + 1) % coords.len()];
        let u = [b[0] - a[0], b[1] - a[1]];
        let v = [c[0] - b[0], c[1] - b[1]];
        u[0] * v[1] - u[1] * v[0] > 0.0
    })
}

#[cfg(test)]
impl HybridStats {
    /// The stage counters must add up to the triangle count and the quad count
    /// must be consistent with the quads in the mesh.
    fn quads_ok(&self) -> bool {
        // Each doublet collapse turns two quads into one, so the final quad
        // count is the quads formed minus the collapses.
        let quads = self.band_quads + self.greedy_quads + self.leftover_quads
            - self.doublet_collapses
            + self.walk_merges;
        quads * 2 + self.leftover_triangles == self.triangles
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::verify;

    fn convex_quads(mesh: &Mesh) -> usize {
        mesh.cells
            .iter()
            .filter(|cell| cell.kind == crate::mesh::Kind::Quad4)
            .filter(|cell| {
                let quad: [usize; 4] = match cell.nodes.as_slice().try_into() {
                    Ok(quad) => quad,
                    Err(_) => return false,
                };
                convex_ccw(&quad_coords(&quad, &mesh.points))
            })
            .count()
    }

    #[test]
    fn a_square_is_quad_dominant_and_conforming() {
        let domain = Domain::rectangle(10.0, 10.0).unwrap();
        let (mesh, stats) = mesh_with_stats(&domain, &Sizing::uniform(2.0).unwrap()).unwrap();
        assert_eq!(mesh.validate(), Ok(()));
        assert!(stats.quads_ok(), "{stats:?}");
        let gate = verify::gate(&mesh);
        assert_eq!(gate.cracks, 0, "cracks: {gate:?}");
        assert_eq!(gate.inverted, 0, "inverted: {gate:?}");
        assert_eq!(gate.overlapping, 0, "overlapping: {gate:?}");
        assert!(gate.quads > gate.triangles, "{gate:?}");
        assert!(verify::boundary_offset(&domain, &mesh) < 1.0e-9);
        assert_eq!(convex_quads(&mesh), gate.quads);
    }

    #[test]
    fn non_default_options_still_produce_a_valid_mesh() {
        let domain = Domain::plate_with_hole(100.0, 60.0, 10.0, 16).unwrap();
        let sizing = Sizing::uniform(4.0).unwrap();
        let options = Options {
            band_layers: 3,
            augment_depth: 4,
            walk_depth: 2,
            walk_budget: 100,
        };
        let mesh = mesh_with_options(&domain, &sizing, &options).unwrap();
        assert_eq!(mesh.validate(), Ok(()));
        let gate = verify::gate(&mesh);
        assert_eq!(gate.cracks, 0, "cracks: {gate:?}");
        assert_eq!(gate.inverted, 0, "inverted: {gate:?}");
        assert_eq!(gate.overlapping, 0, "overlapping: {gate:?}");
        assert!(gate.quads > 0, "{gate:?}");
        assert!(gate.triangles > 0, "{gate:?}");
        assert!(verify::boundary_offset(&domain, &mesh) < 1.0e-9);
    }

    #[test]
    fn a_plate_with_a_hole_is_watertight_and_the_band_is_even() {
        let domain = Domain::plate_with_hole(100.0, 60.0, 10.0, 16).unwrap();
        let (mesh, stats) = mesh_with_stats(&domain, &Sizing::uniform(4.0).unwrap()).unwrap();
        assert!(stats.band_triangles > 0);
        assert_eq!(stats.band_triangles % 2, 0);
        assert!(stats.band_quads > 0);
        assert!(stats.quads_ok(), "{stats:?}");
        let topo = mesh.topology();
        assert_eq!(topo.cracks, 0);
        assert_eq!(topo.inconsistent, 0);
        assert_eq!(topo.non_manifold, 0);
        let gate = verify::gate(&mesh);
        assert_eq!(gate.inverted, 0, "{gate:?}");
        assert_eq!(gate.overlapping, 0, "{gate:?}");
        assert_eq!(convex_quads(&mesh), gate.quads);
        assert!(verify::boundary_offset(&domain, &mesh) < 1.0e-9);
    }

    #[test]
    fn a_greedy_merge_of_a_sliver_is_refused() {
        // Two triangles sharing the edge 0-1, with the other two vertices a
        // hair apart: the union is a valid but very thin kite.
        let points: Vec<Node> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 0.001, 0.0],
            [0.5, -0.001, 0.0],
        ];
        let tris = [[0usize, 1, 2], [1, 0, 3]];
        // The interior floor refuses it (its minimum scaled Jacobian is ~0.002).
        assert!(candidate_for(&points, &tris, 0, 1, GREEDY_FLOOR).is_none());
        // Without a floor the same pair is a perfectly valid convex quad: the
        // refusal is the quality guard, not a broken construction.
        let loose = Floor {
            min_scaled_jacobian: 0.0,
            min_angle_deg: 0.0,
            max_angle_deg: 180.0,
        };
        let candidate = candidate_for(&points, &tris, 0, 1, loose);
        assert!(candidate.is_some());
        assert!(
            candidate.map(|c| c.score < GREEDY_FLOOR.min_scaled_jacobian) == Some(true),
            "the kite should be below the greedy floor"
        );
    }

    #[test]
    fn the_walk_brings_two_residual_triangles_together() {
        // A quad with one triangle below it and one on its right: the two
        // triangles are not adjacent, so only the walk can join them.
        //
        //   4 ---- 3
        //   |      | \
        //   |      |  5
        //   |      | /
        //   0 ---- 2
        //     \   /
        //       1
        let points: Vec<Node> = vec![
            [0.0, 0.0, 0.0],
            [1.0, -1.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [3.0, 1.0, 0.0],
        ];
        let mut quads = vec![Some([0usize, 2, 3, 4])];
        let mut tris = vec![Some([0usize, 1, 2]), Some([2usize, 5, 3])];
        let merges = merge_residuals(&points, &mut quads, &mut tris, &Options::default());
        assert_eq!(
            merges, 1,
            "the walk should bring the two triangles together"
        );
        assert!(tris.iter().all(Option::is_none), "{tris:?}");
        let quad_count = quads.iter().flatten().count();
        assert_eq!(quad_count, 2, "{quads:?}");

        // The resulting mesh must be manifold: no two cells share more than one
        // edge, and the boundary is one closed polyline.
        let cells: Vec<Cell> = quads.iter().flatten().map(|q| Cell::quad4(*q)).collect();
        let mesh = Mesh::new(points, cells);
        let topo = mesh.topology();
        assert_eq!(topo.non_manifold, 0, "{topo:?}");
        assert_eq!(topo.inconsistent, 0, "{topo:?}");
        assert_eq!(topo.cracks, 0, "{topo:?}");
        let area: f64 = mesh
            .cells
            .iter()
            .map(|c| quality::volume(c, &mesh.points))
            .sum();
        // Original patch area: quad 4 + two triangles of area 1 each.
        assert!((area - 6.0).abs() < 1.0e-12, "{area}");
        for quad in quads.iter().flatten() {
            assert!(
                quad_min_sj(quad, &mesh.points) >= 0.2,
                "the walk created a quad below the floor: {quad:?}"
            );
        }
    }

    #[test]
    fn a_domain_with_no_area_is_refused() {
        // Collinear outer loop: the domain has no area and the route returns an
        // error instead of a degenerate mesh.
        let domain = Domain::new(vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], Vec::new()).unwrap();
        assert!(mesh(&domain, &Sizing::uniform(0.5).unwrap()).is_err());
    }
}
