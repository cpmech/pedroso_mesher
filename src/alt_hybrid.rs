//! Hybrid fast indirect quad meshing: Blossom-Quad on the boundary band and
//! greedy triangle merging in the interior.
//!
//! This implements the hybrid scheme of Zint and Grosso (2021), *A hybrid
//! approach to fast indirect quadrilateral mesh generation* (corpus
//! `zint-2021`), using the shipped Blossom-Quad matching of Remacle et al.
//! (2012) (corpus `remacle-2012`) for the boundary band. The steps, in the
//! order the route runs them:
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
//! **Measured (corpus).** This is the speed route, not a gate route. On the
//! seven corpus parts it leaves 4, 10, 16, 20, 52, 68 and 190 triangles (4-190
//! per part), so no raw row passes [`crate::verify::gate`], which requires
//! every element to be a quad. The Sect. 3.3 walk is what makes it competitive:
//! before the walk the band, greedy and leftover passes left 880 residual
//! triangles summed over the parts, and the walk reduced that to 362 (-59 %) in
//! the pass that introduced it. The 0.6 mm performance part runs in 0.84 s raw,
//! against about 19 s for the shipped Blossom-Quad baseline (`two_d::mesh_2d`,
//! 18.5 s in the final corpus table). The common post-pass raises the raw
//! minimum scaled Jacobian from 0.17-0.23 to 0.32-0.70, but a post-pass cannot
//! remove a triangle, so the gate still fails on the triangle count. See
//! `examples/alt_hybrid.rs` and `results/hybrid.txt`.
//!
//! **Approximated.** The paper's even parity is per boundary region; here all
//! boundaries are merged into one band and a single adjacent interior triangle
//! is taken when that band is odd ([`first_adjacent_interior`]). The matching
//! repair is a depth-limited augmenting-path heuristic, not a maximum matching,
//! and the residual walk is bounded both in depth and in expanded states. The
//! greedy interior never recomputes a candidate score, which is exact because
//! merging moves no node. These are deliberate speed/quality trade-offs of the
//! route, not hidden parts of the paper's method.
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
/// [`Options::default`] reproduces the measured tables exactly. Every field is
/// a tuning parameter of the heuristic, not a property of the input geometry.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Options {
    /// Radius of the boundary band, in triangle-edge hops (dimensionless;
    /// default 2, the paper's two layers). A triangle is in the band when all
    /// three of its vertices are at most this many hops from a boundary vertex;
    /// 1 gives a thin band, larger values grow the band and the Blossom cost.
    /// Invariant: the vertex BFS in [`boundary_hops`] stops at this depth.
    pub band_layers: u8,

    /// Maximum length of an alternating augmenting path, in dual-graph hops
    /// (default 8, [`AUGMENT_DEPTH`]). The greedy repair only searches this
    /// far, which keeps it cheaper than a global Blossom solve. A value of 0
    /// admits only a direct unmatched-unmatched pair, which the greedy pass has
    /// already exhausted, so it effectively turns the repair off.
    pub augment_depth: usize,

    /// Maximum number of pentagon splits (triangle moves) in one Sect. 3.3 walk
    /// (default 5, [`WALK_DEPTH`]). 0 means the walk can only merge a residual
    /// triangle that is already adjacent to another one.
    pub walk_depth: usize,

    /// Maximum number of walk states expanded per residual triangle (default
    /// 400, [`WALK_BUDGET`]). It bounds the backtracking search; when the
    /// budget is exhausted the triangle is left as a triangle. Invariant: one
    /// decrement per expanded state, never below zero.
    pub walk_budget: usize,
}

impl Default for Options {
    /// The measured defaults: two band layers, depth-8 augmenting repair,
    /// depth-5 walk and 400 states per residual triangle.
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

/// A quality floor for candidate quads. The three thresholds are evaluated
/// together by [`candidate_for`]; a candidate must clear all of them.
#[derive(Clone, Copy, Debug)]
struct Floor {
    /// Smallest allowed minimum scaled Jacobian over the four corners
    /// (dimensionless, -1..1; a value of 0.2 is the usual gate floor).
    min_scaled_jacobian: f64,

    /// Smallest allowed minimum corner angle, in degrees (0..180).
    min_angle_deg: f64,

    /// Largest allowed corner angle, in degrees (0..180); it rejects a quad
    /// that is folded back on itself even when its Jacobian stays positive.
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
///
/// The pair is stored with `a < b`, so `(a, b)` is a canonical key in the
/// candidate map. The quad has already passed the [`Floor`], the convexity and
/// the exact-area checks of [`candidate_for`]; it is never revalidated by the
/// heap, only rechecked for its triangles still being unmatched.
#[derive(Clone, Copy, Debug)]
struct Candidate {
    /// Minimum scaled Jacobian of the quad's four corners (dimensionless,
    /// -1..1; higher is better). The primary heap key.
    score: f64,

    /// Minimum corner angle of the quad, in degrees (0..180; higher is better).
    /// The tie-break key and a needle guard.
    angle: f64,

    /// First triangle id; always the smaller of the two.
    a: usize,

    /// Second triangle id; always the larger of the two.
    b: usize,

    /// The quad the two triangles would form, counter-clockwise.
    quad: [usize; 4],
}

impl PartialEq for Candidate {
    /// Equality is the pair key plus the two scores, not the geometry: it
    /// exists so the heap can compare candidates and so a test can compare
    /// them without requiring `f64: Eq`.
    fn eq(&self, other: &Self) -> bool {
        self.a == other.a
            && self.b == other.b
            && self.score.to_bits() == other.score.to_bits()
            && self.angle.to_bits() == other.angle.to_bits()
    }
}

impl Eq for Candidate {}

impl Ord for Candidate {
    /// Total order for the max-heaps: higher score first, then higher minimum
    /// angle, then smaller triangle ids. The total order (rather than a
    /// partial one) is what makes the route deterministic.
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
    /// Delegates to [`Ord`], so `BinaryHeap` and any sort see the same order.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Counters for one hybrid run, for the tests and for diagnosing a run.
///
/// These are stage counters, not a census of the final mesh. A quad is built
/// from two triangles, and a walk merge turns two triangles into one quad, so
/// the `quads_ok` test checks the arithmetic
/// `2 * (band_quads + greedy_quads + leftover_quads - doublet_collapses +
/// walk_merges) + leftover_triangles == triangles`.
///
/// Caveat: a doublet collapse folds two quads into one element (four of the
/// original triangles), so its term in that identity is the element delta, not
/// the area delta; the test domains are chosen to have no collapse. The exact
/// area identity would need `+ 2 * doublet_collapses` on the left side (each
/// collapse leaves the one element covering two extra triangle-units).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HybridStats {
    /// Triangles produced by the triangulation (the substrate count).
    pub triangles: usize,

    /// Triangles in the boundary band, after the even-parity adjustment.
    pub band_triangles: usize,

    /// Quads the Blossom band matching produced (two band triangles each).
    pub band_quads: usize,

    /// Quads the interior greedy merge produced.
    pub greedy_quads: usize,

    /// Quads the leftover merge produced.
    pub leftover_quads: usize,

    /// Quads removed by a doublet collapse; each collapse replaces two quads
    /// by one, so this is also the number of quads the final mesh lost.
    pub doublet_collapses: usize,

    /// Quad edge swaps taken by the valence clean-up (geometry is unchanged;
    /// only the two quads around the swapped edge are replaced).
    pub cleanup_swaps: usize,

    /// Pairs of residual triangles joined by the dual-graph walk. Each merge
    /// adds one quad to the mesh and removes two triangles.
    pub walk_merges: usize,

    /// Triangles that stayed triangles (the final quad-dominant shortfall).
    pub leftover_triangles: usize,
}

impl HybridStats {
    /// Share of the original triangle substrate that ended up inside quads:
    /// `(triangles - leftover_triangles) / triangles`.
    ///
    /// Note that this is **not** `quads / (quads + triangles)` when triangles
    /// remain: a quad covers two of the original triangles, so the value is
    /// twice the element-count share. It is the fraction the route's own
    /// bookkeeping makes convenient, and it is 1 exactly when the mesh is
    /// all-quad.
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
///
/// This is the convenience entry point; it returns the same mesh as
/// [`mesh_with_options`] with [`Options::default`] and discards the stage
/// counters. Errors are the triangulation failures (no area, no triangles).
pub fn mesh(domain: &Domain, sizing: &Sizing) -> Result<Mesh, String> {
    mesh_with_options(domain, sizing, &Options::default())
}

/// Mesh a planar domain with the hybrid route and explicit options.
///
/// Thin wrapper over [`mesh_with_options_with_stats`] that drops the counters.
/// `domain` must have positive area and a counter-clockwise outer loop;
/// `sizing.size` is the element size used by the triangulation. The result
/// tiles the domain exactly (the route never moves or drops a point) and is
/// quad-dominant: any unpaired triangle is returned as a `Tri3` cell.
pub fn mesh_with_options(
    domain: &Domain,
    sizing: &Sizing,
    options: &Options,
) -> Result<Mesh, String> {
    mesh_with_options_with_stats(domain, sizing, options).map(|(mesh, _)| mesh)
}

/// Mesh a planar domain with the default options and return the stage counters.
///
/// Same result as [`mesh`], plus the [`HybridStats`] used by the tests to check
/// the stage bookkeeping and by `examples/alt_hybrid.rs` to report the band,
/// greedy, leftover and walk contributions.
pub fn mesh_with_stats(domain: &Domain, sizing: &Sizing) -> Result<(Mesh, HybridStats), String> {
    mesh_with_options_with_stats(domain, sizing, &Options::default())
}

/// Mesh a planar domain with explicit options and return the stage counters.
///
/// This is the whole pipeline, in the order of the module documentation:
///
/// 1. triangulate the domain (constrained Delaunay + Ruppert);
/// 2. mark the boundary band from a vertex BFS and make its size even;
/// 3. match the band with [`pair_triangles_with`] under [`BAND_GUARD`];
/// 4. greedily merge the interior (and its augmenting repair);
/// 5. merge the leftovers of both under [`LEFTOVER_FLOOR`];
/// 6. walk the remaining triangles through quads and merge them;
/// 7. collapse doublets, swap edges by valence, and emit the cells.
///
/// Invariants: the band is even before the matching, every quad passes the
/// convexity and exact-area checks of [`candidate_for`], and the final mesh
/// keeps every input point. Errors are returned before any partial mesh would
/// be observable (the function builds and returns a complete `Mesh`).
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
    // An edge owned by exactly one triangle is a domain boundary edge, so its
    // endpoints are the BFS seeds. The hop count is a vertex distance, not a
    // triangle distance: a triangle is in the band only when all three of its
    // vertices are within the radius (a triangle with one deep vertex is an
    // interior triangle and would otherwise drag the band inward).
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
    //
    // A perfect matching needs an even number of nodes; taking an interior
    // triangle rather than dropping a band triangle keeps the whole boundary
    // band matched. The parity test is on the band as a whole, which is the
    // approximation noted in the module documentation.
    if in_band.iter().filter(|&&b| b).count() % 2 == 1 {
        if let Some(t) = first_adjacent_interior(&tris, &in_band) {
            in_band[t] = true;
        }
    }
    let band_ids: Vec<usize> = (0..tris.len()).filter(|&t| in_band[t]).collect();
    stats.band_triangles = band_ids.len();

    // Blossom on the band. `pair_triangles_with` returns the quads and the
    // triangles it could not pair, in order; map the latter back to global ids
    // so the leftover pass can consider them together with the interior ones.
    let band_tris: Vec<[usize; 3]> = band_ids.iter().map(|&t| tris[t]).collect();
    let (mut quads, band_left) = pair_triangles_with(&band_tris, &points, BAND_GUARD);
    stats.band_quads = quads.len();
    let band_left_ids = map_leftovers(&tris, &band_ids, &band_left);

    // ------------------------------------------------------------ interior --
    // Greedy merge only sees the interior triangles; band leftovers are handled
    // by the looser leftover pass below, which is why the two sets are kept
    // apart here.
    let interior_ids: Vec<usize> = (0..tris.len()).filter(|&t| !in_band[t]).collect();
    let (interior_quads, interior_left) =
        greedy_merge(&points, &tris, &interior_ids, GREEDY_FLOOR, options);
    stats.greedy_quads = interior_quads.len();
    quads.extend(interior_quads);

    // ------------------------------------------------------------ leftovers --
    // A band triangle Blossom rejected and an interior triangle the floor
    // rejected are both leftovers; merging them with each other is the paper's
    // Sect. 3.2/4 recovery step.
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
    // The walk edits the quad and triangle lists in place, so both are wrapped
    // in `Option` slots: a consumed cell becomes `None`, which keeps the ids of
    // the others stable. Stats are recomputed after the walk from what is left.
    let mut slot_quads: Vec<Option<[usize; 4]>> = quads.into_iter().map(Some).collect();
    let mut tri_slots: Vec<Option<[usize; 3]>> = leftover_tris.iter().copied().map(Some).collect();
    stats.walk_merges = merge_residuals(&points, &mut slot_quads, &mut tri_slots, options);
    let leftover_tris: Vec<[usize; 3]> = tri_slots.iter().flatten().copied().collect();
    stats.leftover_triangles = leftover_tris.len();

    // -------------------------------------------------------------- cleanup --
    // Boundary nodes are frozen by both clean-up operators (`boundary` is only
    // read), so a collapse/swap can never pull a node off the domain boundary.
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
    // The second pass is not allowed to undo a swap: it only sees the topology
    // the swap left, and the gate in `collapse_doublets` rejects any merged quad
    // that loses too much quality.
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
    // Quads first, then the triangles that survived; unused slots are skipped.
    // The point array is returned unchanged, so every boundary node is still on
    // the input boundary and `verify::boundary_offset` measures ~0.
    let mut cells: Vec<Cell> = slot_quads
        .iter()
        .flatten()
        .map(|quad| Cell::quad4(*quad))
        .collect();
    cells.extend(leftover_tris.iter().map(|tri| Cell::tri3(*tri)));
    Ok((Mesh::new(points, cells), stats))
}

/// Vertex ids used by a boundary edge (an edge owned by exactly one triangle).
///
/// `tris` is the triangulation and `npoints` sizes the returned mask. The mask
/// has one `bool` per input point; a point used by no edge is not a boundary
/// point. Invariant: a mesh boundary is exactly the set of edges with one
/// incident cell, so this is the same predicate [`boundary_vertices_all`] uses
/// on the final mixed mesh.
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
///
/// This is the same one-use edge predicate as [`boundary_vertices`], but on the
/// post-merge cell lists (with `None` slots skipped). It sizes the mask from
/// the largest node id actually used, so it may be shorter than the point array
/// when the mesh leaves a high-numbered node isolated; every caller therefore
/// reads it with `.get(v).copied().unwrap_or(false)`.
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
///
/// The graph is the triangle edge graph: two vertices are adjacent when they
/// share a triangle edge. The BFS is seeded with every boundary vertex (hop 0)
/// and stops at [`Options::band_layers`], so a vertex farther away keeps
/// `u8::MAX`; the band test `hop[v] <= options.band_layers` is false for those.
/// The returned vector has one entry per entry of `boundary` (that is, per
/// point), and the distance is in edge hops, not millimetres.
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
    // `u8::MAX` is the sentinel "farther than the band (or unreachable)", so a
    // vertex is discovered exactly once and never revisited with a longer path.
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
///
/// Scans band triangles in id order and, for each, its three edges in vertex
/// order, returning the first interior owner in that scan order. The scan order
/// is the determinism contract of the even-parity adjustment: the same
/// triangulation always grows the band by the same triangle. Returns `None`
/// only when no interior triangle shares an edge with the band (e.g. the
/// bandwidth covers the whole mesh).
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
///
/// [`pair_triangles_with`] returns the leftover geometry, not indices into the
/// caller's array, so this looks each leftover up by its sorted vertex key and
/// keeps the matching ids of `ids` in their original order. The keys are unique
/// in a triangulation (no two triangles have the same three vertices), so the
/// lookup is exact.
fn map_leftovers(tris: &[[usize; 3]], ids: &[usize], leftover: &[[usize; 3]]) -> Vec<usize> {
    let keys: HashSet<[usize; 3]> = leftover.iter().map(sorted_key).collect();
    ids.iter()
        .copied()
        .filter(|&t| keys.contains(&sorted_key(&tris[t])))
        .collect()
}

/// The canonical key of a triangle: its three vertex ids sorted ascending.
///
/// This is used only for identity lookup and walk-cycle detection; the order
/// carries no orientation, so two triangles that are the same set of vertices
/// map to the same key.
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
///
/// `points` and `tris` are the full mesh; only the triangles in `ids` are
/// matched. The returned quads are the [`Candidate::quad`] values built by
/// [`candidate_for`], so they are counter-clockwise and pass `floor` by
/// construction. `ids` need not be sorted, but the output order follows it.
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
    // The augmenting repair searches this adjacency, so it must be the graph of
    // *accepted* candidates only: an edge that failed the floor can never be
    // used to re-pair a triangle.
    for ring in adjacency.iter_mut() {
        ring.sort_unstable();
        ring.dedup();
    }

    // `usize::MAX` is the unmatched sentinel. `local` holds candidates around
    // the quad just made (the paper's local continuation), `global` holds all
    // candidates; both are max-heaps under `Candidate::cmp`.
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
///
/// The classic alternating-path search that turns a maximal matching into a
/// larger one, with two limits that keep it local: `depth_limit` caps the
/// number of matched pairs on the path, and the DFS marks visited triangles
/// with a per-search generation stamp. It mutates `mate` only when a full path
/// is found, so a failed search leaves the matching unchanged.
struct Augmenter<'a> {
    /// The candidate graph: triangle -> the triangles it can be paired with.
    adjacency: &'a [Vec<usize>],

    /// Current matching, `usize::MAX` for unmatched; mutated in place.
    mate: &'a mut [usize],

    /// Per-triangle visit stamp; `visited[t] == stamp` means "seen in this
    /// search". Reusing one vector avoids clearing it per start.
    visited: Vec<u32>,

    /// Generation counter for `visited`; incremented per call to
    /// [`Augmenter::augment_from`] and wrapped with a full clear on overflow.
    stamp: u32,

    /// Maximum number of matched pairs on one augmenting path, in dual-graph
    /// hops (from [`Options::augment_depth`]).
    depth_limit: usize,
}

impl Augmenter<'_> {
    /// Find an augmenting path from `start` and, if one exists, flip the
    /// matching along it: the two previously unmatched endpoints become matched
    /// and each matched pair on the path is re-paired with its neighbours.
    ///
    /// Returns `true` when the matching grew. Invariant: on `false`, `mate` is
    /// untouched. The generation stamp is bumped once per call; on wrap it is
    /// reset together with `visited`, which keeps the marks meaningful.
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
        // `find` leaves the alternating path as `v1, mate(v1), v2, ...`, with
        // the last vertex unmatched. Prefixing `start` gives the full chain;
        // pairing it in steps of two flips every edge of the path: the old
        // matched edges disappear, the old unmatched ones become the matching.
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
    ///
    /// `current` is always an unmatched-side triangle in the alternating
    /// sense; `depth` counts matched pairs already on the path. The search
    /// stops at `depth_limit`, skips the edge back to the current mate, and
    /// marks both triangles of a matched pair when entering it so the same
    /// pair cannot appear twice on one path.
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
///
/// This is the single gate every merged pair passes, whichever stage builds it:
/// the union must be a convex counter-clockwise quad, its area must equal the
/// sum of the two triangle areas (so the diagonal really is the shared edge and
/// the pair covers exactly the same region), and its minimum scaled Jacobian and
/// corner angles must clear `floor`. Returns `None` on any failure; the caller
/// treats that as "this pair is not a candidate".
///
/// The score is the minimum over the four corners, i.e. the quad's worst
/// Jacobian, because a single bad corner is what a stress analysis sees. The
/// angle is the minimum corner angle, used only as a tie-break and needle guard.
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

/// The three directed edges of a triangle, in vertex order.
///
/// Used only to enumerate candidate shared edges; the direction is the
/// triangle's own orientation and is not normalized.
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

/// Signed area of a quad, in square millimetres (positive when the vertices go
/// counter-clockwise). Shoelace formula on the four vertices; the caller uses
/// it both as an area and as an orientation test.
fn polygon_signed_area(quad: &[usize; 4], points: &[Node]) -> f64 {
    let mut sum = 0.0;
    for i in 0..4 {
        let a = points[quad[i]];
        let b = points[quad[(i + 1) % 4]];
        sum += a[0] * b[1] - b[0] * a[1];
    }
    sum / 2.0
}

/// Signed area of a triangle, in square millimetres (positive when the three
/// vertices are counter-clockwise). It is the cross product of two edges, so a
/// degenerate triangle gives ~0 and the area check in [`candidate_for`] then
/// rejects the pair.
fn triangle_area(tri: &[usize; 3], points: &[Node]) -> f64 {
    let a = points[tri[0]];
    let b = points[tri[1]];
    let c = points[tri[2]];
    ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2.0
}

/// The XY coordinates of a quad's four vertices, in the quad's order.
fn quad_coords(quad: &[usize; 4], points: &[Node]) -> [[f64; 2]; 4] {
    [
        [points[quad[0]][0], points[quad[0]][1]],
        [points[quad[1]][0], points[quad[1]][1]],
        [points[quad[2]][0], points[quad[2]][1]],
        [points[quad[3]][0], points[quad[3]][1]],
    ]
}

/// Strictly convex and counter-clockwise.
///
/// A thin wrapper over [`convex_ccw_poly`] that fixes the meaning of the
/// predicate for callers: every turn must be strictly left (cross product >
/// 0) and the enclosed area must be positive, so a degenerate or collinear
/// corner is rejected.
fn convex_ccw(quad: &[[f64; 2]; 4]) -> bool {
    convex_ccw_poly(quad)
}

/// Minimum scaled Jacobian of a quad over its four corners (dimensionless,
/// -1..1; higher is better). `f64::INFINITY` only if the quality function
/// returns no corner.
fn quad_min_sj(quad: &[usize; 4], points: &[Node]) -> f64 {
    quality::scaled_jacobian(&Cell::quad4(*quad), points)
        .into_iter()
        .fold(f64::INFINITY, f64::min)
}

/// Minimum corner angle of a quad, in degrees (0..180; higher is better).
/// It is the tie-break guard used by the clean-up and the doublet collapse.
fn quad_min_angle(quad: &[usize; 4], points: &[Node]) -> f64 {
    quality::quad_angles(&Cell::quad4(*quad), points)
        .into_iter()
        .fold(f64::INFINITY, f64::min)
}

/// Canonical undirected edge key: the two vertex ids in ascending order.
///
/// Every edge map in this module is keyed this way, so the direction a cell
/// happens to store does not affect lookup.
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
///
/// The gates are: exact area conservation (1e-9 relative), minimum scaled
/// Jacobian >= 0.2, a loss of at most 0.15 from the worse of the two parents,
/// minimum corner angle >= 20 degrees, and longest edge <= 2.5 `size`. The
/// quality gates are tuning choices: a doublet is not geometrically invalid, but
/// its 180-degree corner has zero Jacobian, so removing it is worth a small
/// quality cost. Boundary vertices are skipped because a boundary doublet can
/// only slide along the straight segment (see the cross-field route, where that
/// is fatal).
///
/// The outer loop is capped at 8 passes and restarts the scan after every
/// accepted collapse, so a vertex that becomes a doublet because a neighbour
/// collapsed is still found. Returns the number of quads removed.
fn collapse_doublets(
    points: &[Node],
    quads: &mut Vec<Option<[usize; 4]>>,
    tris: &[[usize; 3]],
    boundary: &[bool],
    size: f64,
) -> usize {
    let mut collapses = 0usize;
    for _ in 0..8 {
        // `incident` lists the quads around each vertex; `cell_count` counts
        // every incident cell, including triangles. A doublet has exactly two
        // incident cells and both must be quads, so a triangle touching the
        // vertex disqualifies it (merging would leave a crack).
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
///
/// The centroid is the mean of the four vertices. Sorting by `atan2` is valid
/// only when the polygon is star-shaped around that centroid; the caller then
/// checks convexity, so a wrongly ordered set is rejected rather than used.
/// Ties on the angle are broken by vertex id, which keeps the result
/// deterministic. Returns `None` unless there are exactly four vertices.
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

/// Longest edge of a quad, in millimetres. Used by [`collapse_doublets`] to
/// keep a collapsed quad compatible with the local size field (2.5 `size`).
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
///
/// Boundary vertices are excluded from the irregularity sum (their valence is
/// fixed by the domain boundary) but still count for the geometry checks. The
/// round loop is capped at 32 and each round rebuilds all indices from scratch,
/// so a swap's effect is seen by the next round; within a round the edge set is
/// updated eagerly so two swaps cannot create the same diagonal. Returns the
/// number of swaps taken.
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
            // The two candidate diagonals of the hexagon. The delta is the
            // change each one makes to the four shared vertices: x and y lose
            // one incident edge, the new diagonal's endpoints gain one.
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
                    // An existing diagonal would give two cells the same edge
                    // twice, i.e. a non-manifold mesh; the edge set includes
                    // the triangle edges, so a quad cannot cut through one.
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

/// Irregularity of a set of vertices: `sum |valence - 4|` over the vertices
/// that are not on the domain boundary.
///
/// Unitless (a sum of valence differences). Boundary vertices are excluded
/// because their valence is imposed by the boundary loops and cannot be made 4
/// by an interior swap; including them would bias the clean-up toward moving
/// the irregularity onto the boundary. A lower value is better.
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
///
/// The walk is run for up to [`MAX_WALK_PASSES`] passes over the current
/// residual list; a pass that merges nothing ends the loop, so it cannot cycle.
/// It is the step that made the route competitive: measured over the seven
/// corpus parts, it reduced the residuals from 880 (band+greedy+leftover only) to
/// 362 (-59 %). Returns the number of pairs merged.
fn merge_residuals(
    points: &[Node],
    quads: &mut Vec<Option<[usize; 4]>>,
    tris: &mut [Option<[usize; 3]>],
    options: &Options,
) -> usize {
    let mut merges = 0usize;
    for _ in 0..MAX_WALK_PASSES {
        // `nq` is the current quad/triangle cell-id boundary: quads are 0..nq,
        // residual triangles are nq+tid. A merge appends a quad, so the boundary
        // moves and the edge index is rebuilt; triangle slot ids stay fixed.
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
///
/// This is the *base* snapshot for one walk pass: it encodes the mesh before
/// any branch of the backtracking search is applied. [`Walk`] reads it directly
/// for unchanged edges and keeps a per-branch override map for the edges a move
/// touches, so a rollback is an undo-log pop rather than a rebuild.
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
///
/// The pair `(tri, quad)` is the replacement partition of the same pentagon:
/// the moved triangle and the reshaped quad. Both are fully validated in
/// [`move_splits`] before a `SplitOption` is built, so applying one only has to
/// keep the cell and edge bookkeeping in sync.
struct SplitOption {
    /// The moved triangle, counter-clockwise, with the same area as before.
    tri: [usize; 3],

    /// The reshaped quad, counter-clockwise and convex.
    quad: [usize; 4],

    /// Minimum scaled Jacobian of `quad` (dimensionless); higher is better and
    /// the primary ordering key of [`move_splits`].
    quad_sj: f64,

    /// Minimum corner angle of `tri`, in degrees; the secondary ordering key
    /// and the guard against a degenerate moved triangle (>= 2 degrees).
    tri_angle: f64,
}

/// One backtracking branch of the Sect. 3.3 walk.
///
/// The walk is a depth-first search over "move the triangle through one of its
/// adjacent quads" moves. It mutates the shared `quads`/`tris` arrays while a
/// branch is explored and undoes every move on backtracking; `overrides` and
/// `log` are the edge-index delta of the current branch.
struct Walk<'a> {
    /// Mesh coordinates (never mutated by the walk).
    points: &'a [Node],

    /// Quads, with `None` for a cell consumed by the current branch. A move
    /// appends the new quad; ids of existing quads never change.
    quads: &'a mut Vec<Option<[usize; 4]>>,

    /// Residual triangles, indexed by slot id; `None` means merged or moved.
    tris: &'a mut [Option<[usize; 3]>],

    /// The pre-branch edge index from [`build_mesh_edges`].
    base: &'a HashMap<(usize, usize), Vec<usize>>,

    /// Quad/triangle cell-id boundary of `base`: quads are `0..nq`, triangle
    /// cell ids are `nq + slot`.
    nq: usize,

    /// Edge owners changed by the moves applied on the current branch.
    overrides: HashMap<(usize, usize), Vec<usize>>,

    /// Undo log for `overrides`: `(edge, previous override)`.
    log: Vec<((usize, usize), Option<Vec<usize>>)>,

    /// Remaining search states for this triangle; decremented once per expanded
    /// state and never below zero (see [`Options::walk_budget`]).
    budget: usize,

    /// Sorted triangle keys already on the current path. It rejects a branch
    /// that would revisit a triangle, which is what makes the DFS terminate on
    /// a cyclic dual graph.
    visited: HashSet<[usize; 3]>,
}

impl<'a> Walk<'a> {
    /// A walk over one pass's mesh snapshot, with a fresh empty branch state.
    ///
    /// `base` must have been built with the same `nq`; `walk_budget` is copied
    /// into `budget` and consumed across the whole search of this triangle.
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

    /// Whether `edge` exists in the mesh as modified by the current branch.
    ///
    /// Overrides can be empty (the edge was removed by a move), which is why
    /// this cannot just test `overrides.contains_key`.
    fn edge_present(&self, edge: (usize, usize)) -> bool {
        match self.overrides.get(&edge) {
            Some(list) => !list.is_empty(),
            None => self.base.contains_key(&edge),
        }
    }

    /// Merge the current triangle with an adjacent residual triangle, if the
    /// union is a convex quad with scaled Jacobian at least 0.2.
    ///
    /// Targets are collected from the triangle's three edges, then tried in
    /// ascending slot order, so the first valid partner is deterministic. A
    /// merge consumes both triangles and appends the new quad; it does not go
    /// through the override log because the walk ends immediately afterwards.
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
    ///
    /// The search order is: (1) merge with an adjacent residual, (2) stop when
    /// `depth` or the budget is exhausted, (3) reject a triangle key already on
    /// the path, (4) try every adjacent quad and every valid pentagon split of
    /// it, best split first. Returns `true` only when a merge was committed; in
    /// that case the branch is *not* rolled back (the pass ends and rebuilds its
    /// edge index).
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
            // The triangle is already on the current path; moving it again
            // would just rotate the same pentagon partition around a cycle.
            return false;
        }

        // Only quads can be moved through; `owner < nq` selects a quad cell id,
        // while a triangle owner is a *target* for `merge_adjacent`, not a move.
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
                // Failed branch: undo the edge overrides and restore the two
                // cells. The log is popped back to `mark`; every entry the
                // recursive call added is above it.
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
    ///
    /// Only edges whose incidence can change are touched: the old and new
    /// triangle edges and the old and new quad edges. For each, the current
    /// owners are taken (base or override), the two cells are removed, and the
    /// cell is re-added only if its new polygon actually has the edge. The
    /// previous override is logged first so [`Walk::rollback`] can restore it.
    fn apply_split(
        &mut self,
        tid: usize,
        qid: usize,
        old_tri: &[usize; 3],
        old_quad: &[usize; 4],
        split: &SplitOption,
    ) {
        // The triangle's cell id is its slot id offset by the quad count; the
        // quad keeps its own id. A move therefore never renumbers a cell.
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

    /// Undo the edge-owner overrides recorded above `mark`.
    ///
    /// `mark` is the log length before a split was applied. Every entry the
    /// failed branch (or its recursion) added is popped and its previous value
    /// restored, so the edge index returns to exactly the state at `mark`. The
    /// cell arrays are restored by the caller, which holds the old values.
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

/// Whether the polygon `nodes` uses `edge` in either direction.
///
/// Works for a triangle or a quad; it reads the polygon cyclically, so the
/// stored vertex order is the cell's boundary order.
fn cell_has_edge(nodes: &[usize], edge: (usize, usize)) -> bool {
    (0..nodes.len()).any(|i| edge_key(nodes[i], nodes[(i + 1) % nodes.len()]) == edge)
}

/// The pentagon `p, s, t, q, r` covered by triangle `(p,q,r)` and quad
/// `(q,p,s,t)`, or `None` when they do not share exactly one edge.
///
/// The shared edge is `p -> q` in the triangle and `q -> p` in the quad, which
/// is the only way two counter-clockwise cells can share an edge. The returned
/// cycle walks the pentagon once counter-clockwise starting at `p`; it is the
/// union polygon the moved triangle is cut from.
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
///
/// A pentagon split takes three consecutive corners as the new triangle and the
/// remaining four as the new quad; there are five of them, one for each starting
/// corner. A split is kept only when the new triangle is not degenerate
/// (`area > 1e-14`, minimum angle >= 2 degrees), the new quad is convex with
/// scaled Jacobian >= 0.2, the two areas still sum to the pentagon area, and the
/// new diagonal is not already a mesh edge (which would create a non-manifold
/// edge). The returned list is sorted by descending quad Jacobian, then
/// descending triangle angle, so the DFS tries the best split first. These
/// thresholds are the walk's guards; they are tuning choices, not paper
/// constants.
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

    // Five ways to choose the new triangle's first corner; each also defines
    // the new quad as the complementary four corners. The old partition is
    // skipped by its sorted key, not by a comparison of the arrays, because the
    // starting corner may differ while the triangle is the same.
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

/// Minimum corner angle of a triangle, in degrees (0..180). The walk uses it to
/// reject a moved triangle that would become a needle (floor 2 degrees).
fn triangle_min_angle(tri: &[usize; 3], points: &[Node]) -> f64 {
    quality::corner_angles(&Cell::tri3(*tri), points)
        .into_iter()
        .fold(f64::INFINITY, f64::min)
}

/// Merge two triangles sharing an edge, if the union is a convex quad with a
/// scaled Jacobian of at least 0.2 and exactly the same area.
///
/// This is the walk's final step (and the only merge the walk commits): unlike
/// [`candidate_for`] it has no adjustable floor, because the residual pair is
/// the last chance to avoid leaving a triangle. The area equality is the
/// conformity proof: a quad whose area differs from the two triangles would
/// either overlap a neighbour or leave a gap.
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

/// Signed area of a polygon given by its XY coordinates, in square
/// millimetres. Shoelace formula; it works for the 5-gon of the walk as well as
/// for a quad, and its sign is the polygon's orientation.
fn coords_area(coords: &[[f64; 2]]) -> f64 {
    let mut sum = 0.0;
    for i in 0..coords.len() {
        let a = coords[i];
        let b = coords[(i + 1) % coords.len()];
        sum += a[0] * b[1] - b[0] * a[1];
    }
    sum / 2.0
}

/// Strictly convex, counter-clockwise, with positive area, for a polygon given
/// by its coordinates.
///
/// Every turn must be strictly left (cross product > 0) and the signed area must
/// be positive. A collinear corner (cross product 0) is rejected, which is what
/// keeps a degenerate quad out of the mesh.
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
        // count is the quads formed minus the collapses. The test domains are
        // chosen so no collapse fires; if one did, the two quads covered four
        // of the original triangles and the left side would be short by
        // `2 * doublet_collapses` (add it back for an area identity).
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

    /// Count the quads that are strictly convex and counter-clockwise, using the
    /// same predicate as the route; compared against `gate.quads` so the test
    /// checks that every emitted quad is convex, not just non-inverted.
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

    /// A small square: the mesh must be quad-dominant, conforming, watertight,
    /// with boundary offset ~0 and every emitted quad convex.
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

    /// Non-default options must still produce a valid, conforming mesh, even
    /// when they deliberately weaken every heuristic (thin band, shallow
    /// augmenting repair and walk).
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

    /// The plate-with-hole test checks the even-parity contract directly
    /// (`band_triangles % 2 == 0`), plus watertightness and the stage sums.
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

    /// The floor is a quality guard, not a broken construction: the same sliver
    /// pair is a valid convex quad when the floor is loosened, and is refused by
    /// [`GREEDY_FLOOR`] because its worst Jacobian is below 0.05.
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

    /// The walk is the only operator that can join two residual triangles that
    /// are not adjacent; after the merge the mesh must be manifold, crack-free
    /// and area-preserving, and every new quad must clear the 0.2 floor.
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

    /// A zero-area domain is a rejected request, not a degenerate mesh: the
    /// route must return `Err` rather than panic.
    #[test]
    fn a_domain_with_no_area_is_refused() {
        // Collinear outer loop: the domain has no area and the route returns an
        // error instead of a degenerate mesh.
        let domain = Domain::new(vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], Vec::new()).unwrap();
        assert!(mesh(&domain, &Sizing::uniform(0.5).unwrap()).is_err());
    }
}
