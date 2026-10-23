//! Q-Morph: converting a triangle mesh into quads by advancing a front.
//!
//! Implements the Q-Morph construction of Owen, Staten, Canann & Saigal (1999,
//! "Q-Morph: an indirect approach to advancing front quad meshing"; corpus
//! `owen-1999`). The module follows [[Pedroso-Mesher-W3-Design]] for the
//! data-structure choices. Quads and triangles are cells of **one** mesh
//! ([`CellMesh`]), so every operation — edge flip, node insertion, cell
//! replacement — is expressed on one adjacency structure, and a flip can never
//! cut through a quad that has already been placed.
//!
//! The front is the interface between the unconverted region and the meshed one.
//! For a front edge `(a, b)` the procedure places two side nodes `c`, `d` at the
//! local size along the inward normal, recovers the three missing sides by flips
//! among triangles, and replaces the triangles enclosed by `(a, b, d, c)` with
//! one quad. The policy that decides *which* nodes is the part the earlier
//! attempts got wrong; it is spelled out here rather than tuned:
//!
//! 1. **Side nodes from the reachable unconverted region first.** Candidates are
//!    nodes of the live triangles reachable from the front edge, whose segment
//!    from the front vertex stays inside live triangles, and which lie ahead of
//!    the front. The pair that maximises the resulting quad's shape is used.
//!    The design's 0.5–1.5 × local size is measured against the triangulation:
//!    Ruppert only bounds the circumradius, so nodes can be up to ~2 × the field
//!    size away (a 4 mm plate had a 3.9 mm front edge whose nearest inward node
//!    was 7.53 mm away). The range used here is 0.5–2.2 ×; with 1.5 the front
//!    emptied its candidate set on that part at 40 % coverage.
//! 2. **Insert only on a size mismatch.** A side node is inserted only when the
//!    front edge itself is outside 0.5–1.5 × the local size, never as a rescue
//!    for a failed quad. The measured failure of the fallback is in the design
//!    note: insertion on every failure turned 604 triangles into 5 022. In the
//!    measured runs this rule almost never fires, and that is the point.
//! 3. **Never roll back by cloning.** Inserted nodes and recovery flips are
//!    ordinary cells of the unconverted region, so a partial attempt leaves a
//!    valid mesh.
//! 4. **Block around progress.** A failed edge is blocked; a placed quad
//!    unblocks only the edges that touch its nodes.
//!
//! Boundary loops are processed by ascending perimeter, so a hole's rows are
//! laid before the outer loop's coarse rows reach it.
//!
//! **State machine.** `front_loop` repeatedly takes the front loops in
//! ascending perimeter, tries the shortest edge of a loop, and restarts after a
//! success. A refused edge is blocked until a quad touching one of its nodes is
//! placed, which is rule 4. `try_quad` is the per-edge sequence: seed and
//! orient the front triangle, find side nodes, pick and rank side pairs, recover
//! the three unused sides, validate that the enclosed region is exactly the
//! quad, optionally check the even-region guard, then commit. [`Refusals`]
//! counts the refusing stage; a refusal is not an error, it is "try again after
//! the front changes". Two hard stops bound a runaway front: `max_attempts`
//! (200 per input triangle plus 10 000) and `max_triangles` (4 per input
//! triangle plus 100).
//!
//! **Not implemented: the transition rules.** A front edge that disagrees with
//! the field is not split or merged, and two fronts that meet are not stitched
//! with a seam. The front fragments into small loops (many of them one
//! triangle, three front edges) and those cannot be quadrangulated without
//! changing the loop's parity. The leftover triangles are then handed to
//! [`crate::sun::close_residuals`], Sun et al.'s residual-triangle walk (corpus
//! `sun-2015`, `sun-2016`) with the boundary M2/M3 and M6 operations of Sun
//! 2015 §5. On the six parts it closes plate-hole 4/2 mm and stepped completely
//! (68→0, 292→0, 382→0) and leaves 2/3/6 triangles on plate-hole 1 mm,
//! l-bracket and ring (962→2, 78→3, 64→6); those are isolated interior slivers
//! that no pentagon recombination can move. The module documentation there
//! records the geometric reason.
//!
//! **The policy is not enforced by this code.** Nothing here refuses a run that
//! ignores it; measure the mesh that came out with [`crate::verify::gate`].

use std::collections::{HashMap, HashSet};

use crate::mesh::Node;

/// A cell of the converting mesh: either a triangle or a quad.
///
/// The conversion replaces a group of live triangles by one quad; both kinds of
/// cell stay in the same [`CellMesh`] so the front's adjacency always sees the
/// quads that are already placed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cell {
    /// An unconverted triangle, counter-clockwise.
    Tri([usize; 3]),

    /// A placed quad, counter-clockwise.
    Quad([usize; 4]),
}

impl Cell {
    /// Number of nodes of the cell (3 or 4).
    fn n_nodes(&self) -> usize {
        match self {
            Cell::Tri(_) => 3,
            Cell::Quad(_) => 4,
        }
    }

    /// The `i`-th node; `i` must be less than [`Cell::n_nodes`].
    fn node(&self, i: usize) -> usize {
        match self {
            Cell::Tri(t) => t[i],
            Cell::Quad(q) => q[i],
        }
    }

    /// The `i`-th edge as a sorted pair, so the two sides share one key.
    fn edge(&self, i: usize) -> (usize, usize) {
        let (a, b) = (self.node(i), self.node((i + 1) % self.n_nodes()));
        (a.min(b), a.max(b))
    }

    /// True for a triangle, false for a quad.
    fn is_tri(&self) -> bool {
        matches!(self, Cell::Tri(_))
    }
}

/// Reusable adjacency for [`CellMesh::front_loops`].
///
/// Kept across calls because rebuilding the vertex->edge map for every placed
/// quad was the hottest part of the converter.
#[derive(Default)]
struct LoopScratch {
    /// Edge ids incident to each vertex, for this call only; cleared at the end.
    at: Vec<Vec<usize>>,

    /// Per-edge "already walked" flags, indexed like the sorted edge list.
    used: Vec<bool>,
}

/// Reusable marks for [`CellMesh::removing_leaves_odd_component`] and
/// [`CellMesh::live_tri_component_sizes`].
///
/// The stamp/generation pair avoids clearing a `Vec<bool>` over all cells for
/// every candidate placement: a cell is marked when its stamp equals the current
/// generation, and a generation wraps only after `u32::MAX` calls.
#[derive(Default)]
struct FragmentCheck {
    /// Per-cell generation stamp; `stamp[id] == generation` means "already seen".
    stamp: Vec<u32>,

    /// Current generation; incremented per call and reset on wrap-around.
    generation: u32,

    /// DFS stack reused between components.
    stack: Vec<usize>,
}

/// A mesh whose cells are triangles and quads, with edge adjacency, a bucket
/// index over edges (for recovery) and the current front.
///
/// Invariants maintained by `index_cell` / `unindex_cell`: every live cell
/// appears exactly once in every edge list of its own edges; `front` is exactly
/// the set of edges with one live triangle and at most one other (quad or no
/// cell); `live_tris` equals the number of `Some(Cell::Tri(_))` cells; and
/// `edge_buckets`/`node_edges` contain exactly the edges present in `edges`.
/// A removed cell is `None`, not physically deleted, so ids stay stable.
pub struct CellMesh {
    /// All nodes in the XY plane (`z = 0`); placement may append to this.
    pub points: Vec<Node>,

    /// Cells by id; `None` is a removed cell. Removed ids are never reused.
    pub cells: Vec<Option<Cell>>,

    /// Edge key -> owning cell ids. An edge has at most two owners in a valid
    /// mesh; a single owner is a front or domain-boundary edge.
    edges: HashMap<(usize, usize), Vec<usize>>,

    /// Bucket index over edges: recovery asks for the edges a segment crosses,
    /// and scanning the edge map for that is O(cells) per flip.
    edge_buckets: HashMap<(i64, i64), Vec<(usize, usize)>>,

    /// Bucket side length, in millimetres; the size field, clamped to positive
    /// at [`CellMesh::new`]. It also sets the insertion-spacing floor.
    bucket: f64,

    /// Edges at the interface between the converted and unconverted regions.
    front: HashSet<(usize, usize)>,

    /// Unconverted triangles still alive; kept as a counter so the runaway guard
    /// is O(1) rather than a scan of the cell list.
    live_tris: usize,

    /// Edges incident to each node, for side-edge definition by angle.
    node_edges: HashMap<usize, Vec<(usize, usize)>>,

    /// Live triangle ids, in insertion order. The parity guard walks the
    /// unconverted triangles; scanning the whole cell array (which keeps
    /// removed ids as `None`) made each guard call pay for the converted half
    /// of the mesh too. Removed triangles stay in the list with
    /// `tri_alive[id] == false`; ids are never reused, so this is a stable
    /// append-only log.
    tri_list: Vec<usize>,

    /// `tri_alive[id]` is true while cell `id` is a live triangle.
    tri_alive: Vec<bool>,

    /// `tri_listed[id]` is true once `id` has been appended to `tri_list`;
    /// a flip re-indexes the same triangle, and the log must not duplicate it.
    tri_listed: Vec<bool>,
}

impl CellMesh {
    /// Build the converting mesh from an input triangulation.
    ///
    /// Every input triangle is a live cell and every edge is indexed; the front
    /// starts as the domain boundary (edges with exactly one triangle and no
    /// quad). `size` is the target edge length in millimetres and also sets the
    /// edge-bucket size; a non-positive value is clamped to 1 there so the
    /// bucket keys stay finite.
    pub fn new(points: &[Node], tris: &[[usize; 3]], size: f64) -> Self {
        let mut mesh = CellMesh {
            points: points.to_vec(),
            cells: tris.iter().map(|t| Some(Cell::Tri(*t))).collect(),
            edges: HashMap::new(),
            edge_buckets: HashMap::new(),
            bucket: if size > 1e-9 { size } else { 1.0 },
            front: HashSet::new(),
            // `index_cell` below counts the initial triangles; starting at
            // `tris.len()` here counted them twice, which made the runaway guard
            // and the fragment guard see twice the real region.
            live_tris: 0,
            node_edges: HashMap::new(),
            tri_list: Vec::new(),
            tri_alive: vec![false; tris.len()],
            tri_listed: vec![false; tris.len()],
        };
        for id in 0..mesh.cells.len() {
            mesh.index_cell(id);
        }
        mesh
    }

    /// Integer bucket coordinate of `p` for [`CellMesh::edge_buckets`].
    fn bucket_key(&self, p: Node) -> (i64, i64) {
        (
            (p[0] / self.bucket).floor() as i64,
            (p[1] / self.bucket).floor() as i64,
        )
    }

    /// Insert `edge` into every bucket its bounding box touches, and into the
    /// two endpoints' node-edge lists.
    ///
    /// The edge is registered in all cells of the box (not just the endpoint
    /// cells) because [`CellMesh::first_crossing`] scans a segment's whole
    /// bounding box; over-registering is safe, missing an edge would make
    /// recovery fail.
    fn add_edge_bucket(&mut self, edge: (usize, usize)) {
        let (a, b) = (self.points[edge.0], self.points[edge.1]);
        let (i0, j0) = self.bucket_key(a);
        let (i1, j1) = self.bucket_key(b);
        let (lo_i, hi_i) = (i0.min(i1), i0.max(i1));
        let (lo_j, hi_j) = (j0.min(j1), j0.max(j1));
        for i in lo_i..=hi_i {
            for j in lo_j..=hi_j {
                self.edge_buckets.entry((i, j)).or_default().push(edge);
            }
        }
        self.node_edges.entry(edge.0).or_default().push(edge);
        self.node_edges.entry(edge.1).or_default().push(edge);
    }

    /// Remove `edge` from the same buckets and node-edge lists it was added to.
    fn remove_edge_bucket(&mut self, edge: (usize, usize)) {
        let (a, b) = (self.points[edge.0], self.points[edge.1]);
        let (i0, j0) = self.bucket_key(a);
        let (i1, j1) = self.bucket_key(b);
        let (lo_i, hi_i) = (i0.min(i1), i0.max(i1));
        let (lo_j, hi_j) = (j0.min(j1), j0.max(j1));
        for i in lo_i..=hi_i {
            for j in lo_j..=hi_j {
                if let Some(list) = self.edge_buckets.get_mut(&(i, j)) {
                    list.retain(|&e| e != edge);
                    if list.is_empty() {
                        self.edge_buckets.remove(&(i, j));
                    }
                }
            }
        }
        for v in [edge.0, edge.1] {
            if let Some(list) = self.node_edges.get_mut(&v) {
                list.retain(|&e| e != edge);
                if list.is_empty() {
                    self.node_edges.remove(&v);
                }
            }
        }
    }

    /// Add the cell's edges to the adjacency and refresh the front around them.
    ///
    /// Only the first owner of an edge registers it in the bucket index; the
    /// second owner just appends to `edges`, and `refresh_front` then sees two
    /// owners and removes the edge from the front.
    fn index_cell(&mut self, id: usize) {
        let Some(Some(cell)) = self.cells.get(id).copied() else {
            return;
        };
        if cell.is_tri() {
            self.live_tris += 1;
            if self.tri_alive.len() <= id {
                self.tri_alive.resize(id + 1, false);
                self.tri_listed.resize(id + 1, false);
            }
            self.tri_alive[id] = true;
            if !self.tri_listed[id] {
                self.tri_listed[id] = true;
                self.tri_list.push(id);
            }
        }
        for i in 0..cell.n_nodes() {
            let edge = cell.edge(i);
            let was_empty = {
                let list = self.edges.entry(edge).or_default();
                let empty = list.is_empty();
                list.push(id);
                empty
            };
            if was_empty {
                self.add_edge_bucket(edge);
            }
            self.refresh_front(edge);
        }
    }

    /// Remove the cell's edges from the adjacency, dropping an edge entirely
    /// when its last owner goes away, and refresh the front around each.
    fn unindex_cell(&mut self, id: usize) {
        let Some(Some(cell)) = self.cells.get(id).copied() else {
            return;
        };
        if cell.is_tri() {
            self.live_tris = self.live_tris.saturating_sub(1);
            if let Some(alive) = self.tri_alive.get_mut(id) {
                *alive = false;
            }
        }
        for i in 0..cell.n_nodes() {
            let edge = cell.edge(i);
            let mut empty = false;
            if let Some(list) = self.edges.get_mut(&edge) {
                list.retain(|&c| c != id);
                empty = list.is_empty();
            }
            if empty {
                self.edges.remove(&edge);
                self.remove_edge_bucket(edge);
            }
            self.refresh_front(edge);
        }
    }

    /// Recompute whether `edge` is a front edge and insert/remove it.
    ///
    /// Called after every index change on an edge; `is_front` is cheap (the
    /// owner list is at most two entries), so no incremental bookkeeping is
    /// kept beyond the set itself.
    fn refresh_front(&mut self, edge: (usize, usize)) {
        if self.is_front(edge) {
            self.front.insert(edge);
        } else {
            self.front.remove(&edge);
        }
    }

    /// True when the edge separates the unconverted region from the meshed one:
    /// exactly one adjacent cell is a triangle and there is at most one other
    /// cell (a quad). A domain-boundary edge has one triangle and no other cell;
    /// an interface edge has one triangle and one quad.
    fn is_front(&self, edge: (usize, usize)) -> bool {
        let Some(owners) = self.edges.get(&edge) else {
            return false;
        };
        if owners.len() > 2 {
            return false;
        }
        let tris = owners
            .iter()
            .filter(|&&c| {
                self.cells
                    .get(c)
                    .and_then(|cell| cell.as_ref())
                    .is_some_and(Cell::is_tri)
            })
            .count();
        tris == 1
    }

    /// Ids of the live (`Some`) cells, in ascending order.
    fn cell_ids(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.cells.len()).filter(|&i| self.cells[i].is_some())
    }

    /// Owners of `edge`, or an empty slice when it is not in the mesh.
    fn edge_cells(&self, edge: (usize, usize)) -> &[usize] {
        self.edges.get(&edge).map(Vec::as_slice).unwrap_or(&[])
    }

    /// The unconverted triangles, in cell order.
    ///
    /// Cell order, not geometric order: the same mesh always yields the same
    /// vector, which keeps the downstream Sun closure deterministic.
    fn triangles(&self) -> Vec<[usize; 3]> {
        self.cell_ids()
            .filter_map(|id| match self.cells[id] {
                Some(Cell::Tri(t)) => Some(t),
                _ => None,
            })
            .collect()
    }

    /// Sizes of the connected components of the unconverted triangles, by shared
    /// edge, largest first. **One entry** means the remaining region is a single
    /// triangulation — the state the two-stage conversion hands to the matching
    /// stage. More than one means the front stranded islands, and a matching
    /// stage would leave one triangle per odd island.
    ///
    /// The sizes include only triangles; a quad between two islands does not
    /// connect them. Returns an empty vector when no unconverted triangle is
    /// left. Each component is found once (iterative DFS, shared-edge
    /// adjacency), so the cost is linear in the number of live cells.
    pub fn live_tri_component_sizes(&self) -> Vec<usize> {
        let mut seen = vec![false; self.cells.len()];
        let mut sizes = Vec::new();
        let mut stack = Vec::new();
        for index in 0..self.tri_list.len() {
            let start = self.tri_list[index];
            if seen[start] || !self.tri_alive[start] {
                continue;
            }
            seen[start] = true;
            stack.push(start);
            let mut size = 0usize;
            while let Some(id) = stack.pop() {
                size += 1;
                let Some(Some(cell)) = self.cells.get(id).copied() else {
                    continue;
                };
                for i in 0..cell.n_nodes() {
                    for &other in self.edge_cells(cell.edge(i)) {
                        if other != id
                            && !seen[other]
                            && self
                                .cells
                                .get(other)
                                .and_then(|c| c.as_ref())
                                .is_some_and(Cell::is_tri)
                        {
                            seen[other] = true;
                            stack.push(other);
                        }
                    }
                }
            }
            sizes.push(size);
        }
        sizes.sort_unstable_by(|a, b| b.cmp(a));
        sizes
    }

    /// True when removing `region` (all triangles) would leave an unconverted
    /// component with an **odd** number of triangles.
    ///
    /// A component of odd size can never be paired completely — matching leaves
    /// one triangle — so the front must not create one. Even components are
    /// fine: parity of a triangulated region equals parity of its boundary, so
    /// an even component has an even boundary and a perfect pairing is possible
    /// (edge flips fix the structures where it is not).
    /// The guard is conservative: it returns `true` as soon as **any** remaining
    /// component is odd, even if the region being removed is not adjacent to it.
    /// That is what the two-stage conversion wants — it refuses the placement,
    /// not just the bad component. `region` ids are the triangles about to be
    /// consumed by the new quad; `scratch` is reused across calls.
    fn removing_leaves_odd_component(&self, region: &[usize], scratch: &mut FragmentCheck) -> bool {
        let remaining = self.live_tris.saturating_sub(region.len());
        if remaining == 0 {
            return false;
        }
        if scratch.stamp.len() < self.cells.len() {
            scratch.stamp.resize(self.cells.len(), 0);
        }
        scratch.generation = scratch.generation.wrapping_add(1);
        if scratch.generation == 0 {
            for s in &mut scratch.stamp {
                *s = 0;
            }
            scratch.generation = 1;
        }
        let generation = scratch.generation;
        for &id in region {
            scratch.stamp[id] = generation;
        }
        for start in 0..self.tri_list.len() {
            let start = self.tri_list[start];
            if scratch.stamp[start] == generation || !self.tri_alive[start] {
                continue;
            }
            scratch.stamp[start] = generation;
            scratch.stack.push(start);
            let mut size = 0usize;
            while let Some(id) = scratch.stack.pop() {
                size += 1;
                let Some(Some(cell)) = self.cells.get(id).copied() else {
                    continue;
                };
                for i in 0..cell.n_nodes() {
                    for &other in self.edge_cells(cell.edge(i)) {
                        if scratch.stamp[other] == generation {
                            continue;
                        }
                        if self
                            .cells
                            .get(other)
                            .and_then(|c| c.as_ref())
                            .is_some_and(Cell::is_tri)
                        {
                            scratch.stamp[other] = generation;
                            scratch.stack.push(other);
                        }
                    }
                }
            }
            if size % 2 == 1 {
                return true;
            }
        }
        false
    }

    /// Replace cell `id`, keeping the adjacency consistent (unindex, assign,
    /// index).
    fn set_cell(&mut self, id: usize, cell: Cell) {
        self.unindex_cell(id);
        self.cells[id] = Some(cell);
        self.index_cell(id);
    }

    /// Append a cell and return its new id; the id is never reused later.
    fn push_cell(&mut self, cell: Cell) -> usize {
        self.cells.push(Some(cell));
        let id = self.cells.len() - 1;
        self.index_cell(id);
        id
    }

    /// Mark cell `id` as removed and unindex it. The slot becomes `None`; ids of
    /// the other cells do not shift.
    fn remove_cell(&mut self, id: usize) {
        self.unindex_cell(id);
        self.cells[id] = None;
    }

    /// True when the undirected edge `u`-`v` is present.
    fn has_edge(&self, u: usize, v: usize) -> bool {
        self.edges.contains_key(&(u.min(v), u.max(v)))
    }

    /// Side nodes from **existing edges** at `p`, chosen by angle to the ideal
    /// side vector of Owen (1999) §3.3.1.
    ///
    /// The ideal vector is the bisector of the current front edge and the other
    /// front edge at `p`; on a straight boundary the two are opposite and the
    /// bisector degenerates, so the inward perpendicular is used. An existing
    /// edge within 30° of the ideal is a side edge; otherwise the caller falls
    /// back to the reachable-node search.
    fn edge_side_candidates(&self, p: usize, other: usize, target: f64) -> Vec<usize> {
        let po = self.points[p];
        let u = [self.points[other][0] - po[0], self.points[other][1] - po[1]];
        let un = (u[0] * u[0] + u[1] * u[1]).sqrt();
        if un <= 1e-12 {
            return Vec::new();
        }
        let u = [u[0] / un, u[1] / un];
        let mut ideals = Vec::new();
        if let Some(edges) = self.node_edges.get(&p) {
            for &edge in edges {
                if !self.front.contains(&edge) {
                    continue;
                }
                let q = if edge.0 == p { edge.1 } else { edge.0 };
                if q == other {
                    continue;
                }
                let v = [self.points[q][0] - po[0], self.points[q][1] - po[1]];
                let vn = (v[0] * v[0] + v[1] * v[1]).sqrt();
                if vn <= 1e-12 {
                    continue;
                }
                let v = [v[0] / vn, v[1] / vn];
                let b = [u[0] + v[0], u[1] + v[1]];
                let bn = (b[0] * b[0] + b[1] * b[1]).sqrt();
                if bn > 0.2 {
                    ideals.push([b[0] / bn, b[1] / bn]);
                }
            }
        }
        if ideals.is_empty() {
            ideals.push([-u[1], u[0]]);
        }
        let Some(edges) = self.node_edges.get(&p) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for &edge in edges {
            let q = if edge.0 == p { edge.1 } else { edge.0 };
            if q == other || q == p {
                continue;
            }
            // The side edge has to touch the unconverted region; an edge
            // between two quads is already behind the front.
            let touches_live = self.edge_cells(edge).iter().any(|&id| {
                self.cells
                    .get(id)
                    .and_then(|cell| cell.as_ref())
                    .is_some_and(Cell::is_tri)
            });
            if !touches_live {
                continue;
            }
            let qp = self.points[q];
            let d = [qp[0] - po[0], qp[1] - po[1]];
            let dn = (d[0] * d[0] + d[1] * d[1]).sqrt();
            if dn < 0.4 * target || dn > 2.5 * target {
                continue;
            }
            let d = [d[0] / dn, d[1] / dn];
            if ideals
                .iter()
                .any(|ideal| d[0] * ideal[0] + d[1] * ideal[1] >= 0.866)
            {
                out.push(q);
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// The front edges grouped into boundary loops.
    ///
    /// Front edges normally form closed cycles of degree two. Where two fronts
    /// meet a vertex can have degree four; the walk then takes any unused edge,
    /// and a path that cannot close is returned as an open path rather than
    /// dropped — its edges still have to be tried.
    ///
    /// `scratch` keeps the per-vertex adjacency between calls. Building a fresh
    /// `HashMap` for every placed quad was the hottest part of the converter
    /// (1.16 s of a 2.15 s run on the performance part).
    fn front_loops(&self, scratch: &mut LoopScratch) -> Vec<Vec<(usize, usize)>> {
        let mut edges: Vec<(usize, usize)> = self.front.iter().copied().collect();
        // Deterministic: HashSet iteration order would otherwise make the whole
        // conversion depend on the process hash seed.
        edges.sort_unstable();
        if scratch.at.len() < self.points.len() {
            scratch.at.resize_with(self.points.len(), Vec::new);
        }
        let mut touched = Vec::new();
        for (id, &(a, b)) in edges.iter().enumerate() {
            if scratch.at[a].is_empty() {
                touched.push(a);
            }
            scratch.at[a].push(id);
            if scratch.at[b].is_empty() {
                touched.push(b);
            }
            scratch.at[b].push(id);
        }
        scratch.used.clear();
        scratch.used.resize(edges.len(), false);
        let mut loops = Vec::new();
        for start in 0..edges.len() {
            if scratch.used[start] {
                continue;
            }
            scratch.used[start] = true;
            let mut path = vec![edges[start]];
            let start_vertex = edges[start].0;
            let mut here = edges[start].1;
            loop {
                let next = scratch.at[here]
                    .iter()
                    .copied()
                    .find(|&id| !scratch.used[id]);
                let Some(next) = next else {
                    break;
                };
                scratch.used[next] = true;
                path.push(edges[next]);
                let (a, b) = edges[next];
                here = if a == here { b } else { a };
                if here == start_vertex || path.len() >= edges.len() {
                    break;
                }
            }
            loops.push(path);
        }
        for v in touched {
            scratch.at[v].clear();
        }
        loops
    }

    /// Live triangles reachable from `seed` whose nodes are within `radius` of
    /// `origin`.
    ///
    /// A triangle is included when at least one of its nodes is within the
    /// radius; the search does not cross a quad, so the region is a connected
    /// patch of the unconverted triangulation. Used by
    /// [`CellMesh::side_candidates`] and [`CellMesh::insert_near`].
    fn live_region(&self, seed: usize, origin: [f64; 2], radius: f64) -> Vec<usize> {
        let mut region = Vec::new();
        let mut seen = HashSet::new();
        let mut stack = vec![seed];
        while let Some(id) = stack.pop() {
            if !seen.insert(id) {
                continue;
            }
            let Some(Some(Cell::Tri(t))) = self.cells.get(id) else {
                continue;
            };
            let far = t.iter().all(|&v| {
                let p = self.points[v];
                ((p[0] - origin[0]).powi(2) + (p[1] - origin[1]).powi(2)).sqrt() > radius
            });
            if far {
                continue;
            }
            region.push(id);
            let tri = Cell::Tri(*t);
            for i in 0..3 {
                let edge = tri.edge(i);
                for &other in self.edge_cells(edge) {
                    if other == id {
                        continue;
                    }
                    if self
                        .cells
                        .get(other)
                        .and_then(|cell| cell.as_ref())
                        .is_some_and(Cell::is_tri)
                    {
                        stack.push(other);
                    }
                }
            }
        }
        region
    }

    /// True when the open segment `p..q` stays inside the live triangles of
    /// `region`.
    ///
    /// Every edge of the region the segment crosses must be shared by two live
    /// triangles; crossing a boundary edge or a quad edge means the segment has
    /// left the unconverted region. Edges incident to `p` or `q` are ignored
    /// (they are the segment's own endpoints, not a crossing). A segment that
    /// passes exactly through a vertex is not classified as crossing that
    /// vertex's edges; the caller's candidate set is small enough that this
    /// degeneracy is handled by the later region validation.
    fn segment_inside(&self, p: usize, q: usize, region: &[usize]) -> bool {
        let mut edges = HashSet::new();
        for &id in region {
            if let Some(Some(cell)) = self.cells.get(id) {
                for i in 0..cell.n_nodes() {
                    edges.insert(cell.edge(i));
                }
            }
        }
        let (pp, qq) = (self.points[p], self.points[q]);
        for &(u, v) in &edges {
            if u == p || v == p || u == q || v == q {
                continue;
            }
            if !segments_cross(pp, qq, self.points[u], self.points[v]) {
                continue;
            }
            let owners = self.edge_cells((u.min(v), u.max(v)));
            let both_live_tri = owners.len() == 2
                && owners.iter().all(|&c| {
                    self.cells
                        .get(c)
                        .and_then(|cell| cell.as_ref())
                        .is_some_and(Cell::is_tri)
                });
            if !both_live_tri {
                return false;
            }
        }
        true
    }

    /// Nodes that could be a side node for `p`: inside the reachable region, at
    /// 0.5–1.5 × `target` from `p`, and ahead of the front along the inward
    /// normal `n`.
    ///
    /// The measured corrections to the design's literal range are at
    /// [`SIDE_MIN_FACTOR`], [`SIDE_MAX_FACTOR`] and [`SIDE_MAX_LATERAL`].
    fn side_candidates(
        &self,
        p: usize,
        other: usize,
        seed: usize,
        n: [f64; 2],
        target: f64,
    ) -> Vec<usize> {
        let origin = self.points[p];
        let region = self.live_region(
            seed,
            [origin[0], origin[1]],
            (SIDE_MAX_FACTOR + 0.8) * target,
        );
        let mut nodes = HashSet::new();
        for &id in &region {
            if let Some(Some(Cell::Tri(t))) = self.cells.get(id) {
                for &v in t {
                    nodes.insert(v);
                }
            }
        }
        let mut out = Vec::new();
        for &v in &nodes {
            if v == p || v == other {
                continue;
            }
            let q = self.points[v];
            let d = [q[0] - origin[0], q[1] - origin[1]];
            let dist = (d[0] * d[0] + d[1] * d[1]).sqrt();
            if dist < SIDE_MIN_FACTOR * target || dist > SIDE_MAX_FACTOR * target {
                continue;
            }
            let along = d[0] * n[0] + d[1] * n[1];
            if along < 0.25 * target {
                continue;
            }
            let across = (d[0] * n[1] - d[1] * n[0]).abs();
            if across > SIDE_MAX_LATERAL * along {
                continue;
            }
            if !self.segment_inside(p, v, &region) {
                continue;
            }
            out.push(v);
        }
        out.sort_unstable();
        out
    }

    /// Insert a node at `p`, or as close to it as the reachable region allows.
    ///
    /// Searches the live region around `seed` for a triangle that strictly
    /// contains `p`; if none does, the final scan covers every triangle in the
    /// mesh (a fallback for a point that fell outside the region, e.g. after a
    /// recovery flip). Returns the new node id, or `None` when `p` is too close
    /// to a node of the triangle that would be split. Insertion is only called
    /// for a size mismatch; it is never a rescue for a failed placement.
    fn insert_near(&mut self, seed: usize, p: [f64; 2]) -> Option<usize> {
        // Do not split a triangle with a point this close to an existing node:
        // the split would create an edge of a few percent of the local size, and
        // a quad built on that edge has a corner near zero (measured 0.004 x
        // size on the perf part, and the resulting 0.02 quad then fails the gate
        // for good). Insertion is a response to a size mismatch, not a licence to
        // make a degenerate edge.
        let floor = MIN_INSERT_SPACING * self.bucket;
        // Neither the seed front triangle nor the triangle that actually gets
        // split may have a node this close to the new point.
        if !self.point_clear(p, &[seed], floor) {
            return None;
        }
        let region = self.live_region(seed, p, 3.0 * self.bucket);
        for &id in &region {
            let Some(Some(Cell::Tri(t))) = self.cells.get(id).copied() else {
                continue;
            };
            if point_strictly_inside(&self.points, t, p) {
                // Check the triangle that will actually be split, not the seed:
                // its three nodes are the ones the new point gets edges to.
                if !self.point_clear(p, &[id], floor) {
                    return None;
                }
                return Some(self.split_triangle(id, t, p));
            }
        }
        for id in 0..self.cells.len() {
            let Some(Some(Cell::Tri(t))) = self.cells.get(id).copied() else {
                continue;
            };
            if point_strictly_inside(&self.points, t, p) {
                if !self.point_clear(p, &[id], floor) {
                    return None;
                }
                return Some(self.split_triangle(id, t, p));
            }
        }
        None
    }

    /// True when `p` is at least `floor` from every node of `cells`.
    /// True when `p` is at least `floor` from every node of `cells`.
    fn point_clear(&self, p: [f64; 2], cells: &[usize], floor: f64) -> bool {
        for &id in cells {
            let Some(Some(cell)) = self.cells.get(id) else {
                continue;
            };
            for i in 0..cell.n_nodes() {
                let q = self.points[cell.node(i)];
                if ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt() < floor {
                    return false;
                }
            }
        }
        true
    }

    /// Split triangle `t` at `p` into three triangles around the new node.
    ///
    /// The first third reuses cell `id`; the other two get fresh ids. Returns
    /// the new node id. The three new triangles keep `t`'s winding, because each
    /// is `[t[k], t[k+1], v]` in the original cyclic order.
    fn split_triangle(&mut self, id: usize, t: [usize; 3], p: [f64; 2]) -> usize {
        self.points.push([p[0], p[1], 0.0]);
        let v = self.points.len() - 1;
        self.set_cell(id, Cell::Tri([t[0], t[1], v]));
        self.push_cell(Cell::Tri([t[1], t[2], v]));
        self.push_cell(Cell::Tri([t[2], t[0], v]));
        v
    }

    /// The triangle cell that owns the front edge `(u, v)`.
    ///
    /// A front edge has exactly one triangle by definition; this returns the
    /// first triangle owner so a caller can recover the orientation.
    fn tri_owner(&self, u: usize, v: usize) -> Option<usize> {
        self.edge_cells((u.min(v), u.max(v)))
            .iter()
            .copied()
            .find(|&c| {
                self.cells
                    .get(c)
                    .and_then(|cell| cell.as_ref())
                    .is_some_and(Cell::is_tri)
            })
    }

    /// The triangle cells enclosed by the quad's four sides, if that region is
    /// exactly the quad.
    ///
    /// Grows from `seed` without crossing any of the quad's four sides, then
    /// checks (a) no cell outside the quad is included, (b) every node of the
    /// region is inside the quad, and (c) the summed triangle area equals the
    /// quad area. Returns `None` if any of those fail, which means replacing the
    /// region by the quad would overlap a neighbour or leave a gap.
    fn region(&self, seed: usize, quad: &[usize; 4]) -> Option<Vec<usize>> {
        let mut sides = HashSet::new();
        for i in 0..4 {
            let (a, b) = (quad[i], quad[(i + 1) % 4]);
            sides.insert((a.min(b), a.max(b)));
        }
        let mut region = vec![seed];
        let mut seen = HashSet::new();
        seen.insert(seed);
        let mut head = 0;
        while head < region.len() {
            let id = region[head];
            head += 1;
            let cell = self.cells[id]?;
            for i in 0..cell.n_nodes() {
                let edge = cell.edge(i);
                if sides.contains(&edge) {
                    continue;
                }
                for &other in self.edge_cells(edge) {
                    if seen.insert(other) {
                        if !self.cells[other].is_some_and(|c| c.is_tri()) {
                            return None;
                        }
                        region.push(other);
                    }
                }
            }
        }
        for &id in &region {
            let cell = self.cells[id]?;
            for i in 0..cell.n_nodes() {
                if !inside_quad(&self.points, quad, cell.node(i)) {
                    return None;
                }
            }
        }
        // The region has the quad's boundary; the areas must agree too, or the
        // replacement would overlap a neighbour or leave a gap.
        let mut area = 0.0;
        for &id in &region {
            if let Some(Some(Cell::Tri(t))) = self.cells.get(id) {
                let (a, b, c) = (self.points[t[0]], self.points[t[1]], self.points[t[2]]);
                area += cross(a, b, c).abs() / 2.0;
            }
        }
        let expected = quad_area(&self.points, quad).abs();
        if (area - expected).abs() > 1e-9 * expected.max(1.0) {
            return None;
        }
        Some(region)
    }

    /// Flip `(u, v)` when both adjacent cells are triangles forming a convex quad.
    ///
    /// The new diagonal is the other one of the quadrilateral; the two new
    /// triangles are wound from `t1`'s traversal of the shared edge (see the
    /// long comment in the body). Returns `false` without changing anything when
    /// the quad is not strictly convex or the edge does not have exactly two
    /// triangle owners.
    fn flip(&mut self, u: usize, v: usize) -> bool {
        let owners = self.edge_cells((u.min(v), u.max(v)));
        if owners.len() != 2 {
            return false;
        }
        let (owners0, owners1) = (owners[0], owners[1]);
        let (Some(Cell::Tri(t1)), Some(Cell::Tri(t2))) = (self.cells[owners0], self.cells[owners1])
        else {
            return false;
        };
        let p = *t1
            .iter()
            .find(|&&n| n != u && n != v)
            .unwrap_or(&usize::MAX);
        let q = *t2
            .iter()
            .find(|&&n| n != u && n != v)
            .unwrap_or(&usize::MAX);
        if p == usize::MAX || q == usize::MAX || p == q {
            return false;
        }
        if side(self.points[u], self.points[v], self.points[p])
            * side(self.points[u], self.points[v], self.points[q])
            >= 0.0
            || side(self.points[p], self.points[q], self.points[u])
                * side(self.points[p], self.points[q], self.points[v])
                >= 0.0
        {
            return false;
        }
        // The other diagonal splits the quadrilateral into two triangles, and
        // the new winding must follow the winding of `t1`'s traversal of the
        // shared edge: the boundary is `p,u,q,v` when `t1` traverses u→v, and
        // `u,p,v,q` when it traverses v→u. Writing one order unconditionally —
        // which this code did — leaves both new triangles **inverted** on half
        // the flips. That was invisible to the quality report (a `Tri3` has a
        // scaled Jacobian of 1.0 by construction) and every later pairing, merge
        // or cavity then refused those triangles.
        let mut t1_forward = false;
        for i in 0..3 {
            if t1[i] == u && t1[(i + 1) % 3] == v {
                t1_forward = true;
            }
        }
        if t1_forward {
            self.set_cell(owners0, Cell::Tri([p, u, q]));
            self.set_cell(owners1, Cell::Tri([q, v, p]));
        } else {
            self.set_cell(owners0, Cell::Tri([p, q, u]));
            self.set_cell(owners1, Cell::Tri([q, p, v]));
        }
        true
    }

    /// Recover edge `(u, v)` by flipping crossing edges between live triangles.
    ///
    /// A partial recovery is a valid mesh; the caller does not roll it back.
    /// The loop is bounded by the cell count (each iteration either succeeds or
    /// flips an edge, and a flip cannot increase the cell count). Returns whether
    /// the edge exists at the end.
    fn recover(&mut self, u: usize, v: usize) -> bool {
        for _ in 0..self.cells.len() {
            if self.has_edge(u, v) {
                return true;
            }
            let Some(edge) = self.first_crossing(u, v) else {
                return self.has_edge(u, v);
            };
            if !self.flip(edge.0, edge.1) {
                return false;
            }
        }
        self.has_edge(u, v)
    }

    /// A live edge the open segment `u..v` crosses, chosen from the edge buckets
    /// over the segment's bounding box rather than by scanning every edge.
    ///
    /// Candidates are sorted before testing, so the same mesh and segment always
    /// flip the same edge; the bucket index can return an edge several times, so
    /// it is deduplicated. Edges incident to `u` or `v` are skipped (the segment
    /// touches them at its endpoints, it does not cross them).
    fn first_crossing(&self, u: usize, v: usize) -> Option<(usize, usize)> {
        let (p, q) = (self.points[u], self.points[v]);
        let (i0, j0) = self.bucket_key([p[0].min(q[0]), p[1].min(q[1]), 0.0]);
        let (i1, j1) = self.bucket_key([p[0].max(q[0]), p[1].max(q[1]), 0.0]);
        let mut candidates = Vec::new();
        for i in i0..=i1 {
            for j in j0..=j1 {
                if let Some(list) = self.edge_buckets.get(&(i, j)) {
                    candidates.extend_from_slice(list);
                }
            }
        }
        candidates.sort_unstable();
        candidates.dedup();
        for (a, b) in candidates {
            if a == u || a == v || b == u || b == v {
                continue;
            }
            if segments_cross(p, q, self.points[a], self.points[b]) {
                return Some((a, b));
            }
        }
        None
    }
}

/// True when `p` is strictly inside `tri` (no edge or vertex counts).
///
/// All three cross products must have the same sign with a `1e-12` tolerance;
/// this accepts triangles wound either way, which is what the front needs before
/// [`CellMesh::split_triangle`] has normalised anything.
fn point_strictly_inside(points: &[Node], tri: [usize; 3], p: [f64; 2]) -> bool {
    let q = [p[0], p[1], 0.0];
    let d0 = cross(points[tri[0]], points[tri[1]], q);
    let d1 = cross(points[tri[1]], points[tri[2]], q);
    let d2 = cross(points[tri[2]], points[tri[0]], q);
    (d0 > 1e-12 && d1 > 1e-12 && d2 > 1e-12) || (d0 < -1e-12 && d1 < -1e-12 && d2 < -1e-12)
}

/// True when node `v` is inside the (convex) quad, boundary included.
///
/// A point is inside when it is not strictly on one side of every edge; for a
/// convex CCW quad that is exactly "no edge has it on the right". Concave quads
/// are not expected at this call site (the quad was built from a candidate
/// pair), but the test would be conservative for them.
fn inside_quad(points: &[Node], quad: &[usize; 4], v: usize) -> bool {
    let p = points[v];
    let mut positive = 0;
    let mut negative = 0;
    for i in 0..4 {
        let s = cross(points[quad[i]], points[quad[(i + 1) % 4]], p);
        if s > 1e-12 {
            positive += 1;
        } else if s < -1e-12 {
            negative += 1;
        }
    }
    positive == 0 || negative == 0
}

/// Absolute area of a quad (the shoelace sum halved).
fn quad_area(points: &[Node], quad: &[usize; 4]) -> f64 {
    let mut twice = 0.0;
    for i in 0..4 {
        let a = points[quad[i]];
        let b = points[quad[(i + 1) % 4]];
        twice += a[0] * b[1] - b[0] * a[1];
    }
    twice / 2.0
}

/// Twice the signed area of triangle `a-b-c`: positive when counter-clockwise.
fn cross(a: Node, b: Node, c: Node) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// Side of the directed line `a -> b` on which `c` lies (the same cross product
/// as [`cross`], named for the predicate it implements).
fn side(a: Node, b: Node, c: Node) -> f64 {
    cross(a, b, c)
}

/// True when the open segments `p`-`q` and `r`-`s` cross at a single interior
/// point. Collinear or touching segments return `false`; the front only needs
/// strict crossings, and treating a touch as a crossing would make recovery
/// flip forever.
fn segments_cross(p: Node, q: Node, r: Node, s: Node) -> bool {
    let d1 = side(p, q, r);
    let d2 = side(p, q, s);
    let d3 = side(r, s, p);
    let d4 = side(r, s, q);
    d1 * d2 < 0.0 && d3 * d4 < 0.0
}

/// Euclidean length of an edge, in millimetres.
fn length(points: &[Node], edge: (usize, usize)) -> f64 {
    let (a, b) = (points[edge.0], points[edge.1]);
    ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt()
}

/// A side node is taken from the reachable region at 0.5–1.5 × the local size.
/// The upper bound was measured: Ruppert only bounds the *circumradius*, so the
/// triangulation contains edges up to ~2 × the field size (a 4 mm plate had a
/// front edge of 3.9 mm whose nearest inward node was 7.53 mm away = 1.88 × target).
/// With 1.5 the front emptied its candidate set on that part and stopped at 40 %
/// coverage; the bound below is the measured node spacing, not a tuning knob.
const SIDE_MIN_FACTOR: f64 = 0.5;
const SIDE_MAX_FACTOR: f64 = 2.2;
/// Lateral tolerance, as a multiple of the distance along the inward normal.
/// A tight cone (tan 31°) starved the front on non-isotropic triangles: the six
/// parts fell to 52–74 % coverage with 66–1258 triangles left, against 74–85 %
/// and 66–1012 with the joint shape selection doing the rejecting. The cone is
/// a prefilter, not the policy.
const SIDE_MAX_LATERAL: f64 = 2.5;

/// A quad is accepted only if it is convex, has a real Jacobian and is not a
/// sliver. The gate wants 0.2 / 40° on the final mesh; placing a 0.1° quad now
/// cannot be recovered later, so the placement floor is stricter than the gate.
const MIN_QUAD_SJ: f64 = 0.3;
const MIN_QUAD_ANGLE_DEG: f64 = 25.0;

/// Number of best-scoring side pairs examined before the edge is declared
/// failed. This is candidate selection among existing nodes, not insertion.
const MAX_PAIR_TRIALS: usize = 8;

/// A node is never inserted closer than this fraction of the local size to an
/// existing node: such a split creates an edge the quads cannot absorb.
const MIN_INSERT_SPACING: f64 = 0.25;

/// Why one front edge could not become a quad.
///
/// This is an internal classification: [`try_quad`] returns it, [`front_loop`]
/// turns it into a [`Refusals`] counter and blocks the edge. It is not an error
/// — the edge may become feasible after a neighbouring quad is placed.
#[derive(Clone, Copy, Debug)]
enum Refusal {
    /// No seed triangle, no usable side node, or the front edge is degenerate.
    Placement,

    /// Side pairs were found but none met the shape floors
    /// (`MIN_QUAD_SJ`, `MIN_QUAD_ANGLE_DEG`).
    Shape,

    /// One of the three missing sides could not be recovered by flips.
    Recovery,

    /// The triangles enclosed by the candidate quad did not exactly fill it
    /// (wrong boundary or wrong area).
    Region,

    /// The quad would have left an unconverted region of odd triangle count.
    OddRegion,
}

/// Counters for the stage at which a front edge failed.
///
/// One run produces one `Refusals`; the counters are independent (an edge can
/// contribute to `attempts` several times and to one refusal stage per attempt),
/// so they are diagnostic, not a partition. They are copied into
/// [`FrontResult::refusals`] and, after the Sun closure, into
/// [`convert_with_stats`].
#[derive(Debug, Default, Clone, Copy)]
pub struct Refusals {
    /// Number of times the front had no loop left to process.
    pub front_empty: usize,

    /// Edges refused at `Refusal::Placement`.
    pub placement: usize,

    /// Edges refused at `Refusal::Shape` (all candidate pairs too poor).
    pub invalid_quad: usize,

    /// Edges whose three missing sides could not be recovered.
    pub recovery: usize,

    /// Candidate quads whose enclosed region was not exactly the quad.
    pub region: usize,

    /// Placements refused to keep every unconverted region even.
    pub odd_region: usize,

    /// Nodes inserted because the front edge disagreed with the size field.
    pub mismatch_insertions: usize,

    /// Front edges examined (including edges that later succeed).
    pub attempts: usize,

    /// The Sun residual-triangle walk that closes the leftovers.
    pub closure: crate::sun::CloseStats,
}

/// Convert a triangle mesh into quads.
///
/// Returns the quads, the triangles that could not be converted, and the node
/// set the conversion ended with (placement inserts nodes).
///
/// This is the full Q-Morph plus Sun route ([`convert_with_stats`]); the
/// two-stage experimental route uses [`run_front`] directly so it can keep the
/// front's state. The input triangles are not modified; the output cells tile
/// the same domain, with no input triangle dropped silently.
pub fn convert(
    tris: &[[usize; 3]],
    points: &[Node],
    size: f64,
) -> (Vec<[usize; 4]>, Vec<[usize; 3]>, Vec<Node>) {
    convert_with_stats(tris, points, size).0
}

/// How the front decides it has finished.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrontStop {
    /// Convert until no front edge can advance. The leftovers are whatever the
    /// stalled loops left behind, often several disconnected islands.
    Exhaust,
    /// Refuse a placement that would leave an unconverted component with an
    /// odd number of triangles. The front then stops with every remaining
    /// component even, which is the condition the matching stage needs to pair
    /// it completely; the front may still split the region into several even
    /// components (it does so naturally as it advances).
    EvenComponents,
}

/// The front's state when it stops.
///
/// This is a snapshot, not a mesh with adjacency: the quads, the leftover
/// triangles and the (possibly extended) point set are the inputs a closure or
/// matching stage needs. The quads and the leftover triangles partition the
/// input: every input triangle is either inside a placed quad or in `triangles`.
#[derive(Debug, Default)]
pub struct FrontResult {
    /// Nodes at the end of the front; may contain nodes inserted for a size
    /// mismatch, so it is not necessarily the input slice.
    pub points: Vec<Node>,

    /// Quads placed by the front, in placement order, counter-clockwise.
    pub quads: Vec<[usize; 4]>,

    /// The unconverted triangles, in cell order.
    pub triangles: Vec<[usize; 3]>,

    /// Per-stage refusal counters for this run.
    pub refusals: Refusals,

    /// Connected-component sizes of `triangles`, largest first. One entry means
    /// the remainder is one triangulation.
    pub region_components: Vec<usize>,
}

/// Run the Q-Morph front and return its state, **without** the Sun closure.
///
/// [`convert_with_stats`] is this plus [`crate::sun::close_residuals`]; the
/// two-stage conversion is this plus a triangle matching stage.
///
/// `size` is the target edge length in millimetres. `stop` selects the finishing
/// rule: [`FrontStop::Exhaust`] drains the front, [`FrontStop::EvenComponents`]
/// refuses a placement that would leave an odd unconverted component (see
/// `CellMesh::removing_leaves_odd_component`). The returned
/// [`FrontResult::points`] may be longer than `points` by the inserted nodes.
pub fn run_front(tris: &[[usize; 3]], points: &[Node], size: f64, stop: FrontStop) -> FrontResult {
    let mut refusals = Refusals::default();
    let mut mesh = CellMesh::new(points, tris, size);
    let mut quads: Vec<[usize; 4]> = Vec::new();
    front_loop(&mut mesh, tris.len(), size, &mut quads, &mut refusals, stop);
    let region_components = mesh.live_tri_component_sizes();
    let triangles = mesh.triangles();
    FrontResult {
        points: mesh.points,
        quads,
        triangles,
        refusals,
        region_components,
    }
}

/// The outer Q-Morph state machine: choose a front loop, choose an edge, try it.
///
/// `tris_len` is the *input* triangle count and sets both runaway bounds. The
/// loop order and the per-loop edge order are deterministic (perimeter and
/// length, with the edge key as the tie-break), so a run is reproducible.
///
/// The loop terminates in one of three ways: the front becomes empty
/// (`front_empty` is incremented), no loop made progress (all remaining edges
/// are blocked), or a runaway bound is hit. Every path leaves a valid mesh: a
/// refused edge has changed nothing, and a successful placement removed only the
/// triangles it covered.
fn front_loop(
    mesh: &mut CellMesh,
    tris_len: usize,
    size: f64,
    quads: &mut Vec<[usize; 4]>,
    refusals: &mut Refusals,
    stop: FrontStop,
) {
    // Rule 4's memory: an edge that failed stays blocked until a placed quad
    // touches one of its nodes. The set only grows during a pass and is pruned
    // on success, so it cannot stall the front permanently.
    let mut blocked: HashSet<(usize, usize)> = HashSet::new();
    let max_attempts = 200 * tris_len + 10_000;
    let max_triangles = 4 * tris_len + 100;
    let mut scratch = LoopScratch::default();
    let mut fragment = FragmentCheck::default();

    'outer: loop {
        let mut loops: Vec<(f64, Vec<(usize, usize)>)> = mesh
            .front_loops(&mut scratch)
            .into_iter()
            .map(|edges| (loop_perimeter(&mesh.points, &edges), edges))
            .collect();
        if loops.is_empty() {
            refusals.front_empty += 1;
            break;
        }
        // Ascending perimeter: a hole's rows are laid before the outer loop's
        // coarse rows reach it. The perimeter is precomputed; recomputing it in
        // the comparator made this the hottest loop in the converter.
        loops.sort_by(|left, right| {
            left.0
                .partial_cmp(&right.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.1.cmp(&right.1))
        });
        let mut progressed = false;
        'loops: for (_, edges) in loops {
            let mut candidates: Vec<(usize, usize)> = edges
                .into_iter()
                .filter(|edge| mesh.front.contains(edge) && !blocked.contains(edge))
                .collect();
            if candidates.is_empty() {
                continue;
            }
            candidates.sort_by(|&e1, &e2| {
                length(&mesh.points, e1)
                    .partial_cmp(&length(&mesh.points, e2))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| e1.cmp(&e2))
            });
            for edge in candidates {
                if !mesh.front.contains(&edge) || blocked.contains(&edge) {
                    continue;
                }
                refusals.attempts += 1;
                if refusals.attempts > max_attempts || mesh.live_tris > max_triangles {
                    break 'outer;
                }
                match try_quad(mesh, edge, size, quads, refusals, stop, &mut fragment) {
                    Ok(()) => {
                        progressed = true;
                        // Rule 4: only edges touching the new quad can have
                        // become feasible again.
                        if let Some(quad) = quads.last() {
                            blocked.retain(|&(u, v)| !quad.contains(&u) && !quad.contains(&v));
                        }
                        // The front changed: restart from the smallest loop.
                        break 'loops;
                    }
                    Err(refusal) => {
                        match refusal {
                            Refusal::Placement => refusals.placement += 1,
                            Refusal::Shape => refusals.invalid_quad += 1,
                            Refusal::Recovery => refusals.recovery += 1,
                            Refusal::Region => refusals.region += 1,
                            Refusal::OddRegion => refusals.odd_region += 1,
                        }
                        blocked.insert(edge);
                    }
                }
            }
        }
        if !progressed {
            break;
        }
    }
}

/// Run the shipped Q-Morph route and return its cells plus the front counters.
///
/// This is [`run_front`] with [`FrontStop::Exhaust`] followed by the Sun
/// residual closure ([`crate::sun::close_residuals`]). The returned tuple is
/// `((quads, leftover_triangles, points), refusals)`. The closure may insert a
/// boundary midpoint, which is why the returned points are the closure's, not
/// the front's. `refusals.closure` describes the closure stage.
pub fn convert_with_stats(
    tris: &[[usize; 3]],
    points: &[Node],
    size: f64,
) -> ((Vec<[usize; 4]>, Vec<[usize; 3]>, Vec<Node>), Refusals) {
    let front = run_front(tris, points, size, FrontStop::Exhaust);
    let mut refusals = front.refusals;
    let (points, quads, leftover, closure) =
        crate::sun::close_residuals(&front.points, &front.quads, &front.triangles);
    refusals.closure = closure;
    ((quads, leftover, points), refusals)
}

/// Summed edge length of a front loop, in millimetres.
fn loop_perimeter(points: &[Node], edges: &[(usize, usize)]) -> f64 {
    edges.iter().map(|&e| length(points, e)).sum()
}

/// The design's size-mismatch test on the front edge itself.
///
/// A front edge outside `[0.5, 1.5] x target` is a candidate for having a node
/// inserted; an edge inside the band must be placed with existing nodes. This is
/// deliberately strict — insertion on every failure turned 604 triangles into
/// 5 022 on the corpus.
fn size_mismatch(span: f64, target: f64) -> bool {
    span < 0.5 * target || span > 1.5 * target
}

/// Try to build one quad on the front edge.
///
/// This is the per-edge step of the state machine, in order:
///
/// 1. find the seed triangle and orient the edge so the triangle is on the left;
/// 2. find side nodes at both endpoints (existing edges by angle, then the
///    reachable-region search) and insert only on a size mismatch;
/// 3. rank the side pairs by `min(scaled Jacobian, sin(min angle))` and keep the
///    best `MAX_PAIR_TRIALS`;
/// 4. for each pair, recover the three missing sides by flips, check that the
///    enclosed triangles are exactly the quad, and (in `EvenComponents` mode)
///    that removing them leaves every unconverted component even;
/// 5. commit: remove the enclosed triangles, add the quad, push it to `quads`.
///
/// A failure returns the [`Refusal`] of the last stage reached. Recovery flips
/// are deliberately **not** rolled back: they are valid mesh operations and the
/// next pair (or the next edge) benefits from them. Nothing is committed until
/// step 5, so a failed attempt never loses a cell.
fn try_quad(
    mesh: &mut CellMesh,
    edge: (usize, usize),
    size: f64,
    quads: &mut Vec<[usize; 4]>,
    refusals: &mut Refusals,
    stop: FrontStop,
    fragment: &mut FragmentCheck,
) -> Result<(), Refusal> {
    let seed = mesh.tri_owner(edge.0, edge.1).ok_or(Refusal::Placement)?;
    let Some(Cell::Tri(tri)) = mesh.cells[seed] else {
        return Err(Refusal::Placement);
    };
    // Orient the front edge so its triangle lies on the left; the edge key is
    // (min, max) and carries no direction.
    let third = *tri
        .iter()
        .find(|&&n| n != edge.0 && n != edge.1)
        .unwrap_or(&edge.0);
    let (a, b) = {
        let p = mesh.points[edge.0];
        let q = mesh.points[edge.1];
        let r = mesh.points[third];
        if cross(p, q, r) > 0.0 {
            (edge.0, edge.1)
        } else {
            (edge.1, edge.0)
        }
    };
    if a == b {
        return Err(Refusal::Placement);
    }
    let span = length(&mesh.points, (a, b));
    let target = if size > 1e-12 { size } else { span };
    if target <= 1e-12 {
        return Err(Refusal::Placement);
    }
    // Inward normal: from the edge midpoint towards the triangle's centroid.
    let pa = mesh.points[a];
    let pb = mesh.points[b];
    // Inward normal, perpendicular to the front edge and pointing at the
    // unconverted triangle. The old midpoint-to-centroid direction is skewed by
    // the shape of the first triangle, which let a node on the *opposing* front
    // (45° off the side direction) win the candidate search; the pair was then
    // non-convex and the corner stalled.
    let tangent = [pb[0] - pa[0], pb[1] - pa[1]];
    let tlen = (tangent[0] * tangent[0] + tangent[1] * tangent[1]).sqrt();
    if tlen <= 1e-12 {
        return Err(Refusal::Placement);
    }
    let n = [-tangent[1] / tlen, tangent[0] / tlen];

    // Rule 1: side nodes from the reachable unconverted region.
    let mut ca = mesh.edge_side_candidates(a, b, target);
    ca.extend(mesh.side_candidates(a, b, seed, n, target));
    ca.sort_unstable();
    ca.dedup();
    if ca.is_empty() {
        // Rule 2: insertion is a response to a size mismatch on the *front
        // edge*, never a rescue for a failed placement.
        if !size_mismatch(span, target) {
            return Err(Refusal::Placement);
        }
        let Some(c) = insert_along(mesh, seed, [pa[0], pa[1]], n, target) else {
            return Err(Refusal::Placement);
        };
        ca.push(c);
        refusals.mismatch_insertions += 1;
    }
    let mut cb = mesh.edge_side_candidates(b, a, target);
    cb.extend(mesh.side_candidates(b, a, seed, n, target));
    cb.sort_unstable();
    cb.dedup();
    if cb.is_empty() {
        if !size_mismatch(span, target) {
            return Err(Refusal::Placement);
        }
        let Some(d) = insert_along(mesh, seed, [pb[0], pb[1]], n, target) else {
            return Err(Refusal::Placement);
        };
        cb.push(d);
        refusals.mismatch_insertions += 1;
    }

    // Rule 1 (selection): the pair that maximises the resulting quad's shape,
    // jointly over both side nodes.
    let mut pairs: Vec<(f64, [usize; 4])> = Vec::new();
    for &c in &ca {
        if c == a || c == b {
            continue;
        }
        for &d in &cb {
            if d == a || d == b || d == c {
                continue;
            }
            let quad = [a, b, d, c];
            let Some((sj, angle)) = quad_metrics(&mesh.points, &quad) else {
                continue;
            };
            if !sj.is_finite() || !angle.is_finite() {
                continue;
            }
            if sj < MIN_QUAD_SJ || angle < MIN_QUAD_ANGLE_DEG {
                continue;
            }
            let score = sj.min(angle.to_radians().sin());
            pairs.push((score, quad));
        }
    }
    if pairs.is_empty() {
        return Err(Refusal::Shape);
    }
    pairs.sort_by(|left, right| {
        right
            .0
            .partial_cmp(&left.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| left.1.cmp(&right.1))
    });
    pairs.dedup_by(|left, right| left.1 == right.1);

    // Try the best pairs in order. The three recovered sides are the quad's
    // sides that are not the front edge: a-c, b-d and c-d (the front edge is
    // a-b). Recovery may flip triangles the earlier pairs relied on, so the
    // seed triangle is looked up again inside the loop; a pair that no longer
    // encloses a region is simply skipped.
    let mut last = Refusal::Recovery;
    for &(_, quad) in pairs.iter().take(MAX_PAIR_TRIALS) {
        let seed = mesh.tri_owner(edge.0, edge.1).ok_or(Refusal::Placement)?;
        let mut recovered = true;
        for (u, v) in [(quad[0], quad[3]), (quad[1], quad[2]), (quad[2], quad[3])] {
            if !mesh.has_edge(u, v) && !mesh.recover(u, v) {
                recovered = false;
                break;
            }
        }
        if !recovered {
            last = Refusal::Recovery;
            continue;
        }
        let Some(region) = mesh.region(seed, &quad) else {
            last = Refusal::Region;
            continue;
        };
        if region.is_empty() {
            last = Refusal::Region;
            continue;
        }
        // Two-stage: never leave an unconverted region of odd size.
        if stop == FrontStop::EvenComponents
            && mesh.removing_leaves_odd_component(&region, fragment)
        {
            last = Refusal::OddRegion;
            continue;
        }
        for &id in &region {
            mesh.remove_cell(id);
        }
        mesh.push_cell(Cell::Quad(quad));
        quads.push(quad);
        return Ok(());
    }
    Err(last)
}

/// Insert the missing side node at the ideal position, clamped towards `origin`
/// until it lands inside a live triangle.
///
/// The shares 1.0, 0.9, ..., 0.3 walk the point back along the inward normal in
/// 10 % steps. The ideal position is one target length inward; a shorter move is
/// accepted when the ideal one would fall outside the reachable region or too
/// close to an existing node. Returns `None` if no share works.
fn insert_along(
    mesh: &mut CellMesh,
    seed: usize,
    origin: [f64; 2],
    n: [f64; 2],
    target: f64,
) -> Option<usize> {
    for share in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3] {
        let p = [
            origin[0] + share * target * n[0],
            origin[1] + share * target * n[1],
        ];
        if let Some(v) = mesh.insert_near(seed, p) {
            return Some(v);
        }
    }
    None
}

/// Scaled Jacobian and smallest corner angle of a quad, or `None` when the quad
/// is not strictly convex.
///
/// Returns `(min scaled Jacobian, min corner angle in degrees)`. The convexity
/// test is on the signed area (must be positive) and on the four cross products
/// (must be positive), so a bowtie or a collinear corner returns `None` rather
/// than a misleading metric.
fn quad_metrics(points: &[Node], quad: &[usize; 4]) -> Option<(f64, f64)> {
    let coords = [
        [points[quad[0]][0], points[quad[0]][1]],
        [points[quad[1]][0], points[quad[1]][1]],
        [points[quad[2]][0], points[quad[2]][1]],
        [points[quad[3]][0], points[quad[3]][1]],
    ];
    // Strictly convex, counter-clockwise.
    let mut twice_area = 0.0;
    for i in 0..4 {
        let a = points[quad[i]];
        let b = points[quad[(i + 1) % 4]];
        twice_area += a[0] * b[1] - b[0] * a[1];
    }
    if twice_area <= 1e-12 {
        return None;
    }
    for i in 0..4 {
        if cross(
            points[quad[(i + 3) % 4]],
            points[quad[i]],
            points[quad[(i + 1) % 4]],
        ) <= 1e-12
        {
            return None;
        }
    }
    let sj = min_scaled_jacobian(&coords);
    let mut angle = f64::INFINITY;
    for i in 0..4 {
        let here = coords[i];
        let prev = coords[(i + 3) % 4];
        let next = coords[(i + 1) % 4];
        let u = [prev[0] - here[0], prev[1] - here[1]];
        let v = [next[0] - here[0], next[1] - here[1]];
        let denom = (u[0] * u[0] + u[1] * u[1]).sqrt() * (v[0] * v[0] + v[1] * v[1]).sqrt();
        if denom <= 0.0 {
            return None;
        }
        let degrees = ((u[0] * v[0] + u[1] * v[1]) / denom)
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        angle = angle.min(degrees);
    }
    Some((sj, angle))
}

/// The quality contract's scaled Jacobian, evaluated at the four corners.
///
/// This is the same bilinear map as `quality::scaled_jacobian`: at each
/// reference corner, the determinant of the Jacobian divided by the product of
/// its two column norms. The result is in `[-1, 1]` and is `NaN` for a quad with
/// a zero-length Jacobian column; callers guard with `is_finite`.
fn min_scaled_jacobian(p: &[[f64; 2]; 4]) -> f64 {
    const XI: [[f64; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
    let mut min = f64::INFINITY;
    for &xi in &XI {
        let mut j = [[0.0f64; 2]; 2];
        for (i, local) in XI.iter().enumerate() {
            let d0 = 0.25 * local[0] * (1.0 + local[1] * xi[1]);
            let d1 = 0.25 * local[1] * (1.0 + local[0] * xi[0]);
            j[0][0] += d0 * p[i][0];
            j[1][0] += d0 * p[i][1];
            j[0][1] += d1 * p[i][0];
            j[1][1] += d1 * p[i][1];
        }
        let c0 = (j[0][0] * j[0][0] + j[1][0] * j[1][0]).sqrt();
        let c1 = (j[0][1] * j[0][1] + j[1][1] * j[1][1]).sqrt();
        let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
        let sj = if c0 * c1 > 0.0 {
            det / (c0 * c1)
        } else {
            f64::NAN
        };
        min = min.min(sj);
    }
    min
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
    fn a_square_split_in_two_becomes_one_quad() {
        let points = square_points();
        let (quads, leftover, _) = convert(&[[0, 1, 2], [0, 2, 3]], &points, 1.0);
        assert_eq!(quads.len(), 1, "quads {quads:?} leftover {leftover:?}");
        assert!(leftover.is_empty());
        // The starting front edge decides the rotation, not the corner order.
        let q = quads[0];
        assert!(
            (0..4).any(|k| [q[k], q[(k + 1) % 4], q[(k + 2) % 4], q[(k + 3) % 4]] == [0, 1, 2, 3]),
            "quad {q:?}"
        );
    }

    #[test]
    fn a_three_by_three_grid_becomes_four_quads() {
        let mut points = Vec::new();
        for j in 0..3 {
            for i in 0..3 {
                points.push([i as f64, j as f64, 0.0]);
            }
        }
        let id = |i: usize, j: usize| j * 3 + i;
        let mut tris = Vec::new();
        for j in 0..2 {
            for i in 0..2 {
                let (a, b, c, d) = (id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1));
                tris.push([a, b, c]);
                tris.push([a, c, d]);
            }
        }
        let (quads, leftover, _) = convert(&tris, &points, 1.0);
        assert!(leftover.is_empty(), "leftover {leftover:?}");
        assert_eq!(quads.len(), 4, "quads {quads:?}");
    }

    #[test]
    fn the_mesh_tiles_after_a_conversion() {
        // Every edge has one or two cells; every quad is convex with a positive
        // Jacobian; the front is empty at the end.
        let points = square_points();
        let (quads, leftover, out_points) = convert(&[[0, 1, 2], [0, 2, 3]], &points, 1.0);
        let mut mesh = CellMesh::new(&out_points, &leftover, 1.0);
        for q in &quads {
            mesh.push_cell(Cell::Quad(*q));
        }
        let mut counts: HashMap<(usize, usize), usize> = HashMap::new();
        for id in mesh.cell_ids() {
            if let Some(cell) = mesh.cells[id] {
                for i in 0..cell.n_nodes() {
                    *counts.entry(cell.edge(i)).or_insert(0) += 1;
                }
            }
        }
        assert!(counts.values().all(|&n| n == 1 || n == 2), "{counts:?}");
        for q in &quads {
            let (sj, _) = quad_metrics(&out_points, q).unwrap();
            assert!(sj > 0.0);
        }
        assert!(mesh.front.is_empty(), "front {:?}", mesh.front);
    }

    /// A 3 × 3 grid of nodes (2 × 2 cells), each cell split into two triangles.
    fn grid_3x3() -> (Vec<Node>, Vec<[usize; 3]>) {
        let mut points = Vec::new();
        for j in 0..3 {
            for i in 0..3 {
                points.push([i as f64, j as f64, 0.0]);
            }
        }
        let id = |i: usize, j: usize| j * 3 + i;
        let mut tris = Vec::new();
        for j in 0..2 {
            for i in 0..2 {
                let (a, b, c, d) = (id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1));
                tris.push([a, b, c]);
                tris.push([a, c, d]);
            }
        }
        (points, tris)
    }

    #[test]
    fn the_even_region_guard_leaves_only_even_components() {
        let (points, tris) = grid_3x3();
        let front = run_front(&tris, &points, 1.0, FrontStop::EvenComponents);
        assert!(
            front.region_components.iter().all(|n| n % 2 == 0),
            "odd region left: {:?}",
            front.region_components
        );
        // The guard may refuse placements, but it must not lose the quads it
        // placed or the triangles it kept: the union tiles the input.
        let placed: usize = front.quads.len() * 4 + front.triangles.len() * 3;
        let source: usize = tris.len() * 3;
        assert!(placed <= source);
    }

    #[test]
    fn the_front_keeps_every_triangle_wound_counter_clockwise() {
        let (points, tris) = grid_3x3();
        let front = run_front(&tris, &points, 1.0, FrontStop::Exhaust);
        for t in &front.triangles {
            let (a, b, c) = (front.points[t[0]], front.points[t[1]], front.points[t[2]]);
            let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            assert!(area >= 0.0, "clockwise triangle {t:?} after the front");
        }
    }

    #[test]
    fn a_flip_keeps_both_triangles_counter_clockwise() {
        // Both windings of the shared edge: the new pair must be CCW in either.
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        for tris in [
            // t1 = (0, 1, 2) traverses the edge 2→0 (v→u).
            vec![[0, 1, 2], [0, 2, 3]],
            // t1 = (0, 2, 3) traverses it 0→2 (u→v).
            vec![[0, 2, 3], [2, 0, 1]],
        ] {
            let mut mesh = CellMesh::new(&points, &tris, 1.0);
            let before = 1.0;
            assert!(mesh.flip(0, 2), "flip must be legal: {tris:?}");
            let mut area = 0.0;
            for id in mesh.cell_ids() {
                match mesh.cells[id] {
                    Some(Cell::Tri(t)) => {
                        let (a, b, c) = (mesh.points[t[0]], mesh.points[t[1]], mesh.points[t[2]]);
                        let signed =
                            ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2.0;
                        assert!(
                            signed > 0.0,
                            "clockwise triangle {t:?} after flip: {tris:?}"
                        );
                        area += signed;
                    }
                    other => panic!("unexpected cell {other:?}"),
                }
            }
            assert!((area - before).abs() < 1e-12, "area changed: {tris:?}");
        }
    }

    #[test]
    fn insertion_never_makes_a_tiny_edge() {
        let points = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let mut mesh = CellMesh::new(&points, &[[0, 1, 2]], 1.0);
        assert!(
            mesh.insert_near(0, [0.01, 0.01]).is_none(),
            "a point 0.014 from node 0 must not be inserted"
        );
        assert!(
            mesh.insert_near(0, [0.3, 0.3]).is_some(),
            "a point in the middle of the triangle must be inserted"
        );
    }

    #[test]
    fn segments_cross_only_when_they_really_do() {
        let p = [0.0, 0.0, 0.0];
        let q = [1.0, 1.0, 0.0];
        assert!(segments_cross(p, q, [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]));
        assert!(!segments_cross(p, q, [2.0, 0.0, 0.0], [2.0, 1.0, 0.0]));
    }
}
