//! Complete triangle-to-quad conversion: Sun's residual-triangle walk.
//!
//! Q-Morph (Owen et al. 1999, corpus `owen-1999`) stops with an even number of
//! residual triangles per domain. Sun, Yeh, Lin & Zhao (2015, *Eng. Comput.*
//! 31:675-692; corpus `sun-2015`; 2016, *Terr. Atmos. Ocean. Sci.* 27:139-152;
//! corpus `sun-2016`) close them without subdividing a triangle and without
//! touching a quad that is not on the path:
//!
//! 1. **Move the residual, not the target.** For a residual triangle `abh` and
//!    the centre `Pm` of the nearest other residual triangle, form the three
//!    triangles `Pm-a-b`, `Pm-b-h`, `Pm-h-a` and take the edge whose triangle
//!    has the smallest scaled Jacobian (Sun 2015, Eqs. 2-3). That is the edge
//!    that faces `Pm`; it is a Jacobian choice, not a distance choice.
//! 2. **Destroy one quad, keep one triangle.** The moving edge has a triangle on
//!    the other side; the background quad containing that triangle is removed.
//!    There are two recombinations (Sun 2016, §2.4.2): merge the residual with
//!    the connected triangle, or exchange the quad's common edge and merge with
//!    the other half. The candidate whose new residual is closer to `Pm` is
//!    taken, subject to a positive-Jacobian guard on the new quad.
//! 3. **Pair.** Repeat until the moving residual shares an edge with the target
//!    and merge the two into a quad.
//! 4. **Last triangle.** When one residual remains, move it to a boundary edge,
//!    insert the edge midpoint (a B-spline point on a curved boundary; this
//!    mesher's boundaries are straight segments), and apply the transformation
//!    template of Sun 2016, Fig. 6: the triangle plus its two adjacent quads
//!    become three quads around the new boundary point.
//!
//! The template is conforming only on a geometric boundary. On an internal
//! boundary shared by two sub-domains the inserted point would break
//! conformity; this mesher's conversion runs on one domain, and
//! [`CloseStats::remaining`] carries a triangle out rather than breaking it.
//!
//! **Boundary topological optimisation.** The Fig. 6 template cannot finish a
//! boundary sliver when the adjacent Q-Morph quad already has a third node on
//! the same straight boundary line: inserting `P` then puts `P`, `b` and that
//! node on one line and the outer template quad is degenerate. Two Sun 2015 §5
//! operations are implemented for it:
//!
//! - **M2/M3 edge swap** (`WalkMesh::repair_quad`): after the template, swap
//!   the degenerate quad's shared edge with an adjacent quad; if the pair of
//!   new quads is positive, the degeneracy is gone.
//! - **M6-style absorption** (`WalkMesh::absorb_sliver`): when `e-b-c-f` are
//!   collinear on the boundary, remove the sliver plus its two quads and
//!   replace them by two quads, deleting one boundary node. The boundary
//!   geometry is unchanged because the four nodes are collinear.
//!
//! **Measured on the Q-Morph output (2026-10-22).** The walk plus those modes,
//! alternating with W2 smoothing (the smoother moves the nodes that made a
//! pentagon recombination concave), gives all-quad plate-hole 4 mm, 2 mm and
//! stepped; plate-hole 1 mm has 2 triangles, l-bracket 1, ring 6 and the 0.6 mm
//! performance part 8. The remaining ones are isolated interior slivers: every
//! pentagon split around them is concave and they cannot reach the boundary, so
//! M2/M3 has nothing to swap. A local cavity/edge-flip remesh is the next
//! operation. The 40° floor is not met either; only l-bracket reaches 42.6°
//! (with one triangle left). [`CloseStats::remaining`] carries the triangles
//! out; the gate is not claimed to pass.
//!
//! **Clean room.** Implemented from the papers above; no source code was read
//! or ported.
//!
//! **Public API.** [`close_residuals`] is the whole pipeline used by
//! [`crate::qmorph::convert_with_stats`]; [`close_by_cavity`] is the local
//! cavity fallback for residuals the walk could not move; and
//! [`repair_low_quads`], [`improve_quality`], [`repair_degenerate_quads`] and
//! [`perturb_low_quads`] are post-processing passes on a finished quad mesh.
//! All of them take `(points, quads, tris)` slices and return new vectors plus a
//! count; none mutates its input or drops a cell. The indices in `quads`/`tris`
//! must be valid for `points`; that is a precondition of the whole module, not
//! something the functions re-validate.
//!
//! **Approximated or missing.** The boundary template inserts the midpoint of a
//! straight boundary edge, not a point projected onto a curved boundary; the
//! cavity quadrangulation (`close_by_cavity`) is an extension of this mesher,
//! not a Sun template; and a residual that cannot reach a boundary and whose
//! cavity is not simple is carried out in `CloseStats::remaining` rather than
//! forced.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::mesh::Node;

/// A cell of the closing mesh.
///
/// The walk keeps the residual triangles and the quads in one mesh so that a
/// merge/recombination can delete both kinds and insert both kinds in one
/// adjacency update.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cell {
    /// An unconverted residual triangle.
    Tri([usize; 3]),

    /// A quad, either from Q-Morph, a merge or a template.
    Quad([usize; 4]),
}

impl Cell {
    fn n_nodes(self) -> usize {
        match self {
            Cell::Tri(_) => 3,
            Cell::Quad(_) => 4,
        }
    }

    fn node(self, i: usize) -> usize {
        match self {
            Cell::Tri(t) => t[i],
            Cell::Quad(q) => q[i],
        }
    }

    fn edge(self, i: usize) -> (usize, usize) {
        let (a, b) = (self.node(i), self.node((i + 1) % self.n_nodes()));
        (a.min(b), a.max(b))
    }

    fn is_tri(self) -> bool {
        matches!(self, Cell::Tri(_))
    }
}

/// What the residual walk did on one conversion.
///
/// The counters describe attempts, not a partition: a residual can be attempted
/// by the walk, then by the boundary template, then by the cavity, and still be
/// counted in `remaining`. `merged_pairs + remaining` is not the input count
/// either, because a merge removes two triangles and adds one quad (and the
/// cavity adds several).
#[derive(Clone, Copy, Debug, Default)]
pub struct CloseStats {
    /// Triangles in the Q-Morph output.
    pub start_triangles: usize,

    /// Pairs merged into a quad.
    pub merged_pairs: usize,

    /// Pentagon recombinations performed while moving.
    pub moves: usize,

    /// Residual triangle finished with the Sun 2016 boundary template.
    pub boundary_templates: usize,

    /// Sun 2015 §5.2.1 M2/M3 edge swaps that made a template quad valid.
    pub boundary_repairs: usize,

    /// Sun 2015 §5.2.1 M6-style absorptions: a boundary sliver plus its two
    /// quads become two quads, removing the collinear boundary node.
    pub boundary_absorptions: usize,

    /// Residuals closed by a local cavity quadrangulation.
    pub cavity_closures: usize,

    /// Cavity regions tried (a region can fail and be retried larger).
    pub cavity_attempts: usize,

    /// Cavity attempts rejected because the region boundary was not a single
    /// simple loop (a hole, several loops, or a vertex of degree not two).
    pub cavity_boundary_fail: usize,

    /// Cavity attempts rejected because the boundary had odd length and no
    /// domain-boundary edge was available to split.
    pub cavity_odd_fail: usize,

    /// Cavity attempts where the dynamic-programming quadrangulation found no
    /// valid tiling.
    pub cavity_dp_fail: usize,

    /// Cavity attempts rejected after quadrangulation by the area/Jacobian/
    /// manifold validation.
    pub cavity_valid_fail: usize,

    /// Boundary residuals whose template was tried.
    pub template_attempts: usize,

    /// Template attempts rejected before adding cells (not simple).
    pub template_raw_failures: usize,

    /// Templates whose degenerate quads M2/M3 could not repair.
    pub template_repair_failures: usize,

    /// Triangles that could not be moved or merged.
    pub stuck: usize,

    /// Candidate quads refused because of the Jacobian guard.
    pub quality_rejections: usize,

    /// Adjacent residual pairs whose union was not a convex quad.
    pub merge_failures: usize,

    /// Triangles left in the returned mesh.
    pub remaining: usize,
}

/// The new quad must stay usable: the gate's Jacobian floor is applied during
/// the walk, not discovered at the end.
const MOVE_MIN_SJ: f64 = 0.2;

/// A residual triangle collapsed to less than this area is rejected.
const MIN_TRI_AREA2: f64 = 1e-14;

/// A cavity quad must leave usable quality; otherwise the cavity is rejected.
const CAVITY_MIN_SJ: f64 = 0.0;

/// The mutable mesh on which the residual walk operates.
///
/// It is a stripped-down [`crate::qmorph::CellMesh`]: no front and no buckets,
/// because the walk only needs edge adjacency, the set of live triangles and the
/// current domain-boundary edges. Removed cells are `None` so ids stay stable
/// (the move/recombination logic compares ids). Every cell is stored
/// counter-clockwise and canonically rotated; all coordinates are in the XY
/// plane.
#[derive(Clone)]
struct WalkMesh {
    /// Nodes in the plane; the boundary template and the cavity fallback append
    /// to this, so ids in the input slices remain valid.
    points: Vec<Node>,

    /// Cells by id; `None` is a removed cell.
    cells: Vec<Option<Cell>>,

    /// Edge key -> owning cell ids; the source of truth for adjacency.
    edges: HashMap<(usize, usize), Vec<usize>>,

    /// Ids of the live triangles. Kept as a set (not a counter) because the walk
    /// iterates the triangles by id.
    tris: HashSet<usize>,

    /// Domain-boundary edges (exactly one owner), sorted. Rebuilt by
    /// `recompute_boundary` after an operation that can change it.
    boundary: Vec<(usize, usize)>,

    /// Pentagons recombined while moving a residual.
    moves: usize,

    /// Residual pairs merged into a quad.
    merged: usize,

    /// Candidate quads rejected by `move_min_sj`.
    quality_rejections: usize,

    /// Residual pairs whose union was not a valid quad.
    merge_failures: usize,

    /// M2/M3 edge swaps applied while repairing a template quad.
    repairs: usize,

    /// Fig. 6 boundary templates applied.
    templates: usize,

    /// Cavity quadrangulations applied.
    cavities: usize,

    /// Cavity regions attempted.
    cavity_attempts: usize,

    /// Cavity attempts rejected by `region_boundary`.
    cavity_boundary_fail: usize,

    /// Cavity attempts rejected because the loop was odd and no boundary edge
    /// could be split.
    cavity_odd_fail: usize,

    /// Cavity attempts rejected by the DP quadrangulation.
    cavity_dp_fail: usize,

    /// Cavity attempts rejected by the area/Jacobian/manifold validation.
    cavity_valid_fail: usize,

    /// Current Jacobian floor for a recombination step: `0.2`, then `0.1`,
    /// then `0.0` in the staged pass of [`close_residuals`].
    move_min_sj: f64,

    /// Boundary-template attempts.
    template_attempts: usize,

    /// Template attempts rejected before any cell was added.
    template_raw_failures: usize,

    /// Template attempts that could not be repaired to a positive mesh.
    template_repair_failures: usize,

    /// M6-style boundary absorptions applied.
    absorptions: usize,
}

impl WalkMesh {
    /// Build the closing mesh from a Q-Morph output.
    ///
    /// Quads are indexed first, then the residual triangles, both after
    /// normalisation (counter-clockwise and canonically rotated). The initial
    /// boundary is every edge with exactly one owner, sorted so the boundary
    /// scan is deterministic.
    fn new(points: &[Node], quads: &[[usize; 4]], tris: &[[usize; 3]]) -> Self {
        let mut mesh = WalkMesh {
            points: points.to_vec(),
            cells: Vec::with_capacity(quads.len() + tris.len()),
            edges: HashMap::new(),
            tris: HashSet::new(),
            boundary: Vec::new(),
            moves: 0,
            merged: 0,
            quality_rejections: 0,
            merge_failures: 0,
            repairs: 0,
            templates: 0,
            cavities: 0,
            cavity_attempts: 0,
            cavity_boundary_fail: 0,
            cavity_odd_fail: 0,
            cavity_dp_fail: 0,
            cavity_valid_fail: 0,
            move_min_sj: MOVE_MIN_SJ,
            template_attempts: 0,
            template_raw_failures: 0,
            template_repair_failures: 0,
            absorptions: 0,
        };
        // Every cell is stored counter-clockwise. The walk reasons about the
        // winding of the moving edge; the normalisation makes the module safe on
        // any input.
        for q in quads {
            let cell = Cell::Quad(normalize_quad(&mesh.points, *q));
            let id = mesh.push_raw(cell);
            mesh.index(id);
        }
        for t in tris {
            let cell = Cell::Tri(normalize_tri(&mesh.points, *t));
            let id = mesh.push_raw(cell);
            mesh.index(id);
        }
        mesh.boundary = mesh
            .edges
            .iter()
            .filter(|(_, owners)| owners.len() == 1)
            .map(|(&edge, _)| edge)
            .collect();
        mesh.boundary.sort_unstable();
        mesh
    }

    /// Append a cell without indexing; the caller must call [`WalkMesh::index`].
    fn push_raw(&mut self, cell: Cell) -> usize {
        self.cells.push(Some(cell));
        self.cells.len() - 1
    }

    /// Register a cell's edges and, for a triangle, its id in `tris`.
    fn index(&mut self, id: usize) {
        let Some(Some(cell)) = self.cells.get(id).copied() else {
            return;
        };
        for i in 0..cell.n_nodes() {
            self.edges.entry(cell.edge(i)).or_default().push(id);
        }
        if cell.is_tri() {
            self.tris.insert(id);
        }
    }

    /// Append and index a cell; returns its new id.
    fn add(&mut self, cell: Cell) -> usize {
        let id = self.push_raw(cell);
        self.index(id);
        id
    }

    /// Remove cell `id` from the mesh and return it, or `None` if already gone.
    fn remove(&mut self, id: usize) -> Option<Cell> {
        let cell = self.cells.get_mut(id)?.take()?;
        for i in 0..cell.n_nodes() {
            let edge = cell.edge(i);
            let mut empty = false;
            if let Some(owners) = self.edges.get_mut(&edge) {
                owners.retain(|&c| c != id);
                empty = owners.is_empty();
            }
            if empty {
                self.edges.remove(&edge);
            }
        }
        self.tris.remove(&id);
        Some(cell)
    }

    /// The live cell with id `id`, or `None` if removed/out of range.
    fn cell(&self, id: usize) -> Option<Cell> {
        self.cells.get(id).copied().flatten()
    }

    /// True when `id` is a live triangle.
    fn is_tri(&self, id: usize) -> bool {
        self.cell(id).is_some_and(Cell::is_tri)
    }

    /// Live triangle ids in ascending order (deterministic walk order).
    fn tri_ids(&self) -> Vec<usize> {
        let mut ids: Vec<usize> = self.tris.iter().copied().collect();
        ids.sort_unstable();
        ids
    }

    /// Number of live residual triangles.
    fn tri_count(&self) -> usize {
        self.tris.len()
    }

    /// Owners of `edge`, or an empty slice if absent.
    fn edge_cells(&self, edge: (usize, usize)) -> &[usize] {
        self.edges.get(&edge).map(Vec::as_slice).unwrap_or(&[])
    }

    /// True when `edge` has exactly one owner (a domain-boundary edge).
    fn is_boundary_edge(&self, edge: (usize, usize)) -> bool {
        self.edge_cells(edge).len() == 1
    }

    /// The triangle nodes of `id`, or `None` if it is not a live triangle.
    fn tri(&self, id: usize) -> Option<[usize; 3]> {
        match self.cell(id) {
            Some(Cell::Tri(t)) => Some(t),
            _ => None,
        }
    }

    /// The quad nodes of `id`, or `None` if it is not a live quad.
    fn quad(&self, id: usize) -> Option<[usize; 4]> {
        match self.cell(id) {
            Some(Cell::Quad(q)) => Some(q),
            _ => None,
        }
    }

    /// XY coordinates of node `i`.
    fn p(&self, i: usize) -> [f64; 2] {
        [self.points[i][0], self.points[i][1]]
    }

    /// Centroid of a live cell, or `None` if the id is gone.
    fn center(&self, id: usize) -> Option<[f64; 2]> {
        let cell = self.cell(id)?;
        self.center_of(cell)
    }

    /// Arithmetic mean of a cell's nodes; used as the distance reference for
    /// the nearest-residual search.
    fn center_of(&self, cell: Cell) -> Option<[f64; 2]> {
        let n = cell.n_nodes();
        let mut c = [0.0, 0.0];
        for i in 0..n {
            let p = self.points[cell.node(i)];
            c[0] += p[0];
            c[1] += p[1];
        }
        Some([c[0] / n as f64, c[1] / n as f64])
    }

    /// True when the two live triangles share an edge.
    ///
    /// Edge keys are sorted, so the test is orientation-independent. Used by
    /// [`WalkMesh::walk_to`] to stop as soon as a merge is possible.
    fn shares_edge(&self, t1: usize, t2: usize) -> bool {
        let (Some(a), Some(b)) = (self.tri(t1), self.tri(t2)) else {
            return false;
        };
        for i in 0..3 {
            let e = Cell::Tri(a).edge(i);
            for j in 0..3 {
                if Cell::Tri(b).edge(j) == e {
                    return true;
                }
            }
        }
        false
    }

    /// All other residual triangles, nearest first by centroid distance, with
    /// the triangle id as the tie-break so the order is deterministic.
    fn nearest_tris(&self, id: usize) -> Vec<usize> {
        let Some(c) = self.center(id) else {
            return Vec::new();
        };
        let mut all: Vec<(f64, usize)> = Vec::new();
        for &other in &self.tris {
            if other == id {
                continue;
            }
            let Some(o) = self.center(other) else {
                continue;
            };
            let d = (o[0] - c[0]).powi(2) + (o[1] - c[1]).powi(2);
            all.push((d, other));
        }
        all.sort_by(|left, right| {
            left.0
                .partial_cmp(&right.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.1.cmp(&right.1))
        });
        all.into_iter().map(|(_, id)| id).collect()
    }

    /// Merge two residual triangles sharing an edge into one quad.
    ///
    /// Tries every edge of `t1` and looks for the reverse traversal in `t2`, so
    /// the quad is built in each triangle's own winding (the `quad_from_pair`
    /// rule). The merge is accepted only if the quad is simple, has a positive
    /// Jacobian and passes [`WalkMesh::valid_new_cell`]; otherwise the mesh is
    /// untouched and `merge_failures` is incremented. Returns whether a merge
    /// happened.
    fn merge(&mut self, t1: usize, t2: usize) -> bool {
        let (Some(a), Some(b)) = (self.tri(t1), self.tri(t2)) else {
            return false;
        };
        let a = normalize_tri(&self.points, a);
        let b = normalize_tri(&self.points, b);
        for i in 0..3 {
            let x = a[i];
            let y = a[(i + 1) % 3];
            let third_a = a[(i + 2) % 3];
            // The neighbouring triangle has to traverse the same edge the other
            // way; otherwise the mesh is inconsistent.
            let Some(j) = (0..3).find(|&j| b[j] == y && b[(j + 1) % 3] == x) else {
                continue;
            };
            let third_b = b[(j + 2) % 3];
            let quad = normalize_quad(&self.points, [y, third_a, x, third_b]);
            let Some(sj) = quad_sj(&self.points, &quad) else {
                continue;
            };
            if sj <= 0.0 || !self.valid_new_cell(Cell::Quad(quad), &[t1, t2]) {
                continue;
            }
            self.remove(t1);
            self.remove(t2);
            self.add(Cell::Quad(quad));
            self.merged += 1;
            return true;
        }
        self.merge_failures += 1;
        false
    }

    /// One pentagon recombination towards `pm`.
    ///
    /// Sun 2015 Eqs. (2)-(3): rank the residual's three edges by the scaled
    /// Jacobian of the triangle `pm`-edge, smallest first (negative means `pm`
    /// lies across that edge), then take the first edge not in `skip` whose
    /// other side is a quad and whose recombination is valid. `skip` lets a
    /// caller reject one edge after a failed merge and try the next.
    fn move_toward(&mut self, id: usize, pm: [f64; 2], skip: Option<(usize, usize)>) -> Move {
        let Some(t) = self.tri(id) else {
            return Move::Stuck;
        };
        let t = normalize_tri(&self.points, t);
        // Sun 2015, Eqs. (2)-(3): the edge whose triangle with Pm has the
        // smallest scaled Jacobian. A negative value means Pm lies across that
        // edge, so it is the edge the residual has to cross.
        let mut order: Vec<(f64, usize, usize)> = Vec::with_capacity(3);
        for i in 0..3 {
            let (u, v) = (t[i], t[(i + 1) % 3]);
            order.push((point_edge_sj(pm, self.p(u), self.p(v)), u, v));
        }
        order.sort_by(|left, right| {
            left.0
                .partial_cmp(&right.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| (left.1, left.2).cmp(&(right.1, right.2)))
        });
        for (_, u, v) in order {
            let edge = (u.min(v), u.max(v));
            if skip == Some(edge) {
                continue;
            }
            let owners = self.edge_cells(edge);
            if owners.len() != 2 {
                continue;
            }
            let other = if owners[0] == id {
                owners[1]
            } else {
                owners[0]
            };
            if self.is_tri(other) {
                return Move::Adjacent { id: other, edge };
            }
            if self.quad(other).is_some() {
                if let Some((quad, residual)) = self.recombination(t, id, other, u, v, pm, None) {
                    self.remove(id);
                    self.remove(other);
                    self.add(Cell::Quad(quad));
                    let next = self.add(Cell::Tri(residual));
                    self.moves += 1;
                    return Move::Moved(next);
                }
            }
        }
        Move::Stuck
    }

    /// Build one of the two recombinations of `tri` with the quad across its
    /// edge `u -> v`; `tri` is counter-clockwise and contains `u -> v`.
    ///
    /// With `target` set, the residual has to contain that boundary edge, which
    /// is how the last triangle is walked onto the boundary.
    #[allow(clippy::too_many_arguments)]
    fn recombination(
        &self,
        tri: [usize; 3],
        tri_id: usize,
        quad_id: usize,
        u: usize,
        v: usize,
        pm: [f64; 2],
        target: Option<(usize, usize)>,
    ) -> Option<([usize; 4], [usize; 3])> {
        let a = *tri.iter().find(|&&n| n != u && n != v)?;
        let q = normalize_quad(&self.points, self.quad(quad_id)?);
        // The quad traverses the shared edge the other way: v -> u.
        let i = (0..4).find(|&i| q[i] == v && q[(i + 1) % 4] == u)?;
        let s = q[(i + 2) % 4];
        let t = q[(i + 3) % 4];
        let candidates = [
            // (a) merge with the connected triangle (v, u, s)
            ([a, u, s, v], [v, s, t]),
            // (b) exchange the quad's common edge and merge with (v, u, t)
            ([a, u, t, v], [u, s, t]),
        ];
        // Prefer a convex new quad; a shallow concave one is still allowed as a
        // step, because the residual immediately moves on and a later
        // recombination (or the smoother) removes it. Refusing every concave
        // step strands the residual instead, and Sun lets the quality step fix
        // the shape.
        let mut best: Option<((u8, f64, f64), ([usize; 4], [usize; 3]))> = None;
        for (quad, residual) in candidates {
            let quad = normalize_quad(&self.points, quad);
            let residual = normalize_tri(&self.points, residual);
            if tri_area2(&self.points, residual).abs() < MIN_TRI_AREA2 {
                continue;
            }
            if !quad_simple(&self.points, &quad) {
                continue;
            }
            let sj = quad_min_sj(&self.points, &quad);
            if sj <= 0.0 || sj < self.move_min_sj {
                continue;
            }
            if let Some(edge) = target {
                if !cell_has_edge(Cell::Tri(residual), edge) {
                    continue;
                }
            }
            let ignore = [tri_id, quad_id];
            if !self.valid_new_cell(Cell::Quad(quad), &ignore)
                || !self.valid_new_cell(Cell::Tri(residual), &ignore)
            {
                continue;
            }
            let Some(c) = self.center_of(Cell::Tri(residual)) else {
                continue;
            };
            let d = (c[0] - pm[0]).powi(2) + (c[1] - pm[1]).powi(2);
            let class = if sj > 0.0 { 0u8 } else { 1u8 };
            let key = (class, d, -sj);
            if best.as_ref().is_none_or(|(bk, _)| *bk > key) {
                best = Some((key, (quad, residual)));
            }
        }
        best.map(|(_, candidate)| candidate)
    }

    /// Walk `id` until it shares an edge with `target`, then merge.  On failure
    /// the current residual id is returned, because a partial walk has already
    /// moved it (a failed attempt leaves a valid mesh, as in Q-Morph).
    fn walk_to(&mut self, mut id: usize, target: usize, pm: [f64; 2]) -> Result<(), usize> {
        let cap = self.cells.len() + 16;
        let mut seen: HashSet<[usize; 3]> = HashSet::new();
        let mut skip: Option<(usize, usize)> = None;
        for _ in 0..cap {
            if !self.is_tri(id) || !self.is_tri(target) {
                return Err(id);
            }
            if self.shares_edge(id, target) {
                if self.merge(id, target) {
                    return Ok(());
                }
                return Err(id);
            }
            let Some(t) = self.tri(id) else {
                return Err(id);
            };
            let mut key = normalize_tri(&self.points, t);
            key.sort_unstable();
            if !seen.insert(key) {
                return Err(id);
            }
            match self.move_toward(id, pm, skip) {
                Move::Moved(next) => {
                    id = next;
                    skip = None;
                }
                Move::Adjacent { id: other, edge } => {
                    // Any residual neighbour can be paired; reaching the target
                    // is not required once two residuals touch.
                    if self.merge(id, other) {
                        return Ok(());
                    }
                    // Its union is not convex. Try the next moving edge rather
                    // than declaring the whole walk impossible.
                    skip = Some(edge);
                }
                Move::Stuck => return Err(id),
            }
        }
        Err(id)
    }

    /// True when the triangle has at least one domain-boundary edge.
    fn tri_has_boundary(&self, id: usize) -> bool {
        let Some(t) = self.tri(id) else {
            return false;
        };
        (0..3).any(|i| self.is_boundary_edge(Cell::Tri(t).edge(i)))
    }

    /// A domain-boundary edge of a quad that is adjacent to the residual, i.e.
    /// a boundary edge the residual could be moved onto in one recombination.
    fn adjacent_boundary_edge(&self, id: usize) -> Option<(usize, usize)> {
        let t = self.tri(id)?;
        for i in 0..3 {
            let edge = Cell::Tri(t).edge(i);
            for &other in self.edge_cells(edge) {
                if other == id {
                    continue;
                }
                if let Some(q) = self.quad(other) {
                    for j in 0..4 {
                        let e = Cell::Quad(q).edge(j);
                        if self.is_boundary_edge(e) {
                            return Some(e);
                        }
                    }
                }
            }
        }
        None
    }

    /// The domain-boundary edge whose midpoint is nearest to `p`.
    ///
    /// Ties are broken by the smaller edge key. `None` only when the mesh has no
    /// boundary (which cannot happen for a triangulated domain, but is guarded).
    fn nearest_boundary_edge(&self, p: [f64; 2]) -> Option<(usize, usize)> {
        let mut best: Option<(f64, (usize, usize))> = None;
        for &edge in &self.boundary {
            let m = midpoint(self.p(edge.0), self.p(edge.1));
            let d = (m[0] - p[0]).powi(2) + (m[1] - p[1]).powi(2);
            if best.is_none_or(|(bd, be)| d < bd || (d == bd && edge < be)) {
                best = Some((d, edge));
            }
        }
        best.map(|(_, edge)| edge)
    }

    /// Move the residual into a triangle that contains `target` (a boundary
    /// edge) by recombining with the quad that owns that edge.
    fn move_onto_boundary(&mut self, id: usize, target: (usize, usize)) -> Option<usize> {
        let t = normalize_tri(&self.points, self.tri(id)?);
        for i in 0..3 {
            let (u, v) = (t[i], t[(i + 1) % 3]);
            let edge = (u.min(v), u.max(v));
            let owners = self.edge_cells(edge);
            if owners.len() != 2 {
                continue;
            }
            let other = if owners[0] == id {
                owners[1]
            } else {
                owners[0]
            };
            let quad = self.quad(other)?;
            let Some(mid) = self.center_of(Cell::Quad(quad)) else {
                continue;
            };
            if let Some((new_quad, residual)) =
                self.recombination(t, id, other, u, v, mid, Some(target))
            {
                self.remove(id);
                self.remove(other);
                self.add(Cell::Quad(new_quad));
                let next = self.add(Cell::Tri(residual));
                self.moves += 1;
                return Some(next);
            }
        }
        None
    }

    /// Rebuild the boundary-edge list from the current edge adjacency.
    fn recompute_boundary(&mut self) {
        self.boundary = self
            .edges
            .iter()
            .filter(|(_, owners)| owners.len() == 1)
            .map(|(&edge, _)| edge)
            .collect();
        self.boundary.sort_unstable();
    }

    /// Sun 2015 §5.2.1 M6-style absorption for the straight-boundary
    /// staircase. A residual `a-b-c` with `b-c` on the boundary, between quads
    /// `b-a-d-e` and `a-c-f-g`, is removed together with the collinear boundary
    /// node `b` (or `c`): the three cells become the two quads `c-f-g-a` +
    /// `c-a-d-e` (or `b-f-g-a` + `b-a-d-e`). This is a hexagon quadrangulation,
    /// not the Fig. 6 template: no point is inserted, one boundary node is
    /// removed, and the operation is only valid when `e-b-c-f` are collinear,
    /// so the boundary geometry is unchanged.
    fn absorb_one_boundary_sliver(&mut self) -> bool {
        let ids = self.tri_ids();
        for id in ids {
            if self.tri_has_boundary(id) && self.absorb_sliver(id) {
                return true;
            }
        }
        false
    }

    fn absorb_sliver(&mut self, id: usize) -> bool {
        let Some(t) = self.tri(id) else {
            return false;
        };
        let t = normalize_tri(&self.points, t);
        for i in 0..3 {
            let b = t[i];
            let c = t[(i + 1) % 3];
            let a = t[(i + 2) % 3];
            let edge = (b.min(c), b.max(c));
            if !self.is_boundary_edge(edge) {
                continue;
            }
            let Some(q1_id) = self.quad_neighbor(a, b, id) else {
                continue;
            };
            let Some(q2_id) = self.quad_neighbor(a, c, id) else {
                continue;
            };
            let Some(q1) = self.quad(q1_id) else {
                continue;
            };
            let Some(q2) = self.quad(q2_id) else {
                continue;
            };
            let q1 = normalize_quad(&self.points, q1);
            let q2 = normalize_quad(&self.points, q2);
            // Q1 traverses b -> a and Q2 traverses a -> c.
            let Some(i1) = (0..4).find(|&k| q1[k] == b && q1[(k + 1) % 4] == a) else {
                continue;
            };
            let d = q1[(i1 + 2) % 4];
            let e = q1[(i1 + 3) % 4];
            let Some(i2) = (0..4).find(|&k| q2[k] == a && q2[(k + 1) % 4] == c) else {
                continue;
            };
            let f = q2[(i2 + 2) % 4];
            let g = q2[(i2 + 3) % 4];
            // The staircase is only removable when e, b, c, f lie on one
            // straight boundary line; otherwise the chord e-c (or b-f) would
            // cut the geometry.
            if !collinear(self.p(e), self.p(b), self.p(c))
                || !collinear(self.p(b), self.p(c), self.p(f))
            {
                continue;
            }
            // The union is a hexagon once b or c is removed. A hexagon has
            // three quadrangulations (the diagonals v0-v3, v1-v4, v2-v5);
            // evaluate all six and keep the best pair.
            let original_area = (tri_area2(&self.points, t).abs()
                + quad_area(&self.points, &q1)
                + quad_area(&self.points, &q2))
            .abs();
            let hexagons = [
                [c, f, g, a, d, e], // remove b
                [b, f, g, a, d, e], // remove c
            ];
            let mut best: Option<([[usize; 4]; 2], f64)> = None;
            for hex in hexagons {
                for k in 0..3 {
                    let pair = [
                        normalize_quad(
                            &self.points,
                            [hex[k], hex[(k + 1) % 6], hex[(k + 2) % 6], hex[(k + 3) % 6]],
                        ),
                        normalize_quad(
                            &self.points,
                            [hex[(k + 3) % 6], hex[(k + 4) % 6], hex[(k + 5) % 6], hex[k]],
                        ),
                    ];
                    if !pair.iter().all(|quad| quad_simple(&self.points, quad)) {
                        continue;
                    }
                    let pair_area =
                        quad_area(&self.points, &pair[0]) + quad_area(&self.points, &pair[1]);
                    if (pair_area - original_area).abs() > 1.0e-6 * original_area.max(1.0) {
                        continue;
                    }
                    let ignore = [id, q1_id, q2_id];
                    if !self.valid_new_cell(Cell::Quad(pair[0]), &ignore)
                        || !self.valid_new_cell(Cell::Quad(pair[1]), &ignore)
                    {
                        continue;
                    }
                    let score = pair
                        .iter()
                        .map(|quad| quad_min_sj(&self.points, quad))
                        .fold(f64::INFINITY, f64::min);
                    if score <= 0.0 {
                        continue;
                    }
                    if best.as_ref().is_none_or(|(_, bs)| score > *bs) {
                        best = Some((pair, score));
                    }
                }
            }
            let Some((best, _)) = best else {
                continue;
            };
            self.remove(id);
            self.remove(q1_id);
            self.remove(q2_id);
            for quad in best {
                self.add(Cell::Quad(quad));
            }
            self.recompute_boundary();
            self.absorptions += 1;
            return true;
        }
        false
    }

    /// Drop a boundary vertex that lies on a straight domain boundary between
    /// its two loop neighbours. No outside cell uses those two boundary edges,
    /// so merging them into one straight edge stays conforming and removes the
    /// collinear nodes that make the cavity DP degenerate.
    fn simplify_boundary_loop(&self, loopv: &mut Vec<usize>) {
        loop {
            let n = loopv.len();
            if n <= 4 {
                return;
            }
            let mut removed = false;
            for i in 0..n {
                let a = loopv[(i + n - 1) % n];
                let b = loopv[i];
                let c = loopv[(i + 1) % n];
                let ab = (a.min(b), a.max(b));
                let bc = (b.min(c), b.max(c));
                if self.is_boundary_edge(ab)
                    && self.is_boundary_edge(bc)
                    && collinear(self.p(a), self.p(b), self.p(c))
                {
                    loopv.remove(i);
                    removed = true;
                    break;
                }
            }
            if !removed {
                return;
            }
        }
    }

    /// The boundary loop of a region, counter-clockwise.
    ///
    /// Returns `None` when the region is not simply connected (a hole or several
    /// loops), when an edge is used more than twice, or when the walk cannot
    /// close. The loop is the sequence of boundary vertices, not edges, and the
    /// closing edge is implicit.
    fn region_boundary(&self, region: &HashSet<usize>) -> Option<Vec<usize>> {
        let mut counts: HashMap<(usize, usize), usize> = HashMap::new();
        for &id in region {
            let cell = self.cell(id)?;
            for i in 0..cell.n_nodes() {
                *counts.entry(cell.edge(i)).or_default() += 1;
            }
        }
        let mut boundary: Vec<(usize, usize)> = counts
            .iter()
            .filter(|(_, count)| **count == 1)
            .map(|(&edge, _)| edge)
            .collect();
        if boundary.is_empty() {
            return None;
        }
        boundary.sort_unstable();
        let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
        for (a, b) in boundary {
            adj.entry(a).or_default().push(b);
            adj.entry(b).or_default().push(a);
        }
        if adj.values().any(|neighbours| neighbours.len() != 2) {
            return None;
        }
        let start = *adj.keys().min()?;
        let mut loopv = Vec::new();
        let mut previous = usize::MAX;
        let mut current = start;
        loop {
            loopv.push(current);
            let neighbours = adj.get(&current)?;
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
            if loopv.len() > adj.len() {
                return None;
            }
        }
        if loopv.len() != adj.len() {
            return None;
        }
        // Counter-clockwise for the quadrangulator.
        let twice: f64 = (0..loopv.len())
            .map(|i| {
                let a = self.p(loopv[i]);
                let b = self.p(loopv[(i + 1) % loopv.len()]);
                a[0] * b[1] - b[0] * a[1]
            })
            .sum();
        if twice < 0.0 {
            loopv.reverse();
        }
        Some(loopv)
    }

    /// A quad adjacent to the region, for one growth step.
    ///
    /// The smallest id is returned so the growth order is deterministic; only
    /// quads (not residuals) are returned, because the region must stay a
    /// quad patch around the residual.
    fn frontier_quad(&self, region: &HashSet<usize>) -> Option<usize> {
        let mut candidates = Vec::new();
        for &id in region {
            let cell = self.cell(id)?;
            for i in 0..cell.n_nodes() {
                for &other in self.edge_cells(cell.edge(i)) {
                    if region.contains(&other) || self.quad(other).is_none() {
                        continue;
                    }
                    candidates.push(other);
                }
            }
        }
        candidates.sort_unstable();
        candidates.dedup();
        candidates.first().copied()
    }

    /// Close one residual by quadrangulating a small cavity around it. Two
    /// residuals are used as seeds when available, so the cavity boundary has
    /// even parity; a lone residual grows until the cavity touches a domain
    /// boundary edge, which is split with one point (the non-local Fig. 6).
    /// Try to close one residual by a cavity: the first live triangle whose
    /// cavity succeeds, in ascending id order. Returns whether one closed.
    fn cavity_one(&mut self, size: f64) -> bool {
        let ids = self.tri_ids();
        for id in ids {
            if self.cavity_close(id, size) {
                return true;
            }
        }
        false
    }

    /// Close the residual `seed` with the first cavity strategy that works.
    ///
    /// Strategies, in order: a compact disk around the seed; a corridor to the
    /// nearest other residual; a compact region around both; a corridor to the
    /// boundary; and finally one-quad-at-a-time growth. Each strategy retries
    /// with growing radii/regions before the next; `size` is the element size in
    /// millimetres and scales the compact radii (zero disables the compact
    /// strategies). A failed strategy leaves the mesh unchanged (the cavity
    /// validation restores the points it inserted).
    fn cavity_close(&mut self, seed: usize, size: f64) -> bool {
        // 0. Compact region grown by Euclidean radius. A one-cell corridor is
        // often too thin to quadrangulate; a disk around the residual is not.
        if size > 0.0 {
            // One growing region reused across the factors: the radii are
            // increasing, so each factor only adds the ring the previous one
            // did not reach. The set is the same one a fresh grow returns.
            let mut region: HashSet<usize> = HashSet::from([seed]);
            if let Some(center) = self.center(seed) {
                for factor in [
                    2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 30.0, 40.0,
                ] {
                    self.grow_compact_region(&mut region, center, factor * size);
                    if region.len() > 1 && self.try_cavity_region(&region) {
                        return true;
                    }
                }
            }
        }
        // 1. Corridor to another residual: the pair gives an even boundary and
        // the DP is not forced to quadrangulate the sliver alone.
        if let Some(partner) = self.nearest_tris(seed).into_iter().next() {
            if partner != seed {
                if let Some(mut region) = self.path_region(seed, partner, 120) {
                    for _ in 0..4 {
                        if self.try_cavity_region(&region) {
                            return true;
                        }
                        if !self.expand_region(&mut region, 200) {
                            break;
                        }
                    }
                }
            }
        }
        // 1b. Compact region around both residuals (no hole between them).
        if let Some(partner) = self.nearest_tris(seed).into_iter().next() {
            if partner != seed && size > 0.0 {
                if let Some(center) = self.both_center(seed, partner) {
                    let mut region: HashSet<usize> = HashSet::from([seed, partner]);
                    for factor in [
                        2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 30.0, 40.0,
                    ] {
                        self.grow_compact_region(&mut region, center, factor * size);
                        if region.len() > 2 && self.try_cavity_region(&region) {
                            return true;
                        }
                    }
                }
            }
        }
        // 2. Corridor to the boundary: one triangle + quads has odd parity, so
        // the boundary edge is split before the quadrangulation.
        if let Some(target) = self.nearest_boundary_cell(seed) {
            if target != seed {
                if let Some(mut region) = self.path_region(seed, target, 160) {
                    for _ in 0..4 {
                        if self.try_cavity_region(&region) {
                            return true;
                        }
                        if !self.expand_region(&mut region, 200) {
                            break;
                        }
                    }
                }
            }
        }
        // 3. Fallback: grow a compact region around the seed.
        let mut region: HashSet<usize> = HashSet::from([seed]);
        for _ in 0..=20 {
            if self.try_cavity_region(&region) {
                return true;
            }
            match self.frontier_quad(&region) {
                Some(next) => {
                    region.insert(next);
                }
                None => break,
            }
        }
        false
    }

    /// Grow `region` to every edge-connected quad whose centroid is within
    /// `radius` of `center`, in place.
    ///
    /// This is the same fixed point as growing a fresh region from the seed:
    /// membership is by reachability through quads inside the radius, so the
    /// starting set only has to be a subset of the answer (it is, for a larger
    /// radius than the one that produced it). The cavity search asks for a
    /// sequence of growing radii around one seed; reusing the previous region
    /// turns twelve independent growths into one incremental one. The visit is
    /// a queue over the region rather than a rescan of it per growth step, so a
    /// cell is examined once per call.
    fn grow_compact_region(&self, region: &mut HashSet<usize>, center: [f64; 2], radius: f64) {
        let mut queue: Vec<usize> = region.iter().copied().collect();
        while let Some(id) = queue.pop() {
            let Some(cell) = self.cell(id) else {
                continue;
            };
            for i in 0..cell.n_nodes() {
                for &other in self.edge_cells(cell.edge(i)) {
                    if region.contains(&other) || self.quad(other).is_none() {
                        continue;
                    }
                    let Some(c) = self.center(other) else {
                        continue;
                    };
                    if (c[0] - center[0]).hypot(c[1] - center[1]) <= radius {
                        region.insert(other);
                        queue.push(other);
                    }
                }
            }
        }
    }

    /// Midpoint of two cells' centres, or `None` if either is gone.
    fn both_center(&self, a: usize, b: usize) -> Option<[f64; 2]> {
        let ca = self.center(a)?;
        let cb = self.center(b)?;
        Some([(ca[0] + cb[0]) * 0.5, (ca[1] + cb[1]) * 0.5])
    }

    /// Shortest path from `start` to `target` through quad cells (the end
    /// points may be triangles). Returns the cells on the path.
    /// Shortest path from `start` to `target` through quad cells (the end
    /// points may be triangles). Returns the cells on the path.
    ///
    /// BFS through quad-only adjacency; a triangle other than `target` is not
    /// traversed. `max_cells` caps both the BFS frontier and the final path;
    /// hitting it returns `None` rather than a truncated path, because a hole in
    /// the region would make the boundary non-simple.
    fn path_region(&self, start: usize, target: usize, max_cells: usize) -> Option<HashSet<usize>> {
        let mut previous: HashMap<usize, usize> = HashMap::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        previous.insert(start, usize::MAX);
        let mut found = false;
        while let Some(id) = queue.pop_front() {
            if previous.len() > max_cells {
                break;
            }
            let Some(cell) = self.cell(id) else {
                continue;
            };
            for i in 0..cell.n_nodes() {
                for &other in self.edge_cells(cell.edge(i)) {
                    if other == id || previous.contains_key(&other) {
                        continue;
                    }
                    if other == target {
                        previous.insert(other, id);
                        found = true;
                        break;
                    }
                    if self.quad(other).is_some() {
                        previous.insert(other, id);
                        queue.push_back(other);
                    }
                }
                if found {
                    break;
                }
            }
            if found {
                break;
            }
        }
        if !found {
            return None;
        }
        let mut region = HashSet::new();
        let mut current = target;
        while current != usize::MAX {
            region.insert(current);
            current = *previous.get(&current)?;
        }
        if region.len() > max_cells {
            return None;
        }
        Some(region)
    }

    /// Add all quads adjacent to the region, up to `limit` cells.
    ///
    /// Returns whether at least one cell was added; when the limit is reached
    /// the remaining candidates are dropped (the caller tries the region again
    /// before expanding further).
    fn expand_region(&self, region: &mut HashSet<usize>, limit: usize) -> bool {
        let mut added = false;
        let mut candidates = Vec::new();
        for &id in region.iter() {
            let Some(cell) = self.cell(id) else {
                continue;
            };
            for i in 0..cell.n_nodes() {
                for &other in self.edge_cells(cell.edge(i)) {
                    if region.contains(&other) || self.quad(other).is_none() {
                        continue;
                    }
                    candidates.push(other);
                }
            }
        }
        candidates.sort_unstable();
        candidates.dedup();
        for other in candidates {
            if region.len() >= limit {
                break;
            }
            if region.insert(other) {
                added = true;
            }
        }
        added
    }

    /// The quad owning the domain boundary edge nearest to the seed.
    ///
    /// Boundary edges owned by a residual are skipped; the return is a quad id,
    /// because the corridor strategy needs a quad to grow through.
    fn nearest_boundary_cell(&self, seed: usize) -> Option<usize> {
        let center = self.center(seed)?;
        let mut best: Option<(f64, usize)> = None;
        for &edge in &self.boundary {
            let mid = midpoint(self.p(edge.0), self.p(edge.1));
            let d = (mid[0] - center[0]).powi(2) + (mid[1] - center[1]).powi(2);
            let Some(&owner) = self.edge_cells(edge).first() else {
                continue;
            };
            if self.quad(owner).is_none() {
                continue;
            }
            if best.is_none_or(|(bd, _)| d < bd) {
                best = Some((d, owner));
            }
        }
        best.map(|(_, owner)| owner)
    }

    /// Quadrangulate `region` and, if the result is valid, replace it.
    ///
    /// Steps: extract the region boundary loop (`None` => not simple), drop
    /// collinear boundary vertices, split one domain-boundary edge when the loop
    /// is odd (inserting a point, removed again on failure), run the DP
    /// quadrangulation, then validate area, Jacobian and manifoldness. The
    /// replacement is atomic: either every region cell is removed and every new
    /// quad is added, or nothing changes and one of the `cavity_*_fail` counters
    /// is incremented.
    fn try_cavity_region(&mut self, region: &HashSet<usize>) -> bool {
        self.cavity_attempts += 1;
        let mut loopv = match self.region_boundary(region) {
            Some(loopv) => loopv,
            None => {
                self.cavity_boundary_fail += 1;
                return false;
            }
        };
        self.simplify_boundary_loop(&mut loopv);
        let mut inserted = None;
        if loopv.len() % 2 == 1 {
            // Split a domain boundary edge so the polygon becomes even.
            let mut split = None;
            for i in 0..loopv.len() {
                let a = loopv[i];
                let b = loopv[(i + 1) % loopv.len()];
                let edge = (a.min(b), a.max(b));
                if self.is_boundary_edge(edge) {
                    split = Some((i, a, b));
                    break;
                }
            }
            let Some((index, a, b)) = split else {
                self.cavity_odd_fail += 1;
                return false;
            };
            let p = midpoint(self.p(a), self.p(b));
            self.points.push([p[0], p[1], 0.0]);
            let point = self.points.len() - 1;
            inserted = Some(point);
            loopv.insert(index + 1, point);
        }
        let quads = match quadrangulate(&self.points, &loopv) {
            Some(quads) => quads,
            None => {
                self.cavity_dp_fail += 1;
                if inserted.is_some() {
                    self.points.pop();
                }
                return false;
            }
        };
        // Deterministic: the ids are removed in this order, and `remove` leaves a
        // hole that later cells' ids are measured against. Sorting keeps the
        // removal order (and therefore the cell ids) independent of `HashSet`
        // iteration order; the area sum is commutative either way.
        let mut region_ids: Vec<usize> = region.iter().copied().collect();
        region_ids.sort_unstable();
        let mut original_area = 0.0;
        for &id in &region_ids {
            if let Some(cell) = self.cell(id) {
                original_area += match cell {
                    Cell::Tri(t) => tri_area2(&self.points, t).abs(),
                    Cell::Quad(q) => quad_area(&self.points, &q),
                };
            }
        }
        let new_area: f64 = quads.iter().map(|q| quad_area(&self.points, q)).sum();
        let area_ok = (new_area - original_area).abs() <= 1.0e-6 * original_area.max(1.0);
        let valid = area_ok
            && quads.iter().all(|q| {
                self.valid_new_cell(Cell::Quad(*q), &region_ids)
                    && quad_min_sj(&self.points, q) >= CAVITY_MIN_SJ
            });
        if !valid {
            self.cavity_valid_fail += 1;
            if inserted.is_some() {
                self.points.pop();
            }
            return false;
        }
        for id in region_ids {
            self.remove(id);
        }
        for q in quads {
            self.add(Cell::Quad(q));
        }
        self.recompute_boundary();
        self.cavities += 1;
        true
    }

    /// Quads whose bilinear Jacobian is not strictly positive.
    ///
    /// `1e-9` rather than `0`: a quad that is positive only at the noise level
    /// is treated as degenerate, because M2/M3 has to make it clearly positive,
    /// not barely so.
    fn bad_quad_ids(&self) -> Vec<usize> {
        let mut out = Vec::new();
        for id in 0..self.cells.len() {
            if let Some(q) = self.quad(id) {
                let q = normalize_quad(&self.points, q);
                if quad_min_sj(&self.points, &q) <= 1e-9 {
                    out.push(id);
                }
            }
        }
        out
    }

    /// Best valid edge swap between two quads: replace their shared edge by
    /// the other diagonal of the hexagon. Returns the score (minimum Jacobian
    /// of the pair), and the two new quads.
    fn best_swap_between(
        &self,
        q1_id: usize,
        q2_id: usize,
    ) -> Option<(f64, [usize; 4], [usize; 4])> {
        let q1 = self.quad(q1_id)?;
        let q2 = self.quad(q2_id)?;
        let q1 = normalize_quad(&self.points, q1);
        let q2 = normalize_quad(&self.points, q2);
        let original_area = quad_area(&self.points, &q1) + quad_area(&self.points, &q2);
        let mut best: Option<(f64, [usize; 4], [usize; 4])> = None;
        for i in 0..4 {
            let x = q1[i];
            let y = q1[(i + 1) % 4];
            // Q2 has to traverse the shared edge in the opposite direction.
            let Some(j) = (0..4).find(|&j| q2[j] == y && q2[(j + 1) % 4] == x) else {
                continue;
            };
            let p = q1[(i + 2) % 4];
            let q = q1[(i + 3) % 4];
            let r = q2[(j + 2) % 4];
            let s = q2[(j + 3) % 4];
            for (left, right) in [([p, q, x, r], [p, r, s, y]), ([q, x, r, s], [q, s, y, p])] {
                let left = normalize_quad(&self.points, left);
                let right = normalize_quad(&self.points, right);
                if !quad_simple(&self.points, &left) || !quad_simple(&self.points, &right) {
                    continue;
                }
                let pair_area = quad_area(&self.points, &left) + quad_area(&self.points, &right);
                if (pair_area - original_area).abs() > 1.0e-6 * original_area.max(1.0) {
                    continue;
                }
                let ignore = [q1_id, q2_id];
                if !self.valid_new_cell(Cell::Quad(left), &ignore)
                    || !self.valid_new_cell(Cell::Quad(right), &ignore)
                {
                    continue;
                }
                let sl = quad_min_sj(&self.points, &left);
                let sr = quad_min_sj(&self.points, &right);
                // A shallow intermediate degeneracy is allowed: the swap is
                // one step of M2/M3 and the next iteration repairs the new
                // boundary quad. The final check in `try_template_repair`
                // requires every quad to be positive again.
                if sl <= -0.05 || sr <= -0.05 {
                    continue;
                }
                let score = sl.min(sr);
                if best.as_ref().is_none_or(|(bs, _, _)| score > *bs) {
                    best = Some((score, left, right));
                }
            }
        }
        best
    }

    /// The smallest Jacobian over all quads (1.0 when there are none).
    fn min_quad_sj(&self) -> f64 {
        let mut min = f64::INFINITY;
        for id in 0..self.cells.len() {
            if let Some(q) = self.quad(id) {
                let q = normalize_quad(&self.points, q);
                min = min.min(quad_min_sj(&self.points, &q));
            }
        }
        if min.is_finite() {
            min
        } else {
            1.0
        }
    }

    /// Apply the M2/M3 swap that best improves quad `id` with one neighbour.
    ///
    /// Chooses the highest-score valid swap among the quad's four neighbours;
    /// returns `false` when no swap is valid. The caller (`try_template_repair`)
    /// loops this over all bad quads.
    fn repair_quad(&mut self, id: usize) -> bool {
        let Some(q1) = self.quad(id) else {
            return false;
        };
        let q1 = normalize_quad(&self.points, q1);
        let mut best: Option<(f64, usize, [usize; 4], [usize; 4])> = None;
        for i in 0..4 {
            let edge = (q1[i].min(q1[(i + 1) % 4]), q1[i].max(q1[(i + 1) % 4]));
            let owners = self.edge_cells(edge);
            if owners.len() != 2 {
                continue;
            }
            let other = if owners[0] == id {
                owners[1]
            } else {
                owners[0]
            };
            let Some((score, left, right)) = self.best_swap_between(id, other) else {
                continue;
            };
            if best.as_ref().is_none_or(|(bs, _, _, _)| score > *bs) {
                best = Some((score, other, left, right));
            }
        }
        if let Some((_, other, left, right)) = best {
            self.remove(id);
            self.remove(other);
            self.add(Cell::Quad(left));
            self.add(Cell::Quad(right));
            self.repairs += 1;
            true
        } else {
            false
        }
    }

    /// Apply the Fig. 6 template and, if it created degenerate boundary quads
    /// (the case where an adjacent Q-Morph quad already has a third node on the
    /// same boundary line), repair them with M2/M3. If that cannot make every
    /// quad positive, the mesh is restored and the residual is left alone.
    fn try_template_repair(&mut self, id: usize) -> bool {
        self.template_attempts += 1;
        let backup = self.clone();
        if !self.apply_template_raw(id) {
            self.template_raw_failures += 1;
            return false;
        }
        for _ in 0..12 {
            let bad = self.bad_quad_ids();
            if bad.is_empty() {
                // Do not accept a template that leaves a near-degenerate quad;
                // the ring regression was one at SJ ~ 0.000.
                return self.min_quad_sj() >= 0.05;
            }
            let mut changed = false;
            for q in &bad {
                if self.quad(*q).is_some() && self.repair_quad(*q) {
                    changed = true;
                    break;
                }
            }
            if !changed {
                break;
            }
        }
        *self = backup;
        self.template_repair_failures += 1;
        false
    }

    /// Template one boundary residual, if any is ready.
    ///
    /// Tries M6 absorption first (the straight-staircase case), then the Fig. 6
    /// template with its M2/M3 repair, in ascending residual id order. Returns
    /// whether one residual was closed.
    fn template_one_boundary(&mut self) -> bool {
        let ids = self.tri_ids();
        for id in ids {
            if self.tri_has_boundary(id) {
                // A single boundary sliver can be absorbed directly when the
                // adjacent quads form the straight staircase (M6), otherwise
                // the Fig. 6 template with its M2/M3 repair is used.
                if self.absorb_sliver(id) {
                    return true;
                }
                if self.try_template_repair(id) {
                    return true;
                }
            }
        }
        false
    }

    /// The Sun 2016, Fig. 6 template: one residual triangle with its boundary
    /// edge and its two adjacent quads become three quads.
    ///
    /// The midpoint of the boundary edge is inserted and the three cells are
    /// replaced by three quads around it (`[P,d,e,b]`, `[P,g,a,d]`, `[P,c,f,g]`
    /// for the local labels). The operation is accepted only if all three are
    /// simple and manifold with respect to the rest of the mesh; the midpoint is
    /// popped again on rejection. Returns whether the template was applied; the
    /// caller then repairs any non-positive quad with M2/M3 or restores a backup.
    fn apply_template_raw(&mut self, id: usize) -> bool {
        let Some(t) = self.tri(id) else {
            return false;
        };
        let t = normalize_tri(&self.points, t);
        for i in 0..3 {
            let b = t[i];
            let c = t[(i + 1) % 3];
            let a = t[(i + 2) % 3];
            let edge = (b.min(c), b.max(c));
            if !self.is_boundary_edge(edge) {
                continue;
            }
            let Some(q1_id) = self.quad_neighbor(a, b, id) else {
                continue;
            };
            let Some(q2_id) = self.quad_neighbor(a, c, id) else {
                continue;
            };
            let Some(q1) = self.quad(q1_id) else {
                continue;
            };
            let Some(q2) = self.quad(q2_id) else {
                continue;
            };
            let q1 = normalize_quad(&self.points, q1);
            let q2 = normalize_quad(&self.points, q2);
            // Q1 is traversed b -> a, so its far edge is d -> e.
            let Some(i1) = (0..4).find(|&k| q1[k] == b && q1[(k + 1) % 4] == a) else {
                continue;
            };
            let d = q1[(i1 + 2) % 4];
            let e = q1[(i1 + 3) % 4];
            // Q2 is traversed a -> c, so its far edge is f -> g.
            let Some(i2) = (0..4).find(|&k| q2[k] == a && q2[(k + 1) % 4] == c) else {
                continue;
            };
            let f = q2[(i2 + 2) % 4];
            let g = q2[(i2 + 3) % 4];
            let p = midpoint(self.p(b), self.p(c));

            // Validate with the new point in place; it is popped again if the
            // template would self-intersect.
            self.points.push([p[0], p[1], 0.0]);
            let inserted = self.points.len() - 1;
            let quads = [
                normalize_quad(&self.points, [inserted, d, e, b]),
                normalize_quad(&self.points, [inserted, g, a, d]),
                normalize_quad(&self.points, [inserted, c, f, g]),
            ];
            // Sun notes that the middle quad may be concave, and on a straight
            // boundary an outer quad can be degenerate when the adjacent
            // Q-Morph quad also touches the line. Accept any simple quad here;
            // `try_template_repair` then runs M2/M3 or restores the mesh.
            let simple = quads.iter().all(|quad| quad_simple(&self.points, quad));
            let ignore = [id, q1_id, q2_id];
            let manifold = quads
                .iter()
                .all(|quad| self.valid_new_cell(Cell::Quad(*quad), &ignore));
            if !simple || !manifold {
                self.points.pop();
                continue;
            }
            self.remove(id);
            self.remove(q1_id);
            self.remove(q2_id);
            for quad in quads {
                self.add(Cell::Quad(quad));
            }
            // The boundary discretisation changed: b-c became b-P and P-c.
            self.boundary.retain(|&e| e != edge);
            self.boundary.push((b.min(inserted), b.max(inserted)));
            self.boundary.push((c.min(inserted), c.max(inserted)));
            self.boundary.sort_unstable();
            self.templates += 1;
            return true;
        }
        false
    }

    /// The quad across edge `a`-`b` of the residual, excluding `exclude`.
    ///
    /// A residual has a triangle on one side of the edge and (for this call) the
    /// quad to be consumed on the other; the function returns that quad.
    fn quad_neighbor(&self, a: usize, b: usize, exclude: usize) -> Option<usize> {
        let edge = (a.min(b), a.max(b));
        self.edge_cells(edge)
            .iter()
            .copied()
            .find(|&c| c != exclude && self.quad(c).is_some())
    }

    /// True when adding `cell` keeps the tiling manifold: every edge on its
    /// boundary belongs to at most one cell that is not being replaced, and no
    /// such cell shares more than one edge with it. Two cells sharing two edges
    /// overlap (the l-bracket regression this check was added for).
    fn valid_new_cell(&self, cell: Cell, ignore: &[usize]) -> bool {
        let mut per_cell: HashMap<usize, usize> = HashMap::new();
        for i in 0..cell.n_nodes() {
            let edge = cell.edge(i);
            let mut outside = 0usize;
            for &owner in self.edge_cells(edge) {
                if ignore.contains(&owner) {
                    continue;
                }
                outside += 1;
                *per_cell.entry(owner).or_default() += 1;
            }
            if outside > 1 {
                return false;
            }
        }
        per_cell.values().all(|&count| count <= 1)
    }

    /// Try to finish the last residual: absorb/template it if it is on the
    /// boundary, otherwise walk it to the boundary and use the template there.
    ///
    /// `cap` bounds the walk (cells + 64); a failed walk leaves the residual
    /// where it got to (valid mesh, still a triangle) and returns `false`.
    fn finish_last(&mut self, mut id: usize) -> bool {
        let cap = self.cells.len() + 64;
        for _ in 0..cap {
            if !self.is_tri(id) {
                return false;
            }
            if self.tri_has_boundary(id) {
                // A single boundary sliver can be absorbed directly when the
                // adjacent quads form the straight staircase (M6), otherwise
                // the Fig. 6 template with its M2/M3 repair is used.
                if self.absorb_sliver(id) {
                    return true;
                }
                if self.try_template_repair(id) {
                    return true;
                }
            }
            if let Some(edge) = self.adjacent_boundary_edge(id) {
                if let Some(next) = self.move_onto_boundary(id, edge) {
                    id = next;
                    continue;
                }
            }
            let Some(center) = self.center(id) else {
                return false;
            };
            let Some(edge) = self.nearest_boundary_edge(center) else {
                return false;
            };
            let pm = midpoint(self.p(edge.0), self.p(edge.1));
            match self.move_toward(id, pm, None) {
                Move::Moved(next) => id = next,
                Move::Adjacent { id: other, .. } => return self.merge(id, other),
                Move::Stuck => return false,
            }
        }
        false
    }

    /// The live cells as `(quads, triangles)`, in id order.
    ///
    /// This is the canonical output order: ids never decrease, so a run with the
    /// same operations produces the same vectors.
    fn collect(&self) -> (Vec<[usize; 4]>, Vec<[usize; 3]>) {
        let mut quads = Vec::new();
        let mut tris = Vec::new();
        for cell in self.cells.iter().flatten() {
            match cell {
                Cell::Quad(q) => quads.push(*q),
                Cell::Tri(t) => tris.push(*t),
            }
        }
        (quads, tris)
    }
}

/// The outcome of one [`WalkMesh::move_toward`] step.
enum Move {
    /// The residual moved; the payload is its new cell id.
    Moved(usize),

    /// The residual reached another residual; `id` is that neighbour and `edge`
    /// is the shared edge (the caller may need to try a different edge if the
    /// merge fails).
    Adjacent { id: usize, edge: (usize, usize) },

    /// No usable edge: the walk cannot move this residual.
    Stuck,
}

/// Close the residual triangles of a Q-Morph conversion.
///
/// Implements the Sun et al. walk of the module documentation: repeatedly pair
/// residuals (moving one towards the other with pentagon recombinations), then
/// finish a boundary residual with the M6 absorption or the Fig. 6 template, and
/// finally try the last one with `WalkMesh::finish_last`. The walk is staged
/// at Jacobian floors `0.2`, `0.1`, `0.0`: a residual that cannot move at a
/// strict floor is retried with a looser one, which is how the last slivers get
/// closed without accepting a bad mesh (the gate is checked afterwards).
///
/// # Postconditions
///
/// The returned `(points, quads, triangles)` tile the input: every input quad
/// and triangle is either still present or was replaced by cells covering the
/// same area. `points` may be longer than the input (the boundary template
/// inserts a midpoint). Every returned triangle is a residual the walk could not
/// move; `CloseStats::remaining` gives its count. The input indices must be valid
/// for `points`; downstream code assumes that, so malformed input is outside the
/// contract.
pub fn close_residuals(
    points: &[Node],
    quads: &[[usize; 4]],
    tris: &[[usize; 3]],
) -> (Vec<Node>, Vec<[usize; 4]>, Vec<[usize; 3]>, CloseStats) {
    let mut stats = CloseStats {
        start_triangles: tris.len(),
        ..CloseStats::default()
    };
    if tris.is_empty() {
        return (points.to_vec(), quads.to_vec(), Vec::new(), stats);
    }
    let mut mesh = WalkMesh::new(points, quads, tris);
    let max_passes = mesh.cells.len().max(1) * 2 + 64;
    // Strict quality first; the lower guards are only used after the 0.2 pass
    // and the boundary absorption/template pass have stalled, for the last few
    // slivers that no 0.2 recombinations can move.
    for guard in [MOVE_MIN_SJ, 0.1, 0.0] {
        mesh.move_min_sj = guard;
        for _ in 0..max_passes {
            let before = mesh.tri_count();
            if before <= 1 {
                break;
            }
            // Pair residuals until the walk stalls at the 0.2 guard.
            loop {
                let ids = mesh.tri_ids();
                if ids.len() <= 1 {
                    break;
                }
                let mut merged_this_pass = 0usize;
                for id in ids {
                    if !mesh.is_tri(id) || mesh.tri_count() <= 1 {
                        continue;
                    }
                    let mut current = id;
                    let mut done = false;
                    // A merge can be geometrically impossible with the nearest
                    // target (the two triangles make a non-convex quad). Try further
                    // residuals before giving up on this one; a failed walk leaves
                    // the moving residual at its new position, which is where the
                    // next target is measured from.
                    let targets = mesh.nearest_tris(current);
                    for target in targets {
                        if mesh.tri_count() <= 1 || !mesh.is_tri(current) || !mesh.is_tri(target) {
                            break;
                        }
                        if mesh.shares_edge(current, target) {
                            if mesh.merge(current, target) {
                                done = true;
                                break;
                            }
                            continue;
                        }
                        let Some(pm) = mesh.center(target) else {
                            continue;
                        };
                        match mesh.walk_to(current, target, pm) {
                            Ok(()) => {
                                done = true;
                                break;
                            }
                            Err(next) => current = next,
                        }
                    }
                    if done {
                        merged_this_pass += 1;
                    } else {
                        stats.stuck += 1;
                    }
                }
                if merged_this_pass == 0 {
                    break;
                }
            }
            if mesh.tri_count() <= 1 {
                break;
            }
            // Pairing stalled. A boundary residual can be finished directly with
            // the Fig. 6 template; M2/M3 repairs the degenerate boundary quads it
            // creates. One at a time, because removing it can make the rest pair.
            // A straight-boundary staircase is finished by M6-style absorption
            // (two quads, one fewer node) before the Fig. 6 template, which would
            // insert a point on the same line and create a degenerate quad.
            if mesh.absorb_one_boundary_sliver() {
                stats.boundary_absorptions += 1;
            } else {
                mesh.template_one_boundary();
            }
            if mesh.tri_count() == before {
                break;
            }
        }
        if mesh.tri_count() <= 1 {
            break;
        }
    }

    if mesh.tri_count() == 1 {
        if let Some(&id) = mesh.tri_ids().first() {
            mesh.finish_last(id);
        }
    }
    stats.merged_pairs = mesh.merged;
    stats.moves = mesh.moves;
    stats.quality_rejections = mesh.quality_rejections;
    stats.merge_failures = mesh.merge_failures;
    stats.boundary_repairs = mesh.repairs;
    stats.boundary_absorptions = mesh.absorptions;
    stats.boundary_templates = mesh.templates;
    stats.cavity_closures = mesh.cavities;
    stats.cavity_attempts = mesh.cavity_attempts;
    stats.cavity_boundary_fail = mesh.cavity_boundary_fail;
    stats.cavity_odd_fail = mesh.cavity_odd_fail;
    stats.cavity_dp_fail = mesh.cavity_dp_fail;
    stats.cavity_valid_fail = mesh.cavity_valid_fail;
    stats.cavity_closures = mesh.cavities;
    stats.cavity_attempts = mesh.cavity_attempts;
    stats.cavity_boundary_fail = mesh.cavity_boundary_fail;
    stats.cavity_odd_fail = mesh.cavity_odd_fail;
    stats.cavity_dp_fail = mesh.cavity_dp_fail;
    stats.cavity_valid_fail = mesh.cavity_valid_fail;
    stats.template_attempts = mesh.template_attempts;
    stats.template_raw_failures = mesh.template_raw_failures;
    stats.template_repair_failures = mesh.template_repair_failures;
    let (quads, tris) = mesh.collect();
    stats.remaining = tris.len();
    (mesh.points, quads, tris, stats)
}

/// Quadrangulate a simple polygon by dynamic programming over non-crossing
/// diagonals. Returns the quads with the largest possible minimum Jacobian.
///
/// `loopv` is the boundary loop in order (counter-clockwise for a positive
/// area); it must have even length, otherwise the polygon cannot be tiled by
/// quads and the result is `None`. The DP state `(i, j)` is the sub-polygon
/// `i..=j`; a state is a quad `(i,k,l,j)` plus three independently solved
/// sub-polygons `(i,k)`, `(k,l)`, `(l,j)`. The objective is the minimum Jacobian
/// over the whole tiling, so a tiling is only as good as its worst quad. The
/// reconstructed area is checked against the polygon area (a crossed or
/// inconsistent tiling is rejected).
fn quadrangulate(points: &[Node], loopv: &[usize]) -> Option<Vec<[usize; 4]>> {
    let n = loopv.len();
    if n < 4 || n % 2 == 1 {
        return None;
    }
    // Dense memo: state `(i, j)` is stored at `i * n + j`. A `HashMap` here is
    // the inner loop of an O(n^4) recursion (n is the simplified loop length,
    // up to a few hundred), and the hash of a pair of small integers cost more
    // than the recurrences around it.
    type State = Option<(f64, usize, usize)>;
    let mut memo: Vec<Option<State>> = vec![None; n * n];
    fn solve(
        i: usize,
        j: usize,
        n: usize,
        loopv: &[usize],
        points: &[Node],
        memo: &mut [Option<State>],
    ) -> State {
        if let Some(cached) = memo[i * n + j] {
            return cached;
        }
        if j == i + 1 {
            memo[i * n + j] = Some(Some((f64::INFINITY, 0, 0)));
            return Some((f64::INFINITY, 0, 0));
        }
        if j == i + 3 {
            let quad = [loopv[i], loopv[i + 1], loopv[i + 2], loopv[j]];
            let quad = normalize_quad(points, quad);
            let result = if quad_simple(points, &quad) {
                let sj = quad_min_sj(points, &quad);
                if sj > 0.0 {
                    Some((sj, i + 1, i + 2))
                } else {
                    None
                }
            } else {
                None
            };
            memo[i * n + j] = Some(result);
            return result;
        }
        let mut best: Option<(f64, usize, usize)> = None;
        let mut k = i + 1;
        while k < j {
            let mut l = k + 1;
            while l < j {
                if let (Some((s1, _, _)), Some((s2, _, _)), Some((s3, _, _))) = (
                    solve(i, k, n, loopv, points, memo),
                    solve(k, l, n, loopv, points, memo),
                    solve(l, j, n, loopv, points, memo),
                ) {
                    let quad = normalize_quad(points, [loopv[i], loopv[k], loopv[l], loopv[j]]);
                    if quad_simple(points, &quad) {
                        let sq = quad_min_sj(points, &quad);
                        if sq > 0.0 {
                            let score = sq.min(s1).min(s2).min(s3);
                            if best.as_ref().is_none_or(|(bs, _, _)| score > *bs) {
                                best = Some((score, k, l));
                            }
                        }
                    }
                }
                l += 2;
            }
            k += 2;
        }
        memo[i * n + j] = Some(best);
        best
    }
    let (_, top_k, top_l) = solve(0, n - 1, n, loopv, points, &mut memo)?;
    fn build(
        i: usize,
        j: usize,
        n: usize,
        loopv: &[usize],
        memo: &[Option<State>],
        out: &mut Vec<[usize; 4]>,
    ) {
        if j <= i + 1 {
            return;
        }
        let Some(Some((_, k, l))) = memo[i * n + j] else {
            return;
        };
        if j == i + 3 {
            out.push([loopv[i], loopv[i + 1], loopv[i + 2], loopv[j]]);
        } else {
            // The quad at this node is part of the tiling; the recursions
            // only cover the three sub-polygons.
            out.push([loopv[i], loopv[k], loopv[l], loopv[j]]);
            build(i, k, n, loopv, memo, out);
            build(k, l, n, loopv, memo, out);
            build(l, j, n, loopv, memo, out);
        }
    }
    let mut quads = Vec::new();
    if n == 4 {
        quads.push([loopv[0], loopv[1], loopv[2], loopv[3]]);
    } else {
        build(0, n - 1, n, loopv, &memo, &mut quads);
        let _ = (top_k, top_l);
    }
    let polygon_area: f64 = (0..n)
        .map(|i| {
            let a = [points[loopv[i]][0], points[loopv[i]][1]];
            let b = [points[loopv[(i + 1) % n]][0], points[loopv[(i + 1) % n]][1]];
            a[0] * b[1] - b[0] * a[1]
        })
        .sum::<f64>()
        .abs()
        * 0.5;
    let quad_area_sum: f64 = quads.iter().map(|q| quad_area(points, q)).sum();
    // quad_area is twice the geometric area; the polygon shoelace is twice
    // the area as well, so compare in the same (doubled) units.
    if (quad_area_sum - 2.0 * polygon_area).abs() > 1.0e-6 * (2.0 * polygon_area).max(1.0) {
        return None;
    }
    Some(quads)
}

/// Repair quads below `threshold` with the M2/M3 edge swap, repeatedly, until
/// none can be improved. Returns the mesh and the number of swaps.
///
/// A quad is a candidate when its `quad_min_sj` is below `threshold`; the swap
/// is the best M2/M3 pair with one of its neighbours (see
/// `WalkMesh::best_swap_between`). Unlike the template repair, a standalone
/// quality repair may **not** leave an inverted quad: if a swap would make the
/// mesh minimum non-positive, the whole pass is rolled back and the input copy
/// is returned with zero swaps. `rounds` bounds the number of accepted swaps,
/// not the number of candidates examined.
pub fn repair_low_quads(
    points: &[Node],
    quads: &[[usize; 4]],
    tris: &[[usize; 3]],
    threshold: f64,
    rounds: usize,
) -> (Vec<Node>, Vec<[usize; 4]>, Vec<[usize; 3]>, usize) {
    let mut mesh = WalkMesh::new(points, quads, tris);
    let backup = mesh.clone();
    let mut swaps = 0usize;
    for _ in 0..rounds {
        let mut changed = false;
        for id in 0..mesh.cells.len() {
            if let Some(q) = mesh.quad(id) {
                let q = normalize_quad(&mesh.points, q);
                if quad_min_sj(&mesh.points, &q) < threshold && mesh.repair_quad(id) {
                    if mesh.min_quad_sj() <= 0.0 {
                        // M2/M3 is allowed to step through a shallow concavity
                        // inside the template repair, but a standalone quality
                        // repair must not leave an inverted quad.
                        let (quads, tris) = backup.collect();
                        return (backup.points, quads, tris, 0);
                    }
                    changed = true;
                    swaps += 1;
                    break;
                }
            }
        }
        if !changed {
            break;
        }
    }
    let (quads, tris) = mesh.collect();
    (mesh.points, quads, tris, swaps)
}

/// Greedy quality pass: swap the shared edge of two quads when the minimum
/// corner angle of the pair improves and both new quads keep SJ >= 0.2. This is
/// the topological half the node smoother cannot do; it runs on the final mesh.
///
/// This is an angle-driven improvement, not the degenerate-Jack repair of
/// [`repair_degenerate_quads`]: it only considers swaps that increase
/// `min(corner angle)` and requires a clearly positive Jacobian, so it never
/// makes a good mesh worse and never chases a collinear (180 degree) corner.
/// `rounds` bounds the accepted swaps; `tris` is carried through untouched.
pub fn improve_quality(
    points: &[Node],
    quads: &[[usize; 4]],
    tris: &[[usize; 3]],
    rounds: usize,
) -> (Vec<Node>, Vec<[usize; 4]>, Vec<[usize; 3]>, usize) {
    let mut mesh = WalkMesh::new(points, quads, tris);
    let mut swaps = 0usize;
    for _ in 0..rounds {
        let mut best: Option<(f64, usize, usize, [usize; 4], [usize; 4])> = None;
        let mut seen: HashSet<(usize, usize)> = HashSet::new();
        for id in 0..mesh.cells.len() {
            let Some(q1) = mesh.quad(id) else {
                continue;
            };
            let q1 = normalize_quad(&mesh.points, q1);
            for i in 0..4 {
                let edge = (q1[i].min(q1[(i + 1) % 4]), q1[i].max(q1[(i + 1) % 4]));
                let owners = mesh.edge_cells(edge);
                if owners.len() != 2 {
                    continue;
                }
                let other = if owners[0] == id {
                    owners[1]
                } else {
                    owners[0]
                };
                if mesh.quad(other).is_none() {
                    continue;
                }
                let key = (id.min(other), id.max(other));
                if !seen.insert(key) {
                    continue;
                }
                let Some((_, left, right)) = mesh.best_swap_between(id, other) else {
                    continue;
                };
                if quad_min_sj(&mesh.points, &left) < 0.2 || quad_min_sj(&mesh.points, &right) < 0.2
                {
                    continue;
                }
                let old = quad_min_angle(&mesh.points, &q1).min(quad_min_angle(
                    &mesh.points,
                    &normalize_quad(&mesh.points, mesh.quad(other).unwrap_or([0; 4])),
                ));
                let new =
                    quad_min_angle(&mesh.points, &left).min(quad_min_angle(&mesh.points, &right));
                let gain = new - old;
                if gain > 1.0e-6 && best.as_ref().is_none_or(|(bg, _, _, _, _)| gain > *bg) {
                    best = Some((gain, id, other, left, right));
                }
            }
        }
        let Some((_, a, b, left, right)) = best else {
            break;
        };
        mesh.remove(a);
        mesh.remove(b);
        mesh.add(Cell::Quad(left));
        mesh.add(Cell::Quad(right));
        swaps += 1;
    }
    let (quads, tris) = mesh.collect();
    (mesh.points, quads, tris, swaps)
}

/// Smallest interior corner angle of a quad, in degrees.
///
/// Returns `0.0` for a degenerate corner (zero-length edge) or a non-finite
/// result, so the caller's comparison always has a real number. The angle is
/// computed from the two edge vectors at each node and clamped to `[-1, 1]`
/// before `acos`, so a nearly collinear corner is 180 degrees, not NaN.
fn quad_min_angle(points: &[Node], quad: &[usize; 4]) -> f64 {
    let p = |i: usize| [points[i][0], points[i][1]];
    let c = [p(quad[0]), p(quad[1]), p(quad[2]), p(quad[3])];
    let mut min = f64::INFINITY;
    for i in 0..4 {
        let here = c[i];
        let prev = c[(i + 3) % 4];
        let next = c[(i + 1) % 4];
        let u = [prev[0] - here[0], prev[1] - here[1]];
        let v = [next[0] - here[0], next[1] - here[1]];
        let denom = (u[0] * u[0] + u[1] * u[1]).sqrt() * (v[0] * v[0] + v[1] * v[1]).sqrt();
        if denom <= 0.0 {
            return 0.0;
        }
        let angle = ((u[0] * v[0] + u[1] * v[1]) / denom)
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        min = min.min(angle);
    }
    if min.is_finite() {
        min
    } else {
        0.0
    }
}

/// Move the nodes of low-quality quads slightly, maximising the **gate margin**
/// of the incident quads: `min(angle / 30°, min_sj / 0.2)`.
///
/// This is the geometric companion of [`repair_degenerate_quads`]. It serves two
/// cases that no swap reaches: a quad with a ~0 corner Jacobian (three collinear
/// nodes), where the corner node has to leave the line; and a transition quad
/// whose corner is below the 30° floor because its two opposite edges differ in
/// length, where sliding a node rebalances it. The step is a small fraction of
/// the local size, and a move is kept only when it raises the worst incident
/// margin, so a good area of the mesh does not move.
///
/// `size` is the local element size in millimetres and sets the search radii
/// (`0.02`, `0.05`, `0.1` of it). `domain` is required to keep boundary nodes on
/// the geometry: with `None`, a node that lies on a mesh boundary edge is left
/// alone; with `Some`, it may slide along its nearest domain segment. Domain
/// sharp corners are never moved. `rounds` bounds the number of accepted moves.
///
/// # Postconditions
///
/// Only points move; the connectivity and the cell lists are unchanged. A move
/// is accepted only if it strictly raises the minimum gate margin over the
/// incident quads (a negative margin is a real score, so a move that lifts a
/// degenerate quad can win). The return count is the number of moves.
pub fn perturb_low_quads(
    points: &[Node],
    quads: &[[usize; 4]],
    tris: &[[usize; 3]],
    size: f64,
    rounds: usize,
    domain: Option<&crate::two_d::Domain>,
) -> (Vec<Node>, Vec<[usize; 4]>, Vec<[usize; 3]>, usize) {
    let mut pts = points.to_vec();
    let quads = quads.to_vec();
    let tris = tris.to_vec();
    // Boundary nodes may only slide **along** the domain boundary. Moving them
    // freely improves the margin while pushing the mesh off the geometry it was
    // built for (measured: 0.57 x size on plate-hole 1 mm and 1.92 on the ring;
    // the gate cannot see it). Without a domain the boundary nodes are left alone.
    let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();
    for cell in quads
        .iter()
        .map(|q| q.as_slice())
        .chain(tris.iter().map(|t| t.as_slice()))
    {
        for i in 0..cell.len() {
            let a = cell[i];
            let b = cell[(i + 1) % cell.len()];
            *edge_count.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    let mut on_boundary = vec![false; pts.len()];
    for ((a, b), n) in edge_count {
        if n == 1 {
            if a < on_boundary.len() {
                on_boundary[a] = true;
            }
            if b < on_boundary.len() {
                on_boundary[b] = true;
            }
        }
    }
    // A domain corner must stay at the corner: sliding it along one of the two
    // adjacent edges cuts the corner while `boundary_offset` stays 0 (measured:
    // (0,0) moved to (0,1.08) by the projection).
    let mut at_corner = vec![false; pts.len()];
    if let Some(d) = domain {
        let corners = d.sharp_corners();
        for (v, p) in pts.iter().enumerate() {
            if !on_boundary[v] {
                continue;
            }
            for c in &corners {
                let dist = ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2)).sqrt();
                if dist < 1.0e-9 {
                    at_corner[v] = true;
                    break;
                }
            }
        }
    }
    let mut incident: HashMap<usize, Vec<usize>> = HashMap::new();
    for (id, q) in quads.iter().enumerate() {
        for &v in q {
            incident.entry(v).or_default().push(id);
        }
    }
    // The score is the worst incident margin, so a move that improves one quad
    // while inverting another lowers the score and is not taken. A negative value
    // (inverted or degenerate quad) is a real score, not a skip: a move that lifts
    // a 0 Jacobian to 0.3 must be allowed to beat it.
    let margin = |pts: &[Node], v: usize| -> f64 {
        let mut score = f64::INFINITY;
        for &id in incident.get(&v).map(Vec::as_slice).unwrap_or(&[]) {
            let sj = quad_min_sj(pts, &quads[id]);
            let angle = quad_min_angle(pts, &quads[id]);
            score = score.min((angle / 30.0).min(sj / 0.2));
        }
        score
    };
    let mut moved = 0usize;
    // Deterministic: when two nodes have the same best score the first one visited
    // wins, so the visit order must not come from the `HashMap` seed. Ascending
    // node id is the canonical order; the score rule itself is unchanged.
    let mut nodes: Vec<usize> = incident.keys().copied().collect();
    nodes.sort_unstable();
    for _ in 0..rounds {
        let mut best: Option<(f64, usize, [f64; 2])> = None;
        for &v in &nodes {
            let current = margin(&pts, v);
            if current >= 1.0 {
                continue;
            }
            let here = [pts[v][0], pts[v][1]];
            for radius in [0.02 * size, 0.05 * size, 0.1 * size] {
                for k in 0..8 {
                    let a = std::f64::consts::TAU * k as f64 / 8.0;
                    if at_corner[v] {
                        continue;
                    }
                    let mut candidate = [here[0] + radius * a.cos(), here[1] + radius * a.sin()];
                    if on_boundary[v] {
                        // A boundary node may slide along the polyline, never off it.
                        match domain {
                            Some(d) => {
                                let (p, q, _) = d.nearest_segment(candidate);
                                candidate = project_on_segment(candidate, p, q);
                            }
                            None => continue,
                        }
                    }
                    pts[v] = [candidate[0], candidate[1], 0.0];
                    let score = margin(&pts, v);
                    pts[v] = [here[0], here[1], 0.0];
                    if score > current + 1.0e-9 && best.as_ref().is_none_or(|(b, _, _)| score > *b)
                    {
                        best = Some((score, v, candidate));
                    }
                }
            }
        }
        let Some((_, v, p)) = best else {
            break;
        };
        pts[v] = [p[0], p[1], 0.0];
        moved += 1;
    }
    (pts, quads, tris, moved)
}

/// Re-split pairs of quads across their shared edge when the pair's **worst
/// corner Jacobian** improves, for quads below `floor`.
///
/// This is the repair for a quad whose corner Jacobian is ~0 because three of its
/// nodes are collinear: `quad_min_sj` sees it, the corner-angle criterion does not
/// (the collinear corner is 180°), and an angle-driven swap never picks it. The
/// swap only touches pairs with a bad member, and both new quads must clear the
/// floor, so an all-good mesh is left untouched.
///
/// "Degenerate" here means `quad_min_sj <= floor`, not necessarily inverted: the
/// pass only considers a pair when at least one member is at or below `floor`,
/// and requires the new minimum to be strictly above both `floor` and the old
/// value. `rounds` bounds the accepted swaps; `tris` is carried through.
pub fn repair_degenerate_quads(
    points: &[Node],
    quads: &[[usize; 4]],
    tris: &[[usize; 3]],
    floor: f64,
    rounds: usize,
) -> (Vec<Node>, Vec<[usize; 4]>, Vec<[usize; 3]>, usize) {
    let mut mesh = WalkMesh::new(points, quads, tris);
    let mut repaired = 0usize;
    for _ in 0..rounds {
        let mut best: Option<(f64, usize, usize, [usize; 4], [usize; 4])> = None;
        let mut seen: HashSet<(usize, usize)> = HashSet::new();
        for id in 0..mesh.cells.len() {
            let Some(q1) = mesh.quad(id) else {
                continue;
            };
            let q1 = normalize_quad(&mesh.points, q1);
            let old1 = quad_min_sj(&mesh.points, &q1);
            for i in 0..4 {
                let edge = (q1[i].min(q1[(i + 1) % 4]), q1[i].max(q1[(i + 1) % 4]));
                let owners = mesh.edge_cells(edge);
                if owners.len() != 2 {
                    continue;
                }
                let other = if owners[0] == id {
                    owners[1]
                } else {
                    owners[0]
                };
                let Some(q2) = mesh.quad(other) else {
                    continue;
                };
                let q2 = normalize_quad(&mesh.points, q2);
                let old = old1.min(quad_min_sj(&mesh.points, &q2));
                if old >= floor {
                    continue;
                }
                let key = (id.min(other), id.max(other));
                if !seen.insert(key) {
                    continue;
                }
                let Some((score, left, right)) = mesh.best_swap_between(id, other) else {
                    continue;
                };
                if score < floor || score <= old + 1.0e-6 {
                    continue;
                }
                if best.as_ref().is_none_or(|(bs, _, _, _, _)| score > *bs) {
                    best = Some((score, id, other, left, right));
                }
            }
        }
        let Some((_, a, b, left, right)) = best else {
            break;
        };
        mesh.remove(a);
        mesh.remove(b);
        mesh.add(Cell::Quad(left));
        mesh.add(Cell::Quad(right));
        repaired += 1;
    }
    let (quads, tris) = mesh.collect();
    (mesh.points, quads, tris, repaired)
}

/// Post-process the leftovers with the local cavity quadrangulation. This runs
/// after the ordinary pairing/absorption/template pipeline has stalled; the
/// cavity is deliberately not part of that pipeline because it changes the
/// ordering the front needs.
///
/// Runs at most 64 successful cavities; `size` is the element size in
/// millimetres and scales the compact regions (zero disables them). Returns the
/// new mesh and the number of cells closed. Residuals that no cavity can close
/// are returned unchanged; the function never forces a tiling.
pub fn close_by_cavity(
    points: &[Node],
    quads: &[[usize; 4]],
    tris: &[[usize; 3]],
    size: f64,
) -> (Vec<Node>, Vec<[usize; 4]>, Vec<[usize; 3]>, usize) {
    let mut mesh = WalkMesh::new(points, quads, tris);
    let mut closed = 0usize;
    for _ in 0..64 {
        if !mesh.cavity_one(size) {
            break;
        }
        closed += 1;
    }
    let (quads, tris) = mesh.collect();
    (mesh.points, quads, tris, closed)
}

/// Midpoint of two points, in the XY plane.
fn midpoint(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
}

/// Twice the signed area of triangle `a-b-c` (positive when counter-clockwise).
fn area2(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

fn tri_area2(points: &[Node], t: [usize; 3]) -> f64 {
    let p = |i: usize| [points[i][0], points[i][1]];
    area2(p(t[0]), p(t[1]), p(t[2]))
}

/// The scaled Jacobian of the triangle formed by `pm` and the edge `a -> b`.
/// It is negative when `pm` lies across the edge, which is the direction the
/// residual has to move.
fn point_edge_sj(pm: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let u = [a[0] - pm[0], a[1] - pm[1]];
    let v = [b[0] - pm[0], b[1] - pm[1]];
    let det = u[0] * v[1] - u[1] * v[0];
    let norm = (u[0] * u[0] + u[1] * u[1]).sqrt() * (v[0] * v[0] + v[1] * v[1]).sqrt();
    if norm > 0.0 {
        det / norm
    } else {
        0.0
    }
}

/// True when the cell's boundary contains the undirected edge.
fn cell_has_edge(cell: Cell, edge: (usize, usize)) -> bool {
    (0..cell.n_nodes()).any(|i| cell.edge(i) == edge)
}

/// Put a triangle in counter-clockwise order and rotate it to start at the
/// smallest node id.
///
/// The rotation is what makes the walk deterministic: tie-breaks follow the
/// stored node order, so `(1, 2, 0)` and `(0, 1, 2)` must compare equal before
/// any state is stored. This is the fix recorded in the `the_closure_does_not_
/// depend_on_triangle_rotation` test.
fn normalize_tri(points: &[Node], t: [usize; 3]) -> [usize; 3] {
    let t = if tri_area2(points, t) < 0.0 {
        [t[0], t[2], t[1]]
    } else {
        t
    };
    // Canonical rotation: start at the smallest node id. The walk's tie-breaks
    // follow the stored node order, so the same triangle written as (1, 2, 0)
    // instead of (0, 1, 2) sent the closure down a different path. Normalising
    // the rotation makes the result depend on the cells, not on the order the
    // caller happened to write them.
    let i = if t[0] <= t[1] && t[0] <= t[2] {
        0
    } else if t[1] <= t[2] {
        1
    } else {
        2
    };
    [t[i], t[(i + 1) % 3], t[(i + 2) % 3]]
}

/// Put a quad in counter-clockwise order and rotate it to start at the smallest
/// node id, for the same determinism reason as [`normalize_tri`].
///
/// Reversing a clockwise quad must preserve its cyclic neighbour order:
/// `[q0, q3, q2, q1]` is the reversal of the cycle, not a swap of two corners.
fn normalize_quad(points: &[Node], q: [usize; 4]) -> [usize; 4] {
    let p = |i: usize| [points[i][0], points[i][1]];
    let twice: f64 = (0..4)
        .map(|i| {
            let a = p(q[i]);
            let b = p(q[(i + 1) % 4]);
            a[0] * b[1] - b[0] * a[1]
        })
        .sum();
    let q = if twice < 0.0 {
        [q[0], q[3], q[2], q[1]]
    } else {
        q
    };
    // Canonical rotation, for the same reason as `normalize_tri`.
    let mut i = 0;
    for k in 1..4 {
        if q[k] < q[i] {
            i = k;
        }
    }
    [q[i], q[(i + 1) % 4], q[(i + 2) % 4], q[(i + 3) % 4]]
}

/// Closest point to `p` on the segment `a`-`b`.
///
/// The projection parameter is clamped, so the result is always on the closed
/// segment (a boundary node cannot slide off its segment). A degenerate segment
/// returns `a`.
fn project_on_segment(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    let d = [b[0] - a[0], b[1] - a[1]];
    let len2 = d[0] * d[0] + d[1] * d[1];
    if len2 <= 1.0e-24 {
        return a;
    }
    let t = (((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / len2).clamp(0.0, 1.0);
    [a[0] + t * d[0], a[1] + t * d[1]]
}

/// The bilinear scaled Jacobian minima used by the quality contract.
///
/// It may be negative when a corner is inverted; [`quad_sj`] is the positive-only
/// view. The value is the minimum over the four reference corners of
/// `det J / (|J_col0| |J_col1|)`, exactly the gate's metric; `-1.0` is returned
/// when the map is degenerate, so callers can compare without special-casing
/// NaN.
fn quad_min_sj(points: &[Node], quad: &[usize; 4]) -> f64 {
    const XI: [[f64; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];
    let p = [
        [points[quad[0]][0], points[quad[0]][1]],
        [points[quad[1]][0], points[quad[1]][1]],
        [points[quad[2]][0], points[quad[2]][1]],
        [points[quad[3]][0], points[quad[3]][1]],
    ];
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
        let sj = if c0 * c1 > 0.0 { det / (c0 * c1) } else { -1.0 };
        min = min.min(sj);
    }
    if min.is_finite() {
        min
    } else {
        -1.0
    }
}

/// Twice the absolute area of a simple quad.
///
/// The doubling is deliberate and consistent across this module: the triangle
/// helper [`tri_area2`] also returns twice the area, and all area comparisons
/// (`absorb_sliver`, `try_cavity_region`, `quadrangulate`) are made in these
/// doubled units. Do not mix it with `two_d::quad_area`, which halves.
fn quad_area(points: &[Node], quad: &[usize; 4]) -> f64 {
    let p = |i: usize| [points[i][0], points[i][1]];
    let a = p(quad[0]);
    let b = p(quad[1]);
    let c = p(quad[2]);
    let d = p(quad[3]);
    (area2(a, b, c) + area2(a, c, d)).abs()
}

/// The positive-only scaled Jacobian of a quad, or `None` when it is inverted.
fn quad_sj(points: &[Node], quad: &[usize; 4]) -> Option<f64> {
    let sj = quad_min_sj(points, quad);
    if sj > 0.0 {
        Some(sj)
    } else {
        None
    }
}

/// A simple (non-self-intersecting) quad with non-zero area.
fn quad_simple(points: &[Node], quad: &[usize; 4]) -> bool {
    let p = |i: usize| [points[i][0], points[i][1]];
    let c = [p(quad[0]), p(quad[1]), p(quad[2]), p(quad[3])];
    if area2(c[0], c[1], c[2]).abs() < 1e-14 && area2(c[1], c[2], c[3]).abs() < 1e-14 {
        return false;
    }
    if area2(c[0], c[1], c[2]).abs() < 1e-14 && area2(c[2], c[3], c[0]).abs() < 1e-14 {
        return false;
    }
    !segments_cross(c[0], c[1], c[2], c[3]) && !segments_cross(c[1], c[2], c[3], c[0])
}

/// Three points on one line, relative to the segment lengths.
fn collinear(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> bool {
    let cross = area2(a, b, c).abs();
    let scale = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2))
        .sqrt()
        .max(((c[0] - b[0]).powi(2) + (c[1] - b[1]).powi(2)).sqrt())
        .max(((c[0] - a[0]).powi(2) + (c[1] - a[1]).powi(2)).sqrt());
    cross <= 1.0e-9 * scale * scale
}

/// True when the open segments `p`-`q` and `r`-`s` cross; touching or collinear
/// segments return `false`.
fn segments_cross(p: [f64; 2], q: [f64; 2], r: [f64; 2], s: [f64; 2]) -> bool {
    let d1 = area2(p, q, r);
    let d2 = area2(p, q, s);
    let d3 = area2(r, s, p);
    let d4 = area2(r, s, q);
    d1 * d2 < 0.0 && d3 * d4 < 0.0
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::mesh::{Cell as MeshCell, Mesh};

    fn square() -> Vec<Node> {
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    }

    #[test]
    fn a_boundary_doublet_is_not_fixable_by_perturbation() {
        // Three bottom nodes on the domain boundary, collinear: the middle one is
        // a boundary doublet (a 180-degree corner of its quad). Sliding it along
        // the boundary cannot remove the collinearity, and moving it off the
        // boundary is forbidden, so the perturb leaves it alone. Removing it needs
        // a boundary-doublet collapse / local re-quadrangulation, not a move.
        let domain = crate::two_d::Domain::rectangle(2.0, 1.0).unwrap();
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let quads = [[0, 1, 2, 3]];
        let (pts, _, _, _) = perturb_low_quads(&points, &quads, &[], 1.0, 20, Some(&domain));
        let d = domain.distance_to_boundary([pts[1][0], pts[1][1]]);
        assert!(d < 1.0e-12, "the boundary doublet left the boundary: {d}");
        assert!(quad_min_sj(&pts, &quads[0]) <= 1.0e-9);
    }

    #[test]
    fn a_domain_corner_node_is_never_moved() {
        let domain = crate::two_d::Domain::rectangle(1.0, 1.0).unwrap();
        // Node 0 is the domain corner (0,0) and the quad is badly shaped there;
        // the perturb must leave the corner exactly where it is.
        let points = vec![
            [0.0, 0.0, 0.0],
            [0.9, 0.05, 0.0],
            [1.0, 1.0, 0.0],
            [0.02, 0.9, 0.0],
        ];
        let quads = [[0, 1, 2, 3]];
        let (pts, _, _, _) = perturb_low_quads(&points, &quads, &[], 1.0, 50, Some(&domain));
        assert!(pts[0][0].abs() < 1.0e-12 && pts[0][1].abs() < 1.0e-12);
    }

    #[test]
    fn without_a_domain_the_boundary_nodes_are_left_alone() {
        let points = vec![
            [0.0, 0.0, 0.0],
            [0.5, 0.25, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let quads = [[0, 1, 2, 3]];
        let (pts, _, _, _) = perturb_low_quads(&points, &quads, &[], 1.0, 20, None);
        for (a, b) in points.iter().zip(pts.iter()) {
            assert!((a[0] - b[0]).abs() < 1.0e-12 && (a[1] - b[1]).abs() < 1.0e-12);
        }
    }

    #[test]
    fn a_point_projects_onto_its_segment() {
        let p = project_on_segment([0.5, 0.3], [0.0, 0.0], [1.0, 0.0]);
        assert!((p[0] - 0.5).abs() < 1.0e-12 && p[1].abs() < 1.0e-12);
        let p = project_on_segment([-1.0, 2.0], [0.0, 0.0], [1.0, 0.0]);
        assert!(p[0].abs() < 1.0e-12 && p[1].abs() < 1.0e-12);
    }

    #[test]
    fn the_degenerate_repair_leaves_a_good_mesh_alone() {
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
        ];
        let quads = [[0, 1, 2, 3], [1, 5, 4, 2]];
        let (_, out, _, repaired) = repair_degenerate_quads(&points, &quads, &[], 0.25, 8);
        assert_eq!(repaired, 0);
        assert_eq!(out, quads);
    }

    #[test]
    fn the_closure_does_not_depend_on_triangle_rotation() {
        // The front's cells, with every triangle rotated by one: the closure must
        // produce the same quads. Before the canonical rotation, the walk's
        // tie-breaks followed the stored node order and the result changed.
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 1.6, 0.0],
        ];
        let quads = [[0, 1, 2, 3]];
        let tris = [[1, 4, 2], [0, 1, 3]];
        let rotated: Vec<[usize; 3]> = tris.iter().map(|t| [t[1], t[2], t[0]]).collect();
        let (_, q1, t1, _) = close_residuals(&points, &quads, &tris);
        let (_, q2, t2, _) = close_residuals(&points, &quads, &rotated);
        let mut a: Vec<[usize; 4]> = q1.clone();
        a.sort_unstable();
        let mut b: Vec<[usize; 4]> = q2.clone();
        b.sort_unstable();
        assert_eq!(a, b, "closure changed with rotation: {q1:?} vs {q2:?}");
        assert_eq!(t1.len(), t2.len());
    }

    #[test]
    fn two_triangles_become_one_quad() {
        let (points, quads, tris, stats) = close_residuals(&square(), &[], &[[0, 1, 2], [0, 2, 3]]);
        assert!(tris.is_empty(), "{tris:?}");
        assert_eq!(quads.len(), 1);
        assert_eq!(stats.merged_pairs, 1);
        assert_eq!(stats.remaining, 0);
        let mesh = Mesh::new(points, quads.iter().map(|q| MeshCell::quad4(*q)).collect());
        assert_eq!(mesh.topology().cracks, 0);
    }

    #[test]
    fn an_all_quad_mesh_is_unchanged() {
        let (points, quads, tris, stats) = close_residuals(&square(), &[[0, 1, 2, 3]], &[]);
        assert_eq!(points.len(), 4);
        assert_eq!(quads.len(), 1);
        assert!(tris.is_empty());
        assert_eq!(stats.start_triangles, 0);
    }

    #[test]
    fn a_lone_boundary_triangle_uses_the_sun_template() {
        // The Fig. 6 layout: triangle a-b-c with b-c on the boundary, and the
        // two quads (a, d, e, b) and (a, c, f, g) on its other sides.
        let points = vec![
            [0.5, 1.0, 0.0],  // a
            [0.0, 0.0, 0.0],  // b
            [1.0, 0.0, 0.0],  // c
            [-1.0, 1.5, 0.0], // d
            [-1.0, 0.5, 0.0], // e
            [2.0, 0.5, 0.0],  // f
            [2.0, 1.5, 0.0],  // g
        ];
        let quads = [[0, 3, 4, 1], [0, 2, 5, 6]];
        let tris = [[0, 1, 2]];
        let (points, quads, tris, stats) = close_residuals(&points, &quads, &tris);
        assert!(tris.is_empty(), "{tris:?}");
        assert_eq!(quads.len(), 3, "{quads:?}");
        assert_eq!(points.len(), 8);
        assert_eq!(stats.boundary_templates, 1);
        assert!(
            quads.iter().all(|q| quad_min_sj(&points, q) > 0.0),
            "{quads:?}"
        );
        let mesh = Mesh::new(points, quads.iter().map(|q| MeshCell::quad4(*q)).collect());
        assert_eq!(mesh.topology().cracks, 0);
    }

    #[test]
    fn a_straight_boundary_staircase_is_absorbed() {
        // A boundary sliver a-b-c on y = 0, between quads b-a-d-e and a-c-f-g,
        // with e, b, c, f collinear. The M6-style absorption removes one
        // boundary node and replaces the three cells by two quads.
        let points = vec![
            [0.5, 1.0, 0.0],  // a
            [0.0, 0.0, 0.0],  // b
            [1.0, 0.0, 0.0],  // c
            [0.5, 2.0, 0.0],  // d
            [-1.0, 0.0, 0.0], // e
            [2.0, 0.0, 0.0],  // f
            [1.5, 2.0, 0.0],  // g
        ];
        let quads = [[1, 0, 3, 4], [0, 2, 5, 6]];
        let tris = [[0, 1, 2]];
        let (points, quads, tris, stats) = close_residuals(&points, &quads, &tris);
        assert!(tris.is_empty(), "{tris:?}");
        assert_eq!(quads.len(), 2, "{quads:?}");
        assert_eq!(stats.boundary_absorptions, 1, "{stats:?}");
        assert_eq!(stats.boundary_templates, 0, "{stats:?}");
        assert!(
            quads.iter().all(|q| quad_min_sj(&points, q) > 0.0),
            "{quads:?}"
        );
        let mesh = Mesh::new(points, quads.iter().map(|q| MeshCell::quad4(*q)).collect());
        assert_eq!(mesh.topology().cracks, 0);
    }

    #[test]
    fn a_regular_hexagon_quadrangulates() {
        let points: Vec<Node> = (0..6)
            .map(|i| {
                let a = std::f64::consts::TAU * i as f64 / 6.0;
                [a.cos(), a.sin(), 0.0]
            })
            .collect();
        let loopv: Vec<usize> = (0..6).collect();
        assert!(
            quadrangulate(&points, &loopv).is_some(),
            "hexagon did not quadrangulate"
        );
    }

    #[test]
    fn a_regular_octagon_quadrangulates() {
        let points: Vec<Node> = (0..8)
            .map(|i| {
                let a = std::f64::consts::TAU * i as f64 / 8.0;
                [a.cos(), a.sin(), 0.0]
            })
            .collect();
        let loopv: Vec<usize> = (0..8).collect();
        let quads = quadrangulate(&points, &loopv);
        assert!(quads.is_some(), "octagon did not quadrangulate");
        assert_eq!(quads.map(|q| q.len()), Some(3));
    }

    #[test]
    fn a_residual_crosses_a_quad_to_reach_its_target() {
        // T1 | quad | T2 in a 3 x 1 strip.
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
        ];
        let quads = [[1, 4, 5, 2]];
        let tris = [[0, 1, 2], [4, 7, 5]];
        let (_points, quads, tris, stats) = close_residuals(&points, &quads, &tris);
        assert_eq!(stats.merged_pairs, 1, "{stats:?}");
        assert!(tris.is_empty(), "{tris:?}");
        assert_eq!(quads.len(), 2, "{quads:?}");
    }
}
