//! Complete triangle-to-quad conversion: Sun's residual-triangle walk.
//!
//! Q-Morph (Owen et al. 1999) stops with an even number of residual triangles
//! per domain. Sun, Yeh, Lin & Zhao (2015, *Eng. Comput.* 31:675-692; 2016,
//! *Terr. Atmos. Ocean. Sci.* 27:139-152) close them without subdividing a
//! triangle and without touching a quad that is not on the path:
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
//! - **M2/M3 edge swap** ([`WalkMesh::repair_quad`]): after the template, swap
//!   the degenerate quad's shared edge with an adjacent quad; if the pair of
//!   new quads is positive, the degeneracy is gone.
//! - **M6-style absorption** ([`WalkMesh::absorb_sliver`]): when `e-b-c-f` are
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

use std::collections::{HashMap, HashSet};

use crate::mesh::Node;

/// A cell of the closing mesh.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cell {
    Tri([usize; 3]),
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
    pub cavity_attempts: usize,
    pub cavity_boundary_fail: usize,
    pub cavity_odd_fail: usize,
    pub cavity_dp_fail: usize,
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

#[derive(Clone)]
struct WalkMesh {
    points: Vec<Node>,
    cells: Vec<Option<Cell>>,
    edges: HashMap<(usize, usize), Vec<usize>>,
    tris: HashSet<usize>,
    boundary: Vec<(usize, usize)>,
    moves: usize,
    merged: usize,
    quality_rejections: usize,
    merge_failures: usize,
    repairs: usize,
    templates: usize,
    cavities: usize,
    cavity_attempts: usize,
    cavity_boundary_fail: usize,
    cavity_odd_fail: usize,
    cavity_dp_fail: usize,
    cavity_valid_fail: usize,
    move_min_sj: f64,
    template_attempts: usize,
    template_raw_failures: usize,
    template_repair_failures: usize,
    absorptions: usize,
}

impl WalkMesh {
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

    fn push_raw(&mut self, cell: Cell) -> usize {
        self.cells.push(Some(cell));
        self.cells.len() - 1
    }

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

    fn add(&mut self, cell: Cell) -> usize {
        let id = self.push_raw(cell);
        self.index(id);
        id
    }

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

    fn cell(&self, id: usize) -> Option<Cell> {
        self.cells.get(id).copied().flatten()
    }

    fn is_tri(&self, id: usize) -> bool {
        self.cell(id).is_some_and(Cell::is_tri)
    }

    fn tri_ids(&self) -> Vec<usize> {
        let mut ids: Vec<usize> = self.tris.iter().copied().collect();
        ids.sort_unstable();
        ids
    }

    fn tri_count(&self) -> usize {
        self.tris.len()
    }

    fn edge_cells(&self, edge: (usize, usize)) -> &[usize] {
        self.edges.get(&edge).map(Vec::as_slice).unwrap_or(&[])
    }

    fn is_boundary_edge(&self, edge: (usize, usize)) -> bool {
        self.edge_cells(edge).len() == 1
    }

    fn tri(&self, id: usize) -> Option<[usize; 3]> {
        match self.cell(id) {
            Some(Cell::Tri(t)) => Some(t),
            _ => None,
        }
    }

    fn quad(&self, id: usize) -> Option<[usize; 4]> {
        match self.cell(id) {
            Some(Cell::Quad(q)) => Some(q),
            _ => None,
        }
    }

    fn p(&self, i: usize) -> [f64; 2] {
        [self.points[i][0], self.points[i][1]]
    }

    fn center(&self, id: usize) -> Option<[f64; 2]> {
        let cell = self.cell(id)?;
        self.center_of(cell)
    }

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

    fn tri_has_boundary(&self, id: usize) -> bool {
        let Some(t) = self.tri(id) else {
            return false;
        };
        (0..3).any(|i| self.is_boundary_edge(Cell::Tri(t).edge(i)))
    }

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

    /// The boundary cycle of a connected region of cells, or `None` when the
    /// region is not simply connected (a hole or several loops).
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
    fn cavity_one(&mut self) -> bool {
        let ids = self.tri_ids();
        for id in ids {
            if self.cavity_close(id) {
                return true;
            }
        }
        false
    }

    fn cavity_close(&mut self, seed: usize) -> bool {
        let mut seed_sets = vec![vec![seed]];
        if let Some(partner) = self.nearest_tris(seed).into_iter().next() {
            if partner != seed {
                seed_sets.push(vec![seed, partner]);
            }
        }
        for seeds in seed_sets {
            let mut region: HashSet<usize> = seeds.iter().copied().collect();
            for _ in 0..=10 {
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
        }
        false
    }

    fn try_cavity_region(&mut self, region: &HashSet<usize>) -> bool {
        self.cavity_attempts += 1;
        let mut loopv = match self.region_boundary(region) {
            Some(loopv) => loopv,
            None => {
                self.cavity_boundary_fail += 1;
                return false;
            }
        };
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
        let region_ids: Vec<usize> = region.iter().copied().collect();
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
            && quads
                .iter()
                .all(|q| self.valid_new_cell(Cell::Quad(*q), &region_ids));
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

    /// Apply the best swap with one neighbour of `id`.
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
                return true;
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
            if self.cavity_close(id) {
                return true;
            }
        }
        false
    }

    /// The Sun 2016, Fig. 6 template: one residual triangle with its boundary
    /// edge and its two adjacent quads become three quads.
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
            if self.cavity_close(id) {
                return true;
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

enum Move {
    Moved(usize),
    Adjacent { id: usize, edge: (usize, usize) },
    Stuck,
}

/// Close the residual triangles of a Q-Morph conversion.
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
            } else if !mesh.template_one_boundary() {
                // Last resort: a local cavity quadrangulation around an
                // isolated interior sliver.
                mesh.cavity_one();
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
fn quadrangulate(points: &[Node], loopv: &[usize]) -> Option<Vec<[usize; 4]>> {
    let n = loopv.len();
    if n < 4 || n % 2 == 1 {
        return None;
    }
    type Memo = HashMap<(usize, usize), Option<(f64, usize, usize)>>;
    fn solve(
        i: usize,
        j: usize,
        loopv: &[usize],
        points: &[Node],
        memo: &mut Memo,
    ) -> Option<(f64, usize, usize)> {
        if let Some(cached) = memo.get(&(i, j)) {
            return *cached;
        }
        if j == i + 1 {
            memo.insert((i, j), Some((f64::INFINITY, 0, 0)));
            return Some((f64::INFINITY, 0, 0));
        }
        if j == i + 3 {
            let quad = [loopv[i], loopv[i + 1], loopv[i + 2], loopv[j]];
            let quad = normalize_quad(points, quad);
            let result = if quad_simple(points, &quad) {
                let sj = quad_min_sj(points, &quad);
                if sj > -0.05 {
                    Some((sj, i + 1, i + 2))
                } else {
                    None
                }
            } else {
                None
            };
            memo.insert((i, j), result);
            return result;
        }
        let mut best: Option<(f64, usize, usize)> = None;
        let mut k = i + 1;
        while k + 2 < j {
            let mut l = k + 1;
            while l + 1 < j {
                if let (Some((s1, _, _)), Some((s2, _, _)), Some((s3, _, _))) = (
                    solve(i, k, loopv, points, memo),
                    solve(k, l, loopv, points, memo),
                    solve(l, j, loopv, points, memo),
                ) {
                    let quad = normalize_quad(points, [loopv[i], loopv[k], loopv[l], loopv[j]]);
                    if quad_simple(points, &quad) {
                        let sq = quad_min_sj(points, &quad);
                        if sq > -0.05 {
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
        memo.insert((i, j), best);
        best
    }
    let mut memo: Memo = HashMap::new();
    let (_, top_k, top_l) = solve(0, n - 1, loopv, points, &mut memo)?;
    fn build(i: usize, j: usize, loopv: &[usize], memo: &Memo, out: &mut Vec<[usize; 4]>) {
        if j <= i + 1 {
            return;
        }
        let Some((_, k, l)) = memo.get(&(i, j)).copied().flatten() else {
            return;
        };
        if j == i + 3 {
            out.push([loopv[i], loopv[i + 1], loopv[i + 2], loopv[j]]);
        } else {
            build(i, k, loopv, memo, out);
            build(k, l, loopv, memo, out);
            build(l, j, loopv, memo, out);
        }
    }
    let mut quads = Vec::new();
    if n == 4 {
        quads.push([loopv[0], loopv[1], loopv[2], loopv[3]]);
    } else {
        build(0, n - 1, loopv, &memo, &mut quads);
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
    if (quad_area_sum - polygon_area).abs() > 1.0e-6 * polygon_area.max(1.0) {
        return None;
    }
    Some(quads)
}

/// Greedy quality pass: swap the shared edge of two quads when the minimum
/// corner angle of the pair improves and both new quads keep SJ >= 0.2. This is
/// the topological half the node smoother cannot do; it runs on the final mesh.
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

fn midpoint(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
}

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

fn cell_has_edge(cell: Cell, edge: (usize, usize)) -> bool {
    (0..cell.n_nodes()).any(|i| cell.edge(i) == edge)
}

fn normalize_tri(points: &[Node], t: [usize; 3]) -> [usize; 3] {
    if tri_area2(points, t) < 0.0 {
        [t[0], t[2], t[1]]
    } else {
        t
    }
}

fn normalize_quad(points: &[Node], q: [usize; 4]) -> [usize; 4] {
    let p = |i: usize| [points[i][0], points[i][1]];
    let twice: f64 = (0..4)
        .map(|i| {
            let a = p(q[i]);
            let b = p(q[(i + 1) % 4]);
            a[0] * b[1] - b[0] * a[1]
        })
        .sum();
    if twice < 0.0 {
        [q[0], q[3], q[2], q[1]]
    } else {
        q
    }
}

/// The bilinear scaled Jacobian minima used by the quality contract. It may be
/// negative when a corner is inverted; [`quad_sj`] is the positive-only view.
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

/// The absolute area of a simple quad.
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
