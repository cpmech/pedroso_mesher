//! Q-Morph: converting a triangle mesh into quads by advancing a front.
//!
//! This is the second attempt at Owen's algorithm (1999), written from scratch
//! after the first one (`quadify`) plateaued. The structural difference is the
//! one its failures pointed at: **quads and triangles are cells of the same
//! mesh**, so every operation — edge flip, node insertion, cell replacement — is
//! expressed on one adjacency structure. The first attempt kept the quads in a
//! separate list and had to protect them by refusing flips, which is what capped
//! its coverage at ~86 %.
//!
//! The front is the set of edges with a triangle on one side and a quad (or
//! nothing) on the other. For each front edge the procedure places two side
//! nodes at the local size along the inward normal — inserting them when no
//! existing node is close — recovers the three sides by flips among triangles,
//! and replaces the triangles enclosed by the quad with one quad.

use crate::mesh::Node;

/// A cell of the converting mesh: either a triangle or a quad.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cell {
    Tri([usize; 3]),
    Quad([usize; 4]),
}

impl Cell {
    fn nodes(&self) -> Vec<usize> {
        match self {
            Cell::Tri(t) => t.to_vec(),
            Cell::Quad(q) => q.to_vec(),
        }
    }

    fn is_tri(&self) -> bool {
        matches!(self, Cell::Tri(_))
    }
}

/// A mesh whose cells are triangles and quads, with edge adjacency.
pub struct CellMesh {
    pub points: Vec<Node>,
    pub cells: Vec<Option<Cell>>,
    edges: std::collections::HashMap<(usize, usize), Vec<usize>>,
}

impl CellMesh {
    pub fn new(points: &[Node], tris: &[[usize; 3]]) -> Self {
        let mut mesh = CellMesh {
            points: points.to_vec(),
            cells: tris.iter().map(|t| Some(Cell::Tri(*t))).collect(),
            edges: std::collections::HashMap::new(),
        };
        for id in 0..mesh.cells.len() {
            mesh.index_cell(id);
        }
        mesh
    }

    fn index_cell(&mut self, id: usize) {
        if let Some(Some(cell)) = self.cells.get(id) {
            for edge in cell_edges(cell) {
                self.edges.entry(edge).or_default().push(id);
            }
        }
    }

    fn unindex_cell(&mut self, id: usize) {
        if let Some(Some(cell)) = self.cells.get(id) {
            for edge in cell_edges(cell) {
                if let Some(list) = self.edges.get_mut(&edge) {
                    list.retain(|&c| c != id);
                    if list.is_empty() {
                        self.edges.remove(&edge);
                    }
                }
            }
        }
    }

    fn set_cell(&mut self, id: usize, cell: Cell) {
        self.unindex_cell(id);
        self.cells[id] = Some(cell);
        self.index_cell(id);
    }

    fn push_cell(&mut self, cell: Cell) -> usize {
        self.cells.push(Some(cell));
        let id = self.cells.len() - 1;
        self.index_cell(id);
        id
    }

    fn remove_cell(&mut self, id: usize) {
        self.unindex_cell(id);
        self.cells[id] = None;
    }

    fn cell_ids(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.cells.len()).filter(|&i| self.cells[i].is_some())
    }

    fn edge_cells(&self, edge: (usize, usize)) -> Vec<usize> {
        self.edges.get(&edge).cloned().unwrap_or_default()
    }

    /// True when the edge separates the unconverted region from the meshed one.
    ///
    /// The interface edge has a triangle on one side and a quad on the other, so
    /// it has *two* cells — the first version tested for exactly one and therefore
    /// only ever saw the domain boundary, which is why the front never advanced.
    fn front_edge(&self, edge: (usize, usize)) -> bool {
        let cells = self.edge_cells(edge);
        let tris = cells
            .iter()
            .filter(|&&c| self.cells[c].is_some_and(|cell| cell.is_tri()))
            .count();
        tris == 1 && cells.len() <= 2
    }

    /// Flip `(u, v)` when both adjacent cells are triangles forming a convex quad.
    fn flip(&mut self, u: usize, v: usize) -> bool {
        let owners = self.edge_cells((u.min(v), u.max(v)));
        if owners.len() != 2 {
            return false;
        }
        let (Some(Cell::Tri(t1)), Some(Cell::Tri(t2))) =
            (self.cells[owners[0]], self.cells[owners[1]])
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
        // convex quad p-u-q-v: p and q on opposite sides of (u, v), and back
        let s = |o: usize, a: usize, b: usize| {
            let o = self.points[o];
            let a = self.points[a];
            let b = self.points[b];
            (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        };
        if s(u, v, p) * s(u, v, q) >= 0.0 || s(p, q, u) * s(p, q, v) >= 0.0 {
            return false;
        }
        self.set_cell(owners[0], Cell::Tri([p, q, u]));
        self.set_cell(owners[1], Cell::Tri([q, p, v]));
        true
    }

    /// Split the triangle containing `p` into three; returns the new node.
    fn insert(&mut self, p: [f64; 2]) -> Option<usize> {
        let mut found = None;
        for id in self.cell_ids() {
            let Some(Cell::Tri(t)) = self.cells[id] else {
                continue;
            };
            if point_strictly_inside(&self.points, t, p) {
                found = Some((id, t));
                break;
            }
        }
        let (id, t) = found?;
        self.points.push([p[0], p[1], 0.0]);
        let v = self.points.len() - 1;
        self.set_cell(id, Cell::Tri([t[0], t[1], v]));
        self.push_cell(Cell::Tri([t[1], t[2], v]));
        self.push_cell(Cell::Tri([t[2], t[0], v]));
        Some(v)
    }

    /// The triangle cells enclosed by the quad's four sides, if the region is
    /// exactly that quad.
    fn region(&self, seed: usize, quad: &[usize; 4]) -> Option<Vec<usize>> {
        let mut sides = std::collections::HashSet::new();
        for i in 0..4 {
            let (a, b) = (quad[i], quad[(i + 1) % 4]);
            sides.insert((a.min(b), a.max(b)));
        }
        let mut region = vec![seed];
        let mut seen = std::collections::HashSet::new();
        seen.insert(seed);
        let mut head = 0;
        while head < region.len() {
            let id = region[head];
            head += 1;
            let cell = self.cells[id]?;
            for edge in cell_edges(&cell) {
                if sides.contains(&edge) {
                    continue;
                }
                for other in self.edge_cells(edge) {
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
            for v in cell.nodes() {
                if !inside_quad(&self.points, quad, v) {
                    return None;
                }
            }
        }
        Some(region)
    }
}

fn cell_edges(cell: &Cell) -> Vec<(usize, usize)> {
    let nodes = cell.nodes();
    (0..nodes.len())
        .map(|i| {
            let (a, b) = (nodes[i], nodes[(i + 1) % nodes.len()]);
            (a.min(b), a.max(b))
        })
        .collect()
}

fn point_strictly_inside(points: &[Node], tri: [usize; 3], p: [f64; 2]) -> bool {
    let q = [p[0], p[1], 0.0];
    let d0 = cross(points[tri[0]], points[tri[1]], q);
    let d1 = cross(points[tri[1]], points[tri[2]], q);
    let d2 = cross(points[tri[2]], points[tri[0]], q);
    (d0 > 1e-12 && d1 > 1e-12 && d2 > 1e-12) || (d0 < -1e-12 && d1 < -1e-12 && d2 < -1e-12)
}

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

fn cross(a: Node, b: Node, c: Node) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// Convert a triangle mesh into quads.
///
/// Returns the quads, the triangles that could not be converted, and the node
/// set the conversion ended with (placement inserts nodes).
pub fn convert(
    tris: &[[usize; 3]],
    points: &[Node],
    size: f64,
) -> (Vec<[usize; 4]>, Vec<[usize; 3]>, Vec<Node>) {
    convert_with_stats(tris, points, size).0
}

/// Counters for the stage at which a front edge failed.
#[derive(Debug, Default, Clone, Copy)]
pub struct Refusals {
    pub front_empty: usize,
    pub placement: usize,
    pub invalid_quad: usize,
    pub recovery: usize,
    pub region: usize,
}

pub fn convert_with_stats(
    tris: &[[usize; 3]],
    points: &[Node],
    size: f64,
) -> ((Vec<[usize; 4]>, Vec<[usize; 3]>, Vec<Node>), Refusals) {
    let mut refusals = Refusals::default();
    let mut mesh = CellMesh::new(points, tris);
    let mut quads: Vec<[usize; 4]> = Vec::new();
    let mut blocked: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();

    let mut iterations = 0usize;
    loop {
        iterations += 1;
        if iterations > 20 * tris.len() + 100 {
            break;
        }
        let front: Vec<(usize, usize)> = mesh
            .edges
            .iter()
            .filter(|(&edge, _)| mesh.front_edge(edge) && !blocked.contains(&edge))
            .map(|(&edge, _)| edge)
            .collect();
        if front.is_empty() {
            break;
        }
        // shortest first: the front advances where the quads are smallest
        let mut order = front;
        order.sort_by(|&e1, &e2| {
            length(&mesh.points, e1)
                .partial_cmp(&length(&mesh.points, e2))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut progressed = false;
        for edge in order {
            match try_quad(&mut mesh, edge, size, &mut quads) {
                Ok(()) => {
                    progressed = true;
                    break;
                }
                Err(stage) => {
                    match stage {
                        0 => refusals.placement += 1,
                        1 => refusals.invalid_quad += 1,
                        2 => refusals.recovery += 1,
                        _ => refusals.region += 1,
                    }
                    blocked.insert(edge);
                }
            }
        }
        if !progressed {
            break;
        }
        // only the edges around the new quad can have become feasible again
        if let Some(quad) = quads.last() {
            blocked.retain(|&(u, v)| !quad.contains(&u) && !quad.contains(&v));
        }
    }

    let leftover: Vec<[usize; 3]> = mesh
        .cell_ids()
        .filter_map(|id| match mesh.cells[id] {
            Some(Cell::Tri(t)) => Some(t),
            _ => None,
        })
        .collect();
    ((quads, leftover, mesh.points), refusals)
}

fn length(points: &[Node], edge: (usize, usize)) -> f64 {
    let (a, b) = (points[edge.0], points[edge.1]);
    ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt()
}

/// Try to build one quad on the front edge, committing only on success.
fn try_quad(
    mesh: &mut CellMesh,
    edge: (usize, usize),
    size: f64,
    quads: &mut Vec<[usize; 4]>,
) -> Result<(), u8> {
    let owners = mesh.edge_cells(edge);
    let Some(&seed) = owners
        .iter()
        .find(|&&c| mesh.cells[c].is_some_and(|cell| cell.is_tri()))
    else {
        return Err(0);
    };
    let Some(Cell::Tri(tri)) = mesh.cells[seed] else {
        return Err(0);
    };
    // orient the front edge so the triangle is on its left: the edge key is
    // (min, max), which carries no direction
    let third = *tri
        .iter()
        .find(|&&n| n != edge.0 && n != edge.1)
        .unwrap_or(&edge.0);
    let (a, b) = {
        let p = mesh.points[edge.0];
        let q = mesh.points[edge.1];
        let r = mesh.points[third];
        if (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]) > 0.0 {
            (edge.0, edge.1)
        } else {
            (edge.1, edge.0)
        }
    };
    let s = length(&mesh.points, (a, b)).max(1e-12);
    let target = if s > 1e-9 { s } else { size };

    // inward normal: from the edge midpoint towards the triangle's centroid
    let pa = mesh.points[a];
    let pb = mesh.points[b];
    let mid = [(pa[0] + pb[0]) / 2.0, (pa[1] + pb[1]) / 2.0];
    let pc = mesh.points[third];
    let n = [pc[0] - mid[0], pc[1] - mid[1]];
    let nn = (n[0] * n[0] + n[1] * n[1]).sqrt();
    if nn <= 1e-12 {
        return Err(0);
    }
    let n = [n[0] / nn, n[1] / nn];

    let ideal_c = [pa[0] + target * n[0], pa[1] + target * n[1]];
    let ideal_d = [pb[0] + target * n[0], pb[1] + target * n[1]];

    // work on a copy: placement inserts nodes, and a failed quad must not leave
    // them behind
    let mut trial = mesh.clone_for_trial();
    let c = match nearest_node(&trial.points, ideal_c, 1.2 * target) {
        Some(v) => v,
        None => match trial.insert(ideal_c) {
            Some(v) => v,
            None => {
                if std::env::var("QMORPH_DEBUG").is_ok() {
                    eprintln!(
                        "RUSTDBG a={:?} b={:?} n={:?} ideal_c={:?} inside_tri={} cells={}",
                        mesh.points[a], mesh.points[b], n, ideal_c,
                        trial.cell_ids().any(|i| matches!(trial.cells[i], Some(Cell::Tri(t)) if point_strictly_inside(&trial.points, t, ideal_c))),
                        trial.cell_ids().count()
                    );
                }
                return Err(0);
            }
        },
    };
    let d = match nearest_node(&trial.points, ideal_d, 1.2 * target) {
        Some(v) => v,
        None => match trial.insert(ideal_d) {
            Some(v) => v,
            None => return Err(0),
        },
    };
    if c == d || c == a || c == b || d == a || d == b {
        return Err(0);
    }
    let quad = [a, b, d, c];
    if !valid(&trial.points, &quad) {
        return Err(1);
    }
    // a quad that merely is convex can still be a sliver: require a real
    // Jacobian, as the quality contract does
    let coords = [
        [trial.points[quad[0]][0], trial.points[quad[0]][1]],
        [trial.points[quad[1]][0], trial.points[quad[1]][1]],
        [trial.points[quad[2]][0], trial.points[quad[2]][1]],
        [trial.points[quad[3]][0], trial.points[quad[3]][1]],
    ];
    if crate::two_d::quad_min_sj_public(&coords) < 0.3 {
        return Err(1);
    }
    for (u, v) in [(a, c), (b, d), (c, d)] {
        if !trial.has_edge(u, v) && !trial.recover(u, v) {
            return Err(2);
        }
    }
    let Some(region) = trial.region(seed, &quad) else {
        return Err(3);
    };
    if region.is_empty() {
        return Err(3);
    }
    for id in region {
        trial.remove_cell(id);
    }
    trial.push_cell(Cell::Quad(quad));
    *mesh = trial;
    quads.push(quad);
    Ok(())
}

fn valid(points: &[Node], quad: &[usize; 4]) -> bool {
    let mut area = 0.0;
    for i in 0..4 {
        let a = points[quad[i]];
        let b = points[quad[(i + 1) % 4]];
        area += a[0] * b[1] - b[0] * a[1];
    }
    if area <= 1e-12 {
        return false;
    }
    (0..4).all(|i| {
        cross(
            points[quad[(i + 3) % 4]],
            points[quad[i]],
            points[quad[(i + 1) % 4]],
        ) > 0.0
    })
}

fn nearest_node(points: &[Node], p: [f64; 2], radius: f64) -> Option<usize> {
    let mut best = None;
    let mut best_d = radius;
    for (i, q) in points.iter().enumerate() {
        let d = ((q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2)).sqrt();
        if d < best_d {
            best_d = d;
            best = Some(i);
        }
    }
    best
}

impl CellMesh {
    fn clone_for_trial(&self) -> CellMesh {
        CellMesh {
            points: self.points.clone(),
            cells: self.cells.clone(),
            edges: self.edges.clone(),
        }
    }

    fn has_edge(&self, u: usize, v: usize) -> bool {
        self.edges.contains_key(&(u.min(v), u.max(v)))
    }

    /// Recover edge `(u, v)` by flipping crossing edges between triangles.
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

    fn first_crossing(&self, u: usize, v: usize) -> Option<(usize, usize)> {
        let p = self.points[u];
        let q = self.points[v];
        for (&edge, owners) in &self.edges {
            if owners.len() != 2 {
                continue;
            }
            let (a, b) = (edge.0, edge.1);
            if a == u || a == v || b == u || b == v {
                continue;
            }
            let (ra, rb) = (self.points[a], self.points[b]);
            let d1 = side_of(p, q, ra);
            let d2 = side_of(p, q, rb);
            let d3 = side_of(ra, rb, p);
            let d4 = side_of(ra, rb, q);
            if d1 * d2 < 0.0 && d3 * d4 < 0.0 {
                return Some(edge);
            }
        }
        None
    }
}

fn side_of(a: Node, b: Node, c: Node) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}
