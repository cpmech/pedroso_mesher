//! A uniform grid over a planar mesh, for point location.
//!
//! Every placement question in the Q-Morph front — *is this point inside an
//! unconverted triangle?*, *does this segment stay in the unconverted region?* —
//! is a point location. Done by scanning the cell list it is O(cells) per query
//! and the conversion simply does not finish; that is what this index removes.

use std::collections::HashMap;

use crate::mesh::Node;

/// A cell of the mesh, as far as the index is concerned: its node indices.
pub type CellNodes = Vec<usize>;

/// Uniform bucket grid over cells.
pub struct Grid {
    size: f64,
    buckets: HashMap<(i64, i64), Vec<usize>>,
    boxes: Vec<([f64; 2], [f64; 2])>,
}

impl Grid {
    /// Index `cells` (each a node list) laid out on `points`.
    ///
    /// `size` is the bucket edge; it should be about the element size, so that a
    /// cell touches one or a few buckets.
    pub fn build(points: &[Node], cells: &[CellNodes], size: f64) -> Self {
        let size = if size > 1e-9 { size } else { 1.0 };
        let mut grid = Grid {
            size,
            buckets: HashMap::new(),
            boxes: vec![([0.0, 0.0], [0.0, 0.0]); cells.len()],
        };
        for (id, cell) in cells.iter().enumerate() {
            if cell.is_empty() {
                continue;
            }
            let mut lo = [f64::INFINITY; 2];
            let mut hi = [f64::NEG_INFINITY; 2];
            for &v in cell {
                let p = points[v];
                lo[0] = lo[0].min(p[0]);
                lo[1] = lo[1].min(p[1]);
                hi[0] = hi[0].max(p[0]);
                hi[1] = hi[1].max(p[1]);
            }
            grid.boxes[id] = (lo, hi);
            let (i0, j0) = grid.key(lo);
            let (i1, j1) = grid.key(hi);
            for i in i0..=i1 {
                for j in j0..=j1 {
                    grid.buckets.entry((i, j)).or_default().push(id);
                }
            }
        }
        grid
    }

    fn key(&self, p: [f64; 2]) -> (i64, i64) {
        (
            (p[0] / self.size).floor() as i64,
            (p[1] / self.size).floor() as i64,
        )
    }

    /// Cell ids whose bounding box may contain `p`.
    pub fn candidates(&self, p: [f64; 2]) -> Vec<usize> {
        let (i, j) = self.key(p);
        let mut out = Vec::new();
        for di in -1..=1 {
            for dj in -1..=1 {
                if let Some(ids) = self.buckets.get(&(i + di, j + dj)) {
                    for &id in ids {
                        if !out.contains(&id) {
                            out.push(id);
                        }
                    }
                }
            }
        }
        out
    }

    /// Whether `p` is inside the (convex) cell `id`, boundary included.
    pub fn contains(&self, id: usize, points: &[Node], cell: &[usize], p: [f64; 2]) -> bool {
        let (lo, hi) = self.boxes[id];
        if p[0] < lo[0] - 1e-9 || p[0] > hi[0] + 1e-9 || p[1] < lo[1] - 1e-9 || p[1] > hi[1] + 1e-9
        {
            return false;
        }
        let q = [p[0], p[1], 0.0];
        let n = cell.len();
        let mut positive = 0;
        let mut negative = 0;
        for i in 0..n {
            let a = points[cell[i]];
            let b = points[cell[(i + 1) % n]];
            let s = (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]);
            if s > 1e-12 {
                positive += 1;
            } else if s < -1e-12 {
                negative += 1;
            }
        }
        positive == 0 || negative == 0
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn two_triangles() -> (Vec<Node>, Vec<CellNodes>) {
        (
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            vec![vec![0, 1, 2], vec![0, 2, 3]],
        )
    }

    #[test]
    fn a_point_inside_a_cell_is_found() {
        let (points, cells) = two_triangles();
        let grid = Grid::build(&points, &cells, 0.5);
        let hit = grid
            .candidates([0.7, 0.3])
            .into_iter()
            .find(|&id| grid.contains(id, &points, &cells[id], [0.7, 0.3]));
        assert!(hit.is_some(), "no cell found for an interior point");
    }

    #[test]
    fn a_point_outside_every_cell_is_not_found() {
        let (points, cells) = two_triangles();
        let grid = Grid::build(&points, &cells, 0.5);
        let hit = grid
            .candidates([2.0, 2.0])
            .into_iter()
            .find(|&id| grid.contains(id, &points, &cells[id], [2.0, 2.0]));
        assert!(hit.is_none());
    }

    #[test]
    fn cell_centroids_locate_themselves() {
        let (points, cells) = two_triangles();
        let grid = Grid::build(&points, &cells, 0.5);
        for (id, cell) in cells.iter().enumerate() {
            let mut c = [0.0, 0.0];
            for &v in cell {
                c[0] += points[v][0] / cell.len() as f64;
                c[1] += points[v][1] / cell.len() as f64;
            }
            let found = grid
                .candidates(c)
                .into_iter()
                .any(|k| grid.contains(k, &points, &cells[k], c));
            assert!(found, "centroid of cell {id} not located");
        }
    }
}
