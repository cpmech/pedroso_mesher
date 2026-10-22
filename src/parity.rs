//! Boundary-loop parity control for the Q-Morph input.
//!
//! Sun et al. (2016, §3.3) show that an odd boundary loop produces an odd number
//! of triangles, and an odd number of residuals leaves one triangle that cannot
//! be paired. The fix is to insert one point on the longest edge of every odd
//! loop before conversion, which makes the loop even and flips the triangle
//! parity. The point is the straight-edge midpoint; the mesher's boundaries are
//! polylines, so it lies on the domain boundary.

use std::collections::HashMap;

use crate::mesh::Node;

/// Make every boundary loop of the triangulation even.
///
/// Returns the points, the triangles and the number of inserted points. A
/// triangle carrying a split edge is divided into two, preserving winding.
pub fn even_boundary_loops(
    points: &[Node],
    tris: &[[usize; 3]],
) -> (Vec<Node>, Vec<[usize; 3]>, usize) {
    let mut points = points.to_vec();
    let mut tris = tris.to_vec();
    let mut splits = 0usize;
    for _ in 0..16 {
        let loops = boundary_loops(&tris);
        let Some(odd) = loops.iter().find(|edges| edges.len() % 2 == 1) else {
            break;
        };
        let Some(&edge) = odd.iter().max_by(|left, right| {
            let l = length(&points, **left);
            let r = length(&points, **right);
            l.partial_cmp(&r).unwrap_or(std::cmp::Ordering::Equal)
        }) else {
            break;
        };
        let Some(tri_id) = tris.iter().position(|t| {
            (0..3).any(|i| {
                let (a, b) = (t[i], t[(i + 1) % 3]);
                (a.min(b), a.max(b)) == edge
            })
        }) else {
            break;
        };
        let tri = tris[tri_id];
        let Some(i) = (0..3).find(|&i| {
            let (a, b) = (tri[i], tri[(i + 1) % 3]);
            (a.min(b), a.max(b)) == edge
        }) else {
            break;
        };
        let (a, b, c) = (tri[i], tri[(i + 1) % 3], tri[(i + 2) % 3]);
        let p = midpoint(points[a], points[b]);
        points.push([p[0], p[1], 0.0]);
        let point = points.len() - 1;
        tris[tri_id] = [a, point, c];
        tris.push([point, b, c]);
        splits += 1;
    }
    (points, tris, splits)
}

fn boundary_loops(tris: &[[usize; 3]]) -> Vec<Vec<(usize, usize)>> {
    let mut owners: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (id, t) in tris.iter().enumerate() {
        for i in 0..3 {
            let (a, b) = (t[i], t[(i + 1) % 3]);
            owners.entry((a.min(b), a.max(b))).or_default().push(id);
        }
    }
    let mut edges: Vec<(usize, usize)> = owners
        .iter()
        .filter(|(_, ids)| ids.len() == 1)
        .map(|(&edge, _)| edge)
        .collect();
    edges.sort_unstable();
    let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
    for (a, b) in edges {
        adj.entry(a).or_default().push(b);
        adj.entry(b).or_default().push(a);
    }
    if adj.values().any(|neighbours| neighbours.len() != 2) {
        return Vec::new();
    }
    let mut loops = Vec::new();
    let mut used: HashMap<usize, bool> = HashMap::new();
    let mut starts: Vec<usize> = adj.keys().copied().collect();
    starts.sort_unstable();
    for start in starts {
        if used.get(&start).copied().unwrap_or(false) {
            continue;
        }
        let mut loop_edges = Vec::new();
        let mut previous = usize::MAX;
        let mut current = start;
        loop {
            used.insert(current, true);
            let Some(neighbours) = adj.get(&current) else {
                break;
            };
            let next = if neighbours[0] == previous {
                neighbours[1]
            } else {
                neighbours[0]
            };
            loop_edges.push((current.min(next), current.max(next)));
            previous = current;
            current = next;
            if current == start {
                break;
            }
            if loop_edges.len() > adj.len() {
                return Vec::new();
            }
        }
        loops.push(loop_edges);
    }
    loops
}

fn length(points: &[Node], edge: (usize, usize)) -> f64 {
    let a = points[edge.0];
    let b = points[edge.1];
    ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt()
}

fn midpoint(a: Node, b: Node) -> [f64; 2] {
    [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn an_odd_boundary_loop_is_split() {
        let points = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let (points, tris, splits) = even_boundary_loops(&points, &[[0, 1, 2]]);
        assert_eq!(splits, 1);
        assert_eq!(points.len(), 4);
        assert_eq!(tris.len(), 2);
        let loops = boundary_loops(&tris);
        assert!(loops.iter().all(|loop_edges| loop_edges.len() % 2 == 0));
    }
}
