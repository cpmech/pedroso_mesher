//! Recovering a prescribed edge in a triangle mesh by local swaps.
//!
//! This is the primitive behind Q-Morph's quad construction (Owen 1999, §3.3.2,
//! Algorithm 1): once the two side edges of a quad are known, the top edge has to
//! *exist* in the triangulation before the quad can be formed. If it does not,
//! every edge that crosses the wanted segment is flipped, one at a time, until it
//! does — the same edge-recovery used by boundary-constrained Delaunay meshers
//! (Jones, Sloan, George).

use crate::mesh::Node;

/// A triangle as three node indices.
pub type Tri = [usize; 3];

/// Make edge `(a, b)` an edge of the triangulation, flipping crossing edges.
///
/// Returns `false` when the segment runs through a node, when a crossing edge
/// cannot be flipped (its two triangles do not form a convex quad), or when the
/// flips do not converge. The mesh is then in a partially recovered state; the
/// caller is expected to work on a copy.
pub fn recover_edge(tris: &mut [Tri], points: &[Node], a: usize, b: usize) -> bool {
    if a == b {
        return false;
    }
    for _ in 0..tris.len() {
        if has_edge(tris, a, b) {
            return true;
        }
        let Some((u, v)) = first_crossing(tris, points, a, b) else {
            return has_edge(tris, a, b);
        };
        if !flip_edge(tris, points, u, v) {
            return false;
        }
    }
    has_edge(tris, a, b)
}

/// True when some triangle has `a` and `b` as consecutive nodes.
pub fn has_edge(tris: &[Tri], a: usize, b: usize) -> bool {
    tris.iter()
        .any(|t| (0..3).any(|i| t[i] == a && t[(i + 1) % 3] == b))
}

/// An edge that the open segment `a..b` crosses, chosen deterministically.
fn first_crossing(tris: &[Tri], points: &[Node], a: usize, b: usize) -> Option<(usize, usize)> {
    let p = points[a];
    let q = points[b];
    for t in tris {
        for i in 0..3 {
            let (u, v) = (t[i], t[(i + 1) % 3]);
            if u == a || u == b || v == a || v == b {
                continue;
            }
            if segments_cross(p, q, points[u], points[v]) {
                return Some((u, v));
            }
        }
    }
    None
}

/// Flip edge `(u, v)` when its two triangles form a convex quadrilateral.
fn flip_edge(tris: &mut [Tri], points: &[Node], u: usize, v: usize) -> bool {
    let mut owners: Vec<usize> = (0..tris.len())
        .filter(|&i| tris[i].contains(&u) && tris[i].contains(&v))
        .collect();
    if owners.len() != 2 {
        return false;
    }
    let (p, q) = (apex(&tris[owners[0]], u, v), apex(&tris[owners[1]], u, v));
    if p == q {
        return false;
    }
    // the quad p-u-q-v must be convex: p and q on opposite sides of (u, v)
    if side(points[u], points[v], points[p]) * side(points[u], points[v], points[q]) >= 0.0 {
        return false;
    }
    if side(points[p], points[q], points[u]) * side(points[p], points[q], points[v]) >= 0.0 {
        return false;
    }
    owners.sort_unstable();
    tris[owners[0]] = [p, q, u];
    tris[owners[1]] = [q, p, v];
    true
}

fn apex(tri: &Tri, u: usize, v: usize) -> usize {
    *tri.iter()
        .find(|&&n| n != u && n != v)
        .unwrap_or(&usize::MAX)
}

/// Signed area of the triangle (o, a, b): positive when `b` is left of `o -> a`.
fn side(o: Node, a: Node, b: Node) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

/// True when the open segments `p-q` and `r-s` cross properly.
fn segments_cross(p: Node, q: Node, r: Node, s: Node) -> bool {
    let d1 = side(p, q, r);
    let d2 = side(p, q, s);
    let d3 = side(r, s, p);
    let d4 = side(r, s, q);
    d1 * d2 < 0.0 && d3 * d4 < 0.0
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    /// Unit square, one diagonal.
    fn square() -> (Vec<Node>, Vec<Tri>) {
        (
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            vec![[0, 1, 2], [0, 2, 3]],
        )
    }

    #[test]
    fn an_existing_edge_is_recognised() {
        let (_, tris) = square();
        assert!(has_edge(&tris, 0, 2));
        assert!(!has_edge(&tris, 1, 3));
    }

    #[test]
    fn the_other_diagonal_is_recovered_with_one_flip() {
        let (points, mut tris) = square();
        assert!(recover_edge(&mut tris, &points, 1, 3));
        assert!(has_edge(&tris, 1, 3));
        assert!(!has_edge(&tris, 0, 2));
        // both triangles still cover the square
        let area: f64 = tris
            .iter()
            .map(|t| {
                let (a, b, c) = (points[t[0]], points[t[1]], points[t[2]]);
                0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])).abs()
            })
            .sum();
        assert!((area - 1.0).abs() < 1e-12);
    }

    #[test]
    fn recovering_an_adjacent_edge_is_a_no_op() {
        let (points, mut tris) = square();
        assert!(recover_edge(&mut tris, &points, 0, 1));
        assert_eq!(tris, vec![[0, 1, 2], [0, 2, 3]]);
    }

    #[test]
    fn a_segment_ending_at_a_node_does_not_cycle() {
        let (points, mut tris) = square();
        assert!(recover_edge(&mut tris, &points, 2, 3));
        assert_eq!(tris.len(), 2);
    }

    #[test]
    fn a_degenerate_request_is_refused() {
        let (points, mut tris) = square();
        assert!(!recover_edge(&mut tris, &points, 2, 2));
    }
}
