//! Boundary-loop parity control for the Q-Morph input.
//!
//! A triangulated polygon with `n` boundary edges has `n - 2` triangles, so its
//! triangle count has the same parity as `n`. An odd boundary loop therefore
//! yields an odd number of triangles, and an odd number of residuals leaves one
//! triangle that cannot be paired into a quad. Sun et al. (2016, §3.3; corpus
//! `sun-2016`) make the loop even by inserting one point on it before
//! conversion, which flips both parities at once.
//!
//! This module implements that split for straight polyline boundaries: the
//! inserted point is the **midpoint** of the longest edge of the loop, so it
//! lies exactly on the domain boundary and the two new boundary edges are
//! collinear. The triangle that owns the split edge is divided into two,
//! preserving its winding.
//!
//! **Approximated or missing.** On a curved boundary the natural point would be
//! the projection onto the true curve; this mesher only has polylines, so the
//! chord midpoint is used. The loop is capped
//! at 16 splits ([`even_boundary_loops`]) and a boundary graph that is not a set
//! of degree-two cycles (two loops touching at a vertex, a non-manifold edge)
//! is left untouched, so the function does not *guarantee* evenness for such
//! inputs.
//!
//! **Not to be confused with** a triangle's own winding or with the even-region
//! guard of [`crate::twostage`]: this is purely a boundary-count fix made before
//! Q-Morph runs.

use std::collections::HashMap;

use crate::mesh::Node;

/// Make every boundary loop of the triangulation even.
///
/// Returns the points, the triangles and the number of inserted points. A
/// triangle carrying a split edge is divided into two, preserving winding.
///
/// # Preconditions
///
/// `tris` must index `points`; the triangles may be wound either way. The mesh
/// should be a manifold triangulation of the domain: the boundary is found as
/// the set of edges with a single owner.
///
/// # Postconditions
///
/// Every closed boundary loop that could be found has an even number of edges.
/// The new points are appended and have `z = 0`; existing points and triangle
/// indices are unchanged. Every input triangle is still covered (the split one
/// becomes two), and the winding of both replacements is the input's. `splits`
/// is the number of inserted points, not the number of loops examined.
///
/// # Limits
///
/// At most 16 edges are split; a domain with more than 16 odd loops is only
/// partially normalised. A loop is skipped when `boundary_loops` cannot form a
/// single cycle (a vertex of degree other than two); the function then returns
/// without changing that loop rather than guessing. These are documented
/// limitations, not recoverable errors, because the function cannot report a
/// failure through its tuple return.
pub fn even_boundary_loops(
    points: &[Node],
    tris: &[[usize; 3]],
) -> (Vec<Node>, Vec<[usize; 3]>, usize) {
    let mut points = points.to_vec();
    let mut tris = tris.to_vec();
    let mut splits = 0usize;
    // One split per iteration, then the loops are rebuilt because the split
    // changes which loop contains the edge. 16 is a hard cap: more than 16 odd
    // loops is pathological for a planar domain, and an unbounded loop would
    // hide an input bug. The cap is a documented limitation (see the function
    // docs), not a tuning parameter.
    for _ in 0..16 {
        let loops = boundary_loops(&tris);
        let Some(odd) = loops.iter().find(|edges| edges.len() % 2 == 1) else {
            break;
        };
        // Split the longest edge: the midpoint is least likely to create a very
        // short boundary edge on a polyline whose segments vary in length, and
        // any edge of the loop is equally valid for parity.
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
        // Split edge a-b at its midpoint. The two replacements keep the winding:
        // [a, point, c] and [point, b, c] are both in the cyclic order of the
        // original [a, b, c] (or the reverse if the input was clockwise).
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

/// The boundary loops of a triangle mesh, as sorted edge lists.
///
/// Builds the edge-owner map, keeps the edges with exactly one owner (the
/// boundary), then walks each connected component through vertex adjacency. The
/// walk needs every boundary vertex to have degree two; if any has another
/// degree (two loops touching at a point, a non-manifold edge, an open chain),
/// the whole function returns an empty vector. An empty vector therefore means
/// "cannot form simple loops", not "no boundary".
///
/// The walk emits an edge list per loop, in the order it traversed them; the
/// edge endpoints are sorted so the loop's direction is not stored.
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

/// Euclidean length of a boundary edge, in the XY plane.
fn length(points: &[Node], edge: (usize, usize)) -> f64 {
    let a = points[edge.0];
    let b = points[edge.1];
    ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt()
}

/// Midpoint of two nodes, in the XY plane.
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
