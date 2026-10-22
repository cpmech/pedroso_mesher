//! Converting a triangle mesh into quads by advancing a front (Q-Morph, Owen 1999).
//!
//! The front is the boundary between the meshed region and the unmeshed one. For
//! a front edge `(a, b)` the procedure looks for two nodes `c` (a neighbour of
//! `a`) and `d` (a neighbour of `b`) such that `(a, b, d, c)` is a convex quad
//! whose four sides are all mesh edges; the triangles inside that quad are
//! deleted and the quad takes their place. The front then moves on to `(d, c)`.
//!
//! Differences from Owen's algorithm, recorded rather than hidden: the side
//! nodes are chosen among *existing* neighbours instead of being placed along the
//! inward normal, so no node is inserted; the top edge is not recovered by swaps
//! (the caller may do that later); and there is no transition handling. What it
//! does have is the property that motivated the change: a quad is formed from the
//! whole region at once, never by pairing two leftovers.

use crate::mesh::Node;

/// A quad as four node indices, counter-clockwise.
pub type Quad = [usize; 4];

/// Convert `tris` into quads, returning the quads and the triangles that could
/// not be converted (the mesh may then be left quad-dominant).
pub fn quadify(tris: &[[usize; 3]], points: &[Node]) -> (Vec<Quad>, Vec<[usize; 3]>) {
    let mut tris: Vec<[usize; 3]> = tris.to_vec();
    let mut alive = vec![true; tris.len()];
    let mut quads: Vec<Quad> = Vec::new();

    loop {
        let front = front_edges(&tris, &alive, points);
        if front.is_empty() {
            break;
        }
        // shortest first: the front advances where the quads are smallest
        let mut order: Vec<usize> = (0..front.len()).collect();
        order.sort_by(|&i, &j| {
            let li = edge_length(points, front[i].0, front[i].1);
            let lj = edge_length(points, front[j].0, front[j].1);
            li.partial_cmp(&lj).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut progressed = false;
        'outer: for &k in &order {
            let (a, b) = (front[k].0, front[k].1);
            let Some(seed) = alive_triangle_containing(&tris, &alive, a, b) else {
                continue;
            };
            for (c, d) in candidate_corners(&tris, &alive, points, a, b) {
                let quad = [a, b, d, c];
                if !valid_quad(&quad, points) {
                    continue;
                }
                // the top edge must exist; if it does not, recover it on a copy
                let mut recovered = tris.clone();
                if !has_edge(&recovered, &alive, c, d)
                    && !crate::recovery::recover_edge(&mut recovered, points, c, d)
                {
                    continue;
                }
                let Some(region) = region_inside(&recovered, &alive, seed, &quad, points) else {
                    continue;
                };
                for &t in &region {
                    alive[t] = false;
                }
                tris = recovered;
                quads.push(quad);
                progressed = true;
                break 'outer;
            }
        }
        if !progressed {
            break;
        }
    }

    let leftover = tris
        .iter()
        .enumerate()
        .filter(|(i, _)| alive[*i])
        .map(|(_, t)| *t)
        .collect();
    (quads, leftover)
}

/// Directed edges with exactly one alive triangle on the left.
fn front_edges(tris: &[[usize; 3]], alive: &[bool], points: &[Node]) -> Vec<(usize, usize)> {
    let mut count: std::collections::HashMap<(usize, usize), i32> =
        std::collections::HashMap::new();
    let mut directed: std::collections::HashMap<(usize, usize), (usize, usize)> =
        std::collections::HashMap::new();
    for (i, t) in tris.iter().enumerate() {
        if !alive[i] {
            continue;
        }
        for k in 0..3 {
            let (u, v) = (t[k], t[(k + 1) % 3]);
            let key = (u.min(v), u.max(v));
            *count.entry(key).or_insert(0) += 1;
            // keep the direction in which the triangle lies on the left
            let third = t[(k + 2) % 3];
            if cross(points[u], points[v], points[third]) > 0.0 {
                directed.insert(key, (u, v));
            } else {
                directed.insert(key, (v, u));
            }
        }
    }
    count
        .iter()
        .filter(|(_, &n)| n == 1)
        .filter_map(|(key, _)| directed.get(key).copied())
        .collect()
}

/// Pairs `(c, d)` of existing neighbours that close a quad with `(a, b)`.
fn candidate_corners(
    tris: &[[usize; 3]],
    alive: &[bool],
    points: &[Node],
    a: usize,
    b: usize,
) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let na = neighbours(tris, alive, a);
    let nb = neighbours(tris, alive, b);
    for &c in &na {
        if c == b {
            continue;
        }
        for &d in &nb {
            if d == a || d == c {
                continue;
            }
            out.push((c, d));
        }
    }
    // prefer the pair closest to a square: longer top edge, shorter sides
    out.sort_by(|&(c1, d1), &(c2, d2)| {
        let s1 = shape_score(points, a, b, d1, c1);
        let s2 = shape_score(points, a, b, d2, c2);
        s2.partial_cmp(&s1).unwrap_or(std::cmp::Ordering::Equal)
    });
    out
}

fn shape_score(points: &[Node], a: usize, b: usize, d: usize, c: usize) -> f64 {
    let quad = [a, b, d, c];
    let mut worst = f64::INFINITY;
    for i in 0..4 {
        let prev = points[quad[(i + 3) % 4]];
        let here = points[quad[i]];
        let next = points[quad[(i + 1) % 4]];
        let u = [prev[0] - here[0], prev[1] - here[1]];
        let v = [next[0] - here[0], next[1] - here[1]];
        let denom = (u[0] * u[0] + u[1] * u[1]).sqrt() * (v[0] * v[0] + v[1] * v[1]).sqrt();
        if denom <= 0.0 {
            return -1.0;
        }
        let angle = ((u[0] * v[0] + u[1] * v[1]) / denom)
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        worst = worst.min(angle);
    }
    worst
}

fn neighbours(tris: &[[usize; 3]], alive: &[bool], v: usize) -> Vec<usize> {
    let mut out = Vec::new();
    for (i, t) in tris.iter().enumerate() {
        if !alive[i] || !t.contains(&v) {
            continue;
        }
        for &u in t {
            if u != v && !out.contains(&u) {
                out.push(u);
            }
        }
    }
    out
}

fn has_edge(tris: &[[usize; 3]], alive: &[bool], a: usize, b: usize) -> bool {
    tris.iter().enumerate().any(|(i, t)| {
        alive[i]
            && (0..3).any(|k| {
                let (u, v) = (t[k], t[(k + 1) % 3]);
                (u.min(v), u.max(v)) == (a.min(b), a.max(b))
            })
    })
}

fn alive_triangle_containing(
    tris: &[[usize; 3]],
    alive: &[bool],
    a: usize,
    b: usize,
) -> Option<usize> {
    (0..tris.len()).find(|&i| {
        alive[i]
            && (0..3).any(|k| {
                let (u, v) = (tris[i][k], tris[i][(k + 1) % 3]);
                (u.min(v), u.max(v)) == (a.min(b), a.max(b))
            })
    })
}

/// The alive triangles enclosed by the quad's four sides, if that region is
/// exactly the quad.
fn region_inside(
    tris: &[[usize; 3]],
    alive: &[bool],
    seed: usize,
    quad: &Quad,
    points: &[Node],
) -> Option<Vec<usize>> {
    let mut enclosed: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    for i in 0..4 {
        let (u, v) = (quad[i], quad[(i + 1) % 4]);
        enclosed.insert((u.min(v), u.max(v)));
    }
    let mut region = vec![seed];
    let mut seen = vec![false; tris.len()];
    seen[seed] = true;
    let mut head = 0;
    while head < region.len() {
        let t = region[head];
        head += 1;
        for k in 0..3 {
            let (u, v) = (tris[t][k], tris[t][(k + 1) % 3]);
            let key = (u.min(v), u.max(v));
            if enclosed.contains(&key) {
                continue; // a side of the quad: the boundary of the region
            }
            for (i, other) in tris.iter().enumerate() {
                if seen[i] || !alive[i] || i == t {
                    continue;
                }
                let shares = (0..3).any(|m| {
                    let (x, y) = (other[m], other[(m + 1) % 3]);
                    (x.min(y), x.max(y)) == key
                });
                if shares {
                    seen[i] = true;
                    region.push(i);
                }
            }
        }
    }
    // every vertex of the region must be inside or on the quad
    for &t in &region {
        for &v in &tris[t] {
            if !on_or_inside(quad, v, points) {
                return None;
            }
        }
    }
    Some(region)
}

fn on_or_inside(quad: &Quad, v: usize, points: &[Node]) -> bool {
    let p = points[v];
    let mut positive = 0;
    let mut negative = 0;
    for i in 0..4 {
        let a = points[quad[i]];
        let b = points[quad[(i + 1) % 4]];
        let s = cross(a, b, p);
        if s > 1e-12 {
            positive += 1;
        } else if s < -1e-12 {
            negative += 1;
        }
    }
    positive == 0 || negative == 0
}

fn valid_quad(quad: &Quad, points: &[Node]) -> bool {
    let mut area = 0.0;
    for i in 0..4 {
        let a = points[quad[i]];
        let b = points[quad[(i + 1) % 4]];
        area += a[0] * b[1] - b[0] * a[1];
    }
    area > 1e-12 && shape_score(points, quad[0], quad[1], quad[2], quad[3]) > 15.0
}

fn cross(a: Node, b: Node, c: Node) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

fn edge_length(points: &[Node], a: usize, b: usize) -> f64 {
    ((points[b][0] - points[a][0]).powi(2) + (points[b][1] - points[a][1]).powi(2)).sqrt()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn a_single_quad_triangulated_two_ways_becomes_one_quad() {
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let (quads, leftover) = quadify(&[[0, 1, 2], [0, 2, 3]], &points);
        assert_eq!(quads.len(), 1, "quads {quads:?} leftover {leftover:?}");
        assert!(leftover.is_empty());
    }

    #[test]
    fn a_square_split_into_four_quads_is_recognised() {
        // 3x3 grid of points, 8 triangles, should become 4 quads
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
        let (quads, leftover) = quadify(&tris, &points);
        assert!(leftover.is_empty(), "leftover {leftover:?}");
        assert_eq!(quads.len(), 4, "quads {quads:?}");
    }
}
