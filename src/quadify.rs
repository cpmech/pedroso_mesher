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
pub fn quadify(tris: &[[usize; 3]], points: &[Node], size: f64) -> (Vec<Quad>, Vec<[usize; 3]>) {
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
            for (c, d) in candidate_corners(&tris, &alive, points, a, b, size) {
                let quad = [a, b, d, c];
                if !valid_quad(&quad, points) {
                    continue;
                }
                // all three missing sides must exist: a-c, b-d and c-d
                let mut recovered = tris.clone();
                let mut ok = true;
                for (u, v) in [(a, c), (b, d), (c, d)] {
                    if !has_edge(&recovered, &alive, u, v)
                        && !crate::recovery::recover_edge_live(&mut recovered, points, u, v, &alive)
                    {
                        ok = false;
                        break;
                    }
                }
                if !ok {
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

/// Pairs `(c, d)` that close a quad with `(a, b)`, chosen geometrically.
///
/// Q-Morph puts the side nodes along the inward normal at the local element
/// size; taking only graph neighbours (as the first slice did) restricts the
/// quad to the current triangulation and caps its shape. Here any node ahead of
/// the front and at a plausible distance is a candidate, and the three missing
/// sides are recovered afterwards.
fn candidate_corners(
    tris: &[[usize; 3]],
    alive: &[bool],
    points: &[Node],
    a: usize,
    b: usize,
    size: f64,
) -> Vec<(usize, usize)> {
    // Only nodes reachable from the front edge through live triangles: a side
    // node across already-converted territory cannot be reached without cutting
    // through a quad, and the recovery would fail anyway.
    let near_a = live_neighbourhood(tris, alive, a, 2);
    let near_b = live_neighbourhood(tris, alive, b, 2);
    let pa = points[a];
    let pb = points[b];
    let length = edge_length(points, a, b);
    let target = if length > 1e-9 { length } else { size };
    // inward normal: rotate the front direction by +90 degrees (alive region left)
    let n = [-(pb[1] - pa[1]), (pb[0] - pa[0])];
    let nn = (n[0] * n[0] + n[1] * n[1]).sqrt();
    if nn <= 1e-12 {
        return Vec::new();
    }
    let n = [n[0] / nn, n[1] / nn];

    let ahead = |p: Node, origin: Node| -> bool {
        let d = [p[0] - origin[0], p[1] - origin[1]];
        let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
        if len < 1e-9 {
            return false;
        }
        let cos = (d[0] * n[0] + d[1] * n[1]) / len;
        cos > 0.35 && len >= 0.5 * target && len <= 1.8 * target
    };

    let mut out = Vec::new();
    for &c in &near_a {
        if c == a || c == b || !ahead(points[c], pa) {
            continue;
        }
        for &d in &near_b {
            if d == a || d == b || d == c || !ahead(points[d], pb) {
                continue;
            }
            // the sides must stay inside the live region, or the recovery has to
            // cut through a quad and will fail
            if inside_live(tris, alive, points, a, c)
                && inside_live(tris, alive, points, b, d)
                && inside_live(tris, alive, points, c, d)
            {
                out.push((c, d));
            }
        }
    }
    out.sort_by(|&(c1, d1), &(c2, d2)| {
        let s1 = shape_score(points, a, b, d1, c1);
        let s2 = shape_score(points, a, b, d2, c2);
        s2.partial_cmp(&s1).unwrap_or(std::cmp::Ordering::Equal)
    });
    out.truncate(12);
    out
}

/// True when the open segment `a-b` stays inside live triangles.
fn inside_live(tris: &[[usize; 3]], alive: &[bool], points: &[Node], a: usize, b: usize) -> bool {
    for step in 1..8 {
        let t = step as f64 / 8.0;
        let px = points[a][0] + t * (points[b][0] - points[a][0]);
        let py = points[a][1] + t * (points[b][1] - points[a][1]);
        let inside = tris.iter().enumerate().any(|(i, tri)| {
            if !alive[i] {
                return false;
            }
            let (p0, p1, p2) = (points[tri[0]], points[tri[1]], points[tri[2]]);
            let d0 = (p1[0] - p0[0]) * (py - p0[1]) - (p1[1] - p0[1]) * (px - p0[0]);
            let d1 = (p2[0] - p1[0]) * (py - p1[1]) - (p2[1] - p1[1]) * (px - p1[0]);
            let d2 = (p0[0] - p2[0]) * (py - p2[1]) - (p0[1] - p2[1]) * (px - p2[0]);
            (d0 >= -1e-12 && d1 >= -1e-12 && d2 >= -1e-12)
                || (d0 <= 1e-12 && d1 <= 1e-12 && d2 <= 1e-12)
        });
        if !inside {
            return false;
        }
    }
    true
}

/// Nodes within `depth` triangle steps of `start`, using live triangles only.
fn live_neighbourhood(
    tris: &[[usize; 3]],
    alive: &[bool],
    start: usize,
    depth: usize,
) -> Vec<usize> {
    let mut seen = std::collections::HashSet::new();
    let mut frontier = vec![start];
    seen.insert(start);
    for _ in 0..depth {
        let mut next = Vec::new();
        for &v in &frontier {
            for (i, t) in tris.iter().enumerate() {
                if !alive[i] || !t.contains(&v) {
                    continue;
                }
                for &u in t {
                    if seen.insert(u) {
                        next.push(u);
                    }
                }
            }
        }
        frontier = next;
        if frontier.is_empty() {
            break;
        }
    }
    seen.into_iter().collect()
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
    if area <= 1e-12 {
        return false;
    }
    // strictly convex: every corner turns left
    let convex = (0..4).all(|i| {
        let a = points[quad[(i + 3) % 4]];
        let b = points[quad[i]];
        let c = points[quad[(i + 1) % 4]];
        cross(a, b, c) > 0.0
    });
    if !convex {
        return false;
    }
    // real Jacobian, not just corners: a quad can have usable angles and still be
    // a sliver once the bilinear map is evaluated
    crate::two_d::quad_min_sj_public(&coords_of_quad(quad, points)) > 0.2
}

fn coords_of_quad(quad: &Quad, points: &[Node]) -> [[f64; 2]; 4] {
    [
        [points[quad[0]][0], points[quad[0]][1]],
        [points[quad[1]][0], points[quad[1]][1]],
        [points[quad[2]][0], points[quad[2]][1]],
        [points[quad[3]][0], points[quad[3]][1]],
    ]
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
        let (quads, leftover) = quadify(&[[0, 1, 2], [0, 2, 3]], &points, 1.0);
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
        let (quads, leftover) = quadify(&tris, &points, 1.0);
        assert!(leftover.is_empty(), "leftover {leftover:?}");
        assert_eq!(quads.len(), 4, "quads {quads:?}");
    }
}
