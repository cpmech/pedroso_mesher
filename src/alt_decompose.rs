//! Domain decomposition + templates: an all-quadrilateral route.
//!
//! The method follows Yang et al. (2024), *Boundary constrained quad mesh
//! generation based on domain decomposition and templates*, and Miranda &
//! Martha (2017), *Hierarchical template-based quadrilateral mesh generation*:
//!
//! 1. the boundary loops are discretised once, at the requested size, and the
//!    discretisation is shared by every patch (this is what makes the result
//!    conforming);
//! 2. the domain is recursively bisected by *chords* between boundary nodes
//!    until every subregion has exactly four corners. Candidate chords are the
//!    reflex-vertex ray casts (Tam & Armstrong's medial-axis idea, as used by
//!    Yang et al. §3.2/§3.3), diagonals between corners and side midpoints; the
//!    one that removes a reflex vertex and balances the two areas is taken;
//! 3. a four-corner patch is meshed by a template that depends only on the
//!    number of segments on its four sides: the plain tensor grid (T0), the
//!    4-block transition of Miranda's T1 (one pair of opposite sides equal) or
//!    T2 (both pairs unequal, which recurses into T1).  The templates are
//!    applied in a discrete transfinite (Coons) parameterisation of the patch,
//!    so all interior nodes are computed by the mapping and all boundary nodes
//!    are the shared ones;
//! 4. patch parity is fixed on the chords: an even number of boundary segments
//!    per patch is a necessary condition for a pure quad mesh.  The chord
//!    parities solve a GF(2) system on the patch-adjacency tree, and the loop
//!    totals are made even before the boundary is generated;
//! 5. a domain with one hole is meshed as an O-grid of eight patches, one per
//!    half outer side, with the hole split radially to those eight points;
//! 6. the assembled mesh is relaxed by forty interior Laplacian sweeps and a
//!    local interior-node polish (`sun::perturb_low_quads`, boundary nodes
//!    fixed), which removes the small shear that the transition templates
//!    leave inside the O-grid block corners.
//!
//! Implemented: boundary discretisation, recursive domain decomposition,
//! transfinite mapping, T0/T1/T2 templates, one-hole O-grid, parity fixing.
//! Not implemented: triangular/pentagonal template families (a triangular
//! *domain* is refused), several holes, and curved (non-planar) patches.  A
//! subregion whose four corners are not convex is passed to the templates and
//! may fold; such a mesh is returned, and the gate is the judge.
//!
//! Measured limit: the eight-block O-grid of a rectangle with a coarse
//! polygonal hole still has its worst cells in the first ring outside the
//! hole at the two extreme side points, where a radial chord meets a short
//! hole arc.  The mesh is all-quad, watertight and conforming with a high
//! share of valence-4 nodes and a small aspect ratio; the 4 mm and 2 mm plate
//! rows clear the gate, while the 1 mm and performance rows stay at about
//! 28-30° there.

use std::collections::{HashMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::mesh::{Cell, Kind, Mesh, Node};
use crate::two_d::{Domain, Sizing};

/// A vertex is a corner when its boundary turns by more than this angle.
const CORNER_DEG: f64 = 30.0;
/// Safety limit on the recursion depth of the templates.
const MAX_DEPTH: usize = 64;
/// Safety limit on the number of patches.
const MAX_PATCHES: usize = 4096;
/// Safety limit on any discretisation count.
const MAX_COUNT: usize = 2_000_000;

/// Options for the decomposition + templates route.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Options {
    /// A vertex is a corner when its boundary turns by more than this angle
    /// (degrees).
    pub corner_deg: f64,
    /// Safety limit on the recursion depth of the templates.
    pub max_depth: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            corner_deg: CORNER_DEG,
            max_depth: MAX_DEPTH,
        }
    }
}

/// Mesh a planar domain with the decomposition + templates route.
pub fn mesh(domain: &Domain, sizing: &Sizing) -> Result<Mesh, String> {
    mesh_with_options(domain, sizing, &Options::default())
}

/// Mesh a planar domain with the decomposition + templates route, using
/// `options` for the tunable constants.
pub fn mesh_with_options(
    domain: &Domain,
    sizing: &Sizing,
    options: &Options,
) -> Result<Mesh, String> {
    if !sizing.size.is_finite() || sizing.size <= 0.0 {
        return Err("the element size must be positive and finite".to_string());
    }
    let size = sizing.size;
    if domain.outer.len() < 3 {
        return Err("the outer loop needs at least three points".to_string());
    }
    let outer = normalize_loop(&domain.outer, true);
    let holes: Vec<Vec<[f64; 2]>> = domain
        .holes
        .iter()
        .map(|h| normalize_loop(h, false))
        .collect();
    let mesh = match holes.len() {
        0 => mesh_simply_connected(&outer, size, options)?,
        1 => mesh_one_hole(&outer, &holes[0], size, options)?,
        n => {
            return Err(format!(
                "not supported yet: {n} holes (the O-grid handles one hole)"
            ))
        }
    };
    let mesh = relax(mesh, 40);
    Ok(polish(mesh, size))
}

/// Local interior-node polish.
///
/// The four-point Laplacian is blind to the corner Jacobian, so a few nodes in
/// the first ring outside the hole still sit at a 29-30° corner.  The library's
/// [`crate::sun::perturb_low_quads`] moves only the nodes of quads that are
/// below the gate margin; passing `None` for the domain keeps every boundary
/// node fixed, so the geometry is untouched.  On the perfect tensor blocks the
/// margin is already above one and no node moves.
fn polish(mesh: Mesh, size: f64) -> Mesh {
    let mut quads: Vec<[usize; 4]> = Vec::new();
    let mut tris: Vec<[usize; 3]> = Vec::new();
    for cell in &mesh.cells {
        if cell.kind == Kind::Quad4 && cell.nodes.len() == 4 {
            quads.push([cell.nodes[0], cell.nodes[1], cell.nodes[2], cell.nodes[3]]);
        } else if cell.kind == Kind::Tri3 && cell.nodes.len() == 3 {
            tris.push([cell.nodes[0], cell.nodes[1], cell.nodes[2]]);
        }
    }
    let (points, quads, tris, _) =
        crate::sun::perturb_low_quads(&mesh.points, &quads, &tris, size, 200, None);
    let mut cells = Vec::with_capacity(quads.len() + tris.len());
    for q in &quads {
        cells.push(Cell::quad4(*q));
    }
    for t in &tris {
        cells.push(Cell::tri3(*t));
    }
    Mesh::new(points, cells)
}

/// Interior Laplacian relaxation of the assembled patches.
///
/// The transition templates are exact on the boundary but leave a little shear
/// in the first rings inside the corners of an O-grid block.  Forty sweeps of
/// the plain four-point Laplacian on the **interior** nodes remove it; boundary
/// nodes are never moved, so the mesh stays on the geometry (the boundary
/// offset is unchanged).
fn relax(mut mesh: Mesh, sweeps: usize) -> Mesh {
    use std::collections::HashMap;
    let n = mesh.points.len();
    let mut edge_use: HashMap<(usize, usize), usize> = HashMap::new();
    for cell in &mesh.cells {
        for i in 0..cell.nodes.len() {
            let a = cell.nodes[i];
            let b = cell.nodes[(i + 1) % cell.nodes.len()];
            *edge_use.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    let mut boundary = vec![false; n];
    for ((a, b), uses) in edge_use {
        if uses == 1 {
            boundary[a] = true;
            boundary[b] = true;
        }
    }
    let mut incident: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (ci, cell) in mesh.cells.iter().enumerate() {
        for &v in &cell.nodes {
            incident[v].push(ci);
        }
    }
    for _ in 0..sweeps {
        let old = mesh.points.clone();
        for v in 0..n {
            if boundary[v] || incident[v].is_empty() {
                continue;
            }
            let mut ring: Vec<usize> = Vec::new();
            for &ci in &incident[v] {
                for &u in &mesh.cells[ci].nodes {
                    if u != v && !ring.contains(&u) {
                        ring.push(u);
                    }
                }
            }
            if ring.len() < 3 {
                continue;
            }
            let mut x = 0.0;
            let mut y = 0.0;
            for &u in &ring {
                x += old[u][0];
                y += old[u][1];
            }
            mesh.points[v] = [x / ring.len() as f64, y / ring.len() as f64, 0.0];
        }
    }
    mesh
}

// ---------------------------------------------------------------------------
// small geometry helpers
// ---------------------------------------------------------------------------

fn signed_area(points: &[[f64; 2]]) -> f64 {
    let mut sum = 0.0;
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        sum += a[0] * b[1] - b[0] * a[1];
    }
    0.5 * sum
}

/// Make the outer loop counter-clockwise and a hole loop clockwise.
fn normalize_loop(points: &[[f64; 2]], counter_clockwise: bool) -> Vec<[f64; 2]> {
    let mut p = points.to_vec();
    let area = signed_area(&p);
    if (counter_clockwise && area < 0.0) || (!counter_clockwise && area > 0.0) {
        p.reverse();
    }
    p
}

fn cross(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn norm(a: [f64; 2]) -> f64 {
    (a[0] * a[0] + a[1] * a[1]).sqrt()
}

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    norm(sub(a, b))
}

fn lerp(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
}

/// Signed turn (degrees) of the polyline `a -> b -> c`, positive to the left.
fn turn_deg(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    let u = sub(b, a);
    let v = sub(c, b);
    if norm(u) < 1.0e-14 || norm(v) < 1.0e-14 {
        return 0.0;
    }
    cross(u, v).atan2(u[0] * v[0] + u[1] * v[1]).to_degrees()
}

/// Indices of the input vertices where the boundary turns sharply.
fn detect_corners(points: &[[f64; 2]], options: &Options) -> Vec<usize> {
    let n = points.len();
    let mut out = Vec::new();
    for i in 0..n {
        let a = points[(i + n - 1) % n];
        let b = points[i];
        let c = points[(i + 1) % n];
        if turn_deg(a, b, c).abs() > options.corner_deg {
            out.push(i);
        }
    }
    out
}

/// Four vertices roughly a quarter of the perimeter apart (for a smooth loop).
fn pseudo_corners(points: &[[f64; 2]]) -> Vec<usize> {
    let n = points.len();
    if n <= 4 {
        return (0..n).collect();
    }
    let mut lengths = Vec::with_capacity(n);
    let mut total = 0.0;
    for i in 0..n {
        let l = dist(points[i], points[(i + 1) % n]);
        lengths.push(l);
        total += l;
    }
    let mut out = Vec::with_capacity(4);
    for k in 0..4 {
        let target = total * k as f64 / 4.0;
        let mut acc = 0.0;
        let mut idx = 0;
        for (i, &l) in lengths.iter().enumerate() {
            if acc + l >= target {
                let t = if l > 0.0 { (target - acc) / l } else { 0.0 };
                idx = if t < 0.5 { i } else { (i + 1) % n };
                break;
            }
            acc += l;
        }
        if !out.contains(&idx) {
            out.push(idx);
        }
    }
    if out.len() != 4 {
        // Degenerate/regular sampling: fall back to even index spacing.
        out = (0..4).map(|k| k * n / 4).collect();
        out.sort_unstable();
        out.dedup();
        while out.len() < 4 {
            let next = (out.len() * n / 4).min(n - 1);
            if !out.contains(&next) {
                out.push(next);
            } else {
                for i in 0..n {
                    if !out.contains(&i) {
                        out.push(i);
                        break;
                    }
                }
            }
        }
        out.sort_unstable();
    }
    out
}

/// Polyline from `points[start]` to `points[end]` walking forward, wrapping.
fn chain_points(points: &[[f64; 2]], start: usize, end: usize) -> Vec<[f64; 2]> {
    let n = points.len();
    if start == end {
        let mut v = points.to_vec();
        v.push(points[start]);
        return v;
    }
    let mut out = Vec::new();
    let mut i = start;
    out.push(points[i]);
    while i != end {
        i = (i + 1) % n;
        out.push(points[i]);
    }
    out
}

fn count_for(length: f64, size: f64) -> usize {
    let raw = length / size;
    if !raw.is_finite() {
        return 1;
    }
    let n = raw.round().max(1.0).min(MAX_COUNT as f64);
    n as usize
}

/// Split a polyline into `count` equal-length pieces; returns `count + 1`
/// points, the first and last being the polyline ends.
fn distribute(poly: &[[f64; 2]], count: usize) -> Result<Vec<[f64; 2]>, String> {
    if poly.len() < 2 {
        return Err("a boundary chain has fewer than two points".to_string());
    }
    let mut cumulative = Vec::with_capacity(poly.len());
    let mut total = 0.0;
    cumulative.push(0.0);
    for i in 1..poly.len() {
        total += dist(poly[i - 1], poly[i]);
        cumulative.push(total);
    }
    if !total.is_finite() || total <= 1.0e-12 {
        return Err("a boundary chain has zero length".to_string());
    }
    let mut out = Vec::with_capacity(count + 1);
    let mut seg = 0usize;
    for k in 0..=count {
        let target = total * k as f64 / count as f64;
        while seg + 2 < poly.len() && cumulative[seg + 1] < target {
            seg += 1;
        }
        let l = cumulative[seg + 1] - cumulative[seg];
        let t = if l > 0.0 {
            ((target - cumulative[seg]) / l).clamp(0.0, 1.0)
        } else {
            0.0
        };
        out.push(lerp(poly[seg], poly[seg + 1], t));
    }
    let last = out.len() - 1;
    out[0] = poly[0];
    out[last] = poly[poly.len() - 1];
    Ok(out)
}

fn point_key(p: [f64; 2]) -> (i64, i64) {
    ((p[0] * 1.0e9).round() as i64, (p[1] * 1.0e9).round() as i64)
}

// ---------------------------------------------------------------------------
// the discrete boundary and the region decomposition
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Chord {
    a: usize,
    b: usize,
    count: usize,
}

/// Boundary nodes and chords.  Only boundary nodes appear in the region
/// cycles; chord edges are single straight edges between boundary nodes, so a
/// later refinement of the chord never changes the topology of the
/// decomposition.
struct Decomp {
    bpts: Vec<[f64; 2]>,
    bmap: HashMap<(i64, i64), usize>,
    bnext: HashMap<usize, usize>,
    chords: Vec<Chord>,
    chord_of: HashMap<(usize, usize), usize>,
    scale: f64,
}

impl Decomp {
    fn new(scale: f64) -> Self {
        Decomp {
            bpts: Vec::new(),
            bmap: HashMap::new(),
            bnext: HashMap::new(),
            chords: Vec::new(),
            chord_of: HashMap::new(),
            scale: scale.max(1.0),
        }
    }

    fn add_bpt(&mut self, p: [f64; 2]) -> usize {
        let key = point_key(p);
        if let Some(&id) = self.bmap.get(&key) {
            return id;
        }
        let id = self.bpts.len();
        self.bpts.push(p);
        self.bmap.insert(key, id);
        id
    }

    fn p(&self, id: usize) -> [f64; 2] {
        self.bpts[id]
    }

    fn eps(&self) -> f64 {
        1.0e-10 * self.scale
    }

    fn is_boundary_edge(&self, u: usize, v: usize) -> bool {
        self.bnext.get(&u) == Some(&v) || self.bnext.get(&v) == Some(&u)
    }

    fn link(&mut self, u: usize, w: usize) {
        self.bnext.insert(u, w);
    }

    /// Insert `p` between the boundary nodes `u` and `v`.
    fn insert_boundary(&mut self, u: usize, v: usize, p: [f64; 2]) -> usize {
        let id = self.add_bpt(p);
        if self.bnext.get(&u) == Some(&v) {
            let next = v;
            self.link(u, id);
            self.link(id, next);
        } else if self.bnext.get(&v) == Some(&u) {
            let next = u;
            self.link(v, id);
            self.link(id, next);
        } else {
            // Not a boundary edge any more (should not happen); leave it
            // dangling rather than corrupting the loop.
            return id;
        }
        id
    }

    fn set_cycle(&mut self, cycle: &[usize]) {
        let n = cycle.len();
        for i in 0..n {
            let u = cycle[i];
            let v = cycle[(i + 1) % n];
            self.bnext.insert(u, v);
        }
    }

    /// Signed turn at every position of a region cycle, in degrees.
    fn turns(&self, cycle: &[usize]) -> Vec<f64> {
        let n = cycle.len();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let a = self.p(cycle[(i + n - 1) % n]);
            let b = self.p(cycle[i]);
            let c = self.p(cycle[(i + 1) % n]);
            out.push(turn_deg(a, b, c));
        }
        out
    }

    fn corners(&self, cycle: &[usize], options: &Options) -> Vec<(usize, f64)> {
        self.turns(cycle)
            .into_iter()
            .enumerate()
            .filter(|(_, t)| t.abs() > options.corner_deg)
            .collect()
    }

    /// First boundary edge hit by the ray `p + t*dir`, `t > 0`.
    fn cast_ray(&self, cycle: &[usize], start: usize, dir: [f64; 2]) -> Option<(usize, f64)> {
        let n = cycle.len();
        let p = self.p(cycle[start]);
        let dir_len = norm(dir);
        if dir_len < 1.0e-14 {
            return None;
        }
        let mut best: Option<(usize, f64)> = None;
        let mut best_t = f64::INFINITY;
        for e in 0..n {
            if e == start || (e + 1) % n == start {
                continue;
            }
            let a = self.p(cycle[e]);
            let b = self.p(cycle[(e + 1) % n]);
            let seg = sub(b, a);
            let den = cross(seg, dir);
            if den.abs() <= 1.0e-14 * dir_len * norm(seg) {
                continue;
            }
            let pa = sub(p, a);
            let t = cross(pa, seg) / den;
            let s = cross(pa, dir) / den;
            if t > self.eps() && (-1.0e-9..=1.0 + 1.0e-9).contains(&s) && t < best_t {
                best_t = t;
                best = Some((e, s.clamp(0.0, 1.0)));
            }
        }
        best
    }
}

fn cycle_points(d: &Decomp, cycle: &[usize]) -> Vec<[f64; 2]> {
    cycle.iter().map(|&id| d.p(id)).collect()
}

fn point_in_polygon(p: [f64; 2], poly: &[[f64; 2]]) -> bool {
    let mut inside = false;
    let n = poly.len();
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        if (a[1] > p[1]) != (b[1] > p[1]) {
            let t = (p[1] - a[1]) / (b[1] - a[1]);
            if p[0] < a[0] + t * (b[0] - a[0]) {
                inside = !inside;
            }
        }
    }
    inside
}

fn orientation(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    cross(sub(b, a), sub(c, a))
}

fn segments_cross(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2], eps: f64) -> bool {
    let o1 = orientation(a, b, c);
    let o2 = orientation(a, b, d);
    let o3 = orientation(c, d, a);
    let o4 = orientation(c, d, b);
    ((o1 > eps && o2 < -eps) || (o1 < -eps && o2 > eps))
        && ((o3 > eps && o4 < -eps) || (o3 < -eps && o4 > eps))
}

fn point_on_segment_interior(p: [f64; 2], a: [f64; 2], b: [f64; 2], eps: f64) -> bool {
    let ab = sub(b, a);
    let len = norm(ab);
    if len < eps {
        return false;
    }
    let ap = sub(p, a);
    let cr = cross(ab, ap);
    if cr.abs() > eps * len {
        return false;
    }
    let t = (ap[0] * ab[0] + ap[1] * ab[1]) / (len * len);
    t > 1.0e-9 && t < 1.0 - 1.0e-9
}

/// True when the straight segment between `cycle[ip]` and `cycle[iq]` lies in
/// the closed region.
fn chord_inside(d: &Decomp, cycle: &[usize], ip: usize, iq: usize) -> bool {
    let n = cycle.len();
    let a = d.p(cycle[ip]);
    let b = d.p(cycle[iq]);
    let len = dist(a, b);
    if len <= 1.0e-12 * d.scale {
        return false;
    }
    let eps = 1.0e-12 * d.scale;
    let poly = cycle_points(d, cycle);
    for e in 0..n {
        let f = (e + 1) % n;
        if e == ip || e == iq || f == ip || f == iq {
            continue;
        }
        let c = d.p(cycle[e]);
        let dd = d.p(cycle[f]);
        if segments_cross(a, b, c, dd, eps) {
            return false;
        }
        if point_on_segment_interior(c, a, b, eps) || point_on_segment_interior(dd, a, b, eps) {
            return false;
        }
        // An endpoint of the chord must not fall in the middle of another edge.
        if point_on_segment_interior(a, c, dd, eps) || point_on_segment_interior(b, c, dd, eps) {
            // Collinear boundary edges at the chord's start/end are allowed
            // when the projections touch only at the endpoint; the interior
            // test above already excludes a genuine overlap.
            if dist(a, c) > eps && dist(a, dd) > eps && dist(b, c) > eps && dist(b, dd) > eps {
                return false;
            }
        }
    }
    let mid = lerp(a, b, 0.5);
    point_in_polygon(mid, &poly)
}

fn polygon_area(d: &Decomp, cycle: &[usize]) -> f64 {
    let poly = cycle_points(d, cycle);
    signed_area(&poly).abs()
}

// ---------------------------------------------------------------------------
// templates
// ---------------------------------------------------------------------------

struct Builder {
    points: Vec<Node>,
    map: HashMap<(i64, i64), usize>,
    cells: Vec<Cell>,
}

impl Builder {
    fn with_boundary(bpts: &[[f64; 2]]) -> Self {
        let mut b = Builder {
            points: Vec::with_capacity(bpts.len()),
            map: HashMap::new(),
            cells: Vec::new(),
        };
        for &p in bpts {
            let id = b.points.len();
            b.points.push([p[0], p[1], 0.0]);
            b.map.entry(point_key(p)).or_insert(id);
        }
        b
    }

    fn add(&mut self, p: [f64; 2]) -> usize {
        let key = point_key(p);
        if let Some(&id) = self.map.get(&key) {
            return id;
        }
        let id = self.points.len();
        self.points.push([p[0], p[1], 0.0]);
        self.map.insert(key, id);
        id
    }

    fn coord(&self, id: usize) -> [f64; 2] {
        [self.points[id][0], self.points[id][1]]
    }

    fn quad(&mut self, q: [usize; 4]) {
        if q[0] != q[1] && q[1] != q[2] && q[2] != q[3] && q[3] != q[0] {
            self.cells.push(Cell::quad4(q));
        }
    }
}

fn rev(v: &[usize]) -> Vec<usize> {
    v.iter().rev().copied().collect()
}

fn poly_at(b: &Builder, nodes: &[usize], t: f64) -> [f64; 2] {
    let n = nodes.len().saturating_sub(1);
    if n == 0 {
        return b.coord(nodes[0]);
    }
    let x = t.clamp(0.0, 1.0) * n as f64;
    let mut i = x.floor() as usize;
    if i >= n {
        i = n - 1;
    }
    let f = x - i as f64;
    let p = b.coord(nodes[i]);
    let q = b.coord(nodes[i + 1]);
    lerp(p, q, f)
}

/// Discrete transfinite (Coons) map of a four-sided patch.
///
/// The sides are in counter-clockwise order `S0: P0->P1`, `S1: P1->P2`,
/// `S2: P2->P3`, `S3: P3->P0`.
fn coons(b: &Builder, sides: &[Vec<usize>; 4], u: f64, v: f64) -> [f64; 2] {
    let s0 = poly_at(b, &sides[0], u);
    let s1 = poly_at(b, &sides[1], v);
    let s2 = poly_at(b, &sides[2], 1.0 - u);
    let s3 = poly_at(b, &sides[3], 1.0 - v);
    let p0 = b.coord(sides[0][0]);
    let p1 = b.coord(sides[1][0]);
    let p2 = b.coord(sides[2][0]);
    let p3 = b.coord(sides[3][0]);
    let x = (1.0 - v) * s0[0] + v * s2[0] + (1.0 - u) * s3[0] + u * s1[0]
        - ((1.0 - u) * (1.0 - v) * p0[0]
            + u * (1.0 - v) * p1[0]
            + u * v * p2[0]
            + (1.0 - u) * v * p3[0]);
    let y = (1.0 - v) * s0[1] + v * s2[1] + (1.0 - u) * s3[1] + u * s1[1]
        - ((1.0 - u) * (1.0 - v) * p0[1]
            + u * (1.0 - v) * p1[1]
            + u * v * p2[1]
            + (1.0 - u) * v * p3[1]);
    [x, y]
}

fn add_line(b: &mut Builder, pts: impl Iterator<Item = [f64; 2]>) -> Vec<usize> {
    pts.map(|p| b.add(p)).collect()
}

fn mesh_patch(
    b: &mut Builder,
    sides: [Vec<usize>; 4],
    depth: usize,
    options: &Options,
) -> Result<(), String> {
    if depth > options.max_depth {
        return Err("template recursion is too deep".to_string());
    }
    let counts = [
        sides[0].len().saturating_sub(1),
        sides[1].len().saturating_sub(1),
        sides[2].len().saturating_sub(1),
        sides[3].len().saturating_sub(1),
    ];
    if counts.contains(&0) {
        return Err("a patch side has no segments".to_string());
    }
    let total = counts.iter().sum::<usize>();
    if !total.is_multiple_of(2) {
        return Err("a patch has an odd number of boundary segments".to_string());
    }
    if counts[0] == counts[2] && counts[1] == counts[3] {
        return mesh_t0(b, sides);
    }
    // T1 needs one pair of opposite sides equal; rotate to right == left and
    // bottom >= top.
    if counts[1] == counts[3] {
        let mut s = sides;
        if counts[0] < counts[2] {
            s = rotate(&s, 2);
        }
        return mesh_t1(b, s);
    }
    if counts[0] == counts[2] {
        let mut s = rotate(&sides, 1);
        // counts after the rotation: bottom = n1, right = n2, top = n3,
        // left = n0; right == left because n2 == n0.
        let c = [
            s[0].len() - 1,
            s[1].len() - 1,
            s[2].len() - 1,
            s[3].len() - 1,
        ];
        if c[0] < c[2] {
            s = rotate(&s, 2);
        }
        return mesh_t1(b, s);
    }
    let c = [
        sides[0].len() - 1,
        sides[1].len() - 1,
        sides[2].len() - 1,
        sides[3].len() - 1,
    ];
    let k = (0..4)
        .find(|&k| c[k] >= c[(k + 2) % 4] && c[(k + 1) % 4] >= c[(k + 3) % 4])
        .ok_or_else(|| "cannot orient a four-sided patch".to_string())?;
    let s = rotate(&sides, k);
    mesh_t2(b, s, depth + 1, options)
}

fn rotate(sides: &[Vec<usize>; 4], k: usize) -> [Vec<usize>; 4] {
    let mut out: [Vec<usize>; 4] = Default::default();
    for i in 0..4 {
        out[i] = sides[(i + k) % 4].clone();
    }
    out
}

/// Plain tensor grid; requires equal opposite counts.
fn mesh_t0(b: &mut Builder, sides: [Vec<usize>; 4]) -> Result<(), String> {
    let n0 = sides[0].len() - 1;
    let n1 = sides[1].len() - 1;
    let n2 = sides[2].len() - 1;
    let n3 = sides[3].len() - 1;
    if n0 != n2 || n1 != n3 {
        return Err("the tensor template needs equal opposite sides".to_string());
    }
    if sides[0][n0] != sides[1][0]
        || sides[1][n1] != sides[2][0]
        || sides[2][n2] != sides[3][0]
        || sides[3][n3] != sides[0][0]
    {
        return Err("the four sides of a patch do not meet".to_string());
    }
    let mut grid = vec![vec![0usize; n1 + 1]; n0 + 1];
    for i in 0..=n0 {
        grid[i][0] = sides[0][i];
    }
    for j in 0..=n1 {
        grid[n0][j] = sides[1][j];
    }
    for i in 0..=n0 {
        grid[i][n1] = sides[2][n2 - i];
    }
    for j in 0..=n1 {
        grid[0][j] = sides[3][n3 - j];
    }
    for i in 1..n0 {
        for j in 1..n1 {
            let p = coons(b, &sides, i as f64 / n0 as f64, j as f64 / n1 as f64);
            grid[i][j] = b.add(p);
        }
    }
    for i in 0..n0 {
        for j in 0..n1 {
            b.quad([
                grid[i][j],
                grid[i + 1][j],
                grid[i + 1][j + 1],
                grid[i][j + 1],
            ]);
        }
    }
    Ok(())
}

/// Miranda's T1: `right == left`, `bottom > top` and the difference is even.
fn mesh_t1(b: &mut Builder, sides: [Vec<usize>; 4]) -> Result<(), String> {
    let n0 = sides[0].len() - 1;
    let n1 = sides[1].len() - 1;
    let n2 = sides[2].len() - 1;
    let n3 = sides[3].len() - 1;
    if n1 != n3 {
        return Err("T1 needs equal left and right side counts".to_string());
    }
    if n0 <= n2 || !(n0 - n2).is_multiple_of(2) {
        return Err("T1 needs an even, positive bottom-top difference".to_string());
    }
    let band = (n0 - n2) / 2;
    let u1 = band as f64 / n0 as f64;
    let u2 = (band + n2) as f64 / n0 as f64;
    let v = (n1 as f64 / (n1 + band) as f64).clamp(0.02, 0.98);

    // Internal split lines, computed with the patch's own transfinite map.
    let horizontal_pts: Vec<[f64; 2]> = (0..=n2)
        .map(|k| coons(b, &sides, u1 + (u2 - u1) * k as f64 / n2 as f64, v))
        .collect();
    let horizontal = add_line(b, horizontal_pts.into_iter());
    let slant_left_pts: Vec<[f64; 2]> = (0..=band)
        .map(|k| {
            let t = k as f64 / band as f64;
            coons(b, &sides, u1 * t, 1.0 - (1.0 - v) * t)
        })
        .collect();
    let slant_left = add_line(b, slant_left_pts.into_iter());
    let slant_right_pts: Vec<[f64; 2]> = (0..=band)
        .map(|k| {
            let t = k as f64 / band as f64;
            coons(b, &sides, u2 + (1.0 - u2) * t, v + (1.0 - v) * t)
        })
        .collect();
    let slant_right = add_line(b, slant_right_pts.into_iter());
    let vertical1_pts: Vec<[f64; 2]> = (0..=n1)
        .map(|k| coons(b, &sides, u1, v * k as f64 / n1 as f64))
        .collect();
    let vertical1 = add_line(b, vertical1_pts.into_iter());
    let vertical2_pts: Vec<[f64; 2]> = (0..=n1)
        .map(|k| coons(b, &sides, u2, v * k as f64 / n1 as f64))
        .collect();
    let vertical2 = add_line(b, vertical2_pts.into_iter());

    let r1 = [
        horizontal.clone(),
        slant_right.clone(),
        sides[2].clone(),
        slant_left.clone(),
    ];
    let r2 = [
        sides[0][0..=band].to_vec(),
        vertical1.clone(),
        rev(&slant_left),
        sides[3].clone(),
    ];
    let r3 = [
        sides[0][band..=band + n2].to_vec(),
        vertical2.clone(),
        rev(&horizontal),
        rev(&vertical1),
    ];
    let r4 = [
        sides[0][n0 - band..=n0].to_vec(),
        sides[1].clone(),
        rev(&slant_right),
        rev(&vertical2),
    ];
    mesh_t0(b, r1)?;
    mesh_t0(b, r2)?;
    mesh_t0(b, r3)?;
    mesh_t0(b, r4)?;
    Ok(())
}

/// Miranda's T2: both pairs of opposite sides differ.  Three subregions, one
/// of which (R3) recurses into T0 or T1.
fn mesh_t2(
    b: &mut Builder,
    sides: [Vec<usize>; 4],
    depth: usize,
    options: &Options,
) -> Result<(), String> {
    if depth > options.max_depth {
        return Err("template recursion is too deep".to_string());
    }
    let n0 = sides[0].len() - 1;
    let n1 = sides[1].len() - 1;
    let n2 = sides[2].len() - 1;
    let n3 = sides[3].len() - 1;
    if n0 <= n2 || n1 <= n3 {
        return Err("T2 needs bottom > top and right > left".to_string());
    }
    let band = (n0 - n2).min(n1 - n3);
    if band == 0 {
        return Err("T2 with a zero-width band".to_string());
    }
    let u1 = band as f64 / n0 as f64;
    let v = (n1 - band) as f64 / n1 as f64;
    let right_node = n1 - band;

    let horizontal_pts: Vec<[f64; 2]> = (0..=n2)
        .map(|k| {
            let t = k as f64 / n2 as f64;
            coons(b, &sides, u1 + (1.0 - u1) * t, v)
        })
        .collect();
    let mut horizontal = add_line(b, horizontal_pts.into_iter());
    if let Some(last) = horizontal.last_mut() {
        *last = sides[1][right_node];
    }
    let vertical1_pts: Vec<[f64; 2]> = (0..=n3)
        .map(|k| coons(b, &sides, u1, v * k as f64 / n3 as f64))
        .collect();
    let vertical1 = add_line(b, vertical1_pts.into_iter());
    let slant_pts: Vec<[f64; 2]> = (0..=band)
        .map(|k| {
            let t = k as f64 / band as f64;
            coons(b, &sides, u1 * t, 1.0 - (1.0 - v) * t)
        })
        .collect();
    let slant = add_line(b, slant_pts.into_iter());

    let r1 = [
        horizontal.clone(),
        sides[1][right_node..=n1].to_vec(),
        sides[2].clone(),
        slant.clone(),
    ];
    let r2 = [
        sides[0][0..=band].to_vec(),
        vertical1.clone(),
        rev(&slant),
        sides[3].clone(),
    ];
    let r3 = [
        sides[0][band..=n0].to_vec(),
        sides[1][0..=right_node].to_vec(),
        rev(&horizontal),
        rev(&vertical1),
    ];
    mesh_t0(b, r1)?;
    mesh_t0(b, r2)?;
    mesh_patch(b, r3, depth + 1, options)
}

// ---------------------------------------------------------------------------
// patch construction
// ---------------------------------------------------------------------------

/// Walk a region cycle from `start` to `end` (inclusive, wrapping).
fn walk(cycle: &[usize], start: usize, end: usize) -> Vec<usize> {
    let n = cycle.len();
    let mut out = vec![cycle[start]];
    let mut i = start;
    if start == end {
        return out;
    }
    loop {
        i = (i + 1) % n;
        out.push(cycle[i]);
        if i == end {
            break;
        }
    }
    out
}

/// A patch ready to be meshed: four walks of boundary/chord edges.
struct PatchSides {
    walks: [Vec<usize>; 4],
}

fn patch_sides(d: &Decomp, cycle: &[usize], options: &Options) -> Result<PatchSides, String> {
    let corners = d.corners(cycle, options);
    if corners.len() != 4 {
        return Err(format!(
            "a decomposed patch has {} corners instead of 4",
            corners.len()
        ));
    }
    let pos: Vec<usize> = corners.iter().map(|&(i, _)| i).collect();
    let mut walks: [Vec<usize>; 4] = Default::default();
    for i in 0..4 {
        walks[i] = walk(cycle, pos[i], pos[(i + 1) % 4]);
    }
    Ok(PatchSides { walks })
}

fn side_chords(d: &Decomp, w: &[usize], out: &mut Vec<usize>) -> usize {
    let mut boundary = 0usize;
    for pair in w.windows(2) {
        let key = (pair[0].min(pair[1]), pair[0].max(pair[1]));
        match d.chord_of.get(&key) {
            Some(&ci) => out.push(ci),
            None => boundary += 1,
        }
    }
    boundary
}

/// Expand a patch side walk into its node list, inserting the interior nodes
/// of every chord.  Chord nodes are shared between the two patches because the
/// builder deduplicates by coordinate.
fn expand_side(b: &mut Builder, d: &Decomp, w: &[usize]) -> Vec<usize> {
    let mut out = Vec::new();
    out.push(w[0]);
    for pair in w.windows(2) {
        let (u, v) = (pair[0], pair[1]);
        let key = (u.min(v), u.max(v));
        if let Some(&ci) = d.chord_of.get(&key) {
            let k = d.chords[ci].count;
            let (pa, pb) = (b.coord(u), b.coord(v));
            for i in 1..k {
                out.push(b.add(lerp(pa, pb, i as f64 / k as f64)));
            }
        }
        out.push(v);
    }
    out
}

// ---------------------------------------------------------------------------
// boundary loop construction
// ---------------------------------------------------------------------------

/// Generate the boundary nodes of one loop from its `corners` (input-vertex
/// indices, in loop order) and return the node cycle and the corner node ids.
fn build_loop(
    d: &mut Decomp,
    loop_pts: &[[f64; 2]],
    corners: &[usize],
    size: f64,
    fixed_counts: Option<&[usize]>,
) -> Result<(Vec<usize>, Vec<usize>), String> {
    if corners.is_empty() {
        return Err("a boundary loop has no corners".to_string());
    }
    let mut chains = Vec::with_capacity(corners.len());
    let mut lengths = Vec::with_capacity(corners.len());
    for i in 0..corners.len() {
        let start = corners[i];
        let end = corners[(i + 1) % corners.len()];
        let poly = chain_points(loop_pts, start, end);
        let mut len = 0.0;
        for w in poly.windows(2) {
            len += dist(w[0], w[1]);
        }
        chains.push(poly);
        lengths.push(len);
    }
    let mut counts = match fixed_counts {
        Some(c) => {
            if c.len() != chains.len() {
                return Err("wrong number of fixed boundary counts".to_string());
            }
            c.to_vec()
        }
        None => lengths.iter().map(|&l| count_for(l, size)).collect(),
    };
    if counts.iter().any(|&c| c == 0 || c > MAX_COUNT) {
        return Err("a boundary discretisation count is out of range".to_string());
    }
    if fixed_counts.is_none() {
        // A loop must contribute an even number of boundary segments: the
        // patch parity system is solvable exactly when this holds.
        let total: usize = counts.iter().sum();
        if !total.is_multiple_of(2) {
            let (best, _) = lengths
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .ok_or_else(|| "empty boundary loop".to_string())?;
            counts[best] += 1;
        }
    }
    let mut cycle = Vec::new();
    let mut corner_ids = Vec::with_capacity(chains.len());
    for (poly, &count) in chains.iter().zip(counts.iter()) {
        let pts = distribute(poly, count)?;
        let ids: Vec<usize> = pts.iter().map(|&p| d.add_bpt(p)).collect();
        corner_ids.push(ids[0]);
        for id in ids.iter().take(count) {
            cycle.push(*id);
        }
    }
    if cycle.len() < 3 {
        return Err("a boundary loop has fewer than three nodes".to_string());
    }
    d.set_cycle(&cycle);
    Ok((cycle, corner_ids))
}

// ---------------------------------------------------------------------------
// simply connected domain
// ---------------------------------------------------------------------------

fn mesh_simply_connected(outer: &[[f64; 2]], size: f64, options: &Options) -> Result<Mesh, String> {
    let scale = outer
        .iter()
        .map(|p| p[0].abs().max(p[1].abs()))
        .fold(1.0f64, f64::max);
    let mut d = Decomp::new(scale);
    let mut corners = detect_corners(outer, options);
    if corners.len() < 3 {
        // A smooth loop is given four pseudo-corners so that it can still be
        // parameterised as a four-sided patch.
        corners = pseudo_corners(outer);
    }
    if corners.len() == 3 {
        return Err(
            "not supported yet: a triangular domain needs the three-sided template".to_string(),
        );
    }
    if corners.len() < 3 {
        return Err("the outer boundary does not define a polygon".to_string());
    }
    let (cycle, _corner_ids) = build_loop(&mut d, outer, &corners, size, None)?;

    // Recursive bisection.
    let mut patches: Vec<PatchSides> = Vec::new();
    let mut stack: Vec<Vec<usize>> = vec![cycle];
    let mut guard = 0usize;
    while let Some(cycle) = stack.pop() {
        guard += 1;
        if guard > MAX_PATCHES * 8 {
            return Err("domain decomposition did not terminate".to_string());
        }
        let corners = d.corners(&cycle, options);
        let k = corners.len();
        if k == 4 {
            patches.push(patch_sides(&d, &cycle, options)?);
            if patches.len() > MAX_PATCHES {
                return Err("too many patches".to_string());
            }
            continue;
        }
        if k < 4 {
            return Err(format!(
                "not supported yet: a subregion with {k} corners (triangle template missing)"
            ));
        }
        let mut best: Option<Eval> = None;
        for cand in candidates(&d, &cycle, &corners, options) {
            if let Some(ev) = evaluate(&mut d, &cycle, &cand, k, options) {
                if best.as_ref().is_none_or(|b| ev.score < b.score) {
                    best = Some(ev);
                }
            }
        }
        let Some(ev) = best else {
            return Err(format!(
                "domain decomposition stalled on a subregion with {k} corners"
            ));
        };
        // Materialise the selected chord endpoint if it is a point on a
        // boundary edge (evaluate() already added the point to the node list;
        // here we update the boundary adjacency).
        if let Target::EdgePoint { edge, p } = ev.target {
            let u = cycle[edge];
            let w = cycle[(edge + 1) % cycle.len()];
            let id = d.add_bpt(p);
            let point = d.bpts[id];
            d.insert_boundary(u, w, point);
        }
        register_chord(&mut d, ev.aug[ev.ip], ev.aug[ev.iq]);
        let (i, j) = if ev.ip < ev.iq {
            (ev.ip, ev.iq)
        } else {
            (ev.iq, ev.ip)
        };
        let r1 = ev.aug[i..=j].to_vec();
        let mut r2 = ev.aug[j..].to_vec();
        r2.extend_from_slice(&ev.aug[..=i]);
        stack.push(r1);
        stack.push(r2);
    }

    finalize(&d, &patches, size, options)
}

fn register_chord(d: &mut Decomp, u: usize, v: usize) {
    let key = (u.min(v), u.max(v));
    if d.chord_of.contains_key(&key) {
        return;
    }
    let index = d.chords.len();
    d.chords.push(Chord {
        a: key.0,
        b: key.1,
        count: 2,
    });
    d.chord_of.insert(key, index);
}

#[derive(Clone, Copy)]
enum Target {
    /// Position of an existing cycle vertex.
    Node(usize),
    /// A point on the edge from cycle[edge] to cycle[edge+1].
    EdgePoint { edge: usize, p: [f64; 2] },
}

#[derive(Clone, Copy)]
struct Candidate {
    a_pos: usize,
    target: Target,
    from_reflex: bool,
}

struct Eval {
    aug: Vec<usize>,
    ip: usize,
    iq: usize,
    score: f64,
    target: Target,
}

fn augment(
    d: &mut Decomp,
    cycle: &[usize],
    cand: &Candidate,
) -> Option<(Vec<usize>, usize, usize)> {
    let n = cycle.len();
    match cand.target {
        Target::Node(pos) => {
            if pos >= n {
                return None;
            }
            Some((cycle.to_vec(), cand.a_pos, pos))
        }
        Target::EdgePoint { edge, p } => {
            if edge >= n {
                return None;
            }
            let id = d.add_bpt(p);
            let mut aug = cycle.to_vec();
            let ip = if cand.a_pos > edge {
                cand.a_pos + 1
            } else {
                cand.a_pos
            };
            if edge + 1 == n {
                aug.push(id);
            } else {
                aug.insert(edge + 1, id);
            }
            Some((aug, ip, edge + 1))
        }
    }
}

fn candidates(
    d: &Decomp,
    cycle: &[usize],
    corners: &[(usize, f64)],
    options: &Options,
) -> Vec<Candidate> {
    let n = cycle.len();
    let mut out = Vec::new();
    let corner_positions: Vec<usize> = corners.iter().map(|&(i, _)| i).collect();

    // 1. Ray casts from reflex corners along the extensions of their edges.
    for &(pos, turn) in corners {
        if turn >= -options.corner_deg {
            continue;
        }
        let prev = cycle[(pos + n - 1) % n];
        let next = cycle[(pos + 1) % n];
        let v = d.p(cycle[pos]);
        let dirs = [sub(v, d.p(prev)), sub(v, d.p(next))];
        for dir in dirs {
            let Some((edge, s)) = d.cast_ray(cycle, pos, dir) else {
                continue;
            };
            let u = cycle[edge];
            let w = cycle[(edge + 1) % n];
            let hit = lerp(d.p(u), d.p(w), s);
            let seg_len = dist(d.p(u), d.p(w));
            let target = if s * seg_len <= d.eps() {
                Target::Node(edge)
            } else if (1.0 - s) * seg_len <= d.eps() {
                Target::Node((edge + 1) % n)
            } else if d.is_boundary_edge(u, w) {
                Target::EdgePoint { edge, p: hit }
            } else {
                // A hit on an existing chord: attach to the nearer endpoint.
                if dist(hit, d.p(u)) <= dist(hit, d.p(w)) {
                    Target::Node(edge)
                } else {
                    Target::Node((edge + 1) % n)
                }
            };
            out.push(Candidate {
                a_pos: pos,
                target,
                from_reflex: true,
            });
        }
    }

    // 2. Diagonals between two corners.
    for (i, &a) in corner_positions.iter().enumerate() {
        for &b in corner_positions.iter().skip(i + 1) {
            out.push(Candidate {
                a_pos: a,
                target: Target::Node(b),
                from_reflex: false,
            });
        }
    }

    // 3. Midpoint of a side to a corner.  Needed for pentagons, where every
    // corner-corner diagonal cuts off a triangle.
    if corners.len() >= 5 {
        for i in 0..corner_positions.len() {
            let p0 = corner_positions[i];
            let p1 = corner_positions[(i + 1) % corner_positions.len()];
            let len = (p1 + n - p0) % n;
            if len < 2 {
                continue;
            }
            let mid = (p0 + len / 2) % n;
            for &c in &corner_positions {
                if c == p0 || c == p1 || c == mid {
                    continue;
                }
                out.push(Candidate {
                    a_pos: c,
                    target: Target::Node(mid),
                    from_reflex: false,
                });
            }
        }
    }
    out
}

fn evaluate(
    d: &mut Decomp,
    cycle: &[usize],
    cand: &Candidate,
    parent_corners: usize,
    options: &Options,
) -> Option<Eval> {
    let (aug, ip, iq) = augment(d, cycle, cand)?;
    let n = aug.len();
    let circ = ip.abs_diff(iq);
    if circ == 0 || circ == 1 || circ == n - 1 {
        return None;
    }
    if !chord_inside(d, &aug, ip, iq) {
        return None;
    }
    let (i, j) = if ip < iq { (ip, iq) } else { (iq, ip) };
    let r1 = aug[i..=j].to_vec();
    let mut r2 = aug[j..].to_vec();
    r2.extend_from_slice(&aug[..=i]);
    let k1 = d.corners(&r1, options).len();
    let k2 = d.corners(&r2, options).len();
    // Both children must make progress and must not be triangles.
    if k1 < 4 || k2 < 4 || k1 >= parent_corners || k2 >= parent_corners {
        return None;
    }
    let a1 = polygon_area(d, &r1);
    let a2 = polygon_area(d, &r2);
    if !a1.is_finite() || !a2.is_finite() || a1 <= 0.0 || a2 <= 0.0 {
        return None;
    }
    let imbalance = (a1 - a2).abs() / (a1 + a2);
    let chord = dist(d.p(aug[ip]), d.p(aug[iq]));
    let size_pen = chord / (d.scale * 4.0);
    let reflex_bonus = if cand.from_reflex { -0.6 } else { 0.0 };
    let score = 0.08 * (k1 + k2) as f64 + 1.0 * imbalance + 0.05 * size_pen + reflex_bonus;
    Some(Eval {
        aug,
        ip,
        iq,
        score,
        target: cand.target,
    })
}

// ---------------------------------------------------------------------------
// final parity and meshing
// ---------------------------------------------------------------------------

fn chord_length(d: &Decomp, ci: usize) -> f64 {
    dist(d.bpts[d.chords[ci].a], d.bpts[d.chords[ci].b])
}

fn finalize(
    d: &Decomp,
    patches: &[PatchSides],
    size: f64,
    options: &Options,
) -> Result<Mesh, String> {
    let mut builder = Builder::with_boundary(&d.bpts);
    let pcount = patches.len();
    let mut patch_chords: Vec<Vec<usize>> = Vec::with_capacity(pcount);
    let mut residual = vec![0u8; pcount];
    let mut use_by_chord: Vec<Vec<usize>> = vec![Vec::new(); d.chords.len()];

    for (pi, patch) in patches.iter().enumerate() {
        let mut chords = Vec::new();
        let mut boundary = 0usize;
        for w in &patch.walks {
            boundary += side_chords(d, w, &mut chords);
        }
        let mut parity = (boundary % 2) as u8;
        for &ci in &chords {
            let length = chord_length(d, ci);
            let count = count_for(length, size);
            parity ^= (count % 2) as u8;
            use_by_chord[ci].push(pi);
        }
        residual[pi] = parity;
        patch_chords.push(chords);
    }

    // Nominal chord counts.
    let mut counts: Vec<usize> = (0..d.chords.len())
        .map(|ci| count_for(chord_length(d, ci), size))
        .collect();

    // GF(2) parity solve on the patch graph: find a spanning forest, settle the
    // non-tree chords at their nominal parity, then fix the tree chords
    // bottom-up.
    let mut adjacency: Vec<Vec<(usize, usize)>> = vec![Vec::new(); pcount];
    for (ci, users) in use_by_chord.iter().enumerate() {
        if users.len() == 2 && users[0] != users[1] {
            adjacency[users[0]].push((ci, users[1]));
            adjacency[users[1]].push((ci, users[0]));
        } else if users.len() > 2 {
            return Err("a chord is shared by more than two patches".to_string());
        }
    }
    let mut parent: Vec<Option<(usize, usize)>> = vec![None; pcount];
    let mut children: Vec<Vec<(usize, usize)>> = vec![Vec::new(); pcount];
    let mut order = Vec::with_capacity(pcount);
    let mut seen = vec![false; pcount];
    let mut queue = VecDeque::new();
    for root in 0..pcount {
        if seen[root] {
            continue;
        }
        seen[root] = true;
        queue.push_back(root);
        while let Some(p) = queue.pop_front() {
            order.push(p);
            for &(ci, q) in &adjacency[p] {
                if !seen[q] {
                    seen[q] = true;
                    parent[q] = Some((p, ci));
                    children[p].push((ci, q));
                    queue.push_back(q);
                }
            }
        }
    }
    let mut flip = vec![0u8; d.chords.len()];
    for &p in order.iter().rev() {
        let mut acc = residual[p];
        for &(ci, _) in &children[p] {
            acc ^= flip[ci];
        }
        match parent[p] {
            Some((_, ci)) => flip[ci] = acc,
            None => {
                if acc != 0 {
                    return Err("the patch parity system has no solution".to_string());
                }
            }
        }
    }
    for (ci, &f) in flip.iter().enumerate() {
        if f == 1 {
            let n = counts[ci];
            counts[ci] = if n % 2 == 1 && n >= 3 { n - 1 } else { n + 1 };
        }
    }

    let mut chords: Vec<Chord> = d.chords.clone();
    for (ci, count) in counts.iter().enumerate() {
        if *count == 0 || *count > MAX_COUNT {
            return Err("a chord discretisation count is out of range".to_string());
        }
        chords[ci].count = *count;
    }
    // Patch totals must now be even.
    for (pi, patch) in patches.iter().enumerate() {
        let mut total = 0usize;
        for w in &patch.walks {
            for pair in w.windows(2) {
                let key = (pair[0].min(pair[1]), pair[0].max(pair[1]));
                total += match d.chord_of.get(&key) {
                    Some(&ci) => chords[ci].count,
                    None => 1,
                };
            }
        }
        if !total.is_multiple_of(2) {
            return Err(format!("patch {pi} still has an odd side total"));
        }
    }

    let view = Decomp {
        bpts: d.bpts.clone(),
        bmap: d.bmap.clone(),
        bnext: d.bnext.clone(),
        chords,
        chord_of: d.chord_of.clone(),
        scale: d.scale,
    };
    for patch in patches {
        let mut sides: [Vec<usize>; 4] = Default::default();
        for (k, w) in patch.walks.iter().enumerate() {
            sides[k] = expand_side(&mut builder, &view, w);
        }
        mesh_patch(&mut builder, sides, 0, options)?;
    }
    let mesh = prune(Mesh::new(builder.points, builder.cells));
    mesh.validate().map_err(|e| e.to_string())?;
    Ok(mesh)
}

/// Drop the nodes that no cell uses.  Candidate evaluation adds a few orphan
/// points to the boundary list; they must not leak into the returned mesh.
fn prune(mesh: Mesh) -> Mesh {
    let mut used = vec![false; mesh.points.len()];
    for cell in &mesh.cells {
        for &id in &cell.nodes {
            if id < used.len() {
                used[id] = true;
            }
        }
    }
    let mut map = vec![usize::MAX; mesh.points.len()];
    let mut points = Vec::new();
    for (old, &u) in used.iter().enumerate() {
        if u {
            map[old] = points.len();
            points.push(mesh.points[old]);
        }
    }
    let mut cells = mesh.cells;
    for cell in &mut cells {
        for id in &mut cell.nodes {
            *id = map[*id];
        }
    }
    Mesh::new(points, cells)
}

// ---------------------------------------------------------------------------
// one hole: a four-patch O-grid
// ---------------------------------------------------------------------------

/// Insert the arc-length midpoint of each of the four outer sides as a real
/// vertex, and return the new loop together with the structural indices
/// `[C0, M0, C1, M1, C2, M2, C3, M3]` in loop order.
fn insert_side_midpoints(
    outer: &[[f64; 2]],
    corners: &[usize],
) -> Result<(Vec<[f64; 2]>, [usize; 8]), String> {
    let n = outer.len();
    let mut cum = vec![0.0; n + 1];
    for i in 0..n {
        cum[i + 1] = cum[i] + dist(outer[i], outer[(i + 1) % n]);
    }
    let total = cum[n];
    if !total.is_finite() || total <= 1.0e-12 {
        return Err("the outer loop has zero length".to_string());
    }
    let mut insertion: Vec<(usize, f64, [f64; 2])> = Vec::with_capacity(4);
    for k in 0..4 {
        let start = corners[k];
        let end = corners[(k + 1) % 4];
        let side_len = if end > start {
            cum[end] - cum[start]
        } else {
            total - cum[start] + cum[end]
        };
        let mut pos = cum[start] + 0.5 * side_len;
        if pos >= total {
            pos -= total;
        }
        let mut j = 0usize;
        while j + 1 < n && cum[j + 1] < pos {
            j += 1;
        }
        let seg = cum[j + 1] - cum[j];
        let t = if seg > 0.0 {
            ((pos - cum[j]) / seg).clamp(0.0, 1.0)
        } else {
            0.0
        };
        insertion.push((j, t, lerp(outer[j], outer[(j + 1) % n], t)));
    }
    let mut new_points = Vec::with_capacity(n + 4);
    let mut vertex_new = vec![usize::MAX; n];
    let mut mid_new = [usize::MAX; 4];
    for j in 0..n {
        vertex_new[j] = new_points.len();
        new_points.push(outer[j]);
        for (k, &(sj, t, point)) in insertion.iter().enumerate() {
            if sj != j {
                continue;
            }
            if t <= 1.0e-9 {
                mid_new[k] = vertex_new[j];
            } else if t >= 1.0 - 1.0e-9 {
                // handled after all vertices are known
            } else {
                mid_new[k] = new_points.len();
                new_points.push(point);
            }
        }
    }
    for (k, &(sj, t, _)) in insertion.iter().enumerate() {
        if t >= 1.0 - 1.0e-9 {
            mid_new[k] = vertex_new[(sj + 1) % n];
        }
    }
    let mut structure = [0usize; 8];
    for k in 0..4 {
        structure[2 * k] = vertex_new[corners[k]];
        structure[2 * k + 1] = mid_new[k];
        if mid_new[k] == vertex_new[corners[k]] || mid_new[k] == vertex_new[corners[(k + 1) % 4]] {
            return Err("an outer side is too short to split at its midpoint".to_string());
        }
    }
    Ok((new_points, structure))
}

fn angle_diff(a: f64, b: f64) -> f64 {
    let mut d = (a - b) % std::f64::consts::TAU;
    if d > std::f64::consts::PI {
        d -= std::f64::consts::TAU;
    }
    if d < -std::f64::consts::PI {
        d += std::f64::consts::TAU;
    }
    d.abs()
}

/// One hole: an eight-block O-grid, one block per half outer side.
///
/// The outer boundary is cut at the four corners and the four side midpoints,
/// and the hole at the eight nodes radially facing those points.  Block `s` is
/// the straight half-side `S_s -> S_{s+1}`, a chord to `H_{s+1}`, the hole arc
/// from `H_{s+1}` to `H_s`, and a chord back to `S_s`.  All sides are straight
/// or a single hole arc, so the transfinite maps have no kink, and the corner
/// chords are no longer tangent to the hole (the failure mode of the
/// corner-to-hole four-block O-grid).
fn mesh_one_hole(
    outer: &[[f64; 2]],
    hole: &[[f64; 2]],
    size: f64,
    options: &Options,
) -> Result<Mesh, String> {
    let scale = outer
        .iter()
        .map(|p| p[0].abs().max(p[1].abs()))
        .fold(1.0f64, f64::max);
    let mut d = Decomp::new(scale);
    let outer_corners = detect_corners(outer, options);
    if outer_corners.len() != 4 {
        return Err(format!(
            "not supported yet: the O-grid needs a four-corner outer loop (found {})",
            outer_corners.len()
        ));
    }
    let (outer2, structure) = insert_side_midpoints(outer, &outer_corners)?;

    let hole_corners = detect_corners(hole, options);
    let (hole_cycle, _) = if hole_corners.len() >= 3 {
        build_loop(&mut d, hole, &hole_corners, size, None)?
    } else {
        build_loop(&mut d, hole, &[0], size, None)?
    };
    let nh = hole_cycle.len();
    if nh < 8 {
        return Err("the hole discretisation is too coarse for an O-grid".to_string());
    }

    // Eight hole points: the hole node nearest the radial direction of each of
    // the eight structural outer points.  Their cyclic order is the reverse of
    // the structural order, so the four corner chords enter the hole inside the
    // corner's visible cone instead of running tangentially along the arc.
    let centroid: [f64; 2] = {
        let mut c = [0.0, 0.0];
        for &node in &hole_cycle {
            let p = d.p(node);
            c[0] += p[0];
            c[1] += p[1];
        }
        [c[0] / nh as f64, c[1] / nh as f64]
    };
    let mut hidx = [0usize; 8];
    let mut used = vec![false; nh];
    for s in 0..8 {
        let p = outer2[structure[s]];
        let target = (p[1] - centroid[1]).atan2(p[0] - centroid[0]);
        let mut best = (f64::INFINITY, 0usize);
        for (j, &node) in hole_cycle.iter().enumerate() {
            if used[j] {
                continue;
            }
            let q = d.p(node);
            let a = (q[1] - centroid[1]).atan2(q[0] - centroid[0]);
            let diff = angle_diff(a, target);
            if diff < best.0 {
                best = (diff, j);
            }
        }
        used[best.1] = true;
        hidx[s] = best.1;
    }
    // Block s uses the hole arc H[s+1] -> H[s] in stored order.  The eight arcs
    // must partition the hole.
    let mut arcs = [0usize; 8];
    let mut valid = true;
    let mut sum = 0usize;
    for s in 0..8 {
        let arc = (hidx[s] + nh - hidx[(s + 1) % 8]) % nh;
        if arc == 0 {
            valid = false;
            break;
        }
        arcs[s] = arc;
        sum += arc;
    }
    if !valid || sum != nh {
        // Fallback: eight nearly-even points in the required cyclic order.
        let mut best_cost = f64::INFINITY;
        let mut best = [0usize; 8];
        for base in 0..nh {
            let mut cand = [0usize; 8];
            for s in 0..8 {
                cand[s] = (base + nh - (s * nh / 8) % nh) % nh;
            }
            let mut ok = true;
            let mut total = 0usize;
            for s in 0..8 {
                let arc = (cand[s] + nh - cand[(s + 1) % 8]) % nh;
                if arc == 0 {
                    ok = false;
                    break;
                }
                total += arc;
            }
            if !ok || total != nh {
                continue;
            }
            let mut cost = 0.0;
            for s in 0..8 {
                let p = outer2[structure[s]];
                let q = d.p(hole_cycle[cand[s]]);
                let a = (q[1] - centroid[1]).atan2(q[0] - centroid[0]);
                let target = (p[1] - centroid[1]).atan2(p[0] - centroid[0]);
                cost += angle_diff(a, target);
            }
            if cost < best_cost {
                best_cost = cost;
                best = cand;
            }
        }
        if !best_cost.is_finite() {
            return Err("could not place the eight hole split points".to_string());
        }
        hidx = best;
        for s in 0..8 {
            arcs[s] = (hidx[s] + nh - hidx[(s + 1) % 8]) % nh;
        }
    }

    // One chord per structural point, one private outer half-side per block.
    let mut chord_counts = [0usize; 8];
    for s in 0..8 {
        chord_counts[s] = count_for(dist(outer2[structure[s]], d.p(hole_cycle[hidx[s]])), size);
    }
    let mut chain_len = [0.0f64; 8];
    for s in 0..8 {
        let from = structure[s];
        let to = structure[(s + 1) % 8];
        let mut i = from;
        while i != to {
            let next = (i + 1) % outer2.len();
            chain_len[s] += dist(outer2[i], outer2[next]);
            i = next;
        }
    }
    let mut counts: Vec<usize> = chain_len.iter().map(|&l| count_for(l, size)).collect();
    for s in 0..8 {
        let parity = (counts[s] + chord_counts[s] + chord_counts[(s + 1) % 8] + arcs[s]) % 2;
        if parity != 0 {
            counts[s] += 1;
        }
    }
    let (outer_cycle, ids) = build_loop(&mut d, &outer2, &structure, size, Some(&counts))?;
    for s in 0..8 {
        let key = (
            ids[s].min(hole_cycle[hidx[s]]),
            ids[s].max(hole_cycle[hidx[s]]),
        );
        if !d.chord_of.contains_key(&key) {
            let index = d.chords.len();
            d.chords.push(Chord {
                a: key.0,
                b: key.1,
                count: chord_counts[s],
            });
            d.chord_of.insert(key, index);
        }
    }

    let mut patches: Vec<PatchSides> = Vec::with_capacity(8);
    for s in 0..8 {
        let s0 = ids[s];
        let s1 = ids[(s + 1) % 8];
        let p0 = position_of(&outer_cycle, s0)?;
        let p1 = position_of(&outer_cycle, s1)?;
        let outer_side = walk(&outer_cycle, p0, p1);
        let h1 = hole_cycle[hidx[(s + 1) % 8]];
        let h0 = hole_cycle[hidx[s]];
        let hp = position_of(&hole_cycle, h1)?;
        let hq = position_of(&hole_cycle, h0)?;
        let hole_arc = walk(&hole_cycle, hp, hq);
        let walks = [outer_side, vec![s1, h1], hole_arc, vec![h0, s0]];
        patches.push(PatchSides { walks });
    }
    finalize_ogrid(&d, &patches, options)
}

fn finalize_ogrid(d: &Decomp, patches: &[PatchSides], options: &Options) -> Result<Mesh, String> {
    // The O-grid chords already carry their counts; check the parity of each
    // patch and build the mesh.
    let mut builder = Builder::with_boundary(&d.bpts);
    for (pi, patch) in patches.iter().enumerate() {
        let mut total = 0usize;
        for w in &patch.walks {
            for pair in w.windows(2) {
                let key = (pair[0].min(pair[1]), pair[0].max(pair[1]));
                total += match d.chord_of.get(&key) {
                    Some(&ci) => d.chords[ci].count,
                    None => 1,
                };
            }
        }
        if !total.is_multiple_of(2) {
            return Err(format!("O-grid patch {pi} has an odd side total"));
        }
    }
    for patch in patches {
        let mut sides: [Vec<usize>; 4] = Default::default();
        for (k, w) in patch.walks.iter().enumerate() {
            sides[k] = expand_side(&mut builder, d, w);
        }
        mesh_patch(&mut builder, sides, 0, options)?;
    }
    let mesh = prune(Mesh::new(builder.points, builder.cells));
    mesh.validate().map_err(|e| e.to_string())?;
    Ok(mesh)
}

fn position_of(cycle: &[usize], node: usize) -> Result<usize, String> {
    cycle
        .iter()
        .position(|&x| x == node)
        .ok_or_else(|| "a patch corner is missing from its boundary loop".to_string())
}

// ---------------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::verify;

    fn check_all_quads(domain: &Domain, size: f64) -> Mesh {
        let sizing = Sizing::uniform(size).unwrap();
        let mesh = mesh(domain, &sizing).unwrap();
        assert!(mesh
            .cells
            .iter()
            .all(|c| c.kind == crate::mesh::Kind::Quad4));
        assert!(!mesh.cells.is_empty());
        let offset = verify::boundary_offset(domain, &mesh);
        assert!(offset < 1.0e-9, "boundary offset {offset}");
        let topo = mesh.topology();
        assert_eq!(topo.cracks, 0, "cracks");
        let gate = verify::gate(&mesh);
        assert_eq!(gate.triangles, 0);
        assert_eq!(gate.inverted, 0);
        mesh
    }

    #[test]
    fn square_is_a_tensor_grid() {
        let domain = Domain::rectangle(10.0, 10.0).unwrap();
        let mesh = check_all_quads(&domain, 1.0);
        assert_eq!(mesh.cells.len(), 100);
        let gate = verify::gate(&mesh);
        assert!(gate.passes(), "{:?}", gate.reason());
    }

    #[test]
    fn rectangle_is_a_tensor_grid() {
        let domain = Domain::rectangle(20.0, 10.0).unwrap();
        let mesh = check_all_quads(&domain, 1.0);
        assert_eq!(mesh.cells.len(), 200);
        let gate = verify::gate(&mesh);
        assert!(gate.passes(), "{:?}", gate.reason());
    }

    #[test]
    fn l_bracket_decomposes_into_two_patches() {
        let domain = Domain::new(
            vec![
                [0.0, 0.0],
                [80.0, 0.0],
                [80.0, 20.0],
                [20.0, 20.0],
                [20.0, 60.0],
                [0.0, 60.0],
            ],
            vec![],
        )
        .unwrap();
        let mesh = check_all_quads(&domain, 2.0);
        let gate = verify::gate(&mesh);
        assert!(gate.passes(), "{:?}", gate.reason());
        assert_eq!(mesh.cells.len(), 600);
    }

    #[test]
    fn stepped_part_is_all_quads() {
        let domain = Domain::new(
            vec![
                [0.0, 0.0],
                [60.0, 0.0],
                [60.0, 40.0],
                [30.0, 40.0],
                [30.0, 30.0],
                [20.0, 30.0],
                [20.0, 40.0],
                [0.0, 40.0],
            ],
            vec![],
        )
        .unwrap();
        let mesh = check_all_quads(&domain, 1.0);
        let gate = verify::gate(&mesh);
        assert!(gate.passes(), "{:?}", gate.reason());
    }

    #[test]
    fn quarter_annulus_uses_the_transition_template() {
        let arc = |r: f64, a0: f64, a1: f64, n: usize| -> Vec<[f64; 2]> {
            (0..=n)
                .map(|i| {
                    let a = a0 + (a1 - a0) * i as f64 / n as f64;
                    [r * a.cos(), r * a.sin()]
                })
                .collect()
        };
        let mut outer = arc(20.0, 0.0, std::f64::consts::FRAC_PI_2, 24);
        let inner = arc(10.0, std::f64::consts::FRAC_PI_2, 0.0, 12);
        outer.extend(inner);
        let domain = Domain::new(outer, vec![]).unwrap();
        let mesh = check_all_quads(&domain, 1.0);
        let gate = verify::gate(&mesh);
        assert!(gate.passes(), "{:?}", gate.reason());
    }

    #[test]
    fn plate_with_hole_is_a_conforming_o_grid() {
        let domain = Domain::plate_with_hole(100.0, 60.0, 10.0, 16).unwrap();
        let mesh = check_all_quads(&domain, 4.0);
        let gate = verify::gate(&mesh);
        assert_eq!(gate.cracks, 0);
        assert_eq!(gate.triangles, 0);
        assert_eq!(gate.inverted, 0);
        assert_eq!(gate.overlapping, 0);
        assert!(gate.passes(), "{:?}", gate.reason());
    }

    #[test]
    fn triangle_domain_is_refused_not_meshed() {
        let domain = Domain::new(vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], vec![]).unwrap();
        let err = mesh(&domain, &Sizing::uniform(0.2).unwrap()).unwrap_err();
        assert!(err.contains("triangular"), "{err}");
    }

    #[test]
    fn bad_size_is_refused() {
        let domain = Domain::rectangle(1.0, 1.0).unwrap();
        let sizing = Sizing {
            size: 0.0,
            all_quad: false,
        };
        assert!(mesh(&domain, &sizing).is_err());
    }

    #[test]
    fn non_default_options_produce_a_valid_mesh() {
        let domain = Domain::new(
            vec![
                [0.0, 0.0],
                [80.0, 0.0],
                [80.0, 20.0],
                [20.0, 20.0],
                [20.0, 60.0],
                [0.0, 60.0],
            ],
            vec![],
        )
        .unwrap();
        let sizing = Sizing::uniform(2.0).unwrap();
        let options = Options {
            corner_deg: 45.0,
            max_depth: 8,
        };
        assert_ne!(options.corner_deg, Options::default().corner_deg);
        assert_ne!(options.max_depth, Options::default().max_depth);
        let mesh = mesh_with_options(&domain, &sizing, &options).unwrap();
        assert!(mesh
            .cells
            .iter()
            .all(|c| c.kind == crate::mesh::Kind::Quad4));
        assert!(!mesh.cells.is_empty());
        let offset = verify::boundary_offset(&domain, &mesh);
        assert!(offset < 1.0e-9, "boundary offset {offset}");
        let topo = mesh.topology();
        assert_eq!(topo.cracks, 0, "cracks");
        let gate = verify::gate(&mesh);
        assert_eq!(gate.triangles, 0);
        assert_eq!(gate.inverted, 0);
        assert!(gate.passes(), "{:?}", gate.reason());
    }
}
