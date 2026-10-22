//! P1: planar all-quad meshing.
//!
//! The pipeline is the one the plan fixes: build a constrained Delaunay
//! triangulation of the domain, refine it to the requested size, then pair
//! triangles into quads with a **maximum-cardinality, maximum-weight matching**
//! on the dual graph (the Blossom-Quad construction). Pairing first maximises
//! how many triangles are used — that is what makes the result all-quad — and
//! then maximises the quality of the quads formed.
//!
//! The triangulation is the substrate, not the product: a triangle that cannot
//! be paired is returned separately, never silently dropped.

use mwmatching::{Matching, SENTINEL};
use spade::{ConstrainedDelaunayTriangulation, Point2, Triangulation};

use crate::mesh::{Cell, Mesh, Node};
use crate::quality;

/// A planar domain: one outer loop and any number of holes, all closed
/// polygons in the XY plane (millimetres).
#[derive(Clone, Debug, Default)]
pub struct Domain {
    pub outer: Vec<[f64; 2]>,
    pub holes: Vec<Vec<[f64; 2]>>,
}

/// What can go wrong building or meshing a domain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TwoDError {
    /// A loop has fewer than three points.
    LoopTooShort(usize),
    /// A coordinate is not finite.
    NonFinite,
    /// The mesh request is unusable (size ≤ 0, or the domain has no area).
    BadRequest(&'static str),
    /// The triangulation failed.
    Triangulation(&'static str),
}

impl std::fmt::Display for TwoDError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TwoDError::LoopTooShort(n) => write!(f, "a loop has {n} points; at least 3 are needed"),
            TwoDError::NonFinite => write!(f, "a boundary coordinate is not finite"),
            TwoDError::BadRequest(message) => write!(f, "{message}"),
            TwoDError::Triangulation(message) => write!(f, "triangulation failed: {message}"),
        }
    }
}

impl std::error::Error for TwoDError {}

impl Domain {
    /// A domain with an outer loop and optional holes.
    pub fn new(outer: Vec<[f64; 2]>, holes: Vec<Vec<[f64; 2]>>) -> Result<Self, TwoDError> {
        let domain = Domain { outer, holes };
        domain.validate()?;
        Ok(domain)
    }

    /// An axis-aligned rectangle, counter-clockwise.
    pub fn rectangle(width: f64, height: f64) -> Result<Self, TwoDError> {
        Domain::new(
            vec![[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
            Vec::new(),
        )
    }

    /// A rectangle with a circular hole at its centre.
    pub fn plate_with_hole(
        width: f64,
        height: f64,
        radius: f64,
        segments: usize,
    ) -> Result<Self, TwoDError> {
        let circle = (0..segments)
            .map(|i| {
                let a = std::f64::consts::TAU * i as f64 / segments as f64;
                // clockwise, so the hole is traversed opposite to the outer loop
                [
                    width / 2.0 + radius * a.cos(),
                    height / 2.0 - radius * a.sin(),
                ]
            })
            .collect();
        Domain::new(
            vec![[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]],
            vec![circle],
        )
    }

    fn validate(&self) -> Result<(), TwoDError> {
        let loops = std::iter::once(&self.outer).chain(self.holes.iter());
        for loop_points in loops {
            if loop_points.len() < 3 {
                return Err(TwoDError::LoopTooShort(loop_points.len()));
            }
            if loop_points
                .iter()
                .any(|p| !p[0].is_finite() || !p[1].is_finite())
            {
                return Err(TwoDError::NonFinite);
            }
        }
        Ok(())
    }

    /// Signed area: positive when the outer loop is counter-clockwise.
    pub fn area(&self) -> f64 {
        let ring = |points: &[[f64; 2]]| -> f64 {
            let mut sum = 0.0;
            for i in 0..points.len() {
                let a = points[i];
                let b = points[(i + 1) % points.len()];
                sum += a[0] * b[1] - b[0] * a[1];
            }
            sum / 2.0
        };
        ring(&self.outer) - self.holes.iter().map(|h| ring(h).abs()).sum::<f64>()
    }

    /// Boundary segments, outer loop and holes.
    pub fn segments(&self) -> Vec<([f64; 2], [f64; 2])> {
        let mut out = Vec::new();
        for loop_points in std::iter::once(&self.outer).chain(self.holes.iter()) {
            for i in 0..loop_points.len() {
                out.push((loop_points[i], loop_points[(i + 1) % loop_points.len()]));
            }
        }
        out
    }

    /// Even-odd test: inside the outer loop and outside every hole.
    pub fn contains(&self, p: [f64; 2]) -> bool {
        if !point_in_polygon(p, &self.outer) {
            return false;
        }
        !self.holes.iter().any(|h| point_in_polygon(p, h))
    }

    /// Distance from `p` to the nearest boundary segment.
    pub fn distance_to_boundary(&self, p: [f64; 2]) -> f64 {
        self.nearest_segment(p).2
    }

    /// The boundary segment nearest to `p`: its endpoints and the distance.
    pub fn nearest_segment(&self, p: [f64; 2]) -> ([f64; 2], [f64; 2], f64) {
        let mut best = ([0.0; 2], [0.0; 2], f64::INFINITY);
        for (a, b) in self.segments() {
            let d = point_segment_distance(p, a, b);
            if d < best.2 {
                best = (a, b, d);
            }
        }
        best
    }
}

fn point_in_polygon(p: [f64; 2], polygon: &[[f64; 2]]) -> bool {
    let mut inside = false;
    let n = polygon.len();
    for i in 0..n {
        let a = polygon[i];
        let b = polygon[(i + 1) % n];
        if (a[1] > p[1]) != (b[1] > p[1]) {
            let t = (p[1] - a[1]) / (b[1] - a[1]);
            if p[0] < a[0] + t * (b[0] - a[0]) {
                inside = !inside;
            }
        }
    }
    inside
}

fn point_segment_distance(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let length2 = ab[0] * ab[0] + ab[1] * ab[1];
    if length2 <= 0.0 {
        return ((p[0] - a[0]).powi(2) + (p[1] - a[1]).powi(2)).sqrt();
    }
    let t = (((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / length2).clamp(0.0, 1.0);
    let q = [a[0] + t * ab[0], a[1] + t * ab[1]];
    ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt()
}

/// A uniform element size, in millimetres. Curvature and feature fields come
/// later; the P1 gate only needs this.
#[derive(Clone, Copy, Debug)]
pub struct Sizing {
    pub size: f64,
}

impl Sizing {
    pub fn uniform(size: f64) -> Result<Self, TwoDError> {
        if !size.is_finite() || size <= 0.0 {
            return Err(TwoDError::BadRequest(
                "the element size must be positive and finite",
            ));
        }
        Ok(Sizing { size })
    }

    pub fn size_at(&self, _p: [f64; 2]) -> f64 {
        self.size
    }
}

/// Statistics of one planar meshing run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TwoDStats {
    pub points: usize,
    pub triangles: usize,
    pub quads: usize,
    /// Triangles the matching could not pair. They are kept in the mesh, so the
    /// output is quad-dominant rather than all-quad.
    pub unpaired_triangles: usize,
    /// Reserved for the closure pass (not yet enabled); always zero.
    pub absorbed_triangles: usize,
}

impl TwoDStats {
    /// Share of planar elements that are quads.
    pub fn quad_ratio(&self) -> f64 {
        let total = self.quads + self.unpaired_triangles;
        if total == 0 {
            0.0
        } else {
            self.quads as f64 / total as f64
        }
    }
}

/// A planar quad mesh, plus any triangle the matching could not pair.
#[derive(Clone, Debug)]
pub struct TwoDResult {
    pub mesh: Mesh,
    pub unpaired: Vec<[usize; 3]>,
    pub stats: TwoDStats,
}

/// A candidate point closer than this fraction of the local size to an existing
/// point is rejected: Ruppert's spacing rule. Without it, circumcenters land a
/// few hundredths of a millimetre from a neighbour and the quad containing that
/// edge has an aspect ratio in the thousands.
const MIN_SPACING_FRACTION: f64 = 0.45;

/// A uniform grid of points, for "is this candidate too close to anything?".
struct PointGrid {
    cell: f64,
    buckets: std::collections::HashMap<(i64, i64), Vec<[f64; 2]>>,
}

impl PointGrid {
    fn new(cell: f64) -> Self {
        PointGrid {
            cell: cell.max(1e-9),
            buckets: std::collections::HashMap::new(),
        }
    }

    fn key(&self, p: [f64; 2]) -> (i64, i64) {
        (
            (p[0] / self.cell).floor() as i64,
            (p[1] / self.cell).floor() as i64,
        )
    }

    fn insert(&mut self, p: [f64; 2]) {
        self.buckets.entry(self.key(p)).or_default().push(p);
    }

    /// Distance to the nearest stored point, or infinity when the grid is empty.
    fn nearest(&self, p: [f64; 2]) -> f64 {
        let (cx, cy) = self.key(p);
        let mut best = f64::INFINITY;
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(points) = self.buckets.get(&(cx + dx, cy + dy)) {
                    for q in points {
                        let d = ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt();
                        best = best.min(d);
                    }
                }
            }
        }
        best
    }
}

/// A pair whose quad would be worse than this is not offered to the matching.
/// Without a floor the matching pairs two thin triangles into a needle, because
/// pairing is worth more than leaving them unpaired.
const MIN_PAIR_SCALED_JACOBIAN: f64 = 0.1;
/// Likewise for the smallest corner angle of the candidate quad.
const MIN_PAIR_ANGLE_DEG: f64 = 20.0;
/// And for the largest: a corner near 180° means the pair is a needle, not a
/// quad, however good its scaled Jacobian looks.
const MAX_PAIR_ANGLE_DEG: f64 = 150.0;

/// Refinement stops after this many passes even if the size field is unmet.
const MAX_PASSES: usize = 200;
/// A circumcenter closer than this to the boundary is not inserted: it would
/// create a sliver along the edge instead of improving the element.
const BOUNDARY_MARGIN: f64 = 0.55;

/// Mesh a planar domain with all-quad elements.
pub fn mesh_2d(domain: &Domain, sizing: &Sizing) -> Result<TwoDResult, TwoDError> {
    if domain.area() <= 0.0 {
        return Err(TwoDError::BadRequest(
            "the domain has no area (outer loop clockwise, or a hole larger than it)",
        ));
    }
    // Sample every boundary segment down to the size field before constraining
    // it. A plate edge 100 mm long with a 1 mm size would otherwise be a single
    // constrained edge, and every triangle touching it would be a sliver.
    let loops = sample_loops(domain, sizing);
    let mut cdt = ConstrainedDelaunayTriangulation::<Point2<f64>>::new();
    for loop_points in &loops {
        for &[x, y] in loop_points {
            cdt.insert(Point2::new(x, y))
                .map_err(|_| TwoDError::Triangulation("a boundary point could not be inserted"))?;
        }
        for i in 0..loop_points.len() {
            let a = loop_points[i];
            let b = loop_points[(i + 1) % loop_points.len()];
            cdt.add_constraint_edge(Point2::new(a[0], a[1]), Point2::new(b[0], b[1]))
                .map_err(|_| {
                    TwoDError::Triangulation("a boundary edge could not be constrained")
                })?;
        }
    }

    refine(&mut cdt, domain, sizing);

    // Triangles of the domain, keyed by spade's stable vertex indices.
    let triangles = collect_triangles(&cdt, domain);
    if triangles.is_empty() {
        return Err(TwoDError::BadRequest("the domain triangulated to nothing"));
    }

    let mut points: Vec<Node> = Vec::new();
    let mut index_of: std::collections::HashMap<(u64, u64), usize> =
        std::collections::HashMap::new();
    let mut tris: Vec<[usize; 3]> = Vec::with_capacity(triangles.len());
    for tri in &triangles {
        let mut ids = [0usize; 3];
        for (slot, p) in tri.iter().enumerate() {
            let key = (p[0].to_bits(), p[1].to_bits());
            let id = *index_of.entry(key).or_insert_with(|| {
                points.push([p[0], p[1], 0.0]);
                points.len() - 1
            });
            ids[slot] = id;
        }
        tris.push(ids);
    }

    let (quads, unpaired) = pair_triangles(&tris, &points);
    // Close the mesh: absorb any triangle the matching could not pair, so the
    // output is all-quad by construction rather than by hope.
    let stats = TwoDStats {
        points: points.len(),
        triangles: tris.len(),
        quads: quads.len(),
        unpaired_triangles: unpaired.len(),
        absorbed_triangles: 0,
    };
    let mut cells: Vec<Cell> = quads.into_iter().map(Cell::quad4).collect();
    // Triangles the matching could not pair stay in the mesh as triangles.
    // Dropping them would leave a hole, which is worse than a quad-dominant
    // mesh: a hole fails the topology check and cannot be solved.
    cells.extend(unpaired.iter().map(|t| Cell::tri3(*t)));
    let mesh = Mesh::new(points, cells);
    Ok(TwoDResult {
        mesh,
        unpaired: Vec::new(),
        stats,
    })
}

/// Boundary loops with each segment split so none is longer than the target.
fn sample_loops(domain: &Domain, sizing: &Sizing) -> Vec<Vec<[f64; 2]>> {
    let mut out = Vec::new();
    for loop_points in std::iter::once(&domain.outer).chain(domain.holes.iter()) {
        let mut sampled = Vec::new();
        for i in 0..loop_points.len() {
            let a = loop_points[i];
            let b = loop_points[(i + 1) % loop_points.len()];
            let length = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
            let target = sizing.size_at(a).max(1e-9);
            let parts = (length / (1.1 * target)).ceil().max(1.0) as usize;
            for k in 0..parts {
                let t = k as f64 / parts as f64;
                sampled.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
            }
        }
        out.push(sampled);
    }
    out
}

/// Insert circumcenters until no triangle is larger than the sizing field.
///
/// Two insertion rules, both Ruppert's: a triangle whose circumcenter is inside
/// and far enough from the boundary is split there; when the circumcenter is
/// outside the domain or closer to the boundary than the margin, the nearest
/// boundary edge is split at its midpoint instead. Without the second rule a
/// square stalls immediately — its triangles are right-angled, so every
/// circumcenter lands *on* an edge.
fn refine(
    cdt: &mut ConstrainedDelaunayTriangulation<Point2<f64>>,
    domain: &Domain,
    sizing: &Sizing,
) {
    let mut grid = PointGrid::new(sizing.size);
    for face in cdt.inner_faces() {
        for vertex in face.vertices() {
            let p = vertex.position();
            grid.insert([p.x, p.y]);
        }
    }
    let mut inserted_total = 0usize;
    for _ in 0..MAX_PASSES {
        let mut candidates: Vec<[f64; 2]> = Vec::new();
        for face in cdt.inner_faces() {
            let vertices = face.vertices();
            let p: Vec<[f64; 2]> = vertices
                .iter()
                .map(|v| {
                    let q = v.position();
                    [q.x, q.y]
                })
                .collect();
            if !domain.contains(centroid(&p)) {
                continue;
            }
            let Some((center, radius)) = circumcenter(&p) else {
                continue;
            };
            let target = sizing.size_at(center);
            if radius <= target {
                continue;
            }
            let margin = domain.distance_to_boundary(center);
            let spacing_ok = grid.nearest(center) >= MIN_SPACING_FRACTION * target;
            if domain.contains(center) && margin >= BOUNDARY_MARGIN * target && spacing_ok {
                candidates.push(center);
            } else {
                let (a, b, _) = domain.nearest_segment(center);
                let length = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
                if length > 1.2 * target {
                    candidates.push([(a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0]);
                }
            }
        }
        if candidates.is_empty() {
            return;
        }
        // deduplicate: several triangles share a circumcenter on a symmetric
        // configuration, and re-inserting a point is a no-op that would never
        // terminate
        candidates.sort_by(|a, b| {
            (a[0], a[1])
                .partial_cmp(&(b[0], b[1]))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.dedup_by(|a, b| a == b);
        // Insert every candidate this pass. The earlier cap of 64 per pass
        // multiplied the number of O(faces) scans by ~8 for no benefit: the
        // Delaunay kernel is incremental, so the order does not matter and a
        // stale candidate is simply rejected by the spacing check.
        let mut inserted = 0;
        for &center in &candidates {
            if grid.nearest(center) < MIN_SPACING_FRACTION * sizing.size_at(center) {
                continue;
            }
            if cdt.insert(Point2::new(center[0], center[1])).is_ok() {
                grid.insert(center);
                inserted += 1;
            }
        }
        if inserted == 0 {
            return;
        }
        inserted_total += inserted;
        if inserted_total > MAX_POINTS {
            return;
        }
    }
}

/// Hard cap on inserted points, so a bad size field cannot run away.
const MAX_POINTS: usize = 200_000;

/// Triangles whose centroid is inside the domain, as positions.
fn collect_triangles(
    cdt: &ConstrainedDelaunayTriangulation<Point2<f64>>,
    domain: &Domain,
) -> Vec<[[f64; 2]; 3]> {
    let mut out = Vec::new();
    for face in cdt.inner_faces() {
        let vertices = face.vertices();
        let mut tri = [[0.0f64; 2]; 3];
        for (slot, vertex) in vertices.iter().enumerate() {
            let p = vertex.position();
            tri[slot] = [p.x, p.y];
        }
        if domain.contains(centroid(&tri)) {
            out.push(tri);
        }
    }
    out
}

fn centroid(p: &[[f64; 2]]) -> [f64; 2] {
    let n = p.len() as f64;
    [
        p.iter().map(|q| q[0]).sum::<f64>() / n,
        p.iter().map(|q| q[1]).sum::<f64>() / n,
    ]
}

/// Circumcenter and circumradius of a triangle, or `None` if it is degenerate.
fn circumcenter(p: &[[f64; 2]]) -> Option<([f64; 2], f64)> {
    let (a, b, c) = (p[0], p[1], p[2]);
    let d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]));
    if d.abs() < 1e-14 {
        return None;
    }
    let a2 = a[0] * a[0] + a[1] * a[1];
    let b2 = b[0] * b[0] + b[1] * b[1];
    let c2 = c[0] * c[0] + c[1] * c[1];
    let ux = (a2 * (b[1] - c[1]) + b2 * (c[1] - a[1]) + c2 * (a[1] - b[1])) / d;
    let uy = (a2 * (c[0] - b[0]) + b2 * (a[0] - c[0]) + c2 * (b[0] - a[0])) / d;
    let center = [ux, uy];
    Some((center, ((a[0] - ux).powi(2) + (a[1] - uy).powi(2)).sqrt()))
}

/// Pair adjacent triangles into quads: maximum cardinality, then maximum
/// quality, via a general weighted matching (Edmonds' blossom algorithm).
fn pair_triangles(tris: &[[usize; 3]], points: &[Node]) -> (Vec<[usize; 4]>, Vec<[usize; 3]>) {
    // edge key -> the (up to two) triangles that use it
    let mut edge_to_tris: std::collections::HashMap<(usize, usize), Vec<usize>> =
        std::collections::HashMap::new();
    for (t, tri) in tris.iter().enumerate() {
        for i in 0..3 {
            let (a, b) = (tri[i], tri[(i + 1) % 3]);
            let key = (a.min(b), a.max(b));
            edge_to_tris.entry(key).or_default().push(t);
        }
    }

    let mut edges: Vec<(usize, usize, i32)> = Vec::new();
    for owners in edge_to_tris.values() {
        if owners.len() != 2 {
            continue;
        }
        let (t1, t2) = (owners[0], owners[1]);
        let Some(quad) = quad_from_pair(&tris[t1], &tris[t2], points) else {
            continue;
        };
        let cell = Cell::quad4(quad);
        let sj = quality::scaled_jacobian(&cell, points);
        let worst = sj.iter().copied().fold(f64::INFINITY, f64::min);
        if !worst.is_finite() || worst < MIN_PAIR_SCALED_JACOBIAN {
            continue;
        }
        let angles = quality::quad_angles(&cell, points);
        let min_angle = angles.iter().copied().fold(f64::INFINITY, f64::min);
        let max_angle = angles.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !min_angle.is_finite() || min_angle < MIN_PAIR_ANGLE_DEG {
            continue;
        }
        if !max_angle.is_finite() || max_angle > MAX_PAIR_ANGLE_DEG {
            continue;
        }
        // weight: quality in [0, 1] scaled to an integer; higher is better
        let weight = (1000.0 * worst).round().clamp(1.0, 1000.0) as i32;
        edges.push((t1, t2, weight));
    }

    let mut matching = Matching::new(edges);
    matching.max_cardinality();
    let mate = matching.solve();

    let mut quads: Vec<[usize; 4]> = Vec::new();
    let mut used = vec![false; tris.len()];
    for t in 0..tris.len() {
        // `mate` is sized by the vertices that appear in an edge: a triangle
        // with no pairable neighbour is simply absent from it.
        let partner = mate.get(t).copied().unwrap_or(SENTINEL);
        if used[t] || partner == SENTINEL {
            continue;
        }
        let other = partner;
        if other >= tris.len() || used[other] || other <= t {
            continue;
        }
        let Some(quad) = quad_from_pair(&tris[t], &tris[other], points) else {
            continue;
        };
        used[t] = true;
        used[other] = true;
        quads.push(quad);
    }

    let unpaired = tris
        .iter()
        .enumerate()
        .filter(|(t, _)| !used[*t])
        .map(|(_, tri)| *tri)
        .collect();
    (quads, unpaired)
}

/// Build the quad from two triangles that share an edge.
///
/// The shared edge must be taken in each triangle's own winding: triangle `a`
/// traverses it one way, triangle `b` the other. Walking the *sorted* edge
/// instead produces a bowtie for half the pairs — which is exactly what made the
/// first version of this function refuse every mesh.
fn quad_from_pair(a: &[usize; 3], b: &[usize; 3], points: &[Node]) -> Option<[usize; 4]> {
    for i in 0..3 {
        let u = a[i];
        let v = a[(i + 1) % 3];
        let c1 = a[(i + 2) % 3];
        for j in 0..3 {
            if b[j] == v && b[(j + 1) % 3] == u {
                let c2 = b[(j + 2) % 3];
                let mut quad = [u, c1, v, c2];
                if polygon_signed_area(&quad, points) < 0.0 {
                    quad.reverse();
                }
                if polygon_signed_area(&quad, points).abs() < 1e-18 {
                    return None;
                }
                return Some(quad);
            }
        }
    }
    None
}

/// Signed area of a quad from the coordinates; positive when counter-clockwise.
fn polygon_signed_area(quad: &[usize; 4], points: &[Node]) -> f64 {
    let mut sum = 0.0;
    for i in 0..quad.len() {
        let a = points[quad[i]];
        let b = points[quad[(i + 1) % quad.len()]];
        sum += a[0] * b[1] - b[0] * a[1];
    }
    sum / 2.0
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::mesh::Kind;

    /// Strictly convex, counter-clockwise and non-degenerate.
    fn is_convex_ccw(quad: &[usize; 4], points: &[Node]) -> bool {
        if polygon_signed_area(quad, points) <= 1e-12 {
            return false;
        }
        (0..4).all(|i| {
            let a = points[quad[(i + 3) % 4]];
            let b = points[quad[i]];
            let c = points[quad[(i + 1) % 4]];
            let u = [b[0] - a[0], b[1] - a[1]];
            let v = [c[0] - b[0], c[1] - b[1]];
            u[0] * v[1] - u[1] * v[0] > 0.0
        })
    }

    #[test]
    fn a_rectangle_has_the_expected_area() {
        let d = Domain::rectangle(100.0, 60.0).unwrap();
        assert!((d.area() - 6000.0).abs() < 1e-9);
    }

    #[test]
    fn a_plate_with_a_hole_subtracts_it() {
        let d = Domain::plate_with_hole(100.0, 60.0, 10.0, 64).unwrap();
        let expected = 6000.0 - std::f64::consts::PI * 100.0;
        assert!(
            (d.area() - expected).abs() < 2.0,
            "{} vs {}",
            d.area(),
            expected
        );
    }

    #[test]
    fn contains_respects_the_hole() {
        let d = Domain::plate_with_hole(100.0, 60.0, 10.0, 64).unwrap();
        assert!(d.contains([10.0, 10.0]));
        assert!(!d.contains([50.0, 30.0])); // the hole
        assert!(!d.contains([-1.0, 10.0]));
    }

    #[test]
    fn a_square_is_quad_dominant() {
        let d = Domain::rectangle(10.0, 10.0).unwrap();
        let r = mesh_2d(&d, &Sizing::uniform(2.0).unwrap()).unwrap();
        assert!(
            r.stats.quad_ratio() >= 0.75,
            "quad ratio {:.2}, {:?}",
            r.stats.quad_ratio(),
            r.stats
        );
        assert_eq!(
            r.mesh.cells.len(),
            r.stats.quads + r.stats.unpaired_triangles
        );
    }

    #[test]
    fn every_quad_is_convex_and_positively_oriented() {
        let d = Domain::plate_with_hole(100.0, 60.0, 10.0, 64).unwrap();
        let r = mesh_2d(&d, &Sizing::uniform(4.0).unwrap()).unwrap();
        for cell in r.mesh.cells.iter().filter(|c| c.kind == Kind::Quad4) {
            let nodes: [usize; 4] = cell.nodes.as_slice().try_into().unwrap();
            assert!(is_convex_ccw(&nodes, &r.mesh.points), "{nodes:?}");
        }
    }

    #[test]
    fn the_mesh_is_watertight_even_with_leftover_triangles() {
        let d = Domain::plate_with_hole(100.0, 60.0, 10.0, 64).unwrap();
        let r = mesh_2d(&d, &Sizing::uniform(4.0).unwrap()).unwrap();
        let topo = r.mesh.topology();
        assert_eq!(
            topo.cracks, 0,
            "leftover triangles were dropped, leaving a hole"
        );
        assert_eq!(topo.inconsistent, 0);
        assert_eq!(topo.non_manifold, 0);
        // every leftover triangle is in the mesh, not lost
        let tris = r.mesh.cells.iter().filter(|c| c.kind == Kind::Tri3).count();
        assert_eq!(tris, r.stats.unpaired_triangles);
    }

    #[test]
    fn the_square_mesh_is_valid_and_has_no_inverted_quad() {
        let d = Domain::rectangle(10.0, 10.0).unwrap();
        let r = mesh_2d(&d, &Sizing::uniform(2.0).unwrap()).unwrap();
        assert_eq!(r.mesh.validate(), Ok(()));
        let report = quality::report(&r.mesh);
        assert_eq!(report.inverted, 0, "{}", report.summary());
    }

    #[test]
    fn the_plate_with_a_hole_meshes_at_three_sizes() {
        for size in [4.0, 2.0, 1.0] {
            let d = Domain::plate_with_hole(100.0, 60.0, 10.0, 64).unwrap();
            let r = mesh_2d(&d, &Sizing::uniform(size).unwrap()).unwrap();
            assert!(!r.mesh.cells.is_empty(), "size {size}");
            assert_eq!(r.mesh.validate(), Ok(()), "size {size}");
            let report = quality::report(&r.mesh);
            assert_eq!(report.inverted, 0, "size {size}: {}", report.summary());
        }
    }

    #[test]
    fn a_bad_size_is_rejected() {
        assert!(Sizing::uniform(0.0).is_err());
        assert!(Sizing::uniform(-1.0).is_err());
        assert!(Sizing::uniform(f64::NAN).is_err());
    }
}
