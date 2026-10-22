//! Acceptance gate and geometric conformity checks for a converted mesh.
//!
//! The placement policy in the design note (reachable side nodes first,
//! insertion only on a size mismatch, no trial-clone rollback, blocking around
//! progress) is **not enforced by the conversion code**. Nothing in `qmorph`,
//! `quadify` or the `alt_*` routes refuses a run that ignores it; those modules
//! only produce a mesh. This module is the check that a claim has to pass: it
//! measures the mesh the run actually produced, so "all-quad on the six parts"
//! is a verdict with numbers behind it rather than a reading of the intent.
//!
//! Two kinds of check live here:
//!
//! * [`gate`] measures a mesh against the W3 contract (all-quad, positive
//!   scaled Jacobian, no overlapping cells, no cracks, minimum scaled Jacobian
//!   and minimum corner angle). The quality numbers follow Knupp's Jacobian
//!   framework (`knupp-2000-part1` for the planar 2x2 case,
//!   `knupp-2000-part2` for the 3x3 case); see [`crate::quality`].
//! * [`boundary_offset`] and [`corner_offset`] measure geometric conformity,
//!   which the gate's quality numbers cannot see: whether the mesh still sits
//!   on the domain it was built for. These are standard distance checks, not
//!   an algorithm from the corpus.
//!
//! What is approximated or missing: the gate is a 2D contract. It does not
//! measure the 3D metrics (dihedral angle, aspect ratio) that
//! [`crate::quality::report`] computes. It does not prove that the mesh covers
//! the whole domain, only that boundary nodes lie on the boundary and that every
//! sharp corner keeps a node. It cannot see a non-conforming seam built from
//! duplicated nodes, because both patches then have a well-formed boundary of
//! their own (see `overlapping_pairs`). And the placement policy itself is not
//! checked at all: the gate is an outcome, not a description of the run.

use std::collections::HashMap;

use crate::mesh::{Kind, Mesh};
use crate::quality;

/// The W3 acceptance gate, measured on one mesh.
///
/// Every field is a measurement of the mesh that was actually produced, never a
/// claim about the algorithm that produced it. The type is `Copy` and derives
/// `Serialize`/`Deserialize` because it travels in the run statistics and in
/// the CLI's JSON output.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Gate {
    /// Number of `Quad4` cells. Informational: the all-quad rule is
    /// `triangles == 0`, not a target for this count.
    pub quads: usize,

    /// Number of `Tri3` cells. Must be zero: W3 promises an all-quad mesh on
    /// the converted parts, so a leftover triangle is a failure, not a variant.
    pub triangles: usize,

    /// Elements with non-positive orientation: for a `Quad4`/`Hex8` any corner
    /// whose scaled Jacobian is `<= 0` (tangled or folded), and for a `Tri3` a
    /// signed area `<= 0` (see `cell_inverted`). Must be zero.
    pub inverted: usize,

    /// Pairs of cells that share more than one edge. Two cells of a conforming
    /// mesh share at most one edge, so a count above one means they overlap.
    /// Must be zero.
    pub overlapping: usize,

    /// Smallest scaled Jacobian over every corner of every cell; dimensionless,
    /// in `[-1, 1]`, with 1 for a right-angled cell. Must be `>= 0.2`.
    /// Triangles report `1.0` here (see [`crate::quality::scaled_jacobian`]);
    /// their orientation is caught by `inverted` instead.
    pub min_scaled_jacobian: f64,

    /// Smallest corner angle over the planar cells, in degrees. Must be
    /// `>= 30.0` (the user relaxed the earlier bar on 2026-10-22). The value is
    /// NaN when the mesh has no planar cells, and then the gate fails closed.
    pub min_corner_angle_deg: f64,

    /// Boundary vertices whose boundary-edge degree is not 2, i.e. the boundary
    /// polyline does not close. Must be zero.
    pub cracks: usize,
}

impl Gate {
    /// The bar: all-quad, SJ ≥ 0.2, angle ≥ 30° (relaxed by the user on
    /// 2026-10-22), watertight, no overlapping cells.
    ///
    /// The exact rules, in the order [`Gate::reason`] reports them:
    ///
    /// 1. `triangles == 0` — every cell is a quad;
    /// 2. `inverted == 0` — no cell has a non-positive orientation anywhere;
    /// 3. `overlapping == 0` — no cell pair shares more than one edge;
    /// 4. `cracks == 0` — the boundary polyline closes (no boundary vertex of
    ///    degree other than 2);
    /// 5. `min_scaled_jacobian >= 0.2`;
    /// 6. `min_corner_angle_deg >= 30.0`.
    ///
    /// This is a **floor, not a description**. The test is one-sided: it only
    /// asks whether the worst corner clears the minima and whether the topology
    /// counts are zero. It can reject a mesh, but it cannot say how good an
    /// accepted mesh is — every cell may sit exactly at SJ 0.2 and 30°, the
    /// aspect ratio may be large, and warpage/skew are not measured at all. The
    /// gate is the property every shippable run must have, not a summary of the
    /// run. It is also not the placement policy: a route that ignores the
    /// policy can still pass on these numbers alone. That is why
    /// `stats::Stats::margin` treats the gate as the floor and why the stats
    /// carry distributions alongside it.
    pub fn passes(&self) -> bool {
        self.triangles == 0
            && self.inverted == 0
            && self.overlapping == 0
            && self.cracks == 0
            && self.min_scaled_jacobian >= 0.2
            && self.min_corner_angle_deg >= 30.0
    }

    /// The first reason the gate fails, or `None` when it passes.
    ///
    /// The order is topology first (triangles, inversion, overlap, cracks), then
    /// the two quality minima, matching [`Gate::passes`] so that the message
    /// names the most structural failure first. Only the first failure is
    /// returned; fixing it may reveal the next one.
    pub fn reason(&self) -> Option<String> {
        if self.triangles > 0 {
            return Some(format!("{} triangle(s) left", self.triangles));
        }
        if self.inverted > 0 {
            return Some(format!("{} inverted element(s)", self.inverted));
        }
        if self.overlapping > 0 {
            return Some(format!("{} overlapping cell pair(s)", self.overlapping));
        }
        if self.cracks > 0 {
            return Some(format!("{} crack(s)", self.cracks));
        }
        if self.min_scaled_jacobian < 0.2 {
            return Some(format!(
                "min scaled Jacobian {:.3} < 0.2",
                self.min_scaled_jacobian
            ));
        }
        if self.min_corner_angle_deg < 30.0 {
            return Some(format!(
                "min corner angle {:.1}° < 30°",
                self.min_corner_angle_deg
            ));
        }
        None
    }
}

/// Which nodes are on the mesh boundary (an edge used by exactly one cell).
///
/// Edge keys are undirected (`min`, `max`), so the two cells on either side of
/// an interior edge both count the same key; a key seen once is a boundary
/// edge. Only the consecutive node pairs of each cell are treated as edges,
/// which is right for `Quad4` and `Tri3` but **not** for `Hex8` (whose edges
/// are given by `Kind::edges`); these conformity helpers are for planar meshes.
fn boundary_node_mask(mesh: &Mesh) -> Vec<bool> {
    let mut count: HashMap<(usize, usize), usize> = HashMap::new();
    for cell in &mesh.cells {
        // Walk the cell's node ring; the last node closes back onto the first.
        for i in 0..cell.nodes.len() {
            let a = cell.nodes[i];
            let b = cell.nodes[(i + 1) % cell.nodes.len()];
            *count.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    // A node is on the boundary when at least one incident edge is used once.
    // The `mask.len()` guards are unnecessary for a validated mesh, but keep a
    // malformed mesh from panicking here.
    let mut mask = vec![false; mesh.points.len()];
    for ((a, b), n) in count {
        if n == 1 {
            if a < mask.len() {
                mask[a] = true;
            }
            if b < mask.len() {
                mask[b] = true;
            }
        }
    }
    mask
}

/// The boundary nodes as 2D points, with z dropped.
///
/// Planar meshes live in z = 0 by convention (see [`crate::mesh`]), so
/// [`boundary_offset`] measures in the xy plane and only needs two coordinates.
fn boundary_points(mesh: &Mesh) -> Vec<[f64; 2]> {
    let mask = boundary_node_mask(mesh);
    mesh.points
        .iter()
        .enumerate()
        .filter(|(i, _)| mask[*i])
        .map(|(_, p)| [p[0], p[1]])
        .collect()
}

/// Largest distance of a **boundary node** from the domain boundary, in the
/// domain's length unit (millimetres).
///
/// A conforming planar mesh has every boundary node on the domain polyline, so
/// this is ~1e-15 (floating point). A larger value means the mesh no longer sits
/// on the geometry it was built for — the gate's quality numbers cannot see that,
/// because moving a boundary node inward can even improve them. Measured on the
/// ring: a perturbation pass moved boundary nodes up to 1.92 x size off the
/// domain. `stats::Stats` compares the value against `1e-6 * diagonal` to decide
/// whether the run is conforming.
///
/// The measure is one-sided: it only says that boundary nodes are on the
/// polyline. It does not say that every part of the polyline is covered — a
/// whole domain corner can be missing while every remaining node is still on an
/// edge (that is why [`corner_offset`] exists) — and an interior node outside
/// the domain is not a boundary node and is not measured.
pub fn boundary_offset(domain: &crate::two_d::Domain, mesh: &Mesh) -> f64 {
    boundary_points(mesh)
        .into_iter()
        .map(|p| domain.distance_to_boundary(p))
        .fold(0.0, f64::max)
}

/// Largest distance from a **sharp domain corner** to the nearest boundary node,
/// in the domain's length unit (millimetres).
///
/// `boundary_offset` only says the nodes are on the polyline; a node sitting on
/// a corner can slide along an adjacent edge and stay on the polyline while the
/// corner is cut away. Measured: a corner node slid 1.0 along an edge while the
/// boundary offset stayed 0. Every sharp corner must keep a mesh node, so this
/// check closes that hole.
///
/// Only corners returned by `Domain::sharp_corners` count: a shallow polyline
/// vertex (for example one vertex of a 24-gon circle approximation) is not a
/// feature and dropping it is not a geometry error. Returns `INFINITY` (fail
/// closed) when the mesh has no boundary node at all, and `0.0` when the domain
/// has no sharp corner.
pub fn corner_offset(domain: &crate::two_d::Domain, mesh: &Mesh) -> f64 {
    let boundary = boundary_points(mesh);
    if boundary.is_empty() {
        return f64::INFINITY;
    }
    // For each feature corner, take the distance to the closest boundary node,
    // then keep the worst corner. A large `worst` means at least one domain
    // corner has no node near it.
    let mut worst: f64 = 0.0;
    for c in domain.sharp_corners() {
        let mut best = f64::INFINITY;
        for p in &boundary {
            let d = ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2)).sqrt();
            if d < best {
                best = d;
            }
        }
        worst = worst.max(best);
    }
    worst
}

/// Measure a mesh against the gate.
///
/// One pass over the cells and one over the edges: the quality minima come from
/// [`crate::quality::report`], the topology counts from [`Mesh::topology`], and
/// the cell types and inversions are counted here. The result is a snapshot of
/// this mesh only; the gate is not cumulative and compares nothing across runs.
pub fn gate(mesh: &Mesh) -> Gate {
    let report = quality::report(mesh);
    let topo = mesh.topology();
    let quads = mesh.cells.iter().filter(|c| c.kind == Kind::Quad4).count();
    let triangles = mesh.cells.iter().filter(|c| c.kind == Kind::Tri3).count();
    let inverted = mesh
        .cells
        .iter()
        .filter(|c| cell_inverted(c, &mesh.points))
        .count();
    Gate {
        quads,
        triangles,
        inverted,
        overlapping: overlapping_pairs(mesh),
        min_scaled_jacobian: report.min_scaled_jacobian,
        min_corner_angle_deg: report.min_corner_angle_deg,
        cracks: topo.cracks,
    }
}

/// True when a cell has non-positive orientation, at any corner.
///
/// A `Quad4` or `Hex8` is inverted when any corner's scaled Jacobian is `<= 0`
/// (the folded-corner test of `knupp-2000-part1`/`part2`). A `Tri3` must be
/// checked by **signed area** instead: the quality report gives every triangle a
/// scaled Jacobian of 1.0 (there is no parametric corner Jacobian to measure),
/// so a triangle wound clockwise is invisible to it. That hole let a conversion
/// whose recovery flips produced inverted triangles report "no inverted element"
/// — and every later pairing, merge or cavity silently refused those triangles.
///
/// The signed area is the z component of the edge cross product,
/// `(b-a) x (c-a) / 2` without the division, which is positive for the
/// counter-clockwise order of [`crate::mesh::Kind`]. `<= 0` includes the
/// degenerate (collinear) case: a zero-area triangle is not a valid cell either.
fn cell_inverted(cell: &crate::mesh::Cell, points: &[crate::mesh::Node]) -> bool {
    if cell.kind == Kind::Tri3 {
        let (a, b, c) = (
            points[cell.nodes[0]],
            points[cell.nodes[1]],
            points[cell.nodes[2]],
        );
        let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
        return area <= 0.0;
    }
    quality::scaled_jacobian(cell, points)
        .iter()
        .any(|&s| s <= 0.0)
}

/// Pairs of cells that share more than one edge. Such cells overlap; the
/// boundary-degree and Jacobian checks cannot see it.
///
/// The first pass maps an undirected edge to the ids of the cells that use it.
/// The second counts, for every cell pair, how many edges they share; a count
/// above one is one overlapping pair (the count is per pair, not per extra
/// edge). Two conforming cells share at most one edge, so this catches duplicated
/// or stacked cells, not a general geometric intersection between cells that
/// share no edge and use distinct nodes — that would need a polygon-intersection
/// test this check does not do. Like [`boundary_node_mask`], it walks the
/// consecutive node ring and is therefore a planar (quad/triangle) check.
fn overlapping_pairs(mesh: &Mesh) -> usize {
    let mut edges: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (id, cell) in mesh.cells.iter().enumerate() {
        for i in 0..cell.nodes.len() {
            let (a, b) = (cell.nodes[i], cell.nodes[(i + 1) % cell.nodes.len()]);
            edges.entry((a.min(b), a.max(b))).or_default().push(id);
        }
    }
    let mut shared: HashMap<(usize, usize), usize> = HashMap::new();
    for owners in edges.values() {
        for i in 0..owners.len() {
            for j in (i + 1)..owners.len() {
                let pair = (owners[i].min(owners[j]), owners[i].max(owners[j]));
                *shared.entry(pair).or_default() += 1;
            }
        }
    }
    shared.values().filter(|&&count| count > 1).count()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::mesh::Cell;

    fn unit_quad() -> Mesh {
        Mesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            vec![Cell::quad4([0, 1, 2, 3])],
        )
    }

    #[test]
    fn a_single_square_passes_the_gate() {
        let g = gate(&unit_quad());
        assert_eq!(g.triangles, 0);
        assert_eq!(g.cracks, 0);
        assert!(g.min_corner_angle_deg >= 89.0);
        assert!(g.passes(), "{g:?} {:?}", g.reason());
    }

    #[test]
    fn a_leftover_triangle_fails_the_gate() {
        let mut mesh = unit_quad();
        mesh.cells.push(Cell::tri3([0, 1, 0]));
        let g = gate(&mesh);
        assert_eq!(g.triangles, 1);
        assert!(!g.passes());
        assert!(g.reason().unwrap_or_default().contains("triangle"));
    }

    #[test]
    fn a_reversed_quad_counts_as_inverted() {
        let mut mesh = unit_quad();
        mesh.cells[0] = Cell::quad4([1, 0, 3, 2]);
        let g = gate(&mesh);
        assert_eq!(g.inverted, 1);
        assert!(!g.passes());
        assert!(g.reason().unwrap_or_default().contains("inverted"));
    }

    #[test]
    fn a_boundary_node_off_the_domain_is_measured() {
        use crate::two_d::Domain;
        let domain = Domain::rectangle(1.0, 1.0).unwrap();
        let mesh = Mesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [0.5, 0.25, 0.0], // pulled inside the bottom edge
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            vec![Cell::quad4([0, 1, 2, 3])],
        );
        let offset = boundary_offset(&domain, &mesh);
        assert!((offset - 0.25).abs() < 1.0e-12, "{offset}");
        let on = Mesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            vec![Cell::quad4([0, 1, 2, 3])],
        );
        assert!(boundary_offset(&domain, &on) < 1.0e-12);
    }

    #[test]
    fn a_domain_corner_cut_away_is_measured() {
        use crate::two_d::Domain;
        let domain = Domain::rectangle(1.0, 1.0).unwrap();
        // The corner (0,0) node slid up the left edge: still on the polyline,
        // so the boundary offset is 0, but the corner is cut.
        let mesh = Mesh::new(
            vec![
                [0.0, 0.3, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            vec![Cell::quad4([0, 1, 2, 3])],
        );
        assert!(boundary_offset(&domain, &mesh) < 1.0e-12);
        let d = corner_offset(&domain, &mesh);
        assert!((d - 0.3).abs() < 1.0e-12, "{d}");
        let on = Mesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            vec![Cell::quad4([0, 1, 2, 3])],
        );
        assert!(corner_offset(&domain, &on) < 1.0e-12);
    }

    #[test]
    fn a_shallow_boundary_vertex_is_not_a_corner() {
        use crate::two_d::Domain;
        // A 24-gon approximation of a circle: every vertex turns 15 degrees, so
        // none of them is a feature and dropping one is not a geometry error.
        let pts: Vec<[f64; 2]> = (0..24)
            .map(|i| {
                let a = std::f64::consts::TAU * i as f64 / 24.0;
                [1.0 + a.cos(), 1.0 + a.sin()]
            })
            .collect();
        let domain = Domain::new(pts.clone(), vec![]).unwrap();
        assert!(domain.sharp_corners().is_empty());
        let mesh = Mesh::new(
            vec![
                [pts[0][0], pts[0][1], 0.0],
                [pts[1][0], pts[1][1], 0.0],
                [1.0, 1.0, 0.0],
            ],
            vec![Cell::tri3([0, 1, 2])],
        );
        assert!(corner_offset(&domain, &mesh) < 1.0e-12);
    }

    #[test]
    fn a_reversed_triangle_counts_as_inverted() {
        let mut mesh = unit_quad();
        mesh.cells.push(Cell::tri3([0, 2, 1]));
        let g = gate(&mesh);
        assert_eq!(g.inverted, 1, "{g:?}");
        assert!(!g.passes());
    }

    #[test]
    fn overlapping_cells_fail_the_gate() {
        // Two identical quads share four edges and overlap completely.
        let mesh = Mesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            vec![Cell::quad4([0, 1, 2, 3]), Cell::quad4([0, 1, 2, 3])],
        );
        let g = gate(&mesh);
        assert_eq!(g.overlapping, 1, "{g:?}");
        assert!(!g.passes());
        assert!(g.reason().unwrap_or_default().contains("overlapping"));
    }

    #[test]
    fn the_reason_lists_the_first_failure() {
        let mut mesh = unit_quad();
        mesh.cells.push(Cell::tri3([0, 1, 2]));
        let g = gate(&mesh);
        assert_eq!(g.reason().unwrap_or_default(), "1 triangle(s) left");
    }
}
