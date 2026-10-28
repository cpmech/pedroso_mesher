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
        // A mesh with no 2D cells is not a quad mesh (see `gate`).
        self.quads + self.triangles > 0
            && self.triangles == 0
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
    // The gate is a 2D contract: an empty mesh, or one with no quad/triangle
    // cells (e.g. a pure hex mesh), cannot pass it. `quality::report` returns
    // INFINITY minima for an empty mesh, and every one-sided comparison in
    // `passes` would then hold, so this early return is load-bearing.
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

/// Geometric checks that are too expensive for every smoothing iteration.
///
/// Unlike [`Gate`], this report checks actual polygon intersections, containment
/// in the domain, coincident nodes, geometric T-junctions, non-manifold
/// connectivity and total material area. It is intended for the final API
/// boundary. `overlapping_pairs` counts pairs with positive-area intersection;
/// touching at an edge or vertex is not an overlap. `coincident_points` counts
/// duplicate referenced nodes after the first copy; unused point-storage entries
/// are ignored.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GeometryReport {
    /// Planar cell pairs whose interiors intersect.
    pub overlapping_pairs: usize,
    /// First intersecting cell pair, if any.
    pub first_overlap: Option<(usize, usize)>,
    /// Planar cells with a vertex, edge or interior sample outside the domain.
    pub outside_cells: usize,
    /// First cell extending outside the domain, if any.
    pub first_outside_cell: Option<usize>,
    /// Extra referenced nodes at exactly the same 3D coordinates as an earlier node.
    pub coincident_points: usize,
    /// First pair of coincident referenced nodes, if any.
    pub first_coincident_nodes: Option<(usize, usize)>,
    /// Referenced nodes lying in the interior of a cell edge that does not use them.
    pub t_junctions: usize,
    /// First T-junction as `(node, edge_start, edge_end)`, if any.
    pub first_t_junction: Option<(usize, usize, usize)>,
    /// Edges shared by more than two planar cells.
    pub non_manifold_edges: usize,
    /// First non-manifold edge as its two node indices, if any.
    pub first_non_manifold_edge: Option<(usize, usize)>,
    /// Interior edges traversed in the same direction by both adjacent cells.
    pub inconsistent_edges: usize,
    /// First inconsistently oriented edge as its two node indices, if any.
    pub first_inconsistent_edge: Option<(usize, usize)>,
    /// Cells other than `Quad4` and `Tri3`, which this planar report cannot check.
    pub unsupported_cells: usize,
    /// First unsupported cell, if any.
    pub first_unsupported_cell: Option<usize>,
    /// Absolute difference between mesh area and domain area, in area units.
    pub area_error: f64,
    /// Allowed absolute area difference, relative to the domain area.
    pub area_tolerance: f64,
}

impl GeometryReport {
    /// Whether every geometric acceptance criterion passes.
    pub fn passes(&self) -> bool {
        self.overlapping_pairs == 0
            && self.outside_cells == 0
            && self.coincident_points == 0
            && self.t_junctions == 0
            && self.non_manifold_edges == 0
            && self.inconsistent_edges == 0
            && self.unsupported_cells == 0
            && self.area_error.is_finite()
            && self.area_error <= self.area_tolerance
    }

    /// The first failure, in structural-then-geometric order.
    pub fn reason(&self) -> Option<String> {
        if self.unsupported_cells > 0 {
            return Some(format!(
                "{} cell(s) are not planar quads or triangles",
                self.unsupported_cells
            ));
        }
        if self.non_manifold_edges > 0 {
            return Some(format!("{} non-manifold edge(s)", self.non_manifold_edges));
        }
        if self.inconsistent_edges > 0 {
            return Some(format!(
                "{} inconsistently oriented interior edge(s)",
                self.inconsistent_edges
            ));
        }
        if self.coincident_points > 0 {
            return Some(format!(
                "{} coincident duplicate node(s)",
                self.coincident_points
            ));
        }
        if self.t_junctions > 0 {
            return Some(format!(
                "{} geometric T-junction node/edge incidence(s)",
                self.t_junctions
            ));
        }
        if self.overlapping_pairs > 0 {
            return Some(format!(
                "{} geometrically overlapping cell pair(s)",
                self.overlapping_pairs
            ));
        }
        if self.outside_cells > 0 {
            return Some(format!(
                "{} cell(s) extend outside the domain",
                self.outside_cells
            ));
        }
        if !self.area_error.is_finite() || self.area_error > self.area_tolerance {
            return Some(format!(
                "mesh area differs from domain area by {:.3e} (tolerance {:.3e})",
                self.area_error, self.area_tolerance
            ));
        }
        None
    }

    /// The first failure with its cell, node or edge location when applicable.
    pub fn located_reason(&self) -> Option<String> {
        let reason = self.reason()?;
        let location = if self.unsupported_cells > 0 {
            self.first_unsupported_cell
                .map(|cell| format!("cell {cell}"))
        } else if self.non_manifold_edges > 0 {
            self.first_non_manifold_edge
                .map(|(a, b)| format!("edge {a}-{b}"))
        } else if self.inconsistent_edges > 0 {
            self.first_inconsistent_edge
                .map(|(a, b)| format!("edge {a}-{b}"))
        } else if self.coincident_points > 0 {
            self.first_coincident_nodes
                .map(|(a, b)| format!("nodes {a} and {b}"))
        } else if self.t_junctions > 0 {
            self.first_t_junction
                .map(|(node, a, b)| format!("node {node} on edge {a}-{b}"))
        } else if self.overlapping_pairs > 0 {
            self.first_overlap
                .map(|(a, b)| format!("cells {a} and {b}"))
        } else if self.outside_cells > 0 {
            self.first_outside_cell.map(|cell| format!("cell {cell}"))
        } else if !self.area_error.is_finite() || self.area_error > self.area_tolerance {
            Some("domain coverage".to_string())
        } else {
            None
        };
        Some(match location {
            Some(location) => format!("{reason} ({location})"),
            None => reason,
        })
    }
}

#[derive(Clone, Debug)]
struct CellGeometry {
    vertices: Vec<[f64; 2]>,
    centroid: [f64; 2],
    min: [f64; 2],
    max: [f64; 2],
}

/// Run the full planar geometric acceptance checks.
///
/// Structural validation runs first, so malformed indices return [`MeshError`]
/// instead of reaching geometric code. Area uses a relative `1e-8` tolerance;
/// containment uses `1e-12` of the domain diagonal only to absorb round-off.
/// The check assumes the simple, convex cells required by [`Gate`]. Inverted or
/// self-intersecting cells remain the quality gate's responsibility.
pub fn geometry(
    domain: &crate::two_d::Domain,
    mesh: &Mesh,
) -> Result<GeometryReport, crate::mesh::MeshError> {
    mesh.validate()?;
    let topology = mesh.topology();
    let mut used = vec![false; mesh.points.len()];
    for cell in &mesh.cells {
        for &id in &cell.nodes {
            used[id] = true;
        }
    }
    let mut coincident: HashMap<(u64, u64, u64), usize> = HashMap::new();
    let mut coincident_points = 0usize;
    let mut first_coincident_nodes = None;
    for (id, &p) in mesh.points.iter().enumerate() {
        if !used[id] {
            continue;
        }
        if let Some(&first) = coincident.get(&coordinate_key(p)) {
            coincident_points += 1;
            first_coincident_nodes.get_or_insert((first, id));
        } else {
            coincident.insert(coordinate_key(p), id);
        }
    }

    let mut cells = Vec::with_capacity(mesh.cells.len());
    let mut unsupported_cells = 0usize;
    let mut first_unsupported_cell = None;
    let mut area = 0.0;
    let mut compensation = 0.0;
    for (id, cell) in mesh.cells.iter().enumerate() {
        if cell.kind == Kind::Hex8 {
            unsupported_cells += 1;
            first_unsupported_cell.get_or_insert(id);
            continue;
        }
        let value = quality::volume(cell, &mesh.points).abs();
        let y = value - compensation;
        let t = area + y;
        compensation = (t - area) - y;
        area = t;
        cells.push(cell_geometry(&mesh.points, cell));
    }

    let (overlapping_pairs, first_overlap) = count_geometric_overlaps(&cells);
    let domain_area = domain.area();
    let area_error = if domain_area.is_finite() && domain_area > 0.0 && area.is_finite() {
        (area - domain_area).abs()
    } else {
        f64::INFINITY
    };
    let area_tolerance = if domain_area.is_finite() && domain_area > 0.0 {
        1.0e-8 * domain_area
    } else {
        0.0
    };
    let mut outside_cells = 0usize;
    let mut first_outside_cell = None;
    if unsupported_cells == 0 {
        for id in 0..mesh.cells.len() {
            if cell_extends_outside(domain, &mesh.points, &mesh.cells[id], &cells[id]) {
                outside_cells += 1;
                first_outside_cell.get_or_insert(id);
            }
        }
    }
    let (t_junctions, first_t_junction) = if unsupported_cells == 0 {
        count_t_junctions(mesh, &used, 1.0e-12 * domain_diagonal(domain))
    } else {
        (0, None)
    };

    Ok(GeometryReport {
        overlapping_pairs,
        first_overlap,
        outside_cells,
        first_outside_cell,
        coincident_points,
        first_coincident_nodes,
        t_junctions,
        first_t_junction,
        non_manifold_edges: topology.non_manifold,
        first_non_manifold_edge: if topology.non_manifold > 0 {
            first_non_manifold_edge(mesh)
        } else {
            None
        },
        inconsistent_edges: topology.inconsistent,
        first_inconsistent_edge: if topology.inconsistent > 0 {
            first_inconsistent_edge(mesh)
        } else {
            None
        },
        unsupported_cells,
        first_unsupported_cell,
        area_error,
        area_tolerance,
    })
}

fn first_non_manifold_edge(mesh: &Mesh) -> Option<(usize, usize)> {
    first_special_edge(mesh, true)
}

fn first_inconsistent_edge(mesh: &Mesh) -> Option<(usize, usize)> {
    first_special_edge(mesh, false)
}

fn first_special_edge(mesh: &Mesh, non_manifold: bool) -> Option<(usize, usize)> {
    let mut seen: HashMap<(usize, usize), (usize, (usize, usize))> = HashMap::new();
    for cell in &mesh.cells {
        for i in 0..cell.nodes.len() {
            let a = cell.nodes[i];
            let b = cell.nodes[(i + 1) % cell.nodes.len()];
            let key = (a.min(b), a.max(b));
            let (count, first) = seen.entry(key).or_insert((0, (a, b)));
            *count += 1;
            if (*count == 3 && non_manifold)
                || (*count == 2 && !non_manifold && (a < b) == (first.0 < first.1))
            {
                return Some(key);
            }
        }
    }
    None
}

fn coordinate_key(p: crate::mesh::Node) -> (u64, u64, u64) {
    fn bits(x: f64) -> u64 {
        if x == 0.0 {
            0.0f64.to_bits()
        } else {
            x.to_bits()
        }
    }
    (bits(p[0]), bits(p[1]), bits(p[2]))
}

fn cell_geometry(points: &[crate::mesh::Node], cell: &crate::mesh::Cell) -> CellGeometry {
    let mut vertices = Vec::with_capacity(cell.nodes.len());
    let mut centroid = [0.0; 2];
    let mut min = [f64::INFINITY; 2];
    let mut max = [f64::NEG_INFINITY; 2];
    for &id in &cell.nodes {
        let p = [points[id][0], points[id][1]];
        vertices.push(p);
        centroid[0] += p[0] / cell.nodes.len() as f64;
        centroid[1] += p[1] / cell.nodes.len() as f64;
        min[0] = min[0].min(p[0]);
        min[1] = min[1].min(p[1]);
        max[0] = max[0].max(p[0]);
        max[1] = max[1].max(p[1]);
    }
    CellGeometry {
        vertices,
        centroid,
        min,
        max,
    }
}

fn count_geometric_overlaps(cells: &[CellGeometry]) -> (usize, Option<(usize, usize)>) {
    let mut order: Vec<usize> = (0..cells.len()).collect();
    order.sort_by(|&a, &b| {
        cells[a].min[0]
            .total_cmp(&cells[b].min[0])
            .then_with(|| a.cmp(&b))
    });
    let mut active: Vec<usize> = Vec::new();
    let mut overlapping = 0usize;
    let mut first = None;
    for current in order {
        active.retain(|&other| cells[other].max[0] >= cells[current].min[0]);
        for &other in &active {
            if boxes_overlap(&cells[other], &cells[current])
                && polygons_overlap(&cells[other], &cells[current])
            {
                overlapping += 1;
                first.get_or_insert((other, current));
            }
        }
        active.push(current);
    }
    (overlapping, first)
}

fn boxes_overlap(a: &CellGeometry, b: &CellGeometry) -> bool {
    a.min[0] <= b.max[0] && a.max[0] >= b.min[0] && a.min[1] <= b.max[1] && a.max[1] >= b.min[1]
}

fn count_t_junctions(
    mesh: &Mesh,
    used: &[bool],
    tolerance: f64,
) -> (usize, Option<(usize, usize, usize)>) {
    #[derive(Clone, Copy)]
    struct Edge {
        nodes: [usize; 2],
        a: [f64; 2],
        b: [f64; 2],
        min: [f64; 2],
        max: [f64; 2],
    }

    let mut edges = Vec::with_capacity(mesh.cells.len() * 4);
    for cell in &mesh.cells {
        for i in 0..cell.nodes.len() {
            let na = cell.nodes[i];
            let nb = cell.nodes[(i + 1) % cell.nodes.len()];
            let a = [mesh.points[na][0], mesh.points[na][1]];
            let b = [mesh.points[nb][0], mesh.points[nb][1]];
            edges.push(Edge {
                nodes: [na, nb],
                a,
                b,
                min: [a[0].min(b[0]), a[1].min(b[1])],
                max: [a[0].max(b[0]), a[1].max(b[1])],
            });
        }
    }
    edges.sort_by(|a, b| {
        a.min[0]
            .total_cmp(&b.min[0])
            .then_with(|| a.nodes.cmp(&b.nodes))
    });
    let mut nodes: Vec<usize> = (0..mesh.points.len()).filter(|&id| used[id]).collect();
    nodes.sort_by(|&a, &b| {
        mesh.points[a][0]
            .total_cmp(&mesh.points[b][0])
            .then_with(|| a.cmp(&b))
    });

    let mut active = Vec::new();
    let mut next_edge = 0usize;
    let mut count = 0usize;
    let mut first = None;
    for node in nodes {
        let p = [mesh.points[node][0], mesh.points[node][1]];
        while next_edge < edges.len() && edges[next_edge].min[0] <= p[0] + tolerance {
            active.push(next_edge);
            next_edge += 1;
        }
        active.retain(|&edge| edges[edge].max[0] >= p[0] - tolerance);
        for &edge in &active {
            let e = edges[edge];
            if e.nodes.contains(&node) || p[1] < e.min[1] - tolerance || p[1] > e.max[1] + tolerance
            {
                continue;
            }
            let ab = [e.b[0] - e.a[0], e.b[1] - e.a[1]];
            let ap = [p[0] - e.a[0], p[1] - e.a[1]];
            let length_squared = ab[0] * ab[0] + ab[1] * ab[1];
            if length_squared == 0.0 {
                continue;
            }
            let length = length_squared.sqrt();
            let projection = (ap[0] * ab[0] + ap[1] * ab[1]) / length;
            let distance = (ab[0] * ap[1] - ab[1] * ap[0]).abs() / length;
            if projection > tolerance && projection < length - tolerance && distance <= tolerance {
                count += 1;
                first.get_or_insert((node, e.nodes[0], e.nodes[1]));
            }
        }
    }
    (count, first)
}

fn polygons_overlap(a: &CellGeometry, b: &CellGeometry) -> bool {
    for i in 0..a.vertices.len() {
        let a0 = a.vertices[i];
        let a1 = a.vertices[(i + 1) % a.vertices.len()];
        for j in 0..b.vertices.len() {
            let b0 = b.vertices[j];
            let b1 = b.vertices[(j + 1) % b.vertices.len()];
            if proper_segment_intersection(a0, a1, b0, b1) {
                return true;
            }
        }
    }
    if a.vertices
        .iter()
        .any(|&p| strictly_inside_convex(p, &b.vertices))
        || b.vertices
            .iter()
            .any(|&p| strictly_inside_convex(p, &a.vertices))
    {
        return true;
    }
    strictly_inside_convex(a.centroid, &b.vertices)
        || strictly_inside_convex(b.centroid, &a.vertices)
}

fn cross(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
}

fn proper_segment_intersection(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
    let ab_c = cross(a, b, c);
    let ab_d = cross(a, b, d);
    let cd_a = cross(c, d, a);
    let cd_b = cross(c, d, b);
    ((ab_c > 0.0 && ab_d < 0.0) || (ab_c < 0.0 && ab_d > 0.0))
        && ((cd_a > 0.0 && cd_b < 0.0) || (cd_a < 0.0 && cd_b > 0.0))
}

fn strictly_inside_convex(p: [f64; 2], polygon: &[[f64; 2]]) -> bool {
    strictly_inside_convex_with_tolerance(p, polygon, 0.0)
}

fn strictly_inside_convex_with_tolerance(
    p: [f64; 2],
    polygon: &[[f64; 2]],
    tolerance: f64,
) -> bool {
    let mut sign = 0i8;
    for i in 0..polygon.len() {
        let a = polygon[i];
        let b = polygon[(i + 1) % polygon.len()];
        let value = cross(a, b, p);
        let length = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
        if length == 0.0 || value.abs() <= tolerance * length {
            return false;
        }
        let current = if value > 0.0 { 1 } else { -1 };
        if sign == 0 {
            sign = current;
        } else if sign != current {
            return false;
        }
    }
    sign != 0
}

fn cell_extends_outside(
    domain: &crate::two_d::Domain,
    points: &[crate::mesh::Node],
    cell: &crate::mesh::Cell,
    geometry: &CellGeometry,
) -> bool {
    let tolerance = 1.0e-12 * domain_diagonal(domain);
    let inside = |p: [f64; 2]| domain.contains(p) || domain.distance_to_boundary(p) <= tolerance;
    if cell.nodes.iter().any(|&id| points[id][2].abs() > tolerance)
        || geometry.vertices.iter().any(|&p| !inside(p))
        || !inside(geometry.centroid)
    {
        return true;
    }
    let segments = domain.segments();
    for i in 0..geometry.vertices.len() {
        let a = geometry.vertices[i];
        let b = geometry.vertices[(i + 1) % geometry.vertices.len()];
        let midpoint = [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])];
        if !inside(midpoint)
            || segments
                .iter()
                .any(|&(c, d)| proper_segment_intersection(a, b, c, d))
        {
            return true;
        }
    }
    for &(a, b) in &segments {
        let midpoint = [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])];
        if strictly_inside_convex_with_tolerance(a, &geometry.vertices, tolerance)
            || strictly_inside_convex_with_tolerance(b, &geometry.vertices, tolerance)
            || strictly_inside_convex_with_tolerance(midpoint, &geometry.vertices, tolerance)
        {
            return true;
        }
    }
    false
}

fn domain_diagonal(domain: &crate::two_d::Domain) -> f64 {
    let mut min = [f64::INFINITY; 2];
    let mut max = [f64::NEG_INFINITY; 2];
    for (a, b) in domain.segments() {
        for p in [a, b] {
            min[0] = min[0].min(p[0]);
            min[1] = min[1].min(p[1]);
            max[0] = max[0].max(p[0]);
            max[1] = max[1].max(p[1]);
        }
    }
    ((max[0] - min[0]).powi(2) + (max[1] - min[1]).powi(2))
        .sqrt()
        .max(1.0e-12)
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
    fn an_empty_mesh_does_not_pass_the_gate() {
        // `quality::report` refuses the empty mesh, but the gate copies only the
        // minima (INFINITY) and every one-sided comparison would hold.
        let empty = Mesh::new(Vec::new(), Vec::new());
        assert!(!gate(&empty).passes());
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

    fn two_quad_rectangle() -> (crate::two_d::Domain, Mesh) {
        let domain = crate::two_d::Domain::rectangle(2.0, 1.0).unwrap();
        let mesh = Mesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
            ],
            vec![Cell::quad4([0, 1, 4, 3]), Cell::quad4([1, 2, 5, 4])],
        );
        (domain, mesh)
    }

    #[test]
    fn strict_geometry_accepts_a_shared_edge_and_full_coverage() {
        let (domain, mesh) = two_quad_rectangle();
        let report = geometry(&domain, &mesh).unwrap();
        assert!(report.passes(), "{report:?}");
        assert_eq!(report.overlapping_pairs, 0);
        assert_eq!(report.area_error, 0.0);
    }

    #[test]
    fn strict_geometry_ignores_unused_duplicate_point_entries() {
        let (domain, mut mesh) = two_quad_rectangle();
        mesh.points.push([0.0, 0.0, 0.0]);
        mesh.points.push([1.0, 1.0, 0.0]);
        let report = geometry(&domain, &mesh).unwrap();
        assert_eq!(report.coincident_points, 0);
        assert!(report.passes(), "{report:?}");
    }

    fn t_junction_rectangle(scale: f64, offset: [f64; 2]) -> (crate::two_d::Domain, Mesh) {
        let map = |p: [f64; 2]| [offset[0] + scale * p[0], offset[1] + scale * p[1], 0.0];
        let domain = crate::two_d::Domain::new(
            vec![
                [offset[0], offset[1]],
                [offset[0] + 2.0 * scale, offset[1]],
                [offset[0] + 2.0 * scale, offset[1] + scale],
                [offset[0], offset[1] + scale],
            ],
            vec![],
        )
        .unwrap();
        let points = vec![
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
            [1.0, 0.5],
            [2.0, 0.5],
        ]
        .into_iter()
        .map(map)
        .collect();
        let mesh = Mesh::new(
            points,
            vec![
                Cell::quad4([0, 1, 4, 3]),
                Cell::quad4([1, 2, 7, 6]),
                Cell::quad4([6, 7, 5, 4]),
            ],
        );
        (domain, mesh)
    }

    #[test]
    fn strict_geometry_detects_a_t_junction() {
        let (domain, mesh) = t_junction_rectangle(1.0, [0.0, 0.0]);
        assert_eq!(gate(&mesh).overlapping, 0);
        let report = geometry(&domain, &mesh).unwrap();
        assert_eq!(report.t_junctions, 1);
        assert_eq!(report.first_t_junction, Some((6, 1, 4)));
        assert_eq!(report.overlapping_pairs, 0);
        assert!(report.reason().unwrap_or_default().contains("T-junction"));
        assert!(report
            .located_reason()
            .unwrap_or_default()
            .contains("node 6 on edge 1-4"));
        assert!(!report.passes());
    }

    #[test]
    fn strict_geometry_detects_a_translated_and_scaled_t_junction() {
        let (domain, mesh) = t_junction_rectangle(1.0e6, [-3.0e7, 2.0e7]);
        let report = geometry(&domain, &mesh).unwrap();
        assert_eq!(report.t_junctions, 1);
        assert!(!report.passes());
    }

    #[test]
    fn strict_geometry_detects_overlap_without_shared_node_indices() {
        let domain = crate::two_d::Domain::rectangle(3.0, 3.0).unwrap();
        let mesh = Mesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 2.0, 0.0],
                [0.0, 2.0, 0.0],
                [1.0, 1.0, 0.0],
                [3.0, 1.0, 0.0],
                [3.0, 3.0, 0.0],
                [1.0, 3.0, 0.0],
            ],
            vec![Cell::quad4([0, 1, 2, 3]), Cell::quad4([4, 5, 6, 7])],
        );
        assert_eq!(gate(&mesh).overlapping, 0);
        let report = geometry(&domain, &mesh).unwrap();
        assert_eq!(report.overlapping_pairs, 1);
        assert!(!report.passes());
    }

    #[test]
    fn strict_geometry_detects_a_duplicated_cell_with_new_indices() {
        let domain = crate::two_d::Domain::rectangle(1.0, 1.0).unwrap();
        let points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let mesh = Mesh::new(
            points,
            vec![Cell::quad4([0, 1, 2, 3]), Cell::quad4([4, 5, 6, 7])],
        );
        let report = geometry(&domain, &mesh).unwrap();
        assert_eq!(report.overlapping_pairs, 1);
        assert_eq!(report.coincident_points, 4);
        assert!(!report.passes());
    }

    #[test]
    fn strict_geometry_rejects_missing_domain_area() {
        let domain = crate::two_d::Domain::rectangle(2.0, 2.0).unwrap();
        let report = geometry(&domain, &unit_quad()).unwrap();
        assert_eq!(report.overlapping_pairs, 0);
        assert_eq!(report.outside_cells, 0);
        assert_eq!(report.area_error, 3.0);
        assert!(!report.passes());
        assert!(report.reason().unwrap_or_default().contains("area"));
    }

    #[test]
    fn strict_geometry_rejects_a_cell_outside_with_the_right_area() {
        let domain = crate::two_d::Domain::rectangle(1.0, 1.0).unwrap();
        let mesh = Mesh::new(
            vec![
                [2.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [3.0, 1.0, 0.0],
                [2.0, 1.0, 0.0],
            ],
            vec![Cell::quad4([0, 1, 2, 3])],
        );
        let report = geometry(&domain, &mesh).unwrap();
        assert!(report.area_error < 1.0e-12);
        assert_eq!(report.outside_cells, 1);
        assert!(!report.passes());
    }

    #[test]
    fn strict_geometry_rejects_a_cell_covering_a_hole() {
        let domain = crate::two_d::Domain::new(
            vec![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
            vec![vec![[1.5, 1.5], [2.5, 1.5], [2.5, 2.5], [1.5, 2.5]]],
        )
        .unwrap();
        let mesh = Mesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [4.0, 4.0, 0.0],
                [0.0, 4.0, 0.0],
            ],
            vec![Cell::quad4([0, 1, 2, 3])],
        );
        let report = geometry(&domain, &mesh).unwrap();
        assert_eq!(report.outside_cells, 1);
        assert_eq!(report.area_error, 1.0);
        assert!(!report.passes());
    }

    #[test]
    fn strict_geometry_rejects_non_manifold_edges() {
        let domain = crate::two_d::Domain::rectangle(1.0, 1.0).unwrap();
        let mesh = Mesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 1.0, 0.0],
                [0.5, -1.0, 0.0],
            ],
            vec![
                Cell::tri3([0, 1, 2]),
                Cell::tri3([1, 0, 3]),
                Cell::tri3([0, 1, 3]),
            ],
        );
        let report = geometry(&domain, &mesh).unwrap();
        assert_eq!(report.non_manifold_edges, 1);
        assert_eq!(report.first_non_manifold_edge, Some((0, 1)));
        assert!(report
            .located_reason()
            .unwrap_or_default()
            .contains("edge 0-1"));
        assert!(!report.passes());
    }

    #[test]
    fn strict_geometry_locates_inconsistent_orientation() {
        let (domain, mut mesh) = two_quad_rectangle();
        mesh.cells[1].nodes = vec![1, 4, 5, 2];
        let report = geometry(&domain, &mesh).unwrap();
        assert_eq!(report.inconsistent_edges, 1);
        assert_eq!(report.first_inconsistent_edge, Some((1, 4)));
        assert!(report
            .located_reason()
            .unwrap_or_default()
            .contains("edge 1-4"));
    }

    #[test]
    fn strict_geometry_locates_area_failure_at_domain_level() {
        let domain = crate::two_d::Domain::rectangle(2.0, 2.0).unwrap();
        let report = geometry(&domain, &unit_quad()).unwrap();
        assert!(report
            .located_reason()
            .unwrap_or_default()
            .contains("domain coverage"));
    }

    #[test]
    fn strict_geometry_treats_a_shared_hole_edge_as_boundary() {
        let domain = crate::two_d::Domain::plate_with_hole(100.0, 60.0, 10.0, 16).unwrap();
        let mesh = Mesh::new(
            vec![
                [50.0, 40.0, 0.0],
                [53.8268343236509, 39.23879532511287, 0.0],
                [51.431769192423, 45.05907372400746, 0.0],
            ],
            vec![Cell::tri3([0, 1, 2])],
        );
        let report = geometry(&domain, &mesh).unwrap();
        assert_eq!(report.outside_cells, 0);
    }

    #[test]
    fn qmorph_smoothing_preserves_plate_hole_coverage() {
        let domain = crate::two_d::Domain::plate_with_hole(100.0, 60.0, 10.0, 16).unwrap();
        let sizing = crate::two_d::Sizing::uniform(4.0).unwrap();
        let mut config = crate::config::MeshConfig::new(crate::config::Route::Qmorph);
        config.smoothing.enabled = false;
        let out = crate::config::generate(&domain, &sizing, &config).unwrap();
        let report = geometry(&domain, &out.mesh).unwrap();
        assert!(report.passes(), "{report:?}");
    }

    #[test]
    fn strict_geometry_validates_indices_before_geometric_access() {
        let domain = crate::two_d::Domain::rectangle(1.0, 1.0).unwrap();
        let mesh = Mesh::new(vec![[0.0, 0.0, 0.0]; 4], vec![Cell::quad4([0, 1, 2, 9])]);
        assert_eq!(
            geometry(&domain, &mesh),
            Err(crate::mesh::MeshError::IndexOutOfRange {
                cell: 0,
                node: 3,
                index: 9,
            })
        );
    }
}
