//! Checking a conversion against the W3 gate.
//!
//! The placement policy in the design note (reachable side nodes first, insertion
//! only on a size mismatch, no trial-clone rollback, blocking around progress) is
//! **not enforced by the conversion code**. Nothing in `qmorph` or `quadify`
//! refuses a run that ignores it; the modules only produce a mesh. This module is
//! the check that a claim has to pass: it measures the mesh the run actually
//! produced, so "all-quad on the six parts" is a verdict with numbers behind it
//! rather than a reading of the intent.

use std::collections::HashMap;

use crate::mesh::{Kind, Mesh};
use crate::quality;

/// The W3 acceptance gate, measured on one mesh.
#[derive(Debug, Clone, Copy)]
pub struct Gate {
    /// Quads and triangles in the mesh.
    pub quads: usize,
    pub triangles: usize,
    /// Elements whose scaled Jacobian is not positive.
    pub inverted: usize,
    /// Pairs of cells sharing more than one edge (overlapping cells).
    pub overlapping: usize,
    pub min_scaled_jacobian: f64,
    pub min_corner_angle_deg: f64,
    /// Boundary vertices whose boundary-edge degree is not 2.
    pub cracks: usize,
}

impl Gate {
    /// The bar: all-quad, SJ ≥ 0.2, angle ≥ 30° (relaxed by the user on
    /// 2026-10-22), watertight, no overlapping cells.
    pub fn passes(&self) -> bool {
        self.triangles == 0
            && self.inverted == 0
            && self.overlapping == 0
            && self.cracks == 0
            && self.min_scaled_jacobian >= 0.2
            && self.min_corner_angle_deg >= 30.0
    }

    /// The first reason the gate fails, or `None`.
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

/// Largest distance of a **boundary node** from the domain boundary.
///
/// A conforming planar mesh has every boundary node on the domain polyline, so
/// this is ~1e-15 (floating point). A larger value means the mesh no longer sits
/// on the geometry it was built for — the gate's quality numbers cannot see that,
/// because moving a boundary node inward can even improve them. Measured on the
/// ring: a perturbation pass moved boundary nodes up to 0.1 x size per round and
/// the mesh stuck out of the quarter annulus by ~0.1 mm.
pub fn boundary_offset(domain: &crate::two_d::Domain, mesh: &Mesh) -> f64 {
    let mut count: HashMap<(usize, usize), usize> = HashMap::new();
    for cell in &mesh.cells {
        for i in 0..cell.nodes.len() {
            let a = cell.nodes[i];
            let b = cell.nodes[(i + 1) % cell.nodes.len()];
            *count.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    let mut on_boundary = vec![false; mesh.points.len()];
    for ((a, b), n) in count {
        if n == 1 {
            if a < on_boundary.len() {
                on_boundary[a] = true;
            }
            if b < on_boundary.len() {
                on_boundary[b] = true;
            }
        }
    }
    let mut worst: f64 = 0.0;
    for (i, p) in mesh.points.iter().enumerate() {
        if on_boundary[i] {
            worst = worst.max(domain.distance_to_boundary([p[0], p[1]]));
        }
    }
    worst
}

/// Measure a mesh against the gate.
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

/// True when a cell has non-positive orientation.
///
/// A `Tri3` must be checked by **signed area**: the quality report gives every
/// triangle a scaled Jacobian of 1.0 (there is no corner Jacobian to measure), so
/// a triangle wound clockwise is invisible to it. That hole let a conversion
/// whose recovery flips produced inverted triangles report "no inverted element"
/// — and every later pairing, merge or cavity silently refused those triangles.
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
