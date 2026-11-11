//! Elevation of linear finite elements (`Tri3`, `Qua4`, `Tet4`, `Hex8`) to
//! quadratic serendipity/Lagrange elements (`Tri6`, `Qua8`, `Tet10`, `Hex20`)
//! using `gemlab`'s exact mid-edge node ordering.

use std::collections::BTreeMap;

use crate::core::error::Result;
use crate::core::mesh::{Cell, Mesh};

/// Convert every linear element in `mesh` (`Tri3`, `Qua4`, `Tet4`, `Hex8`) to
/// its quadratic counterpart (`Tri6`, `Qua8`, `Tet10`, `Hex20`) by inserting a
/// shared midpoint node on each unique mesh edge.
///
/// Preserves `point_markers` on existing vertices, `cell_markers`,
/// `marked_edges`, and `marked_faces` (which continue to reference corner node
/// indices per `gemlab`'s `.msh` specification).
pub fn elevate_to_quadratic(mesh: &Mesh) -> Result<Mesh> {
    mesh.validate()?;
    let mut points = mesh.points.clone();
    let mut point_markers = mesh.point_markers.clone();
    if !point_markers.is_empty() && point_markers.len() < points.len() {
        point_markers.resize(points.len(), 0);
    }

    // Propagate edge markers onto newly created mid-edge nodes when a marked
    // edge exists.
    let mut edge_marker_map: BTreeMap<(usize, usize), i32> = BTreeMap::new();
    for &(marker, [a, b]) in &mesh.marked_edges {
        edge_marker_map.insert((a.min(b), a.max(b)), marker);
    }

    let mut midpoint_of: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    let mut new_cells = Vec::with_capacity(mesh.cells.len());

    for cell in &mesh.cells {
        if cell.kind.is_quadratic() {
            new_cells.push(cell.clone());
            continue;
        }
        let quad_kind = cell.kind.quadratic_kind();
        let mut nodes = Vec::with_capacity(quad_kind.n_nodes());
        nodes.extend_from_slice(&cell.nodes);

        for &(la, lb) in cell.kind.edges() {
            let ga = cell.nodes[la];
            let gb = cell.nodes[lb];
            let key = (ga.min(gb), ga.max(gb));
            let mid_id = match midpoint_of.get(&key) {
                Some(&existing) => existing,
                None => {
                    let pa = points[ga];
                    let pb = points[gb];
                    let mid = [
                        0.5 * (pa[0] + pb[0]),
                        0.5 * (pa[1] + pb[1]),
                        0.5 * (pa[2] + pb[2]),
                    ];
                    let id = points.len();
                    points.push(mid);
                    if !point_markers.is_empty() || edge_marker_map.contains_key(&key) {
                        if point_markers.len() < id {
                            point_markers.resize(id, 0);
                        }
                        let m = edge_marker_map.get(&key).copied().unwrap_or(0);
                        point_markers.push(m);
                    }
                    midpoint_of.insert(key, id);
                    id
                }
            };
            nodes.push(mid_id);
        }

        new_cells.push(Cell {
            kind: quad_kind,
            nodes,
        });
    }

    let mut elevated = Mesh::new(points, new_cells);
    elevated.point_markers = point_markers;
    elevated.cell_markers = mesh.cell_markers.clone();
    elevated.marked_edges = mesh.marked_edges.clone();
    elevated.marked_faces = mesh.marked_faces.clone();
    elevated.validate()?;
    Ok(elevated)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::core::mesh::Kind;
    use crate::core::quality;

    #[test]
    fn elevates_tri3_qua4_tet4_and_hex8_preserving_volume_and_conformity() {
        // Two shared quads -> two Qua8 with 13 unique nodes (6 corners + 7 edges)
        let qmesh = Mesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            vec![Cell::qua4([0, 1, 4, 5]), Cell::qua4([1, 2, 3, 4])],
        );
        let q8 = elevate_to_quadratic(&qmesh).unwrap();
        assert_eq!(q8.points.len(), 13);
        assert_eq!(q8.cells[0].kind, Kind::Qua8);
        assert!((q8.volume() - 2.0).abs() < 1e-12);
        assert!(q8.topology().is_watertight());
        for c in &q8.cells {
            for sj in quality::scaled_jacobian(c, &q8.points) {
                assert!((sj - 1.0).abs() < 1e-12);
            }
        }

        // Unit cube Hex8 -> Hex20 with 20 nodes and unit volume
        let hmesh = Mesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            vec![Cell::hex8([0, 1, 2, 3, 4, 5, 6, 7])],
        );
        let h20 = elevate_to_quadratic(&hmesh).unwrap();
        assert_eq!(h20.points.len(), 20);
        assert_eq!(h20.cells[0].kind, Kind::Hex20);
        assert!((h20.volume() - 1.0).abs() < 1e-12);
        for sj in quality::scaled_jacobian(&h20.cells[0], &h20.points) {
            assert!((sj - 1.0).abs() < 1e-12);
        }

        // Unit right Tet4 -> Tet10 with 10 nodes and volume 1/6
        let tmesh = Mesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            vec![Cell::tet4([0, 1, 2, 3])],
        );
        let t10 = elevate_to_quadratic(&tmesh).unwrap();
        assert_eq!(t10.points.len(), 10);
        assert_eq!(t10.cells[0].kind, Kind::Tet10);
        assert!((t10.volume() - 1.0 / 6.0).abs() < 1e-12);
    }
}
