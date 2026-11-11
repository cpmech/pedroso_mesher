//! Universal quality-constrained 2D triangular FEA mesh generator (`Tri3`, `Tri6`).

use crate::core::error::{Error, Result};
use crate::core::high_order::elevate_to_quadratic;
use crate::core::mesh::{Cell, Mesh};
use crate::papers::foundations::shewchuk_2002::{self, Domain, QualityRequest};

/// Generate a quality-constrained 2D triangular mesh (`Tri3`, or `Tri6` when `quadratic == true`).
pub fn generate_triangles<F>(
    domain: &Domain,
    size_field: F,
    request: QualityRequest,
    quadratic: bool,
) -> Result<Mesh>
where
    F: Fn([f64; 2]) -> f64,
{
    let refined = shewchuk_2002::triangulate(domain, size_field, request)
        .map_err(|e| Error::Algorithm(e.to_string()))?;
    let tri = refined.triangulation();
    let points: Vec<[f64; 3]> = tri.points().iter().map(|&[x, y]| [x, y, 0.0]).collect();
    let mut cells = Vec::with_capacity(tri.triangles().len());
    let mut cell_markers = Vec::with_capacity(tri.triangles().len());
    for t in tri.triangles() {
        cells.push(Cell::tri3(t.vertices));
        cell_markers.push(t.material);
    }
    let mut mesh = Mesh::new(points, cells);
    mesh.cell_markers = cell_markers;
    for c in tri.constraints() {
        if c.marker != 0 {
            mesh.marked_edges.push((c.marker, c.vertices));
        }
    }
    if quadratic {
        elevate_to_quadratic(&mesh)
    } else {
        Ok(mesh)
    }
}
