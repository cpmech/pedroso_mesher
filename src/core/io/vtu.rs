//! VTK UnstructuredGrid XML (`.vtu`) writer for ParaView visualization.
//!
//! Exports all eight FEA element kinds (`Tri3`, `Tri6`, `Qua4`, `Qua8`,
//! `Tet4`, `Tet10`, `Hex8`, `Hex20`) together with cell region markers and
//! minimum per-cell scaled Jacobian.

use std::fmt::Write as _;
use std::path::Path;

use crate::mesh::{Kind, Mesh};
use crate::quality;

/// VTK unstructured cell type code (VTK file formats specification).
pub fn vtk_cell_type(kind: Kind) -> u8 {
    match kind {
        Kind::Tri3 => 5,
        Kind::Qua4 => 9,
        Kind::Tet4 => 10,
        Kind::Hex8 => 12,
        Kind::Tri6 => 22,
        Kind::Qua8 => 23,
        Kind::Tet10 => 24,
        Kind::Hex20 => 25,
    }
}

/// Render a validated [`Mesh`] as an ASCII VTK XML UnstructuredGrid (`.vtu`).
pub fn to_vtu(mesh: &Mesh) -> Result<String, crate::mesh::MeshError> {
    mesh.validate()?;
    let mut xml = String::with_capacity(512 + mesh.points.len() * 40 + mesh.cells.len() * 48);
    xml.push_str("<?xml version=\"1.0\"?>\n");
    xml.push_str(
        "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n",
    );
    xml.push_str("  <UnstructuredGrid>\n");
    let _ = writeln!(
        xml,
        "    <Piece NumberOfPoints=\"{}\" NumberOfCells=\"{}\">",
        mesh.points.len(),
        mesh.cells.len()
    );

    // Points
    xml.push_str("      <Points>\n");
    xml.push_str("        <DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"ascii\">\n          ");
    for (i, p) in mesh.points.iter().enumerate() {
        if i > 0 {
            xml.push(' ');
        }
        let _ = write!(xml, "{} {} {}", p[0], p[1], p[2]);
    }
    xml.push_str("\n        </DataArray>\n      </Points>\n");

    // Cells
    xml.push_str("      <Cells>\n");
    xml.push_str(
        "        <DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n          ",
    );
    let mut first = true;
    for cell in &mesh.cells {
        for &n in &cell.nodes {
            if !first {
                xml.push(' ');
            }
            first = false;
            let _ = write!(xml, "{n}");
        }
    }
    xml.push_str("\n        </DataArray>\n");

    xml.push_str(
        "        <DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n          ",
    );
    let mut offset = 0usize;
    for (i, cell) in mesh.cells.iter().enumerate() {
        offset += cell.nodes.len();
        if i > 0 {
            xml.push(' ');
        }
        let _ = write!(xml, "{offset}");
    }
    xml.push_str("\n        </DataArray>\n");

    xml.push_str("        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n          ");
    for (i, cell) in mesh.cells.iter().enumerate() {
        if i > 0 {
            xml.push(' ');
        }
        let _ = write!(xml, "{}", vtk_cell_type(cell.kind));
    }
    xml.push_str("\n        </DataArray>\n      </Cells>\n");

    // CellData: marker & min_scaled_jacobian
    xml.push_str("      <CellData Scalars=\"marker\">\n");
    xml.push_str("        <DataArray type=\"Int32\" Name=\"marker\" format=\"ascii\">\n          ");
    for i in 0..mesh.cells.len() {
        if i > 0 {
            xml.push(' ');
        }
        let _ = write!(xml, "{}", mesh.cell_marker(i));
    }
    xml.push_str("\n        </DataArray>\n");

    xml.push_str("        <DataArray type=\"Float64\" Name=\"min_scaled_jacobian\" format=\"ascii\">\n          ");
    for (i, cell) in mesh.cells.iter().enumerate() {
        if i > 0 {
            xml.push(' ');
        }
        let min_sj = quality::scaled_jacobian(cell, &mesh.points)
            .into_iter()
            .fold(f64::INFINITY, f64::min);
        let val = if min_sj.is_finite() { min_sj } else { 0.0 };
        let _ = write!(xml, "{val:.6}");
    }
    xml.push_str("\n        </DataArray>\n      </CellData>\n");

    xml.push_str("    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n");
    Ok(xml)
}

/// Validate and write a `.vtu` file to disk.
pub fn write_vtu(mesh: &Mesh, path: &Path) -> Result<(), crate::msh::WriteError> {
    let xml = to_vtu(mesh).map_err(crate::msh::WriteError::Invalid)?;
    std::fs::write(path, xml).map_err(crate::msh::WriteError::Io)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::mesh::Cell;

    #[test]
    fn writes_valid_vtu_xml() {
        let mesh = Mesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            vec![Cell::qua4([0, 1, 2, 3])],
        );
        let xml = to_vtu(&mesh).unwrap();
        assert!(xml.contains("NumberOfPoints=\"4\" NumberOfCells=\"1\""));
        assert!(xml.contains("<DataArray type=\"UInt8\" Name=\"types\""));
        assert!(xml.contains(">\n          9\n"));
    }
}
