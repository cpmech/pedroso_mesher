//! Writing a mesh in gemlab's MSH text format.
//!
//! The format is the interchange used by `gemlab` and `pmsim`, so a mesh made
//! here can be solved there without this crate depending on either (and without
//! dragging BLAS into a mesher that must build anywhere):
//!
//! ```text
//! # header
//! # ndim npoint ncell nmarked_edge nmarked_face
//!      2      8     5            0            0
//!
//! # points
//! # id marker x   y
//!    0 0 0.0 0.0
//!
//! # cells
//! # id marker kind points
//!    0   1 qua4  1 4 6 3
//! ```
//!
//! Cell kinds are gemlab's names: `tri3`, `qua4`, `hex8`. A mesh with no
//! hexahedra is written as 2D (two coordinates per point); otherwise as 3D.

use std::fmt::Write as _;
use std::io;
use std::path::Path;

use crate::mesh::{Kind, Mesh};

/// Cell kind as gemlab spells it.
pub fn kind_name(kind: Kind) -> &'static str {
    match kind {
        Kind::Tri3 => "tri3",
        Kind::Quad4 => "qua4",
        Kind::Hex8 => "hex8",
    }
}

/// Markers are not part of the P1 contract yet; one region, marker 1.
const MARKER: i32 = 1;

/// Render a mesh as MSH text.
pub fn to_msh(mesh: &Mesh) -> String {
    let is_3d =
        mesh.cells.iter().any(|c| c.kind == Kind::Hex8) || mesh.points.iter().any(|p| p[2] != 0.0);
    let ndim = if is_3d { 3 } else { 2 };

    let mut out = String::new();
    out.push_str("# header\n# ndim npoint ncell nmarked_edge nmarked_face\n");
    let _ = writeln!(
        out,
        "{} {} {} 0 0",
        ndim,
        mesh.points.len(),
        mesh.cells.len()
    );

    out.push_str("\n# points\n# id marker");
    out.push_str(if is_3d { " x y z\n" } else { " x y\n" });
    for (id, p) in mesh.points.iter().enumerate() {
        let _ = if is_3d {
            writeln!(out, "{id} 0 {} {} {}", p[0], p[1], p[2])
        } else {
            writeln!(out, "{id} 0 {} {}", p[0], p[1])
        };
    }

    out.push_str("\n# cells\n# id marker kind points\n");
    for (id, cell) in mesh.cells.iter().enumerate() {
        out.push_str(&format!("{id} {MARKER} {}", kind_name(cell.kind)));
        for n in &cell.nodes {
            let _ = write!(out, " {n}");
        }
        out.push('\n');
    }
    out
}

/// Write a mesh to `path`, validating it first.
pub fn write_msh(mesh: &Mesh, path: &Path) -> Result<(), WriteError> {
    mesh.validate().map_err(WriteError::Invalid)?;
    std::fs::write(path, to_msh(mesh)).map_err(WriteError::Io)
}

/// Why a mesh could not be written.
#[derive(Debug)]
pub enum WriteError {
    Invalid(crate::mesh::MeshError),
    Io(io::Error),
}

impl std::fmt::Display for WriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WriteError::Invalid(e) => write!(f, "the mesh is invalid: {e}"),
            WriteError::Io(e) => write!(f, "cannot write the file: {e}"),
        }
    }
}

impl std::error::Error for WriteError {}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::mesh::Cell;

    fn square() -> Mesh {
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
    fn a_planar_mesh_is_written_as_2d() {
        let text = to_msh(&square());
        assert!(text.contains("# header"), "{text}");
        assert!(text.contains("2 4 1 0 0"), "{text}");
        assert!(text.contains("0 0 0 0"), "{text}");
        assert!(text.contains("0 1 qua4 0 1 2 3"), "{text}");
        assert!(
            !text.contains(" 0.0 0.0 0.0"),
            "z must not be written in 2D"
        );
    }

    #[test]
    fn a_hexahedral_mesh_is_written_as_3d() {
        let mesh = Mesh::new(
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
        let text = to_msh(&mesh);
        assert!(text.contains("3 8 1 0 0"), "{text}");
        assert!(text.contains("0 1 hex8 0 1 2 3 4 5 6 7"), "{text}");
    }

    #[test]
    fn a_triangle_uses_gemlabs_spelling() {
        let mesh = Mesh::new(
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![Cell::tri3([0, 1, 2])],
        );
        assert!(to_msh(&mesh).contains("0 1 tri3 0 1 2"));
    }

    #[test]
    fn the_header_counts_match_the_mesh() {
        let text = to_msh(&square());
        let header = text.lines().nth(2).unwrap();
        let fields: Vec<usize> = header
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        assert_eq!(fields, vec![2, 4, 1, 0, 0]);
    }

    #[test]
    fn writing_validates_first() {
        let mut mesh = square();
        mesh.points[0][0] = f64::NAN;
        let error = write_msh(&mesh, Path::new("/tmp/never-written.msh")).unwrap_err();
        assert!(matches!(error, WriteError::Invalid(_)));
    }

    #[test]
    fn every_line_is_parseable_shape() {
        // the reader splits on whitespace and expects an id, a marker and then
        // the coordinates, so no line may be empty where data is expected
        let text = to_msh(&square());
        let points: Vec<&str> = text
            .lines()
            .skip_while(|l| !l.starts_with("# points"))
            .skip(2)
            .take_while(|l| !l.is_empty())
            .collect();
        assert_eq!(points.len(), 4);
        for line in points {
            let fields: Vec<&str> = line.split_whitespace().collect();
            assert_eq!(fields.len(), 4, "{line}");
        }
    }
}
