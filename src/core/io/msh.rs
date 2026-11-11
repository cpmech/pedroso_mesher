//! Reading and writing meshes in `gemlab`'s `.msh` text format.
//!
//! Supports all eight FEA element shapes (`tri3`, `tri6`, `qua4`, `qua8`,
//! `tet4`, `tet10`, `hex8`, `hex20`) plus point markers, cell region markers,
//! `# marked edges`, and `# marked faces`.

use std::fmt::Write as _;
use std::io;
use std::path::Path;

use crate::mesh::{Kind, Mesh};

/// Cell kind string as `gemlab` spells it.
pub fn kind_name(kind: Kind) -> &'static str {
    kind.msh_name()
}

/// Render a [`Mesh`] in `gemlab`'s `.msh` text format.
pub fn to_msh(mesh: &Mesh) -> String {
    let is_3d = mesh.cells.iter().any(|c| c.kind.dim() == 3)
        || mesh.points.iter().any(|p| p[2] != 0.0)
        || !mesh.marked_faces.is_empty();
    let ndim = if is_3d { 3 } else { 2 };

    let mut out = String::new();
    out.push_str("# header\n# ndim npoint ncell nmarked_edge nmarked_face\n");
    let _ = writeln!(
        out,
        "{} {} {} {} {}",
        ndim,
        mesh.points.len(),
        mesh.cells.len(),
        mesh.marked_edges.len(),
        mesh.marked_faces.len()
    );

    out.push_str("\n# points\n# id marker");
    out.push_str(if is_3d { " x y z\n" } else { " x y\n" });
    for (id, p) in mesh.points.iter().enumerate() {
        let pm = mesh.point_marker(id);
        let _ = if is_3d {
            writeln!(out, "{id} {pm} {} {} {}", p[0], p[1], p[2])
        } else {
            writeln!(out, "{id} {pm} {} {}", p[0], p[1])
        };
    }

    out.push_str("\n# cells\n# id marker kind points\n");
    for (id, cell) in mesh.cells.iter().enumerate() {
        let cm = mesh.cell_marker(id);
        let _ = write!(out, "{id} {cm} {}", kind_name(cell.kind));
        for n in &cell.nodes {
            let _ = write!(out, " {n}");
        }
        out.push('\n');
    }

    if !mesh.marked_edges.is_empty() {
        out.push_str("\n# marked edges\n# marker p1 p2\n");
        for &(marker, [p1, p2]) in &mesh.marked_edges {
            let _ = writeln!(out, "{marker} {p1} {p2}");
        }
    }

    if !mesh.marked_faces.is_empty() {
        out.push_str("\n# marked faces\n# marker p1 p2 p3 {p4}\n");
        for (marker, nodes) in &mesh.marked_faces {
            let _ = write!(out, "{marker}");
            for &n in nodes {
                let _ = write!(out, " {n}");
            }
            out.push('\n');
        }
    }

    out
}

/// Validate and write a mesh to `path` in `.msh` format.
pub fn write_msh(mesh: &Mesh, path: &Path) -> Result<(), WriteError> {
    mesh.validate().map_err(WriteError::Invalid)?;
    std::fs::write(path, to_msh(mesh)).map_err(WriteError::Io)
}

/// Errors returned by [`write_msh`].
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
            vec![Cell::qua4([0, 1, 2, 3])],
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
    fn writes_marked_edges_and_cell_markers() {
        let mut mesh = square();
        mesh.set_cell_marker(0, 7);
        mesh.set_point_marker(0, -1);
        mesh.marked_edges.push((-100, [0, 1]));
        let text = to_msh(&mesh);
        assert!(text.contains("2 4 1 1 0"), "{text}");
        assert!(text.contains("0 -1 0 0"), "{text}");
        assert!(text.contains("0 7 qua4 0 1 2 3"), "{text}");
        assert!(
            text.contains("# marked edges\n# marker p1 p2\n-100 0 1\n"),
            "{text}"
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
    fn writing_validates_first() {
        let mut mesh = square();
        mesh.points[0][0] = f64::NAN;
        let error = write_msh(&mesh, Path::new("/tmp/opencode/never-written.msh")).unwrap_err();
        assert!(matches!(error, WriteError::Invalid(_)));
    }
}
