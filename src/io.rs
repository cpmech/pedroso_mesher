//! Reading a mesh from JSON.
//!
//! The P0 interchange format is deliberately small: it is what the tests, the
//! benchmark harness and the CLI use before an MSH/VTU writer exists.
//!
//! ```json
//! {
//!   "points": [[0,0,0], [1,0,0], [1,1,0], [0,1,0]],
//!   "cells": [{"kind": "quad4", "nodes": [0,1,2,3]}]
//! }
//! ```

use std::fmt;

use serde::Deserialize;

use crate::mesh::{Cell, Kind, Mesh, MeshError};

/// Anything that can go wrong reading a mesh.
#[derive(Debug)]
pub enum IoError {
    Json(String),
    UnknownKind(String),
    Mesh(MeshError),
}

impl fmt::Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IoError::Json(message) => write!(f, "invalid mesh JSON: {message}"),
            IoError::UnknownKind(kind) => {
                write!(
                    f,
                    "unknown cell kind {kind:?}; expected \"quad4\" or \"hex8\""
                )
            }
            IoError::Mesh(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for IoError {}

impl From<MeshError> for IoError {
    fn from(error: MeshError) -> Self {
        IoError::Mesh(error)
    }
}

#[derive(Deserialize)]
struct MeshJson {
    points: Vec<[f64; 3]>,
    cells: Vec<CellJson>,
}

#[derive(Deserialize)]
struct CellJson {
    kind: String,
    nodes: Vec<usize>,
}

/// Parse and validate a mesh.
pub fn read_mesh_json(text: &str) -> Result<Mesh, IoError> {
    let parsed: MeshJson = serde_json::from_str(text).map_err(|e| IoError::Json(e.to_string()))?;
    let mut cells = Vec::with_capacity(parsed.cells.len());
    for cell in parsed.cells {
        let kind = match cell.kind.as_str() {
            "quad4" | "Quad4" => Kind::Quad4,
            "hex8" | "Hex8" => Kind::Hex8,
            other => return Err(IoError::UnknownKind(other.to_string())),
        };
        cells.push(Cell {
            kind,
            nodes: cell.nodes,
        });
    }
    let mesh = Mesh::new(parsed.points, cells);
    mesh.validate()?;
    Ok(mesh)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    const SQUARE: &str = r#"{
        "points": [[0,0,0], [1,0,0], [1,1,0], [0,1,0]],
        "cells": [{"kind": "quad4", "nodes": [0,1,2,3]}]
    }"#;

    #[test]
    fn reads_a_quad() {
        let mesh = read_mesh_json(SQUARE).unwrap();
        assert_eq!(mesh.points.len(), 4);
        assert_eq!(mesh.cells.len(), 1);
        assert_eq!(mesh.cells[0].kind, Kind::Quad4);
    }

    #[test]
    fn rejects_an_unknown_kind() {
        let text = SQUARE.replace("quad4", "tri3");
        assert!(matches!(read_mesh_json(&text), Err(IoError::UnknownKind(k)) if k == "tri3"));
    }

    #[test]
    fn rejects_bad_json() {
        assert!(matches!(read_mesh_json("{"), Err(IoError::Json(_))));
    }

    #[test]
    fn validates_indices_on_the_way_in() {
        let text = SQUARE.replace("[0,1,2,3]", "[0,1,2,9]");
        assert!(matches!(read_mesh_json(&text), Err(IoError::Mesh(_))));
    }
}
