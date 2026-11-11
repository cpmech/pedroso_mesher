//! Reading and writing meshes in JSON format via `serde_json`.

use std::fmt;

use serde::{Deserialize, Serialize};

use crate::mesh::{Cell, Kind, Mesh, MeshError};

/// Errors produced when reading or writing mesh JSON.
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
            IoError::UnknownKind(kind) => write!(
                f,
                "unknown cell kind {kind:?}; expected tri3, tri6, qua4, qua8, tet4, tet10, hex8, or hex20"
            ),
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

#[derive(Deserialize, Serialize)]
struct MeshJson {
    points: Vec<[f64; 3]>,
    cells: Vec<CellJson>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    point_markers: Vec<i32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    cell_markers: Vec<i32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    marked_edges: Vec<(i32, [usize; 2])>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    marked_faces: Vec<(i32, Vec<usize>)>,
}

#[derive(Deserialize, Serialize)]
struct CellJson {
    kind: String,
    nodes: Vec<usize>,
}

/// Parse and structurally validate a mesh from JSON text.
pub fn read_mesh_json(text: &str) -> Result<Mesh, IoError> {
    let parsed: MeshJson = serde_json::from_str(text).map_err(|e| IoError::Json(e.to_string()))?;
    let mut cells = Vec::with_capacity(parsed.cells.len());
    for cell in parsed.cells {
        let Some(kind) = Kind::from_name(&cell.kind) else {
            return Err(IoError::UnknownKind(cell.kind));
        };
        cells.push(Cell {
            kind,
            nodes: cell.nodes,
        });
    }
    let mut mesh = Mesh::new(parsed.points, cells);
    mesh.point_markers = parsed.point_markers;
    mesh.cell_markers = parsed.cell_markers;
    mesh.marked_edges = parsed.marked_edges;
    mesh.marked_faces = parsed.marked_faces;
    mesh.validate()?;
    Ok(mesh)
}

/// Serialize a validated mesh to pretty-printed JSON text.
pub fn write_mesh_json(mesh: &Mesh) -> Result<String, IoError> {
    mesh.validate()?;
    let payload = MeshJson {
        points: mesh.points.clone(),
        cells: mesh
            .cells
            .iter()
            .map(|c| CellJson {
                kind: c.kind.name().to_string(),
                nodes: c.nodes.clone(),
            })
            .collect(),
        point_markers: mesh.point_markers.clone(),
        cell_markers: mesh.cell_markers.clone(),
        marked_edges: mesh.marked_edges.clone(),
        marked_faces: mesh.marked_faces.clone(),
    };
    serde_json::to_string_pretty(&payload).map_err(|e| IoError::Json(e.to_string()))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    const SQUARE: &str = r#"{
        "points": [[0,0,0], [1,0,0], [1,1,0], [0,1,0]],
        "cells": [{"kind": "qua4", "nodes": [0,1,2,3]}]
    }"#;

    #[test]
    fn reads_and_roundtrips_all_supported_kinds() {
        let mesh = read_mesh_json(SQUARE).unwrap();
        assert_eq!(mesh.points.len(), 4);
        assert_eq!(mesh.cells.len(), 1);
        assert_eq!(mesh.cells[0].kind, Kind::Qua4);

        let text = write_mesh_json(&mesh).unwrap();
        let reloaded = read_mesh_json(&text).unwrap();
        assert_eq!(reloaded.cells, mesh.cells);
    }

    #[test]
    fn rejects_an_unknown_kind() {
        let text = SQUARE.replace("qua4", "prism6");
        assert!(matches!(read_mesh_json(&text), Err(IoError::UnknownKind(k)) if k == "prism6"));
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
