//! Mesh and figure input/output formats (`JSON`, `gemlab` `.msh`, `ParaView` `.vtu`, and `SVG`).

pub mod json;
pub mod msh;
pub mod svg;
pub mod vtu;

pub use json::{read_mesh_json, write_mesh_json, IoError};
pub use msh::{kind_name, to_msh, write_msh, WriteError};
pub use vtu::{to_vtu, vtk_cell_type, write_vtu};
