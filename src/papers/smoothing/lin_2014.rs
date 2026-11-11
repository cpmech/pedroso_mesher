//! Lin, Guan, Chang & Lo (2014) — Vertex-Ball Spring Smoothing (VerBSS): An
//! efficient method for unstructured dynamic hybrid meshes.
//!
//! Reference: Lin T.J., Guan Z.Q., Chang J.H., Lo S.H., 2014. *Computers &
//! Structures*, 136: 24–33.

use crate::core::error::{Error, Result};
use crate::core::mesh::Mesh;

#[derive(Clone, Debug, Default)]
pub struct Params {}

pub fn run(_mesh: &Mesh, _params: &Params) -> Result<Mesh> {
    Err(Error::NotSupportedYet(
        "lin_2014 Vertex-Ball Spring Smoothing VerBSS (Phase 2B)".to_string(),
    ))
}
