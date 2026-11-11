//! Schneiders (1996) — A grid-based algorithm for the generation of hexahedral
//! element meshes.
//!
//! Reference: Schneiders R., 1996. *Engineering with Computers*, 12(3–4): 168–177.

use crate::core::domain3d::Domain3D;
use crate::core::error::{Error, Result};
use crate::core::mesh::Mesh;

#[derive(Clone, Debug, Default)]
pub struct Params {}

pub fn run(_domain: &Domain3D, _params: &Params) -> Result<Mesh> {
    Err(Error::NotSupportedYet(
        "schneiders_1996 grid-based hexahedral core & isomorphic buffer layer (Phase 4)"
            .to_string(),
    ))
}
