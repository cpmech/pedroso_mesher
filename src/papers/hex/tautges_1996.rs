//! Tautges, Blacker & Mitchell (1996) — The whisker weaving algorithm: A
//! connectivity-based method for constructing all-hexahedral finite element meshes.
//!
//! Reference: Tautges T.J., Blacker T., Mitchell S.A., 1996. *IJNME*, 39(19): 3327–3349.

use crate::core::error::{Error, Result};
use crate::core::mesh::Mesh;

#[derive(Clone, Debug, Default)]
pub struct Params {}

pub fn run(_surface_quad_mesh: &Mesh, _params: &Params) -> Result<Mesh> {
    Err(Error::NotSupportedYet(
        "tautges_1996 Spatial Twist Continuum whisker weaving (Phase 4)".to_string(),
    ))
}
