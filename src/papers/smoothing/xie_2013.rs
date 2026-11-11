//! Xie, Sevilla, Hassan & Morgan (2013) — The generation of arbitrary order
//! curved meshes for 3D finite element analysis.
//!
//! Reference: Xie Z.Q., Sevilla R., Hassan O., Morgan K., 2013. *Computational
//! Mechanics*, 51(3): 361–374. DOI: `10.1007/s00466-012-0736-4`.

use crate::core::error::{Error, Result};
use crate::core::mesh::Mesh;

#[derive(Clone, Debug, Default)]
pub struct Params {}

pub fn run(_mesh: &Mesh, _params: &Params) -> Result<Mesh> {
    Err(Error::NotSupportedYet(
        "xie_2013 linear-elasticity curving for high-order Tri6/Qua8/Tet10/Hex20 (Phase 2D/3/4)"
            .to_string(),
    ))
}
