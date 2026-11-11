//! Menéndez-Díaz, González-Nicieza & Álvarez-Vigil (2005) — Hexahedral mesh
//! smoothing using a direct method (3D geometric-stiffness direct method).
//!
//! Reference: Menéndez-Díaz A. et al., 2005. *Computers & Geosciences*, 31(4): 453–463.

use crate::core::error::{Error, Result};
use crate::core::mesh::Mesh;

#[derive(Clone, Debug, Default)]
pub struct Params {}

pub fn run(_mesh: &Mesh, _params: &Params) -> Result<Mesh> {
    Err(Error::NotSupportedYet(
        "menendez_diaz_2005 3D hexahedral geometric-stiffness direct smoothing (Phase 4)"
            .to_string(),
    ))
}
