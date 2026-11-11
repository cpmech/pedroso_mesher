//! Zhang, Chiang, Fan & Mundilova (2023) — Local decomposition of hexahedral
//! singular nodes into singular curves.
//!
//! Reference: Zhang P. et al., 2023. *Computer-Aided Design*, 158: 103484.
//! DOI: `10.1016/j.cad.2023.103484`.

use crate::core::error::{Error, Result};
use crate::core::mesh::Mesh;

#[derive(Clone, Debug, Default)]
pub struct Params {}

pub fn run(_hex_mesh: &Mesh, _params: &Params) -> Result<Mesh> {
    Err(Error::NotSupportedYet(
        "zhang_2023 hexahedral singular node sheet-inflation decomposition (Phase 4)".to_string(),
    ))
}
