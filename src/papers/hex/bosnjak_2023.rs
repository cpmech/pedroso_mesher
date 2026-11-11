//! Bošnjak, Pepe, Schussnig, Schmalstieg & Fries (2023) — Higher-order
//! block-structured hex meshing of tubular structures.
//!
//! Reference: Bošnjak D. et al., 2023. *Engineering with Computers*.
//! DOI: `10.1007/s00366-023-01834-7`.

use crate::core::domain3d::Domain3D;
use crate::core::error::{Error, Result};
use crate::core::mesh::Mesh;

#[derive(Clone, Debug, Default)]
pub struct Params {}

pub fn run(_domain: &Domain3D, _params: &Params) -> Result<Mesh> {
    Err(Error::NotSupportedYet(
        "bosnjak_2023 skeleton block-structured higher-order tubular hex meshing (Phase 4)"
            .to_string(),
    ))
}
