//! Merhof, Grosso, Tremel & Greiner (2007) — Anisotropic quadrilateral mesh
//! generation: An indirect approach.
//!
//! Reference: Merhof D. et al., 2007. *Advances in Engineering Software*, 38(8–9): 560–568.

use crate::core::error::{Error, Result};
use crate::core::mesh::Mesh;
use crate::papers::common::two_d::{Domain, Sizing};

#[derive(Clone, Debug, Default)]
pub struct Params {}

pub fn run(_domain: &Domain, _sizing: &Sizing, _params: &Params) -> Result<Mesh> {
    Err(Error::NotSupportedYet(
        "merhof_2007 multi-pass 3-step anisotropic indirect quad meshing (Phase 2C Tier 3)"
            .to_string(),
    ))
}
