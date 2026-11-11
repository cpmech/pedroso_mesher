//! Pochet, Celes, Lopes & Gattass (2017) — A new quadtree-based approach for
//! automatic quadrilateral mesh generation.
//!
//! Reference: Pochet A. et al., 2017. *Engineering with Computers*, 33(2): 275–292.

use crate::core::error::{Error, Result};
use crate::core::mesh::Mesh;
use crate::papers::common::two_d::{Domain, Sizing};

#[derive(Clone, Debug, Default)]
pub struct Params {}

pub fn run(_domain: &Domain, _sizing: &Sizing, _params: &Params) -> Result<Mesh> {
    Err(Error::NotSupportedYet(
        "pochet_2017 extended quadtree S1/S2 subdivision & corner attraction (Phase 2C Tier 2)"
            .to_string(),
    ))
}
