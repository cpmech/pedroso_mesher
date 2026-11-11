//! Gao, Xiao, Xu, Shen & Xu (2024) — Quadrilateral surface mesh generation with
//! improved quality by combination of triangles.
//!
//! Reference: Gao J. et al., 2024. *IJNME*, 125.

use crate::core::error::{Error, Result};
use crate::core::mesh::Mesh;
use crate::papers::common::two_d::{Domain, Sizing};

#[derive(Clone, Debug, Default)]
pub struct Params {}

pub fn run(_domain: &Domain, _sizing: &Sizing, _params: &Params) -> Result<Mesh> {
    Err(Error::NotSupportedYet(
        "gao_2024 local-global remeshing, Blossom & connectivity optimization (Phase 2C Tier 2)"
            .to_string(),
    ))
}
