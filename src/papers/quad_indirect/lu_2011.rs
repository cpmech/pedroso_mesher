//! Lu, Qian, Shi & Liu (2011) — Quading triangular meshes with certain
//! topological constraints.
//!
//! Reference: Lu L., Qian X., Shi X., Liu F., 2011. *Journal of Computational
//! and Applied Mathematics*, 235(13): 3826–3836.

use crate::core::error::{Error, Result};
use crate::core::mesh::Mesh;
use crate::papers::common::two_d::{Domain, Sizing};

#[derive(Clone, Debug, Default)]
pub struct Params {}

pub fn run(_domain: &Domain, _sizing: &Sizing, _params: &Params) -> Result<Mesh> {
    Err(Error::NotSupportedYet(
        "lu_2011 homotopic-basis topological quading (Phase 2C Tier 3)".to_string(),
    ))
}
