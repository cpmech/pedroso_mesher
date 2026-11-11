//! Nasri, Sabin & Yasseen (2009) — Filling $n$-sided regions by quad meshes for
//! subdivision surfaces.
//!
//! Reference: Nasri A., Sabin M., Yasseen Z., 2009. *Computer Graphics Forum*,
//! 28(6): 1644–1658.

use crate::core::error::{Error, Result};
use crate::core::mesh::Mesh;
use crate::papers::common::two_d::{Domain, Sizing};

#[derive(Clone, Debug, Default)]
pub struct Params {}

pub fn run(_domain: &Domain, _sizing: &Sizing, _params: &Params) -> Result<Mesh> {
    Err(Error::NotSupportedYet(
        "nasri_2009 n-sided refractors & generalized discrete Coons patches (Phase 2C Tier 3)"
            .to_string(),
    ))
}
