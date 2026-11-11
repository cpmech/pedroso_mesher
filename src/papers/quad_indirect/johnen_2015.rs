//! Johnen, Ernst & Geuzaine (2015) — Sequential decision-making approach for
//! quadrangular mesh generation.
//!
//! Reference: Johnen A., Ernst D., Geuzaine C., 2015. *Engineering with Computers*,
//! 31(4): 719–730.

use crate::core::error::{Error, Result};
use crate::core::mesh::Mesh;
use crate::papers::common::two_d::{Domain, Sizing};

#[derive(Clone, Debug, Default)]
pub struct Params {}

pub fn run(_domain: &Domain, _sizing: &Sizing, _params: &Params) -> Result<Mesh> {
    Err(Error::NotSupportedYet(
        "johnen_2015 selective look-ahead tree recombination (Phase 2C Tier 3)".to_string(),
    ))
}
