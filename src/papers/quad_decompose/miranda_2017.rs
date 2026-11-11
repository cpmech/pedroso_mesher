//! Miranda & Martha (2017) — Hierarchical template-based quadrilateral mesh
//! generation.
//!
//! Reference: Miranda A.C.O., Martha L.F., 2017. *Engineering with Computers*,
//! 33(2): 123–138.

use crate::core::error::{Error, Result};
use crate::core::mesh::Mesh;
use crate::papers::common::two_d::{Domain, Sizing};

#[derive(Clone, Debug, Default)]
pub struct Params {}

pub fn run(_domain: &Domain, _sizing: &Sizing, _params: &Params) -> Result<Mesh> {
    Err(Error::NotSupportedYet(
        "miranda_2017 standalone 4-, 3-, and 2-curve hierarchical templates (Phase 2C Tier 2)"
            .to_string(),
    ))
}
