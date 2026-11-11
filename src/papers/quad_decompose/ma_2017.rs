//! Ma, Zhao & Sun (2017) — Sub-domain decomposition and quadrilateral meshing
//! with line constraints.
//!
//! Reference: Ma X., Zhao G., Sun L., 2017. *Advances in Engineering Software*,
//! 109: 26–38.

use crate::core::error::{Error, Result};
use crate::core::mesh::Mesh;
use crate::papers::common::two_d::{Domain, Sizing};

#[derive(Clone, Debug, Default)]
pub struct Params {}

pub fn run(_domain: &Domain, _sizing: &Sizing, _params: &Params) -> Result<Mesh> {
    Err(Error::NotSupportedYet(
        "ma_2017 line-constrained sub-domain decomposition & ILP parity (Phase 2C Tier 2)"
            .to_string(),
    ))
}
