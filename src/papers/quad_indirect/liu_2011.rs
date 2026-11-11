//! Liu, Xing & Guan (2011) — An indirect approach for automatic generation of
//! quadrilateral meshes with arbitrary line constraints.
//!
//! Reference: Liu Y., Xing H.L., Guan Z., 2011. *IJNME*, 87(9): 906–924.

use crate::core::error::{Error, Result};
use crate::core::mesh::Mesh;
use crate::papers::common::two_d::{Domain, Sizing};

#[derive(Clone, Debug, Default)]
pub struct Params {}

pub fn run(_domain: &Domain, _sizing: &Sizing, _params: &Params) -> Result<Mesh> {
    Err(Error::NotSupportedYet(
        "liu_2011 line-constrained indirect advancing-front quad meshing (Phase 2C Tier 3)"
            .to_string(),
    ))
}
