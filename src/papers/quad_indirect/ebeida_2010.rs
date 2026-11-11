//! Ebeida, Karamete, Mestreau & Dey (2010) — Q-TRAN: A new approach to
//! transform triangular meshes into quadrilateral meshes locally.
//!
//! Reference: Ebeida M.S. et al., 2010. *Proceedings of the 19th International
//! Meshing Roundtable*, Springer, pp. 23–34.

use crate::core::error::{Error, Result};
use crate::core::mesh::Mesh;
use crate::papers::common::two_d::{Domain, Sizing};

#[derive(Clone, Debug, Default)]
pub struct Params {}

pub fn run(_domain: &Domain, _sizing: &Sizing, _params: &Params) -> Result<Mesh> {
    Err(Error::NotSupportedYet(
        "ebeida_2010 Q-TRAN local triangle-to-quad transformation (Phase 2C Tier 2)".to_string(),
    ))
}
