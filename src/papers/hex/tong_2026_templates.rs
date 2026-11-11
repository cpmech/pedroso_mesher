//! Tong & Zhang (2026) — Element-saving hexahedral 3-refinement templates.
//!
//! Reference: Tong H., Zhang Y.J., 2026. *Computer Aided Geometric Design*, 102550.
//! DOI: `10.1016/j.cagd.2026.102550`.

use crate::core::error::{Error, Result};
use crate::core::mesh::Mesh;

#[derive(Clone, Debug, Default)]
pub struct Params {}

pub fn run(_hex_mesh: &Mesh, _params: &Params) -> Result<Mesh> {
    Err(Error::NotSupportedYet(
        "tong_2026_templates element-saving hexahedral 3-refinement templates (Phase 4)"
            .to_string(),
    ))
}
