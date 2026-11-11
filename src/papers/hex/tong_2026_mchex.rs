//! Tong & Zhang (2026) — MCHex: marching cubes based adaptive hexahedral mesh
//! generation with guaranteed positive Jacobian.
//!
//! Reference: Tong H., Zhang Y.J., 2026. *ACM Transactions on Graphics*.
//! DOI: `10.1145/3858338`.

use crate::core::domain3d::Domain3D;
use crate::core::error::{Error, Result};
use crate::core::mesh::Mesh;

#[derive(Clone, Debug, Default)]
pub struct Params {}

pub fn run(_domain: &Domain3D, _params: &Params) -> Result<Mesh> {
    Err(Error::NotSupportedYet(
        "tong_2026_mchex marching-cubes positive-Jacobian adaptive hex meshing (Phase 4)"
            .to_string(),
    ))
}
