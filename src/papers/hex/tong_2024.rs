//! Tong, Halilaj & Zhang (2024) — HybridOctree_Hex: Hybrid octree-based
//! adaptive all-hexahedral mesh generation with Jacobian control.
//!
//! Reference: Tong H., Halilaj E., Zhang Y.J., 2024. *Journal of Computational
//! Science*, 78: 102278. DOI: `10.1016/j.jocs.2024.102278`.

use crate::core::domain3d::Domain3D;
use crate::core::error::{Error, Result};
use crate::core::mesh::Mesh;

#[derive(Clone, Debug, Default)]
pub struct Params {}

pub fn run(_domain: &Domain3D, _params: &Params) -> Result<Mesh> {
    Err(Error::NotSupportedYet(
        "tong_2024 HybridOctree_Hex strongly balanced octree dual & buffer optimization (Phase 4)"
            .to_string(),
    ))
}
