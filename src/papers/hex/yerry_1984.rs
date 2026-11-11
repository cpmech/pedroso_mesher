//! Yerry & Shephard (1984) — Automatic three-dimensional mesh generation by the
//! modified-octree technique.
//!
//! Reference: Yerry M.A., Shephard M.S., 1984. *IJNME*, 20(11): 1965–1990.
//! DOI: `10.1002/nme.1620201103`.

use crate::core::domain3d::Domain3D;
use crate::core::error::{Error, Result};
use crate::core::mesh::Mesh;

#[derive(Clone, Debug, Default)]
pub struct Params {}

pub fn run(_domain: &Domain3D, _params: &Params) -> Result<Mesh> {
    Err(Error::NotSupportedYet(
        "yerry_1984 modified-octree 3D mesh generation (Phase 4)".to_string(),
    ))
}
