//! Pellenard, Alliez & Morvan (2011) — Isotropic 2D quadrangle meshing with
//! size and orientation control.
//!
//! Reference: Pellenard B., Alliez P., Morvan J.-M., 2011. "Isotropic 2D
//! quadrangle meshing with size and orientation control." *Proceedings of the
//! 20th International Meshing Roundtable*, Springer, pp. 81–98.

use crate::core::error::{Error, Result};
use crate::core::mesh::Mesh;
use crate::papers::common::two_d::{Domain, Sizing};

/// Paper-defined parameters for Pellenard et al. (2011).
#[derive(Clone, Debug, Default)]
pub struct Params {}

/// Run Pellenard et al. (2011) tile-relaxation & local-parameterization quad meshing.
pub fn run(_domain: &Domain, _sizing: &Sizing, _params: &Params) -> Result<Mesh> {
    Err(Error::NotSupportedYet(
        "pellenard_2011 standalone tile relaxation (Phase 2C Tier 2)".to_string(),
    ))
}
