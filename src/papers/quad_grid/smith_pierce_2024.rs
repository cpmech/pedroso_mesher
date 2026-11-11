//! Smith-Pierce, Ruffin & Dement (2024) — Automated Unstructured Quad/Hex
//! Meshing for High-Order Discontinuous Galerkin CFD.
//!
//! Reference: Smith-Pierce M., Ruffin S., Dement D., 2024. *AIAA SciTech 2024
//! Forum*, AIAA 2024-0384. DOI: `10.2514/6.2024-0384`.

use crate::core::error::{Error, Result};
use crate::core::mesh::Mesh;
use crate::papers::common::two_d::{Domain, Sizing};

#[derive(Clone, Debug, Default)]
pub struct Params {}

pub fn run(_domain: &Domain, _sizing: &Sizing, _params: &Params) -> Result<Mesh> {
    Err(Error::NotSupportedYet(
        "smith_pierce_2024 uncertainty-driven Cartesian core & extruded wall layers (Phase 2C Tier 3)"
            .to_string(),
    ))
}
