//! Carrier-Baudouin, Remacle, Marchandise & Lambrechts (2014) — A frontal
//! approach to hex-dominant mesh generation.
//!
//! Reference: Carrier-Baudouin T. et al., 2014. *Advanced Modeling and Simulation
//! in Engineering Sciences*, 1: 8. DOI: `10.1186/2213-7467-1-8`.

use crate::core::domain3d::Domain3D;
use crate::core::error::{Error, Result};
use crate::core::mesh::Mesh;

#[derive(Clone, Debug, Default)]
pub struct Params {}

pub fn run(_domain: &Domain3D, _params: &Params) -> Result<Mesh> {
    Err(Error::NotSupportedYet(
        "carrier_baudouin_2014 frontal vertex alignment & hex recombination (Phase 4)".to_string(),
    ))
}
