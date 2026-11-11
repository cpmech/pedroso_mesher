//! Durand, Pantoja-Rosero & Oliveira (2019) — A general mesh smoothing method
//! for finite elements.
//!
//! Reference: Durand R., Pantoja-Rosero B.G., Oliveira V., 2019. *Finite Elements
//! in Analysis and Design*, 158: 17–30.

use crate::core::error::{Error, Result};
use crate::core::mesh::Mesh;

#[derive(Clone, Debug, Default)]
pub struct Params {}

pub fn run(_mesh: &Mesh, _params: &Params) -> Result<Mesh> {
    Err(Error::NotSupportedYet(
        "durand_2019 least-squares general finite-element mesh smoothing (Phase 2B)".to_string(),
    ))
}
