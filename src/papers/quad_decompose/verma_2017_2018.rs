//! Verma & Suresh (2017, 2018) — Minimum Singularity Templates (MST) and
//! $\alpha$MST unified quad mesh adaptation.
//!
//! References:
//! - Verma C.S., Suresh K., 2017. *Computer-Aided Design*, 85: 99–110.
//! - Verma C.S., Suresh K., 2018. *IJNME*, 113: 513–536.

use crate::core::error::{Error, Result};
use crate::core::mesh::Mesh;

#[derive(Clone, Debug)]
pub struct Params {
    pub alpha: f64,
}

impl Default for Params {
    fn default() -> Self {
        Self { alpha: 1.0 }
    }
}

pub fn run(_mesh: &Mesh, _params: &Params) -> Result<Mesh> {
    Err(Error::NotSupportedYet(
        "verma_2017_2018 full BFS patch extraction & MST/alpha-MST templates (Phase 2C Tier 1)"
            .to_string(),
    ))
}
