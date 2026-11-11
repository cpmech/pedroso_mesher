//! Yang, Zhang, Ren & Xu (2020) — 2D Mesh smoothing based on Markov chain method.
//!
//! Reference: Yang F., Zhang D., Ren H., Xu J., 2020. *Engineering with Computers*,
//! 36: 1615–1626. DOI: `10.1007/s00366-019-00786-1`.

use crate::core::error::{Error, Result};
use crate::core::mesh::Mesh;

#[derive(Clone, Debug)]
pub struct Params {
    pub seed: u64,
}

impl Default for Params {
    fn default() -> Self {
        Self { seed: 42 }
    }
}

pub fn run(_mesh: &Mesh, _params: &Params) -> Result<Mesh> {
    Err(Error::NotSupportedYet(
        "yang_2020 Markov-chain Monte Carlo 2D mesh smoothing (Phase 2B)".to_string(),
    ))
}
