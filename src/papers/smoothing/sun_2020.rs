//! Sun, Gou & Geng (2020) — Simultaneous Smoothing and Untangling of 2D Meshes
//! Based on Explicit Element Geometric Transformation and Element Stitching.
//!
//! Reference: Sun S., Gou Z., Geng M., 2020. *Applied Sciences*, 10(14): 4994.

use crate::core::error::{Error, Result};
use crate::core::mesh::Mesh;

#[derive(Clone, Debug, Default)]
pub struct Params {}

pub fn run(_mesh: &Mesh, _params: &Params) -> Result<Mesh> {
    Err(Error::NotSupportedYet(
        "sun_2020 EEGT + weighted element stitching smoothing & untangling (Phase 2B)".to_string(),
    ))
}
