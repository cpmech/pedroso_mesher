//! Universal quality-constrained 3D hexahedral FEA mesh generator (`Hex8`, `Hex20`).

use crate::core::domain3d::Domain3D;
use crate::core::error::{Error, Result};
use crate::core::mesh::Mesh;
use crate::core::sizing::QualityTargets;

/// Generate a quality-constrained 3D all-hexahedral FEA mesh (`Hex8`, or `Hex20` when `quadratic == true`).
pub fn generate_hexahedra(
    _domain: &Domain3D,
    _targets: &QualityTargets,
    _quadratic: bool,
) -> Result<Mesh> {
    Err(Error::NotSupportedYet(
        "3D hexahedral FEA mesh generation (scheduled for Phase 4)".to_string(),
    ))
}
