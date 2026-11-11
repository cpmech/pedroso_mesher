//! Universal quality-constrained 2D quadrilateral FEA mesh generator (`Qua4`, `Qua8`).

use crate::core::error::{Error, Result};
use crate::core::high_order::elevate_to_quadratic;
use crate::core::mesh::Mesh;
use crate::papers::common::config::{self, MeshConfig};
use crate::papers::common::two_d::{Domain, Sizing};

/// Generate a strictly verified 2D all-quadrilateral FEA mesh (`Qua4`, or `Qua8` when `quadratic == true`).
pub fn generate_quads(
    domain: &Domain,
    sizing: &Sizing,
    config: &MeshConfig,
    quadratic: bool,
) -> Result<Mesh> {
    let quad_mesh = config::generate_quad(domain, sizing, config)
        .map_err(|e| Error::Algorithm(e.to_string()))?;
    if quadratic {
        elevate_to_quadratic(&quad_mesh.mesh)
    } else {
        Ok(quad_mesh.mesh)
    }
}
