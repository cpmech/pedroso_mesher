//! Element sizing and quality control specifications for 2D and 3D FEA meshers.

use serde::{Deserialize, Serialize};

/// Target quality and sizing constraints for FEA mesh generation.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct QualityTargets {
    /// Characteristic target edge length $h > 0$ (when `Some`).
    pub target_edge_length: Option<f64>,
    /// Maximum allowed element area (2D) or volume (3D).
    pub max_measure: Option<f64>,
    /// Target minimum planar corner angle or 3D dihedral angle in degrees.
    pub min_angle_deg: Option<f64>,
    /// Target minimum scaled Jacobian in `(0, 1]`.
    pub min_scaled_jacobian: Option<f64>,
}

impl Default for QualityTargets {
    fn default() -> Self {
        Self {
            target_edge_length: None,
            max_measure: None,
            min_angle_deg: Some(25.0),
            min_scaled_jacobian: Some(0.2),
        }
    }
}
