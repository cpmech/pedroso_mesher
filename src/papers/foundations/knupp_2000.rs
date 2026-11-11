//! Knupp (2000) — Finite element mesh quality via optimization of the Jacobian
//! matrix norm and associated quantities (Part I: surface; Part II: volume).
//!
//! References:
//! - Knupp P.M., 2000. *IJNME*, 48(3): 401–420 (Part I: surface mesh optimization).
//! - Knupp P.M., 2000. *IJNME*, 48(8): 1165–1185 (Part II: volume mesh optimization
//!   and the condition number of the Jacobian matrix).
//!
//! Phase 2A/3/4 expands this module with Knupp's full matrix-norm objective
//! families and local modified-Newton vertex optimization.

pub use crate::core::quality::{
    aspect_ratio, corner_angles, dihedral_angles, quad_angles, report, report_with,
    scaled_jacobian, volume, Report, Thresholds, Verdict,
};
