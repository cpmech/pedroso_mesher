//! Shared, paper-agnostic FEA mesh model, domains, graph matching, verification,
//! quality metrics, high-order elevation, and I/O (`JSON`, `.msh`, `.vtu`, `SVG`).
//!
//! All linear algebra required by paper modules or smoothers is performed via
//! [`algebra_linear`].

pub mod bench;
pub mod domain2d;
pub mod domain3d;
pub mod error;
pub mod high_order;
pub mod io;
pub mod matching;
pub mod mesh;
pub mod quality;
pub mod sizing;
pub mod stats;
pub mod verify;
