//! Shewchuk (1997) — Adaptive precision floating-point arithmetic and fast
//! robust geometric predicates.
//!
//! Reference: Shewchuk J.R., 1997. "Adaptive Precision Floating-Point Arithmetic
//! and Fast Robust Geometric Predicates." *Discrete & Computational Geometry*,
//! 18(3): 305–363. DOI: `10.1007/PL00009321`.
//!
//! Currently re-exports the clean-room 2D filtered predicates (`orient2d`,
//! `incircle`, `encroaches`); Phase 2A/3 upgrades this module to full adaptive
//! expansion arithmetic including 3D `orient3d` and `insphere`.

pub use super::delaunay::{encroaches, incircle, orient2d};
