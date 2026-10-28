//! Robust planar geometry for the constrained Delaunay kernel.
//!
//! This is a separate input model from [`crate::two_d::Domain`] while the new
//! kernel is being built. Existing quad routes are not migrated by this module.

mod domain;
mod exact;
mod geometry;
mod predicates;
mod refinement;
mod triangulation;

pub use domain::{Domain, DomainError, Feature, Loop, PointLocation, Region};
pub use geometry::Point;
pub use predicates::{encroaches, incircle, orient2d};
pub use refinement::{
    triangulate, QualityRequest, QualityStats, RefinedMesh, RefinementError, RefinementReport,
};
pub use triangulation::{Constraint, ConstraintKind, Triangle, Triangulation, TriangulationError};
