//! Shewchuk (2002) — Delaunay refinement algorithms for triangular mesh generation.
//!
//! Reference: Shewchuk J.R., 2002. "Delaunay refinement algorithms for triangular
//! mesh generation." *Computational Geometry*, 22(1–3): 21–74.
//! DOI: `10.1016/S0925-7721(01)00047-5`.
//!
//! Phase 2A extends the current clean-room constrained Delaunay refinement with
//! Chew's off-center variants and Shewchuk's **Terminator** concentric-shell
//! acute-angle handling.

pub use super::delaunay::{
    triangulate, AdapterError, CdtAdapter, CdtLocation, Constraint, ConstraintKind, Domain,
    DomainError, Feature, Loop, Point, PointLocation, QualityRequest, QualityStats, RefinedMesh,
    RefinementError, RefinementReport, Region, Triangle, Triangulation, TriangulationError,
};
