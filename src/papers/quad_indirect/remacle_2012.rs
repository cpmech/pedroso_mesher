//! Remacle et al. (2012) — Blossom-Quad: A non-uniform quadrilateral mesh
//! generator using a minimum-cost perfect-matching algorithm.
//!
//! Reference: Remacle J.-F., Lambrechts J., Seny B., Marchandise E., Johnen A.,
//! Geuzaine C., 2012. *IJNME*, 89(9): 1102–1119. DOI: `10.1002/nme.3279`.
//!
//! Phase 2C Tier 1 upgrades this module from basic dual-graph Blossom pairing
//! (`two_d::mesh_2d`) to Remacle et al.'s exact augmented graph $G'$ (including
//! extra boundary/internal edges for odd-parity & non-convex triangle pairs) and
//! full §6 topological cleanup.

pub use crate::papers::common::two_d::{
    mesh_2d, pair_triangles_with, Domain, PairGuard, Sizing, TwoDError, TwoDResult, TwoDStats,
};
