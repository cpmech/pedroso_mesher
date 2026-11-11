//! 3D FEA Piecewise Linear Complex (PLC) domain specification with regions,
//! holes, and marked vertices, edges, and facets.

use serde::{Deserialize, Serialize};

/// Region seed point assigning a solid material `marker` and optional `max_volume`
/// to the 3D enclosed region containing `point`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RegionSeed3D {
    pub point: [f64; 3],
    pub marker: i32,
    pub max_volume: Option<f64>,
}

/// Planar polygonal facet (typically a triangle or quad from a CAD B-rep tessellation)
/// carrying an FEA face `marker`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MarkedFacet3D {
    pub vertices: Vec<usize>,
    pub marker: i32,
}

/// 3D Piecewise Linear Complex (PLC) for tetrahedral and hexahedral mesh generation.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Domain3D {
    /// 3D vertex coordinates `[x, y, z]`.
    pub points: Vec<[f64; 3]>,
    /// Optional per-vertex markers (`0` = unmarked).
    pub point_markers: Vec<i32>,
    /// Marked feature/constraint edges `(marker, [p0, p1])`.
    pub marked_edges: Vec<(i32, [usize; 2])>,
    /// Boundary and internal interface facets.
    pub facets: Vec<MarkedFacet3D>,
    /// Solid region classification seeds.
    pub regions: Vec<RegionSeed3D>,
    /// Interior seed points identifying void cavities/holes.
    pub holes: Vec<[f64; 3]>,
}
