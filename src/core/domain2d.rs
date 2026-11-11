//! Planar 2D FEA domain specification (Piecewise Straight-Line Graph / PSLG)
//! with regions, holes, and vertex/edge boundary markers.
//!
//! Supports defining regions and holes either by closed boundary loops or by
//! interior seed points, matching `gemlab` FEA conventions.

use serde::{Deserialize, Serialize};

/// Region seed point assigning a material/region `marker` and optional `max_area`
/// to the connected planar sub-domain containing `point`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RegionSeed2D {
    pub point: [f64; 2],
    pub marker: i32,
    pub max_area: Option<f64>,
}

/// Marked planar vertex `(coords, marker)`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MarkedVertex2D {
    pub coords: [f64; 2],
    pub marker: i32,
}

/// Constrained planar segment indexing two vertices in [`Domain2D::vertices`],
/// carrying an optional FEA boundary/interface `marker`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MarkedSegment2D {
    pub endpoints: [usize; 2],
    pub marker: i32,
}

/// Planar Piecewise Straight-Line Graph (PSLG) domain for 2D FEA mesh generation.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Domain2D {
    /// Input PSLG vertices with optional point markers.
    pub vertices: Vec<MarkedVertex2D>,
    /// Constrained boundary, hole, material-interface, or internal feature segments.
    pub segments: Vec<MarkedSegment2D>,
    /// Region classification seeds; if empty, interior faces receive marker `1`.
    pub regions: Vec<RegionSeed2D>,
    /// Interior seed points identifying holes (void sub-domains to remove).
    pub holes: Vec<[f64; 2]>,
}

impl Domain2D {
    /// Construct an empty PSLG domain.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a vertex `(x, y)` with `marker` (`0` = unmarked), returning its index.
    pub fn add_vertex(&mut self, coords: [f64; 2], marker: i32) -> usize {
        let id = self.vertices.len();
        self.vertices.push(MarkedVertex2D { coords, marker });
        id
    }

    /// Add a constrained segment `[a, b]` with `marker` (`0` = unmarked).
    pub fn add_segment(&mut self, a: usize, b: usize, marker: i32) {
        self.segments.push(MarkedSegment2D {
            endpoints: [a, b],
            marker,
        });
    }

    /// Add a closed polygonal loop with a uniform segment `marker`.
    pub fn add_closed_loop(&mut self, loop_pts: &[[f64; 2]], marker: i32) -> Vec<usize> {
        let mut ids = Vec::with_capacity(loop_pts.len());
        for &pt in loop_pts {
            ids.push(self.add_vertex(pt, 0));
        }
        let n = ids.len();
        for i in 0..n {
            self.add_segment(ids[i], ids[(i + 1) % n], marker);
        }
        ids
    }
}
