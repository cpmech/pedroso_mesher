//! One configuration for every quad route.
//!
//! [`MeshConfig`] follows the `russell_nonlin` pattern — defaults that reproduce
//! the measured tables, plus chainable setters — and it is also
//! `Serialize`/`Deserialize`, so the same configuration can be loaded from JSON,
//! recorded with a study, and passed across the wasm boundary.

use serde::{Deserialize, Serialize};

use crate::mesh::Mesh;
use crate::smooth::{self, SmoothOptions, SmoothOutcome};
use crate::stats::MeshStatistics;
use crate::two_d::{Domain, Sizing};

/// Which quad generator to run.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Route {
    /// The shipped Q-Morph conversion (supported on 4 of the 7 corpus parts).
    Qmorph,
    /// Blacker's direct all-quad advancing front plus a matched core.
    Paving,
    /// Right-angled triangles plus bipartite labelling (all-quad by construction).
    Bipartite,
    /// Cleanup-free strongly balanced quadtree (the general default).
    #[default]
    Quadtree,
    /// 4-sided patch decomposition with transfinite templates.
    Decompose,
    /// Quad-dominant: Blossom boundary band plus greedy interior.
    Hybrid,
    /// Cross-field-aligned boundary rows plus a matched core.
    Crossfield,
    /// The product pipeline plus the log-barrier untangler (a post-pass route).
    Untangle,
    /// Frontal Delaunay in the L∞ (Chebyshev) norm.
    L1,
}

impl Route {
    /// All routes, in shipping order.
    pub const ALL: [Route; 9] = [
        Route::Qmorph,
        Route::Paving,
        Route::Bipartite,
        Route::Quadtree,
        Route::Decompose,
        Route::Hybrid,
        Route::Crossfield,
        Route::Untangle,
        Route::L1,
    ];

    /// The name used in tables, JSON and logs.
    pub fn name(self) -> &'static str {
        match self {
            Route::Qmorph => "qmorph",
            Route::Paving => "paving",
            Route::Bipartite => "bipartite",
            Route::Quadtree => "quadtree",
            Route::Decompose => "decompose",
            Route::Hybrid => "hybrid",
            Route::Crossfield => "crossfield",
            Route::Untangle => "untangle",
            Route::L1 => "l1",
        }
    }
}

/// The configuration of one mesh generation: the route, its own options, and
/// the shared smoothing stage.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MeshConfig {
    /// Which generator to run.
    pub route: Route,
    /// The smoother applied after the generator (on by default).
    pub smoothing: SmoothOptions,
    pub quadtree: crate::alt_quadtree::Options,
    pub paving: crate::alt_paving::Options,
    pub bipartite: crate::alt_bipartite::Options,
    pub l1: crate::alt_l1::Options,
    pub crossfield: crate::alt_crossfield::Options,
    pub hybrid: crate::alt_hybrid::Options,
    pub untangle: crate::alt_untangle::Options,
    pub decompose: crate::alt_decompose::Options,
}

impl MeshConfig {
    /// A configuration for one route, everything else at its default.
    pub fn new(route: Route) -> Self {
        Self {
            route,
            ..Self::default()
        }
    }

    /// Select the route.
    pub fn set_route(&mut self, route: Route) -> &mut Self {
        self.route = route;
        self
    }

    /// Replace the smoothing options.
    pub fn set_smoothing(&mut self, smoothing: SmoothOptions) -> &mut Self {
        self.smoothing = smoothing;
        self
    }

    /// Serialise to pretty JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Parse the JSON form.
    pub fn from_json(text: &str) -> Result<Self, String> {
        serde_json::from_str(text).map_err(|e| e.to_string())
    }
}

/// What one generation produced.
#[derive(Clone, Debug)]
pub struct MeshOutput {
    /// The returned mesh (smoothed when the smoother was accepted).
    pub mesh: Mesh,
    /// Statistics of the returned mesh.
    pub statistics: MeshStatistics,
    /// Statistics right after the generator, before smoothing.
    pub before: MeshStatistics,
    /// The route that ran.
    pub route: Route,
    /// The smoothing run, if it was attempted.
    pub smoothing: Option<SmoothOutcome>,
    /// Wall time of the whole generation.
    pub seconds: f64,
}

/// Run `config.route` and then the shared smoother.
///
/// `Ok` does not imply that the mesh passes the gate: inspect
/// [`MeshOutput::statistics`] (or `generate_quad` for the
/// `Unsupported` error). `Err` means the route itself could not produce a mesh.
pub fn generate(
    domain: &Domain,
    sizing: &Sizing,
    config: &MeshConfig,
) -> Result<MeshOutput, String> {
    let started = std::time::Instant::now();
    let mesh = match config.route {
        Route::Qmorph => match crate::quad::mesh_quads(domain, sizing) {
            Ok(q) => q.mesh,
            Err(crate::quad::QuadError::Unsupported(u)) => u.mesh,
            Err(e) => return Err(e.to_string()),
        },
        Route::Paving => crate::alt_paving::mesh_with_options(domain, sizing, &config.paving)?,
        Route::Bipartite => {
            crate::alt_bipartite::mesh_with_options(domain, sizing, &config.bipartite)?
        }
        Route::Quadtree => {
            crate::alt_quadtree::mesh_with_options(domain, sizing, &config.quadtree)?
        }
        Route::Decompose => {
            crate::alt_decompose::mesh_with_options(domain, sizing, &config.decompose)?
        }
        Route::Hybrid => crate::alt_hybrid::mesh_with_options(domain, sizing, &config.hybrid)?,
        Route::Crossfield => {
            crate::alt_crossfield::mesh_with_options(domain, sizing, &config.crossfield)?
        }
        Route::Untangle => {
            crate::alt_untangle::mesh_with_options(domain, sizing, &config.untangle)?
        }
        Route::L1 => crate::alt_l1::mesh_with_options(domain, sizing, &config.l1)?,
    };
    let before = MeshStatistics::of(domain, &mesh);
    let outcome = smooth::smooth(domain, &mesh, sizing.size, &config.smoothing);
    let (mesh, statistics) = if outcome.accepted {
        (outcome.smoothed.clone(), outcome.after.clone())
    } else {
        (mesh, before.clone())
    };
    Ok(MeshOutput {
        mesh,
        statistics,
        before,
        route: config.route,
        smoothing: Some(outcome),
        seconds: started.elapsed().as_secs_f64(),
    })
}

/// Run any route and wrap the result in the quad module's [`crate::quad::QuadMesh`].
///
/// A mesh that does not pass the gate is returned as
/// [`crate::quad::QuadError::Unsupported`] carrying the mesh, exactly like the
/// shipped Q-Morph entry point.
pub fn generate_quad(
    domain: &Domain,
    sizing: &Sizing,
    config: &MeshConfig,
) -> Result<crate::quad::QuadMesh, crate::quad::QuadError> {
    use crate::quad::{QuadError, QuadMesh, Route as Conversion, Stats, Timing, Unsupported};
    let out = generate(domain, sizing, config).map_err(QuadError::Route)?;
    let gate = out.statistics.gate;
    let stats = Stats {
        rounds: 0,
        cavity_closures: 0,
        low_repairs: 0,
        degenerate_repairs: 0,
        quality_swaps: 0,
        perturbations: 0,
        parity_splits: 0,
        quads_only_sj: out.statistics.min_scaled_jacobian,
        quads_only_angle_deg: out.statistics.min_corner_angle_deg,
        boundary_offset: out.statistics.boundary_offset,
        corner_offset: out.statistics.corner_offset,
        two_stage: None,
    };
    let timing = Timing {
        triangulation: 0.0,
        conversion: 0.0,
        total: out.seconds,
    };
    let refusals = crate::qmorph::Refusals::default();
    if !gate.passes() {
        return Err(QuadError::Unsupported(Box::new(Unsupported {
            gate,
            raw_gate: gate,
            stats,
            timing,
            route: Conversion::Front,
            reason: gate.reason().unwrap_or_default(),
            refusals,
            mesh: out.mesh,
        })));
    }
    Ok(QuadMesh {
        mesh: out.mesh,
        gate,
        raw_gate: gate,
        stats,
        timing,
        refusals,
        route: Conversion::Front,
        generator: config.route,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::two_d::Sizing;

    fn square() -> (Domain, Sizing) {
        (
            Domain::rectangle(2.0, 2.0).unwrap(),
            Sizing::uniform(0.5).unwrap(),
        )
    }

    #[test]
    fn the_default_config_round_trips_through_json() {
        let text = MeshConfig::default().to_json();
        let back = MeshConfig::from_json(&text).unwrap();
        assert_eq!(back.route, MeshConfig::default().route);
        assert_eq!(
            back.quadtree.cell_factor,
            MeshConfig::default().quadtree.cell_factor
        );
    }

    #[test]
    fn every_route_generates_a_mesh_on_a_square() {
        let (domain, sizing) = square();
        for route in Route::ALL {
            let config = MeshConfig::new(route);
            let out = generate(&domain, &sizing, &config).unwrap();
            assert!(
                !out.mesh.cells.is_empty(),
                "{} produced no cells",
                route.name()
            );
            assert_eq!(out.route, route);
        }
    }

    #[test]
    fn the_default_route_is_the_quadtree_and_it_passes_a_square() {
        let (domain, sizing) = square();
        let config = MeshConfig::default();
        assert_eq!(config.route, Route::Quadtree);
        let out = generate(&domain, &sizing, &config).unwrap();
        assert!(out.statistics.passes);
        assert!(out.statistics.conforming);
    }

    #[test]
    fn the_quad_view_records_the_generator() {
        let (domain, sizing) = square();
        let config = MeshConfig::new(Route::Quadtree);
        let q = generate_quad(&domain, &sizing, &config).unwrap();
        assert_eq!(q.generator, Route::Quadtree);
    }

    #[test]
    fn smoothing_can_be_switched_off() {
        let (domain, sizing) = square();
        let mut config = MeshConfig::new(Route::Bipartite);
        config.smoothing.enabled = false;
        let out = generate(&domain, &sizing, &config).unwrap();
        assert!(out.smoothing.is_some());
        assert_eq!(out.statistics.quads, out.before.quads);
    }

    /// The stepped part of the route measurements (the one whose quad count used
    /// to alternate between 2197 and 2198).
    fn stepped() -> Domain {
        Domain::new(
            vec![
                [0.0, 0.0],
                [60.0, 0.0],
                [60.0, 40.0],
                [30.0, 40.0],
                [30.0, 30.0],
                [20.0, 30.0],
                [20.0, 40.0],
                [0.0, 40.0],
            ],
            vec![],
        )
        .unwrap()
    }

    /// Two runs of the same configuration, in the same process, must produce
    /// identical meshes.
    ///
    /// The hash collections in the shared stages are iterated in a canonical
    /// (sorted-key) order precisely so that this holds: before that, the process
    /// hash seed changed cavity/removal order, matching edge order and node
    /// perturbation order, and the same binary returned 2197 or 2198 quads on the
    /// stepped part depending on the run.
    #[test]
    fn two_generations_in_one_process_are_identical() {
        // The all-quad cascade is the most order-sensitive two_d path.
        let plate = Domain::plate_with_hole(20.0, 12.0, 3.0, 16).unwrap();
        let sizing = Sizing::uniform(2.0).unwrap().all_quad();
        let first = crate::two_d::mesh_2d(&plate, &sizing).unwrap();
        let again = crate::two_d::mesh_2d(&plate, &sizing).unwrap();
        assert_eq!(
            first.mesh.points, again.mesh.points,
            "two_d::mesh_2d moved its points between in-process runs"
        );
        assert_eq!(
            first.mesh.cells, again.mesh.cells,
            "two_d::mesh_2d changed its cells between in-process runs"
        );

        let stepped = stepped();
        // Coarse enough to keep the test quick; at this size the matching used to
        // pick different equal-weight pairs under two hash seeds.
        let sizing = Sizing::uniform(1.75).unwrap();
        let mut config = MeshConfig::new(Route::Crossfield);
        // The generator is what the hash order reaches; the smoother is a fixed
        // Jacobi/objective sweep and only multiplies the test's cost.
        config.smoothing.enabled = false;
        let first = generate(&stepped, &sizing, &config).unwrap();
        let again = generate(&stepped, &sizing, &config).unwrap();
        assert_eq!(
            first.mesh.points, again.mesh.points,
            "crossfield moved its points between in-process runs"
        );
        assert_eq!(
            first.mesh.cells, again.mesh.cells,
            "crossfield changed its cells between in-process runs"
        );
    }
}
