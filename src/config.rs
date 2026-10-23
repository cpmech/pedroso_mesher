//! The shipped configuration API: one [`MeshConfig`] selects a [`Route`],
//! carries that route's own `Options`, and runs the shared smoothing stage after
//! the generator.
//!
//! [`generate`] is the product entry point: it dispatches to the route, measures
//! the result, runs [`smooth::smooth`], and keeps the smoothed mesh only when
//! the smoother accepted it (otherwise the generator's mesh and its statistics
//! are returned). [`generate_quad`] adapts the same run to the legacy
//! [`crate::quad::QuadMesh`] view: a mesh that fails the gate is
//! [`crate::quad::QuadError::Unsupported`], and a route that cannot generate at
//! all is [`crate::quad::QuadError::Route`].
//!
//! Nine routes are shipped behind this enum. Each is implemented from the public
//! paper named on its variant and measured on the crate corpus (six planar parts
//! plus the 0.6 mm performance part); the alternative routes' `Options` defaults
//! reproduce the constants they were measured with. The papers are cited by
//! corpus id (`owen-1999`, `blacker-1991`, ...), never copied as code; see
//! `AGENTS.md` for the clean-room rules.
//!
//! The design follows the `russell_nonlin` pattern: defaults reproduce the
//! measured tables, the setters are chainable, and `Serialize`/`Deserialize`
//! makes the same configuration loadable from JSON, recordable with a study, and
//! passable across the wasm boundary.
//!
//! Shipping pattern (the quadtree is the default route):
//!
//! ```
//! use pedroso_mesher::config::{generate, generate_quad, MeshConfig, Route};
//! use pedroso_mesher::two_d::{Domain, Sizing};
//!
//! # fn main() -> Result<(), String> {
//! let domain = Domain::rectangle(2.0, 2.0).map_err(|e| e.to_string())?;
//! let sizing = Sizing::uniform(0.5).map_err(|e| e.to_string())?;
//!
//! // One config, one call. `statistics.passes` is the acceptance gate; the
//! // generator's own numbers are in `before`.
//! let config = MeshConfig::new(Route::Quadtree);
//! let out = generate(&domain, &sizing, &config)?;
//! assert!(out.statistics.passes);
//!
//! // The legacy view returns the same mesh with the conversion statistics.
//! let quad = generate_quad(&domain, &sizing, &config).map_err(|e| e.to_string())?;
//! assert_eq!(quad.generator, Route::Quadtree);
//! # Ok(())
//! # }
//! ```

use serde::{Deserialize, Serialize};

use crate::mesh::Mesh;
use crate::smooth::{self, SmoothOptions, SmoothOutcome};
use crate::stats::MeshStatistics;
use crate::two_d::{Domain, Sizing};

/// Which quad generator [`MeshConfig::route`] selects.
///
/// The variants are in shipping order ([`Route::ALL`]). Each doc names the
/// paper (by corpus id), the route's character (direct, indirect, grid or
/// post-pass, and whether it promises all-quad), and the measured table that
/// its example program prints. Measurements are on the crate corpus
/// (six planar parts plus the 0.6 mm performance part) with the route's default
/// `Options`; `raw` is the route output and `post` is after the common
/// smoothing/swap/perturbation post-pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Route {
    /// **Q-Morph** — indirect. Paper `owen-1999` (advancing front over a
    /// triangle substrate), with `sun-2015`/`sun-2016` closing the residual
    /// triangles. Character: the gate-checked shipped conversion; it returns a
    /// mesh only if it passes the 30 degree / SJ 0.2 gate, so 4 of the 7 corpus
    /// parts are supported.
    /// Measured table: `examples/qmorph.rs` (raw gate rows; `UNSUPPORTED` rows
    /// are the carried failures, not shipped meshes).
    Qmorph,

    /// **Paving** — direct. Paper `blacker-1991` (advancing rows, with the
    /// §2.8 seam). Character: all-quad boundary rows towards the interior; the
    /// residual core is closed by the shared CDT + matching fallback.
    /// Measured table: `examples/alt_paving.rs`; 7/7 PASS raw and post, raw SJ
    /// 0.62–0.70 and corner 38–44 degrees, about 15k quads at 0.6 mm.
    Paving,

    /// **Bipartite** — indirect. Paper `georgiadis-2022` (bipartite topological
    /// labelling). Character: a right-angled triangle mesh is 2-coloured so the
    /// monochromatic edges are a perfect matching; all-quad by construction.
    /// Measured table: `examples/alt_bipartite.rs`; 7/7 PASS raw and post, post
    /// SJ 0.69–0.82 and corner 47.5–61.8 degrees (0.6 mm post about 11 s).
    Bipartite,

    /// **Quadtree** — grid-based, the general default. Papers `rushdi-2017` and
    /// its precursor `rushdi-2015` (cleanup-free strongly balanced quadtree).
    /// Character: grid/geometry intersection plus 2-refinement templates;
    /// all-quad with no cleanup, but the mesh stays grid-aligned (no boundary
    /// layer, curved boundaries become chords).
    /// Measured table: `examples/alt_quadtree.rs`; 7/7 PASS raw and post, post
    /// SJ 0.63–1.00 and corner 42.9–90 degrees, 14 900 quads at 0.6 mm in
    /// 0.16 s raw.
    #[default]
    Quadtree,

    /// **Decompose** — decomposition and templates. Papers `yang-2024`
    /// (boundary-constrained decomposition) and `miranda-2017` (T0/T1/T2
    /// transfinite templates). Character: recursive chords until every patch
    /// has four corners, then a discrete Coons map; exact on axis-aligned
    /// parts, but the 8-patch O-grid ring around a hole limits quality.
    /// Measured table: `examples/alt_decompose.rs`; all-quad and watertight
    /// 7/7, raw 7/7 PASS; after the common post-pass the plate-hole 1 mm and
    /// 0.6 mm rows can fall below the 30 degree gate on that ring.
    Decompose,

    /// **Hybrid** — indirect, quad-dominant. Paper `zint-2021`, using the
    /// shipped Blossom-Quad matching (`remacle-2012`) on the boundary band.
    /// Character: fast Blossom band plus greedy interior merging and a residual
    /// triangle walk; it does not promise all-quad, and trades leftover
    /// triangles for speed. Measured table: `examples/alt_hybrid.rs`; 0.6 mm
    /// raw in about 0.8 s with 190 triangles left (FAIL), an order of magnitude
    /// faster than the matching routes.
    Hybrid,

    /// **Crossfield** — direct-ish. Papers `pellenard-2014`, `pellenard-2011`
    /// and `remacle-2013` §3. Character: a cross field on a background grid
    /// drives field-aligned paving rows; the interior is closed by the
    /// Q-Morph/matching stage. Measured table: `examples/alt_crossfield.rs`;
    /// all-quad 7/7 and all seven PASS after the common post-pass (post SJ
    /// 0.23–0.78, corner 30–55 degrees); the 0.6 mm part needs the pre-walk
    /// fallback and gains about 5.1k quads (20 586).
    Crossfield,

    /// **Untangle** — post-pass over the product pipeline. Papers `sastry-2014`
    /// (log-barrier untangling) and `verma-2017`/`verma-2018` (singularity
    /// reduction by edge swaps). Character: keeps the mesh a failing gate
    /// returned, then moves interior nodes and swaps interior edges; it cannot
    /// create or remove cells, so leftover triangles stay.
    /// Measured table: `examples/alt_untangle.rs`; 4/7 PASS; the plate-hole
    /// 1 mm boundary doublet (SJ 0) and the ring/performance triangles stay
    /// failing, while the valence-4 share rises by 1–6 points on passing parts.
    Untangle,

    /// **L1** — indirect (frontal Delaunay plus recombination). Paper
    /// `remacle-2013`; the paper's "L-norm" is the infinity (Chebyshev) norm,
    /// as `alt_l1` explains. Character: cross-field-driven L-infinity point
    /// placement, then maximum-weight Blossom matching of triangles into quads
    /// (exact below `EXACT_MATCHING_TRIANGLES`, blocked plus greedy above it).
    /// Measured table: `examples/alt_l1.rs`; all-quad 7/7 and post PASS 7/7
    /// with the best quality of the shipped routes (post SJ 0.66–0.98, corner
    /// 41.6–78.7 degrees; the 0.6 mm row uses the blocked matching path).
    L1,

    /// **Sized** — direct (the Simons 2024 thesis pipeline). Chapter 2 grid and
    /// boundary channel (`simons_channel`); Chapters 3–5 are available as
    /// `simons_godf`, `simons_sizefamily` and `simons_qgp`. Character: a
    /// regular grid interior, a carved channel and a boundary ribbon (or the
    /// template cover), all-quad and exactly on the boundary. Measured table
    /// (`generate -- sized <part>`, with the shared smoother + shape pass):
    /// plate-hole 4 mm 262 quads SJ 0.608 / 38.2 degrees, 2 mm 1240 / 0.564 /
    /// 42.5, 1 mm 5324 / 0.481 / 35.3, 0.6 mm 15520 / 0.513 / 36.4;
    /// l-bracket 600 / 1.000 / 90 and stepped 2300 / 1.000 / 90 (perfect
    /// grids); ring 48 / 0.998 / 87.5 (two-chain strip). **7 of the 7 corpus
    /// parts** pass the gate with the shared smoother + shape pass.
    Sized,

    /// **Features** — the Simons 2024 interior-feature constraints (Chapter 2):
    /// a through-going polyline splits the domain and both halves are meshed
    /// with `sized`/fallbacks and welded along one sampled seam; a terminating
    /// polyline becomes a thin slot (a hole) at the measured size. The feature
    /// polylines come from [`MeshConfig::features`]; the route needs at least
    /// one.
    Features,
}

impl Route {
    /// All eleven routes, in shipping order (the order the tables and the
    /// configuration examples use). Every route in this array is callable by
    /// [`generate`]; not every route passes the gate on every part.
    pub const ALL: [Route; 11] = [
        Route::Qmorph,
        Route::Paving,
        Route::Bipartite,
        Route::Quadtree,
        Route::Decompose,
        Route::Hybrid,
        Route::Crossfield,
        Route::Untangle,
        Route::L1,
        Route::Sized,
        Route::Features,
    ];

    /// The stable lowercase name used in tables, JSON and logs.
    ///
    /// These strings are the command-line route names accepted by
    /// `examples/generate.rs`; changing one breaks saved configurations.
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
            Route::Sized => "sized",
            Route::Features => "features",
        }
    }
}

/// One complete mesh generation request: the route, that route's own options,
/// and the shared post-generation smoother.
///
/// The defaults are the constants each route was measured with, so
/// `MeshConfig::default()` is the default product configuration: the quadtree
/// route with smoothing on. The struct is `Serialize`/`Deserialize`; the route
/// options are embedded by value, so a saved JSON file is a complete recipe.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MeshConfig {
    /// Which generator to run. See [`Route`] for the paper, character and
    /// measured table of each variant.
    pub route: Route,

    /// The shared smoother applied after the generator (on by default). It
    /// moves interior nodes only and is rejected when it does not improve the
    /// gate margin; its fields are documented on [`SmoothOptions`].
    pub smoothing: SmoothOptions,

    /// Options of the `quadtree` route: grid density, number of alignments,
    /// repelling distance and retry budget.
    pub quadtree: crate::alt_quadtree::Options,

    /// Options of the `paving` route: corner margin, minimum edge, front gap,
    /// row budget and §2.8 seaming.
    pub paving: crate::alt_paving::Options,

    /// Options of the `bipartite` route: right-angle flip, labelling, repair
    /// and Steiner budgets.
    pub bipartite: crate::alt_bipartite::Options,

    /// Options of the `l1` route: the resolution threshold `h_max` and the
    /// minimum insertion spacing.
    pub l1: crate::alt_l1::Options,

    /// Options of the `crossfield` route: Gauss-Seidel sweeps of the field.
    pub crossfield: crate::alt_crossfield::Options,

    /// Options of the `hybrid` route: boundary-band depth and the augmenting
    /// path / triangle-walk budgets.
    pub hybrid: crate::alt_hybrid::Options,

    /// Options of the `untangle` route: coordinate-search rounds, accepted-swap
    /// cap and evaluation budget.
    pub untangle: crate::alt_untangle::Options,

    /// Options of the `decompose` route: corner threshold (degrees) and the
    /// recursion-depth safety limit.
    pub decompose: crate::alt_decompose::Options,

    /// Options of the `sized` route (the Simons 2024 channel route; currently
    /// empty because the thesis knobs are fixed at their measured defaults).
    pub sized: crate::sized::Options,

    /// Interior feature polylines for [`Route::Features`] (empty otherwise).
    /// A through-going polyline (both endpoints on the domain boundary) splits
    /// the region; a terminating one becomes a slot (a thin hole).
    pub features: Vec<Vec<[f64; 2]>>,
}

impl MeshConfig {
    /// A configuration for one route, with that route's options, the smoother
    /// and every other route's options at their defaults.
    ///
    /// This is the recommended constructor: it keeps the non-selected option
    /// blocks at the measured defaults, which is what `MeshConfig::default()`
    /// also does.
    pub fn new(route: Route) -> Self {
        Self {
            route,
            ..Self::default()
        }
    }

    /// Select the route and return `self` for chaining.
    ///
    /// The other fields, including all option blocks, are unchanged; switching
    /// route therefore keeps defaults for the new route.
    pub fn set_route(&mut self, route: Route) -> &mut Self {
        self.route = route;
        self
    }

    /// Replace the shared smoothing options and return `self` for chaining.
    pub fn set_smoothing(&mut self, smoothing: SmoothOptions) -> &mut Self {
        self.smoothing = smoothing;
        self
    }

    /// Serialise the whole configuration to pretty JSON.
    ///
    /// Serialisation of this plain struct cannot fail; the fallback `"{}"` only
    /// keeps the return type infallible (a caller that needs the error should
    /// use `serde_json` directly).
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Parse a configuration from JSON produced by [`Self::to_json`].
    ///
    /// The error is the parser's message as text; a missing field is an error
    /// (there are no serde defaults on [`MeshConfig`]), so a saved file must
    /// come from a compatible version.
    pub fn from_json(text: &str) -> Result<Self, String> {
        serde_json::from_str(text).map_err(|e| e.to_string())
    }
}

/// What one [`generate`] call produced.
///
/// `statistics` describes `mesh`; `before` is the same measurement immediately
/// after the route and before smoothing. Keeping both makes the effect of the
/// smoother explicit, and `smoothing` retains the full outcome (including a
/// candidate the smoother rejected).
#[derive(Clone, Debug)]
pub struct MeshOutput {
    /// The returned mesh: the smoothed mesh when the smoother accepted it,
    /// otherwise the generator's mesh unchanged. Coordinates are in the
    /// domain's units (millimetres for the corpus).
    pub mesh: Mesh,

    /// Statistics of [`Self::mesh`]: after the accepted smoothing, if any, and
    /// otherwise equal to [`Self::before`].
    pub statistics: MeshStatistics,

    /// Statistics immediately after the route and before smoothing. This is
    /// the mesh the route's measured table is computed on.
    pub before: MeshStatistics,

    /// The route that ran, copied from [`MeshConfig::route`].
    pub route: Route,

    /// The smoothing run, if it was attempted. `Some` even when the smoother
    /// was disabled or rejected the candidate; [`SmoothOutcome::accepted`]
    /// says whether it replaced the mesh.
    pub smoothing: Option<SmoothOutcome>,

    /// Wall-clock seconds for the route and the smoother together (not the
    /// caller's domain or size setup).
    pub seconds: f64,
}

/// Ship one mesh: run the configured route, measure it, run the shared
/// smoother, and return the accepted result.
///
/// The order is fixed:
///
/// 1. dispatch on [`MeshConfig::route`] and generate with that route's options.
///    [`Route::Qmorph`] runs the gate-checked [`crate::quad::mesh_quads`] and
///    keeps the mesh carried by `QuadError::Unsupported` (the carried mesh is
///    exactly what the measurement and the smoother need);
/// 2. measure the generator result with [`MeshStatistics::of`] (`before`);
/// 3. run [`smooth::smooth`] with [`MeshConfig::smoothing`];
/// 4. return the smoothed mesh and its statistics when the smoother accepted
///    them, otherwise the generator mesh and `before` unchanged.
///
/// [`Sizing::size`] is the target edge length in domain units; both the
/// generator and the smoother use it. `Ok` does not imply that the gate
/// passes: inspect [`MeshOutput::statistics`] (`passes`) or use
/// [`generate_quad`] for the gate-as-error contract. `Err` means a route could
/// not produce a mesh at all; the string is that route's error (for the
/// Q-Morph route, an input or conversion error).
pub fn generate(
    domain: &Domain,
    sizing: &Sizing,
    config: &MeshConfig,
) -> Result<MeshOutput, String> {
    let started = std::time::Instant::now();
    // The route is the only branch in this function; each arm either returns
    // its own error type (converted by `?`) or a Mesh. The Q-Morph arm is the
    // only one whose error can still carry a usable mesh.
    let mesh = match config.route {
        Route::Qmorph => match crate::quad::mesh_quads(domain, sizing) {
            Ok(q) => q.mesh,
            // The shipped route refuses a mesh below the gate; the carried mesh
            // is still a valid geometry and is what the smoother can improve.
            Err(crate::quad::QuadError::Unsupported(u)) => u.mesh,
            // Input/route errors have no usable mesh, so they propagate.
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
        Route::Sized => crate::sized::mesh_with_options(domain, sizing, &config.sized)?,
        Route::Features => {
            if config.features.is_empty() {
                return Err("the features route needs at least one feature polyline".into());
            }
            let features = crate::simons_features::Features {
                polylines: config.features.clone(),
            };
            crate::simons_features::mesh(domain, &features, sizing)?
        }
    };
    // Step 2: the generator's own row. Measuring before smoothing is what lets
    // a caller compare a route against its measured table and see the smoother's
    // contribution separately.
    let before = MeshStatistics::of(domain, &mesh);
    // Step 3: one smoother for every route. It moves interior nodes only;
    // `reject_if_worse` decides whether its candidate replaces the input.
    let outcome = smooth::smooth(domain, &mesh, sizing.size, &config.smoothing);
    // Step 4: only an accepted smoothing replaces the mesh. A rejected candidate
    // stays in `smoothing` for diagnostics, but the returned statistics stay the
    // generator's.
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
        // The clock starts before the dispatch, so this includes every stage
        // from route generation through the smoother.
        seconds: started.elapsed().as_secs_f64(),
    })
}

/// Run any route through [`generate`] and adapt the result to the legacy
/// [`crate::quad::QuadMesh`] view.
///
/// The two failure classes are deliberately distinct:
///
/// * a route that cannot generate at all is [`crate::quad::QuadError::Route`];
///   the string is the route's own error, e.g. an unusable domain;
/// * a generated mesh that does not pass the gate is
///   [`crate::quad::QuadError::Unsupported`], carrying the mesh, the measured
///   gate and the statistics, exactly like [`crate::quad::mesh_quads`].
///
/// The front-only counters (`rounds`, `cavity_closures`, `low_repairs`, ...) are
/// zero for the alternative routes; the quality fields come from the statistics
/// of the accepted mesh. The conversion route is reported as
/// [`crate::quad::Route::Front`] because this adapter only wraps the front
/// pipeline, while the new `generator` field records the actual [`Route`].
///
/// Note: unlike [`crate::quad::mesh_quads`], this adapter enforces the mesh gate
/// but not its separate boundary/corner offset checks. The offsets are carried
/// in [`crate::quad::Stats`], so a caller that needs the full geometry contract
/// must test them or call the front route directly.
pub fn generate_quad(
    domain: &Domain,
    sizing: &Sizing,
    config: &MeshConfig,
) -> Result<crate::quad::QuadMesh, crate::quad::QuadError> {
    use crate::quad::{QuadError, QuadMesh, Route as Conversion, Stats, Timing, Unsupported};
    // Route failures stay distinct from gate failures: only the latter carry a
    // mesh and a measured gate to diagnose.
    let out = generate(domain, sizing, config).map_err(QuadError::Route)?;
    let gate = out.statistics.gate;
    // The front's own counters do not exist for the alternative routes. The
    // quality fields are genuine measurements, so a report can still compare
    // shapes; `two_stage` stays None because no two-stage conversion ran.
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
    // Only the total time is known at this level; the route did not report its
    // own triangulation/conversion split (the Q-Morph front's `Timing` is not
    // preserved by `generate`).
    let timing = Timing {
        triangulation: 0.0,
        conversion: 0.0,
        total: out.seconds,
    };
    let refusals = crate::qmorph::Refusals::default();
    // The metric gate and the geometry guards are separate: a mesh can pass the
    // gate with a boundary node moved inward or a corner slid along an edge.
    // `mesh_quads` checks both, so this entry point must too.
    if !gate.passes() || !out.statistics.conforming {
        // Carry the mesh, exactly as `mesh_quads` does: the caller may draw or
        // inspect the failure, but must not ship it as an accepted result.
        // `raw_gate` is the same measurement here because this adapter has no
        // post-conversion gate separate from the accepted-mesh gate.
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
#[allow(clippy::unwrap_used, clippy::panic)]
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
        for route in Route::ALL {
            // The channel route needs room for an interior grid next to its
            // 1.4 h boundary cull; the other routes are happy on the small test
            // square.
            let (domain, sizing, features) = match route {
                Route::Sized => (
                    Domain::rectangle(10.0, 10.0).unwrap(),
                    crate::two_d::Sizing::uniform(1.0).unwrap(),
                    Vec::new(),
                ),
                Route::Features => (
                    Domain::rectangle(10.0, 10.0).unwrap(),
                    crate::two_d::Sizing::uniform(1.0).unwrap(),
                    vec![vec![[5.0, 0.0], [5.0, 10.0]]],
                ),
                _ => {
                    let (d, s) = square();
                    (d, s, Vec::new())
                }
            };
            let mut config = MeshConfig::new(route);
            config.features = features;
            let out = generate(&domain, &sizing, &config)
                .unwrap_or_else(|e| panic!("{} failed on its test domain: {e}", route.name()));
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
