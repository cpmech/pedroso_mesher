//! The Simons (2024) end-to-end thesis pipeline on a planar domain.
//!
//! This module runs the chain of the thesis on one input: the background
//! triangulation (Chapter 2's grid), the size-aware cross field (Chapter 3's
//! GODF solve), the QGP regions and their Coons patches (Chapter 5), the
//! stitch to the boundary with the Chapter 2 channel (Section 5.1.8.2), and
//! the angle + stretch smoothing of Section 2.2.10.
//!
//! # The pipeline
//!
//! 1. **Triangulation (Section 2.2.2).** An axis-aligned rectangle is filled
//!    with a structured grid of `base_size` cells, each split into two
//!    triangles and kept when its centroid is inside the domain. Any other
//!    domain (holes, a rotated or non-convex boundary, several loops) is
//!    triangulated with [`crate::two_d::triangulate`] at `base_size`, which
//!    already samples the boundary, constrains it and filters the triangles
//!    outside the domain.
//! 2. **Size function.** `sigma(x, y) = 1 + growth * x` by default, or any
//!    positive callback through [`mesh_with_size`]. It is sampled at the
//!    triangulation vertices.
//! 3. **Graded tile grid (Chapter 5, Sections 5.1.1-5.1.3, measured path).**
//!    When `sigma_max / sigma_min > 1.5`, [`Options::graded`] is set and the
//!    domain is a rectangle (or a rectangle with one axis-aligned rectangular
//!    hole), [`crate::simons_graded::mesh`] decomposes it into a tile grid
//!    snapped to the size metric, builds the one-edge-per-side QGP graph, runs
//!    the integer solve of eq. (5.1), and Coons-meshes every tile. This path
//!    needs no cross field: the tiling is built from the size function.
//!    Otherwise, or when it fails, the next steps run.
//!
//!    A second, opt-in graded route goes through Chapter 5 instead:
//!    [`Options::seeded`] runs the cross field, injects the Section 5.1.4
//!    valence-3/5 pairs ([`crate::simons_seeded`]), builds the seeded
//!    motorcycle graph and simple tiles, meshes their Coons patches and
//!    stitches the result. Its measured numbers are in "The graded path and
//!    its measured numbers" below: it does **not** beat the metric tile grid
//!    on the two task cases, so the metric route stays the default.
//!
//! 4. **Cross field (Section 3.3, Algorithm 1).**
//!    [`crate::simons_godf::solve_with_options`] with `accurate: true` (the
//!    curvature-aware GODF Appendix D integrals) and `holomorphic: 0` (the
//!    Dirichlet energy `A = Delta` the thesis uses).
//! 5. **QGP and stitch (Sections 5.0-5.1 and 5.1.8.2).**
//!    [`crate::simons_regions::build`] finds the singularities, the
//!    motorcycle graph and the simple tiles; [`Regions::mesh`] meshes every
//!    tile with its bilinear Coons patch; and [`crate::simons_regions::stitch`]
//!    carves the quads near the outer boundary and fills the band with
//!    [`crate::simons_channel`]. If any of those steps fails, the pipeline
//!    falls back to the Chapter 2 channel route ([`crate::sized::mesh`]) at a
//!    uniform size.
//! 6. **Smoothing (Section 2.2.10).** [`crate::simons_smooth`] with
//!    `stretch_weight = 1.0` and the boundary nodes fixed. The before/after
//!    statistics are returned in [`Trace::smoothing`].
//!
//! # Size accuracy
//!
//! Every quad's size is `sqrt(area)`; its target is `sigma` at the quad
//! centroid. The reported ratio is `max(r, 1/r)`, so 1 is exact and larger is
//! worse. [`Report::size_ratio_mean`] and [`Report::size_ratio_worst`] are the
//! mean and maximum over the quads.
//!
//! # Faithful, simplified and measured deviations
//!
//! Faithful: the structured grid interior, the size-aware GODF solve with the
//! accurate helper integrals, the singularity/motorcycle-graph/simple-tile
//! regions, the QGP integer counts and Coons patches, the carve + channel
//! stitch of Section 5.1.8.2, and the angle + stretch smoothing.
//!
//! Simplified:
//!
//! * The rectangle grid is generated from the axis-aligned bounding box, so a
//!   rectangle whose outer loop is not axis-aligned takes the CDT path.
//! * The CDT path uses the crate's uniform-size triangulation at `base_size`
//!   and only applies `sigma` afterwards, per vertex. The thesis refines the
//!   background mesh with the size field; that refinement is not implemented.
//! * The size function has no Chapter 4 minimum-anisotropy optimisation.
//! * The smoothing uses the crate's 2D simplex, not SciPy's Nelder-Mead, and
//!   its `size` argument is the mean `sqrt(area)` of the mesh.
//!
//! # The graded path and its measured numbers
//!
//! A size field that only grows with `x` produces a cross field with no
//! singularity on a simply connected rectangle (measured: `(0, 0)` for the
//! 10x10 square at growth 0.5). The motorcycle graph is then the single
//! boundary face, and its QGP solve collapses it (Section 5.1.6): the left and
//! right sides want about 10 and 1.7 subdivisions, so the zero count is the
//! energy optimum. That is why the cross-field route cannot grade this case.
//!
//! [`crate::simons_graded`] replaces the degenerate single tile by a grid of
//! simple tiles whose corners follow the size metric. Measured on the 10x10
//! square with `sigma = 1 + 0.5 x`, derived tile grid `4 x 4` (16 tiles,
//! 40 edges, every count 1): 16 quads, `SJ = 1.0`, `min_angle = 90`,
//! `cracks = 0`, `size_ratio_mean = 1.262`, `size_ratio_worst = 1.494`,
//! `corr(sqrt(area), sigma) = 0.996` (the uniform channel fallback was
//! `worst = 1.875`). For the 2D gradient `sigma = 1 + 0.3 x + 0.3 y` the
//! derived grid is `3 x 3`: 9 quads, `mean = 1.181`, `worst = 1.392`. An
//! explicit request for `8 x 8` tiles is *physically infeasible* on this
//! domain (the expected number of quads is `integral integral dA/sigma^2 ~
//! 9`, so 64 tiles cannot each carry one); the path reports
//! `requested = (8, 8)` and `coarsened = true` and meshes the derived grid.
//! Increasing the linear grid to `K = 5` gives `worst = 1.696` and `K = 6`
//! gives `1.878`; `K = 8` collapses (every column ideal count is below 0.5),
//! which is the stepwise-count trend behind the derived `K = 4`.
//!
//! ## The seeded Section 5.1.4 field path (measured, opt-in)
//!
//! [`Options::seeded`] runs GODF with `accurate = true` and
//! [`crate::simons_godf::BoundaryMode::Tangent`], injects
//! `pairs = 3` valence-3/5 pairs along the size gradient
//! ([`crate::simons_seeded::pair_positions`]), and builds the QGP tiles and the
//! stitch from the seeded field. Measured on the 10x10 square with
//! `sigma = 1 + 0.5 x`, `base_size = 1`:
//!
//! * the seeded field detects `(5, 5)` singularities and the motorcycle graph
//!   keeps 4 tiles (19 collapsed); the interior is 7 quads covering 25.9 % of
//!   the area with `worst = 1.557`. The pipeline rejects the attempt at the
//!   coverage guard (`< 98 %`) instead of stitching a mesh with holes; even
//!   bypassing the guard, the stitch fails for 1-4 pairs (every carve radius
//!   between `1.4 sigma` and `3.6 sigma` removes every quad, or the QGP solve
//!   collapses every tile);
//! * the 2D case `sigma = 1 + 0.3 x + 0.3 y` is the same (1 tile, 2 quads,
//!   21.4 % coverage, `worst = 1.410`).
//!
//! The reason is structural: at the target density (~17 quads over 10x10) the
//! sparse pair set cuts the domain into tiles that are still larger than the
//! local size, so most of them collapse (Section 5.1.6) and only a quarter of
//! the area is meshed. The one-tile
//! §5.1.4 relaxation cannot fix it either: for the whole-domain ideals
//! `[3.6, 1.7, 3.6, 10]`, a pair that relieves the vertical discrepancy
//! `10 - 1.7` also unbalances the horizontal pair by the same amount, so the
//! redirection penalty keeps the strict (collapsed) counts
//! ([`crate::simons_seeded::relaxed_counts`] returns `k = 0`). The pairs only
//! pay off after the motorcycle graph has already cut the domain into tiles
//! whose ideals are nearly consistent. Therefore `seeded = false` is the
//! default and [`SOURCE_GRADED`] remains the shipped graded route.
//!
//! When the graded path is not applicable or fails, the cross-field QGP +
//! stitch route runs exactly as before. When that route fails on a graded
//! rectangle, the channel fallback is still uniform (its first size is the
//! geometric mean `sqrt(sigma_min sigma_max)`, and four fractions around it
//! are also tried and the gate-passing mesh with the smallest worst ratio is
//! kept); its measured numbers are reported rather than claimed as a growth
//! result. Constant size, and domains the graded shape detector rejects, keep
//! the full QGP + stitch path.
//!
//! # Public API
//!
//! ```
//! use pedroso_mesher::simons_pipeline::{demo, mesh, Options};
//! use pedroso_mesher::Domain;
//!
//! # fn main() -> Result<(), String> {
//! let domain = Domain::rectangle(10.0, 10.0).map_err(|e| e.to_string())?;
//! let options = Options { base_size: 1.0, growth: 0.5, smooth: true, ..Options::default() };
//! let report = mesh(&domain, &options)?;
//! assert_eq!(report.triangles, 0);
//! let _ = demo()?;
//! # Ok(())
//! # }
//! ```
//!
//! The module has no external dependencies, never panics in non-test code, and
//! builds for `wasm32-unknown-unknown`.

use crate::mesh::{Kind, Mesh, Node};
use crate::quality;
use crate::simons_godf;
use crate::simons_graded;
use crate::simons_regions;
use crate::simons_seeded;
use crate::simons_smooth;
use crate::two_d::{self, Domain, Sizing};
use crate::verify;

/// Iterations of the inverse-power GODF solve (Algorithm 1).
const GODF_ITERATIONS: usize = 400;

/// Sweeps of the angle + stretch smoother.
const SMOOTH_ITERATIONS: usize = 30;

/// Stretch weight of the Section 2.2.10 objective.
const STRETCH_WEIGHT: f64 = 1.0;

/// Largest number of grid columns (or rows) the structured rectangle path
/// accepts. Beyond this the background mesh is not a useful input anyway.
const MAX_GRID: usize = 2048;

/// Largest number of structured grid cells the rectangle path accepts.
const MAX_CELLS: u128 = 1_000_000;

/// The QGP + stitch route.
pub const SOURCE_QGP: &str = "qgp+stitch";

/// The graded tile-grid QGP route of [`crate::simons_graded`].
pub const SOURCE_GRADED: &str = "graded-qgp";

/// The Chapter 5 seeded-field route of [`crate::simons_seeded`].
pub const SOURCE_SEEDED: &str = "seeded-qgp";

/// The Chapter 2 channel fallback route.
pub const SOURCE_CHANNEL: &str = "channel";

/// Size variation `sigma_max / sigma_min` above which the graded tile-grid path
/// is tried (the task's threshold).
pub const GRADED_VARIATION_THRESHOLD: f64 = 1.5;

/// Options for [`mesh`].
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Options {
    /// Edge length of the background grid, in domain units. The size function,
    /// not this value, sets the output element size; `base_size` only controls
    /// how finely the field and the motorcycle graph are resolved. Must be
    /// finite and positive.
    pub base_size: f64,

    /// Linear growth of the default size function `sigma(x, y) = 1 + growth *
    /// x`. Must be finite; every sampled `sigma` must stay positive.
    pub growth: f64,

    /// Run the angle + stretch smoothing of Section 2.2.10.
    pub smooth: bool,

    /// Requested `(K, L)` tile grid of the graded path. `(0, 0)` (the default)
    /// derives it from the size variation. Each entry is at most
    /// [`crate::simons_graded::MAX_TILES`].
    pub tiles: (usize, usize),

    /// Try the graded tile-grid path when `sigma_max / sigma_min` exceeds
    /// [`GRADED_VARIATION_THRESHOLD`] and the domain is a graded rectangle.
    /// Constant-size and general domains keep the QGP/stitch and channel
    /// routes.
    pub graded: bool,

    /// Prefer the Chapter 5 seeded-field route ([`SOURCE_SEEDED`]) on a graded
    /// rectangle: GODF with the accurate integrals and a tangent boundary, the
    /// Section 5.1.4 valence-3/5 pairs of [`crate::simons_seeded`], the seeded
    /// QGP tiles and the stitch. The default, `false`, keeps the metric tile
    /// grid. The seeded route falls back to the metric grid (or, when that is
    /// disabled, to the ordinary QGP/stitch and channel routes) when it cannot
    /// produce a domain-covering mesh. Measured: it does not beat the metric
    /// grid on the task cases (see the module documentation).
    pub seeded: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            base_size: 1.0,
            growth: 0.5,
            smooth: true,
            tiles: (0, 0),
            graded: true,
            seeded: false,
        }
    }
}

/// The before/after numbers of one smoothing run.
#[derive(Clone, Copy, Debug)]
pub struct Smoothing {
    /// Statistics of the mesh before smoothing.
    pub before: simons_smooth::Stats,
    /// Statistics of the mesh after smoothing.
    pub after: simons_smooth::Stats,
    /// Sweeps actually executed.
    pub sweeps: usize,
    /// Distinct nodes moved at least once.
    pub moved_nodes: usize,
}

/// How the pipeline reached the result. This is where the deviations are
/// reported, so a caller can print them instead of inferring them from the
/// mesh.
#[derive(Clone, Debug, Default)]
pub struct Trace {
    /// The graded tile-grid run, when that path was selected and succeeded.
    pub graded: Option<simons_graded::Info>,
    /// Why the graded path was not used, when it was requested and failed.
    pub graded_error: Option<String>,
    /// Number of Section 5.1.4 pairs placed by the seeded-field attempt.
    pub seeded_pairs: usize,
    /// `(valence-3, valence-5)` singularities the seeded field detected.
    pub seeded_detected: (usize, usize),
    /// Simple tiles of the seeded motorcycle graph.
    pub seeded_tiles: usize,
    /// Quads of the seeded interior mesh (before the stitch).
    pub seeded_interior_quads: usize,
    /// Fraction of the domain area covered by the seeded interior mesh.
    pub seeded_coverage: f64,
    /// Why the seeded-field path was not used or did not produce a mesh.
    pub seeded_error: Option<String>,
    /// The sampled size variation `sigma_max / sigma_min`.
    pub size_variation: f64,
    /// The QGP failure that forced the channel fallback, if any.
    pub qgp_error: Option<String>,
    /// Simple tiles that [`simons_regions::build`] produced (0 when it failed).
    pub qgp_tiles: usize,
    /// Quads of the QGP interior mesh before the carve (0 when it failed).
    pub qgp_interior_quads: usize,
    /// Degenerate boundary tiles that [`simons_regions::build`] skipped.
    pub qgp_degenerate_tiles: usize,
    /// QGP tiles collapsed by the integer solve (Section 5.1.6).
    pub qgp_collapsed_tiles: usize,
    /// Uniform size used by the channel fallback, if it ran.
    pub fallback_size: Option<f64>,
    /// The smoothing run, when it succeeded.
    pub smoothing: Option<Smoothing>,
    /// Why smoothing was not applied, when it was requested and failed.
    pub smoothing_error: Option<String>,
}

/// The measured result of the pipeline.
#[derive(Clone, Debug)]
pub struct Report {
    /// The final mesh (after smoothing when it ran).
    pub mesh: Mesh,
    /// Number of `Quad4` cells.
    pub quads: usize,
    /// Number of `Tri3` cells (0 for a passing all-quad mesh).
    pub triangles: usize,
    /// Boundary vertices whose boundary-edge degree is not 2.
    pub cracks: usize,
    /// Smallest corner scaled Jacobian over all quads.
    pub min_scaled_jacobian: f64,
    /// Smallest corner angle over all quads, in degrees.
    pub min_angle_deg: f64,
    /// Mean of `max(sqrt(area)/sigma, sigma/sqrt(area))` over the quads.
    pub size_ratio_mean: f64,
    /// Maximum of `max(sqrt(area)/sigma, sigma/sqrt(area))` over the quads.
    pub size_ratio_worst: f64,
    /// `(valence-3, valence-5)` singularities of the solved cross field. The
    /// graded path builds its tiles from the size function, not from a cross
    /// field, so it reports `(0, 0)`.
    pub singularities: (usize, usize),
    /// [`SOURCE_GRADED`], [`SOURCE_QGP`] or [`SOURCE_CHANNEL`].
    pub source: &'static str,
}

/// Run the pipeline with the default size function `sigma(x, y) = 1 + growth *
/// x`.
pub fn mesh(domain: &Domain, options: &Options) -> Result<Report, String> {
    Ok(mesh_with_trace(domain, options)?.0)
}

/// Run the pipeline with an explicit size callback.
///
/// `sigma` is evaluated at every triangulation vertex, at every quad centroid
/// and, when the graded path runs, at the tile corners and metric samples; it
/// must return a positive, finite value everywhere. The route selection is the
/// one described in the module header: the graded tile grid first when the
/// sampled variation exceeds [`GRADED_VARIATION_THRESHOLD`], then the
/// cross-field QGP + stitch, then the uniform channel fallback (whose first
/// size is the geometric mean of the smallest and largest sampled size, plus
/// four fractions around it).
pub fn mesh_with_size<F>(domain: &Domain, options: &Options, sigma: F) -> Result<Report, String>
where
    F: Fn(f64, f64) -> f64,
{
    Ok(run(domain, options, &sigma)?.0)
}

/// [`mesh`], also returning the [`Trace`] of the graded/QGP/fallback/smoothing
/// decisions.
pub fn mesh_with_trace(domain: &Domain, options: &Options) -> Result<(Report, Trace), String> {
    let growth = options.growth;
    run(domain, options, &move |x, _y| 1.0 + growth * x)
}

/// Run the pipeline on the demo case: a 10 x 10 square at `base_size = 1.0`
/// and `growth = 0.5`, smoothed, and prints the measured numbers.
///
/// The demo prints the source, the requested/effective graded tile grid, the
/// QGP failure when the fallback ran, the uniform fallback size, and the
/// smoothing before/after values. The returned [`Report`] is the same
/// measurement in structured form.
pub fn demo() -> Result<Report, String> {
    let domain = Domain::rectangle(10.0, 10.0)
        .map_err(|error| format!("simons_pipeline demo: the domain is invalid: {error}"))?;
    let options = Options {
        base_size: 1.0,
        growth: 0.5,
        smooth: true,
        ..Options::default()
    };
    let (report, trace) = mesh_with_trace(&domain, &options)?;
    println!(
        "simons_pipeline demo: source={} quads={} triangles={} cracks={} \
         SJ={:.4} min_angle={:.2} size_ratio mean={:.3} worst={:.3} \
         singularities=({}, {})",
        report.source,
        report.quads,
        report.triangles,
        report.cracks,
        report.min_scaled_jacobian,
        report.min_angle_deg,
        report.size_ratio_mean,
        report.size_ratio_worst,
        report.singularities.0,
        report.singularities.1,
    );
    if let Some(graded) = &trace.graded {
        println!(
            "  graded path: requested tiles {:?} -> effective {:?} (coarsened {}), \
             {} regions, {} edges, min count {}, {} quads",
            graded.requested_tiles,
            graded.tiles,
            graded.coarsened,
            graded.regions,
            graded.edges,
            graded.min_count,
            graded.cells,
        );
    }
    if let Some(error) = &trace.graded_error {
        println!("  graded path failed: {error}");
    }
    if trace.seeded_pairs > 0 || trace.seeded_error.is_some() {
        println!(
            "  seeded path: pairs={} detected={:?} tiles={} interior={} coverage={:.3}",
            trace.seeded_pairs,
            trace.seeded_detected,
            trace.seeded_tiles,
            trace.seeded_interior_quads,
            trace.seeded_coverage,
        );
    }
    if let Some(error) = &trace.seeded_error {
        println!("  seeded path failed: {error}");
    }
    if let Some(error) = &trace.qgp_error {
        println!("  QGP path failed: {error}");
    }
    if let Some(size) = trace.fallback_size {
        println!(
            "  channel fallback: uniform size {size:.4} \
             (chosen by the fallback size search)"
        );
    }
    if let Some(smoothing) = &trace.smoothing {
        println!(
            "  smoothing: before SJ={:.4} angle={:.2} stretch={:.3} -> \
             after SJ={:.4} angle={:.2} stretch={:.3} ({} sweeps, {} nodes moved)",
            smoothing.before.min_scaled_jacobian,
            smoothing.before.min_angle_deg,
            smoothing.before.max_stretch,
            smoothing.after.min_scaled_jacobian,
            smoothing.after.min_angle_deg,
            smoothing.after.max_stretch,
            smoothing.sweeps,
            smoothing.moved_nodes,
        );
    }
    if let Some(error) = &trace.smoothing_error {
        println!("  smoothing skipped: {error}");
    }
    Ok(report)
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// The shared implementation, parameterised by the size function.
fn run<F>(domain: &Domain, options: &Options, sigma: &F) -> Result<(Report, Trace), String>
where
    F: Fn(f64, f64) -> f64,
{
    check_options(options)?;
    if let Some(reason) = domain_problem(domain) {
        return Err(format!("simons_pipeline: {reason}"));
    }
    let (points, tris) = triangulate(domain, options.base_size)?;
    if points.is_empty() || tris.is_empty() {
        return Err("simons_pipeline: the triangulation is empty".to_string());
    }
    let size = vertex_sizes(&points, sigma)?;
    let variation = size_variation(&size)?;

    let mut trace = Trace {
        size_variation: variation,
        ..Trace::default()
    };

    // The graded paths: the metric tile grid (the shipped default) needs no
    // cross field, and the opt-in Section 5.1.4 seeded field goes through the
    // Chapter 3-5 chain. When neither is applicable or either fails, the
    // original Chapter 3-5 route (or the channel fallback) runs unchanged.
    let graded_shape = simons_graded::shape(domain).is_some();
    let graded_requested = options.graded && variation > GRADED_VARIATION_THRESHOLD && graded_shape;
    let seeded_requested = options.seeded && variation > GRADED_VARIATION_THRESHOLD && graded_shape;
    let (mesh, source, singularities) = if seeded_requested {
        match seeded_field_path(domain, &points, &tris, sigma, &size, &mut trace) {
            Ok((mesh, singularities)) => (mesh, SOURCE_SEEDED, singularities),
            Err(error) => {
                trace.seeded_error = Some(error);
                metric_or_qgp(domain, &points, &tris, sigma, &size, options, &mut trace)?
            }
        }
    } else if graded_requested {
        metric_or_qgp(domain, &points, &tris, sigma, &size, options, &mut trace)?
    } else {
        qgp_or_channel(domain, &points, &tris, sigma, &size, &mut trace)?
    };

    let (mesh, smoothing_error) = if options.smooth {
        match smooth_mesh(domain, &mesh, sigma) {
            Ok((smoothed, smoothing)) => {
                trace.smoothing = Some(smoothing);
                (smoothed, None)
            }
            Err(error) => (mesh, Some(error)),
        }
    } else {
        (mesh, None)
    };
    trace.smoothing_error = smoothing_error;

    let report = finish(mesh, sigma, singularities, source)?;
    Ok((report, trace))
}

/// The sampled size variation `sigma_max / sigma_min`.
fn size_variation(size: &[f64]) -> Result<f64, String> {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &value in size {
        if !value.is_finite() || value <= 0.0 {
            return Err(format!(
                "simons_pipeline: a sampled size is {value}; it must be positive \
                 and finite"
            ));
        }
        min = min.min(value);
        max = max.max(value);
    }
    if !min.is_finite() || !max.is_finite() || min <= 0.0 {
        return Err("simons_pipeline: the size range is not positive and finite".to_string());
    }
    Ok(max / min)
}

/// The original Chapter 3-5 route: the size-aware cross field, the QGP tiles
/// and stitch, then the channel fallback.
fn qgp_or_channel<F>(
    domain: &Domain,
    points: &[Node],
    tris: &[[usize; 3]],
    sigma: &F,
    size: &[f64],
    trace: &mut Trace,
) -> Result<(Mesh, &'static str, (usize, usize)), String>
where
    F: Fn(f64, f64) -> f64,
{
    // Chapter 3: the size-aware cross field. The thesis' Dirichlet energy
    // (holomorphic = 0) with the accurate Appendix D integrals.
    let field_options = simons_godf::Options {
        accurate: true,
        holomorphic: 0.0,
        ..simons_godf::Options::default()
    };
    let alpha =
        simons_godf::solve_with_options(points, tris, size, &field_options, GODF_ITERATIONS)
            .map_err(|error| format!("simons_pipeline: the cross-field solve failed: {error}"))?;
    let singularities = simons_godf::count_singularities(&alpha, tris);

    match qgp_stitch(points, tris, &alpha, size, trace) {
        Ok(mesh) => Ok((mesh, SOURCE_QGP, singularities)),
        Err(qgp_error) => {
            trace.qgp_error = Some(qgp_error.clone());
            let (mesh, fallback_size) = channel_fallback(domain, size, sigma).map_err(|error| {
                format!(
                    "simons_pipeline: the QGP path failed ({qgp_error}) and the \
                     channel fallback failed too ({error})"
                )
            })?;
            trace.fallback_size = Some(fallback_size);
            Ok((mesh, SOURCE_CHANNEL, singularities))
        }
    }
}

/// The metric tile grid when it is enabled and applicable, otherwise the
/// ordinary QGP/stitch + channel route.
fn metric_or_qgp<F>(
    domain: &Domain,
    points: &[Node],
    tris: &[[usize; 3]],
    sigma: &F,
    size: &[f64],
    options: &Options,
    trace: &mut Trace,
) -> Result<(Mesh, &'static str, (usize, usize)), String>
where
    F: Fn(f64, f64) -> f64,
{
    if options.graded && simons_graded::shape(domain).is_some() {
        let graded_options = simons_graded::Options {
            tiles: options.tiles,
        };
        match simons_graded::mesh(domain, &graded_options, sigma) {
            Ok((mesh, info)) => {
                trace.graded = Some(info);
                Ok((mesh, SOURCE_GRADED, (0usize, 0usize)))
            }
            Err(error) => {
                trace.graded_error = Some(error);
                qgp_or_channel(domain, points, tris, sigma, size, trace)
            }
        }
    } else {
        qgp_or_channel(domain, points, tris, sigma, size, trace)
    }
}

/// The number of Section 5.1.4 pairs for a sampled size variation.
///
/// One pair per doubling of the size gives roughly one chord redirection per
/// scale, which is what the grading needs; the result is clamped to
/// `1..=4` so a very aggressive range does not turn the field into noise.
fn seeded_pair_count(variation: f64) -> usize {
    if !variation.is_finite() || variation <= 1.0 {
        return 1;
    }
    let count = variation.log2().round();
    if !count.is_finite() {
        return 1;
    }
    (count.max(1.0) as usize).min(4)
}

/// The Chapter 5 seeded-field path: GODF with the accurate integrals and a
/// tangent boundary, the Section 5.1.4 pairs, the seeded QGP tiles, the stitch
/// and (by the caller) the target-area smoothing.
///
/// It returns an error, and the caller falls back to the metric path, when the
/// seeded field cannot produce a domain-covering stitched mesh. The trace
/// always records what the attempt measured, including on failure.
fn seeded_field_path<F>(
    domain: &Domain,
    points: &[Node],
    tris: &[[usize; 3]],
    sigma: &F,
    size: &[f64],
    trace: &mut Trace,
) -> Result<(Mesh, (usize, usize)), String>
where
    F: Fn(f64, f64) -> f64,
{
    // Chapter 3 with the thesis' Dirichlet energy, accurate Appendix D
    // integrals and the tangent boundary the field path asks for.
    let field_options = simons_godf::Options {
        accurate: true,
        holomorphic: 0.0,
        boundary: simons_godf::BoundaryMode::Tangent,
        ..simons_godf::Options::default()
    };
    let alpha =
        simons_godf::solve_with_options(points, tris, size, &field_options, GODF_ITERATIONS)
            .map_err(|error| format!("the cross-field solve failed: {error}"))?;

    // Section 5.1.4: place the neutral-holonomy pairs and seed the field.
    let requested = seeded_pair_count(trace.size_variation);
    let pairs = simons_seeded::pair_positions(domain, sigma, requested)
        .map_err(|error| format!("the pair placement failed: {error}"))?;
    trace.seeded_pairs = pairs.len();
    if pairs.is_empty() {
        return Err(format!(
            "no valence-3/5 pair fits inside the domain (requested {requested})"
        ));
    }
    let seeded = simons_seeded::seed_field(points, &alpha, &pairs)
        .map_err(|error| format!("the field seeding failed: {error}"))?;
    trace.seeded_detected = simons_godf::count_singularities(&seeded, tris);

    // The seeded motorcycle graph and simple tiles. Its interior numbers are
    // recorded even when the stitch later fails.
    let regions = simons_regions::build(points, tris, &seeded, size)
        .map_err(|error| format!("the seeded QGP regions failed: {error}"))?;
    trace.seeded_tiles = regions.tiles.len();
    let interior = regions
        .mesh()
        .map_err(|error| format!("the seeded QGP tile mesh failed: {error}"))?;
    trace.seeded_interior_quads = interior.cells.len();
    let interior_area = mesh_area(&interior);
    let domain_area = domain.area();
    trace.seeded_coverage = if domain_area > 0.0 {
        interior_area / domain_area
    } else {
        0.0
    };
    if trace.seeded_coverage < 0.98 {
        return Err(format!(
            "the seeded QGP tiles cover {:.1} % of the domain (need 98 %)",
            100.0 * trace.seeded_coverage
        ));
    }

    // Stitch the seeded tiles to the boundary with the Chapter 2 channel
    // (Section 5.1.8.2).
    let mesh = simons_regions::stitch(points, tris, &seeded, size)
        .map_err(|error| format!("the seeded stitch failed: {error}"))?;
    mesh.validate()
        .map_err(|error| format!("the stitched seeded mesh is invalid: {error}"))?;
    let coverage = if domain_area > 0.0 {
        mesh_area(&mesh) / domain_area
    } else {
        0.0
    };
    if !(0.98..=1.02).contains(&coverage) {
        return Err(format!(
            "the stitched seeded mesh covers {:.1} % of the domain (need 98-102 %)",
            100.0 * coverage
        ));
    }
    let gate = verify::gate(&mesh);
    if gate.triangles != 0 || gate.inverted != 0 || gate.overlapping != 0 || gate.cracks != 0 {
        return Err(format!(
            "the stitched seeded mesh failed the topology gate: {}",
            gate.reason()
                .unwrap_or_else(|| "unknown reason".to_string())
        ));
    }
    Ok((mesh, trace.seeded_detected))
}

/// The area of the `Quad4` cells of a planar mesh.
fn mesh_area(mesh: &Mesh) -> f64 {
    mesh.cells
        .iter()
        .filter(|cell| cell.kind == Kind::Quad4)
        .map(|cell| quality::volume(cell, &mesh.points).abs())
        .sum()
}

/// Validate the options before doing any work.
fn check_options(options: &Options) -> Result<(), String> {
    if !options.base_size.is_finite() || options.base_size <= 0.0 {
        return Err(format!(
            "simons_pipeline: base_size must be positive and finite (got {})",
            options.base_size
        ));
    }
    if !options.growth.is_finite() {
        return Err(format!(
            "simons_pipeline: growth must be finite (got {})",
            options.growth
        ));
    }
    let (tiles_x, tiles_y) = options.tiles;
    if (tiles_x == 0) != (tiles_y == 0) {
        return Err(format!(
            "simons_pipeline: tiles must be (0, 0) (derived) or both positive (got \
             ({tiles_x}, {tiles_y}))"
        ));
    }
    if tiles_x > simons_graded::MAX_TILES || tiles_y > simons_graded::MAX_TILES {
        return Err(format!(
            "simons_pipeline: the tile grid ({tiles_x}, {tiles_y}) exceeds the limit \
             of {} per direction",
            simons_graded::MAX_TILES
        ));
    }
    Ok(())
}

/// Reject a domain the planar pipeline cannot use.
fn domain_problem(domain: &Domain) -> Option<String> {
    for (index, point) in domain
        .outer
        .iter()
        .chain(domain.holes.iter().flatten())
        .enumerate()
    {
        if !point[0].is_finite() || !point[1].is_finite() {
            return Some(format!("boundary point {index} is not finite"));
        }
    }
    let area = domain.area();
    if !area.is_finite() || area <= 0.0 {
        return Some(format!(
            "the domain area is {area}; the outer loop must be counter-clockwise \
             and larger than its holes"
        ));
    }
    None
}

/// Sample the size callback at every triangulation vertex.
fn vertex_sizes<F>(points: &[Node], sigma: &F) -> Result<Vec<f64>, String>
where
    F: Fn(f64, f64) -> f64,
{
    let mut sizes = Vec::with_capacity(points.len());
    for (index, point) in points.iter().enumerate() {
        let size = sigma(point[0], point[1]);
        if !size.is_finite() || size <= 0.0 {
            return Err(format!(
                "simons_pipeline: the size function returned {size} at vertex {index} \
                 ({}, {}); it must be positive and finite",
                point[0], point[1]
            ));
        }
        sizes.push(size);
    }
    Ok(sizes)
}

// ---------------------------------------------------------------------------
// Step 1: triangulation
// ---------------------------------------------------------------------------

/// Triangulate the domain: a structured grid for an axis-aligned rectangle,
/// the crate's CDT otherwise.
fn triangulate(domain: &Domain, base_size: f64) -> Result<(Vec<Node>, Vec<[usize; 3]>), String> {
    match rectangle_bounds(domain) {
        Some(bounds) => structured_grid(domain, bounds, base_size),
        None => {
            let sizing = Sizing::uniform(base_size).map_err(|error| {
                format!("simons_pipeline: the triangulation size is invalid: {error}")
            })?;
            two_d::triangulate(domain, &sizing)
                .map_err(|error| format!("simons_pipeline: the CDT failed: {error}"))
        }
    }
}

/// The axis-aligned bounding box of a four-corner rectangle, or `None` when
/// the domain is anything else (a hole, a rotated rectangle, a polygon).
fn rectangle_bounds(domain: &Domain) -> Option<[f64; 4]> {
    if !domain.holes.is_empty() || domain.outer.len() != 4 {
        return None;
    }
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for point in &domain.outer {
        min_x = min_x.min(point[0]);
        max_x = max_x.max(point[0]);
        min_y = min_y.min(point[1]);
        max_y = max_y.max(point[1]);
    }
    if max_x <= min_x || max_y <= min_y {
        return None;
    }
    let scale = 1.0
        + min_x
            .abs()
            .max(max_x.abs())
            .max(min_y.abs())
            .max(max_y.abs());
    let tolerance = 1e-9 * scale;
    let mut corners = [false; 4];
    for point in &domain.outer {
        let column = if (point[0] - min_x).abs() <= tolerance {
            0
        } else if (point[0] - max_x).abs() <= tolerance {
            1
        } else {
            return None;
        };
        let row = if (point[1] - min_y).abs() <= tolerance {
            0
        } else if (point[1] - max_y).abs() <= tolerance {
            1
        } else {
            return None;
        };
        corners[row * 2 + column] = true;
    }
    if corners.iter().all(|&present| present) {
        Some([min_x, max_x, min_y, max_y])
    } else {
        None
    }
}

/// The number of structured grid cells along one side.
fn grid_count(length: f64, base_size: f64) -> Result<usize, String> {
    let count = (length / base_size).ceil();
    if !count.is_finite() || count < 1.0 || count > MAX_GRID as f64 {
        return Err(format!(
            "simons_pipeline: base_size {base_size} needs {count} grid cells along \
             a side of length {length}; the structured grid limit is {MAX_GRID}"
        ));
    }
    Ok(count as usize)
}

/// A structured `nx` x `ny` grid, two counter-clockwise triangles per cell.
fn structured_grid(
    domain: &Domain,
    bounds: [f64; 4],
    base_size: f64,
) -> Result<(Vec<Node>, Vec<[usize; 3]>), String> {
    let [min_x, max_x, min_y, max_y] = bounds;
    let width = max_x - min_x;
    let height = max_y - min_y;
    let nx = grid_count(width, base_size)?;
    let ny = grid_count(height, base_size)?;
    let cells = nx as u128 * ny as u128;
    if cells > MAX_CELLS {
        return Err(format!(
            "simons_pipeline: {nx} x {ny} background cells exceed the limit of \
             {MAX_CELLS}; increase base_size"
        ));
    }
    let point_count = (nx as u128 + 1) * (ny as u128 + 1);
    let mut points: Vec<Node> = Vec::with_capacity(point_count as usize);
    let dx = width / nx as f64;
    let dy = height / ny as f64;
    for j in 0..=ny {
        for i in 0..=nx {
            points.push([min_x + i as f64 * dx, min_y + j as f64 * dy, 0.0]);
        }
    }
    let index = |i: usize, j: usize| j * (nx + 1) + i;
    let mut tris: Vec<[usize; 3]> = Vec::with_capacity(2 * cells as usize);
    for j in 0..ny {
        for i in 0..nx {
            let a = index(i, j);
            let b = index(i + 1, j);
            let c = index(i + 1, j + 1);
            let d = index(i, j + 1);
            // Clip: the cell is kept when its centre is inside the domain. For
            // the axis-aligned rectangle that `rectangle_bounds` accepted
            // every cell is inside; the test keeps the path honest if the
            // detector is ever widened.
            let centre = [
                0.25 * (points[a][0] + points[b][0] + points[c][0] + points[d][0]),
                0.25 * (points[a][1] + points[b][1] + points[c][1] + points[d][1]),
            ];
            if !domain.contains(centre) {
                continue;
            }
            tris.push([a, b, c]);
            tris.push([a, c, d]);
        }
    }
    Ok((points, tris))
}

// ---------------------------------------------------------------------------
// Step 4: QGP, mesh and stitch
// ---------------------------------------------------------------------------

/// Build the QGP regions, mesh the tiles and stitch them to the boundary.
fn qgp_stitch(
    points: &[[f64; 3]],
    tris: &[[usize; 3]],
    alpha: &[f64],
    size: &[f64],
    trace: &mut Trace,
) -> Result<Mesh, String> {
    let regions = simons_regions::build(points, tris, alpha, size)
        .map_err(|error| format!("QGP regions: {error}"))?;
    trace.qgp_tiles = regions.tiles.len();
    trace.qgp_degenerate_tiles = regions.degenerate_tiles;
    trace.qgp_collapsed_tiles = regions.collapsed_tiles;
    let interior = regions
        .mesh()
        .map_err(|error| format!("QGP tile mesh: {error}"))?;
    trace.qgp_interior_quads = interior.cells.len();
    let mesh = simons_regions::stitch(points, tris, alpha, size)
        .map_err(|error| format!("QGP stitch: {error}"))?;
    mesh.validate()
        .map_err(|error| format!("the stitched mesh is invalid: {error}"))?;
    Ok(mesh)
}

/// The Chapter 2 channel fallback.
///
/// A uniform mesh cannot follow a varying size, so `sqrt(min * max)` (the
/// geometric mean, which minimises the worst ratio over the range) is the
/// first candidate. Four fractions around it are measured as well, and the
/// gate-passing mesh with the smallest worst size ratio is kept (ties: the
/// smaller mean, then the first candidate). This search is a failure-recovery
/// heuristic, not part of the thesis.
fn channel_fallback<F>(domain: &Domain, size: &[f64], sigma: &F) -> Result<(Mesh, f64), String>
where
    F: Fn(f64, f64) -> f64,
{
    let (min, max) = size
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &value| {
            (lo.min(value), hi.max(value))
        });
    if !min.is_finite() || !max.is_finite() || min <= 0.0 || max <= 0.0 {
        return Err(format!(
            "the size range [{min}, {max}] is not positive and finite"
        ));
    }
    let uniform = (min * max).sqrt();
    if !uniform.is_finite() || uniform <= 0.0 {
        return Err(format!("the fallback size {uniform} is not positive"));
    }
    let candidates = [
        uniform,
        0.75 * uniform,
        1.5 * uniform,
        0.5 * uniform,
        2.0 * uniform,
    ];
    let mut best_passing: Option<(f64, f64, Mesh, f64)> = None;
    let mut first: Option<(Mesh, f64)> = None;
    let mut last_error: Option<String> = None;
    for candidate in candidates {
        if !candidate.is_finite() || candidate <= 0.0 {
            continue;
        }
        let sizing = match Sizing::uniform(candidate) {
            Ok(sizing) => sizing,
            Err(error) => {
                last_error = Some(error.to_string());
                continue;
            }
        };
        let mesh = match crate::sized::mesh(domain, &sizing) {
            Ok(mesh) => mesh,
            Err(error) => {
                last_error = Some(format!("channel: {error}"));
                continue;
            }
        };
        if mesh.cells.is_empty() {
            last_error = Some("channel: the fallback produced an empty mesh".to_string());
            continue;
        }
        if first.is_none() {
            first = Some((mesh.clone(), candidate));
        }
        let gate = verify::gate(&mesh);
        if !gate.passes() {
            continue;
        }
        let (mean, worst) = match size_ratios(&mesh, sigma) {
            Ok(ratios) => ratios,
            Err(error) => {
                last_error = Some(error);
                continue;
            }
        };
        let better = match &best_passing {
            None => true,
            Some((best_mean, best_worst, _, _)) => {
                worst < *best_worst - 1e-12
                    || ((worst - *best_worst).abs() <= 1e-12 && mean < *best_mean - 1e-12)
            }
        };
        if better {
            best_passing = Some((mean, worst, mesh, candidate));
        }
    }
    if let Some((_, _, mesh, chosen)) = best_passing {
        return Ok((mesh, chosen));
    }
    if let Some(found) = first {
        return Ok(found);
    }
    Err(last_error.unwrap_or_else(|| "the channel route produced no mesh".to_string()))
}

// ---------------------------------------------------------------------------
// Step 5: smoothing
// ---------------------------------------------------------------------------

/// Smooth with the thesis angle + stretch objective, boundary fixed, then with
/// the product smoother's **target-area** term against the size field.
fn smooth_mesh<F>(domain: &Domain, mesh: &Mesh, sigma: &F) -> Result<(Mesh, Smoothing), String>
where
    F: Fn(f64, f64) -> f64,
{
    let size = mean_quad_size(mesh)?;
    let options = simons_smooth::Options {
        iterations: SMOOTH_ITERATIONS,
        stretch_weight: STRETCH_WEIGHT,
        allow_boundary: false,
        simplex: true,
    };
    let outcome = simons_smooth::smooth_with_stats(domain, mesh, size, &options)
        .map_err(|error| format!("smoothing: {error}"))?;
    let mut smoothed = outcome.mesh.clone();
    // The thesis smoother has no size term; the product smoother does. Pull
    // `sqrt(area)` towards sigma at each node, keeping the mesh only when the
    // gate survives (the smoother's own acceptance also checks the size).
    let targets: Vec<f64> = smoothed
        .points
        .iter()
        .map(|point| sigma(point[0], point[1]))
        .collect();
    let product = crate::smooth::smooth_with_targets(
        domain,
        &smoothed,
        &targets,
        &crate::smooth::SmoothOptions {
            size_weight: 1.0,
            ..crate::smooth::SmoothOptions::default()
        },
    );
    if product.accepted && product.after.passes {
        smoothed = product.smoothed;
    }
    let after = crate::simons_smooth::measure(&smoothed).unwrap_or(outcome.after);
    Ok((
        smoothed,
        Smoothing {
            before: outcome.before,
            after,
            sweeps: outcome.sweeps,
            moved_nodes: outcome.moved_nodes,
        },
    ))
}

/// The mean `sqrt(area)` over the quads, used as the smoother's step scale.
fn mean_quad_size(mesh: &Mesh) -> Result<f64, String> {
    let mut total = 0.0;
    let mut count = 0usize;
    for cell in &mesh.cells {
        if cell.kind != Kind::Quad4 {
            continue;
        }
        let area = quality::volume(cell, &mesh.points).abs();
        if !area.is_finite() || area <= 0.0 {
            continue;
        }
        total += area.sqrt();
        count += 1;
    }
    if count == 0 {
        return Err("smoothing: the mesh has no positive-area quad".to_string());
    }
    let mean = total / count as f64;
    if !mean.is_finite() || mean <= 0.0 {
        return Err("smoothing: the mean quad size is not positive".to_string());
    }
    Ok(mean)
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

/// Measure the final mesh and assemble the report.
fn finish<F>(
    mesh: Mesh,
    sigma: &F,
    singularities: (usize, usize),
    source: &'static str,
) -> Result<Report, String>
where
    F: Fn(f64, f64) -> f64,
{
    mesh.validate()
        .map_err(|error| format!("simons_pipeline: the final mesh is invalid: {error}"))?;
    let gate = verify::gate(&mesh);
    let (size_ratio_mean, size_ratio_worst) = size_ratios(&mesh, sigma)?;
    Ok(Report {
        mesh,
        quads: gate.quads,
        triangles: gate.triangles,
        cracks: gate.cracks,
        min_scaled_jacobian: gate.min_scaled_jacobian,
        min_angle_deg: gate.min_corner_angle_deg,
        size_ratio_mean,
        size_ratio_worst,
        singularities,
        source,
    })
}

/// The mean and worst size ratio over the quads.
fn size_ratios<F>(mesh: &Mesh, sigma: &F) -> Result<(f64, f64), String>
where
    F: Fn(f64, f64) -> f64,
{
    let mut total = 0.0;
    let mut worst = 1.0f64;
    let mut count = 0usize;
    for (index, cell) in mesh.cells.iter().enumerate() {
        if cell.kind != Kind::Quad4 {
            continue;
        }
        let area = quality::volume(cell, &mesh.points).abs();
        if !area.is_finite() || area <= 0.0 {
            return Err(format!(
                "simons_pipeline: quad {index} has non-positive area {area}"
            ));
        }
        let mut centre_x = 0.0;
        let mut centre_y = 0.0;
        for &node in &cell.nodes {
            let point = match mesh.points.get(node) {
                Some(point) => point,
                None => {
                    return Err(format!(
                        "simons_pipeline: quad {index} references node {node} out of range"
                    ));
                }
            };
            centre_x += point[0];
            centre_y += point[1];
        }
        let count_nodes = cell.nodes.len();
        if count_nodes == 0 {
            return Err(format!("simons_pipeline: quad {index} has no nodes"));
        }
        centre_x /= count_nodes as f64;
        centre_y /= count_nodes as f64;
        let target = sigma(centre_x, centre_y);
        if !target.is_finite() || target <= 0.0 {
            return Err(format!(
                "simons_pipeline: the size function returned {target} at quad {index}'s \
                 centroid ({centre_x}, {centre_y})"
            ));
        }
        let length = area.sqrt();
        let ratio = (length / target).max(target / length);
        if !ratio.is_finite() {
            return Err(format!(
                "simons_pipeline: quad {index} has a non-finite size ratio"
            ));
        }
        total += ratio;
        worst = worst.max(ratio);
        count += 1;
    }
    if count == 0 {
        return Err("simons_pipeline: the final mesh has no quad to measure".to_string());
    }
    Ok((total / count as f64, worst))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn graded_square_pipeline_is_measured() {
        let domain = Domain::rectangle(10.0, 10.0).unwrap();
        let options = Options {
            base_size: 1.0,
            growth: 0.5,
            smooth: true,
            ..Options::default()
        };
        let start = Instant::now();
        let (report, trace) = mesh_with_trace(&domain, &options).unwrap();
        let elapsed = start.elapsed();
        let gate = verify::gate(&report.mesh);
        println!(
            "graded 10x10 square (base 1.0, growth 0.5): source={} quads={} \
             triangles={} cracks={} SJ={:.4} min_angle={:.2} \
             size_ratio mean={:.3} worst={:.3} singularities=({}, {}) \
             qgp_tiles={} qgp_interior={} qgp_degenerate={} qgp_collapsed={} \
             fallback_size={:?} elapsed={:?}",
            report.source,
            report.quads,
            report.triangles,
            report.cracks,
            report.min_scaled_jacobian,
            report.min_angle_deg,
            report.size_ratio_mean,
            report.size_ratio_worst,
            report.singularities.0,
            report.singularities.1,
            trace.qgp_tiles,
            trace.qgp_interior_quads,
            trace.qgp_degenerate_tiles,
            trace.qgp_collapsed_tiles,
            trace.fallback_size,
            elapsed,
        );
        if let Some(graded) = &trace.graded {
            println!(
                "  graded: requested={:?} effective={:?} coarsened={} regions={} \
                 edges={} min_count={} cells={}",
                graded.requested_tiles,
                graded.tiles,
                graded.coarsened,
                graded.regions,
                graded.edges,
                graded.min_count,
                graded.cells,
            );
        }
        if let Some(error) = &trace.graded_error {
            println!("  graded error: {error}");
        }
        if let Some(error) = &trace.qgp_error {
            println!("  qgp error: {error}");
        }
        if let Some(smoothing) = &trace.smoothing {
            println!(
                "  smoothing: before SJ={:.4} angle={:.2} stretch={:.3} -> \
                 after SJ={:.4} angle={:.2} stretch={:.3}",
                smoothing.before.min_scaled_jacobian,
                smoothing.before.min_angle_deg,
                smoothing.before.max_stretch,
                smoothing.after.min_scaled_jacobian,
                smoothing.after.min_angle_deg,
                smoothing.after.max_stretch,
            );
        }
        if let Some(error) = &trace.smoothing_error {
            println!("  smoothing skipped: {error}");
        }
        assert!(report.quads > 0);
        assert_eq!(report.triangles, 0, "not all-quad");
        assert_eq!(report.cracks, 0, "cracks: {}", report.cracks);
        assert!(gate.passes(), "gate failed: {:?}", gate.reason());
        assert_eq!(
            report.source, SOURCE_GRADED,
            "graded error: {:?}",
            trace.graded_error
        );
        assert!(
            report.size_ratio_worst < 1.8,
            "the graded path should beat the uniform fallback (1.875); worst {}",
            report.size_ratio_worst
        );
        assert!(
            report.size_ratio_mean >= 1.0 - 1e-12
                && report.size_ratio_mean <= report.size_ratio_worst + 1e-12,
            "size ratios: mean {} worst {}",
            report.size_ratio_mean,
            report.size_ratio_worst
        );
        let trend = simons_graded::size_stats(&report.mesh, &|x, _y| 1.0 + 0.5 * x).unwrap();
        assert!(
            trend.correlation > 0.9,
            "the size trend is not visible after smoothing: correlation {}",
            trend.correlation
        );
        assert!(
            elapsed.as_secs_f64() < 10.0,
            "the pipeline took {elapsed:?}"
        );
    }

    #[test]
    fn two_dimensional_gradient_uses_the_graded_path() {
        let domain = Domain::rectangle(10.0, 10.0).unwrap();
        let options = Options {
            base_size: 1.0,
            growth: 0.0,
            smooth: true,
            tiles: (8, 8),
            ..Options::default()
        };
        let sigma = |x: f64, y: f64| 1.0 + 0.3 * x + 0.3 * y;
        let (report, trace) = run(&domain, &options, &sigma).unwrap();
        let gate = verify::gate(&report.mesh);
        let stats = simons_graded::size_stats(&report.mesh, &sigma).unwrap();
        println!(
            "2D graded 10x10 (sigma = 1 + 0.3x + 0.3y): source={} quads={} \
             triangles={} cracks={} SJ={:.4} min_angle={:.2} size mean={:.3} \
             worst={:.3} correlation={:.3}",
            report.source,
            report.quads,
            report.triangles,
            report.cracks,
            report.min_scaled_jacobian,
            report.min_angle_deg,
            stats.mean,
            stats.worst,
            stats.correlation,
        );
        let graded = trace.graded.as_ref().expect("the graded path did not run");
        println!(
            "  graded: requested={:?} effective={:?} coarsened={} regions={} \
             edges={} min_count={}",
            graded.requested_tiles,
            graded.tiles,
            graded.coarsened,
            graded.regions,
            graded.edges,
            graded.min_count,
        );
        assert_eq!(report.source, SOURCE_GRADED);
        assert_eq!(graded.requested_tiles, (8, 8));
        assert!(graded.coarsened, "the 8 x 8 grid should collapse");
        assert_eq!(report.triangles, 0, "not all-quad");
        assert_eq!(report.cracks, 0, "cracks: {}", report.cracks);
        assert!(gate.passes(), "gate failed: {:?}", gate.reason());
        assert!(report.size_ratio_worst < 1.8);
        assert!(stats.correlation > 0.9);
    }

    #[test]
    fn a_plate_with_a_rectangular_hole_takes_the_graded_path() {
        let domain = Domain::new(
            vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
            vec![vec![[4.0, 4.0], [6.0, 4.0], [6.0, 6.0], [4.0, 6.0]]],
        )
        .unwrap();
        let options = Options {
            base_size: 1.0,
            growth: 0.2,
            smooth: false,
            ..Options::default()
        };
        let (report, trace) = mesh_with_trace(&domain, &options).unwrap();
        let gate = verify::gate(&report.mesh);
        let stats = simons_graded::size_stats(&report.mesh, &|x, _y| 1.0 + 0.2 * x).unwrap();
        println!(
            "graded plate with a rectangular hole: source={} quads={} cracks={} \
             SJ={:.4} size mean={:.3} worst={:.3}",
            report.source,
            report.quads,
            report.cracks,
            report.min_scaled_jacobian,
            stats.mean,
            stats.worst,
        );
        assert_eq!(report.source, SOURCE_GRADED);
        assert_eq!(report.triangles, 0);
        assert_eq!(
            report.cracks, 0,
            "cracks around the hole: {}",
            report.cracks
        );
        assert!(gate.passes(), "gate failed: {:?}", gate.reason());
        assert!(report.size_ratio_worst < 2.5);
        let graded = trace.graded.as_ref().expect("the graded path did not run");
        assert_eq!(graded.regions, report.quads);
    }

    #[test]
    fn the_graded_flag_can_be_disabled() {
        let domain = Domain::rectangle(10.0, 10.0).unwrap();
        let options = Options {
            base_size: 1.0,
            growth: 0.5,
            smooth: false,
            graded: false,
            ..Options::default()
        };
        let (report, trace) = mesh_with_trace(&domain, &options).unwrap();
        assert!(trace.graded.is_none());
        assert_eq!(report.triangles, 0);
        assert_eq!(report.cracks, 0);
        let gate = verify::gate(&report.mesh);
        assert!(gate.passes(), "gate failed: {:?}", gate.reason());
    }

    #[test]
    fn the_seeded_pair_count_follows_the_size_variation() {
        assert_eq!(seeded_pair_count(1.0), 1);
        assert_eq!(seeded_pair_count(2.0), 1);
        assert_eq!(seeded_pair_count(6.0), 3);
        assert_eq!(seeded_pair_count(16.0), 4);
        assert_eq!(seeded_pair_count(f64::NAN), 1);
        assert_eq!(seeded_pair_count(0.5), 1);
    }

    #[test]
    fn the_seeded_field_path_is_opt_in_and_measured() {
        assert!(!Options::default().seeded, "the metric path is the default");
        let domain = Domain::rectangle(10.0, 10.0).unwrap();
        let cases: [(&str, Box<dyn Fn(f64, f64) -> f64>); 2] = [
            ("linear-x", Box::new(|x: f64, _y: f64| 1.0 + 0.5 * x)),
            (
                "2d-gradient",
                Box::new(|x: f64, y: f64| 1.0 + 0.3 * x + 0.3 * y),
            ),
        ];
        for (name, sigma) in cases {
            let options = Options {
                base_size: 1.0,
                smooth: true,
                seeded: true,
                ..Options::default()
            };
            let (report, trace) = run(&domain, &options, &sigma).expect("the pipeline ran");
            println!(
                "seeded {name}: source={} quads={} triangles={} cracks={} \
                 mean={:.3} worst={:.3} seeded_pairs={} detected={:?} tiles={} \
                 interior={} coverage={:.3} error={:?}",
                report.source,
                report.quads,
                report.triangles,
                report.cracks,
                report.size_ratio_mean,
                report.size_ratio_worst,
                trace.seeded_pairs,
                trace.seeded_detected,
                trace.seeded_tiles,
                trace.seeded_interior_quads,
                trace.seeded_coverage,
                trace.seeded_error,
            );
            // The seeded attempt must be recorded, and when it did not win the
            // reason must be reported. It must never silently ship a mesh that
            // does not cover the domain: the metric path is the fallback and
            // it covers it with the measured 1.494 / 1.392.
            assert!(trace.seeded_pairs > 0, "no pair was placed");
            if report.source == SOURCE_SEEDED {
                assert!(trace.seeded_error.is_none());
                assert_eq!(report.triangles, 0);
                assert_eq!(report.cracks, 0);
            } else {
                assert!(
                    trace.seeded_error.is_some(),
                    "the seeded path was not selected but reported no error"
                );
                assert_eq!(report.source, SOURCE_GRADED);
                let gate = verify::gate(&report.mesh);
                assert!(gate.passes(), "the fallback failed: {:?}", gate.reason());
            }
        }
    }

    #[test]
    fn constant_size_square_is_near_uniform_and_uses_qgp() {
        let domain = Domain::rectangle(10.0, 10.0).unwrap();
        let options = Options {
            base_size: 1.0,
            growth: 0.0,
            smooth: true,
            ..Options::default()
        };
        let (report, trace) = mesh_with_trace(&domain, &options).unwrap();
        let gate = verify::gate(&report.mesh);
        println!(
            "constant 10x10 square: source={} quads={} triangles={} cracks={} \
             SJ={:.4} min_angle={:.2} size_ratio mean={:.3} worst={:.3} \
             singularities=({}, {}) qgp_tiles={} qgp_interior={}",
            report.source,
            report.quads,
            report.triangles,
            report.cracks,
            report.min_scaled_jacobian,
            report.min_angle_deg,
            report.size_ratio_mean,
            report.size_ratio_worst,
            report.singularities.0,
            report.singularities.1,
            trace.qgp_tiles,
            trace.qgp_interior_quads,
        );
        assert_eq!(
            report.source, SOURCE_QGP,
            "qgp error: {:?}",
            trace.qgp_error
        );
        assert_eq!(report.triangles, 0);
        assert_eq!(report.cracks, 0);
        assert_eq!(report.singularities, (0, 0));
        assert!(gate.passes(), "gate failed: {:?}", gate.reason());
        assert!(
            report.size_ratio_worst < 1.5,
            "a constant size should be near-uniform; worst ratio {}",
            report.size_ratio_worst
        );
        assert!(trace.smoothing.is_some(), "smoothing did not run");
    }

    #[test]
    fn a_callback_size_matches_the_growth_formula() {
        let domain = Domain::rectangle(6.0, 6.0).unwrap();
        let options = Options {
            base_size: 1.0,
            growth: 0.25,
            smooth: false,
            ..Options::default()
        };
        let growth = options.growth;
        let report = mesh_with_size(&domain, &options, |x, _y| 1.0 + growth * x).unwrap();
        let same = mesh(&domain, &options).unwrap();
        assert_eq!(report.quads, same.quads);
        assert_eq!(report.triangles, 0);
        assert_eq!(report.cracks, 0);
        assert!((report.size_ratio_worst - same.size_ratio_worst).abs() < 1e-12);
        assert!((report.size_ratio_mean - same.size_ratio_mean).abs() < 1e-12);
    }

    #[test]
    fn an_l_bracket_takes_the_general_domain_path() {
        let domain = Domain::new(
            vec![
                [0.0, 0.0],
                [6.0, 0.0],
                [6.0, 2.0],
                [2.0, 2.0],
                [2.0, 6.0],
                [0.0, 6.0],
            ],
            Vec::new(),
        )
        .unwrap();
        let options = Options {
            base_size: 1.0,
            growth: 0.0,
            smooth: false,
            ..Options::default()
        };
        let (report, trace) = mesh_with_trace(&domain, &options).unwrap();
        println!(
            "L-bracket: source={} quads={} triangles={} cracks={} SJ={:.4} \
             min_angle={:.2} size_ratio mean={:.3} worst={:.3} qgp_error={:?}",
            report.source,
            report.quads,
            report.triangles,
            report.cracks,
            report.min_scaled_jacobian,
            report.min_angle_deg,
            report.size_ratio_mean,
            report.size_ratio_worst,
            trace.qgp_error,
        );
        assert!(report.quads > 0);
        assert_eq!(report.triangles, 0);
        assert_eq!(report.cracks, 0);
        let gate = verify::gate(&report.mesh);
        assert!(gate.passes(), "gate failed: {:?}", gate.reason());
        assert!(
            report.size_ratio_worst < 2.0,
            "the fallback size search should keep the L-bracket sized; worst {}",
            report.size_ratio_worst
        );
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        let domain = Domain::rectangle(4.0, 4.0).unwrap();
        assert!(mesh(
            &domain,
            &Options {
                base_size: 0.0,
                growth: 0.5,
                smooth: false,
                ..Options::default()
            }
        )
        .is_err());
        assert!(mesh(
            &domain,
            &Options {
                base_size: 1.0,
                growth: f64::NAN,
                smooth: false,
                ..Options::default()
            }
        )
        .is_err());
        assert!(mesh_with_size(
            &domain,
            &Options {
                base_size: 1.0,
                growth: 0.0,
                smooth: false,
                ..Options::default()
            },
            |_, _| 0.0,
        )
        .is_err());
        assert!(mesh_with_size(
            &domain,
            &Options {
                base_size: 1.0,
                growth: 0.0,
                smooth: false,
                ..Options::default()
            },
            |_, _| f64::INFINITY,
        )
        .is_err());
        let clockwise = Domain::new(
            vec![[0.0, 0.0], [0.0, 4.0], [4.0, 4.0], [4.0, 0.0]],
            Vec::new(),
        )
        .unwrap();
        assert!(mesh(
            &clockwise,
            &Options {
                base_size: 1.0,
                growth: 0.0,
                smooth: false,
                ..Options::default()
            }
        )
        .is_err());
    }

    #[test]
    fn demo_numbers_are_measured() {
        let report = demo().unwrap();
        assert!(report.quads > 0);
        assert_eq!(report.triangles, 0);
        assert_eq!(report.cracks, 0);
        assert!(report.size_ratio_worst < 1.8);
    }
}
