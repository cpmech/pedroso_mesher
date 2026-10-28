//! Quad meshing of a planar domain, **gate-checked**: the product all-quad entry
//! point.
//!
//! [`mesh_quads`] runs the W3 pipeline in this order:
//!
//! 1. **triangulation** — [`triangulate`] (constrained Delaunay + Ruppert)
//!    builds the substrate at [`Sizing::size`];
//! 2. **parity** — [`even_boundary_loops`] splits one edge of every odd
//!    boundary loop (`sun-2016` §3.3), because an all-quad mesh needs an even
//!    number of edges on each loop;
//! 3. **conversion** — the Q-Morph advancing front (`owen-1999`) converts
//!    triangles into quads; [`Route::TwoStage`] replaces it with the
//!    experimental even-region front plus Blossom matching;
//! 4. **closure** — leftover triangles are closed by alternating smoothing with
//!    Sun's residual-triangle walk (`sun::close_residuals`, `sun-2015` §5), the
//!    cavity pass, and watertight residual refinement. If cleanup made the mesh
//!    inconsistent, refinement retries from the preserved raw conversion;
//! 5. **quality passes** — low-quad repair, degenerate/collinear repair,
//!    angle-driven edge swaps, a three-radius smoothing polish, and a final
//!    degenerate repair plus boundary-aware perturbation;
//! 6. **gate** — [`verify::gate`] (all-quad, SJ >= 0.2, corner >= 30 degrees,
//!    no inversion, overlap or crack) plus the explicit boundary and corner
//!    offsets.
//!
//! A mesh that meets the gate is returned as [`QuadMesh`]; one that does not is
//! [`QuadError::Unsupported`] with the measured numbers, the first reason and
//! the failing mesh (kept for diagnosis, not for shipping). This is deliberate:
//! the gate is the contract, not a target to report against. On the planar
//! corpus that makes the ring section at 1 mm and the 0.6 mm performance part
//! *unsupported* for now — the ring leaves two zero-area triangles and the
//! 0.6 mm part leaves five triangles and a scaled Jacobian of 0.156.
//!
//! `two_d::mesh_2d` remains the **default** product route (quad-dominant,
//! matching-based, no gate). This module is the all-quad route.
//!
//! Clean room: Q-Morph follows Owen et al. 1999 (`owen-1999`) and the residual
//! closure follows Sun 2015/2016 (`sun-2015`, `sun-2016`); no implementation
//! was read or ported.

use crate::mesh::{Cell, Kind, Mesh};
use crate::parity::even_boundary_loops;
use crate::qmorph::{convert_with_stats, Refusals};
use crate::quality;
use crate::sun;
use crate::two_d::{close_by_refinement, smooth_mesh, triangulate, Domain, Sizing, TwoDError};
use crate::twostage;
use crate::verify::{self, Gate};

pub use crate::config::{
    generate_quad as mesh_quads_with_config, MeshConfig, MeshOutput, Route as Generator,
};

/// Which conversion stage feeds the pipeline.
///
/// Both variants share everything after conversion — the Sun residual closure,
/// the cavity pass, the quality repairs and the gate. Only stage 3 differs.
/// The measured tables are printed by `examples/qmorph.rs`; passing the
/// `two-stage` argument selects [`Route::TwoStage`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// **Front**: Q-Morph front to exhaustion, then Sun's residual closure.
    ///
    /// Character: the shipped conversion, and the route the 30 degree / SJ 0.2
    /// gate is measured on. Measured: 4 of the 7 corpus parts are supported; the
    /// ring, the plate-hole 1 mm and the l-bracket leave a few isolated
    /// triangles and come back as [`QuadError::Unsupported`].
    Front,

    /// **TwoStage**: Q-Morph front under the even-region guard, then Blossom
    /// matching of the remainder.
    ///
    /// Character: experimental. The guard forbids leaving an odd unconverted
    /// component so the matcher can pair it. Measured negative (1/6): only one
    /// of the six planar parts improved, coverage moved by -3 % to +6 % (not
    /// monotone), and the protected remainder is thin sliver components whose
    /// two-triangle unions are non-convex, so the matcher pairs only 20–35 % of
    /// them. Kept as an experiment; [`Stats::two_stage`] carries the counters.
    TwoStage,
}

/// Counters for the stages that ran, plus the quality measurements of the final
/// mesh.
///
/// Counters are totals for the run; the quality fields are measured on the same
/// mesh the gate saw. Distances are in the domain's units (millimetres for the
/// corpus) and angles are degrees. The front-only counters are zero when the
/// route is not Q-Morph; [`crate::config::generate_quad`] reports them as zero
/// for the alternative generators.
#[derive(Debug, Default, Clone)]
pub struct Stats {
    /// Rounds of the alternating smooth/walk closure (`close_rounds`, 0–10).
    pub rounds: usize,

    /// Cavity closures accepted in `close_cavities` (at most three rounds).
    pub cavity_closures: usize,

    /// Quads repaired by `sun::repair_low_quads` (scaled Jacobian below 0.05).
    pub low_repairs: usize,

    /// Quads repaired by `sun::repair_degenerate_quads` (scaled Jacobian below
    /// 0.25 or a collinear corner).
    pub degenerate_repairs: usize,

    /// Angle-driven edge swaps accepted by `sun::improve_quality`.
    pub quality_swaps: usize,

    /// Perturbation moves accepted by `sun::perturb_low_quads` (boundary-aware).
    pub perturbations: usize,

    /// Boundary edges split by `parity::even_boundary_loops` to make every
    /// boundary loop even.
    pub parity_splits: usize,

    /// Worst scaled Jacobian of the quads alone (the triangles a failing mesh
    /// keeps would otherwise mask it). Dimensionless, in `(0, 1]` for a
    /// non-degenerate quad.
    pub quads_only_sj: f64,

    /// Worst corner angle of the quads alone, in degrees.
    pub quads_only_angle_deg: f64,

    /// Largest distance of a boundary node from the domain boundary; 0 when the
    /// mesh sits on its geometry. The gate cannot see a boundary moved inward,
    /// so this is checked separately at the end of [`mesh_quads_with_route`].
    pub boundary_offset: f64,

    /// Largest distance from a domain corner to the nearest boundary node; 0
    /// when every corner is kept. A corner node sliding along an edge cuts the
    /// corner while `boundary_offset` stays 0, which is why this second guard
    /// exists. Units are the domain's units.
    pub corner_offset: f64,

    /// Present when [`Route::TwoStage`] ran; `None` for [`Route::Front`].
    pub two_stage: Option<twostage::TwoStageStats>,
}

/// Wall-clock seconds of the pipeline stages.
///
/// The values are informational; they do not affect the gate or the mesh. They
/// are measured with `std::time::Instant` in [`mesh_quads_with_route`], and the
/// alternative-route adapter [`crate::config::generate_quad`] only knows the
/// total.
#[derive(Debug, Default, Clone, Copy)]
pub struct Timing {
    /// Triangulation plus the parity pre-pass (stage 1 and stage 2).
    pub triangulation: f64,

    /// Triangle-to-quad conversion (stage 3), front or two-stage.
    pub conversion: f64,

    /// The whole of [`mesh_quads_with_route`], so it also covers the closure,
    /// the cavity, the quality passes and the gate checks.
    pub total: f64,
}

/// A quad mesh that met the gate.
///
/// Invariants: [`Self::gate`] passes and is the gate of [`Self::mesh`];
/// [`Self::raw_gate`] is the same measurement before the closure and quality
/// passes and may fail. The mesh can be drawn or written, but the struct exists
/// only after the acceptance checks in [`mesh_quads_with_route`] have passed.
pub struct QuadMesh {
    /// The accepted mesh. On the Q-Morph route every cell is `Quad4`; the
    /// adapter [`crate::config::generate_quad`] also only accepts gate-passing
    /// meshes, so no triangle is silently hidden here (a triangle would fail
    /// the gate or remain visible to the caller).
    pub mesh: Mesh,

    /// The gate of the returned mesh; it passes by construction.
    pub gate: Gate,

    /// The gate right after conversion, before the closure and quality passes.
    /// Keeping it shows how much the post-passes bought even when it failed.
    pub raw_gate: Gate,

    /// Stage counters and the quality measurements of `mesh`.
    pub stats: Stats,

    /// Wall-clock breakdown; `timing.total` covers the whole run.
    pub timing: Timing,

    /// Why the Q-Morph front refused edges during conversion. It is empty for
    /// the alternative generators adapted by [`crate::config::generate_quad`].
    pub refusals: Refusals,

    /// Which conversion route ran ([`Route::Front`] or [`Route::TwoStage`]).
    pub route: Route,

    /// The generator that produced the mesh. The conversion route stays in
    /// [`Self::route`]; this field records the [`crate::config::Route`].
    /// [`mesh_quads`] always reports [`crate::config::Route::Qmorph`], while
    /// [`crate::config::generate_quad`] copies the configured generator.
    pub generator: crate::config::Route,
}

/// Why [`mesh_quads`] did not return a mesh: the measured result of a
/// configuration that the route does not support yet.
///
/// This is the payload of [`QuadError::Unsupported`], boxed to keep the error
/// variant small. Every field is a measurement of the *failing* mesh, so the
/// caller can reproduce the table row or draw the geometry; none of it is
/// evidence that the mesh is shippable.
#[derive(Debug)]
pub struct Unsupported {
    /// The gate measured on the final mesh; it fails.
    pub gate: Gate,

    /// The gate right after conversion, before the closure and quality passes.
    pub raw_gate: Gate,

    /// Stage counters and the quality measurements of the failing mesh.
    pub stats: Stats,

    /// Wall-clock breakdown of the failed run.
    pub timing: Timing,

    /// The conversion route that ran.
    pub route: Route,

    /// The gate's first reason, in the gate's priority order, e.g.
    /// `"2 triangle(s) left"` or `"min corner angle 28.4 < 30"`.
    pub reason: String,

    /// Front refusal counters; useful to tell a topological stop (empty front,
    /// region) from a placement or quality stop.
    pub refusals: Refusals,

    /// The mesh that failed the gate. Kept for diagnosis (drawing it is how the
    /// failures are understood); it must not be shipped as a result.
    pub mesh: Mesh,
}

/// Why [`mesh_quads`] did not return a mesh.
///
/// The variants separate an unusable input, a generated mesh that fails the
/// acceptance gate, and an alternative-route failure. A gate failure is not an
/// input error: it is a measured "not supported yet".
#[derive(Debug)]
pub enum QuadError {
    /// The domain or the size field is unusable (short/non-finite loop, size
    /// <= 0, or zero area).
    Input(TwoDError),

    /// The pipeline finished but the mesh cannot meet the gate. This
    /// configuration is outside the supported set for now; the boxed
    /// [`Unsupported`] carries the measurements and the failing mesh.
    Unsupported(Box<Unsupported>),

    /// An alternative route (see [`crate::config::MeshConfig`]) could not
    /// produce a mesh at all; the string is its error. A mesh that was produced
    /// but failed the gate is [`QuadError::Unsupported`] instead, see
    /// [`crate::config::generate_quad`].
    Route(String),
}

impl std::fmt::Display for QuadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuadError::Input(e) => write!(f, "{e}"),
            // The "not supported yet" prefix is the public signal that the
            // geometry is valid but the route cannot meet the gate on it.
            QuadError::Unsupported(u) => write!(f, "not supported yet: {}", u.reason),
            QuadError::Route(e) => write!(f, "route failed: {e}"),
        }
    }
}

impl std::error::Error for QuadError {}

// A triangulation error is an input error at this level; `?` on
// `two_d::triangulate` therefore lands in [`QuadError::Input`].
impl From<TwoDError> for QuadError {
    fn from(e: TwoDError) -> Self {
        QuadError::Input(e)
    }
}

/// Mesh a planar domain with quads, or say why it cannot be done yet.
///
/// Runs [`Route::Front`] (the shipped Q-Morph conversion) and applies the whole
/// gate contract described in the module documentation. `sizing.size` is the
/// target edge length in the domain's units. Returns [`QuadError::Input`] for an
/// unusable domain or size, and [`QuadError::Unsupported`] when the generated
/// mesh fails the gate; it never returns a mesh it knows fails.
pub fn mesh_quads(domain: &Domain, sizing: &Sizing) -> Result<QuadMesh, QuadError> {
    mesh_quads_with_route(domain, sizing, Route::Front)
}

/// Mesh a planar domain with quads under an explicit conversion [`Route`].
///
/// The pipeline order is fixed (triangulation, parity, conversion, closure,
/// cavity, quality passes, gate); `route` only selects the conversion stage.
/// The final gate is the one in [`QuadMesh::gate`], and the explicit
/// boundary/corner offsets are checked after it because [`verify::gate`] cannot
/// see geometry that moved off the domain.
pub fn mesh_quads_with_route(
    domain: &Domain,
    sizing: &Sizing,
    route: Route,
) -> Result<QuadMesh, QuadError> {
    let size = sizing.size;
    let t0 = std::time::Instant::now();
    // Stage 1: the constrained-Delaunay + Ruppert substrate. Its triangles are
    // the input of the parity pre-pass and of the conversion.
    let (points, tris) = triangulate(domain, sizing)?;
    // Stage 2: Sun 2016 §3.3. An all-quad mesh needs an even number of boundary
    // edges per loop; split one edge of each odd loop before conversion so the
    // conversion never has to repair the parity itself.
    let (points, tris, parity_splits) = even_boundary_loops(&points, &tris);
    let triangulation = t0.elapsed().as_secs_f64();

    let t1 = std::time::Instant::now();
    let mut two_stage = None;
    // Stage 3: conversion. `TwoStage` is opt-in because its measured table is
    // negative (see the variant); both arms return quads plus the triangles
    // they could not convert.
    let ((quads, leftover, points), refusals) = match route {
        Route::Front => convert_with_stats(&tris, &points, size),
        Route::TwoStage => {
            let r = twostage::convert(&tris, &points, size);
            two_stage = Some(r.stats);
            ((r.quads, r.triangles, r.points), r.refusals)
        }
    };
    let conversion = t1.elapsed().as_secs_f64();

    // Quads and leftover triangles share one mesh so the closure can reason
    // about the whole connectivity; the gate is measured on this mixed state.
    let mut cells: Vec<Cell> = quads.iter().map(|q| Cell::quad4(*q)).collect();
    cells.extend(leftover.iter().map(|t| Cell::tri3(*t)));
    let mut mesh = Mesh::new(points, cells);
    // The raw gate is the conversion result, before any closure or repair. It
    // is recorded in the output and in a failure so the post-passes' effect is
    // visible (a raw failure may become a passing final mesh).
    let raw_gate = verify::gate(&mesh);
    let converted = mesh.clone();

    // Stages 4–6, in order: alternating smooth/walk rounds, the cavity pass if
    // triangles survive, residual refinement, then quality repairs and polishing.
    // A refinement candidate must be watertight; when the cleaned mesh cannot
    // provide one, retry from the raw conversion before cleanup. The gate is
    // measured only after all stages.
    let rounds = close_rounds(&mut mesh, domain, size);
    let cavity_closures = close_cavities(&mut mesh, domain, size);
    if !tri_cells(&mesh).is_empty() {
        let refine = |source: &Mesh| {
            let (quads, points, leftover) = close_by_refinement(
                &quad_cells(source),
                &tri_cells(source),
                &source.points,
                domain,
            );
            let candidate = Mesh::new(points, cells_of(&quads, &leftover));
            candidate.topology().is_watertight().then_some(candidate)
        };
        if let Some(candidate) = refine(&mesh).or_else(|| refine(&converted)) {
            mesh = candidate;
        }
    }
    let (low_repairs, degenerate_repairs, quality_swaps, perturbations) =
        quality_passes(&mut mesh, domain, size);
    let gate = verify::gate(&mesh);

    // The quad-only report isolates the quads from any triangle the closure
    // could not remove: a failing mesh's triangles would otherwise mask the
    // quads' real minimum angle/Jacobian in the statistics.
    let quad_mesh = Mesh::new(
        mesh.points.clone(),
        mesh.cells
            .iter()
            .filter(|c| c.kind == Kind::Quad4)
            .cloned()
            .collect::<Vec<_>>(),
    );
    let report = quality::report(&quad_mesh);
    let stats = Stats {
        rounds,
        cavity_closures,
        low_repairs,
        degenerate_repairs,
        quality_swaps,
        perturbations,
        parity_splits,
        quads_only_sj: report.min_scaled_jacobian,
        quads_only_angle_deg: report.min_corner_angle_deg,
        // These are the geometry measurements, not gate criteria; the checks
        // below turn them into errors. They are stored either way so a report
        // can see a mesh that passed the gate but drifted off the geometry.
        boundary_offset: verify::boundary_offset(domain, &mesh),
        corner_offset: verify::corner_offset(domain, &mesh),
        two_stage,
    };
    let timing = Timing {
        triangulation,
        conversion,
        total: t0.elapsed().as_secs_f64(),
    };
    // Geometry conformity: a boundary node moved inward can improve every gate
    // number while the mesh no longer sits on the domain. Check it explicitly.
    // The tolerance is 1e-6 of the largest absolute coordinate (at least 1
    // unit), i.e. round-off scale rather than a geometric allowance.
    let (mut x0, mut y0) = (f64::INFINITY, f64::INFINITY);
    let (mut x1, mut y1) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for p in &mesh.points {
        x0 = x0.min(p[0]);
        y0 = y0.min(p[1]);
        x1 = x1.max(p[0]);
        y1 = y1.max(p[1]);
    }
    let diagonal = ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt().max(1.0e-12);
    let tolerance = 1.0e-6 * diagonal;
    let offset = verify::boundary_offset(domain, &mesh);
    if offset > tolerance {
        return Err(QuadError::Unsupported(Box::new(Unsupported {
            gate,
            raw_gate,
            stats,
            timing,
            route,
            reason: format!("a boundary node is {offset:.2e} off the domain boundary"),
            refusals,
            mesh,
        })));
    }
    // A corner node can slide along an adjacent edge: `boundary_offset` stays 0
    // while the corner itself is cut off. Check the corner separately.
    let corner = verify::corner_offset(domain, &mesh);
    if corner > tolerance {
        return Err(QuadError::Unsupported(Box::new(Unsupported {
            gate,
            raw_gate,
            stats,
            timing,
            route,
            reason: format!("a domain corner is {corner:.2e} from the nearest boundary node"),
            refusals,
            mesh,
        })));
    }
    if !gate.passes() {
        return Err(QuadError::Unsupported(Box::new(Unsupported {
            gate,
            raw_gate,
            stats,
            timing,
            route,
            reason: gate.reason().unwrap_or_default(),
            refusals,
            mesh,
        })));
    }
    // The fast gate cannot see intersections between cells with disjoint node
    // indices or prove domain coverage. Run the full geometric contract once at
    // the final API boundary, after all generation and quality passes.
    let geometry_reason = match verify::geometry(domain, &mesh) {
        Ok(report) => report.located_reason(),
        Err(e) => Some(format!("invalid mesh: {e}")),
    };
    if let Some(reason) = geometry_reason {
        return Err(QuadError::Unsupported(Box::new(Unsupported {
            gate,
            raw_gate,
            stats,
            timing,
            route,
            reason,
            refusals,
            mesh,
        })));
    }
    Ok(QuadMesh {
        mesh,
        gate,
        raw_gate,
        stats,
        timing,
        refusals,
        route,
        // `mesh_quads` is the Q-Morph conversion. The alternative generators
        // are adapted by `config::generate_quad`, which overwrites this field
        // with the configured route.
        generator: crate::config::Route::Qmorph,
    })
}

/// Alternate W2 smoothing with the Sun residual walk, keeping the best state
/// under the gate's own order. Returns how many rounds ran.
///
/// This is the first half of stage 4 (`sun-2015` §5 gives the walk; the W2
/// smoother is the crate's boundary-preserving one). A round smooths, walks the
/// residual triangles into quads, and keeps a smoothed copy scored by
/// [`score`]. The loop stops after 10 rounds, when no triangles remain, or after
/// three consecutive rounds that fail to reduce the triangle count.
fn close_rounds(mesh: &mut Mesh, domain: &Domain, size: f64) -> usize {
    let mut remaining = tri_cells(mesh).len();
    let mut rounds = 0usize;
    let mut stale = 0usize;
    let mut last_remaining = remaining;
    let mut best: Option<(Mesh, Gate)> = None;
    while remaining > 0 && rounds < 10 && stale < 3 {
        // The first two rounds smooth at the requested size so the walk sees
        // the global shape; after that a coarser (4x) relaxation escapes a
        // local stall without erasing the boundary rows.
        let factor = match rounds {
            0..=1 => 1.0,
            _ => 4.0,
        };
        smooth_mesh(mesh, domain, 80, size * factor);
        let (p2, q2, t2, _stats) =
            sun::close_residuals(&mesh.points, &quad_cells(mesh), &tri_cells(mesh));
        *mesh = Mesh::new(p2, cells_of(&q2, &t2));
        remaining = t2.len();
        rounds += 1;
        // "Stale" means the round did not reduce the triangle count; three in a
        // row is the measured point where further rounds stop paying off.
        if remaining >= last_remaining {
            stale += 1;
        } else {
            stale = 0;
        }
        last_remaining = remaining;

        // Score a smoothed candidate, not the raw walk output: the gate order
        // (triangles, inverted, cracks, angle, SJ) is the distribution the
        // final gate will be measured on.
        let mut candidate = mesh.clone();
        smooth_mesh(&mut candidate, domain, 80, size);
        let candidate_gate = verify::gate(&candidate);
        let better = match &best {
            None => true,
            Some((_, current)) => score(&candidate_gate) < score(current),
        };
        if better {
            best = Some((candidate, candidate_gate));
        }
    }
    // If the conversion already had no triangles the loop never ran; the final
    // polish still applies so the returned mesh is smoothed like the others.
    let chosen = match best {
        Some((candidate, _)) => candidate,
        None => {
            smooth_mesh(mesh, domain, 80, size);
            mesh.clone()
        }
    };
    *mesh = chosen;
    rounds
}

/// A final polish: repair near-degenerate and collinear-corner quads, swap for
/// angle, smooth at the size and two finer radii, then repair once more.
///
/// This is stage 5 of the pipeline. The repairs and the angle swaps are Sun's
/// (`sun-2015` for the residual geometry); the three-radius polish and the
/// boundary-aware perturbation are the crate's quality stage. Returns
/// `(low_repairs, degenerate_repairs, quality_swaps, perturbations)` to be
/// accumulated into [`Stats`].
fn quality_passes(mesh: &mut Mesh, domain: &Domain, size: f64) -> (usize, usize, usize, usize) {
    // 0.05 is the "near-zero" Jacobian floor: these are the inverted/near-flat
    // quads the M2/M3 swap must rescue first. 16 rounds bounds the churn.
    let (points, quads, tris, low_repairs) =
        sun::repair_low_quads(&mesh.points, &quad_cells(mesh), &tri_cells(mesh), 0.05, 16);
    *mesh = Mesh::new(points, cells_of(&quads, &tris));

    // 0.25 catches a folded or collinear-corner quad; 8 rounds is the measured
    // budget where the repair either succeeds or the mesh stays as it is.
    let (points, quads, tris, degenerate_repairs) =
        sun::repair_degenerate_quads(&mesh.points, &quad_cells(mesh), &tri_cells(mesh), 0.25, 8);
    *mesh = Mesh::new(points, cells_of(&quads, &tris));

    // 12 angle-driven swap rounds; each swap must keep both quads convex and
    // above the gate floor, so this wins angle without risking the gate.
    let (points, quads, tris, quality_swaps) =
        sun::improve_quality(&mesh.points, &quad_cells(mesh), &tri_cells(mesh), 12);
    *mesh = Mesh::new(points, cells_of(&quads, &tris));

    // Try smoothing at the requested size and at half and quarter size, and
    // keep the candidate with the best gate order. The finer radii resolve
    // local low-angle corners that the size-level sweep cannot move.
    let mut polished: Option<(Mesh, Gate)> = None;
    for factor in [1.0, 0.5, 0.25] {
        let mut candidate = mesh.clone();
        smooth_mesh(&mut candidate, domain, 80, size * factor);
        let candidate_gate = verify::gate(&candidate);
        let better = match &polished {
            None => true,
            Some((_, current)) => score(&candidate_gate) < score(current),
        };
        if better {
            polished = Some((candidate, candidate_gate));
        }
    }
    if let Some((candidate, _)) = polished {
        *mesh = candidate;
    }

    // A final repair for quads whose corner Jacobian is ~0 (three collinear
    // nodes): smoothing and the closure can both create one, and no later pass
    // looks at it, because the corner-angle criterion cannot see a 180° corner.
    // The perturbation then moves only boundary-aware low quads; 200 is the
    // common post-pass's budget, so the route matches the measured table.
    let (points, quads, tris, degenerate_after) =
        sun::repair_degenerate_quads(&mesh.points, &quad_cells(mesh), &tri_cells(mesh), 0.25, 8);
    let (points, quads, tris, perturbations) =
        sun::perturb_low_quads(&points, &quads, &tris, size, 200, Some(domain));
    *mesh = Mesh::new(points, cells_of(&quads, &tris));
    (
        low_repairs,
        // Count both degenerate-repair passes; the second one is part of the
        // same stage from the caller's point of view.
        degenerate_repairs + degenerate_after,
        quality_swaps,
        perturbations,
    )
}

/// If triangles remain after the closure rounds, alternate the cavity with a few
/// walk/smooth rounds: the cavity changes the local topology and the walk can
/// then pair what it unlocked.
///
/// This is the second half of stage 4. The cavity pass can open the local
/// configuration that blocks the Sun walk, so each cavity is followed by up to
/// four smooth+walk attempts before deciding whether the triangle count really
/// improved.
fn close_cavities(mesh: &mut Mesh, domain: &Domain, size: f64) -> usize {
    let mut closed_total = 0usize;
    // Three cavity rounds are enough on the corpus; more would only churn a
    // mesh the walk cannot close.
    for _ in 0..3 {
        if tri_cells(mesh).is_empty() {
            break;
        }
        let before = tri_cells(mesh).len();
        let (points, quads, tris, closed) =
            sun::close_by_cavity(&mesh.points, &quad_cells(mesh), &tri_cells(mesh), size);
        closed_total += closed;
        *mesh = Mesh::new(points, cells_of(&quads, &tris));
        // Let the walk exploit the topology the cavity just changed; the
        // smoothing is what moves the nodes into a pairable configuration.
        for _ in 0..4 {
            if tri_cells(mesh).is_empty() {
                break;
            }
            smooth_mesh(mesh, domain, 80, size);
            let (points, quads, tris, _) =
                sun::close_residuals(&mesh.points, &quad_cells(mesh), &tri_cells(mesh));
            *mesh = Mesh::new(points, cells_of(&quads, &tris));
        }
        // No net progress: another cavity would repeat the same geometry.
        if tri_cells(mesh).len() >= before {
            break;
        }
    }
    closed_total
}

/// Total order on a float for the best-state comparison.
///
/// The values compared by [`score`] are gate measurements (counts, angles and
/// scaled Jacobians), which are finite for every mesh, so the NaN fallback in
/// `cmp` is defensive only; it treats NaN as equal rather than panicking.
#[derive(Clone, Copy, PartialEq)]
struct OrderedF64(f64);

// `f64` is not `Ord`, so this newtype supplies the total order used by the
// best-state comparisons in `close_rounds` and `quality_passes`.
impl Eq for OrderedF64 {}

impl PartialOrd for OrderedF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// The gate's own order: fewest triangles, then no inverted element and no
/// crack, then the highest angle, then the highest Jacobian.
///
/// The tuple is compared lexicographically (smaller is better). The float
/// fields are wrapped in [`std::cmp::Reverse`] so that a higher angle/Jacobian
/// wins, and in [`OrderedF64`] so the tuple implements `Ord`.
fn score(
    g: &Gate,
) -> (
    usize,
    usize,
    usize,
    std::cmp::Reverse<OrderedF64>,
    std::cmp::Reverse<OrderedF64>,
) {
    (
        g.triangles,
        g.inverted,
        g.cracks,
        std::cmp::Reverse(OrderedF64(g.min_corner_angle_deg)),
        std::cmp::Reverse(OrderedF64(g.min_scaled_jacobian)),
    )
}

/// The `Quad4` connectivity of `mesh`, in cell order. Used to hand the mesh to
/// the `sun` passes without copying the whole `Cell` enum.
fn quad_cells(mesh: &Mesh) -> Vec<[usize; 4]> {
    mesh.cells
        .iter()
        .filter(|c| c.kind == Kind::Quad4)
        .map(|c| [c.nodes[0], c.nodes[1], c.nodes[2], c.nodes[3]])
        .collect()
}

/// The `Tri3` connectivity of `mesh`, in cell order. The closure only ever
/// consumes or produces triangles through these arrays.
fn tri_cells(mesh: &Mesh) -> Vec<[usize; 3]> {
    mesh.cells
        .iter()
        .filter(|c| c.kind == Kind::Tri3)
        .map(|c| [c.nodes[0], c.nodes[1], c.nodes[2]])
        .collect()
}

/// Rebuild cells from quad/triangle arrays, quads first. The order matters only
/// for the report and for deterministic output.
fn cells_of(quads: &[[usize; 4]], tris: &[[usize; 3]]) -> Vec<Cell> {
    let mut cells: Vec<Cell> = quads.iter().map(|q| Cell::quad4(*q)).collect();
    cells.extend(tris.iter().map(|t| Cell::tri3(*t)));
    cells
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn arc(cx: f64, cy: f64, r: f64, a0: f64, a1: f64, n: usize) -> Vec<[f64; 2]> {
        (0..=n)
            .map(|i| {
                let a = a0 + (a1 - a0) * i as f64 / n as f64;
                [cx + r * a.cos(), cy + r * a.sin()]
            })
            .collect()
    }

    #[test]
    fn a_square_is_supported_and_comes_back_all_quad() {
        let domain = Domain::rectangle(10.0, 10.0).unwrap();
        let sizing = Sizing::uniform(1.0).unwrap();
        let q = mesh_quads(&domain, &sizing).expect("a square is above the gate");
        assert!(q.gate.passes());
        assert_eq!(q.gate.triangles, 0);
        assert!(q.mesh.cells.iter().all(|c| c.kind == Kind::Quad4));
    }

    #[test]
    fn a_bad_size_is_an_input_error() {
        let domain = Domain::rectangle(10.0, 10.0).unwrap();
        let sizing = Sizing {
            size: 0.0,
            all_quad: false,
        };
        match mesh_quads(&domain, &sizing) {
            Err(QuadError::Input(_)) => {}
            Ok(_) => panic!("a zero size must not mesh"),
            Err(e) => panic!("expected an input error, got {e}"),
        }
    }

    #[test]
    fn a_mesh_below_the_gate_is_unsupported_not_shipped() {
        // The ring section at size 1 leaves two zero-area triangles. If this
        // starts passing, the ring was fixed: update the support claim and turn
        // this into a success test.
        let mut outer = arc(0.0, 0.0, 20.0, 0.0, std::f64::consts::FRAC_PI_2, 24);
        let inner = arc(0.0, 0.0, 10.0, std::f64::consts::FRAC_PI_2, 0.0, 12);
        outer.extend(inner);
        let domain = Domain::new(outer, vec![]).unwrap();
        match mesh_quads(&domain, &Sizing::uniform(1.0).unwrap()) {
            Err(QuadError::Unsupported(u)) => {
                assert!(u.gate.triangles > 0 || !u.gate.passes());
                assert!(!u.reason.is_empty());
                let message = QuadError::Unsupported(Box::new(Unsupported {
                    gate: u.gate,
                    raw_gate: u.gate,
                    stats: Stats::default(),
                    timing: Timing::default(),
                    route: Route::Front,
                    reason: u.reason.clone(),
                    refusals: Refusals::default(),
                    mesh: Mesh::new(Vec::new(), Vec::new()),
                }))
                .to_string();
                assert!(message.starts_with("not supported yet:"), "{message}");
            }
            Ok(q) => panic!(
                "the ring now meets the gate ({} triangles, SJ {:.3}); update the support claim",
                q.gate.triangles, q.gate.min_scaled_jacobian
            ),
            Err(e) => panic!("unexpected error: {e}"),
        }
    }
}
