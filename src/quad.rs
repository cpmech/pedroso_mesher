//! Quad meshing of a planar domain, **gate-checked**: the product entry point.
//!
//! [`mesh_quads`] runs the whole W3 pipeline — triangulation, parity pre-pass,
//! Q-Morph front, Sun residual closure, cavity, quality repairs and polishing —
//! and returns a mesh **only when it meets the acceptance gate**
//! ([`crate::verify::gate`]). A result below the gate comes back as
//! [`QuadError::Unsupported`] with the measured numbers and the first reason
//! instead of as a mesh: the configuration is outside what this route supports
//! yet, and handing back a mesh that the gate rejects would hide that.
//!
//! This is deliberate. The gate is not a target to report against but the
//! contract: `mesh_quads` never returns something it knows fails. On the planar
//! corpus that makes the ring section at size 1 mm and the 0.6 mm performance
//! part *unsupported* for now — the ring leaves two zero-area triangles and the
//! 0.6 mm part leaves five triangles and a scaled Jacobian of 0.156.
//!
//! `two_d::mesh_2d` remains the **default** product route (quad-dominant,
//! matching-based, no gate). This module is the all-quad route.

use crate::mesh::{Cell, Kind, Mesh};
use crate::parity::even_boundary_loops;
use crate::qmorph::{convert_with_stats, Refusals};
use crate::quality;
use crate::sun;
use crate::two_d::{smooth_mesh, triangulate, Domain, Sizing, TwoDError};
use crate::twostage;
use crate::verify::{self, Gate};

/// Which conversion stage feeds the pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    /// Q-Morph front to exhaustion, then the Sun residual closure. The route the
    /// 30° gate is measured on.
    Front,
    /// Q-Morph front under the even-region guard, then Blossom matching.
    /// Measured negative (1/6) and kept as an experiment.
    TwoStage,
}

/// Counters for the stages that ran.
#[derive(Debug, Default, Clone)]
pub struct Stats {
    pub rounds: usize,
    pub cavity_closures: usize,
    pub low_repairs: usize,
    pub degenerate_repairs: usize,
    pub quality_swaps: usize,
    pub perturbations: usize,
    pub parity_splits: usize,
    /// Worst scaled Jacobian and corner angle of the quads alone (the triangles
    /// a failing mesh keeps would otherwise mask them).
    pub quads_only_sj: f64,
    pub quads_only_angle_deg: f64,
    /// Largest distance of a boundary node from the domain boundary (0 when the
    /// mesh sits on its geometry; the gate cannot see a boundary moved inward).
    pub boundary_offset: f64,
    /// Largest distance from a domain corner to the nearest boundary node (0 when
    /// every corner is kept; a corner node sliding along an edge cuts the corner
    /// while the boundary offset stays 0).
    pub corner_offset: f64,
    /// Present when [`Route::TwoStage`] ran.
    pub two_stage: Option<twostage::TwoStageStats>,
}

/// Wall-clock seconds.
#[derive(Debug, Default, Clone, Copy)]
pub struct Timing {
    pub triangulation: f64,
    pub conversion: f64,
    pub total: f64,
}

/// A quad mesh that met the gate.
pub struct QuadMesh {
    pub mesh: Mesh,
    /// The gate of the returned mesh; it passes by construction.
    pub gate: Gate,
    /// The gate right after conversion, before the closure and quality passes.
    pub raw_gate: Gate,
    pub stats: Stats,
    pub timing: Timing,
    pub refusals: Refusals,
    pub route: Route,
}

/// Why [`mesh_quads`] did not return a mesh: the measured result of a
/// configuration that the route does not support yet.
#[derive(Debug)]
pub struct Unsupported {
    pub gate: Gate,
    pub raw_gate: Gate,
    pub stats: Stats,
    pub timing: Timing,
    pub route: Route,
    /// The gate's first reason, e.g. "2 triangle(s) left".
    pub reason: String,
    pub refusals: Refusals,
    /// The mesh that failed the gate. Kept for diagnosis (drawing it is how the
    /// failures are understood); it must not be shipped as a result.
    pub mesh: Mesh,
}

/// Why [`mesh_quads`] did not return a mesh.
#[derive(Debug)]
pub enum QuadError {
    /// The domain or the size field is unusable.
    Input(TwoDError),
    /// The pipeline finished but the mesh cannot meet the gate. This
    /// configuration is outside the supported set for now.
    Unsupported(Box<Unsupported>),
}

impl std::fmt::Display for QuadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuadError::Input(e) => write!(f, "{e}"),
            QuadError::Unsupported(u) => write!(f, "not supported yet: {}", u.reason),
        }
    }
}

impl std::error::Error for QuadError {}

impl From<TwoDError> for QuadError {
    fn from(e: TwoDError) -> Self {
        QuadError::Input(e)
    }
}

/// Mesh a planar domain with quads, or say why it cannot be done yet.
///
/// Runs [`Route::Front`]. See the module documentation for the contract.
pub fn mesh_quads(domain: &Domain, sizing: &Sizing) -> Result<QuadMesh, QuadError> {
    mesh_quads_with_route(domain, sizing, Route::Front)
}

/// Mesh a planar domain with quads under an explicit conversion [`Route`].
pub fn mesh_quads_with_route(
    domain: &Domain,
    sizing: &Sizing,
    route: Route,
) -> Result<QuadMesh, QuadError> {
    let size = sizing.size;
    let t0 = std::time::Instant::now();
    let (points, tris) = triangulate(domain, sizing)?;
    // Sun 2016 §3.3: make every boundary loop even before conversion.
    let (points, tris, parity_splits) = even_boundary_loops(&points, &tris);
    let triangulation = t0.elapsed().as_secs_f64();

    let t1 = std::time::Instant::now();
    let mut two_stage = None;
    let ((quads, leftover, points), refusals) = match route {
        Route::Front => convert_with_stats(&tris, &points, size),
        Route::TwoStage => {
            let r = twostage::convert(&tris, &points, size);
            two_stage = Some(r.stats);
            ((r.quads, r.triangles, r.points), r.refusals)
        }
    };
    let conversion = t1.elapsed().as_secs_f64();

    let mut cells: Vec<Cell> = quads.iter().map(|q| Cell::quad4(*q)).collect();
    cells.extend(leftover.iter().map(|t| Cell::tri3(*t)));
    let mut mesh = Mesh::new(points, cells);
    let raw_gate = verify::gate(&mesh);

    let rounds = close_rounds(&mut mesh, domain, size);
    let cavity_closures = close_cavities(&mut mesh, domain, size);
    let (low_repairs, degenerate_repairs, quality_swaps, perturbations) =
        quality_passes(&mut mesh, domain, size);
    let gate = verify::gate(&mesh);

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
    let diagonal = mesh
        .points
        .iter()
        .fold((f64::NEG_INFINITY, f64::NEG_INFINITY), |acc, p| {
            (acc.0.max(p[0].abs()), acc.1.max(p[1].abs()))
        });
    let tolerance = 1.0e-6 * diagonal.0.max(diagonal.1).max(1.0);
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
    Ok(QuadMesh {
        mesh,
        gate,
        raw_gate,
        stats,
        timing,
        refusals,
        route,
    })
}

/// Alternate W2 smoothing with the Sun residual walk, keeping the best state
/// under the gate's own order. Returns how many rounds ran.
fn close_rounds(mesh: &mut Mesh, domain: &Domain, size: f64) -> usize {
    let mut remaining = tri_cells(mesh).len();
    let mut rounds = 0usize;
    let mut stale = 0usize;
    let mut last_remaining = remaining;
    let mut best: Option<(Mesh, Gate)> = None;
    while remaining > 0 && rounds < 10 && stale < 3 {
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
        if remaining >= last_remaining {
            stale += 1;
        } else {
            stale = 0;
        }
        last_remaining = remaining;

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
fn quality_passes(mesh: &mut Mesh, domain: &Domain, size: f64) -> (usize, usize, usize, usize) {
    let (points, quads, tris, low_repairs) =
        sun::repair_low_quads(&mesh.points, &quad_cells(mesh), &tri_cells(mesh), 0.05, 16);
    *mesh = Mesh::new(points, cells_of(&quads, &tris));

    let (points, quads, tris, degenerate_repairs) =
        sun::repair_degenerate_quads(&mesh.points, &quad_cells(mesh), &tri_cells(mesh), 0.25, 8);
    *mesh = Mesh::new(points, cells_of(&quads, &tris));

    let (points, quads, tris, quality_swaps) =
        sun::improve_quality(&mesh.points, &quad_cells(mesh), &tri_cells(mesh), 12);
    *mesh = Mesh::new(points, cells_of(&quads, &tris));

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
    let (points, quads, tris, degenerate_after) =
        sun::repair_degenerate_quads(&mesh.points, &quad_cells(mesh), &tri_cells(mesh), 0.25, 8);
    let (points, quads, tris, perturbations) =
        sun::perturb_low_quads(&points, &quads, &tris, size, 200, Some(domain));
    *mesh = Mesh::new(points, cells_of(&quads, &tris));
    (
        low_repairs,
        degenerate_repairs + degenerate_after,
        quality_swaps,
        perturbations,
    )
}

/// If triangles remain after the closure rounds, alternate the cavity with a few
/// walk/smooth rounds: the cavity changes the local topology and the walk can
/// then pair what it unlocked.
fn close_cavities(mesh: &mut Mesh, domain: &Domain, size: f64) -> usize {
    let mut closed_total = 0usize;
    for _ in 0..3 {
        if tri_cells(mesh).is_empty() {
            break;
        }
        let before = tri_cells(mesh).len();
        let (points, quads, tris, closed) =
            sun::close_by_cavity(&mesh.points, &quad_cells(mesh), &tri_cells(mesh), size);
        closed_total += closed;
        *mesh = Mesh::new(points, cells_of(&quads, &tris));
        for _ in 0..4 {
            if tri_cells(mesh).is_empty() {
                break;
            }
            smooth_mesh(mesh, domain, 80, size);
            let (points, quads, tris, _) =
                sun::close_residuals(&mesh.points, &quad_cells(mesh), &tri_cells(mesh));
            *mesh = Mesh::new(points, cells_of(&quads, &tris));
        }
        if tri_cells(mesh).len() >= before {
            break;
        }
    }
    closed_total
}

/// Total order on a float for the best-state comparison (values are finite).
#[derive(Clone, Copy, PartialEq)]
struct OrderedF64(f64);

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

fn quad_cells(mesh: &Mesh) -> Vec<[usize; 4]> {
    mesh.cells
        .iter()
        .filter(|c| c.kind == Kind::Quad4)
        .map(|c| [c.nodes[0], c.nodes[1], c.nodes[2], c.nodes[3]])
        .collect()
}

fn tri_cells(mesh: &Mesh) -> Vec<[usize; 3]> {
    mesh.cells
        .iter()
        .filter(|c| c.kind == Kind::Tri3)
        .map(|c| [c.nodes[0], c.nodes[1], c.nodes[2]])
        .collect()
}

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
