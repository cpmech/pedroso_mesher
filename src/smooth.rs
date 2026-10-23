//! Boundary-preserving smoothing, applicable after any of the quad routes.
//!
//! By default the smoother moves **interior** nodes only: boundary nodes stay
//! exactly where the generator put them, and sharp domain corners never move
//! (that is what makes it safe after a structured O-grid). Two opt-in paths can
//! slide a boundary node along the polyline (never off it, and never a corner):
//! `move_boundary` for the relaxation, and `perturb_rounds > 0` for the
//! perturbation. It maximises the worst incident gate margin
//! `min(angle/30 deg, SJ/0.2, aspect_cap/aspect)` by coordinate search, then
//! optionally runs the angle-driven quad edge swaps and the boundary-aware
//! perturbation. It returns both the before/after statistics and whether the
//! smoothed mesh should replace the original.
//!
//! **Where the algorithm comes from.** The coordinate-search smoother is this
//! project's own: it does not minimise a log barrier or solve a linear system;
//! it probes a small fixed stencil of candidate positions per node and keeps
//! the best one. The *idea* it borrows is sastry-2014 (Sastry, Shontz & Vavasis,
//! *A log-barrier method for mesh quality improvement and untangling*): the
//! objective is a barrier that falls to -inf as an element becomes inverted, so
//! a search step cannot cross to the infeasible side and is pushed off it.
//! Here the barrier is the minimum of the normalised gate margins, which is
//! cheap, bounded and derivative-free; the consequence is the same, because a
//! candidate that inverts a cell scores -1 and is never kept. The edge swaps
//! are purely geometric: the shared edge of two quads is re-diagonalised when
//! the pair's worst corner angle improves while both new quads stay above the
//! gate SJ. They do **not** change the valence of a node, so they are not the
//! singularity reduction of verma-2017 (*Combinatorial approach to reduce
//! singularities in quadrilateral meshes*): that paper's atomic operation, an
//! interior edge swap accepted when it lowers `sum |valence - 4|`, is the idea
//! behind the valence histogram in [`MeshStatistics`], and it lives in the
//! untangle pass, not here. The full MST patch templates of verma-2017 are
//! **not** implemented in this crate. The accept/reject criterion is the
//! project's contract, not a paper: [`crate::verify::Gate`] (all-quad,
//! SJ >= 0.2, corner angle >= 30 deg, watertight, no inverted or overlapping
//! cells; angle floor relaxed by the user on 2026-10-22).
//!
//! **Why the Pareto guards exist (measured).** An earlier version maximised
//! only the angle. On the paving route it raised the minimum angle while
//! dropping the global minimum scaled Jacobian from 0.685 to 0.347 and creating
//! quads of aspect ratio 125 to 1094: the search traded one hidden failure for
//! another, because the angle of the incident cells is a local objective and
//! nothing in it forbade a long thin cell. The guards in [`relax_interior`] and
//! in the swap acceptance make the objective genuinely Pareto: a move may not
//! lower the global min SJ nor raise the global max aspect past
//! `max(current max, aspect_cap)`, so a smoothed mesh can only improve the tails
//! it is judged on. The same idea in its simplest form is [`better`]: keep a
//! pass, lift a failure, never lower the margin floor.
//!
//! **What is approximated / missing.** Default runs move interior nodes only;
//! the boundary is as the generator left it unless `move_boundary` is set (the
//! relaxation slides boundary nodes along the polyline) or a perturbation round
//! touches a low quad on the boundary (it slides them the same way). Sharp
//! corners are held in both cases. The coordinate search is a 17-point stencil
//! (centroid plus two rings), not a continuous optimisation, so a node can stop
//! at a local optimum; there is no line search and no remeshing. Triangles get
//! the angle objective without SJ or aspect (quality has no triangle corner
//! Jacobian). The smoother cannot add or remove a singularity or a triangle;
//! the swaps only re-diagonalise a pair of quads, so a defect that needs an MST
//! patch stays.

use crate::mesh::{Kind, Mesh};
use crate::quality;
use crate::simons_smooth;
use crate::stats::MeshStatistics;
use crate::sun;
use crate::two_d::{smooth_mesh, Domain};

/// Options for [`smooth`].
///
/// The defaults are the measured constants of the route (ten sweeps, swaps on,
/// perturbation off), so `SmoothOptions::default()` is what the examples run.
/// A setup that disables every pass is recognised by [`smooth`] and returns the
/// input without work.
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct SmoothOptions {
    /// Run the smoother at all (default `true`). When `false`, [`smooth`]
    /// returns the input unchanged with `accepted = true`; this is the switch
    /// the route configuration uses to compare a route with and without the
    /// pass.
    pub enabled: bool,

    /// Node-relaxation sweeps over the interior nodes (default 10). One sweep
    /// visits every interior node once; the loop stops early when a sweep moves
    /// nothing. Cost is O(sweeps * nodes * 17 candidate evaluations).
    pub iterations: usize,

    /// Also let boundary nodes slide along the polyline (off by default).
    /// Sharp domain corners stay fixed. The default keeps every boundary node
    /// where the generator put it; this mode uses
    /// [`crate::two_d::smooth_mesh`] instead of the interior coordinate search.
    pub move_boundary: bool,

    /// Run the angle-driven quad edge swaps (default `true`): up to 12 rounds
    /// of [`crate::sun::improve_quality`]. The whole candidate mesh is kept only
    /// if it does not lower the global min SJ nor raise the aspect tail past
    /// `max(current max, aspect_cap)` (see the module note).
    pub improve_angles: bool,

    /// Rounds of the boundary-aware perturbation (0 = off): the nodes of low
    /// quads are nudged by a fraction of the local size, boundary nodes only
    /// along the boundary ([`crate::sun::perturb_low_quads`]). Off by default
    /// because on the ring it moved boundary nodes up to 1.92 x size off the
    /// domain; it is for the cases no swap reaches.
    pub perturb_rounds: usize,

    /// If the smoothed mesh is worse than the input, reject it (default
    /// `true`). With `false` the caller takes [`SmoothOutcome::smoothed`]
    /// unconditionally — useful to inspect a bad run, not to build a mesh.
    pub reject_if_worse: bool,

    /// Aspect ratio above which a quad is treated as a sliver (default
    /// [`ASPECT_CAP`], 30). Dimensionless (`longest / shortest` edge). It
    /// enters the per-cell margin as `aspect_cap / aspect` and the accept
    /// guards as a ceiling; lowering it makes the smoother more conservative
    /// about long thin quads.
    pub aspect_cap: f64,

    /// Weight of the thesis angle+stretch objective in the final shape pass
    /// (default 1.0). [`crate::simons_smooth`] adds a stretch term (longest
    /// diagonal / shortest edge) to the angle deviation, so the optimiser
    /// cannot turn a square into an elongated rectangle of the right area.
    /// Measured on a perturbed 6x6 grid: angle-only raises the aspect max
    /// 1.666 -> 2.009, angle+stretch keeps it at 1.594 while still lifting the
    /// minimum angle 64.96 -> 78.57 degrees. Set to 0 to disable the term.
    pub stretch_weight: f64,

    /// Run the angle+stretch shape pass after the other passes (default true,
    /// all-quad meshes only; a mesh with triangles skips it).
    pub shape_pass: bool,
}

impl Default for SmoothOptions {
    fn default() -> Self {
        Self {
            enabled: true,
            iterations: 10,
            move_boundary: false,
            improve_angles: true,
            perturb_rounds: 0,
            reject_if_worse: true,
            aspect_cap: ASPECT_CAP,
            stretch_weight: 1.0,
            shape_pass: true,
        }
    }
}

/// The result of one smoothing run: both measurements and the smoothed mesh.
///
/// `before` and `after` are full [`MeshStatistics`], so a caller can log the
/// change (or the reason it was rejected) without re-measuring. `smoothed` is
/// returned even when it was rejected, so a failing mesh can still be written
/// out for diagnosis; the caller decides whether to use it with `accepted`.
#[derive(Clone, Debug)]
pub struct SmoothOutcome {
    /// Statistics of the input mesh, measured before any move.
    pub before: MeshStatistics,

    /// Statistics of [`Self::smoothed`], measured after the last pass.
    pub after: MeshStatistics,

    /// The smoothed mesh, whether or not it was accepted. Node coordinates are
    /// cloned from the input; cell order can change when swaps run (quads
    /// first, then triangles).
    pub smoothed: Mesh,

    /// Whether the smoothed mesh improves the gate margin (or lifts a failure).
    /// Also `true` when smoothing is disabled: nothing changed, so the input is
    /// accepted as it is.
    pub accepted: bool,
}

/// Smooth a mesh after a generator.
///
/// Pipeline: (1) interior-node coordinate search ([`relax_interior`]) or, with
/// `move_boundary`, the boundary-sliding relaxation of
/// [`crate::two_d::smooth_mesh`]; (2) optional angle swaps
/// ([`crate::sun::improve_quality`]) accepted only under the Pareto guards;
/// (3) optional perturbation ([`crate::sun::perturb_low_quads`]); (4) measure
/// and accept or reject with [`better`]. `size` is the target element size in
/// length units; it sets the search radii (`0.25 * size`, `0.08 * size`) and
/// the perturbation step.
///
/// The input is not modified: the mesh is cloned first. No candidate position
/// outside the domain is ever accepted, and the function never panics.
pub fn smooth(domain: &Domain, mesh: &Mesh, size: f64, options: &SmoothOptions) -> SmoothOutcome {
    let before = MeshStatistics::of(domain, mesh);
    // Nothing to do: return the input as its own result. `accepted` is true
    // because the input is not being replaced by something worse.
    if !options.enabled
        || (options.iterations == 0 && !options.improve_angles && options.perturb_rounds == 0)
    {
        return SmoothOutcome {
            before: before.clone(),
            after: before,
            smoothed: mesh.clone(),
            accepted: true,
        };
    }
    let mut work = mesh.clone();
    // Two different relaxations: the default is the interior coordinate search
    // with the Pareto guards; `move_boundary` delegates to the older
    // boundary-sliding relaxation, which may move boundary nodes.
    if options.move_boundary {
        smooth_mesh(&mut work, domain, options.iterations.max(1), size);
    } else {
        relax_interior(
            domain,
            &mut work,
            size,
            options.iterations,
            options.aspect_cap,
        );
    }
    if options.improve_angles {
        // The swaps are a discrete change, so they are accepted as a whole
        // mesh, not per swap: a swapped pair can raise its own angle and still
        // leave a long thin quad behind. The same two guards as the node
        // search (measured failure without them: paving SJ 0.685 -> 0.347,
        // aspect 125 to 1094).
        let before_swaps = MeshStatistics::of(domain, &work);
        let (points, quads, tris) = split(&work);
        let (p, q, t, _) = sun::improve_quality(&points, &quads, &tris, 12);
        let candidate = join(p, &q, &t);
        let s = MeshStatistics::of(domain, &candidate);
        if s.min_scaled_jacobian >= before_swaps.min_scaled_jacobian - 1.0e-9
            && s.aspect_ratio.max <= before_swaps.aspect_ratio.max.max(options.aspect_cap) + 1.0e-9
        {
            work = candidate;
        }
    }
    if options.perturb_rounds > 0 {
        // Perturbation is last and is not re-guarded afterwards: the caller's
        // `better` check is the final veto, and on the ring the boundary was
        // moved up to 1.92 x size, which the conforming flags then report.
        let (points, quads, tris) = split(&work);
        let (p, q, t, _) = sun::perturb_low_quads(
            &points,
            &quads,
            &tris,
            size,
            options.perturb_rounds,
            Some(domain),
        );
        work = join(p, &q, &t);
    }
    // The thesis section 2.2.10 shape pass. The angle-only objective above can
    // still leave elongated rectangles (the user's observation): it raises the
    // minimum angle while the aspect tail grows. The stretch term is the
    // reference answer, and it is kept only when it actually reduces the tail
    // and the mesh stays above the gate, so it cannot trade a pass for shape.
    if options.shape_pass && options.stretch_weight > 0.0 {
        let all_quads = work.cells.iter().all(|c| c.kind == Kind::Quad4);
        if all_quads {
            let before_shape = MeshStatistics::of(domain, &work);
            let thesis = simons_smooth::Options {
                iterations: 30,
                stretch_weight: options.stretch_weight,
                allow_boundary: false,
                simplex: true,
            };
            if let Ok(candidate) = simons_smooth::smooth(domain, &work, size, &thesis) {
                let s = MeshStatistics::of(domain, &candidate);
                let shape_gain = s.aspect_ratio.max < before_shape.aspect_ratio.max - 1.0e-9
                    || s.aspect_ratio.p95 < before_shape.aspect_ratio.p95 - 1.0e-9;
                let gate_safe = s.margin() >= 1.0 - 1.0e-9;
                let topology_ok = s.cracks <= before_shape.cracks
                    && s.overlapping <= before_shape.overlapping
                    && s.inverted <= before_shape.inverted;
                if shape_gain && gate_safe && topology_ok {
                    work = candidate;
                }
            }
        }
    }
    let after = MeshStatistics::of(domain, &work);
    // Reject-by-default is the safe choice for an automatic pipeline; the
    // comparison itself is [`better`].
    let accepted = !options.reject_if_worse || better(&before, &after);
    SmoothOutcome {
        before,
        after,
        smoothed: work,
        accepted,
    }
}

/// Decides whether a smoothed mesh may replace the input.
///
/// Order matters: a failure lifted to a pass is always accepted, a pass turned
/// into a failure is always rejected (topology and gate thresholds dominate),
/// and between two meshes of the same verdict the gate margin floor may not go
/// down. Two failing meshes still compare by margin, so a run that improves a
/// failure without fixing it is accepted — the smoother is not expected to
/// remove a triangle or a crack, so the gate verdict can stay FAIL.
pub fn better(before: &MeshStatistics, after: &MeshStatistics) -> bool {
    if !before.passes && after.passes {
        // A pass is always an improvement, but the topology must not regress.
        return after.cracks == 0 && after.overlapping == 0 && after.inverted == 0;
    }
    if before.passes && !after.passes {
        return false;
    }
    // Between two failing meshes, a smoother that gains a crack, an overlap or
    // an inverted cell is not an improvement even if its angle/SJ margin rose.
    if after.cracks > before.cracks
        || after.overlapping > before.overlapping
        || after.inverted > before.inverted
    {
        return false;
    }
    // Nor is one that blows the aspect tail past the configured cap.
    if after.aspect_ratio.max > before.aspect_ratio.max.max(ASPECT_CAP) + 1.0e-9 {
        return false;
    }
    after.margin() >= before.margin() - 1.0e-9
}

/// The aspect ratio above which a quad is considered a sliver for smoothing.
///
/// Dimensionless and fixed at 30. A rectangle 30 times longer than it is wide
/// still has scaled Jacobian 1 and 90 deg corners, which is exactly why this
/// cap exists: the gate minima cannot see a long thin cell, and it is bad for a
/// finite-element solve anyway. Beyond the cap the margin starts to fall off
/// (`aspect_cap / aspect < 1`) and the accept guards watch the maximum.
const ASPECT_CAP: f64 = 30.0;

/// Worst gate margin of one cell: `min(angle/30 deg, SJ/0.2, aspect_cap/aspect)`.
///
/// Dimensionless, normalised so 1.0 is exactly on the gate. A non-positive area
/// or a non-finite angle returns `-1.0`, the barrier's infeasible side: such a
/// candidate is never selected (and is also filtered by the Pareto guards).
/// The aspect term is what keeps the relaxation from trading a sliver for a
/// better angle: without it the search produced quads of aspect 125 to 1094
/// while raising the minimum angle (measured on the paving route; global min SJ
/// fell 0.685 -> 0.347). Triangles have no corner Jacobian in this code base,
/// so they are scored by angle alone.
fn margin(cell: &crate::mesh::Cell, points: &[crate::mesh::Node], aspect_cap: f64) -> f64 {
    // Signed volume: a non-positive value (or a NaN from a zero-length edge)
    // means the cell is inverted or degenerate, and no quality score is
    // defined there.
    let area = quality::volume(cell, points);
    let angles = quality::corner_angles(cell, points);
    let angle = angles.into_iter().fold(f64::INFINITY, f64::min);
    if area.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) || !angle.is_finite() {
        return -1.0;
    }
    if cell.kind == Kind::Quad4 {
        let sj = quality::scaled_jacobian(cell, points)
            .into_iter()
            .fold(f64::INFINITY, f64::min);
        let aspect = quality::aspect_ratio(cell, points);
        // `max(1.0)` protects against a non-finite aspect: `max` ignores NaN,
        // so a broken cell falls back to the angle/SJ terms instead of scoring
        // a spurious infinity.
        (angle / 30.0)
            .min(sj / 0.2)
            .min(aspect_cap / aspect.max(1.0))
    } else {
        angle / 30.0
    }
}

/// Explode a [`Mesh`] into the three parallel arrays the `sun` helpers use.
///
/// The point ids are kept as they are, so [`join`] can rebuild the same mesh;
/// any cell kind other than quad or triangle is dropped (the 2D routes never
/// produce one).
fn split(mesh: &Mesh) -> (Vec<[f64; 3]>, Vec<[usize; 4]>, Vec<[usize; 3]>) {
    let mut quads = Vec::new();
    let mut tris = Vec::new();
    for cell in &mesh.cells {
        match cell.kind {
            Kind::Quad4 => quads.push([cell.nodes[0], cell.nodes[1], cell.nodes[2], cell.nodes[3]]),
            Kind::Tri3 => tris.push([cell.nodes[0], cell.nodes[1], cell.nodes[2]]),
            _ => {}
        }
    }
    (mesh.points.clone(), quads, tris)
}

/// Rebuild a [`Mesh`] from the arrays produced by [`split`].
///
/// Quads are emitted before triangles, so the cell order changes relative to
/// the input (the geometry and node ids do not).
fn join(points: Vec<[f64; 3]>, quads: &[[usize; 4]], tris: &[[usize; 3]]) -> Mesh {
    let mut cells: Vec<crate::mesh::Cell> =
        quads.iter().map(|q| crate::mesh::Cell::quad4(*q)).collect();
    cells.extend(tris.iter().map(|t| crate::mesh::Cell::tri3(*t)));
    Mesh::new(points, cells)
}

/// Which nodes lie on the mesh boundary: an edge used by exactly one cell.
///
/// Same topological definition as `verify::boundary_node_mask` and the stats
/// module. The mask is computed once per relaxation call and never updated,
/// which is safe because boundary nodes are never moved by [`relax_interior`].
fn boundary_mask(mesh: &Mesh) -> Vec<bool> {
    let mut count: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
    for cell in &mesh.cells {
        for i in 0..cell.nodes.len() {
            let a = cell.nodes[i];
            let b = cell.nodes[(i + 1) % cell.nodes.len()];
            *count.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    let mut mask = vec![false; mesh.points.len()];
    for ((a, b), n) in count {
        if n == 1 {
            if a < mask.len() {
                mask[a] = true;
            }
            if b < mask.len() {
                mask[b] = true;
            }
        }
    }
    mask
}

/// Coordinate search for each interior node, maximising its worst incident
/// margin. Boundary nodes are never touched.
///
/// For each interior node the stencil is the ring centroid plus two rings of 8
/// points at `0.25 * size` and `0.08 * size`; a candidate is skipped when it
/// leaves the domain, when it would lower the global min SJ, or when it would
/// raise the global max aspect beyond `max(initial max, aspect_cap)`. A move is
/// taken only when the worst incident margin improves, so a node in a good
/// neighbourhood does not drift. Returns the number of moves accepted over all
/// sweeps (informational; [`smooth`] does not consume it).
fn relax_interior(
    domain: &Domain,
    mesh: &mut Mesh,
    size: f64,
    iterations: usize,
    aspect_cap: f64,
) -> usize {
    let n = mesh.points.len();
    let mut incident: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (c, cell) in mesh.cells.iter().enumerate() {
        for &v in &cell.nodes {
            if v < n {
                incident[v].push(c);
            }
        }
    }
    let on_boundary = boundary_mask(mesh);
    // Pareto guards: a node move may not lower the global minimum SJ nor raise
    // the global maximum aspect ratio; inside those bounds it maximises margin.
    // The global minima only change through the moved node's incident cells, so
    // checking those cells against the frozen global floor is equivalent to
    // checking the global floor, and much cheaper. The floor is frozen at the
    // *initial* mesh, so an improvement made later in the sweep cannot be spent
    // on a new sliver: no cell is ever allowed below where the input started.
    // Measured motivation: the unguarded objective lost paving SJ 0.685 ->
    // 0.347 and produced aspect 125 to 1094.
    let probe = MeshStatistics::of(domain, mesh);
    let sj_floor = probe.min_scaled_jacobian;
    let aspect_ceiling = probe.aspect_ratio.max.max(aspect_cap);
    let mut moved = 0usize;
    for _ in 0..iterations {
        let mut sweep = 0usize;
        for v in 0..n {
            if incident[v].is_empty() || on_boundary[v] {
                continue;
            }
            let here = [mesh.points[v][0], mesh.points[v][1]];
            // The one-ring: every other node of every incident cell. Its
            // centroid is the cheap "regular position" candidate.
            let mut ring: Vec<usize> = Vec::new();
            for &c in &incident[v] {
                for &u in &mesh.cells[c].nodes {
                    if u != v && u < n && !ring.contains(&u) {
                        ring.push(u);
                    }
                }
            }
            if ring.len() < 3 {
                continue;
            }
            let centroid = [
                ring.iter().map(|&u| mesh.points[u][0]).sum::<f64>() / ring.len() as f64,
                ring.iter().map(|&u| mesh.points[u][1]).sum::<f64>() / ring.len() as f64,
            ];
            let score = |m: &Mesh, v: usize| -> f64 {
                incident[v]
                    .iter()
                    .map(|&c| margin(&m.cells[c], &m.points, aspect_cap))
                    .fold(f64::INFINITY, f64::min)
            };
            // The same incident cells, checked against the global guards.
            // Only quads are checked: a triangle has no corner Jacobian in this
            // code base, and its angle is already in `score`.
            let allowed = |m: &Mesh, v: usize| -> bool {
                let mut sj = f64::INFINITY;
                let mut aspect: f64 = 1.0;
                for &c in &incident[v] {
                    let cell = &m.cells[c];
                    if cell.kind == Kind::Quad4 {
                        sj = sj.min(
                            quality::scaled_jacobian(cell, &m.points)
                                .into_iter()
                                .fold(f64::INFINITY, f64::min),
                        );
                        aspect = aspect.max(quality::aspect_ratio(cell, &m.points));
                    }
                }
                sj >= sj_floor - 1.0e-12 && aspect <= aspect_ceiling + 1.0e-12
            };
            let current = score(mesh, v);
            let mut best = current;
            let mut best_pos = here;
            let mut candidates = vec![centroid];
            // Two radii: the coarse one can jump past a whole cell, the fine
            // one polishes locally. 8 points per ring keeps the cost
            // deterministic (17 candidates per node).
            for (radius, steps) in [(0.25 * size, 8), (0.08 * size, 8)] {
                for k in 0..steps {
                    let a = std::f64::consts::TAU * k as f64 / steps as f64;
                    candidates.push([here[0] + radius * a.cos(), here[1] + radius * a.sin()]);
                }
            }
            for cand in candidates {
                // `contains` is the only geometric restriction: a node may not
                // leave the domain even if the margin would improve.
                if !domain.contains(cand) {
                    continue;
                }
                mesh.points[v][0] = cand[0];
                mesh.points[v][1] = cand[1];
                let m = score(mesh, v);
                if !allowed(mesh, v) {
                    continue;
                }
                if m > best + 1.0e-9 {
                    best = m;
                    best_pos = cand;
                }
            }
            // Restore the node even when no candidate beat `here`.
            mesh.points[v][0] = best_pos[0];
            mesh.points[v][1] = best_pos[1];
            if best > current + 1.0e-9 {
                sweep += 1;
            }
        }
        moved += sweep;
        if sweep == 0 {
            break;
        }
    }
    moved
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::mesh::Cell;

    fn grid_with_center(center: [f64; 2]) -> (Domain, Mesh) {
        let mut p = Vec::new();
        for j in 0..3 {
            for i in 0..3 {
                p.push([i as f64, j as f64, 0.0]);
            }
        }
        p[4] = [center[0], center[1], 0.0];
        let mut cells = Vec::new();
        for j in 0..2 {
            for i in 0..2 {
                let a = j * 3 + i;
                cells.push(Cell::quad4([a, a + 1, a + 4, a + 3]));
            }
        }
        (Domain::rectangle(2.0, 2.0).unwrap(), Mesh::new(p, cells))
    }

    #[test]
    fn a_skewed_interior_node_is_pulled_back() {
        let (d, m) = grid_with_center([0.8, 0.5]);
        let before = MeshStatistics::of(&d, &m);
        let out = smooth(&d, &m, 1.0, &SmoothOptions::default());
        assert!(out.after.min_corner_angle_deg > before.min_corner_angle_deg);
        assert!((out.smoothed.points[4][0] - 1.0).abs() < 0.2);
        assert!(out.accepted);
    }

    #[test]
    fn boundary_nodes_never_move() {
        let (d, m) = grid_with_center([0.8, 0.5]);
        let out = smooth(&d, &m, 1.0, &SmoothOptions::default());
        let mask = boundary_mask(&m);
        for (v, &b) in mask.iter().enumerate() {
            if b {
                assert!((out.smoothed.points[v][0] - m.points[v][0]).abs() < 1.0e-12);
                assert!((out.smoothed.points[v][1] - m.points[v][1]).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn a_topology_regression_is_not_better() {
        let (d, m) = grid_with_center([1.0, 1.0]);
        let mut before = MeshStatistics::of(&d, &m);
        before.passes = false; // pretend both fail for a reason other than topology
        let mut after = before.clone();
        after.passes = false;
        after.cracks += 1;
        assert!(!better(&before, &after));
        let mut after = before.clone();
        after.passes = false;
        after.overlapping += 1;
        assert!(!better(&before, &after));
        let mut after = before.clone();
        after.passes = false;
        after.inverted += 1;
        assert!(!better(&before, &after));
        // A pure quality gain is still accepted.
        let mut after = before.clone();
        after.passes = false;
        after.min_corner_angle_deg += 5.0;
        assert!(better(&before, &after));
    }

    #[test]
    fn the_shape_pass_never_worsens_the_aspect_tail() {
        // A wavy boundary plus perturbed interior nodes leaves rectangles that
        // the angle-only passes accept; the thesis stretch pass must not make
        // the aspect tail worse, and the gate must stay safe.
        let mut points = Vec::new();
        for j in 0..7 {
            for i in 0..7 {
                let mut x = i as f64;
                let mut y = j as f64;
                let boundary = i == 0 || i == 6 || j == 0 || j == 6;
                if boundary {
                    x += 0.18 * (3.0 * y).sin();
                    y += 0.18 * (2.5 * x).cos();
                } else {
                    x += 0.12 * ((i * 7 + j * 3) as f64 * 0.7).sin();
                    y += 0.12 * ((i * 5 + j * 11) as f64 * 0.9).cos();
                }
                points.push([x, y, 0.0]);
            }
        }
        let mut cells = Vec::new();
        for j in 0..6 {
            for i in 0..6 {
                let a = j * 7 + i;
                cells.push(crate::mesh::Cell::quad4([a, a + 1, a + 8, a + 7]));
            }
        }
        let mesh = Mesh::new(points, cells);
        let domain = Domain::rectangle(6.0, 6.0).unwrap();
        let with = SmoothOptions::default();
        let without = SmoothOptions {
            shape_pass: false,
            ..SmoothOptions::default()
        };
        let a = smooth(&domain, &mesh, 1.0, &with);
        let b = smooth(&domain, &mesh, 1.0, &without);
        assert!(a.after.passes, "{}", a.after.line());
        assert!(
            a.after.aspect_ratio.max <= b.after.aspect_ratio.max + 1.0e-9,
            "shape pass made the aspect tail worse: {} > {}",
            a.after.aspect_ratio.max,
            b.after.aspect_ratio.max
        );
        assert!(a.after.aspect_ratio.p95 <= b.after.aspect_ratio.p95 + 1.0e-9);
        // The boundary is fixed in both runs.
        for (v, p) in mesh.points.iter().enumerate() {
            if v % 7 == 0 || v % 7 == 6 || v < 7 || v >= 42 {
                assert!((a.smoothed.points[v][0] - p[0]).abs() < 1.0e-12);
                assert!((a.smoothed.points[v][1] - p[1]).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn a_regular_grid_is_left_alone() {
        let (d, m) = grid_with_center([1.0, 1.0]);
        let out = smooth(&d, &m, 1.0, &SmoothOptions::default());
        assert!(out.accepted);
        assert!((out.smoothed.points[4][0] - 1.0).abs() < 1.0e-12);
        assert!((out.smoothed.points[4][1] - 1.0).abs() < 1.0e-12);
    }
}
