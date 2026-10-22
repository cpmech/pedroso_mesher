//! Boundary-preserving smoothing, applicable after any of the quad routes.
//!
//! The smoother moves **interior** nodes only: boundary nodes stay exactly
//! where the generator put them, and sharp domain corners never move (that is
//! what makes it safe after a structured O-grid). It maximises the worst
//! incident gate margin `min(angle/30°, SJ/0.2)` by coordinate search, then
//! optionally runs the angle-driven edge swaps and the boundary-aware
//! perturbation. It returns both the before/after statistics and whether the
//! smoothed mesh should replace the original.

use crate::mesh::{Kind, Mesh};
use crate::quality;
use crate::stats::MeshStatistics;
use crate::sun;
use crate::two_d::{smooth_mesh, Domain};

/// Options for [`smooth`].
#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct SmoothOptions {
    /// Run the smoother at all (default true).
    pub enabled: bool,
    /// Node-relaxation sweeps (interior nodes only).
    pub iterations: usize,
    /// Also let boundary nodes slide along the polyline (off by default).
    pub move_boundary: bool,
    /// Run the angle-driven quad edge swaps.
    pub improve_angles: bool,
    /// Rounds of the boundary-aware perturbation (0 = off).
    pub perturb_rounds: usize,
    /// If the smoothed mesh is worse than the input, reject it (default true).
    pub reject_if_worse: bool,
    /// Aspect ratio above which a quad is treated as a sliver (default 30).
    pub aspect_cap: f64,
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
        }
    }
}

/// The result of one smoothing run: both measurements and the smoothed mesh.
#[derive(Clone, Debug)]
pub struct SmoothOutcome {
    pub before: MeshStatistics,
    pub after: MeshStatistics,
    /// The smoothed mesh, whether or not it was accepted.
    pub smoothed: Mesh,
    /// Whether the smoothed mesh improves the gate margin (or lifts a failure).
    pub accepted: bool,
}

/// Smooth a mesh after a generator.
pub fn smooth(domain: &Domain, mesh: &Mesh, size: f64, options: &SmoothOptions) -> SmoothOutcome {
    let before = MeshStatistics::of(domain, mesh);
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
    let after = MeshStatistics::of(domain, &work);
    let accepted = !options.reject_if_worse || better(&before, &after);
    SmoothOutcome {
        before,
        after,
        smoothed: work,
        accepted,
    }
}

/// A smoothed mesh replaces the input when it keeps a pass it had or lifts a
/// failure, and does not lower the gate margin floor.
pub fn better(before: &MeshStatistics, after: &MeshStatistics) -> bool {
    if !before.passes && after.passes {
        return true;
    }
    if before.passes && !after.passes {
        return false;
    }
    after.margin() >= before.margin() - 1.0e-9
}

/// The aspect ratio above which a quad is considered a sliver for smoothing.
const ASPECT_CAP: f64 = 30.0;

/// Worst gate margin of one cell: `min(angle/30°, SJ/0.2, 30/aspect)`.
///
/// The aspect term is what keeps the relaxation from trading a sliver for a
/// better angle: without it the search produced quads of aspect 125-1094 while
/// raising the minimum angle.
fn margin(cell: &crate::mesh::Cell, points: &[crate::mesh::Node], aspect_cap: f64) -> f64 {
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
        (angle / 30.0)
            .min(sj / 0.2)
            .min(aspect_cap / aspect.max(1.0))
    } else {
        angle / 30.0
    }
}

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

fn join(points: Vec<[f64; 3]>, quads: &[[usize; 4]], tris: &[[usize; 3]]) -> Mesh {
    let mut cells: Vec<crate::mesh::Cell> =
        quads.iter().map(|q| crate::mesh::Cell::quad4(*q)).collect();
    cells.extend(tris.iter().map(|t| crate::mesh::Cell::tri3(*t)));
    Mesh::new(points, cells)
}

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
            for (radius, steps) in [(0.25 * size, 8), (0.08 * size, 8)] {
                for k in 0..steps {
                    let a = std::f64::consts::TAU * k as f64 / steps as f64;
                    candidates.push([here[0] + radius * a.cos(), here[1] + radius * a.sin()]);
                }
            }
            for cand in candidates {
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
    fn a_regular_grid_is_left_alone() {
        let (d, m) = grid_with_center([1.0, 1.0]);
        let out = smooth(&d, &m, 1.0, &SmoothOptions::default());
        assert!(out.accepted);
        assert!((out.smoothed.points[4][0] - 1.0).abs() < 1.0e-12);
        assert!((out.smoothed.points[4][1] - 1.0).abs() < 1.0e-12);
    }
}
