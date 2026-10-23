//! Singularity seeding for the Chapter 5 separatrix-QGP path (Section 5.1.4).
//!
//! The thesis relaxes the strict QGP region constraint of eq. (5.1) by adding
//! *valence-3/valence-5 singularity pairs* to a simple tile. A pair preserves
//! the neutral holonomy of the tile (the indices `-1` and `+1` cancel), but it
//! lets a chord entering the tile leave it in the other direction: the number
//! of subdivisions on opposite sides may differ, as long as the two
//! discrepancies have the same magnitude. That is what makes a single tile
//! able to carry a different number of chords on its dense and its sparse
//! side.
//!
//! This module implements two of the pieces of that section:
//!
//! 1. **The pair field.** A cross-field angle `alpha` is modified by
//!    `(arg(z - p3) - arg(z - p5)) / 4`. The `1/4` is the cross-field exponent:
//!    a full turn of the argument produces a quarter turn of `alpha`, so the
//!    winding of `exp(4 i alpha)` is `-2 pi` around `p3` (valence 3) and
//!    `+2 pi` around `p5` (valence 5), exactly the pair of Section 5.1.4. The
//!    superposition is neutral at infinity (the two arguments cancel), so the
//!    boundary data of the underlying field is unchanged; the pair only
//!    redirects chords between the two points.
//!
//! 2. **The relaxed integer counts.** For a tile with ideal counts
//!    `s_hat = [bottom, right, top, left]` (eq. 5.1's `integral ds / sigma`),
//!    [`relaxed_counts`] minimises the thesis energy
//!    `sum (s_i / s_hat_i - 1)^2` plus `penalty * k` over integer counts that
//!    satisfy `|s0 - s2| = |s1 - s3| = k`, with `k` the number of pairs used in
//!    the tile (Section 5.1.4's relaxation; `k = 0` is eq. (5.1) itself). The
//!    penalty is the thesis' "penalty to each chord redirected in this way".
//!
//! # Faithful
//!
//! The neutral-holonomy pair (`1/4` exponent, opposite windings), the
//! relaxation `|s0 - s2| = |s1 - s3|`, the eq. (5.1) energy and the
//! redirection penalty. [`seed_field`] changes only the per-vertex field; the
//! singularity detection, the motorcycle graph, the simple tiles, the integer
//! solve and the Coons patches are the existing Chapter 3-5 code
//! ([`crate::simons_regions`], [`crate::simons_qgp`]).
//!
//! # Simplified / not implemented
//!
//! * The pair positions are a heuristic ([`pair_positions`]): they are spread
//!   along the iso-size line through the domain centre, separated along the
//!   size gradient by `0.75 sigma`, with no search over the combinatorial
//!   placement (Section 5.1.4 calls the optimal placement an open problem).
//! * The chord-redirection *mesh* of Figure 5.4 (a tile meshed with `k` pairs
//!   and unequal opposite counts) is not implemented. The tiles produced here
//!   are still meshed by the tensor-product Coons patch of
//!   [`crate::simons_qgp::mesh_tile_curves`], which requires equal opposite
//!   counts. [`relaxed_counts`] reports the counts the thesis would use; the
//!   pipeline currently runs the seeded field through the existing strict QGP
//!   solve, so the pairs can only help by changing the motorcycle graph.
//! * There is no singularity merging (Section 5.1.5.2) and no scale-factor
//!   search (Section 5.1.7).
//!
//! # Measured result (2026-10-23)
//!
//! On the two grading cases of the task the seeded field does **not** beat the
//! metric tile grid:
//!
//! * `sigma = 1 + 0.5 x` on a 10x10 square, `base_size = 1`, `pairs = 3`:
//!   five valence-3 and five valence-5 singularities are detected, but the
//!   motorcycle graph leaves only 4 tiles (19 collapsed); the interior is 7
//!   quads covering 25.9 % of the area with worst size ratio 1.557. The
//!   pipeline rejects the attempt at its 98 % coverage guard (and never
//!   stitches a mesh with holes); bypassing the guard, the stitch still fails
//!   for 1-4 pairs because every carve radius removes every quad. The metric
//!   path gives 16 quads at `worst = 1.494`.
//! * `sigma = 1 + 0.3 x + 0.3 y`: the same story (1 tile, 2 quads, 21.4 %
//!   coverage, worst 1.410); the metric path gives 9 quads at `worst = 1.392`.
//!
//! The [[`relaxed_counts`]] solve explains why a single tile cannot fix this:
//! for the one-tile ideals `[3.6, 1.7, 3.6, 10]` a pair would have to unbalance
//! the horizontal pair by the same amount that it relieves the vertical pair,
//! so with the thesis' redirection penalty the energy still prefers the strict
//! (collapsed) counts. The pairs only pay off after the motorcycle graph has
//! already cut the domain into tiles whose own ideals are nearly consistent.
//!
//! The cause is structural, not a tuning problem: with ~17 target quads over a
//! 10x10 domain, most of the motorcycle-graph tiles are larger than the local
//! size, so the integer solve collapses them (Section 5.1.6) and the kept
//! tiles cannot cover the domain. Seeding therefore stays opt-in
//! ([`crate::simons_pipeline::Options::seeded`]) and the metric path remains
//! the default.
//!
//! # Public API
//!
//! ```
//! use pedroso_mesher::simons_seeded::{relaxed_counts, Pair};
//!
//! let pair = Pair { valence3: [3.0, 3.0], valence5: [4.0, 4.0] };
//! assert_eq!(pair.valence3[0], 3.0);
//! let (counts, used) = relaxed_counts([10.0, 1.7, 3.6, 3.6], 8, 0.5).unwrap();
//! assert_eq!(counts.len(), 4);
//! let _ = used;
//! ```

use crate::mesh::{Kind, Mesh};
use crate::quality;
use crate::simons_godf;
use crate::simons_regions::{self, Regions};
use crate::two_d::Domain;

/// A valence-3 / valence-5 singularity pair of Section 5.1.4.
///
/// `valence3` is the `-1` index of the pair and `valence5` the `+1` index:
/// their sum is the neutral holonomy that keeps the tile's degree unchanged.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pair {
    /// Position of the valence-3 singularity.
    pub valence3: [f64; 2],
    /// Position of the valence-5 singularity.
    pub valence5: [f64; 2],
}

/// What [`seeded_mesh`] found.
#[derive(Clone, Debug, Default)]
pub struct Info {
    /// `(valence-3, valence-5)` singularities detected in the seeded field.
    pub detected: (usize, usize),
    /// Simple tiles kept by the QGP solve.
    pub tiles: usize,
    /// Faces rejected as degenerate (not four-sided).
    pub degenerate_tiles: usize,
    /// Tiles collapsed by the integer solve.
    pub collapsed_tiles: usize,
    /// Quads of the welded tile mesh.
    pub quads: usize,
    /// Area of the meshed tiles.
    pub interior_area: f64,
    /// Area of the input triangle mesh.
    pub domain_area: f64,
}

impl Info {
    /// Fraction of the domain area covered by the meshed tiles.
    pub fn coverage(&self) -> f64 {
        if self.domain_area > 0.0 {
            self.interior_area / self.domain_area
        } else {
            0.0
        }
    }
}

/// One measured case of [`demo`].
#[derive(Clone, Debug, Default)]
pub struct DemoCase {
    /// Short name of the case.
    pub name: &'static str,
    /// Number of pairs requested.
    pub requested_pairs: usize,
    /// `(valence-3, valence-5)` singularities detected after seeding.
    pub detected: (usize, usize),
    /// Simple tiles kept.
    pub tiles: usize,
    /// Tiles collapsed by the integer solve.
    pub collapsed_tiles: usize,
    /// Quads of the interior mesh.
    pub quads: usize,
    /// Fraction of the domain covered by the interior mesh.
    pub coverage: f64,
    /// Worst `max(sqrt(area)/sigma, sigma/sqrt(area))` of the interior.
    pub worst_ratio: f64,
    /// Failure of the field path, when it could not produce a mesh.
    pub error: Option<String>,
}

/// Numbers measured by [`demo`].
#[derive(Clone, Debug, Default)]
pub struct DemoNumbers {
    /// `sigma = 1 + 0.5 x` on a 10x10 square.
    pub linear_x: DemoCase,
    /// `sigma = 1 + 0.3 x + 0.3 y` on a 10x10 square.
    pub two_dimensional: DemoCase,
    /// The relaxed counts of `[10, 1.7, 3.6, 3.6]` with `k <= 8`, penalty 0.5.
    pub relaxed: ([usize; 4], usize),
}

/// Add the `1/4`-exponent pair fields of `pairs` to a per-vertex cross-field
/// angle.
///
/// `alpha` is the representation returned by [`crate::simons_godf::solve`]
/// (`u_i = exp(4 i alpha_i)`); the result has one value per input vertex and is
/// *not* wrapped, which is fine because every consumer evaluates `exp(4 i
/// alpha)` and wraps transitions. Returns an error when `alpha` does not have
/// exactly one finite value per point, or when a pair is not finite.
pub fn seed_field(points: &[[f64; 3]], alpha: &[f64], pairs: &[Pair]) -> Result<Vec<f64>, String> {
    if points.len() != alpha.len() {
        return Err(format!(
            "simons_seeded: {} field values for {} points",
            alpha.len(),
            points.len()
        ));
    }
    for (index, &value) in alpha.iter().enumerate() {
        if !value.is_finite() {
            return Err(format!(
                "simons_seeded: the field value {index} is not finite ({value})"
            ));
        }
    }
    for (index, pair) in pairs.iter().enumerate() {
        if !pair.valence3[0].is_finite()
            || !pair.valence3[1].is_finite()
            || !pair.valence5[0].is_finite()
            || !pair.valence5[1].is_finite()
        {
            return Err(format!(
                "simons_seeded: pair {index} has a non-finite position"
            ));
        }
    }
    let mut seeded = Vec::with_capacity(points.len());
    for (point, &value) in points.iter().zip(alpha.iter()) {
        let mut angle = value;
        for pair in pairs {
            let d3 = [point[0] - pair.valence3[0], point[1] - pair.valence3[1]];
            let d5 = [point[0] - pair.valence5[0], point[1] - pair.valence5[1]];
            angle += 0.25 * (d3[1].atan2(d3[0]) - d5[1].atan2(d5[0]));
        }
        seeded.push(angle);
    }
    Ok(seeded)
}

/// Place `count` pairs along the size-gradient direction through the domain.
///
/// The pairs are spread on the iso-size line through the domain centre (the
/// direction perpendicular to `grad sigma`), and each pair is separated along
/// `grad sigma` by `0.75 sigma` at its centre. This is the "highest
/// size-gradient locations" heuristic of the task, not the thesis' open
/// combinatorial placement: it is deterministic, it keeps the pair inside the
/// domain when the domain contains its bounding-box centre, and it cancels to
/// a neutral field far away.
///
/// Returns an error for a non-positive `count`, a domain without a finite
/// bounding box, or a size callback that is not positive and finite at the
/// sampled positions. Pairs whose two points would fall outside the domain are
/// dropped (so the result may be shorter than `count`).
pub fn pair_positions<F>(domain: &Domain, sigma: &F, count: usize) -> Result<Vec<Pair>, String>
where
    F: Fn(f64, f64) -> f64,
{
    if count == 0 {
        return Err("simons_seeded: the pair count must be positive".to_string());
    }
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for point in &domain.outer {
        if !point[0].is_finite() || !point[1].is_finite() {
            return Err("simons_seeded: the domain has a non-finite point".to_string());
        }
        min_x = min_x.min(point[0]);
        max_x = max_x.max(point[0]);
        min_y = min_y.min(point[1]);
        max_y = max_y.max(point[1]);
    }
    if max_x <= min_x || max_y <= min_y {
        return Err("simons_seeded: the domain bounding box is degenerate".to_string());
    }
    let width = max_x - min_x;
    let height = max_y - min_y;
    let centre = [0.5 * (min_x + max_x), 0.5 * (min_y + max_y)];
    let h = 1.0e-3 * width.min(height).max(1.0e-6);
    let left = sample_size(sigma, centre[0] - h, centre[1])?;
    let right = sample_size(sigma, centre[0] + h, centre[1])?;
    let bottom = sample_size(sigma, centre[0], centre[1] - h)?;
    let top = sample_size(sigma, centre[0], centre[1] + h)?;
    let mut gx = (right - left) / (2.0 * h);
    let mut gy = (top - bottom) / (2.0 * h);
    let norm = (gx * gx + gy * gy).sqrt();
    if !norm.is_finite() || norm <= 1.0e-12 {
        // Constant size: any direction works; use +x.
        gx = 1.0;
        gy = 0.0;
    } else {
        gx /= norm;
        gy /= norm;
    }
    // Perpendicular (the iso-size direction).
    let px = -gy;
    let py = gx;
    let span = width * px.abs() + height * py.abs();
    let step = if count > 1 {
        span / (count as f64 + 1.0)
    } else {
        0.0
    };
    let mut pairs = Vec::with_capacity(count);
    for index in 0..count {
        let offset = if count > 1 {
            (index as f64 + 1.0 - 0.5 * (count as f64 + 1.0)) * step
        } else {
            0.0
        };
        let base = [centre[0] + px * offset, centre[1] + py * offset];
        let local = sample_size(sigma, base[0], base[1])?;
        let separation = 0.75 * local;
        let p3 = [
            base[0] - 0.5 * separation * gx,
            base[1] - 0.5 * separation * gy,
        ];
        let p5 = [
            base[0] + 0.5 * separation * gx,
            base[1] + 0.5 * separation * gy,
        ];
        let inside = |p: [f64; 2]| domain.contains(p);
        if inside(p3) && inside(p5) {
            pairs.push(Pair {
                valence3: p3,
                valence5: p5,
            });
        }
    }
    Ok(pairs)
}

/// A positive, finite size sample.
fn sample_size<F>(sigma: &F, x: f64, y: f64) -> Result<f64, String>
where
    F: Fn(f64, f64) -> f64,
{
    let value = sigma(x, y);
    if !value.is_finite() || value <= 0.0 {
        return Err(format!(
            "simons_seeded: the size function returned {value} at ({x}, {y}); \
             it must be positive and finite"
        ));
    }
    Ok(value)
}

/// The §5.1.4 relaxed integer counts of one tile.
///
/// `ideals` is `[bottom, right, top, left]` (`s_hat = integral ds / sigma`);
/// the thesis minimises `sum (s_i / s_hat_i - 1)^2` under `s_i >= 0` and
/// `|s0 - s2| = |s1 - s3| = k`, where `k` is the number of singularity pairs
/// in the tile, and adds `penalty * k` to bias against redirecting chords.
/// `max_pairs` bounds `k` (0 is the strict eq. (5.1) solve).
///
/// Returns the counts and the `k` that was used. The search is exhaustive over
/// `k`, the two signs of the discrepancy, and integer counts up to twice the
/// largest ideal plus `max_pairs` (clamped to 64), so it is inexpensive for
/// one tile but not intended as a global solver.
pub fn relaxed_counts(
    ideals: [f64; 4],
    max_pairs: usize,
    penalty: f64,
) -> Result<([usize; 4], usize), String> {
    for (index, &ideal) in ideals.iter().enumerate() {
        if !ideal.is_finite() || ideal <= 0.0 {
            return Err(format!(
                "simons_seeded: ideal count {index} must be positive and finite \
                 (got {ideal})"
            ));
        }
    }
    if !penalty.is_finite() || penalty < 0.0 {
        return Err(format!(
            "simons_seeded: the redirection penalty must be finite and \
             non-negative (got {penalty})"
        ));
    }
    let largest = ideals.iter().copied().fold(0.0f64, f64::max);
    let bound = (2 * largest.ceil() as usize + max_pairs + 2).min(64);
    let mut best: Option<([usize; 4], usize, f64)> = None;
    for pairs in 0..=max_pairs {
        for &sign in &[-1i64, 1] {
            for s0 in 0..=bound {
                for s1 in 0..=bound {
                    let s2 = s0 as i64 + sign * pairs as i64;
                    let s3 = s1 as i64 - sign * pairs as i64;
                    if s2 < 0 || s3 < 0 {
                        continue;
                    }
                    let counts = [s0, s1, s2 as usize, s3 as usize];
                    let energy = counts_energy(counts, ideals) + penalty * pairs as f64;
                    let better = match best {
                        None => true,
                        Some((current, current_pairs, current_energy)) => {
                            energy < current_energy - 1.0e-12
                                || ((energy - current_energy).abs() <= 1.0e-12
                                    && (pairs, counts) < (current_pairs, current))
                        }
                    };
                    if better {
                        best = Some((counts, pairs, energy));
                    }
                }
            }
        }
    }
    match best {
        Some((counts, pairs, _)) => Ok((counts, pairs)),
        None => Err("simons_seeded: the relaxed count search is empty".to_string()),
    }
}

/// The eq. (5.1) energy of one tile.
fn counts_energy(counts: [usize; 4], ideals: [f64; 4]) -> f64 {
    let mut energy = 0.0;
    for index in 0..4 {
        let ratio = counts[index] as f64 / ideals[index] - 1.0;
        energy += ratio * ratio;
    }
    energy
}

/// Seed the field, build the QGP regions and mesh every kept tile.
///
/// This is the Chapter 5 machinery unchanged ([`simons_regions::build`] and
/// [`Regions::mesh`]) on the pair-seeded field; it does **not** stitch to the
/// boundary (that is [`crate::simons_regions::stitch`], which the pipeline
/// calls with the same seeded field).
pub fn seeded_mesh(
    points: &[[f64; 3]],
    tris: &[[usize; 3]],
    alpha: &[f64],
    size: &[f64],
    pairs: &[Pair],
) -> Result<(Mesh, Info), String> {
    if points.len() != size.len() {
        return Err(format!(
            "simons_seeded: {} sizes for {} points",
            size.len(),
            points.len()
        ));
    }
    for (index, &value) in size.iter().enumerate() {
        if !value.is_finite() || value <= 0.0 {
            return Err(format!(
                "simons_seeded: size {index} is {value}; it must be positive and finite"
            ));
        }
    }
    let seeded = seed_field(points, alpha, pairs)?;
    let detected = simons_godf::count_singularities(&seeded, tris);
    let regions: Regions = simons_regions::build(points, tris, &seeded, size)?;
    let mesh = regions.mesh()?;
    let mut interior_area = 0.0;
    for cell in &mesh.cells {
        if cell.kind == Kind::Quad4 {
            interior_area += quality::volume(cell, &mesh.points).abs();
        }
    }
    let info = Info {
        detected,
        tiles: regions.tiles.len(),
        degenerate_tiles: regions.degenerate_tiles,
        collapsed_tiles: regions.collapsed_tiles,
        quads: mesh.cells.len(),
        interior_area,
        domain_area: triangle_area(points, tris),
    };
    Ok((mesh, info))
}

/// The area of a planar triangle mesh.
fn triangle_area(points: &[[f64; 3]], tris: &[[usize; 3]]) -> f64 {
    let mut total = 0.0;
    for triangle in tris {
        let [a, b, c] = *triangle;
        if a >= points.len() || b >= points.len() || c >= points.len() {
            continue;
        }
        let ab = [points[b][0] - points[a][0], points[b][1] - points[a][1]];
        let ac = [points[c][0] - points[a][0], points[c][1] - points[a][1]];
        total += 0.5 * (ab[0] * ac[1] - ab[1] * ac[0]).abs();
    }
    total
}

/// The worst and mean size ratio of a quad mesh against `sigma`.
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
                "simons_seeded: quad {index} has non-positive area {area}"
            ));
        }
        let mut cx = 0.0;
        let mut cy = 0.0;
        for &node in &cell.nodes {
            let point = match mesh.points.get(node) {
                Some(point) => point,
                None => {
                    return Err(format!(
                        "simons_seeded: quad {index} references node {node} out of range"
                    ))
                }
            };
            cx += point[0];
            cy += point[1];
        }
        let nodes = cell.nodes.len();
        if nodes == 0 {
            return Err(format!("simons_seeded: quad {index} has no nodes"));
        }
        cx /= nodes as f64;
        cy /= nodes as f64;
        let target = sigma(cx, cy);
        if !target.is_finite() || target <= 0.0 {
            return Err(format!(
                "simons_seeded: the size function returned {target} at quad {index}'s \
                 centre ({cx}, {cy})"
            ));
        }
        let length = area.sqrt();
        let ratio = (length / target).max(target / length);
        total += ratio;
        worst = worst.max(ratio);
        count += 1;
    }
    if count == 0 {
        return Err("simons_seeded: the mesh has no quad".to_string());
    }
    Ok((total / count as f64, worst))
}

/// The measured demo: the two grading cases of the task.
///
/// Both are 10x10 squares at `base_size = 1`, solved with the accurate,
/// tangent-boundary field the pipeline's field path uses, seeded with three
/// pairs. The returned numbers are the interior QGP mesh (before the stitch),
/// which is what the pairs can affect; the stitch result is reported by the
/// pipeline.
pub fn demo() -> Result<DemoNumbers, String> {
    let domain = Domain::rectangle(10.0, 10.0)
        .map_err(|error| format!("simons_seeded demo: the domain is invalid: {error}"))?;
    let linear_x = measure_case("linear-x", &domain, &|x, _y| 1.0 + 0.5 * x, 3)?;
    let two_dimensional = measure_case("2d-gradient", &domain, &|x, y| 1.0 + 0.3 * x + 0.3 * y, 3)?;
    let relaxed = relaxed_counts([10.0, 1.7, 3.6, 3.6], 8, 0.5)?;
    Ok(DemoNumbers {
        linear_x,
        two_dimensional,
        relaxed,
    })
}

/// One [`demo`] case.
fn measure_case<F>(
    name: &'static str,
    domain: &Domain,
    sigma: &F,
    pairs: usize,
) -> Result<DemoCase, String>
where
    F: Fn(f64, f64) -> f64,
{
    let (points, tris) = rectangle_grid(10, 10, 1.0)?;
    let size: Vec<f64> = points
        .iter()
        .map(|point| sigma(point[0], point[1]))
        .collect();
    let field_options = simons_godf::Options {
        accurate: true,
        holomorphic: 0.0,
        boundary: simons_godf::BoundaryMode::Tangent,
        ..simons_godf::Options::default()
    };
    let alpha = simons_godf::solve_with_options(&points, &tris, &size, &field_options, 400)?;
    let placed = pair_positions(domain, sigma, pairs)?;
    let mut case = DemoCase {
        name,
        requested_pairs: placed.len(),
        ..DemoCase::default()
    };
    match seeded_mesh(&points, &tris, &alpha, &size, &placed) {
        Ok((mesh, info)) => {
            case.detected = info.detected;
            case.tiles = info.tiles;
            case.collapsed_tiles = info.collapsed_tiles;
            case.quads = info.quads;
            case.coverage = info.coverage();
            match size_ratios(&mesh, sigma) {
                Ok((_, worst)) => case.worst_ratio = worst,
                Err(error) => case.error = Some(error),
            }
        }
        Err(error) => case.error = Some(error),
    }
    Ok(case)
}

/// A structured `n x n` grid of a unit-cell rectangle, two triangles per cell.
fn rectangle_grid(n: usize, m: usize, h: f64) -> Result<(Vec<[f64; 3]>, Vec<[usize; 3]>), String> {
    if n == 0 || m == 0 {
        return Err("simons_seeded: the demo grid is empty".to_string());
    }
    if !h.is_finite() || h <= 0.0 {
        return Err(format!("simons_seeded: the demo cell size {h} is invalid"));
    }
    let mut points: Vec<[f64; 3]> = Vec::with_capacity((n + 1) * (m + 1));
    for j in 0..=m {
        for i in 0..=n {
            points.push([i as f64 * h, j as f64 * h, 0.0]);
        }
    }
    let index = |i: usize, j: usize| j * (n + 1) + i;
    let mut tris: Vec<[usize; 3]> = Vec::with_capacity(2 * n * m);
    for j in 0..m {
        for i in 0..n {
            let a = index(i, j);
            let b = index(i + 1, j);
            let c = index(i + 1, j + 1);
            let d = index(i, j + 1);
            tris.push([a, b, c]);
            tris.push([a, c, d]);
        }
    }
    Ok((points, tris))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn square() -> Domain {
        Domain::rectangle(10.0, 10.0).unwrap()
    }

    #[test]
    fn a_pair_seeds_one_valence_three_and_one_valence_five() {
        let (points, tris) = rectangle_grid(10, 10, 1.0).unwrap();
        let alpha = vec![0.0; points.len()];
        let pair = Pair {
            valence3: [3.0, 4.0],
            valence5: [5.0, 6.0],
        };
        let seeded = seed_field(&points, &alpha, &[pair]).unwrap();
        assert_eq!(seeded.len(), points.len());
        let detected = simons_godf::count_singularities(&seeded, &tris);
        println!("one pair: detected = {detected:?}");
        assert_eq!(detected, (1, 1));
        // The pair is neutral at infinity: the mean angle change over the
        // boundary is small.
        assert!(seeded.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn seeding_validates_its_inputs() {
        let points = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        assert!(seed_field(&points, &[0.0], &[]).is_err());
        assert!(seed_field(&points, &[f64::NAN, 0.0], &[]).is_err());
        let pair = Pair {
            valence3: [f64::NAN, 0.0],
            valence5: [1.0, 1.0],
        };
        assert!(seed_field(&points, &[0.0, 0.0], &[pair]).is_err());
    }

    #[test]
    fn pair_positions_follow_the_size_gradient_and_stay_inside() {
        let domain = square();
        let sigma = |x: f64, _y: f64| 1.0 + 0.5 * x;
        let pairs = pair_positions(&domain, &sigma, 3).unwrap();
        println!("linear-x pairs = {pairs:?}");
        assert_eq!(pairs.len(), 3);
        for pair in &pairs {
            assert!(domain.contains(pair.valence3));
            assert!(domain.contains(pair.valence5));
            assert!(pair.valence5[0] > pair.valence3[0]);
            assert!((pair.valence5[0] - pair.valence3[0]).abs() > 0.1);
        }
        let diagonal = pair_positions(&domain, &|x, y| 1.0 + 0.3 * x + 0.3 * y, 2);
        assert!(diagonal.is_ok());
        // A constant size still produces valid pairs along +x.
        let constant = pair_positions(&domain, &|_, _| 2.0, 2).unwrap();
        assert_eq!(constant.len(), 2);
        assert!(pair_positions(&domain, &|_, _| 0.0, 2).is_err());
    }

    #[test]
    fn the_strict_relaxation_reduces_to_eq_five_one() {
        let (counts, used) = relaxed_counts([2.0, 3.0, 2.0, 3.0], 0, 0.0).unwrap();
        println!("strict counts = {counts:?}, k = {used}");
        assert_eq!(used, 0);
        assert_eq!(counts[0], counts[2]);
        assert_eq!(counts[1], counts[3]);
        assert_eq!(counts, [2, 3, 2, 3]);
    }

    #[test]
    fn the_one_tile_gradient_prefers_the_strict_counts() {
        // A single tile spanning the whole 10x10 gradient: relieving the
        // vertical pair (left 10, right 1.7) forces the horizontal pair
        // (bottom/top 3.6) to differ by the same amount, which costs more than
        // it saves. With the thesis' redirection penalty the strict counts win.
        let (counts, used) = relaxed_counts([3.6, 1.7, 3.6, 10.0], 8, 0.5).unwrap();
        println!("one-tile gradient: counts = {counts:?}, k = {used}");
        assert_eq!(used, 0);
        assert_eq!(counts, [4, 2, 4, 2]);
    }

    #[test]
    fn the_relaxation_carries_a_different_count_on_opposite_sides() {
        // A tile whose four ideals already differ consistently: bottom 10,
        // right 4, top 9, left 3. One pair (k = 1) fits all four sides with
        // zero energy; the strict q. (5.1) solve has to compromise.
        let (counts, used) = relaxed_counts([10.0, 4.0, 9.0, 3.0], 8, 0.0).unwrap();
        println!("relaxed counts = {counts:?}, k = {used}");
        assert_eq!(used, 1);
        let d0 = (counts[0] as i64 - counts[2] as i64).unsigned_abs() as usize;
        let d1 = (counts[1] as i64 - counts[3] as i64).unsigned_abs() as usize;
        assert_eq!(d0, used);
        assert_eq!(d1, used);
        // The two opposite pairs are out of balance by the same amount, which
        // is exactly the neutral-holonomy pair of Section 5.1.4.
        assert!(counts[0] != counts[2]);
        assert!(counts[1] != counts[3]);
    }

    #[test]
    fn the_relaxation_penalty_prefers_straight_chords() {
        let (_, free) = relaxed_counts([2.4, 2.4, 2.4, 2.4], 4, 0.0).unwrap();
        let (counts, paid) = relaxed_counts([2.4, 2.4, 2.4, 2.4], 4, 5.0).unwrap();
        println!("free k = {free}, paid k = {paid}, paid counts = {counts:?}");
        assert_eq!(free, 0, "no redirection is needed when the ideals agree");
        assert_eq!(paid, 0);
        assert!(relaxed_counts([1.0, 1.0, 1.0, 1.0], 2, -1.0).is_err());
    }

    #[test]
    fn seeded_mesh_reports_coverage_and_fails_cleanly() {
        let (points, tris) = rectangle_grid(10, 10, 1.0).unwrap();
        let size: Vec<f64> = points.iter().map(|p| 1.0 + 0.5 * p[0]).collect();
        let alpha = vec![0.0; points.len()];
        let pair = Pair {
            valence3: [3.0, 4.0],
            valence5: [4.0, 5.0],
        };
        match seeded_mesh(&points, &tris, &alpha, &size, &[pair]) {
            Ok((mesh, info)) => {
                println!(
                    "seeded mesh: tiles={} quads={} coverage={:.3} area={:.2}/{:.2}",
                    info.tiles,
                    info.quads,
                    info.coverage(),
                    info.interior_area,
                    info.domain_area
                );
                assert!(info.domain_area > 0.0);
                assert!(!mesh.cells.is_empty());
                assert!(info.coverage() > 0.0 && info.coverage() <= 1.0 + 1e-9);
            }
            Err(error) => println!("seeded mesh failed cleanly: {error}"),
        }
        assert!(seeded_mesh(&points, &tris, &alpha, &size[..size.len() - 1], &[]).is_err());
    }

    #[test]
    fn demo_numbers_are_measured() {
        let numbers = demo().unwrap();
        println!(
            "demo {}: requested={} detected={:?} tiles={} collapsed={} quads={} \
             coverage={:.3} worst={:.3} error={:?}",
            numbers.linear_x.name,
            numbers.linear_x.requested_pairs,
            numbers.linear_x.detected,
            numbers.linear_x.tiles,
            numbers.linear_x.collapsed_tiles,
            numbers.linear_x.quads,
            numbers.linear_x.coverage,
            numbers.linear_x.worst_ratio,
            numbers.linear_x.error,
        );
        println!(
            "demo {}: requested={} detected={:?} tiles={} collapsed={} quads={} \
             coverage={:.3} worst={:.3} error={:?}",
            numbers.two_dimensional.name,
            numbers.two_dimensional.requested_pairs,
            numbers.two_dimensional.detected,
            numbers.two_dimensional.tiles,
            numbers.two_dimensional.collapsed_tiles,
            numbers.two_dimensional.quads,
            numbers.two_dimensional.coverage,
            numbers.two_dimensional.worst_ratio,
            numbers.two_dimensional.error,
        );
        println!(
            "relaxed [10, 1.7, 3.6, 3.6]: counts={:?} k={}",
            numbers.relaxed.0, numbers.relaxed.1
        );
        assert_eq!(numbers.linear_x.requested_pairs, 3);
        assert!(numbers.linear_x.error.is_some() || numbers.linear_x.quads > 0);
        // The one-tile gradient does not profit from a pair (see the module
        // documentation); the relaxed solve with a consistent tile does.
        assert_eq!(numbers.relaxed.1, 0);
        let (_, used) = relaxed_counts([10.0, 4.0, 9.0, 3.0], 8, 0.0).unwrap();
        assert_eq!(used, 1);
    }
}
