//! A graded realization of the Simons (2024) QGP pipeline on a rectangle.
//!
//! Chapter 5 asks for a *simple tile* decomposition of the domain and one
//! integer subdivision count per motorcycle-graph edge. On a graded rectangle
//! the size-aware cross field is regular (no singularity), so the motorcycle
//! graph degenerates to the single boundary face and the integer solve of
//! eq. (5.1) collapses that face (Section 5.1.6). This module sidesteps the
//! degenerate field by **constructing** a simple-tile decomposition whose tile
//! corners follow the size function, and then running the same Chapter 5
//! machinery on it:
//!
//! 1. **Tile grid.** The rectangle (or a rectangle with one axis-aligned
//!    rectangular hole) is split into `K x L` rectangular tiles. The split
//!    lines are the level sets of the cumulative size metric
//!    `W(x) = integral dx / sigma`: equal increments of `W` give tile widths
//!    proportional to `sigma`, which is the level-set spacing the size function
//!    asks for. For the linear gradient `sigma = 1 + g x` this is exactly a set
//!    of `sigma` level sets. On a plate the hole edges are forced to be tile
//!    lines, so every tile is either fully inside or fully inside the hole.
//! 2. **Edge graph.** Each tile side is a chain of exactly one edge. A side
//!    shared by two tiles is the same edge id, so the QGP constraint (5.1) is
//!    enforced across the whole grid. The ideal count of an edge is
//!    `s_hat = integral ds / sigma`, evaluated with the trapezoid rule from the
//!    two end points, as in Section 5.1.1.
//! 3. **Integer counts (Section 5.1.2).** [`crate::simons_qgp::solve_counts`]
//!    minimizes `sum (s_i / s_hat_i - 1)^2` under eq. (5.1) and `s_i >= 0`.
//! 4. **Coons patches (Section 5.1.3).** Every tile is meshed with
//!    [`crate::simons_qgp::mesh_tile_curves`] at its solved `n_u x n_v`, and the
//!    tiles are welded by coordinates.
//!
//! # Why the tile counts are chosen from the size variation
//!
//! A tile side is a single edge, so its ideal count is the number of quads that
//! fit on that side: `length / sigma` (roughly). When a requested `K` makes
//! every column thinner than half a quad, the energy of eq. (5.1) prefers
//! `s = 0` for the whole column and the QGP solve collapses it. This is the
//! same degenerate-tile collapse as Section 5.1.6, and it is *physical*: on a
//! `10 x 10` square with `sigma = 1 + 0.5 x` there are only about
//! `integral dx / sigma = 3.58` quads across the domain, so 8 columns cannot
//! each carry one quad. The default (`tiles = (0, 0)`) therefore derives the
//! tile counts from
//!
//! ```text
//! Q = integral integral dA / sigma^2
//! K = round(sqrt(Q * width / height)),  L = round(sqrt(Q * height / width))
//! ```
//!
//! which is the expected number of quads of size `sigma`. An explicit
//! `tiles = (K, L)` is honoured when that grid is feasible; otherwise the path
//! falls back to the derived counts (and, if those also collapse, halves the
//! offending direction, the simplified form of the collapse of Section 5.1.6).
//! Both the requested and the effective counts are reported in [`Info`].
//!
//! # Faithful / simplified / not implemented
//!
//! * Faithful: the simple-tile/edge model, the one-edge-per-side graph, eq.
//!   (5.1), the ideal count `integral ds / sigma` (trapezoid), the integer
//!   energy and its augmenting-path solver (from [`crate::simons_qgp`]), and the
//!   bilinear Coons patch of eq. (5.2).
//! * Simplified: the tiles are a Cartesian grid aligned with the size metric,
//!   not the separatrix graph of a solved cross field (the field has no
//!   singularity here). The tile lines are the level sets of the *cumulative*
//!   metric, which is the natural generalisation of the `sigma` level sets when
//!   `sigma` is not monotone. There are no extra singularity pairs (Section
//!   5.1.4), singularity merging (5.1.5.2) or nautilus handling.
//! * Not implemented: arbitrary non-convex domains and non-rectangular holes;
//!   the pipeline keeps its existing QGP/stitch and channel routes for those.
//!   The cross field itself is not constructed here because the grid is
//!   axis-aligned.
//!
//! # Public API
//!
//! ```
//! use pedroso_mesher::simons_graded::{self, Options};
//! use pedroso_mesher::Domain;
//!
//! # fn main() -> Result<(), String> {
//! let domain = Domain::rectangle(10.0, 10.0).map_err(|e| e.to_string())?;
//! let (mesh, info) = simons_graded::mesh(&domain, &Options::default(), &|x, _y| 1.0 + 0.5 * x)?;
//! assert_eq!(info.tiles, (4, 4));
//! assert!(mesh.cells.len() > 0);
//! # Ok(())
//! # }
//! ```
//!
//! The module has no external dependencies, never panics in non-test code, and
//! builds for `wasm32-unknown-unknown`.

use std::collections::HashMap;

use crate::mesh::{Cell, Mesh, Node};
use crate::quality;
use crate::simons_qgp::{self, BoundaryCurves, Region};
use crate::two_d::Domain;

/// Number of intervals used to integrate the cumulative size metric.
const METRIC_SAMPLES: usize = 256;

/// Number of samples across a tile line when a 2D size function is averaged.
const PERPENDICULAR_SAMPLES: usize = 32;

/// Number of cells per direction for the `integral dA / sigma^2` estimate.
const AUTO_SAMPLES: usize = 48;

/// Largest tile grid coordinate accepted. Beyond this the solver graph, not the
/// mesh, becomes the cost.
pub const MAX_TILES: usize = 64;

/// Largest number of halving steps used to recover from a collapsed grid.
const MAX_COARSEN: usize = 8;

/// Weld tolerance, relative to the domain diagonal.
const WELD_RELATIVE: f64 = 1e-9;

/// Options of the graded path.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Options {
    /// Requested tile grid. `(0, 0)` (the default) derives `K` and `L` from the
    /// size variation; any other pair is used when that grid does not collapse.
    pub tiles: (usize, usize),
}

/// The requested and effective tile counts and the resulting QGP graph.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Info {
    /// The grid requested through [`Options::tiles`] (the derived grid when the
    /// request was `(0, 0)`).
    pub requested_tiles: (usize, usize),
    /// The grid that was actually meshed.
    pub tiles: (usize, usize),
    /// True when the requested grid collapsed and a smaller one was meshed.
    pub coarsened: bool,
    /// Number of simple tiles (QGP regions).
    pub regions: usize,
    /// Number of motorcycle-graph edges (tile sides).
    pub edges: usize,
    /// Smallest solved subdivision count over the used edges.
    pub min_count: usize,
    /// Quads produced by the Coons patches.
    pub cells: usize,
}

/// Size accuracy of a mesh: `sqrt(area)` against `sigma` at the quad centroid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SizeStats {
    /// Mean of `max(sqrt(area)/sigma, sigma/sqrt(area))`.
    pub mean: f64,
    /// Maximum of the same ratio.
    pub worst: f64,
    /// Pearson correlation between `sqrt(area)` and `sigma`; near 1 means the
    /// size trend is visible, near 0 means it is not.
    pub correlation: f64,
}

/// One measured demo case.
#[derive(Clone, Debug, PartialEq)]
pub struct DemoCase {
    /// The grid requested (derived grid for the automatic case).
    pub requested_tiles: (usize, usize),
    /// The grid that was meshed.
    pub tiles: (usize, usize),
    /// True when the requested grid collapsed.
    pub coarsened: bool,
    /// Number of quads.
    pub quads: usize,
    /// The measured size accuracy.
    pub stats: SizeStats,
}

/// The three demo cases: the linear gradient, the 2D gradient with derived
/// tiles, and the 2D gradient with an explicit `8 x 8` request.
#[derive(Clone, Debug, PartialEq)]
pub struct DemoNumbers {
    /// `sigma = 1 + 0.5 x` on `10 x 10`, derived tiles.
    pub linear: DemoCase,
    /// `sigma = 1 + 0.3 x + 0.3 y` on `10 x 10`, derived tiles.
    pub two_d_auto: DemoCase,
    /// The same 2D case with an explicit `8 x 8` request.
    pub two_d_eight: DemoCase,
}

// ---------------------------------------------------------------------------
// Domain shape
// ---------------------------------------------------------------------------

/// The two planar shapes the graded grid supports.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    /// An axis-aligned rectangle in `[min_x, max_x, min_y, max_y]`.
    Rectangle([f64; 4]),
    /// An outer rectangle with one axis-aligned rectangular hole; both are
    /// `[min_x, max_x, min_y, max_y]`.
    RectangleWithHole {
        /// Outer rectangle.
        outer: [f64; 4],
        /// Hole rectangle, strictly inside `outer`.
        hole: [f64; 4],
    },
}

impl Shape {
    /// The outer bounding box `[min_x, max_x, min_y, max_y]`.
    pub fn bounds(self) -> [f64; 4] {
        match self {
            Shape::Rectangle(bounds) => bounds,
            Shape::RectangleWithHole { outer, .. } => outer,
        }
    }

    /// The hole box, when there is one.
    pub fn hole(self) -> Option<[f64; 4]> {
        match self {
            Shape::Rectangle(_) => None,
            Shape::RectangleWithHole { hole, .. } => Some(hole),
        }
    }
}

/// Recognise a rectangle or a rectangle with one rectangular hole.
///
/// Returns `None` for any other boundary (a general polygon, a circular hole,
/// a rotated rectangle, more than one hole). The outer loop must be four
/// distinct axis-aligned corners; a hole must be four distinct axis-aligned
/// corners strictly inside the outer box.
pub fn shape(domain: &Domain) -> Option<Shape> {
    let outer = axis_aligned_rect(&domain.outer)?;
    match domain.holes.len() {
        0 => Some(Shape::Rectangle(outer)),
        1 => {
            let hole_points = domain.holes.first()?;
            let hole = axis_aligned_rect(hole_points)?;
            let [ox0, ox1, oy0, oy1] = outer;
            let [hx0, hx1, hy0, hy1] = hole;
            let scale = 1.0
                + ox0
                    .abs()
                    .max(ox1.abs())
                    .max(oy0.abs())
                    .max(oy1.abs())
                    .max(hx0.abs())
                    .max(hx1.abs())
                    .max(hy0.abs())
                    .max(hy1.abs());
            let tolerance = 1e-9 * scale;
            if hx0 <= ox0 + tolerance
                || hx1 >= ox1 - tolerance
                || hy0 <= oy0 + tolerance
                || hy1 >= oy1 - tolerance
            {
                return None;
            }
            Some(Shape::RectangleWithHole { outer, hole })
        }
        _ => None,
    }
}

/// The bounding box of four distinct axis-aligned rectangle corners.
fn axis_aligned_rect(points: &[[f64; 2]]) -> Option<[f64; 4]> {
    if points.len() != 4 {
        return None;
    }
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for point in points {
        if !point[0].is_finite() || !point[1].is_finite() {
            return None;
        }
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
    for point in points {
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

// ---------------------------------------------------------------------------
// Size metric and tile lines
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Axis {
    X,
    Y,
}

/// `1 / sigma(x, y)`, rejecting a non-positive or non-finite size.
fn inverse_size<F>(sigma: &F, x: f64, y: f64) -> Result<f64, String>
where
    F: Fn(f64, f64) -> f64,
{
    let size = sigma(x, y);
    if !size.is_finite() || size <= 0.0 {
        return Err(format!(
            "simons_graded: the size function returned {size} at ({x}, {y}); \
             it must be positive and finite"
        ));
    }
    Ok(1.0 / size)
}

/// Mean of `1 / sigma` across the direction perpendicular to `axis`.
///
/// For [`Axis::X`] the fixed coordinate is `position` (an `x`) and the integral
/// runs over `[lo, hi]` in `y`; for [`Axis::Y`] the roles are swapped. Simpson's
/// rule is used so a linear `sigma` is integrated exactly.
fn mean_inverse<F>(sigma: &F, axis: Axis, position: f64, lo: f64, hi: f64) -> Result<f64, String>
where
    F: Fn(f64, f64) -> f64,
{
    let n = PERPENDICULAR_SAMPLES;
    if n < 2 || hi <= lo {
        return Err(format!(
            "simons_graded: the perpendicular interval [{lo}, {hi}] is not \
             positive"
        ));
    }
    let h = (hi - lo) / n as f64;
    let mut sum = 0.0;
    for i in 0..=n {
        let t = lo + i as f64 * h;
        let value = match axis {
            Axis::X => inverse_size(sigma, position, t)?,
            Axis::Y => inverse_size(sigma, t, position)?,
        };
        let weight = if i == 0 || i == n {
            1.0
        } else if i % 2 == 0 {
            2.0
        } else {
            4.0
        };
        sum += weight * value;
    }
    Ok(sum * h / 3.0 / (hi - lo))
}

/// `integral ds / sigma` along one straight segment, by Simpson's rule in the
/// size metric.
fn metric_integral<F>(
    sigma: &F,
    axis: Axis,
    lo: f64,
    hi: f64,
    perpendicular: [f64; 2],
) -> Result<f64, String>
where
    F: Fn(f64, f64) -> f64,
{
    let n = METRIC_SAMPLES;
    if hi <= lo {
        return Err(format!(
            "simons_graded: the metric interval [{lo}, {hi}] is not positive"
        ));
    }
    let h = (hi - lo) / n as f64;
    let mut sum = 0.0;
    for i in 0..=n {
        let t = lo + i as f64 * h;
        let value = mean_inverse(sigma, axis, t, perpendicular[0], perpendicular[1])?;
        let weight = if i == 0 || i == n {
            1.0
        } else if i % 2 == 0 {
            2.0
        } else {
            4.0
        };
        sum += weight * value;
    }
    Ok(sum * h / 3.0)
}

/// Split `[lo, hi]` into `count` intervals of equal cumulative size metric.
///
/// The result has `count + 1` strictly increasing lines. Equal increments of
/// `integral ds / sigma` make the tile width proportional to `sigma`, which is
/// the level-set spacing of the size function.
fn metric_lines<F>(
    sigma: &F,
    axis: Axis,
    lo: f64,
    hi: f64,
    count: usize,
    perpendicular: [f64; 2],
) -> Result<(Vec<f64>, f64), String>
where
    F: Fn(f64, f64) -> f64,
{
    if count == 0 {
        return Err("simons_graded: a tile direction needs at least one interval".to_string());
    }
    if hi <= lo {
        return Err(format!(
            "simons_graded: the tile interval [{lo}, {hi}] is not positive"
        ));
    }
    let n = METRIC_SAMPLES;
    let h = (hi - lo) / n as f64;
    let mut positions = Vec::with_capacity(n + 1);
    let mut weights = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let position = lo + i as f64 * h;
        positions.push(position);
        weights.push(mean_inverse(
            sigma,
            axis,
            position,
            perpendicular[0],
            perpendicular[1],
        )?);
    }
    let mut cumulative = Vec::with_capacity(n + 1);
    cumulative.push(0.0);
    for i in 0..n {
        let previous = cumulative[i];
        cumulative.push(previous + 0.5 * (weights[i] + weights[i + 1]) * h);
    }
    let total = match cumulative.last() {
        Some(&value) => value,
        None => return Err("simons_graded: the metric integral is empty".to_string()),
    };
    if !total.is_finite() || total <= 0.0 {
        return Err(format!(
            "simons_graded: the size metric integral is {total}; it must be \
             positive"
        ));
    }
    let mut lines = Vec::with_capacity(count + 1);
    lines.push(lo);
    for k in 1..count {
        let target = total * k as f64 / count as f64;
        // Find the sample bracket that contains `target`.
        let mut left = 0usize;
        let mut right = n;
        while right - left > 1 {
            let middle = (left + right) / 2;
            if cumulative[middle] < target {
                left = middle;
            } else {
                right = middle;
            }
        }
        let c0 = cumulative[left];
        let c1 = cumulative[right];
        let fraction = if c1 > c0 {
            ((target - c0) / (c1 - c0)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        lines.push(positions[left] + fraction * (positions[right] - positions[left]));
    }
    lines.push(hi);
    for pair in lines.windows(2) {
        if pair[1].partial_cmp(&pair[0]) != Some(std::cmp::Ordering::Greater) {
            return Err(
                "simons_graded: the size metric produced a repeated tile line; \
                 the size variation is too extreme for this grid"
                    .to_string(),
            );
        }
    }
    Ok((lines, total))
}

/// Distribute `total` intervals over segments proportional to `weights`.
///
/// Every segment with a positive weight gets at least one interval, so a hole
/// edge always remains a tile line. The total may be raised to the number of
/// positive segments.
fn allocate_intervals(total: usize, weights: &[f64]) -> Result<Vec<usize>, String> {
    let positive = weights.iter().filter(|&&weight| weight > 0.0).count();
    if positive == 0 {
        return Err("simons_graded: every tile segment has a zero size metric".to_string());
    }
    let total = total.max(positive);
    let sum: f64 = weights.iter().filter(|&&weight| weight > 0.0).sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err("simons_graded: the size metric sum is not positive".to_string());
    }
    let ideal: Vec<f64> = weights
        .iter()
        .map(|&weight| total as f64 * weight.max(0.0) / sum)
        .collect();
    let mut counts: Vec<usize> = ideal
        .iter()
        .map(|&value| value.floor().max(0.0) as usize)
        .collect();
    for (index, &weight) in weights.iter().enumerate() {
        if weight > 0.0 {
            if let Some(count) = counts.get_mut(index) {
                if *count == 0 {
                    *count = 1;
                }
            }
        }
    }
    let mut assigned: usize = counts.iter().sum();
    while assigned < total {
        let mut best = 0usize;
        let mut best_fraction = f64::NEG_INFINITY;
        for (index, &value) in ideal.iter().enumerate() {
            let count = counts.get(index).copied().unwrap_or(0);
            let fraction = value - count as f64;
            if fraction > best_fraction + 1e-15 {
                best_fraction = fraction;
                best = index;
            }
        }
        if let Some(count) = counts.get_mut(best) {
            *count += 1;
            assigned += 1;
        } else {
            return Err("simons_graded: interval allocation lost a segment".to_string());
        }
    }
    while assigned > total {
        let mut best: Option<usize> = None;
        let mut best_excess = f64::INFINITY;
        for (index, &value) in ideal.iter().enumerate() {
            let count = counts.get(index).copied().unwrap_or(0);
            if count <= 1 {
                continue;
            }
            let excess = count as f64 - value;
            if excess < best_excess - 1e-15 {
                best_excess = excess;
                best = Some(index);
            }
        }
        match best {
            Some(index) => {
                if let Some(count) = counts.get_mut(index) {
                    *count -= 1;
                    assigned -= 1;
                }
            }
            None => {
                return Err(
                    "simons_graded: cannot reduce the interval allocation further".to_string(),
                )
            }
        }
    }
    Ok(counts)
}

/// The tile lines of one axis, with the hole edges forced when there is a hole.
fn axis_lines<F>(
    sigma: &F,
    axis: Axis,
    bounds: [f64; 4],
    perpendicular: [f64; 2],
    count: usize,
    hole: Option<[f64; 2]>,
) -> Result<Vec<f64>, String>
where
    F: Fn(f64, f64) -> f64,
{
    let (lo, hi) = match axis {
        Axis::X => (bounds[0], bounds[1]),
        Axis::Y => (bounds[2], bounds[3]),
    };
    let segments: Vec<(f64, f64)> = match hole {
        Some([h0, h1]) if h0 > lo && h1 < hi => vec![(lo, h0), (h0, h1), (h1, hi)],
        _ => vec![(lo, hi)],
    };
    let mut weights = Vec::with_capacity(segments.len());
    for &(a, b) in &segments {
        weights.push(metric_integral(sigma, axis, a, b, perpendicular)?);
    }
    let counts = allocate_intervals(count, &weights)?;
    let mut lines: Vec<f64> = Vec::new();
    for (index, &(a, b)) in segments.iter().enumerate() {
        let segment_count = counts.get(index).copied().unwrap_or(0);
        if segment_count == 0 {
            continue;
        }
        let (mut segment, _) = metric_lines(sigma, axis, a, b, segment_count, perpendicular)?;
        if lines.is_empty() {
            lines.append(&mut segment);
        } else if !segment.is_empty() {
            // The first point is the end of the previous segment.
            segment.remove(0);
            lines.append(&mut segment);
        }
    }
    // Drop a repeated join point, if floating point made one.
    lines.dedup_by(|a, b| (*a - *b).abs() <= 1e-12 * (1.0 + a.abs().max(b.abs())));
    if lines.len() < 2 {
        return Err("simons_graded: the tile lines collapsed".to_string());
    }
    Ok(lines)
}

/// The number of intervals in each direction derived from the size variation.
///
/// `Q = integral integral dA / sigma^2` is the expected number of quads when a
/// quad has side `sigma`; the split between the two directions follows the
/// domain aspect ratio.
pub fn auto_tiles<F>(domain: &Domain, sigma: &F) -> Result<(usize, usize), String>
where
    F: Fn(f64, f64) -> f64,
{
    let shape = shape(domain)
        .ok_or_else(|| "simons_graded: the domain is not a graded rectangle".to_string())?;
    let [min_x, max_x, min_y, max_y] = shape.bounds();
    let width = max_x - min_x;
    let height = max_y - min_y;
    if width <= 0.0 || height <= 0.0 {
        return Err("simons_graded: the bounding box is degenerate".to_string());
    }
    let n = AUTO_SAMPLES;
    let dx = width / n as f64;
    let dy = height / n as f64;
    let mut quads = 0.0;
    for j in 0..n {
        let y = min_y + (j as f64 + 0.5) * dy;
        for i in 0..n {
            let x = min_x + (i as f64 + 0.5) * dx;
            if !domain.contains([x, y]) {
                continue;
            }
            let inverse = inverse_size(sigma, x, y)?;
            quads += inverse * inverse * dx * dy;
        }
    }
    if !quads.is_finite() || quads <= 0.0 {
        return Err(format!(
            "simons_graded: the estimated quad count is {quads}; it must be \
             positive"
        ));
    }
    let k = (quads * width / height).sqrt().round();
    let l = (quads * height / width).sqrt().round();
    let clamp = |value: f64| -> Result<usize, String> {
        if !value.is_finite() {
            return Err("simons_graded: the derived tile count is not finite".to_string());
        }
        Ok((value.max(1.0) as usize).min(MAX_TILES))
    };
    Ok((clamp(k)?, clamp(l)?))
}

// ---------------------------------------------------------------------------
// Tile grid, edge graph and QGP counts
// ---------------------------------------------------------------------------

/// The tile grid with its used edges and QGP regions.
struct Grid {
    xs: Vec<f64>,
    ys: Vec<f64>,
    h_id: Vec<Vec<Option<usize>>>,
    v_id: Vec<Vec<Option<usize>>>,
    ideal: Vec<f64>,
    regions: Vec<Region>,
    tiles: Vec<(usize, usize)>,
}

impl Grid {
    fn k(&self) -> usize {
        self.xs.len().saturating_sub(1)
    }

    fn l(&self) -> usize {
        self.ys.len().saturating_sub(1)
    }
}

/// `integral ds / sigma` of a horizontal edge, trapezoid at the end points.
fn horizontal_ideal<F>(sigma: &F, x0: f64, x1: f64, y: f64) -> Result<f64, String>
where
    F: Fn(f64, f64) -> f64,
{
    let left = inverse_size(sigma, x0, y)?;
    let right = inverse_size(sigma, x1, y)?;
    Ok(0.5 * (x1 - x0) * (left + right))
}

/// `integral ds / sigma` of a vertical edge, trapezoid at the end points.
fn vertical_ideal<F>(sigma: &F, x: f64, y0: f64, y1: f64) -> Result<f64, String>
where
    F: Fn(f64, f64) -> f64,
{
    let bottom = inverse_size(sigma, x, y0)?;
    let top = inverse_size(sigma, x, y1)?;
    Ok(0.5 * (y1 - y0) * (bottom + top))
}

/// Build the used-edge graph and the QGP regions for one tile grid.
fn build_grid<F>(domain: &Domain, sigma: &F, xs: &[f64], ys: &[f64]) -> Result<Grid, String>
where
    F: Fn(f64, f64) -> f64,
{
    let k = xs.len().saturating_sub(1);
    let l = ys.len().saturating_sub(1);
    if k == 0 || l == 0 {
        return Err("simons_graded: the tile grid has no cell".to_string());
    }
    let mut present = vec![vec![false; l]; k];
    for j in 0..l {
        let y = 0.5 * (ys[j] + ys[j + 1]);
        for i in 0..k {
            let x = 0.5 * (xs[i] + xs[i + 1]);
            present[i][j] = domain.contains([x, y]);
        }
    }
    let any = present.iter().flatten().any(|&value| value);
    if !any {
        return Err("simons_graded: the tile grid is empty".to_string());
    }

    let mut h_used = vec![vec![false; l + 1]; k];
    let mut v_used = vec![vec![false; l]; k + 1];
    for j in 0..l {
        for i in 0..k {
            if present[i][j] {
                h_used[i][j] = true;
                h_used[i][j + 1] = true;
                v_used[i][j] = true;
                v_used[i + 1][j] = true;
            }
        }
    }

    let mut ideal: Vec<f64> = Vec::new();
    let mut h_id: Vec<Vec<Option<usize>>> = vec![vec![None; l + 1]; k];
    for j in 0..=l {
        for i in 0..k {
            if h_used[i][j] {
                let value = horizontal_ideal(sigma, xs[i], xs[i + 1], ys[j])?;
                h_id[i][j] = Some(ideal.len());
                ideal.push(value);
            }
        }
    }
    let mut v_id: Vec<Vec<Option<usize>>> = vec![vec![None; l]; k + 1];
    for j in 0..l {
        for i in 0..=k {
            if v_used[i][j] {
                let value = vertical_ideal(sigma, xs[i], ys[j], ys[j + 1])?;
                v_id[i][j] = Some(ideal.len());
                ideal.push(value);
            }
        }
    }

    let mut regions = Vec::new();
    let mut tiles = Vec::new();
    for j in 0..l {
        for i in 0..k {
            if !present[i][j] {
                continue;
            }
            let bottom = h_id[i][j]
                .ok_or_else(|| "simons_graded: a tile is missing its bottom edge".to_string())?;
            let top = h_id[i][j + 1]
                .ok_or_else(|| "simons_graded: a tile is missing its top edge".to_string())?;
            let left = v_id[i][j]
                .ok_or_else(|| "simons_graded: a tile is missing its left edge".to_string())?;
            let right = v_id[i + 1][j]
                .ok_or_else(|| "simons_graded: a tile is missing its right edge".to_string())?;
            regions.push(Region::new([
                vec![bottom],
                vec![right],
                vec![top],
                vec![left],
            ]));
            tiles.push((i, j));
        }
    }

    Ok(Grid {
        xs: xs.to_vec(),
        ys: ys.to_vec(),
        h_id,
        v_id,
        ideal,
        regions,
        tiles,
    })
}

/// The integer count that minimizes `sum (s / s_hat - 1)^2` on one side.
fn best_side_count(ideals: &[f64]) -> Result<usize, String> {
    if ideals.is_empty() {
        return Ok(0);
    }
    let mut max_ideal = 0.0f64;
    for &ideal in ideals {
        if !ideal.is_finite() || ideal <= 0.0 {
            return Err(format!(
                "simons_graded: an edge ideal count is {ideal}; it must be \
                 positive"
            ));
        }
        max_ideal = max_ideal.max(ideal);
    }
    let limit = (max_ideal.ceil().max(0.0) as usize).saturating_add(2);
    let energy = |s: usize| -> f64 {
        ideals
            .iter()
            .map(|&ideal| {
                let difference = s as f64 / ideal - 1.0;
                difference * difference
            })
            .sum()
    };
    let mut best = 0usize;
    let mut best_energy = energy(0);
    for s in 1..=limit {
        let value = energy(s);
        if value < best_energy - 1e-12 {
            best_energy = value;
            best = s;
        }
    }
    Ok(best)
}

/// Whether the two directions of the grid admit positive integer counts.
///
/// One side of a tile is one edge, so the grid has one count per column strip
/// (for the horizontal edges) and one per row strip (for the vertical edges).
/// If the best count of any strip is zero, the QGP solve collapses it.
fn feasibility(grid: &Grid) -> Result<(bool, bool), String> {
    let k = grid.k();
    let l = grid.l();
    let mut horizontal = true;
    for i in 0..k {
        let mut ideals = Vec::new();
        for j in 0..=l {
            if let Some(id) = grid.h_id[i][j] {
                if let Some(&value) = grid.ideal.get(id) {
                    ideals.push(value);
                }
            }
        }
        if best_side_count(&ideals)? == 0 {
            horizontal = false;
            break;
        }
    }
    let mut vertical = true;
    for j in 0..l {
        let mut ideals = Vec::new();
        for i in 0..=k {
            if let Some(id) = grid.v_id[i][j] {
                if let Some(&value) = grid.ideal.get(id) {
                    ideals.push(value);
                }
            }
        }
        if best_side_count(&ideals)? == 0 {
            vertical = false;
            break;
        }
    }
    Ok((horizontal, vertical))
}

/// Count the tile sides whose solved subdivision count is zero.
fn zero_sides(grid: &Grid, counts: &[usize]) -> Result<usize, String> {
    let mut zeros = 0usize;
    for (index, region) in grid.regions.iter().enumerate() {
        let mut horizontal = 0usize;
        let mut vertical = 0usize;
        for &edge in &region.sides[0] {
            horizontal += *counts.get(edge).ok_or_else(|| {
                format!("simons_graded: region {index} references edge {edge} out of range")
            })?;
        }
        for &edge in &region.sides[1] {
            vertical += *counts.get(edge).ok_or_else(|| {
                format!("simons_graded: region {index} references edge {edge} out of range")
            })?;
        }
        if horizontal == 0 || vertical == 0 {
            zeros += 1;
        }
    }
    Ok(zeros)
}

/// One solved grid.
struct Solved {
    grid: Grid,
    counts: Vec<usize>,
    tiles: (usize, usize),
    requested: (usize, usize),
    coarsened: bool,
}

/// Build and solve a grid at one tile count.
fn try_tiles<F>(
    domain: &Domain,
    sigma: &F,
    requested: (usize, usize),
    candidate: (usize, usize),
) -> Result<Option<Solved>, String>
where
    F: Fn(f64, f64) -> f64,
{
    let shape = shape(domain)
        .ok_or_else(|| "simons_graded: the domain is not a graded rectangle".to_string())?;
    let bounds = shape.bounds();
    let (k, l) = candidate;
    let xs = axis_lines(
        sigma,
        Axis::X,
        bounds,
        [bounds[2], bounds[3]],
        k,
        shape.hole().map(|hole| [hole[0], hole[1]]),
    )?;
    let ys = axis_lines(
        sigma,
        Axis::Y,
        bounds,
        [bounds[0], bounds[1]],
        l,
        shape.hole().map(|hole| [hole[2], hole[3]]),
    )?;
    let grid = build_grid(domain, sigma, &xs, &ys)?;
    let (horizontal, vertical) = feasibility(&grid)?;
    if !(horizontal && vertical) {
        return Ok(None);
    }
    let counts = simons_qgp::solve_counts(&grid.regions, &grid.ideal)
        .map_err(|error| format!("simons_graded: the QGP count solve failed: {error}"))?;
    if zero_sides(&grid, &counts)? > 0 {
        return Ok(None);
    }
    Ok(Some(Solved {
        grid,
        counts,
        tiles: (xs.len().saturating_sub(1), ys.len().saturating_sub(1)),
        requested,
        coarsened: (xs.len().saturating_sub(1), ys.len().saturating_sub(1)) != requested,
    }))
}

/// Realize the graded tile grid and its QGP counts.
fn solve_grid<F>(domain: &Domain, sigma: &F, options: &Options) -> Result<Solved, String>
where
    F: Fn(f64, f64) -> f64,
{
    if options.tiles != (0, 0) && (options.tiles.0 > MAX_TILES || options.tiles.1 > MAX_TILES) {
        return Err(format!(
            "simons_graded: the requested tile grid ({}, {}) exceeds the \
             limit of {MAX_TILES} per direction",
            options.tiles.0, options.tiles.1
        ));
    }
    let derived = auto_tiles(domain, sigma)?;
    let requested = if options.tiles == (0, 0) {
        derived
    } else {
        options.tiles
    };

    let mut candidates: Vec<(usize, usize)> = vec![requested];
    if derived != requested {
        candidates.push(derived);
    }
    for candidate in candidates {
        if let Some(solved) = try_tiles(domain, sigma, requested, candidate)? {
            return Ok(solved);
        }
    }

    // Both the request and the derived grid collapsed: halve the offending
    // direction, the simplified Section 5.1.6 collapse.
    let mut k = derived.0;
    let mut l = derived.1;
    for _ in 0..MAX_COARSEN {
        if k == 1 && l == 1 {
            break;
        }
        if let Some(solved) = try_tiles(domain, sigma, requested, (k, l))? {
            return Ok(solved);
        }
        let next_k = if k > 1 { (k / 2).max(1) } else { 1 };
        let next_l = if l > 1 { (l / 2).max(1) } else { 1 };
        if next_k == k && next_l == l {
            break;
        }
        k = next_k;
        l = next_l;
    }
    Err(format!(
        "simons_graded: no tile grid with positive QGP counts was found \
         (requested ({}, {}), derived ({}, {}))",
        requested.0, requested.1, derived.0, derived.1
    ))
}

// ---------------------------------------------------------------------------
// Coons tiles and welding
// ---------------------------------------------------------------------------

/// Node welding by coordinates, with a tolerance relative to the domain.
struct Welder {
    tolerance: f64,
    points: Vec<Node>,
    buckets: HashMap<(i64, i64), Vec<usize>>,
}

impl Welder {
    fn new(scale: f64) -> Welder {
        Welder {
            tolerance: WELD_RELATIVE * scale.max(1.0),
            points: Vec::new(),
            buckets: HashMap::new(),
        }
    }

    fn key(&self, point: Node) -> (i64, i64) {
        let x = (point[0] / self.tolerance).round();
        let y = (point[1] / self.tolerance).round();
        (x as i64, y as i64)
    }

    fn add(&mut self, point: Node) -> usize {
        let (key_x, key_y) = self.key(point);
        let tolerance_squared = self.tolerance * self.tolerance;
        for dx in -1i64..=1 {
            for dy in -1i64..=1 {
                let key = (key_x + dx, key_y + dy);
                if let Some(ids) = self.buckets.get(&key) {
                    for &id in ids {
                        if let Some(existing) = self.points.get(id) {
                            let difference_x = existing[0] - point[0];
                            let difference_y = existing[1] - point[1];
                            if difference_x * difference_x + difference_y * difference_y
                                <= tolerance_squared
                            {
                                return id;
                            }
                        }
                    }
                }
            }
        }
        let id = self.points.len();
        self.points.push(point);
        self.buckets.entry((key_x, key_y)).or_default().push(id);
        id
    }
}

/// Coons-mesh every tile and weld the shared sides by coordinates.
fn mesh_tiles(solved: &Solved, scale: f64) -> Result<Mesh, String> {
    let grid = &solved.grid;
    let mut welder = Welder::new(scale);
    let mut cells: Vec<Cell> = Vec::new();
    for &(i, j) in &grid.tiles {
        let bottom = grid.h_id[i][j]
            .ok_or_else(|| "simons_graded: a tile is missing its bottom edge".to_string())?;
        let left = grid.v_id[i][j]
            .ok_or_else(|| "simons_graded: a tile is missing its left edge".to_string())?;
        let n_u = *solved
            .counts
            .get(bottom)
            .ok_or_else(|| "simons_graded: the bottom edge has no count".to_string())?;
        let n_v = *solved
            .counts
            .get(left)
            .ok_or_else(|| "simons_graded: the left edge has no count".to_string())?;
        if n_u == 0 || n_v == 0 {
            return Err(format!(
                "simons_graded: tile ({i}, {j}) has a zero subdivision count"
            ));
        }
        let x0 = *grid
            .xs
            .get(i)
            .ok_or_else(|| "simons_graded: tile column out of range".to_string())?;
        let x1 = *grid
            .xs
            .get(i + 1)
            .ok_or_else(|| "simons_graded: tile column out of range".to_string())?;
        let y0 = *grid
            .ys
            .get(j)
            .ok_or_else(|| "simons_graded: tile row out of range".to_string())?;
        let y1 = *grid
            .ys
            .get(j + 1)
            .ok_or_else(|| "simons_graded: tile row out of range".to_string())?;
        let curves = BoundaryCurves::from_corners([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]);
        let tile = simons_qgp::mesh_tile_curves(&curves, n_u, n_v)
            .map_err(|error| format!("simons_graded: tile ({i}, {j}): {error}"))?;
        let mut local = Vec::with_capacity(tile.points.len());
        for point in &tile.points {
            local.push(welder.add([point[0], point[1], 0.0]));
        }
        for cell in &tile.cells {
            if cell.nodes.len() != 4 {
                return Err(format!(
                    "simons_graded: tile ({i}, {j}) produced a non-quad cell"
                ));
            }
            let a = *local
                .get(cell.nodes[0])
                .ok_or_else(|| "simons_graded: a tile node is out of range".to_string())?;
            let b = *local
                .get(cell.nodes[1])
                .ok_or_else(|| "simons_graded: a tile node is out of range".to_string())?;
            let c = *local
                .get(cell.nodes[2])
                .ok_or_else(|| "simons_graded: a tile node is out of range".to_string())?;
            let d = *local
                .get(cell.nodes[3])
                .ok_or_else(|| "simons_graded: a tile node is out of range".to_string())?;
            cells.push(Cell::quad4([a, b, c, d]));
        }
    }
    let mesh = Mesh::new(welder.points, cells);
    mesh.validate()
        .map_err(|error| format!("simons_graded: the welded mesh is invalid: {error}"))?;
    Ok(mesh)
}

/// The domain diagonal, used as the welding scale.
fn domain_scale(shape: Shape) -> f64 {
    let [min_x, max_x, min_y, max_y] = shape.bounds();
    let width = max_x - min_x;
    let height = max_y - min_y;
    (width * width + height * height).sqrt()
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Mesh a supported domain with the graded tile-grid QGP path.
///
/// `sigma` is evaluated at the tile corners, at the metric samples and at the
/// quad centroids of [`size_stats`]; it must return a positive, finite value.
/// Both the requested and the effective tile grid are returned in [`Info`].
pub fn mesh<F>(domain: &Domain, options: &Options, sigma: &F) -> Result<(Mesh, Info), String>
where
    F: Fn(f64, f64) -> f64,
{
    let shape = shape(domain)
        .ok_or_else(|| "simons_graded: the domain is not a graded rectangle".to_string())?;
    let solved = solve_grid(domain, sigma, options)?;
    let mesh = mesh_tiles(&solved, domain_scale(shape))?;
    let gate = crate::verify::gate(&mesh);
    if gate.triangles != 0 || gate.inverted != 0 || gate.overlapping != 0 || gate.cracks != 0 {
        return Err(format!(
            "simons_graded: the welded mesh failed the topology check: {}",
            gate.reason()
                .unwrap_or_else(|| "unknown reason".to_string())
        ));
    }
    let min_count = solved.counts.iter().copied().min().unwrap_or(0);
    let info = Info {
        requested_tiles: solved.requested,
        tiles: solved.tiles,
        coarsened: solved.coarsened,
        regions: solved.grid.regions.len(),
        edges: solved.grid.ideal.len(),
        min_count,
        cells: mesh.cells.len(),
    };
    Ok((mesh, info))
}

/// The size accuracy of a quad mesh against an arbitrary size function.
pub fn size_stats<F>(mesh: &Mesh, sigma: &F) -> Result<SizeStats, String>
where
    F: Fn(f64, f64) -> f64,
{
    let mut total_ratio = 0.0;
    let mut worst = 1.0f64;
    let mut count = 0usize;
    let mut sum_size = 0.0;
    let mut sum_target = 0.0;
    let mut sum_size_squared = 0.0;
    let mut sum_target_squared = 0.0;
    let mut sum_product = 0.0;
    for (index, cell) in mesh.cells.iter().enumerate() {
        if cell.kind != crate::mesh::Kind::Quad4 {
            continue;
        }
        let area = quality::volume(cell, &mesh.points).abs();
        if !area.is_finite() || area <= 0.0 {
            return Err(format!(
                "simons_graded: quad {index} has non-positive area {area}"
            ));
        }
        let mut centre_x = 0.0;
        let mut centre_y = 0.0;
        for &node in &cell.nodes {
            let point = mesh
                .points
                .get(node)
                .ok_or_else(|| format!("simons_graded: quad {index} references node {node}"))?;
            centre_x += point[0];
            centre_y += point[1];
        }
        let nodes = cell.nodes.len();
        if nodes == 0 {
            return Err(format!("simons_graded: quad {index} has no nodes"));
        }
        centre_x /= nodes as f64;
        centre_y /= nodes as f64;
        let target = sigma(centre_x, centre_y);
        if !target.is_finite() || target <= 0.0 {
            return Err(format!(
                "simons_graded: the size function returned {target} at quad \
                 {index}'s centroid"
            ));
        }
        let size = area.sqrt();
        let ratio = (size / target).max(target / size);
        if !ratio.is_finite() {
            return Err(format!(
                "simons_graded: quad {index} has a non-finite size ratio"
            ));
        }
        total_ratio += ratio;
        worst = worst.max(ratio);
        sum_size += size;
        sum_target += target;
        sum_size_squared += size * size;
        sum_target_squared += target * target;
        sum_product += size * target;
        count += 1;
    }
    if count == 0 {
        return Err("simons_graded: the mesh has no quad to measure".to_string());
    }
    let count_f = count as f64;
    let mean = total_ratio / count_f;
    let covariance = sum_product / count_f - (sum_size / count_f) * (sum_target / count_f);
    let variance_size = (sum_size_squared / count_f - (sum_size / count_f).powi(2)).max(0.0);
    let variance_target = (sum_target_squared / count_f - (sum_target / count_f).powi(2)).max(0.0);
    let denominator = (variance_size * variance_target).sqrt();
    let correlation = if denominator > 0.0 {
        (covariance / denominator).clamp(-1.0, 1.0)
    } else {
        0.0
    };
    Ok(SizeStats {
        mean,
        worst,
        correlation,
    })
}

// ---------------------------------------------------------------------------
// Demo numbers
// ---------------------------------------------------------------------------

fn measure_case<F>(domain: &Domain, options: &Options, sigma: &F) -> Result<DemoCase, String>
where
    F: Fn(f64, f64) -> f64,
{
    let (mesh, info) = mesh(domain, options, sigma)?;
    let stats = size_stats(&mesh, sigma)?;
    Ok(DemoCase {
        requested_tiles: info.requested_tiles,
        tiles: info.tiles,
        coarsened: info.coarsened,
        quads: info.cells,
        stats,
    })
}

/// Run the three measured demo cases and print their numbers.
pub fn demo() -> Result<DemoNumbers, String> {
    let domain = Domain::rectangle(10.0, 10.0)
        .map_err(|error| format!("simons_graded demo: the domain is invalid: {error}"))?;

    let linear = measure_case(&domain, &Options::default(), &|x, _y: f64| 1.0 + 0.5 * x)?;
    println!(
        "simons_graded linear 10x10 (sigma = 1 + 0.5x): requested={:?} \
         tiles={:?} coarsened={} quads={} size mean={:.3} worst={:.3} \
         correlation={:.3}",
        linear.requested_tiles,
        linear.tiles,
        linear.coarsened,
        linear.quads,
        linear.stats.mean,
        linear.stats.worst,
        linear.stats.correlation,
    );

    let two_d_auto = measure_case(&domain, &Options::default(), &|x, y| {
        1.0 + 0.3 * x + 0.3 * y
    })?;
    println!(
        "simons_graded 2D 10x10 (sigma = 1 + 0.3x + 0.3y, derived): \
         requested={:?} tiles={:?} coarsened={} quads={} size mean={:.3} \
         worst={:.3} correlation={:.3}",
        two_d_auto.requested_tiles,
        two_d_auto.tiles,
        two_d_auto.coarsened,
        two_d_auto.quads,
        two_d_auto.stats.mean,
        two_d_auto.stats.worst,
        two_d_auto.stats.correlation,
    );

    let two_d_eight = measure_case(&domain, &Options { tiles: (8, 8) }, &|x, y| {
        1.0 + 0.3 * x + 0.3 * y
    })?;
    println!(
        "simons_graded 2D 10x10 (sigma = 1 + 0.3x + 0.3y, requested 8x8): \
         requested={:?} tiles={:?} coarsened={} quads={} size mean={:.3} \
         worst={:.3} correlation={:.3}",
        two_d_eight.requested_tiles,
        two_d_eight.tiles,
        two_d_eight.coarsened,
        two_d_eight.quads,
        two_d_eight.stats.mean,
        two_d_eight.stats.worst,
        two_d_eight.stats.correlation,
    );

    Ok(DemoNumbers {
        linear,
        two_d_auto,
        two_d_eight,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::verify;

    fn gate_of(mesh: &Mesh) -> verify::Gate {
        verify::gate(mesh)
    }

    #[test]
    fn shape_recognises_a_rectangle_and_rejects_other_domains() {
        let rectangle = Domain::rectangle(10.0, 4.0).unwrap();
        assert_eq!(
            shape(&rectangle),
            Some(Shape::Rectangle([0.0, 10.0, 0.0, 4.0]))
        );
        let polygon = Domain::new(
            vec![[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [1.0, 4.0], [0.0, 3.0]],
            Vec::new(),
        )
        .unwrap();
        assert_eq!(shape(&polygon), None);
        let plate = Domain::new(
            vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
            vec![vec![[4.0, 4.0], [6.0, 4.0], [6.0, 6.0], [4.0, 6.0]]],
        )
        .unwrap();
        assert_eq!(
            shape(&plate),
            Some(Shape::RectangleWithHole {
                outer: [0.0, 10.0, 0.0, 10.0],
                hole: [4.0, 6.0, 4.0, 6.0],
            })
        );
    }

    #[test]
    fn auto_tiles_follow_the_size_variation() {
        let domain = Domain::rectangle(10.0, 10.0).unwrap();
        assert_eq!(auto_tiles(&domain, &|x, _y| 1.0 + 0.5 * x).unwrap(), (4, 4));
        assert_eq!(
            auto_tiles(&domain, &|x, y| 1.0 + 0.3 * x + 0.3 * y).unwrap(),
            (3, 3)
        );
    }

    #[test]
    fn the_graded_linear_square_hits_the_size_target() {
        let domain = Domain::rectangle(10.0, 10.0).unwrap();
        let (mesh, info) = mesh(&domain, &Options::default(), &|x, _y| 1.0 + 0.5 * x).unwrap();
        let gate = gate_of(&mesh);
        let stats = size_stats(&mesh, &|x, _y| 1.0 + 0.5 * x).unwrap();
        println!(
            "graded linear: requested={:?} tiles={:?} coarsened={} regions={} \
             edges={} min_count={} quads={} SJ={:.4} angle={:.2} size \
             mean={:.3} worst={:.3} correlation={:.3}",
            info.requested_tiles,
            info.tiles,
            info.coarsened,
            info.regions,
            info.edges,
            info.min_count,
            info.cells,
            gate.min_scaled_jacobian,
            gate.min_corner_angle_deg,
            stats.mean,
            stats.worst,
            stats.correlation,
        );
        assert_eq!(gate.triangles, 0, "not all-quad");
        assert_eq!(gate.inverted, 0, "inverted quads");
        assert_eq!(gate.cracks, 0, "cracks: {}", gate.cracks);
        assert!(gate.passes(), "gate failed: {:?}", gate.reason());
        assert!(stats.worst < 1.8, "worst size ratio {}", stats.worst);
        assert!(
            stats.mean > 1.0 - 1e-12 && stats.mean <= stats.worst + 1e-12,
            "size ratios: mean {} worst {}",
            stats.mean,
            stats.worst
        );
        assert!(
            stats.correlation > 0.9,
            "the size trend is not visible: correlation {}",
            stats.correlation
        );
        assert_eq!(info.tiles, (4, 4));
        assert!(!info.coarsened);
    }

    #[test]
    fn the_two_dimensional_gradient_is_graded() {
        let domain = Domain::rectangle(10.0, 10.0).unwrap();
        let sigma = |x: f64, y: f64| 1.0 + 0.3 * x + 0.3 * y;
        let (auto_mesh, auto_info) = mesh(&domain, &Options::default(), &sigma).unwrap();
        let gate = gate_of(&auto_mesh);
        let stats = size_stats(&auto_mesh, &sigma).unwrap();
        println!(
            "graded 2D: requested={:?} tiles={:?} coarsened={} quads={} \
             SJ={:.4} size mean={:.3} worst={:.3} correlation={:.3}",
            auto_info.requested_tiles,
            auto_info.tiles,
            auto_info.coarsened,
            auto_info.cells,
            gate.min_scaled_jacobian,
            stats.mean,
            stats.worst,
            stats.correlation,
        );
        assert_eq!(gate.triangles, 0);
        assert_eq!(gate.cracks, 0);
        assert!(gate.passes(), "gate failed: {:?}", gate.reason());
        assert!(stats.worst < 1.8, "worst size ratio {}", stats.worst);
        assert!(stats.correlation > 0.9);
        assert_eq!(auto_info.tiles, (3, 3));

        // The explicit 8 x 8 request is physically infeasible: the expected
        // number of quads is only about 9, so 64 tiles cannot each carry one.
        // The path reports the requested grid and meshes the derived one.
        let (requested_mesh, requested_info) =
            mesh(&domain, &Options { tiles: (8, 8) }, &sigma).unwrap();
        let gate = gate_of(&requested_mesh);
        let stats = size_stats(&requested_mesh, &sigma).unwrap();
        println!(
            "graded 2D (requested 8x8): effective={:?} coarsened={} quads={} \
             size mean={:.3} worst={:.3} correlation={:.3}",
            requested_info.tiles,
            requested_info.coarsened,
            requested_info.cells,
            stats.mean,
            stats.worst,
            stats.correlation,
        );
        assert_eq!(requested_info.requested_tiles, (8, 8));
        assert!(requested_info.coarsened, "the 8 x 8 grid should collapse");
        assert_eq!(gate.cracks, 0);
        assert!(gate.passes(), "gate failed: {:?}", gate.reason());
        assert!(stats.worst < 2.0, "worst size ratio {}", stats.worst);
    }

    #[test]
    fn increasing_the_tile_count_worsens_then_collapses() {
        let domain = Domain::rectangle(10.0, 10.0).unwrap();
        let sigma = |x: f64, _y: f64| 1.0 + 0.5 * x;
        let requests = [(4usize, 4usize), (5, 4), (6, 4), (8, 4), (16, 4)];
        let mut measured: Vec<(usize, usize, (usize, usize), bool, f64)> = Vec::new();
        for &(k, l) in &requests {
            let (mesh, info) = mesh(&domain, &Options { tiles: (k, l) }, &sigma).unwrap();
            let gate = gate_of(&mesh);
            let stats = size_stats(&mesh, &sigma).unwrap();
            println!(
                "tile trend: requested=({k}, {l}) effective={:?} \
                 coarsened={} quads={} size_ratio_worst={:.3}",
                info.tiles, info.coarsened, info.cells, stats.worst,
            );
            assert_eq!(gate.triangles, 0);
            assert_eq!(gate.cracks, 0);
            assert!(gate.passes(), "gate failed: {:?}", gate.reason());
            measured.push((k, l, info.tiles, info.coarsened, stats.worst));
        }
        assert_eq!(measured.len(), requests.len());
        let (_, _, first_tiles, first_coarsened, first_worst) = measured[0];
        assert_eq!(first_tiles, (4, 4));
        assert!(!first_coarsened);
        assert!(first_worst < 1.8);
        // The stepwise counts get coarser as the tile gets thinner: 5 columns
        // are worse than 4 and 6 columns are worse still, above the target.
        let (_, _, _, _, second_worst) = measured[1];
        let (_, _, _, _, third_worst) = measured[2];
        assert!(second_worst > first_worst);
        assert!(third_worst > second_worst);
        assert!(third_worst > 1.8);
        // Beyond that the grid is infeasible and the derived 4 x 4 is meshed.
        for entry in measured.iter().skip(3) {
            assert_eq!(entry.2, (4, 4));
            assert!(entry.3);
        }
    }

    #[test]
    fn a_rectangle_with_a_rectangular_hole_is_graded() {
        let domain = Domain::new(
            vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
            vec![vec![[4.0, 4.0], [6.0, 4.0], [6.0, 6.0], [4.0, 6.0]]],
        )
        .unwrap();
        let sigma = |x: f64, y: f64| 1.0 + 0.2 * x + 0.1 * y;
        let (mesh, info) = mesh(&domain, &Options::default(), &sigma).unwrap();
        let gate = gate_of(&mesh);
        let stats = size_stats(&mesh, &sigma).unwrap();
        println!(
            "graded plate: tiles={:?} coarsened={} quads={} SJ={:.4} \
             cracks={} size mean={:.3} worst={:.3}",
            info.tiles,
            info.coarsened,
            info.cells,
            gate.min_scaled_jacobian,
            gate.cracks,
            stats.mean,
            stats.worst,
        );
        assert_eq!(gate.triangles, 0);
        assert_eq!(gate.cracks, 0, "cracks around the hole: {}", gate.cracks);
        assert!(gate.passes(), "gate failed: {:?}", gate.reason());
        assert!(stats.worst < 2.5, "worst size ratio {}", stats.worst);
    }

    #[test]
    fn a_constant_size_is_uniform_and_uses_a_positive_count() {
        let domain = Domain::rectangle(10.0, 10.0).unwrap();
        let (mesh, info) = mesh(&domain, &Options::default(), &|_, _| 2.0).unwrap();
        let gate = gate_of(&mesh);
        let stats = size_stats(&mesh, &|_, _| 2.0).unwrap();
        assert!(gate.passes(), "gate failed: {:?}", gate.reason());
        assert!(info.min_count >= 1);
        assert!(stats.worst < 1.2, "worst size ratio {}", stats.worst);
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        let domain = Domain::rectangle(10.0, 10.0).unwrap();
        assert!(mesh(&domain, &Options::default(), &|_, _| 0.0).is_err());
        assert!(mesh(&domain, &Options::default(), &|_, _| f64::NAN).is_err());
        assert!(mesh(&domain, &Options { tiles: (999, 999) }, &|_, _| 1.0).is_err());
        let polygon = Domain::new(
            vec![[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [1.0, 4.0], [0.0, 3.0]],
            Vec::new(),
        )
        .unwrap();
        assert!(mesh(&polygon, &Options::default(), &|_, _| 1.0).is_err());
    }

    #[test]
    fn the_demo_numbers_are_measured() {
        let numbers = demo().unwrap();
        assert!(numbers.linear.stats.worst < 1.8);
        assert!(numbers.linear.stats.correlation > 0.9);
        assert!(numbers.two_d_auto.stats.worst < 1.8);
        assert!(numbers.two_d_eight.coarsened);
    }
}
