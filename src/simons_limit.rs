//! Limit-cycle detection and removal for a 4-RoSy cross field.
//!
//! Implements Simons (2024), *Size-Driven All-Quad Meshing for Geographic
//! Data*, Section 4.12 ("Removable limit cycles"), together with the
//! streamline tracing of Section 4.3 and the discussion of cycles in
//! Sections 4.8-4.9.
//!
//! # What is implemented faithfully
//!
//! - **Cycles and limit cycles (Section 4.12).** A *cycle* is a closed
//!   streamline; a *limit cycle* is a cycle into which (or away from which)
//!   neighbouring streamlines spiral. [`detect_limit_cycles`] traces
//!   streamlines with classical RK4 on the piecewise-linear, complex-valued
//!   field of Chapter 3, and accepts a closed orbit when the path comes back
//!   within a small tolerance of an earlier point with tangent continuity.
//!   The orbit is then classified with two offset neighbour traces: if the
//!   neighbours on **both** sides approach the orbit it is a stable limit
//!   cycle, if they recede on both sides it is unstable, and if the distance
//!   stays roughly constant the orbit is an ordinary (allowed) cycle and is
//!   not reported. The distinction is exactly the one drawn in the thesis:
//!   "A stable limit cycle gamma is one in which, as we traverse gamma, the
//!   nearby streamlines on either side either all approach or all move away
//!   from gamma (depending on our direction of travel on gamma)."
//! - **The removal rule (Section 4.12).** The thesis removes a stable simple
//!   limit cycle by "smoothly add[ing] a twist of pi/2 to alpha in the
//!   neighborhood of gamma", so that each streamline spiralling into the
//!   neighbourhood is connected to a streamline crossing gamma and vice versa,
//!   without adding singular points. [`avoid_limit_cycles`] implements that
//!   rule on the vertex angles: the angle gets `pi/2` inside the cycle and `0`
//!   outside, with a smooth (`C^1`) transition across an annulus of the given
//!   width around it. Because a cross field is unchanged by a global `pi/2`
//!   rotation, only the transition annulus changes the field, exactly the
//!   "region in which the swap occurs" that the thesis says "can be made
//!   arbitrarily thin".
//! - **Size scaling.** The integration step, the closure tolerance and the
//!   neighbour offsets are all proportional to the mean size function, so the
//!   detector is scale-free: an element of size `s` behaves the same as one of
//!   size `1`.
//!
//! # How the field is evaluated
//!
//! `alpha` is given per vertex modulo `pi/2`, as returned by
//! [`crate::simons_godf::solve`]. Inside a triangle the complex value
//! `u = exp(4 i alpha)` is interpolated linearly (the Chapter 3
//! representation), and the angle is `arg(u)/4`. This is the same
//! barycentric interpolation as [`crate::simons_godf::interp_angle`], so a
//! `pi/2` jump in the stored angles does not create an artificial
//! discontinuity. At a singular point, where the interpolated `u` vanishes,
//! the direction is undefined and a trace stops there.
//!
//! # Deviation: discrete removal
//!
//! The thesis lets the transition annulus be arbitrarily thin. Here it is
//! bounded below by the background mesh: [`avoid_limit_cycles`] rotates only
//! the mesh vertices, so a width of a couple of elements is needed for the
//! discrete field to remain resolvable. A very small width concentrates the
//! `grad(alpha)` in one triangle and will be poorly represented by the
//! piecewise-linear field. The function reports the field it produced; it
//! makes no promise that the discrete re-trace finds no limit cycle, and
//! [`demo`] measures what actually happens on its example.
//!
//! # Simplifications and limits
//!
//! - Detection is sampling-based: a regular `SEED_GRID x SEED_GRID` grid of
//!   seeds, traced along both orientations of both orthogonal families. A
//!   limit cycle with a basin smaller than a grid cell can be missed.
//! - Closure uses a fixed tolerance (`h/2`); a very slowly converging spiral
//!   needs many turns and may exceed the step cap.
//! - Stable limit cycles are the ones that matter (the thesis notes that
//!   unstable cycles disappear under small smooth perturbations); a closed
//!   loop is returned for an unstable cycle only when a seed happens to land
//!   close enough for its own trace to close. The public [`LimitCycle`] does
//!   not expose the stable/unstable classification, only the geometry.
//! - The analysis is planar: the `z` coordinate is ignored. A 3-D surface
//!   version and the motorcycle-graph "nautilus" handling of Section 5.1.4
//!   are out of scope.
//! - The removal is applied to the angle field as a post-process. It is not
//!   followed by a re-solve of the Chapter 3 eigenproblem, and it does not
//!   update sigma/tau of Chapter 4.
//!
//! # Public API
//!
//! ```
//! # use pedroso_mesher::simons_limit::{detect_limit_cycles, avoid_limit_cycles};
//! // A constant field on a small square: a straight streamline, no cycle.
//! let points = vec![
//!     [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
//!     [0.0, 1.0, 0.0], [1.0, 1.0, 0.0],
//! ];
//! let tris = vec![[0, 1, 3], [0, 3, 2]];
//! let alpha = vec![0.0; 4];
//! let size = vec![1.0; 4];
//! let cycles = detect_limit_cycles(&points, &tris, &alpha, &size).unwrap();
//! assert!(cycles.is_empty());
//! // With no cycle there is nothing to twist, so the field is returned as is.
//! let twisted = avoid_limit_cycles(&points, &tris, &alpha, &cycles, 0.5).unwrap();
//! assert_eq!(twisted, alpha);
//! ```

use std::collections::HashMap;
use std::f64::consts::FRAC_PI_2;

/// Seeds per axis of the seeding grid.
const SEED_GRID: usize = 6;
/// Hard cap on the number of RK4 steps of one trace.
const MAX_TRACE_STEPS: usize = 1800;
/// Minimum number of steps of arc between two points of one orbit.
const MIN_LOOP_STEPS: usize = 10;
/// Closure tolerance as a fraction of the integration step.
const CLOSURE_FRACTION: f64 = 0.5;
/// Minimum tangent cosine at closure (15 degrees).
const TANGENT_COS: f64 = 0.965_925_826_289_068_3;
/// Neighbour offset in units of the integration step.
const NEIGHBOUR_OFFSET: f64 = 1.0;
/// Number of sample points of a cycle used for the neighbour classification.
const CLASSIFY_SAMPLES: usize = 4;
/// Factor by which a neighbour must approach to count as converging.
const CONVERGE_FACTOR: f64 = 0.5;
/// Factor by which a neighbour must recede to count as diverging.
const DIVERGE_FACTOR: f64 = 1.75;
/// Mean-size fraction used as the RK4 step.
const STEP_FRACTION: f64 = 0.25;
/// A triangle with a smaller absolute double area is degenerate.
const DEGENERATE_AREA2: f64 = 1.0e-14;
/// Barycentric tolerance for point location.
const BARY_TOL: f64 = 1.0e-10;

/// A closed streamline of the cross field.
///
/// `polyline` is the traced orbit without a duplicated endpoint: the last
/// point is within the closure tolerance of the first one. `closed` says
/// whether the polyline closes on itself. This implementation only emits
/// closed orbits, so it is `true` for every value returned by
/// [`detect_limit_cycles`]; the flag is kept in the API so a caller can
/// distinguish a closed orbit from a future spiral-only report.
#[derive(Clone, Debug, PartialEq)]
pub struct LimitCycle {
    /// Vertices of the traced orbit, in order.
    pub polyline: Vec<[f64; 2]>,
    /// Whether the final point returns to the first one (always `true` here).
    pub closed: bool,
}

/// The stable/unstable classification of a traced orbit.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Stable,
    Unstable,
    Ordinary,
}

// ---------------------------------------------------------------------------
// Small vector helpers
// ---------------------------------------------------------------------------

fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn dot(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

fn cross(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

fn norm(a: [f64; 2]) -> f64 {
    (a[0] * a[0] + a[1] * a[1]).sqrt()
}

fn normalize(a: [f64; 2]) -> Option<[f64; 2]> {
    let n = norm(a);
    if n.is_finite() && n > 1.0e-15 {
        Some([a[0] / n, a[1] / n])
    } else {
        None
    }
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    norm(sub(a, b))
}

// ---------------------------------------------------------------------------
// Background triangle mesh with piecewise-linear u = exp(4 i alpha)
// ---------------------------------------------------------------------------

/// Planar triangle tessellation carrying the cross-field angle.
struct Field {
    pts: Vec<[f64; 2]>,
    tris: Vec<[usize; 3]>,
    tri_lo: Vec<[f64; 2]>,
    tri_hi: Vec<[f64; 2]>,
    den: Vec<f64>,
    alpha: Vec<f64>,
    /// `[xmin, ymin, xmax, ymax]`.
    bbox: [f64; 4],
}

impl Field {
    fn build(
        points: &[[f64; 3]],
        tris: &[[usize; 3]],
        alpha: &[f64],
        size: &[f64],
    ) -> Result<Field, String> {
        let n = points.len();
        if n < 3 {
            return Err("limit-cycle detection needs at least 3 points".to_string());
        }
        if alpha.len() != n {
            return Err(format!(
                "alpha has {} values but there are {n} points",
                alpha.len()
            ));
        }
        if size.len() != n {
            return Err(format!(
                "size has {} values but there are {n} points",
                size.len()
            ));
        }
        if tris.is_empty() {
            return Err("limit-cycle detection needs at least one triangle".to_string());
        }
        let mut pts = Vec::with_capacity(n);
        let mut bbox = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        for (i, p) in points.iter().enumerate() {
            if !p[0].is_finite() || !p[1].is_finite() {
                return Err(format!("point {i} has a non-finite x or y coordinate"));
            }
            pts.push([p[0], p[1]]);
            bbox[0] = bbox[0].min(p[0]);
            bbox[1] = bbox[1].min(p[1]);
            bbox[2] = bbox[2].max(p[0]);
            bbox[3] = bbox[3].max(p[1]);
        }
        if bbox[2] <= bbox[0] || bbox[3] <= bbox[1] {
            return Err("the input points span a degenerate bounding box".to_string());
        }
        for (i, a) in alpha.iter().enumerate() {
            if !a.is_finite() || a.abs() > 1.0e6 {
                return Err(format!("alpha[{i}] is not a finite usable angle"));
            }
        }
        for (i, s) in size.iter().enumerate() {
            if !s.is_finite() || *s <= 0.0 {
                return Err(format!("size[{i}] must be finite and positive"));
            }
        }

        let mut valid = 0usize;
        let mut tri_lo = Vec::with_capacity(tris.len());
        let mut tri_hi = Vec::with_capacity(tris.len());
        let mut den = Vec::with_capacity(tris.len());
        for (ti, t) in tris.iter().enumerate() {
            if t[0] >= n || t[1] >= n || t[2] >= n {
                return Err(format!(
                    "triangle {ti} references a point that does not exist"
                ));
            }
            let p0 = pts[t[0]];
            let p1 = pts[t[1]];
            let p2 = pts[t[2]];
            let d = cross(sub(p1, p0), sub(p2, p0));
            tri_lo.push([p0[0].min(p1[0]).min(p2[0]), p0[1].min(p1[1]).min(p2[1])]);
            tri_hi.push([p0[0].max(p1[0]).max(p2[0]), p0[1].max(p1[1]).max(p2[1])]);
            den.push(d);
            if d.abs() > DEGENERATE_AREA2 {
                valid += 1;
            }
        }
        if valid == 0 {
            return Err("all triangles are degenerate".to_string());
        }
        Ok(Field {
            pts,
            tris: tris.to_vec(),
            tri_lo,
            tri_hi,
            den,
            alpha: alpha.to_vec(),
            bbox,
        })
    }

    fn diagonal(&self) -> f64 {
        let [x0, y0, x1, y1] = self.bbox;
        ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt()
    }

    fn bary_raw(&self, p: [f64; 2], ti: usize) -> Option<[f64; 3]> {
        let tri = *self.tris.get(ti)?;
        let p0 = *self.pts.get(tri[0])?;
        let p1 = *self.pts.get(tri[1])?;
        let p2 = *self.pts.get(tri[2])?;
        let d = *self.den.get(ti)?;
        if d.abs() <= DEGENERATE_AREA2 {
            return None;
        }
        let e1 = sub(p1, p0);
        let e2 = sub(p2, p0);
        let w = sub(p, p0);
        let l1 = cross(w, e2) / d;
        let l2 = cross(e1, w) / d;
        Some([1.0 - l1 - l2, l1, l2])
    }

    /// Locate `p`; returns the triangle and clamped barycentric coordinates.
    fn locate(&self, p: [f64; 2]) -> Option<(usize, [f64; 3])> {
        let mut best: Option<(usize, [f64; 3], f64)> = None;
        for ti in 0..self.tris.len() {
            let lo = *self.tri_lo.get(ti)?;
            let hi = *self.tri_hi.get(ti)?;
            if p[0] < lo[0] || p[1] < lo[1] || p[0] > hi[0] || p[1] > hi[1] {
                continue;
            }
            let Some(l) = self.bary_raw(p, ti) else {
                continue;
            };
            let min_l = l[0].min(l[1]).min(l[2]);
            if min_l >= -BARY_TOL && best.as_ref().is_none_or(|(_, _, bm)| min_l > *bm) {
                best = Some((ti, l, min_l));
            }
        }
        best.map(|(ti, l, _)| (ti, clamp_bary(l)))
    }

    /// Cross-field angle at `p`, in `[0, pi/2)`, or `None` at a singularity.
    fn alpha_at(&self, p: [f64; 2]) -> Option<f64> {
        let (ti, w) = self.locate(p)?;
        let tri = *self.tris.get(ti)?;
        let mut re = 0.0;
        let mut im = 0.0;
        for k in 0..3 {
            let a = *self.alpha.get(tri[k])?;
            re += w[k] * (4.0 * a).cos();
            im += w[k] * (4.0 * a).sin();
        }
        if re * re + im * im <= 1.0e-24 {
            return None;
        }
        let angle = im.atan2(re) / 4.0;
        Some(angle.rem_euclid(FRAC_PI_2))
    }

    /// Direction of the branch of `alpha_at` closest to `prev`.
    fn direction_at(&self, p: [f64; 2], prev: [f64; 2]) -> Option<[f64; 2]> {
        let a = self.alpha_at(p)?;
        let mut best = [0.0f64; 2];
        let mut best_dot = -2.0f64;
        for k in 0..4 {
            let theta = a + f64::from(k) * FRAC_PI_2;
            let d = [theta.cos(), theta.sin()];
            let dp = dot(d, prev);
            if dp > best_dot {
                best = d;
                best_dot = dp;
            }
        }
        if best_dot > 0.0 {
            Some(best)
        } else {
            None
        }
    }

    /// One RK4 step of length `h`, continuing the branch closest to `d0`.
    fn rk4(&self, x: [f64; 2], d0: [f64; 2], h: f64) -> Option<([f64; 2], [f64; 2])> {
        let k1 = self.direction_at(x, d0)?;
        let x2 = [x[0] + 0.5 * h * k1[0], x[1] + 0.5 * h * k1[1]];
        let k2 = self.direction_at(x2, k1)?;
        let x3 = [x[0] + 0.5 * h * k2[0], x[1] + 0.5 * h * k2[1]];
        let k3 = self.direction_at(x3, k2)?;
        let x4 = [x[0] + h * k3[0], x[1] + h * k3[1]];
        let k4 = self.direction_at(x4, k3)?;
        let v = [
            (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
            (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
        ];
        let d = normalize(v)?;
        let next = [x[0] + h * d[0], x[1] + h * d[1]];
        if !next[0].is_finite() || !next[1].is_finite() {
            return None;
        }
        self.locate(next)?;
        Some((next, d))
    }

    /// Trace `steps` RK4 steps from `start` in the direction closest to `dir`.
    fn trace(
        &self,
        start: [f64; 2],
        dir: [f64; 2],
        h: f64,
        steps: usize,
    ) -> Vec<([f64; 2], [f64; 2])> {
        let mut out = Vec::with_capacity(steps.min(512) + 1);
        let mut d = match normalize(dir) {
            Some(d) => d,
            None => return out,
        };
        let mut x = start;
        out.push((x, d));
        for _ in 0..steps {
            match self.rk4(x, d, h) {
                Some((next, next_dir)) => {
                    out.push((next, next_dir));
                    x = next;
                    d = next_dir;
                }
                None => break,
            }
        }
        out
    }

    /// Trace until the path returns close to an earlier point with tangent
    /// continuity; returns the closed sub-polyline when it does.
    fn trace_until_closed(&self, start: [f64; 2], dir: [f64; 2], h: f64) -> Option<Vec<[f64; 2]>> {
        let tolerance = CLOSURE_FRACTION * h;
        let cell = 4.0 * tolerance;
        let mut hash: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
        let mut points: Vec<[f64; 2]> = Vec::new();
        let mut directions: Vec<[f64; 2]> = Vec::new();
        let mut arc: Vec<f64> = Vec::new();
        let mut d = normalize(dir)?;
        let mut x = start;
        for step in 0..MAX_TRACE_STEPS {
            points.push(x);
            directions.push(d);
            let previous_arc = arc.last().copied().unwrap_or(0.0);
            arc.push(previous_arc + h);
            if step >= MIN_LOOP_STEPS {
                if let Some(found) = self.find_recurrence(
                    &hash,
                    &points,
                    &directions,
                    &arc,
                    x,
                    step,
                    h,
                    tolerance,
                    cell,
                ) {
                    let mut polyline = points[found..].to_vec();
                    if polyline.len() >= MIN_LOOP_STEPS {
                        // Close implicitly; keep the traced points only.
                        polyline.shrink_to_fit();
                        return Some(polyline);
                    }
                }
            }
            let key = ((x[0] / cell).floor() as i64, (x[1] / cell).floor() as i64);
            hash.entry(key).or_default().push(step);
            match self.rk4(x, d, h) {
                Some((next, next_dir)) => {
                    x = next;
                    d = next_dir;
                }
                None => break,
            }
        }
        None
    }

    #[allow(clippy::too_many_arguments)]
    fn find_recurrence(
        &self,
        hash: &HashMap<(i64, i64), Vec<usize>>,
        points: &[[f64; 2]],
        directions: &[[f64; 2]],
        arc: &[f64],
        x: [f64; 2],
        current: usize,
        h: f64,
        tolerance: f64,
        cell: f64,
    ) -> Option<usize> {
        let cx = (x[0] / cell).floor() as i64;
        let cy = (x[1] / cell).floor() as i64;
        let minimum_arc = 6.0 * h;
        // Prefer the most recent qualifying point: that gives the shortest
        // orbit instead of several turns of the same spiral.
        let mut best: Option<usize> = None;
        for dx in -1..=1 {
            for dy in -1..=1 {
                let Some(indices) = hash.get(&(cx + dx, cy + dy)) else {
                    continue;
                };
                for &index in indices {
                    if index + MIN_LOOP_STEPS > current {
                        continue;
                    }
                    if arc[current] - arc[index] < minimum_arc {
                        continue;
                    }
                    let Some(&earlier) = points.get(index) else {
                        continue;
                    };
                    let dist = distance(x, earlier);
                    if dist > tolerance {
                        continue;
                    }
                    let Some(&d0) = directions.get(index) else {
                        continue;
                    };
                    if dot(d0, directions[current]) < TANGENT_COS {
                        continue;
                    }
                    if best.is_none_or(|bd| index > bd) {
                        best = Some(index);
                    }
                }
            }
        }
        best
    }
}

fn clamp_bary(l: [f64; 3]) -> [f64; 3] {
    let c = [l[0].max(0.0), l[1].max(0.0), l[2].max(0.0)];
    let sum = c[0] + c[1] + c[2];
    if sum > 0.0 && sum.is_finite() {
        [c[0] / sum, c[1] / sum, c[2] / sum]
    } else {
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    }
}

// ---------------------------------------------------------------------------
// Geometry of a closed polyline
// ---------------------------------------------------------------------------

/// Distance from `p` to the closed polyline through `poly`.
fn point_to_cycle(p: [f64; 2], poly: &[[f64; 2]]) -> f64 {
    if poly.is_empty() {
        return f64::INFINITY;
    }
    if poly.len() == 1 {
        return distance(p, poly[0]);
    }
    let mut best = f64::INFINITY;
    for i in 0..poly.len() {
        let a = poly[i];
        let b = poly[(i + 1) % poly.len()];
        best = best.min(point_to_segment(p, a, b));
    }
    best
}

fn point_to_segment(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let ab = sub(b, a);
    let squared = dot(ab, ab);
    if squared <= 1.0e-30 {
        return distance(p, a);
    }
    let t = (dot(sub(p, a), ab) / squared).clamp(0.0, 1.0);
    distance(p, [a[0] + t * ab[0], a[1] + t * ab[1]])
}

fn cycle_perimeter(poly: &[[f64; 2]]) -> f64 {
    if poly.len() < 2 {
        return 0.0;
    }
    let mut length = 0.0;
    for i in 0..poly.len() {
        length += distance(poly[i], poly[(i + 1) % poly.len()]);
    }
    length
}

/// Even-odd point-in-polygon test for the implicitly closed polyline.
fn point_in_cycle(p: [f64; 2], poly: &[[f64; 2]]) -> bool {
    if poly.len() < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let pi = poly[i];
        let pj = poly[j];
        if (pi[1] > p[1]) != (pj[1] > p[1]) {
            let t = (p[1] - pi[1]) / (pj[1] - pi[1]);
            if p[0] < pi[0] + t * (pj[0] - pi[0]) {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

fn cycle_tangent(poly: &[[f64; 2]], index: usize) -> Option<[f64; 2]> {
    let n = poly.len();
    if n < 3 {
        return None;
    }
    let previous = poly[(index + n - 1) % n];
    let next = poly[(index + 1) % n];
    normalize(sub(next, previous))
}

// ---------------------------------------------------------------------------
// Classification of a traced orbit
// ---------------------------------------------------------------------------

fn classify(field: &Field, cycle: &[[f64; 2]], h: f64) -> Kind {
    let perimeter = cycle_perimeter(cycle);
    if !perimeter.is_finite() || perimeter <= 0.0 || cycle.len() < 3 {
        return Kind::Ordinary;
    }
    let lap_steps = ((perimeter / h).round() as usize).clamp(60, 500);
    let offset = NEIGHBOUR_OFFSET * h;
    let mut converge = 0usize;
    let mut diverge = 0usize;
    let mut tested = 0usize;
    for sample in 0..CLASSIFY_SAMPLES {
        let index = (sample * cycle.len()) / CLASSIFY_SAMPLES;
        let Some(tangent) = cycle_tangent(cycle, index) else {
            continue;
        };
        let normal = [-tangent[1], tangent[0]];
        for side in [-1.0f64, 1.0] {
            let start = [
                cycle[index][0] + side * offset * normal[0],
                cycle[index][1] + side * offset * normal[1],
            ];
            if field.alpha_at(start).is_none() {
                continue;
            }
            let d0 = point_to_cycle(start, cycle);
            if !d0.is_finite() || d0 <= 1.0e-12 {
                continue;
            }
            let trace = field.trace(start, [tangent[0], tangent[1]], h, 2 * lap_steps);
            if trace.len() < 8 {
                continue;
            }
            let mut minimum = f64::INFINITY;
            let mut maximum = 0.0f64;
            for (point, _) in &trace {
                let d = point_to_cycle(*point, cycle);
                minimum = minimum.min(d);
                maximum = maximum.max(d);
            }
            if !minimum.is_finite() || !maximum.is_finite() {
                continue;
            }
            tested += 1;
            if minimum < CONVERGE_FACTOR * d0 {
                converge += 1;
            } else if maximum > DIVERGE_FACTOR * d0 {
                diverge += 1;
            }
        }
    }
    if tested < 2 {
        return Kind::Ordinary;
    }
    if converge == tested {
        Kind::Stable
    } else if diverge == tested {
        Kind::Unstable
    } else {
        Kind::Ordinary
    }
}

fn similar_cycles(a: &LimitCycle, b: &LimitCycle, h: f64) -> bool {
    if a.closed != b.closed || a.polyline.is_empty() || b.polyline.is_empty() {
        return false;
    }
    let stride = (a.polyline.len() / 8).max(1);
    let mut sum = 0.0;
    let mut count = 0usize;
    for point in a.polyline.iter().step_by(stride) {
        sum += point_to_cycle(*point, &b.polyline);
        count += 1;
    }
    count > 0 && sum / (count as f64) < 1.5 * h
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

/// A regular grid of seeds inside the tessellation.
fn seeds(field: &Field) -> Vec<[f64; 2]> {
    let [x0, y0, x1, y1] = field.bbox;
    let mut out = Vec::new();
    for j in 0..SEED_GRID {
        for i in 0..SEED_GRID {
            let fx = (i as f64 + 0.5) / SEED_GRID as f64;
            let fy = (j as f64 + 0.5) / SEED_GRID as f64;
            let p = [x0 + (x1 - x0) * fx, y0 + (y1 - y0) * fy];
            if field.alpha_at(p).is_some() {
                out.push(p);
            }
        }
    }
    if out.is_empty() {
        for t in field.tris.iter().take(4) {
            let p = [
                (field.pts[t[0]][0] + field.pts[t[1]][0] + field.pts[t[2]][0]) / 3.0,
                (field.pts[t[0]][1] + field.pts[t[1]][1] + field.pts[t[2]][1]) / 3.0,
            ];
            if field.alpha_at(p).is_some() {
                out.push(p);
            }
        }
    }
    out
}

/// Detect the limit cycles of a cross field.
///
/// `points` are the vertices of a planar triangle tessellation (`z` is
/// ignored), `tris` its triangles, `alpha` the cross-field angle at every
/// vertex (radians, modulo `pi/2`, as returned by
/// [`crate::simons_godf::solve`]), and `size` the size function at every
/// vertex. The mean size sets the RK4 step (`0.25 * mean size`), the closure
/// tolerance (`step/2`) and the offsets of the neighbour traces.
///
/// Streamlines are traced from a regular grid of seeds along both orthogonal
/// families. A trace that returns within the tolerance of an earlier point
/// with tangent continuity (dot product of the two directions at least
/// `cos 15 degrees`) yields an orbit. The orbit is kept only if its offset
/// neighbours on both sides spiral toward it (a stable limit cycle) or away
/// from it (an unstable one); an ordinary cycle, whose neighbours stay
/// parallel, is allowed by the thesis and is not reported. Duplicate orbits
/// found from different seeds are merged.
///
/// The returned polylines do not duplicate the first point; they close
/// implicitly. `LimitCycle::closed` is `true`. The function never panics and
/// returns an error for inconsistent or degenerate input.
pub fn detect_limit_cycles(
    points: &[[f64; 3]],
    tris: &[[usize; 3]],
    alpha: &[f64],
    size: &[f64],
) -> Result<Vec<LimitCycle>, String> {
    let field = Field::build(points, tris, alpha, size)?;
    let mean_size = size.iter().sum::<f64>() / size.len().max(1) as f64;
    if !mean_size.is_finite() || mean_size <= 0.0 {
        return Err("the mean size is not finite and positive".to_string());
    }
    let h = STEP_FRACTION * mean_size;
    let diagonal = field.diagonal();
    if !diagonal.is_finite() || diagonal <= 0.0 {
        return Err("the input points span a degenerate bounding box".to_string());
    }
    let seeds = seeds(&field);
    if seeds.is_empty() {
        return Err("no seed point lies inside the input tessellation".to_string());
    }

    let mut cycles: Vec<LimitCycle> = Vec::new();
    for seed in seeds {
        let Some(alpha0) = field.alpha_at(seed) else {
            continue;
        };
        for family in 0..2 {
            let theta = alpha0 + f64::from(family) * FRAC_PI_2;
            let dir = [theta.cos(), theta.sin()];
            // A line field has no orientation: a seed outside an attracting
            // cycle may have to be followed in the reverse direction. Both
            // orientations of both families are traced.
            for orientation in [1.0f64, -1.0] {
                let oriented = [orientation * dir[0], orientation * dir[1]];
                let Some(polyline) = field.trace_until_closed(seed, oriented, h) else {
                    continue;
                };
                let cycle = LimitCycle {
                    polyline,
                    closed: true,
                };
                if cycle.polyline.len() < MIN_LOOP_STEPS {
                    continue;
                }
                if classify(&field, &cycle.polyline, h) == Kind::Ordinary {
                    continue;
                }
                if cycles
                    .iter()
                    .any(|existing| similar_cycles(existing, &cycle, h))
                {
                    continue;
                }
                cycles.push(cycle);
            }
        }
    }
    Ok(cycles)
}

// ---------------------------------------------------------------------------
// Removal: the pi/2 twist of Section 4.12
// ---------------------------------------------------------------------------

/// Apply the thesis's removable-limit-cycle twist to a cross field.
///
/// For every vertex the signed distance `d` to the nearest point of a cycle is
/// computed (negative inside the cycle, positive outside). The angle is then
/// rotated by
///
/// ```text
/// alpha <- alpha + (pi/2) * smoothstep(0.5 - d / width)
/// ```
///
/// clamped to `1` inside and `0` outside. A global `pi/2` rotation leaves an
/// unoriented cross field unchanged, so only the annulus of width `width`
/// around the cycle changes; that is the discrete version of the thesis's
/// "smoothly add a twist of pi/2 to alpha in the neighborhood of gamma", which
/// swaps the two streamline families and removes a simple stable limit cycle
/// without adding singular points. The result is reduced to `[0, pi/2)`.
///
/// `cycles` is normally the output of [`detect_limit_cycles`]; an empty list
/// returns `alpha` unchanged. Cycles are applied in order, so overlapping
/// neighbourhoods accumulate twists (only meaningful for disjoint simple
/// cycles). `width` must be finite and positive.
///
/// Returns an error when the mesh or `alpha` is invalid, when a cycle has
/// fewer than three points, or when `width` is not finite and positive.
pub fn avoid_limit_cycles(
    points: &[[f64; 3]],
    tris: &[[usize; 3]],
    alpha: &[f64],
    cycles: &[LimitCycle],
    width: f64,
) -> Result<Vec<f64>, String> {
    if !width.is_finite() || width <= 0.0 {
        return Err(format!(
            "the twist width {width} is not finite and positive"
        ));
    }
    if alpha.len() != points.len() {
        return Err(format!(
            "alpha has {} values but there are {} points",
            alpha.len(),
            points.len()
        ));
    }
    if cycles.iter().any(|cycle| cycle.polyline.len() < 3) {
        return Err("a limit cycle has fewer than three polyline points".to_string());
    }
    // Validate the mesh itself even when there is nothing to twist.
    let sizes = vec![1.0; points.len()];
    let field = Field::build(points, tris, alpha, &sizes)?;
    if cycles.is_empty() {
        return Ok(alpha.to_vec());
    }

    let mut out = Vec::with_capacity(field.pts.len());
    for (index, point) in field.pts.iter().enumerate() {
        let mut value = alpha[index];
        for cycle in cycles {
            let inside = point_in_cycle(*point, &cycle.polyline);
            let distance = point_to_cycle(*point, &cycle.polyline);
            if !distance.is_finite() {
                continue;
            }
            let signed = if inside { -distance } else { distance };
            let t = (0.5 - signed / width).clamp(0.0, 1.0);
            let smooth = t * t * (3.0 - 2.0 * t);
            value += FRAC_PI_2 * smooth;
        }
        out.push(value.rem_euclid(FRAC_PI_2));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

/// Numbers measured by [`demo`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DemoNumbers {
    /// Limit cycles detected in the stable spiral field (expected: one).
    pub limit_cycles: usize,
    /// Limit cycles detected in the purely circular field (expected: zero,
    /// because those are ordinary cycles).
    pub ordinary_cycles: usize,
    /// Vertices whose angle changed by the removal twist.
    pub twisted_vertices: usize,
    /// Largest 4-RoSy angle change of the twist, in radians (the change is
    /// measured modulo `pi/2`, because a cross field is unoriented).
    pub max_twist: f64,
    /// Limit cycles still detected after the twist.
    pub cycles_after_twist: usize,
}

/// Regular `n x n` triangulation of the square `[-2, 2]^2`.
fn demo_mesh(n: usize) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let n = n.max(1);
    let mut points = Vec::with_capacity((n + 1) * (n + 1));
    for j in 0..=n {
        for i in 0..=n {
            let x = -2.0 + 4.0 * i as f64 / n as f64;
            let y = -2.0 + 4.0 * j as f64 / n as f64;
            points.push([x, y, 0.0]);
        }
    }
    let mut tris = Vec::with_capacity(2 * n * n);
    for j in 0..n {
        for i in 0..n {
            let a = j * (n + 1) + i;
            let b = a + 1;
            let c = a + n + 2;
            let d = a + n + 1;
            tris.push([a, b, c]);
            tris.push([a, c, d]);
        }
    }
    (points, tris)
}

/// A stable limit cycle at the unit circle, or a purely circular field.
///
/// With `spiral = true` the direction is that of the vector field
/// `v = (x (1 - r^2) - y, y (1 - r^2) + x)`: streamlines spiral toward the
/// unit circle from both sides, which is therefore a stable limit cycle. With
/// `spiral = false` the direction is tangent to all concentric circles
/// (`v = (-y, x)`), so every streamline is an ordinary cycle and none may be
/// reported as a limit cycle. In both cases the origin is a singular point and
/// its angle is set to zero.
fn circular_alpha(points: &[[f64; 3]], spiral: bool) -> Vec<f64> {
    points
        .iter()
        .map(|p| {
            let (x, y) = (p[0], p[1]);
            let r2 = x * x + y * y;
            let (vx, vy) = if spiral {
                let factor = 1.0 - r2;
                (x * factor - y, y * factor + x)
            } else {
                (-y, x)
            };
            if vx == 0.0 && vy == 0.0 {
                0.0
            } else {
                vy.atan2(vx).rem_euclid(FRAC_PI_2)
            }
        })
        .collect()
}

/// Run the three measurements reported in the module documentation.
///
/// 1. Detect the stable limit cycle of the spiral field.
/// 2. Detect on the purely circular field and confirm that ordinary cycles
///    are filtered out.
/// 3. Apply [`avoid_limit_cycles`] to the spiral field and re-detect.
///
/// The fields live on a `14 x 14` triangulation of `[-2, 2]^2` with a unit
/// size function; the twist width is `0.75`.
pub fn demo() -> Result<DemoNumbers, String> {
    let (points, tris) = demo_mesh(14);
    let size = vec![1.0; points.len()];
    let spiral = circular_alpha(&points, true);
    let limit = detect_limit_cycles(&points, &tris, &spiral, &size)?;
    let circle = circular_alpha(&points, false);
    let ordinary = detect_limit_cycles(&points, &tris, &circle, &size)?;

    let width = 0.75;
    let twisted = avoid_limit_cycles(&points, &tris, &spiral, &limit, width)?;
    let mut twisted_vertices = 0usize;
    let mut max_twist = 0.0f64;
    for (&before, &after) in spiral.iter().zip(twisted.iter()) {
        let change = (after - before).rem_euclid(FRAC_PI_2);
        let change = change.min(FRAC_PI_2 - change);
        if change > 1.0e-12 {
            twisted_vertices += 1;
        }
        max_twist = max_twist.max(change);
    }
    let after = detect_limit_cycles(&points, &tris, &twisted, &size)?;
    Ok(DemoNumbers {
        limit_cycles: limit.len(),
        ordinary_cycles: ordinary.len(),
        twisted_vertices,
        max_twist,
        cycles_after_twist: after.len(),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use std::f64::consts::TAU;

    #[test]
    fn demo_measures_the_section_4_12_numbers() -> Result<(), String> {
        let numbers = demo()?;
        println!(
            "limit cycles on the stable spiral field: {}",
            numbers.limit_cycles
        );
        println!(
            "limit cycles on the purely circular field: {}",
            numbers.ordinary_cycles
        );
        println!(
            "twist: {} vertices changed, max change {:.6} rad, {} limit cycles after",
            numbers.twisted_vertices, numbers.max_twist, numbers.cycles_after_twist
        );
        assert_eq!(
            numbers.limit_cycles, 1,
            "expected exactly one stable limit cycle"
        );
        assert_eq!(
            numbers.ordinary_cycles, 0,
            "ordinary cycles must not be reported as limit cycles"
        );
        assert!(numbers.twisted_vertices > 0, "the twist changed no vertex");
        assert!(numbers.max_twist > 0.1, "the twist is too small");
        assert!(
            numbers.cycles_after_twist <= numbers.limit_cycles,
            "the twist created more limit cycles"
        );
        Ok(())
    }

    #[test]
    fn constant_field_has_no_cycles() -> Result<(), String> {
        let (points, tris) = demo_mesh(6);
        let alpha = vec![0.0; points.len()];
        let size = vec![1.0; points.len()];
        let cycles = detect_limit_cycles(&points, &tris, &alpha, &size)?;
        println!("constant field: {} cycles", cycles.len());
        assert!(cycles.is_empty());
        Ok(())
    }

    #[test]
    fn plain_circular_field_reports_no_limit_cycle() -> Result<(), String> {
        let (points, tris) = demo_mesh(12);
        let size = vec![1.0; points.len()];
        let alpha = circular_alpha(&points, false);
        let cycles = detect_limit_cycles(&points, &tris, &alpha, &size)?;
        println!("purely circular field: {} cycles", cycles.len());
        assert!(cycles.is_empty());
        Ok(())
    }

    #[test]
    fn stable_spiral_reports_one_limit_cycle() -> Result<(), String> {
        let (points, tris) = demo_mesh(14);
        let size = vec![1.0; points.len()];
        let alpha = circular_alpha(&points, true);
        let cycles = detect_limit_cycles(&points, &tris, &alpha, &size)?;
        println!("stable spiral field: {} cycle(s)", cycles.len());
        for cycle in &cycles {
            assert!(cycle.closed);
            assert!(cycle.polyline.len() >= MIN_LOOP_STEPS);
            for p in &cycle.polyline {
                assert!(p[0].is_finite() && p[1].is_finite());
            }
            // The limit cycle is the unit circle: every vertex is at radius 1
            // up to the closure tolerance.
            let worst = cycle
                .polyline
                .iter()
                .map(|p| ((p[0] * p[0] + p[1] * p[1]).sqrt() - 1.0).abs())
                .fold(0.0f64, f64::max);
            println!(
                "  polyline points = {}, max |radius - 1| = {:.4e}, perimeter = {:.4}",
                cycle.polyline.len(),
                worst,
                cycle_perimeter(&cycle.polyline)
            );
            assert!(worst < 0.2, "not close to the unit circle: {worst}");
        }
        assert_eq!(cycles.len(), 1);
        Ok(())
    }

    #[test]
    fn twist_changes_the_annulus_and_not_the_far_field() -> Result<(), String> {
        let (points, tris) = demo_mesh(14);
        let size = vec![1.0; points.len()];
        let alpha = circular_alpha(&points, true);
        let cycles = detect_limit_cycles(&points, &tris, &alpha, &size)?;
        assert_eq!(cycles.len(), 1);
        let width = 1.0;
        let twisted = avoid_limit_cycles(&points, &tris, &alpha, &cycles, width)?;
        assert_eq!(twisted.len(), alpha.len());
        let mut far_changed = 0usize;
        let mut near_changed = 0usize;
        for (p, (&before, &after)) in points.iter().zip(alpha.iter().zip(twisted.iter())) {
            let distance = point_to_cycle([p[0], p[1]], &cycles[0].polyline);
            let change = (after - before).rem_euclid(FRAC_PI_2);
            let change = change.min(FRAC_PI_2 - change);
            if distance < 0.5 * width && change > 1.0e-6 {
                near_changed += 1;
            }
            if distance > 2.0 * width && change > 1.0e-6 {
                far_changed += 1;
            }
        }
        println!(
            "twist width {width}: {near_changed} vertices changed near the cycle, {far_changed} far away"
        );
        assert!(near_changed > 0);
        assert_eq!(far_changed, 0);
        Ok(())
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        let (points, tris) = demo_mesh(4);
        let size = vec![1.0; points.len()];
        let alpha = vec![0.0; points.len()];
        assert!(detect_limit_cycles(&points, &tris, &alpha[..2], &size).is_err());
        assert!(detect_limit_cycles(&points, &tris, &alpha, &size[..2]).is_err());
        assert!(detect_limit_cycles(&points, &[], &alpha, &size).is_err());
        assert!(detect_limit_cycles(&points, &[[0, 1, points.len()]], &alpha, &size).is_err());
        let mut bad_size = size.clone();
        bad_size[0] = 0.0;
        assert!(detect_limit_cycles(&points, &tris, &alpha, &bad_size).is_err());
        let cycles = vec![LimitCycle {
            polyline: vec![[0.0, 0.0], [1.0, 0.0]],
            closed: true,
        }];
        assert!(avoid_limit_cycles(&points, &tris, &alpha, &cycles, 0.5).is_err());
        assert!(avoid_limit_cycles(&points, &tris, &alpha, &[], 0.0).is_err());
        assert!(avoid_limit_cycles(&points, &tris, &alpha[..2], &[], 0.5).is_err());
        // The empty cycle list is a no-op.
        let same = avoid_limit_cycles(&points, &tris, &alpha, &[], 0.5);
        assert_eq!(same, Ok(alpha));
    }

    #[test]
    fn degenerate_tessellation_is_rejected() {
        let points = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let tris = vec![[0, 1, 2]];
        let size = vec![1.0; 3];
        let alpha = vec![0.0; 3];
        assert!(detect_limit_cycles(&points, &tris, &alpha, &size).is_err());
    }

    #[test]
    fn closure_helper_measures_the_unit_circle() {
        let polyline: Vec<[f64; 2]> = (0..16)
            .map(|k| {
                let theta = TAU * k as f64 / 16.0;
                [theta.cos(), theta.sin()]
            })
            .collect();
        let perimeter = cycle_perimeter(&polyline);
        assert!((perimeter - TAU).abs() < 0.05);
        assert!(point_in_cycle([0.0, 0.0], &polyline));
        assert!(!point_in_cycle([2.0, 0.0], &polyline));
        assert!((point_to_cycle([0.0, 1.1], &polyline) - 0.1).abs() < 1.0e-2);
        println!("unit-circle polyline: perimeter = {perimeter:.6}");
    }
}
