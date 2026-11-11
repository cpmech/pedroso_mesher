//! Size-function families for a 4-RoSy cross field.
//!
//! Implements Simons (2024), *Size-Driven All-Quad Meshing for Geographic
//! Data*, Chapter 4, sections 4.3-4.12: the local orthogonal frame
//! `(sigma, tau, alpha)`, the characteristic PDEs (4.3)/(4.4) for the two
//! length functions along streamlines, the anisotropy `A` of eq. (4.7), and
//! the streamline-based optimisation of the per-line scalar multipliers of
//! section 4.11.
//!
//! # What is implemented faithfully
//!
//! - **Cross field and frame (section 4.3).** The field is an angle
//!   `alpha(x, y) mod pi/2`; at a regular point the frame is
//!   `K = [sigma cos a, -tau sin a; sigma sin a, tau cos a]`. [`inverse_jacobian`]
//!   is eq. (4.1) and [`metric`] is the Riemannian metric `K^{-T} K^{-1}` of
//!   eq. (4.2).
//! - **Streamlines (section 4.3).** [`trace_streamline`] traces a field line
//!   with classical RK4. At every stage the branch `alpha + k pi/2` closest to
//!   the previous direction is chosen, which continues the same field line
//!   through the 4-RoSy ambiguity (a branch jump of `pi/2` would turn onto the
//!   orthogonal family).
//! - **Integration of sigma and tau (section 4.7).** The length function of a
//!   streamline family obeys, along its own tangent, `f' = f grad(alpha) . n`
//!   with `n` the left normal of the tangent; for the family with tangent
//!   `(-sin a, cos a)` this is exactly eq. (4.3) and for the orthogonal family
//!   with tangent `(cos a, sin a)` it is exactly eq. (4.4). The profile is
//!   integrated in log space with the exponential factor of each polyline
//!   segment: for the piecewise-constant `grad(alpha)` that the triangle
//!   discretisation provides this is the exact solution of the ODE on the
//!   segment. It is first-order accurate in the step, like the forward Euler
//!   used by the thesis (section 4.14.1), but the exponential form never
//!   changes the sign of `sigma` or `tau`.
//! - **Storage (section 4.14.1).** `alpha` is given per vertex and linearly
//!   interpolated over a triangle mesh, so `grad(alpha)` is constant per
//!   triangle. The values are periodic modulo `pi/2`; for every triangle they
//!   are first shifted by multiples of `pi/2` to the branch closest to the
//!   first vertex. A triangle whose shifted values still spread by more than
//!   `pi/4` is *singular* (barycentric interpolation of the angle is not
//!   meaningful there) and the workaround proposed by the thesis for singular
//!   triangles is applied: `grad(sigma) = grad(tau) = 0` on it.
//! - **Anisotropy (section 4.10).** `A = (sigma/tau + tau/sigma - 2)^(2k)`
//!   with `k >= 1`, eq. (4.7). It is zero iff `sigma = tau`; larger `k`
//!   penalises the worst anisotropy more strongly.
//! - **Optimisation (section 4.11).** A sparse set of representative
//!   streamlines is seeded on a regular grid (both families). Each line is
//!   integrated from the target size at its seed and then carries a scalar
//!   multiplier; `A` is evaluated at every intersection of the two families;
//!   the multipliers are adjusted by coordinate descent, one exact 1-D
//!   minimisation per line, until a pass brings no improvement. In the log
//!   variable the per-line objective is convex, because
//!   `A = (2 cosh u - 2)^(2k)` with `u = ln(sigma/tau)`, so the minimisation
//!   is robust.
//! - **Isotropic case (sections 4.5/4.6).** When `sigma = tau`, eqs. (4.3)
//!   and (4.4) reduce to the Cauchy-Riemann-like system of (4.5)/(4.6).
//!   Note a sign slip in the printed equations: combining (4.3) and (4.4)
//!   with `sigma = tau` gives `sigma_x = sigma alpha_y` and
//!   `sigma_y = -sigma alpha_x`, i.e. both printed equations with their signs
//!   flipped (the same as replacing `alpha` by `-alpha`). The thesis's own
//!   singular-point example, `sigma = (4/3)(x^2+y^2)^(1/8)` with
//!   `alpha = atan(y/x)/4`, satisfies the derived signs. Only (4.3)/(4.4) are
//!   used for the numerics, so the slip does not affect the results; it is
//!   documented for whoever compares the equations.
//! - **Size accuracy (section 4.13).** `sqrt(sigma tau) / s_hat` is reported,
//!   at the sample points, as [`FieldAnalysis::max_size_ratio`].
//!
//! # Simplifications
//!
//! - The tessellation and `alpha` are inputs: there is no cross-field solve,
//!   no topology-driven choice of representative lines (separatrices, limit
//!   cycles, "close to every point of `T`"), no limit-cycle removal
//!   (section 4.12), no visualisation, and no surface (3-D) version.
//! - Seeds are a regular `SEED_GRID x SEED_GRID` grid over the bounding box
//!   (both families), not a topology-driven set. Duplicate representatives are
//!   not removed, so one intersection can be counted more than once.
//! - The global scale of `sigma` and `tau` is fixed by matching the geometric
//!   mean of `sqrt(sigma tau)` to the target size at the sample points, instead
//!   of the thesis's `integral(sigma^2 + tau^2) dA = 1`. `A` is invariant
//!   under a common scaling of `sigma` and `tau`; only the reported size ratio
//!   depends on this convention.
//! - The analysis is planar: the `z` coordinate of the points is ignored.
//!
//! # Public API
//!
//! ```
//! # use pedroso_mesher::simons_sizefamily::{analyze, trace_streamline};
//! // A 2x1 rectangle split into right triangles.
//! let points = vec![
//!     [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0],
//!     [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0],
//! ];
//! let tris = vec![[0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4]];
//! let alpha = vec![0.0; points.len()];
//! let size = vec![1.0; points.len()];
//! let report = analyze(&points, &tris, &alpha, &size).unwrap();
//! assert_eq!(report.max_anisotropy, 0.0);
//! assert!(report.samples > 0);
//! let path = trace_streamline(&alpha, &points, &tris, [0.5, 0.5], 10, 0.05);
//! assert!(path.len() > 1);
//! ```

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};

/// Default exponent `k` of the anisotropy (eq. 4.7) used by [`analyze`].
///
/// `k = 2` keeps the mean anisotropy informative while still penalising the
/// worst sample harder than `k = 1`; the thesis leaves `k` as a tuning
/// parameter. [`analyze_with_k`] exposes it.
pub const DEFAULT_K: u32 = 2;

/// Number of seeds per axis of the regular seeding grid.
const SEED_GRID: usize = 4;

/// Maximum number of RK4 steps taken in each direction from a seed.
const TRACE_STEPS: usize = 48;

/// Maximum number of coordinate-descent sweeps.
const MAX_SWEEPS: usize = 60;

/// Golden-section iterations for the one-dimensional line minimisation.
const GOLDEN_ITERS: usize = 100;

/// Half-width of the golden-section bracket in `ln(multiplier)`.
const BRACKET: f64 = 30.0;

/// Clamp on `|ln r|` inside the anisotropy evaluation, which keeps `A` finite.
const LOG_CLAMP: f64 = 40.0;

/// Clamp on `|ln value|` of an integrated length profile.
const VALUE_CLAMP: f64 = 60.0;

/// Anisotropy returned for invalid inputs and for ratios beyond the clamp.
const MAX_ANISOTROPY: f64 = 1.0e300;

/// Barycentric tolerance for point location.
const BARY_TOL: f64 = 1.0e-9;

/// A triangle with a smaller absolute double area is treated as degenerate.
const DEGENERATE_AREA2: f64 = 1.0e-14;

/// Safety cap on the number of intersections kept for the optimisation.
const MAX_INTERSECTIONS: usize = 20_000;

/// Hard cap on the number of steps accepted by [`trace_streamline`].
const MAX_TRACE_STEPS: usize = 10_000;

/// Angle spread above which a triangle is treated as singular.
const SINGULAR_SPREAD: f64 = FRAC_PI_4;

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Outcome of the size-function family analysis on one cross field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldAnalysis {
    /// Worst anisotropy `A` (eq. 4.7) over the sample points.
    pub max_anisotropy: f64,
    /// Mean anisotropy `A` over the sample points.
    pub mean_anisotropy: f64,
    /// Number of sample points: intersections of the representative
    /// streamlines of the two families (section 4.11).
    pub samples: usize,
    /// Worst-case size accuracy `max(r, 1/r)` with
    /// `r = sqrt(sigma tau) / s_hat`, over the sample points. `1.0` means
    /// `sqrt(sigma tau)` matches the target size exactly; larger values mean a
    /// worse match in either direction.
    pub max_size_ratio: f64,
}

// ---------------------------------------------------------------------------
// Small vector algebra
// ---------------------------------------------------------------------------

fn dot(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

fn cross(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

fn left_perp(a: [f64; 2]) -> [f64; 2] {
    [-a[1], a[0]]
}

// ---------------------------------------------------------------------------
// Section 4.3/4.4: the inverse Jacobian and the metric
// ---------------------------------------------------------------------------

/// Inverse Jacobian `K^{-1}` of the local parameterisation, eq. (4.1).
///
/// `K = [[sigma cos a, -tau sin a], [sigma sin a, tau cos a]]` maps
/// infinitesimal axis-aligned squares of parameter space to rectangles of the
/// cross field, and
/// `K^{-1} = [[cos(-a)/sigma, -sin(-a)/sigma],
///            [ sin(-a)/tau,   cos(-a)/tau  ]]`.
///
/// Returns `None` when `sigma` or `tau` is not finite and positive, or when
/// `alpha` is not finite.
pub fn inverse_jacobian(sigma: f64, tau: f64, alpha: f64) -> Option<[[f64; 2]; 2]> {
    if !sigma.is_finite() || sigma <= 0.0 || !tau.is_finite() || tau <= 0.0 {
        return None;
    }
    if !alpha.is_finite() {
        return None;
    }
    let (c, s) = (alpha.cos(), alpha.sin());
    Some([[c / sigma, s / sigma], [-s / tau, c / tau]])
}

/// Riemannian metric `K^{-T} K^{-1}` of eq. (4.2).
///
/// The squared length of a plane vector `v` measured in parameter space is
/// `v^T M v`; for `sigma = tau = s` the metric is `(1/s^2) I` (conformal,
/// isothermal case of section 4.5).
///
/// Returns `None` on the same conditions as [`inverse_jacobian`].
pub fn metric(sigma: f64, tau: f64, alpha: f64) -> Option<[[f64; 2]; 2]> {
    let k_inv = inverse_jacobian(sigma, tau, alpha)?;
    // (K^{-T} K^{-1})_{ij} = sum_l K^{-1}_{li} K^{-1}_{lj}.
    let mut m = [[0.0f64; 2]; 2];
    for (i, row) in m.iter_mut().enumerate() {
        for (j, value) in row.iter_mut().enumerate() {
            let mut sum = 0.0;
            for l in 0..2 {
                sum += k_inv[l][i] * k_inv[l][j];
            }
            *value = sum;
        }
    }
    if m.iter().all(|r| r.iter().all(|v| v.is_finite())) {
        Some(m)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Section 4.10: anisotropy
// ---------------------------------------------------------------------------

/// Anisotropy `A = (sigma/tau + tau/sigma - 2)^(2k)`, eq. (4.7).
///
/// `k` is clamped to `1..=8` and the ratio is clamped to `exp(+-40)` so that
/// the result is always finite; non-positive or non-finite inputs return
/// `1e300`.
pub fn anisotropy(sigma: f64, tau: f64, k: u32) -> f64 {
    if !sigma.is_finite() || sigma <= 0.0 || !tau.is_finite() || tau <= 0.0 {
        return MAX_ANISOTROPY;
    }
    aniso_log(sigma.ln() - tau.ln(), k)
}

/// `A` from `u = ln(sigma/tau)`: `(2 cosh u - 2)^(2k) = (4 sinh^2(u/2))^(2k)`.
fn aniso_log(u: f64, k: u32) -> f64 {
    if !u.is_finite() {
        return MAX_ANISOTROPY;
    }
    let u = u.clamp(-LOG_CLAMP, LOG_CLAMP);
    let k = k.clamp(1, 8);
    let half_sinh = (0.5 * u).sinh();
    let base = 4.0 * half_sinh * half_sinh;
    let exponent = 2 * k;
    let value = base.powi(exponent as i32);
    if value.is_finite() && value >= 0.0 {
        value.min(MAX_ANISOTROPY)
    } else {
        MAX_ANISOTROPY
    }
}

// ---------------------------------------------------------------------------
// Background triangle mesh with piecewise-linear alpha
// ---------------------------------------------------------------------------

/// Planar triangle tessellation carrying `alpha` and the target size.
struct Background {
    pts: Vec<[f64; 2]>,
    tris: Vec<[usize; 3]>,
    tri_lo: Vec<[f64; 2]>,
    tri_hi: Vec<[f64; 2]>,
    /// Signed double area of every triangle.
    den: Vec<f64>,
    /// Branch-corrected `alpha` at the three vertices of every triangle.
    alpha: Vec<[f64; 3]>,
    /// `grad(alpha)`, constant per triangle; zero on singular triangles.
    grad: Vec<[f64; 2]>,
    singular: Vec<bool>,
    size: Vec<f64>,
    /// `[xmin, ymin, xmax, ymax]`.
    bbox: [f64; 4],
}

impl Background {
    fn build(
        points: &[[f64; 3]],
        tris: &[[usize; 3]],
        alpha: &[f64],
        target: Option<&[f64]>,
    ) -> Result<Background, String> {
        let n = points.len();
        if n < 3 {
            return Err("the size-function analysis needs at least 3 points".to_string());
        }
        if alpha.len() != n {
            return Err(format!(
                "alpha has {} values but there are {n} points",
                alpha.len()
            ));
        }
        if let Some(sizes) = target {
            if sizes.len() != n {
                return Err(format!(
                    "target_size has {} values but there are {n} points",
                    sizes.len()
                ));
            }
        }
        if tris.is_empty() {
            return Err("the size-function analysis needs at least one triangle".to_string());
        }

        let mut pts = Vec::with_capacity(n);
        let mut bbox = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        for (i, p) in points.iter().enumerate() {
            if !p[0].is_finite() || !p[1].is_finite() || !p[2].is_finite() {
                return Err(format!("point {i} has a non-finite coordinate"));
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
                return Err(format!("alpha[{i}] is not a usable finite angle"));
            }
        }
        let mut sizes = Vec::with_capacity(n);
        match target {
            Some(values) => {
                for (i, v) in values.iter().enumerate() {
                    if !v.is_finite() || *v <= 0.0 {
                        return Err(format!("target_size[{i}] must be finite and positive"));
                    }
                    sizes.push(*v);
                }
            }
            None => sizes = vec![1.0; n],
        }

        let mut tri_lo = Vec::with_capacity(tris.len());
        let mut tri_hi = Vec::with_capacity(tris.len());
        let mut den = Vec::with_capacity(tris.len());
        let mut alpha_tri = Vec::with_capacity(tris.len());
        let mut grad = Vec::with_capacity(tris.len());
        let mut singular = Vec::with_capacity(tris.len());

        for (ti, t) in tris.iter().enumerate() {
            if t[0] >= n || t[1] >= n || t[2] >= n {
                return Err(format!(
                    "triangle {ti} references a point that does not exist"
                ));
            }
            let p0 = pts[t[0]];
            let p1 = pts[t[1]];
            let p2 = pts[t[2]];
            let e1 = [p1[0] - p0[0], p1[1] - p0[1]];
            let e2 = [p2[0] - p0[0], p2[1] - p0[1]];
            let d = cross(e1, e2);

            tri_lo.push([p0[0].min(p1[0]).min(p2[0]), p0[1].min(p1[1]).min(p2[1])]);
            tri_hi.push([p0[0].max(p1[0]).max(p2[0]), p0[1].max(p1[1]).max(p2[1])]);
            den.push(d);

            // Lift each vertex angle by a multiple of pi/2 to the representative
            // closest to vertex 0, which removes the pi/2 periodicity across
            // the triangle.
            let f0 = alpha[t[0]];
            let f1 = alpha[t[1]] + FRAC_PI_2 * ((f0 - alpha[t[1]]) / FRAC_PI_2).round();
            let f2 = alpha[t[2]] + FRAC_PI_2 * ((f0 - alpha[t[2]]) / FRAC_PI_2).round();
            let lo = f0.min(f1).min(f2);
            let hi = f0.max(f1).max(f2);
            let is_singular = hi - lo > SINGULAR_SPREAD;
            let g = if is_singular {
                [0.0, 0.0]
            } else {
                triangle_gradient(p0, p1, p2, f0, f1, f2, d).unwrap_or([0.0, 0.0])
            };
            alpha_tri.push([f0, f1, f2]);
            grad.push(g);
            singular.push(is_singular);
        }

        Ok(Background {
            pts,
            tris: tris.to_vec(),
            tri_lo,
            tri_hi,
            den,
            alpha: alpha_tri,
            grad,
            singular,
            size: sizes,
            bbox,
        })
    }

    fn diagonal(&self) -> f64 {
        let [x0, y0, x1, y1] = self.bbox;
        ((x1 - x0).powi(2) + (y1 - y0).powi(2)).sqrt()
    }

    /// Barycentric coordinates of `p` in triangle `ti`, without clamping.
    fn bary_raw(&self, p: [f64; 2], ti: usize) -> Option<[f64; 3]> {
        let tri = *self.tris.get(ti)?;
        let p0 = *self.pts.get(tri[0])?;
        let p1 = *self.pts.get(tri[1])?;
        let p2 = *self.pts.get(tri[2])?;
        let d = *self.den.get(ti)?;
        if d.abs() <= DEGENERATE_AREA2 {
            return None;
        }
        let e1 = [p1[0] - p0[0], p1[1] - p0[1]];
        let e2 = [p2[0] - p0[0], p2[1] - p0[1]];
        let w = [p[0] - p0[0], p[1] - p0[1]];
        let l1 = cross(w, e2) / d;
        let l2 = cross(e1, w) / d;
        Some([1.0 - l1 - l2, l1, l2])
    }

    /// Locate `p`; returns the triangle and clamped barycentric coordinates.
    fn locate_bary(&self, p: [f64; 2]) -> Option<(usize, [f64; 3])> {
        let mut best: Option<(usize, [f64; 3], f64)> = None;
        for ti in 0..self.tris.len() {
            let lo = self.tri_lo[ti];
            let hi = self.tri_hi[ti];
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

    /// Interpolated cross-field angle at `p`.
    fn alpha_at(&self, p: [f64; 2]) -> Option<f64> {
        let (ti, l) = self.locate_bary(p)?;
        let a = self.alpha.get(ti)?;
        Some(l[0] * a[0] + l[1] * a[1] + l[2] * a[2])
    }

    /// `grad(alpha)` at `p`; zero on singular triangles.
    fn grad_at(&self, p: [f64; 2]) -> Option<[f64; 2]> {
        let (ti, _) = self.locate_bary(p)?;
        if *self.singular.get(ti)? {
            Some([0.0, 0.0])
        } else {
            self.grad.get(ti).copied()
        }
    }

    /// Interpolated target size at `p`.
    fn size_at(&self, p: [f64; 2]) -> Option<f64> {
        let (ti, l) = self.locate_bary(p)?;
        let t = self.tris.get(ti)?;
        let s0 = *self.size.get(t[0])?;
        let s1 = *self.size.get(t[1])?;
        let s2 = *self.size.get(t[2])?;
        Some(l[0] * s0 + l[1] * s1 + l[2] * s2)
    }

    /// Direction of the cross-field branch closest to `prev` at `p`.
    fn branch_dir(&self, p: [f64; 2], prev: [f64; 2]) -> Option<[f64; 2]> {
        let a = self.alpha_at(p)?;
        let mut best = [a.cos(), a.sin()];
        let mut best_dot = dot(best, prev);
        for k in 1u32..4 {
            let th = a + f64::from(k) * FRAC_PI_2;
            let d = [th.cos(), th.sin()];
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

    /// `ln` of the multiplicative factor of one polyline segment.
    ///
    /// Along a streamline of either family, `d ln f / ds = grad(alpha) . n`,
    /// with `n` the left normal of the direction of travel (eqs. 4.3 and 4.4).
    fn log_step(&self, from: [f64; 2], to: [f64; 2]) -> f64 {
        let d = [to[0] - from[0], to[1] - from[1]];
        let len = (d[0] * d[0] + d[1] * d[1]).sqrt();
        if !len.is_finite() || len < 1.0e-15 {
            return 0.0;
        }
        let n = [-d[1] / len, d[0] / len];
        let mid = [0.5 * (from[0] + to[0]), 0.5 * (from[1] + to[1])];
        match self.grad_at(mid) {
            Some(g) => dot(g, n) * len,
            // The chord of a non-convex tessellation can leave the mesh even
            // when both ends are inside; clamping the contribution to zero is
            // the conservative choice.
            None => 0.0,
        }
    }

    /// Integrate `ln f` along the polyline, starting at `seed` with `seed_log`.
    fn integrate_log(&self, pts: &[[f64; 2]], seed: usize, seed_log: f64) -> Vec<f64> {
        let n = pts.len();
        let mut out = vec![seed_log; n];
        let mut i = seed;
        while i + 1 < n {
            let step = self.log_step(pts[i], pts[i + 1]);
            out[i + 1] = (out[i] + step).clamp(-VALUE_CLAMP, VALUE_CLAMP);
            i += 1;
        }
        let mut i = seed;
        while i > 0 {
            let step = self.log_step(pts[i], pts[i - 1]);
            out[i - 1] = (out[i] + step).clamp(-VALUE_CLAMP, VALUE_CLAMP);
            i -= 1;
        }
        out
    }

    /// Regular grid of seeds inside the mesh, with a centroid fallback.
    fn grid_seeds(&self) -> Vec<[f64; 2]> {
        let [x0, y0, x1, y1] = self.bbox;
        let mut out = Vec::new();
        for j in 0..SEED_GRID {
            for i in 0..SEED_GRID {
                let fx = (i as f64 + 0.5) / SEED_GRID as f64;
                let fy = (j as f64 + 0.5) / SEED_GRID as f64;
                let p = [x0 + (x1 - x0) * fx, y0 + (y1 - y0) * fy];
                if self.locate_bary(p).is_some() {
                    out.push(p);
                }
            }
        }
        if out.is_empty() {
            for t in self.tris.iter().take(4) {
                let p = [
                    (self.pts[t[0]][0] + self.pts[t[1]][0] + self.pts[t[2]][0]) / 3.0,
                    (self.pts[t[0]][1] + self.pts[t[1]][1] + self.pts[t[2]][1]) / 3.0,
                ];
                if self.locate_bary(p).is_some() {
                    out.push(p);
                }
            }
        }
        out
    }
}

/// Clamp barycentric coordinates to the closed triangle and normalise them.
fn clamp_bary(l: [f64; 3]) -> [f64; 3] {
    let c = [l[0].max(0.0), l[1].max(0.0), l[2].max(0.0)];
    let sum = c[0] + c[1] + c[2];
    if sum > 0.0 && sum.is_finite() {
        [c[0] / sum, c[1] / sum, c[2] / sum]
    } else {
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    }
}

/// Constant `grad(alpha)` of a triangle from its three vertex values.
///
/// `grad(phi_i)` is the outward-perpendicular of the opposite edge divided by
/// the signed double area, so `grad(alpha) = sum_i f_i grad(phi_i)`.
fn triangle_gradient(
    p0: [f64; 2],
    p1: [f64; 2],
    p2: [f64; 2],
    f0: f64,
    f1: f64,
    f2: f64,
    den: f64,
) -> Option<[f64; 2]> {
    if den.abs() <= DEGENERATE_AREA2 || !den.is_finite() {
        return None;
    }
    let g0 = left_perp([p2[0] - p1[0], p2[1] - p1[1]]);
    let g1 = left_perp([p0[0] - p2[0], p0[1] - p2[1]]);
    let g2 = left_perp([p1[0] - p0[0], p1[1] - p0[1]]);
    let gx = (f0 * g0[0] + f1 * g1[0] + f2 * g2[0]) / den;
    let gy = (f0 * g0[1] + f1 * g1[1] + f2 * g2[1]) / den;
    if gx.is_finite() && gy.is_finite() {
        Some([gx, gy])
    } else {
        None
    }
}

/// Gradient of the isotropic length function, eqs. (4.5)/(4.6).
///
/// When `tau = sigma` the characteristic equations reduce to a
/// Cauchy-Riemann-like system for `sigma`. Combining eqs. (4.3) and (4.4) gives
/// `sigma_x = sigma alpha_y` and `sigma_y = -sigma alpha_x`; the printed
/// (4.5)/(4.6) have both signs flipped (see the module documentation), which is
/// the same as replacing `alpha` by `-alpha`. Given a positive `sigma` and
/// `grad(alpha)`, this returns that gradient.
///
/// The thesis's singular-point example, `sigma = (4/3)(x^2+y^2)^(1/8)` and
/// `alpha = atan(y/x)/4`, reproduces
/// `grad(sigma) = (1/3)(x^2+y^2)^(-7/8)(x, y)`.
pub fn isotropic_scale_gradient(sigma: f64, grad_alpha: [f64; 2]) -> [f64; 2] {
    [sigma * grad_alpha[1], -sigma * grad_alpha[0]]
}

// ---------------------------------------------------------------------------
// Streamline tracing (sections 4.3 and 4.7)
// ---------------------------------------------------------------------------

/// The two streamline families of the cross field.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Family {
    /// Length function `sigma`, tangent `(-sin a, cos a)` (eq. 4.3).
    Sigma,
    /// Length function `tau`, tangent `(cos a, sin a)` (eq. 4.4).
    Tau,
}

impl Family {
    fn start_dir(self, alpha: f64) -> [f64; 2] {
        match self {
            Family::Sigma => [-alpha.sin(), alpha.cos()],
            Family::Tau => [alpha.cos(), alpha.sin()],
        }
    }
}

/// One RK4 step on the cross field, continuing the branch closest to `d0`.
fn rk4_step(bg: &Background, x: [f64; 2], d0: [f64; 2], h: f64) -> Option<([f64; 2], [f64; 2])> {
    let k1 = bg.branch_dir(x, d0)?;
    let x2 = [x[0] + 0.5 * h * k1[0], x[1] + 0.5 * h * k1[1]];
    let k2 = bg.branch_dir(x2, k1)?;
    let x3 = [x[0] + 0.5 * h * k2[0], x[1] + 0.5 * h * k2[1]];
    let k3 = bg.branch_dir(x3, k2)?;
    let x4 = [x[0] + h * k3[0], x[1] + h * k3[1]];
    let k4 = bg.branch_dir(x4, k3)?;
    let v = [
        (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0,
        (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0,
    ];
    let len = (v[0] * v[0] + v[1] * v[1]).sqrt();
    if !len.is_finite() || len < 1.0e-12 {
        return None;
    }
    let d = [v[0] / len, v[1] / len];
    let next = [x[0] + h * d[0], x[1] + h * d[1]];
    if !next[0].is_finite() || !next[1].is_finite() {
        return None;
    }
    // Stop at the boundary of the tessellation.
    bg.locate_bary(next)?;
    Some((next, d))
}

/// Trace `steps` RK4 steps from `start` in the direction `dir`.
fn trace_dir(
    bg: &Background,
    start: [f64; 2],
    dir: [f64; 2],
    steps: usize,
    step: f64,
) -> Vec<[f64; 2]> {
    let mut out = Vec::with_capacity(steps.min(256) + 1);
    out.push(start);
    let mut x = start;
    let mut d = dir;
    for _ in 0..steps {
        match rk4_step(bg, x, d, step) {
            Some((next, next_dir)) => {
                out.push(next);
                x = next;
                d = next_dir;
            }
            None => break,
        }
    }
    out
}

/// Trace a streamline of the cross field.
///
/// The path starts at `start` in the `+alpha` direction (the `tau` family,
/// tangent `(cos a, sin a)`) and takes up to `steps` RK4 steps of length
/// `step`. At every stage the branch `alpha + k pi/2` closest to the previous
/// direction is used, so the polyline continues the same field line instead of
/// turning onto the orthogonal family. The seed is the first returned point;
/// the trace stops early when the next point leaves the tessellation.
///
/// Returns an empty vector when the inputs cannot be interpreted (too few
/// points or triangles, inconsistent `alpha`, `start` outside the mesh, a
/// non-positive step). The function never panics; `steps` is capped at
/// `10_000`.
pub fn trace_streamline(
    alpha: &[f64],
    points: &[[f64; 3]],
    tris: &[[usize; 3]],
    start: [f64; 2],
    steps: usize,
    step: f64,
) -> Vec<[f64; 2]> {
    if !step.is_finite() || step <= 0.0 || steps == 0 {
        return Vec::new();
    }
    let bg = match Background::build(points, tris, alpha, None) {
        Ok(bg) => bg,
        Err(_) => return Vec::new(),
    };
    let alpha0 = match bg.alpha_at(start) {
        Some(a) => a,
        None => return Vec::new(),
    };
    let steps = steps.min(MAX_TRACE_STEPS);
    let dir = Family::Tau.start_dir(alpha0);
    trace_dir(&bg, start, dir, steps, step)
}

// ---------------------------------------------------------------------------
// Representative lines and their sigma/tau profiles
// ---------------------------------------------------------------------------

/// One representative streamline carrying one length function.
struct Line {
    pts: Vec<[f64; 2]>,
    /// `ln` of the integrated length function at every polyline point.
    log_base: Vec<f64>,
    /// Scalar multiplier of the whole line (section 4.11), in log space.
    log_mult: f64,
    bbox: [f64; 4],
}

impl Line {
    /// Trace a full line through `seed` (both directions) and integrate its
    /// length function, seeded with the target size at `seed`.
    fn trace(
        bg: &Background,
        seed: [f64; 2],
        family: Family,
        steps: usize,
        step: f64,
    ) -> Option<Line> {
        if !step.is_finite() || step <= 0.0 {
            return None;
        }
        let alpha0 = bg.alpha_at(seed)?;
        let dir0 = family.start_dir(alpha0);
        let forward = trace_dir(bg, seed, dir0, steps, step);
        let backward = trace_dir(bg, seed, [-dir0[0], -dir0[1]], steps, step);
        if forward.len() < 2 && backward.len() < 2 {
            return None;
        }

        let mut pts: Vec<[f64; 2]> = Vec::with_capacity(forward.len() + backward.len());
        for p in backward.iter().skip(1).rev() {
            pts.push(*p);
        }
        let seed_index = pts.len();
        pts.extend(forward.iter().copied());
        if pts.len() < 2 {
            return None;
        }

        let mut lo = [f64::INFINITY, f64::INFINITY];
        let mut hi = [f64::NEG_INFINITY, f64::NEG_INFINITY];
        for p in &pts {
            lo[0] = lo[0].min(p[0]);
            lo[1] = lo[1].min(p[1]);
            hi[0] = hi[0].max(p[0]);
            hi[1] = hi[1].max(p[1]);
        }

        let seed_value = bg.size_at(seed)?;
        let log_base = bg.integrate_log(&pts, seed_index, seed_value.ln());
        Some(Line {
            pts,
            log_base,
            log_mult: 0.0,
            bbox: [lo[0], lo[1], hi[0], hi[1]],
        })
    }

    /// `ln` of the integrated profile at segment `seg`, fraction `t`.
    fn eval_log(&self, seg: usize, t: f64) -> f64 {
        match (self.log_base.get(seg), self.log_base.get(seg + 1)) {
            (Some(a), Some(b)) => a + (b - a) * t.clamp(0.0, 1.0),
            _ => self.log_base.get(seg).copied().unwrap_or(0.0),
        }
    }

    /// `ln` of the whole profile including the current multiplier.
    fn total_log(&self, seg: usize, t: f64) -> f64 {
        self.eval_log(seg, t) + self.log_mult
    }
}

/// One intersection of a `sigma` line and a `tau` line (a sample point).
struct Intersection {
    si: usize,
    ti: usize,
    s_seg: usize,
    s_t: f64,
    t_seg: usize,
    t_t: f64,
    pos: [f64; 2],
}

/// Segment-segment intersection, returning the parameters on both segments.
fn seg_intersect(p: [f64; 2], p2: [f64; 2], q: [f64; 2], q2: [f64; 2]) -> Option<(f64, f64)> {
    let r = [p2[0] - p[0], p2[1] - p[1]];
    let s = [q2[0] - q[0], q2[1] - q[1]];
    let den = cross(r, s);
    if !den.is_finite() || den.abs() < 1.0e-14 {
        return None;
    }
    let qp = [q[0] - p[0], q[1] - p[1]];
    let t = cross(qp, s) / den;
    let u = cross(qp, r) / den;
    let tol = 1.0e-9;
    if t < -tol || t > 1.0 + tol || u < -tol || u > 1.0 + tol {
        return None;
    }
    Some((t.clamp(0.0, 1.0), u.clamp(0.0, 1.0)))
}

fn bbox_overlap(a: [f64; 4], b: [f64; 4]) -> bool {
    let tol = 1.0e-12;
    a[0] <= b[2] + tol && b[0] <= a[2] + tol && a[1] <= b[3] + tol && b[1] <= a[3] + tol
}

/// All intersections of the two families, capped at [`MAX_INTERSECTIONS`].
fn intersections(sig: &[Line], tau: &[Line]) -> Vec<Intersection> {
    let mut out = Vec::new();
    for (si, a) in sig.iter().enumerate() {
        for (ti, b) in tau.iter().enumerate() {
            if !bbox_overlap(a.bbox, b.bbox) {
                continue;
            }
            for i in 0..a.pts.len().saturating_sub(1) {
                for j in 0..b.pts.len().saturating_sub(1) {
                    let Some((t, u)) =
                        seg_intersect(a.pts[i], a.pts[i + 1], b.pts[j], b.pts[j + 1])
                    else {
                        continue;
                    };
                    let seg = [a.pts[i + 1][0] - a.pts[i][0], a.pts[i + 1][1] - a.pts[i][1]];
                    out.push(Intersection {
                        si,
                        ti,
                        s_seg: i,
                        s_t: t,
                        t_seg: j,
                        t_t: u,
                        pos: [a.pts[i][0] + t * seg[0], a.pts[i][1] + t * seg[1]],
                    });
                    if out.len() >= MAX_INTERSECTIONS {
                        return out;
                    }
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Section 4.11: coordinate descent over the line multipliers
// ---------------------------------------------------------------------------

/// Golden-section minimisation of a unimodal function on `[lo, hi]`.
fn golden_min<F: Fn(f64) -> f64>(f: F, lo: f64, hi: f64, iters: usize) -> f64 {
    const INV_PHI: f64 = 0.618_033_988_749_894_9;
    let mut a = lo;
    let mut b = hi;
    let mut c = b - INV_PHI * (b - a);
    let mut d = a + INV_PHI * (b - a);
    let mut fc = f(c);
    let mut fd = f(d);
    for _ in 0..iters {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - INV_PHI * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + INV_PHI * (b - a);
            fd = f(d);
        }
    }
    0.5 * (a + b)
}

/// Sum of the anisotropy over all sample points.
fn total_of(sig: &[Line], tau: &[Line], inters: &[Intersection], k: u32) -> f64 {
    let mut sum = 0.0;
    for it in inters {
        let u = sig[it.si].total_log(it.s_seg, it.s_t) - tau[it.ti].total_log(it.t_seg, it.t_t);
        sum += aniso_log(u, k);
    }
    sum
}

/// Resize one line so that the summed anisotropy of its intersections is
/// minimised, leaving the other family fixed.
///
/// In the log multiplier `s` the objective is a sum of
/// `(2 cosh(g + sign*s) - 2)^(2k)` terms, which is convex, so the exact 1-D
/// minimum is found by golden section.
fn optimize_line(
    which_is_sigma: bool,
    idx: usize,
    sig: &mut [Line],
    tau: &mut [Line],
    inters: &[Intersection],
    k: u32,
) {
    let (terms, current) = {
        let own = if which_is_sigma { &sig[idx] } else { &tau[idx] };
        let mut terms: Vec<(f64, f64)> = Vec::new();
        for it in inters {
            let (own_seg, own_t, other_total) = if which_is_sigma {
                if it.si != idx {
                    continue;
                }
                (it.s_seg, it.s_t, tau[it.ti].total_log(it.t_seg, it.t_t))
            } else {
                if it.ti != idx {
                    continue;
                }
                (it.t_seg, it.t_t, sig[it.si].total_log(it.s_seg, it.s_t))
            };
            let own_base = own.eval_log(own_seg, own_t);
            let g = if which_is_sigma {
                own_base - other_total
            } else {
                other_total - own_base
            };
            terms.push((g, if which_is_sigma { 1.0 } else { -1.0 }));
        }
        (terms, own.log_mult)
    };
    if terms.is_empty() {
        return;
    }
    let eval = |s: f64| -> f64 {
        let mut sum = 0.0;
        for &(g, sign) in &terms {
            sum += aniso_log(g + sign * s, k);
        }
        sum
    };
    let before = eval(current);
    if before <= 0.0 {
        return;
    }
    let s = golden_min(eval, current - BRACKET, current + BRACKET, GOLDEN_ITERS);
    if eval(s) < before {
        let s = s.clamp(-VALUE_CLAMP, VALUE_CLAMP);
        if which_is_sigma {
            sig[idx].log_mult = s;
        } else {
            tau[idx].log_mult = s;
        }
    }
}

/// Representative lines, their intersections and the exponent `k`.
struct Run {
    sig: Vec<Line>,
    tau: Vec<Line>,
    inters: Vec<Intersection>,
    k: u32,
}

impl Run {
    fn log_sigma(&self, it: &Intersection) -> f64 {
        self.sig[it.si].total_log(it.s_seg, it.s_t)
    }

    fn log_tau(&self, it: &Intersection) -> f64 {
        self.tau[it.ti].total_log(it.t_seg, it.t_t)
    }

    fn log_ratio(&self, it: &Intersection) -> f64 {
        self.log_sigma(it) - self.log_tau(it)
    }

    fn optimize(&mut self) {
        let k = self.k;
        let inters = &self.inters;
        let sig = &mut self.sig;
        let tau = &mut self.tau;
        for _ in 0..MAX_SWEEPS {
            let before = total_of(sig, tau, inters, k);
            if before <= 1e-30 {
                break;
            }
            for idx in 0..sig.len() {
                optimize_line(true, idx, sig, tau, inters, k);
            }
            for idx in 0..tau.len() {
                optimize_line(false, idx, sig, tau, inters, k);
            }
            let after = total_of(sig, tau, inters, k);
            if before - after <= 1.0e-12 * (1.0 + before) {
                break;
            }
        }
    }

    /// Fix the free common scale of `sigma` and `tau`: `A` is scale
    /// invariant, so the geometric mean of `sqrt(sigma tau) / s_hat` over the
    /// sample points is set to one.
    fn normalize(&mut self, bg: &Background) {
        let mut sum = 0.0;
        let mut count = 0usize;
        for it in &self.inters {
            if let Some(s_hat) = bg.size_at(it.pos) {
                sum += 0.5 * (self.log_sigma(it) + self.log_tau(it)) - s_hat.ln();
                count += 1;
            }
        }
        if count == 0 {
            return;
        }
        let c = sum / count as f64;
        if !c.is_finite() {
            return;
        }
        for line in self.sig.iter_mut().chain(self.tau.iter_mut()) {
            line.log_mult -= c;
        }
    }

    fn analysis(&self, bg: &Background) -> FieldAnalysis {
        let mut max_anisotropy = 0.0f64;
        let mut sum_anisotropy = 0.0f64;
        let mut max_abs_log_ratio = 0.0f64;
        for it in &self.inters {
            max_anisotropy = max_anisotropy.max(aniso_log(self.log_ratio(it), self.k));
            sum_anisotropy += aniso_log(self.log_ratio(it), self.k);
            if let Some(s_hat) = bg.size_at(it.pos) {
                let r = 0.5 * (self.log_sigma(it) + self.log_tau(it)) - s_hat.ln();
                max_abs_log_ratio = max_abs_log_ratio.max(r.abs());
            }
        }
        let samples = self.inters.len();
        FieldAnalysis {
            max_anisotropy,
            mean_anisotropy: if samples > 0 {
                sum_anisotropy / samples as f64
            } else {
                0.0
            },
            samples,
            max_size_ratio: max_abs_log_ratio.exp().min(MAX_ANISOTROPY),
        }
    }
}

/// Build the representative lines, intersect them and optimise the multipliers.
fn build_run(
    points: &[[f64; 3]],
    tris: &[[usize; 3]],
    alpha: &[f64],
    target_size: &[f64],
    k: u32,
) -> Result<(Run, Background), String> {
    let bg = Background::build(points, tris, alpha, Some(target_size))?;
    let diag = bg.diagonal();
    if diag <= 0.0 || !diag.is_finite() {
        return Err("the input points span a degenerate bounding box".to_string());
    }
    let step = diag / TRACE_STEPS as f64;
    let seeds = bg.grid_seeds();
    if seeds.is_empty() {
        return Err("no seed point lies inside the input tessellation".to_string());
    }

    let mut sig: Vec<Line> = Vec::new();
    let mut tau: Vec<Line> = Vec::new();
    for seed in &seeds {
        if let Some(line) = Line::trace(&bg, *seed, Family::Sigma, TRACE_STEPS, step) {
            sig.push(line);
        }
        if let Some(line) = Line::trace(&bg, *seed, Family::Tau, TRACE_STEPS, step) {
            tau.push(line);
        }
    }
    if sig.is_empty() || tau.is_empty() {
        return Err("could not trace representative streamlines of both families".to_string());
    }

    let inters = intersections(&sig, &tau);
    if inters.is_empty() {
        return Err(
            "the representative streamlines of the two families do not intersect".to_string(),
        );
    }

    let mut run = Run {
        sig,
        tau,
        inters,
        k: k.max(1),
    };
    run.optimize();
    Ok((run, bg))
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Analyse the size-function families of a cross field.
///
/// `points` are the vertices of a planar triangle tessellation (`z` is
/// ignored), `tris` its triangles, and `alpha` / `target_size` have one value
/// per vertex. `alpha` is the cross-field angle in radians modulo `pi/2` and is
/// linearly interpolated over each triangle; `target_size` is the desired
/// element size `s_hat` and is also interpolated linearly.
///
/// The function seeds representative streamlines on a regular grid, integrates
/// `sigma` along one family and `tau` along the other (eqs. 4.3/4.4), optimises
/// the per-line multipliers of section 4.11 against `A` of eq. (4.7) with
/// [`DEFAULT_K`], and reports the anisotropy and the size accuracy
/// `sqrt(sigma tau) / s_hat` at the intersections of the two families.
///
/// Returns an error when the arrays are inconsistent, a coordinate or size is
/// not finite, the tessellation is degenerate, or no representative lines or
/// intersections could be produced. The function never panics.
pub fn analyze(
    points: &[[f64; 3]],
    tris: &[[usize; 3]],
    alpha: &[f64],
    target_size: &[f64],
) -> Result<FieldAnalysis, String> {
    analyze_with_k(points, tris, alpha, target_size, DEFAULT_K)
}

/// Like [`analyze`], with an explicit anisotropy exponent `k` (clamped to
/// `1..=8`). `k = 1` is the plain squared anisotropy of eq. (4.7).
pub fn analyze_with_k(
    points: &[[f64; 3]],
    tris: &[[usize; 3]],
    alpha: &[f64],
    target_size: &[f64],
    k: u32,
) -> Result<FieldAnalysis, String> {
    let (mut run, bg) = build_run(points, tris, alpha, target_size, k)?;
    run.normalize(&bg);
    Ok(run.analysis(&bg))
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

/// Regular triangulation of an axis-aligned rectangle.
fn rectangle_mesh(
    nx: usize,
    ny: usize,
    x0: f64,
    y0: f64,
    width: f64,
    height: f64,
) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let nx = nx.max(1);
    let ny = ny.max(1);
    let mut pts = Vec::with_capacity((nx + 1) * (ny + 1));
    for j in 0..=ny {
        for i in 0..=nx {
            let x = x0 + width * i as f64 / nx as f64;
            let y = y0 + height * j as f64 / ny as f64;
            pts.push([x, y, 0.0]);
        }
    }
    let mut tris = Vec::with_capacity(2 * nx * ny);
    for j in 0..ny {
        for i in 0..nx {
            let a = j * (nx + 1) + i;
            let b = a + 1;
            let c = a + nx + 2;
            let d = a + nx + 1;
            tris.push([a, b, c]);
            tris.push([a, c, d]);
        }
    }
    (pts, tris)
}

/// Run the five small experiments reported in the module documentation and
/// return their [`FieldAnalysis`] numbers.
///
/// The cases are: a constant field with constant size; a harmonic uniform
/// field with a linear target size (`k = 1`); `alpha = theta/2` around the
/// origin (a singular point); a uniform size of `2.0` on a rectangle; and the
/// non-harmonic field `alpha = x^2 / 2`.
pub fn demo() -> Result<Vec<(&'static str, FieldAnalysis)>, String> {
    let mut out = Vec::new();

    let (pts, tris) = rectangle_mesh(16, 16, 0.0, 0.0, 1.0, 1.0);
    let alpha = vec![0.0; pts.len()];
    let size = vec![1.0; pts.len()];
    out.push((
        "constant alpha = 0, constant size 1",
        analyze(&pts, &tris, &alpha, &size)?,
    ));

    let (pts, tris) = rectangle_mesh(16, 16, 0.0, 0.0, 2.0, 1.0);
    let alpha = vec![0.37; pts.len()];
    let size: Vec<f64> = pts.iter().map(|p| 1.0 + 0.5 * p[0]).collect();
    out.push((
        "uniform harmonic field, linear target size (k = 1)",
        analyze_with_k(&pts, &tris, &alpha, &size, 1)?,
    ));

    let (pts, tris) = rectangle_mesh(16, 16, -1.0, -1.0, 2.0, 2.0);
    let alpha: Vec<f64> = pts.iter().map(|p| 0.5 * p[1].atan2(p[0])).collect();
    let size = vec![1.0; pts.len()];
    out.push((
        "alpha = theta / 2 around the origin (singular point)",
        analyze(&pts, &tris, &alpha, &size)?,
    ));

    let (pts, tris) = rectangle_mesh(16, 16, 0.0, 0.0, 2.0, 1.0);
    let alpha = vec![0.0; pts.len()];
    let size = vec![2.0; pts.len()];
    out.push((
        "constant alpha = 0, uniform size 2",
        analyze(&pts, &tris, &alpha, &size)?,
    ));

    let (pts, tris) = rectangle_mesh(16, 16, 0.0, 0.0, 2.0, 1.0);
    let alpha: Vec<f64> = pts.iter().map(|p| 0.5 * p[0] * p[0]).collect();
    let size = vec![1.0; pts.len()];
    out.push((
        "non-harmonic alpha = x^2 / 2 (k = 1)",
        analyze_with_k(&pts, &tris, &alpha, &size, 1)?,
    ));

    Ok(out)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn triangle_mesh(
        nx: usize,
        ny: usize,
        x0: f64,
        y0: f64,
        w: f64,
        h: f64,
    ) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
        rectangle_mesh(nx, ny, x0, y0, w, h)
    }

    #[test]
    fn constant_field_has_zero_anisotropy() {
        let (pts, tris) = triangle_mesh(12, 12, 0.0, 0.0, 1.0, 1.0);
        let alpha = vec![0.0; pts.len()];
        let size = vec![1.0; pts.len()];
        let a = analyze(&pts, &tris, &alpha, &size).unwrap();
        println!("constant field: {a:?}");
        assert!(a.max_anisotropy.abs() < 1.0e-12);
        assert!(a.mean_anisotropy.abs() < 1.0e-12);
        assert!(a.samples > 0);
        assert!((a.max_size_ratio - 1.0).abs() < 1.0e-9);
    }

    #[test]
    fn harmonic_uniform_field_with_linear_size_is_isotropic() {
        let (pts, tris) = triangle_mesh(12, 12, 0.0, 0.0, 2.0, 1.0);
        let alpha = vec![0.37; pts.len()];
        let size: Vec<f64> = pts.iter().map(|p| 1.0 + 0.5 * p[0]).collect();

        let a = analyze_with_k(&pts, &tris, &alpha, &size, 1).unwrap();
        println!("uniform field, linear size: {a:?}");
        assert!(a.max_anisotropy < 1.0e-6, "max A = {}", a.max_anisotropy);
        assert!(a.mean_anisotropy < 1.0e-7, "mean A = {}", a.mean_anisotropy);

        // "sigma = tau up to a constant": inspect the optimised profiles.
        let (mut run, bg) = build_run(&pts, &tris, &alpha, &size, 1).unwrap();
        run.normalize(&bg);
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        let mut worst = 0.0f64;
        for it in &run.inters {
            let ls = run.log_sigma(it);
            let lt = run.log_tau(it);
            lo = lo.min(ls);
            hi = hi.max(ls);
            worst = worst.max((ls - lt).abs());
        }
        println!(
            "sigma spread = {:.3e}, worst |ln(sigma/tau)| = {:.3e}",
            hi - lo,
            worst
        );
        assert!(hi - lo < 0.02, "sigma spread = {}", hi - lo);
        assert!(worst < 0.02, "|ln(sigma/tau)| = {worst}");
    }

    #[test]
    fn rotating_field_traces_and_stays_finite() {
        let (pts, tris) = triangle_mesh(12, 12, -1.0, -1.0, 2.0, 2.0);
        let alpha: Vec<f64> = pts.iter().map(|p| 0.5 * p[1].atan2(p[0])).collect();

        let path = trace_streamline(&alpha, &pts, &tris, [0.4, 0.3], 40, 0.02);
        println!("rotating field path length = {}", path.len());
        assert!(path.len() > 10, "only {} points", path.len());
        for p in &path {
            assert!(p[0].is_finite() && p[1].is_finite());
        }
        for w in path.windows(2) {
            let dx = w[1][0] - w[0][0];
            let dy = w[1][1] - w[0][1];
            let d = (dx * dx + dy * dy).sqrt();
            assert!((d - 0.02).abs() < 1.0e-9, "chord {d}");
        }
        for w in path.windows(3) {
            let a = [w[1][0] - w[0][0], w[1][1] - w[0][1]];
            let b = [w[2][0] - w[1][0], w[2][1] - w[1][1]];
            let cos = (a[0] * b[0] + a[1] * b[1]) / (0.02 * 0.02);
            let turn = cos.clamp(-1.0, 1.0).acos();
            assert!(turn < 0.5, "turn {turn}");
        }

        let size = vec![1.0; pts.len()];
        let a = analyze(&pts, &tris, &alpha, &size).unwrap();
        println!("rotating field: {a:?}");
        assert!(a.samples > 0);
        assert!(a.max_anisotropy.is_finite());
        assert!(a.mean_anisotropy.is_finite());
        assert!(a.max_anisotropy > 0.0);
    }

    #[test]
    fn size_ratio_is_reported() {
        let (pts, tris) = triangle_mesh(10, 10, 0.0, 0.0, 2.0, 1.0);
        let alpha = vec![0.0; pts.len()];
        let size = vec![2.0; pts.len()];
        let a = analyze(&pts, &tris, &alpha, &size).unwrap();
        println!("uniform size 2: {a:?}");
        assert!((a.max_size_ratio - 1.0).abs() < 1.0e-9);

        // A growing target size cannot be matched by a constant sigma = tau;
        // the reported worst-case ratio must be larger than one.
        let size: Vec<f64> = pts.iter().map(|p| 1.0 + 0.5 * p[0]).collect();
        let b = analyze_with_k(&pts, &tris, &alpha, &size, 1).unwrap();
        println!("linear size 1 + x/2: {b:?}");
        assert!(b.max_size_ratio > 1.0);
        assert!(b.max_size_ratio.is_finite());
    }

    #[test]
    fn anisotropy_matches_the_definition() {
        assert!(anisotropy(1.0, 1.0, 1).abs() < 1.0e-15);
        assert!((anisotropy(2.0, 1.0, 1) - 0.25).abs() < 1.0e-12);
        assert!((anisotropy(1.0, 2.0, 1) - 0.25).abs() < 1.0e-12);
        assert!((anisotropy(2.0, 1.0, 2) - 0.0625).abs() < 1.0e-12);
        assert!(anisotropy(0.0, 1.0, 1) > 1.0e299);
        assert!(anisotropy(f64::NAN, 1.0, 1) > 1.0e299);
    }

    #[test]
    fn jacobian_and_metric_definitions() {
        let (sigma, tau, alpha): (f64, f64, f64) = (2.0, 3.0, 0.4);
        let c = alpha.cos();
        let s = alpha.sin();
        let k = [[sigma * c, -tau * s], [sigma * s, tau * c]];
        let k_inv = inverse_jacobian(sigma, tau, alpha).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = 0.0;
                for l in 0..2 {
                    sum += k[i][l] * k_inv[l][j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((sum - expected).abs() < 1.0e-12, "K K^-1 at {i},{j}");
            }
        }
        let m = metric(sigma, tau, alpha).unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let mut sum = 0.0;
                for l in 0..2 {
                    sum += k_inv[l][i] * k_inv[l][j];
                }
                assert!((m[i][j] - sum).abs() < 1.0e-12, "metric at {i},{j}");
            }
        }
        let iso = metric(2.0, 2.0, 0.4).unwrap();
        assert!((iso[0][0] - 0.25).abs() < 1.0e-12);
        assert!(iso[0][1].abs() < 1.0e-12);
        assert!((iso[1][1] - 0.25).abs() < 1.0e-12);
        assert!(inverse_jacobian(0.0, 1.0, 0.0).is_none());
    }

    #[test]
    fn tracer_is_straight_in_a_constant_field() {
        let (pts, tris) = triangle_mesh(8, 8, 0.0, 0.0, 1.0, 1.0);
        let alpha = vec![0.0; pts.len()];
        let path = trace_streamline(&alpha, &pts, &tris, [0.5, 0.5], 20, 0.02);
        assert!(path.len() > 1);
        for p in &path {
            assert!((p[1] - 0.5).abs() < 1.0e-12);
        }
        // A start outside the mesh yields an empty path.
        assert!(trace_streamline(&alpha, &pts, &tris, [2.0, 2.0], 5, 0.02).is_empty());
    }

    #[test]
    fn non_harmonic_field_develops_anisotropy() {
        let (pts, tris) = triangle_mesh(16, 16, 0.0, 0.0, 2.0, 1.0);
        let alpha: Vec<f64> = pts.iter().map(|p| 0.5 * p[0] * p[0]).collect();
        let size = vec![1.0; pts.len()];

        let (mut run, _bg) = build_run(&pts, &tris, &alpha, &size, 1).unwrap();
        let optimized = total_of(&run.sig, &run.tau, &run.inters, 1);
        for line in run.sig.iter_mut().chain(run.tau.iter_mut()) {
            line.log_mult = 0.0;
        }
        let baseline = total_of(&run.sig, &run.tau, &run.inters, 1);
        println!(
            "non-harmonic field: baseline sum A = {baseline:.6e}, optimised = {optimized:.6e}"
        );
        assert!(run.inters.len() > 50);
        assert!(optimized < 0.1 * baseline);

        let a = analyze_with_k(&pts, &tris, &alpha, &size, 1).unwrap();
        println!("non-harmonic field: {a:?}");
        assert!(a.max_anisotropy > 1.0e-6);
        assert!(a.mean_anisotropy > 0.0);
    }

    #[test]
    fn isotropic_gradient_matches_the_thesis_example() {
        // sigma = (4/3) r^(1/4) and alpha = atan(y/x)/4 at (x, y) = (2, 0):
        // grad(alpha) = (-y, x)/(4 r^2) = (0, 1/8) and, from (4.5)/(4.6),
        // grad(sigma) = (1/3) r^(-7/4) (x, y).
        let (x, y) = (2.0f64, 0.0f64);
        let r = (x * x + y * y).sqrt();
        let sigma = (4.0 / 3.0) * r.powf(0.25);
        let grad_alpha = [-y / (4.0 * r * r), x / (4.0 * r * r)];
        let g = isotropic_scale_gradient(sigma, grad_alpha);
        let expected = (1.0 / 3.0) * r.powf(-1.75);
        println!("isotropic gradient = {g:?}, expected ({}, 0)", expected * x);
        assert!((g[0] - expected * x).abs() < 1.0e-12);
        assert!(g[1].abs() < 1.0e-12);
    }

    #[test]
    fn demo_reports_the_chapter_four_numbers() {
        let rows = demo().unwrap();
        assert_eq!(rows.len(), 5);
        for (name, a) in &rows {
            println!(
                "{name}: max A = {:.6e}, mean A = {:.6e}, samples = {}, size ratio = {:.6}",
                a.max_anisotropy, a.mean_anisotropy, a.samples, a.max_size_ratio
            );
            assert!(a.samples > 0);
            assert!(a.max_anisotropy.is_finite());
            assert!(a.mean_anisotropy.is_finite());
            assert!(a.max_size_ratio >= 1.0 - 1.0e-12);
        }
        assert!(rows[0].1.max_anisotropy.abs() < 1.0e-12);
        assert!(rows[1].1.max_anisotropy < 1.0e-6);
    }
}
