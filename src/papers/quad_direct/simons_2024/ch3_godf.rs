//! Size-aware cross fields on a planar triangle mesh (Simons 2024, Chapter 3).
//!
//! This module follows the *Globally Optimal Direction Fields* construction of
//! Knöppel et al. as adapted by Simons, *Size-Driven All-Quad Meshing for
//! Geographic Data* (2024), Sections 3.2.5 and 3.3. The size function is used
//! as a conformal Riemannian metric `g = sigma^2 I`; parallel transport over a
//! mesh edge therefore induces a rotation, which enters the finite-element
//! matrices as the complex coefficients `r_pq` (Algorithm 1, lines 1-11).
//!
//! The cross field is stored at every vertex as the fourth power `u_i` of a
//! direction, i.e. `u_i = exp(i * 4 * theta_i)`. The smooth field is the
//! eigenvector of the generalized Hermitian eigenproblem
//!
//! ```text
//! A u = lambda M u
//! ```
//!
//! belonging to the smallest eigenvalue of the shifted pencil `A + 1e-8 M`
//! (Algorithm 1, line 33). The eigenvector is obtained by shifted inverse
//! power iteration; every inner linear solve is a complex Hermitian
//! conjugate-gradient solve. Both matrices are stored as a real diagonal plus
//! one complex value per mesh edge, so the storage is `O(|V| + |E|)` and a
//! matrix-vector product is `O(|E|)`.
//!
//! # Faithful to the thesis
//!
//! * Transport coefficients exactly as Algorithm 1, lines 2-11: `sigma` at the
//!   midpoint of the edge, `g = (g_i + g_j)/2`, `S = p_j - p_i`,
//!   `dVx = (gx Sx + gy Sy)/sigma`, `dVy = (gy Sx - gx Sy)/sigma`,
//!   `theta = atan2(dVy, 1 + dVx)`, `rho = -n theta` with `n = 4`, and
//!   `r = cos(rho) + i sin(rho)` (eqs. 3.4 and 3.5).
//! * `M` and `A` are Hermitian and are accumulated triangle by triangle; the
//!   off-diagonal entry `(p, q)` of a counter-clockwise triangle is
//!   `conj(r_pq)` times the real (or complex, for the accurate path) triangle
//!   integral, as printed on p. 78 of the thesis and in GODF Eqs. 17-18.
//! * The `10^-8 M` regularization of line 33 is used.
//! * The holomorphic / anti-holomorphic family of Eq. 18 (GODF Eq. 18) is
//!   implemented through [`Options::holomorphic`], the parameter `s`:
//!
//!   ```text
//!   A_ii = Delta_ii - s (Omega_ijk / |t_ijk|) M_ii
//!   A_jk = Delta_jk - s (Omega_ijk / |t_ijk|) M_jk + s eps_jk i conj(r_jk) / 2
//!   ```
//!
//!   with `eps_jk = +1` for the counter-clockwise edge `j -> k` of a
//!   counter-clockwise triangle (the orientation used internally). `s = 0` is
//!   the Dirichlet energy (`A = Delta`, the historical behaviour), `s = 1` is
//!   the holomorphic energy (fewer singularities) and `s = -1` the
//!   anti-holomorphic energy; `s` is validated to lie in `[-1, 1]`. The
//!   `i conj(r_jk)/2` term is the boundary term of the anti-holomorphic minus
//!   holomorphic energy; it cancels on interior edges and survives on the mesh
//!   boundary, as in GODF Section 6.1.1.
//! * The vector associated with the smallest eigenvalue of the shifted pencil
//!   is computed; the phase of the field is the arbitrary global phase that the
//!   thesis describes in Section 3.3.1.
//! * The direction at a query point is the barycentric interpolation of the
//!   complex vertex values followed by `arg(u)/4` (Section 3.3.1); this is what
//!   [`interp_angle`] does.
//! * Singularities are counted with the triangle holonomy (Section 3.1.6):
//!   the winding of `exp(i 4 theta)` around each triangle is `0` or `+/-1`,
//!   giving valences 4, 3 or 5.
//!
//! # Guidance / alignment (Section 3.3)
//!
//! [`solve_with_guidance`] biases the field toward a per-vertex target. The
//! target angles `g_i` use the same representation as the returned field (the
//! cross-field angle `theta`, whose fourth power is the complex value), so the
//! complex target is `G_i = exp(4 i g_i)`. With weight `s >= 0` the guided
//! field solves the normal equations
//!
//! ```text
//! (A + s M) u = s M G.
//! ```
//!
//! Equivalently, `s M` is added to the energy matrix and the right-hand side
//! of the generalized problem is `s b` with `b = M G`. These are the normal
//! equations of the soft alignment energy
//!
//! ```text
//! E(u) = u^H A u + s ||u - G||_M^2,
//! ```
//!
//! so `s = 0` recovers the unguided eigenproblem of [`solve`], and
//! `s -> infinity` recovers `u = G`. GODF (Eq. 16 of the paper) instead uses a
//! Lagrange multiplier, `(A - lambda_t M) u = M q`, with a norm constraint;
//! the thesis does not print an alignment line in Algorithm 1, and this module
//! implements the soft-penalty form requested for it. The guidance weight is
//! unrelated to the holomorphic parameter `s` of Eq. 18.
//!
//! # Boundary constraints (Section 3.3 and feature boundaries)
//!
//! The unconstrained problem uses the homogeneous Neumann condition of Section
//! 3.3 (`du/dn = 0`), which is what [`BoundaryMode::None`] (the default)
//! implements. [`BoundaryMode::Tangent`] adds an inhomogeneous Dirichlet
//! condition at every boundary vertex: `u_i = G_i` with
//! `G_i = exp(4 i theta_i)` and `theta_i` the average of the tangents of the
//! incident boundary edges, averaged in the fourth-power representation (so the
//! two perpendicular tangents of a square corner are identical). The interior
//! minimises the same Dirichlet energy subject to that data. Because the data
//! is inhomogeneous, the generalized eigenproblem becomes a linear
//! energy-minimisation problem; it is solved by eliminating the boundary
//! unknowns exactly (their columns move to the right-hand side and their rows
//! become the identity). [`BoundaryMode::Weak`]`(w)` is the penalty version:
//! `w M_ii (u_i - G_i)` is added at each boundary vertex, i.e. `w M_ii` on the
//! diagonal and `w M_ii G_i` on the right-hand side. `Weak(0)` is exactly
//! `None`; large `w` approaches `Tangent` (measured max difference
//! `1.8e-5 rad` at `w = 1e6`). Guidance may be combined with either mode: the
//! boundary rows win on the boundary, and the soft guidance term acts on the
//! interior. An error is returned when a boundary mode is requested on a mesh
//! with no boundary edges.
//!
//! # Anisotropy (Section 4.10)
//!
//! [`Options::anisotropy`] is the ratio `a = sigma/tau` and
//! [`Options::anisotropy_angle`] is the direction `phi` of the `sigma` axis.
//! The field is aligned by the soft energy
//!
//! ```text
//! E_aniso(u) = w(a) ||u - exp(4 i phi)||_M^2,
//! w(a)       = (a + 1/a - 2)^2.
//! ```
//!
//! The bracket is the inner term of Eq. 4.7 with `k = 1`; the weight is a
//! modelling choice. It is added to the same normal equations as guidance, so
//! `a = 1` gives `w = 0` and exactly the historical field, while `a = 4` adds
//! `w ~ 5.06`. Because the alignment term is inhomogeneous the solve is a
//! single linear system, not an eigenproblem.
//!
//! *Deviation.* Chapter 4's anisotropy is an *output* measure of a cross field,
//! not an input to the Chapter 3 solve. I derived the Levi-Civita connection
//! for the anisotropic metric `g = diag(sigma^2, sigma^2/a^2)`; in the
//! principal frame its transport generalises Eq. 3.5 to
//! `dVy = a^2 uy Sx - ux Sy` (reducing exactly to Eq. 3.5 at `a = 1`). That
//! extension is **not** used. A constant anisotropic metric has zero Dirichlet
//! energy for every constant cross field, so transport alone cannot break the
//! global-phase degeneracy, and on the coarse stretched-rectangle test the
//! `a^2` term produced strong boundary-layer distortion. The transport is
//! therefore exactly the thesis's Eq. 3.5, and anisotropy enters only through
//! the alignment energy above.
//!
//! # Adaptive background refinement (Sections 3.2.5.1 and 3.4.2)
//!
//! [`Options::adaptive`]` = theta_max` enables one round. After the first
//! solve, a triangle is flagged when the maximum 4-RoSy angle difference over
//! its three vertices exceeds `theta_max`. All edges of flagged triangles are
//! split (red-green: 1, 2 or 3 split edges give 2, 3 or 4 children), the size
//! is linearly interpolated at the midpoints, guidance targets are interpolated
//! in the fourth-power representation, and the field is re-solved on the
//! refined mesh. Original vertices keep their indices, so the returned values
//! at them are the refined solution evaluated there (no further interpolation
//! is needed).
//!
//! *Simplification.* The thesis's predicate is a bound on the rotation that a
//! pair of vertices would induce (from the size gradient) during background
//! mesh generation, plus a curvature estimate. Here the variation of the
//! *solved* field over each triangle is used (as requested), and only a single
//! refinement round is performed. The refined mesh is not fed back into the
//! downstream QGP extraction.
//!
//! # Accurate mass and Dirichlet integrals
//!
//! The flat-triangle limits quoted by the thesis are
//!
//! ```text
//! M_ii = |t|/6,                 M_ij = conj(r_ij) |t|/12
//! A_ii = |p_jk|^2 / (4|t|)
//! A_ij = conj(r_ij) <p_ki, p_jk> / (4|t|)
//! ```
//!
//! with `p_jk = p_k - p_j`, `p_ki = p_i - p_k` and cyclic permutations. They
//! are used when [`Options::accurate`] is `false`.
//!
//! With `accurate = true` the curvature-aware helper integrals of GODF
//! Appendix D are used (Appendix D of the thesis's reference [KCPS13]; the
//! thesis quotes them as Eqs. 17-18). Let `Omega = arg(r_ij r_jk r_ki)` be the
//! triangle holonomy, `r_pq = exp(i rho_pq)` the transport coefficient on the
//! counter-clockwise edge `p -> q`, and
//!
//! ```text
//! g11 = |p_p - p_r|^2,  g12 = <p_p - p_r, p_q - p_r>,  g22 = |p_q - p_r|^2
//! ```
//!
//! for the ordered pair `(p, q)` with third vertex `r`. Then
//!
//! ```text
//! MassIJ(Omega) = (6 e^{i Omega} - 6 - 6 i Omega + 3 Omega^2 + i Omega^3)
//!                 / (3 Omega^4),
//! M_pq          = conj(r_pq) |t| MassIJ(Omega),
//!
//! DirichletIJ(Omega, g11, g12, g22) =
//!     [ (3 g11 + 4 g12 + 3 g22)
//!       + i Omega (g11 + g12 + g22)
//!       - i Omega^3 g12 / 6
//!       + Omega^4 (g11 - 2 g12 + g22) / 24
//!       - i Omega^5 (g11 - 2 g12 + g22) / 60
//!       + ( -(3 g11 + 4 g12 + 3 g22)
//!           + i Omega (2 g11 + 3 g12 + 2 g22)
//!           + Omega^2 (g11 + 2 g12 + g22) / 2 ) e^{i Omega} ] / Omega^4,
//! A_pq          = conj(r_pq) DirichletIJ(Omega, g11, g12, g22) / |t|,
//!
//! Delta_cc      = [ |p_uv|^2
//!                   + Omega^2 (|p_cu|^2 + <p_cu, p_cv> + |p_cv|^2) / 90 ]
//!                 / (4 |t|),
//! ```
//!
//! where `c` is the centre vertex and `u, v` the other two vertices in
//! counter-clockwise order. `MassIJ` and `DirichletIJ` have the removable
//! singularities `MassIJ(0) = 1/12` and
//! `DirichletIJ(0, g11, g12, g22) = -g12/4`; both are evaluated with a Taylor
//! series for `|Omega| < 0.1` and with the closed forms otherwise.
//!
//! # Deviations (documented on purpose)
//!
//! * **Guidance form.** The soft penalty above is used instead of GODF's
//!   Lagrange-multiplier alignment. The thesis only asks for a field that can
//!   be biased toward a target, not for a norm-constrained alignment solve.
//! * **Vertex gradients.** Algorithm 1 consumes a per-vertex gradient `g_i`.
//!   It is obtained here as the area-weighted average of the constant
//!   gradients of the incident triangles. A uniform average would change the
//!   coefficients only slightly.
//! * **Shift scaling.** Line 33 literally adds `10^-8 M`. To remain meaningful
//!   on meshes of very different length scales the same relative shift
//!   `1e-8 * trace(A)/trace(M)` is used; it is `1e-8` for a unit-scale mesh.
//! * **Eigen-solve.** GODF/SciPy use a sparse shift-invert Arnoldi solver. Here
//!   the smallest eigenpair of the shifted pencil is found with shifted inverse
//!   power iteration and Jacobi-preconditioned complex CG. On the constant-size
//!   test the smallest eigenvector is the constant/harmonic mode (`lambda = 0`
//!   before the shift), which is the smooth constant field; the shift makes
//!   that eigenvalue `1e-8` rather than exactly zero, so the returned
//!   eigenpair is the smallest of the shifted pencil, as in Algorithm 1.
//! * **Guided / constrained iteration budget.** Guidance, anisotropy and weak
//!   boundary penalties make the problem a single linear solve, not an
//!   eigen-iteration. `iterations` still has to be positive; the
//!   conjugate-gradient cap is `max(400 + 8|V|, iterations)`, bounded by 6000.
//! * **Boundary Dirichlet solve.** `Tangent` eliminates the boundary unknowns
//!   instead of adding a finite penalty. The result is the exact minimiser of
//!   the Dirichlet energy with the prescribed tangent data, not the smallest
//!   eigenpair of the original pencil (which is not defined for inhomogeneous
//!   data). Singularities are then counted exactly as for every other solve.
//! * **Anisotropy.** The thesis does not feed anisotropy into the field solve,
//!   and its conformal transport cannot choose a global phase. A soft alignment
//!   energy is used instead; the anisotropic transport formula is documented
//!   above but not applied. This is the main deviation of this module.
//! * **Adaptive predicate.** The refinement test uses the solved field's
//!   variation over a triangle, not the size-gradient rotation bound of Section
//!   3.2.5.1; only one round is performed.
//! * **Accurate integral evaluation.** The closed forms lose precision near
//!   `Omega = 0`; a Taylor series (terms through `Omega^8`) is used for
//!   `|Omega| < 0.1`, which is far below the accuracy of the double-precision
//!   closed form on either side of the switch.
//! * **Holomorphic parameter.** [`Options::holomorphic`] is validated to lie in
//!   `[-1, 1]`, the range GODF gives for `s`, and non-finite values are
//!   rejected. The `M_jk` used in the Eq. 18 correction is the same mass entry
//!   as the rest of the assembly (the accurate `MassIJ` when
//!   [`Options::accurate`] is set, the flat `1/12` otherwise).
//!
//! # Not implemented
//!
//! Iterative multi-round adaptive refinement, the anisotropic transport derived
//! above, singularity placement, the extraction of a quad mesh, and the
//! compatibility analysis of Chapter 4 (the `sigma`/`tau` fields implied by a
//! cross field). The input is assumed planar: `z` is ignored and only `x, y`
//! are used.

use std::collections::hash_map::Entry;
use std::collections::HashMap;

/// Order of the rotational symmetry: a cross field is a 4-RoSy field.
const FIELD_SYMMETRY: f64 = 4.0;
/// Relative regularization added to the energy matrix (Algorithm 1, line 33).
const ENERGY_SHIFT: f64 = 1e-8;
/// Relative tolerance of the outer (inverse power) iteration.
const OUTER_TOLERANCE: f64 = 1e-8;
/// Relative tolerance of the inner conjugate-gradient solve.
const INNER_TOLERANCE: f64 = 1e-9;
/// Below this holonomy the helper integrals are evaluated by Taylor series.
const INTEGRAL_SERIES_THRESHOLD: f64 = 0.1;

/// How the field is constrained on the mesh boundary.
///
/// The thesis uses homogeneous Neumann conditions for the unconstrained
/// Helmholtz problem (Section 3.3). When a boundary is a *feature* (a plate
/// edge, a hole), the field should instead have one of its four axes along the
/// boundary tangent; this enum selects that constraint.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum BoundaryMode {
    /// No boundary constraint: the historical homogeneous-Neumann field.
    #[default]
    None,
    /// Fix every boundary vertex to the boundary tangent (a Dirichlet
    /// constraint). The field at a boundary vertex is `exp(4 i theta)` with
    /// `theta` the tangent angle, taken modulo `pi/2`. The interior is the
    /// minimum-energy extension of that data.
    Tangent,
    /// Penalty form: add `weight * diag(M)` to the energy diagonal on boundary
    /// vertices and `weight * diag(M) * G` to the right-hand side, with `G` the
    /// tangent target. Large weight approaches [`BoundaryMode::Tangent`];
    /// `weight = 0` is the same as [`BoundaryMode::None`]. The penalty is
    /// relative to the (lumped) mass matrix, so it has the same scale as the
    /// Dirichlet energy on meshes of any element size.
    Weak(f64),
}

/// Options for [`solve_with_options`].
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Use the curvature-aware `MassIJ`/`DirichletIJ` integrals of GODF
    /// Appendix D (quoted by the thesis as Eqs. 17-18) instead of their
    /// flat-triangle limits. The default, `false`, is the flat limit used by
    /// the original implementation.
    pub accurate: bool,
    /// Holomorphic parameter `s` of GODF Eq. 18 (the thesis's Eq. 18).
    ///
    /// `s = 0` (the default) is the Dirichlet energy `A = Delta`, i.e. the
    /// historical field; `s = 1` is the holomorphic energy (fewer
    /// singularities); `s = -1` is the anti-holomorphic energy. Values outside
    /// `[-1, 1]` and non-finite values are rejected by the solvers.
    pub holomorphic: f64,
    /// Anisotropy ratio `sigma / tau` (Section 4.10 of the thesis; `a >= 1`
    /// means the `sigma` axis is longer). The default, `1`, is the isotropic
    /// metric of Chapter 3 and reproduces the historical field.
    ///
    /// The ratio adds a soft alignment energy toward the frame rotated by
    /// [`Options::anisotropy_angle`], with weight `(a + 1/a - 2)^2`; the exact
    /// formula is in the module documentation. `a = 1` is exactly isotropic
    /// and reproduces the historical field. The anisotropic transport formula
    /// is documented there too, but it is not applied (see the deviation).
    /// Non-finite and non-positive values are rejected.
    pub anisotropy: f64,
    /// Direction (radians, counter-clockwise from `+x`) of the `sigma` axis of
    /// the anisotropic metric. The default, `0`, is the `+x` axis. Ignored
    /// when [`Options::anisotropy`] is `1`. Non-finite values are rejected.
    pub anisotropy_angle: f64,
    /// Boundary constraint (the thesis's homogeneous Neumann condition is the
    /// default).
    pub boundary: BoundaryMode,
    /// Field-variation threshold (radians of cross-field angle) for one round
    /// of adaptive background refinement (Sections 3.2.5.1 and 3.4.2). A
    /// triangle is refined (red-green) when the maximum 4-RoSy angle
    /// difference over its three vertices exceeds this value; the field is
    /// then re-solved on the refined mesh. `0` (the default) disables
    /// refinement. Non-finite and negative values are rejected.
    pub adaptive: f64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            accurate: false,
            holomorphic: 0.0,
            anisotropy: 1.0,
            anisotropy_angle: 0.0,
            boundary: BoundaryMode::None,
            adaptive: 0.0,
        }
    }
}

/// A per-vertex target cross field for [`solve_with_guidance`].
///
/// `angle[i]` is the target cross-field angle at vertex `i`, in the same
/// representation as the angles returned by the solver (a rotation of the
/// cross, with `arg(u_i) = 4 * angle[i]`). Values are taken modulo `pi/2`.
/// `weight` is the non-negative soft-alignment weight `s`; `weight = 0`
/// reproduces the unguided field.
#[derive(Clone, Debug)]
pub struct Guidance {
    /// Non-negative alignment weight `s`.
    pub weight: f64,
    /// One target angle per vertex, in radians.
    pub angle: Vec<f64>,
}

/// A complex number. The public API deals only in angles, so this stays private.
#[derive(Clone, Copy, Debug, Default)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    const ZERO: Self = Self { re: 0.0, im: 0.0 };
    const ONE: Self = Self { re: 1.0, im: 0.0 };

    fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    fn from_angle(angle: f64) -> Self {
        Self {
            re: angle.cos(),
            im: angle.sin(),
        }
    }

    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }

    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    fn scale(self, factor: f64) -> Self {
        Self {
            re: self.re * factor,
            im: self.im * factor,
        }
    }

    fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    fn argument(self) -> f64 {
        self.im.atan2(self.re)
    }
}

/// Hermitian inner product `sum_i conj(a_i) b_i`.
fn dot(a: &[Complex], b: &[Complex]) -> Complex {
    let mut sum = Complex::ZERO;
    for (ai, bi) in a.iter().zip(b.iter()) {
        sum = sum.add(ai.conj().mul(*bi));
    }
    sum
}

fn euclidean_norm(values: &[Complex]) -> f64 {
    dot(values, values).re.max(0.0).sqrt()
}

/// The high-accuracy `MassIJ` integral of GODF Appendix D (thesis Eq. 17).
///
/// The flat limit is `1/12`. The closed form has a removable `0/0` at
/// `Omega = 0`, so a Taylor series through `Omega^8` is used for small `Omega`.
fn mass_integral(omega: f64) -> Complex {
    if omega.abs() < INTEGRAL_SERIES_THRESHOLD {
        let o2 = omega * omega;
        let o4 = o2 * o2;
        let o6 = o4 * o2;
        let o8 = o4 * o4;
        let re = 1.0 / 12.0 - o2 / 360.0 + o4 / 20160.0 - o6 / 1814400.0 + o8 / 239_500_800.0;
        let im = omega * (1.0 / 60.0 - o2 / 2520.0 + o4 / 181_440.0 - o6 / 19_958_400.0);
        Complex::new(re, im)
    } else {
        let phase = Complex::from_angle(omega);
        let numerator = phase
            .scale(6.0)
            .sub(Complex::new(6.0, 0.0))
            .sub(Complex::new(0.0, 6.0 * omega))
            .add(Complex::new(3.0 * omega * omega, 0.0))
            .add(Complex::new(0.0, omega * omega * omega));
        numerator.scale((3.0 * omega * omega * omega * omega).recip())
    }
}

/// The high-accuracy `DirichletIJ` integral of GODF Appendix D (thesis
/// Eq. 18), with `g11`, `g12`, `g22` defined for the ordered edge.
///
/// The flat limit is `-g12/4`. A Taylor series through `Omega^8` is used for
/// small `Omega` to avoid catastrophic cancellation in the closed form.
fn dirichlet_integral(omega: f64, g11: f64, g12: f64, g22: f64) -> Complex {
    if omega.abs() < INTEGRAL_SERIES_THRESHOLD {
        let o2 = omega * omega;
        let o4 = o2 * o2;
        let o6 = o4 * o2;
        let o8 = o4 * o4;
        let edges = g11 + g22;
        let re = -0.25 * g12
            + edges * (o2 / 120.0 - o4 / 2688.0 + o6 / 129_600.0 - o8 / 10_644_480.0)
            + g12 * (o2 / 45.0 - o4 / 1120.0 + o6 / 56_700.0 - o8 / 4_790_016.0);
        let im = omega
            * (edges * (-1.0 / 24.0 + o2 / 504.0 - o4 / 17_280.0 + o6 / 1_108_800.0)
                + g12 * (-1.0 / 24.0 + 5.0 * o2 / 1008.0 - 7.0 * o4 / 51_840.0 + o6 / 492_800.0));
        Complex::new(re, im)
    } else {
        let o2 = omega * omega;
        let o3 = o2 * omega;
        let o4 = o2 * o2;
        let o5 = o4 * omega;
        let a = 3.0 * g11 + 4.0 * g12 + 3.0 * g22;
        let b = g11 + g12 + g22;
        let c = g11 - 2.0 * g12 + g22;
        let d = g11 + 2.0 * g12 + g22;
        let e = 2.0 * g11 + 3.0 * g12 + 2.0 * g22;
        let tail = Complex::new(-a + 0.5 * o2 * d, omega * e).mul(Complex::from_angle(omega));
        let bracket = Complex::new(
            a + o4 * c / 24.0,
            omega * b - o3 * g12 / 6.0 - o5 * c / 60.0,
        )
        .add(tail);
        bracket.scale(o4.recip())
    }
}

/// Real diagonal plus one complex value for every edge, for a Hermitian matrix.
///
/// Entry `(a, b)` of edge `e` is `off[e]`; entry `(b, a)` is `conj(off[e])`.
struct Hermitian {
    size: usize,
    diagonal: Vec<f64>,
    edge_a: Vec<usize>,
    edge_b: Vec<usize>,
    off_diagonal: Vec<Complex>,
}

impl Hermitian {
    fn zero(size: usize, edge_a: Vec<usize>, edge_b: Vec<usize>) -> Self {
        let edges = edge_a.len();
        Self {
            size,
            diagonal: vec![0.0; size],
            edge_a,
            edge_b,
            off_diagonal: vec![Complex::ZERO; edges],
        }
    }

    /// `self + shift * mass`, with both matrices sharing the same edge list.
    fn shifted(&self, shift: f64, mass: &Self) -> Result<Self, String> {
        if self.size != mass.size || self.edge_a != mass.edge_a || self.edge_b != mass.edge_b {
            return Err("internal error: mass and energy matrices have different sparsity".into());
        }
        let mut diagonal = vec![0.0; self.size];
        for i in 0..self.size {
            diagonal[i] = self.diagonal[i] + shift * mass.diagonal[i];
        }
        let mut off_diagonal = Vec::with_capacity(self.off_diagonal.len());
        for e in 0..self.off_diagonal.len() {
            off_diagonal.push(self.off_diagonal[e].add(mass.off_diagonal[e].scale(shift)));
        }
        Ok(Self {
            size: self.size,
            diagonal,
            edge_a: self.edge_a.clone(),
            edge_b: self.edge_b.clone(),
            off_diagonal,
        })
    }

    fn apply(&self, x: &[Complex]) -> Vec<Complex> {
        let mut y: Vec<Complex> = (0..self.size)
            .map(|i| x[i].scale(self.diagonal[i]))
            .collect();
        for e in 0..self.off_diagonal.len() {
            let a = self.edge_a[e];
            let b = self.edge_b[e];
            y[a] = y[a].add(self.off_diagonal[e].mul(x[b]));
            y[b] = y[b].add(self.off_diagonal[e].conj().mul(x[a]));
        }
        y
    }

    fn trace(&self) -> f64 {
        self.diagonal.iter().sum()
    }
}

fn sub2(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn dot2(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

fn cross2(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

fn rotate_90(a: [f64; 2]) -> [f64; 2] {
    [-a[1], a[0]]
}

fn scale2(a: [f64; 2], factor: f64) -> [f64; 2] {
    [a[0] * factor, a[1] * factor]
}

/// The `Delta_cc` diagonal of the Dirichlet matrix (GODF Eq. 18).
///
/// `center`, `next` and `previous` are consecutive vertices in
/// counter-clockwise order; `area` is the triangle area. The `Omega^2` term is
/// the one printed by the thesis (`|p_uv|^2` plus the three squared/dot edge
/// terms over 90).
fn dirichlet_diagonal(
    omega: f64,
    area: f64,
    center: [f64; 2],
    next: [f64; 2],
    previous: [f64; 2],
) -> f64 {
    let p_cu = sub2(next, center);
    let p_cv = sub2(previous, center);
    let opposite = sub2(previous, next);
    let numerator = dot2(opposite, opposite)
        + omega * omega * (dot2(p_cu, p_cu) + dot2(p_cu, p_cv) + dot2(p_cv, p_cv)) / 90.0;
    numerator / (4.0 * area)
}

/// Geometry of one triangle, with its vertices already ordered counter-clockwise.
#[derive(Clone, Copy, Debug)]
struct TriangleGeometry {
    vertices: [usize; 3],
    positions: [[f64; 2]; 3],
    area: f64,
}

/// Reads one triangle, reorders it counter-clockwise and rejects degenerate ones.
fn triangle_geometry(
    points: &[[f64; 3]],
    triangle: &[usize; 3],
    index: usize,
) -> Result<TriangleGeometry, String> {
    let mut vertices = *triangle;
    let mut positions = [
        [points[vertices[0]][0], points[vertices[0]][1]],
        [points[vertices[1]][0], points[vertices[1]][1]],
        [points[vertices[2]][0], points[vertices[2]][1]],
    ];
    let twice_signed_area = cross2(
        sub2(positions[1], positions[0]),
        sub2(positions[2], positions[0]),
    );
    if twice_signed_area < 0.0 {
        vertices.swap(1, 2);
        positions.swap(1, 2);
    }
    let area = 0.5 * twice_signed_area.abs();
    let longest = dot2(
        sub2(positions[1], positions[0]),
        sub2(positions[1], positions[0]),
    )
    .max(dot2(
        sub2(positions[2], positions[1]),
        sub2(positions[2], positions[1]),
    ))
    .max(dot2(
        sub2(positions[0], positions[2]),
        sub2(positions[0], positions[2]),
    ));
    if !area.is_finite() || area <= 1e-14 * longest {
        return Err(format!(
            "triangle {index} is degenerate (area {area}, longest squared edge {longest})"
        ));
    }
    Ok(TriangleGeometry {
        vertices,
        positions,
        area,
    })
}

/// Constant gradient of the linear interpolation of `sigma` over a triangle.
///
/// For a counter-clockwise triangle the hat-function gradients are
/// `rot90(p_k - p_j) / (2|t|)` and cyclic permutations.
fn triangle_gradient(positions: &[[f64; 2]; 3], sigma: [f64; 3], area: f64) -> [f64; 2] {
    let inverse_two_area = (2.0 * area).recip();
    let g0 = scale2(
        rotate_90(sub2(positions[2], positions[1])),
        inverse_two_area,
    );
    let g1 = scale2(
        rotate_90(sub2(positions[0], positions[2])),
        inverse_two_area,
    );
    let g2 = scale2(
        rotate_90(sub2(positions[1], positions[0])),
        inverse_two_area,
    );
    [
        sigma[0] * g0[0] + sigma[1] * g1[0] + sigma[2] * g2[0],
        sigma[0] * g0[1] + sigma[1] * g1[1] + sigma[2] * g2[1],
    ]
}

/// The transport coefficient `r_ab` for the canonical edge `a < b`.
///
/// Implements Algorithm 1, lines 2-11 (eqs. 3.4 and 3.5).
fn transport_coefficient(
    a: usize,
    b: usize,
    points: &[[f64; 3]],
    size: &[f64],
    gradient: &[[f64; 2]],
) -> Result<Complex, String> {
    let sigma = 0.5 * (size[a] + size[b]);
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(format!(
            "edge ({a}, {b}) has non-positive average size {sigma}"
        ));
    }
    let g = [
        0.5 * (gradient[a][0] + gradient[b][0]),
        0.5 * (gradient[a][1] + gradient[b][1]),
    ];
    let s = [points[b][0] - points[a][0], points[b][1] - points[a][1]];
    let dvx = (g[0] * s[0] + g[1] * s[1]) / sigma;
    let dvy = (g[1] * s[0] - g[0] * s[1]) / sigma;
    let theta = dvy.atan2(1.0 + dvx);
    let rho = -FIELD_SYMMETRY * theta;
    Ok(Complex::from_angle(rho))
}

/// Validates the public inputs and returns the number of vertices.
fn validate(
    points: &[[f64; 3]],
    tris: &[[usize; 3]],
    size: &[f64],
    iterations: usize,
) -> Result<usize, String> {
    if iterations == 0 {
        return Err("iterations must be positive".into());
    }
    if points.is_empty() {
        return Err("the mesh has no vertices".into());
    }
    if tris.is_empty() {
        return Err("the mesh has no triangles".into());
    }
    if size.len() != points.len() {
        return Err(format!(
            "size has {} values but the mesh has {} vertices",
            size.len(),
            points.len()
        ));
    }
    for (i, point) in points.iter().enumerate() {
        if !point[0].is_finite() || !point[1].is_finite() {
            return Err(format!("point {i} has a non-finite x or y coordinate"));
        }
    }
    for (i, value) in size.iter().enumerate() {
        if !value.is_finite() || *value <= 0.0 {
            return Err(format!("size[{i}] = {value} is not finite and positive"));
        }
    }
    for (t, triangle) in tris.iter().enumerate() {
        for &vertex in triangle.iter() {
            if vertex >= points.len() {
                return Err(format!(
                    "triangle {t} uses vertex {vertex}, but there are only {} vertices",
                    points.len()
                ));
            }
        }
    }
    Ok(points.len())
}

/// Validates a guidance request against a mesh with `vertices` vertices.
fn validate_guidance(vertices: usize, guidance: &Guidance) -> Result<(), String> {
    if !guidance.weight.is_finite() || guidance.weight < 0.0 {
        return Err(format!(
            "guidance weight {} is not finite and non-negative",
            guidance.weight
        ));
    }
    if guidance.angle.len() != vertices {
        return Err(format!(
            "guidance has {} angles but the mesh has {vertices} vertices",
            guidance.angle.len()
        ));
    }
    for (i, angle) in guidance.angle.iter().enumerate() {
        if !angle.is_finite() {
            return Err(format!("guidance angle[{i}] = {angle} is not finite"));
        }
    }
    Ok(())
}

/// Validates the new options.
fn validate_options(options: &Options) -> Result<(), String> {
    if !options.holomorphic.is_finite() || !(-1.0..=1.0).contains(&options.holomorphic) {
        return Err(format!(
            "the holomorphic parameter s = {} is not finite and in [-1, 1]",
            options.holomorphic
        ));
    }
    if !options.anisotropy.is_finite() || options.anisotropy <= 0.0 {
        return Err(format!(
            "the anisotropy ratio a = {} is not finite and positive",
            options.anisotropy
        ));
    }
    if !options.anisotropy_angle.is_finite() {
        return Err(format!(
            "the anisotropy direction {} is not finite",
            options.anisotropy_angle
        ));
    }
    if !options.adaptive.is_finite() || options.adaptive < 0.0 {
        return Err(format!(
            "the adaptive threshold {} is not finite and non-negative",
            options.adaptive
        ));
    }
    match options.boundary {
        BoundaryMode::None | BoundaryMode::Tangent => {}
        BoundaryMode::Weak(weight) => {
            if !weight.is_finite() || weight < 0.0 {
                return Err(format!(
                    "the weak boundary weight {weight} is not finite and non-negative"
                ));
            }
        }
    }
    Ok(())
}

/// Soft alignment weight of the anisotropic metric (Section 4.10).
///
/// Chapter 4 measures anisotropy by `(sigma/tau + tau/sigma - 2)^(2k)` (Eq.
/// 4.7). With `k = 1` the bracket is `a + 1/a - 2`, and this implementation
/// uses its square `(a + 1/a - 2)^2` as the soft weight in the mass inner
/// product. That makes `a = 1` exactly isotropic (weight zero) and gives
/// `a = 4` a weight of about `5.06`, strong enough to overcome the smoothing
/// energy on the test rectangle. The weight is a modelling choice, documented
/// as a deviation because the thesis does not feed anisotropy back into the
/// field solve.
fn anisotropy_alignment_weight(anisotropy: f64) -> f64 {
    if anisotropy == 1.0 {
        return 0.0;
    }
    let measure = anisotropy + anisotropy.recip() - 2.0;
    measure * measure
}

/// Boundary vertices and their complex tangent targets `exp(4 i theta)`.
///
/// An edge is a boundary edge when it belongs to exactly one triangle. The
/// tangent thetas of all boundary edges incident to a vertex are averaged in
/// the fourth-power representation, which is exactly the average that a
/// 4-RoSy field is invariant under (a square corner's two tangents are
/// perpendicular, hence identical after taking the fourth power).
fn boundary_targets(
    points: &[[f64; 3]],
    tris: &[[usize; 3]],
    vertices: usize,
) -> Result<(Vec<bool>, Vec<Complex>), String> {
    let mut counts: HashMap<(usize, usize), usize> = HashMap::new();
    for triangle in tris {
        for (u, v) in [
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ] {
            let key = if u < v { (u, v) } else { (v, u) };
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    let mut is_boundary = vec![false; vertices];
    let mut sum = vec![Complex::ZERO; vertices];
    let mut first = vec![Complex::ONE; vertices];
    let mut has_first = vec![false; vertices];
    let mut edges = 0usize;
    for triangle in tris {
        for (u, v) in [
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ] {
            let key = if u < v { (u, v) } else { (v, u) };
            if counts.get(&key).copied().unwrap_or(0) != 1 {
                continue;
            }
            edges += 1;
            let tangent = (points[v][1] - points[u][1]).atan2(points[v][0] - points[u][0]);
            let value = Complex::from_angle(FIELD_SYMMETRY * tangent);
            for vertex in [u, v] {
                is_boundary[vertex] = true;
                sum[vertex] = sum[vertex].add(value);
                if !has_first[vertex] {
                    has_first[vertex] = true;
                    first[vertex] = value;
                }
            }
        }
    }
    if edges == 0 {
        return Err("the mesh has no boundary edges".into());
    }
    let mut targets = vec![Complex::ZERO; vertices];
    for vertex in 0..vertices {
        if !is_boundary[vertex] {
            continue;
        }
        let norm = (sum[vertex].re * sum[vertex].re + sum[vertex].im * sum[vertex].im).sqrt();
        targets[vertex] = if norm.is_finite() && norm > 1e-12 {
            sum[vertex].scale(norm.recip())
        } else {
            first[vertex]
        };
    }
    Ok((is_boundary, targets))
}

/// Fixes the rows of a Hermitian matrix for an exact Dirichlet condition.
///
/// Off-diagonal entries incident to a boundary vertex are dropped (their
/// contribution to the interior rows is moved to the right-hand side by the
/// caller) and the boundary diagonal is set to one, so the corresponding
/// unknown is exactly its prescribed value.
fn dirichlet_rows(mut matrix: Hermitian, is_boundary: &[bool]) -> Hermitian {
    for edge in 0..matrix.off_diagonal.len() {
        if is_boundary[matrix.edge_a[edge]] || is_boundary[matrix.edge_b[edge]] {
            matrix.off_diagonal[edge] = Complex::ZERO;
        }
    }
    for (vertex, boundary) in is_boundary.iter().enumerate() {
        if *boundary {
            matrix.diagonal[vertex] = 1.0;
        }
    }
    matrix
}

/// Maximum 4-RoSy angle difference over the vertices of a triangle.
fn triangle_field_variation(alpha: &[f64], triangle: &[usize; 3]) -> f64 {
    let mut variation = 0.0f64;
    for (u, v) in [
        (triangle[0], triangle[1]),
        (triangle[1], triangle[2]),
        (triangle[2], triangle[0]),
    ] {
        let a = alpha[u];
        let b = alpha[v];
        let d = (a - b).rem_euclid(std::f64::consts::FRAC_PI_2);
        variation = variation.max(d.min(std::f64::consts::FRAC_PI_2 - d));
    }
    variation
}

/// One round of red-green refinement on the triangles whose field variation
/// exceeds `threshold`.
///
/// Returns the refined mesh, the extended size values, the number of flagged
/// (parent) triangles, and the ordered edge list `(a, b)` of the midpoints (so
/// that a guidance field can be interpolated on the new vertices). The
/// original vertices keep their indices, so the solution on the refined mesh
/// can be read back at them directly (there is nothing to interpolate at an
/// original vertex). A midpoint inherits the average of its edge's endpoint
/// sizes, which is the exact piecewise-linear interpolation.
fn refine_once(
    points: &[[f64; 3]],
    tris: &[[usize; 3]],
    size: &[f64],
    alpha: &[f64],
    threshold: f64,
) -> (
    Vec<[f64; 3]>,
    Vec<[usize; 3]>,
    Vec<f64>,
    usize,
    Vec<(usize, usize)>,
) {
    let mut split_edge: HashMap<(usize, usize), usize> = HashMap::new();
    let mut edge_order: Vec<(usize, usize)> = Vec::new();
    let mut flagged = 0usize;
    for triangle in tris {
        if triangle_field_variation(alpha, triangle) <= threshold {
            continue;
        }
        flagged += 1;
        for (u, v) in [
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ] {
            let key = if u < v { (u, v) } else { (v, u) };
            if let Entry::Vacant(entry) = split_edge.entry(key) {
                entry.insert(usize::MAX);
                edge_order.push(key);
            }
        }
    }

    let mut refined_points = points.to_vec();
    let mut refined_size = size.to_vec();
    for key in &edge_order {
        let index = refined_points.len();
        split_edge.insert(*key, index);
        refined_points.push([
            0.5 * (points[key.0][0] + points[key.1][0]),
            0.5 * (points[key.0][1] + points[key.1][1]),
            0.0,
        ]);
        refined_size.push(0.5 * (size[key.0] + size[key.1]));
    }

    let midpoint = |u: usize, v: usize| -> Option<usize> {
        let key = if u < v { (u, v) } else { (v, u) };
        split_edge
            .get(&key)
            .copied()
            .filter(|&index| index != usize::MAX)
    };

    let mut refined_tris: Vec<[usize; 3]> = Vec::with_capacity(tris.len() * 2);
    for triangle in tris {
        let [a, b, c] = *triangle;
        let m_ab = midpoint(a, b);
        let m_bc = midpoint(b, c);
        let m_ca = midpoint(c, a);
        match (m_ab, m_bc, m_ca) {
            (None, None, None) => refined_tris.push([a, b, c]),
            (Some(p), None, None) => {
                refined_tris.push([a, p, c]);
                refined_tris.push([p, b, c]);
            }
            (None, Some(q), None) => {
                refined_tris.push([a, b, q]);
                refined_tris.push([a, q, c]);
            }
            (None, None, Some(r)) => {
                refined_tris.push([a, b, r]);
                refined_tris.push([r, b, c]);
            }
            (Some(p), Some(q), None) => {
                // Marked edges meet at `b`.
                refined_tris.push([b, p, q]);
                refined_tris.push([p, a, c]);
                refined_tris.push([p, c, q]);
            }
            (None, Some(q), Some(r)) => {
                // Marked edges meet at `c`.
                refined_tris.push([c, q, r]);
                refined_tris.push([q, b, a]);
                refined_tris.push([q, a, r]);
            }
            (Some(p), None, Some(r)) => {
                // Marked edges meet at `a`.
                refined_tris.push([a, p, r]);
                refined_tris.push([p, b, c]);
                refined_tris.push([p, c, r]);
            }
            (Some(p), Some(q), Some(r)) => {
                refined_tris.push([a, p, r]);
                refined_tris.push([p, b, q]);
                refined_tris.push([r, q, c]);
                refined_tris.push([p, q, r]);
            }
        }
    }
    (
        refined_points,
        refined_tris,
        refined_size,
        flagged,
        edge_order,
    )
}

/// Extends a guided target field to the vertices added by [`refine_once`].
///
/// New vertices are edge midpoints, so their target is the circular mean of
/// the two endpoint targets in the fourth-power representation. The original
/// vertices keep their targets (the first `guidance.angle.len()` entries).
fn extend_guidance(
    guidance: &Guidance,
    edge_order: &[(usize, usize)],
    refined_vertices: usize,
) -> Guidance {
    let mut angle = Vec::with_capacity(refined_vertices);
    angle.extend_from_slice(&guidance.angle);
    for &(a, b) in edge_order {
        let first = Complex::from_angle(FIELD_SYMMETRY * guidance.angle[a]);
        let second = Complex::from_angle(FIELD_SYMMETRY * guidance.angle[b]);
        let sum = first.add(second);
        let norm = (sum.re * sum.re + sum.im * sum.im).sqrt();
        angle.push(if norm.is_finite() && norm > 1e-12 {
            cross_angle(sum)
        } else {
            guidance.angle[a]
        });
    }
    Guidance {
        weight: guidance.weight,
        angle,
    }
}

/// Jacobi-preconditioned conjugate gradients for the Hermitian system `matrix x = rhs`.
///
/// Returns the iterate and the number of iterations actually performed.
fn conjugate_gradient(
    matrix: &Hermitian,
    rhs: &[Complex],
    tolerance: f64,
    max_iterations: usize,
) -> Result<(Vec<Complex>, usize), String> {
    let n = rhs.len();
    let rhs_norm = euclidean_norm(rhs);
    if !rhs_norm.is_finite() || rhs_norm <= 0.0 {
        return Err("the conjugate-gradient right-hand side has zero norm".into());
    }
    let mut inverse_diagonal = vec![0.0; n];
    for i in 0..n {
        let value = matrix.diagonal[i];
        if !value.is_finite() || value <= 0.0 {
            return Err(format!(
                "the shifted energy matrix has a non-positive diagonal at vertex {i} ({value})"
            ));
        }
        inverse_diagonal[i] = value.recip();
    }

    let mut x = vec![Complex::ZERO; n];
    let mut residual = rhs.to_vec();
    let mut preconditioned: Vec<Complex> = (0..n)
        .map(|i| residual[i].scale(inverse_diagonal[i]))
        .collect();
    let mut direction = preconditioned.clone();
    let mut residual_dot = dot(&preconditioned, &residual).re;
    if !residual_dot.is_finite() || residual_dot <= 0.0 {
        return Err("the conjugate-gradient initial residual is not positive".into());
    }

    for iteration in 0..max_iterations {
        let applied = matrix.apply(&direction);
        let denominator = dot(&direction, &applied).re;
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err(
                "conjugate-gradient breakdown: the shifted energy matrix is not positive definite"
                    .into(),
            );
        }
        let alpha = residual_dot / denominator;
        for i in 0..n {
            x[i] = x[i].add(direction[i].scale(alpha));
            residual[i] = residual[i].sub(applied[i].scale(alpha));
        }
        if euclidean_norm(&residual) <= tolerance * rhs_norm {
            return Ok((x, iteration + 1));
        }
        for i in 0..n {
            preconditioned[i] = residual[i].scale(inverse_diagonal[i]);
        }
        let next_dot = dot(&preconditioned, &residual).re;
        if !next_dot.is_finite() || next_dot <= 0.0 {
            return Err("conjugate-gradient breakdown: lost positive definiteness".into());
        }
        let beta = next_dot / residual_dot;
        for i in 0..n {
            direction[i] = preconditioned[i].add(direction[i].scale(beta));
        }
        residual_dot = next_dot;
    }
    Err(format!(
        "conjugate gradient did not converge in {max_iterations} iterations"
    ))
}

/// Normalizes `x` in the mass inner product `x^H M x`.
fn normalize_in_mass(x: &mut [Complex], mass: &Hermitian) -> Result<(), String> {
    let applied = mass.apply(x);
    let squared = dot(x, &applied).re;
    if !squared.is_finite() || squared <= 0.0 {
        return Err("the mass matrix is not positive definite on the current iterate".into());
    }
    let norm = squared.sqrt();
    if !norm.is_finite() || norm <= 0.0 {
        return Err("the mass norm of the iterate is not positive".into());
    }
    for value in x.iter_mut() {
        *value = value.scale(norm.recip());
    }
    Ok(())
}

/// Wraps an angle to `(-pi, pi]`.
fn wrap_to_pi(angle: f64) -> f64 {
    let mut wrapped =
        (angle + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
    if wrapped <= -std::f64::consts::PI {
        wrapped += std::f64::consts::TAU;
    }
    wrapped
}

/// Converts a complex fourth-power value to a cross-field angle in `[0, pi/2)`.
fn cross_angle(value: Complex) -> f64 {
    let mut angle = value.argument() / FIELD_SYMMETRY;
    angle = angle.rem_euclid(std::f64::consts::FRAC_PI_2);
    if !(0.0..std::f64::consts::FRAC_PI_2).contains(&angle) {
        angle = 0.0;
    }
    angle
}

/// Converts the complex vertex values to cross-field angles.
fn angles_from(values: &[Complex]) -> Result<Vec<f64>, String> {
    let mut angles = Vec::with_capacity(values.len());
    for value in values {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err("the computed field contains a non-finite value".into());
        }
        angles.push(cross_angle(*value));
    }
    Ok(angles)
}

/// Shared implementation of the unguided and guided solves.
fn solve_mesh(
    points: &[[f64; 3]],
    tris: &[[usize; 3]],
    size: &[f64],
    options: &Options,
    guidance: Option<&Guidance>,
    iterations: usize,
) -> Result<Vec<f64>, String> {
    let n = validate(points, tris, size, iterations)?;
    validate_options(options)?;
    if let Some(guidance) = guidance {
        validate_guidance(n, guidance)?;
    }

    // Pass 1: per-triangle geometry, per-vertex area-weighted size gradients and
    // the unique edge list.
    let mut geometries: Vec<TriangleGeometry> = Vec::with_capacity(tris.len());
    let mut gradient_sum = vec![[0.0f64; 2]; n];
    let mut gradient_weight = vec![0.0f64; n];
    let mut edge_map: HashMap<(usize, usize), usize> = HashMap::new();
    let mut edge_a: Vec<usize> = Vec::new();
    let mut edge_b: Vec<usize> = Vec::new();

    for (t, triangle) in tris.iter().enumerate() {
        let geometry = triangle_geometry(points, triangle, t)?;
        let [i, j, k] = geometry.vertices;
        let sigma = [size[i], size[j], size[k]];
        let gradient = triangle_gradient(&geometry.positions, sigma, geometry.area);
        for &vertex in geometry.vertices.iter() {
            gradient_sum[vertex][0] += geometry.area * gradient[0];
            gradient_sum[vertex][1] += geometry.area * gradient[1];
            gradient_weight[vertex] += geometry.area;
        }
        for (u, v) in [(i, j), (j, k), (k, i)] {
            let key = if u < v { (u, v) } else { (v, u) };
            if let Entry::Vacant(entry) = edge_map.entry(key) {
                entry.insert(edge_a.len());
                edge_a.push(key.0);
                edge_b.push(key.1);
            }
        }
        geometries.push(geometry);
    }

    let mut gradient = vec![[0.0f64; 2]; n];
    for vertex in 0..n {
        let weight = gradient_weight[vertex];
        if !weight.is_finite() || weight <= 0.0 {
            return Err(format!(
                "vertex {vertex} is not incident to any non-degenerate triangle"
            ));
        }
        gradient[vertex] = [
            gradient_sum[vertex][0] / weight,
            gradient_sum[vertex][1] / weight,
        ];
        if !gradient[vertex][0].is_finite() || !gradient[vertex][1].is_finite() {
            return Err(format!(
                "the size gradient at vertex {vertex} is not finite"
            ));
        }
    }

    // Transport coefficients, one per undirected edge (Algorithm 1, lines 2-11).
    let mut transport = Vec::with_capacity(edge_a.len());
    for e in 0..edge_a.len() {
        transport.push(transport_coefficient(
            edge_a[e], edge_b[e], points, size, &gradient,
        )?);
    }

    // Pass 2: Hermitian mass and energy matrices (Algorithm 1, lines 12-32,
    // with the appendix helper integrals when `options.accurate`).
    let mut mass = Hermitian::zero(n, edge_a.clone(), edge_b.clone());
    let mut energy = Hermitian::zero(n, edge_a.clone(), edge_b.clone());
    for geometry in &geometries {
        let [i, j, k] = geometry.vertices;
        let area = geometry.area;
        let p = geometry.positions;

        mass.diagonal[i] += area / 6.0;
        mass.diagonal[j] += area / 6.0;
        mass.diagonal[k] += area / 6.0;

        // Holonomy of the counter-clockwise triangle (Eq. 13).
        let mut holonomy = Complex::ONE;
        for (u, v) in [(i, j), (j, k), (k, i)] {
            let key = if u < v { (u, v) } else { (v, u) };
            let edge = match edge_map.get(&key) {
                Some(&edge) => edge,
                None => {
                    return Err(
                        "internal error: a triangle edge is missing from the edge list".into(),
                    )
                }
            };
            let r_uv = if u < v {
                transport[edge]
            } else {
                transport[edge].conj()
            };
            holonomy = holonomy.mul(r_uv);
        }
        let omega = holonomy.argument();

        if options.accurate {
            energy.diagonal[i] += dirichlet_diagonal(omega, area, p[0], p[1], p[2]);
            energy.diagonal[j] += dirichlet_diagonal(omega, area, p[1], p[2], p[0]);
            energy.diagonal[k] += dirichlet_diagonal(omega, area, p[2], p[0], p[1]);
        } else {
            let quarter_area = (4.0 * area).recip();
            let p_jk = sub2(p[2], p[1]);
            let p_ki = sub2(p[0], p[2]);
            let p_ij = sub2(p[1], p[0]);
            energy.diagonal[i] += dot2(p_jk, p_jk) * quarter_area;
            energy.diagonal[j] += dot2(p_ki, p_ki) * quarter_area;
            energy.diagonal[k] += dot2(p_ij, p_ij) * quarter_area;
        }

        // Eq. 18 diagonal term: `-s (Omega/|t|) M_ii` with `M_ii = |t|/6`,
        // so the correction is `-s Omega/6` for the three vertices.
        let holomorphic = options.holomorphic;
        if holomorphic != 0.0 {
            let diagonal_correction = -holomorphic * omega / 6.0;
            energy.diagonal[i] += diagonal_correction;
            energy.diagonal[j] += diagonal_correction;
            energy.diagonal[k] += diagonal_correction;
        }

        // The three counter-clockwise oriented pairs `(p, q)` with the third
        // vertex `r`, expressed as local indices into `geometry.positions`.
        let pairs = [(0usize, 1usize, 2usize), (1, 2, 0), (2, 0, 1)];
        for &(p_local, q_local, r_local) in pairs.iter() {
            let p_vertex = geometry.vertices[p_local];
            let q_vertex = geometry.vertices[q_local];
            let key = if p_vertex < q_vertex {
                (p_vertex, q_vertex)
            } else {
                (q_vertex, p_vertex)
            };
            let edge = match edge_map.get(&key) {
                Some(&edge) => edge,
                None => {
                    return Err(
                        "internal error: a triangle edge is missing from the edge list".into(),
                    )
                }
            };
            let r_pq = if p_vertex < q_vertex {
                transport[edge]
            } else {
                transport[edge].conj()
            };
            let p_p = geometry.positions[p_local];
            let p_q = geometry.positions[q_local];
            let p_r = geometry.positions[r_local];
            let from_r_to_p = sub2(p_p, p_r);
            let from_r_to_q = sub2(p_q, p_r);
            let g11 = dot2(from_r_to_p, from_r_to_p);
            let g12 = dot2(from_r_to_p, from_r_to_q);
            let g22 = dot2(from_r_to_q, from_r_to_q);

            let mass_scalar = if options.accurate {
                mass_integral(omega)
            } else {
                Complex::new(1.0 / 12.0, 0.0)
            };
            let mass_entry = r_pq.conj().mul(mass_scalar).scale(area);
            let mut energy_entry = if options.accurate {
                r_pq.conj()
                    .mul(dirichlet_integral(omega, g11, g12, g22))
                    .scale(area.recip())
            } else {
                r_pq.conj().scale(-g12 * (4.0 * area).recip())
            };

            // Eq. 18 off-diagonal terms:
            //   `-s (Omega/|t|) M_pq + s i conj(r_pq) / 2`,
            // with `eps_pq = +1` because the pair is traversed counter-clockwise
            // (the sign in the second term is tied to that orientation).
            if holomorphic != 0.0 {
                energy_entry = energy_entry
                    .sub(mass_entry.scale(holomorphic * omega / area))
                    .add(Complex::new(0.0, 0.5 * holomorphic).mul(r_pq.conj()));
            }

            // Store the entry for the canonical edge `(a, b)` with `a < b`;
            // the transposed entry is generated as its conjugate.
            let (mass_value, energy_value) = if p_vertex < q_vertex {
                (mass_entry, energy_entry)
            } else {
                (mass_entry.conj(), energy_entry.conj())
            };
            mass.off_diagonal[edge] = mass.off_diagonal[edge].add(mass_value);
            energy.off_diagonal[edge] = energy.off_diagonal[edge].add(energy_value);
        }
    }

    let energy_trace = energy.trace();
    let mass_trace = mass.trace();
    if !energy_trace.is_finite() || energy_trace <= 0.0 {
        return Err("the energy matrix has a non-positive trace".into());
    }
    if !mass_trace.is_finite() || mass_trace <= 0.0 {
        return Err("the mass matrix has a non-positive trace".into());
    }

    // Algorithm 1, line 33, made scale-relative; see the module documentation.
    let shift = ENERGY_SHIFT * energy_trace / mass_trace;

    // Algorithm 1, line 33: the shifted energy pencil.
    let shifted_energy = energy.shifted(shift, &mass)?;

    // Soft terms of the normal equations: the guidance penalty of
    // `solve_with_guidance` and the anisotropy alignment of Section 4.10.
    // Each term contributes `w M` to the energy and `w M G` to the right-hand
    // side, i.e. the linear system `(A + shift M + sum w M) u = sum w M G`.
    let mut soft: Vec<(f64, Vec<Complex>)> = Vec::new();
    if let Some(guidance) = guidance {
        if guidance.weight > 0.0 {
            let target: Vec<Complex> = guidance
                .angle
                .iter()
                .map(|&angle| Complex::from_angle(FIELD_SYMMETRY * angle))
                .collect();
            soft.push((guidance.weight, target));
        }
    }
    let anisotropy_weight = anisotropy_alignment_weight(options.anisotropy);
    if anisotropy_weight > 0.0 {
        soft.push((
            anisotropy_weight,
            vec![Complex::from_angle(FIELD_SYMMETRY * options.anisotropy_angle); n],
        ));
    }

    let weak_weight = match options.boundary {
        BoundaryMode::Weak(weight) => weight,
        BoundaryMode::None | BoundaryMode::Tangent => 0.0,
    };
    let constrained = matches!(options.boundary, BoundaryMode::Tangent) || weak_weight > 0.0;
    if constrained {
        let (is_boundary, targets) = boundary_targets(points, tris, n)?;
        let max_iterations = (400 + 8 * n).max(iterations).min(6000);
        if options.boundary == BoundaryMode::Tangent {
            // Exact Dirichlet elimination: the boundary unknowns are known, so
            // their columns move to the right-hand side and their rows become
            // the identity. The interior then minimises the same energy.
            let boundary_data: Vec<Complex> = (0..n)
                .map(|i| {
                    if is_boundary[i] {
                        targets[i]
                    } else {
                        Complex::ZERO
                    }
                })
                .collect();
            let coupling = shifted_energy.apply(&boundary_data);
            let mut system = shifted_energy;
            let mut rhs = vec![Complex::ZERO; n];
            for (weight, target) in &soft {
                let interior: Vec<Complex> = target
                    .iter()
                    .enumerate()
                    .map(|(i, value)| {
                        if is_boundary[i] {
                            Complex::ZERO
                        } else {
                            *value
                        }
                    })
                    .collect();
                let applied = mass.apply(&interior);
                for i in 0..n {
                    if !is_boundary[i] {
                        rhs[i] = rhs[i].add(applied[i].scale(*weight));
                    }
                }
                system = system.shifted(*weight, &mass)?;
            }
            for i in 0..n {
                if is_boundary[i] {
                    rhs[i] = targets[i];
                } else {
                    rhs[i] = rhs[i].sub(coupling[i]);
                }
            }
            let system = dirichlet_rows(system, &is_boundary);
            let (solution, _) = conjugate_gradient(&system, &rhs, INNER_TOLERANCE, max_iterations)?;
            return angles_from(&solution);
        }
        // Weak penalty, lumped to the mass diagonal: `+ w M_ii u_i = w M_ii G_i`
        // at boundary vertices.
        let mut system = shifted_energy;
        let mut rhs = vec![Complex::ZERO; n];
        for i in 0..n {
            if is_boundary[i] {
                let penalty = weak_weight * mass.diagonal[i];
                system.diagonal[i] += penalty;
                rhs[i] = targets[i].scale(penalty);
            }
        }
        for (weight, target) in &soft {
            let applied = mass.apply(target);
            for i in 0..n {
                rhs[i] = rhs[i].add(applied[i].scale(*weight));
            }
            system = system.shifted(*weight, &mass)?;
        }
        let (solution, _) = conjugate_gradient(&system, &rhs, INNER_TOLERANCE, max_iterations)?;
        return angles_from(&solution);
    }

    if !soft.is_empty() {
        let mut system = shifted_energy;
        let mut rhs = vec![Complex::ZERO; n];
        for (weight, target) in &soft {
            let applied = mass.apply(target);
            for i in 0..n {
                rhs[i] = rhs[i].add(applied[i].scale(*weight));
            }
            system = system.shifted(*weight, &mass)?;
        }
        let max_iterations = (400 + 8 * n).max(iterations).min(6000);
        let (solution, _) = conjugate_gradient(&system, &rhs, INNER_TOLERANCE, max_iterations)?;
        return angles_from(&solution);
    }

    // Unguided solve: smallest eigenpair of the shifted pencil.
    let shifted = shifted_energy;
    let average_diagonal = energy_trace / n as f64;

    // Deterministic start: a constant direction plus a small complex ramp over
    // the bounding box. The ramp breaks the symmetry so that a non-constant
    // lowest mode is not missed, while the constant part carries the harmonic
    // mode when one exists.
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for point in points {
        min_x = min_x.min(point[0]);
        max_x = max_x.max(point[0]);
        min_y = min_y.min(point[1]);
        max_y = max_y.max(point[1]);
    }
    let center_x = 0.5 * (min_x + max_x);
    let center_y = 0.5 * (min_y + max_y);
    let span = (max_x - min_x).max(max_y - min_y);
    let inverse_span = if span > 0.0 { span.recip() } else { 1.0 };
    let mut x: Vec<Complex> = points
        .iter()
        .map(|point| {
            Complex::ONE.add(
                Complex::new(
                    (point[0] - center_x) * inverse_span,
                    (point[1] - center_y) * inverse_span,
                )
                .scale(0.05),
            )
        })
        .collect();
    normalize_in_mass(&mut x, &mass)?;

    let inner_max = (400 + 8 * n).min(6000);
    let mut converged = false;
    for _outer in 0..iterations {
        let rhs = mass.apply(&x);
        let (mut next, _inner_iterations) =
            conjugate_gradient(&shifted, &rhs, INNER_TOLERANCE, inner_max)?;
        normalize_in_mass(&mut next, &mass)?;

        let mass_next = mass.apply(&next);
        let energy_next = shifted.apply(&next);
        let denominator = dot(&next, &mass_next).re;
        if !denominator.is_finite() || denominator <= 0.0 {
            return Err("the eigenvalue Rayleigh quotient has a non-positive denominator".into());
        }
        let lambda = dot(&next, &energy_next).re / denominator;
        if !lambda.is_finite() {
            return Err("the eigenvalue estimate is not finite".into());
        }
        let residual: Vec<Complex> = energy_next
            .iter()
            .zip(mass_next.iter())
            .map(|(energy_value, mass_value)| energy_value.sub(mass_value.scale(lambda)))
            .collect();
        let relative = euclidean_norm(&residual) / average_diagonal;
        if relative <= OUTER_TOLERANCE {
            x = next;
            converged = true;
            break;
        }
        x = next;
    }
    if !converged {
        return Err(format!(
            "shifted inverse power iteration did not converge in {iterations} iterations"
        ));
    }

    angles_from(&x)
}

/// Shared implementation: validate, solve, and optionally refine once.
fn solve_impl(
    points: &[[f64; 3]],
    tris: &[[usize; 3]],
    size: &[f64],
    options: &Options,
    guidance: Option<&Guidance>,
    iterations: usize,
) -> Result<Vec<f64>, String> {
    let n = validate(points, tris, size, iterations)?;
    validate_options(options)?;
    if let Some(guidance) = guidance {
        validate_guidance(n, guidance)?;
    }
    let alpha = solve_mesh(points, tris, size, options, guidance, iterations)?;
    if options.adaptive <= 0.0 {
        return Ok(alpha);
    }

    // One refinement round on the field's own variation (Sections 3.2.5.1 and
    // 3.4.2), then a re-solve on the refined background mesh. The original
    // vertices keep their indices, so reading their new values is the
    // interpolation back to the original mesh.
    let (refined_points, refined_tris, refined_size, flagged, edge_order) =
        refine_once(points, tris, size, &alpha, options.adaptive);
    if flagged == 0 {
        return Ok(alpha);
    }
    let mut inner = *options;
    inner.adaptive = 0.0;
    let extended;
    let refined_guidance = match guidance {
        Some(guidance) if guidance.weight > 0.0 => {
            extended = extend_guidance(guidance, &edge_order, refined_points.len());
            Some(&extended)
        }
        _ => None,
    };
    let refined_alpha = solve_mesh(
        &refined_points,
        &refined_tris,
        &refined_size,
        &inner,
        refined_guidance,
        iterations,
    )?;
    if refined_alpha.len() < n {
        return Err("internal error: the refined field lost original vertices".into());
    }
    Ok(refined_alpha[..n].to_vec())
}

/// Computes a size-aware cross field on a planar triangle mesh.
///
/// `points` are the mesh vertices (`z` is ignored), `tris` the triangles
/// (either orientation; they are reordered counter-clockwise internally) and
/// `size` the size function sampled at the vertices. The returned vector holds,
/// for every vertex, the field angle `arg(u_i)/4` in radians, reduced to
/// `[0, pi/2)` modulo the four-fold symmetry of a cross.
///
/// `iterations` bounds the number of shifted inverse-power iterations. The
/// function returns an error when the mesh, the sizes or the iterations are
/// invalid, when a triangle is degenerate, or when the eigen-solve does not
/// converge.
///
/// This is the `s = 0` case of [`solve_with_guidance`]; it uses the flat
/// triangle integrals of [`Options::default`]. Use [`solve_with_options`] for
/// the accurate integrals.
///
/// # Example
///
/// ```no_run
/// let points = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
/// let tris = vec![[0, 1, 2]];
/// let size = vec![1.0, 1.0, 1.0];
/// if let Ok(angle) = pedroso_mesher::simons_godf::solve(&points, &tris, &size, 50) {
///     assert_eq!(angle.len(), 3);
/// }
/// ```
pub fn solve(
    points: &[[f64; 3]],
    tris: &[[usize; 3]],
    size: &[f64],
    iterations: usize,
) -> Result<Vec<f64>, String> {
    solve_impl(points, tris, size, &Options::default(), None, iterations)
}

/// Computes a size-aware cross field with explicit [`Options`].
///
/// With `options.accurate == true` the curvature-aware `MassIJ` and
/// `DirichletIJ` integrals of GODF Appendix D replace the flat limits. The
/// other fields enable the holomorphic energy ([`Options::holomorphic`]),
/// boundary tangent constraints ([`Options::boundary`]), the anisotropic
/// alignment of Section 4.10 ([`Options::anisotropy`] and
/// [`Options::anisotropy_angle`]) and one round of adaptive refinement
/// ([`Options::adaptive`]). With all defaults the result is identical to
/// [`solve`].
///
/// # Example
///
/// ```no_run
/// let options = pedroso_mesher::simons_godf::Options {
///     accurate: true,
///     ..pedroso_mesher::simons_godf::Options::default()
/// };
/// let alpha = pedroso_mesher::simons_godf::solve_with_options(
///     &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
///     &[[0, 1, 2]],
///     &[1.0, 1.0, 1.0],
///     &options,
///     50,
/// )?;
/// assert_eq!(alpha.len(), 3);
/// # Ok::<(), String>(())
/// ```
pub fn solve_with_options(
    points: &[[f64; 3]],
    tris: &[[usize; 3]],
    size: &[f64],
    options: &Options,
    iterations: usize,
) -> Result<Vec<f64>, String> {
    solve_impl(points, tris, size, options, None, iterations)
}

/// Computes a size-aware cross field biased toward a per-vertex target field.
///
/// The guided field solves `(A + s M) u = s M G` with `s = guidance.weight`,
/// `G_i = exp(4 i guidance.angle[i])`, and `A`, `M` the Dirichlet energy and
/// mass matrices. `guidance.weight == 0` returns exactly the unguided field of
/// [`solve`]; larger weights favour the target. The returned angles have the
/// same representation and `[0, pi/2)` range as [`solve`]. This entry point
/// uses [`Options::default`]; use [`solve_with_guidance_and_options`] for
/// boundary, anisotropy and adaptive options.
///
/// Returns an error when the mesh is invalid, when `guidance.weight` is
/// negative or non-finite, or when `guidance.angle` does not have one finite
/// value per vertex.
///
/// # Example
///
/// ```no_run
/// let points = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
/// let tris = vec![[0, 1, 2]];
/// let size = vec![1.0, 1.0, 1.0];
/// let guidance = pedroso_mesher::simons_godf::Guidance {
///     weight: 1.0,
///     angle: vec![0.3, 0.3, 0.3],
/// };
/// if let Ok(alpha) = pedroso_mesher::simons_godf::solve_with_guidance(
///     &points, &tris, &size, &guidance, 50,
/// ) {
///     assert_eq!(alpha.len(), 3);
/// }
/// ```
pub fn solve_with_guidance(
    points: &[[f64; 3]],
    tris: &[[usize; 3]],
    size: &[f64],
    guidance: &Guidance,
    iterations: usize,
) -> Result<Vec<f64>, String> {
    solve_impl(
        points,
        tris,
        size,
        &Options::default(),
        Some(guidance),
        iterations,
    )
}

/// Computes a guided cross field with explicit [`Options`].
///
/// Combines the soft alignment of [`solve_with_guidance`] with every option of
/// [`solve_with_options`] (accurate integrals, holomorphic energy, boundary
/// constraints, anisotropy and adaptive refinement). When a boundary
/// constraint is active, the boundary tangent wins over `guidance` on the
/// boundary vertices.
pub fn solve_with_guidance_and_options(
    points: &[[f64; 3]],
    tris: &[[usize; 3]],
    size: &[f64],
    guidance: &Guidance,
    options: &Options,
    iterations: usize,
) -> Result<Vec<f64>, String> {
    solve_impl(points, tris, size, options, Some(guidance), iterations)
}

/// Interpolates the cross-field angle at a planar query point.
///
/// The vertex values `alpha` are the fourth-power angles returned by [`solve`]
/// (`u_i = exp(i 4 alpha_i)`). Within a triangle the complex values are
/// interpolated with the barycentric coordinates and the result is converted
/// back with `arg(u)/4`, exactly as in Section 3.3.1. Triangles are scanned in
/// order; a point on a shared edge is found in whichever triangle comes first,
/// and the two interpolations agree because the field is continuous.
///
/// Returns `NaN` when `alpha` does not match `points`, when no triangle
/// contains the point, or when the interpolated complex value vanishes (for
/// example exactly at a singularity, where the direction is undefined).
pub fn interp_angle(alpha: &[f64], points: &[[f64; 3]], tris: &[[usize; 3]], p: [f64; 2]) -> f64 {
    if alpha.len() != points.len() {
        return f64::NAN;
    }
    if alpha.iter().any(|value| !value.is_finite()) {
        return f64::NAN;
    }
    for triangle in tris {
        let [i, j, k] = *triangle;
        if i >= points.len() || j >= points.len() || k >= points.len() {
            continue;
        }
        let p_i = [points[i][0], points[i][1]];
        let p_j = [points[j][0], points[j][1]];
        let p_k = [points[k][0], points[k][1]];
        let denominator = cross2(sub2(p_j, p_i), sub2(p_k, p_i));
        if !denominator.is_finite() || denominator.abs() <= f64::MIN_POSITIVE {
            continue;
        }
        let w_i = cross2(sub2(p_j, p), sub2(p_k, p)) / denominator;
        let w_j = cross2(sub2(p_k, p), sub2(p_i, p)) / denominator;
        let w_k = cross2(sub2(p_i, p), sub2(p_j, p)) / denominator;
        let tolerance = 1e-12;
        if w_i < -tolerance
            || w_j < -tolerance
            || w_k < -tolerance
            || w_i > 1.0 + tolerance
            || w_j > 1.0 + tolerance
            || w_k > 1.0 + tolerance
        {
            continue;
        }
        let interpolated = Complex::from_angle(4.0 * alpha[i])
            .scale(w_i)
            .add(Complex::from_angle(4.0 * alpha[j]).scale(w_j))
            .add(Complex::from_angle(4.0 * alpha[k]).scale(w_k));
        if interpolated.re == 0.0 && interpolated.im == 0.0 {
            return f64::NAN;
        }
        return cross_angle(interpolated);
    }
    f64::NAN
}

/// Counts the singular triangles of a cross field by their holonomy.
///
/// For every triangle the winding of `exp(i 4 theta)` is accumulated from the
/// three vertex transitions; each transition is wrapped to `(-pi, pi]`, so the
/// sum is `-2 pi`, `0` or `+2 pi` and the triangle holds a valence-5, regular
/// or valence-3 singularity respectively (Section 3.1.6). The returned pair is
/// `(valence_3, valence_5)`. Triangles with out-of-range or non-finite data are
/// skipped; at most one singularity is counted per triangle.
pub fn count_singularities(alpha: &[f64], tris: &[[usize; 3]]) -> (usize, usize) {
    let mut valence_3 = 0;
    let mut valence_5 = 0;
    for triangle in tris {
        let [i, j, k] = *triangle;
        if i >= alpha.len() || j >= alpha.len() || k >= alpha.len() {
            continue;
        }
        let a = alpha[i];
        let b = alpha[j];
        let c = alpha[k];
        if !a.is_finite() || !b.is_finite() || !c.is_finite() {
            continue;
        }
        let winding =
            wrap_to_pi(4.0 * (b - a)) + wrap_to_pi(4.0 * (c - b)) + wrap_to_pi(4.0 * (a - c));
        let charge = (winding / std::f64::consts::TAU).round() as i64;
        if charge > 0 {
            valence_3 += 1;
        } else if charge < 0 {
            valence_5 += 1;
        }
    }
    (valence_3, valence_5)
}

/// Regular counter-clockwise triangulation of the unit square, used by
/// [`demo_holomorphic`].
fn demo_square_mesh(n: usize) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let n = n.max(1);
    let mut points = Vec::with_capacity((n + 1) * (n + 1));
    for j in 0..=n {
        for i in 0..=n {
            points.push([i as f64 / n as f64, j as f64 / n as f64, 0.0]);
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

/// Counter-clockwise triangulation of an `nx` by `ny` rectangle.
fn demo_grid_mesh(
    nx: usize,
    ny: usize,
    width: f64,
    height: f64,
) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let nx = nx.max(1);
    let ny = ny.max(1);
    let mut points = Vec::with_capacity((nx + 1) * (ny + 1));
    for j in 0..=ny {
        for i in 0..=nx {
            points.push([
                width * i as f64 / nx as f64,
                height * j as f64 / ny as f64,
                0.0,
            ]);
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
    (points, tris)
}

/// An annular plate with a circular hole.
///
/// The inner ring is a circle of radius `0.35` and the outer ring a concentric
/// circle of radius `1.0`; both are genuine circles, so a field that follows
/// the hole can also be tangent to the outer boundary without singularities.
/// Used by [`demo_boundary`].
fn demo_plate_with_hole(radial: usize, angular: usize) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
    let radial = radial.max(1);
    let angular = angular.max(4);
    let inner = 0.35f64;
    let mut points = Vec::with_capacity((radial + 1) * angular);
    for ring in 0..=radial {
        let fraction = ring as f64 / radial as f64;
        for sector in 0..angular {
            let phi = std::f64::consts::TAU * sector as f64 / angular as f64;
            let (sin, cos) = phi.sin_cos();
            let radius = inner + (1.0 - inner) * fraction;
            points.push([radius * cos, radius * sin, 0.0]);
        }
    }
    let mut tris = Vec::with_capacity(2 * radial * angular);
    for ring in 0..radial {
        for sector in 0..angular {
            let next = (sector + 1) % angular;
            let a = ring * angular + sector;
            let b = (ring + 1) * angular + sector;
            let c = (ring + 1) * angular + next;
            let d = ring * angular + next;
            tris.push([a, b, c]);
            tris.push([a, c, d]);
        }
    }
    (points, tris)
}

/// Unique edges that belong to exactly one triangle (the mesh boundary).
fn boundary_edges(tris: &[[usize; 3]]) -> Vec<(usize, usize)> {
    let mut counts: HashMap<(usize, usize), usize> = HashMap::new();
    for triangle in tris {
        for (u, v) in [
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ] {
            let key = if u < v { (u, v) } else { (v, u) };
            *counts.entry(key).or_insert(0) += 1;
        }
    }
    let mut edges = Vec::new();
    for triangle in tris {
        for (u, v) in [
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ] {
            let key = if u < v { (u, v) } else { (v, u) };
            if counts.get(&key).copied().unwrap_or(0) == 1 {
                edges.push((u, v));
            }
        }
    }
    edges
}

/// 4-RoSy distance between two cross-field angles.
fn cross_angle_distance(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(std::f64::consts::FRAC_PI_2);
    d.min(std::f64::consts::FRAC_PI_2 - d)
}

/// Circular mean of a cross field, in `[0, pi/2)`.
fn mean_cross_angle(alpha: &[f64]) -> f64 {
    let mut sum = Complex::ZERO;
    for &angle in alpha {
        sum = sum.add(Complex::from_angle(FIELD_SYMMETRY * angle));
    }
    cross_angle(sum)
}

/// Mean and maximum 4-RoSy distance to the boundary tangent over the vertices
/// in a band of width `band`.
///
/// The tangent at a boundary vertex is the average (in the fourth-power
/// representation) of the tangents of its incident boundary edges, exactly the
/// target used by [`BoundaryMode::Tangent`]; this matters on a smooth boundary,
/// where a single edge's chord direction is offset by half a sector. A vertex
/// is in the band when its distance to the nearest boundary vertex is at most
/// `band`. When `radius_limit` is given, only vertices whose distance to the
/// origin is at most that value are considered; this isolates the inner
/// boundary of the plate-with-hole demo. Returns `(mean, maximum, count)`; an
/// empty band gives `(0, 0, 0)`.
fn boundary_band_measure(
    points: &[[f64; 3]],
    tris: &[[usize; 3]],
    alpha: &[f64],
    band: f64,
    radius_limit: Option<f64>,
) -> (f64, f64, usize) {
    let edges = boundary_edges(tris);
    let mut sum = vec![Complex::ZERO; points.len()];
    let mut is_boundary = vec![false; points.len()];
    for (u, v) in edges {
        let tangent = (points[v][1] - points[u][1]).atan2(points[v][0] - points[u][0]);
        let value = Complex::from_angle(FIELD_SYMMETRY * tangent);
        for vertex in [u, v] {
            is_boundary[vertex] = true;
            sum[vertex] = sum[vertex].add(value);
        }
    }
    let boundary: Vec<(usize, f64)> = (0..points.len())
        .filter(|&vertex| is_boundary[vertex])
        .map(|vertex| (vertex, cross_angle(sum[vertex])))
        .collect();

    let mut total = 0.0f64;
    let mut maximum = 0.0f64;
    let mut count = 0usize;
    for (vertex, point) in points.iter().enumerate() {
        if vertex >= alpha.len() {
            continue;
        }
        if let Some(limit) = radius_limit {
            if (point[0] * point[0] + point[1] * point[1]).sqrt() > limit {
                continue;
            }
        }
        let mut nearest = f64::INFINITY;
        let mut target = 0.0f64;
        for &(boundary_vertex, boundary_angle) in &boundary {
            let dx = point[0] - points[boundary_vertex][0];
            let dy = point[1] - points[boundary_vertex][1];
            let distance = (dx * dx + dy * dy).sqrt();
            if distance < nearest {
                nearest = distance;
                target = boundary_angle;
            }
        }
        if nearest <= band {
            let d = cross_angle_distance(alpha[vertex], target);
            total += d;
            maximum = maximum.max(d);
            count += 1;
        }
    }
    if count == 0 {
        (0.0, 0.0, 0)
    } else {
        (total / count as f64, maximum, count)
    }
}

/// Measurements of the holomorphic parameter `s` on a guided, graded square.
///
/// The case is a `10 x 10` unit square with the graded size `sigma = 1 + x/2`,
/// the accurate integrals, and a constant guidance target of `0.3 rad` with
/// weight `1` (the guidance selects one representative of the global phase and
/// makes the solve a single linear system). For each `s` in
/// `[-1, -0.5, 0.5, 1]` the tuple is
/// `(s, mean 4-RoSy angle difference to s = 0, max difference to s = 0)`, in
/// radians. The measurements are returned for reporting; they are also
/// asserted in the module tests.
///
/// Returns an error only if one of the linear solves fails.
pub fn demo_holomorphic() -> Result<Vec<(f64, f64, f64)>, String> {
    let (points, tris) = demo_square_mesh(10);
    let size: Vec<f64> = points.iter().map(|point| 1.0 + 0.5 * point[0]).collect();
    let guidance = Guidance {
        weight: 1.0,
        angle: vec![0.3; points.len()],
    };
    let base = solve_with_guidance_and_options(
        &points,
        &tris,
        &size,
        &guidance,
        &Options {
            accurate: true,
            holomorphic: 0.0,
            ..Options::default()
        },
        100,
    )?;
    let mut out = Vec::new();
    for s in [-1.0, -0.5, 0.5, 1.0] {
        let alpha = solve_with_guidance_and_options(
            &points,
            &tris,
            &size,
            &guidance,
            &Options {
                accurate: true,
                holomorphic: s,
                ..Options::default()
            },
            100,
        )?;
        let mut mean = 0.0f64;
        let mut maximum = 0.0f64;
        for (&a, &b) in alpha.iter().zip(base.iter()) {
            let d = (a - b).rem_euclid(std::f64::consts::FRAC_PI_2);
            let d = d.min(std::f64::consts::FRAC_PI_2 - d);
            mean += d;
            maximum = maximum.max(d);
        }
        mean /= alpha.len() as f64;
        out.push((s, mean, maximum));
    }
    Ok(out)
}

/// Measurements of [`BoundaryMode`] on the square and the plate-with-hole.
#[derive(Clone, Debug, Default)]
pub struct BoundaryReport {
    /// Mean 4-RoSy distance to the boundary tangent in the square band, before.
    pub square_before_mean: f64,
    /// Mean 4-RoSy distance to the boundary tangent in the square band, after.
    pub square_after_mean: f64,
    /// Maximum distance in the square band, before.
    pub square_before_max: f64,
    /// Maximum distance in the square band, after.
    pub square_after_max: f64,
    /// Number of vertices in the square band.
    pub square_band_vertices: usize,
    /// `(valence-3, valence-5)` singularities on the square, before.
    pub square_singularities_before: (usize, usize),
    /// `(valence-3, valence-5)` singularities on the square, after.
    pub square_singularities_after: (usize, usize),
    /// Mean 4-RoSy distance to the circular-hole tangent, before.
    pub hole_before_mean: f64,
    /// Mean 4-RoSy distance to the circular-hole tangent, after.
    pub hole_after_mean: f64,
    /// Maximum distance in the hole band, before.
    pub hole_before_max: f64,
    /// Maximum distance in the hole band, after.
    pub hole_after_max: f64,
    /// Number of vertices in the hole band.
    pub hole_band_vertices: usize,
    /// `(valence-3, valence-5)` singularities on the annular plate, before.
    pub hole_singularities_before: (usize, usize),
    /// `(valence-3, valence-5)` singularities on the annular plate, after.
    pub hole_singularities_after: (usize, usize),
}

/// Measures the boundary-tangent constraint on a `10 x 10` graded square and on
/// an annular plate with a circular hole.
///
/// The square uses `sigma = 1 + x/2`, so its unconstrained field rotates with
/// `x`; the annular plate uses a constant size. "Before" is
/// [`BoundaryMode::None`], "after" is [`BoundaryMode::Tangent`]. The plate's
/// annulus is topologically compatible with a smooth tangent field, so both
/// solves have zero singularities.
///
/// Returns an error only if one of the solves fails.
pub fn demo_boundary() -> Result<BoundaryReport, String> {
    let (square_points, square_tris) = demo_square_mesh(10);
    let square_size: Vec<f64> = square_points
        .iter()
        .map(|point| 1.0 + 0.5 * point[0])
        .collect();
    let square_before = solve_with_options(
        &square_points,
        &square_tris,
        &square_size,
        &Options::default(),
        100,
    )?;
    let square_after = solve_with_options(
        &square_points,
        &square_tris,
        &square_size,
        &Options {
            boundary: BoundaryMode::Tangent,
            ..Options::default()
        },
        100,
    )?;
    let (square_before_mean, square_before_max, square_band_vertices) =
        boundary_band_measure(&square_points, &square_tris, &square_before, 0.15, None);
    let (square_after_mean, square_after_max, _) =
        boundary_band_measure(&square_points, &square_tris, &square_after, 0.15, None);

    let (plate_points, plate_tris) = demo_plate_with_hole(4, 48);
    let plate_size = vec![1.0; plate_points.len()];
    let plate_before = solve_with_options(
        &plate_points,
        &plate_tris,
        &plate_size,
        &Options::default(),
        100,
    )?;
    let plate_after = solve_with_options(
        &plate_points,
        &plate_tris,
        &plate_size,
        &Options {
            boundary: BoundaryMode::Tangent,
            ..Options::default()
        },
        100,
    )?;
    let hole_limit = 0.55;
    let (hole_before_mean, hole_before_max, hole_band_vertices) = boundary_band_measure(
        &plate_points,
        &plate_tris,
        &plate_before,
        0.2,
        Some(hole_limit),
    );
    let (hole_after_mean, hole_after_max, _) = boundary_band_measure(
        &plate_points,
        &plate_tris,
        &plate_after,
        0.2,
        Some(hole_limit),
    );

    Ok(BoundaryReport {
        square_before_mean,
        square_after_mean,
        square_before_max,
        square_after_max,
        square_band_vertices,
        square_singularities_before: count_singularities(&square_before, &square_tris),
        square_singularities_after: count_singularities(&square_after, &square_tris),
        hole_before_mean,
        hole_after_mean,
        hole_before_max,
        hole_after_max,
        hole_band_vertices,
        hole_singularities_before: count_singularities(&plate_before, &plate_tris),
        hole_singularities_after: count_singularities(&plate_after, &plate_tris),
    })
}

/// Measurements of the anisotropy option on a stretched rectangle.
#[derive(Clone, Debug, Default)]
pub struct AnisotropyReport {
    /// Direction of the `sigma` axis, in radians.
    pub frame_angle: f64,
    /// One row per tested ratio: `(a, mean distance to the frame, mean angle,
    /// maximum adjacent angle step)`, in radians.
    pub rows: Vec<(f64, f64, f64, f64)>,
}

/// Measures `a = 1, 2, 4` on a `2 x 1` rectangle with `sigma = 1 + x/2` and
/// anisotropy direction `0.4 rad`.
///
/// A `a = 1` row is the isotropic (historical) field, which has an arbitrary
/// global phase. Larger ratios add the alignment energy described in the
/// module documentation and should bring the mean angle close to the frame.
///
/// Returns an error only if one of the solves fails.
pub fn demo_anisotropy() -> Result<AnisotropyReport, String> {
    let (points, tris) = demo_grid_mesh(16, 8, 2.0, 1.0);
    let size: Vec<f64> = points.iter().map(|point| 1.0 + 0.5 * point[0]).collect();
    let frame_angle = 0.4f64;
    // The guidance (`0.0 rad`, weight 1) anchors the global phase so that the
    // before/after angles are comparable; the anisotropy term then rotates the
    // field toward the frame.
    let guidance = Guidance {
        weight: 1.0,
        angle: vec![0.0; points.len()],
    };
    let mut rows = Vec::new();
    for anisotropy in [1.0, 2.0, 4.0] {
        let alpha = solve_with_guidance_and_options(
            &points,
            &tris,
            &size,
            &guidance,
            &Options {
                anisotropy,
                anisotropy_angle: frame_angle,
                ..Options::default()
            },
            100,
        )?;
        let mean_distance = alpha
            .iter()
            .map(|&angle| cross_angle_distance(angle, frame_angle))
            .sum::<f64>()
            / alpha.len() as f64;
        let mut max_step = 0.0f64;
        for triangle in &tris {
            for (u, v) in [
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ] {
                max_step = max_step.max(cross_angle_distance(alpha[u], alpha[v]));
            }
        }
        rows.push((
            anisotropy,
            mean_distance,
            mean_cross_angle(&alpha),
            max_step,
        ));
    }
    Ok(AnisotropyReport { frame_angle, rows })
}

/// Measurements of one round of adaptive background refinement.
#[derive(Clone, Debug, Default)]
pub struct AdaptiveReport {
    /// Variation threshold in radians of cross-field angle.
    pub threshold: f64,
    /// Number of parent triangles whose variation exceeded the threshold.
    pub flagged_triangles: usize,
    /// Number of triangles before refinement.
    pub original_triangles: usize,
    /// Number of triangles after red-green refinement.
    pub refined_triangles: usize,
    /// Number of vertices before refinement.
    pub original_vertices: usize,
    /// Number of vertices after refinement (original vertices included).
    pub refined_vertices: usize,
    /// Mean 4-RoSy difference at the original vertices, before vs after.
    pub mean_field_difference: f64,
    /// Maximum 4-RoSy difference at the original vertices.
    pub max_field_difference: f64,
    /// `(valence-3, valence-5)` singularities on the original mesh.
    pub singularities_before: (usize, usize),
    /// `(valence-3, valence-5)` singularities on the refined mesh.
    pub singularities_after: (usize, usize),
}

/// Measures one round of adaptive refinement on a guided, graded `10 x 10`
/// square.
///
/// The guidance (`0.3 rad`, weight `1`) fixes the otherwise free global phase
/// so that the before/after angles are directly comparable. The refinement
/// threshold is `0.08 rad`; the refined field is solved with the same options
/// and the original vertices are read back directly (their indices are kept).
///
/// Returns an error only if one of the solves fails.
pub fn demo_adaptive() -> Result<AdaptiveReport, String> {
    let (points, tris) = demo_square_mesh(10);
    let size: Vec<f64> = points.iter().map(|point| 1.0 + point[0]).collect();
    let guidance = Guidance {
        weight: 1.0,
        angle: vec![0.3; points.len()],
    };
    let base = solve_with_guidance(&points, &tris, &size, &guidance, 100)?;
    let threshold = 0.08f64;
    let (refined_points, refined_tris, refined_size, flagged, edge_order) =
        refine_once(&points, &tris, &size, &base, threshold);
    if flagged == 0 {
        return Ok(AdaptiveReport {
            threshold,
            original_triangles: tris.len(),
            original_vertices: points.len(),
            refined_triangles: tris.len(),
            refined_vertices: points.len(),
            singularities_before: count_singularities(&base, &tris),
            ..AdaptiveReport::default()
        });
    }
    let extended = extend_guidance(&guidance, &edge_order, refined_points.len());
    let refined = solve_with_guidance(
        &refined_points,
        &refined_tris,
        &refined_size,
        &extended,
        100,
    )?;
    let mut mean = 0.0f64;
    let mut maximum = 0.0f64;
    for (before, after) in base.iter().zip(refined.iter()) {
        let d = cross_angle_distance(*before, *after);
        mean += d;
        maximum = maximum.max(d);
    }
    mean /= base.len() as f64;
    Ok(AdaptiveReport {
        threshold,
        flagged_triangles: flagged,
        original_triangles: tris.len(),
        refined_triangles: refined_tris.len(),
        original_vertices: points.len(),
        refined_vertices: refined_points.len(),
        mean_field_difference: mean,
        max_field_difference: maximum,
        singularities_before: count_singularities(&base, &tris),
        singularities_after: count_singularities(&refined, &refined_tris),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const HALF_PI: f64 = std::f64::consts::FRAC_PI_2;

    /// Counter-clockwise triangulation of the unit square with `n` cells per side.
    fn square_mesh(n: usize) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
        let mut points = Vec::with_capacity((n + 1) * (n + 1));
        let mut index = HashMap::new();
        for j in 0..=n {
            for i in 0..=n {
                index.insert((i, j), points.len());
                points.push([i as f64 / n as f64, j as f64 / n as f64, 0.0]);
            }
        }
        let mut tris = Vec::with_capacity(2 * n * n);
        for j in 0..n {
            for i in 0..n {
                let a = index[&(i, j)];
                let b = index[&(i + 1, j)];
                let c = index[&(i + 1, j + 1)];
                let d = index[&(i, j + 1)];
                tris.push([a, b, c]);
                tris.push([a, c, d]);
            }
        }
        (points, tris)
    }

    /// Distance between two cross-field angles, modulo `pi/2`.
    fn rosy_distance(a: f64, b: f64) -> f64 {
        let d = (a - b).rem_euclid(HALF_PI);
        d.min(HALF_PI - d)
    }

    fn max_pairwise_distance(alpha: &[f64]) -> f64 {
        let mut spread = 0.0f64;
        for i in 0..alpha.len() {
            for j in 0..alpha.len() {
                spread = spread.max(rosy_distance(alpha[i], alpha[j]));
            }
        }
        spread
    }

    fn max_edge_distance(alpha: &[f64], tris: &[[usize; 3]]) -> f64 {
        let mut maximum = 0.0f64;
        for triangle in tris {
            for (u, v) in [
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ] {
                maximum = maximum.max(rosy_distance(alpha[u], alpha[v]));
            }
        }
        maximum
    }

    /// Circular mean of the fourth-power angles, back in `[0, pi/2)`.
    fn mean_angle(alpha: &[f64]) -> f64 {
        let mut sum = Complex::ZERO;
        for &value in alpha {
            sum = sum.add(Complex::from_angle(4.0 * value));
        }
        cross_angle(sum)
    }

    /// Mean 4-RoSy distance to a constant target angle.
    fn mean_distance_to(alpha: &[f64], target: f64) -> f64 {
        let total: f64 = alpha
            .iter()
            .map(|&value| rosy_distance(value, target))
            .sum();
        total / alpha.len() as f64
    }

    #[test]
    fn helper_integrals_have_the_flat_limit() {
        let mass = mass_integral(0.0);
        assert!((mass.re - 1.0 / 12.0).abs() < 1e-15);
        assert!(mass.im.abs() < 1e-15);

        let dirichlet = dirichlet_integral(0.0, 3.0, -0.7, 2.0);
        assert!((dirichlet.re - 0.175).abs() < 1e-15);
        assert!(dirichlet.im.abs() < 1e-15);

        // The series and closed-form branches must join continuously at the
        // switch. (The closed form itself loses ~1e-11 relative accuracy at
        // Omega = 0.1 because of cancellation, which is why the series is used
        // below the threshold.)
        for omega in [
            INTEGRAL_SERIES_THRESHOLD - 1e-12,
            -INTEGRAL_SERIES_THRESHOLD + 1e-12,
        ] {
            let series_mass = mass_integral(omega);
            let closed_mass = mass_integral(omega.signum() * (INTEGRAL_SERIES_THRESHOLD + 1e-12));
            assert!((series_mass.re - closed_mass.re).abs() < 1e-10);
            assert!((series_mass.im - closed_mass.im).abs() < 1e-10);

            let series_dirichlet = dirichlet_integral(omega, 3.0, -0.7, 2.0);
            let closed_dirichlet = dirichlet_integral(
                omega.signum() * (INTEGRAL_SERIES_THRESHOLD + 1e-12),
                3.0,
                -0.7,
                2.0,
            );
            assert!((series_dirichlet.re - closed_dirichlet.re).abs() < 1e-10);
            assert!((series_dirichlet.im - closed_dirichlet.im).abs() < 1e-10);
        }
    }

    #[test]
    fn constant_size_gives_a_constant_field() -> Result<(), String> {
        let n = 8;
        let (points, tris) = square_mesh(n);
        let size = vec![1.0; points.len()];
        let alpha = solve(&points, &tris, &size, 100)?;

        assert_eq!(alpha.len(), points.len());
        for value in &alpha {
            assert!(value.is_finite(), "non-finite angle {value}");
            assert!(
                (0.0..HALF_PI).contains(value),
                "angle {value} outside [0, pi/2)"
            );
        }
        let spread = max_pairwise_distance(&alpha);
        let edge_spread = max_edge_distance(&alpha, &tris);
        println!(
            "constant-size {}x{} square (flat): vertices={}, triangles={}, \
             max pairwise 4-RoSy spread={spread:.3e} rad, max edge spread={edge_spread:.3e} rad",
            n,
            n,
            points.len(),
            tris.len()
        );
        assert!(spread < 0.05, "angle spread {spread} is not small");
        Ok(())
    }

    #[test]
    fn constant_size_accurate_gives_a_constant_field() -> Result<(), String> {
        let n = 8;
        let (points, tris) = square_mesh(n);
        let size = vec![1.0; points.len()];
        let alpha = solve_with_options(
            &points,
            &tris,
            &size,
            &Options {
                accurate: true,
                holomorphic: 0.0,
                ..Options::default()
            },
            100,
        )?;

        let spread = max_pairwise_distance(&alpha);
        let edge_spread = max_edge_distance(&alpha, &tris);
        println!(
            "constant-size {}x{} square (accurate): vertices={}, triangles={}, \
             max pairwise 4-RoSy spread={spread:.3e} rad, max edge spread={edge_spread:.3e} rad",
            n,
            n,
            points.len(),
            tris.len()
        );
        assert!(
            spread < 1e-6,
            "accurate constant-size angle spread {spread} exceeds 1e-6 rad"
        );
        assert!(
            edge_spread < 1e-6,
            "accurate constant-size edge spread {edge_spread} exceeds 1e-6 rad"
        );
        Ok(())
    }

    #[test]
    fn linear_growth_rotates_smoothly() -> Result<(), String> {
        let n = 8;
        let (points, tris) = square_mesh(n);
        let size: Vec<f64> = points.iter().map(|p| 1.0 + 0.5 * p[0]).collect();
        let alpha = solve(&points, &tris, &size, 100)?;

        let edge_spread = max_edge_distance(&alpha, &tris);
        let spread = max_pairwise_distance(&alpha);
        let (valence_3, valence_5) = count_singularities(&alpha, &tris);
        println!(
            "linear growth sigma=1+0.5x on a {n}x{n} square (flat): \
             max adjacent 4-RoSy angle step={edge_spread:.4} rad, \
             max pairwise 4-RoSy spread={spread:.4} rad, \
             singularities: valence-3={valence_3}, valence-5={valence_5}"
        );

        assert!(
            edge_spread < 0.12,
            "adjacent angle step {edge_spread} is too large"
        );
        assert!(
            spread > 0.2,
            "the field did not rotate measurably (spread {spread})"
        );
        Ok(())
    }

    #[test]
    fn linear_growth_accurate_rotates_smoothly() -> Result<(), String> {
        let n = 8;
        let (points, tris) = square_mesh(n);
        let size: Vec<f64> = points.iter().map(|p| 1.0 + 0.5 * p[0]).collect();
        let alpha = solve_with_options(
            &points,
            &tris,
            &size,
            &Options {
                accurate: true,
                holomorphic: 0.0,
                ..Options::default()
            },
            100,
        )?;
        let flat = solve(&points, &tris, &size, 100)?;

        let edge_spread = max_edge_distance(&alpha, &tris);
        let spread = max_pairwise_distance(&alpha);
        let (valence_3, valence_5) = count_singularities(&alpha, &tris);
        let mean_difference = alpha
            .iter()
            .zip(flat.iter())
            .map(|(&accurate, &approx)| rosy_distance(accurate, approx))
            .sum::<f64>()
            / alpha.len() as f64;
        println!(
            "linear growth sigma=1+0.5x on a {n}x{n} square (accurate): \
             max adjacent 4-RoSy angle step={edge_spread:.4} rad, \
             max pairwise 4-RoSy spread={spread:.4} rad, \
             singularities: valence-3={valence_3}, valence-5={valence_5}, \
             mean 4-RoSy difference to flat={mean_difference:.3e} rad"
        );

        assert!(
            edge_spread < 0.12,
            "adjacent angle step {edge_spread} is too large"
        );
        assert!(
            spread > 0.2,
            "the field did not rotate measurably (spread {spread})"
        );
        assert!(
            mean_difference < 0.05,
            "the accurate correction moved the field too far from the flat solution \
             (mean difference {mean_difference} rad)"
        );
        Ok(())
    }

    #[test]
    fn guided_square_aligns_better_than_unguided() -> Result<(), String> {
        let n = 8;
        let (points, tris) = square_mesh(n);
        let size = vec![1.0; points.len()];
        let unguided = solve(&points, &tris, &size, 100)?;
        let base = mean_angle(&unguided);
        let target = (base + 0.3).rem_euclid(HALF_PI);
        let guidance = Guidance {
            weight: 1.0,
            angle: vec![target; points.len()],
        };
        let guided = solve_with_guidance(&points, &tris, &size, &guidance, 100)?;

        let distance_unguided = mean_distance_to(&unguided, target);
        let distance_guided = mean_distance_to(&guided, target);
        let guided_spread = max_pairwise_distance(&guided);
        println!(
            "guided {}x{} square: target={target:.6} rad, s=0 mean 4-RoSy distance={distance_unguided:.6} rad, \
             s=1 mean 4-RoSy distance={distance_guided:.3e} rad, guided pairwise spread={guided_spread:.3e} rad",
            n, n
        );

        assert!(
            distance_unguided > 0.2,
            "the s=0 field is already aligned with the target ({distance_unguided})"
        );
        assert!(
            distance_guided < 1e-6,
            "guided field is not aligned with the target: mean distance {distance_guided} rad"
        );
        assert!(
            distance_guided < distance_unguided * 1e-3,
            "guidance did not improve alignment: {distance_guided} vs {distance_unguided}"
        );
        assert!(
            guided_spread < 1e-6,
            "guided constant-size field has spread {guided_spread} rad"
        );
        Ok(())
    }

    #[test]
    fn guided_graded_square_is_biased() -> Result<(), String> {
        let n = 8;
        let (points, tris) = square_mesh(n);
        let size: Vec<f64> = points.iter().map(|p| 1.0 + 0.5 * p[0]).collect();
        let unguided = solve(&points, &tris, &size, 100)?;
        let target = (mean_angle(&unguided) + 0.3).rem_euclid(HALF_PI);
        let guidance = Guidance {
            weight: 1.0,
            angle: vec![target; points.len()],
        };
        let guided = solve_with_guidance(&points, &tris, &size, &guidance, 100)?;
        let stronger = Guidance {
            weight: 10.0,
            angle: vec![target; points.len()],
        };
        let strongly_guided = solve_with_guidance(&points, &tris, &size, &stronger, 100)?;

        let before = mean_distance_to(&unguided, target);
        let after = mean_distance_to(&guided, target);
        let after_strong = mean_distance_to(&strongly_guided, target);
        println!(
            "graded {}x{} square, target={target:.6} rad: s=0 mean 4-RoSy distance={before:.6} rad,              s=1={after:.6} rad, s=10={after_strong:.6} rad",
            n, n
        );
        assert!(before > 0.2, "the graded s=0 field is already aligned");
        assert!(
            after < 0.5 * before,
            "weight 1 did not bias the graded field: {after} vs {before}"
        );
        assert!(
            after_strong < after,
            "a larger weight did not improve the alignment: {after_strong} vs {after}"
        );
        Ok(())
    }

    #[test]
    fn zero_weight_guidance_matches_the_unguided_field() -> Result<(), String> {
        let n = 4;
        let (points, tris) = square_mesh(n);
        let size: Vec<f64> = points.iter().map(|p| 1.0 + 0.5 * p[0]).collect();
        let unguided = solve(&points, &tris, &size, 100)?;
        let guidance = Guidance {
            weight: 0.0,
            angle: vec![0.2; points.len()],
        };
        let guided = solve_with_guidance(&points, &tris, &size, &guidance, 100)?;
        let difference = unguided
            .iter()
            .zip(guided.iter())
            .map(|(&a, &b)| rosy_distance(a, b))
            .fold(0.0f64, f64::max);
        println!("zero-weight guidance: max 4-RoSy difference to solve()={difference:.3e} rad");
        assert!(difference < 1e-12);
        Ok(())
    }

    #[test]
    fn interpolation_matches_vertices_and_rejects_outside_points() -> Result<(), String> {
        let (points, tris) = square_mesh(4);
        let size: Vec<f64> = points.iter().map(|p| 1.0 + 0.5 * p[0]).collect();
        let alpha = solve(&points, &tris, &size, 100)?;

        for (i, point) in points.iter().enumerate() {
            let value = interp_angle(&alpha, &points, &tris, [point[0], point[1]]);
            assert!(value.is_finite(), "interpolation at vertex {i} is NaN");
            assert!(
                rosy_distance(value, alpha[i]) < 1e-9,
                "interpolation at vertex {i} returned {value} instead of {}",
                alpha[i]
            );
        }

        let inside = interp_angle(&alpha, &points, &tris, [0.37, 0.62]);
        assert!(
            inside.is_finite(),
            "interpolation at an interior point is NaN"
        );
        assert!((0.0..HALF_PI).contains(&inside));

        let outside = interp_angle(&alpha, &points, &tris, [2.0, 2.0]);
        assert!(outside.is_nan(), "a point outside the mesh must give NaN");
        Ok(())
    }

    #[test]
    fn singularity_winding_is_detected() -> Result<(), String> {
        let sectors = 7;
        let mut points = vec![[0.0, 0.0, 0.0]];
        let mut positive = vec![0.0];
        let mut negative = vec![0.0];
        for k in 0..sectors {
            let phi = std::f64::consts::TAU * k as f64 / sectors as f64;
            points.push([phi.cos(), phi.sin(), 0.0]);
            positive.push((phi / 4.0).rem_euclid(HALF_PI));
            negative.push((-phi / 4.0).rem_euclid(HALF_PI));
        }
        let mut tris = Vec::new();
        for k in 0..sectors {
            tris.push([0, 1 + k, 1 + (k + 1) % sectors]);
        }
        let p3 = count_singularities(&positive, &tris);
        let p5 = count_singularities(&negative, &tris);
        println!("synthetic fan: positive winding={p3:?}, negative winding={p5:?}");
        assert_eq!(p3, (1, 0));
        assert_eq!(p5, (0, 1));
        Ok(())
    }

    #[test]
    fn single_triangle_has_a_constant_field() -> Result<(), String> {
        let points = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let tris = vec![[0, 1, 2]];
        let size = vec![1.0; 3];
        let alpha = solve(&points, &tris, &size, 50)?;
        let spread = max_pairwise_distance(&alpha);
        let value = interp_angle(&alpha, &points, &tris, [0.25, 0.25]);
        println!(
            "single triangle: max pairwise 4-RoSy spread={spread:.3e} rad, \
             barycentric query={value:.6} rad"
        );
        assert!(spread < 0.05, "the single-triangle field is not constant");
        assert!(value.is_finite());
        Ok(())
    }

    #[test]
    fn invalid_inputs_are_rejected() {
        let (points, tris) = square_mesh(2);
        let n = points.len();
        let size = vec![1.0; n];
        assert!(solve(&points, &tris, &size, 0).is_err());
        assert!(solve(&points, &tris, &size[..n - 1], 10).is_err());
        let mut bad_size = size.clone();
        bad_size[0] = 0.0;
        assert!(solve(&points, &tris, &bad_size, 10).is_err());
        let bad_tris = vec![[0, 1, n]];
        assert!(solve(&points, &bad_tris, &size, 10).is_err());
        let degenerate = vec![[0, 1, 1]];
        assert!(solve(&points, &degenerate, &size, 10).is_err());
        assert!(interp_angle(&size[..n - 1], &points, &tris, [0.5, 0.5]).is_nan());

        // Options must not change the invalid-input contract.
        assert!(solve_with_options(
            &points,
            &tris,
            &size,
            &Options {
                accurate: true,
                holomorphic: 0.0,
                ..Options::default()
            },
            0
        )
        .is_err());

        // Guidance validation.
        let short = Guidance {
            weight: 1.0,
            angle: vec![0.0; n - 1],
        };
        assert!(solve_with_guidance(&points, &tris, &size, &short, 10).is_err());
        let negative = Guidance {
            weight: -1.0,
            angle: vec![0.0; n],
        };
        assert!(solve_with_guidance(&points, &tris, &size, &negative, 10).is_err());
        let infinite = Guidance {
            weight: f64::INFINITY,
            angle: vec![0.0; n],
        };
        assert!(solve_with_guidance(&points, &tris, &size, &infinite, 10).is_err());
        let mut bad_angle = vec![0.0; n];
        bad_angle[0] = f64::NAN;
        let non_finite = Guidance {
            weight: 1.0,
            angle: bad_angle,
        };
        assert!(solve_with_guidance(&points, &tris, &size, &non_finite, 10).is_err());
        assert!(solve_with_guidance_and_options(
            &points,
            &tris,
            &size,
            &Guidance {
                weight: 1.0,
                angle: vec![0.0; n],
            },
            &Options {
                accurate: true,
                holomorphic: 0.0,
                ..Options::default()
            },
            0,
        )
        .is_err());
    }
    #[test]
    fn holomorphic_zero_matches_the_dirichlet_field() -> Result<(), String> {
        let (points, tris) = square_mesh(8);
        let size: Vec<f64> = points.iter().map(|p| 1.0 + 0.5 * p[0]).collect();
        let guidance = Guidance {
            weight: 1.0,
            angle: vec![0.23; points.len()],
        };
        let current = solve_with_guidance(&points, &tris, &size, &guidance, 100)?;
        let s0 = solve_with_guidance_and_options(
            &points,
            &tris,
            &size,
            &guidance,
            &Options {
                accurate: false,
                holomorphic: 0.0,
                ..Options::default()
            },
            100,
        )?;
        let difference = current
            .iter()
            .zip(s0.iter())
            .map(|(&a, &b)| rosy_distance(a, b))
            .fold(0.0f64, f64::max);
        println!("s = 0: max 4-RoSy difference to the historical field = {difference:.3e} rad");
        assert_eq!(current, s0);
        Ok(())
    }

    #[test]
    fn holomorphic_parameter_changes_a_guided_graded_field() -> Result<(), String> {
        let rows = demo_holomorphic()?;
        println!("holomorphic parameter on a guided, graded 10x10 square:");
        for (s, mean, maximum) in &rows {
            println!(
                "  s = {s:+.1}: mean 4-RoSy difference = {mean:.6e} rad, max = {maximum:.6e} rad"
            );
        }
        assert_eq!(rows.len(), 4);
        for (s, mean, maximum) in &rows {
            assert!(mean.is_finite() && maximum.is_finite());
            assert!(*maximum >= *mean);
            assert!(mean.abs() > 1.0e-6, "s = {s} did not change the field");
        }
        // The two extreme energies move the guided field by several degrees.
        let holomorphic = rows
            .iter()
            .find(|row| row.0 == 1.0)
            .ok_or("missing s = 1 row")?;
        let anti = rows
            .iter()
            .find(|row| row.0 == -1.0)
            .ok_or("missing s = -1 row")?;
        assert!(
            holomorphic.1 > 0.01,
            "s = 1 mean difference {} is too small",
            holomorphic.1
        );
        assert!(
            anti.1 > 0.01,
            "s = -1 mean difference {} is too small",
            anti.1
        );
        assert!(
            holomorphic.2 > 0.05,
            "s = 1 max difference {} is too small",
            holomorphic.2
        );
        Ok(())
    }

    #[test]
    fn holomorphic_parameter_is_validated() {
        let (points, tris) = square_mesh(2);
        let size = vec![1.0; points.len()];
        for s in [f64::NAN, f64::INFINITY, 1.5, -1.5] {
            let options = Options {
                accurate: false,
                holomorphic: s,
                ..Options::default()
            };
            assert!(
                solve_with_options(&points, &tris, &size, &options, 10).is_err(),
                "s = {s} was accepted"
            );
        }
    }

    #[test]
    fn boundary_tangent_aligns_the_square_and_the_hole() -> Result<(), String> {
        let report = demo_boundary()?;
        println!(
            "boundary tangent, 10x10 graded square: band vertices={}, \
             mean distance before={:.6e} rad, after={:.6e} rad, \
             max before={:.6e} rad, after={:.6e} rad, \
             singularities before={:?}, after={:?}",
            report.square_band_vertices,
            report.square_before_mean,
            report.square_after_mean,
            report.square_before_max,
            report.square_after_max,
            report.square_singularities_before,
            report.square_singularities_after
        );
        println!(
            "boundary tangent, annular plate with a circular hole: band vertices={}, \
             mean distance before={:.6e} rad, after={:.6e} rad, \
             max before={:.6e} rad, after={:.6e} rad, \
             singularities before={:?}, after={:?}",
            report.hole_band_vertices,
            report.hole_before_mean,
            report.hole_after_mean,
            report.hole_before_max,
            report.hole_after_max,
            report.hole_singularities_before,
            report.hole_singularities_after
        );
        assert!(report.square_band_vertices > 10);
        assert!(
            report.square_before_mean > 0.05,
            "the graded square is already boundary aligned ({})",
            report.square_before_mean
        );
        assert!(
            report.square_after_mean < 0.02,
            "square boundary not aligned after Tangent ({})",
            report.square_after_mean
        );
        assert!(report.square_after_mean < 0.1 * report.square_before_mean);
        assert!(report.hole_band_vertices > 10);
        assert!(
            report.hole_before_mean > 0.1,
            "the plate is already hole aligned ({})",
            report.hole_before_mean
        );
        assert!(
            report.hole_after_mean < 1e-6,
            "hole boundary not aligned after Tangent ({})",
            report.hole_after_mean
        );
        assert!(
            report.hole_after_mean < 0.01 * report.hole_before_mean,
            "the hole band improved only from {} to {}",
            report.hole_before_mean,
            report.hole_after_mean
        );
        assert_eq!(
            report.square_singularities_before,
            report.square_singularities_after
        );
        assert_eq!(
            report.hole_singularities_before,
            report.hole_singularities_after
        );
        Ok(())
    }

    #[test]
    fn weak_boundary_approaches_tangent() -> Result<(), String> {
        let (points, tris) = square_mesh(6);
        let size: Vec<f64> = points.iter().map(|point| 1.0 + 0.5 * point[0]).collect();
        let none = solve_with_options(&points, &tris, &size, &Options::default(), 100)?;
        let zero = solve_with_options(
            &points,
            &tris,
            &size,
            &Options {
                boundary: BoundaryMode::Weak(0.0),
                ..Options::default()
            },
            100,
        )?;
        assert_eq!(none, zero);
        let weak = solve_with_options(
            &points,
            &tris,
            &size,
            &Options {
                boundary: BoundaryMode::Weak(1.0e6),
                ..Options::default()
            },
            100,
        )?;
        let tangent = solve_with_options(
            &points,
            &tris,
            &size,
            &Options {
                boundary: BoundaryMode::Tangent,
                ..Options::default()
            },
            100,
        )?;
        let difference = weak
            .iter()
            .zip(tangent.iter())
            .map(|(&a, &b)| rosy_distance(a, b))
            .fold(0.0f64, f64::max);
        println!(
            "weak boundary w=1e6 vs exact Tangent: max 4-RoSy difference={difference:.3e} rad"
        );
        assert!(
            difference < 1e-4,
            "the weak penalty did not approach Dirichlet ({difference})"
        );
        Ok(())
    }

    #[test]
    fn anisotropy_aligns_a_stretched_rectangle() -> Result<(), String> {
        let report = demo_anisotropy()?;
        println!(
            "anisotropy on a 2x1 rectangle, frame angle={:.3} rad:",
            report.frame_angle
        );
        for (a, mean, angle, step) in &report.rows {
            println!(
                "  a={a:.1}: mean distance to frame={mean:.6e} rad, \
                 mean field angle={angle:.6e} rad, max adjacent step={step:.6e} rad"
            );
        }
        assert_eq!(report.rows.len(), 3);
        let isotropic = report.rows[0];
        let anisotropic = report.rows[2];
        assert_eq!(isotropic.0, 1.0);
        assert_eq!(anisotropic.0, 4.0);
        assert!(
            isotropic.1 > 0.1,
            "the isotropic field is unexpectedly aligned ({})",
            isotropic.1
        );
        assert!(
            anisotropic.1 < 0.3 * isotropic.1,
            "a=4 did not align the field: {} vs {}",
            anisotropic.1,
            isotropic.1
        );
        assert!(
            anisotropic.1 < 0.1,
            "a=4 is not close to the anisotropy frame ({})",
            anisotropic.1
        );
        assert!(
            anisotropic.3 < 0.2,
            "the anisotropic field is not smooth ({})",
            anisotropic.3
        );
        Ok(())
    }

    #[test]
    fn anisotropy_one_reproduces_the_isotropic_field() -> Result<(), String> {
        let (points, tris) = square_mesh(6);
        let size: Vec<f64> = points.iter().map(|point| 1.0 + 0.5 * point[0]).collect();
        let base = solve(&points, &tris, &size, 100)?;
        let options = Options {
            anisotropy: 1.0,
            anisotropy_angle: 0.7,
            ..Options::default()
        };
        let rotated_frame = solve_with_options(&points, &tris, &size, &options, 100)?;
        let difference = base
            .iter()
            .zip(rotated_frame.iter())
            .map(|(&a, &b)| rosy_distance(a, b))
            .fold(0.0f64, f64::max);
        println!(
            "anisotropy a=1 with a rotated frame: max 4-RoSy difference to solve()={difference:.3e} rad"
        );
        assert!(
            difference < 1e-10,
            "a=1 changed the isotropic field ({difference})"
        );
        Ok(())
    }

    #[test]
    fn adaptive_refinement_smooths_the_field() -> Result<(), String> {
        let report = demo_adaptive()?;
        println!(
            "adaptive refinement on a guided 10x10 graded square: threshold={:.3} rad, \
             flagged={} of {} triangles, refined mesh: {} triangles / {} vertices, \
             field difference mean={:.6e} rad, max={:.6e} rad, \
             singularities before={:?}, after={:?}",
            report.threshold,
            report.flagged_triangles,
            report.original_triangles,
            report.refined_triangles,
            report.refined_vertices,
            report.mean_field_difference,
            report.max_field_difference,
            report.singularities_before,
            report.singularities_after
        );
        assert!(
            report.flagged_triangles > 0,
            "no triangle was flagged at threshold {}",
            report.threshold
        );
        assert!(report.refined_triangles > report.original_triangles);
        assert!(report.refined_vertices > report.original_vertices);
        assert!(report.mean_field_difference.is_finite());
        assert!(report.max_field_difference.is_finite());
        assert!(
            report.mean_field_difference > 0.0,
            "the refined field is identical to the original field"
        );
        Ok(())
    }

    #[test]
    fn adaptive_option_returns_the_original_vertex_count() -> Result<(), String> {
        let (points, tris) = square_mesh(10);
        let size: Vec<f64> = points.iter().map(|point| 1.0 + point[0]).collect();
        let guidance = Guidance {
            weight: 1.0,
            angle: vec![0.3; points.len()],
        };
        let base = solve_with_guidance(&points, &tris, &size, &guidance, 100)?;
        let refined = solve_with_guidance_and_options(
            &points,
            &tris,
            &size,
            &guidance,
            &Options {
                adaptive: 0.08,
                ..Options::default()
            },
            100,
        )?;
        assert_eq!(refined.len(), base.len());
        let mean = refined
            .iter()
            .zip(base.iter())
            .map(|(&a, &b)| rosy_distance(a, b))
            .sum::<f64>()
            / base.len() as f64;
        println!("public adaptive option: mean 4-RoSy difference={mean:.3e} rad");
        assert!(mean > 0.0, "the adaptive option did not change the field");
        Ok(())
    }

    #[test]
    fn new_options_are_validated() {
        let (points, tris) = square_mesh(2);
        let size = vec![1.0; points.len()];
        let defaults = Options::default();
        assert_eq!(defaults.anisotropy, 1.0);
        assert_eq!(defaults.anisotropy_angle, 0.0);
        assert_eq!(defaults.boundary, BoundaryMode::None);
        assert_eq!(defaults.adaptive, 0.0);

        for anisotropy in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
            let options = Options {
                anisotropy,
                ..Options::default()
            };
            assert!(
                solve_with_options(&points, &tris, &size, &options, 10).is_err(),
                "anisotropy {anisotropy} was accepted"
            );
        }
        for angle in [f64::NAN, f64::INFINITY] {
            let options = Options {
                anisotropy: 2.0,
                anisotropy_angle: angle,
                ..Options::default()
            };
            assert!(
                solve_with_options(&points, &tris, &size, &options, 10).is_err(),
                "anisotropy angle {angle} was accepted"
            );
        }
        for weight in [f64::NAN, f64::INFINITY, -1.0] {
            let options = Options {
                boundary: BoundaryMode::Weak(weight),
                ..Options::default()
            };
            assert!(
                solve_with_options(&points, &tris, &size, &options, 10).is_err(),
                "weak boundary weight {weight} was accepted"
            );
        }
        for adaptive in [f64::NAN, f64::INFINITY, -0.1] {
            let options = Options {
                adaptive,
                ..Options::default()
            };
            assert!(
                solve_with_options(&points, &tris, &size, &options, 10).is_err(),
                "adaptive threshold {adaptive} was accepted"
            );
        }
    }
}
