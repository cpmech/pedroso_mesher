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
//!   off-diagonal entries are `r_pq` times the real triangle integral, and the
//!   complex conjugate is used for the transposed entry (Algorithm 1, lines
//!   15-32).
//! * The Dirichlet energy (`s = 0`, so `A` is `Delta`) and the `10^-8 M`
//!   regularization of line 33 are used.
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
//! # Deviations (documented on purpose)
//!
//! * **Helper integrals.** The curvature-aware `MassIJ` and `DirichletIJ`
//!   functions of Knöppel et al. (GODF, Appendix D) are not reproduced here.
//!   Their flat-triangle limits are used instead, exactly the approximations
//!   the thesis quotes for the mass matrix:
//!
//!   ```text
//!   M_ii = |t|/6,                 M_ij = r_ij |t|/12
//!   A_ii = |p_jk|^2 / (4|t|)
//!   A_ij = r_ij <p_ki, p_jk> / (4|t|)
//!   ```
//!
//!   with `p_jk = p_k - p_j`, `p_ki = p_i - p_k` and cyclic permutations.
//!   Consequently the `Omega_ijk` correction that GODF carries inside
//!   `MassIJ`/`DirichletIJ` is not applied; only the leading flat term is.
//!   The transport coefficients still carry the size-gradient connection, so
//!   the global graded behaviour of the field is retained, while the
//!   per-triangle curvature corrections are `O(Omega^2)` and are omitted.
//! * **Vertex gradients.** Algorithm 1 consumes a per-vertex gradient `g_i`.
//!   It is obtained here as the area-weighted average of the constant
//!   gradients of the incident triangles, as the task specifies. A uniform
//!   average would change the coefficients only slightly.
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
//!
//! # Not implemented
//!
//! Guidance/alignment terms, the holomorphic/anti-holomorphic split (`s != 0`),
//! anisotropic sizing (Chapter 4), adaptive background-mesh refinement,
//! singularity placement, boundary constraints, and the extraction of a quad
//! mesh. The input is assumed planar: `z` is ignored and only `x, y` are used.

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
    let n = validate(points, tris, size, iterations)?;

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

    // Pass 2: Hermitian mass and energy matrices (Algorithm 1, lines 12-32).
    let mut mass = Hermitian::zero(n, edge_a.clone(), edge_b.clone());
    let mut energy = Hermitian::zero(n, edge_a.clone(), edge_b.clone());
    for geometry in &geometries {
        let [i, j, k] = geometry.vertices;
        let area = geometry.area;
        let p = geometry.positions;

        mass.diagonal[i] += area / 6.0;
        mass.diagonal[j] += area / 6.0;
        mass.diagonal[k] += area / 6.0;

        let p_jk = sub2(p[2], p[1]);
        let p_ki = sub2(p[0], p[2]);
        let p_ij = sub2(p[1], p[0]);
        let quarter_area = (4.0 * area).recip();
        energy.diagonal[i] += dot2(p_jk, p_jk) * quarter_area;
        energy.diagonal[j] += dot2(p_ki, p_ki) * quarter_area;
        energy.diagonal[k] += dot2(p_ij, p_ij) * quarter_area;

        // The three unordered pairs, the geometric Dirichlet coefficient of
        // each, and the corresponding (canonical) edge.
        let pairs = [(i, j), (j, k), (k, i)];
        let coefficients = [
            dot2(p_ki, p_jk) * quarter_area,
            dot2(p_ij, p_ki) * quarter_area,
            dot2(p_jk, p_ij) * quarter_area,
        ];
        for (pair, coefficient) in pairs.iter().zip(coefficients.iter()) {
            let (u, v) = *pair;
            let key = if u < v { (u, v) } else { (v, u) };
            let edge = match edge_map.get(&key) {
                Some(&edge) => edge,
                None => {
                    return Err(
                        "internal error: a triangle edge is missing from the edge list".into(),
                    )
                }
            };
            // The canonical entry (a, b) is r_ab times the real integral; the
            // entry (b, a) is generated as its conjugate by `Hermitian::apply`.
            mass.off_diagonal[edge] =
                mass.off_diagonal[edge].add(transport[edge].scale(area / 12.0));
            energy.off_diagonal[edge] =
                energy.off_diagonal[edge].add(transport[edge].scale(*coefficient));
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
    let shifted = energy.shifted(shift, &mass)?;
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

    let mut angles = Vec::with_capacity(n);
    for value in &x {
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err("the computed field contains a non-finite value".into());
        }
        angles.push(cross_angle(*value));
    }
    Ok(angles)
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
            "constant-size {}x{} square: vertices={}, triangles={}, \
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
    fn linear_growth_rotates_smoothly() -> Result<(), String> {
        let n = 8;
        let (points, tris) = square_mesh(n);
        let size: Vec<f64> = points.iter().map(|p| 1.0 + 0.5 * p[0]).collect();
        let alpha = solve(&points, &tris, &size, 100)?;

        let edge_spread = max_edge_distance(&alpha, &tris);
        let spread = max_pairwise_distance(&alpha);
        let (valence_3, valence_5) = count_singularities(&alpha, &tris);
        println!(
            "linear growth sigma=1+0.5x on a {n}x{n} square: \
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
    }
}
