//! Element quality: the per-element metrics the mesher is judged by.
//!
//! The contract metrics follow Knupp's Jacobian framework. The definitions,
//! formulas, ranges and sources are:
//!
//! * **Scaled Jacobian (SJ)** at a corner: `det J / (||c1|| ||c2|| ||c3||)`,
//!   where `c1, c2, c3` are the columns of the Jacobian `d x / d xi` (the images
//!   of the parametric axes). This is `det(Q)` for the matrix whose columns are
//!   normalised, the quantity Knupp calls the scaled Jacobian
//!   (`knupp-2000-part2`, section 5, "The ideal element"; the CUBIT name is
//!   cited there). The planar case applies the same normalisation to the 2x2
//!   Jacobian of `knupp-2000-part1`, section 2. Range `[-1, 1]`: 1 for a cube or
//!   a right-angled quad,
//!   `sin(theta)` for a sheared cell, 0 for a degenerate one, negative for an
//!   inverted one.
//! * **Corner angle**: the angle between the two element edges meeting at a
//!   node, in degrees, `acos(e_prev . e_next / (||e_prev|| ||e_next||))`.
//!   Standard metric (not from the corpus). Range `[0, 180]`; 90 is ideal for a
//!   quad. For a quad the two measures are related: at a corner `SJ = sin(angle)`.
//! * **Dihedral angle** at a hex edge: `180° - angle(n1, n2)` between the
//!   outward normals of the two faces meeting there. Standard metric (Knupp's
//!   papers compute 2D corner angles, not hex dihedrals). Range `[0, 180]`; 90
//!   for a cube.
//! * **Aspect ratio**: longest edge over shortest edge. Standard metric
//!   (Knupp's aspect-ratio objective is a matrix-norm quantity based on the
//!   diagonal of the metric tensor, `knupp-2000-part1`, not this geometric
//!   ratio). Range `[1, inf)`; 1 for an equilateral element.
//!
//! Everything is computed from the trilinear (hex) or bilinear (quad) map, not
//! from an approximation, so the values match what a finite element code sees.
//! What is approximated or missing: the Jacobian is evaluated at the element's
//! corner nodes only (no interior sampling and no minimum over integration
//! points), there is no warp, taper or skew metric, and a `Tri3` has no
//! parametric Jacobian at all — [`scaled_jacobian`] returns 1.0 for every
//! triangle, so a reversed triangle is invisible here (the gate checks it by
//! signed area instead). The planar metrics use the x and y coordinates only,
//! relying on the z = 0 convention of [`crate::mesh`]; a non-planar quad is
//! measured by its projection.

use crate::mesh::{Cell, Kind, Mesh, Node};

/// Local corner coordinates of a `Hex8`, matching [`Kind`]'s node order.
///
/// Entry `i` is `(xi, eta, zeta)` of node `i` in the trilinear reference cube
/// `[-1, 1]^3`: the bottom ring `0..3` at `zeta = -1`, then the top ring `4..7`
/// at `zeta = +1`, counter-clockwise seen from +Z. Each coordinate is `±1`, so
/// the shape-function derivatives below are the standard
/// `dN_i/dxi = xi_i (1 + eta_i eta)(1 + zeta_i zeta) / 8`.
const HEX_XI: [[f64; 3]; 8] = [
    [-1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0],
    [1.0, 1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0],
    [1.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0],
];

/// Local corner coordinates of a `Quad4`.
///
/// Entry `i` is `(xi, eta)` of node `i` in the bilinear reference square
/// `[-1, 1]^2`, counter-clockwise from `(-1, -1)`, matching [`Kind`]'s node
/// order. The shape-function derivatives are
/// `dN_i/dxi = xi_i (1 + eta_i eta) / 4` (and the symmetric one in `eta`).
const QUAD_XI: [[f64; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];

/// Thresholds a mesh is judged against.
///
/// The defaults are the P0 contract: refuse anything with an inverted cell,
/// target a scaled Jacobian of 0.2, keep hex dihedral angles in the 30..150°
/// band, and accept a geometric aspect ratio up to 10 while refusing above 100.
/// The 0.2 target is the usual scaled-Jacobian floor for a usable element, the
/// same order as the ≈0.300 that Knupp's optimiser reached on whisker-weaved
/// meshes (`knupp-2000-part2`, summary). The remaining numbers are project
/// choices and rules of thumb, not values taken from a corpus paper.
#[derive(Clone, Copy, Debug)]
pub struct Thresholds {
    /// Hard floor for the scaled Jacobian. An element whose smallest corner is
    /// at or below this counts as inverted, and the mesh is refused. Default
    /// 0.0 (so "positive Jacobian" is the validity rule).
    pub hard_scaled_jacobian: f64,

    /// Target for every corner of every element. An element below it is counted
    /// in `Report::below_target`; missing it makes the verdict `BelowTarget`,
    /// not `Refused`. Default 0.2.
    pub target_scaled_jacobian: f64,

    /// Lower bound of the acceptable hex dihedral band, in degrees. Below it
    /// the verdict is `BelowTarget`. Default 30.
    pub min_dihedral_deg: f64,

    /// Upper bound of the acceptable hex dihedral band, in degrees. Above it
    /// the verdict is `BelowTarget`. Default 150.
    pub max_dihedral_deg: f64,

    /// Aspect ratio above this is reported as `BelowTarget`. Default 10.
    pub target_aspect_ratio: f64,

    /// Aspect ratio above this refuses the mesh. Default 100.
    pub hard_aspect_ratio: f64,
}

impl Default for Thresholds {
    fn default() -> Self {
        Thresholds {
            hard_scaled_jacobian: 0.0,
            target_scaled_jacobian: 0.2,
            min_dihedral_deg: 30.0,
            max_dihedral_deg: 150.0,
            target_aspect_ratio: 10.0,
            hard_aspect_ratio: 100.0,
        }
    }
}

/// How a mesh fares against the contract.
///
/// The precedence is `Refused` over `BelowTarget` over `Acceptable`: a hard
/// failure hides the softer target misses, and a target miss is only reported
/// when the mesh is otherwise valid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Every element meets every target and no hard threshold was broken.
    Acceptable,
    /// Valid (no inverted element), but at least one target was missed; usable
    /// with a caveat and reported in `Report::reasons`.
    BelowTarget,

    /// Must not be shipped: an inverted element or a hard threshold was broken.
    Refused,
}

/// Per-mesh quality summary: the aggregate of the per-element metrics plus the
/// verdict and its human-readable reasons.
///
/// All angles are degrees and all lengths are in the mesh's unit (millimetres).
/// The Jacobian fields are dimensionless. Fields that do not apply to a non-empty
/// mesh are NaN, not zero, so a caller cannot mistake "not measured" for
/// "perfect": dihedral angles are NaN when there is no hex, and corner angles
/// are NaN when the mesh is purely 3D. The empty mesh is the exception: it is
/// refused before that normalisation, so its angle fields keep the `INFINITY`
/// sentinels and `worst_element` is `None`.
#[derive(Clone, Debug)]
pub struct Report {
    /// Number of cells in the mesh.
    pub elements: usize,

    /// Smallest scaled Jacobian over every corner of every element;
    /// dimensionless, in `[-1, 1]` for a valid mesh. `INFINITY` for an empty
    /// mesh (nothing was measured).
    pub min_scaled_jacobian: f64,

    /// Arithmetic mean of the scaled Jacobian over every corner of every
    /// element; dimensionless. Triangles contribute 1.0 per corner. 0.0 for an
    /// empty mesh (the mean is not defined there).
    pub mean_scaled_jacobian: f64,

    /// Largest scaled Jacobian over every corner of every element;
    /// dimensionless, at most 1 for a non-inverted element. `-INFINITY` for an
    /// empty mesh.
    pub max_scaled_jacobian: f64,

    /// Index of the element holding the minimum scaled Jacobian; the first one
    /// in cell order when several tie. `None` for an empty mesh.
    pub worst_element: Option<usize>,

    /// Elements whose smallest corner is at or below `hard_scaled_jacobian`, or
    /// NaN (degenerate). Must be 0 for `Acceptable`/`BelowTarget`.
    pub inverted: usize,

    /// Elements whose smallest corner is below `target_scaled_jacobian`. A
    /// count, not a flag; the verdict uses it only in the reason text.
    pub below_target: usize,

    /// Smallest dihedral angle over all 12 edges of all hexes, in degrees;
    /// `[0, 180]`, 90 for a cube. NaN when a non-empty mesh has no hexahedron
    /// (an empty mesh keeps `INFINITY`, see the struct note).
    pub min_dihedral_deg: f64,

    /// Largest dihedral angle over all 12 edges of all hexes, in degrees;
    /// `[0, 180]`, 90 for a cube. NaN when a non-empty mesh has no hexahedron.
    pub max_dihedral_deg: f64,

    /// Largest aspect ratio (longest edge / shortest edge) over all cells;
    /// dimensionless, `>= 1`, `INFINITY` for a degenerate cell.
    pub max_aspect_ratio: f64,

    /// Smallest corner angle over planar elements (quads and triangles), in
    /// degrees; NaN when the mesh is purely 3D. This is the angle the W3 gate
    /// holds to `>= 30°`.
    pub min_corner_angle_deg: f64,

    /// Largest corner angle over planar elements, in degrees; NaN when the mesh
    /// is purely 3D. A value above 180 would indicate a reflex corner, which a
    /// valid planar cell does not have.
    pub max_corner_angle_deg: f64,

    /// The overall verdict; `Refused` takes precedence over `BelowTarget`.
    pub verdict: Verdict,

    /// Human-readable reasons for a non-`Acceptable` verdict, one per broken
    /// hard threshold or missed target. Empty for `Acceptable`.
    pub reasons: Vec<String>,
}

impl Report {
    /// One-line summary for logs and the CLI.
    ///
    /// The format is `N element(s), min SJ …, mean …, dihedral …, aspect …,
    /// verdict`, with the dihedral band replaced by `n/a (no hex)` when no
    /// hexahedron was measured. It is for humans; parse the struct fields or the
    /// JSON output when a machine needs the values.
    pub fn summary(&self) -> String {
        let verdict = match self.verdict {
            Verdict::Acceptable => "acceptable",
            Verdict::BelowTarget => "below target",
            Verdict::Refused => "REFUSED",
        };
        let dihedral = if self.min_dihedral_deg.is_finite() {
            format!(
                "{:.1}..{:.1}°",
                self.min_dihedral_deg, self.max_dihedral_deg
            )
        } else {
            "n/a (no hex)".to_string()
        };
        format!(
            "{} element(s), min SJ {:.4}, mean {:.4}, dihedral {dihedral}, aspect {:.2}, {verdict}",
            self.elements,
            self.min_scaled_jacobian,
            self.mean_scaled_jacobian,
            self.max_aspect_ratio
        )
    }
}

// --------------------------------------------------------------------------- //
// geometry helpers
// --------------------------------------------------------------------------- //

/// Difference of two points (`a - b`), component by component.
fn sub(a: Node, b: Node) -> Node {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Euclidean dot product of two 3D vectors.
fn dot(a: Node, b: Node) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Euclidean norm of a 3D vector.
fn norm(a: Node) -> f64 {
    dot(a, a).sqrt()
}

/// Determinant of a 3x3 matrix stored row-major, by cofactor expansion along
/// the first row.
fn det3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Jacobian `d x / d xi` of the trilinear hex at a local point `xi`.
///
/// Returns a 3x3 matrix `J[row][column]` where the row is the spatial coordinate
/// (x, y, z) and the column is the parametric direction (xi, eta, zeta), i.e.
/// `J = [c0 c1 c2]` with the columns as vectors. The bubble-free trilinear
/// shape functions are
/// `N_i = (1 + xi_i xi)(1 + eta_i eta)(1 + zeta_i zeta) / 8`, differentiated
/// analytically; the `0.125` is `1/8`.
fn hex_jacobian(points: &[Node], cell: &Cell, xi: [f64; 3]) -> [[f64; 3]; 3] {
    let mut j = [[0.0f64; 3]; 3];
    // Accumulate J = sum_i x_i (grad N_i)^T over the eight corners.
    for (i, local) in HEX_XI.iter().enumerate() {
        let p = points[cell.nodes[i]];
        // dN_i/dxi, dN_i/deta, dN_i/dzeta
        let derivatives = [
            0.125 * local[0] * (1.0 + local[1] * xi[1]) * (1.0 + local[2] * xi[2]),
            0.125 * local[1] * (1.0 + local[0] * xi[0]) * (1.0 + local[2] * xi[2]),
            0.125 * local[2] * (1.0 + local[0] * xi[0]) * (1.0 + local[1] * xi[1]),
        ];
        // J[coordinate][parameter] = sum_i dN_i/dparameter * x_i[coordinate]
        for (column, d) in derivatives.into_iter().enumerate() {
            j[0][column] += d * p[0];
            j[1][column] += d * p[1];
            j[2][column] += d * p[2];
        }
    }
    j
}

/// Jacobian of the bilinear quad, as a 2x2 matrix in the xy plane.
///
/// Rows are (x, y), columns are (xi, eta); the z coordinate is ignored because
/// planar meshes live in z = 0. The shape functions are
/// `N_i = (1 + xi_i xi)(1 + eta_i eta) / 4`, again differentiated analytically.
/// For a non-planar quad the matrix is the projection onto the xy plane, not the
/// in-surface metric.
fn quad_jacobian(points: &[Node], cell: &Cell, xi: [f64; 2]) -> [[f64; 2]; 2] {
    let mut j = [[0.0f64; 2]; 2];
    for (i, local) in QUAD_XI.iter().enumerate() {
        let p = points[cell.nodes[i]];
        let derivatives = [
            0.25 * local[0] * (1.0 + local[1] * xi[1]),
            0.25 * local[1] * (1.0 + local[0] * xi[0]),
        ];
        for (column, d) in derivatives.into_iter().enumerate() {
            j[0][column] += d * p[0];
            j[1][column] += d * p[1];
        }
    }
    j
}

// --------------------------------------------------------------------------- //
// per-element metrics
// --------------------------------------------------------------------------- //

/// Scaled Jacobian at every corner, in the cell's node order.
///
/// `SJ = det J / (||c0|| ||c1|| ||c2||)` for a hex and
/// `SJ = det J / (||c0|| ||c1||)` for a quad, with the columns `c_i` of the
/// corner Jacobian; this is `det(Q)` of the column-normalised matrix
/// (`knupp-2000-part2`, section 5). Range `[-1, 1]`: positive for a valid cell,
/// 1 for a right-angled one, negative for an inverted one. A NaN is returned
/// when a column has zero length, i.e. the cell is degenerate; the report treats
/// a NaN as inverted.
///
/// Only the corners are sampled (the `HEX_XI`/`QUAD_XI` points). The returned
/// vector has one entry per cell node, so it can be zipped with the node order
/// for diagnostics.
///
/// A `Tri3` has no corner Jacobian in this parameterisation. The function
/// returns `1.0` for each of its three corners as a neutral placeholder, which
/// means the scaled Jacobian cannot see a reversed or degenerate triangle; the
/// gate checks triangles by signed area instead (see `verify::cell_inverted`).
pub fn scaled_jacobian(cell: &Cell, points: &[Node]) -> Vec<f64> {
    match cell.kind {
        Kind::Hex8 => HEX_XI
            .iter()
            .map(|&xi| {
                let j = hex_jacobian(points, cell, xi);
                let c0 = [j[0][0], j[1][0], j[2][0]];
                let c1 = [j[0][1], j[1][1], j[2][1]];
                let c2 = [j[0][2], j[1][2], j[2][2]];
                let scale = norm(c0) * norm(c1) * norm(c2);
                if scale > 0.0 {
                    det3(&j) / scale
                } else {
                    f64::NAN
                }
            })
            .collect(),
        // Placeholder: see the doc comment above. `verify` does not rely on this
        // value for triangles.
        Kind::Tri3 => vec![1.0; 3],
        Kind::Quad4 => QUAD_XI
            .iter()
            .map(|&xi| {
                let j = quad_jacobian(points, cell, xi);
                let c0 = [j[0][0], j[1][0], 0.0];
                let c1 = [j[0][1], j[1][1], 0.0];
                let scale = norm(c0) * norm(c1);
                let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
                if scale > 0.0 {
                    det / scale
                } else {
                    f64::NAN
                }
            })
            .collect(),
    }
}

/// Signed volume of a hex, or signed area of a quad, by Gauss integration.
///
/// Two Gauss points per direction (at `±1/sqrt(3)`, weight 1) integrate the
/// degree-three Jacobian exactly, so the result is the true measure of the
/// trilinear/bilinear element, not an approximation. The sign follows the cell
/// orientation: positive for the node order of [`crate::mesh::Kind`], negative
/// for an inverted cell, and zero for a degenerate one. Units are the mesh unit
/// cubed for a hex and squared for a planar cell.
///
/// A `Tri3` uses the z component of the edge cross product, i.e. its signed area
/// in the xy plane; that is exact for the z = 0 convention the 2D pipeline
/// keeps. A quad's area is likewise its xy projection, so a non-planar quad is
/// not measured by its surface area.
pub fn volume(cell: &Cell, points: &[Node]) -> f64 {
    let g = 1.0 / 3.0_f64.sqrt();
    match cell.kind {
        Kind::Hex8 => {
            let mut sum = 0.0;
            for &a in &[-g, g] {
                for &b in &[-g, g] {
                    for &c in &[-g, g] {
                        sum += det3(&hex_jacobian(points, cell, [a, b, c]));
                    }
                }
            }
            sum
        }
        Kind::Tri3 => {
            let a = points[cell.nodes[0]];
            let b = points[cell.nodes[1]];
            let c = points[cell.nodes[2]];
            let u = sub(b, a);
            let v = sub(c, a);
            (u[0] * v[1] - u[1] * v[0]) / 2.0
        }
        Kind::Quad4 => {
            let mut sum = 0.0;
            for &a in &[-g, g] {
                for &b in &[-g, g] {
                    let j = quad_jacobian(points, cell, [a, b]);
                    sum += j[0][0] * j[1][1] - j[0][1] * j[1][0];
                }
            }
            sum
        }
    }
}

/// Longest edge over shortest edge.
///
/// A standard geometric metric (not taken from a corpus paper; Knupp's
/// aspect-ratio objective is a different, matrix-norm quantity). Range
/// `[1, inf)`: 1 for an equilateral element, larger for a stretched one, and
/// `INFINITY` when an edge has zero length. Edges come from
/// [`Kind::edges`], so the count is 4 for a quad, 3 for a triangle and 12 for a
/// hex.
pub fn aspect_ratio(cell: &Cell, points: &[Node]) -> f64 {
    // Sentinel extremes: every valid kind has at least three edges, so both are
    // overwritten. A zero-length edge leaves `lo == 0` and yields INFINITY.
    let mut lo = f64::INFINITY;
    let mut hi = 0.0f64;
    for &(a, b) in cell.kind.edges() {
        let e = norm(sub(points[cell.nodes[a]], points[cell.nodes[b]]));
        lo = lo.min(e);
        hi = hi.max(e);
    }
    if lo > 0.0 {
        hi / lo
    } else {
        f64::INFINITY
    }
}

/// Corner angles of a quad, in degrees, in node order.
///
/// At node `i` the angle is between the edges to the previous and next node in
/// the ring, `acos(e_prev . e_next / (||e_prev|| ||e_next||))`, clamped to
/// `[-1, 1]` before the `acos` so that round-off cannot produce a NaN. A
/// standard metric; range `[0, 180]`, 90 for a square. The value is NaN when an
/// incident edge has zero length. The function is also applied to a `Tri3` by
/// [`corner_angles`], which uses the same ring formula.
pub fn quad_angles(cell: &Cell, points: &[Node]) -> Vec<f64> {
    let n = cell.nodes.len();
    (0..n)
        .map(|i| {
            let prev = sub(points[cell.nodes[(i + n - 1) % n]], points[cell.nodes[i]]);
            let next = sub(points[cell.nodes[(i + 1) % n]], points[cell.nodes[i]]);
            let denom = norm(prev) * norm(next);
            if denom > 0.0 {
                (dot(prev, next) / denom)
                    .clamp(-1.0, 1.0)
                    .acos()
                    .to_degrees()
            } else {
                f64::NAN
            }
        })
        .collect()
}

/// Corner angles of a planar element (quad or triangle), in degrees.
///
/// Returns 4 values for a `Quad4`, 3 for a `Tri3` and an empty vector for a
/// `Hex8` (a solid has no planar corner angle). The per-corner definition and
/// range are those of [`quad_angles`].
pub fn corner_angles(cell: &Cell, points: &[Node]) -> Vec<f64> {
    if cell.kind == Kind::Quad4 {
        return quad_angles(cell, points);
    }
    if cell.kind != Kind::Tri3 {
        return Vec::new();
    }
    let n = cell.nodes.len();
    (0..n)
        .map(|i| {
            let prev = sub(points[cell.nodes[(i + n - 1) % n]], points[cell.nodes[i]]);
            let next = sub(points[cell.nodes[(i + 1) % n]], points[cell.nodes[i]]);
            let denom = norm(prev) * norm(next);
            if denom > 0.0 {
                (dot(prev, next) / denom)
                    .clamp(-1.0, 1.0)
                    .acos()
                    .to_degrees()
            } else {
                f64::NAN
            }
        })
        .collect()
}

/// Dihedral angle at each of the 12 hex edges, in degrees, in edge order.
///
/// For each edge, take the two faces that contain it and the angle between their
/// outward normals; the dihedral angle is `180° - that angle` (the normals of a
/// cube's adjacent faces are perpendicular, so the dihedral is 90°). Range
/// `[0, 180]`, clamped against round-off. This is a standard metric; the corpus
/// papers define corner angles, not hex dihedrals. Returns an empty vector for a
/// non-hex cell and NaN for an edge not shared by exactly two faces (only a
/// malformed cell can produce that).
pub fn dihedral_angles(cell: &Cell, points: &[Node]) -> Vec<f64> {
    if cell.kind != Kind::Hex8 {
        return Vec::new();
    }
    let faces = cell.kind.faces();
    // One normal per face. The face loops in `Kind::faces` are outward-facing;
    // for adjacent faces `n1 . n2` is 1 when they are coplanar (dihedral 180°),
    // 0 at a right angle (90°) and -1 when folded flat onto each other (0°).
    let normals: Vec<Node> = faces
        .iter()
        .map(|f| {
            let p: Vec<Node> = f.iter().map(|&i| points[cell.nodes[i]]).collect();
            // Newell's method: robust for a slightly warped quad face.
            let mut n = [0.0f64; 3];
            for i in 0..p.len() {
                let a = p[i];
                let b = p[(i + 1) % p.len()];
                n[0] += (a[1] - b[1]) * (a[2] + b[2]);
                n[1] += (a[2] - b[2]) * (a[0] + b[0]);
                n[2] += (a[0] - b[0]) * (a[1] + b[1]);
            }
            n
        })
        .collect();

    cell.kind
        .edges()
        .iter()
        .map(|&(a, b)| {
            let mut meeting = Vec::new();
            for (fi, face) in faces.iter().enumerate() {
                if face.contains(&a) && face.contains(&b) {
                    meeting.push(fi);
                }
            }
            if meeting.len() != 2 {
                return f64::NAN;
            }
            let (n1, n2) = (normals[meeting[0]], normals[meeting[1]]);
            let denom = norm(n1) * norm(n2);
            if denom <= 0.0 {
                return f64::NAN;
            }
            let angle = (dot(n1, n2) / denom).clamp(-1.0, 1.0).acos().to_degrees();
            (180.0 - angle).clamp(0.0, 180.0)
        })
        .collect()
}

// --------------------------------------------------------------------------- //
// report
// --------------------------------------------------------------------------- //

/// Judge a mesh against the default contract.
///
/// Shorthand for [`report_with`] with [`Thresholds::default`]; this is the call
/// the CLI and the library example use.
pub fn report(mesh: &Mesh) -> Report {
    report_with(mesh, &Thresholds::default())
}

/// Judge a mesh against explicit thresholds.
///
/// One pass over the cells accumulates the minima/maxima/mean, the inversion and
/// target counts, the aspect ratio and the applicable angle metrics, then the
/// verdict is derived from the thresholds. The empty mesh is refused with a
/// reason, and its sentinel `INFINITY` fields must not be read as a perfect
/// mesh. For a non-empty mesh, metrics that do not apply are set to NaN rather
/// than left at their `INFINITY` sentinels (dihedral when there is no hex,
/// corner angles when there is no planar cell).
pub fn report_with(mesh: &Mesh, thresholds: &Thresholds) -> Report {
    let mut report = Report {
        elements: mesh.cells.len(),
        min_scaled_jacobian: f64::INFINITY,
        mean_scaled_jacobian: 0.0,
        max_scaled_jacobian: f64::NEG_INFINITY,
        worst_element: None,
        inverted: 0,
        below_target: 0,
        min_dihedral_deg: f64::INFINITY,
        max_dihedral_deg: f64::NEG_INFINITY,
        max_aspect_ratio: 0.0,
        min_corner_angle_deg: f64::INFINITY,
        max_corner_angle_deg: f64::NEG_INFINITY,
        verdict: Verdict::Acceptable,
        reasons: Vec::new(),
    };
    // An empty mesh has no element to measure. Refuse it here so that no caller
    // reads the sentinel INFINITY fields as "perfect".
    if mesh.cells.is_empty() {
        report.verdict = Verdict::Refused;
        report.reasons.push("the mesh has no elements".to_string());
        return report;
    }

    // Mean over every corner of every cell, not over elements: a hex therefore
    // contributes eight samples and a quad four. That is the documented
    // element-corner average, not a bug.
    let mut total = 0.0;
    let mut corners = 0usize;
    for (index, cell) in mesh.cells.iter().enumerate() {
        let sj = scaled_jacobian(cell, &mesh.points);
        let smallest = sj.iter().copied().fold(f64::INFINITY, f64::min);
        // Strict `<` keeps the first element in cell order on a tie.
        if smallest < report.min_scaled_jacobian {
            report.min_scaled_jacobian = smallest;
            report.worst_element = Some(index);
        }
        report.max_scaled_jacobian = report
            .max_scaled_jacobian
            .max(sj.iter().copied().fold(f64::NEG_INFINITY, f64::max));
        for &value in &sj {
            total += value;
            corners += 1;
        }
        // `<= hard` is the validity rule: a zero Jacobian is degenerate and is
        // as unusable as a negative one. NaN means a zero-length column, also
        // degenerate.
        if smallest <= thresholds.hard_scaled_jacobian || smallest.is_nan() {
            report.inverted += 1;
        }
        if smallest < thresholds.target_scaled_jacobian {
            report.below_target += 1;
        }

        let aspect = aspect_ratio(cell, &mesh.points);
        report.max_aspect_ratio = report.max_aspect_ratio.max(aspect);

        // Corner angles only exist for planar cells; a hex contributes no
        // entry, and a purely 3D mesh is normalised to NaN below.
        if cell.kind.dim() == 2 {
            for value in corner_angles(cell, &mesh.points) {
                if value.is_finite() {
                    report.min_corner_angle_deg = report.min_corner_angle_deg.min(value);
                    report.max_corner_angle_deg = report.max_corner_angle_deg.max(value);
                }
            }
        }
        if cell.kind == Kind::Hex8 {
            for value in dihedral_angles(cell, &mesh.points) {
                if value.is_finite() {
                    report.min_dihedral_deg = report.min_dihedral_deg.min(value);
                    report.max_dihedral_deg = report.max_dihedral_deg.max(value);
                }
            }
        }
    }
    report.mean_scaled_jacobian = if corners > 0 {
        total / corners as f64
    } else {
        f64::NAN
    };

    // Normalise "not measured" from the `INFINITY` accumulators to NaN, so the
    // summary and the callers can tell it apart from a perfect value.
    if !report.min_corner_angle_deg.is_finite() {
        report.min_corner_angle_deg = f64::NAN;
        report.max_corner_angle_deg = f64::NAN;
    }
    if !report.min_dihedral_deg.is_finite() {
        // no hexahedra: dihedral angles do not apply
        report.min_dihedral_deg = f64::NAN;
        report.max_dihedral_deg = f64::NAN;
    }
    // Hard failures come first and override every target check below: an
    // inverted element or an aspect ratio over the hard limit refuses the mesh.
    if report.inverted > 0 {
        report.verdict = Verdict::Refused;
        report.reasons.push(format!(
            "{} element(s) inverted or degenerate (scaled Jacobian <= {})",
            report.inverted, thresholds.hard_scaled_jacobian
        ));
    }
    if report.max_aspect_ratio > thresholds.hard_aspect_ratio {
        report.verdict = Verdict::Refused;
        report.reasons.push(format!(
            "aspect ratio {:.1} exceeds the hard limit {:.1}",
            report.max_aspect_ratio, thresholds.hard_aspect_ratio
        ));
    }
    // Target misses only soften an otherwise valid mesh; they never upgrade a
    // refused one. NaN comparisons are false, so an inapplicable metric (no hex)
    // does not trigger its target.
    if report.verdict != Verdict::Refused {
        if report.min_scaled_jacobian < thresholds.target_scaled_jacobian {
            report.verdict = Verdict::BelowTarget;
            report.reasons.push(format!(
                "min scaled Jacobian {:.3} is below the target {:.2} ({} element(s) affected)",
                report.min_scaled_jacobian, thresholds.target_scaled_jacobian, report.below_target
            ));
        }
        if report.min_dihedral_deg < thresholds.min_dihedral_deg {
            report.verdict = Verdict::BelowTarget;
            report.reasons.push(format!(
                "dihedral angle {:.1}° is below {:.0}°",
                report.min_dihedral_deg, thresholds.min_dihedral_deg
            ));
        }
        if report.max_dihedral_deg > thresholds.max_dihedral_deg {
            report.verdict = Verdict::BelowTarget;
            report.reasons.push(format!(
                "dihedral angle {:.1}° is above {:.0}°",
                report.max_dihedral_deg, thresholds.max_dihedral_deg
            ));
        }
        if report.max_aspect_ratio > thresholds.target_aspect_ratio {
            report.verdict = Verdict::BelowTarget;
            report.reasons.push(format!(
                "aspect ratio {:.1} is above the target {:.1}",
                report.max_aspect_ratio, thresholds.target_aspect_ratio
            ));
        }
    }
    report
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::mesh::Cell;

    fn cube() -> (Vec<Node>, Cell) {
        (
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            Cell::hex8([0, 1, 2, 3, 4, 5, 6, 7]),
        )
    }

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn a_unit_cube_has_unit_scaled_jacobian_everywhere() {
        let (p, c) = cube();
        for value in scaled_jacobian(&c, &p) {
            assert!(approx(value, 1.0, 1e-12), "got {value}");
        }
    }

    #[test]
    fn a_unit_cube_has_unit_volume() {
        let (p, c) = cube();
        assert!(approx(volume(&c, &p), 1.0, 1e-12));
    }

    #[test]
    fn a_box_scales_with_its_dimensions() {
        let (mut p, c) = cube();
        for point in &mut p {
            point[0] *= 2.0;
        }
        assert!(approx(volume(&c, &p), 2.0, 1e-12));
        assert!(approx(aspect_ratio(&c, &p), 2.0, 1e-12));
    }

    #[test]
    fn a_sheared_cube_is_one_over_sqrt_two() {
        // shear top face by +1 in x: the three edges at every corner are
        // orthogonal except the vertical one, which gains a component
        let (mut p, c) = cube();
        for point in p.iter_mut().skip(4) {
            point[0] += 1.0;
        }
        for value in scaled_jacobian(&c, &p) {
            assert!(approx(value, 1.0 / 2.0_f64.sqrt(), 1e-12), "got {value}");
        }
    }

    #[test]
    fn a_mirrored_cube_is_inverted() {
        let (mut p, c) = cube();
        for point in &mut p {
            point[0] = -point[0];
        }
        for value in scaled_jacobian(&c, &p) {
            assert!(value < 0.0, "got {value}");
        }
        assert!(volume(&c, &p) < 0.0);
    }

    #[test]
    fn a_cube_has_ninety_degree_dihedrals() {
        let (p, c) = cube();
        let angles = dihedral_angles(&c, &p);
        assert_eq!(angles.len(), 12);
        for value in angles {
            assert!(approx(value, 90.0, 1e-9), "got {value}");
        }
    }

    #[test]
    fn a_unit_square_has_unit_scaled_jacobian_and_area() {
        let p = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let c = Cell::quad4([0, 1, 2, 3]);
        for value in scaled_jacobian(&c, &p) {
            assert!(approx(value, 1.0, 1e-12));
        }
        assert!(approx(volume(&c, &p), 1.0, 1e-12));
        for angle in quad_angles(&c, &p) {
            assert!(approx(angle, 90.0, 1e-9));
        }
    }

    #[test]
    fn a_rhombus_has_the_sine_of_its_corner_angle() {
        // a unit rhombus sheared to 45 degrees at node 0
        let p = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0 + 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ];
        let c = Cell::quad4([0, 1, 2, 3]);
        let sj = scaled_jacobian(&c, &p);
        for value in sj {
            assert!(approx(value, 1.0 / 2.0_f64.sqrt(), 1e-12), "got {value}");
        }
    }

    #[test]
    fn a_degenerate_quad_is_flagged() {
        let p = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ];
        let c = Cell::quad4([0, 1, 2, 3]);
        assert!(volume(&c, &p).abs() < 1e-12);
        assert!(scaled_jacobian(&c, &p)
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0));
    }

    #[test]
    fn a_valid_cube_reports_acceptable() {
        let (p, c) = cube();
        let r = report(&Mesh::new(p, vec![c]));
        assert_eq!(r.verdict, Verdict::Acceptable, "{:?}", r.reasons);
        assert!(r.reasons.is_empty());
        assert_eq!(r.worst_element, Some(0));
    }

    #[test]
    fn an_inverted_cube_is_refused() {
        let (mut p, c) = cube();
        for point in &mut p {
            point[0] = -point[0];
        }
        let r = report(&Mesh::new(p, vec![c]));
        assert_eq!(r.verdict, Verdict::Refused);
        assert_eq!(r.inverted, 1);
        assert!(r.reasons[0].contains("inverted"));
    }

    #[test]
    fn a_quad_only_mesh_reports_no_dihedral_angles() {
        let p = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let c = Cell::quad4([0, 1, 2, 3]);
        let r = report(&Mesh::new(p, vec![c]));
        assert!(r.min_dihedral_deg.is_nan() && r.max_dihedral_deg.is_nan());
        assert!(r.summary().contains("n/a"));
        assert_eq!(r.verdict, Verdict::Acceptable);
    }

    #[test]
    fn an_empty_mesh_is_refused_with_a_reason() {
        let r = report(&Mesh::default());
        assert_eq!(r.verdict, Verdict::Refused);
        assert!(r.reasons[0].contains("no elements"));
    }

    #[test]
    fn a_below_target_cube_is_reported_but_not_refused() {
        let (mut p, c) = cube();
        // shear hard enough to drop the scaled Jacobian below 0.2
        for point in p.iter_mut().skip(4) {
            point[0] += 10.0;
        }
        let r = report(&Mesh::new(p, vec![c]));
        assert_eq!(r.verdict, Verdict::BelowTarget);
        assert!(r.min_scaled_jacobian < 0.2);
    }
}
