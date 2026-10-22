//! Element quality: the numbers the mesher is judged by.
//!
//! The definitions are the ones from Knupp's Jacobian framework (2000), which
//! the plan adopts as the contract:
//!
//! * **Scaled Jacobian** at a corner: `det J / (|c1| |c2| |c3|)`, where the
//!   columns of `J` are the images of the parametric axes. It is 1 for a cube,
//!   `sin θ` for a sheared cell, and negative for an inverted one.
//! * **Dihedral angle** at a hex edge: `180° - angle(n1, n2)` between the
//!   outward normals of the two faces meeting there; 90° for a cube.
//! * **Aspect ratio**: longest edge over shortest edge.
//!
//! Everything is computed from the trilinear (hex) or bilinear (quad) map, not
//! from an approximation, so the values match what a finite element code sees.

use crate::mesh::{Cell, Kind, Mesh, Node};

/// Local corner coordinates of a `Hex8`, matching [`Kind`]'s node order.
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
const QUAD_XI: [[f64; 2]; 4] = [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]];

/// Thresholds a mesh is judged against. The defaults are the P0 contract:
/// refuse anything with an inverted cell, target a scaled Jacobian of 0.2, and
/// accept down to 0.1 on the benchmark corpus.
#[derive(Clone, Copy, Debug)]
pub struct Thresholds {
    /// Hard floor for the scaled Jacobian; at or below this, refuse.
    pub hard_scaled_jacobian: f64,
    /// Target for every corner of every element.
    pub target_scaled_jacobian: f64,
    /// Dihedral angles outside this band are recorded as violations.
    pub min_dihedral_deg: f64,
    pub max_dihedral_deg: f64,
    /// Aspect ratio above this is reported; above `hard_aspect_ratio`, refused.
    pub target_aspect_ratio: f64,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Every element meets the target.
    Acceptable,
    /// Valid, but below the target; usable with a caveat.
    BelowTarget,
    /// Must not be shipped: an inverted element or a hard threshold was broken.
    Refused,
}

/// Per-mesh quality summary.
#[derive(Clone, Debug)]
pub struct Report {
    pub elements: usize,
    pub min_scaled_jacobian: f64,
    pub mean_scaled_jacobian: f64,
    pub max_scaled_jacobian: f64,
    /// Index of the element holding the minimum.
    pub worst_element: Option<usize>,
    pub inverted: usize,
    pub below_target: usize,
    pub min_dihedral_deg: f64,
    pub max_dihedral_deg: f64,
    pub max_aspect_ratio: f64,
    pub verdict: Verdict,
    pub reasons: Vec<String>,
}

impl Report {
    /// One-line summary for logs and the CLI.
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

fn sub(a: Node, b: Node) -> Node {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: Node, b: Node) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: Node) -> f64 {
    dot(a, a).sqrt()
}

fn det3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Jacobian `d x / d xi` of the trilinear hex at a local point.
fn hex_jacobian(points: &[Node], cell: &Cell, xi: [f64; 3]) -> [[f64; 3]; 3] {
    let mut j = [[0.0f64; 3]; 3];
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

/// Jacobian of the bilinear quad, as a 2x2 matrix in the plane.
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
/// Two Gauss points per direction integrate the degree-three Jacobian exactly,
/// so the result is the true measure of the trilinear/bilinear element.
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
pub fn aspect_ratio(cell: &Cell, points: &[Node]) -> f64 {
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

/// Dihedral angle at each of the 12 hex edges, in degrees, in edge order.
pub fn dihedral_angles(cell: &Cell, points: &[Node]) -> Vec<f64> {
    if cell.kind != Kind::Hex8 {
        return Vec::new();
    }
    let faces = cell.kind.faces();
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
pub fn report(mesh: &Mesh) -> Report {
    report_with(mesh, &Thresholds::default())
}

/// Judge a mesh against explicit thresholds.
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
        verdict: Verdict::Acceptable,
        reasons: Vec::new(),
    };
    if mesh.cells.is_empty() {
        report.verdict = Verdict::Refused;
        report.reasons.push("the mesh has no elements".to_string());
        return report;
    }

    let mut total = 0.0;
    let mut corners = 0usize;
    for (index, cell) in mesh.cells.iter().enumerate() {
        let sj = scaled_jacobian(cell, &mesh.points);
        let smallest = sj.iter().copied().fold(f64::INFINITY, f64::min);
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
        if smallest <= thresholds.hard_scaled_jacobian || smallest.is_nan() {
            report.inverted += 1;
        }
        if smallest < thresholds.target_scaled_jacobian {
            report.below_target += 1;
        }

        let aspect = aspect_ratio(cell, &mesh.points);
        report.max_aspect_ratio = report.max_aspect_ratio.max(aspect);

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

    if !report.min_dihedral_deg.is_finite() {
        // no hexahedra: dihedral angles do not apply
        report.min_dihedral_deg = f64::NAN;
        report.max_dihedral_deg = f64::NAN;
    }
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
