//! Size- and angle-driven refinement of a constrained Delaunay triangulation.
//!
//! Small input angles can prevent guaranteed termination. Every run therefore
//! has an explicit operation budget and reports exhaustion instead of returning
//! a mesh as though it had converged.

use std::fmt;

use super::domain::Domain;
use super::geometry::{same_point, Point};
use super::predicates::encroaches;
use super::triangulation::{Triangulation, TriangulationError};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QualityRequest {
    pub min_angle_deg: f64,
    pub max_operations: usize,
}

impl Default for QualityRequest {
    fn default() -> Self {
        Self {
            min_angle_deg: 25.0,
            max_operations: 100_000,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct QualityStats {
    pub points: usize,
    pub triangles: usize,
    pub min_angle_deg: f64,
    pub max_angle_deg: f64,
    pub min_radius_ratio: f64,
    pub max_area: f64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RefinementReport {
    pub operations: usize,
    pub inserted_circumcentres: usize,
    pub split_constraints: usize,
    pub split_unconstrained_edges: usize,
    pub rejected_circumcentres: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RefinedMesh {
    triangulation: Triangulation,
    quality: QualityStats,
    report: RefinementReport,
}

impl RefinedMesh {
    pub fn triangulation(&self) -> &Triangulation {
        &self.triangulation
    }

    pub fn quality(&self) -> QualityStats {
        self.quality
    }

    pub fn report(&self) -> RefinementReport {
        self.report
    }

    pub fn into_triangulation(self) -> Triangulation {
        self.triangulation
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum RefinementError {
    InvalidAngle(f64),
    ZeroOperationLimit,
    InvalidSize { point: Point, value: f64 },
    OperationLimit { operations: usize },
    Stalled([usize; 2]),
    NonFiniteGeometry,
    Triangulation(TriangulationError),
}

impl fmt::Display for RefinementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAngle(angle) => write!(
                f,
                "minimum angle {angle} is invalid; choose a finite value from 0 to 30 degrees"
            ),
            Self::ZeroOperationLimit => {
                write!(f, "the refinement operation limit must be positive")
            }
            Self::InvalidSize { point, value } => write!(
                f,
                "size field returned {value} at {point:?}; sizes must be finite and positive"
            ),
            Self::OperationLimit { operations } => write!(
                f,
                "refinement did not converge within {operations} operations"
            ),
            Self::Stalled(edge) => write!(
                f,
                "refinement stalled because edge {edge:?} has no distinct binary64 midpoint"
            ),
            Self::NonFiniteGeometry => write!(f, "refinement produced non-finite quality geometry"),
            Self::Triangulation(error) => write!(f, "triangulation failed: {error}"),
        }
    }
}

impl std::error::Error for RefinementError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Triangulation(error) => Some(error),
            _ => None,
        }
    }
}

impl From<TriangulationError> for RefinementError {
    fn from(error: TriangulationError) -> Self {
        Self::Triangulation(error)
    }
}

/// Refine a domain until every longest triangle edge is at most `h(centroid)`
/// and every angle meets the requested minimum. An angle of zero disables the
/// angle criterion. The practical target is capped at 30 degrees.
pub fn triangulate<F: Fn(Point) -> f64>(
    domain: &Domain,
    size_field: F,
    request: QualityRequest,
) -> Result<RefinedMesh, RefinementError> {
    if !request.min_angle_deg.is_finite()
        || request.min_angle_deg < 0.0
        || request.min_angle_deg > 30.0
    {
        return Err(RefinementError::InvalidAngle(request.min_angle_deg));
    }
    if request.max_operations == 0 {
        return Err(RefinementError::ZeroOperationLimit);
    }
    let mut mesh = Triangulation::from_domain(domain)?;
    let mut report = RefinementReport::default();
    loop {
        if let Some(edge) = encroached_constraint(&mesh) {
            begin_operation(&report, request)?;
            split_edge(&mut mesh, edge, &mut report)?;
            continue;
        }
        let Some((triangle, longest_edge)) = bad_triangle(&mesh, &size_field, request)? else {
            let quality = QualityStats::new(&mesh)?;
            return Ok(RefinedMesh {
                triangulation: mesh,
                quality,
                report,
            });
        };
        begin_operation(&report, request)?;

        let vertices = mesh.triangles()[triangle].vertices;
        let center = circumcenter(&mesh, vertices);
        if let Some(center) = center {
            let encroached = mesh
                .constraints()
                .map(|constraint| constraint.vertices)
                .find(|edge| {
                    let [a, b] = *edge;
                    encroaches(mesh.points()[a], mesh.points()[b], center)
                });
            if let Some(edge) = encroached {
                report.rejected_circumcentres += 1;
                split_edge(&mut mesh, edge, &mut report)?;
                continue;
            }
            if !mesh.points().iter().any(|point| same_point(*point, center)) {
                match mesh.insert_point(center) {
                    Ok(_) => {
                        report.operations += 1;
                        report.inserted_circumcentres += 1;
                        continue;
                    }
                    Err(_) => report.rejected_circumcentres += 1,
                }
            } else {
                report.rejected_circumcentres += 1;
            }
        } else {
            report.rejected_circumcentres += 1;
        }
        split_edge(&mut mesh, longest_edge, &mut report)?;
    }
}

fn begin_operation(
    report: &RefinementReport,
    request: QualityRequest,
) -> Result<(), RefinementError> {
    if report.operations >= request.max_operations {
        return Err(RefinementError::OperationLimit {
            operations: report.operations,
        });
    }
    Ok(())
}

fn split_edge(
    mesh: &mut Triangulation,
    edge: [usize; 2],
    report: &mut RefinementReport,
) -> Result<(), RefinementError> {
    let constrained = mesh.constraint(edge).is_some();
    match mesh.split_edge(edge) {
        Ok(_) => {
            report.operations += 1;
            if constrained {
                report.split_constraints += 1;
            } else {
                report.split_unconstrained_edges += 1;
            }
            Ok(())
        }
        Err(TriangulationError::DegenerateInsertion | TriangulationError::DuplicatePoint(_)) => {
            Err(RefinementError::Stalled(edge))
        }
        Err(error) => Err(RefinementError::Triangulation(error)),
    }
}

fn encroached_constraint(mesh: &Triangulation) -> Option<[usize; 2]> {
    for constraint in mesh.constraints() {
        let [a, b] = constraint.vertices;
        if mesh.points().iter().enumerate().any(|(index, point)| {
            index != a && index != b && encroaches(mesh.points()[a], mesh.points()[b], *point)
        }) {
            return Some(constraint.vertices);
        }
    }
    None
}

fn bad_triangle<F: Fn(Point) -> f64>(
    mesh: &Triangulation,
    size_field: &F,
    request: QualityRequest,
) -> Result<Option<(usize, [usize; 2])>, RefinementError> {
    for (index, triangle) in mesh.triangles().iter().enumerate() {
        let [a, b, c] = triangle.vertices;
        let points = [mesh.points()[a], mesh.points()[b], mesh.points()[c]];
        let center = centroid(points);
        let size = size_field(center);
        if !size.is_finite() || size <= 0.0 {
            return Err(RefinementError::InvalidSize {
                point: center,
                value: size,
            });
        }
        let longest_edge = longest_edge(mesh, triangle.vertices);
        let too_large = edge_length(mesh, longest_edge) > size;
        let too_skinny =
            request.min_angle_deg > 0.0 && minimum_angle(points) < request.min_angle_deg;
        if too_large || too_skinny {
            return Ok(Some((index, longest_edge)));
        }
    }
    Ok(None)
}

fn longest_edge(mesh: &Triangulation, vertices: [usize; 3]) -> [usize; 2] {
    let mut edges = [
        [vertices[0], vertices[1]],
        [vertices[1], vertices[2]],
        [vertices[2], vertices[0]],
    ];
    edges.sort_by(|left, right| {
        edge_length(mesh, *right)
            .total_cmp(&edge_length(mesh, *left))
            .then_with(|| left.cmp(right))
    });
    edges[0]
}

fn edge_length(mesh: &Triangulation, edge: [usize; 2]) -> f64 {
    let a = mesh.points()[edge[0]];
    let b = mesh.points()[edge[1]];
    (b[0] - a[0]).hypot(b[1] - a[1])
}

fn centroid(points: [Point; 3]) -> Point {
    let [a, b, c] = points;
    [
        a[0] + (b[0] - a[0]) / 3.0 + (c[0] - a[0]) / 3.0,
        a[1] + (b[1] - a[1]) / 3.0 + (c[1] - a[1]) / 3.0,
    ]
}

fn minimum_angle(points: [Point; 3]) -> f64 {
    let mut minimum = f64::INFINITY;
    for index in 0..3 {
        let angle = angle_at(
            points[index],
            points[(index + 1) % 3],
            points[(index + 2) % 3],
        );
        minimum = minimum.min(angle);
    }
    minimum
}

fn angle_at(origin: Point, first: Point, second: Point) -> f64 {
    let u = [first[0] - origin[0], first[1] - origin[1]];
    let v = [second[0] - origin[0], second[1] - origin[1]];
    let scale = u[0].abs().max(u[1].abs()).max(v[0].abs()).max(v[1].abs());
    if scale == 0.0 || !scale.is_finite() {
        return 0.0;
    }
    let u = [u[0] / scale, u[1] / scale];
    let v = [v[0] / scale, v[1] / scale];
    let cross = (u[0] * v[1] - u[1] * v[0]).abs();
    let dot = u[0] * v[0] + u[1] * v[1];
    cross.atan2(dot).to_degrees()
}

fn circumcenter(mesh: &Triangulation, vertices: [usize; 3]) -> Option<Point> {
    let a = mesh.points()[vertices[0]];
    let b = mesh.points()[vertices[1]];
    let c = mesh.points()[vertices[2]];
    let u = [b[0] - a[0], b[1] - a[1]];
    let v = [c[0] - a[0], c[1] - a[1]];
    let scale = u[0].abs().max(u[1].abs()).max(v[0].abs()).max(v[1].abs());
    if scale == 0.0 || !scale.is_finite() {
        return None;
    }
    let u = [u[0] / scale, u[1] / scale];
    let v = [v[0] / scale, v[1] / scale];
    let denominator = 2.0 * (u[0] * v[1] - u[1] * v[0]);
    if denominator == 0.0 || !denominator.is_finite() {
        return None;
    }
    let uu = u[0] * u[0] + u[1] * u[1];
    let vv = v[0] * v[0] + v[1] * v[1];
    let local = [
        (uu * v[1] - vv * u[1]) / denominator,
        (u[0] * vv - v[0] * uu) / denominator,
    ];
    let center = [a[0] + scale * local[0], a[1] + scale * local[1]];
    if center[0].is_finite() && center[1].is_finite() {
        Some(center)
    } else {
        None
    }
}

impl QualityStats {
    fn new(mesh: &Triangulation) -> Result<Self, RefinementError> {
        let mut result = Self {
            points: mesh.points().len(),
            triangles: mesh.triangles().len(),
            min_angle_deg: f64::INFINITY,
            max_angle_deg: f64::NEG_INFINITY,
            min_radius_ratio: f64::INFINITY,
            max_area: 0.0,
        };
        for triangle in mesh.triangles() {
            let [a, b, c] = triangle.vertices;
            let points = [mesh.points()[a], mesh.points()[b], mesh.points()[c]];
            for index in 0..3 {
                let angle = angle_at(
                    points[index],
                    points[(index + 1) % 3],
                    points[(index + 2) % 3],
                );
                result.min_angle_deg = result.min_angle_deg.min(angle);
                result.max_angle_deg = result.max_angle_deg.max(angle);
            }
            result.min_radius_ratio = result.min_radius_ratio.min(radius_ratio(points));
            let ab = [points[1][0] - points[0][0], points[1][1] - points[0][1]];
            let ac = [points[2][0] - points[0][0], points[2][1] - points[0][1]];
            let area = 0.5 * (ab[0] * ac[1] - ab[1] * ac[0]).abs();
            result.max_area = result.max_area.max(area);
        }
        if !result.min_angle_deg.is_finite()
            || !result.max_angle_deg.is_finite()
            || !result.min_radius_ratio.is_finite()
            || !result.max_area.is_finite()
        {
            return Err(RefinementError::NonFiniteGeometry);
        }
        Ok(result)
    }
}

fn radius_ratio(points: [Point; 3]) -> f64 {
    let mut lengths = [
        (points[1][0] - points[0][0]).hypot(points[1][1] - points[0][1]),
        (points[2][0] - points[1][0]).hypot(points[2][1] - points[1][1]),
        (points[0][0] - points[2][0]).hypot(points[0][1] - points[2][1]),
    ];
    let scale = lengths[0].max(lengths[1]).max(lengths[2]);
    if scale == 0.0 || !scale.is_finite() {
        return 0.0;
    }
    for length in &mut lengths {
        *length /= scale;
    }
    let ab = [points[1][0] - points[0][0], points[1][1] - points[0][1]];
    let ac = [points[2][0] - points[0][0], points[2][1] - points[0][1]];
    let cross = (ab[0] / scale) * (ac[1] / scale) - (ab[1] / scale) * (ac[0] / scale);
    let area = 0.5 * cross.abs();
    let perimeter = lengths[0] + lengths[1] + lengths[2];
    16.0 * area * area / (perimeter * lengths[0] * lengths[1] * lengths[2])
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use super::*;
    use crate::delaunay::{ConstraintKind, Feature, Loop, Region};

    fn request(min_angle_deg: f64) -> QualityRequest {
        QualityRequest {
            min_angle_deg,
            max_operations: 20_000,
        }
    }

    fn square() -> Domain {
        Domain::single(vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]).expect("square")
    }

    fn assert_sizes<F: Fn(Point) -> f64>(mesh: &Triangulation, size_field: F) {
        for triangle in mesh.triangles() {
            let [a, b, c] = triangle.vertices;
            let points = [mesh.points()[a], mesh.points()[b], mesh.points()[c]];
            let size = size_field(centroid(points));
            assert!(edge_length(mesh, longest_edge(mesh, triangle.vertices)) <= size);
        }
    }

    #[test]
    fn uniform_sizing_produces_a_valid_compliant_mesh() {
        let mesh = triangulate(&square(), |_| 0.5, request(0.0)).expect("uniform refinement");
        mesh.triangulation().validate().expect("valid mesh");
        assert!((mesh.triangulation().area() - 4.0).abs() < 1.0e-12);
        assert_sizes(mesh.triangulation(), |_| 0.5);
        assert!(mesh.quality().points > 4);
        assert_eq!(
            mesh.report().operations,
            mesh.report().inserted_circumcentres
                + mesh.report().split_constraints
                + mesh.report().split_unconstrained_edges
        );
    }

    #[test]
    fn sizing_only_mode_does_not_impose_an_angle_target() {
        let mesh = triangulate(&square(), |_| 0.75, request(0.0)).expect("sizing only");
        assert_sizes(mesh.triangulation(), |_| 0.75);
        assert!(mesh.quality().min_angle_deg > 0.0);
    }

    #[test]
    fn quality_refinement_reaches_twenty_degrees_on_a_skinny_domain() {
        let domain = Domain::single(vec![[0.0, 0.0], [4.0, 0.0], [4.0, 0.5], [0.0, 0.5]])
            .expect("skinny rectangle");
        let mesh = triangulate(&domain, |_| 10.0, request(20.0)).expect("quality refinement");
        assert!(mesh.quality().min_angle_deg >= 20.0 - 1.0e-10);
        mesh.triangulation().validate().expect("valid mesh");
    }

    #[test]
    fn spatially_varying_sizes_are_checked_at_each_centroid() {
        let domain = Domain::single(vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
            .expect("unit square");
        let size = |point: Point| 0.2 + 0.8 * point[0];
        let mesh = triangulate(&domain, size, request(0.0)).expect("graded refinement");
        assert_sizes(mesh.triangulation(), size);
        mesh.triangulation().validate().expect("valid mesh");
    }

    #[test]
    fn holes_and_disconnected_regions_remain_separate_and_tagged() {
        let domain = Domain::new(
            vec![
                Region::new(
                    4,
                    Loop::new(vec![[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]], 40),
                ),
                Region::new(
                    8,
                    Loop::new(vec![[7.0, 0.0], [9.0, 0.0], [9.0, 2.0], [7.0, 2.0]], 80),
                ),
            ],
            vec![Loop::new(
                vec![[2.0, 2.0], [3.0, 2.0], [3.0, 3.0], [2.0, 3.0]],
                41,
            )],
            Vec::new(),
            Vec::new(),
        )
        .expect("holed disjoint domain");
        let mesh = triangulate(&domain, |_| 1.0, request(20.0)).expect("refinement");
        assert!((mesh.triangulation().area() - domain.area()).abs() < 1.0e-10);
        assert!(mesh.triangulation().constraints().any(|constraint| {
            constraint.kind == ConstraintKind::HoleBoundary && constraint.marker == 41
        }));
        assert!(mesh
            .triangulation()
            .triangles()
            .iter()
            .any(|triangle| triangle.material == 4));
        assert!(mesh
            .triangulation()
            .triangles()
            .iter()
            .any(|triangle| triangle.material == 8));
        assert_sizes(mesh.triangulation(), |_| 1.0);
        mesh.triangulation().validate().expect("valid mesh");
    }

    #[test]
    fn boundary_material_and_feature_lineage_survives_refinement() {
        let domain = Domain::new(
            vec![Region::new(
                1,
                Loop::new(vec![[0.0, 0.0], [6.0, 0.0], [6.0, 6.0], [0.0, 6.0]], 10),
            )],
            Vec::new(),
            vec![Loop::new(
                vec![[2.0, 2.0], [4.0, 2.0], [4.0, 4.0], [2.0, 4.0]],
                20,
            )],
            vec![Feature::new(vec![[0.5, 1.0], [1.5, 1.0]], 30)],
        )
        .expect("tagged domain");
        let mesh = triangulate(&domain, |_| 1.5, request(20.0)).expect("tagged refinement");
        for (kind, marker) in [
            (ConstraintKind::OuterBoundary, 10),
            (ConstraintKind::MaterialInterface, 20),
            (ConstraintKind::Feature, 30),
        ] {
            assert!(mesh
                .triangulation()
                .constraints()
                .filter(|constraint| constraint.kind == kind)
                .all(|constraint| constraint.marker == marker));
            assert!(mesh
                .triangulation()
                .constraints()
                .any(|constraint| constraint.kind == kind && constraint.marker == marker));
        }
        assert!(mesh
            .triangulation()
            .triangles()
            .iter()
            .any(|triangle| triangle.material == 1));
        assert!(mesh
            .triangulation()
            .triangles()
            .iter()
            .any(|triangle| triangle.material == 20));
        mesh.triangulation().validate().expect("valid mesh");
    }

    #[test]
    fn refinement_is_stable_under_translation_and_scaling() {
        for scale in [1.0e-20, 1.0e20] {
            let offset = 10.0 * scale;
            let ring = vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]
                .into_iter()
                .map(|point| [offset + scale * point[0], offset + scale * point[1]])
                .collect();
            let domain = Domain::single(ring).expect("scaled domain");
            let mesh =
                triangulate(&domain, |_| 0.5 * scale, request(20.0)).expect("scaled refinement");
            assert_sizes(mesh.triangulation(), |_| 0.5 * scale);
            assert!(mesh.quality().min_angle_deg >= 20.0 - 1.0e-10);
            mesh.triangulation().validate().expect("valid mesh");
        }
    }

    #[test]
    fn invalid_sizes_are_rejected_at_the_sample_point() {
        for value in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
            let error = triangulate(&square(), |_| value, request(0.0))
                .expect_err("invalid size must fail");
            assert!(matches!(
                error,
                RefinementError::InvalidSize { value: actual, .. } if actual.to_bits() == value.to_bits()
            ));
        }
    }

    #[test]
    fn invalid_quality_requests_are_rejected_before_sampling() {
        for angle in [-0.1, 30.1, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                triangulate(&square(), |_| 1.0, request(angle)),
                Err(RefinementError::InvalidAngle(actual)) if actual.to_bits() == angle.to_bits()
            ));
        }
        assert_eq!(
            triangulate(
                &square(),
                |_| 1.0,
                QualityRequest {
                    min_angle_deg: 20.0,
                    max_operations: 0,
                }
            ),
            Err(RefinementError::ZeroOperationLimit)
        );
    }

    #[test]
    fn operation_exhaustion_is_not_reported_as_convergence() {
        let error = triangulate(
            &square(),
            |_| 0.05,
            QualityRequest {
                min_angle_deg: 20.0,
                max_operations: 1,
            },
        )
        .expect_err("small operation budget");
        assert_eq!(error, RefinementError::OperationLimit { operations: 1 });
    }

    #[test]
    fn an_unrepresentable_midpoint_is_an_explicit_stall() {
        let base = 1.0e-140_f64;
        let next = f64::from_bits(base.to_bits() + 1);
        let mut mesh = Triangulation::new(
            vec![[base, base], [next, base], [base, next]],
            vec![crate::delaunay::Triangle::new([0, 1, 2], 0)],
            vec![
                crate::delaunay::Constraint::new([0, 1], 0, ConstraintKind::OuterBoundary, 0),
                crate::delaunay::Constraint::new([1, 2], 0, ConstraintKind::OuterBoundary, 0),
                crate::delaunay::Constraint::new([0, 2], 0, ConstraintKind::OuterBoundary, 0),
            ],
        )
        .expect("tiny valid triangulation");
        let error = split_edge(&mut mesh, [1, 2], &mut RefinementReport::default())
            .expect_err("unrepresentable midpoint");
        assert_eq!(error, RefinementError::Stalled([1, 2]));
    }

    #[test]
    fn quality_statistics_and_output_are_deterministic() {
        let unrefined = triangulate(&square(), |_| 10.0, request(25.0)).expect("statistics");
        assert_eq!(unrefined.report().operations, 0);
        assert!((unrefined.quality().min_angle_deg - 45.0).abs() < 1.0e-12);
        assert!((unrefined.quality().max_angle_deg - 90.0).abs() < 1.0e-12);
        assert!(unrefined.quality().min_radius_ratio > 0.82);
        assert!(unrefined.quality().min_radius_ratio < 0.83);
        assert_eq!(unrefined.quality().max_area, 2.0);

        let first = triangulate(&square(), |_| 0.75, request(20.0)).expect("first run");
        let second = triangulate(&square(), |_| 0.75, request(20.0)).expect("second run");
        assert_eq!(first, second);
        first.into_triangulation().validate().expect("owned mesh");
    }
}
