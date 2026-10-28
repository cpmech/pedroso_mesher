//! Mutable constrained triangulation with deterministic Lawson legalisation.
//!
//! Adjacency is derived from triangle edges. Every public mutation is
//! transactional and rechecks topology, geometry, constraints and area before
//! returning successfully.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use super::domain::{Domain, PointLocation};
use super::geometry::{point_on_segment, same_point, strictly_between, Point};
use super::predicates::{incircle, orient2d};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Triangle {
    pub vertices: [usize; 3],
    pub material: i32,
}

impl Triangle {
    pub fn new(vertices: [usize; 3], material: i32) -> Self {
        Self { vertices, material }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConstraintKind {
    OuterBoundary,
    HoleBoundary,
    MaterialInterface,
    Feature,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Constraint {
    pub vertices: [usize; 2],
    pub marker: i32,
    pub kind: ConstraintKind,
    /// Index of the input loop or feature from which this edge originated.
    pub source: usize,
}

impl Constraint {
    pub fn new(vertices: [usize; 2], marker: i32, kind: ConstraintKind, source: usize) -> Self {
        Self {
            vertices: edge_key(vertices[0], vertices[1]),
            marker,
            kind,
            source,
        }
    }

    fn split(self, first: usize, middle: usize, second: usize) -> [Self; 2] {
        [
            Self::new([first, middle], self.marker, self.kind, self.source),
            Self::new([middle, second], self.marker, self.kind, self.source),
        ]
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Triangulation {
    points: Vec<Point>,
    triangles: Vec<Triangle>,
    constraints: BTreeMap<[usize; 2], Constraint>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TriangulationError {
    Empty,
    NonFinitePoint(usize),
    DuplicatePoint(usize),
    PointOutside,
    InvalidVertex(usize),
    MissingEdge([usize; 2]),
    MissingConstraint([usize; 2]),
    ConstraintConflict([usize; 2]),
    DegenerateInsertion,
    LegalisationLimit,
    RecoveryLimit,
    RecoveryFailed([usize; 2]),
    InvalidMesh(String),
}

impl fmt::Display for TriangulationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "a triangulation needs points and triangles"),
            Self::NonFinitePoint(point) => write!(f, "point {point} is not finite"),
            Self::DuplicatePoint(point) => {
                write!(f, "the insertion point coincides with point {point}")
            }
            Self::PointOutside => write!(f, "the insertion point is outside the triangulation"),
            Self::InvalidVertex(point) => write!(f, "point index {point} is out of range"),
            Self::MissingEdge(edge) => write!(f, "edge {edge:?} is absent from the triangulation"),
            Self::MissingConstraint(edge) => write!(f, "edge {edge:?} is not constrained"),
            Self::ConstraintConflict(edge) => {
                write!(f, "constraint {edge:?} conflicts with an existing constraint")
            }
            Self::DegenerateInsertion => {
                write!(f, "the insertion would create a zero-area triangle")
            }
            Self::LegalisationLimit => write!(f, "Delaunay legalisation exceeded its flip budget"),
            Self::RecoveryLimit => write!(f, "constraint recovery exceeded its flip budget"),
            Self::RecoveryFailed(edge) => write!(
                f,
                "constraint {edge:?} cannot be recovered without crossing, overlapping or inverting geometry"
            ),
            Self::InvalidMesh(message) => write!(f, "invalid triangulation: {message}"),
        }
    }
}

impl std::error::Error for TriangulationError {}

impl Triangulation {
    /// Construct and validate a mesh. Boundary edges must be constrained;
    /// internal material and feature constraints may have two incident cells.
    pub fn new(
        points: Vec<Point>,
        triangles: Vec<Triangle>,
        constraints: Vec<Constraint>,
    ) -> Result<Self, TriangulationError> {
        if points.is_empty() || triangles.is_empty() {
            return Err(TriangulationError::Empty);
        }
        let mut constraint_map = BTreeMap::new();
        for constraint in constraints {
            let edge = constraint.vertices;
            if constraint_map.insert(edge, constraint).is_some() {
                return Err(TriangulationError::InvalidMesh(format!(
                    "constraint {edge:?} occurs more than once"
                )));
            }
        }
        let result = Self {
            points,
            triangles,
            constraints: constraint_map,
        };
        result.validate()?;
        Ok(result)
    }

    /// Triangulate a validated domain without adding Steiner points. Every
    /// input vertex and constraint is preserved and every output triangle is
    /// counter-clockwise and locally Delaunay away from constraints.
    pub fn from_domain(domain: &Domain) -> Result<Self, TriangulationError> {
        let mut input_points = Vec::new();
        for region in domain.regions() {
            input_points.extend_from_slice(region.boundary().points());
            for hole in region.holes() {
                input_points.extend_from_slice(hole.points());
            }
        }
        for material in domain.materials() {
            input_points.extend_from_slice(material.points());
        }
        for feature in domain.features() {
            input_points.extend_from_slice(feature.path());
        }
        let (minimum, maximum) = point_bounds(&input_points)?;
        let span = (maximum[0] - minimum[0]).max(maximum[1] - minimum[1]);
        if !span.is_finite() || span <= 0.0 {
            return Err(TriangulationError::InvalidMesh(
                "cannot construct a finite bounding triangle".to_string(),
            ));
        }
        let center = [
            minimum[0] + 0.5 * (maximum[0] - minimum[0]),
            minimum[1] + 0.5 * (maximum[1] - minimum[1]),
        ];
        let artificial = vec![
            [center[0] - 16.0 * span, center[1] - 16.0 * span],
            [center[0] + 16.0 * span, center[1] - 16.0 * span],
            [center[0], center[1] + 16.0 * span],
        ];
        let temporary = vec![
            Constraint::new([0, 1], 0, ConstraintKind::OuterBoundary, usize::MAX),
            Constraint::new([1, 2], 0, ConstraintKind::OuterBoundary, usize::MAX),
            Constraint::new([0, 2], 0, ConstraintKind::OuterBoundary, usize::MAX),
        ];
        let mut mesh = Self::new(artificial, vec![Triangle::new([0, 1, 2], 0)], temporary)?;
        let mut ids = BTreeMap::new();
        for point in input_points {
            let vertex = mesh.insert_point(point)?;
            ids.insert(coordinate_key(point), vertex);
        }

        let mut expected_constraints = 0usize;
        for (region_index, region) in domain.regions().iter().enumerate() {
            mesh.recover_loop(
                &ids,
                region.boundary().points(),
                region.boundary().marker(),
                ConstraintKind::OuterBoundary,
                region_index,
            )?;
            expected_constraints += region.boundary().points().len();
        }
        let hole_offset = domain.regions().len();
        for (hole_index, hole) in domain.holes().enumerate() {
            mesh.recover_loop(
                &ids,
                hole.points(),
                hole.marker(),
                ConstraintKind::HoleBoundary,
                hole_offset + hole_index,
            )?;
            expected_constraints += hole.points().len();
        }
        let material_offset = hole_offset + domain.holes().count();
        for (material_index, material) in domain.materials().iter().enumerate() {
            mesh.recover_loop(
                &ids,
                material.points(),
                material.marker(),
                ConstraintKind::MaterialInterface,
                material_offset + material_index,
            )?;
            expected_constraints += material.points().len();
        }
        for (feature_index, feature) in domain.features().iter().enumerate() {
            for pair in feature.path().windows(2) {
                if pair.len() != 2 {
                    return Err(TriangulationError::InvalidMesh(
                        "a validated feature has an incomplete edge".to_string(),
                    ));
                }
                let a = lookup_vertex(&ids, pair[0])?;
                let b = lookup_vertex(&ids, pair[1])?;
                mesh.recover_constraint(Constraint::new(
                    [a, b],
                    feature.marker(),
                    ConstraintKind::Feature,
                    feature_index,
                ))?;
                expected_constraints += 1;
            }
        }

        mesh.retain_domain(domain)?;
        if mesh.constraints.len() != expected_constraints {
            return Err(TriangulationError::InvalidMesh(format!(
                "expected {expected_constraints} constraints, found {}",
                mesh.constraints.len()
            )));
        }
        mesh.legalize_inner()?;
        mesh.validate()?;
        let area_error = (mesh.area() - domain.area()).abs();
        let tolerance = 1.0e-8 * domain.area().abs().max(f64::MIN_POSITIVE);
        if !area_error.is_finite() || area_error > tolerance {
            return Err(TriangulationError::InvalidMesh(format!(
                "mesh area {} does not match domain area {}",
                mesh.area(),
                domain.area()
            )));
        }
        Ok(mesh)
    }

    fn recover_loop(
        &mut self,
        ids: &BTreeMap<(u64, u64), usize>,
        points: &[Point],
        marker: i32,
        kind: ConstraintKind,
        source: usize,
    ) -> Result<(), TriangulationError> {
        for index in 0..points.len() {
            let a = lookup_vertex(ids, points[index])?;
            let b = lookup_vertex(ids, points[(index + 1) % points.len()])?;
            self.recover_constraint(Constraint::new([a, b], marker, kind, source))?;
        }
        Ok(())
    }

    fn retain_domain(&mut self, domain: &Domain) -> Result<(), TriangulationError> {
        let points = &self.points;
        self.triangles.retain_mut(|triangle| {
            let [a, b, c] = triangle.vertices;
            let pa = points[a];
            let pb = points[b];
            let pc = points[c];
            let centroid = [
                pa[0] + (pb[0] - pa[0]) / 3.0 + (pc[0] - pa[0]) / 3.0,
                pa[1] + (pb[1] - pa[1]) / 3.0 + (pc[1] - pa[1]) / 3.0,
            ];
            match domain.locate(centroid) {
                PointLocation::Inside { material, .. } => {
                    triangle.material = material;
                    true
                }
                PointLocation::Boundary | PointLocation::Outside => false,
            }
        });
        if self.triangles.is_empty() {
            return Err(TriangulationError::Empty);
        }
        let used: BTreeSet<usize> = self
            .triangles
            .iter()
            .flat_map(|triangle| triangle.vertices)
            .collect();
        let edges = self.edge_uses()?;
        self.constraints
            .retain(|edge, constraint| edges.contains_key(edge) && constraint.source != usize::MAX);

        let mut old_to_new = vec![usize::MAX; self.points.len()];
        let mut compact_points = Vec::with_capacity(used.len());
        for (old, point) in self.points.iter().enumerate() {
            if used.contains(&old) {
                old_to_new[old] = compact_points.len();
                compact_points.push(*point);
            }
        }
        for triangle in &mut self.triangles {
            for vertex in &mut triangle.vertices {
                *vertex = old_to_new[*vertex];
                if *vertex == usize::MAX {
                    return Err(TriangulationError::InvalidMesh(
                        "compaction retained an unused triangle vertex".to_string(),
                    ));
                }
            }
        }
        let mut compact_constraints = BTreeMap::new();
        for (_, mut constraint) in std::mem::take(&mut self.constraints) {
            let first = *old_to_new.get(constraint.vertices[0]).ok_or_else(|| {
                TriangulationError::InvalidMesh("constraint endpoint is out of range".to_string())
            })?;
            let second = *old_to_new.get(constraint.vertices[1]).ok_or_else(|| {
                TriangulationError::InvalidMesh("constraint endpoint is out of range".to_string())
            })?;
            if first == usize::MAX || second == usize::MAX {
                return Err(TriangulationError::InvalidMesh(
                    "compaction retained an unused constraint endpoint".to_string(),
                ));
            }
            constraint.vertices = edge_key(first, second);
            if compact_constraints
                .insert(constraint.vertices, constraint)
                .is_some()
            {
                return Err(TriangulationError::InvalidMesh(
                    "compaction duplicated a constraint".to_string(),
                ));
            }
        }
        self.constraints = compact_constraints;
        self.points = compact_points;
        Ok(())
    }

    pub fn points(&self) -> &[Point] {
        &self.points
    }

    pub fn triangles(&self) -> &[Triangle] {
        &self.triangles
    }

    pub fn constraints(&self) -> impl Iterator<Item = &Constraint> {
        self.constraints.values()
    }

    pub fn constraint(&self, edge: [usize; 2]) -> Option<&Constraint> {
        self.constraints.get(&edge_key(edge[0], edge[1]))
    }

    pub fn area(&self) -> f64 {
        let mut total = 0.0;
        let mut correction = 0.0;
        for triangle in &self.triangles {
            let term = triangle_area(&self.points, triangle.vertices);
            let adjusted = term - correction;
            let next = total + adjusted;
            correction = (next - total) - adjusted;
            total = next;
        }
        total
    }

    /// Insert a new point strictly inside a triangle or in the relative
    /// interior of an existing edge, then restore the local Delaunay property.
    pub fn insert_point(&mut self, point: Point) -> Result<usize, TriangulationError> {
        if !point[0].is_finite() || !point[1].is_finite() {
            return Err(TriangulationError::NonFinitePoint(self.points.len()));
        }
        if let Some(index) = self
            .points
            .iter()
            .position(|&existing| same_point(existing, point))
        {
            return Err(TriangulationError::DuplicatePoint(index));
        }
        let location = self.locate(point)?;
        let original = self.clone();
        let old_area = self.area();
        let vertex = self.points.len();
        self.points.push(point);
        let mutation = match location {
            Location::Interior(triangle) => self.insert_in_triangle(triangle, vertex),
            Location::Edge(edge) => self.insert_on_edge(edge, vertex),
        };
        if let Err(error) = mutation {
            *self = original;
            return Err(error);
        }
        if let Err(error) = self
            .legalize_inner()
            .and_then(|()| self.finish_mutation(old_area))
        {
            *self = original;
            return Err(error);
        }
        Ok(vertex)
    }

    /// Split a constrained edge at its binary64 midpoint. Both descendants
    /// retain the original marker, kind and source index.
    pub fn split_constraint(&mut self, edge: [usize; 2]) -> Result<usize, TriangulationError> {
        let key = edge_key(edge[0], edge[1]);
        if !self.constraints.contains_key(&key) {
            return Err(TriangulationError::MissingConstraint(key));
        }
        self.split_edge(key)
    }

    /// Split any existing edge at its binary64 midpoint.
    pub fn split_edge(&mut self, edge: [usize; 2]) -> Result<usize, TriangulationError> {
        for vertex in edge {
            if vertex >= self.points.len() {
                return Err(TriangulationError::InvalidVertex(vertex));
            }
        }
        let key = edge_key(edge[0], edge[1]);
        if !self.edge_uses()?.contains_key(&key) {
            return Err(TriangulationError::MissingEdge(key));
        }
        let a = self.points[key[0]];
        let b = self.points[key[1]];
        let midpoint = [a[0] + 0.5 * (b[0] - a[0]), a[1] + 0.5 * (b[1] - a[1])];
        if same_point(a, midpoint) || same_point(b, midpoint) {
            return Err(TriangulationError::DegenerateInsertion);
        }
        self.insert_point(midpoint)
    }

    /// Recover a segment between existing vertices by flipping unconstrained
    /// edges that properly cross it. Existing collinear vertices split the
    /// request into a chain and receive the same lineage.
    pub fn recover_constraint(&mut self, constraint: Constraint) -> Result<(), TriangulationError> {
        let [a, b] = constraint.vertices;
        if a >= self.points.len() {
            return Err(TriangulationError::InvalidVertex(a));
        }
        if b >= self.points.len() {
            return Err(TriangulationError::InvalidVertex(b));
        }
        if a == b {
            return Err(TriangulationError::DegenerateInsertion);
        }
        let original = self.clone();
        let old_area = self.area();
        let result = self
            .recover_constraint_inner(constraint)
            .and_then(|()| self.legalize_inner())
            .and_then(|()| self.finish_mutation(old_area));
        if let Err(error) = result {
            *self = original;
            return Err(error);
        }
        Ok(())
    }

    fn recover_constraint_inner(
        &mut self,
        constraint: Constraint,
    ) -> Result<(), TriangulationError> {
        let wanted = constraint.vertices;
        let pa = self.points[wanted[0]];
        let pb = self.points[wanted[1]];
        let mut between: Vec<usize> = self
            .points
            .iter()
            .enumerate()
            .filter(|(index, point)| {
                *index != wanted[0] && *index != wanted[1] && strictly_between(**point, pa, pb)
            })
            .map(|(index, _)| index)
            .collect();
        if !between.is_empty() {
            let axis = if (pb[0] - pa[0]).abs() >= (pb[1] - pa[1]).abs() {
                0
            } else {
                1
            };
            between.sort_by(|left, right| {
                self.points[*left][axis]
                    .total_cmp(&self.points[*right][axis])
                    .then_with(|| left.cmp(right))
            });
            let mut chain = vec![wanted[0]];
            chain.extend(between);
            chain.push(wanted[1]);
            for pair in chain.windows(2) {
                if pair.len() != 2 {
                    return Err(TriangulationError::RecoveryFailed(wanted));
                }
                self.recover_constraint_inner(Constraint::new(
                    [pair[0], pair[1]],
                    constraint.marker,
                    constraint.kind,
                    constraint.source,
                ))?;
            }
            return Ok(());
        }

        let budget = self
            .triangles
            .len()
            .saturating_mul(self.triangles.len())
            .saturating_mul(20)
            .clamp(1_000, 2_000_000);
        let mut flips = 0usize;
        loop {
            let uses = self.edge_uses()?;
            if uses.contains_key(&wanted) {
                if let Some(existing) = self.constraints.get(&wanted) {
                    if *existing != constraint {
                        return Err(TriangulationError::ConstraintConflict(wanted));
                    }
                    return Ok(());
                }
                self.constraints.insert(wanted, constraint);
                self.validate()?;
                return Ok(());
            }

            let mut candidate = None;
            for (edge, incident) in uses {
                if incident.len() != 2 {
                    continue;
                }
                if !properly_crosses(pa, pb, self.points[edge[0]], self.points[edge[1]]) {
                    continue;
                }
                if self.constraints.contains_key(&edge) {
                    return Err(TriangulationError::ConstraintConflict(edge));
                }
                if let Some(flip) = self.prepare_flip(edge, incident, false) {
                    candidate = Some(flip);
                    break;
                }
            }
            let Some(flip) = candidate else {
                return Err(TriangulationError::RecoveryFailed(wanted));
            };
            self.triangles[flip.first] = flip.first_triangle;
            self.triangles[flip.second] = flip.second_triangle;
            self.validate()?;
            flips += 1;
            if flips > budget {
                return Err(TriangulationError::RecoveryLimit);
            }
        }
    }

    /// Legalise every flippable unconstrained edge in deterministic edge order.
    pub fn legalize(&mut self) -> Result<(), TriangulationError> {
        let original = self.clone();
        let old_area = self.area();
        if let Err(error) = self
            .legalize_inner()
            .and_then(|()| self.finish_mutation(old_area))
        {
            *self = original;
            return Err(error);
        }
        Ok(())
    }

    /// Check the complete mutable-kernel invariant set.
    pub fn validate(&self) -> Result<(), TriangulationError> {
        if self.points.is_empty() || self.triangles.is_empty() {
            return Err(TriangulationError::Empty);
        }
        for (index, point) in self.points.iter().enumerate() {
            if !point[0].is_finite() || !point[1].is_finite() {
                return Err(TriangulationError::NonFinitePoint(index));
            }
        }
        let mut triangle_keys = BTreeSet::new();
        for (index, triangle) in self.triangles.iter().enumerate() {
            let [a, b, c] = triangle.vertices;
            if a >= self.points.len() || b >= self.points.len() || c >= self.points.len() {
                return Err(TriangulationError::InvalidMesh(format!(
                    "triangle {index} references an unknown point"
                )));
            }
            if a == b || b == c || c == a {
                return Err(TriangulationError::InvalidMesh(format!(
                    "triangle {index} repeats a point"
                )));
            }
            if orient2d(self.points[a], self.points[b], self.points[c]) <= 0 {
                return Err(TriangulationError::InvalidMesh(format!(
                    "triangle {index} is not counter-clockwise"
                )));
            }
            let mut key = triangle.vertices;
            key.sort_unstable();
            if !triangle_keys.insert(key) {
                return Err(TriangulationError::InvalidMesh(format!(
                    "triangle {index} duplicates another triangle"
                )));
            }
        }

        let edges = self.edge_uses()?;
        for (edge, uses) in &edges {
            if uses.len() == 1 && !self.constraints.contains_key(edge) {
                return Err(TriangulationError::InvalidMesh(format!(
                    "boundary edge {edge:?} is not constrained"
                )));
            }
        }
        for edge in self.constraints.keys() {
            if edge[0] >= self.points.len() || edge[1] >= self.points.len() || edge[0] == edge[1] {
                return Err(TriangulationError::InvalidMesh(format!(
                    "constraint {edge:?} has invalid endpoints"
                )));
            }
            if !edges.contains_key(edge) {
                return Err(TriangulationError::InvalidMesh(format!(
                    "constraint {edge:?} is absent from the triangles"
                )));
            }
        }

        let mut boundary_degrees = BTreeMap::new();
        for (edge, uses) in &edges {
            if uses.len() == 1 {
                *boundary_degrees.entry(edge[0]).or_insert(0usize) += 1;
                *boundary_degrees.entry(edge[1]).or_insert(0usize) += 1;
            }
            for (point_index, &point) in self.points.iter().enumerate() {
                if point_index != edge[0]
                    && point_index != edge[1]
                    && point_on_segment(point, self.points[edge[0]], self.points[edge[1]])
                {
                    return Err(TriangulationError::InvalidMesh(format!(
                        "point {point_index} forms a T-junction on edge {edge:?}"
                    )));
                }
            }
        }
        if let Some((point, degree)) = boundary_degrees.iter().find(|(_, degree)| **degree != 2) {
            return Err(TriangulationError::InvalidMesh(format!(
                "boundary point {point} has degree {degree}, expected two"
            )));
        }
        Ok(())
    }

    fn locate(&self, point: Point) -> Result<Location, TriangulationError> {
        for (triangle_index, triangle) in self.triangles.iter().enumerate() {
            let [a, b, c] = triangle.vertices;
            let sides = [
                orient2d(self.points[a], self.points[b], point),
                orient2d(self.points[b], self.points[c], point),
                orient2d(self.points[c], self.points[a], point),
            ];
            if sides.iter().any(|&side| side < 0) {
                continue;
            }
            let zero_count = sides.iter().filter(|&&side| side == 0).count();
            return match zero_count {
                0 => Ok(Location::Interior(triangle_index)),
                1 => {
                    let local = sides.iter().position(|&side| side == 0).ok_or(
                        TriangulationError::InvalidMesh("edge location lost its edge".to_string()),
                    )?;
                    Ok(Location::Edge(edge_key(
                        triangle.vertices[local],
                        triangle.vertices[(local + 1) % 3],
                    )))
                }
                _ => Err(TriangulationError::InvalidMesh(
                    "a non-vertex insertion lies on two triangle edges".to_string(),
                )),
            };
        }
        Err(TriangulationError::PointOutside)
    }

    fn insert_in_triangle(
        &mut self,
        triangle_index: usize,
        point: usize,
    ) -> Result<(), TriangulationError> {
        let Some(&triangle) = self.triangles.get(triangle_index) else {
            return Err(TriangulationError::InvalidMesh(
                "interior insertion references an unknown triangle".to_string(),
            ));
        };
        let [a, b, c] = triangle.vertices;
        let replacement = Triangle::new([a, b, point], triangle.material);
        if orient2d(self.points[a], self.points[b], self.points[point]) <= 0
            || orient2d(self.points[b], self.points[c], self.points[point]) <= 0
            || orient2d(self.points[c], self.points[a], self.points[point]) <= 0
        {
            return Err(TriangulationError::DegenerateInsertion);
        }
        self.triangles[triangle_index] = replacement;
        self.triangles
            .push(Triangle::new([b, c, point], triangle.material));
        self.triangles
            .push(Triangle::new([c, a, point], triangle.material));
        Ok(())
    }

    fn insert_on_edge(&mut self, edge: [usize; 2], point: usize) -> Result<(), TriangulationError> {
        let uses = self
            .edge_uses()?
            .get(&edge)
            .cloned()
            .ok_or(TriangulationError::PointOutside)?;
        if uses.is_empty() || uses.len() > 2 {
            return Err(TriangulationError::InvalidMesh(format!(
                "edge {edge:?} has invalid incidence during insertion"
            )));
        }
        if !strictly_between(
            self.points[point],
            self.points[edge[0]],
            self.points[edge[1]],
        ) {
            return Err(TriangulationError::DegenerateInsertion);
        }
        for (triangle_index, local) in uses {
            let Some(&triangle) = self.triangles.get(triangle_index) else {
                return Err(TriangulationError::InvalidMesh(
                    "edge insertion references an unknown triangle".to_string(),
                ));
            };
            let a = triangle.vertices[local];
            let b = triangle.vertices[(local + 1) % 3];
            let opposite = triangle.vertices[(local + 2) % 3];
            let first = Triangle::new([opposite, a, point], triangle.material);
            let second = Triangle::new([opposite, point, b], triangle.material);
            if orient2d(
                self.points[first.vertices[0]],
                self.points[first.vertices[1]],
                self.points[first.vertices[2]],
            ) <= 0
                || orient2d(
                    self.points[second.vertices[0]],
                    self.points[second.vertices[1]],
                    self.points[second.vertices[2]],
                ) <= 0
            {
                return Err(TriangulationError::DegenerateInsertion);
            }
            self.triangles[triangle_index] = first;
            self.triangles.push(second);
        }
        if let Some(constraint) = self.constraints.remove(&edge) {
            for descendant in constraint.split(edge[0], point, edge[1]) {
                self.constraints.insert(descendant.vertices, descendant);
            }
        }
        Ok(())
    }

    fn legalize_inner(&mut self) -> Result<(), TriangulationError> {
        let budget = self
            .triangles
            .len()
            .saturating_mul(self.triangles.len())
            .saturating_mul(20)
            .clamp(1_000, 2_000_000);
        let mut flips = 0usize;
        loop {
            let uses = self.edge_uses()?;
            let mut candidate = None;
            for (edge, incident) in uses {
                if incident.len() != 2 || self.constraints.contains_key(&edge) {
                    continue;
                }
                if let Some(flip) = self.prepare_flip(edge, incident, true) {
                    candidate = Some(flip);
                    break;
                }
            }
            let Some(flip) = candidate else {
                return Ok(());
            };
            self.triangles[flip.first] = flip.first_triangle;
            self.triangles[flip.second] = flip.second_triangle;
            self.validate()?;
            flips += 1;
            if flips > budget {
                return Err(TriangulationError::LegalisationLimit);
            }
        }
    }

    fn prepare_flip(
        &self,
        edge: [usize; 2],
        incident: Vec<(usize, usize)>,
        delaunay_only: bool,
    ) -> Option<Flip> {
        let (first, first_local) = *incident.first()?;
        let (second, second_local) = *incident.get(1)?;
        let first_triangle = *self.triangles.get(first)?;
        let second_triangle = *self.triangles.get(second)?;
        let a = first_triangle.vertices[first_local];
        let b = first_triangle.vertices[(first_local + 1) % 3];
        let c = first_triangle.vertices[(first_local + 2) % 3];
        let d = second_triangle.vertices[(second_local + 2) % 3];
        if edge_key(a, b) != edge || c == d {
            return None;
        }
        let first_replacement = Triangle::new([a, d, c], first_triangle.material);
        let second_replacement = Triangle::new([b, c, d], second_triangle.material);
        if orient2d(self.points[a], self.points[d], self.points[c]) <= 0
            || orient2d(self.points[b], self.points[c], self.points[d]) <= 0
        {
            return None;
        }
        if delaunay_only
            && incircle(
                self.points[a],
                self.points[b],
                self.points[c],
                self.points[d],
            ) <= 0
        {
            return None;
        }
        Some(Flip {
            first,
            second,
            first_triangle: first_replacement,
            second_triangle: second_replacement,
        })
    }

    fn edge_uses(&self) -> Result<BTreeMap<[usize; 2], Vec<(usize, usize)>>, TriangulationError> {
        let mut result = BTreeMap::new();
        for (triangle_index, triangle) in self.triangles.iter().enumerate() {
            for local in 0..3 {
                let a = triangle.vertices[local];
                let b = triangle.vertices[(local + 1) % 3];
                result
                    .entry(edge_key(a, b))
                    .or_insert_with(Vec::new)
                    .push((triangle_index, local));
            }
        }
        for (edge, uses) in &result {
            if uses.len() > 2 {
                return Err(TriangulationError::InvalidMesh(format!(
                    "edge {edge:?} has more than two incident triangles"
                )));
            }
            if uses.len() == 2 {
                let first = uses[0];
                let second = uses[1];
                let first_edge = directed_edge(self.triangles[first.0], first.1);
                let second_edge = directed_edge(self.triangles[second.0], second.1);
                if first_edge.0 != second_edge.1 || first_edge.1 != second_edge.0 {
                    return Err(TriangulationError::InvalidMesh(format!(
                        "edge {edge:?} has inconsistently oriented neighbours"
                    )));
                }
            }
        }
        Ok(result)
    }

    fn finish_mutation(&self, old_area: f64) -> Result<(), TriangulationError> {
        self.validate()?;
        let new_area = self.area();
        let tolerance = 128.0 * f64::EPSILON * old_area.abs().max(new_area.abs());
        if !new_area.is_finite() || (new_area - old_area).abs() > tolerance {
            return Err(TriangulationError::InvalidMesh(format!(
                "area changed from {old_area} to {new_area} during a topology mutation"
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum Location {
    Interior(usize),
    Edge([usize; 2]),
}

#[derive(Clone, Copy)]
struct Flip {
    first: usize,
    second: usize,
    first_triangle: Triangle,
    second_triangle: Triangle,
}

fn edge_key(a: usize, b: usize) -> [usize; 2] {
    if a < b {
        [a, b]
    } else {
        [b, a]
    }
}

fn directed_edge(triangle: Triangle, local: usize) -> (usize, usize) {
    (triangle.vertices[local], triangle.vertices[(local + 1) % 3])
}

fn properly_crosses(a: Point, b: Point, c: Point, d: Point) -> bool {
    let ab_c = orient2d(a, b, c);
    let ab_d = orient2d(a, b, d);
    let cd_a = orient2d(c, d, a);
    let cd_b = orient2d(c, d, b);
    ab_c != 0 && ab_d != 0 && cd_a != 0 && cd_b != 0 && ab_c == -ab_d && cd_a == -cd_b
}

fn coordinate_key(point: Point) -> (u64, u64) {
    let canonical = |value: f64| if value == 0.0 { 0 } else { value.to_bits() };
    (canonical(point[0]), canonical(point[1]))
}

fn lookup_vertex(
    ids: &BTreeMap<(u64, u64), usize>,
    point: Point,
) -> Result<usize, TriangulationError> {
    ids.get(&coordinate_key(point)).copied().ok_or_else(|| {
        TriangulationError::InvalidMesh("a domain vertex was not inserted".to_string())
    })
}

fn point_bounds(points: &[Point]) -> Result<(Point, Point), TriangulationError> {
    let Some(&first) = points.first() else {
        return Err(TriangulationError::Empty);
    };
    let mut minimum = first;
    let mut maximum = first;
    for &point in &points[1..] {
        for axis in 0..2 {
            minimum[axis] = minimum[axis].min(point[axis]);
            maximum[axis] = maximum[axis].max(point[axis]);
        }
    }
    Ok((minimum, maximum))
}

fn triangle_area(points: &[Point], vertices: [usize; 3]) -> f64 {
    let a = points[vertices[0]];
    let b = points[vertices[1]];
    let c = points[vertices[2]];
    0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])).abs()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::unwrap_used)]

    use super::*;

    fn square() -> Triangulation {
        Triangulation::new(
            vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            vec![Triangle::new([0, 1, 2], 7), Triangle::new([0, 2, 3], 7)],
            vec![
                Constraint::new([0, 1], 11, ConstraintKind::OuterBoundary, 0),
                Constraint::new([1, 2], 11, ConstraintKind::OuterBoundary, 0),
                Constraint::new([2, 3], 11, ConstraintKind::OuterBoundary, 0),
                Constraint::new([0, 3], 11, ConstraintKind::OuterBoundary, 0),
            ],
        )
        .expect("valid square")
    }

    #[test]
    fn interior_insertion_preserves_area_constraints_and_material() {
        let mut mesh = square();
        let point = mesh.insert_point([0.25, 0.5]).expect("insert");
        assert_eq!(point, 4);
        assert_eq!(mesh.triangles().len(), 4);
        assert_eq!(mesh.constraints().count(), 4);
        assert!(mesh
            .triangles()
            .iter()
            .all(|triangle| triangle.material == 7));
        assert!((mesh.area() - 1.0).abs() < 1.0e-14);
        mesh.validate().expect("valid after insertion");
    }

    #[test]
    fn boundary_edge_insertion_preserves_lineage() {
        let mut mesh = square();
        let point = mesh.insert_point([0.5, 0.0]).expect("insert on boundary");
        assert_eq!(point, 4);
        assert_eq!(mesh.triangles().len(), 3);
        assert_eq!(mesh.constraints().count(), 5);
        for edge in [[0, 4], [1, 4]] {
            let constraint = mesh.constraint(edge).expect("descendant constraint");
            assert_eq!(constraint.marker, 11);
            assert_eq!(constraint.kind, ConstraintKind::OuterBoundary);
            assert_eq!(constraint.source, 0);
        }
        assert!(mesh.constraint([0, 1]).is_none());
        mesh.validate().expect("valid split boundary");
    }

    #[test]
    fn internal_constraint_splits_on_both_incident_triangles() {
        let mut mesh = Triangulation::new(
            vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            vec![Triangle::new([0, 1, 2], 7), Triangle::new([0, 2, 3], 9)],
            vec![
                Constraint::new([0, 1], 11, ConstraintKind::OuterBoundary, 0),
                Constraint::new([1, 2], 11, ConstraintKind::OuterBoundary, 0),
                Constraint::new([2, 3], 11, ConstraintKind::OuterBoundary, 0),
                Constraint::new([0, 3], 11, ConstraintKind::OuterBoundary, 0),
                Constraint::new([0, 2], 22, ConstraintKind::MaterialInterface, 3),
            ],
        )
        .expect("two materials");
        mesh.insert_point([0.5, 0.5]).expect("split interface");
        assert_eq!(mesh.triangles().len(), 4);
        assert_eq!(mesh.constraints().count(), 6);
        for edge in [[0, 4], [2, 4]] {
            let constraint = mesh.constraint(edge).expect("interface descendant");
            assert_eq!(constraint.marker, 22);
            assert_eq!(constraint.kind, ConstraintKind::MaterialInterface);
            assert_eq!(constraint.source, 3);
        }
        assert_eq!(
            mesh.triangles()
                .iter()
                .filter(|triangle| triangle.material == 7)
                .count(),
            2
        );
        assert_eq!(
            mesh.triangles()
                .iter()
                .filter(|triangle| triangle.material == 9)
                .count(),
            2
        );
        mesh.validate().expect("valid interface split");
    }

    #[test]
    fn split_constraint_uses_the_same_transaction_and_lineage() {
        let mut mesh = square();
        let point = mesh.split_constraint([0, 1]).expect("split");
        assert_eq!(mesh.points()[point], [0.5, 0.0]);
        assert_eq!(mesh.constraint([0, 4]).map(|c| c.marker), Some(11));
        assert_eq!(mesh.constraint([1, 4]).map(|c| c.marker), Some(11));
    }

    #[test]
    fn split_edge_accepts_free_edges_and_rejects_absent_edges() {
        let mut mesh = square();
        let point = mesh.split_edge([0, 2]).expect("split diagonal");
        assert_eq!(mesh.points()[point], [0.5, 0.5]);
        assert_eq!(mesh.triangles().len(), 4);
        assert!(mesh.constraint([0, 2]).is_none());
        assert_eq!(
            mesh.split_edge([0, 2]),
            Err(TriangulationError::MissingEdge([0, 2]))
        );
        mesh.validate().expect("valid free-edge split");
    }

    #[test]
    fn rejected_insertions_roll_back_exactly() {
        let mut mesh = square();
        let original = mesh.clone();
        assert_eq!(mesh.insert_point([0.5, 0.0]).map(|_| ()), Ok(()));
        let with_boundary_point = mesh.clone();
        assert_eq!(
            mesh.insert_point([0.5, 0.0]),
            Err(TriangulationError::DuplicatePoint(4))
        );
        assert_eq!(mesh, with_boundary_point);
        assert_eq!(
            mesh.insert_point([2.0, 0.5]),
            Err(TriangulationError::PointOutside)
        );
        assert_eq!(mesh, with_boundary_point);

        let mut original = original;
        assert_eq!(
            original.insert_point([f64::NAN, 0.0]),
            Err(TriangulationError::NonFinitePoint(4))
        );
        assert_eq!(original, square());
    }

    #[test]
    fn sequential_insertions_remain_conforming_and_delaunay() {
        let mut mesh = square();
        for point in [
            [0.25, 0.5],
            [0.75, 0.5],
            [0.5, 0.25],
            [0.5, 0.75],
            [0.5, 0.0],
            [0.5, 0.5],
        ] {
            mesh.insert_point(point).expect("sequential insertion");
            mesh.validate().expect("invariants after every insertion");
            assert!((mesh.area() - 1.0).abs() < 1.0e-13);
        }
        assert_eq!(mesh.points().len(), 10);
        assert_free_edges_are_delaunay(&mesh);
    }

    #[test]
    fn legalisation_is_deterministic() {
        let mut first = square();
        let mut second = square();
        for point in [[0.1, 0.4], [0.8, 0.2], [0.6, 0.9], [0.5, 0.5]] {
            first.insert_point(point).expect("first insertion");
            second.insert_point(point).expect("second insertion");
        }
        assert_eq!(first, second);
        assert_free_edges_are_delaunay(&first);
    }

    #[test]
    fn translated_and_scaled_insertions_use_exact_location_tests() {
        for scale in [1.0e-50, 1.0e50] {
            let offset = 10.0 * scale;
            let points = square()
                .points()
                .iter()
                .map(|point| [offset + scale * point[0], offset + scale * point[1]])
                .collect();
            let triangles = square().triangles().to_vec();
            let constraints = square().constraints().copied().collect();
            let mut mesh = Triangulation::new(points, triangles, constraints).expect("scaled mesh");
            mesh.insert_point([offset + 0.25 * scale, offset + 0.5 * scale])
                .expect("scaled insertion");
            mesh.insert_point([offset + 0.5 * scale, offset])
                .expect("scaled boundary insertion");
            mesh.validate().expect("scaled invariants");
        }
    }

    #[test]
    fn validation_rejects_missing_boundary_constraints_and_inversion() {
        let points = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let error = Triangulation::new(
            points.clone(),
            vec![Triangle::new([0, 1, 2], 0)],
            Vec::new(),
        )
        .expect_err("unconstrained boundary");
        assert!(matches!(error, TriangulationError::InvalidMesh(_)));
        let error = Triangulation::new(
            points,
            vec![Triangle::new([0, 2, 1], 0)],
            vec![
                Constraint::new([0, 1], 0, ConstraintKind::OuterBoundary, 0),
                Constraint::new([1, 2], 0, ConstraintKind::OuterBoundary, 0),
                Constraint::new([0, 2], 0, ConstraintKind::OuterBoundary, 0),
            ],
        )
        .expect_err("inverted triangle");
        assert!(matches!(error, TriangulationError::InvalidMesh(_)));
    }

    #[test]
    fn recovery_flips_the_crossing_diagonal_and_preserves_coverage() {
        let mut mesh = square();
        mesh.recover_constraint(Constraint::new([1, 3], 42, ConstraintKind::Feature, 6))
            .expect("recover other diagonal");
        assert_eq!(mesh.triangles().len(), 2);
        let constraint = mesh.constraint([1, 3]).expect("recovered edge");
        assert_eq!(constraint.marker, 42);
        assert_eq!(constraint.kind, ConstraintKind::Feature);
        assert_eq!(constraint.source, 6);
        assert!(mesh.constraint([0, 2]).is_none());
        assert!((mesh.area() - 1.0).abs() < 1.0e-14);
        mesh.validate().expect("valid recovered mesh");
        assert_free_edges_are_delaunay(&mesh);
    }

    #[test]
    fn a_recovered_constraint_can_be_split_without_losing_lineage() {
        let mut mesh = square();
        mesh.recover_constraint(Constraint::new(
            [1, 3],
            42,
            ConstraintKind::MaterialInterface,
            6,
        ))
        .expect("recover");
        let point = mesh.split_constraint([1, 3]).expect("split recovered edge");
        assert_eq!(point, 4);
        for edge in [[1, 4], [3, 4]] {
            let constraint = mesh.constraint(edge).expect("descendant");
            assert_eq!(constraint.marker, 42);
            assert_eq!(constraint.kind, ConstraintKind::MaterialInterface);
            assert_eq!(constraint.source, 6);
        }
        mesh.validate().expect("valid split recovery");
    }

    #[test]
    fn crossing_an_existing_constraint_is_rejected_with_rollback() {
        let mut mesh = square();
        mesh.recover_constraint(Constraint::new(
            [0, 2],
            22,
            ConstraintKind::MaterialInterface,
            3,
        ))
        .expect("fix original diagonal");
        let before = mesh.clone();
        assert_eq!(
            mesh.recover_constraint(Constraint::new([1, 3], 33, ConstraintKind::Feature, 4,)),
            Err(TriangulationError::ConstraintConflict([0, 2]))
        );
        assert_eq!(mesh, before);
    }

    #[test]
    fn recovery_follows_existing_collinear_vertices() {
        let mut mesh = square();
        mesh.insert_point([0.5, 0.0]).expect("boundary midpoint");
        let before = mesh.clone();
        mesh.recover_constraint(Constraint::new(
            [0, 1],
            11,
            ConstraintKind::OuterBoundary,
            0,
        ))
        .expect("existing chain");
        assert_eq!(mesh, before);
        assert_eq!(mesh.constraint([0, 4]).map(|c| c.marker), Some(11));
        assert_eq!(mesh.constraint([1, 4]).map(|c| c.marker), Some(11));
    }

    #[test]
    fn domain_square_is_a_fixed_vertex_cdt() {
        let domain =
            Domain::single(vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]).expect("domain");
        let mesh = Triangulation::from_domain(&domain).expect("fixed CDT");
        assert_eq!(mesh.points().len(), 4);
        assert_eq!(mesh.triangles().len(), 2);
        assert_eq!(mesh.constraints().count(), 4);
        assert_eq!(mesh.area(), 4.0);
        assert_free_edges_are_delaunay(&mesh);
    }

    #[test]
    fn domain_with_a_hole_preserves_every_boundary_and_coverage() {
        let domain = Domain::new(
            vec![crate::delaunay::Region::new(
                7,
                crate::delaunay::Loop::new(
                    vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
                    11,
                ),
            )],
            vec![crate::delaunay::Loop::new(
                vec![[2.0, 2.0], [4.0, 2.0], [4.0, 4.0], [2.0, 4.0]],
                12,
            )],
            Vec::new(),
            Vec::new(),
        )
        .expect("domain with hole");
        let mesh = Triangulation::from_domain(&domain).expect("CDT with hole");
        assert_eq!(mesh.points().len(), 8);
        assert_eq!(mesh.triangles().len(), 8);
        assert_eq!(mesh.constraints().count(), 8);
        assert_eq!(mesh.area(), 96.0);
        assert!(mesh
            .constraints()
            .any(|constraint| constraint.kind == ConstraintKind::HoleBoundary
                && constraint.marker == 12));
        assert!(mesh
            .triangles()
            .iter()
            .all(|triangle| triangle.material == 7));
        assert_free_edges_are_delaunay(&mesh);
    }

    #[test]
    fn concave_domain_does_not_fill_the_reentrant_corner() {
        let domain = Domain::single(vec![
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 2.0],
            [2.0, 2.0],
            [2.0, 4.0],
            [0.0, 4.0],
        ])
        .expect("L domain");
        let mesh = Triangulation::from_domain(&domain).expect("concave CDT");
        assert_eq!(mesh.points().len(), 6);
        assert_eq!(mesh.triangles().len(), 4);
        assert_eq!(mesh.area(), 12.0);
        assert_free_edges_are_delaunay(&mesh);
    }

    #[test]
    fn materials_and_features_keep_their_tags_from_domain_to_mesh() {
        let domain = Domain::new(
            vec![crate::delaunay::Region::new(
                1,
                crate::delaunay::Loop::new(
                    vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
                    10,
                ),
            )],
            Vec::new(),
            vec![crate::delaunay::Loop::new(
                vec![[3.0, 3.0], [7.0, 3.0], [7.0, 7.0], [3.0, 7.0]],
                20,
            )],
            vec![crate::delaunay::Feature::new(
                vec![[1.0, 5.0], [2.0, 5.0]],
                30,
            )],
        )
        .expect("tagged domain");
        let mesh = Triangulation::from_domain(&domain).expect("tagged CDT");
        assert_eq!(mesh.points().len(), 10);
        assert_eq!(mesh.constraints().count(), 9);
        assert!(mesh
            .triangles()
            .iter()
            .any(|triangle| triangle.material == 1));
        assert!(mesh
            .triangles()
            .iter()
            .any(|triangle| triangle.material == 20));
        assert!(mesh.constraints().any(|constraint| {
            constraint.kind == ConstraintKind::MaterialInterface && constraint.marker == 20
        }));
        assert!(mesh.constraints().any(|constraint| {
            constraint.kind == ConstraintKind::Feature && constraint.marker == 30
        }));
        assert_free_edges_are_delaunay(&mesh);
    }

    #[test]
    fn disconnected_regions_have_separate_closed_boundaries() {
        let domain = Domain::new(
            vec![
                crate::delaunay::Region::new(
                    4,
                    crate::delaunay::Loop::new(
                        vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
                        0,
                    ),
                ),
                crate::delaunay::Region::new(
                    8,
                    crate::delaunay::Loop::new(
                        vec![[4.0, 0.0], [5.0, 0.0], [5.0, 1.0], [4.0, 1.0]],
                        0,
                    ),
                ),
            ],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
        .expect("disjoint domain");
        let mesh = Triangulation::from_domain(&domain).expect("disjoint CDT");
        assert_eq!(mesh.points().len(), 8);
        assert_eq!(mesh.triangles().len(), 4);
        assert_eq!(mesh.constraints().count(), 8);
        assert_eq!(mesh.area(), 5.0);
        assert_eq!(
            mesh.triangles()
                .iter()
                .filter(|triangle| triangle.material == 4)
                .count(),
            2
        );
        assert_eq!(
            mesh.triangles()
                .iter()
                .filter(|triangle| triangle.material == 8)
                .count(),
            2
        );
    }

    #[test]
    fn star_domains_with_holes_survive_multi_flip_recovery() {
        for case in 0..12usize {
            let outer_count = 10 + 2 * case;
            let outer: Vec<Point> = (0..outer_count)
                .map(|index| {
                    let angle = std::f64::consts::TAU * index as f64 / outer_count as f64;
                    let radius = 8.0 + (case % 3) as f64 + 2.0 * (index % 2) as f64;
                    [radius * angle.cos(), radius * angle.sin()]
                })
                .collect();
            let hole: Vec<Point> = (0..8)
                .map(|index| {
                    let angle = std::f64::consts::TAU * index as f64 / 8.0;
                    [2.0 * angle.cos(), 2.0 * angle.sin()]
                })
                .collect();
            let domain = Domain::new(
                vec![crate::delaunay::Region::new(
                    case as i32,
                    crate::delaunay::Loop::new(outer, 0),
                )],
                vec![crate::delaunay::Loop::new(hole, 0)],
                Vec::new(),
                Vec::new(),
            )
            .expect("star domain");
            let mesh = Triangulation::from_domain(&domain).expect("star CDT");
            assert_eq!(mesh.points().len(), outer_count + 8);
            assert_eq!(mesh.triangles().len(), outer_count + 8);
            assert_eq!(mesh.constraints().count(), outer_count + 8);
            assert!((mesh.area() - domain.area()).abs() <= 1.0e-8 * domain.area());
            assert_free_edges_are_delaunay(&mesh);
        }
    }

    #[test]
    fn fixed_domain_cdt_supports_later_mutations() {
        let domain = Domain::new(
            vec![crate::delaunay::Region::new(
                7,
                crate::delaunay::Loop::new(
                    vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
                    11,
                ),
            )],
            vec![crate::delaunay::Loop::new(
                vec![[2.0, 2.0], [4.0, 2.0], [4.0, 4.0], [2.0, 4.0]],
                12,
            )],
            Vec::new(),
            Vec::new(),
        )
        .expect("domain");
        let mut mesh = Triangulation::from_domain(&domain).expect("fixed CDT");
        mesh.insert_point([6.0, 6.0])
            .expect("interior refinement point");
        let outer = mesh
            .constraints()
            .find(|constraint| constraint.kind == ConstraintKind::OuterBoundary)
            .copied()
            .expect("outer edge");
        mesh.split_constraint(outer.vertices)
            .expect("boundary refinement");
        mesh.validate().expect("dynamic domain invariants");
        assert!((mesh.area() - 96.0).abs() < 1.0e-12);
        assert_free_edges_are_delaunay(&mesh);
    }

    #[test]
    fn domain_construction_is_stable_under_translation_and_scaling() {
        for scale in [1.0e-20, 1.0e20] {
            let offset = 10.0 * scale;
            let ring: Vec<Point> = vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]]
                .into_iter()
                .map(|point| [offset + scale * point[0], offset + scale * point[1]])
                .collect();
            let domain = Domain::single(ring).expect("scaled domain");
            let mesh = Triangulation::from_domain(&domain).expect("scaled CDT");
            assert_eq!(mesh.points().len(), 4);
            assert_eq!(mesh.triangles().len(), 2);
            assert!((mesh.area() - domain.area()).abs() <= 1.0e-8 * domain.area());
            assert_free_edges_are_delaunay(&mesh);
        }
    }

    fn assert_free_edges_are_delaunay(mesh: &Triangulation) {
        for (edge, uses) in mesh.edge_uses().expect("edges") {
            if uses.len() != 2 || mesh.constraint(edge).is_some() {
                continue;
            }
            let first = mesh.triangles()[uses[0].0];
            let second = mesh.triangles()[uses[1].0];
            let a = first.vertices[uses[0].1];
            let b = first.vertices[(uses[0].1 + 1) % 3];
            let c = first.vertices[(uses[0].1 + 2) % 3];
            let d = second.vertices[(uses[1].1 + 2) % 3];
            let convex = orient2d(mesh.points()[a], mesh.points()[d], mesh.points()[c]) > 0
                && orient2d(mesh.points()[b], mesh.points()[c], mesh.points()[d]) > 0;
            if convex {
                assert!(
                    incircle(
                        mesh.points()[a],
                        mesh.points()[b],
                        mesh.points()[c],
                        mesh.points()[d]
                    ) <= 0,
                    "free edge {edge:?} is illegal"
                );
            }
        }
    }
}
