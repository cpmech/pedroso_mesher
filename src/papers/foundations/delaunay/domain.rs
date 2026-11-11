//! Validated polygonal domains with holes, embedded materials and features.
//!
//! Region and feature markers are input data, not geometric tolerances. Every
//! boundary decision uses exact predicates on the supplied binary64 values.

use std::fmt;

use super::geometry::{
    clean_path, clean_ring, invalid_shared_edge_contact, ring_area, ring_area_sign, ring_location,
    same_point, segments_intersect, Point, RingLocation,
};

/// A closed constraint loop and its physical-group marker.
#[derive(Clone, Debug, PartialEq)]
pub struct Loop {
    points: Vec<Point>,
    marker: i32,
}

impl Loop {
    pub fn new(points: Vec<Point>, marker: i32) -> Self {
        Self { points, marker }
    }

    pub fn points(&self) -> &[Point] {
        &self.points
    }

    pub fn marker(&self) -> i32 {
        self.marker
    }
}

/// One disconnected material component. Its material marker is the default
/// inside the outer boundary unless an embedded material loop overrides it.
#[derive(Clone, Debug, PartialEq)]
pub struct Region {
    boundary: Loop,
    holes: Vec<Loop>,
    material: i32,
}

impl Region {
    pub fn new(material: i32, boundary: Loop) -> Self {
        Self {
            boundary,
            holes: Vec::new(),
            material,
        }
    }

    pub fn boundary(&self) -> &Loop {
        &self.boundary
    }

    pub fn holes(&self) -> &[Loop] {
        &self.holes
    }

    pub fn material(&self) -> i32 {
        self.material
    }
}

/// An open internal feature polyline and its physical-group marker.
#[derive(Clone, Debug, PartialEq)]
pub struct Feature {
    path: Vec<Point>,
    marker: i32,
}

impl Feature {
    pub fn new(path: Vec<Point>, marker: i32) -> Self {
        Self { path, marker }
    }

    pub fn path(&self) -> &[Point] {
        &self.path
    }

    pub fn marker(&self) -> i32 {
        self.marker
    }
}

/// A validated domain. Embedded material loops may nest; the innermost loop
/// determines the material. All other independent constraints are disjoint.
#[derive(Clone, Debug, PartialEq)]
pub struct Domain {
    regions: Vec<Region>,
    materials: Vec<Loop>,
    features: Vec<Feature>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointLocation {
    Outside,
    Boundary,
    Inside { region: usize, material: i32 },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DomainError {
    NoRegions,
    NonFiniteCoordinate { entity: usize },
    LoopTooShort { entity: usize, points: usize },
    FeatureTooShort { feature: usize, points: usize },
    RepeatedVertex { entity: usize },
    DegenerateLoop { entity: usize },
    SelfIntersection { entity: usize },
    FeatureSelfIntersection { feature: usize },
    LoopIntersection { first: usize, second: usize },
    NestedRegions { outer: usize, inner: usize },
    HoleOutsideRegion { hole: usize },
    NestedHoles { outer: usize, inner: usize },
    MaterialOutsideRegion { material: usize },
    MaterialInHole { material: usize },
    FeatureOutsideDomain { feature: usize },
    FeatureIntersectsLoop { feature: usize, entity: usize },
    FeatureIntersection { first: usize, second: usize },
    InvalidArea,
}

impl fmt::Display for DomainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoRegions => write!(f, "the domain needs at least one outer region"),
            Self::NonFiniteCoordinate { entity } => {
                write!(f, "geometry entity {entity} has a non-finite coordinate")
            }
            Self::LoopTooShort { entity, points } => write!(
                f,
                "closed loop {entity} has {points} distinct points; at least three are required"
            ),
            Self::FeatureTooShort { feature, points } => write!(
                f,
                "feature {feature} has {points} distinct points; at least two are required"
            ),
            Self::RepeatedVertex { entity } => {
                write!(f, "geometry entity {entity} repeats a non-adjacent vertex")
            }
            Self::DegenerateLoop { entity } => {
                write!(f, "closed loop {entity} has zero area")
            }
            Self::SelfIntersection { entity } => {
                write!(
                    f,
                    "closed loop {entity} crosses, touches or overlaps itself"
                )
            }
            Self::FeatureSelfIntersection { feature } => {
                write!(f, "feature {feature} crosses, touches or overlaps itself")
            }
            Self::LoopIntersection { first, second } => {
                write!(f, "closed loops {first} and {second} intersect or touch")
            }
            Self::NestedRegions { outer, inner } => write!(
                f,
                "outer region {inner} lies inside region {outer}; outer regions must be disjoint"
            ),
            Self::HoleOutsideRegion { hole } => {
                write!(
                    f,
                    "hole {hole} is not strictly inside exactly one outer region"
                )
            }
            Self::NestedHoles { outer, inner } => {
                write!(f, "hole {inner} lies inside hole {outer}")
            }
            Self::MaterialOutsideRegion { material } => write!(
                f,
                "embedded material {material} is not strictly inside exactly one outer region"
            ),
            Self::MaterialInHole { material } => {
                write!(f, "embedded material {material} lies in a hole")
            }
            Self::FeatureOutsideDomain { feature } => {
                write!(
                    f,
                    "feature {feature} leaves the material domain or touches its boundary"
                )
            }
            Self::FeatureIntersectsLoop { feature, entity } => write!(
                f,
                "feature {feature} intersects or touches closed loop {entity}"
            ),
            Self::FeatureIntersection { first, second } => {
                write!(f, "features {first} and {second} intersect or touch")
            }
            Self::InvalidArea => write!(f, "the domain has zero or non-finite area"),
        }
    }
}

impl std::error::Error for DomainError {}

impl Domain {
    /// Validate and construct a domain. Closed loops may be clockwise or
    /// counter-clockwise, and may repeat their first point at the end.
    pub fn new(
        regions: Vec<Region>,
        holes: Vec<Loop>,
        materials: Vec<Loop>,
        features: Vec<Feature>,
    ) -> Result<Self, DomainError> {
        if regions.is_empty() {
            return Err(DomainError::NoRegions);
        }
        let mut cleaned_regions = Vec::with_capacity(regions.len());
        for (index, region) in regions.into_iter().enumerate() {
            let boundary = validated_loop(region.boundary, index)?;
            cleaned_regions.push(Region::new(region.material, boundary));
        }
        let hole_offset = cleaned_regions.len();
        let mut cleaned_holes = Vec::with_capacity(holes.len());
        for (index, hole) in holes.into_iter().enumerate() {
            cleaned_holes.push(validated_loop(hole, hole_offset + index)?);
        }
        let material_offset = hole_offset + cleaned_holes.len();
        let mut cleaned_materials = Vec::with_capacity(materials.len());
        for (index, material) in materials.into_iter().enumerate() {
            cleaned_materials.push(validated_loop(material, material_offset + index)?);
        }
        let mut cleaned_features = Vec::with_capacity(features.len());
        for (index, feature) in features.into_iter().enumerate() {
            cleaned_features.push(validated_feature(feature, index)?);
        }

        validate_distinct_loops(&cleaned_regions, &cleaned_holes, &cleaned_materials)?;
        validate_region_nesting(&cleaned_regions)?;
        validate_hole_nesting(&cleaned_holes)?;
        assign_holes(&mut cleaned_regions, &cleaned_holes)?;
        validate_material_locations(&cleaned_regions, &cleaned_holes, &cleaned_materials)?;

        let domain = Self {
            regions: cleaned_regions,
            materials: cleaned_materials,
            features: cleaned_features,
        };
        domain.validate_features()?;
        if !domain.area().is_finite() || domain.area() <= 0.0 {
            return Err(DomainError::InvalidArea);
        }
        Ok(domain)
    }

    /// One untagged outer polygon, with no holes or internal constraints.
    pub fn single(boundary: Vec<Point>) -> Result<Self, DomainError> {
        Self::new(
            vec![Region::new(0, Loop::new(boundary, 0))],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
    }

    pub fn regions(&self) -> &[Region] {
        &self.regions
    }

    pub fn materials(&self) -> &[Loop] {
        &self.materials
    }

    pub fn features(&self) -> &[Feature] {
        &self.features
    }

    pub fn holes(&self) -> impl Iterator<Item = &Loop> {
        self.regions.iter().flat_map(|region| region.holes())
    }

    /// Material area, excluding holes. Embedded materials do not change area.
    pub fn area(&self) -> f64 {
        self.regions
            .iter()
            .map(|region| {
                ring_area(region.boundary.points()).abs()
                    - region
                        .holes
                        .iter()
                        .map(|hole| ring_area(hole.points()).abs())
                        .sum::<f64>()
            })
            .sum()
    }

    /// Strict point classification. Any outer, hole, material or feature edge is
    /// reported as `Boundary`; material membership is defined only in the open
    /// interior between constraints.
    pub fn locate(&self, point: Point) -> PointLocation {
        if !point[0].is_finite() || !point[1].is_finite() {
            return PointLocation::Outside;
        }
        let location = self.locate_without_features(point);
        if location != PointLocation::Outside
            && self.features.iter().any(|feature| {
                feature.path.windows(2).any(|edge| {
                    edge.len() == 2 && super::geometry::point_on_segment(point, edge[0], edge[1])
                })
            })
        {
            return PointLocation::Boundary;
        }
        location
    }

    pub fn contains(&self, point: Point) -> bool {
        matches!(self.locate(point), PointLocation::Inside { .. })
    }

    fn locate_without_features(&self, point: Point) -> PointLocation {
        for (region_index, region) in self.regions.iter().enumerate() {
            match ring_location(point, region.boundary.points()) {
                RingLocation::Boundary => return PointLocation::Boundary,
                RingLocation::Outside => continue,
                RingLocation::Inside => {}
            }
            let mut in_hole = false;
            for hole in &region.holes {
                match ring_location(point, hole.points()) {
                    RingLocation::Boundary => return PointLocation::Boundary,
                    RingLocation::Inside => {
                        in_hole = true;
                        break;
                    }
                    RingLocation::Outside => {}
                }
            }
            if in_hole {
                return PointLocation::Outside;
            }
            for material in &self.materials {
                if ring_location(point, material.points()) == RingLocation::Boundary {
                    return PointLocation::Boundary;
                }
            }
            let material = self.effective_material(point, region_index, region.material);
            return PointLocation::Inside {
                region: region_index,
                material,
            };
        }
        PointLocation::Outside
    }

    fn effective_material(&self, point: Point, region: usize, default: i32) -> i32 {
        let mut selected = default;
        let mut greatest_depth = 0usize;
        for (index, material) in self.materials.iter().enumerate() {
            if ring_location(point, material.points()) != RingLocation::Inside {
                continue;
            }
            let owner_matches =
                ring_location(material.points()[0], self.regions[region].boundary.points())
                    == RingLocation::Inside;
            if !owner_matches {
                continue;
            }
            let depth = self
                .materials
                .iter()
                .enumerate()
                .filter(|(other_index, other)| {
                    index != *other_index
                        && ring_location(material.points()[0], other.points())
                            == RingLocation::Inside
                })
                .count();
            if depth >= greatest_depth {
                greatest_depth = depth;
                selected = material.marker();
            }
        }
        selected
    }

    fn validate_features(&self) -> Result<(), DomainError> {
        let loops = self.all_loops();
        for (feature_index, feature) in self.features.iter().enumerate() {
            for edge in feature.path.windows(2) {
                if edge.len() != 2 {
                    return Err(DomainError::FeatureOutsideDomain {
                        feature: feature_index,
                    });
                }
                let midpoint = [
                    edge[0][0] + 0.5 * (edge[1][0] - edge[0][0]),
                    edge[0][1] + 0.5 * (edge[1][1] - edge[0][1]),
                ];
                if self.locate_without_features(edge[0]) != self.locate_without_features(midpoint)
                    || !matches!(
                        self.locate_without_features(edge[0]),
                        PointLocation::Inside { .. }
                    )
                    || !matches!(
                        self.locate_without_features(edge[1]),
                        PointLocation::Inside { .. }
                    )
                {
                    return Err(DomainError::FeatureOutsideDomain {
                        feature: feature_index,
                    });
                }
            }
            for &(entity, loop_points) in &loops {
                if path_intersects_ring(&feature.path, loop_points) {
                    return Err(DomainError::FeatureIntersectsLoop {
                        feature: feature_index,
                        entity,
                    });
                }
            }
            for other in 0..feature_index {
                if paths_intersect(&self.features[other].path, &feature.path) {
                    return Err(DomainError::FeatureIntersection {
                        first: other,
                        second: feature_index,
                    });
                }
            }
        }
        Ok(())
    }

    fn all_loops(&self) -> Vec<(usize, &[Point])> {
        let mut result = Vec::new();
        for (index, region) in self.regions.iter().enumerate() {
            result.push((index, region.boundary.points()));
            for hole in &region.holes {
                result.push((result.len(), hole.points()));
            }
        }
        let material_offset = result.len();
        for (index, material) in self.materials.iter().enumerate() {
            result.push((material_offset + index, material.points()));
        }
        result
    }
}

fn ensure_finite(points: &[Point], entity: usize) -> Result<(), DomainError> {
    if points
        .iter()
        .any(|point| !point[0].is_finite() || !point[1].is_finite())
    {
        return Err(DomainError::NonFiniteCoordinate { entity });
    }
    Ok(())
}

fn validated_loop(input: Loop, entity: usize) -> Result<Loop, DomainError> {
    ensure_finite(input.points(), entity)?;
    let points = clean_ring(input.points());
    if points.len() < 3 {
        return Err(DomainError::LoopTooShort {
            entity,
            points: points.len(),
        });
    }
    validate_unique_vertices(&points, entity)?;
    validate_simple_chain(&points, true, entity)?;
    if ring_area_sign(&points) == 0 {
        return Err(DomainError::DegenerateLoop { entity });
    }
    Ok(Loop::new(points, input.marker()))
}

fn validated_feature(input: Feature, feature: usize) -> Result<Feature, DomainError> {
    ensure_finite(input.path(), feature)?;
    let path = clean_path(input.path());
    if path.len() < 2 {
        return Err(DomainError::FeatureTooShort {
            feature,
            points: path.len(),
        });
    }
    validate_unique_vertices(&path, feature)?;
    validate_simple_chain(&path, false, feature)?;
    Ok(Feature::new(path, input.marker()))
}

fn validate_unique_vertices(points: &[Point], entity: usize) -> Result<(), DomainError> {
    for (index, &point) in points.iter().enumerate() {
        if points[..index]
            .iter()
            .any(|&previous| same_point(previous, point))
        {
            return Err(DomainError::RepeatedVertex { entity });
        }
    }
    Ok(())
}

fn validate_simple_chain(points: &[Point], closed: bool, entity: usize) -> Result<(), DomainError> {
    let edge_count = if closed {
        points.len()
    } else {
        points.len() - 1
    };
    let edge = |index: usize| (points[index], points[(index + 1) % points.len()]);
    for first in 0..edge_count {
        for second in first + 1..edge_count {
            let adjacent = second == first + 1 || closed && first == 0 && second + 1 == edge_count;
            let (a, b) = edge(first);
            let (c, d) = edge(second);
            let invalid = if adjacent {
                invalid_shared_edge_contact(a, b, c, d)
            } else {
                segments_intersect(a, b, c, d)
            };
            if invalid {
                return if closed {
                    Err(DomainError::SelfIntersection { entity })
                } else {
                    Err(DomainError::FeatureSelfIntersection { feature: entity })
                };
            }
        }
    }
    Ok(())
}

fn loop_entities<'a>(
    regions: &'a [Region],
    holes: &'a [Loop],
    materials: &'a [Loop],
) -> Vec<(usize, &'a [Point])> {
    let mut result = Vec::new();
    for (index, region) in regions.iter().enumerate() {
        result.push((index, region.boundary.points()));
    }
    for (index, hole) in holes.iter().enumerate() {
        result.push((regions.len() + index, hole.points()));
    }
    for (index, material) in materials.iter().enumerate() {
        result.push((regions.len() + holes.len() + index, material.points()));
    }
    result
}

fn validate_distinct_loops(
    regions: &[Region],
    holes: &[Loop],
    materials: &[Loop],
) -> Result<(), DomainError> {
    let loops = loop_entities(regions, holes, materials);
    for first in 0..loops.len() {
        for second in first + 1..loops.len() {
            if rings_intersect(loops[first].1, loops[second].1) {
                return Err(DomainError::LoopIntersection {
                    first: loops[first].0,
                    second: loops[second].0,
                });
            }
        }
    }
    Ok(())
}

fn rings_intersect(first: &[Point], second: &[Point]) -> bool {
    (0..first.len()).any(|i| {
        (0..second.len()).any(|j| {
            segments_intersect(
                first[i],
                first[(i + 1) % first.len()],
                second[j],
                second[(j + 1) % second.len()],
            )
        })
    })
}

fn path_intersects_ring(path: &[Point], ring: &[Point]) -> bool {
    path.windows(2).any(|edge| {
        edge.len() == 2
            && (0..ring.len())
                .any(|i| segments_intersect(edge[0], edge[1], ring[i], ring[(i + 1) % ring.len()]))
    })
}

fn paths_intersect(first: &[Point], second: &[Point]) -> bool {
    first.windows(2).any(|a| {
        a.len() == 2
            && second
                .windows(2)
                .any(|b| b.len() == 2 && segments_intersect(a[0], a[1], b[0], b[1]))
    })
}

fn validate_region_nesting(regions: &[Region]) -> Result<(), DomainError> {
    for outer in 0..regions.len() {
        for inner in 0..regions.len() {
            if outer != inner
                && ring_location(
                    regions[inner].boundary.points()[0],
                    regions[outer].boundary.points(),
                ) == RingLocation::Inside
            {
                return Err(DomainError::NestedRegions { outer, inner });
            }
        }
    }
    Ok(())
}

fn validate_hole_nesting(holes: &[Loop]) -> Result<(), DomainError> {
    for outer in 0..holes.len() {
        for inner in 0..holes.len() {
            if outer != inner
                && ring_location(holes[inner].points()[0], holes[outer].points())
                    == RingLocation::Inside
            {
                return Err(DomainError::NestedHoles { outer, inner });
            }
        }
    }
    Ok(())
}

fn assign_holes(regions: &mut [Region], holes: &[Loop]) -> Result<(), DomainError> {
    for (hole_index, hole) in holes.iter().enumerate() {
        let mut owner = None;
        for (region_index, region) in regions.iter().enumerate() {
            if ring_location(hole.points()[0], region.boundary.points()) == RingLocation::Inside {
                if owner.is_some() {
                    return Err(DomainError::HoleOutsideRegion { hole: hole_index });
                }
                owner = Some(region_index);
            }
        }
        let Some(owner) = owner else {
            return Err(DomainError::HoleOutsideRegion { hole: hole_index });
        };
        let Some(region) = regions.get_mut(owner) else {
            return Err(DomainError::HoleOutsideRegion { hole: hole_index });
        };
        region.holes.push(hole.clone());
    }
    Ok(())
}

fn validate_material_locations(
    regions: &[Region],
    holes: &[Loop],
    materials: &[Loop],
) -> Result<(), DomainError> {
    for (material_index, material) in materials.iter().enumerate() {
        let probe = material.points()[0];
        let owners = regions
            .iter()
            .filter(|region| ring_location(probe, region.boundary.points()) == RingLocation::Inside)
            .count();
        if owners != 1 {
            return Err(DomainError::MaterialOutsideRegion {
                material: material_index,
            });
        }
        if holes
            .iter()
            .any(|hole| ring_location(probe, hole.points()) == RingLocation::Inside)
        {
            return Err(DomainError::MaterialInHole {
                material: material_index,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;

    fn square(min: f64, max: f64) -> Vec<Point> {
        vec![[min, min], [max, min], [max, max], [min, max]]
    }

    fn base_domain() -> Domain {
        Domain::single(square(0.0, 10.0)).expect("valid square")
    }

    #[test]
    fn cleans_closing_and_adjacent_duplicate_vertices() {
        let mut ring = square(0.0, 2.0);
        ring.insert(1, ring[0]);
        ring.push(ring[0]);
        let domain = Domain::single(ring).expect("clean ring");
        assert_eq!(domain.regions()[0].boundary().points().len(), 4);
        assert_eq!(domain.area(), 4.0);
    }

    #[test]
    fn classifies_holes_and_preserves_markers() {
        let domain = Domain::new(
            vec![Region::new(7, Loop::new(square(0.0, 10.0), 11))],
            vec![Loop::new(square(2.0, 4.0), 12)],
            vec![Loop::new(square(6.0, 8.0), 21)],
            Vec::new(),
        )
        .expect("domain");
        assert_eq!(domain.regions()[0].boundary().marker(), 11);
        assert_eq!(domain.holes().next().map(Loop::marker), Some(12));
        assert_eq!(
            domain.locate([1.0, 1.0]),
            PointLocation::Inside {
                region: 0,
                material: 7
            }
        );
        assert_eq!(
            domain.locate([7.0, 7.0]),
            PointLocation::Inside {
                region: 0,
                material: 21
            }
        );
        assert_eq!(domain.locate([3.0, 3.0]), PointLocation::Outside);
        assert_eq!(domain.locate([0.0, 3.0]), PointLocation::Boundary);
        assert_eq!(domain.area(), 96.0);
    }

    #[test]
    fn innermost_embedded_material_wins() {
        let domain = Domain::new(
            vec![Region::new(1, Loop::new(square(0.0, 10.0), 0))],
            Vec::new(),
            vec![
                Loop::new(square(1.0, 9.0), 2),
                Loop::new(square(3.0, 7.0), 3),
            ],
            Vec::new(),
        )
        .expect("nested materials");
        assert_eq!(
            domain.locate([2.0, 5.0]),
            PointLocation::Inside {
                region: 0,
                material: 2
            }
        );
        assert_eq!(
            domain.locate([5.0, 5.0]),
            PointLocation::Inside {
                region: 0,
                material: 3
            }
        );
        assert_eq!(domain.locate([3.0, 5.0]), PointLocation::Boundary);
    }

    #[test]
    fn disjoint_regions_and_input_orientation_are_supported() {
        let mut first = square(0.0, 2.0);
        first.reverse();
        let mut second = square(0.0, 1.0);
        for point in &mut second {
            point[0] += 4.0;
        }
        let domain = Domain::new(
            vec![
                Region::new(4, Loop::new(first, 0)),
                Region::new(8, Loop::new(second, 0)),
            ],
            Vec::new(),
            Vec::new(),
            Vec::new(),
        )
        .expect("disjoint regions");
        assert_eq!(domain.area(), 5.0);
        assert_eq!(
            domain.locate([4.5, 0.5]),
            PointLocation::Inside {
                region: 1,
                material: 8
            }
        );
    }

    #[test]
    fn features_are_internal_constraints_with_markers() {
        let domain = Domain::new(
            vec![Region::new(1, Loop::new(square(0.0, 10.0), 2))],
            Vec::new(),
            Vec::new(),
            vec![Feature::new(vec![[2.0, 2.0], [5.0, 5.0], [8.0, 2.0]], 33)],
        )
        .expect("feature");
        assert_eq!(domain.features()[0].marker(), 33);
        assert_eq!(domain.locate([4.0, 4.0]), PointLocation::Boundary);
        assert!(domain.contains([4.0, 4.5]));
    }

    #[test]
    fn signed_zero_does_not_create_distinct_vertices() {
        let error = Domain::single(vec![
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [-0.0, 0.0],
            [0.0, 2.0],
        ])
        .expect_err("signed-zero duplicate");
        assert!(matches!(
            error,
            DomainError::RepeatedVertex { .. } | DomainError::SelfIntersection { .. }
        ));
    }

    #[test]
    fn rejects_non_finite_degenerate_and_self_intersecting_loops() {
        assert!(matches!(
            Domain::single(vec![[0.0, 0.0], [f64::NAN, 0.0], [1.0, 1.0]]),
            Err(DomainError::NonFiniteCoordinate { .. })
        ));
        assert!(matches!(
            Domain::single(vec![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
            Err(DomainError::DegenerateLoop { .. }) | Err(DomainError::SelfIntersection { .. })
        ));
        assert!(matches!(
            Domain::single(vec![[0.0, 0.0], [2.0, 2.0], [0.0, 2.0], [2.0, 0.0]]),
            Err(DomainError::SelfIntersection { .. })
        ));
    }

    #[test]
    fn rejects_backtracking_and_non_adjacent_touches() {
        assert!(matches!(
            Domain::single(vec![
                [0.0, 0.0],
                [4.0, 0.0],
                [2.0, 0.0],
                [2.0, 2.0],
                [0.0, 2.0]
            ]),
            Err(DomainError::SelfIntersection { .. })
        ));
        assert!(matches!(
            Domain::single(vec![
                [0.0, 0.0],
                [4.0, 0.0],
                [4.0, 4.0],
                [2.0, 0.0],
                [0.0, 4.0]
            ]),
            Err(DomainError::SelfIntersection { .. })
        ));
    }

    #[test]
    fn rejects_outside_touching_and_nested_holes() {
        let region = || vec![Region::new(0, Loop::new(square(0.0, 10.0), 0))];
        assert!(matches!(
            Domain::new(
                region(),
                vec![Loop::new(square(12.0, 14.0), 0)],
                Vec::new(),
                Vec::new()
            ),
            Err(DomainError::HoleOutsideRegion { .. })
        ));
        assert!(matches!(
            Domain::new(
                region(),
                vec![Loop::new(square(0.0, 2.0), 0)],
                Vec::new(),
                Vec::new()
            ),
            Err(DomainError::LoopIntersection { .. })
        ));
        assert!(matches!(
            Domain::new(
                region(),
                vec![
                    Loop::new(square(1.0, 8.0), 0),
                    Loop::new(square(2.0, 4.0), 0)
                ],
                Vec::new(),
                Vec::new()
            ),
            Err(DomainError::NestedHoles { .. })
        ));
    }

    #[test]
    fn rejects_misplaced_materials_and_intersections() {
        let region = || vec![Region::new(0, Loop::new(square(0.0, 10.0), 0))];
        assert!(matches!(
            Domain::new(
                region(),
                Vec::new(),
                vec![Loop::new(square(12.0, 14.0), 1)],
                Vec::new()
            ),
            Err(DomainError::MaterialOutsideRegion { .. })
        ));
        assert!(matches!(
            Domain::new(
                region(),
                Vec::new(),
                vec![
                    Loop::new(square(1.0, 8.0), 1),
                    Loop::new(square(4.0, 12.0), 2)
                ],
                Vec::new()
            ),
            Err(DomainError::LoopIntersection { .. })
        ));
    }

    #[test]
    fn rejects_feature_boundary_contact_crossings_and_junctions() {
        let region = || vec![Region::new(0, Loop::new(square(0.0, 10.0), 0))];
        assert!(matches!(
            Domain::new(
                region(),
                Vec::new(),
                Vec::new(),
                vec![Feature::new(vec![[0.0, 0.0], [2.0, 2.0]], 0)]
            ),
            Err(DomainError::FeatureOutsideDomain { .. })
                | Err(DomainError::FeatureIntersectsLoop { .. })
        ));
        assert!(matches!(
            Domain::new(
                region(),
                Vec::new(),
                Vec::new(),
                vec![
                    Feature::new(vec![[2.0, 2.0], [8.0, 8.0]], 0),
                    Feature::new(vec![[2.0, 8.0], [8.0, 2.0]], 0)
                ]
            ),
            Err(DomainError::FeatureIntersection { .. })
        ));
        assert!(matches!(
            Domain::new(
                region(),
                Vec::new(),
                Vec::new(),
                vec![
                    Feature::new(vec![[2.0, 2.0], [5.0, 5.0]], 0),
                    Feature::new(vec![[5.0, 5.0], [8.0, 2.0]], 0)
                ]
            ),
            Err(DomainError::FeatureIntersection { .. })
        ));
    }

    #[test]
    fn exact_validation_survives_translation_and_extreme_scaling() {
        for scale in [1.0e-100, 1.0, 1.0e100] {
            let offset = 1.0e8 * scale;
            let ring = square(0.0, 1.0)
                .into_iter()
                .map(|point| [offset + scale * point[0], offset + scale * point[1]])
                .collect();
            let domain = Domain::single(ring).expect("scaled square");
            assert!(domain.contains([offset + 0.25 * scale, offset + 0.25 * scale]));
        }
    }

    #[test]
    fn base_domain_contains_only_strict_interior() {
        let domain = base_domain();
        assert!(domain.contains([5.0, 5.0]));
        assert!(!domain.contains([10.0, 5.0]));
        assert!(!domain.contains([11.0, 5.0]));
    }
}
