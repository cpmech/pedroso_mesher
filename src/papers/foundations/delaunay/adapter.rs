//! Adapter between route-oriented polygon loops and the robust mutable CDT.

use std::fmt;

use super::domain::{Domain, DomainError, Loop, Region};
use super::geometry::{same_point, Point};
use super::predicates::orient2d;
use super::triangulation::{Triangle, Triangulation, TriangulationError};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CdtLocation {
    Vertex(usize),
    Edge([usize; 2]),
    Triangle(usize),
    Outside,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdapterError {
    Domain(DomainError),
    Triangulation(TriangulationError),
}

impl fmt::Display for AdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Domain(error) => write!(f, "invalid triangulation domain: {error}"),
            Self::Triangulation(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for AdapterError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Domain(error) => Some(error),
            Self::Triangulation(error) => Some(error),
        }
    }
}

impl From<DomainError> for AdapterError {
    fn from(error: DomainError) -> Self {
        Self::Domain(error)
    }
}

impl From<TriangulationError> for AdapterError {
    fn from(error: TriangulationError) -> Self {
        Self::Triangulation(error)
    }
}

/// A route-facing view of outer polygons, their holes and interior Steiner
/// points. Point and triangle indices remain stable for the lifetime of the
/// adapter. Exterior faces are removed when the initial domain is constructed.
#[derive(Clone, Debug, PartialEq)]
pub struct CdtAdapter {
    mesh: Triangulation,
}

impl CdtAdapter {
    pub fn from_loops(outer: Vec<Point>, holes: Vec<Vec<Point>>) -> Result<Self, AdapterError> {
        Self::from_loops_and_points(outer, holes, std::iter::empty())
    }

    pub fn from_loops_and_points<I>(
        outer: Vec<Point>,
        holes: Vec<Vec<Point>>,
        points: I,
    ) -> Result<Self, AdapterError>
    where
        I: IntoIterator<Item = Point>,
    {
        Self::from_regions_and_points(vec![outer], holes, points)
    }

    pub fn from_regions_and_points<I>(
        outers: Vec<Vec<Point>>,
        holes: Vec<Vec<Point>>,
        points: I,
    ) -> Result<Self, AdapterError>
    where
        I: IntoIterator<Item = Point>,
    {
        let domain = Domain::new(
            outers
                .into_iter()
                .map(|outer| Region::new(0, Loop::new(outer, 0)))
                .collect(),
            holes.into_iter().map(|hole| Loop::new(hole, 0)).collect(),
            Vec::new(),
            Vec::new(),
        )?;
        let mesh = Triangulation::from_domain_and_points(&domain, points)?;
        Ok(Self { mesh })
    }

    /// Insert an interior point, returning the existing index on an exact
    /// coordinate duplicate. Boundary duplicates are handled the same way.
    pub fn insert_point(&mut self, point: Point) -> Result<usize, AdapterError> {
        insert_deduplicated(&mut self.mesh, point)
    }

    pub fn points(&self) -> &[Point] {
        self.mesh.points()
    }

    pub fn triangles(&self) -> &[Triangle] {
        self.mesh.triangles()
    }

    pub fn triangulation(&self) -> &Triangulation {
        &self.mesh
    }

    pub fn locate(&self, point: Point) -> CdtLocation {
        if let Some(index) = self
            .mesh
            .points()
            .iter()
            .position(|existing| same_point(*existing, point))
        {
            return CdtLocation::Vertex(index);
        }
        for (index, triangle) in self.mesh.triangles().iter().enumerate() {
            let [a, b, c] = triangle.vertices;
            let sides = [
                orient2d(self.mesh.points()[a], self.mesh.points()[b], point),
                orient2d(self.mesh.points()[b], self.mesh.points()[c], point),
                orient2d(self.mesh.points()[c], self.mesh.points()[a], point),
            ];
            if sides.iter().any(|side| *side < 0) {
                continue;
            }
            if let Some(local) = sides.iter().position(|side| *side == 0) {
                return CdtLocation::Edge([
                    triangle.vertices[local],
                    triangle.vertices[(local + 1) % 3],
                ]);
            }
            return CdtLocation::Triangle(index);
        }
        CdtLocation::Outside
    }
}

fn insert_deduplicated(mesh: &mut Triangulation, point: Point) -> Result<usize, AdapterError> {
    if let Some(index) = mesh
        .points()
        .iter()
        .position(|existing| same_point(*existing, point))
    {
        return Ok(index);
    }
    mesh.insert_point_local(point).map_err(AdapterError::from)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used)]

    use super::*;

    #[test]
    fn adapter_preserves_indices_and_classifies_insertions() {
        let mut cdt = CdtAdapter::from_loops(
            vec![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
            Vec::new(),
        )
        .expect("adapter");
        assert_eq!(cdt.insert_point([4.0, 0.0]), Ok(1));
        let interior = cdt.insert_point([1.0, 1.0]).expect("interior");
        assert_eq!(cdt.insert_point([1.0, 1.0]), Ok(interior));
        assert_eq!(cdt.locate([1.0, 1.0]), CdtLocation::Vertex(interior));
        assert!(matches!(
            cdt.locate([2.0, 2.0]),
            CdtLocation::Triangle(_) | CdtLocation::Edge(_)
        ));
        assert_eq!(cdt.locate([5.0, 2.0]), CdtLocation::Outside);
        cdt.triangulation().validate().expect("valid CDT");
    }

    #[test]
    fn disconnected_regions_share_the_same_adapter() {
        let cdt = CdtAdapter::from_regions_and_points(
            vec![
                vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
                vec![[4.0, 0.0], [6.0, 0.0], [6.0, 2.0], [4.0, 2.0]],
            ],
            Vec::new(),
            vec![[1.0, 1.0], [5.0, 1.0]],
        )
        .expect("disconnected adapter");
        assert_eq!(cdt.points().len(), 10);
        assert_eq!(cdt.triangulation().area(), 8.0);
        assert_eq!(cdt.locate([3.0, 1.0]), CdtLocation::Outside);
        cdt.triangulation().validate().expect("valid CDT");
    }

    #[test]
    fn holes_and_additional_points_are_integrated_before_route_use() {
        let cdt = CdtAdapter::from_loops_and_points(
            vec![[0.0, 0.0], [6.0, 0.0], [6.0, 6.0], [0.0, 6.0]],
            vec![vec![[2.0, 2.0], [4.0, 2.0], [4.0, 4.0], [2.0, 4.0]]],
            vec![[1.0, 3.0], [1.0, 3.0], [0.0, 0.0]],
        )
        .expect("holed adapter");
        assert_eq!(cdt.points().len(), 9);
        assert_eq!(cdt.triangulation().area(), 32.0);
        assert_eq!(cdt.locate([3.0, 3.0]), CdtLocation::Outside);
        assert!(cdt.triangles().iter().all(|triangle| {
            let [a, b, c] = triangle.vertices;
            let centroid = [
                (cdt.points()[a][0] + cdt.points()[b][0] + cdt.points()[c][0]) / 3.0,
                (cdt.points()[a][1] + cdt.points()[b][1] + cdt.points()[c][1]) / 3.0,
            ];
            !(2.0..4.0).contains(&centroid[0]) || !(2.0..4.0).contains(&centroid[1])
        }));
    }
}
