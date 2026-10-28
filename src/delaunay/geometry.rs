//! Exact-topology helpers for polygonal domains.

use super::exact;
use super::predicates::orient2d;

pub type Point = [f64; 2];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RingLocation {
    Inside,
    Boundary,
    Outside,
}

pub(crate) fn same_point(a: Point, b: Point) -> bool {
    a[0] == b[0] && a[1] == b[1]
}

pub(crate) fn clean_ring(points: &[Point]) -> Vec<Point> {
    let mut result = Vec::with_capacity(points.len());
    for &point in points {
        if result.last().is_none_or(|&last| !same_point(last, point)) {
            result.push(point);
        }
    }
    while result.len() > 1
        && result
            .first()
            .zip(result.last())
            .is_some_and(|(&first, &last)| same_point(first, last))
    {
        result.pop();
    }
    result
}

pub(crate) fn clean_path(points: &[Point]) -> Vec<Point> {
    let mut result = Vec::with_capacity(points.len());
    for &point in points {
        if result.last().is_none_or(|&last| !same_point(last, point)) {
            result.push(point);
        }
    }
    result
}

pub(crate) fn point_on_segment(point: Point, a: Point, b: Point) -> bool {
    within_bounds(point, a, b) && orient2d(a, b, point) == 0
}

fn within_bounds(point: Point, a: Point, b: Point) -> bool {
    point[0] >= a[0].min(b[0])
        && point[0] <= a[0].max(b[0])
        && point[1] >= a[1].min(b[1])
        && point[1] <= a[1].max(b[1])
}

pub(crate) fn strictly_between(point: Point, a: Point, b: Point) -> bool {
    point_on_segment(point, a, b) && !same_point(point, a) && !same_point(point, b)
}

/// Inclusive segment intersection. Collinear overlaps and touches count.
pub(crate) fn segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool {
    let ab_c = orient2d(a, b, c);
    let ab_d = orient2d(a, b, d);
    let cd_a = orient2d(c, d, a);
    let cd_b = orient2d(c, d, b);
    if ab_c == 0 && within_bounds(c, a, b)
        || ab_d == 0 && within_bounds(d, a, b)
        || cd_a == 0 && within_bounds(a, c, d)
        || cd_b == 0 && within_bounds(b, c, d)
    {
        return true;
    }
    ab_c != 0 && ab_d != 0 && cd_a != 0 && cd_b != 0 && ab_c == -ab_d && cd_a == -cd_b
}

/// Whether two edges sharing at most an endpoint have any additional contact.
pub(crate) fn invalid_shared_edge_contact(a: Point, b: Point, c: Point, d: Point) -> bool {
    if !segments_intersect(a, b, c, d) {
        return false;
    }
    let shared = same_point(a, c) || same_point(a, d) || same_point(b, c) || same_point(b, d);
    if !shared {
        return true;
    }
    strictly_between(a, c, d)
        || strictly_between(b, c, d)
        || strictly_between(c, a, b)
        || strictly_between(d, a, b)
}

pub(crate) fn ring_location(point: Point, ring: &[Point]) -> RingLocation {
    if ring.len() < 3 {
        return RingLocation::Outside;
    }
    let mut winding = 0i32;
    for i in 0..ring.len() {
        let a = ring[i];
        let b = ring[(i + 1) % ring.len()];
        if point_on_segment(point, a, b) {
            return RingLocation::Boundary;
        }
        if a[1] <= point[1] {
            if b[1] > point[1] && orient2d(a, b, point) > 0 {
                winding += 1;
            }
        } else if b[1] <= point[1] && orient2d(a, b, point) < 0 {
            winding -= 1;
        }
    }
    if winding == 0 {
        RingLocation::Outside
    } else {
        RingLocation::Inside
    }
}

pub(crate) fn ring_area_sign(ring: &[Point]) -> i8 {
    exact::ring_area(ring)
}

/// Approximate area for reporting and coverage comparisons, never for topology.
pub(crate) fn ring_area(ring: &[Point]) -> f64 {
    let Some(&origin) = ring.first() else {
        return 0.0;
    };
    let mut sum = 0.0;
    let mut correction = 0.0;
    for i in 0..ring.len() {
        let a = ring[i];
        let b = ring[(i + 1) % ring.len()];
        let term =
            (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0]);
        let adjusted = term - correction;
        let next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    }
    0.5 * sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_contact_is_exact() {
        assert!(segments_intersect(
            [0.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [2.0, 0.0]
        ));
        assert!(point_on_segment([0.2, 0.2], [0.1, 0.1], [0.3, 0.3]));
        assert!(!point_on_segment([0.4, 0.4], [0.1, 0.1], [0.3, 0.3]));
        assert!(invalid_shared_edge_contact(
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0]
        ));
    }

    #[test]
    fn winding_distinguishes_boundary_and_orientation_does_not_matter() {
        let mut ring = vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]];
        assert_eq!(ring_location([1.0, 1.0], &ring), RingLocation::Inside);
        assert_eq!(ring_location([2.0, 1.0], &ring), RingLocation::Boundary);
        assert_eq!(ring_location([3.0, 1.0], &ring), RingLocation::Outside);
        ring.reverse();
        assert_eq!(ring_location([1.0, 1.0], &ring), RingLocation::Inside);
    }
}
