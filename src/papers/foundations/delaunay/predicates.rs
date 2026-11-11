//! Filtered orientation and in-circle predicates.
//!
//! Inputs must be finite. The domain and triangulation layers enforce that
//! condition before calling these functions. Positive in-circle signs assume a
//! counter-clockwise triangle `(a, b, c)`.

use super::exact;

/// Sign of twice the signed area of `(a, b, c)`: 1, 0 or -1.
pub fn orient2d(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> i8 {
    let left = (a[0] - c[0]) * (b[1] - c[1]);
    let right = (a[1] - c[1]) * (b[0] - c[0]);
    let estimate = left - right;
    let magnitude = left.abs() + right.abs();
    let uncertainty = 8.0 * f64::EPSILON * magnitude;
    if uncertainty.is_normal() {
        if estimate > uncertainty {
            return 1;
        }
        if estimate < -uncertainty {
            return -1;
        }
    }
    exact::orientation(a, b, c)
}

/// In-circle determinant sign for counter-clockwise `(a, b, c)` and query `d`.
///
/// Returns 1 inside the circumcircle, 0 on it and -1 outside.
pub fn incircle(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> i8 {
    let adx = a[0] - d[0];
    let ady = a[1] - d[1];
    let bdx = b[0] - d[0];
    let bdy = b[1] - d[1];
    let cdx = c[0] - d[0];
    let cdy = c[1] - d[1];
    let bc = bdx * cdy - cdx * bdy;
    let ca = cdx * ady - adx * cdy;
    let ab = adx * bdy - bdx * ady;
    let aa = adx * adx + ady * ady;
    let bb = bdx * bdx + bdy * bdy;
    let cc = cdx * cdx + cdy * cdy;
    let estimate = aa * bc + bb * ca + cc * ab;
    let magnitude = (bdx * cdy).abs() * aa
        + (cdx * bdy).abs() * aa
        + (cdx * ady).abs() * bb
        + (adx * cdy).abs() * bb
        + (adx * bdy).abs() * cc
        + (bdx * ady).abs() * cc;
    let uncertainty = 64.0 * f64::EPSILON * magnitude;
    if uncertainty.is_normal() {
        if estimate > uncertainty {
            return 1;
        }
        if estimate < -uncertainty {
            return -1;
        }
    }
    exact::in_circle(a, b, c, d)
}

/// Whether `p` lies strictly inside the diametral circle of segment `(a, b)`.
pub fn encroaches(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> bool {
    let ax = a[0] - p[0];
    let ay = a[1] - p[1];
    let bx = b[0] - p[0];
    let by = b[1] - p[1];
    let left = ax * bx;
    let right = ay * by;
    let estimate = left + right;
    let uncertainty = 8.0 * f64::EPSILON * (left.abs() + right.abs());
    if uncertainty.is_normal() && estimate < -uncertainty {
        return true;
    }
    if uncertainty.is_normal() && estimate > uncertainty {
        return false;
    }
    exact::encroachment_sign(a, b, p) < 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orientation_classifies_exact_and_near_degenerate_cases() {
        assert_eq!(orient2d([0.0, 0.0], [1.0, 0.0], [0.0, 1.0]), 1);
        assert_eq!(orient2d([0.0, 0.0], [0.0, 1.0], [1.0, 0.0]), -1);
        assert_eq!(orient2d([0.1, 0.1], [0.2, 0.2], [0.3, 0.3]), 0);
        assert_eq!(
            orient2d(
                [1.0e300, 1.0e300],
                [1.0e300 + 1.0e290, 1.0e300],
                [1.0e300, 1.0e300 + 1.0e290]
            ),
            1
        );
    }

    #[test]
    fn incircle_handles_cocircular_tiny_and_huge_inputs() {
        let a = [0.0, 0.0];
        let b = [1.0, 0.0];
        let c = [0.0, 1.0];
        assert_eq!(incircle(a, b, c, [0.25, 0.25]), 1);
        assert_eq!(incircle(a, b, c, [1.0, 1.0]), 0);
        assert_eq!(incircle(a, b, c, [2.0, 2.0]), -1);

        let tiny = [1.0e-200, 0.0];
        assert_eq!(
            incircle(tiny, [0.0, 1.0e-200], [-1.0e-200, 0.0], [0.0, -1.0e-200]),
            0
        );
        assert_eq!(
            incircle(
                [1.0e300, 1.0e300],
                [1.0e300 + 1.0e290, 1.0e300],
                [1.0e300, 1.0e300 + 1.0e290],
                [1.0e300 + 1.0e289, 1.0e300 + 1.0e289]
            ),
            1
        );
    }

    #[test]
    fn encroachment_uses_the_diametral_circle_without_a_tolerance() {
        assert!(encroaches([0.0, 0.0], [2.0, 0.0], [1.0, 0.5]));
        assert!(!encroaches([0.0, 0.0], [2.0, 0.0], [1.0, 1.0]));
        assert!(!encroaches([0.0, 0.0], [2.0, 0.0], [2.0, 0.0]));
        assert!(encroaches(
            [1.0e200, 1.0e200],
            [1.0e200 + 1.0e190, 1.0e200],
            [1.0e200 + 5.0e189, 1.0e200 + 1.0e189]
        ));
    }

    fn random(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*state >> 11) as f64) / ((1u64 << 53) as f64)
    }

    #[test]
    fn filters_agree_with_exact_fallbacks_after_translation_and_scaling() {
        let mut state = 0x51a7_c0de_u64;
        for case in 0..1_000 {
            let scale = 10.0f64.powi((random(&mut state) * 16.0) as i32 - 8);
            let offset = if case % 2 == 0 { 0.0 } else { 1.0e8 };
            let point = |state: &mut u64| {
                [
                    offset + scale * random(state),
                    offset - scale * random(state),
                ]
            };
            let a = point(&mut state);
            let b = point(&mut state);
            let c = point(&mut state);
            let d = point(&mut state);
            assert_eq!(orient2d(a, b, c), exact::orientation(a, b, c));
            assert_eq!(incircle(a, b, c, d), exact::in_circle(a, b, c, d));
        }
    }
}
