//! The benchmark corpus: the parts every phase is measured on.
//!
//! A part is a *specification*, not a mesh: the mesher does not exist yet, and
//! the point of fixing the corpus now is that P1 and P4 have something concrete
//! to fail against. Each entry states the geometry, the size range it must be
//! meshed at, and the quality it must reach.
//!
//! The eight 3D parts are the ones the D1 acceptance gate is measured on (min
//! scaled Jacobian ≥ 0.1 on six of eight). The four 2D parts gate P1.

/// One benchmark part.
#[derive(Clone, Copy, Debug)]
pub struct Part {
    /// Short id, used in reports and file names.
    pub id: &'static str,
    /// What the part is.
    pub name: &'static str,
    /// Spatial dimension: 2 or 3.
    pub dim: u8,
    /// Geometry, in millimetres, and the size range to mesh at.
    pub parameters: &'static str,
    /// What the part is there to stress.
    pub requirement: &'static str,
    /// Minimum scaled Jacobian this part must reach.
    pub min_scaled_jacobian: f64,
}

/// Every part, 2D first. Twelve entries: four 2D and eight 3D.
pub const PARTS: &[Part] = &[
    Part {
        id: "plate-hole",
        name: "Plate with a central hole",
        dim: 2,
        parameters: "100 × 60 mm plate, Ø20 hole centred, size 2–8 mm",
        requirement: "all-quad, feature captured at the hole, 45° minimum angle",
        min_scaled_jacobian: 0.1,
    },
    Part {
        id: "l-bracket-2d",
        name: "L-bracket section",
        dim: 2,
        parameters: "80 × 60 mm, 20 mm legs, 8 mm fillet, size 1–4 mm",
        requirement: "concave corner resolved without inverted quads",
        min_scaled_jacobian: 0.1,
    },
    Part {
        id: "ring-section",
        name: "Quarter of an annulus",
        dim: 2,
        parameters: "outer R20, inner R10, 90°, size 1–3 mm",
        requirement: "boundary quads follow both arcs",
        min_scaled_jacobian: 0.2,
    },
    Part {
        id: "stepped-section-2d",
        name: "Stepped section with a notch",
        dim: 2,
        parameters: "60 × 40 mm with a 10 × 30 mm notch, size 1–5 mm",
        requirement: "re-entrant corners, two sharp angles",
        min_scaled_jacobian: 0.1,
    },
    Part {
        id: "cube",
        name: "Unit cube",
        dim: 3,
        parameters: "10 × 10 × 10 mm, size 1–5 mm",
        requirement: "the sanity part: everything must be ≈ 1.0",
        min_scaled_jacobian: 0.2,
    },
    Part {
        id: "hollow-cylinder",
        name: "Hollow cylinder",
        dim: 3,
        parameters: "outer R20, inner R12, height 40 mm, size 2–6 mm",
        requirement: "curved boundary layer, closed annulus",
        min_scaled_jacobian: 0.1,
    },
    Part {
        id: "filleted-block",
        name: "Block with a filleted edge",
        dim: 3,
        parameters: "60 × 40 × 20 mm, fillet 6 mm, size 2–5 mm",
        requirement: "fillet captured, no inverted buffer cells",
        min_scaled_jacobian: 0.1,
    },
    Part {
        id: "thin-wall",
        name: "Thin wall",
        dim: 3,
        parameters: "50 × 50 × 3 mm, size 1–2 mm",
        requirement: "at least two elements through the thickness",
        min_scaled_jacobian: 0.1,
    },
    Part {
        id: "stepped-shaft",
        name: "Stepped shaft",
        dim: 3,
        parameters: "Ø30 → Ø20, 60 mm long, size 2–5 mm",
        requirement: "circular steps and a sharp shoulder",
        min_scaled_jacobian: 0.1,
    },
    Part {
        id: "bracket-rib",
        name: "Bracket with a rib",
        dim: 3,
        parameters: "80 × 60 × 40 mm plate, 6 mm rib, size 2–5 mm",
        requirement: "rib root, two concave corners",
        min_scaled_jacobian: 0.1,
    },
    Part {
        id: "torus-section",
        name: "Quarter torus",
        dim: 3,
        parameters: "R30, r10, 90° sweep, size 2–5 mm",
        requirement: "doubly curved boundary, no twisted cells",
        min_scaled_jacobian: 0.1,
    },
    Part {
        id: "sliver-feature",
        name: "Block with a 0.5 mm slot",
        dim: 3,
        parameters: "40 × 40 × 20 mm, slot 0.5 mm wide, size 0.5–2 mm",
        requirement: "the deliberately awkward one: narrow region",
        min_scaled_jacobian: 0.05,
    },
];

/// The 2D parts (P1 gate).
pub fn parts_2d() -> impl Iterator<Item = &'static Part> {
    PARTS.iter().filter(|p| p.dim == 2)
}

/// The 3D parts (P4 acceptance, six of eight at 0.1).
pub fn parts_3d() -> impl Iterator<Item = &'static Part> {
    PARTS.iter().filter(|p| p.dim == 3)
}

/// The part with this id, if it exists.
pub fn part(id: &str) -> Option<&'static Part> {
    PARTS.iter().find(|p| p.id == id)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    #[test]
    fn the_corpus_has_twelve_parts() {
        assert_eq!(PARTS.len(), 12);
    }

    #[test]
    fn the_split_is_four_2d_and_eight_3d() {
        assert_eq!(parts_2d().count(), 4);
        assert_eq!(parts_3d().count(), 8);
    }

    #[test]
    fn every_id_is_unique_and_findable() {
        for p in PARTS {
            assert!(part(p.id).is_some(), "{}", p.id);
        }
        let mut ids: Vec<&str> = PARTS.iter().map(|p| p.id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), PARTS.len());
    }

    #[test]
    fn every_part_states_a_requirement_and_a_positive_bar() {
        for part in PARTS {
            assert!(!part.requirement.is_empty(), "{}", part.id);
            assert!(!part.parameters.is_empty(), "{}", part.id);
            assert!(part.min_scaled_jacobian > 0.0, "{}", part.id);
        }
    }

    #[test]
    fn the_sliver_is_the_awkward_one() {
        let sliver = part("sliver-feature").expect("sliver-feature exists");
        let strictest = parts_3d()
            .map(|p| p.min_scaled_jacobian)
            .fold(1.0, f64::min);
        assert_eq!(sliver.min_scaled_jacobian, strictest);
    }
}
