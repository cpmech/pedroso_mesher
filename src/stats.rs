//! Public mesh statistics: the distributions behind the gate.
//!
//! `verify::gate` reports minima (min angle, min scaled Jacobian). That is the
//! acceptance bar, not a description of a mesh: a mesh can pass with 58 %
//! valence-4 and one 1:191 616 sliver. [`MeshStatistics`] is the data structure
//! with the full picture (distributions, valence histogram, conformity), it is
//! serialisable, and [`MeshStatistics::to_json`] writes it for logging or
//! post-processing.

use serde::{Deserialize, Serialize};

use crate::mesh::{Kind, Mesh};
use crate::quality;
use crate::two_d::Domain;
use crate::verify::{self, Gate};

/// Min / mean / median / p95 / max of a set of values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Distribution {
    pub min: f64,
    pub mean: f64,
    pub median: f64,
    pub p95: f64,
    pub max: f64,
}

impl Distribution {
    /// Build a distribution, ignoring non-finite values.
    pub fn of(values: &[f64]) -> Self {
        let mut v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
        if v.is_empty() {
            return Self::default();
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = v.len();
        Self {
            min: v[0],
            mean: v.iter().sum::<f64>() / n as f64,
            median: v[n / 2],
            p95: v[(n * 95) / 100],
            max: v[n - 1],
        }
    }
}

/// Everything measurable about one mesh, as a public data structure.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeshStatistics {
    pub quads: usize,
    pub triangles: usize,
    pub inverted: usize,
    pub overlapping: usize,
    pub cracks: usize,
    /// Minimum over all quads of the corner angle, and its distribution.
    pub min_corner_angle_deg: f64,
    pub angle: Distribution,
    /// Minimum over all quads of the scaled Jacobian, and its distribution.
    pub min_scaled_jacobian: f64,
    pub scaled_jacobian: Distribution,
    pub aspect_ratio: Distribution,
    /// Histogram of the valence of interior nodes; the last bucket is `8+`.
    pub valence: Vec<usize>,
    /// Share (0..1) of interior nodes with exactly four incident quads.
    pub interior_valence4_share: f64,
    /// Largest distance of a boundary node from the domain boundary.
    pub boundary_offset: f64,
    /// Largest distance from a domain sharp corner to the nearest boundary node.
    pub corner_offset: f64,
    /// The gate verdict on this mesh.
    pub passes: bool,
    /// Boundary nodes on the polyline and sharp corners kept.
    pub conforming: bool,
    /// The gate itself (minima, counts).
    pub gate: Gate,
}

impl MeshStatistics {
    /// Measure a mesh against a domain.
    pub fn of(domain: &Domain, mesh: &Mesh) -> Self {
        let gate = verify::gate(mesh);
        let mut angles = Vec::new();
        let mut sjs = Vec::new();
        let mut aspects = Vec::new();
        for cell in &mesh.cells {
            if cell.kind != Kind::Quad4 {
                continue;
            }
            angles.push(
                quality::quad_angles(cell, &mesh.points)
                    .into_iter()
                    .fold(f64::INFINITY, f64::min),
            );
            sjs.push(
                quality::scaled_jacobian(cell, &mesh.points)
                    .into_iter()
                    .fold(f64::INFINITY, f64::min),
            );
            aspects.push(quality::aspect_ratio(cell, &mesh.points));
        }

        // Boundary nodes: an edge used by exactly one cell.
        let mut edge_count: std::collections::HashMap<(usize, usize), usize> =
            std::collections::HashMap::new();
        for cell in &mesh.cells {
            for i in 0..cell.nodes.len() {
                let a = cell.nodes[i];
                let b = cell.nodes[(i + 1) % cell.nodes.len()];
                *edge_count.entry((a.min(b), a.max(b))).or_default() += 1;
            }
        }
        let mut on_boundary = vec![false; mesh.points.len()];
        for ((a, b), n) in edge_count {
            if n == 1 {
                if a < on_boundary.len() {
                    on_boundary[a] = true;
                }
                if b < on_boundary.len() {
                    on_boundary[b] = true;
                }
            }
        }
        let mut valence = vec![0usize; mesh.points.len()];
        for cell in &mesh.cells {
            for &v in &cell.nodes {
                if v < valence.len() {
                    valence[v] += 1;
                }
            }
        }
        let mut histogram = vec![0usize; 9];
        let mut interior = 0usize;
        let mut four = 0usize;
        for (v, &k) in valence.iter().enumerate() {
            if on_boundary[v] || k == 0 {
                continue;
            }
            interior += 1;
            if k == 4 {
                four += 1;
            }
            histogram[k.min(8)] += 1;
        }

        let boundary_offset = verify::boundary_offset(domain, mesh);
        let corner_offset = verify::corner_offset(domain, mesh);
        let diagonal = mesh
            .points
            .iter()
            .fold(1.0_f64, |acc, p| acc.max(p[0].abs()).max(p[1].abs()));
        let tolerance = 1.0e-6 * diagonal;
        Self {
            quads: gate.quads,
            triangles: gate.triangles,
            inverted: gate.inverted,
            overlapping: gate.overlapping,
            cracks: gate.cracks,
            min_corner_angle_deg: gate.min_corner_angle_deg,
            angle: Distribution::of(&angles),
            min_scaled_jacobian: gate.min_scaled_jacobian,
            scaled_jacobian: Distribution::of(&sjs),
            aspect_ratio: Distribution::of(&aspects),
            valence: histogram,
            interior_valence4_share: four as f64 / interior.max(1) as f64,
            boundary_offset,
            corner_offset,
            passes: gate.passes(),
            conforming: boundary_offset <= tolerance && corner_offset <= tolerance,
            gate,
        }
    }

    /// The floor of the gate margin: `min(angle/30°, SJ/0.2)`.
    pub fn margin(&self) -> f64 {
        (self.min_corner_angle_deg / 30.0).min(self.min_scaled_jacobian / 0.2)
    }

    /// One line, for tables.
    pub fn line(&self) -> String {
        format!(
            "q={} tri={} inv={} ovl={} sj={:.3} ang={:.1} v4={:.1}% ar95={:.2} armax={:.2} {}",
            self.quads,
            self.triangles,
            self.inverted,
            self.overlapping,
            self.min_scaled_jacobian,
            self.min_corner_angle_deg,
            100.0 * self.interior_valence4_share,
            self.aspect_ratio.p95,
            self.aspect_ratio.max,
            if self.passes { "PASS" } else { "FAIL" }
        )
    }

    /// Serialise to pretty JSON (the user-facing form of the statistics).
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Parse the JSON form back.
    pub fn from_json(text: &str) -> Result<Self, String> {
        serde_json::from_str(text).map_err(|e| e.to_string())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::mesh::Cell;

    fn square() -> (Domain, Mesh) {
        let domain = Domain::rectangle(1.0, 1.0).unwrap();
        let mesh = Mesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            vec![Cell::quad4([0, 1, 2, 3])],
        );
        (domain, mesh)
    }

    #[test]
    fn a_square_has_ninety_degree_angles_and_unit_aspect() {
        let (d, m) = square();
        let s = MeshStatistics::of(&d, &m);
        assert_eq!(s.quads, 1);
        assert_eq!(s.triangles, 0);
        assert!((s.angle.mean - 90.0).abs() < 1.0e-9);
        assert!((s.aspect_ratio.max - 1.0).abs() < 1.0e-12);
        assert!(s.passes);
        assert!(s.conforming);
        // one quad, four boundary nodes, no interior node
        assert_eq!(s.valence[4], 0);
        assert_eq!(s.interior_valence4_share, 0.0);
    }

    #[test]
    fn the_distribution_reports_min_median_p95_max() {
        let d = Distribution::of(&[4.0, 1.0, 3.0, 2.0, 5.0]);
        assert_eq!(d.min, 1.0);
        assert_eq!(d.max, 5.0);
        assert_eq!(d.median, 3.0);
        assert!((d.mean - 3.0).abs() < 1.0e-12);
    }

    #[test]
    fn statistics_round_trip_through_json() {
        let (d, m) = square();
        let s = MeshStatistics::of(&d, &m);
        let back = MeshStatistics::from_json(&s.to_json()).unwrap();
        assert_eq!(back.quads, s.quads);
        assert_eq!(back.passes, s.passes);
        assert!((back.margin() - s.margin()).abs() < 1.0e-12);
    }
}
