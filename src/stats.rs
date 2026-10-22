//! Public mesh statistics: the full picture behind the gate.
//!
//! [`crate::verify::gate`] reports minima (min corner angle, min scaled
//! Jacobian) and the topology counts. That is the acceptance bar, not a
//! description of a mesh: a mesh can pass with 58 % valence-4 and one 1:191 616
//! sliver (measured on the ring). [`MeshStatistics`] is the public data
//! structure with the whole picture (angle, scaled-Jacobian and aspect-ratio
//! distributions, valence histogram, conformity offsets, the gate itself). It
//! is `serde`-serialisable, and [`MeshStatistics::to_json`] writes it for
//! logging or post-processing.
//!
//! **Metrics.** The corner scaled Jacobian and the edge-length aspect ratio are
//! the surface metrics of knupp-2000-part1 (Knupp, *Mesh quality via Jacobian
//! matrix norm, Part I: surface*): `SJ = det(J) / (|J_col0| |J_col1|)` at each
//! corner (dimensionless; 1 for a square, <= 0 for an inverted or degenerate
//! cell) and `aspect = longest edge / shortest edge` (dimensionless; 1 for a
//! square). Corner angles are plain Euclidean angles in degrees.
//!
//! The gate thresholds (`SJ >= 0.2`, `angle >= 30 deg`, all-quad, watertight,
//! no inverted or overlapping cells) are this project's **contract**, not a
//! paper constant; they are defined and enforced in [`crate::verify`]. This
//! module only measures: statistics never feed back into generation.
//!
//! **What is approximated.** Valence is counted as incident *cells*, so on a
//! quad-dominant mesh a node touching a triangle gets one count that is not a
//! quad; the valence-4 share is exact only on an all-quad mesh. Boundary nodes
//! are found topologically (an edge used by exactly one cell), which is what
//! the conformity checks want, but it cannot see a mesh whose boundary edge
//! degree is wrong — that is what `cracks` counts. The conformity tolerance is
//! `1e-6 * max |coordinate|` (a box-extent scale, not the true diagonal).

use serde::{Deserialize, Serialize};

use crate::mesh::{Kind, Mesh};
use crate::quality;
use crate::two_d::Domain;
use crate::verify::{self, Gate};

/// Min / mean / median / p95 / max of a set of values.
///
/// Units follow the measured quantity: degrees for angles, dimensionless for
/// the scaled Jacobian and the aspect ratio. Percentiles use the nearest-rank
/// convention on the ascending sorted sample (`v[(n * 95) / 100]`), and the
/// median of an even sample is the upper of the two middle values; both are
/// deterministic and need no interpolation. All fields are finite by
/// construction (see [`Distribution::of`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Distribution {
    /// Smallest finite value in the sample, in the sample's own unit; the value
    /// the gate floor is compared with.
    pub min: f64,

    /// Arithmetic mean over the finite values, in the sample's unit; hides a
    /// single bad cell, which is why `min` and `p95` are reported next to it.
    pub mean: f64,

    /// Middle value (upper middle for an even count), in the sample's unit; a
    /// robust "typical" cell.
    pub median: f64,

    /// 95th percentile by nearest rank, in the sample's unit; shows how heavy
    /// the bad tail is without being dominated by one element the way `max` is.
    pub p95: f64,

    /// Largest finite value, in the sample's unit; the single worst cell. Tail
    /// guards are expressed on this field (the smoother uses it for the aspect
    /// ratio).
    pub max: f64,
}

impl Distribution {
    /// Build a distribution from a sample, ignoring non-finite values.
    ///
    /// Non-finite inputs (NaN from a degenerate corner, infinity from a
    /// zero-length edge) are dropped rather than poisoning the whole
    /// distribution; an empty or all-non-finite sample returns all zeros
    /// (`Self::default`), which is a sentinel, not a valid measurement. The
    /// sample is copied and sorted, so the caller's order is not changed.
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
///
/// This is the object the gate and the post-processing examples carry around:
/// it is `Clone`, `Debug` and `serde`, and [`Self::to_json`] is the
/// user-facing form. Lengths are in the domain's length unit; all ratios and
/// shares are dimensionless.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MeshStatistics {
    /// Number of `Quad4` cells — the target cell type of the gate.
    pub quads: usize,

    /// Number of `Tri3` cells; a mesh that passes this project's gate has 0.
    pub triangles: usize,

    /// Cells with non-positive orientation: a quad with any corner scaled
    /// Jacobian <= 0, or a triangle with signed area <= 0.
    pub inverted: usize,

    /// Pairs of cells sharing more than one edge. Such cells overlap; the
    /// Jacobian and boundary-degree checks cannot see it.
    pub overlapping: usize,

    /// Boundary vertices whose number of boundary edges is not 2 (an open
    /// chain, a T-junction or a non-manifold fan): the watertightness hole in
    /// the topology.
    pub cracks: usize,

    /// Minimum over all quads of the smallest corner angle, in degrees
    /// (0..180). The gate floor is 30 deg.
    pub min_corner_angle_deg: f64,

    /// Distribution of the per-quad minimum corner angle, in degrees. `min`
    /// repeats [`Self::min_corner_angle_deg`]; the mean and p95 say whether a
    /// single bad quad or the whole mesh is responsible.
    pub angle: Distribution,

    /// Minimum over all quads of the corner scaled Jacobian (dimensionless).
    /// A square corner is 1.0; the gate floor is 0.2; <= 0 is inverted.
    pub min_scaled_jacobian: f64,

    /// Distribution of the per-quad minimum corner scaled Jacobian
    /// (dimensionless); `.min` repeats [`Self::min_scaled_jacobian`].
    pub scaled_jacobian: Distribution,

    /// Distribution of the per-quad edge-length aspect ratio
    /// (`longest / shortest`, dimensionless; 1.0 for a square). There is no
    /// gate floor: the smoother uses this distribution as its tail guard,
    /// because the gate minima did not see aspect 1094 (measured; see
    /// [`crate::smooth`]).
    pub aspect_ratio: Distribution,

    /// Histogram of the valence of interior nodes: `valence[k]` counts the
    /// interior nodes incident to exactly `k` cells for `k = 1..=7`, and
    /// `valence[8]` counts every interior node of valence 8 or more.
    /// `valence[0]` is always 0 (a node with no cell is not a mesh node).
    /// Valence is counted in incident cells, so it is the quad valence only on
    /// an all-quad mesh. Why it matters: valence 4 is the regular interior quad
    /// node; the other values are the singularities whose reduction is the
    /// subject of verma-2017.
    pub valence: Vec<usize>,

    /// Share (0..1) of interior nodes with exactly four incident cells — the
    /// regular fraction of the interior. 1.0 is a fully regular interior;
    /// measured: 58 % on the ring, and 43.6 -> 49.3 % after the singularity
    /// swaps. 0.0 on a mesh with no interior node.
    pub interior_valence4_share: f64,

    /// Largest distance (length unit) of a mesh boundary node from the domain
    /// boundary. A conforming planar mesh gives ~1e-15; a larger value means
    /// the mesh no longer sits on the geometry it was built for, which the
    /// quality minima cannot detect (moving a boundary node inward can even
    /// improve them). Measured on the ring: a perturbation pass put boundary
    /// nodes up to 1.92 x size off the domain.
    pub boundary_offset: f64,

    /// Largest distance (length unit) from a domain sharp corner to the nearest
    /// boundary node. Catches a corner that lost its node: the node can slide
    /// along an adjacent edge, so `boundary_offset` stays 0 while the corner is
    /// cut off (measured: a corner node moved 1.08 along an edge).
    pub corner_offset: f64,

    /// The gate verdict: `true` when every criterion of
    /// [`crate::verify::Gate::passes`] holds.
    pub passes: bool,

    /// Whether the mesh is conforming to the geometry: every boundary node on
    /// the polyline and every sharp corner kept, both within
    /// `1e-6 * max |coordinate|`. `true` implies nothing about quality.
    pub conforming: bool,

    /// The gate itself (minima and counts), kept so a caller can format the
    /// failing reason with [`crate::verify::Gate::reason`] without recomputing
    /// it.
    pub gate: Gate,
}

impl MeshStatistics {
    /// Measure a mesh against a domain.
    ///
    /// One pass over the cells collects the per-quad minimum angle, minimum
    /// scaled Jacobian and aspect ratio; [`verify::gate`] supplies the counts
    /// and minima. Boundary and conformity work needs a second pass over the
    /// edges and one over the nodes. The domain is used only for
    /// `boundary_offset` and `corner_offset`; the mesh is not modified.
    pub fn of(domain: &Domain, mesh: &Mesh) -> Self {
        let gate = verify::gate(mesh);
        let mut angles = Vec::new();
        let mut sjs = Vec::new();
        let mut aspects = Vec::new();
        // Only Quad4 cells enter the distributions: a Tri3 has no corner
        // Jacobian (quality returns 1.0 by convention), so mixing it in would
        // bias SJ and aspect towards a value that was never measured.
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

        // Boundary nodes: an edge used by exactly one cell. This is the
        // topological definition the conformity checks need; a mesh with a
        // wrong boundary edge degree shows up in `gate.cracks`, not here.
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
        // Valence is the number of incident cells, not edges; for a quad that
        // is the same count, for a triangle it is one per triangle.
        let mut valence = vec![0usize; mesh.points.len()];
        for cell in &mesh.cells {
            for &v in &cell.nodes {
                if v < valence.len() {
                    valence[v] += 1;
                }
            }
        }
        // Nine buckets, `0..=8`, so the histogram has a fixed shape in the
        // JSON; the last bucket absorbs every valence of 8 or more.
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
        // Conformity tolerance: 1e-6 of the largest absolute coordinate. This
        // is a box-extent scale, not the geometric diagonal; see the module
        // note. The offsets are floating-point ~1e-15 on a truly conforming
        // mesh, so the tolerance only has to absorb round-off.
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

    /// The floor of the gate margin: `min(angle / 30 deg, SJ / 0.2)`.
    ///
    /// Dimensionless and normalised to the gate: 1.0 is exactly on the bar,
    /// below 1.0 means at least one criterion fails, above 1.0 is margin. This
    /// is the comparison the smoother uses to decide whether a mesh improved.
    pub fn margin(&self) -> f64 {
        (self.min_corner_angle_deg / 30.0).min(self.min_scaled_jacobian / 0.2)
    }

    /// One line, for tables and logs.
    ///
    /// Abbreviations: `q` quads, `tri` triangles, `inv` inverted, `ovl`
    /// overlapping, `sj` min scaled Jacobian, `ang` min corner angle (deg),
    /// `v4` interior valence-4 share (%), `ar95` / `armax` aspect p95 / max.
    /// The last token is `PASS` or `FAIL`; no other token contains a space.
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
    ///
    /// Field names are the struct field names; each distribution keeps
    /// `min/mean/median/p95/max`. Serialisation of this all-finite data cannot
    /// fail in practice; the `{}` fallback keeps the function panic-free.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|_| "{}".to_string())
    }

    /// Parse the JSON form back.
    ///
    /// Used by tests and post-processing; a malformed or older document is an
    /// error string, never a panic. The result is only as trustworthy as the
    /// document: nothing re-measures the mesh.
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
