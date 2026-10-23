//! Interior constraints for the Simons 2024 all-quad pipeline.
//!
//! Thesis reference: Simons, *Size-Driven All-Quad Meshing for Geographic Data*
//! (2024).  This module implements the **interior features** of the input PSLG
//! described in Chapter 2, §2.2.1, and the split/stitch idea used when an
//! interior path divides a region (Chapter 5, §5.1.8.2: the interior mesh is
//! stitched to the perimeter/channel).
//!
//! # What the thesis says and what this file does
//!
//! * §2.2.1: "We allow the PSLG to include interior features, which are paths
//!   that either terminate inside the region or divide the region into
//!   subregions. Both sides of any path that terminates in the interior are
//!   included in the boundary of the region ... Interior paths that split the
//!   region must also have an even number of edges."  This file validates that
//!   kind of input (a [`Features`] set of open polylines) and implements the
//!   two cases.
//! * **Through-going feature** (both endpoints on the outer boundary).  The
//!   outer loop is split by the feature into two loops (a planar subdivision of
//!   one loop into two).  The feature is *sampled once* at the element size;
//!   the sampled nodes are inserted into **both** sub-domain loops.  Each
//!   sub-domain is meshed by an existing all-quad route and the two meshes are
//!   welded into one.  Because the two boundary chains along the feature carry
//!   the same coordinates, the weld makes every feature edge an interior edge
//!   used by one quad on each side: `Mesh::topology().cracks == 0`, no
//!   T-junction and no duplicated seam.
//! * **Terminating feature** (one endpoint on the boundary, the other strictly
//!   inside).  The thesis turns both sides of the path into boundary and carves
//!   a thin channel around it.  The simplest faithful version is implemented:
//!   the stub is offset into a thin closed *slot* (two parallel polylines plus
//!   a round tip), and the slot is submitted to the mesher as a **hole**.  The
//!   feature is therefore represented by a boundary loop, exactly as the thesis
//!   wants, but the slot has a finite width instead of the thesis' zero-width
//!   slit (see [`SLOT_WIDTH_FACTOR`] and [`SLOT_GAP_FACTOR`]).
//!
//! # Faithful
//!
//! * PSLG validation of §2.2.1: each path has at least two points, lies inside
//!   the domain, has no self-intersection, and its classification is exactly
//!   "terminates inside" or "divides the region".
//! * The planar loop split for a through feature: one loop becomes two loops
//!   that share the feature polyline; the domain's holes are assigned to the
//!   sub-domain that contains them.
//! * The feature is sampled once at the requested size and the same sampled
//!   coordinates are used on both sides, which is what makes the merged mesh
//!   conforming.
//! * The terminating feature becomes a boundary (a slot hole), so its two sides
//!   are boundary edges of the output mesh.
//!
//! # Simplified or not implemented
//!
//! * **One through feature per call.**  A through feature plus a terminating
//!   feature in the same call is rejected ("not supported yet"); several
//!   through features are rejected as well.  Several **terminating** features
//!   are supported (one slot per feature) as long as they are disjoint.
//! * **A feature that touches neither boundary is rejected.**  Treating a
//!   floating polyline as a stub would invent a connection to the boundary
//!   that the input does not contain, so the module returns a clear `Err`
//!   instead (the behaviour is fixed and documented, not silent).
//! * **Route order and the seam.**  A through feature is first split with the
//!   *sampled* loops and meshed by [`crate::sized`] (Chapter 2's grid +
//!   channel).  On the square demo this route does keep the sampled seam and
//!   returns two perfect 5x10 grids.  When a route re-samples the boundary
//!   (the channel ribbon may re-discretise it to the inner grid loop) the two
//!   seam chains differ or the sampled nodes are missing; that attempt is then
//!   rejected.  Plain loops (only the feature's own vertices) are tried with
//!   every route afterwards, and the seam-preserving routes
//!   ([`crate::alt_decompose`], [`crate::alt_bipartite`],
//!   [`crate::alt_untangle`], which discretise the boundary once and keep
//!   input vertices) provide the candidates that can honour a prescribed seam
//!   at any size.  The first attempt that (a) gives the same seam chain on
//!   both sides, (b) welds to zero cracks and (c) passes
//!   [`crate::verify::gate`] is returned.  The chosen route and whether the
//!   sampled seam was kept are reported by [`demo`].
//! * **Slot geometry.**  The slot width is one element size
//!   ([`SLOT_WIDTH_FACTOR`] = 1.0) and the slot starts half a width inside the
//!   boundary ([`SLOT_GAP_FACTOR`] = 0.5).  The inset keeps the slot a genuine
//!   closed hole instead of a hole tangent to the outer loop; the price is that
//!   the last half width of the stub is not represented.  A general polyline
//!   stub is offset with averaged vertex normals (a bevel at bends), not with a
//!   full miter/round join; a self-intersecting offset is rejected.
//! * No Chapter 2 channel is built *around* the feature: the feature itself is
//!   the boundary of the sub-domains (through) or of the slot (terminating).
//!   The channel of [`crate::sized`] is still the interior route when it can
//!   close, which is where the thesis' stitching happens.
//!
//! # Public API
//!
//! ```text
//! simons_features::mesh(&domain, &features, &sizing) -> Result<Mesh, String>
//! simons_features::demo()                           -> Result<DemoNumbers, String>
//! ```
//!
//! Every mesh returned by [`mesh`] is all-quad, has `topology().cracks == 0`
//! and passes [`crate::verify::gate`]; the return value is an error otherwise.
//! Boundary nodes lie on the geometry the mesh was built for (the original
//! domain for a through feature, the domain with its slot holes for a
//! terminating feature), so the boundary offset is round-off.
//!
//! # Measured (demo, size 1)
//!
//! * 10x10 square, vertical through feature `(5,0)-(5,10)`: route `sized`,
//!   100 quads (two 5x10 halves), min SJ 1.0000, angles 90/90, cracks 0,
//!   boundary offset 0, gate PASS, 11 seam nodes (the sampled feature).
//! * 10x10 square, stub `(5,0)-(5,5)`: route `bipartite`, 106 quads, min SJ
//!   0.7071, angles 45/90, cracks 0, boundary offset 4.4e-16 (against the
//!   domain with its slot), gate PASS, slot width 1.0, start gap 0.5.
//!
//! The tests assert the two required square cases and the rejected inputs.  A
//! diagonal through feature, a bent through feature, a bent stub, two disjoint
//! stubs and a through feature beside a hole were also exercised during
//! development and all returned conforming gate-passing meshes; they are not
//! asserted here to keep the suite small.

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

use std::collections::HashMap;

use crate::mesh::{Cell, Mesh, Node};
use crate::two_d::{Domain, Sizing};
use crate::verify;

/// Slot width of a terminating feature, in units of the element size.
///
/// The thesis leaves a zero-width slit; a zero-width hole is not a polygon, so
/// the implemented slot is one element wide. This is the "slot width parameter
/// (≈ size)" of the module contract.
pub const SLOT_WIDTH_FACTOR: f64 = 1.0;

/// Distance from the boundary to the start cap of a slot, in units of the slot
/// width (i.e. half a width when the width is one size).
///
/// The inset is what keeps the slot strictly inside the domain (a genuine
/// hole): a slot whose start cap lay on the outer boundary would be tangent to
/// it and would make the domain boundary non-manifold at the contact.
pub const SLOT_GAP_FACTOR: f64 = 0.5;

/// Relative tolerance used for "on the boundary", "same point" and
/// self-intersection. It is multiplied by the domain's diagonal (at least 1).
const REL_TOL: f64 = 1.0e-9;

/// Relative tolerance for the seam chain and the weld.  The feature samples are
/// exact coordinates, so this only absorbs round-off in a route's geometry.
const SEAM_TOL: f64 = 1.0e-7;

/// Segments on the round tip of a slot.
const SLOT_TIP_SEGMENTS: usize = 8;

/// A set of interior feature polylines: rivers, material/political boundaries
/// and fault lines of §2.2.1.
///
/// The polylines are **open** (a feature that terminates inside) or run from
/// boundary to boundary (a feature that divides the region). A closed polyline
/// is not a feature; it is a hole and belongs in [`Domain::holes`].
#[derive(Clone, Debug, Default)]
pub struct Features {
    /// One open polyline per feature, as `[x, y]` millimetres.
    pub polylines: Vec<Vec<[f64; 2]>>,
}

/// How a feature interacts with the domain boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeatureKind {
    /// Both endpoints lie on the outer boundary; the feature divides the
    /// region into two sub-domains.
    Through,
    /// One endpoint lies on the outer boundary and the other strictly inside;
    /// the feature becomes a thin slot (a hole).
    Terminating,
}

impl Features {
    /// Collect polylines into a feature set (no validation; call
    /// [`Features::validate`] or [`mesh`]).
    pub fn new(polylines: Vec<Vec<[f64; 2]>>) -> Self {
        Features { polylines }
    }

    /// Validate the PSLG contract of §2.2.1 against `domain`.
    ///
    /// Checks, for every polyline: at least two points, finite coordinates, no
    /// zero-length segment, no self-intersection (within tolerance), every
    /// point inside the closed domain, exactly one of the two legal
    /// classifications (through or terminating), and no crossing of the
    /// boundary except at the declared endpoint(s).  Two features may not
    /// intersect each other.  A feature that touches neither boundary is
    /// rejected (it is neither a divider nor a stub).
    pub fn validate(&self, domain: &Domain) -> Result<(), String> {
        self.analyze(domain).map(|_| ())
    }

    /// The classification of every polyline, after the same validation as
    /// [`Features::validate`].
    pub fn kinds(&self, domain: &Domain) -> Result<Vec<FeatureKind>, String> {
        self.analyze(domain)
    }

    /// Validation + classification in one pass.
    fn analyze(&self, domain: &Domain) -> Result<Vec<FeatureKind>, String> {
        if domain.outer.len() < 3 {
            return Err("the domain outer loop has fewer than three points".to_string());
        }
        if self.polylines.is_empty() {
            return Err("no features: the feature set is empty".to_string());
        }
        let tol = tolerance(domain);
        let boundary_segments = domain.segments();

        let mut kinds = Vec::with_capacity(self.polylines.len());
        for (index, line) in self.polylines.iter().enumerate() {
            if line.len() < 2 {
                return Err(format!(
                    "feature {index} has {} point(s); a feature needs at least two",
                    line.len()
                ));
            }
            for point in line {
                if !point[0].is_finite() || !point[1].is_finite() {
                    return Err(format!("feature {index} has a non-finite coordinate"));
                }
            }
            for pair in line.windows(2) {
                if distance(pair[0], pair[1]) <= tol {
                    return Err(format!(
                        "feature {index} has a zero-length segment at {:?}",
                        pair[0]
                    ));
                }
            }
            if polyline_self_intersects(line, tol) {
                return Err(format!("feature {index} self-intersects"));
            }

            // Every vertex must be in the closed domain, and only the declared
            // endpoints may be on the boundary.
            let mut on_boundary = Vec::with_capacity(line.len());
            for point in line {
                let near = domain.distance_to_boundary(*point) <= tol;
                if !near && !domain.contains(*point) {
                    return Err(format!(
                        "feature {index} has a point outside the domain: {point:?}"
                    ));
                }
                on_boundary.push(near);
            }
            let start_on = on_boundary[0];
            let end_on = on_boundary[line.len() - 1];
            let start_outer = locate_on_loop(&domain.outer, line[0], tol).is_some();
            let end_outer = locate_on_loop(&domain.outer, line[line.len() - 1], tol).is_some();

            let kind = match (start_on, end_on) {
                (true, true) => {
                    if !start_outer || !end_outer {
                        return Err(format!(
                            "feature {index} has an endpoint on a hole; only the outer \
                             boundary may carry feature endpoints"
                        ));
                    }
                    for (point_index, point) in line.iter().enumerate() {
                        if point_index == 0 || point_index + 1 == line.len() {
                            continue;
                        }
                        if on_boundary[point_index] {
                            return Err(format!(
                                "feature {index} touches the boundary at an interior point \
                                 {point:?}; a through feature may only meet the boundary at \
                                 its two endpoints"
                            ));
                        }
                    }
                    FeatureKind::Through
                }
                (true, false) => {
                    if !start_outer {
                        return Err(format!(
                            "feature {index} starts on a hole; a terminating feature must \
                             start on the outer boundary"
                        ));
                    }
                    for point_index in 1..line.len() {
                        if on_boundary[point_index] {
                            return Err(format!(
                                "feature {index} touches the boundary at an interior point \
                                 {:?}; only the start of a terminating feature may lie \
                                 on the boundary",
                                line[point_index]
                            ));
                        }
                    }
                    FeatureKind::Terminating
                }
                _ => {
                    return Err(format!(
                        "feature {index} does not touch the boundary; a floating interior \
                         feature is rejected (only a path that divides the region or \
                         terminates on the boundary is a valid feature)"
                    ));
                }
            };

            // Boundary-crossing test.  A feature may meet the boundary only at
            // its declared endpoint(s); crossing anywhere else, or running
            // along the boundary, is an error.
            let n_segments = line.len() - 1;
            for segment_index in 0..n_segments {
                let segment = (line[segment_index], line[segment_index + 1]);
                let allowed = [
                    (segment_index == 0).then_some(line[0]),
                    (segment_index + 1 == n_segments).then_some(line[line.len() - 1]),
                ];
                for &(a, b) in &boundary_segments {
                    if !segments_intersect(segment.0, segment.1, a, b, tol) {
                        continue;
                    }
                    let endpoint_allowed = allowed.iter().flatten().any(|&p| {
                        point_segment_distance(p, segment.0, segment.1) <= tol
                            && point_segment_distance(p, a, b) <= tol
                    });
                    if !endpoint_allowed {
                        return Err(format!(
                            "feature {index} crosses the boundary away from its endpoint \
                             (segment {:?} to {:?})",
                            segment.0, segment.1
                        ));
                    }
                }
                // Also reject a run along the boundary: sample the segment and
                // require the samples (away from an allowed endpoint) to be
                // strictly inside.
                for sample in 1..=8 {
                    let t = sample as f64 / 9.0;
                    let p = [
                        segment.0[0] + t * (segment.1[0] - segment.0[0]),
                        segment.0[1] + t * (segment.1[1] - segment.0[1]),
                    ];
                    let near_allowed = allowed
                        .iter()
                        .flatten()
                        .any(|&q| distance(p, q) <= 4.0 * tol);
                    if near_allowed {
                        continue;
                    }
                    if domain.distance_to_boundary(p) <= tol {
                        return Err(format!(
                            "feature {index} runs along the boundary near {p:?}"
                        ));
                    }
                    if !domain.contains(p) {
                        return Err(format!("feature {index} leaves the domain near {p:?}"));
                    }
                }
            }
            kinds.push(kind);
        }

        // Features may not intersect each other: the slot/split construction
        // assumes disjoint paths.
        for i in 0..self.polylines.len() {
            for j in (i + 1)..self.polylines.len() {
                if polylines_intersect(&self.polylines[i], &self.polylines[j], tol) {
                    return Err(format!("features {i} and {j} intersect"));
                }
            }
        }
        Ok(kinds)
    }
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// One feature case's measured numbers, returned by [`demo`].
#[derive(Clone, Debug, Default)]
pub struct CaseNumbers {
    /// Case name.
    pub name: String,
    /// Route that produced the mesh ("sized", "quadtree", "decompose",
    /// "bipartite", "untangle").
    pub route: String,
    /// Whether the sampled seam of a through feature was preserved.
    pub sampled_seam: bool,
    /// Number of quads.
    pub quads: usize,
    /// Number of triangles (must be 0).
    pub triangles: usize,
    /// Smallest scaled Jacobian.
    pub min_scaled_jacobian: f64,
    /// Smallest corner angle, in degrees.
    pub min_angle_deg: f64,
    /// Largest corner angle, in degrees.
    pub max_angle_deg: f64,
    /// Topological cracks (`Mesh::topology().cracks`).
    pub cracks: usize,
    /// Largest distance of a boundary node from the measurement domain.
    pub boundary_offset: f64,
    /// Whether [`crate::verify::Gate::passes`] holds.
    pub gate_pass: bool,
    /// Number of nodes shared along the feature seam (through case).
    pub seam_nodes: usize,
    /// Number of sub-domains (2 for a single through feature, 1 otherwise).
    pub subdomains: usize,
    /// Canonical feature samples requested at the element size (through case).
    pub feature_samples: usize,
    /// Slot width (terminating case), in length units.
    pub slot_width: f64,
    /// Slot start inset from the boundary (terminating case).
    pub slot_gap: f64,
}

/// Numbers returned by [`demo`].
#[derive(Clone, Debug, Default)]
pub struct DemoNumbers {
    /// 10x10 square split by the vertical feature (5,0)-(5,10).
    pub through_square: CaseNumbers,
    /// 10x10 square with the stub (5,0)-(5,5) as a thin slot.
    pub stub_square: CaseNumbers,
    /// Canonical number of samples on the vertical feature at size 1.
    pub vertical_feature_samples: usize,
}

/// Mesh a planar domain with interior features.
///
/// `domain` is the outer/holes geometry; `features` are the interior paths of
/// §2.2.1. Both are validated. A through feature is handled by splitting the
/// outer loop and welding two sub-meshes; terminating features become thin
/// slot holes. The returned mesh is all-quad, conforming (`topology().cracks ==
/// 0`) and gate-passing, or an `Err` explains why no such mesh was built.
///
/// # Errors
///
/// Invalid geometry (a short polyline, a point outside, a self-intersection, a
/// floating feature, crossing features), a feature combination this module does
/// not support yet (more than one through feature, or a through feature mixed
/// with terminating ones), and the case where every tried all-quad route fails
/// to produce a conforming gate-passing mesh.
pub fn mesh(domain: &Domain, features: &Features, sizing: &Sizing) -> Result<Mesh, String> {
    let kinds = features.analyze(domain)?;
    if !sizing.size.is_finite() || sizing.size <= 0.0 {
        return Err("the element size must be positive and finite".to_string());
    }
    if domain.area() <= 0.0 {
        return Err("the domain has no positive area".to_string());
    }
    let through = kinds
        .iter()
        .filter(|kind| **kind == FeatureKind::Through)
        .count();
    let terminating: Vec<usize> = kinds
        .iter()
        .enumerate()
        .filter(|(_, kind)| **kind == FeatureKind::Terminating)
        .map(|(index, _)| index)
        .collect();
    match (through, terminating.len()) {
        (0, 0) => Err("no features: the feature set is empty".to_string()),
        (0, _) => {
            let stubs: Vec<&[[f64; 2]]> = terminating
                .iter()
                .map(|&index| features.polylines[index].as_slice())
                .collect();
            mesh_terminating(domain, &stubs, sizing).map(|output| output.mesh)
        }
        (1, 0) => {
            let index = kinds
                .iter()
                .position(|kind| *kind == FeatureKind::Through)
                .ok_or_else(|| "internal error: through feature count mismatched".to_string())?;
            mesh_through(domain, &features.polylines[index], sizing).map(|output| output.mesh)
        }
        (1, _) => Err(
            "not supported yet: a through-going feature mixed with terminating features \
             in one call"
                .to_string(),
        ),
        (_, _) => {
            Err("not supported yet: more than one through-going feature in one call".to_string())
        }
    }
}

/// Run the two demo cases and return their measured numbers.
///
/// * a 10x10 square with the vertical through feature `(5,0)-(5,10)`, size 1;
/// * a 10x10 square with the terminating feature `(5,0)-(5,5)`, size 1.
pub fn demo() -> Result<DemoNumbers, String> {
    let square = Domain::rectangle(10.0, 10.0).map_err(|error| error.to_string())?;
    let sizing = Sizing::uniform(1.0).map_err(|error| error.to_string())?;

    let through_feature = vec![[5.0, 0.0], [5.0, 10.0]];
    let through = mesh_through(&square, &through_feature, &sizing)?;
    let through_numbers = measure(
        "square / vertical through feature",
        &square,
        &through.mesh,
        through.route,
        through.sampled,
        through.seam.len(),
        through.subdomains,
        through.feature_samples,
        0.0,
        0.0,
    );

    let stub_feature = vec![[5.0, 0.0], [5.0, 5.0]];
    let stub = mesh_terminating(&square, &[stub_feature.as_slice()], &sizing)?;
    let mut holes = square.holes.clone();
    holes.push(stub.slot.clone());
    let slot_domain =
        Domain::new(square.outer.clone(), holes).map_err(|error| error.to_string())?;
    let stub_numbers = measure(
        "square / stub slot",
        &slot_domain,
        &stub.mesh,
        stub.route,
        false,
        0,
        1,
        0,
        stub.slot_width,
        stub.slot_gap,
    );

    Ok(DemoNumbers {
        through_square: through_numbers,
        stub_square: stub_numbers,
        vertical_feature_samples: sample_polyline(&through_feature, sizing.size)
            .map(|samples| samples.len())
            .unwrap_or(0),
    })
}

// ---------------------------------------------------------------------------
// Feature classification and loop splitting
// ---------------------------------------------------------------------------

/// Sample a polyline once at `size`: every segment is split into
/// `ceil(length / size)` equal pieces, so the spacing is at most `size` and
/// every input vertex is kept. This is the single sampling that both sides of a
/// through feature use.
pub fn sample_polyline(points: &[[f64; 2]], size: f64) -> Result<Vec<[f64; 2]>, String> {
    if points.len() < 2 {
        return Err("cannot sample a feature with fewer than two points".to_string());
    }
    if !size.is_finite() || size <= 0.0 {
        return Err("the sampling size must be positive and finite".to_string());
    }
    let mut out = Vec::with_capacity(points.len() + 1);
    out.push(points[0]);
    for pair in points.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let length = distance(a, b);
        if length <= 0.0 {
            return Err("cannot sample a zero-length feature segment".to_string());
        }
        let pieces = (length / size).ceil().max(1.0) as usize;
        for step in 1..=pieces {
            let t = step as f64 / pieces as f64;
            out.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
        }
    }
    Ok(out)
}

/// Build the thin closed slot loop of a terminating feature.
///
/// The stub is offset by `±width/2` with averaged vertex normals; the interior
/// end gets a semicircular tip and the boundary end a flat cap placed
/// [`SLOT_GAP_FACTOR`] widths inside the start point. The returned loop is a
/// simple closed polygon (a hole candidate); it does not yet know about the
/// domain. `width` is normally the element size; the main entry point uses
/// [`SLOT_WIDTH_FACTOR`] × `sizing.size`.
pub fn slot_loop(stub: &[[f64; 2]], width: f64) -> Result<Vec<[f64; 2]>, String> {
    if stub.len() < 2 {
        return Err("a terminating feature needs at least two points".to_string());
    }
    if !width.is_finite() || width <= 0.0 {
        return Err("the slot width must be positive and finite".to_string());
    }
    let mut points = Vec::with_capacity(stub.len());
    for point in stub {
        if !point[0].is_finite() || !point[1].is_finite() {
            return Err("the stub has a non-finite coordinate".to_string());
        }
        if let Some(last) = points.last() {
            if distance(*last, *point) <= tolerance_from_scale(scale_of_points(stub)) {
                return Err("the stub has a zero-length segment".to_string());
            }
        }
        points.push(*point);
    }

    let tol = tolerance_from_scale(scale_of_points(stub));
    let gap = SLOT_GAP_FACTOR * width;
    let half = width / 2.0;

    // Move the start into the domain so that the flat start cap is strictly
    // inside: the slot is a hole, not a notch tangent to the outer loop.
    let first_dir = unit_vector(points[0], points[1])?;
    let mut shifted = points.clone();
    shifted[0] = [
        points[0][0] + first_dir[0] * gap,
        points[0][1] + first_dir[1] * gap,
    ];
    if distance(points[0], points[1]) <= gap + tol {
        return Err("the stub is too short for a slot with this width".to_string());
    }

    // Unit tangents at the vertices and the corresponding normals.
    let mut tangents = Vec::with_capacity(shifted.len());
    for i in 0..shifted.len() {
        let tangent = if i == 0 {
            unit_vector(shifted[0], shifted[1])?
        } else if i + 1 == shifted.len() {
            unit_vector(shifted[i - 1], shifted[i])?
        } else {
            let a = unit_vector(shifted[i - 1], shifted[i])?;
            let b = unit_vector(shifted[i], shifted[i + 1])?;
            let sum = [a[0] + b[0], a[1] + b[1]];
            let length = (sum[0] * sum[0] + sum[1] * sum[1]).sqrt();
            if length <= 1.0e-9 {
                return Err("the stub folds back on itself; cannot offset it".to_string());
            }
            [sum[0] / length, sum[1] / length]
        };
        tangents.push(tangent);
    }

    let mut left = Vec::with_capacity(shifted.len());
    let mut right = Vec::with_capacity(shifted.len());
    for (point, tangent) in shifted.iter().zip(tangents.iter()) {
        let normal = [-tangent[1], tangent[0]];
        left.push([point[0] + half * normal[0], point[1] + half * normal[1]]);
        right.push([point[0] - half * normal[0], point[1] - half * normal[1]]);
    }

    let mut loop_points = Vec::new();
    for point in &left {
        loop_points.push(*point);
    }
    // Round tip: from the left offset, clockwise through the extension of the
    // stub, to the right offset.  The first point of the arc is the left end
    // already in `loop_points`.
    let last = shifted.len() - 1;
    let end_normal = [-tangents[last][1], tangents[last][0]];
    let start_angle = end_normal[1].atan2(end_normal[0]);
    for step in 1..=SLOT_TIP_SEGMENTS {
        let angle = start_angle - std::f64::consts::PI * step as f64 / SLOT_TIP_SEGMENTS as f64;
        loop_points.push([
            shifted[last][0] + half * angle.cos(),
            shifted[last][1] + half * angle.sin(),
        ]);
    }
    // Right side back to the start.  The arc ended at the right offset of the
    // tip, so start one vertex earlier.
    for i in (0..last).rev() {
        loop_points.push(right[i]);
    }

    let loop_points = clean_loop(&loop_points, tol)?;
    if loop_points.len() < 3 {
        return Err("the slot loop is degenerate".to_string());
    }
    if !loop_is_simple(&loop_points, tol) {
        return Err(
            "the slot loop self-intersects; the stub is too short or too \
                    sharply bent for this width"
                .to_string(),
        );
    }
    Ok(loop_points)
}

/// A through feature split into its two sub-domains.
struct Split {
    first: Domain,
    second: Domain,
    feature: Vec<[f64; 2]>,
}

/// Split `outer` along `feature` into two positive-area loops.
///
/// The feature's first and last points must already lie on `outer` (within
/// `tol`). The returned loops share the feature points, so a mesher that keeps
/// input boundary vertices produces the same seam on both sides.
fn split_domain(domain: &Domain, feature: &[[f64; 2]], tol: f64) -> Result<Split, String> {
    if feature.len() < 2 {
        return Err("a through feature needs at least two points".to_string());
    }
    let a_loc = locate_on_loop(&domain.outer, feature[0], tol)
        .ok_or_else(|| "the first feature endpoint is not on the outer boundary".to_string())?;
    let b_loc = locate_on_loop(&domain.outer, feature[feature.len() - 1], tol)
        .ok_or_else(|| "the last feature endpoint is not on the outer boundary".to_string())?;
    let a = point_on_loop(&domain.outer, a_loc.0, a_loc.1);
    let b = point_on_loop(&domain.outer, b_loc.0, b_loc.1);
    if distance(a, b) <= tol {
        return Err("the feature's two boundary endpoints coincide".to_string());
    }

    // Snap the endpoints onto the boundary and keep the interior vertices.
    let mut snapped = Vec::with_capacity(feature.len());
    snapped.push(a);
    for point in &feature[1..feature.len() - 1] {
        if distance(*point, a) <= tol || distance(*point, b) <= tol {
            continue;
        }
        snapped.push(*point);
    }
    snapped.push(b);

    let arc_ab = boundary_arc(&domain.outer, a_loc, b_loc, tol);
    let arc_ba = boundary_arc(&domain.outer, b_loc, a_loc, tol);

    let mut first_loop = arc_ab;
    for index in (1..snapped.len() - 1).rev() {
        push_distinct(&mut first_loop, snapped[index], tol);
    }
    let mut second_loop = arc_ba;
    for point in &snapped[1..snapped.len() - 1] {
        push_distinct(&mut second_loop, *point, tol);
    }

    let first_loop = orient_loop(first_loop, tol)?;
    let second_loop = orient_loop(second_loop, tol)?;
    if polygon_area(&first_loop).abs() <= tol * tol || polygon_area(&second_loop).abs() <= tol * tol
    {
        return Err("the feature does not cut a positive-area sub-domain".to_string());
    }

    // Assign every hole to the side that contains it.
    let mut first_holes = Vec::new();
    let mut second_holes = Vec::new();
    for hole in &domain.holes {
        if hole.len() < 3 {
            return Err("a domain hole has fewer than three points".to_string());
        }
        let probe = hole[0];
        let in_first = point_in_polygon(probe, &first_loop);
        let in_second = point_in_polygon(probe, &second_loop);
        if in_first == in_second {
            return Err(
                "a domain hole could not be assigned to one side of the through feature"
                    .to_string(),
            );
        }
        if in_first {
            first_holes.push(hole.clone());
        } else {
            second_holes.push(hole.clone());
        }
    }

    let first = Domain::new(first_loop, first_holes).map_err(|error| error.to_string())?;
    let second = Domain::new(second_loop, second_holes).map_err(|error| error.to_string())?;
    if first.area() <= tol * tol || second.area() <= tol * tol {
        return Err("the split produced a sub-domain with no positive area".to_string());
    }
    Ok(Split {
        first,
        second,
        feature: snapped,
    })
}

/// The boundary arc from `from` to `to` in the loop's stored (counter-clockwise)
/// order, with the two interpolated endpoint points.
fn boundary_arc(
    points: &[[f64; 2]],
    from: (usize, f64),
    to: (usize, f64),
    tol: f64,
) -> Vec<[f64; 2]> {
    let n = points.len();
    let mut out = Vec::new();
    let start = point_on_loop(points, from.0, from.1);
    out.push(start);
    let (to_segment, to_t) = to;
    // Same segment: the arc is the piece between the two points, no vertex in
    // between.
    if from.0 == to_segment && from.1 < to_t {
        push_distinct(&mut out, point_on_loop(points, to_segment, to_t), tol);
        return out;
    }
    let mut index = (from.0 + 1) % n;
    let mut guard = 0usize;
    loop {
        guard += 1;
        if guard > n + 1 {
            break;
        }
        push_distinct(&mut out, points[index], tol);
        if index == to_segment {
            break;
        }
        index = (index + 1) % n;
    }
    if to_t > 0.0 {
        push_distinct(&mut out, point_on_loop(points, to_segment, to_t), tol);
    }
    out
}

/// Find where a point lies on a closed loop: `(segment index, parameter)`.
///
/// A point at a vertex is reported at parameter 0 of the segment that starts
/// there, so an arc never has to special-case a vertex endpoint.
fn locate_on_loop(points: &[[f64; 2]], target: [f64; 2], tol: f64) -> Option<(usize, f64)> {
    let n = points.len();
    if n < 2 {
        return None;
    }
    for (index, point) in points.iter().enumerate() {
        if distance(*point, target) <= tol {
            return Some((index, 0.0));
        }
    }
    for index in 0..n {
        let a = points[index];
        let b = points[(index + 1) % n];
        if point_segment_distance(target, a, b) <= tol {
            let ab = [b[0] - a[0], b[1] - a[1]];
            let length2 = ab[0] * ab[0] + ab[1] * ab[1];
            if length2 <= 0.0 {
                continue;
            }
            let t = (((target[0] - a[0]) * ab[0] + (target[1] - a[1]) * ab[1]) / length2)
                .clamp(0.0, 1.0);
            return Some((index, t));
        }
    }
    None
}

/// Interpolated point at parameter `t` on loop segment `segment`.
fn point_on_loop(points: &[[f64; 2]], segment: usize, t: f64) -> [f64; 2] {
    if points.is_empty() {
        return [0.0, 0.0];
    }
    let a = points[segment % points.len()];
    let b = points[(segment + 1) % points.len()];
    [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
}

/// Remove consecutive duplicate points, drop a repeated closing point and
/// require at least three points.
fn clean_loop(points: &[[f64; 2]], tol: f64) -> Result<Vec<[f64; 2]>, String> {
    let mut out: Vec<[f64; 2]> = Vec::with_capacity(points.len());
    for &point in points {
        push_distinct(&mut out, point, tol);
    }
    while out.len() > 1 {
        let first = out[0];
        let last = out[out.len() - 1];
        if distance(first, last) <= tol {
            out.pop();
        } else {
            break;
        }
    }
    if out.len() < 3 {
        return Err("a loop has fewer than three distinct points".to_string());
    }
    Ok(out)
}

/// Make a loop counter-clockwise (positive area), rejecting a degenerate one.
fn orient_loop(mut points: Vec<[f64; 2]>, tol: f64) -> Result<Vec<[f64; 2]>, String> {
    points = clean_loop(&points, tol)?;
    let area = polygon_area(&points);
    if area.abs() <= tol * tol {
        return Err("a loop has no area".to_string());
    }
    if area < 0.0 {
        points.reverse();
    }
    Ok(points)
}

fn push_distinct(out: &mut Vec<[f64; 2]>, point: [f64; 2], tol: f64) {
    if let Some(last) = out.last() {
        if distance(*last, point) <= tol {
            return;
        }
    }
    out.push(point);
}

// ---------------------------------------------------------------------------
// Mesh generation: through features
// ---------------------------------------------------------------------------

/// One successful through-feature construction.
struct ThroughOutput {
    mesh: Mesh,
    route: &'static str,
    sampled: bool,
    seam: Vec<[f64; 2]>,
    subdomains: usize,
    feature_samples: usize,
}

/// A route that turns a domain into an all-quad mesh.
type Mesher = fn(&Domain, &Sizing) -> Result<Mesh, String>;

/// The tried routes, in order.  The first two are the ones named by the task
/// (`sized` is Chapter 2's grid + channel; `quadtree` is the requested
/// fallback); the next three discretise the boundary once and keep input
/// boundary vertices, which is what a prescribed seam needs.
const ROUTES: [(&str, Mesher); 5] = [
    ("sized", crate::sized::mesh),
    ("quadtree", crate::alt_quadtree::mesh),
    ("decompose", crate::alt_decompose::mesh),
    ("bipartite", crate::alt_bipartite::mesh),
    ("untangle", crate::alt_untangle::mesh),
];

/// The route order for a slot (a domain with one or more holes).  The same
/// sized/quadtree attempts come first; the decomposition route supports only
/// one hole and is the slowest with a thin slot, so the triangulation-based
/// routes (which keep every boundary vertex and handle several holes) come
/// before it.
const SLOT_ROUTES: [(&str, Mesher); 5] = [
    ("sized", crate::sized::mesh),
    ("quadtree", crate::alt_quadtree::mesh),
    ("bipartite", crate::alt_bipartite::mesh),
    ("untangle", crate::alt_untangle::mesh),
    ("decompose", crate::alt_decompose::mesh),
];

/// Mesh a through feature by splitting the outer loop and welding two meshes.
fn mesh_through(
    domain: &Domain,
    feature: &[[f64; 2]],
    sizing: &Sizing,
) -> Result<ThroughOutput, String> {
    let tol = tolerance(domain);
    let plain = snap_feature(domain, feature, tol)?;
    let sampled = sample_polyline(&plain, sizing.size)?;
    let mut errors: Vec<String> = Vec::new();

    // Prefer the exact sampled seam; fall back to a route-chosen seam only if
    // no seam-preserving route can build the sampled loops.
    for &(use_sampled, seam_feature) in &[(true, &sampled), (false, &plain)] {
        let split = match split_domain(domain, seam_feature, tol) {
            Ok(split) => split,
            Err(error) => {
                errors.push(format!(
                    "{} loops: {error}",
                    if use_sampled { "sampled" } else { "plain" }
                ));
                continue;
            }
        };
        for &(route, mesher) in &ROUTES {
            match try_pair(domain, &split, sizing, route, mesher, tol) {
                Ok((mesh, seam)) => {
                    let seam_tol = seam_tolerance(domain);
                    if use_sampled && !contains_points(&sampled, &seam, seam_tol) {
                        if errors.len() < 12 {
                            errors.push(format!(
                                "{route} (sampled seam): the sampled feature nodes were \
                                 not all preserved"
                            ));
                        }
                        continue;
                    }
                    return Ok(ThroughOutput {
                        mesh,
                        route,
                        sampled: use_sampled,
                        seam,
                        subdomains: 2,
                        feature_samples: sampled.len(),
                    });
                }
                Err(error) => {
                    if errors.len() < 12 {
                        errors.push(format!(
                            "{route} ({} seam): {error}",
                            if use_sampled { "sampled" } else { "plain" }
                        ));
                    }
                }
            }
        }
    }
    Err(format!(
        "no all-quad route produced a conforming through-feature mesh: {}",
        errors.join("; ")
    ))
}

/// Snap a through feature's endpoints onto the outer loop.
fn snap_feature(domain: &Domain, feature: &[[f64; 2]], tol: f64) -> Result<Vec<[f64; 2]>, String> {
    if feature.len() < 2 {
        return Err("a through feature needs at least two points".to_string());
    }
    let a_loc = locate_on_loop(&domain.outer, feature[0], tol)
        .ok_or_else(|| "the first feature endpoint is not on the outer boundary".to_string())?;
    let b_loc = locate_on_loop(&domain.outer, feature[feature.len() - 1], tol)
        .ok_or_else(|| "the last feature endpoint is not on the outer boundary".to_string())?;
    let mut out = Vec::with_capacity(feature.len());
    out.push(point_on_loop(&domain.outer, a_loc.0, a_loc.1));
    out.extend_from_slice(&feature[1..feature.len() - 1]);
    out.push(point_on_loop(&domain.outer, b_loc.0, b_loc.1));
    Ok(out)
}

/// Mesh the two sub-domains with one route and weld them.
fn try_pair(
    parent: &Domain,
    split: &Split,
    sizing: &Sizing,
    route: &'static str,
    mesher: Mesher,
    tol: f64,
) -> Result<(Mesh, Vec<[f64; 2]>), String> {
    let first =
        mesher(&split.first, sizing).map_err(|error| format!("first sub-domain: {error}"))?;
    let second =
        mesher(&split.second, sizing).map_err(|error| format!("second sub-domain: {error}"))?;
    let seam_tol = seam_tolerance(parent).max(tol);
    // The smoother never moves a boundary node, so a route whose raw boundary
    // is already off the sub-domain geometry cannot be repaired by it.
    // Checking before smoothing avoids spending the smoothing pass on it.
    for (side, sub_domain, mesh) in [
        ("first", &split.first, &first),
        ("second", &split.second, &second),
    ] {
        let offset = verify::boundary_offset(sub_domain, mesh);
        if offset > seam_tol {
            return Err(format!(
                "{route}: the {side} sub-mesh boundary is {offset:.3e} off the geometry"
            ));
        }
    }
    let first = finish_pair_side(&split.first, first, sizing);
    let second = finish_pair_side(&split.second, second, sizing);
    let first_seam = seam_chain(&first, &split.feature, seam_tol)?;
    let second_seam = seam_chain(&second, &split.feature, seam_tol)?;
    if first_seam.len() < 2 || second_seam.len() < 2 {
        return Err(format!(
            "{route}: the feature is not represented on both sides \
             ({} and {} seam nodes)",
            first_seam.len(),
            second_seam.len()
        ));
    }
    if !same_chain(&first_seam, &second_seam, seam_tol) {
        return Err(format!(
            "{route}: the two sub-meshes have different seam chains ({} vs {} nodes)",
            first_seam.len(),
            second_seam.len()
        ));
    }

    let weld_tol = seam_tol;
    let merged = weld(&first, &second, weld_tol)?;
    let gate = verify::gate(&merged);
    if !gate.passes() {
        return Err(format!(
            "{route}: the merged mesh fails the gate: {}",
            gate.reason()
                .unwrap_or_else(|| "unknown reason".to_string())
        ));
    }
    let topology = merged.topology();
    if topology.cracks != 0 || topology.non_manifold != 0 || topology.inconsistent != 0 {
        return Err(format!(
            "{route}: the merged mesh is not conforming (cracks {}, non-manifold {}, \
             inconsistent {})",
            topology.cracks, topology.non_manifold, topology.inconsistent
        ));
    }
    let offset = verify::boundary_offset(parent, &merged);
    if offset > seam_tol {
        return Err(format!(
            "{route}: the merged boundary is {offset:.3e} off the input geometry"
        ));
    }
    Ok((merged, first_seam))
}

/// If a route's raw mesh does not pass the gate, try the boundary-preserving
/// smoother and keep whichever candidate is better.  Boundary nodes are never
/// moved by the default smoother, so the seam stays valid.
fn finish_pair_side(domain: &Domain, mesh: Mesh, sizing: &Sizing) -> Mesh {
    let raw = verify::gate(&mesh);
    if raw.passes() {
        return mesh;
    }
    let outcome = crate::smooth::smooth(
        domain,
        &mesh,
        sizing.size,
        &crate::smooth::SmoothOptions::default(),
    );
    if gate_score(&verify::gate(&outcome.smoothed)) > gate_score(&raw) {
        outcome.smoothed
    } else {
        mesh
    }
}

/// A scalar score for comparing two candidate meshes: 1.0 is exactly on the
/// gate floor, larger is better, a structural failure is negative.
fn gate_score(gate: &verify::Gate) -> f64 {
    if gate.triangles > 0 || gate.inverted > 0 || gate.overlapping > 0 || gate.cracks > 0 {
        return -1.0;
    }
    (gate.min_scaled_jacobian / 0.2).min(gate.min_corner_angle_deg / 30.0)
}

/// The boundary nodes of `mesh` that lie on the feature polyline, ordered along
/// the feature from its first to its last point.
fn seam_chain(mesh: &Mesh, feature: &[[f64; 2]], tol: f64) -> Result<Vec<[f64; 2]>, String> {
    let boundary = boundary_mask(mesh);
    let mut found: Vec<(f64, [f64; 2])> = Vec::new();
    for (index, point) in mesh.points.iter().enumerate() {
        if !boundary.get(index).copied().unwrap_or(false) {
            continue;
        }
        let p = [point[0], point[1]];
        if let Some(parameter) = polyline_parameter(feature, p, tol) {
            found.push((parameter, p));
        }
    }
    found.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut chain = Vec::with_capacity(found.len());
    for (_, point) in found {
        push_distinct(&mut chain, point, tol);
    }
    if chain.len() < 2 {
        return Err("the feature seam has fewer than two boundary nodes".to_string());
    }
    Ok(chain)
}

/// Whether both boundary chains carry the same nodes in the same order.
fn same_chain(first: &[[f64; 2]], second: &[[f64; 2]], tol: f64) -> bool {
    first.len() == second.len()
        && first
            .iter()
            .zip(second.iter())
            .all(|(a, b)| distance(*a, *b) <= tol)
}

/// Whether every expected point has a match in `chain`.
fn contains_points(expected: &[[f64; 2]], chain: &[[f64; 2]], tol: f64) -> bool {
    expected.iter().all(|point| {
        chain
            .iter()
            .any(|candidate| distance(*point, *candidate) <= tol)
    })
}

/// Weld two meshes by coordinate.  Points closer than `tol` become one node;
/// cells are re-indexed and the result is structurally validated.
fn weld(first: &Mesh, second: &Mesh, tol: f64) -> Result<Mesh, String> {
    let cell = 10.0 * tol.max(1.0e-15);
    let mut points: Vec<Node> = first.points.clone();
    let mut buckets: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (index, point) in points.iter().enumerate() {
        buckets
            .entry(point_key(*point, cell))
            .or_default()
            .push(index);
    }

    let mut map = vec![0usize; second.points.len()];
    for (index, point) in second.points.iter().enumerate() {
        let key = point_key(*point, cell);
        let mut best: Option<(f64, usize)> = None;
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(candidates) = buckets.get(&(key.0 + dx, key.1 + dy)) {
                    for &candidate in candidates {
                        let other = points[candidate];
                        let d = ((other[0] - point[0]).powi(2)
                            + (other[1] - point[1]).powi(2)
                            + (other[2] - point[2]).powi(2))
                        .sqrt();
                        if d <= tol && best.is_none_or(|(best_d, _)| d < best_d) {
                            best = Some((d, candidate));
                        }
                    }
                }
            }
        }
        match best {
            Some((_, candidate)) => map[index] = candidate,
            None => {
                let id = points.len();
                points.push(*point);
                buckets.entry(key).or_default().push(id);
                map[index] = id;
            }
        }
    }

    let mut cells: Vec<Cell> = first.cells.clone();
    for cell in &second.cells {
        let mut nodes = Vec::with_capacity(cell.nodes.len());
        for &node in &cell.nodes {
            let mapped = map
                .get(node)
                .copied()
                .ok_or_else(|| "internal error: welded cell index out of range".to_string())?;
            nodes.push(mapped);
        }
        if nodes.len() != 4 {
            return Err("the welded mesh received a non-quad cell".to_string());
        }
        cells.push(Cell::quad4([nodes[0], nodes[1], nodes[2], nodes[3]]));
    }
    let mesh = Mesh::new(points, cells);
    mesh.validate()
        .map_err(|error| format!("the welded mesh is invalid: {error}"))?;
    Ok(mesh)
}

fn point_key(point: Node, cell: f64) -> (i64, i64) {
    (
        (point[0] / cell).round() as i64,
        (point[1] / cell).round() as i64,
    )
}

/// The planar-index of the nearest point on a polyline, or `None` if farther
/// than `tol`.
fn polyline_parameter(points: &[[f64; 2]], target: [f64; 2], tol: f64) -> Option<f64> {
    if points.len() < 2 {
        return None;
    }
    let mut base = 0.0;
    let mut best: Option<(f64, f64)> = None;
    for pair in points.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let ab = [b[0] - a[0], b[1] - a[1]];
        let length2 = ab[0] * ab[0] + ab[1] * ab[1];
        let segment_length = length2.sqrt();
        if segment_length <= 0.0 {
            continue;
        }
        let t =
            (((target[0] - a[0]) * ab[0] + (target[1] - a[1]) * ab[1]) / length2).clamp(0.0, 1.0);
        let projection = [a[0] + t * ab[0], a[1] + t * ab[1]];
        let d = distance(projection, target);
        if d <= tol {
            let parameter = base + t * segment_length;
            if best.is_none_or(|(best_d, _)| d < best_d) {
                best = Some((d, parameter));
            }
        }
        base += segment_length;
    }
    best.map(|(_, parameter)| parameter)
}

// ---------------------------------------------------------------------------
// Mesh generation: terminating features
// ---------------------------------------------------------------------------

/// One successful terminating-feature construction.
struct TerminatingOutput {
    mesh: Mesh,
    route: &'static str,
    slot: Vec<[f64; 2]>,
    slot_width: f64,
    slot_gap: f64,
}

/// Build the slot holes and mesh the domain with them.
fn mesh_terminating(
    domain: &Domain,
    stubs: &[&[[f64; 2]]],
    sizing: &Sizing,
) -> Result<TerminatingOutput, String> {
    if stubs.is_empty() {
        return Err("no terminating feature was supplied".to_string());
    }
    let tol = tolerance(domain);
    let width = SLOT_WIDTH_FACTOR * sizing.size;
    let gap = SLOT_GAP_FACTOR * width;
    let mut holes = domain.holes.clone();
    let mut slots: Vec<Vec<[f64; 2]>> = Vec::with_capacity(stubs.len());
    for (index, stub) in stubs.iter().enumerate() {
        let slot = slot_loop(stub, width)
            .map_err(|error| format!("terminating feature {index}: {error}"))?;
        validate_slot(domain, &slot, tol)
            .map_err(|error| format!("terminating feature {index}: {error}"))?;
        for (other_index, other) in slots.iter().enumerate() {
            if loop_polygons_intersect(&slot, other, tol) {
                return Err(format!(
                    "terminating features {other_index} and {index} produce overlapping slots"
                ));
            }
        }
        slots.push(slot.clone());
        holes.push(slot);
    }
    let domain_with_slots =
        Domain::new(domain.outer.clone(), holes).map_err(|error| error.to_string())?;
    if domain_with_slots.area() <= tol * tol {
        return Err("the slot leaves the domain with no area".to_string());
    }

    let mut errors = Vec::new();
    for &(route, mesher) in &SLOT_ROUTES {
        let mesh = match mesher(&domain_with_slots, sizing) {
            Ok(mesh) => mesh,
            Err(error) => {
                if errors.len() < 8 {
                    errors.push(format!("{route}: {error}"));
                }
                continue;
            }
        };
        // Smoothing preserves every boundary node, so a route whose raw
        // boundary is off the slot geometry cannot be repaired by it.
        let raw_offset = verify::boundary_offset(&domain_with_slots, &mesh);
        if raw_offset > seam_tolerance(domain) {
            if errors.len() < 8 {
                errors.push(format!(
                    "{route}: boundary offset {raw_offset:.3e} is too large"
                ));
            }
            continue;
        }
        let mesh = finish_terminating(&domain_with_slots, mesh, sizing);
        let gate = verify::gate(&mesh);
        if !gate.passes() {
            if errors.len() < 8 {
                errors.push(format!(
                    "{route}: gate: {}",
                    gate.reason()
                        .unwrap_or_else(|| "unknown reason".to_string())
                ));
            }
            continue;
        }
        let topology = mesh.topology();
        if topology.cracks != 0 || topology.non_manifold != 0 || topology.inconsistent != 0 {
            if errors.len() < 8 {
                errors.push(format!(
                    "{route}: not conforming (cracks {}, non-manifold {}, inconsistent {})",
                    topology.cracks, topology.non_manifold, topology.inconsistent
                ));
            }
            continue;
        }
        let offset = verify::boundary_offset(&domain_with_slots, &mesh);
        if offset > seam_tolerance(domain) {
            if errors.len() < 8 {
                errors.push(format!(
                    "{route}: boundary offset {offset:.3e} is too large"
                ));
            }
            continue;
        }
        return Ok(TerminatingOutput {
            mesh,
            route,
            slot: slots.first().cloned().unwrap_or_default(),
            slot_width: width,
            slot_gap: gap,
        });
    }
    Err(format!(
        "no all-quad route produced a conforming slot mesh: {}",
        errors.join("; ")
    ))
}

/// Smooth only if the raw mesh does not already pass the gate.
fn finish_terminating(domain: &Domain, mesh: Mesh, sizing: &Sizing) -> Mesh {
    let raw = verify::gate(&mesh);
    if raw.passes() {
        return mesh;
    }
    let outcome = crate::smooth::smooth(
        domain,
        &mesh,
        sizing.size,
        &crate::smooth::SmoothOptions::default(),
    );
    if gate_score(&verify::gate(&outcome.smoothed)) > gate_score(&raw) {
        outcome.smoothed
    } else {
        mesh
    }
}

/// Every slot point must be inside the original domain (or on its boundary) and
/// the slot must not leave it.
fn validate_slot(domain: &Domain, slot: &[[f64; 2]], tol: f64) -> Result<(), String> {
    for point in slot {
        if !domain.contains(*point) && domain.distance_to_boundary(*point) > tol {
            return Err(format!("the slot leaves the domain at {point:?}"));
        }
    }
    for pair in slot
        .iter()
        .copied()
        .zip(slot.iter().copied().cycle().skip(1))
        .take(slot.len())
    {
        for (a, b) in domain.segments() {
            if segments_intersect(pair.0, pair.1, a, b, tol) {
                return Err("the slot crosses the domain boundary".to_string());
            }
        }
    }
    Ok(())
}

/// Measure one case, used by [`demo`].
fn measure(
    name: &str,
    domain: &Domain,
    mesh: &Mesh,
    route: &str,
    sampled_seam: bool,
    seam_nodes: usize,
    subdomains: usize,
    feature_samples: usize,
    slot_width: f64,
    slot_gap: f64,
) -> CaseNumbers {
    let stats = crate::stats::MeshStatistics::of(domain, mesh);
    CaseNumbers {
        name: name.to_string(),
        route: route.to_string(),
        sampled_seam,
        quads: stats.quads,
        triangles: stats.triangles,
        min_scaled_jacobian: stats.min_scaled_jacobian,
        min_angle_deg: stats.angle.min,
        max_angle_deg: stats.angle.max,
        cracks: stats.cracks,
        boundary_offset: stats.boundary_offset,
        gate_pass: stats.passes,
        seam_nodes,
        subdomains,
        feature_samples,
        slot_width,
        slot_gap,
    }
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

/// Tolerance for `domain`, scaled by its diagonal.
fn tolerance(domain: &Domain) -> f64 {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for point in std::iter::once(&domain.outer)
        .chain(domain.holes.iter())
        .flatten()
    {
        min_x = min_x.min(point[0]);
        min_y = min_y.min(point[1]);
        max_x = max_x.max(point[0]);
        max_y = max_y.max(point[1]);
    }
    let diagonal = ((max_x - min_x).powi(2) + (max_y - min_y).powi(2)).sqrt();
    tolerance_from_scale(diagonal)
}

fn seam_tolerance(domain: &Domain) -> f64 {
    let scale = tolerance(domain) / REL_TOL;
    SEAM_TOL * scale.max(1.0)
}

fn tolerance_from_scale(scale: f64) -> f64 {
    REL_TOL * scale.max(1.0)
}

fn scale_of_points(points: &[[f64; 2]]) -> f64 {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for point in points {
        min_x = min_x.min(point[0]);
        min_y = min_y.min(point[1]);
        max_x = max_x.max(point[0]);
        max_y = max_y.max(point[1]);
    }
    ((max_x - min_x).powi(2) + (max_y - min_y).powi(2)).sqrt()
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn unit_vector(a: [f64; 2], b: [f64; 2]) -> Result<[f64; 2], String> {
    let delta = [b[0] - a[0], b[1] - a[1]];
    let length = (delta[0] * delta[0] + delta[1] * delta[1]).sqrt();
    if length <= 0.0 || !length.is_finite() {
        return Err("a zero-length feature segment cannot be offset".to_string());
    }
    Ok([delta[0] / length, delta[1] / length])
}

fn point_segment_distance(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let length2 = ab[0] * ab[0] + ab[1] * ab[1];
    if length2 <= 0.0 {
        return distance(p, a);
    }
    let t = (((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / length2).clamp(0.0, 1.0);
    distance(p, [a[0] + t * ab[0], a[1] + t * ab[1]])
}

fn polygon_area(points: &[[f64; 2]]) -> f64 {
    let mut sum = 0.0;
    let n = points.len();
    for i in 0..n {
        let a = points[i];
        let b = points[(i + 1) % n];
        sum += a[0] * b[1] - b[0] * a[1];
    }
    sum / 2.0
}

/// Classical even-odd point-in-polygon test.
fn point_in_polygon(p: [f64; 2], polygon: &[[f64; 2]]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    for i in 0..n {
        let a = polygon[i];
        let b = polygon[(i + 1) % n];
        if (a[1] > p[1]) != (b[1] > p[1]) {
            let t = (p[1] - a[1]) / (b[1] - a[1]);
            if p[0] < a[0] + t * (b[0] - a[0]) {
                inside = !inside;
            }
        }
    }
    inside
}

fn orientation(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

fn point_on_segment(p: [f64; 2], a: [f64; 2], b: [f64; 2], tol: f64) -> bool {
    point_segment_distance(p, a, b) <= tol
}

/// Whether two closed segments intersect or touch, with `tol` absorbing
/// round-off.  Collinear overlap counts as an intersection.
fn segments_intersect(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2], tol: f64) -> bool {
    let o1 = orientation(a, b, c);
    let o2 = orientation(a, b, d);
    let o3 = orientation(c, d, a);
    let o4 = orientation(c, d, b);
    let eps = tol * (distance(a, b) + distance(c, d)).max(tol);
    if ((o1 > eps && o2 < -eps) || (o1 < -eps && o2 > eps))
        && ((o3 > eps && o4 < -eps) || (o3 < -eps && o4 > eps))
    {
        return true;
    }
    (o1.abs() <= eps && point_on_segment(c, a, b, tol))
        || (o2.abs() <= eps && point_on_segment(d, a, b, tol))
        || (o3.abs() <= eps && point_on_segment(a, c, d, tol))
        || (o4.abs() <= eps && point_on_segment(b, c, d, tol))
}

/// Whether two closed segment lists intersect.  Adjacent segments are allowed
/// to share their common endpoint.
fn loop_is_simple(points: &[[f64; 2]], tol: f64) -> bool {
    let n = points.len();
    if n < 3 {
        return false;
    }
    for i in 0..n {
        if distance(points[i], points[(i + 1) % n]) <= tol {
            return false;
        }
    }
    for i in 0..n {
        let a = points[i];
        let b = points[(i + 1) % n];
        for j in (i + 1)..n {
            let c = points[j];
            let d = points[(j + 1) % n];
            // Adjacent segments share a vertex by construction.  A spike that
            // folds back along the same line is still a self-intersection; the
            // two adjacency cases share a different vertex, so the non-shared
            // endpoints are tested explicitly.
            if j == i + 1 {
                // `(a, b)` and `(c, d)` share `b == c`.
                if point_on_segment(a, c, d, tol) || point_on_segment(d, a, b, tol) {
                    return false;
                }
                continue;
            }
            if i == 0 && j + 1 == n {
                // The first and last segments share `a == d`.
                if point_on_segment(b, c, d, tol) || point_on_segment(c, a, b, tol) {
                    return false;
                }
                continue;
            }
            if segments_intersect(a, b, c, d, tol) {
                return false;
            }
        }
    }
    true
}

/// Whether an open polyline self-intersects.  Adjacent segments share their
/// common vertex and are allowed; a fold back along the same line is not.
fn polyline_self_intersects(points: &[[f64; 2]], tol: f64) -> bool {
    let n = points.len();
    if n < 3 {
        return false;
    }
    for i in 0..n - 1 {
        if distance(points[i], points[i + 1]) <= tol {
            return true;
        }
    }
    for i in 0..n - 1 {
        let a = points[i];
        let b = points[i + 1];
        for j in (i + 1)..n - 1 {
            let c = points[j];
            let d = points[j + 1];
            if j == i + 1 {
                // Adjacent segments share `b == c`.  A forward turn is fine;
                // a fold back along the same line (one endpoint lying on the
                // other segment) is a self-overlap.
                if point_on_segment(a, c, d, tol) || point_on_segment(d, a, b, tol) {
                    return true;
                }
                continue;
            }
            if segments_intersect(a, b, c, d, tol) {
                return true;
            }
        }
    }
    false
}

/// Whether two open polylines intersect.
fn polylines_intersect(first: &[[f64; 2]], second: &[[f64; 2]], tol: f64) -> bool {
    if first.len() < 2 || second.len() < 2 {
        return false;
    }
    for a_pair in first.windows(2) {
        for b_pair in second.windows(2) {
            if segments_intersect(a_pair[0], a_pair[1], b_pair[0], b_pair[1], tol) {
                return true;
            }
        }
    }
    false
}

/// Whether two closed loops intersect (including containment-free crossing).
fn loop_polygons_intersect(first: &[[f64; 2]], second: &[[f64; 2]], tol: f64) -> bool {
    let n = first.len();
    let m = second.len();
    if n < 3 || m < 3 {
        return false;
    }
    for i in 0..n {
        for j in 0..m {
            if segments_intersect(
                first[i],
                first[(i + 1) % n],
                second[j],
                second[(j + 1) % m],
                tol,
            ) {
                return true;
            }
        }
    }
    point_in_polygon(first[0], second) || point_in_polygon(second[0], first)
}

/// Which nodes of `mesh` are boundary nodes (an edge used by exactly one cell).
fn boundary_mask(mesh: &Mesh) -> Vec<bool> {
    let mut count: HashMap<(usize, usize), usize> = HashMap::new();
    for cell in &mesh.cells {
        for i in 0..cell.nodes.len() {
            let a = cell.nodes[i];
            let b = cell.nodes[(i + 1) % cell.nodes.len()];
            *count.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    let mut mask = vec![false; mesh.points.len()];
    for ((a, b), uses) in count {
        if uses == 1 {
            if a < mask.len() {
                mask[a] = true;
            }
            if b < mask.len() {
                mask[b] = true;
            }
        }
    }
    mask
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn square() -> Domain {
        Domain::rectangle(10.0, 10.0).unwrap()
    }

    fn sizes() -> Sizing {
        Sizing::uniform(1.0).unwrap()
    }

    /// The canonical vertical samples at size 1: y = 0, 1, ..., 10.
    fn vertical_samples() -> Vec<[f64; 2]> {
        (0..=10).map(|k| [5.0, k as f64]).collect()
    }

    /// Every mesh node at x = 5 with y in [0, 10]: the feature seam.  The
    /// seam is interior in the merged mesh, so it is found by coordinate, not
    /// by the boundary mask.
    fn seam_node_ys(mesh: &Mesh) -> Vec<f64> {
        let mut ys: Vec<f64> = mesh
            .points
            .iter()
            .filter(|point| {
                (point[0] - 5.0).abs() < 1.0e-9 && point[1] > -1.0e-9 && point[1] < 10.0 + 1.0e-9
            })
            .map(|point| point[1])
            .collect();
        ys.sort_by(|a, b| a.total_cmp(b));
        ys.dedup_by(|a, b| (*a - *b).abs() < 1.0e-9);
        ys
    }

    /// How many cells use the edge between the nodes nearest `a` and `b`.
    fn edge_uses(mesh: &Mesh, a: [f64; 2], b: [f64; 2]) -> usize {
        let find = |p: [f64; 2]| -> Option<usize> {
            mesh.points
                .iter()
                .position(|q| distance([q[0], q[1]], p) < 1.0e-8)
        };
        let (Some(ia), Some(ib)) = (find(a), find(b)) else {
            return 0;
        };
        let key = (ia.min(ib), ia.max(ib));
        let mut uses = 0;
        for cell in &mesh.cells {
            for i in 0..cell.nodes.len() {
                let u = cell.nodes[i];
                let v = cell.nodes[(i + 1) % cell.nodes.len()];
                if (u.min(v), u.max(v)) == key {
                    uses += 1;
                }
            }
        }
        uses
    }

    #[test]
    fn a_vertical_through_feature_splits_the_square() {
        let domain = square();
        let features = Features::new(vec![vec![[5.0, 0.0], [5.0, 10.0]]]);
        let mesh = mesh(&domain, &features, &sizes()).unwrap();
        let stats = crate::stats::MeshStatistics::of(&domain, &mesh);
        let gate = verify::gate(&mesh);
        assert!(gate.passes(), "{:?}", gate.reason());
        assert_eq!(stats.triangles, 0);
        assert_eq!(stats.cracks, 0);
        assert!(stats.boundary_offset < 1.0e-9, "{}", stats.boundary_offset);
        // Two 5x10 halves at size 1.
        assert_eq!(stats.quads, 100, "quads = {}", stats.quads);
        // The feature is sampled once, and every canonical sample is a mesh
        // node shared by both sides.
        let ys = seam_node_ys(&mesh);
        let expected = vertical_samples();
        assert!(
            contains_points(
                &expected,
                &ys.iter().map(|y| [5.0, *y]).collect::<Vec<_>>(),
                1.0e-7
            ),
            "seam nodes {ys:?} do not contain the canonical samples"
        );
        // The feature is an interior edge chain: each canonical segment is
        // used by one quad on each side, not a boundary edge.
        for pair in expected.windows(2) {
            assert_eq!(edge_uses(&mesh, pair[0], pair[1]), 2, "{:?}", pair);
        }
        assert_eq!(mesh.topology().cracks, 0);
    }

    #[test]
    fn a_terminating_feature_becomes_a_slit() {
        let domain = square();
        let features = Features::new(vec![vec![[5.0, 0.0], [5.0, 5.0]]]);
        let mesh = mesh(&domain, &features, &sizes()).unwrap();
        let gate = verify::gate(&mesh);
        assert!(gate.passes(), "{:?}", gate.reason());
        assert_eq!(gate.triangles, 0);
        assert_eq!(gate.cracks, 0);
        // The slot is a genuine internal boundary: the mesh has an outer loop
        // and a slot loop, and the slot nodes are inside the original square.
        let mut holes = domain.holes.clone();
        holes.push(slot_loop(&[[5.0, 0.0], [5.0, 5.0]], 1.0).unwrap());
        let slot_domain = Domain::new(domain.outer.clone(), holes).unwrap();
        let stats = crate::stats::MeshStatistics::of(&slot_domain, &mesh);
        assert!(stats.boundary_offset < 1.0e-9, "{}", stats.boundary_offset);
        let original_offset = verify::boundary_offset(&domain, &mesh);
        assert!(
            original_offset > 0.2,
            "a slit is a new boundary inside the original domain: {original_offset}"
        );
        assert!(stats.quads > 0);
    }

    #[test]
    fn a_floating_feature_is_rejected() {
        let domain = square();
        let features = Features::new(vec![vec![[4.0, 4.0], [6.0, 6.0]]]);
        let error = mesh(&domain, &features, &sizes()).unwrap_err();
        assert!(error.contains("does not touch the boundary"), "{error}");
    }

    #[test]
    fn invalid_features_are_rejected() {
        let domain = square();
        let sizing = sizes();
        let cases = vec![
            vec![vec![[5.0, 5.0]]],
            vec![vec![[5.0, 5.0], [5.0, 5.0]]],
            vec![vec![[-1.0, 5.0], [11.0, 5.0]]],
            vec![vec![[5.0, 0.0], [15.0, 5.0], [5.0, 10.0]]],
            vec![vec![[1.0, 1.0], [9.0, 9.0], [9.0, 1.0], [1.0, 9.0]]],
            vec![vec![[5.0, 0.0], [5.0, 9.0], [5.0, 2.0]]],
            vec![],
        ];
        for polylines in cases {
            let features = Features::new(polylines);
            assert!(
                mesh(&domain, &features, &sizing).is_err(),
                "an invalid feature set was accepted"
            );
        }
    }

    #[test]
    fn the_two_sub_domains_are_both_used() {
        // A feature nearer one side: the two halves must have different areas,
        // so this is not a accidentally symmetric case.
        let domain = square();
        let features = Features::new(vec![vec![[3.0, 0.0], [3.0, 10.0]]]);
        let mesh = mesh(&domain, &features, &sizes()).unwrap();
        let stats = crate::stats::MeshStatistics::of(&domain, &mesh);
        assert!(stats.passes, "{:?}", stats.gate.reason());
        assert_eq!(stats.cracks, 0);
        assert!(stats.quads > 0);
        // Every canonical sample at x = 3 is a node of the merged mesh.
        let expected: Vec<[f64; 2]> = (0..=10).map(|k| [3.0, k as f64]).collect();
        let chain: Vec<[f64; 2]> = mesh
            .points
            .iter()
            .filter(|p| (p[0] - 3.0).abs() < 1.0e-9)
            .map(|p| [p[0], p[1]])
            .collect();
        assert!(contains_points(&expected, &chain, 1.0e-7));
    }

    #[test]
    fn a_mixed_feature_set_is_rejected() {
        let domain = square();
        let features = Features::new(vec![
            vec![[5.0, 0.0], [5.0, 10.0]],
            vec![[2.0, 0.0], [2.0, 5.0]],
        ]);
        assert!(mesh(&domain, &features, &sizes()).is_err());
    }

    #[test]
    fn demo_numbers_are_gate_passing() {
        let numbers = demo().unwrap();
        assert!(numbers.through_square.gate_pass);
        assert_eq!(numbers.through_square.cracks, 0);
        assert!(numbers.through_square.boundary_offset < 1.0e-9);
        assert_eq!(numbers.through_square.quads, 100);
        assert_eq!(numbers.through_square.subdomains, 2);
        assert!(numbers.through_square.sampled_seam);
        assert_eq!(numbers.vertical_feature_samples, 11);
        assert!(numbers.stub_square.gate_pass);
        assert_eq!(numbers.stub_square.cracks, 0);
        assert!(numbers.stub_square.boundary_offset < 1.0e-9);
        assert!((numbers.stub_square.slot_width - 1.0).abs() < 1.0e-12);
        assert!((numbers.stub_square.slot_gap - 0.5).abs() < 1.0e-12);
        assert!(numbers.stub_square.quads > 0);
    }

    #[test]
    fn sampling_keeps_every_vertex_and_spaces_at_most_size() {
        let samples = sample_polyline(&[[0.0, 0.0], [3.2, 0.0]], 1.0).unwrap();
        assert!(samples.len() >= 5);
        assert!(samples
            .windows(2)
            .all(|pair| distance(pair[0], pair[1]) <= 1.0 + 1.0e-12));
        let samples = sample_polyline(&[[0.0, 0.0], [0.3, 0.0]], 1.0).unwrap();
        assert_eq!(samples.len(), 2);
    }

    #[test]
    fn another_size_splits_with_the_same_quality_band() {
        let domain = square();
        let features = Features::new(vec![vec![[5.0, 0.0], [5.0, 10.0]]]);
        let sizing = Sizing::uniform(2.0).unwrap();
        let mesh = mesh(&domain, &features, &sizing).unwrap();
        let stats = crate::stats::MeshStatistics::of(&domain, &mesh);
        assert!(stats.passes, "{:?}", stats.gate.reason());
        assert_eq!(stats.cracks, 0);
        assert!(stats.quads > 0);
        // At size 2 the samples are y = 0, 2, ..., 10.
        let chain: Vec<[f64; 2]> = mesh
            .points
            .iter()
            .filter(|p| (p[0] - 5.0).abs() < 1.0e-9)
            .map(|p| [p[0], p[1]])
            .collect();
        let expected: Vec<[f64; 2]> = (0..=5).map(|k| [5.0, 2.0 * k as f64]).collect();
        assert!(contains_points(&expected, &chain, 1.0e-7));
    }
}
