//! Interior constraints for the Simons 2024 all-quad pipeline.
//!
//! Thesis reference: Simons, *Size-Driven All-Quad Meshing for Geographic Data*
//! (2024). This module implements the **interior features** of the input PSLG
//! of Chapter 2 §2.2.1, the planar subdivision induced by a set of through-going
//! paths, the **channel** carve of §2.2.2 for terminating paths, and the
//! stitching/welding of the pieces (Chapter 5 §5.1.8.2: the interior mesh is
//! stitched to the perimeter/channel).
//!
//! # What the thesis says and what this file does
//!
//! * §2.2.1: "We allow the PSLG to include interior features, which are paths
//!   that either terminate inside the region or divide the region into
//!   subregions. Both sides of any path that terminates in the interior are
//!   included in the boundary of the region ... Interior paths that split the
//!   region must also have an even number of edges."  This file validates that
//!   input (a [`Features`] set of open polylines), classifies every path, and
//!   implements both cases.
//! * **Through-going features** (both endpoints on the outer boundary).  The
//!   outer loop is subdivided successively: for each feature, every piece
//!   between two crossings with the current region boundaries is inserted into
//!   the region that contains it, splitting that region's loop in two.  The
//!   result is the planar subdivision induced by the features.  Features may
//!   cross each other (the crossing is inserted on the fly and needs no input
//!   vertex).  Every feature is *sampled once* at the element size and the same
//!   sampled chain is used on both sides of every seam.  Each face is meshed by
//!   an existing all-quad route and all faces are welded into one mesh.  Because
//!   both sides carry the same coordinates, every feature edge is an interior
//!   edge used by one quad per side: `topology().cracks == 0`, no T-junction
//!   and no duplicated seam.
//! * **Terminating features** (one endpoint on the boundary, the other strictly
//!   inside).  The thesis turns both sides of the path into boundary and carves
//!   a thin channel around it.  Two representations are implemented, selected
//!   by [`Options::terminating`]:
//!   - [`TerminatingMode::Slot`] (default): the stub is offset into a thin
//!     closed *slot* (two parallel polylines plus a round tip) submitted as a
//!     **hole**, i.e. a boundary loop, exactly as the thesis wants, with a
//!     finite width [`SLOT_WIDTH_FACTOR`] instead of the zero-width slit;
//!   - [`TerminatingMode::Channel`]: the stub is a **carved channel**. Its two
//!     sides (a slit of [`CHANNEL_SLIT_FACTOR`]) form the feature boundary, a
//!     band of width [`CHANNEL_WIDTH_FACTOR`] (§2.2.2's δ = 1.4h) along it is
//!     built as one quad per chain edge (a meshed boundary layer), and the
//!     remainder of the region is meshed by a route and welded to the band's
//!     outer chain.  The feature is therefore a boundary of the mesh with a
//!     meshed band, not a hole.
//! * **Mixed sets** are supported: the through features subdivide first, then
//!   each terminating feature becomes a slot or channel inside the face that
//!   contains it.
//!
//! # Faithful
//!
//! * PSLG validation of §2.2.1: each path has at least two points, lies inside
//!   the domain, has no self-intersection, and is classified exactly as
//!   "terminates inside" or "divides the region"; two through features may
//!   cross (a planar arrangement), while any other feature-feature contact and
//!   any collinear overlap is rejected.
//! * The planar loop subdivision and loop split: one loop becomes two loops
//!   that share a feature chain; holes are assigned to the sub-domain that
//!   contains them.
//! * One sampling per through feature at the requested size, used unchanged on
//!   both sides of every seam, which is what makes the welded mesh conforming.
//! * The terminating feature becomes boundary geometry in both modes (a hole
//!   loop in slot mode, the band's inner chain in channel mode).
//! * The channel band follows §2.2.2: a thin band between the feature and the
//!   interior mesh, meshed separately and joined to the interior.
//!
//! # Simplified or not implemented
//!
//! * **Seam preservation is a route property.**  A through feature is split
//!   with the *sampled* loops and meshed by the route that can honour the
//!   sampled chains.  The seam-preserving routes (`sized`'s grid fits the demo
//!   case; [`crate::alt_bipartite`], [`crate::alt_untangle`] and
//!   [`crate::alt_decompose`] discretise the boundary once and keep input
//!   vertices) provide the candidates.  The first route that gives the same
//!   chain on both sides and welds to zero cracks, zero overlapping pairs and a
//!   passing gate is returned; if no route preserves the samples, the plain
//!   feature vertices are retried and the route then chooses the seam.
//! * **Slot geometry.**  Slot width is one element size ([`SLOT_WIDTH_FACTOR`])
//!   and the slot starts half a width inside the boundary ([`SLOT_GAP_FACTOR`]);
//!   the inset keeps the slot a genuine closed hole.  A general polyline stub
//!   is offset with averaged vertex normals (a bevel at bends), and a
//!   self-intersecting offset is rejected.
//! * **Channel geometry.**  The band has one node per inner-chain edge; both
//!   mouth ends are projected onto the region boundary and must lie on a
//!   straight boundary segment, so the stub has to meet the boundary roughly
//!   perpendicularly.  The river slit is 0.5h wide (the thesis' zero-width slit
//!   is not a polygon) and the band width is δ = 1.4h, constant as in §2.2.2
//!   (`Sizing` is uniform here, so there is no size field to grade it with).
//!   A stub bent tighter than the band radius produces a distorted band quad,
//!   and one whose band would cross a through seam leaves its face; both are
//!   rejected with the measured reason.  There is no patch/template cover
//!   *inside* the band (the band is a regular strip), and §2.2.10 smoothing is
//!   the crate's shared smoother, not the thesis' angle/stretch Nelder-Mead.
//! * **A channel in a subdivided face** is attempted, but a narrow face (the
//!   band plus the through seam) may leave a remainder that no route can mesh
//!   while preserving the band's outer chain; such a run is rejected with that
//!   reason.  The mixed through + terminating case is demonstrated with the
//!   default slot representation.
//! * The interior of each face is meshed by a whole-domain route; this module
//!   does not inlay a separate interior grid in every face (the routes build
//!   their own interior).
//!
//! # Public API
//!
//! ```text
//! simons_features::mesh(&domain, &features, &sizing) -> Result<Mesh, String>
//! simons_features::mesh_with_options(&domain, &features, &sizing, &options)
//! simons_features::demo() -> Result<DemoNumbers, String>
//! ```
//!
//! Every mesh returned by [`mesh`]/[`mesh_with_options`] is all-quad, has
//! `topology().cracks == 0` and passes [`crate::verify::gate`]; the return value
//! is an error otherwise. Boundary nodes lie on the geometry the mesh was built
//! for (the original domain, plus the slot or river-slit holes), so the boundary
//! offset is round-off.
//!
//! # Measured (demo, size 1, 10x10 square)
//!
//! | case | route | quads | SJ | angle min/max | cracks | offset | gate |
//! |---|---|---|---|---|---|---|---|
//! | vertical through `(5,0)-(5,10)` | sized | 100 | 1.0000 | 90.0/90.0 | 0 | 0 | PASS |
//! | stub slot `(5,0)-(5,5)` | bipartite | 106 | 0.7071 | 45.0/90.0 | 0 | 4.4e-16 | PASS |
//! | cross (vertical + horizontal) | sized | 100 | 1.0000 | 90.0/90.0 | 0 | 0 | PASS |
//! | through + stub `(2,0)-(2,5)` | bipartite | 113 | 0.2660 | 32.0/90.0 | 0 | 4.4e-16 | PASS |
//! | stub channel | sized | 90 (18 band) | 0.6226 | 38.5/90.0 | 0 | 1.8e-15 | PASS |
//!
//! The cross has 4 sub-domains and 21 seam nodes (11 + 11 canonical samples, the
//! centre shared); the mixed case has 2 sub-domains and one slot; the channel
//! case has one river slit as its measured hole and an 18-quad band.  The tests
//! assert these cases, the sampled seams (every canonical sample is a node and
//! each seam edge is used by exactly two quads), and the rejected inputs.

#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

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

/// Width of the channel band of a terminating feature, in units of the element
/// size.  The thesis uses δ = 1.4h for the channel that separates the interior
/// mesh from the boundary (§2.2.2); the band option reuses that measured value.
pub const CHANNEL_WIDTH_FACTOR: f64 = 1.4;

/// Width of the (unmeshed) river slit inside a channel band, in units of the
/// element size.
///
/// The thesis leaves the feature a zero-width slit. A zero-width polygon is
/// degenerate, so the channel's inner boundary is the two sides of the stub
/// offset by half this width (see [`TerminatingMode::Channel`]).
pub const CHANNEL_SLIT_FACTOR: f64 = 0.5;

/// How a terminating feature is represented in the mesh.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum TerminatingMode {
    /// The stub becomes a thin **slot**: a closed hole of width
    /// [`SLOT_WIDTH_FACTOR`] × size whose start cap is inset by
    /// [`SLOT_GAP_FACTOR`] × width.  This is the module's original behaviour.
    #[default]
    Slot,
    /// The stub is **carved**: the stub itself (a slit of
    /// [`CHANNEL_SLIT_FACTOR`] × size) is the feature boundary, a band of
    /// width [`CHANNEL_WIDTH_FACTOR`] × size along it is meshed as a quad
    /// strip (the thesis' channel, §2.2.2), and the rest of the region is
    /// meshed and welded to the band's outer chain.  The feature is therefore
    /// a boundary of the mesh with a meshed boundary layer, not a hole.
    Channel,
}

/// Options of [`mesh_with_options`].
///
/// The shipped configuration keeps the measured defaults; the struct is
/// additive so the orchestrator can expose the channel option (it is embedded
/// in `MeshConfig::feature_options`).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Options {
    /// How terminating features are represented.
    pub terminating: TerminatingMode,
    /// Width of the channel band, in units of the element size.  Only used by
    /// [`TerminatingMode::Channel`]; the default is [`CHANNEL_WIDTH_FACTOR`].
    pub channel_width: f64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            terminating: TerminatingMode::Slot,
            channel_width: CHANNEL_WIDTH_FACTOR,
        }
    }
}

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

        // Feature pairs.  Two through features may cross: the planar
        // subdivision splits each of them at the crossing (the crossing needs
        // no input vertex, it is inserted on the fly).  A collinear overlap and
        // any contact involving a terminating feature are rejected, because a
        // slot or a channel cannot be crossed without destroying its boundary.
        for i in 0..self.polylines.len() {
            for j in (i + 1)..self.polylines.len() {
                if polylines_overlap_collinear(&self.polylines[i], &self.polylines[j], tol) {
                    return Err(format!("features {i} and {j} overlap along a segment"));
                }
                if !polylines_intersect(&self.polylines[i], &self.polylines[j], tol) {
                    continue;
                }
                if kinds[i] == FeatureKind::Through && kinds[j] == FeatureKind::Through {
                    continue;
                }
                return Err(format!(
                    "features {i} and {j} intersect; only two through-going features may \
                     cross (a terminating feature must be disjoint from every feature)"
                ));
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
    /// Route that produced the mesh ("sized", "quadtree", "bipartite",
    /// "untangle", "decompose").
    pub route: String,
    /// Whether the sampled seams of the through features were preserved.
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
    /// Nodes on the through-feature seams (all of them, in the merged mesh).
    pub seam_nodes: usize,
    /// Number of sub-domains (one per mesh region; 4 for a cross).
    pub subdomains: usize,
    /// Canonical samples requested on all through features (at the element
    /// size).
    pub feature_samples: usize,
    /// Slot width (slot mode), in length units.
    pub slot_width: f64,
    /// Slot start inset from the boundary (slot mode).
    pub slot_gap: f64,
    /// How terminating features were represented ("slot" or "channel").
    pub terminating_mode: String,
    /// Channel band width (channel mode), in length units.
    pub channel_width: f64,
    /// Quads of the pre-built channel bands (channel mode).
    pub channel_quads: usize,
    /// Number of unmeshed removal loops: one per slot or river strip.
    pub removals: usize,
}

/// Numbers returned by [`demo`].
#[derive(Clone, Debug, Default)]
pub struct DemoNumbers {
    /// 10x10 square split by the vertical feature (5,0)-(5,10).
    pub through_square: CaseNumbers,
    /// 10x10 square with the stub (5,0)-(5,5) as a thin slot.
    pub stub_square: CaseNumbers,
    /// 10x10 square with a cross: the vertical and the horizontal through
    /// features at the centre.
    pub cross_square: CaseNumbers,
    /// 10x10 square with the vertical through feature and the stub
    /// (2,0)-(2,5) (mixed through + terminating).
    pub t_square: CaseNumbers,
    /// 10x10 square with the stub (5,0)-(5,5) carved as a channel band.
    pub channel_square: CaseNumbers,
    /// Canonical number of samples on the vertical feature at size 1.
    pub vertical_feature_samples: usize,
}

/// Mesh a planar domain with interior features, with the default options.
///
/// Same as [`mesh_with_options`] with [`Options::default`] (a terminating
/// feature becomes a slot).
///
/// # Errors
///
/// Invalid geometry (a short polyline, a point outside, a self-intersection, a
/// floating feature, a terminating feature meeting another feature), and the
/// case where every tried all-quad route fails to produce a conforming
/// gate-passing mesh.
pub fn mesh(domain: &Domain, features: &Features, sizing: &Sizing) -> Result<Mesh, String> {
    mesh_with_options(domain, features, sizing, &Options::default())
}

/// Mesh a planar domain with interior features and explicit options.
///
/// `domain` is the outer/holes geometry; `features` are the interior paths of
/// §2.2.1. Both are validated.
///
/// * Several **through-going** features are supported: the outer loop is
///   subdivided successively into one sub-domain per face of the planar
///   subdivision (features may cross each other), each face is meshed with the
///   existing all-quad routes and the sub-meshes are welded on the sampled
///   seams.
/// * A **terminating** feature is either a thin slot (the default) or, with
///   [`TerminatingMode::Channel`], a carved channel band (both selectable in
///   [`Options`]); terminating and through features may be mixed in one call.
///
/// The returned mesh is all-quad, conforming (`topology().cracks == 0`) and
/// gate-passing, or an `Err` explains why no such mesh was built.
///
/// # Errors
///
/// Invalid geometry, an option that is not positive and finite, and the case
/// where every tried all-quad route fails on at least one sub-domain or the
/// welded mesh fails the gate.
pub fn mesh_with_options(
    domain: &Domain,
    features: &Features,
    sizing: &Sizing,
    options: &Options,
) -> Result<Mesh, String> {
    mesh_detailed(domain, features, sizing, options).map(|outcome| outcome.mesh)
}

/// Run the demo cases and return their measured numbers.
///
/// * a 10x10 square with the vertical through feature `(5,0)-(5,10)`, size 1;
/// * a 10x10 square with the terminating feature `(5,0)-(5,5)`, size 1 (slot);
/// * a 10x10 square with the cross `(5,0)-(5,10)` + `(0,5)-(10,5)`, size 1;
/// * a 10x10 square with the through feature `(5,0)-(5,10)` and the stub
///   `(2,0)-(2,5)`, size 1 (mixed);
/// * the same stub as the channel option (band width 1.4).
pub fn demo() -> Result<DemoNumbers, String> {
    let square = Domain::rectangle(10.0, 10.0).map_err(|error| error.to_string())?;
    let sizing = Sizing::uniform(1.0).map_err(|error| error.to_string())?;
    let default = Options::default();

    let through_feature = vec![[5.0, 0.0], [5.0, 10.0]];
    let through = mesh_detailed(
        &square,
        &Features::new(vec![through_feature.clone()]),
        &sizing,
        &default,
    )?;
    let through_numbers = measure("square / vertical through feature", &through);

    let stub_feature = vec![[5.0, 0.0], [5.0, 5.0]];
    let stub = mesh_detailed(
        &square,
        &Features::new(vec![stub_feature.clone()]),
        &sizing,
        &default,
    )?;
    let stub_numbers = measure("square / stub slot", &stub);

    let cross = mesh_detailed(
        &square,
        &Features::new(vec![
            vec![[5.0, 0.0], [5.0, 10.0]],
            vec![[0.0, 5.0], [10.0, 5.0]],
        ]),
        &sizing,
        &default,
    )?;
    let cross_numbers = measure("square / cross of two through features", &cross);

    let mixed = mesh_detailed(
        &square,
        &Features::new(vec![
            vec![[5.0, 0.0], [5.0, 10.0]],
            vec![[2.0, 0.0], [2.0, 5.0]],
        ]),
        &sizing,
        &default,
    )?;
    let t_numbers = measure("square / through + terminating", &mixed);

    let channel_options = Options {
        terminating: TerminatingMode::Channel,
        ..Options::default()
    };
    let channel = mesh_detailed(
        &square,
        &Features::new(vec![stub_feature.clone()]),
        &sizing,
        &channel_options,
    )?;
    let channel_numbers = measure("square / stub channel", &channel);

    Ok(DemoNumbers {
        through_square: through_numbers,
        stub_square: stub_numbers,
        cross_square: cross_numbers,
        t_square: t_numbers,
        channel_square: channel_numbers,
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
    Ok(Split { first, second })
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

/// One successful, fully measured feature construction.
struct Outcome {
    mesh: Mesh,
    route: &'static str,
    sampled: bool,
    seam_nodes: usize,
    subdomains: usize,
    feature_samples: usize,
    terminating_mode: TerminatingMode,
    slot_width: f64,
    slot_gap: f64,
    channel_width: f64,
    channel_quads: usize,
    removals: usize,
    measured: Domain,
}

/// One region of the planar subdivision that a route has to mesh.
struct Piece {
    /// Geometry handed to the route (a sub-domain with its slot holes, or the
    /// remainder left after a channel band is carved).
    domain: Domain,
    /// A pre-built all-quad channel band, welded to the route's mesh.
    band: Option<Mesh>,
    /// Outer chain of the band, which the route's boundary must preserve.
    band_seam: Vec<[f64; 2]>,
}

/// The complete plan: one piece per sub-domain, the measurement domain, and the
/// through seams the sampled check has to find.
struct Plan {
    pieces: Vec<Piece>,
    /// Original outer loop + original holes + every unmeshed removal loop
    /// (slot or river strip).
    measured: Domain,
    through: Vec<Vec<[f64; 2]>>,
    /// Canonical samples to look for in the merged mesh (empty for the plain
    /// seam fallback).
    samples: Vec<Vec<[f64; 2]>>,
    slot_width: f64,
    slot_gap: f64,
    channel_width: f64,
    channel_quads: usize,
    removals: usize,
    mode: TerminatingMode,
}

/// A route that turns a domain into an all-quad mesh.
type Mesher = fn(&Domain, &Sizing) -> Result<Mesh, String>;

/// The tried routes, in order.  `sized` is Chapter 2's grid + channel;
/// `quadtree` is the grid fallback; the last three discretise the boundary
/// once and keep input boundary vertices, which is what a prescribed seam and
/// a band chain need.
const ROUTES: [(&str, Mesher); 5] = [
    ("sized", crate::sized::mesh),
    ("quadtree", crate::alt_quadtree::mesh),
    ("bipartite", crate::alt_bipartite::mesh),
    ("untangle", crate::alt_untangle::mesh),
    ("decompose", crate::alt_decompose::mesh),
];

/// Validation, dispatch, and the sampled-seam / plain-seam fallbacks.
fn mesh_detailed(
    domain: &Domain,
    features: &Features,
    sizing: &Sizing,
    options: &Options,
) -> Result<Outcome, String> {
    let kinds = features.analyze(domain)?;
    if !sizing.size.is_finite() || sizing.size <= 0.0 {
        return Err("the element size must be positive and finite".to_string());
    }
    if !options.channel_width.is_finite() || options.channel_width <= 0.0 {
        return Err("the channel width must be positive and finite".to_string());
    }
    if domain.area() <= 0.0 {
        return Err("the domain has no positive area".to_string());
    }
    let tol = tolerance(domain);
    let through_indices: Vec<usize> = kinds
        .iter()
        .enumerate()
        .filter(|(_, kind)| **kind == FeatureKind::Through)
        .map(|(index, _)| index)
        .collect();
    let stubs: Vec<Vec<[f64; 2]>> = kinds
        .iter()
        .enumerate()
        .filter(|(_, kind)| **kind == FeatureKind::Terminating)
        .map(|(index, _)| features.polylines[index].clone())
        .collect();

    // No through feature: the planar subdivision is the domain itself.
    if through_indices.is_empty() {
        let plan = build_plan(domain, &[], &stubs, sizing, options, false)?;
        let (mesh, route) = try_plan(&plan, sizing, tol)?;
        return Ok(finish_outcome(plan, mesh, route));
    }

    // Prefer the exact sampled seams; fall back to route-chosen seams only if
    // no seam-preserving route can build the sampled loops.
    let plain: Vec<Vec<[f64; 2]>> = through_indices
        .iter()
        .map(|&index| snap_feature(domain, &features.polylines[index], tol))
        .collect::<Result<Vec<_>, _>>()?;
    let sampled: Vec<Vec<[f64; 2]>> = plain
        .iter()
        .map(|feature| sample_polyline(feature, sizing.size))
        .collect::<Result<Vec<_>, _>>()?;
    let mut errors: Vec<String> = Vec::new();
    for &(use_sampled, lines) in &[(true, &sampled), (false, &plain)] {
        let plan = match build_plan(domain, lines, &stubs, sizing, options, use_sampled) {
            Ok(plan) => plan,
            Err(error) => {
                if errors.len() < 12 {
                    errors.push(format!(
                        "{} seams: {error}",
                        if use_sampled { "sampled" } else { "plain" }
                    ));
                }
                continue;
            }
        };
        match try_plan(&plan, sizing, tol) {
            Ok((mesh, route)) => return Ok(finish_outcome(plan, mesh, route)),
            Err(error) => {
                if errors.len() < 12 {
                    errors.push(format!(
                        "{}: {error}",
                        if use_sampled {
                            "sampled seams"
                        } else {
                            "plain seams"
                        }
                    ));
                }
            }
        }
    }
    Err(format!(
        "no all-quad route produced a conforming feature mesh: {}",
        errors.join("; ")
    ))
}

/// Package a successful mesh with its measured numbers.
fn finish_outcome(plan: Plan, mesh: Mesh, route: &'static str) -> Outcome {
    let tol = seam_tolerance(&plan.measured);
    let points: Vec<[f64; 2]> = mesh.points.iter().map(|p| [p[0], p[1]]).collect();
    let seam_nodes = points
        .iter()
        .filter(|point| {
            plan.through
                .iter()
                .any(|line| polyline_parameter(line, **point, tol).is_some())
        })
        .count();
    let feature_samples: usize = plan.samples.iter().map(Vec::len).sum();
    Outcome {
        mesh,
        route,
        sampled: !plan.samples.is_empty(),
        seam_nodes,
        subdomains: plan.pieces.len(),
        feature_samples,
        terminating_mode: plan.mode,
        slot_width: plan.slot_width,
        slot_gap: plan.slot_gap,
        channel_width: plan.channel_width,
        channel_quads: plan.channel_quads,
        removals: plan.removals,
        measured: plan.measured,
    }
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

// ---------------------------------------------------------------------------
// The planar subdivision and the channel carve
// ---------------------------------------------------------------------------

/// Build the subdivision plan: one piece per face, then the terminating
/// features assigned to the face that contains them.
fn build_plan(
    domain: &Domain,
    through: &[Vec<[f64; 2]>],
    stubs: &[Vec<[f64; 2]>],
    sizing: &Sizing,
    options: &Options,
    sampled: bool,
) -> Result<Plan, String> {
    let tol = tolerance(domain);
    let mut regions: Vec<Domain> = vec![domain.clone()];
    for feature in through {
        if feature.len() < 2 {
            return Err("a through feature needs at least two points".to_string());
        }
        subdivide(&mut regions, feature, tol)?;
    }
    let mut pieces: Vec<Piece> = regions
        .into_iter()
        .map(|region| Piece {
            domain: region,
            band: None,
            band_seam: Vec::new(),
        })
        .collect();

    let slot_width = SLOT_WIDTH_FACTOR * sizing.size;
    let slot_gap = SLOT_GAP_FACTOR * slot_width;
    let channel_width = options.channel_width * sizing.size;
    let channel_slit = CHANNEL_SLIT_FACTOR * sizing.size;
    let mut slot_count = 0usize;
    let mut channel_quads = 0usize;
    let mut removals: Vec<Vec<[f64; 2]>> = Vec::new();
    let mut keep_out: Vec<Vec<[f64; 2]>> = Vec::new();

    for (number, stub) in stubs.iter().enumerate() {
        let index = locate_piece(&pieces, stub, tol)
            .map_err(|error| format!("terminating feature {number}: {error}"))?;
        if locate_on_loop(&pieces[index].domain.outer, stub[0], tol).is_none() {
            return Err(format!(
                "terminating feature {number}: the mouth does not lie on the outer boundary \
                 of its sub-region"
            ));
        }
        match options.terminating {
            TerminatingMode::Slot => {
                let slot = slot_loop(stub, slot_width)
                    .map_err(|error| format!("terminating feature {number}: {error}"))?;
                validate_slot(&pieces[index].domain, &slot, tol)
                    .map_err(|error| format!("terminating feature {number}: {error}"))?;
                for (other, polygon) in keep_out.iter().enumerate() {
                    if loop_polygons_intersect(&slot, polygon, tol) {
                        return Err(format!(
                            "terminating feature {number} overlaps removal {other} \
                             (features must be disjoint)"
                        ));
                    }
                }
                pieces[index].domain.holes.push(slot.clone());
                removals.push(slot.clone());
                keep_out.push(slot);
                slot_count += 1;
            }
            TerminatingMode::Channel => {
                let path = sample_polyline(stub, sizing.size)
                    .map_err(|error| format!("terminating feature {number}: {error}"))?;
                let channel = build_channel(
                    &pieces[index].domain,
                    &path,
                    channel_width,
                    channel_slit,
                    tol,
                )
                .map_err(|error| format!("terminating feature {number}: {error}"))?;
                for (other, polygon) in keep_out.iter().enumerate() {
                    if loop_polygons_intersect(&channel.outline, polygon, tol) {
                        return Err(format!(
                            "terminating feature {number} overlaps removal {other} \
                             (features must be disjoint)"
                        ));
                    }
                }
                channel_quads += channel.quads;
                removals.push(channel.river.clone());
                keep_out.push(channel.outline);
                keep_out.push(channel.river);
                pieces[index].domain = channel.remainder;
                pieces[index].band = Some(channel.band);
                pieces[index].band_seam = channel.outer;
            }
        }
    }

    let mut holes = domain.holes.clone();
    holes.extend(removals.iter().cloned());
    let measured = Domain::new(domain.outer.clone(), holes).map_err(|error| error.to_string())?;
    if measured.area() <= tol * tol {
        return Err("the removal loops leave the measured domain with no area".to_string());
    }
    let (slot_width, slot_gap, channel_width) = match options.terminating {
        TerminatingMode::Slot if slot_count > 0 => (slot_width, slot_gap, 0.0),
        TerminatingMode::Channel => (0.0, 0.0, channel_width),
        _ => (0.0, 0.0, 0.0),
    };
    Ok(Plan {
        pieces,
        measured,
        through: through.to_vec(),
        samples: if sampled {
            through.to_vec()
        } else {
            Vec::new()
        },
        slot_width,
        slot_gap,
        channel_width,
        channel_quads,
        removals: removals.len(),
        mode: options.terminating,
    })
}

/// The piece whose interior contains the terminating feature's inner end.
fn locate_piece(pieces: &[Piece], stub: &[[f64; 2]], tol: f64) -> Result<usize, String> {
    let probe = *stub
        .last()
        .ok_or_else(|| "a terminating feature needs at least two points".to_string())?;
    let mut best: Option<(f64, usize)> = None;
    for (index, piece) in pieces.iter().enumerate() {
        if !piece.domain.contains(probe) {
            continue;
        }
        let d = piece.domain.distance_to_boundary(probe);
        if best.is_none_or(|(best_d, _)| d > best_d) {
            best = Some((d, index));
        }
    }
    match best {
        Some((d, index)) if d > tol => Ok(index),
        Some(_) => Err("the interior end of the feature lies on a sub-region boundary".to_string()),
        None => Err(
            "the interior end of the feature lies outside every sub-region (does it cross a \
             through feature or a hole?)"
                .to_string(),
        ),
    }
}

/// Split one region at a time until every piece of `feature` lies in one
/// region.
fn subdivide(regions: &mut Vec<Domain>, feature: &[[f64; 2]], tol: f64) -> Result<(), String> {
    let pieces = split_feature_at_regions(feature, regions, tol)?;
    for piece in pieces {
        let probe = polyline_point_at(&piece, 0.5);
        let mut best: Option<(f64, usize)> = None;
        for (index, region) in regions.iter().enumerate() {
            if !region.contains(probe) {
                continue;
            }
            let d = region.distance_to_boundary(probe);
            if best.is_none_or(|(best_d, _)| d > best_d) {
                best = Some((d, index));
            }
        }
        let (d, index) = best.ok_or_else(|| {
            "a piece of a through feature lies outside every sub-region".to_string()
        })?;
        if d <= tol {
            return Err(
                "a piece of a through feature lies on a sub-region boundary; the feature \
                 has a grazing crossing"
                    .to_string(),
            );
        }
        let split = split_domain(&regions[index], &piece, tol)?;
        regions[index] = split.first;
        regions.push(split.second);
    }
    Ok(())
}

/// Split a through feature at every crossing with the current region
/// boundaries.
fn split_feature_at_regions(
    feature: &[[f64; 2]],
    regions: &[Domain],
    tol: f64,
) -> Result<Vec<Vec<[f64; 2]>>, String> {
    if feature.len() < 2 {
        return Err("a through feature needs at least two points".to_string());
    }
    let vertex_parameters = polyline_vertex_parameters(feature);
    let total = *vertex_parameters.last().unwrap_or(&0.0);
    if total <= tol {
        return Err("a through feature has no length".to_string());
    }
    let mut cuts: Vec<(f64, [f64; 2])> =
        vec![(0.0, feature[0]), (total, feature[feature.len() - 1])];
    let mut base = 0.0;
    for pair in feature.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let length = distance(a, b);
        if length <= 0.0 {
            continue;
        }
        for region in regions {
            let n = region.outer.len();
            if n < 2 {
                continue;
            }
            for index in 0..n {
                let c = region.outer[index];
                let d = region.outer[(index + 1) % n];
                if let Some(t) = segment_intersection_parameter(a, b, c, d, tol) {
                    cuts.push((
                        base + t * length,
                        [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])],
                    ));
                }
            }
        }
        base += length;
    }
    cuts.sort_by(|x, y| x.0.total_cmp(&y.0));
    let mut unique: Vec<(f64, [f64; 2])> = Vec::with_capacity(cuts.len());
    for (parameter, point) in cuts {
        if let Some((_, last)) = unique.last() {
            if distance(*last, point) <= tol {
                continue;
            }
        }
        unique.push((parameter, point));
    }
    let mut pieces = Vec::new();
    for window in unique.windows(2) {
        let (s0, p0) = window[0];
        let (s1, p1) = window[1];
        if s1 - s0 <= tol {
            continue;
        }
        let mut piece = vec![p0];
        for (index, point) in feature.iter().enumerate() {
            let s = vertex_parameters[index];
            if s > s0 + tol && s < s1 - tol {
                push_distinct(&mut piece, *point, tol);
            }
        }
        push_distinct(&mut piece, p1, tol);
        if piece.len() >= 2 {
            pieces.push(piece);
        }
    }
    if pieces.is_empty() {
        return Err("the feature could not be split into sub-region pieces".to_string());
    }
    Ok(pieces)
}

/// Parameter `t` on segment `a-b` of the intersection with segment `c-d`.
///
/// Endpoints count; parallel (including collinear) segments return `None`, so a
/// collinear overlap is not silently cut into pieces.  The collinear overlap is
/// rejected separately by [`polylines_overlap_collinear`].
fn segment_intersection_parameter(
    a: [f64; 2],
    b: [f64; 2],
    c: [f64; 2],
    d: [f64; 2],
    tol: f64,
) -> Option<f64> {
    let r = [b[0] - a[0], b[1] - a[1]];
    let s = [d[0] - c[0], d[1] - c[1]];
    let r_length = (r[0] * r[0] + r[1] * r[1]).sqrt();
    let s_length = (s[0] * s[0] + s[1] * s[1]).sqrt();
    if r_length <= 0.0 || s_length <= 0.0 {
        return None;
    }
    let denominator = r[0] * s[1] - r[1] * s[0];
    if denominator.abs() <= 1.0e-12 * r_length * s_length {
        return None;
    }
    let q = [c[0] - a[0], c[1] - a[1]];
    let t = (q[0] * s[1] - q[1] * s[0]) / denominator;
    let u = (q[0] * r[1] - q[1] * r[0]) / denominator;
    let t_margin = tol / r_length;
    let u_margin = tol / s_length;
    if t < -t_margin || t > 1.0 + t_margin || u < -u_margin || u > 1.0 + u_margin {
        return None;
    }
    Some(t.clamp(0.0, 1.0))
}

/// Cumulative arc-length parameter of every vertex of a polyline.
fn polyline_vertex_parameters(points: &[[f64; 2]]) -> Vec<f64> {
    let mut out = Vec::with_capacity(points.len());
    let mut total = 0.0;
    out.push(0.0);
    for pair in points.windows(2) {
        total += distance(pair[0], pair[1]);
        out.push(total);
    }
    out
}

/// Point at a fraction of the polyline's arc length.
fn polyline_point_at(points: &[[f64; 2]], fraction: f64) -> [f64; 2] {
    if points.len() < 2 {
        return points.first().copied().unwrap_or([0.0, 0.0]);
    }
    let parameters = polyline_vertex_parameters(points);
    let total = *parameters.last().unwrap_or(&0.0);
    if total <= 0.0 {
        return points[0];
    }
    let target = fraction.clamp(0.0, 1.0) * total;
    for (index, pair) in points.windows(2).enumerate() {
        let s0 = parameters[index];
        let s1 = parameters[index + 1];
        if target <= s1 || index + 2 == points.len() {
            let length = s1 - s0;
            if length <= 0.0 {
                return pair[0];
            }
            let t = ((target - s0) / length).clamp(0.0, 1.0);
            return [
                pair[0][0] + t * (pair[1][0] - pair[0][0]),
                pair[0][1] + t * (pair[1][1] - pair[0][1]),
            ];
        }
    }
    points[points.len() - 1]
}

/// A carved channel band and the region left around it.
struct ChannelPiece {
    band: Mesh,
    remainder: Domain,
    /// River slit polygon (a hole of the measured domain).
    river: Vec<[f64; 2]>,
    /// Closed outline of the band, used for overlap tests between features.
    outline: Vec<[f64; 2]>,
    /// Outer chain (on the remainder side): the route's prescribed seam.
    outer: Vec<[f64; 2]>,
    quads: usize,
}

/// Build the channel band along `path`, carve it out of `region`, and return
/// both.
///
/// The band is the thesis' channel (§2.2.2): the stub is the feature boundary
/// (two sides offset by half the slit width plus a round tip), a band of
/// `width` is meshed as one quad per chain edge, and the region left on the far
/// side of the band is a simple polygon routed and welded back to the band's
/// outer chain.  The mouth has to lie on a straight boundary segment: both
/// offset chains are projected onto the region boundary there.
fn build_channel(
    region: &Domain,
    path: &[[f64; 2]],
    width: f64,
    slit: f64,
    tol: f64,
) -> Result<ChannelPiece, String> {
    if path.len() < 2 {
        return Err("a channel needs a stub with at least two points".to_string());
    }
    if !width.is_finite() || width <= 0.0 {
        return Err("the channel width must be positive and finite".to_string());
    }
    if !slit.is_finite() || slit <= 0.0 {
        return Err("the channel slit width must be positive and finite".to_string());
    }
    let half = slit / 2.0;

    // Tangents at the stub vertices, averaged at a bend, then the two sides
    // and a round tip: the inner chain of the band.
    let mut tangents = Vec::with_capacity(path.len());
    for i in 0..path.len() {
        let tangent = if i == 0 {
            unit_vector(path[0], path[1])?
        } else if i + 1 == path.len() {
            unit_vector(path[i - 1], path[i])?
        } else {
            let a = unit_vector(path[i - 1], path[i])?;
            let b = unit_vector(path[i], path[i + 1])?;
            let sum = [a[0] + b[0], a[1] + b[1]];
            let length = (sum[0] * sum[0] + sum[1] * sum[1]).sqrt();
            if length <= 1.0e-9 {
                return Err("the stub folds back on itself; cannot carve it".to_string());
            }
            [sum[0] / length, sum[1] / length]
        };
        tangents.push(tangent);
    }
    let mut inner: Vec<[f64; 2]> = Vec::with_capacity(2 * path.len() + SLOT_TIP_SEGMENTS);
    for (point, tangent) in path.iter().zip(tangents.iter()) {
        let normal = [-tangent[1], tangent[0]];
        inner.push([point[0] + half * normal[0], point[1] + half * normal[1]]);
    }
    let last = path.len() - 1;
    let end_normal = [-tangents[last][1], tangents[last][0]];
    let start_angle = end_normal[1].atan2(end_normal[0]);
    for step in 1..=SLOT_TIP_SEGMENTS {
        let angle = start_angle - std::f64::consts::PI * step as f64 / SLOT_TIP_SEGMENTS as f64;
        inner.push([
            path[last][0] + half * angle.cos(),
            path[last][1] + half * angle.sin(),
        ]);
    }
    for i in (0..last).rev() {
        let normal = [-tangents[i][1], tangents[i][0]];
        inner.push([path[i][0] - half * normal[0], path[i][1] - half * normal[1]]);
    }

    // The two ends of the chain are the mouth: put them on the region
    // boundary (a no-op for a stub perpendicular to a straight boundary).
    let first = project_to_boundary(region, inner[0]);
    let end = project_to_boundary(region, inner[inner.len() - 1]);
    inner[0] = first;
    let inner_len = inner.len();
    inner[inner_len - 1] = end;

    // Outer chain: offset the inner chain by the band width, away from the
    // river, and project its mouth ends onto the boundary too.
    let mut outer: Vec<[f64; 2]> = Vec::with_capacity(inner.len());
    for i in 0..inner.len() {
        let tangent = chain_tangent(&inner, i)?;
        let normal = [-tangent[1], tangent[0]];
        outer.push([
            inner[i][0] + width * normal[0],
            inner[i][1] + width * normal[1],
        ]);
    }
    let outer_len = outer.len();
    outer[0] = project_to_boundary(region, outer[0]);
    outer[outer_len - 1] = project_to_boundary(region, outer[outer_len - 1]);

    // The mouth caps have to lie on the boundary; otherwise the band would
    // cross it.  For a straight boundary segment this is exact.
    for (a, b) in [
        (inner[0], outer[0]),
        (inner[inner_len - 1], outer[outer_len - 1]),
    ] {
        let middle = [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])];
        if region.distance_to_boundary(middle) > tol {
            return Err(
                "the channel mouth is not on a straight boundary segment (the stub meets \
                 the boundary too obliquely for the band)"
                    .to_string(),
            );
        }
    }
    for point in inner.iter().chain(outer.iter()) {
        if !region.contains(*point) && region.distance_to_boundary(*point) > tol {
            return Err("the channel band leaves the region".to_string());
        }
    }

    // The closed band polygon: inner chain A..B, outer chain reversed B'..A'.
    let mut outline = inner.clone();
    outline.extend(outer.iter().rev().copied());
    if !loop_is_simple(&outline, tol) {
        return Err(
            "the channel band self-intersects (the stub is too short or too \
                    sharply bent for this width)"
                .to_string(),
        );
    }
    if polygon_area(&outline).abs() <= tol * tol {
        return Err("the channel band has no area".to_string());
    }

    // Band quads.
    let n = inner.len();
    let mut points: Vec<Node> = Vec::with_capacity(2 * n);
    points.extend(inner.iter().map(|p| [p[0], p[1], 0.0]));
    points.extend(outer.iter().map(|p| [p[0], p[1], 0.0]));
    let mut cells = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        let quad = [i, i + 1, n + i + 1, n + i];
        let corners = [
            points[quad[0]],
            points[quad[1]],
            points[quad[2]],
            points[quad[3]],
        ];
        let area = (corners[1][0] - corners[0][0]) * (corners[3][1] - corners[0][1])
            - (corners[1][1] - corners[0][1]) * (corners[3][0] - corners[0][0]);
        if area <= 0.0 {
            return Err("the channel band would contain an inverted quad".to_string());
        }
        cells.push(Cell::quad4(quad));
    }
    let band = Mesh::new(points, cells);
    band.validate()
        .map_err(|error| format!("the channel band is invalid: {error}"))?;
    let quality = crate::quality::report(&band);
    if quality.min_scaled_jacobian < 0.2 {
        return Err(format!(
            "the channel band would contain a distorted quad (min SJ {:.3})",
            quality.min_scaled_jacobian
        ));
    }

    // The unmeshed river slit: the inner chain closed across the mouth.
    let river = inner.clone();
    if !loop_is_simple(&river, tol) {
        return Err("the river slit self-intersects".to_string());
    }
    if polygon_area(&river).abs() <= tol * tol {
        return Err("the river slit has no area".to_string());
    }

    // Carve: the remainder boundary walks the region boundary from the second
    // mouth point back to the first, then the outer chain.
    let a_loc = locate_on_loop(&region.outer, outer[0], tol)
        .ok_or_else(|| "the channel's outer chain does not start on the boundary".to_string())?;
    let b_loc = locate_on_loop(&region.outer, outer[outer_len - 1], tol)
        .ok_or_else(|| "the channel's outer chain does not end on the boundary".to_string())?;
    let mut remainder_outer = boundary_arc(&region.outer, b_loc, a_loc, tol);
    for point in &outer {
        push_distinct(&mut remainder_outer, *point, tol);
    }
    let remainder_outer = orient_loop(remainder_outer, tol)?;
    if !loop_is_simple(&remainder_outer, tol) {
        return Err("the region left around the channel self-intersects".to_string());
    }
    let remainder =
        Domain::new(remainder_outer, region.holes.clone()).map_err(|error| error.to_string())?;
    if remainder.area() <= tol * tol {
        return Err("the channel leaves no positive remainder area".to_string());
    }

    Ok(ChannelPiece {
        band,
        remainder,
        river,
        outline,
        outer,
        quads: n - 1,
    })
}

/// Unit tangent at chain vertex `i`, averaged at an interior vertex.
fn chain_tangent(chain: &[[f64; 2]], i: usize) -> Result<[f64; 2], String> {
    if chain.len() < 2 || i >= chain.len() {
        return Err("a chain needs at least two points".to_string());
    }
    if i == 0 {
        return unit_vector(chain[0], chain[1]);
    }
    if i + 1 == chain.len() {
        return unit_vector(chain[i - 1], chain[i]);
    }
    let a = unit_vector(chain[i - 1], chain[i])?;
    let b = unit_vector(chain[i], chain[i + 1])?;
    let sum = [a[0] + b[0], a[1] + b[1]];
    let length = (sum[0] * sum[0] + sum[1] * sum[1]).sqrt();
    if length <= 1.0e-9 {
        return Err("a channel chain folds back on itself".to_string());
    }
    Ok([sum[0] / length, sum[1] / length])
}

/// Nearest point of the region's outer loop.
fn project_to_boundary(region: &Domain, point: [f64; 2]) -> [f64; 2] {
    nearest_on_loop(&region.outer, point).0
}

/// Nearest point on a closed loop and its distance.
fn nearest_on_loop(points: &[[f64; 2]], target: [f64; 2]) -> ([f64; 2], f64) {
    let n = points.len();
    if n < 2 {
        return (points.first().copied().unwrap_or([0.0, 0.0]), f64::INFINITY);
    }
    let mut best = (points[0], f64::INFINITY);
    for index in 0..n {
        let a = points[index];
        let b = points[(index + 1) % n];
        let ab = [b[0] - a[0], b[1] - a[1]];
        let length2 = ab[0] * ab[0] + ab[1] * ab[1];
        let t = if length2 <= 0.0 {
            0.0
        } else {
            (((target[0] - a[0]) * ab[0] + (target[1] - a[1]) * ab[1]) / length2).clamp(0.0, 1.0)
        };
        let candidate = [a[0] + t * ab[0], a[1] + t * ab[1]];
        let d = distance(candidate, target);
        if d < best.1 {
            best = (candidate, d);
        }
    }
    best
}

/// Whether two polylines overlap along a segment (a collinear shared piece).
fn polylines_overlap_collinear(first: &[[f64; 2]], second: &[[f64; 2]], tol: f64) -> bool {
    for a in first.windows(2) {
        for b in second.windows(2) {
            if segments_overlap_collinear(a[0], a[1], b[0], b[1], tol) {
                return true;
            }
        }
    }
    false
}

/// Whether two closed segments are collinear and share a piece longer than
/// `tol`.
fn segments_overlap_collinear(
    a: [f64; 2],
    b: [f64; 2],
    c: [f64; 2],
    d: [f64; 2],
    tol: f64,
) -> bool {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let length = (ab[0] * ab[0] + ab[1] * ab[1]).sqrt();
    if length <= 0.0 {
        return false;
    }
    let line_distance =
        |p: [f64; 2]| ((p[0] - a[0]) * ab[1] - (p[1] - a[1]) * ab[0]).abs() / length;
    if line_distance(c) > tol || line_distance(d) > tol {
        return false;
    }
    let project = |p: [f64; 2]| ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / (length * length);
    let tc = project(c);
    let td = project(d);
    let low = tc.min(td);
    let high = tc.max(td);
    let margin = tol / length;
    high > margin && low < 1.0 - margin && (high - low) * length > tol
}

/// Try every route on the whole plan and keep the first conforming result.
fn try_plan(plan: &Plan, sizing: &Sizing, tol: f64) -> Result<(Mesh, &'static str), String> {
    let mut errors: Vec<String> = Vec::new();
    for &(route, mesher) in &ROUTES {
        match try_plan_route(plan, sizing, mesher, tol) {
            Ok(mesh) => return Ok((mesh, route)),
            Err(error) => {
                if errors.len() < 8 {
                    errors.push(format!("{route}: {error}"));
                }
            }
        }
    }
    Err(format!(
        "no all-quad route produced a conforming feature mesh: {}",
        errors.join("; ")
    ))
}

/// Mesh every piece with one route, weld the channel bands and the pieces, and
/// check the merged result.
fn try_plan_route(plan: &Plan, sizing: &Sizing, mesher: Mesher, tol: f64) -> Result<Mesh, String> {
    let plan_tol = seam_tolerance(&plan.measured).max(tol);
    let mut meshes = Vec::with_capacity(plan.pieces.len());
    for (index, piece) in plan.pieces.iter().enumerate() {
        let piece_tol = seam_tolerance(&piece.domain).max(tol);
        let raw = mesher(&piece.domain, sizing)
            .map_err(|error| format!("piece {index}: the route failed: {error}"))?;
        // The smoother never moves a boundary node, so a route whose raw
        // boundary is already off the piece geometry cannot be repaired by it.
        // Checking before smoothing avoids spending the smoothing pass on it.
        let offset = verify::boundary_offset(&piece.domain, &raw);
        if offset > piece_tol {
            return Err(format!(
                "piece {index}: the route's boundary is {offset:.3e} off the geometry"
            ));
        }
        let finished = finish_side(&piece.domain, raw, sizing);
        let mesh = match &piece.band {
            None => finished,
            Some(band) => {
                let seam = seam_chain(&finished, &piece.band_seam, piece_tol)
                    .map_err(|error| format!("piece {index}: the band chain: {error}"))?;
                if !same_chain(&seam, &piece.band_seam, piece_tol) {
                    return Err(format!(
                        "piece {index}: the route changed the band chain ({} vs {} nodes)",
                        seam.len(),
                        piece.band_seam.len()
                    ));
                }
                weld(band, &finished, piece_tol)?
            }
        };
        meshes.push(mesh);
    }
    let merged = weld_all(&meshes, plan_tol)?;
    let gate = verify::gate(&merged);
    if !gate.passes() {
        return Err(format!(
            "the merged mesh fails the gate: {}",
            gate.reason()
                .unwrap_or_else(|| "unknown reason".to_string())
        ));
    }
    let topology = merged.topology();
    if topology.cracks != 0 || topology.non_manifold != 0 || topology.inconsistent != 0 {
        return Err(format!(
            "the merged mesh is not conforming (cracks {}, non-manifold {}, \
             inconsistent {})",
            topology.cracks, topology.non_manifold, topology.inconsistent
        ));
    }
    let offset = verify::boundary_offset(&plan.measured, &merged);
    if offset > plan_tol {
        return Err(format!(
            "the merged boundary is {offset:.3e} off the input geometry"
        ));
    }
    if !plan.samples.is_empty() {
        let points: Vec<[f64; 2]> = merged.points.iter().map(|p| [p[0], p[1]]).collect();
        for (index, samples) in plan.samples.iter().enumerate() {
            if !contains_points(samples, &points, plan_tol) {
                return Err(format!(
                    "through feature {index}: the sampled seam nodes were not all preserved"
                ));
            }
        }
    }
    Ok(merged)
}

/// Weld a list of meshes pairwise by coordinate.
fn weld_all(meshes: &[Mesh], tol: f64) -> Result<Mesh, String> {
    let mut iter = meshes.iter();
    let first = iter
        .next()
        .ok_or_else(|| "internal error: the feature plan has no pieces".to_string())?;
    let mut merged = first.clone();
    for mesh in iter {
        merged = weld(&merged, mesh, tol)?;
    }
    Ok(merged)
}

/// If a route's raw mesh does not pass the gate, try the boundary-preserving
/// smoother and keep a measured improvement. Boundary nodes are never moved, so
/// the seam stays valid.
fn finish_side(domain: &Domain, mesh: Mesh, sizing: &Sizing) -> Mesh {
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
// Slots
// ---------------------------------------------------------------------------

/// Every slot point must be inside its region (or on the region boundary) and
/// the slot must not cross it.
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
fn measure(name: &str, outcome: &Outcome) -> CaseNumbers {
    let stats = crate::stats::MeshStatistics::of(&outcome.measured, &outcome.mesh);
    CaseNumbers {
        name: name.to_string(),
        route: outcome.route.to_string(),
        sampled_seam: outcome.sampled,
        quads: stats.quads,
        triangles: stats.triangles,
        min_scaled_jacobian: stats.min_scaled_jacobian,
        min_angle_deg: stats.angle.min,
        max_angle_deg: stats.angle.max,
        cracks: stats.cracks,
        boundary_offset: stats.boundary_offset,
        gate_pass: stats.passes,
        seam_nodes: outcome.seam_nodes,
        subdomains: outcome.subdomains,
        feature_samples: outcome.feature_samples,
        slot_width: outcome.slot_width,
        slot_gap: outcome.slot_gap,
        terminating_mode: match outcome.terminating_mode {
            TerminatingMode::Slot => "slot".to_string(),
            TerminatingMode::Channel => "channel".to_string(),
        },
        channel_width: outcome.channel_width,
        channel_quads: outcome.channel_quads,
        removals: outcome.removals,
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
            // A terminating feature may not end on a through feature.
            vec![vec![[5.0, 0.0], [5.0, 10.0]], vec![[2.0, 0.0], [5.0, 5.0]]],
            // Two terminating features may not cross.
            vec![vec![[2.0, 0.0], [2.0, 5.0]], vec![[0.0, 2.0], [4.0, 2.0]]],
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
    fn a_cross_of_two_through_features_makes_four_sub_meshes() {
        let domain = square();
        let features = Features::new(vec![
            vec![[5.0, 0.0], [5.0, 10.0]],
            vec![[0.0, 5.0], [10.0, 5.0]],
        ]);
        let outcome = mesh_detailed(&domain, &features, &sizes(), &Options::default()).unwrap();
        assert!(outcome.sampled, "the sampled seams should be preserved");
        assert_eq!(outcome.subdomains, 4);
        // 11 + 11 canonical samples, the centre one shared.
        assert_eq!(outcome.seam_nodes, 21);
        let mesh = &outcome.mesh;
        let stats = crate::stats::MeshStatistics::of(&domain, mesh);
        assert!(stats.passes, "{:?}", stats.gate.reason());
        assert_eq!(stats.triangles, 0);
        assert_eq!(stats.cracks, 0);
        assert!(stats.boundary_offset < 1.0e-9, "{}", stats.boundary_offset);
        // Four 5x5 quadrant grids at size 1.
        assert_eq!(stats.quads, 100, "quads = {}", stats.quads);
        let points: Vec<[f64; 2]> = mesh.points.iter().map(|p| [p[0], p[1]]).collect();
        let vertical = vertical_samples();
        let horizontal: Vec<[f64; 2]> = (0..=10).map(|k| [k as f64, 5.0]).collect();
        assert!(contains_points(&vertical, &points, 1.0e-7));
        assert!(contains_points(&horizontal, &points, 1.0e-7));
        // Both seams are interior edge chains: one quad on each side.
        for pair in vertical.windows(2) {
            assert_eq!(edge_uses(mesh, pair[0], pair[1]), 2, "{:?}", pair);
        }
        for pair in horizontal.windows(2) {
            assert_eq!(edge_uses(mesh, pair[0], pair[1]), 2, "{:?}", pair);
        }
        // The centre is one node shared by the four quadrants.
        let centres = mesh
            .points
            .iter()
            .filter(|p| distance([p[0], p[1]], [5.0, 5.0]) < 1.0e-8)
            .count();
        assert_eq!(centres, 1);
    }

    #[test]
    fn a_cross_at_a_non_sampled_point_still_welds() {
        // At size 2 the samples are on even coordinates, so the crossing
        // (5,5) is not a sample of either feature: the subdivision inserts it
        // into both seam chains.
        let domain = square();
        let features = Features::new(vec![
            vec![[5.0, 0.0], [5.0, 10.0]],
            vec![[0.0, 5.0], [10.0, 5.0]],
        ]);
        let sizing = Sizing::uniform(2.0).unwrap();
        let outcome = mesh_detailed(&domain, &features, &sizing, &Options::default()).unwrap();
        assert!(outcome.sampled);
        assert_eq!(outcome.subdomains, 4);
        let mesh = &outcome.mesh;
        let stats = crate::stats::MeshStatistics::of(&domain, mesh);
        assert!(stats.passes, "{:?}", stats.gate.reason());
        assert_eq!(stats.triangles, 0);
        assert_eq!(stats.cracks, 0);
        assert!(stats.boundary_offset < 1.0e-9, "{}", stats.boundary_offset);
        let points: Vec<[f64; 2]> = mesh.points.iter().map(|p| [p[0], p[1]]).collect();
        assert!(contains_points(
            &(0..=5).map(|k| [5.0, 2.0 * k as f64]).collect::<Vec<_>>(),
            &points,
            1.0e-7
        ));
        assert!(contains_points(
            &(0..=5).map(|k| [2.0 * k as f64, 5.0]).collect::<Vec<_>>(),
            &points,
            1.0e-7
        ));
        // The crossing is one welded node.
        assert_eq!(
            mesh.points
                .iter()
                .filter(|p| distance([p[0], p[1]], [5.0, 5.0]) < 1.0e-8)
                .count(),
            1
        );
    }

    #[test]
    fn three_through_features_make_six_sub_meshes() {
        let domain = square();
        let features = Features::new(vec![
            vec![[3.0, 0.0], [3.0, 10.0]],
            vec![[7.0, 0.0], [7.0, 10.0]],
            vec![[0.0, 5.0], [10.0, 5.0]],
        ]);
        let outcome = mesh_detailed(&domain, &features, &sizes(), &Options::default()).unwrap();
        assert!(outcome.sampled);
        assert_eq!(outcome.subdomains, 6);
        let mesh = &outcome.mesh;
        let stats = crate::stats::MeshStatistics::of(&domain, mesh);
        assert!(stats.passes, "{:?}", stats.gate.reason());
        assert_eq!(stats.triangles, 0);
        assert_eq!(stats.cracks, 0);
        assert!(stats.boundary_offset < 1.0e-9, "{}", stats.boundary_offset);
        assert_eq!(stats.quads, 100, "quads = {}", stats.quads);
        let points: Vec<[f64; 2]> = mesh.points.iter().map(|p| [p[0], p[1]]).collect();
        for x in [3.0, 7.0] {
            assert!(contains_points(
                &(0..=10).map(|k| [x, k as f64]).collect::<Vec<_>>(),
                &points,
                1.0e-7
            ));
        }
        assert!(contains_points(
            &(0..=10).map(|k| [k as f64, 5.0]).collect::<Vec<_>>(),
            &points,
            1.0e-7
        ));
    }

    #[test]
    fn a_through_and_a_terminating_feature_are_mixed() {
        let domain = square();
        let features = Features::new(vec![
            vec![[5.0, 0.0], [5.0, 10.0]],
            vec![[2.0, 0.0], [2.0, 5.0]],
        ]);
        let outcome = mesh_detailed(&domain, &features, &sizes(), &Options::default()).unwrap();
        assert!(outcome.sampled);
        assert_eq!(outcome.subdomains, 2);
        assert_eq!(outcome.removals, 1);
        assert_eq!(outcome.terminating_mode, TerminatingMode::Slot);
        let mesh = &outcome.mesh;
        let stats = crate::stats::MeshStatistics::of(&outcome.measured, mesh);
        assert!(stats.passes, "{:?}", stats.gate.reason());
        assert_eq!(stats.triangles, 0);
        assert_eq!(stats.cracks, 0);
        assert!(stats.boundary_offset < 1.0e-9, "{}", stats.boundary_offset);
        // The through seam is sampled on both halves and welded.
        let points: Vec<[f64; 2]> = mesh.points.iter().map(|p| [p[0], p[1]]).collect();
        assert!(contains_points(&vertical_samples(), &points, 1.0e-7));
        for pair in vertical_samples().windows(2) {
            assert_eq!(edge_uses(mesh, pair[0], pair[1]), 2, "{:?}", pair);
        }
        // The slot is a boundary of the left half: it is inside the original
        // square but on the measured domain's boundary.
        let original_offset = verify::boundary_offset(&domain, mesh);
        assert!(original_offset > 0.4, "{original_offset}");
        assert!(stats.quads > 0);
    }

    #[test]
    fn a_carved_channel_is_a_meshed_boundary_band() {
        let domain = square();
        let features = Features::new(vec![vec![[5.0, 0.0], [5.0, 5.0]]]);
        let options = Options {
            terminating: TerminatingMode::Channel,
            ..Options::default()
        };
        let outcome = mesh_detailed(&domain, &features, &sizes(), &options).unwrap();
        assert_eq!(outcome.terminating_mode, TerminatingMode::Channel);
        assert!(outcome.channel_quads > 0, "the band has no quads");
        assert!((outcome.channel_width - CHANNEL_WIDTH_FACTOR).abs() < 1.0e-12);
        assert_eq!(outcome.slot_width, 0.0);
        assert_eq!(outcome.removals, 1);
        let mesh = &outcome.mesh;
        let stats = crate::stats::MeshStatistics::of(&outcome.measured, mesh);
        assert!(stats.passes, "{:?}", stats.gate.reason());
        assert_eq!(stats.triangles, 0);
        assert_eq!(stats.cracks, 0);
        assert!(stats.boundary_offset < 1.0e-9, "{}", stats.boundary_offset);
        // The river slit is the only hole of the measured domain and its
        // boundary nodes are mesh nodes: the feature is a boundary.
        assert_eq!(outcome.measured.holes.len(), 1);
        let river = &outcome.measured.holes[0];
        let on_river = mesh
            .points
            .iter()
            .filter(|p| {
                let q = [p[0], p[1]];
                let n = river.len();
                (0..n).any(|i| point_segment_distance(q, river[i], river[(i + 1) % n]) < 1.0e-8)
            })
            .count();
        assert!(on_river >= 4, "only {on_river} nodes on the river slit");
        // The band quads are the newest contributor: the mesh is larger than
        // the slot mesh at the same size.
        let slot = mesh_detailed(&domain, &features, &sizes(), &Options::default()).unwrap();
        assert!(stats.quads > slot.mesh.cells.len() - outcome.channel_quads);
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
        assert_eq!(numbers.stub_square.terminating_mode, "slot");
        assert!(numbers.stub_square.quads > 0);

        assert!(numbers.cross_square.gate_pass);
        assert_eq!(numbers.cross_square.cracks, 0);
        assert!(numbers.cross_square.boundary_offset < 1.0e-9);
        assert_eq!(numbers.cross_square.quads, 100);
        assert_eq!(numbers.cross_square.subdomains, 4);
        assert_eq!(numbers.cross_square.seam_nodes, 21);
        assert!(numbers.cross_square.sampled_seam);

        assert!(numbers.t_square.gate_pass);
        assert_eq!(numbers.t_square.cracks, 0);
        assert!(numbers.t_square.boundary_offset < 1.0e-9);
        assert_eq!(numbers.t_square.subdomains, 2);
        assert_eq!(numbers.t_square.removals, 1);
        assert_eq!(numbers.t_square.terminating_mode, "slot");
        assert!(numbers.t_square.sampled_seam);

        assert!(numbers.channel_square.gate_pass);
        assert_eq!(numbers.channel_square.cracks, 0);
        assert!(numbers.channel_square.boundary_offset < 1.0e-9);
        assert_eq!(numbers.channel_square.terminating_mode, "channel");
        assert!((numbers.channel_square.channel_width - CHANNEL_WIDTH_FACTOR).abs() < 1.0e-12);
        assert!(numbers.channel_square.channel_quads > 0);
        assert_eq!(numbers.channel_square.slot_width, 0.0);
        assert_eq!(numbers.channel_square.removals, 1);
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
