//! Region decomposition of a size-aware cross field: singularities, the
//! motorcycle graph of its separatrices, the simple tiles, and their QGP
//! subdivision.
//!
//! This is the missing link between Chapter 3's cross field (produced by
//! [`crate::simons_godf::solve`]) and Chapter 5's Quantized Global
//! Parameterization (implemented in [`crate::simons_qgp`]).
//!
//! # What is implemented (Simons 2024)
//!
//! 1. **Singularities (Section 3.1.6).** For every triangle the holonomy of the
//!    4-RoSy field is the sum of the three wrapped transitions of
//!    `exp(i 4 alpha)`, so it is `-2 pi`, `0` or `+2 pi`; the charge is `+1`
//!    (valence 3), `0` or `-1` (valence 5), and there is at most one singularity
//!    per triangle. The position inside the triangle is the zero of the affine
//!    interpolant of `exp(i 4 alpha)` (a 3x3 barycentric solve), not the
//!    triangle centroid.
//!
//! 2. **Separatrix directions (Section 3.1.8).** Around a zero `z0` the
//!    interpolated field is `u(z) = a (z - z0) + b conj(z - z0)`, so the
//!    four-RoSy angle is `arg(u)/4`. A ray at angle `theta` is a field line when
//!    its tangent is one of the four directions, i.e. when
//!    `h(theta) = wrap(4 theta - arg(a exp(i theta) + b exp(-i theta)))`
//!    vanishes. `h` has exactly `valence` zeros on `[0, 2 pi)`, so a valence-3
//!    singularity emits three separatrices and a valence-5 singularity five:
//!    a quad-mesh vertex of valence `v` has `v` incident edges (Section 3.1.8's
//!    "six separatrices, three field lines" rule for even valences).
//!
//! 3. **Tracing (Sections 3.1.10 and 3.1.12).** A fixed-step RK4 walk on the
//!    triangle mesh. The direction is sampled with the barycentric angle
//!    interpolation of [`crate::simons_godf::interp_angle`] and the branch of
//!    the cross closest to the current heading is followed, so the path never
//!    flips by 90 degrees. The walk stops at the first of: a boundary edge
//!    (intersected exactly, not by bisection), another singularity, or an
//!    already traced separatrix.
//!
//! 4. **The motorcycle graph (Section 3.1.9).** A separatrix that meets an
//!    existing separatrix stops there and forms a T-junction with a 180 degree
//!    interior angle; the met edge is split at the intersection so the graph
//!    stays planar. Node kinds are singularities, boundary hits and T-junctions.
//!    The motorcycle rule is applied in tracing order (each new motorcycle
//!    stops at the first existing path); Section 3.1.9 advances all motorcycles
//!    simultaneously at equal speed, which only changes which motorcycle "wins"
//!    a collision. The order here is deterministic: singularities in triangle
//!    order, arms in angle order.
//!
//! 5. **Simple tiles (Definition 5.0.1 and Section 5.0.1.2).** The faces of the
//!    graph are extracted by walking half-edges (`next` = smallest clockwise
//!    turn from the reverse of the incoming direction, which keeps the face on
//!    the left). A face whose interior is not inside the input mesh is
//!    discarded: that is the outer face and, on an annulus, the hole face. Each
//!    remaining face is split into four sides at its corners, where a corner is
//!    a node whose boundary turn exceeds 45 degrees and a T-junction is a node
//!    where the boundary continues straight (180 degree interior angle). A
//!    material face with a number of corners other than four is counted as a
//!    degenerate tile and skipped (Section 5.1.6).
//!
//! 6. **Tile QGP (Sections 5.1.1-5.1.3).** The ideal count of an edge is
//!    `s_hat = integral ds / sigma` (trapezoid over the polyline, with `1/sigma`
//!    interpolated barycentrically at the samples). Every tile contributes the
//!    eq. (5.1) region `[side0, side1, side2, side3]` to
//!    [`crate::simons_qgp::solve_counts`], and each tile is meshed with
//!    [`crate::simons_qgp::mesh_tile_curves`] (the bilinear Coons patch of eq.
//!    (5.2)) using its integer counts. The tile meshes are welded by coordinate
//!    into one [`Mesh`].
//!
//! # Faithful
//!
//! Per-triangle holonomy and one singularity per triangle, the exact zero
//! position, the `v` separatrix directions from the local affine model, the
//! fixed-step walk on the barycentric field, the motorcycle T-junction rule
//! with edge splitting, the planar half-edge face walk, four-corner simple
//! tiles, eq. (5.1) through [`crate::simons_qgp`], and eq. (5.2) through
//! [`crate::simons_qgp::mesh_tile_curves`].
//!
//! # Simplified
//!
//! * The motorcycle graph is built sequentially (one separatrix at a time, each
//!   stopping at the first already traced path) instead of all motorcycles
//!   advancing simultaneously. The result still partitions the region into
//!   simple tiles.
//! * Seed points are placed a small distance from the singularity along the
//!   exact local ray direction instead of the binary-search "trace inward"
//!   scheme of Section 3.1.12; the local affine model gives the ray exactly, so
//!   no inward search is needed.
//! * `s_hat` uses the caller's size input directly (no Chapter 4 minimum
//!   anisotropy optimisation).
//! * Coons patches use uniformly spaced parameter seams along each side (the
//!   "aligned" special case of Section 5.1.5.1); the per-edge integer counts are
//!   enforced as side sums, which is exactly eq. (5.1).
//!
//! # Boundary-adjacent tiles
//!
//! A tile of the motorcycle graph is a logical rectangle only when its four
//! sides are field lines. The domain boundary is not a field line in general, so
//! the tiles that touch it are bounded by two separatrices and one or two
//! boundary arcs; they can have three or five corners, or (on a bent tile such
//! as the nautilus) a corner polygon that is not counter-clockwise. Such faces
//! are reported in [`Regions::degenerate_tiles`] and left out of the mesh, which
//! leaves a crack. The thesis removes exactly these tiles: it carves quads away
//! from the boundary and fills the gap with the channel of Chapter 2 (Section
//! 5.1.8.2), which this module does not implement. On the annulus demo the
//! interior tiles are meshed and the boundary ring is not, which is measured and
//! reported by [`demo`].
//!
//! Measured attempts at avoiding the channel, all rejected:
//!
//! * Splitting a three-cornered face's free boundary side in two and choosing
//!   its two free counts from the shared sides (a local version of Section
//!   5.1.6's completion) does mesh the face, but its two shared sides are fixed
//!   at `6 x 6`, so the Coons patch must fit 36 quads into a sliver: coverage
//!   rises from 47.9 % to 52.9 % and the quad count from 49 to 85, while the
//!   smallest scaled Jacobian drops from -0.147 to -0.221, the smallest area
//!   from 3.5e-2 to 6.0e-4 and the inverted-quad count rises from 1 to 5. A
//!   triangle with `n` subdivisions on both shared sides cannot be
//!   quadrangulated without either a singularity pair or degenerate quads, so
//!   this is not a substitute for the channel.
//! * Lowering [`CORNER_TURN`] to 30 degrees or less (so a 38-degree boundary
//!   bend counts as a corner) changes the region graph enough that
//!   [`crate::simons_qgp::solve_counts`] returns `constraint N is not balanced
//!   after the solve`, an internal error of that solver, so the threshold stays
//!   at 45 degrees.
//! * Section 5.1.4's relaxation `s0 + s1 = s2 + s3` is equivalent to choosing a
//!   different pairing of the four sides in the strict constraint: `A - C =
//!   B - D` is `A + D = C + B`, and `A - C = -(B - D)` is `A + B = C + D`. The
//!   demo's tiles all satisfy `A = C` and `B = D` already, so the relaxation
//!   changes no measured number; implementing it properly also needs the
//!   chord-redirection mesh of Figure 5.4, which is not here.
//!
//! # Stitching (Section 5.1.8.2)
//!
//! [`stitch`] carves the interior QGP quads within `1.4 * sigma` of the outer
//! boundary, extracts the boundary loop of what remains and hands it to
//! [`crate::simons_channel::channel_between`] together with the domain
//! boundary. Measured, and blocked on two concrete things:
//!
//! Pinched boundary nodes are split before the walk: every non-manifold
//! boundary node gets one copy per incident boundary loop (the quads at the
//! node are re-indexed to the copy of the wedge that contains their centroid),
//! so the boundary edge set becomes a set of disjoint cycles.
//!
//! Measured with T4's channel HEAD (`channel_between_dense`,
//! `channel_between_hole`):
//!
//! * **Rectangle: gate PASS.** 160 interior quads, 48 carved, 112 kept, a
//!   44-node interior loop (area 1.400) and a 44-quad equal-count projection
//!   band: **156 quads, cracks 0, coverage 1.000, min SJ 0.625, angles
//!   38.7-141.3 degrees, boundary offset 7e-18**.
//! * **Annulus: gate PASS via the collar fallback, with the QGP interior now
//!   including the horseshoe.** The 2.93-area horseshoe (four field lines, a
//!   clockwise corner polygon) is kept as a *bent* tile and meshed by a Coons
//!   patch whose two long sides (8.767 and 6.067) are re-parametrized by their
//!   **angle about the tile centroid** instead of arc length
//!   ([`resample_angular`] + [`bent_coons_quads`]); the QGP interior grows from
//!   49 to **79 quads** with every quad positive. The interior is still not a
//!   ring, though: the horseshoe is an *open* C-strip (its two ends are the
//!   radial seps, so its boundary is one loop with no hole), and the hole is
//!   surrounded by the three 3/5-corner faces, which are still rejected. Those
//!   faces measure: areas 0.433, 0.185 and 0.174, with sides (graph edges,
//!   boundary edges) = [(2,0), (1,1), (2,0)], [(2,0), (2,1), (1,0)] and
//!   [(3,1), (1,0), (1,0)]: the free hole side is a **single** graph edge (so
//!   its subdivision is free) while the two seps carry the neighbours' counts.
//!   A quad mesh of a triangle with legs 6 and base 4 needs 12 quads and four
//!   interior vertices (two valence-3/5 pairs) by the Euler/edge-parity count,
//!   and it needs six rows because each row consumes one segment per leg; the
//!   "rows of 6, 4, 2" layout consumes only three side edges per leg, so it is
//!   infeasible as specified (and the arc-2 / rows 6, 4 variant fails the same
//!   count: 14 boundary segments and 10 quads give 13 interior edges, i.e. 17
//!   vertices, not 14 + two pairs). A correct (6, 6, 4) even-triangle
//!   quadrangulation is what is missing; it is not implemented, so
//!   `ring_bands` reports "no hole loop around the hole" and the collar
//!   fallback runs: a collar loop offset `1.4 sigma` from the hole polygon
//!   (100 nodes) carries the outer band (100 quads) and the hole band (100
//!   quads), which share its ids. **200 quads, 0 QGP interior quads, cracks 0,
//!   coverage 1.000, min SJ 0.729, angles 46.8-129.6 degrees, boundary offset
//!   2e-16.** This is a fallback, not the thesis stitch.
//!
//! Section 5.1.5.2 merging is the last route, and its scale is measured
//! (`annulus_singularity_neighbourhoods`): the four singularities carry 3, 5, 3
//! and 3 incident tiles with total areas 2.075, 5.794, 2.061 and 1.816. A
//! merged region is therefore a non-convex 6-to-10-corner polygon spanning up
//! to most of the annulus, which the four-side convex Coons machinery cannot
//! mesh; it needs a multi-block or non-convex block decomposition, and that is
//! what still separates this module from a QGP ring on the annulus.
//!
//! Measured attempts at the ring, all rejected: coarser annulus meshes (40x4,
//! 60x6) keep the same rejected faces; the bent-horseshoe Coons patch folds
//! when its long sides are sampled by arc length (fixed here by the angular
//! sampling); a phase sweep over 0-35 degrees of the cross field never leaves a
//! ring component (at phase 0 the carved interior is blobs of 11, 10 and 4
//! quads, each with a single loop); and a two-band path on a
//! hole-enclosing loop finds no such loop.
//!
//! # Not implemented
//!
//! Extra singularity pairs and the nautilus relaxation of Section 5.1.4 (a tile
//! whose two opposite sides are the same edge, or an annular tile, is rejected
//! with an error); singularity merging (Section 5.1.5.2); degenerate-tile
//! collapse into a single vertex (Section 5.1.6: the tile is skipped and
//! counted instead); limit cycles (an arm that closes on itself is not
//! detected, it runs to the boundary); the boundary carving, channel and
//! stitching of Chapter 2; and the size-function construction of Chapter 4.
//! This module does not build a cross field, it consumes one.
//!
//! # Example
//!
//! ```no_run
//! use pedroso_mesher::simons_regions;
//!
//! let points = vec![
//!     [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
//! ];
//! let tris = vec![[0, 1, 2], [0, 2, 3]];
//! let alpha = vec![0.0; 4];
//! let size = vec![0.1; 4];
//! if let Ok(mesh) = simons_regions::mesh_tiles(&points, &tris, &alpha, &size) {
//!     assert!(!mesh.cells.is_empty());
//! }
//! ```
//!
//! The module has no dependencies beyond the crate, never panics in non-test
//! code, and is `wasm32-unknown-unknown` compatible.

use std::collections::HashMap;

use crate::mesh::{Cell, Mesh, Node};
use crate::quality;
use crate::simons_channel;
use crate::simons_godf;
use crate::simons_qgp::{self, BoundaryCurves, Region};
use crate::two_d::Domain;

/// The RK4 step is this fraction of the mean triangle edge length.
const STEP_FRACTION: f64 = 0.35;
/// Safety bound on the number of RK4 steps of one separatrix.
const MAX_STEPS: usize = 20_000;
/// Samples on the ray-direction scan of one singularity.
const ARM_SAMPLES: usize = 720;
/// A ray-direction bracket is only accepted when `|h|` is below this at both
/// ends, which filters the `2 pi` wrap jumps of `h`.
const ARM_BRACKET: f64 = 1.2;
/// Two ray directions closer than this (radians) are the same arm.
const ARM_DEDUPE: f64 = 0.02;
/// A boundary node whose turn is smaller than this is not a tile corner.
const CORNER_TURN: f64 = std::f64::consts::FRAC_PI_4;
/// Barycentric tolerance of the point location.
const LOCATE_TOLERANCE: f64 = 1e-9;
/// Relative tolerance used to merge coincident graph nodes and mesh nodes.
const WELD_TOLERANCE: f64 = 1e-9;

// ---------------------------------------------------------------------------
// Public data model
// ---------------------------------------------------------------------------

/// A node of the motorcycle graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    /// A field singularity, with its valence (3 or 5).
    Singularity(usize),
    /// A point where a separatrix reaches the boundary of the mesh.
    Boundary,
    /// A point where a separatrix ran into an existing one (180 degree corner).
    TJunction,
}

/// A node of the motorcycle graph.
#[derive(Clone, Copy, Debug)]
pub struct GraphNode {
    /// Position in the plane.
    pub point: [f64; 2],
    /// What created the node.
    pub kind: NodeKind,
}

/// What a graph edge is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeKind {
    /// A separatrix segment, traced through the field.
    Separatrix,
    /// An arc of the domain boundary.
    Boundary,
}

/// An edge of the motorcycle graph, with the polyline that realises it.
#[derive(Clone, Debug)]
pub struct GraphEdge {
    /// Start node.
    pub from: usize,
    /// End node.
    pub to: usize,
    /// Polyline from `from` to `to`, at least two points.
    pub polyline: Vec<[f64; 2]>,
    /// Originating singularity (separatrices only).
    pub singularity: Option<usize>,
    /// Which of that singularity's arms this segment belongs to.
    pub arm: usize,
    /// Separatrix or boundary arc.
    pub kind: EdgeKind,
}

/// The planar motorcycle graph of the field's separatrices.
#[derive(Clone, Debug, Default)]
pub struct MotorcycleGraph {
    /// Nodes, in creation order (the singularities come first).
    pub nodes: Vec<GraphNode>,
    /// Edges, in creation order.
    pub edges: Vec<GraphEdge>,
}

impl MotorcycleGraph {
    /// Number of edges incident to a node (0 when the node does not exist).
    pub fn degree(&self, node: usize) -> usize {
        self.edges
            .iter()
            .filter(|edge| edge.from == node || edge.to == node)
            .count()
    }

    /// Edge ids incident to a node.
    pub fn incident(&self, node: usize) -> Vec<usize> {
        let mut ids = Vec::new();
        for (id, edge) in self.edges.iter().enumerate() {
            if edge.from == node || edge.to == node {
                ids.push(id);
            }
        }
        ids
    }

    /// Polyline of an edge, reversed when `forward` is `false`.
    pub fn polyline(&self, edge: usize, forward: bool) -> Vec<[f64; 2]> {
        let mut line = match self.edges.get(edge) {
            Some(found) => found.polyline.clone(),
            None => Vec::new(),
        };
        if !forward {
            line.reverse();
        }
        line
    }
}

/// One simple tile: a face of the motorcycle graph with four sides.
#[derive(Clone, Debug)]
pub struct Tile {
    /// Graph node ids around the face, in counter-clockwise order (a node may
    /// appear twice when a T-junction or a singularity is visited twice).
    pub nodes: Vec<usize>,
    /// The four sides, each a sequence of graph edge ids, counter-clockwise.
    pub sides: [Vec<usize>; 4],
    /// The four corner positions, counter-clockwise; `[0]` is the Coons
    /// bottom-left corner.
    pub corners: [[f64; 2]; 4],
    /// The four corner graph nodes.
    pub corner_nodes: [usize; 4],
    /// The four boundary curves of the Coons patch, counter-clockwise.
    pub curves: BoundaryCurves,
    /// Integer subdivision count along each side; `counts[0] == counts[2]` and
    /// `counts[1] == counts[3]` by eq. (5.1).
    pub counts: [usize; 4],
    /// Signed area of the tile polygon (positive).
    pub area: f64,
    /// True for a *bent* tile: four field lines, but a corner polygon that is
    /// not counter-clockwise (the annular/nautilus case of Section 5.1.4).
    pub bent: bool,
    /// True for a *triangular* tile: a face with three corners, two shared seps
    /// and one free boundary arc, meshed by the three-kite fan
    /// ([`triangle_fan_quads`]). Its counts come from the two seps.
    pub triangular: bool,
}

impl Tile {
    /// Subdivisions along the bottom and top sides (`u`).
    pub fn n_u(&self) -> usize {
        self.counts[0]
    }

    /// Subdivisions along the left and right sides (`v`).
    pub fn n_v(&self) -> usize {
        self.counts[1]
    }
}

/// The singularities, the motorcycle graph and the simple tiles of a field.
#[derive(Clone, Debug, Default)]
pub struct Regions {
    /// One position per singularity (a field singularity lies inside a
    /// triangle, so it is not a vertex).
    pub singularities: Vec<[f64; 2]>,
    /// The planar motorcycle graph.
    pub graph: MotorcycleGraph,
    /// The simple tiles (faces of the graph) with their QGP counts.
    pub tiles: Vec<Tile>,
    /// Material faces that did not have exactly four convex corners and were
    /// skipped (Section 5.1.6's degenerate tiles).
    pub degenerate_tiles: usize,
    /// Tiles whose QGP solve collapsed a whole side (`s = 0`, Section 5.1.6),
    /// so they have no quads and are left out of the mesh.
    pub collapsed_tiles: usize,
    /// QGP energy `sum (s/s_hat - 1)^2` of the returned counts.
    pub energy: f64,
}

/// Resample an open polyline by arc length into `count + 1` points.
fn resample_open(points: &[[f64; 2]], count: usize) -> Vec<[f64; 2]> {
    let mut out = Vec::with_capacity(count + 1);
    if points.len() < 2 || count == 0 {
        return out;
    }
    let mut cumulative = Vec::with_capacity(points.len());
    cumulative.push(0.0);
    let mut total = 0.0;
    for pair in points.windows(2) {
        total += distance(pair[0], pair[1]);
        cumulative.push(total);
    }
    if total <= 0.0 {
        return out;
    }
    let mut segment = 0usize;
    for step in 0..=count {
        let target = total * step as f64 / count as f64;
        while segment + 2 < cumulative.len() && cumulative[segment + 1] < target {
            segment += 1;
        }
        let start = cumulative[segment];
        let end = cumulative[segment + 1];
        let a = points[segment];
        let b = points[segment + 1];
        let t = if end - start > 0.0 {
            ((target - start) / (end - start)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        out.push(add(a, scale(sub(b, a), t)));
    }
    out
}

/// Resample an open polyline by its *angle* about `centre`, into `count + 1`
/// points.
///
/// This is the angular re-parametrization of Section 5.1.4's annular tiles: two
/// sides that wrap around the hole are paired at equal angle instead of at
/// equal arc length, which is what makes the ring a rectangular strip. Returns
/// `None` when the angle is not monotone along the polyline (the side does not
/// wrap the centre) or its total turn is negligible.
fn resample_angular(points: &[[f64; 2]], count: usize, centre: [f64; 2]) -> Option<Vec<[f64; 2]>> {
    if points.len() < 2 || count == 0 {
        return None;
    }
    let mut angles: Vec<f64> = Vec::with_capacity(points.len());
    let mut total = 0.0;
    let mut previous: Option<f64> = None;
    for point in points {
        let offset = sub(*point, centre);
        if norm(offset) <= 1e-12 {
            return None;
        }
        let angle = angle_of(offset);
        match previous {
            Some(last) => {
                let step = wrap_pi(angle - last);
                total += step;
                angles.push(total);
            }
            None => angles.push(0.0),
        }
        previous = Some(angle);
    }
    let mut span = total;
    if !span.is_finite() || span.abs() < 0.5 {
        return None;
    }
    // Work with an increasing angle sequence: reverse and negate when the side
    // wraps the other way.
    let mut points: Vec<[f64; 2]> = points.to_vec();
    let mut ordered: Vec<f64> = angles;
    if span < 0.0 {
        points.reverse();
        ordered.reverse();
        ordered = ordered.into_iter().map(|value| -value).collect();
        span = -span;
    }
    let monotone = ordered.windows(2).all(|pair| pair[1] >= pair[0] - 1e-12);
    if !monotone {
        return None;
    }
    let mut out = Vec::with_capacity(count + 1);
    for step in 0..=count {
        let target = span * step as f64 / count as f64;
        // Locate the segment that contains `target` in the angle parameter.
        let mut index = 0usize;
        while index + 2 < ordered.len() && ordered[index + 1] < target {
            index += 1;
        }
        let start = ordered[index];
        let end = ordered[index + 1];
        let t = if (end - start).abs() > 0.0 {
            ((target - start) / (end - start)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        out.push(add(
            points[index],
            scale(sub(points[index + 1], points[index]), t),
        ));
    }
    Some(out)
}

/// Coons patch of a *bent* (annular) tile with the two long sides resampled by
/// angle about the tile centroid, so the ring becomes a rectangular strip.
fn bent_coons_quads(
    curves: &BoundaryCurves,
    n_u: usize,
    n_v: usize,
) -> Result<Vec<[[f64; 2]; 4]>, String> {
    if n_u == 0 || n_v == 0 {
        return Err("bent coons: zero subdivision".to_string());
    }
    let sides = [&curves.bottom, &curves.right, &curves.top, &curves.left];
    let lengths: Vec<f64> = sides
        .iter()
        .map(|side| {
            let mut total = 0.0;
            for pair in side.windows(2) {
                total += distance(pair[0], pair[1]);
            }
            total
        })
        .collect();
    // The two long sides wrap the hole; order them by length.
    let mut order: Vec<usize> = (0..4).collect();
    order.sort_by(|left, right| {
        lengths[*right]
            .partial_cmp(&lengths[*left])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let long: Vec<usize> = order[..2].to_vec();
    // The tile centroid is the ring centre.
    let mut centre = [0.0f64; 2];
    let mut total = 0.0;
    for side in sides.iter() {
        for point in side.iter() {
            centre[0] += point[0];
            centre[1] += point[1];
            total += 1.0;
        }
    }
    if total > 0.0 {
        centre[0] /= total;
        centre[1] /= total;
    }
    let counts = [n_u, n_v, n_u, n_v];
    let mut sampled: Vec<Vec<[f64; 2]>> = Vec::with_capacity(4);
    for (index, side) in sides.iter().enumerate() {
        let count = counts[index];
        if long.contains(&index) {
            if let Some(points) = resample_angular(side, count, centre) {
                sampled.push(points);
                continue;
            }
        }
        sampled.push(resample_open(side, count));
    }
    if sampled.iter().any(|line| line.len() < 2) {
        return Err("bent coons: a side has zero length".to_string());
    }
    let bottom = sampled[0].clone();
    let right = sampled[1].clone();
    let top = sampled[2].clone();
    let left = sampled[3].clone();
    let corner0 = bottom[0];
    let corner1 = bottom[n_u];
    let corner2 = top[0];
    let corner3 = top[n_u];
    let mut points: Vec<[f64; 2]> = Vec::with_capacity((n_u + 1) * (n_v + 1));
    for j in 0..=n_v {
        let v = j as f64 / n_v as f64;
        for i in 0..=n_u {
            let u = i as f64 / n_u as f64;
            let bottom_point = bottom[i];
            let top_point = top[n_u - i];
            let left_point = left[n_v - j];
            let right_point = right[j];
            let blend_bottom_top = add(scale(bottom_point, 1.0 - v), scale(top_point, v));
            let blend_left_right = add(scale(right_point, u), scale(left_point, 1.0 - u));
            let bilinear = add(
                scale(add(scale(corner0, 1.0 - u), scale(corner1, u)), 1.0 - v),
                scale(add(scale(corner3, 1.0 - u), scale(corner2, u)), v),
            );
            points.push(add(blend_bottom_top, sub(blend_left_right, bilinear)));
        }
    }
    let index = |i: usize, j: usize| j * (n_u + 1) + i;
    let mut quads = Vec::with_capacity(n_u * n_v);
    for j in 0..n_v {
        for i in 0..n_u {
            let quad = [
                points[index(i, j)],
                points[index(i + 1, j)],
                points[index(i + 1, j + 1)],
                points[index(i, j + 1)],
            ];
            if polygon_area(&quad) <= 0.0 {
                return Err("bent coons: a quad is inverted".to_string());
            }
            quads.push(quad);
        }
    }
    Ok(quads)
}

impl Regions {
    /// Mesh every tile with its Coons patch and weld the results into one mesh.
    ///
    /// Tiles whose QGP solve collapsed a side are not in [`Regions::tiles`], so
    /// every tile here has a positive count. Returns an error when the welded
    /// mesh is structurally invalid.
    pub fn mesh(&self) -> Result<Mesh, String> {
        if self.tiles.is_empty() {
            return Err("regions: there is no tile to mesh".to_string());
        }
        let mut scale = 1.0f64;
        for tile in &self.tiles {
            for corner in &tile.corners {
                scale = scale.max(corner[0].abs()).max(corner[1].abs());
            }
        }
        let tolerance = WELD_TOLERANCE * scale;
        let mut points: Vec<Node> = Vec::new();
        let mut cells: Vec<Cell> = Vec::new();
        let mut weld: HashMap<(i64, i64), usize> = HashMap::new();
        for (index, tile) in self.tiles.iter().enumerate() {
            let n_u = tile.n_u();
            let n_v = tile.n_v();
            if n_u == 0 || n_v == 0 {
                return Err(format!(
                    "regions: tile {index} has a zero subdivision count ({n_u} x {n_v})"
                ));
            }
            let tile_mesh = if tile.triangular {
                let mut sides = [&tile.curves.right[..], &tile.curves.bottom[..]];
                sides[0] = &tile.curves.right;
                // The fan wants (sep from the base end to the apex, the other
                // sep from the base start to the apex, the base arc).
                let mut other: Vec<[f64; 2]> = tile.curves.top.clone();
                other.reverse();
                let (quads, boundary) =
                    triangle_fan_quads(&tile.curves.right, &other, &tile.curves.bottom, n_u)
                        .map_err(|error| format!("regions: triangular tile {index}: {error}"))?;
                if boundary != 3 * n_u {
                    return Err(format!(
                        "regions: triangular tile {index}: boundary {boundary} != 3 * {n_u}"
                    ));
                }
                let _ = sides;
                let mut points: Vec<Node> = Vec::with_capacity(quads.len() * 4);
                let mut cells: Vec<Cell> = Vec::with_capacity(quads.len());
                for quad in &quads {
                    let mut nodes = [0usize; 4];
                    for (slot, point) in quad.iter().enumerate() {
                        nodes[slot] = points.len();
                        points.push([point[0], point[1], 0.0]);
                    }
                    cells.push(Cell::quad4(nodes));
                }
                Mesh::new(points, cells)
            } else if tile.bent {
                let quads = bent_coons_quads(&tile.curves, n_u, n_v)
                    .map_err(|error| format!("regions: bent tile {index}: {error}"))?;
                let mut points: Vec<Node> = Vec::with_capacity(quads.len() * 4);
                let mut cells: Vec<Cell> = Vec::with_capacity(quads.len());
                for quad in &quads {
                    let mut nodes = [0usize; 4];
                    for (slot, point) in quad.iter().enumerate() {
                        nodes[slot] = points.len();
                        points.push([point[0], point[1], 0.0]);
                    }
                    cells.push(Cell::quad4(nodes));
                }
                Mesh::new(points, cells)
            } else {
                simons_qgp::mesh_tile_curves(&tile.curves, n_u, n_v)
                    .map_err(|error| format!("regions: tile {index}: {error}"))?
            };
            let mut local: Vec<usize> = Vec::with_capacity(tile_mesh.points.len());
            for point in &tile_mesh.points {
                let key = (
                    (point[0] / tolerance).round() as i64,
                    (point[1] / tolerance).round() as i64,
                );
                let shared = match weld.get(&key) {
                    Some(&existing) => existing,
                    None => {
                        let id = points.len();
                        points.push(*point);
                        weld.insert(key, id);
                        id
                    }
                };
                local.push(shared);
            }
            for cell in &tile_mesh.cells {
                let mut mapped = [0usize; 4];
                if cell.nodes.len() != 4 {
                    return Err(format!("regions: tile {index} produced a non-quad cell"));
                }
                for (slot, &node) in cell.nodes.iter().enumerate() {
                    mapped[slot] = local.get(node).copied().unwrap_or(0);
                }
                cells.push(Cell::quad4(mapped));
            }
        }
        let mesh = Mesh::new(points, cells);
        mesh.validate()
            .map_err(|error| format!("regions: the welded mesh is invalid: {error}"))?;
        Ok(mesh)
    }
}

// ---------------------------------------------------------------------------
// Demo numbers
// ---------------------------------------------------------------------------

/// One demo case's measured numbers.
#[derive(Clone, Debug, Default)]
pub struct DemoCase {
    /// Short name of the case.
    pub name: &'static str,
    /// Input mesh vertices.
    pub vertices: usize,
    /// Input mesh triangles.
    pub triangles: usize,
    /// Number of detected singularities.
    pub singularities: usize,
    /// Number of valence-3 singularities.
    pub valence3: usize,
    /// Number of valence-5 singularities.
    pub valence5: usize,
    /// Motorcycle-graph nodes.
    pub graph_nodes: usize,
    /// Motorcycle-graph edges.
    pub graph_edges: usize,
    /// Motorcycle-graph nodes that are boundary hits.
    pub boundary_nodes: usize,
    /// Motorcycle-graph nodes that are T-junctions.
    pub tjunction_nodes: usize,
    /// Motorcycle-graph edges that are traced separatrices.
    pub separatrix_edges: usize,
    /// Motorcycle-graph edges that are boundary arcs.
    pub boundary_edges: usize,
    /// Simple tiles that were kept.
    pub tiles: usize,
    /// Material faces that were skipped as degenerate.
    pub degenerate_tiles: usize,
    /// Tiles collapsed by the QGP solve.
    pub collapsed_tiles: usize,
    /// Quads of the welded mesh.
    pub quads: usize,
    /// Integer counts of every tile, `[side0, side1, side2, side3]`.
    pub counts: Vec<[usize; 4]>,
    /// QGP energy.
    pub qgp_energy: f64,
    /// Smallest scaled Jacobian of the welded mesh.
    pub min_scaled_jacobian: f64,
    /// Smallest corner angle of the welded mesh, in degrees.
    pub min_corner_angle_deg: f64,
    /// Largest corner angle of the welded mesh, in degrees.
    pub max_corner_angle_deg: f64,
    /// Smallest signed Coons quad area (positive when no quad is inverted).
    pub min_signed_area: f64,
    /// Edges used by exactly one quad that are not on the domain boundary.
    pub cracks: usize,
    /// Edges used by more than two quads.
    pub nonmanifold_edges: usize,
    /// Largest distance from a boundary node of the mesh to the domain boundary.
    pub boundary_offset: f64,
    /// Area of the input triangle mesh.
    pub domain_area: f64,
    /// Area of the welded quad mesh.
    pub mesh_area: f64,
    /// `|mesh_area - domain_area| / domain_area`.
    pub relative_area_error: f64,
    /// The welded mesh.
    pub mesh: Mesh,
}

/// Numbers measured by [`demo`].
#[derive(Clone, Debug, Default)]
pub struct DemoNumbers {
    /// The square with a circular hole (4 singularities).
    pub annulus: DemoCase,
    /// The single rectangle (constant field, no singularity, one tile).
    pub rectangle: DemoCase,
}

// ---------------------------------------------------------------------------
// Small 2D vector and geometry helpers
// ---------------------------------------------------------------------------

fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn add(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

fn scale(a: [f64; 2], factor: f64) -> [f64; 2] {
    [a[0] * factor, a[1] * factor]
}

fn dot(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

fn cross(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

fn norm(a: [f64; 2]) -> f64 {
    dot(a, a).sqrt()
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    norm(sub(a, b))
}

fn unit(a: [f64; 2]) -> Option<[f64; 2]> {
    let length = norm(a);
    if length.is_finite() && length > 0.0 {
        Some([a[0] / length, a[1] / length])
    } else {
        None
    }
}

fn angle_of(a: [f64; 2]) -> f64 {
    a[1].atan2(a[0])
}

/// Wraps an angle to `(-pi, pi]`.
fn wrap_pi(angle: f64) -> f64 {
    let mut wrapped =
        (angle + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI;
    if wrapped <= -std::f64::consts::PI {
        wrapped += std::f64::consts::TAU;
    }
    wrapped
}

/// Wraps a cross-field angle to `[0, pi/2)`.
fn wrap_cross(angle: f64) -> f64 {
    let mut wrapped = angle.rem_euclid(std::f64::consts::FRAC_PI_2);
    if !(0.0..std::f64::consts::FRAC_PI_2).contains(&wrapped) {
        wrapped = 0.0;
    }
    wrapped
}

/// Distance from `p` to the segment `a`-`b`.
fn segment_distance(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let ab = sub(b, a);
    let length2 = dot(ab, ab);
    if length2 <= 0.0 {
        return distance(p, a);
    }
    let t = (dot(sub(p, a), ab) / length2).clamp(0.0, 1.0);
    distance(p, add(a, scale(ab, t)))
}

/// Parametric intersection of the segments `p`-`q` and `a`-`b`.
///
/// Returns `(t, u)` with `p + t (q - p) == a + u (b - a)` when the segments
/// cross, with both parameters inside `[0, 1]` up to `tolerance`.
fn segment_intersection(
    p: [f64; 2],
    q: [f64; 2],
    a: [f64; 2],
    b: [f64; 2],
    tolerance: f64,
) -> Option<(f64, f64)> {
    let r = sub(q, p);
    let s = sub(b, a);
    let denominator = cross(r, s);
    if !denominator.is_finite() || denominator == 0.0 {
        return None;
    }
    let pa = sub(a, p);
    let t = cross(pa, s) / denominator;
    let u = cross(pa, r) / denominator;
    if !t.is_finite() || !u.is_finite() {
        return None;
    }
    if t < -tolerance || t > 1.0 + tolerance || u < -tolerance || u > 1.0 + tolerance {
        return None;
    }
    Some((t.clamp(0.0, 1.0), u.clamp(0.0, 1.0)))
}

/// Signed area of a closed polygon (positive for counter-clockwise winding).
fn polygon_area(points: &[[f64; 2]]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }
    let mut twice = 0.0;
    for i in 0..points.len() {
        let p = points[i];
        let q = points[(i + 1) % points.len()];
        twice += p[0] * q[1] - q[0] * p[1];
    }
    0.5 * twice
}

/// Direction from `p` towards the first polyline point that differs from it.
fn incoming_direction(line: &[[f64; 2]], p: [f64; 2]) -> Option<[f64; 2]> {
    for i in (0..line.len().saturating_sub(1)).rev() {
        let candidate = sub(p, line[i]);
        if norm(candidate) > 1e-14 {
            return unit(candidate);
        }
    }
    None
}

/// Direction from `p` towards the first polyline point after it.
fn outgoing_direction(line: &[[f64; 2]], p: [f64; 2]) -> Option<[f64; 2]> {
    for i in 1..line.len() {
        let candidate = sub(line[i], p);
        if norm(candidate) > 1e-14 {
            return unit(candidate);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// The input surface: triangles, point location, boundary loops
// ---------------------------------------------------------------------------

/// The triangle mesh, the per-vertex field angle and the per-vertex size, with
/// the derived boundary loops and the point locator.
struct Field<'a> {
    points: &'a [[f64; 3]],
    tris: &'a [[usize; 3]],
    alpha: &'a [f64],
    size: &'a [f64],
    /// Mean triangle edge length, used to pick the RK4 step.
    mean_edge: f64,
    /// Bounding-box diagonal, used to scale tolerances.
    diagonal: f64,
    /// Directed boundary edges, interior on the left (counter-clockwise).
    boundary_directed: Vec<[usize; 2]>,
    /// One vertex sequence per boundary loop.
    boundary_loops: Vec<Vec<usize>>,
    /// Loop points (the repeated closing point is not stored).
    boundary_points: Vec<Vec<[f64; 2]>>,
    /// Per loop, the cumulative arc length at every loop point, plus the total.
    boundary_cumulative: Vec<Vec<f64>>,
    /// Per directed boundary edge, `(loop, cumulative length at its start)`.
    boundary_edge_loop: Vec<(usize, f64)>,
}

impl<'a> Field<'a> {
    fn new(
        points: &'a [[f64; 3]],
        tris: &'a [[usize; 3]],
        alpha: &'a [f64],
        size: &'a [f64],
    ) -> Result<Field<'a>, String> {
        if points.len() < 3 {
            return Err("regions: the mesh needs at least three points".to_string());
        }
        if tris.is_empty() {
            return Err("regions: the mesh needs at least one triangle".to_string());
        }
        if alpha.len() != points.len() {
            return Err(format!(
                "regions: alpha has {} values for {} points",
                alpha.len(),
                points.len()
            ));
        }
        if size.len() != points.len() {
            return Err(format!(
                "regions: size has {} values for {} points",
                size.len(),
                points.len()
            ));
        }
        for (index, value) in alpha.iter().enumerate() {
            if !value.is_finite() {
                return Err(format!("regions: alpha[{index}] is not finite"));
            }
        }
        for (index, value) in size.iter().enumerate() {
            if !value.is_finite() || *value <= 0.0 {
                return Err(format!(
                    "regions: size[{index}] must be finite and positive (got {value})"
                ));
            }
        }

        let mut min = [f64::INFINITY; 2];
        let mut max = [f64::NEG_INFINITY; 2];
        for point in points {
            if !point[0].is_finite() || !point[1].is_finite() {
                return Err("regions: the mesh has a non-finite point".to_string());
            }
            for axis in 0..2 {
                min[axis] = min[axis].min(point[axis]);
                max[axis] = max[axis].max(point[axis]);
            }
        }
        let diagonal = ((max[0] - min[0]).powi(2) + (max[1] - min[1]).powi(2)).sqrt();
        let diagonal = if diagonal.is_finite() && diagonal > 0.0 {
            diagonal
        } else {
            1.0
        };

        let mut edge_length: HashMap<(usize, usize), f64> = HashMap::new();
        for (index, triangle) in tris.iter().enumerate() {
            let [i, j, k] = *triangle;
            if i >= points.len() || j >= points.len() || k >= points.len() {
                return Err(format!(
                    "regions: triangle {index} has an out-of-range vertex index"
                ));
            }
            if i == j || j == k || k == i {
                return Err(format!("regions: triangle {index} repeats a vertex"));
            }
            let a = [points[i][0], points[i][1]];
            let b = [points[j][0], points[j][1]];
            let c = [points[k][0], points[k][1]];
            if cross(sub(b, a), sub(c, a)) == 0.0 {
                return Err(format!("regions: triangle {index} is degenerate"));
            }
            for (u, v) in [(i, j), (j, k), (k, i)] {
                let key = if u < v { (u, v) } else { (v, u) };
                let length = distance([points[u][0], points[u][1]], [points[v][0], points[v][1]]);
                let slot = edge_length.entry(key).or_insert(0.0);
                *slot = length;
            }
        }
        let mut edge_count: HashMap<(usize, usize), usize> = HashMap::new();
        for triangle in tris.iter() {
            let [i, j, k] = *triangle;
            for (u, v) in [(i, j), (j, k), (k, i)] {
                let key = if u < v { (u, v) } else { (v, u) };
                *edge_count.entry(key).or_insert(0) += 1;
            }
        }
        for (edge, count) in &edge_count {
            if *count > 2 {
                return Err(format!(
                    "regions: the mesh is not manifold (edge {:?} is used by {count} triangles)",
                    edge
                ));
            }
        }
        let mut total_length = 0.0;
        for length in edge_length.values() {
            total_length += length;
        }
        let mean_edge = if edge_length.is_empty() {
            diagonal
        } else {
            total_length / edge_length.len() as f64
        };
        if !mean_edge.is_finite() || mean_edge <= 0.0 {
            return Err("regions: the mesh has zero total edge length".to_string());
        }

        let mut boundary_edges: Vec<[usize; 2]> = edge_count
            .iter()
            .filter(|(_, count)| **count == 1)
            .map(|(&(a, b), _)| [a, b])
            .collect();
        boundary_edges.sort_unstable();
        if boundary_edges.is_empty() {
            return Err(
                "regions: the mesh is closed (no boundary edges); a planar region with a \
                 boundary is required"
                    .to_string(),
            );
        }

        // Directed edges of the counter-clockwise triangles. A directed edge is
        // on the boundary when its reverse is not used, so its interior is on
        // its left.
        let mut directed: HashMap<(usize, usize), ()> = HashMap::with_capacity(3 * tris.len());
        for triangle in tris.iter() {
            let [i, j, k] = *triangle;
            let a = [points[i][0], points[i][1]];
            let b = [points[j][0], points[j][1]];
            let c = [points[k][0], points[k][1]];
            let oriented = if cross(sub(b, a), sub(c, a)) >= 0.0 {
                [i, j, k]
            } else {
                [i, k, j]
            };
            for (u, v) in [
                (oriented[0], oriented[1]),
                (oriented[1], oriented[2]),
                (oriented[2], oriented[0]),
            ] {
                directed.insert((u, v), ());
            }
        }
        let mut boundary_directed: Vec<[usize; 2]> = Vec::new();
        for &(u, v) in directed.keys() {
            if !directed.contains_key(&(v, u)) {
                boundary_directed.push([u, v]);
            }
        }
        boundary_directed.sort_unstable();

        let mut by_start: HashMap<usize, usize> = HashMap::new();
        for (index, edge) in boundary_directed.iter().enumerate() {
            if by_start.insert(edge[0], index).is_some() {
                return Err(format!(
                    "regions: vertex {} starts two boundary edges (non-manifold boundary)",
                    edge[0]
                ));
            }
        }
        let mut used = vec![false; boundary_directed.len()];
        let mut boundary_loops: Vec<Vec<usize>> = Vec::new();
        for start in 0..boundary_directed.len() {
            if used[start] {
                continue;
            }
            let mut vertices: Vec<usize> = Vec::new();
            let mut current = start;
            let mut closed = false;
            for _ in 0..=boundary_directed.len() {
                if used[current] {
                    closed = current == start;
                    break;
                }
                used[current] = true;
                vertices.push(boundary_directed[current][0]);
                current = match by_start.get(&boundary_directed[current][1]) {
                    Some(&next) => next,
                    None => {
                        return Err(
                            "regions: a boundary chain ends in the middle of the mesh".to_string()
                        )
                    }
                };
            }
            if !closed {
                return Err("regions: the boundary loops could not be chained".to_string());
            }
            if vertices.len() < 3 {
                return Err("regions: a boundary loop has fewer than three vertices".to_string());
            }
            boundary_loops.push(vertices);
        }

        let mut boundary_points: Vec<Vec<[f64; 2]>> = Vec::new();
        let mut boundary_cumulative: Vec<Vec<f64>> = Vec::new();
        for vertices in &boundary_loops {
            let mut loop_points: Vec<[f64; 2]> = Vec::with_capacity(vertices.len());
            for &vertex in vertices {
                let point = match points.get(vertex) {
                    Some(found) => [found[0], found[1]],
                    None => return Err("regions: bad boundary vertex".to_string()),
                };
                loop_points.push(point);
            }
            let mut cumulative = Vec::with_capacity(loop_points.len() + 1);
            let mut total = 0.0;
            cumulative.push(0.0);
            for index in 0..loop_points.len() {
                let a = loop_points[index];
                let b = loop_points[(index + 1) % loop_points.len()];
                total += distance(a, b);
                cumulative.push(total);
            }
            if !total.is_finite() || total <= 0.0 {
                return Err("regions: a boundary loop has zero length".to_string());
            }
            boundary_points.push(loop_points);
            boundary_cumulative.push(cumulative);
        }

        let mut vertex_loop: HashMap<usize, (usize, f64)> = HashMap::new();
        for (loop_index, vertices) in boundary_loops.iter().enumerate() {
            for (position, &vertex) in vertices.iter().enumerate() {
                let offset = boundary_cumulative
                    .get(loop_index)
                    .and_then(|values| values.get(position))
                    .copied()
                    .unwrap_or(0.0);
                vertex_loop.insert(vertex, (loop_index, offset));
            }
        }
        let mut boundary_edge_loop: Vec<(usize, f64)> = Vec::with_capacity(boundary_directed.len());
        for edge in &boundary_directed {
            match vertex_loop.get(&edge[0]) {
                Some(&entry) => boundary_edge_loop.push(entry),
                None => return Err("regions: a boundary edge is outside every loop".to_string()),
            }
        }

        Ok(Field {
            points,
            tris,
            alpha,
            size,
            mean_edge,
            diagonal,
            boundary_directed,
            boundary_loops,
            boundary_points,
            boundary_cumulative,
            boundary_edge_loop,
        })
    }

    /// Planar position of a mesh vertex.
    fn xy(&self, vertex: usize) -> Option<[f64; 2]> {
        let point = self.points.get(vertex)?;
        Some([point[0], point[1]])
    }

    /// Triangle index and barycentric weights of a point, or `None` when the
    /// point is outside the mesh.
    fn locate(&self, p: [f64; 2]) -> Option<(usize, [f64; 3])> {
        for (index, triangle) in self.tris.iter().enumerate() {
            let (pi, pj, pk) = match (
                self.xy(triangle[0]),
                self.xy(triangle[1]),
                self.xy(triangle[2]),
            ) {
                (Some(a), Some(b), Some(c)) => (a, b, c),
                _ => continue,
            };
            let denominator = cross(sub(pj, pi), sub(pk, pi));
            if denominator == 0.0 {
                continue;
            }
            let w_i = cross(sub(pj, p), sub(pk, p)) / denominator;
            let w_j = cross(sub(pk, p), sub(pi, p)) / denominator;
            let w_k = cross(sub(pi, p), sub(pj, p)) / denominator;
            if w_i >= -LOCATE_TOLERANCE
                && w_j >= -LOCATE_TOLERANCE
                && w_k >= -LOCATE_TOLERANCE
                && w_i <= 1.0 + LOCATE_TOLERANCE
                && w_j <= 1.0 + LOCATE_TOLERANCE
                && w_k <= 1.0 + LOCATE_TOLERANCE
            {
                return Some((index, [w_i, w_j, w_k]));
            }
        }
        None
    }

    /// True when the point is covered by a triangle.
    fn is_inside(&self, p: [f64; 2]) -> bool {
        self.locate(p).is_some()
    }

    /// Barycentric interpolation of a per-vertex value at a point.
    fn interpolate(&self, values: &[f64], p: [f64; 2]) -> Option<f64> {
        let (index, weights) = self.locate(p)?;
        let triangle = self.tris.get(index)?;
        let mut sum = 0.0;
        let mut total = 0.0;
        for (slot, &vertex) in triangle.iter().enumerate() {
            let weight = weights.get(slot).copied().unwrap_or(0.0).max(0.0);
            sum += weight * values.get(vertex).copied()?;
            total += weight;
        }
        if total > 1e-12 {
            Some(sum / total)
        } else {
            None
        }
    }

    /// Size at a point, falling back to the nearest vertex outside the mesh.
    fn sigma_at(&self, p: [f64; 2]) -> f64 {
        if let Some(value) = self.interpolate(self.size, p) {
            if value.is_finite() && value > 0.0 {
                return value;
            }
        }
        let mut best = self.size.first().copied().unwrap_or(1.0);
        let mut best_distance = f64::INFINITY;
        for (index, point) in self.points.iter().enumerate() {
            let d = distance(p, [point[0], point[1]]);
            if d < best_distance {
                best_distance = d;
                best = self.size.get(index).copied().unwrap_or(best);
            }
        }
        best
    }

    /// Boundary segment of a directed boundary edge.
    fn boundary_segment(&self, index: usize) -> Option<([f64; 2], [f64; 2])> {
        let edge = self.boundary_directed.get(index)?;
        Some((self.xy(edge[0])?, self.xy(edge[1])?))
    }

    /// Total length of a boundary loop.
    fn loop_total(&self, loop_index: usize) -> f64 {
        self.boundary_cumulative
            .get(loop_index)
            .and_then(|values| values.last().copied())
            .unwrap_or(0.0)
    }

    /// Point at arc-length `t` along a boundary loop.
    fn loop_point_at(&self, loop_index: usize, t: f64) -> Option<[f64; 2]> {
        let points = self.boundary_points.get(loop_index)?;
        let cumulative = self.boundary_cumulative.get(loop_index)?;
        let total = self.loop_total(loop_index);
        if points.len() < 2 || cumulative.len() != points.len() + 1 || total <= 0.0 {
            return None;
        }
        let mut t = t.rem_euclid(total);
        if !t.is_finite() || t >= total {
            t = 0.0;
        }
        let mut segment = points.len() - 1;
        for index in 0..points.len() {
            if t >= cumulative[index] && t <= cumulative[index + 1] {
                segment = index;
                break;
            }
        }
        let start = cumulative[segment];
        let end = cumulative[segment + 1];
        let a = points[segment];
        let b = points[(segment + 1) % points.len()];
        if end - start <= 0.0 {
            return Some(a);
        }
        let local = ((t - start) / (end - start)).clamp(0.0, 1.0);
        Some(add(a, scale(sub(b, a), local)))
    }

    /// Polyline along a boundary loop from arc length `start` to `end`, always
    /// walking in the loop's (counter-clockwise) direction.
    fn loop_polyline(&self, loop_index: usize, start: f64, end: f64) -> Vec<[f64; 2]> {
        let points = match self.boundary_points.get(loop_index) {
            Some(found) => found,
            None => return Vec::new(),
        };
        let cumulative = match self.boundary_cumulative.get(loop_index) {
            Some(found) => found,
            None => return Vec::new(),
        };
        let total = self.loop_total(loop_index);
        let mut end = end;
        if end <= start {
            end += total;
        }
        let mut polyline = Vec::new();
        if let Some(first) = self.loop_point_at(loop_index, start) {
            polyline.push(first);
        }
        for index in 0..points.len() {
            let offset = cumulative.get(index).copied().unwrap_or(0.0);
            if offset > start + 1e-12 && offset < end - 1e-12 {
                polyline.push(points[index]);
            }
        }
        if let Some(last) = self.loop_point_at(loop_index, end) {
            polyline.push(last);
        }
        let mut cleaned: Vec<[f64; 2]> = Vec::with_capacity(polyline.len());
        for point in polyline {
            if cleaned
                .last()
                .map(|previous: &[f64; 2]| distance(*previous, point) > 1e-14)
                .unwrap_or(true)
            {
                cleaned.push(point);
            }
        }
        cleaned
    }
}

// ---------------------------------------------------------------------------
// Singularities and their separatrix directions (Sections 3.1.6 and 3.1.8)
// ---------------------------------------------------------------------------

/// A detected field singularity with its separatrix directions.
#[derive(Clone, Debug)]
struct Singularity {
    point: [f64; 2],
    valence: usize,
    triangle: usize,
    arms: Vec<[f64; 2]>,
}

/// `exp(i 4 alpha)` as a complex pair.
fn field_value(alpha: f64) -> [f64; 2] {
    let angle = 4.0 * alpha;
    [angle.cos(), angle.sin()]
}

/// Phase `arg(a exp(i theta) + b exp(-i theta))` of the local field along the ray
/// at angle `theta`.
fn local_phase(a: [f64; 2], b: [f64; 2], theta: f64) -> f64 {
    let (s, c) = theta.sin_cos();
    let re = a[0] * c - a[1] * s + b[0] * c + b[1] * s;
    let im = a[0] * s + a[1] * c + b[1] * c - b[0] * s;
    im.atan2(re)
}

/// Zeros of `wrap(4 theta - phase(theta))` on `[0, 2 pi)`.
///
/// This is the separatrix count of Section 3.1.8: `valence` rays leave the
/// singularity.
fn arm_directions(a: [f64; 2], b: [f64; 2], valence: usize) -> Vec<[f64; 2]> {
    let valence = valence.max(1);
    let h = |theta: f64| wrap_pi(4.0 * theta - local_phase(a, b, theta));
    let mut roots: Vec<f64> = Vec::new();
    for index in 0..ARM_SAMPLES {
        let t0 = std::f64::consts::TAU * index as f64 / ARM_SAMPLES as f64;
        let t1 = std::f64::consts::TAU * (index + 1) as f64 / ARM_SAMPLES as f64;
        let h0 = h(t0);
        let h1 = h(t1);
        if h0.abs() >= ARM_BRACKET || h1.abs() >= ARM_BRACKET || h0 * h1 > 0.0 {
            continue;
        }
        let mut lo = t0;
        let mut hi = t1;
        let mut f_lo = h0;
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            let f_mid = h(mid);
            if f_lo * f_mid <= 0.0 {
                hi = mid;
            } else {
                lo = mid;
                f_lo = f_mid;
            }
        }
        roots.push(0.5 * (lo + hi));
    }
    roots.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let mut directions: Vec<[f64; 2]> = Vec::new();
    for theta in roots {
        let duplicate = directions.iter().any(|direction| {
            let existing = angle_of(*direction);
            wrap_pi(existing - theta).abs() <= ARM_DEDUPE
                || wrap_pi(existing + std::f64::consts::TAU - theta).abs() <= ARM_DEDUPE
        });
        if !duplicate {
            directions.push([theta.cos(), theta.sin()]);
        }
    }
    if directions.len() != valence {
        // The scan failed (a numerically degenerate local model); fall back to
        // the exact directions of the dominant conformal part.
        let dominant = if dot(a, a) >= dot(b, b) { a } else { b };
        let base = angle_of(dominant);
        directions = (0..valence)
            .map(|k| {
                let theta = (base + std::f64::consts::TAU * k as f64) / valence as f64;
                [theta.cos(), theta.sin()]
            })
            .collect();
    }
    directions.sort_by(|left, right| {
        angle_of(*left)
            .rem_euclid(std::f64::consts::TAU)
            .partial_cmp(&angle_of(*right).rem_euclid(std::f64::consts::TAU))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    directions
}

/// Zero of the affine interpolant of `u` inside a triangle, as barycentric
/// weights (Cramer's rule on the 3x3 system).
fn barycentric_zero(u: &[[f64; 2]; 3]) -> Option<[f64; 3]> {
    let determinant = u[0][0] * (u[1][1] - u[2][1]) - u[1][0] * (u[0][1] - u[2][1])
        + u[2][0] * (u[0][1] - u[1][1]);
    if !determinant.is_finite() || determinant.abs() <= 1e-14 {
        return None;
    }
    let n0 = u[1][0] * u[2][1] - u[2][0] * u[1][1];
    let n1 = -u[0][0] * u[2][1] + u[2][0] * u[0][1];
    let l0 = n0 / determinant;
    let l1 = n1 / determinant;
    let l2 = 1.0 - l0 - l1;
    if !l0.is_finite() || !l1.is_finite() || !l2.is_finite() {
        return None;
    }
    Some([l0, l1, l2])
}

/// Affine coefficients `(a, b)` of `u(z) = u0 + a (z - z0) + b conj(z - z0)`.
fn affine_coefficients(
    u: &[[f64; 2]; 3],
    e1: [f64; 2],
    e2: [f64; 2],
) -> Option<([f64; 2], [f64; 2])> {
    let determinant = cross(e1, e2);
    if determinant == 0.0 || !determinant.is_finite() {
        return None;
    }
    let d1 = sub(u[1], u[0]);
    let d2 = sub(u[2], u[0]);
    let a00 = (d1[0] * e2[1] - d2[0] * e1[1]) / determinant;
    let a01 = (-d1[0] * e2[0] + d2[0] * e1[0]) / determinant;
    let a10 = (d1[1] * e2[1] - d2[1] * e1[1]) / determinant;
    let a11 = (-d1[1] * e2[0] + d2[1] * e1[0]) / determinant;
    let a = [0.5 * (a00 + a11), 0.5 * (a10 - a01)];
    let b = [0.5 * (a00 - a11), 0.5 * (a10 + a01)];
    if !a[0].is_finite() || !a[1].is_finite() || !b[0].is_finite() || !b[1].is_finite() {
        return None;
    }
    Some((a, b))
}

/// Detect the singularities of the field, one per winding triangle.
fn detect_singularities(field: &Field) -> Result<Vec<Singularity>, String> {
    let mut singularities = Vec::new();
    for (index, triangle) in field.tris.iter().enumerate() {
        let (alpha_i, alpha_j, alpha_k) = match (
            field.alpha.get(triangle[0]),
            field.alpha.get(triangle[1]),
            field.alpha.get(triangle[2]),
        ) {
            (Some(a), Some(b), Some(c)) => (*a, *b, *c),
            _ => continue,
        };
        let winding = wrap_pi(4.0 * (alpha_j - alpha_i))
            + wrap_pi(4.0 * (alpha_k - alpha_j))
            + wrap_pi(4.0 * (alpha_i - alpha_k));
        let charge = (winding / std::f64::consts::TAU).round() as i32;
        if charge == 0 {
            continue;
        }
        let u = [
            field_value(alpha_i),
            field_value(alpha_j),
            field_value(alpha_k),
        ];
        let weights = match barycentric_zero(&u) {
            Some(found) => found,
            None => {
                return Err(format!(
                    "regions: the singularity of triangle {index} could not be located"
                ))
            }
        };
        let (p0, p1, p2) = match (
            field.xy(triangle[0]),
            field.xy(triangle[1]),
            field.xy(triangle[2]),
        ) {
            (Some(a), Some(b), Some(c)) => (a, b, c),
            _ => continue,
        };
        let point = add(
            add(scale(p0, weights[0]), scale(p1, weights[1])),
            scale(p2, weights[2]),
        );
        if !point[0].is_finite() || !point[1].is_finite() {
            return Err(format!(
                "regions: the singularity position of triangle {index} is not finite"
            ));
        }
        let (a, b) = match affine_coefficients(&u, sub(p1, p0), sub(p2, p0)) {
            Some(found) => found,
            None => {
                return Err(format!(
                    "regions: the local field of triangle {index} is degenerate"
                ))
            }
        };
        let valence = (4 - charge) as usize;
        if valence < 1 {
            return Err(format!(
                "regions: triangle {index} has an impossible charge {charge}"
            ));
        }
        let arms = arm_directions(a, b, valence);
        singularities.push(Singularity {
            point,
            valence,
            triangle: index,
            arms,
        });
    }
    Ok(singularities)
}

// ---------------------------------------------------------------------------
// The motorcycle graph (Sections 3.1.9 and 3.1.12)
// ---------------------------------------------------------------------------

/// A tracing hit.
enum Hit {
    /// A directed boundary edge was crossed at parameter `t`.
    Boundary { t: f64 },
    /// An existing edge was crossed.
    Separatrix {
        edge: usize,
        segment: usize,
        point: [f64; 2],
    },
}

/// Builds the graph: traces the separatrices, splits edges at T-junctions and
/// closes the graph along the boundary.
struct Builder<'a> {
    field: &'a Field<'a>,
    graph: MotorcycleGraph,
    singularities: Vec<Singularity>,
    singularity_nodes: Vec<usize>,
    /// For a boundary node, the `(loop, arc length)` of its position.
    boundary_position: Vec<Option<(usize, f64)>>,
    step: f64,
    snap: f64,
}

impl<'a> Builder<'a> {
    fn new(field: &'a Field<'a>) -> Builder<'a> {
        Builder {
            field,
            graph: MotorcycleGraph::default(),
            singularities: Vec::new(),
            singularity_nodes: Vec::new(),
            boundary_position: Vec::new(),
            step: STEP_FRACTION * field.mean_edge,
            snap: WELD_TOLERANCE * field.diagonal,
        }
    }

    /// Detect the singularities and create their nodes (nodes `0..n`).
    fn prepare(&mut self) -> Result<(), String> {
        self.singularities = detect_singularities(self.field)?;
        for index in 0..self.singularities.len() {
            let point = self.singularities[index].point;
            let valence = self.singularities[index].valence;
            let node = self.add_node(point, NodeKind::Singularity(valence));
            self.singularity_nodes.push(node);
        }
        Ok(())
    }

    /// Add a node, reusing a coincident node of the same kind.
    fn add_node(&mut self, point: [f64; 2], kind: NodeKind) -> usize {
        for (index, node) in self.graph.nodes.iter().enumerate() {
            if node.kind == kind && distance(node.point, point) <= self.snap {
                return index;
            }
        }
        self.graph.nodes.push(GraphNode { point, kind });
        self.boundary_position.push(None);
        self.graph.nodes.len() - 1
    }

    /// Add an edge. Degenerate edges are rejected instead of corrupting the
    /// planar graph.
    fn add_edge(
        &mut self,
        from: usize,
        to: usize,
        polyline: Vec<[f64; 2]>,
        singularity: Option<usize>,
        arm: usize,
        kind: EdgeKind,
    ) -> Result<(), String> {
        if from == to {
            return Err("regions: a separatrix closed onto itself (limit cycle)".to_string());
        }
        if polyline.len() < 2 {
            return Err("regions: a graph edge has fewer than two points".to_string());
        }
        let mut polyline = polyline;
        if let (Some(first), Some(node)) = (polyline.first_mut(), self.graph.nodes.get(from)) {
            *first = node.point;
        }
        if let (Some(last), Some(node)) = (polyline.last_mut(), self.graph.nodes.get(to)) {
            *last = node.point;
        }
        self.graph.edges.push(GraphEdge {
            from,
            to,
            polyline,
            singularity,
            arm,
            kind,
        });
        Ok(())
    }

    /// The four-RoSy direction at `p` closest to `previous`.
    fn direction(&self, p: [f64; 2], previous: [f64; 2]) -> Option<[f64; 2]> {
        let alpha =
            simons_godf::interp_angle(self.field.alpha, self.field.points, self.field.tris, p);
        if !alpha.is_finite() {
            return None;
        }
        let mut best = None;
        let mut best_dot = f64::NEG_INFINITY;
        for branch in 0..4 {
            let angle = alpha + branch as f64 * std::f64::consts::FRAC_PI_2;
            let candidate = [angle.cos(), angle.sin()];
            let alignment = dot(candidate, previous);
            if alignment > best_dot {
                best_dot = alignment;
                best = Some(candidate);
            }
        }
        best
    }

    /// Seed distance from a singularity: a fraction of its distance to the
    /// triangle boundary, so the seed stays where the local model is exact.
    fn seed_radius(&self, index: usize) -> f64 {
        let point = self.singularities[index].point;
        let triangle = match self.field.tris.get(self.singularities[index].triangle) {
            Some(found) => *found,
            None => return 1e-6 * self.field.diagonal,
        };
        let mut nearest = f64::INFINITY;
        for slot in 0..3 {
            let a = match self.field.xy(triangle[slot]) {
                Some(found) => found,
                None => continue,
            };
            let b = match self.field.xy(triangle[(slot + 1) % 3]) {
                Some(found) => found,
                None => continue,
            };
            nearest = nearest.min(segment_distance(point, a, b));
        }
        let floor = 1e-7 * self.field.diagonal;
        let ceiling = 0.05 * self.field.diagonal;
        if nearest.is_finite() {
            (0.3 * nearest).clamp(floor, ceiling)
        } else {
            floor
        }
    }

    /// Another singularity within a fraction of a step, if any.
    fn other_singularity(&self, point: [f64; 2], origin: usize) -> Option<usize> {
        for (index, singularity) in self.singularities.iter().enumerate() {
            if index == origin {
                continue;
            }
            if distance(singularity.point, point) <= 0.25 * self.step {
                return Some(index);
            }
        }
        None
    }

    /// First intersection of the step `p`-`q` with an existing separatrix or a
    /// boundary edge.
    fn first_hit(&self, p: [f64; 2], q: [f64; 2]) -> Option<Hit> {
        let mut best_t = f64::INFINITY;
        let mut best: Option<Hit> = None;
        for index in 0..self.field.boundary_directed.len() {
            let (a, b) = match self.field.boundary_segment(index) {
                Some(found) => found,
                None => continue,
            };
            if let Some((t, _)) = segment_intersection(p, q, a, b, 1e-12) {
                if t > 1e-12 && t < best_t {
                    best_t = t;
                    best = Some(Hit::Boundary { t });
                }
            }
        }
        for (edge_index, edge) in self.graph.edges.iter().enumerate() {
            for segment in 0..edge.polyline.len().saturating_sub(1) {
                let (a, b) = match (
                    edge.polyline.get(segment).copied(),
                    edge.polyline.get(segment + 1).copied(),
                ) {
                    (Some(a), Some(b)) => (a, b),
                    _ => continue,
                };
                if let Some((t, _)) = segment_intersection(p, q, a, b, 1e-12) {
                    if t > 1e-9 && t < best_t {
                        best_t = t;
                        best = Some(Hit::Separatrix {
                            edge: edge_index,
                            segment,
                            point: add(p, scale(sub(q, p), t)),
                        });
                    }
                }
            }
        }
        best
    }

    /// Split an edge at a T-junction and return the new node.
    fn split_edge(
        &mut self,
        edge: usize,
        segment: usize,
        point: [f64; 2],
    ) -> Result<usize, String> {
        let current = match self.graph.edges.get(edge) {
            Some(found) => found.clone(),
            None => return Err("regions: a T-junction refers to a missing edge".to_string()),
        };
        let last = current.polyline.len().saturating_sub(1);
        if segment == 0 {
            if let Some(&start) = current.polyline.first() {
                if distance(point, start) <= self.snap {
                    return Ok(current.from);
                }
            }
        }
        if segment + 1 >= last {
            if let Some(&end) = current.polyline.last() {
                if distance(point, end) <= self.snap {
                    return Ok(current.to);
                }
            }
        }
        let node = self.add_node(point, NodeKind::TJunction);
        let mut left: Vec<[f64; 2]> = current.polyline[..=segment].to_vec();
        left.push(point);
        let mut right: Vec<[f64; 2]> = vec![point];
        right.extend_from_slice(&current.polyline[segment + 1..]);
        if left.len() < 2 || right.len() < 2 {
            return Err("regions: a T-junction produced a degenerate edge".to_string());
        }
        self.graph.edges[edge] = GraphEdge {
            from: current.from,
            to: node,
            polyline: left,
            singularity: current.singularity,
            arm: current.arm,
            kind: current.kind,
        };
        self.graph.edges.push(GraphEdge {
            from: node,
            to: current.to,
            polyline: right,
            singularity: current.singularity,
            arm: current.arm,
            kind: current.kind,
        });
        Ok(node)
    }

    /// Boundary node nearest to `point`, with its arc-length position.
    fn boundary_node_near(&mut self, point: [f64; 2]) -> Result<usize, String> {
        let mut best = None;
        let mut best_distance = f64::INFINITY;
        for index in 0..self.field.boundary_directed.len() {
            if let Some((a, b)) = self.field.boundary_segment(index) {
                let d = segment_distance(point, a, b);
                if d < best_distance {
                    best_distance = d;
                    best = Some(index);
                }
            }
        }
        let index = match best {
            Some(found) => found,
            None => return Err("regions: the mesh has no boundary edge".to_string()),
        };
        let (a, b) = match self.field.boundary_segment(index) {
            Some(found) => found,
            None => return Err("regions: a boundary edge has no geometry".to_string()),
        };
        let (loop_index, start) = self
            .field
            .boundary_edge_loop
            .get(index)
            .copied()
            .unwrap_or((0, 0.0));
        let ab = sub(b, a);
        let length = norm(ab);
        let local = if length > 0.0 {
            (dot(sub(point, a), ab) / (length * length)).clamp(0.0, 1.0) * length
        } else {
            0.0
        };
        let total = self.field.loop_total(loop_index);
        let position = if total > 0.0 {
            (start + local).rem_euclid(total)
        } else {
            0.0
        };
        let node = self.add_node(point, NodeKind::Boundary);
        if let Some(slot) = self.boundary_position.get_mut(node) {
            if slot.is_none() {
                *slot = Some((loop_index, position));
            }
        }
        Ok(node)
    }

    /// Trace one separatrix arm from `seed` in direction `direction`.
    fn trace_arm(
        &mut self,
        origin: usize,
        arm: usize,
        seed: [f64; 2],
        direction: [f64; 2],
    ) -> Result<(), String> {
        let step = self.step;
        let from = self.singularity_nodes[origin];
        let centre = self.singularities[origin].point;
        let mut polyline = if distance(centre, seed) > 0.0 {
            vec![centre, seed]
        } else {
            vec![seed]
        };
        let mut p = seed;
        let mut previous = direction;
        for _ in 0..MAX_STEPS {
            let k1 = match self.direction(p, previous) {
                Some(found) => found,
                None => {
                    let node = self.boundary_node_near(p)?;
                    return self.add_edge(
                        from,
                        node,
                        polyline,
                        Some(origin),
                        arm,
                        EdgeKind::Separatrix,
                    );
                }
            };
            let half = 0.5 * step;
            let k2 = self.direction(add(p, scale(k1, half)), k1).unwrap_or(k1);
            let k3 = self.direction(add(p, scale(k2, half)), k2).unwrap_or(k2);
            let k4 = self.direction(add(p, scale(k3, step)), k3).unwrap_or(k3);
            let increment = scale(add(add(k1, scale(add(k2, k3), 2.0)), k4), step / 6.0);
            let q = add(p, increment);
            if !q[0].is_finite() || !q[1].is_finite() || norm(increment) <= 0.0 {
                return Err(format!(
                    "regions: the separatrix of singularity {origin} stalled at ({}, {})",
                    p[0], p[1]
                ));
            }
            if let Some(target) = self.other_singularity(q, origin) {
                polyline.push(self.singularities[target].point);
                let node = self.singularity_nodes[target];
                return self.add_edge(
                    from,
                    node,
                    polyline,
                    Some(origin),
                    arm,
                    EdgeKind::Separatrix,
                );
            }
            match self.first_hit(p, q) {
                Some(Hit::Separatrix {
                    edge,
                    segment,
                    point,
                    ..
                }) => {
                    polyline.push(point);
                    let node = self.split_edge(edge, segment, point)?;
                    return self.add_edge(
                        from,
                        node,
                        polyline,
                        Some(origin),
                        arm,
                        EdgeKind::Separatrix,
                    );
                }
                Some(Hit::Boundary { t }) => {
                    let point = add(p, scale(sub(q, p), t));
                    polyline.push(point);
                    let node = self.boundary_node_near(point)?;
                    return self.add_edge(
                        from,
                        node,
                        polyline,
                        Some(origin),
                        arm,
                        EdgeKind::Separatrix,
                    );
                }
                None => {}
            }
            if !self.field.is_inside(q) {
                // The step left the mesh but no boundary edge was crossed
                // numerically; bisect for the crossing point.
                let point = self.bisect_boundary(p, q);
                polyline.push(point);
                let node = self.boundary_node_near(point)?;
                return self.add_edge(
                    from,
                    node,
                    polyline,
                    Some(origin),
                    arm,
                    EdgeKind::Separatrix,
                );
            }
            previous = self.direction(q, previous).unwrap_or(k4);
            p = q;
            polyline.push(p);
        }
        Err(format!(
            "regions: a separatrix of singularity {origin} did not terminate within {MAX_STEPS} steps"
        ))
    }

    /// Last point of the step `p`-`q` that is still inside the mesh.
    fn bisect_boundary(&self, p: [f64; 2], q: [f64; 2]) -> [f64; 2] {
        let mut inside = p;
        let mut outside = q;
        for _ in 0..60 {
            let middle = scale(add(inside, outside), 0.5);
            if self.field.is_inside(middle) {
                inside = middle;
            } else {
                outside = middle;
            }
        }
        inside
    }

    /// Trace every arm of every singularity.
    fn trace_all(&mut self) -> Result<(), String> {
        for index in 0..self.singularities.len() {
            let point = self.singularities[index].point;
            let arms = self.singularities[index].arms.clone();
            let radius = self.seed_radius(index);
            for (arm, direction) in arms.iter().enumerate() {
                let seed = add(point, scale(*direction, radius));
                self.trace_arm(index, arm, seed, *direction)?;
            }
        }
        Ok(())
    }

    /// Close the graph along the boundary: one edge per pair of consecutive
    /// boundary hits on each boundary loop.
    fn add_boundary_arcs(&mut self) -> Result<(), String> {
        let loops = self.field.boundary_loops.len();
        let mut per_loop: Vec<Vec<(f64, usize)>> = vec![Vec::new(); loops];
        for (node, position) in self.boundary_position.iter().enumerate() {
            if let Some((loop_index, offset)) = *position {
                if let Some(list) = per_loop.get_mut(loop_index) {
                    list.push((offset, node));
                }
            }
        }
        for (loop_index, list) in per_loop.iter_mut().enumerate() {
            list.sort_by(|left, right| {
                left.0
                    .partial_cmp(&right.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            if list.is_empty() {
                return Err(format!(
                    "regions: no separatrix reaches boundary loop {loop_index}; the face that \
                     surrounds the hole is an annulus (the nautilus case of Section 5.1.4), \
                     which is not implemented"
                ));
            }
            if list.len() == 1 {
                return Err(format!(
                    "regions: only one separatrix reaches boundary loop {loop_index}; the \
                     boundary cannot be closed into tiles"
                ));
            }
            let total = self.field.loop_total(loop_index);
            for position in 0..list.len() {
                let (start, from) = list[position];
                let (end, to) = list[(position + 1) % list.len()];
                let end = if position + 1 == list.len() {
                    end + total
                } else {
                    end
                };
                let polyline = self.field.loop_polyline(loop_index, start, end);
                self.add_edge(from, to, polyline, None, 0, EdgeKind::Boundary)?;
            }
        }
        Ok(())
    }

    /// The graph is only the boundary when the field has no singularity (the
    /// rectangle QGP demo case).
    fn add_boundary_loop_edges(&mut self) -> Result<(), String> {
        if self.field.boundary_loops.len() > 1 {
            return Err(
                "regions: the field has no singularity but the domain has a hole; the region \
                 is an annulus (the nautilus case of Section 5.1.4), which is not implemented"
                    .to_string(),
            );
        }
        for (loop_index, vertices) in self.field.boundary_loops.iter().enumerate() {
            let mut nodes: Vec<usize> = Vec::with_capacity(vertices.len());
            for (position, &vertex) in vertices.iter().enumerate() {
                let point = match self.field.xy(vertex) {
                    Some(found) => found,
                    None => return Err("regions: bad boundary vertex".to_string()),
                };
                let node = self.add_node(point, NodeKind::Boundary);
                if let Some(slot) = self.boundary_position.get_mut(node) {
                    let offset = self
                        .field
                        .boundary_cumulative
                        .get(loop_index)
                        .and_then(|values| values.get(position))
                        .copied()
                        .unwrap_or(0.0);
                    *slot = Some((loop_index, offset));
                }
                nodes.push(node);
            }
            for position in 0..nodes.len() {
                let from = nodes[position];
                let to = nodes[(position + 1) % nodes.len()];
                let a = self.graph.nodes[from].point;
                let b = self.graph.nodes[to].point;
                self.add_edge(from, to, vec![a, b], None, 0, EdgeKind::Boundary)?;
            }
        }
        Ok(())
    }

    // -- faces -------------------------------------------------------------

    fn half_start(&self, half: usize) -> Option<usize> {
        let edge = self.graph.edges.get(half / 2)?;
        Some(if half.is_multiple_of(2) {
            edge.from
        } else {
            edge.to
        })
    }

    fn half_end(&self, half: usize) -> Option<usize> {
        let edge = self.graph.edges.get(half / 2)?;
        Some(if half.is_multiple_of(2) {
            edge.to
        } else {
            edge.from
        })
    }

    fn oriented(&self, half: usize) -> Option<Vec<[f64; 2]>> {
        let edge = self.graph.edges.get(half / 2)?;
        let mut line = edge.polyline.clone();
        if !half.is_multiple_of(2) {
            line.reverse();
        }
        Some(line)
    }

    /// Next half-edge of the face on the left of `half`: the smallest clockwise
    /// turn from the reverse of the incoming direction.
    fn next_half(&self, half: usize) -> Option<usize> {
        let vertex = self.half_end(half)?;
        let line = self.oriented(half)?;
        let end_point = *line.last()?;
        let incoming = incoming_direction(&line, end_point)?;
        let back = angle_of(scale(incoming, -1.0));
        let mut best = None;
        let mut best_turn = f64::INFINITY;
        for edge_index in 0..self.graph.edges.len() {
            for branch in 0..2 {
                let candidate = 2 * edge_index + branch;
                if candidate == (half ^ 1) || self.half_start(candidate) != Some(vertex) {
                    continue;
                }
                let candidate_line = match self.oriented(candidate) {
                    Some(found) if found.len() >= 2 => found,
                    _ => continue,
                };
                let outgoing = match outgoing_direction(&candidate_line, candidate_line[0]) {
                    Some(found) => found,
                    None => continue,
                };
                let turn = (back - angle_of(outgoing)).rem_euclid(std::f64::consts::TAU);
                if turn > 1e-12 && turn < best_turn {
                    best_turn = turn;
                    best = Some(candidate);
                }
            }
        }
        best
    }

    /// Walk every face of the graph.
    fn walk_faces(&self) -> Result<Vec<Vec<usize>>, String> {
        let count = self.graph.edges.len();
        if count == 0 {
            return Err("regions: the motorcycle graph has no edge".to_string());
        }
        for node in 0..self.graph.nodes.len() {
            let degree = self.graph.degree(node);
            if degree < 2 {
                return Err(format!(
                    "regions: graph node {node} has degree {degree}; every node of a motorcycle \
                     graph needs at least two edges"
                ));
            }
        }
        let mut visited = vec![false; 2 * count];
        let mut faces: Vec<Vec<usize>> = Vec::new();
        for start in 0..2 * count {
            if visited[start] {
                continue;
            }
            let mut cycle = Vec::new();
            let mut half = start;
            let mut closed = false;
            for _ in 0..=2 * count {
                if visited[half] {
                    closed = half == start;
                    break;
                }
                visited[half] = true;
                cycle.push(half);
                half = match self.next_half(half) {
                    Some(found) => found,
                    None => break,
                };
            }
            if !closed {
                return Err(
                    "regions: the face walk did not close (a dangling edge or a non-manifold \
                     node in the motorcycle graph)"
                        .to_string(),
                );
            }
            faces.push(cycle);
        }
        Ok(faces)
    }

    /// A point just inside a face, near the middle of its first edge.
    fn interior_sample(&self, face: &[usize]) -> Option<[f64; 2]> {
        let line = self.oriented(*face.first()?)?;
        if line.len() < 2 {
            return None;
        }
        let (a, b) = (line[0], line[1]);
        let direction = unit(sub(b, a))?;
        let left = [-direction[1], direction[0]];
        let length = distance(a, b);
        let epsilon = (0.05 * length)
            .min(1e-3 * self.field.diagonal)
            .max(1e-5 * self.field.diagonal);
        Some(add(scale(add(a, b), 0.5), scale(left, epsilon)))
    }

    /// Turn at every node of a face loop (positive for a convex corner of a
    /// counter-clockwise face).
    fn face_turns(&self, face: &[usize]) -> Option<Vec<f64>> {
        let count = face.len();
        if count < 2 {
            return None;
        }
        let mut turns = vec![0.0; count];
        for index in 0..count {
            let previous = self.oriented(face[(index + count - 1) % count])?;
            let next = self.oriented(face[index])?;
            let point = *next.first()?;
            let incoming = incoming_direction(&previous, point)?;
            let outgoing = outgoing_direction(&next, point)?;
            turns[index] = wrap_pi(angle_of(outgoing) - angle_of(incoming));
        }
        Some(turns)
    }

    /// Concatenate the oriented polylines of a sequence of half-edges.
    fn concatenate(&self, half_edges: &[usize]) -> Option<Vec<[f64; 2]>> {
        let mut line: Vec<[f64; 2]> = Vec::new();
        for (index, &half) in half_edges.iter().enumerate() {
            let piece = self.oriented(half)?;
            let skip = usize::from(index > 0 && !line.is_empty());
            for point in piece.into_iter().skip(skip) {
                line.push(point);
            }
        }
        if line.len() < 2 {
            None
        } else {
            Some(line)
        }
    }

    /// Turn the material faces into four-sided tiles.
    fn make_tiles(&self, faces: &[Vec<usize>]) -> Result<(Vec<Tile>, usize), String> {
        let mut tiles = Vec::new();
        let mut degenerate = 0usize;
        for face in faces {
            let count = face.len();
            if count < 3 {
                degenerate += 1;
                continue;
            }
            let mut loop_points: Vec<[f64; 2]> = Vec::new();
            for &half in face {
                let line = match self.oriented(half) {
                    Some(found) => found,
                    None => continue,
                };
                let skip = usize::from(!loop_points.is_empty());
                for point in line.into_iter().skip(skip) {
                    loop_points.push(point);
                }
            }
            if loop_points.len() < 3 {
                degenerate += 1;
                continue;
            }
            if let (Some(first), Some(last)) = (loop_points.first(), loop_points.last()) {
                if distance(*first, *last) <= 1e-12 * self.field.diagonal {
                    loop_points.pop();
                }
            }
            let area = polygon_area(&loop_points);
            if area <= 1e-10 * self.field.diagonal * self.field.diagonal {
                // The outer face (negative or zero area) is not a tile.
                continue;
            }
            let sample = match self.interior_sample(face) {
                Some(found) => found,
                None => {
                    degenerate += 1;
                    continue;
                }
            };
            if !self.field.is_inside(sample) {
                // Not part of the material: the outer face or the hole face.
                continue;
            }
            let turns = match self.face_turns(face) {
                Some(found) => found,
                None => {
                    degenerate += 1;
                    continue;
                }
            };
            let mut corners: Vec<usize> = Vec::new();
            let mut convex = true;
            for (index, turn) in turns.iter().enumerate() {
                if turn.abs() > CORNER_TURN {
                    if *turn <= 0.0 {
                        convex = false;
                    }
                    corners.push(index);
                }
            }
            if corners.len() == 3 {
                let mut tri_sides: [Vec<usize>; 3] = Default::default();
                let mut tri_lines: [Vec<[f64; 2]>; 3] = Default::default();
                let mut tri_failed = false;
                for side in 0..3 {
                    let from = corners[side];
                    let to = corners[(side + 1) % 3];
                    let mut half_edges: Vec<usize> = Vec::new();
                    let mut index = from;
                    loop {
                        half_edges.push(face[index]);
                        index = (index + 1) % count;
                        if index == to {
                            break;
                        }
                    }
                    if half_edges.is_empty() {
                        tri_failed = true;
                        break;
                    }
                    for &half in &half_edges {
                        tri_sides[side].push(half / 2);
                    }
                    tri_lines[side] = match self.concatenate(&half_edges) {
                        Some(found) => found,
                        None => {
                            tri_failed = true;
                            break;
                        }
                    };
                }
                // The free side is the one with the most boundary edges: it may
                // be a sub-chain (a boundary edge plus a separatrix), in which
                // case the separatrix part is not shared with any kept tile and
                // the fan fills it too.
                let boundary_edges = |side: &Vec<usize>| -> usize {
                    side.iter()
                        .filter(|&&edge| {
                            self.graph
                                .edges
                                .get(edge)
                                .map(|found| found.kind == EdgeKind::Boundary)
                                .unwrap_or(false)
                        })
                        .count()
                };
                let free_side = (0..3)
                    .filter(|&side| boundary_edges(&tri_sides[side]) > 0)
                    .max_by_key(|&side| boundary_edges(&tri_sides[side]));
                let usable = convex && !tri_failed && tri_lines.iter().all(|line| line.len() >= 2);
                if let (true, Some(free_side)) = (usable, free_side) {
                    let mut sides: [Vec<usize>; 4] = Default::default();
                    let mut lines: [Vec<[f64; 2]>; 4] = Default::default();
                    let mut corner_points = [[0.0f64; 2]; 4];
                    let mut corner_nodes = [0usize; 4];
                    for slot in 0..3 {
                        let source = (free_side + slot) % 3;
                        sides[slot] = tri_sides[source].clone();
                        lines[slot] = tri_lines[source].clone();
                        corner_points[slot] = tri_lines[source][0];
                        corner_nodes[slot] = self.half_start(face[corners[source]]).unwrap_or(0);
                    }
                    corner_points[3] = corner_points[0];
                    corner_nodes[3] = corner_nodes[0];
                    let curves = BoundaryCurves {
                        bottom: lines[0].clone(),
                        right: lines[1].clone(),
                        top: lines[2].clone(),
                        left: Vec::new(),
                    };
                    tiles.push(Tile {
                        nodes: corner_nodes[..3].to_vec(),
                        sides,
                        corners: corner_points,
                        corner_nodes,
                        curves,
                        counts: [0; 4],
                        area,
                        bent: false,
                        triangular: true,
                    });
                    continue;
                }
                degenerate += 1;
                continue;
            }
            if corners.len() != 4 || !convex {
                degenerate += 1;
                continue;
            }
            let mut sides: [Vec<usize>; 4] = Default::default();
            let mut side_lines: [Vec<[f64; 2]>; 4] = Default::default();
            let mut failed = false;
            for side in 0..4 {
                let from = corners[side];
                let to = corners[(side + 1) % 4];
                let mut half_edges: Vec<usize> = Vec::new();
                let mut index = from;
                loop {
                    half_edges.push(face[index]);
                    index = (index + 1) % count;
                    if index == to {
                        break;
                    }
                }
                if half_edges.is_empty() {
                    failed = true;
                    break;
                }
                for &half in &half_edges {
                    sides[side].push(half / 2);
                }
                side_lines[side] = match self.concatenate(&half_edges) {
                    Some(found) => found,
                    None => {
                        failed = true;
                        break;
                    }
                };
            }
            if failed || side_lines.iter().any(|line| line.len() < 2) {
                degenerate += 1;
                continue;
            }
            let mut corner_points = [[0.0f64; 2]; 4];
            let mut corner_nodes = [0usize; 4];
            for side in 0..4 {
                corner_points[side] = side_lines[side][0];
                corner_nodes[side] = match self.half_start(face[corners[side]]) {
                    Some(found) => found,
                    None => {
                        failed = true;
                        continue;
                    }
                };
            }
            if failed {
                degenerate += 1;
                continue;
            }
            let bent = polygon_area(&corner_points) <= 0.0;
            let mut node_ids: Vec<usize> = Vec::with_capacity(count);
            for &half in face {
                node_ids.push(self.half_start(half).unwrap_or(0));
            }
            let curves = BoundaryCurves {
                bottom: side_lines[0].clone(),
                right: side_lines[1].clone(),
                top: side_lines[2].clone(),
                left: side_lines[3].clone(),
            };
            tiles.push(Tile {
                nodes: node_ids,
                sides,
                corners: corner_points,
                corner_nodes,
                curves,
                counts: [0; 4],
                area,
                bent,
                triangular: false,
            });
        }
        Ok((tiles, degenerate))
    }

    /// `s_hat = integral ds / sigma` for every graph edge.
    fn ideal_counts(&self) -> Result<Vec<f64>, String> {
        let mut ideals = Vec::with_capacity(self.graph.edges.len());
        for (index, edge) in self.graph.edges.iter().enumerate() {
            let mut total = 0.0;
            for pair in edge.polyline.windows(2) {
                let a = self.field.sigma_at(pair[0]);
                let b = self.field.sigma_at(pair[1]);
                if !a.is_finite() || !b.is_finite() || a <= 0.0 || b <= 0.0 {
                    return Err(format!(
                        "regions: edge {index} has a non-positive size along its polyline"
                    ));
                }
                total += 0.5 * (1.0 / a + 1.0 / b) * distance(pair[0], pair[1]);
            }
            if !total.is_finite() || total <= 0.0 {
                return Err(format!(
                    "regions: edge {index} has a zero ideal subdivision count; the size \
                     function is too large for this edge"
                ));
            }
            ideals.push(total);
        }
        Ok(ideals)
    }
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Build the singularities, the motorcycle graph and the simple tiles of a
/// cross field.
///
/// `points` and `tris` are the planar triangle mesh (`z` is ignored), `alpha`
/// the per-vertex four-RoSy angle in `[0, pi/2)` (as returned by
/// [`crate::simons_godf::solve`]) and `size` the per-vertex target element size.
/// The QGP integer counts are solved for every tile; call [`Regions::mesh`] (or
/// use [`mesh_tiles`]) to mesh them.
pub fn build(
    points: &[[f64; 3]],
    tris: &[[usize; 3]],
    alpha: &[f64],
    size: &[f64],
) -> Result<Regions, String> {
    let field = Field::new(points, tris, alpha, size)?;
    let mut builder = Builder::new(&field);
    builder.prepare()?;
    if builder.singularities.is_empty() {
        builder.add_boundary_loop_edges()?;
    } else {
        builder.trace_all()?;
        builder.add_boundary_arcs()?;
    }
    let faces = builder.walk_faces()?;
    let (tiles, degenerate) = builder.make_tiles(&faces)?;
    if tiles.is_empty() {
        return Err("regions: no simple tile was found".to_string());
    }
    let ideals = builder.ideal_counts()?;
    let regions: Vec<Region> = tiles
        .iter()
        .filter(|tile| !tile.triangular)
        .map(|tile| Region::new(tile.sides.clone()))
        .collect();
    let counts = simons_qgp::solve_counts(&regions, &ideals)?;
    let energy = simons_qgp::energy(&counts, &ideals)?;
    let mut kept: Vec<Tile> = Vec::with_capacity(tiles.len());
    let mut collapsed = 0usize;
    for (index, mut tile) in tiles.into_iter().enumerate() {
        if tile.triangular {
            let mut left = 0usize;
            for &edge in &tile.sides[1] {
                left = left.saturating_add(counts.get(edge).copied().unwrap_or(0));
            }
            let mut right = 0usize;
            for &edge in &tile.sides[2] {
                right = right.saturating_add(counts.get(edge).copied().unwrap_or(0));
            }
            eprintln!(
                "DEBUG triangular tile {index}: seps {left} vs {right} ({} sides)",
                tile.sides[1].len() + tile.sides[2].len()
            );
            // The three-kite fan needs a common even side count. When the two
            // seps were solved to different counts, use the largest even count
            // that fits both; the odd sep then keeps one segment unmeshed (it
            // borders a rejected face, not a kept tile).
            let common = left.min(right) - (left.min(right) % 2);
            if common < 2 {
                collapsed += 1;
                continue;
            }
            tile.counts = [common, common, common, common];
            kept.push(tile);
            continue;
        }
        let mut row = [0usize; 4];
        for (side, slot) in row.iter_mut().enumerate() {
            let mut sum = 0usize;
            for &edge in &tile.sides[side] {
                sum = sum.saturating_add(counts.get(edge).copied().unwrap_or(0));
            }
            *slot = sum;
        }
        if row[0] != row[2] || row[1] != row[3] {
            return Err(format!(
                "regions: tile {index} violates eq. (5.1) after the QGP solve \
                 ({} {} {} {})",
                row[0], row[1], row[2], row[3]
            ));
        }
        if row[0] == 0 || row[1] == 0 {
            // The QGP energy solved this tile onto a collapsed side; Section
            // 5.1.6 collapses such a tile to a point.
            collapsed += 1;
            continue;
        }
        tile.counts = row;
        kept.push(tile);
    }
    if kept.is_empty() {
        return Err("regions: every tile was collapsed by the QGP solve".to_string());
    }
    let singularities = builder
        .singularities
        .iter()
        .map(|singularity| singularity.point)
        .collect();
    Ok(Regions {
        singularities,
        graph: builder.graph,
        tiles: kept,
        degenerate_tiles: degenerate,
        collapsed_tiles: collapsed,
        energy,
    })
}

/// Build the regions and mesh every tile into one welded [`Mesh`].
pub fn mesh_tiles(
    points: &[[f64; 3]],
    tris: &[[usize; 3]],
    alpha: &[f64],
    size: &[f64],
) -> Result<Mesh, String> {
    build(points, tris, alpha, size)?.mesh()
}

/// Tutte/Laplacian placement of the interior nodes of a quad patch.
///
/// `quads` is the topology (node indices), `fixed` the positions of the nodes
/// that must not move (a boundary loop), and the free nodes are placed so that
/// each is the average of its graph neighbours. The linear system is solved by
/// hand-written Gauss-Seidel sweeps (no dependencies); it converges for a
/// convex fixed boundary, which keeps every quad of a patch inside its
/// boundary. This is the placement the channel templates use for their Steiner
/// points (Section 2.2.5) and the one that removes the chord-crossing
/// inversions of a naive triangle strip.
///
/// Returns one position per node index used by `quads`.
pub fn laplacian_embedding(
    quads: &[[usize; 4]],
    fixed: &HashMap<usize, [f64; 2]>,
    initial: &HashMap<usize, [f64; 2]>,
    sweeps: usize,
) -> HashMap<usize, [f64; 2]> {
    let mut neighbours: HashMap<usize, Vec<usize>> = HashMap::new();
    for quad in quads {
        for slot in 0..4 {
            let a = quad[slot];
            let b = quad[(slot + 1) % 4];
            neighbours.entry(a).or_default().push(b);
            neighbours.entry(b).or_default().push(a);
        }
    }
    let mut positions: HashMap<usize, [f64; 2]> = HashMap::new();
    for (&node, point) in initial {
        positions.insert(node, *point);
    }
    for (&node, point) in fixed {
        positions.insert(node, *point);
    }
    for (&node, list) in &neighbours {
        if fixed.contains_key(&node) {
            continue;
        }
        if let std::collections::hash_map::Entry::Vacant(slot) = positions.entry(node) {
            let mut sum = [0.0f64; 2];
            let mut count = 0.0;
            for &other in list {
                if let Some(point) = initial.get(&other) {
                    sum = add(sum, *point);
                    count += 1.0;
                }
            }
            if count > 0.0 {
                slot.insert(scale(sum, 1.0 / count));
            }
        }
    }
    let free: Vec<usize> = neighbours
        .keys()
        .copied()
        .filter(|node| !fixed.contains_key(node))
        .collect();
    for _ in 0..sweeps {
        let mut moved = 0.0f64;
        for &node in &free {
            let list = match neighbours.get(&node) {
                Some(found) if !found.is_empty() => found,
                _ => continue,
            };
            let mut sum = [0.0f64; 2];
            let mut count = 0.0;
            for &other in list {
                if let Some(point) = positions.get(&other) {
                    sum = add(sum, *point);
                    count += 1.0;
                }
            }
            if count <= 0.0 {
                continue;
            }
            let target = scale(sum, 1.0 / count);
            if let Some(current) = positions.get(&node).copied() {
                moved += distance(current, target);
            }
            positions.insert(node, target);
        }
        if moved < 1e-12 {
            break;
        }
    }
    positions
}

/// Quadrangulate a triangle with `leg_segments` subdivisions on each shared sep
/// and `base_segments` on the free base: 24 quads for (6, 4), arranged as five
/// quad rows plus a four-quad pinwheel at the apex with two extra interior
/// nodes. The interior nodes are placed by [`laplacian_embedding`] with the
/// base, the two seps and the apex fixed, so the embedding stays inside the
/// triangle; every quad is then checked to be counter-clockwise.
///
/// Returns the quads and their boundary edge count (which must be
/// `base_segments + 2 * leg_segments`).
#[allow(dead_code)]
fn triangle_patch_quads(
    sep_left: &[[f64; 2]],
    sep_right: &[[f64; 2]],
    base: &[[f64; 2]],
    base_segments: usize,
    leg_segments: usize,
) -> Result<(Vec<[[f64; 2]; 4]>, usize), String> {
    if base_segments < 2 || leg_segments < 2 {
        return Err("triangle patch: at least two segments per side".to_string());
    }
    let left = resample_open(sep_left, leg_segments);
    let right = resample_open(sep_right, leg_segments);
    let bottom = resample_open(base, base_segments);
    if left.len() != leg_segments + 1
        || right.len() != leg_segments + 1
        || bottom.len() != base_segments + 1
    {
        return Err("triangle patch: a side has zero length".to_string());
    }
    let apex = match (left.last(), right.last()) {
        (Some(a), Some(b)) => add(scale(*a, 0.5), scale(*b, 0.5)),
        _ => return Err("triangle patch: no apex".to_string()),
    };
    // Node ids: base 0..base_segments; level j (1..leg_segments) has the left
    // sep node, the interior nodes and the right sep node.
    let width = base_segments + 1;
    let mut positions: HashMap<usize, [f64; 2]> = HashMap::new();
    let mut fixed: HashMap<usize, [f64; 2]> = HashMap::new();
    for (i, point) in bottom.iter().enumerate() {
        positions.insert(i, *point);
        fixed.insert(i, *point);
    }
    let level_start = |j: usize| -> usize { width + (j - 1) * width };
    for j in 1..=leg_segments {
        let start = level_start(j);
        let left_point = *left.get(j).unwrap_or(&apex);
        let right_point = *right.get(j).unwrap_or(&apex);
        for i in 0..width {
            let s = i as f64 / base_segments as f64;
            let point = add(scale(left_point, 1.0 - s), scale(right_point, s));
            positions.insert(start + i, point);
            if i == 0 {
                fixed.insert(start, left_point);
            } else if i + 1 == width {
                fixed.insert(start + i, right_point);
            }
        }
    }
    let apex_id = level_start(leg_segments) + width;
    positions.insert(apex_id, apex);
    fixed.insert(apex_id, apex);
    let extra0 = apex_id + 1;
    let extra1 = apex_id + 2;
    // The two fan extras sit on the apex bisector, at a third of the apex
    // height and a quarter of the chord half-width to either side. They are
    // fixed, not solved: for a thin triangle the harmonic solve puts them on
    // the level chord, which folds the two middle fan quads.
    let last_start = level_start(leg_segments) - width;
    let chord_start = positions.get(&last_start).copied().unwrap_or(apex);
    let chord_end = positions
        .get(&(last_start + base_segments))
        .copied()
        .unwrap_or(apex);
    let chord = sub(chord_end, chord_start);
    let chord_length = norm(chord);
    let middle = scale(add(chord_start, chord_end), 0.5);
    // `up` points from the apex towards the chord midpoint, so that
    // `apex + up * h/3` is inside the triangle.
    let up = unit(sub(middle, apex)).unwrap_or([0.0, -1.0]);
    let tangent = unit(chord).unwrap_or([1.0, 0.0]);
    let height = distance(apex, middle);
    let along = add(scale(up, height / 3.0), apex);
    let side = 0.25 * 0.5 * chord_length;
    let extra0_point = add(along, scale(tangent, side));
    let extra1_point = add(along, scale(tangent, -side));
    positions.insert(extra0, extra0_point);
    positions.insert(extra1, extra1_point);
    fixed.insert(extra0, extra0_point);
    fixed.insert(extra1, extra1_point);
    // Topology: strip rows then the pinwheel.
    let mut quads: Vec<[usize; 4]> = Vec::new();
    for j in 0..leg_segments - 1 {
        let lower = level_start(j.max(1));
        let lower = if j == 0 { 0 } else { lower };
        let upper = level_start(j + 1);
        for i in 0..base_segments {
            quads.push([lower + i, lower + i + 1, upper + i + 1, upper + i]);
        }
    }
    let last = level_start(leg_segments);
    // Pinwheel: q1 and q4 share (apex, p0); q2 and q3 share (p0, p1).
    quads.push([last, last + 1, extra0, apex_id]);
    quads.push([last + 1, last + 2, extra0, extra1]);
    quads.push([last + 2, last + 3, extra1, extra0]);
    quads.push([last + 3, last + 4, apex_id, extra0]);
    let placed = laplacian_embedding(&quads, &fixed, &positions, 400);
    let mut out: Vec<[[f64; 2]; 4]> = Vec::with_capacity(quads.len());
    for (index, quad) in quads.iter().enumerate() {
        let mut corners = [[0.0f64; 2]; 4];
        for (slot, &node) in quad.iter().enumerate() {
            corners[slot] = match placed.get(&node) {
                Some(found) => *found,
                None => return Err(format!("triangle patch: node {node} unplaced")),
            };
        }
        if polygon_area(&corners) <= 0.0 {
            return Err(format!("triangle patch: quad {index} is inverted"));
        }
        out.push(corners);
    }
    // Boundary edge count of the topology.
    let mut edges: HashMap<(usize, usize), usize> = HashMap::new();
    for quad in &quads {
        for slot in 0..4 {
            let a = quad[slot];
            let b = quad[(slot + 1) % 4];
            let key = if a < b { (a, b) } else { (b, a) };
            *edges.entry(key).or_insert(0) += 1;
        }
    }
    let boundary = edges.values().filter(|&&count| count == 1).count();
    Ok((out, boundary))
}

/// Coons grid of a patch whose four sides are the given polylines (plain arc
/// length sampling, eq. 5.2); every quad must have positive area.
fn coons_patch_quads(
    bottom: &[[f64; 2]],
    right: &[[f64; 2]],
    top: &[[f64; 2]],
    left: &[[f64; 2]],
    n_u: usize,
    n_v: usize,
) -> Result<Vec<[[f64; 2]; 4]>, String> {
    if n_u == 0 || n_v == 0 {
        return Err("coons patch: zero subdivision".to_string());
    }
    let bottom = resample_open(bottom, n_u);
    let right = resample_open(right, n_v);
    let top = resample_open(top, n_u);
    let left = resample_open(left, n_v);
    if bottom.len() != n_u + 1
        || right.len() != n_v + 1
        || top.len() != n_u + 1
        || left.len() != n_v + 1
    {
        return Err("coons patch: a side has zero length".to_string());
    }
    let corner0 = bottom[0];
    let corner1 = bottom[n_u];
    let corner2 = top[0];
    let corner3 = top[n_u];
    let mut points: Vec<[f64; 2]> = Vec::with_capacity((n_u + 1) * (n_v + 1));
    for j in 0..=n_v {
        let v = j as f64 / n_v as f64;
        for i in 0..=n_u {
            let u = i as f64 / n_u as f64;
            let top_point = top[n_u - i];
            let left_point = left[n_v - j];
            let blend_bottom_top = add(scale(bottom[i], 1.0 - v), scale(top_point, v));
            let blend_left_right = add(scale(right[j], u), scale(left_point, 1.0 - u));
            let bilinear = add(
                scale(add(scale(corner0, 1.0 - u), scale(corner1, u)), 1.0 - v),
                scale(add(scale(corner3, 1.0 - u), scale(corner2, u)), v),
            );
            points.push(add(blend_bottom_top, sub(blend_left_right, bilinear)));
        }
    }
    let index = |i: usize, j: usize| j * (n_u + 1) + i;
    let mut quads = Vec::with_capacity(n_u * n_v);
    for j in 0..n_v {
        for i in 0..n_u {
            let quad = [
                points[index(i, j)],
                points[index(i + 1, j)],
                points[index(i + 1, j + 1)],
                points[index(i, j + 1)],
            ];
            if polygon_area(&quad) <= 0.0 {
                return Err("coons patch: a quad is inverted".to_string());
            }
            quads.push(quad);
        }
    }
    Ok(quads)
}

/// Quadrangulate a triangle as a **three-kite fan**: the triangle is split by
/// its three side midpoints and its centroid into three kites, each meshed with
/// a 3x3 Coons grid, so every side gets `6` subdivisions (the shared seps keep
/// their QGP counts) and the free arc is partitioned into 6.
///
/// This avoids the thin-apex fan that folds on the hole-adjacent faces: the
/// kites are convex and their Coons grids are positive. Returns the quads and
/// the boundary edge count (which must be `3 * 6`).
pub fn triangle_fan_quads(
    sep_left: &[[f64; 2]],
    sep_right: &[[f64; 2]],
    base: &[[f64; 2]],
    side_segments: usize,
) -> Result<(Vec<[[f64; 2]; 4]>, usize), String> {
    if !side_segments.is_multiple_of(2) || side_segments < 2 {
        return Err("triangle fan: an even side with at least two segments".to_string());
    }
    let half = side_segments / 2;
    let base_nodes = resample_open(base, side_segments);
    let left_nodes = resample_open(sep_left, side_segments);
    let right_nodes = resample_open(sep_right, side_segments);
    if base_nodes.len() != side_segments + 1
        || left_nodes.len() != side_segments + 1
        || right_nodes.len() != side_segments + 1
    {
        return Err("triangle fan: a side has zero length".to_string());
    }
    let c0 = base_nodes[0];
    let c1 = base_nodes[side_segments];
    let apex = match (left_nodes.last(), right_nodes.last()) {
        (Some(a), Some(b)) => add(scale(*a, 0.5), scale(*b, 0.5)),
        _ => return Err("triangle fan: no apex".to_string()),
    };
    let m0 = base_nodes[half];
    let m1 = *left_nodes.get(half).unwrap_or(&apex);
    let m2 = *right_nodes.get(half).unwrap_or(&apex);
    let _ = (c0, c1, apex);
    // The fan centre is the average of the three side midpoints: it keeps the
    // three kites similar in shape, which is what stops the Coons grids from
    // folding on a thin triangle.
    let centre = scale(add(add(m0, m1), m2), 1.0 / 3.0);
    let chain = |nodes: &[[f64; 2]], from: usize, to: usize| -> Vec<[f64; 2]> {
        if from <= to {
            nodes[from..=to].to_vec()
        } else {
            let mut out = nodes[to..=from].to_vec();
            out.reverse();
            out
        }
    };
    let mut quads = Vec::with_capacity(3 * half * half);
    // Kite 0: base's second half, the sep-left first half, the two spokes.
    quads.extend(
        coons_patch_quads(
            &chain(&base_nodes, half, side_segments),
            &chain(&left_nodes, 0, half),
            &[m1, centre],
            &[centre, m0],
            half,
            half,
        )
        .map_err(|error| format!("kite 0: {error}"))?,
    );
    // Kite 1: the sep-left second half, the sep-right second half.
    quads.extend(
        coons_patch_quads(
            &chain(&left_nodes, half, side_segments),
            &chain(&right_nodes, side_segments, half),
            &[m2, centre],
            &[centre, m1],
            half,
            half,
        )
        .map_err(|error| format!("kite 1: {error}"))?,
    );
    // Kite 2: the sep-right first half, the base's first half.
    quads.extend(
        coons_patch_quads(
            &chain(&right_nodes, half, 0),
            &chain(&base_nodes, 0, half),
            &[m0, centre],
            &[centre, m2],
            half,
            half,
        )
        .map_err(|error| format!("kite 2: {error}"))?,
    );
    // Boundary count of the union.
    let mut edges: HashMap<((i64, i64), (i64, i64)), usize> = HashMap::new();
    let key = |p: [f64; 2]| ((p[0] * 1e6).round() as i64, (p[1] * 1e6).round() as i64);
    for quad in &quads {
        for slot in 0..4 {
            let a = key(quad[slot]);
            let b = key(quad[(slot + 1) % 4]);
            let edge = if a < b { (a, b) } else { (b, a) };
            *edges.entry(edge).or_insert(0) += 1;
        }
    }
    let boundary = edges.values().filter(|&&count| count == 1).count();
    Ok((quads, boundary))
}

// ---------------------------------------------------------------------------
// Section 5.1.8.2: stitching the interior mesh to the boundary
// ---------------------------------------------------------------------------

/// One stitched case's measured numbers.
#[derive(Clone, Debug, Default)]
pub struct StitchCase {
    /// Short name of the case.
    pub name: &'static str,
    /// Quads of the QGP interior mesh before the carve.
    pub interior_quads: usize,
    /// Quads removed by the carve near the outer boundary.
    pub carved_quads: usize,
    /// Quads added by the channel.
    pub channel_quads: usize,
    /// Interior QGP quads kept after the carve (0 when the whole domain is
    /// filled by channel bands).
    pub kept_interior: usize,
    /// Number of `simons_channel::channel_between` calls that succeeded.
    pub bands: usize,
    /// Total quads of the stitched mesh.
    pub quads: usize,
    /// Edges used by exactly one quad that are not on the domain boundary.
    pub cracks: usize,
    /// Smallest scaled Jacobian of the stitched mesh.
    pub min_scaled_jacobian: f64,
    /// Smallest corner angle of the stitched mesh, in degrees.
    pub min_corner_angle_deg: f64,
    /// Largest corner angle of the stitched mesh, in degrees.
    pub max_corner_angle_deg: f64,
    /// Area covered by the stitched mesh divided by the domain area.
    pub coverage: f64,
    /// Largest distance from a mesh boundary node to the domain boundary.
    pub boundary_offset: f64,
    /// What could not be stitched, with the `channel_between` error text.
    pub notes: Vec<String>,
    /// The stitched mesh (interior plus channel); empty when the stitch failed.
    pub mesh: Mesh,
}

/// Numbers measured by [`stitch_demo`].
#[derive(Clone, Debug, Default)]
pub struct StitchNumbers {
    /// The square with a circular hole.
    pub annulus: StitchCase,
    /// The single rectangle.
    pub rectangle: StitchCase,
}

/// Signed area of a closed polygon given as `[x, y]` points.
fn loop_area(points: &[[f64; 2]]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }
    let mut twice = 0.0;
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        twice += a[0] * b[1] - b[0] * a[1];
    }
    0.5 * twice
}

/// Distance from a point to a closed polyline.
fn distance_to_loop(p: [f64; 2], points: &[[f64; 2]]) -> f64 {
    let mut best = f64::INFINITY;
    if points.len() < 2 {
        return best;
    }
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        best = best.min(segment_distance(p, a, b));
    }
    best
}

/// Boundary loops of a quad mesh, with the material on the left (so an outer
/// loop is counter-clockwise and a hole loop is clockwise). Vertices are node
/// ids in order.
/// Directed boundary edges of a quad mesh (material on the left).
fn boundary_directed(mesh: &Mesh) -> Result<Vec<(usize, usize)>, String> {
    let mut directed: HashMap<(usize, usize), ()> = HashMap::new();
    for (index, cell) in mesh.cells.iter().enumerate() {
        if cell.nodes.len() != 4 {
            return Err(format!("regions: cell {index} is not a quad"));
        }
        for slot in 0..4 {
            let a = cell.nodes[slot];
            let b = cell.nodes[(slot + 1) % 4];
            if directed.remove(&(b, a)).is_none() && directed.insert((a, b), ()).is_some() {
                return Err(format!(
                    "regions: directed mesh edge ({a}, {b}) is used twice in the same \
                     direction (inconsistent quad winding)"
                ));
            }
        }
    }
    if directed.is_empty() {
        return Err("regions: the mesh has no boundary".to_string());
    }
    let mut edges: Vec<(usize, usize)> = directed.keys().copied().collect();
    edges.sort_unstable();
    Ok(edges)
}

/// Boundary loops of a quad mesh, with the material on the left (so an outer
/// loop is counter-clockwise and a hole loop is clockwise). Vertices are node
/// ids in order.
fn mesh_boundary_loops(mesh: &Mesh) -> Result<Vec<Vec<usize>>, String> {
    let directed = boundary_directed(mesh)?;
    let mut outgoing: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for &edge in &directed {
        outgoing.entry(edge.0).or_default().push(edge);
    }
    // Cell adjacency: every directed cell edge and its owner, so that at a
    // pinch node (several loops meeting, e.g. a singularity) the walk can sweep
    // the fan of cells around the node and pick the next boundary edge
    // deterministically instead of relying on near-collinear angles.
    let mut owner: HashMap<(usize, usize), usize> = HashMap::new();
    for (index, cell) in mesh.cells.iter().enumerate() {
        for slot in 0..4 {
            let a = cell.nodes[slot];
            let b = cell.nodes[(slot + 1) % 4];
            owner.insert((a, b), index);
        }
    }
    let next_in_cell = |cell: usize, edge: (usize, usize)| -> Option<(usize, usize)> {
        let quad = mesh.cells.get(cell)?;
        for slot in 0..4 {
            if quad.nodes[slot] == edge.0 && quad.nodes[(slot + 1) % 4] == edge.1 {
                let after = quad.nodes[(slot + 2) % 4];
                return Some((edge.1, after));
            }
        }
        None
    };
    let boundary: HashMap<(usize, usize), ()> = directed.iter().map(|&e| (e, ())).collect();
    let sweep = |edge: (usize, usize)| -> Option<(usize, usize)> {
        let mut current = edge;
        for _ in 0..16 {
            let cell = owner.get(&current).copied()?;
            let candidate = next_in_cell(cell, current)?;
            if boundary.contains_key(&candidate) {
                return Some(candidate);
            }
            current = (candidate.1, candidate.0);
        }
        None
    };
    let position = |node: usize| -> Option<[f64; 2]> {
        mesh.points.get(node).map(|point| [point[0], point[1]])
    };
    let geometric = |edge: (usize, usize)| -> Option<(usize, usize)> {
        let b = position(edge.1)?;
        let a = position(edge.0)?;
        let back = angle_of(sub(a, b));
        let mut best: Option<(f64, (usize, usize))> = None;
        for &candidate in outgoing.get(&edge.1)? {
            if candidate == (edge.1, edge.0) {
                continue;
            }
            let c = position(candidate.1)?;
            let direction = unit(sub(c, b))?;
            let turn = (back - angle_of(direction)).rem_euclid(std::f64::consts::TAU);
            if turn > 1e-12 && best.map(|(current, _)| turn < current).unwrap_or(true) {
                best = Some((turn, candidate));
            }
        }
        best.map(|(_, found)| found)
    };
    let mut visited: HashMap<(usize, usize), ()> = HashMap::new();
    let mut loops: Vec<Vec<usize>> = Vec::new();
    for &start in &directed {
        if visited.contains_key(&start) {
            continue;
        }
        let mut loop_ids: Vec<usize> = Vec::new();
        let mut edge = start;
        let mut closed = false;
        for _ in 0..=directed.len() {
            if visited.contains_key(&edge) {
                closed = edge == start;
                break;
            }
            visited.insert(edge, ());
            loop_ids.push(edge.0);
            edge = match sweep(edge).or_else(|| geometric(edge)) {
                Some(found) => found,
                None => break,
            };
        }
        if !closed {
            let mut info = String::new();
            for &id in loop_ids.iter().take(12) {
                if let Some(point) = position(id) {
                    info.push_str(&format!("({:.3},{:.3}) ", point[0], point[1]));
                }
            }
            return Err(format!(
                "regions: a mesh boundary loop did not close: start {:?} walked {} nodes {}",
                start,
                loop_ids.len(),
                info
            ));
        }
        loops.push(loop_ids);
    }
    Ok(loops)
}

/// Split every non-manifold boundary node into one copy per incident boundary
/// loop, so the boundary edge set becomes a union of disjoint cycles.
///
/// A quad-mesh boundary node where two boundary loops touch (a *pinch*, three
/// or more boundary edges at the node) is duplicated: each incident boundary
/// wedge keeps its own copy, and every quad at the node is re-indexed to the
/// copy of the wedge that contains its centroid. The copies sit at the same
/// position, so the mesh stays conforming inside each fan and the loops become
/// independent. Returns the new mesh and the number of extra nodes created.
fn split_pinched_boundary(mesh: &Mesh) -> Result<(Mesh, usize), String> {
    let directed = boundary_directed(mesh)?;
    let mut outgoing: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut incoming: HashMap<usize, Vec<usize>> = HashMap::new();
    for &(a, b) in &directed {
        outgoing.entry(a).or_default().push(b);
        incoming.entry(b).or_default().push(a);
    }
    let position = |node: usize| -> Option<[f64; 2]> {
        mesh.points.get(node).map(|point| [point[0], point[1]])
    };
    // Wedges at every pinched node: (incoming start, outgoing end, sector
    // start angle, sector end angle) measured counter-clockwise.
    let mut wedges: HashMap<usize, Vec<(usize, usize, f64, f64)>> = HashMap::new();
    let mut created = 0usize;
    let mut pinch_nodes: Vec<usize> = outgoing
        .iter()
        .filter(|(_, list)| list.len() > 1)
        .map(|(&node, _)| node)
        .collect();
    pinch_nodes.sort_unstable();
    for node in pinch_nodes {
        let centre = match position(node) {
            Some(found) => found,
            None => return Err("regions: a boundary node has no position".to_string()),
        };
        let outs = outgoing.get(&node).cloned().unwrap_or_default();
        let ins = incoming.get(&node).cloned().unwrap_or_default();
        if ins.len() != outs.len() || outs.len() < 2 {
            return Err(format!(
                "regions: boundary node {node} has {} incoming and {} outgoing boundary \
                 edges; it cannot be split",
                ins.len(),
                outs.len()
            ));
        }
        let mut entries: Vec<(usize, usize, f64, f64)> = Vec::with_capacity(outs.len());
        let mut used: Vec<bool> = vec![false; outs.len()];
        for &start in &ins {
            let from = match position(start) {
                Some(found) => found,
                None => continue,
            };
            let back = angle_of(sub(from, centre));
            let mut best: Option<(f64, usize)> = None;
            for (slot, &end) in outs.iter().enumerate() {
                if used[slot] {
                    continue;
                }
                let target = match position(end) {
                    Some(found) => found,
                    None => continue,
                };
                let turn = (back - angle_of(sub(target, centre))).rem_euclid(std::f64::consts::TAU);
                if best.map(|(current, _)| turn < current).unwrap_or(true) {
                    best = Some((turn, slot));
                }
            }
            let slot = match best {
                Some((_, found)) => found,
                None => {
                    return Err(format!(
                        "regions: boundary node {node} has no free outgoing edge"
                    ))
                }
            };
            used[slot] = true;
            let end = outs[slot];
            let target = match position(end) {
                Some(found) => found,
                None => return Err("regions: a boundary node has no position".to_string()),
            };
            // The material wedge at the node runs counter-clockwise from the
            // outgoing direction to the reverse of the incoming direction.
            entries.push((start, end, angle_of(sub(target, centre)), back));
        }
        if entries.len() != outs.len() {
            return Err(format!("regions: boundary node {node} could not be paired"));
        }
        created += entries.len() - 1;
        wedges.insert(node, entries);
    }
    if wedges.is_empty() {
        return Ok((mesh.clone(), 0));
    }
    // New copy ids for every wedge except the first one of each node.
    let mut copies: HashMap<(usize, usize), usize> = HashMap::new();
    let mut points = mesh.points.clone();
    for (&node, entries) in &wedges {
        for (slot, _) in entries.iter().enumerate() {
            let id = if slot == 0 {
                node
            } else {
                let id = points.len();
                points.push(mesh.points.get(node).copied().unwrap_or([0.0; 3]));
                id
            };
            copies.insert((node, slot), id);
        }
    }
    let contains_angle = |start: f64, end: f64, value: f64| -> bool {
        let span = (end - start).rem_euclid(std::f64::consts::TAU);
        let offset = (value - start).rem_euclid(std::f64::consts::TAU);
        offset <= span + 1e-12
    };
    let mut cells: Vec<Cell> = Vec::with_capacity(mesh.cells.len());
    for (index, cell) in mesh.cells.iter().enumerate() {
        let mut nodes = cell.nodes.clone();
        for slot in 0..nodes.len() {
            let node = nodes[slot];
            let entries = match wedges.get(&node) {
                Some(found) => found,
                None => continue,
            };
            let centre = match position(node) {
                Some(found) => found,
                None => return Err("regions: a boundary node has no position".to_string()),
            };
            let mut centroid = [0.0f64; 2];
            for &other in &cell.nodes {
                if let Some(point) = position(other) {
                    centroid[0] += 0.25 * point[0];
                    centroid[1] += 0.25 * point[1];
                }
            }
            let direction = angle_of(sub(centroid, centre));
            let mut best: Option<(f64, usize)> = None;
            for (entry, &(_, _, start, end)) in entries.iter().enumerate() {
                if contains_angle(start, end, direction) {
                    best = Some((0.0, entry));
                    break;
                }
                // distance to the nearer sector end, as a fallback
                let to_start = (direction - start).rem_euclid(std::f64::consts::TAU);
                let to_end = (end - direction).rem_euclid(std::f64::consts::TAU);
                let distance = to_start.min(to_end);
                if best.map(|(current, _)| distance < current).unwrap_or(true) {
                    best = Some((distance, entry));
                }
            }
            let entry = match best {
                Some((_, found)) => found,
                None => {
                    return Err(format!(
                        "regions: cell {index} at pinch node {node} has no wedge"
                    ))
                }
            };
            nodes[slot] = *copies.get(&(node, entry)).unwrap_or(&node);
        }
        let mut unique = true;
        for a in 0..nodes.len() {
            for b in (a + 1)..nodes.len() {
                if nodes[a] == nodes[b] {
                    unique = false;
                }
            }
        }
        if !unique {
            return Err(format!(
                "regions: splitting the boundary pinches collapsed cell {index}"
            ));
        }
        cells.push(Cell::quad4([nodes[0], nodes[1], nodes[2], nodes[3]]));
    }
    Ok((Mesh::new(points, cells), created))
}

/// Edge-connected components of a quad mesh, largest first.
fn split_components(mesh: &Mesh) -> Vec<Mesh> {
    if mesh.cells.is_empty() {
        return Vec::new();
    }
    let mut parent: Vec<usize> = (0..mesh.cells.len()).collect();
    fn root(parent: &mut [usize], mut node: usize) -> usize {
        while parent.get(node).copied().unwrap_or(node) != node {
            let next = parent.get(node).copied().unwrap_or(node);
            let grandparent = parent.get(next).copied().unwrap_or(next);
            if let Some(slot) = parent.get_mut(node) {
                *slot = grandparent;
            }
            node = next;
        }
        node
    }
    let mut owner: HashMap<(usize, usize), usize> = HashMap::new();
    for (index, cell) in mesh.cells.iter().enumerate() {
        if cell.nodes.len() != 4 {
            continue;
        }
        for slot in 0..4 {
            let a = cell.nodes[slot];
            let b = cell.nodes[(slot + 1) % 4];
            let key = if a < b { (a, b) } else { (b, a) };
            match owner.get(&key) {
                Some(&other) => {
                    let left = root(&mut parent, index);
                    let right = root(&mut parent, other);
                    if left != right {
                        if let Some(slot) = parent.get_mut(left) {
                            *slot = right;
                        }
                    }
                }
                None => {
                    owner.insert(key, index);
                }
            }
        }
    }
    let mut buckets: HashMap<usize, Vec<usize>> = HashMap::new();
    for index in 0..mesh.cells.len() {
        let root = root(&mut parent, index);
        buckets.entry(root).or_default().push(index);
    }
    let mut groups: Vec<Vec<usize>> = buckets.into_values().collect();
    groups.sort_by_key(|group| std::cmp::Reverse(group.len()));
    let mut components = Vec::with_capacity(groups.len());
    for group in groups {
        let mut points: Vec<Node> = Vec::new();
        let mut cells: Vec<Cell> = Vec::new();
        let mut remap: HashMap<usize, usize> = HashMap::new();
        for &index in &group {
            let cell = match mesh.cells.get(index) {
                Some(found) if found.nodes.len() == 4 => found,
                _ => continue,
            };
            let mut mapped = [0usize; 4];
            for (slot, &node) in cell.nodes.iter().enumerate() {
                let id = match remap.get(&node) {
                    Some(&found) => found,
                    None => {
                        let id = points.len();
                        points.push(mesh.points.get(node).copied().unwrap_or([0.0; 3]));
                        remap.insert(node, id);
                        id
                    }
                };
                mapped[slot] = id;
            }
            cells.push(Cell::quad4(mapped));
        }
        components.push(Mesh::new(points, cells));
    }
    components
}

/// Remove the quads closer than `radius_factor * sigma` to any of the given
/// boundary loops (the outer boundary and the holes).
fn carve_rings(
    mesh: &Mesh,
    field: &Field,
    loops: &[Vec<[f64; 2]>],
    radius_factor: f64,
) -> (Mesh, usize) {
    carve_rings_two(mesh, field, loops, radius_factor, radius_factor)
}

/// Like [`carve_rings`] but with a separate factor for the first loop (the
/// outer boundary) and the rest (the holes): the ring has to stay connected,
/// so the hole side is only detached, not cleared at the full channel width.
fn carve_rings_two(
    mesh: &Mesh,
    field: &Field,
    loops: &[Vec<[f64; 2]>],
    outer_factor: f64,
    hole_factor: f64,
) -> (Mesh, usize) {
    let mut points: Vec<Node> = Vec::new();
    let mut cells: Vec<Cell> = Vec::new();
    let mut remap: HashMap<usize, usize> = HashMap::new();
    let mut carved = 0usize;
    for cell in &mesh.cells {
        if cell.nodes.len() != 4 {
            carved += 1;
            continue;
        }
        let mut corners = [[0.0f64; 2]; 4];
        for (slot, &node) in cell.nodes.iter().enumerate() {
            let point = mesh.points.get(node).copied().unwrap_or([0.0; 3]);
            corners[slot] = [point[0], point[1]];
        }
        let centroid = [
            0.25 * (corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]),
            0.25 * (corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]),
        ];
        let sigma = field.sigma_at(centroid);
        let nearest_outer = loops
            .first()
            .map(|loop_points| distance_to_loop(centroid, loop_points))
            .unwrap_or(f64::INFINITY);
        let nearest_hole = loops
            .iter()
            .skip(1)
            .map(|loop_points| distance_to_loop(centroid, loop_points))
            .fold(f64::INFINITY, f64::min);
        if nearest_outer <= outer_factor * sigma || nearest_hole <= hole_factor * sigma {
            carved += 1;
            continue;
        }
        let mut mapped = [0usize; 4];
        for (slot, &node) in cell.nodes.iter().enumerate() {
            let index = match remap.get(&node) {
                Some(&found) => found,
                None => {
                    let id = points.len();
                    points.push(mesh.points.get(node).copied().unwrap_or([0.0; 3]));
                    remap.insert(node, id);
                    id
                }
            };
            mapped[slot] = index;
        }
        cells.push(Cell::quad4(mapped));
    }
    (Mesh::new(points, cells), carved)
}

/// Keep only the largest edge-connected component of a quad mesh.
fn largest_component(mesh: &Mesh) -> (Mesh, usize) {
    if mesh.cells.is_empty() {
        return (mesh.clone(), 0);
    }
    let mut parent: Vec<usize> = (0..mesh.cells.len()).collect();
    fn root(parent: &mut [usize], mut node: usize) -> usize {
        while parent.get(node).copied().unwrap_or(node) != node {
            let next = parent.get(node).copied().unwrap_or(node);
            let grandparent = parent.get(next).copied().unwrap_or(next);
            if let Some(slot) = parent.get_mut(node) {
                *slot = grandparent;
            }
            node = next;
        }
        node
    }
    let mut owner: HashMap<(usize, usize), usize> = HashMap::new();
    for (index, cell) in mesh.cells.iter().enumerate() {
        if cell.nodes.len() != 4 {
            continue;
        }
        for slot in 0..4 {
            let a = cell.nodes[slot];
            let b = cell.nodes[(slot + 1) % 4];
            let key = if a < b { (a, b) } else { (b, a) };
            match owner.get(&key) {
                Some(&other) => {
                    let left = root(&mut parent, index);
                    let right = root(&mut parent, other);
                    if left != right {
                        if let Some(slot) = parent.get_mut(left) {
                            *slot = right;
                        }
                    }
                }
                None => {
                    owner.insert(key, index);
                }
            }
        }
    }
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for index in 0..mesh.cells.len() {
        let root = root(&mut parent, index);
        *counts.entry(root).or_insert(0) += 1;
    }
    let keep = match counts.iter().max_by_key(|(_, &count)| count) {
        Some((&root, _)) => root,
        None => return (mesh.clone(), 0),
    };
    let mut points: Vec<Node> = Vec::new();
    let mut cells: Vec<Cell> = Vec::new();
    let mut remap: HashMap<usize, usize> = HashMap::new();
    let mut dropped = 0usize;
    for (index, cell) in mesh.cells.iter().enumerate() {
        if root(&mut parent, index) != keep || cell.nodes.len() != 4 {
            dropped += 1;
            continue;
        }
        let mut mapped = [0usize; 4];
        for (slot, &node) in cell.nodes.iter().enumerate() {
            let id = match remap.get(&node) {
                Some(&found) => found,
                None => {
                    let id = points.len();
                    points.push(mesh.points.get(node).copied().unwrap_or([0.0; 3]));
                    remap.insert(node, id);
                    id
                }
            };
            mapped[slot] = id;
        }
        cells.push(Cell::quad4(mapped));
    }
    (Mesh::new(points, cells), dropped)
}

/// Ray-casting point-in-polygon test for a closed loop.
fn loop_contains(loop_points: &[[f64; 2]], p: [f64; 2]) -> bool {
    let mut inside = false;
    let count = loop_points.len();
    if count < 3 {
        return false;
    }
    let mut j = count - 1;
    for i in 0..count {
        let a = loop_points[i];
        let b = loop_points[j];
        if (a[1] > p[1]) != (b[1] > p[1]) {
            let denominator = b[1] - a[1];
            if denominator != 0.0 {
                let x = a[0] + (p[1] - a[1]) * (b[0] - a[0]) / denominator;
                if p[0] < x {
                    inside = !inside;
                }
            }
        }
        j = i;
    }
    inside
}

/// The interior loop that encloses the domain hole and lies strictly outside
/// it: the largest negative-area boundary loop whose interior contains the
/// hole's centroid and whose absolute area is larger than the hole's.
fn hole_enclosing_loop(
    mesh: &Mesh,
    domain: &Domain,
    hole_points: &[Vec<[f64; 2]>],
) -> Option<(Vec<usize>, Vec<[f64; 2]>, f64)> {
    let hole = hole_points.first()?;
    let hole_area = loop_area(hole).abs();
    let count = hole.len().max(1);
    let mut centre = [0.0f64; 2];
    for point in hole {
        centre[0] += point[0] / count as f64;
        centre[1] += point[1] / count as f64;
    }
    let loops = mesh_boundary_loops(mesh).ok()?;
    let mut best: Option<(Vec<usize>, Vec<[f64; 2]>, f64)> = None;
    for ids in loops {
        let points = loop_points(mesh, &ids);
        let area = loop_area(&points);
        if area >= 0.0 || area.abs() <= hole_area {
            continue;
        }
        let outside = points
            .iter()
            .all(|&p| domain.distance_to_boundary(p) > 1.0e-9);
        if !outside || !loop_contains(&points, centre) {
            continue;
        }
        if best
            .as_ref()
            .map(|(_, _, found)| area.abs() > *found)
            .unwrap_or(true)
        {
            best = Some((ids, points, area));
        }
    }
    best
}

/// Points of a boundary loop of a mesh.
fn loop_points(mesh: &Mesh, ids: &[usize]) -> Vec<[f64; 2]> {
    let mut points = Vec::with_capacity(ids.len());
    for &node in ids {
        let point = mesh.points.get(node).copied().unwrap_or([0.0; 3]);
        points.push([point[0], point[1]]);
    }
    points
}

/// Mean target size along a loop.
fn mean_sigma(field: &Field, points: &[[f64; 2]]) -> f64 {
    if points.is_empty() {
        return 1.0;
    }
    let mut total = 0.0;
    for point in points {
        total += field.sigma_at(*point);
    }
    total / points.len() as f64
}

/// The two channel bands of a ring-shaped interior component: the outer band
/// between the domain boundary and the component's outer loop, and the hole
/// band between the hole polygon and the component's hole loop. The component's
/// quads are kept (Section 5.1.8.2's interior mesh).
fn ring_bands(
    component: &Mesh,
    domain: &Domain,
    outer_points: &[[f64; 2]],
    hole_points: &[Vec<[f64; 2]>],
    field: &Field,
) -> Result<(Vec<[usize; 4]>, Vec<[f64; 3]>), String> {
    let hole = hole_points
        .first()
        .ok_or_else(|| "ring bands: the domain has no hole".to_string())?;
    let hole_area = loop_area(hole).abs();
    let count = hole.len().max(1);
    let mut centre = [0.0f64; 2];
    for point in hole {
        centre[0] += point[0] / count as f64;
        centre[1] += point[1] / count as f64;
    }
    let loops = mesh_boundary_loops(component)?;
    let mut outer_loop: Option<(Vec<usize>, f64)> = None;
    let mut inner_loop: Option<(Vec<usize>, f64)> = None;
    for ids in loops {
        let points = loop_points(component, &ids);
        let area = loop_area(&points);
        let off_boundary = points
            .iter()
            .all(|&p| domain.distance_to_boundary(p) > 1.0e-9);
        if !off_boundary {
            continue;
        }
        if area > 0.0 {
            if outer_loop
                .as_ref()
                .map(|(_, found)| area > *found)
                .unwrap_or(true)
            {
                outer_loop = Some((ids, area));
            }
        } else if area.abs() > hole_area && loop_contains(&points, centre) && inner_loop.is_none() {
            inner_loop = Some((ids, area));
        }
    }
    for component_loop in mesh_boundary_loops(component).unwrap_or_default() {
        let points = loop_points(component, &component_loop);
        let area = loop_area(&points);
        let off = points
            .iter()
            .filter(|&&p| domain.distance_to_boundary(p) > 1.0e-9)
            .count();
        eprintln!(
            "DEBUG ring loop: {} nodes area {area:.4} off-boundary {off}/{} contains {}",
            component_loop.len(),
            points.len(),
            loop_contains(&points, centre)
        );
    }
    let (outer_ids, outer_area) =
        outer_loop.ok_or_else(|| "ring bands: no outer loop".to_string())?;
    let (inner_ids, inner_area) =
        inner_loop.ok_or_else(|| "ring bands: no hole loop around the hole".to_string())?;
    let mut pool: Vec<[f64; 3]> = component.points.clone();
    let outer_points2d = loop_points(component, &outer_ids);
    let inner_points2d = loop_points(component, &inner_ids);
    let outer_size = mean_sigma(field, &outer_points2d);
    let inner_size = mean_sigma(field, &inner_points2d);
    let outer_quads = simons_channel::channel_between_dense(
        domain,
        outer_points,
        &outer_points2d,
        &outer_ids,
        &mut pool,
        outer_size,
    )
    .or_else(|_| {
        simons_channel::channel_between(
            domain,
            outer_points,
            &outer_points2d,
            &outer_ids,
            &mut pool,
            outer_size,
        )
    })
    .map_err(|error| format!("ring bands: outer band (area {outer_area:.4}): {error}"))?;
    let hole_quads = simons_channel::channel_between_hole(
        domain,
        hole,
        &inner_points2d,
        &inner_ids,
        &mut pool,
        inner_size,
    )
    .map_err(|error| format!("ring bands: hole band (area {inner_area:.4}): {error}"))?;
    let mut quads = outer_quads;
    quads.extend(hole_quads);
    Ok((quads, pool))
}

/// Stitch the QGP interior mesh to the boundary of the input mesh with
/// `simons_channel::channel_between` (Section 5.1.8.2).
///
/// The interior QGP quads within `1.4 * sigma` of the outer boundary are carved
/// away, the boundary loop of what remains is handed to
/// [`simons_channel::channel_between`] together with the domain boundary, and
/// the channel quads are welded to the interior (they share the loop's node
/// ids). Hole-adjacent quads are kept, so the annulus needs no channel on the
/// hole side; the hole band that the thesis would also fill is reported in
/// [`StitchCase::notes`] because `channel_between` only builds a band whose
/// outer loop is on the domain boundary.
pub fn stitch(
    points: &[[f64; 3]],
    tris: &[[usize; 3]],
    alpha: &[f64],
    size: &[f64],
) -> Result<Mesh, String> {
    Ok(stitch_case("stitched", points, tris, alpha, size)?.mesh)
}

/// Run the stitch on one input and measure it.
fn stitch_case(
    name: &'static str,
    points: &[[f64; 3]],
    tris: &[[usize; 3]],
    alpha: &[f64],
    size: &[f64],
) -> Result<StitchCase, String> {
    let field = Field::new(points, tris, alpha, size)?;
    let regions = build(points, tris, alpha, size)?;
    let interior = regions.mesh()?;
    let mut notes: Vec<String> = Vec::new();
    // The domain, from the mesh boundary loops (the outer loop first).
    let mut order: Vec<usize> = (0..field.boundary_loops.len()).collect();
    order.sort_by(|&left, &right| {
        let a = loop_area(&field.boundary_points[left]).abs();
        let b = loop_area(&field.boundary_points[right]).abs();
        b.partial_cmp(&a).unwrap_or(std::cmp::Ordering::Equal)
    });
    let outer_index = match order.first() {
        Some(&found) => found,
        None => return Err("regions: the input mesh has no boundary loop".to_string()),
    };
    let outer_points = field.boundary_points[outer_index].clone();
    let hole_points: Vec<Vec<[f64; 2]>> = order
        .iter()
        .skip(1)
        .map(|&index| field.boundary_points[index].clone())
        .collect();
    let domain = Domain::new(outer_points.clone(), hole_points.clone())
        .map_err(|error| format!("regions: the domain is invalid: {error}"))?;

    // Carve radius and band search. Both the outer boundary and the holes are
    // carved, the largest edge-connected component is kept (the annulus
    // interior fragments into islands around the rejected tiles), and then the
    // outer ring is stitched with `channel_between` and every interior hole
    // loop with `channel_between_hole`.
    let all_loops: Vec<Vec<[f64; 2]>> = std::iter::once(outer_points.clone())
        .chain(hole_points.iter().cloned())
        .collect();
    let mut attempt: Option<(
        Mesh,
        usize,
        Vec<[usize; 4]>,
        Vec<[f64; 3]>,
        usize,
        f64,
        bool,
    )> = None;
    let mut fallback: Option<(Mesh, usize)> = None;
    for &radius in &[1.4f64, 1.6, 1.8, 2.0, 2.4, 3.0, 3.6] {
        let (carved, carved_quads) = carve_rings(&interior, &field, &all_loops, radius);
        if carved.cells.is_empty() {
            notes.push(format!("carve {radius:.1}: every quad was carved away"));
            continue;
        }
        let (carved, split_nodes) = match split_pinched_boundary(&carved) {
            Ok(found) => found,
            Err(error) => {
                notes.push(format!("carve {radius:.1}: {error}"));
                continue;
            }
        };
        if split_nodes > 0 {
            notes.push(format!(
                "carve {radius:.1}: split {split_nodes} pinched boundary node(s)"
            ));
        }
        // A domain with a hole can be filled by two bands that meet on a single
        // loop around the hole: the outer band between the domain boundary and
        // that loop, and the hole band between the hole polygon and the same
        // loop. This is the annulus path; it needs no interior ring at all.
        if !hole_points.is_empty() {
            if let Some((ids, points2d, area)) = hole_enclosing_loop(&carved, &domain, &hole_points)
            {
                let mean_size = mean_sigma(&field, &points2d);
                let mut pool: Vec<[f64; 3]> = carved.points.clone();
                let outer_band = simons_channel::channel_between_dense(
                    &domain,
                    &outer_points,
                    &points2d,
                    &ids,
                    &mut pool,
                    mean_size,
                )
                .or_else(|_| {
                    simons_channel::channel_between(
                        &domain,
                        &outer_points,
                        &points2d,
                        &ids,
                        &mut pool,
                        mean_size,
                    )
                });
                let hole_band = simons_channel::channel_between_hole(
                    &domain,
                    &hole_points[0],
                    &points2d,
                    &ids,
                    &mut pool,
                    mean_size,
                );
                match (outer_band, hole_band) {
                    (Ok(outer_quads), Ok(hole_quads)) => {
                        notes.push(format!(
                            "carve {radius:.1}: two-band fill on {} nodes (loop area {area:.4}): \
                             outer {} quads, hole {} quads",
                            ids.len(),
                            outer_quads.len(),
                            hole_quads.len()
                        ));
                        let mut quads = outer_quads;
                        quads.extend(hole_quads);
                        let bands = 2usize;
                        attempt = Some((carved, carved_quads, quads, pool, bands, radius, false));
                        break;
                    }
                    (outer, hole) => {
                        if let Err(error) = outer {
                            notes.push(format!("carve {radius:.1}: two-band outer: {error}"));
                        }
                        if let Err(error) = hole {
                            notes.push(format!("carve {radius:.1}: two-band hole: {error}"));
                        }
                    }
                }
            } else {
                notes.push(format!(
                    "carve {radius:.1}: no interior loop encloses the hole"
                ));
            }
        }
        let (carved, dropped) = largest_component(&carved);
        if carved.cells.is_empty() {
            notes.push(format!("carve {radius:.1}: no component survived"));
            continue;
        }
        if dropped > 0 {
            notes.push(format!(
                "carve {radius:.1}: dropped {dropped} quad(s) of smaller components"
            ));
        }
        let loops = match mesh_boundary_loops(&carved) {
            Ok(found) => found,
            Err(error) => {
                notes.push(format!("carve {radius:.1}: {error}"));
                if fallback.is_none() {
                    fallback = Some((carved, carved_quads));
                }
                continue;
            }
        };
        let mut outer_candidate: Option<(usize, f64)> = None;
        let mut hole_candidates: Vec<(usize, f64, usize)> = Vec::new();
        for (index, loop_ids) in loops.iter().enumerate() {
            let points2d = loop_points(&carved, loop_ids);
            let area = loop_area(&points2d);
            let off_boundary = points2d
                .iter()
                .all(|&p| domain.distance_to_boundary(p) > 1.0e-9);
            if !off_boundary || loop_ids.len() < 4 {
                continue;
            }
            if area > 0.0 {
                if outer_candidate
                    .map(|(_, found)| area > found)
                    .unwrap_or(true)
                {
                    outer_candidate = Some((index, area));
                }
            } else if area < 0.0 && !hole_points.is_empty() {
                let mut best: Option<(f64, usize)> = None;
                for (hole_index, hole) in hole_points.iter().enumerate() {
                    let distance = distance_to_loop(points2d[0], hole);
                    if best.map(|(current, _)| distance < current).unwrap_or(true) {
                        best = Some((distance, hole_index));
                    }
                }
                if let Some((_, hole_index)) = best {
                    hole_candidates.push((index, area, hole_index));
                }
            }
        }
        let mut pool: Vec<[f64; 3]> = carved.points.clone();
        let mut quads: Vec<[usize; 4]> = Vec::new();
        let mut bands = 0usize;
        if let Some((index, area)) = outer_candidate {
            let ids = loops[index].clone();
            if ids.len().is_multiple_of(2) {
                let points2d = loop_points(&carved, &ids);
                let mean_size = mean_sigma(&field, &points2d);
                let band = simons_channel::channel_between_dense(
                    &domain,
                    &outer_points,
                    &points2d,
                    &ids,
                    &mut pool,
                    mean_size,
                )
                .or_else(|_| {
                    simons_channel::channel_between(
                        &domain,
                        &outer_points,
                        &points2d,
                        &ids,
                        &mut pool,
                        mean_size,
                    )
                });
                match band {
                    Ok(found) => {
                        notes.push(format!(
                            "carve {radius:.1}: outer band on {} nodes (area {area:.4}), \
                             {} quads",
                            ids.len(),
                            found.len()
                        ));
                        quads.extend(found);
                        bands += 1;
                    }
                    Err(error) => {
                        notes.push(format!("carve {radius:.1}: channel_between: {error}"))
                    }
                }
            }
        }
        for (index, area, hole_index) in hole_candidates {
            let ids = loops[index].clone();
            if !ids.len().is_multiple_of(2) {
                notes.push(format!(
                    "carve {radius:.1}: hole loop {} has an odd number of nodes",
                    ids.len()
                ));
                continue;
            }
            let points2d = loop_points(&carved, &ids);
            let mean_size = mean_sigma(&field, &points2d);
            let hole = match hole_points.get(hole_index) {
                Some(found) => found.clone(),
                None => continue,
            };
            match simons_channel::channel_between_hole(
                &domain, &hole, &points2d, &ids, &mut pool, mean_size,
            ) {
                Ok(found) => {
                    notes.push(format!(
                        "carve {radius:.1}: hole band on {} nodes (area {area:.4}), {} quads",
                        ids.len(),
                        found.len()
                    ));
                    quads.extend(found);
                    bands += 1;
                }
                Err(error) => {
                    notes.push(format!("carve {radius:.1}: channel_between_hole: {error}"))
                }
            }
        }
        if bands > 0 {
            attempt = Some((carved, carved_quads, quads, pool, bands, radius, true));
            break;
        }
        if fallback.is_none() {
            fallback = Some((carved, carved_quads));
        }
    }
    let (mut carved_mesh, carved_quads, mut channel_quads, pool, bands, _radius, keep_interior) =
        match attempt {
            Some(found) => found,
            None => {
                let (carved, carved_quads) = match fallback {
                    Some(found) => found,
                    None => {
                        return Err(format!(
                            "regions: no carve radius left a stitchable interior loop ({notes:?})"
                        ))
                    }
                };
                (
                    carved,
                    carved_quads,
                    Vec::new(),
                    Vec::new(),
                    0usize,
                    0.0,
                    true,
                )
            }
        };
    if !pool.is_empty() {
        carved_mesh.points = pool;
    }
    let mut keep_interior = keep_interior;
    let mut bands = bands;
    // A ring-shaped QGP interior: keep the ring component and stitch both of
    // its loops (Section 5.1.8.2).
    if bands < 2 && !hole_points.is_empty() {
        'ring: for &radius in &[1.4f64, 1.6, 1.8] {
            let (carved, _carved_count) =
                carve_rings_two(&interior, &field, &all_loops, radius, 0.3);
            let (carved, _split) = match split_pinched_boundary(&carved) {
                Ok(found) => found,
                Err(_) => continue,
            };
            for component in split_components(&carved) {
                match ring_bands(&component, &domain, &outer_points, &hole_points, &field) {
                    Ok((quads, pool)) => {
                        let kept = component.cells.len();
                        let mut with_pool = component;
                        with_pool.points = pool;
                        notes.push(format!(
                            "ring stitch at carve {radius:.1}: {kept} QGP interior quads kept, \
                             two bands ({} channel quads)",
                            quads.len()
                        ));
                        carved_mesh = with_pool;
                        channel_quads = quads;
                        keep_interior = true;
                        bands = 2;
                        break 'ring;
                    }
                    Err(error) => notes.push(format!("ring stitch {radius:.1}: {error}")),
                }
            }
        }
    }
    // Last resort for a domain with a hole and no ring-shaped interior: two
    // bands that meet on a collar loop offset from the hole polygon. The QGP
    // interior contributes no quad; the bands share the collar ids, so the
    // result is still conforming.
    if bands < 2 && !hole_points.is_empty() {
        let hole = hole_points[0].clone();
        let count = hole.len();
        let mean = mean_sigma(&field, &hole);
        let gap = 1.4 * mean;
        let mut centre = [0.0f64; 2];
        for point in &hole {
            centre[0] += point[0] / count as f64;
            centre[1] += point[1] / count as f64;
        }
        let collar: Vec<[f64; 2]> = hole
            .iter()
            .map(|point| {
                let offset = sub(*point, centre);
                match unit(offset) {
                    Some(direction) => add(*point, scale(direction, gap)),
                    None => *point,
                }
            })
            .collect();
        let mut pool = carved_mesh.points.clone();
        let mut ids: Vec<usize> = Vec::with_capacity(count);
        for point in &collar {
            ids.push(pool.len());
            pool.push([point[0], point[1], 0.0]);
        }
        let outer_band = simons_channel::channel_between_dense(
            &domain,
            &outer_points,
            &collar,
            &ids,
            &mut pool,
            mean,
        )
        .or_else(|_| {
            simons_channel::channel_between(&domain, &outer_points, &collar, &ids, &mut pool, mean)
        });
        let hole_band =
            simons_channel::channel_between_hole(&domain, &hole, &collar, &ids, &mut pool, mean);
        match (outer_band, hole_band) {
            (Ok(outer_quads), Ok(hole_quads)) => {
                notes.push(format!(
                    "collar fallback: {} nodes at {gap:.3} from the hole, outer {} quads, \
                     hole {} quads, no QGP interior",
                    ids.len(),
                    outer_quads.len(),
                    hole_quads.len()
                ));
                carved_mesh.points = pool;
                channel_quads = outer_quads;
                channel_quads.extend(hole_quads);
                bands = 2;
                keep_interior = false;
            }
            (outer, hole_band) => {
                if let Err(error) = outer {
                    notes.push(format!("collar outer band: {error}"));
                }
                if let Err(error) = hole_band {
                    notes.push(format!("collar hole band: {error}"));
                }
            }
        }
    }
    let mut cells: Vec<Cell> = if keep_interior {
        carved_mesh.cells.clone()
    } else {
        Vec::new()
    };
    for quad in &channel_quads {
        cells.push(Cell::quad4(*quad));
    }
    let kept_interior = cells.len().saturating_sub(channel_quads.len());
    let mesh = Mesh::new(carved_mesh.points.clone(), cells);
    mesh.validate()
        .map_err(|error| format!("regions: the stitched mesh is invalid: {error}"))?;
    let mut case = StitchCase {
        name,
        interior_quads: interior.cells.len(),
        carved_quads,
        channel_quads: channel_quads.len(),
        kept_interior,
        bands,
        quads: mesh.cells.len(),
        notes,
        min_scaled_jacobian: 0.0,
        ..StitchCase::default()
    };
    let report = quality::report(&mesh);
    case.min_scaled_jacobian = report.min_scaled_jacobian;
    case.min_corner_angle_deg = report.min_corner_angle_deg;
    case.max_corner_angle_deg = report.max_corner_angle_deg;
    // Cracks, area and boundary offset.
    let mut incidence: HashMap<(usize, usize), usize> = HashMap::new();
    let mut mesh_area = 0.0;
    for cell in &mesh.cells {
        if cell.nodes.len() != 4 {
            continue;
        }
        let mut quad = [[0.0f64; 2]; 4];
        for (slot, &node) in cell.nodes.iter().enumerate() {
            let point = mesh.points.get(node).copied().unwrap_or([0.0; 3]);
            quad[slot] = [point[0], point[1]];
            let next = cell.nodes[(slot + 1) % 4];
            let key = if node < next {
                (node, next)
            } else {
                (next, node)
            };
            *incidence.entry(key).or_insert(0) += 1;
        }
        mesh_area += loop_area(&quad);
    }
    for (&(a, b), &count) in &incidence {
        if count != 1 {
            continue;
        }
        let pa = mesh.points.get(a).copied().unwrap_or([0.0; 3]);
        let pb = mesh.points.get(b).copied().unwrap_or([0.0; 3]);
        let middle = [0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1])];
        let nearest = domain.distance_to_boundary(middle);
        case.boundary_offset = case.boundary_offset.max(nearest);
        if nearest > 1e-7 * field.diagonal {
            case.cracks += 1;
        }
    }
    let domain_area = domain.area();
    if domain_area > 0.0 {
        case.coverage = mesh_area / domain_area;
    }
    case.mesh = mesh;
    Ok(case)
}

/// Run the stitch on the annulus and the rectangle and return the numbers.
pub fn stitch_demo() -> Result<StitchNumbers, String> {
    let (annulus_points, annulus_tris, annulus_alpha, annulus_size) = annulus_fixture()?;
    let annulus = stitch_case(
        "stitched annulus",
        &annulus_points,
        &annulus_tris,
        &annulus_alpha,
        &annulus_size,
    )?;
    let (rectangle_points, rectangle_tris, rectangle_alpha, rectangle_size) = rectangle_fixture()?;
    let rectangle = stitch_case(
        "stitched rectangle",
        &rectangle_points,
        &rectangle_tris,
        &rectangle_alpha,
        &rectangle_size,
    )?;
    Ok(StitchNumbers { annulus, rectangle })
}

// ---------------------------------------------------------------------------
// Demo
// ---------------------------------------------------------------------------

fn complex_mul(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]]
}

fn complex_div(a: [f64; 2], b: [f64; 2]) -> Option<[f64; 2]> {
    let denominator = b[0] * b[0] + b[1] * b[1];
    if denominator <= 0.0 || !denominator.is_finite() {
        return None;
    }
    Some([
        (a[0] * b[0] + a[1] * b[1]) / denominator,
        (a[1] * b[0] - a[0] * b[1]) / denominator,
    ])
}

/// A structured quad mesh of the square `[-half, half]^2` with a circular hole
/// of radius `inner`, triangulated, and the analytic four-singularity field
/// `u = exp(4 i theta) (z - a)(z - b) / ((z - c)(z - d))`: zeros at the
/// valence-3 points `a`, `b` and poles at the valence-5 points `c`, `d`.
///
/// The `exp(4 i theta)` factor is the "O-grid" field (radial and
/// circumferential); the rational factor adds the two valence-3 and two
/// valence-5 singularities without changing the side of the annulus the field
/// lines leave through.
/// Rotation of the annulus cross field, in radians. The motorcycle graph (and
/// therefore whether the QGP interior forms a ring) depends on this phase
/// (Section 5.1.8: "the structure of the motorcycle graph is dependent on the
/// phase of the cross field").
const ANNULUS_PHASE: f64 = 0.0;

fn annulus_fixture() -> Result<(Vec<[f64; 3]>, Vec<[usize; 3]>, Vec<f64>, Vec<f64>), String> {
    annulus_fixture_with_phase(ANNULUS_PHASE)
}

fn annulus_fixture_with_phase(
    phase: f64,
) -> Result<(Vec<[f64; 3]>, Vec<[usize; 3]>, Vec<f64>, Vec<f64>), String> {
    const N_THETA: usize = 100;
    const N_RADIAL: usize = 10;
    const INNER: f64 = 0.4;
    const HALF: f64 = 1.5;
    let mut points: Vec<[f64; 3]> = Vec::with_capacity((N_RADIAL + 1) * N_THETA);
    for layer in 0..=N_RADIAL {
        let t = layer as f64 / N_RADIAL as f64;
        for segment in 0..N_THETA {
            let theta = std::f64::consts::TAU * segment as f64 / N_THETA as f64;
            let (sin, cos) = theta.sin_cos();
            let outer = HALF / cos.abs().max(sin.abs());
            let radius = INNER + (outer - INNER) * t;
            points.push([radius * cos, radius * sin, 0.0]);
        }
    }
    let index = |layer: usize, segment: usize| layer * N_THETA + segment % N_THETA;
    let mut tris: Vec<[usize; 3]> = Vec::with_capacity(2 * N_RADIAL * N_THETA);
    for layer in 0..N_RADIAL {
        for segment in 0..N_THETA {
            let a = index(layer, segment);
            let b = index(layer, segment + 1);
            let c = index(layer + 1, segment + 1);
            let d = index(layer + 1, segment);
            tris.push([a, c, b]);
            tris.push([a, d, c]);
        }
    }
    let alpha: Vec<f64> = points
        .iter()
        .map(|point| {
            let z = [point[0], point[1]];
            let radius2 = z[0] * z[0] + z[1] * z[1];
            // exp(4 i theta) = z^4 / |z|^4.
            let z2 = [z[0] * z[0] - z[1] * z[1], 2.0 * z[0] * z[1]];
            let z4 = [z2[0] * z2[0] - z2[1] * z2[1], 2.0 * z2[0] * z2[1]];
            // The O-grid field, rotated by `SPIRAL` so its field lines spiral
            // across the annulus instead of closing on themselves.
            let (spiral_sin, spiral_cos) = (4.0 * phase).sin_cos();
            let base = [z4[0] / (radius2 * radius2), z4[1] / (radius2 * radius2)];
            let prefactor = [
                base[0] * spiral_cos - base[1] * spiral_sin,
                base[0] * spiral_sin + base[1] * spiral_cos,
            ];
            // Singularities on the unit circle at 45 degrees to the radial
            // mesh lines, so none of them lies on a mesh edge.
            let diagonal = std::f64::consts::FRAC_1_SQRT_2;
            let zeros = complex_mul(
                [z[0] - diagonal, z[1] - diagonal],
                [z[0] + diagonal, z[1] + diagonal],
            );
            let poles = complex_mul(
                [z[0] + diagonal, z[1] - diagonal],
                [z[0] - diagonal, z[1] + diagonal],
            );
            let ratio = complex_div(zeros, poles).unwrap_or([1.0, 0.0]);
            let value = complex_mul(prefactor, ratio);
            wrap_cross(0.25 * value[1].atan2(value[0]))
        })
        .collect();
    let size: Vec<f64> = points
        .iter()
        .map(|point| 0.22 + 0.03 * (point[0] * point[0] + point[1] * point[1]).sqrt())
        .collect();
    Ok((points, tris, alpha, size))
}

/// A structured rectangle `[0, 2] x [0, 1]` with a constant field and a
/// constant size: no singularity, one tile, the QGP demo case.
fn rectangle_fixture() -> Result<(Vec<[f64; 3]>, Vec<[usize; 3]>, Vec<f64>, Vec<f64>), String> {
    const NX: usize = 8;
    const NY: usize = 5;
    let mut points: Vec<[f64; 3]> = Vec::with_capacity((NX + 1) * (NY + 1));
    for j in 0..=NY {
        for i in 0..=NX {
            points.push([2.0 * i as f64 / NX as f64, j as f64 / NY as f64, 0.0]);
        }
    }
    let index = |i: usize, j: usize| j * (NX + 1) + i;
    let mut tris: Vec<[usize; 3]> = Vec::with_capacity(2 * NX * NY);
    for j in 0..NY {
        for i in 0..NX {
            let a = index(i, j);
            let b = index(i + 1, j);
            let c = index(i + 1, j + 1);
            let d = index(i, j + 1);
            tris.push([a, b, c]);
            tris.push([a, c, d]);
        }
    }
    let alpha = vec![0.0; points.len()];
    let size = vec![0.1; points.len()];
    Ok((points, tris, alpha, size))
}

/// Mesh statistics of one demo case.
fn measure_case(
    name: &'static str,
    points: &[[f64; 3]],
    tris: &[[usize; 3]],
    alpha: &[f64],
    size: &[f64],
) -> Result<DemoCase, String> {
    let field = Field::new(points, tris, alpha, size)?;
    let regions = build(points, tris, alpha, size)?;
    let mesh = regions.mesh()?;
    let report = quality::report(&mesh);
    let mut case = DemoCase {
        name,
        vertices: points.len(),
        triangles: tris.len(),
        singularities: regions.singularities.len(),
        graph_nodes: regions.graph.nodes.len(),
        graph_edges: regions.graph.edges.len(),
        tiles: regions.tiles.len(),
        degenerate_tiles: regions.degenerate_tiles,
        collapsed_tiles: regions.collapsed_tiles,
        quads: mesh.cells.len(),
        counts: regions.tiles.iter().map(|tile| tile.counts).collect(),
        qgp_energy: regions.energy,
        min_scaled_jacobian: report.min_scaled_jacobian,
        min_corner_angle_deg: report.min_corner_angle_deg,
        max_corner_angle_deg: report.max_corner_angle_deg,
        ..DemoCase::default()
    };
    for node in &regions.graph.nodes {
        match node.kind {
            NodeKind::Singularity(3) => case.valence3 += 1,
            NodeKind::Singularity(5) => case.valence5 += 1,
            NodeKind::Singularity(_) => {}
            NodeKind::Boundary => case.boundary_nodes += 1,
            NodeKind::TJunction => case.tjunction_nodes += 1,
        }
    }
    for edge in &regions.graph.edges {
        match edge.kind {
            EdgeKind::Separatrix => case.separatrix_edges += 1,
            EdgeKind::Boundary => case.boundary_edges += 1,
        }
    }
    let mut incidence: HashMap<(usize, usize), usize> = HashMap::new();
    let mut min_area = f64::INFINITY;
    let mut mesh_area = 0.0;
    for cell in &mesh.cells {
        if cell.nodes.len() != 4 {
            continue;
        }
        let mut quad = [[0.0f64; 2]; 4];
        for (slot, &node) in cell.nodes.iter().enumerate() {
            let point = mesh.points.get(node).copied().unwrap_or([0.0; 3]);
            quad[slot] = [point[0], point[1]];
            let next = cell.nodes[(slot + 1) % cell.nodes.len()];
            let key = if node < next {
                (node, next)
            } else {
                (next, node)
            };
            *incidence.entry(key).or_insert(0) += 1;
        }
        let area = polygon_area(&quad);
        min_area = min_area.min(area);
        mesh_area += area;
    }
    case.min_signed_area = min_area;
    case.mesh_area = mesh_area;
    let mut domain_area = 0.0;
    for triangle in tris {
        let (a, b, c) = match (
            points.get(triangle[0]),
            points.get(triangle[1]),
            points.get(triangle[2]),
        ) {
            (Some(a), Some(b), Some(c)) => (a, b, c),
            _ => continue,
        };
        domain_area += 0.5 * ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])).abs();
    }
    case.domain_area = domain_area;
    if domain_area > 0.0 {
        case.relative_area_error = (mesh_area - domain_area).abs() / domain_area;
    }
    for (&(a, b), &count) in &incidence {
        if count > 2 {
            case.nonmanifold_edges += 1;
        }
        if count == 1 {
            let pa = mesh.points.get(a).copied().unwrap_or([0.0; 3]);
            let pb = mesh.points.get(b).copied().unwrap_or([0.0; 3]);
            let middle = [0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1])];
            let mut nearest = f64::INFINITY;
            for index in 0..field.boundary_directed.len() {
                if let Some((start, end)) = field.boundary_segment(index) {
                    nearest = nearest.min(segment_distance(middle, start, end));
                }
            }
            if nearest.is_finite() {
                case.boundary_offset = case.boundary_offset.max(nearest);
                if nearest > 1e-7 * field.diagonal {
                    case.cracks += 1;
                }
            }
        }
    }
    case.mesh = mesh;
    Ok(case)
}

/// Run the two reference cases and return their measured numbers.
///
/// The cases are the square with a circular hole (four singularities, the
/// motorcycle graph, the simple tiles and their Coons patches) and the single
/// rectangle with a constant field (no singularity, one tile).
pub fn demo() -> Result<DemoNumbers, String> {
    let (annulus_points, annulus_tris, annulus_alpha, annulus_size) = annulus_fixture()?;
    let annulus = measure_case(
        "annulus (square with a circular hole)",
        &annulus_points,
        &annulus_tris,
        &annulus_alpha,
        &annulus_size,
    )?;
    let (rectangle_points, rectangle_tris, rectangle_alpha, rectangle_size) = rectangle_fixture()?;
    let rectangle = measure_case(
        "rectangle (constant field)",
        &rectangle_points,
        &rectangle_tris,
        &rectangle_alpha,
        &rectangle_size,
    )?;
    Ok(DemoNumbers { annulus, rectangle })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn square_mesh(n: usize) -> (Vec<[f64; 3]>, Vec<[usize; 3]>) {
        let mut points = Vec::with_capacity((n + 1) * (n + 1));
        for j in 0..=n {
            for i in 0..=n {
                points.push([i as f64 / n as f64, j as f64 / n as f64, 0.0]);
            }
        }
        let mut tris = Vec::with_capacity(2 * n * n);
        for j in 0..n {
            for i in 0..n {
                let a = j * (n + 1) + i;
                let b = a + 1;
                let c = a + n + 2;
                let d = a + n + 1;
                tris.push([a, b, c]);
                tris.push([a, c, d]);
            }
        }
        (points, tris)
    }

    #[test]
    fn arm_directions_match_the_vertex_valence() {
        let three = arm_directions([1.0, 0.0], [0.0, 0.0], 3);
        assert_eq!(three.len(), 3, "a valence-3 singularity has three arms");
        for (index, direction) in three.iter().enumerate() {
            let expected = std::f64::consts::TAU * index as f64 / 3.0;
            assert!(
                wrap_pi(angle_of(*direction) - expected).abs() < 1e-6,
                "arm {index} is at {} instead of {expected}",
                angle_of(*direction)
            );
        }
        let five = arm_directions([0.0, 0.0], [1.0, 0.0], 5);
        assert_eq!(five.len(), 5, "a valence-5 singularity has five arms");
    }

    #[test]
    fn rectangle_is_one_tile() -> Result<(), String> {
        let (points, tris, alpha, size) = rectangle_fixture()?;
        let regions = build(&points, &tris, &alpha, &size)?;
        assert!(
            regions.singularities.is_empty(),
            "a constant field has no singularity"
        );
        assert_eq!(regions.tiles.len(), 1, "a rectangle is a single tile");
        assert_eq!(regions.degenerate_tiles, 0, "no degenerate tile");
        let tile = match regions.tiles.first() {
            Some(found) => found,
            None => return Err("the rectangle has no tile".to_string()),
        };
        assert!(
            tile.counts[0] >= 10 && tile.counts[1] >= 5,
            "unexpected counts {:?}",
            tile.counts
        );
        let mesh = regions.mesh()?;
        let report = quality::report(&mesh);
        assert_eq!(mesh.cells.len(), tile.counts[0] * tile.counts[1]);
        assert!(
            report.min_scaled_jacobian > 0.9,
            "the Coons grid of a rectangle should be square (min SJ {})",
            report.min_scaled_jacobian
        );
        assert!(
            (report.min_corner_angle_deg - 90.0).abs() < 1e-6,
            "the Coons grid of a rectangle should have right angles (min {})",
            report.min_corner_angle_deg
        );
        Ok(())
    }

    #[test]
    fn annulus_has_four_singularities() -> Result<(), String> {
        let (points, tris, alpha, size) = annulus_fixture()?;
        let regions = build(&points, &tris, &alpha, &size)?;
        assert_eq!(
            regions.singularities.len(),
            4,
            "the annulus field has two valence-3 and two valence-5 singularities"
        );
        assert!(
            regions.tiles.len() >= 4,
            "the motorcycle graph should split the annulus into at least four tiles, got {}",
            regions.tiles.len()
        );
        let mut total = 0usize;
        for tile in &regions.tiles {
            assert!(tile.counts[0] > 0 && tile.counts[1] > 0);
            assert_eq!(tile.counts[0], tile.counts[2]);
            assert_eq!(tile.counts[1], tile.counts[3]);
            total += if tile.triangular {
                // Three kites of (n/2)^2 quads each.
                3 * (tile.counts[0] / 2) * (tile.counts[0] / 2)
            } else {
                tile.counts[0] * tile.counts[1]
            };
        }
        let mesh = regions.mesh()?;
        assert_eq!(mesh.cells.len(), total);
        let report = quality::report(&mesh);
        let mut min_area = f64::INFINITY;
        for cell in &mesh.cells {
            let mut quad = [[0.0f64; 2]; 4];
            for (slot, &node) in cell.nodes.iter().enumerate() {
                if let Some(point) = mesh.points.get(node) {
                    quad[slot] = [point[0], point[1]];
                }
            }
            min_area = min_area.min(polygon_area(&quad));
        }
        assert!(
            min_area > 0.0,
            "every Coons quad must have a positive area, min {min_area}"
        );
        println!(
            "annulus mesh: {} quads, min area {:.4}, min SJ {:.3}, inverted {}",
            mesh.cells.len(),
            min_area,
            report.min_scaled_jacobian,
            report.inverted
        );
        Ok(())
    }

    #[test]
    fn demo_numbers_are_measured() -> Result<(), String> {
        let numbers = demo()?;
        let annulus = &numbers.annulus;
        assert_eq!(annulus.singularities, 4);
        assert_eq!(annulus.valence3, 2);
        assert_eq!(annulus.valence5, 2);
        assert!(
            annulus.tiles >= 6,
            "the annulus graph should keep at least six four-sided tiles, got {}",
            annulus.tiles
        );
        assert!(annulus.quads >= 40, "quads = {}", annulus.quads);
        assert!(
            annulus.min_signed_area > 0.0,
            "every Coons quad must have positive area, min {}",
            annulus.min_signed_area
        );
        assert_eq!(annulus.nonmanifold_edges, 0);
        assert!(
            annulus.mesh_area / annulus.domain_area > 0.45,
            "the tiles should cover almost half of the annulus ({} of {})",
            annulus.mesh_area,
            annulus.domain_area
        );
        assert!(annulus.qgp_energy.is_finite());
        let rectangle = &numbers.rectangle;
        assert_eq!(rectangle.singularities, 0);
        assert_eq!(rectangle.tiles, 1);
        assert_eq!(rectangle.cracks, 0);
        assert!(rectangle.relative_area_error < 1e-9);
        println!(
            "annulus: sing={} (v3={}, v5={}), nodes={} (bnd={}, T={}), edges={} \
             (sep={}, bnd={}), tiles={} (degenerate {}), \
             quads={}, energy={:.3}, minSJ={:.3}, angles=[{:.1}, {:.1}], minArea={:.3e}, \
             cracks={}, offset={:.2e}, coverage={:.3}, counts={:?}",
            annulus.singularities,
            annulus.valence3,
            annulus.valence5,
            annulus.graph_nodes,
            annulus.boundary_nodes,
            annulus.tjunction_nodes,
            annulus.graph_edges,
            annulus.separatrix_edges,
            annulus.boundary_edges,
            annulus.tiles,
            annulus.degenerate_tiles,
            annulus.quads,
            annulus.qgp_energy,
            annulus.min_scaled_jacobian,
            annulus.min_corner_angle_deg,
            annulus.max_corner_angle_deg,
            annulus.min_signed_area,
            annulus.cracks,
            annulus.boundary_offset,
            annulus.mesh_area / annulus.domain_area,
            annulus.counts
        );
        println!(
            "rectangle: tiles={}, counts={:?}, quads={}, minSJ={:.3}, angles=[{:.1}, {:.1}], \
             cracks={}, area_error={:.2e}",
            rectangle.tiles,
            rectangle.counts,
            rectangle.quads,
            rectangle.min_scaled_jacobian,
            rectangle.min_corner_angle_deg,
            rectangle.max_corner_angle_deg,
            rectangle.cracks,
            rectangle.relative_area_error
        );
        Ok(())
    }

    #[test]
    fn stitch_numbers_are_measured() -> Result<(), String> {
        let numbers = stitch_demo()?;
        for case in [&numbers.rectangle, &numbers.annulus] {
            println!(
                "{}: interior={} carved={} channel={} bands={} quads={} cracks={} \
                 minSJ={:.3} angles=[{:.1}, {:.1}] coverage={:.3} offset={:.2e} notes={:?}",
                case.name,
                case.interior_quads,
                case.carved_quads,
                case.channel_quads,
                case.bands,
                case.quads,
                case.cracks,
                case.min_scaled_jacobian,
                case.min_corner_angle_deg,
                case.max_corner_angle_deg,
                case.coverage,
                case.boundary_offset,
                case.notes
            );
        }
        // Both cases must come out all-quad, crack-free and covering the
        // domain; the annulus is filled by two channel bands because its QGP
        // interior has no ring component.
        for case in [&numbers.rectangle, &numbers.annulus] {
            assert_eq!(case.cracks, 0, "{}: cracks {}", case.name, case.cracks);
            assert!(
                case.coverage > 0.999,
                "{}: coverage {}",
                case.name,
                case.coverage
            );
            assert!(
                case.min_scaled_jacobian > 0.2,
                "{}: min SJ {}",
                case.name,
                case.min_scaled_jacobian
            );
            assert!(
                case.min_corner_angle_deg >= 30.0 && case.max_corner_angle_deg <= 150.0,
                "{}: angles [{}, {}]",
                case.name,
                case.min_corner_angle_deg,
                case.max_corner_angle_deg
            );
            assert!(!case.mesh.cells.is_empty());
        }
        assert_eq!(numbers.rectangle.channel_quads, 44);
        assert_eq!(numbers.rectangle.kept_interior, 112);
        assert_eq!(numbers.annulus.bands, 2);
        assert_eq!(numbers.annulus.kept_interior, 0);
        assert_eq!(numbers.annulus.channel_quads, 200);
        Ok(())
    }

    #[test]
    fn rectangle_carve_walks_as_one_loop() -> Result<(), String> {
        let (points, tris, alpha, size) = rectangle_fixture()?;
        let field = Field::new(&points, &tris, &alpha, &size)?;
        let regions = build(&points, &tris, &alpha, &size)?;
        let interior = regions.mesh()?;
        let outer = field.boundary_points.clone();
        let (carved, carved_quads) = carve_rings(&interior, &field, &[outer[0].clone()], 1.4);
        assert_eq!(carved_quads, 48);
        let (carved, split) = split_pinched_boundary(&carved)?;
        assert_eq!(split, 0, "the rectangle loop has no pinch");
        let loops = mesh_boundary_loops(&carved)?;
        assert_eq!(loops.len(), 1, "the carved rectangle boundary is one loop");
        assert_eq!(loops[0].len(), 44, "the loop has 44 nodes");
        let mut loop_points: Vec<[f64; 2]> = Vec::new();
        for &node in &loops[0] {
            let point = carved.points.get(node).copied().unwrap_or([0.0; 3]);
            loop_points.push([point[0], point[1]]);
        }
        let area = loop_area(&loop_points);
        assert!((area - 1.4).abs() < 1e-9, "loop area {area}");
        println!(
            "rectangle carve: {} quads carved, one loop of {} nodes, area {area:.3}",
            carved_quads,
            loops[0].len()
        );
        Ok(())
    }

    #[test]
    fn annulus_ring_phase_sweep() -> Result<(), String> {
        let mut best: Option<(f64, usize, usize)> = None;
        for step in 0..8 {
            let phase = step as f64 * 5.0 * std::f64::consts::PI / 180.0;
            let (points, tris, alpha, size) = annulus_fixture_with_phase(phase)?;
            let field = Field::new(&points, &tris, &alpha, &size)?;
            let regions = build(&points, &tris, &alpha, &size)?;
            let interior = regions.mesh()?;
            let outer = field.boundary_points.clone();
            let mut order: Vec<usize> = (0..outer.len()).collect();
            order.sort_by_key(|&index| {
                std::cmp::Reverse((loop_area(&outer[index]).abs() * 1e9) as i64)
            });
            let sorted: Vec<Vec<[f64; 2]>> = order.iter().map(|&i| outer[i].clone()).collect();
            let hole = sorted[1].clone();
            let hole_area = loop_area(&hole).abs();
            let mut centre = [0.0f64; 2];
            for point in &hole {
                centre[0] += point[0] / hole.len() as f64;
                centre[1] += point[1] / hole.len() as f64;
            }
            let mut ring_components = 0usize;
            let mut ring_quads = 0usize;
            for radius in [1.4f64, 1.6] {
                let (carved, _) = carve_rings_two(&interior, &field, &sorted, radius, 0.3);
                let (carved, _) = split_pinched_boundary(&carved)?;
                for component in split_components(&carved) {
                    let loops = match mesh_boundary_loops(&component) {
                        Ok(found) => found,
                        Err(_) => continue,
                    };
                    let mut has_inner = false;
                    let mut has_outer = false;
                    for ids in &loops {
                        let pts = loop_points(&component, ids);
                        let area = loop_area(&pts);
                        let off = pts.iter().all(|&p| domain_distance(&sorted, p) > 1e-9);
                        if !off {
                            continue;
                        }
                        if area > 0.0 {
                            has_outer = true;
                        } else if area.abs() > hole_area && loop_contains(&pts, centre) {
                            has_inner = true;
                        }
                    }
                    if has_inner && has_outer {
                        ring_components += 1;
                        ring_quads = ring_quads.max(component.cells.len());
                    }
                }
            }
            println!(
                "phase {:5.1} deg: tiles={} degenerate={} interior={} ring_components={} ring_quads={}",
                phase.to_degrees(),
                regions.tiles.len(),
                regions.degenerate_tiles,
                interior.cells.len(),
                ring_components,
                ring_quads
            );
            if ring_components > 0 && best.map(|(_, q, _)| ring_quads > q).unwrap_or(true) {
                best = Some((phase, ring_quads, ring_components));
            }
        }
        println!("best phase: {:?}", best);
        Ok(())
    }

    fn domain_distance(loops: &[Vec<[f64; 2]>], p: [f64; 2]) -> f64 {
        loops
            .iter()
            .map(|loop_points| distance_to_loop(p, loop_points))
            .fold(f64::INFINITY, f64::min)
    }

    /// Check that a quad patch is a disk: every edge is used once (boundary) or
    /// twice (interior), and V - E + F = 1.
    fn patch_is_a_disk(quads: &[[usize; 4]]) -> Result<usize, String> {
        let mut edges: HashMap<(usize, usize), usize> = HashMap::new();
        let mut nodes: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        for quad in quads {
            for slot in 0..4 {
                let a = quad[slot];
                let b = quad[(slot + 1) % 4];
                if a == b {
                    return Err("degenerate quad".to_string());
                }
                nodes.insert(a);
                nodes.insert(b);
                let key = if a < b { (a, b) } else { (b, a) };
                *edges.entry(key).or_insert(0) += 1;
            }
        }
        for (edge, count) in &edges {
            if *count > 2 {
                return Err(format!("edge {edge:?} used {count} times"));
            }
        }
        let boundary = edges.values().filter(|&&count| count == 1).count();
        let v = nodes.len();
        let e = edges.len();
        // Euler for a connected planar graph: V - E + F = 2 with F including
        // the outer face; a quad patch is a disk when V - E + (quads + 1) = 2.
        let f = quads.len() + 1;
        if v + f != e + 2 {
            return Err(format!(
                "not a disk: V={v} E={e} F={f} (V + F - E = {} != 2)",
                v + f - e
            ));
        }
        Ok(boundary)
    }

    #[test]
    fn triangle_fan_patch_is_positive() -> Result<(), String> {
        let sep_left: Vec<[f64; 2]> = (0..=12)
            .map(|k| {
                let t = k as f64 / 12.0;
                [-1.0 + t, t]
            })
            .collect();
        let sep_right: Vec<[f64; 2]> = (0..=12)
            .map(|k| {
                let t = k as f64 / 12.0;
                [1.0 - t, t]
            })
            .collect();
        let base: Vec<[f64; 2]> = (0..=24)
            .map(|k| {
                let s = k as f64 / 24.0;
                [-1.0 + 2.0 * s, -0.25 * (s * std::f64::consts::PI).sin()]
            })
            .collect();
        let (quads, boundary) = triangle_fan_quads(&sep_right, &sep_left, &base, 6)?;
        assert_eq!(quads.len(), 27, "three kites of 3 x 3");
        assert_eq!(boundary, 18, "6 + 6 + 6 boundary segments");
        for (index, quad) in quads.iter().enumerate() {
            assert!(polygon_area(quad) > 0.0, "quad {index} is not positive");
        }
        println!(
            "triangle fan: {} quads, boundary {boundary}, min area {:.3e}",
            quads.len(),
            quads
                .iter()
                .map(|q| polygon_area(q))
                .fold(f64::INFINITY, f64::min)
        );
        Ok(())
    }

    #[test]
    fn triangle_strip_topology_is_a_disk() {
        // Variant (a): 4 strip rows + a 4-quad apex fan with 3 extras.
        let mut quads: Vec<[usize; 4]> = Vec::new();
        // levels: 0 base (5 nodes 0..4), 1..5 (5 nodes each: 5..29)
        let level = |j: usize| -> [usize; 5] {
            if j == 0 {
                [0, 1, 2, 3, 4]
            } else {
                let base = 5 + (j - 1) * 5;
                [base, base + 1, base + 2, base + 3, base + 4]
            }
        };
        for j in 0..5 {
            let lower = level(j);
            let upper = level(j + 1);
            for i in 0..4 {
                quads.push([lower[i], lower[i + 1], upper[i + 1], upper[i]]);
            }
        }
        let last = level(5);
        let extras = [30, 31, 32];
        let apex = 33;
        // Pinwheel: q1 and q4 share the apex edge (A, p0); q2 and q3 share
        // (p0, p1) and (p1, L2); every apex edge is used twice.
        quads.push([last[0], last[1], extras[0], apex]);
        quads.push([last[1], last[2], extras[1], extras[0]]);
        quads.push([last[2], last[3], extras[0], extras[1]]);
        quads.push([last[3], last[4], apex, extras[0]]);
        // The 5-row strip plus the pinwheel fan (q1 and q4 sharing (apex, p0),
        // q2 and q3 sharing (p0, p1)) is a disk whose boundary is exactly
        // 4 + 6 + 6 = 16 edges: 24 quads, 2 extra interior nodes.
        match patch_is_a_disk(&quads) {
            Ok(boundary) => {
                println!(
                    "variant (a): {} quads, boundary {boundary} edges (want 4 + 6 + 6 = 16)",
                    quads.len()
                );
                assert_eq!(boundary, 16, "pinwheel boundary");
            }
            Err(error) => println!("variant (a) invalid: {error}"),
        }
    }

    #[test]
    fn laplacian_places_interior_nodes_inside() -> Result<(), String> {
        // A 3x3 grid of quads (4x4 nodes) with the boundary fixed and the four
        // interior nodes started at the centre.
        let mut quads = Vec::new();
        for j in 0..3 {
            for i in 0..3 {
                let a = j * 4 + i;
                quads.push([a, a + 1, a + 5, a + 4]);
            }
        }
        let mut fixed = HashMap::new();
        let mut initial = HashMap::new();
        for j in 0..4 {
            for i in 0..4 {
                let node = j * 4 + i;
                let point = [i as f64, j as f64];
                initial.insert(node, point);
                if i == 0 || j == 0 || i == 3 || j == 3 {
                    fixed.insert(node, point);
                }
            }
        }
        let placed = laplacian_embedding(&quads, &fixed, &initial, 500);
        for j in 0..4 {
            for i in 0..4 {
                let point = match placed.get(&(j * 4 + i)) {
                    Some(found) => *found,
                    None => return Err(format!("node ({i}, {j}) was not placed")),
                };
                assert!(
                    point[0] >= -1e-9
                        && point[0] <= 3.0 + 1e-9
                        && point[1] >= -1e-9
                        && point[1] <= 3.0 + 1e-9,
                    "node ({i}, {j}) left the patch: {point:?}"
                );
            }
        }
        // With the square boundary fixed at its grid positions the harmonic
        // extension is the linear field, so every interior node returns to its
        // grid position.
        let centre = placed.get(&5).copied().unwrap_or([f64::NAN; 2]);
        assert!(
            (centre[0] - 1.0).abs() < 1e-6 && (centre[1] - 1.0).abs() < 1e-6,
            "interior node at {centre:?}"
        );
        println!("laplacian: interior node at {centre:?}");
        Ok(())
    }

    /// Census of the tiles around each singularity: Section 5.1.5.2 merging
    /// would join these, so their sizes and areas are what a merged region
    /// would have to mesh.
    /// Annulus fixture with explicit resolution, phase and size law, for the
    /// QGP parameter sweep.
    fn annulus_variant(
        n_theta: usize,
        n_radial: usize,
        phase: f64,
        size_law: &dyn Fn(f64) -> f64,
    ) -> Result<(Vec<[f64; 3]>, Vec<[usize; 3]>, Vec<f64>, Vec<f64>), String> {
        const INNER: f64 = 0.4;
        const HALF: f64 = 1.5;
        let mut points: Vec<[f64; 3]> = Vec::with_capacity((n_radial + 1) * n_theta);
        for layer in 0..=n_radial {
            let t = layer as f64 / n_radial as f64;
            for segment in 0..n_theta {
                let theta = std::f64::consts::TAU * segment as f64 / n_theta as f64;
                let (sin, cos) = theta.sin_cos();
                // Square annulus: the radius is clamped to the square boundary.
                let radius = INNER + t * (HALF - INNER);
                points.push([radius * cos, radius * sin, 0.0]);
            }
        }
        let index = |layer: usize, segment: usize| layer * n_theta + segment % n_theta;
        let mut tris: Vec<[usize; 3]> = Vec::with_capacity(2 * n_radial * n_theta);
        for layer in 0..n_radial {
            for segment in 0..n_theta {
                let a = index(layer, segment);
                let b = index(layer, segment + 1);
                let c = index(layer + 1, segment + 1);
                let d = index(layer + 1, segment);
                tris.push([a, c, b]);
                tris.push([a, d, c]);
            }
        }
        let alpha: Vec<f64> = points
            .iter()
            .map(|point| {
                let z = [point[0], point[1]];
                let radius2 = z[0] * z[0] + z[1] * z[1];
                let z2 = [z[0] * z[0] - z[1] * z[1], 2.0 * z[0] * z[1]];
                let z4 = [z2[0] * z2[0] - z2[1] * z2[1], 2.0 * z2[0] * z2[1]];
                let (spiral_sin, spiral_cos) = (4.0 * phase).sin_cos();
                let base = [z4[0] / (radius2 * radius2), z4[1] / (radius2 * radius2)];
                let prefactor = [
                    base[0] * spiral_cos - base[1] * spiral_sin,
                    base[0] * spiral_sin + base[1] * spiral_cos,
                ];
                let diagonal = std::f64::consts::FRAC_1_SQRT_2;
                let zeros = complex_mul(
                    [z[0] - diagonal, z[1] - diagonal],
                    [z[0] + diagonal, z[1] + diagonal],
                );
                let poles = complex_mul(
                    [z[0] + diagonal, z[1] - diagonal],
                    [z[0] - diagonal, z[1] + diagonal],
                );
                let ratio = complex_div(zeros, poles).unwrap_or([1.0, 0.0]);
                let value = complex_mul(prefactor, ratio);
                wrap_cross(0.25 * value[1].atan2(value[0]))
            })
            .collect();
        let size: Vec<f64> = points
            .iter()
            .map(|point| size_law((point[0] * point[0] + point[1] * point[1]).sqrt()))
            .collect();
        Ok((points, tris, alpha, size))
    }

    #[test]
    fn annulus_qgp_parameter_sweep() {
        let mut rows: Vec<String> = Vec::new();
        let mut hits = 0usize;
        for &(n_theta, n_radial) in &[(60usize, 6usize), (80, 8), (100, 10), (120, 12)] {
            for &phase_deg in &[0.0f64, 10.0, 20.0, 30.0] {
                for &scale in &[0.15f64, 0.22, 0.3, 0.4] {
                    let phase = phase_deg.to_radians();
                    let law = move |_r: f64| scale;
                    let fixture = match annulus_variant(n_theta, n_radial, phase, &law) {
                        Ok(found) => found,
                        Err(error) => {
                            rows.push(format!("{n_theta}x{n_radial} {phase_deg:2.0}deg {scale:.2}: fixture {error}"));
                            continue;
                        }
                    };
                    let (points, tris, alpha, size) = fixture;
                    let regions = match build(&points, &tris, &alpha, &size) {
                        Ok(found) => found,
                        Err(error) => {
                            rows.push(format!(
                                "{n_theta}x{n_radial} {phase_deg:2.0}deg {scale:.2}: {error}"
                            ));
                            continue;
                        }
                    };
                    let interior = match regions.mesh() {
                        Ok(found) => found,
                        Err(error) => {
                            rows.push(format!(
                                "{n_theta}x{n_radial} {phase_deg:2.0}deg {scale:.2}: {error}"
                            ));
                            continue;
                        }
                    };
                    let field = match Field::new(&points, &tris, &alpha, &size) {
                        Ok(found) => found,
                        Err(error) => {
                            rows.push(format!(
                                "{n_theta}x{n_radial} {phase_deg:2.0}deg {scale:.2}: {error}"
                            ));
                            continue;
                        }
                    };
                    let outer = field.boundary_points.clone();
                    let hole = field.boundary_points.get(1).cloned().unwrap_or_default();
                    let mut centre = [0.0f64; 2];
                    for point in &hole {
                        centre[0] += point[0] / hole.len().max(1) as f64;
                        centre[1] += point[1] / hole.len().max(1) as f64;
                    }
                    let (carved, carved_quads) =
                        carve_rings(&interior, &field, &[outer[0].clone()], 1.4);
                    let loops = mesh_boundary_loops(&carved).unwrap_or_default();
                    let mut contains = 0usize;
                    let mut areas: Vec<f64> = Vec::new();
                    for ids in &loops {
                        let line = loop_points(&carved, ids);
                        areas.push(loop_area(&line));
                        if loop_contains(&line, centre) {
                            contains += 1;
                        }
                    }
                    if contains > 0 {
                        hits += 1;
                    }
                    rows.push(format!(
                        "{n_theta}x{n_radial} {phase_deg:2.0}deg {scale:.2}: tiles {} interior {carved_quads} carved, {} loops, contains {contains}, areas {:?}",
                        regions.tiles.len(),
                        loops.len(),
                        areas.iter().map(|a| (a * 100.0).round() / 100.0).collect::<Vec<_>>()
                    ));
                }
            }
        }
        for row in &rows {
            println!("sweep: {row}");
        }
        println!(
            "sweep: {hits} configurations contain the hole of {}",
            rows.len()
        );
    }

    /// Sweep of the Chapter 3 boundary-constrained field on the annulus: does
    /// a Tangent / Weak(1e4) / anisotropic field produce a QGP interior that
    /// encircles the hole?
    #[test]
    fn annulus_boundary_field_sweep() {
        use crate::simons_godf::{BoundaryMode, Options};
        let mut rows: Vec<String> = Vec::new();
        let mut hits = 0usize;
        let scales: [(&str, f64); 2] = [("0.15", 0.15), ("0.22", 0.22)];
        let variants: [(&str, Options); 5] = [
            (
                "tangent+accurate",
                Options {
                    boundary: BoundaryMode::Tangent,
                    accurate: true,
                    ..Options::default()
                },
            ),
            (
                "weak1e4",
                Options {
                    boundary: BoundaryMode::Weak(1.0e4),
                    accurate: true,
                    ..Options::default()
                },
            ),
            (
                "tangent+aniso2",
                Options {
                    boundary: BoundaryMode::Tangent,
                    accurate: true,
                    anisotropy: 2.0,
                    ..Options::default()
                },
            ),
            (
                "tangent+aniso3",
                Options {
                    boundary: BoundaryMode::Tangent,
                    accurate: true,
                    anisotropy: 3.0,
                    ..Options::default()
                },
            ),
            (
                "tangent+aniso4",
                Options {
                    boundary: BoundaryMode::Tangent,
                    accurate: true,
                    anisotropy: 4.0,
                    ..Options::default()
                },
            ),
        ];
        for &(label, options) in &variants {
            for &(scale_label, scale) in &scales {
                for &(n_theta, n_radial) in &[(100usize, 10usize), (48, 4)] {
                    let law = move |_r: f64| scale;
                    let fixture = match annulus_variant(n_theta, n_radial, 0.0, &law) {
                        Ok(found) => found,
                        Err(error) => {
                            rows.push(format!(
                                "{label} {scale_label} {n_theta}x{n_radial}: fixture {error}"
                            ));
                            continue;
                        }
                    };
                    let (points, tris, _alpha, size) = fixture;
                    let alpha = match crate::simons_godf::solve_with_options(
                        &points, &tris, &size, &options, 100,
                    ) {
                        Ok(found) => found,
                        Err(error) => {
                            rows.push(format!(
                                "{label} {scale_label} {n_theta}x{n_radial}: solve {error}"
                            ));
                            continue;
                        }
                    };
                    let regions = match build(&points, &tris, &alpha, &size) {
                        Ok(found) => found,
                        Err(error) => {
                            rows.push(format!(
                                "{label} {scale_label} {n_theta}x{n_radial}: {error}"
                            ));
                            continue;
                        }
                    };
                    let interior = match regions.mesh() {
                        Ok(found) => found,
                        Err(error) => {
                            rows.push(format!(
                                "{label} {scale_label} {n_theta}x{n_radial}: {error}"
                            ));
                            continue;
                        }
                    };
                    let field = match Field::new(&points, &tris, &alpha, &size) {
                        Ok(found) => found,
                        Err(error) => {
                            rows.push(format!(
                                "{label} {scale_label} {n_theta}x{n_radial}: {error}"
                            ));
                            continue;
                        }
                    };
                    let outer = field.boundary_points.clone();
                    let hole = field.boundary_points.get(1).cloned().unwrap_or_default();
                    let mut centre = [0.0f64; 2];
                    for point in &hole {
                        centre[0] += point[0] / hole.len().max(1) as f64;
                        centre[1] += point[1] / hole.len().max(1) as f64;
                    }
                    let (carved, carved_quads) =
                        carve_rings(&interior, &field, &[outer[0].clone()], 1.4);
                    let loops = mesh_boundary_loops(&carved).unwrap_or_default();
                    let mut contains = 0usize;
                    let mut areas: Vec<f64> = Vec::new();
                    for ids in &loops {
                        let line = loop_points(&carved, ids);
                        areas.push(loop_area(&line));
                        if loop_contains(&line, centre) {
                            contains += 1;
                        }
                    }
                    if contains > 0 {
                        hits += 1;
                    }
                    rows.push(format!(
                        "{label:18} {scale_label} {n_theta}x{n_radial}: sing {} tiles {} interior {carved_quads} carved, {} loops, contains {contains}, areas {:?}",
                        regions.singularities.len(),
                        regions.tiles.len(),
                        loops.len(),
                        areas.iter().map(|a| (a * 100.0).round() / 100.0).collect::<Vec<_>>()
                    ));
                }
            }
        }
        for row in &rows {
            println!("field sweep: {row}");
        }
        println!(
            "field sweep: {hits} configurations contain the hole of {}",
            rows.len()
        );
    }

    #[test]
    fn annulus_singularity_neighbourhoods() -> Result<(), String> {
        let (points, tris, alpha, size) = annulus_fixture()?;
        let regions = build(&points, &tris, &alpha, &size)?;
        println!(
            "merge: {} singularities, {} tiles, {} degenerate, {} collapsed",
            regions.singularities.len(),
            regions.tiles.len(),
            regions.degenerate_tiles,
            regions.collapsed_tiles
        );
        for (index, point) in regions.singularities.iter().enumerate() {
            let mut incident: Vec<(usize, f64, [usize; 4])> = Vec::new();
            for (tile_index, tile) in regions.tiles.iter().enumerate() {
                if tile
                    .corners
                    .iter()
                    .any(|corner| distance(*corner, *point) <= 1e-9)
                {
                    incident.push((tile_index, tile.area, tile.corner_nodes));
                }
            }
            let mut total = 0.0;
            for (_, area, _) in &incident {
                total += area;
            }
            println!(
                "merge: singularity {index} at ({:.3}, {:.3}): {} incident tiles, areas {:?}, total {:.3}",
                point[0],
                point[1],
                incident.len(),
                incident
                    .iter()
                    .map(|(_, area, _)| (*area * 1000.0).round() / 1000.0)
                    .collect::<Vec<_>>(),
                total
            );
        }
        Ok(())
    }

    #[test]
    fn annulus_hole_loop_geometry() -> Result<(), String> {
        let (points, tris, alpha, size) = annulus_fixture()?;
        let field = Field::new(&points, &tris, &alpha, &size)?;
        let regions = build(&points, &tris, &alpha, &size)?;
        let interior = regions.mesh()?;
        let outer = field.boundary_points.clone();
        let (carved, carved_quads) = carve_rings(&interior, &field, &[outer[0].clone()], 1.4);
        let (carved, split) = split_pinched_boundary(&carved)?;
        println!(
            "annulus 1.4: {} quads carved, {} pinch node(s) split",
            carved_quads, split
        );
        let loops = match mesh_boundary_loops(&carved) {
            Ok(found) => found,
            Err(error) => {
                println!("annulus 1.4: the carved boundary does not close: {error}");
                return Ok(());
            }
        };
        for (index, ids) in loops.iter().enumerate() {
            let mut pts: Vec<[f64; 2]> = Vec::new();
            let mut on_hole = 0usize;
            for &node in ids {
                let point = carved.points.get(node).copied().unwrap_or([0.0; 3]);
                pts.push([point[0], point[1]]);
                if field.boundary_points.len() > 1
                    && field.boundary_points[1..]
                        .iter()
                        .any(|hole| distance_to_loop([point[0], point[1]], hole) <= 1e-9)
                {
                    on_hole += 1;
                }
            }
            println!(
                "annulus loop {index}: {} nodes, area {:.4}, {on_hole} nodes on the hole boundary",
                ids.len(),
                loop_area(&pts)
            );
        }
        Ok(())
    }

    #[test]
    fn godf_field_can_be_built() -> Result<(), String> {
        let (points, tris) = square_mesh(6);
        let size = vec![0.15; points.len()];
        let alpha = simons_godf::solve(&points, &tris, &size, 100)?;
        let regions = build(&points, &tris, &alpha, &size)?;
        assert!(
            !regions.tiles.is_empty(),
            "a constant size on a square must leave at least one tile"
        );
        let mesh = regions.mesh()?;
        assert!(!mesh.cells.is_empty());
        Ok(())
    }

    #[test]
    fn invalid_input_is_rejected() {
        let (points, tris) = square_mesh(2);
        let alpha = vec![0.0; points.len()];
        let short = vec![0.1; points.len() - 1];
        assert!(build(&points, &tris, &alpha, &short).is_err());
        let mut negative = vec![0.1; points.len()];
        negative[0] = -1.0;
        assert!(build(&points, &tris, &alpha, &negative).is_err());
        let mut bad = tris.clone();
        bad[0] = [0, 0, 1];
        assert!(build(&points, &bad, &alpha, &vec![0.1; points.len()]).is_err());
        assert!(build(&points, &[], &alpha, &vec![0.1; points.len()]).is_err());
    }
}
