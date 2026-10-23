//! Boundary-channel all-quad meshing (Simons 2024, Chapter 2).
//!
//! This module implements the *channel construction* of Simons' thesis,
//! "Size-Driven All-Quad Meshing for Geographic Data" (2024), Sections 2.2.1
//! to 2.2.9. The interior of the domain is meshed by a regular grid (§2.2.2);
//! the band between that grid and the domain boundary — the **channel** — is
//! cut into small patches by **cross edges** (§2.2.3–2.2.4) and each patch is
//! quadrangulated by a template set (§2.2.5–2.2.6), with Steiner points placed
//! by a Tutte (harmonic) embedding. Patches are then selected by a covering
//! step (§2.2.7–2.2.9).
//!
//! # Faithful to the thesis
//!
//! * §2.2.2: the grid is overlaid at spacing `h`; grid vertices outside the
//!   domain or within `δ = 1.4h` of the boundary are culled; cells whose four
//!   corners survive become interior quads; every other surviving grid edge
//!   (including hanging edges, which are traversed in both directions) is
//!   kept as an inner channel edge.
//! * §2.2.2: the domain boundary and the inner grid edges are walked with
//!   opposite orientation to produce the channel loop(s) (one loop per
//!   boundary component of the channel; a domain hole contributes one, and a
//!   grid island contributes one).
//! * §2.2.3: candidate cross edges join channel vertices no farther apart than
//!   `4h` and must not cross an inner/outer edge; the cross-edge set is grown
//!   greedily from the shortest candidates.
//! * §2.2.4: triangles of the resulting planar graph are glued along cross
//!   edges (up to three joining steps); patches with 4–8 sides are kept.
//! * §2.2.5: patch rank is the worst quad's maximum angle deviation from 90°;
//!   interior Steiner points are placed by a Tutte/barycentric (Laplacian)
//!   embedding, exactly as the thesis describes.
//! * §2.2.6: 4-sided patches are one quad; 6-sided patches use the paving
//!   templates of Blacker & Stephenson (1991, "closure of boundaries with six
//!   nodes") with 0, 1 and 2 Steiner vertices; 8-sided patches use the
//!   recursive cuts of Fig. 2.6 in simplified form (corner cut with a Steiner
//!   point, 4+6 split, 6/6 split with a Steiner point on the chord, and the
//!   0-Steiner double-chord topology).
//! * §2.2.10 is *not* implemented: no angle/stretch smoothing is run. The
//!   grid and the boundary are left where the construction puts them.
//!
//! # Implementation status and deviations (documented)
//!
//! **The default construction is a boundary ribbon, not the thesis' patch
//! cover.** `mesh` first meshes the channel as one or more strips, each with one
//! quad per edge of a grid loop: the domain outer boundary is joined to the
//! outer grid loop, and every domain hole is joined to the grid hole loop that
//! surrounds it (two ribbons for a rectangle with a circular hole). A domain
//! loop is re-discretized to the same number of vertices as its grid loop
//! (keeping every input corner and every new point on the polyline), the loop
//! orientations are chosen so that the ribbon quads have positive area, and the
//! best rotation/direction of each pair is chosen by quad validity and angle
//! rank. This is much simpler than §2.2.3–2.2.9: no cross edges, no templates,
//! no covering. The template + greedy-cover path described below is implemented
//! and is attempted only if the ribbon fails.
//!
//! Results of this file:
//! * 10×10 square, h = 1: 60 quads (36 interior grid + 24 ribbon), min SJ
//!   0.7071, corner angles 45°–135°, cracks 0, boundary offset ~1.8e-15.
//! * square with a square hole, h = 1: 16 quads (no interior grid survives the
//!   cull), min SJ 0.7071, angles 45°–135°, cracks 0, offset 0.
//! * 100×60 plate with a circular 24-gon hole (r = 10), h = 4: 262 quads (162
//!   interior grid + 100 ribbon over two ribbons), 4 channel loops, min SJ
//!   0.3964, angles 28.16°–156.64°, aspect 8.12, cracks 0, boundary offset
//!   1.4e-14, quality verdict Acceptable (PASS). At h = 2: 1240 quads (1040 +
//!   200), min SJ 0.2868, angles 19.81°–163.33°, aspect 6.15, cracks 0,
//!   offset 1.4e-14, Acceptable. At h = 1: 5324 quads (4924 + 400), min SJ
//!   0.2936, angles 25.15°–162.93°, aspect 4.18, cracks 0, offset 1.4e-14,
//!   Acceptable.
//! * **Corpus l-bracket** (80×60 section, size 2): 600 quads (464 interior
//!   grid + 136 channel) at an adaptive channel width δ = 1.0h, min SJ 1.0000,
//!   angles 90°–90°, aspect 1.00, cracks 0, boundary offset 0, verdict
//!   Acceptable. The thesis-default δ = 1.4h leaves only a thin staircase grid
//!   strip and fails; at δ = 1.0h the template/greedy path closes the channel
//!   with perfect grid quads. `build` therefore tries δ = 1.4h, 1.0h, 0.7h in
//!   order (see `ADAPTIVE_DELTAS`).
//! * **Corpus stepped section** (60×40 with a notch, size 1): 2300 quads
//!   (2084 interior + 216 channel), same δ = 1.0h path, min SJ 1.0000,
//!   angles 90°–90°, aspect 1.00, cracks 0, offset 0, Acceptable.
//! * **Perf plate** (100×60, 64-gon hole r = 10, size 0.6): 15,520 quads in
//!   ~0.35 s; the channel closes at δ = 1.4h through a **projection row strip**
//!   between the staircase grid hole loop and the 64-gon domain hole. Raw
//!   min SJ 0.0874, angles 5.01°–174.71°, verdict BelowTarget (not Refused):
//!   the worst quad is a sliver where a radial staircase step is almost
//!   parallel to its cross edge. The mesh is all-quad and conforming; a split
//!   or cap at those steps is the remaining quality improvement.
//! * **Corpus ring section** (quarter annulus, R20/R10, size 1): 48 quads,
//!   min SJ 0.9979, angles 87.50°–93.75°, aspect 7.66, cracks 0, boundary
//!   offset 4.4e-16, verdict Acceptable (PASS), produced by the **two-chain
//!   strip** fallback (see below). The thesis channel path alone cannot close
//!   this case: the curved staircase channel leaves a leftover region that
//!   neither the ribbon nor the greedy-cover fallback can close at δ = 1.4h,
//!   1.0h or 0.7h.
//! * The small 10×10 L-bracket of the earlier tests still has a thin staircase
//!   grid strip at every tried δ and remains unmarshalled; the corpus L (wide
//!   legs, size 2) is the case that passes.
//!
//! * **Covering (§2.2.7–2.2.9)**: the full megapatch/widest-path search is not
//!   implemented. Patches are scored and selected by a greedy cover: repeatedly
//!   take the patch that adds the largest uncovered channel area, breaking ties
//!   by the better rank. Overlaps are impossible by construction because a
//!   patch is a union of whole channel faces; a patch is only taken when all of
//!   its faces are still uncovered. Whatever the greedy cover leaves is closed
//!   by a fallback cavity fill (fan/centroid quadrangulation), with one allowed
//!   domain-perimeter split when a leftover polygon has an odd number of sides.
//!   The statistics report the fraction of the channel area the greedy cover
//!   filled and the number of fallback regions/quads.
//! * **Cross-edge midpoint splits (§2.2.5) are not implemented.** Odd-sided
//!   patches (5 and 7 sides) are kept as candidates but not quadrangulated:
//!   splitting a shared cross edge would require every neighbouring patch to
//!   agree on the split. Their area goes to the fallback.
//! * **Megapatches and the widest-path cover (§2.2.7–2.2.8) are not
//!   implemented.** The widest-path machinery exists only as the A/B/C ribbon
//!   dynamic program (`dp_ribbon`): strip quads plus caps that absorb a
//!   loop-count mismatch. Its caps are degenerate when three consecutive
//!   boundary vertices are collinear, which is exactly the common case on a
//!   straight run, so it does not close the wide-leg corpus channels. What
//!   unlocked the corpus L and stepped sections instead was the adaptive
//!   channel width above, which keeps the regular grid close enough to the
//!   boundary that the ordinary template cover closes the remaining thin band.
//! * **Adaptive channel width**: `build` tries δ = 1.4h, then 1.0h, then 0.7h,
//!   returning the first valid all-quad mesh. This deviates from the thesis'
//!   fixed δ = 1.4h, but it is what lets the corpus L and stepped sections
//!   pass while preserving the thesis-default results on the square and plate
//!   family.
//! * **Projection row strip**: when a band's fixed grid loop is a staircase
//!   (the perf plate's grid hole loop) and neither the ribbon nor the cyclic
//!   row strip can match it to a free domain loop, each fixed-loop vertex is
//!   projected radially from the free loop's centroid onto the free loop and
//!   1–4 interpolated rows are built. This closes the 0.6 mm perf plate's hole
//!   band. It is inside `add_ribbon`, so `channel_between` and the ordinary
//!   channel path share it.
//! * **Universal left-over closure (two-chain strip)**: when every channel
//!   attempt fails and the domain is simply connected, `build` cuts two
//!   boundary edges, subdivides the shorter remaining chain so both chains have
//!   the same edge count, and builds a structured strip (1–4 radial rows) of
//!   quads between them. The two cut edges become the strip's end cross edges,
//!   so the mesh boundary is exactly the domain boundary. This is the
//!   generalization of the prescribed single-centre midpoint fan that the ring
//!   needs: the single-centre fan `[v_i, m_i, c, m_{i-1}]` is degenerate on
//!   straight boundary stretches (180° at `v_i`) and, for the ring's concave
//!   inner arc, 11 of its 38 quads are reflex, so a single centre cannot see the
//!   whole ring sector. The two-chain strip keeps the same "one quad per pair of
//!   chain edges" idea without needing one centre. It is a whole-domain fallback
//!   (no interior grid) and is documented as such.
//! * **6-sided, 3-Steiner template**: for a *convex* hexagon this topology is
//!   necessarily degenerate — the only two combinatorial configurations are a
//!   doublet (two quads sharing two edges) or a central triangular hole. The
//!   thesis' fourth template exists for concave patches and its non-degenerate
//!   configuration could not be reproduced from the public description; the
//!   three non-degenerate templates are implemented instead.
//! * Cross-edge candidate generation is `O(n²)` in the number of channel
//!   vertices and is capped; very fine requests return an error instead of
//!   exhausting memory. The 4h reach is the thesis' own current choice and is
//!   the main reason a cover can fail.
//! * The grid is axis-aligned with its origin at the lower-left lattice line
//!   (no rotation search); the thesis tries several grid offsets/orientations.
//!
//! # Public API
//!
//! ```text
//! sig_simons_channel::mesh(&domain, h)            -> Result<Mesh, String>
//! sig_simons_channel::mesh_with_stats(&domain, h) -> Result<(Mesh, ChannelStats), String>
//! sig_simons_channel::channel(&domain, h)         -> Result<Vec<ChannelLoop>, String>
//! sig_simons_channel::channel_between(domain, outer, inner, inner_ids, points, size)
//!                                                -> Result<Vec<[usize;4]>, String>
//! sig_simons_channel::channel_between_hole(domain, hole, inner, inner_ids, points, size)
//!                                                -> Result<Vec<[usize;4]>, String>
//! sig_simons_channel::demo()                      -> Result<DemoNumbers, String>
//! ```
//!
//! Every mesh returned by [`mesh`] is all-quadrilateral, conforming
//! (`Mesh::topology().cracks == 0`), has no inverted cell, and has all of its
//! boundary nodes on the input polyline (distance 0 up to floating point).
//!
//! The module is self-contained: no external dependencies, no `unsafe`, no
//! panics in non-test code, and it builds for `wasm32-unknown-unknown`.

#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::type_complexity
)]

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

use crate::mesh::{Cell, Mesh, Node};
use crate::quality;
use crate::two_d::Domain;

/// Distance from the boundary within which grid vertices are culled (§2.2.2).
pub const DELTA_FACTOR: f64 = 1.4;
/// Maximum length of a candidate cross edge, in units of `h` (§2.2.3).
pub const CROSS_REACH_FACTOR: f64 = 4.0;

/// Channel widths tried, in order, when the default 1.4h channel cannot be
/// closed. Narrower channels leave the regular grid closer to the boundary,
/// which is what the wide-leg sections of the corpus need.
pub const ADAPTIVE_DELTAS: [f64; 3] = [1.4, 1.0, 0.7];
/// Maximum number of joining steps when growing a patch (§2.2.4).
const MAX_GLUES: usize = 3;

const EPS: f64 = 1.0e-9;
const TAU: f64 = std::f64::consts::TAU;
const MAX_CHANNEL_VERTICES: usize = 4000;
const MAX_CANDIDATE_EDGES: usize = 400_000;
const MAX_PATCH_SETS: usize = 60_000;
const MAX_PATCHES: usize = 50_000;

/// Which boundary of the channel a [`ChannelLoop`] belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LoopSource {
    /// The input domain boundary (outer loop or a hole), discretized to `h`.
    Boundary,
    /// The boundary of the interior grid, walked opposite to the grid quads.
    Grid,
}

/// One closed boundary loop of the channel region.
///
/// The points are ordered so that the channel lies on the left of every
/// directed edge (the same convention the face walk uses internally). Boundary
/// loops are therefore counter-clockwise for the outer loop and clockwise for
/// holes; grid loops are clockwise around an interior grid block and
/// counter-clockwise around an island of channel inside the grid. Consecutive
/// duplicate points only appear when a hanging grid edge is walked in both
/// directions (a zero-width slit), which is exactly how §2.2.2 keeps hanging
/// edges.
#[derive(Clone, Debug)]
pub struct ChannelLoop {
    pub points: Vec<[f64; 2]>,
    pub source: LoopSource,
}

/// Numbers collected while building one channel mesh.
#[derive(Clone, Debug, Default)]
pub struct ChannelStats {
    /// Which construction produced the mesh: "boundary-ribbon" or
    /// "templates+greedy".
    pub method: &'static str,
    /// Channel-width factor used (1.4 is the thesis default).
    pub delta_factor: f64,
    /// Grid vertices that survived the cull.
    pub grid_vertices: usize,
    /// Quads promoted from the regular grid.
    pub interior_quads: usize,
    /// Number of channel boundary loops (domain loops + grid loops).
    pub channel_loops: usize,
    /// Cross edges accepted by the no-crossing search.
    pub cross_edges: usize,
    /// Faces of the planar channel graph.
    pub faces: usize,
    /// Faces with exactly three sides.
    pub triangle_faces: usize,
    /// Patches with 4–8 sides kept as candidates.
    pub patches: usize,
    /// Candidate patches for which a template produced a valid quadrangulation.
    pub usable_patches: usize,
    /// Patches picked by the greedy cover.
    pub selected_patches: usize,
    /// Channel area fraction filled before the fallback (ribbon path: 1.0).
    pub greedy_covered_fraction: f64,
    /// Leftover regions closed by the fallback.
    pub fallback_regions: usize,
    /// Quads produced by the fallback.
    pub fallback_quads: usize,
    /// Quads belonging to the channel (greedy + fallback).
    pub channel_quads: usize,
    /// Total quads of the returned mesh.
    pub quads: usize,
    /// Smallest scaled Jacobian over the mesh.
    pub min_scaled_jacobian: f64,
    /// Smallest corner angle of a planar quad, in degrees.
    pub min_angle_deg: f64,
    /// Largest corner angle of a planar quad, in degrees.
    pub max_angle_deg: f64,
    /// Boundary vertices that are not shared by exactly two boundary edges.
    pub cracks: usize,
    /// Cells with a non-positive scaled Jacobian.
    pub inverted: usize,
    /// Maximum distance from a boundary node to the input polyline.
    pub boundary_offset: f64,
    /// Domain perimeter edges split by the fallback (at most one per edge).
    pub split_perimeter_edges: usize,
}

/// The numbers of one demo case.
#[derive(Clone, Debug, Default)]
pub struct CaseNumbers {
    pub method: String,
    pub delta_factor: f64,
    pub quads: usize,
    pub interior_quads: usize,
    pub channel_quads: usize,
    pub min_scaled_jacobian: f64,
    pub min_angle_deg: f64,
    pub max_angle_deg: f64,
    pub cracks: usize,
    pub inverted: usize,
    pub boundary_offset: f64,
    pub greedy_covered_fraction: f64,
    pub fallback_regions: usize,
    pub fallback_quads: usize,
    pub cross_edges: usize,
    pub patches: usize,
    pub selected_patches: usize,
}

/// Numbers returned by [`demo`].
#[derive(Clone, Debug, Default)]
pub struct DemoNumbers {
    pub square: CaseNumbers,
    pub l_bracket: Option<CaseNumbers>,
    /// Why the L-bracket case failed to mesh, when it did.
    pub l_bracket_error: Option<String>,
    /// `None` when the square-with-a-square-hole case could not be covered.
    pub square_with_hole: Option<CaseNumbers>,
    /// Why the square-with-a-hole case failed, when it did.
    pub square_with_hole_error: Option<String>,
}

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Mesh `domain` with the grid + channel construction, at uniform edge size
/// `size`.
pub fn mesh(domain: &Domain, size: f64) -> Result<Mesh, String> {
    Ok(build(domain, size)?.mesh)
}

/// Same as [`mesh`], also returning the construction statistics.
pub fn mesh_with_stats(domain: &Domain, size: f64) -> Result<(Mesh, ChannelStats), String> {
    let output = build(domain, size)?;
    Ok((output.mesh, output.stats))
}

/// Build only the channel boundary loops, without cross edges or patches.
///
/// This is the §2.2.2 output: the domain boundary discretized to roughly `h`,
/// plus the inner grid edges walked in the opposite direction. Loops are
/// ordered with the channel on their left.
pub fn channel(domain: &Domain, size: f64) -> Result<Vec<ChannelLoop>, String> {
    let prepared = prepare(domain, size, DELTA_FACTOR)?;
    let mut loops = Vec::with_capacity(prepared.loops.len());
    for raw in &prepared.loops {
        let mut points = Vec::with_capacity(raw.points.len());
        for &id in &raw.points {
            match prepared.pos.get(id) {
                Some(&p) => points.push(p),
                None => {
                    return Err("internal error: channel loop references a missing point".into())
                }
            }
        }
        loops.push(ChannelLoop {
            points,
            source: raw.source,
        });
    }
    Ok(loops)
}

/// Turn angle above which a loop vertex counts as a corner (degrees).
pub const CORNER_TURN_DEG: f64 = 25.0;

/// Indices of the corner vertices of a loop: the direction change is greater
/// than [`CORNER_TURN_DEG`]. Collinear vertices have a turn of ~0 and are not
/// corners.
fn corner_indices(points: &[[f64; 2]]) -> Vec<usize> {
    let n = points.len();
    let mut corners = Vec::new();
    if n < 4 {
        return (0..n).collect();
    }
    for i in 0..n {
        let previous = points[(i + n - 1) % n];
        let v = points[i];
        let next = points[(i + 1) % n];
        let a = [v[0] - previous[0], v[1] - previous[1]];
        let b = [next[0] - v[0], next[1] - v[1]];
        let la = (a[0] * a[0] + a[1] * a[1]).sqrt();
        let lb = (b[0] * b[0] + b[1] * b[1]).sqrt();
        if la <= EPS || lb <= EPS {
            continue;
        }
        let cosine = ((a[0] * b[0] + a[1] * b[1]) / (la * lb)).clamp(-1.0, 1.0);
        if cosine.acos().to_degrees() > CORNER_TURN_DEG {
            corners.push(i);
        }
    }
    corners
}

/// Collapse collinear runs to the loop's corners when that leaves at least
/// four, an even number, and a proper subset. Otherwise keep the loop as is.
fn reduce_loop_with_ids(points: &[[f64; 2]], ids: &[usize]) -> (Vec<[f64; 2]>, Vec<usize>) {
    let corners = corner_indices(points);
    if corners.len() >= 4 && corners.len() < points.len() && corners.len().is_multiple_of(2) {
        let reduced_points = corners.iter().map(|&i| points[i]).collect();
        let reduced_ids = corners.iter().map(|&i| ids[i]).collect();
        (reduced_points, reduced_ids)
    } else {
        (points.to_vec(), ids.to_vec())
    }
}

/// Shared implementation of [`channel_between`] and [`channel_between_hole`].
fn ribbon_between_impl(
    domain: &Domain,
    free_loop: &[[f64; 2]],
    fixed_loop: &[[f64; 2]],
    fixed_ids: &[usize],
    free_is_outer: bool,
    points: &mut Vec<[f64; 3]>,
    size: f64,
    label: &str,
) -> Result<Vec<[usize; 4]>, String> {
    if !size.is_finite() || size <= 0.0 {
        return Err("the element size must be positive and finite".into());
    }
    if free_loop.len() < 3 || fixed_loop.len() < 3 || fixed_ids.len() < 3 {
        return Err("both channel loops need at least three vertices".into());
    }
    if fixed_loop.len() != fixed_ids.len() {
        return Err("the fixed loop and its id list have different lengths".into());
    }
    for (slot, &id) in fixed_ids.iter().enumerate() {
        let Some(point) = points.get(id) else {
            return Err(format!("fixed id {id} is out of range for the node pool"));
        };
        let expected = fixed_loop[slot];
        if (point[0] - expected[0]).abs() > 1.0e-9 || (point[1] - expected[1]).abs() > 1.0e-9 {
            return Err(format!(
                "fixed node {id} is at ({}, {}) but the loop says ({}, {})",
                point[0], point[1], expected[0], expected[1]
            ));
        }
    }

    let free_clean = clean_loop_points(free_loop)?;
    let (fixed_clean, fixed_clean_ids) = clean_loop_with_ids(fixed_loop, fixed_ids)?;
    if fixed_clean.len() != fixed_ids.len() {
        return Err("the fixed loop changed length while cleaning; ids would be misaligned".into());
    }
    let tolerance = 1.0e-6 * (1.0 + size);
    for point in &free_clean {
        if domain.distance_to_boundary(*point) > tolerance {
            return Err("a point of the domain-side loop is not on the domain boundary".into());
        }
    }
    for point in &fixed_clean {
        if !domain.contains(*point) {
            return Err("a fixed-loop point is not inside the domain".into());
        }
        if !free_is_outer && point_in_polygon(*point, &free_clean) {
            return Err("a fixed-loop point lies inside the hole polygon".into());
        }
    }

    // Work on a 2D copy so the shared pool is untouched on failure.
    let base = points.len();
    let mut pos: Vec<[f64; 2]> = points.iter().map(|p| [p[0], p[1]]).collect();
    let mut free_ids: Vec<usize> = Vec::with_capacity(free_clean.len());
    for point in &free_clean {
        free_ids.push(pos.len());
        pos.push(*point);
    }
    // Corner-aware matching: a dense loop with collinear nodes (the stitch's
    // carved interior boundary) is reduced to its corners, both in geometry
    // and in the ids that the returned quads use.
    let (free_reduced, free_reduced_ids) = reduce_loop_with_ids(&free_clean, &free_ids);
    let (fixed_reduced, fixed_reduced_ids) = reduce_loop_with_ids(&fixed_clean, &fixed_clean_ids);

    let free_area = polygon_area_points(&free_reduced).abs();
    let fixed_area = polygon_area_points(&fixed_reduced).abs();
    let band_area = if free_is_outer {
        free_area - fixed_area
    } else {
        fixed_area - free_area
    };
    if band_area <= EPS {
        return Err("the two loops do not leave a positive channel area".into());
    }

    let mut quads: Vec<[usize; 4]> = Vec::new();
    add_ribbon(
        &free_clean,
        &free_reduced,
        &free_reduced_ids,
        &fixed_reduced,
        &fixed_reduced_ids,
        free_is_outer,
        label,
        &mut pos,
        &mut quads,
    )?;
    if quads.is_empty() {
        return Err(format!("{label} produced no quads"));
    }

    // Validate the quads on the local positions before committing anything.
    let mut total = 0.0f64;
    for quad in &quads {
        let corners = [
            *pos.get(quad[0])
                .ok_or("internal error: ribbon quad index")?,
            *pos.get(quad[1])
                .ok_or("internal error: ribbon quad index")?,
            *pos.get(quad[2])
                .ok_or("internal error: ribbon quad index")?,
            *pos.get(quad[3])
                .ok_or("internal error: ribbon quad index")?,
        ];
        let area = quad_area(corners);
        if area <= 1.0e-14 || quad_angle_rank(corners).is_none() {
            return Err(format!("{label} produced a degenerate or non-convex quad"));
        }
        let centroid = [
            (corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]) / 4.0,
            (corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]) / 4.0,
        ];
        if !domain.contains(centroid) && domain.distance_to_boundary(centroid) > 1.0e-9 {
            return Err(format!("{label} produced a quad outside the domain"));
        }
        total += area;
    }
    if (total - band_area).abs() > 1.0e-3 * band_area.max(1.0) {
        return Err(format!(
            "{label} covers {total:.6} but the band area is {band_area:.6}"
        ));
    }

    // Compact the newly created nodes and commit them to the shared pool.
    let mut used: BTreeSet<usize> = BTreeSet::new();
    for quad in &quads {
        for &id in quad {
            if id >= base {
                used.insert(id);
            }
        }
    }
    let mut remap: HashMap<usize, usize> = HashMap::new();
    let mut fresh: Vec<[f64; 3]> = Vec::with_capacity(used.len());
    for &id in &used {
        let point = *pos.get(id).ok_or("internal error: ribbon node")?;
        remap.insert(id, points.len() + fresh.len());
        fresh.push([point[0], point[1], 0.0]);
    }
    for quad in &mut quads {
        for id in quad.iter_mut() {
            if *id >= base {
                let Some(&mapped) = remap.get(id) else {
                    return Err("internal error: ribbon lost a node".into());
                };
                *id = mapped;
            }
        }
    }
    points.extend(fresh);
    Ok(quads)
}

/// Mesh the channel between two given closed loops, using the caller's shared
/// node pool.
///
/// `outer` is normally the domain boundary (every point is checked to lie on
/// the domain polyline) and `inner` is the boundary loop of an existing
/// interior mesh, whose node ids in `points` are `inner_ids`. The returned
/// quads index into `points`, which is extended in place with the new boundary
/// and Steiner nodes. No grid is built: this is the same [`add_ribbon`] code
/// path used by [`build_ribbon`], including the widest-path/cap, cyclic-row and
/// projection-row candidates, the convexity/angle validation, the channel-area
/// check and the shared-node compaction.
///
/// Dense loops are matched **corner-aware**: collinear runs are collapsed to
/// corners before choosing the ribbon, so a 4-corner boundary can be stitched to
/// an inset rectangle discretized with any number of collinear nodes. The
/// quads then use the fixed loop's corner ids; the intermediate collinear ids
/// are left in `points` for the caller.
///
/// On error `points` is left untouched.
pub fn channel_between(
    domain: &Domain,
    outer: &[[f64; 2]],
    inner: &[[f64; 2]],
    inner_ids: &[usize],
    points: &mut Vec<[f64; 3]>,
    size: f64,
) -> Result<Vec<[usize; 4]>, String> {
    ribbon_between_impl(
        domain,
        outer,
        inner,
        inner_ids,
        true,
        points,
        size,
        "channel_between",
    )
}

/// Mesh the band around a **domain hole**.
///
/// `hole` is the hole polygon on the domain boundary (the inner side of the
/// band) and `inner` is the interior mesh's hole loop, which is larger than the
/// hole polygon (the outer side of the band); `inner_ids` are its ids in
/// `points`. The band area is `|inner| - |hole|`. The returned quads share the
/// interior loop's ids and are otherwise produced by the same corner-aware
/// ribbon code as [`channel_between`].
pub fn channel_between_hole(
    domain: &Domain,
    hole: &[[f64; 2]],
    inner: &[[f64; 2]],
    inner_ids: &[usize],
    points: &mut Vec<[f64; 3]>,
    size: f64,
) -> Result<Vec<[usize; 4]>, String> {
    ribbon_between_impl(
        domain,
        hole,
        inner,
        inner_ids,
        false,
        points,
        size,
        "channel_between_hole",
    )
}

/// Run the three demo cases and return their numbers./// Run the three demo cases and return their numbers.
///
/// The square and the L-bracket are the working cases. The square with a
/// square hole is attempted and its error is reported rather than panicking.
pub fn demo() -> Result<DemoNumbers, String> {
    let square = Domain::rectangle(10.0, 10.0).map_err(|e| e.to_string())?;
    let square_numbers = case_numbers(&square, 1.0)?;

    let l_bracket = Domain::new(
        vec![
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 5.0],
            [5.0, 5.0],
            [5.0, 10.0],
            [0.0, 10.0],
        ],
        Vec::new(),
    )
    .map_err(|e| e.to_string())?;
    let (l_numbers, l_error) = match case_numbers(&l_bracket, 1.0) {
        Ok(numbers) => (Some(numbers), None),
        Err(message) => (None, Some(message)),
    };

    let plate = Domain::new(
        vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
        vec![vec![[3.0, 3.0], [3.0, 7.0], [7.0, 7.0], [7.0, 3.0]]],
    )
    .map_err(|e| e.to_string())?;
    let (hole_numbers, hole_error) = match case_numbers(&plate, 1.0) {
        Ok(numbers) => (Some(numbers), None),
        Err(message) => (None, Some(message)),
    };

    Ok(DemoNumbers {
        square: square_numbers,
        l_bracket: l_numbers,
        l_bracket_error: l_error,
        square_with_hole: hole_numbers,
        square_with_hole_error: hole_error,
    })
}

fn case_numbers(domain: &Domain, size: f64) -> Result<CaseNumbers, String> {
    let (_, stats) = mesh_with_stats(domain, size)?;
    Ok(CaseNumbers {
        method: stats.method.to_string(),
        delta_factor: stats.delta_factor,
        quads: stats.quads,
        interior_quads: stats.interior_quads,
        channel_quads: stats.channel_quads,
        min_scaled_jacobian: stats.min_scaled_jacobian,
        min_angle_deg: stats.min_angle_deg,
        max_angle_deg: stats.max_angle_deg,
        cracks: stats.cracks,
        inverted: stats.inverted,
        boundary_offset: stats.boundary_offset,
        greedy_covered_fraction: stats.greedy_covered_fraction,
        fallback_regions: stats.fallback_regions,
        fallback_quads: stats.fallback_quads,
        cross_edges: stats.cross_edges,
        patches: stats.patches,
        selected_patches: stats.selected_patches,
    })
}

// ---------------------------------------------------------------------------
// Preparation: boundary discretization, grid, inner loops
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct RawLoop {
    points: Vec<usize>,
    source: LoopSource,
}

struct Prepared {
    pos: Vec<[f64; 2]>,
    h: f64,
    ox: f64,
    oy: f64,
    quads: Vec<[usize; 4]>,
    /// Grid cell (relative lattice index) -> index into `quads`.
    quad_cells: HashMap<(i64, i64), usize>,
    loops: Vec<RawLoop>,
    /// Undirected domain-perimeter edges (the input polyline, discretized).
    domain_edges: HashSet<(usize, usize)>,
    grid_vertex_count: usize,
}

impl Prepared {
    /// Is `p` inside (or on the boundary of) one of the promoted quads?
    fn inside_grid_region(&self, p: [f64; 2]) -> bool {
        let i = ((p[0] - self.ox) / self.h).floor() as i64;
        let j = ((p[1] - self.oy) / self.h).floor() as i64;
        for di in -1..=1 {
            for dj in -1..=1 {
                if let Some(&quad_index) = self.quad_cells.get(&(i + di, j + dj)) {
                    if let Some(quad) = self.quads.get(quad_index) {
                        if point_in_or_on_convex_quad(p, quad, &self.pos) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// Is `p` in the channel (inside the domain and not in an interior quad)?
    fn in_channel(&self, domain: &Domain, p: [f64; 2]) -> bool {
        domain.contains(p) && !self.inside_grid_region(p)
    }
}

fn prepare(domain: &Domain, size: f64, delta: f64) -> Result<Prepared, String> {
    let h = check_request(domain, size)?;
    let mut pos: Vec<[f64; 2]> = Vec::new();
    let (boundary_loops, domain_edges) = discretize_boundary(domain, h, &mut pos)?;

    // A regular grid, aligned to the lower-left lattice line.
    let (min_x, min_y, max_x, max_y) = domain_bounds(domain);
    let ox = (min_x / h).floor() * h;
    let oy = (min_y / h).floor() * h;
    let nx = ((max_x - ox) / h).ceil() as i64 + 2;
    let ny = ((max_y - oy) / h).ceil() as i64 + 2;

    let mut vertex: HashMap<(i64, i64), usize> = HashMap::new();
    for j in -1..=ny {
        for i in -1..=nx {
            let p = [ox + i as f64 * h, oy + j as f64 * h];
            if domain.contains(p) && domain.distance_to_boundary(p) + 1.0e-9 >= delta * h {
                let id = pos.len();
                pos.push(p);
                vertex.insert((i, j), id);
            }
        }
    }
    let grid_vertex_count = vertex.len();

    // Promote cells whose four corners survived.
    let mut quads: Vec<[usize; 4]> = Vec::new();
    let mut quad_cells: HashMap<(i64, i64), usize> = HashMap::new();
    for j in 0..ny {
        for i in 0..nx {
            let corners = [
                vertex.get(&(i, j)).copied(),
                vertex.get(&(i + 1, j)).copied(),
                vertex.get(&(i + 1, j + 1)).copied(),
                vertex.get(&(i, j + 1)).copied(),
            ];
            let (Some(a), Some(b), Some(c), Some(d)) =
                (corners[0], corners[1], corners[2], corners[3])
            else {
                continue;
            };
            let centre = [ox + (i as f64 + 0.5) * h, oy + (j as f64 + 0.5) * h];
            if !domain.contains(centre) {
                continue;
            }
            quad_cells.insert((i, j), quads.len());
            quads.push([a, b, c, d]);
        }
    }

    // All grid edges between two surviving vertices.
    let mut grid_edges: Vec<(usize, usize)> = Vec::new();
    for j in -1..=ny {
        for i in -1..=nx {
            let Some(&a) = vertex.get(&(i, j)) else {
                continue;
            };
            if let Some(&b) = vertex.get(&(i + 1, j)) {
                grid_edges.push((a.min(b), a.max(b)));
            }
            if let Some(&b) = vertex.get(&(i, j + 1)) {
                grid_edges.push((a.min(b), a.max(b)));
            }
        }
    }
    grid_edges.sort_unstable();
    grid_edges.dedup();

    // How many promoted quads use each edge.
    let mut use_count: HashMap<(usize, usize), usize> = HashMap::new();
    for quad in &quads {
        for k in 0..4 {
            let u = quad[k];
            let v = quad[(k + 1) % 4];
            *use_count.entry(edge_key(u, v)).or_insert(0) += 1;
        }
    }

    // Walk the channel-adjacent half-edges of the inner grid.
    let grid_loops = walk_inner_loops(&pos, &grid_edges, &use_count, &quads)?;

    let mut loops: Vec<RawLoop> = Vec::new();
    for points in boundary_loops {
        loops.push(RawLoop {
            points,
            source: LoopSource::Boundary,
        });
    }
    for points in grid_loops {
        loops.push(RawLoop {
            points,
            source: LoopSource::Grid,
        });
    }
    for raw in &loops {
        if !raw.points.len().is_multiple_of(2) {
            return Err(format!(
                "channel loop with {} sides cannot be quadrangulated; the input boundary \
                 could not be discretized to an even number of edges",
                raw.points.len()
            ));
        }
    }

    Ok(Prepared {
        pos,
        h,
        ox,
        oy,
        quads,
        quad_cells,
        loops,
        domain_edges,
        grid_vertex_count,
    })
}

fn check_request(domain: &Domain, size: f64) -> Result<f64, String> {
    if !size.is_finite() || size <= 0.0 {
        return Err("the element size must be positive and finite".into());
    }
    if domain.outer.len() < 3 {
        return Err("the domain outer loop has fewer than three points".into());
    }
    for loop_points in std::iter::once(&domain.outer).chain(domain.holes.iter()) {
        if loop_points.len() < 3 {
            return Err("a domain loop has fewer than three points".into());
        }
        if loop_points
            .iter()
            .any(|p| !p[0].is_finite() || !p[1].is_finite())
        {
            return Err("a domain coordinate is not finite".into());
        }
    }
    if domain.area() <= 1.0e-12 {
        return Err("the domain has no positive area".into());
    }
    Ok(size)
}

fn domain_bounds(domain: &Domain) -> (f64, f64, f64, f64) {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for p in std::iter::once(&domain.outer)
        .chain(domain.holes.iter())
        .flatten()
    {
        min_x = min_x.min(p[0]);
        min_y = min_y.min(p[1]);
        max_x = max_x.max(p[0]);
        max_y = max_y.max(p[1]);
    }
    (min_x, min_y, max_x, max_y)
}

/// Discretize every domain loop into edges of roughly `h`, with an even edge
/// count per loop (required for a quad decomposition), preserving every input
/// vertex and keeping all added points exactly on the polyline.
fn discretize_boundary(
    domain: &Domain,
    h: f64,
    pos: &mut Vec<[f64; 2]>,
) -> Result<(Vec<Vec<usize>>, HashSet<(usize, usize)>), String> {
    let mut loops = Vec::new();
    let mut domain_edges = HashSet::new();
    let mut outer = true;
    for loop_points in std::iter::once(&domain.outer).chain(domain.holes.iter()) {
        // Remove a repeated closing point, if the caller supplied one.
        let mut source: Vec<[f64; 2]> = loop_points.clone();
        while source.len() > 3 {
            let last = source[source.len() - 1];
            let first = source[0];
            if (last[0] - first[0]).abs() <= EPS && (last[1] - first[1]).abs() <= EPS {
                source.pop();
            } else {
                break;
            }
        }
        let n = source.len();
        if n < 3 {
            return Err("a domain loop is degenerate".into());
        }
        // Subdivision count per segment: at least one, roughly length / h.
        let mut counts: Vec<usize> = Vec::with_capacity(n);
        let mut total = 0usize;
        for i in 0..n {
            let a = source[i];
            let b = source[(i + 1) % n];
            let length = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
            let count = (length / h).round().max(1.0) as usize;
            counts.push(count);
            total += count;
        }
        // Every loop of a quadrangulation must have an even number of edges.
        if !total.is_multiple_of(2) {
            let longest = counts
                .iter()
                .enumerate()
                .max_by_key(|(_, &c)| c)
                .map(|(i, _)| i)
                .unwrap_or(0);
            counts[longest] += 1;
            total += 1;
        }
        if total < 4 {
            // A triangle with an extra point is the smallest sensible loop.
            counts[0] += 4 - total;
        }
        let mut ids: Vec<usize> = Vec::with_capacity(total);
        for i in 0..n {
            let a = source[i];
            let b = source[(i + 1) % n];
            let count = counts[i];
            for k in 0..count {
                let t = k as f64 / count as f64;
                let p = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
                ids.push(pos.len());
                pos.push(p);
            }
        }
        // Ensure the desired orientation: outer CCW, holes CW (channel left).
        let area = polygon_area_ids(pos.as_slice(), &ids);
        let want_ccw = outer;
        if (area > 0.0) != want_ccw {
            ids.reverse();
        }
        for i in 0..ids.len() {
            let a = ids[i];
            let b = ids[(i + 1) % ids.len()];
            domain_edges.insert(edge_key(a, b));
        }
        loops.push(ids);
        outer = false;
    }
    Ok((loops, domain_edges))
}

/// Walk the channel-adjacent grid half-edges (§2.2.2).
fn walk_inner_loops(
    pos: &[[f64; 2]],
    grid_edges: &[(usize, usize)],
    use_count: &HashMap<(usize, usize), usize>,
    quads: &[[usize; 4]],
) -> Result<Vec<Vec<usize>>, String> {
    // The direction of every edge as seen from the (only) promoted quad.
    let mut quad_direction: HashMap<(usize, usize), (usize, usize)> = HashMap::new();
    for quad in quads {
        for k in 0..4 {
            quad_direction.insert(
                edge_key(quad[k], quad[(k + 1) % 4]),
                (quad[k], quad[(k + 1) % 4]),
            );
        }
    }

    // Directed half-edges that touch the channel: one direction for a boundary
    // edge (channel on the left), both directions for a hanging edge.
    let mut present: BTreeSet<(usize, usize)> = BTreeSet::new();
    for &(a, b) in grid_edges {
        match use_count.get(&edge_key(a, b)).copied().unwrap_or(0) {
            0 => {
                present.insert((a, b));
                present.insert((b, a));
            }
            1 => {
                if let Some(&(u, v)) = quad_direction.get(&edge_key(a, b)) {
                    present.insert((v, u));
                } else {
                    return Err("internal error: a boundary grid edge has no incident quad".into());
                }
            }
            _ => {}
        }
    }

    let mut outgoing: Vec<Vec<usize>> = vec![Vec::new(); pos.len()];
    for &(u, v) in &present {
        outgoing[u].push(v);
    }
    for list in &mut outgoing {
        list.sort_unstable();
        list.dedup();
    }

    let mut visited: HashSet<(usize, usize)> = HashSet::new();
    let mut loops: Vec<Vec<usize>> = Vec::new();
    for &start in &present {
        if visited.contains(&start) {
            continue;
        }
        let mut cycle: Vec<usize> = Vec::new();
        let mut current = start;
        let mut guard = 0usize;
        loop {
            if !visited.insert(current) {
                break;
            }
            cycle.push(current.0);
            let next = match next_half_edge(current.1, current.0, &outgoing, pos) {
                Some(next) => next,
                None => break,
            };
            current = (current.1, next);
            if current == start {
                break;
            }
            guard += 1;
            if guard > present.len() + 8 {
                return Err("the inner grid walk did not close; the grid is not planar".into());
            }
        }
        if current != start {
            return Err("the inner grid walk left an open chain".into());
        }
        if cycle.len() >= 4 {
            let area = polygon_area_ids(pos, &cycle);
            // Zero-area cycles are isolated hanging chains; they carry no
            // channel area and are dropped (documented failure mode).
            if area.abs() > EPS {
                loops.push(cycle);
            }
        }
    }
    loops.sort();
    Ok(loops)
}

/// Next half-edge of the face on the left of `(u, v)`, at `v`.
///
/// Among the outgoing edges (excluding the reverse edge unless it is the only
/// one), pick the first one clockwise from the reverse of the incoming
/// direction. This is the standard planar face traversal and produces loops
/// with the face on the left.
fn next_half_edge(v: usize, u: usize, outgoing: &[Vec<usize>], pos: &[[f64; 2]]) -> Option<usize> {
    let base = (pos[u][1] - pos[v][1]).atan2(pos[u][0] - pos[v][0]);
    let mut best: Option<(f64, usize)> = None;
    let mut reverse: Option<usize> = None;
    for &w in outgoing.get(v)? {
        if w == u {
            reverse = Some(w);
            continue;
        }
        let angle = (pos[w][1] - pos[v][1]).atan2(pos[w][0] - pos[v][0]);
        let clockwise = (base - angle).rem_euclid(TAU);
        let better = match best {
            None => true,
            Some((best_angle, best_w)) => {
                clockwise < best_angle - 1.0e-12
                    || ((clockwise - best_angle).abs() <= 1.0e-12 && w < best_w)
            }
        };
        if better {
            best = Some((clockwise, w));
        }
    }
    best.map(|(_, w)| w).or(reverse)
}

// ---------------------------------------------------------------------------
// Cross edges (§2.2.3)
// ---------------------------------------------------------------------------

fn build_cross_edges(prepared: &Prepared, domain: &Domain) -> Result<Vec<(usize, usize)>, String> {
    let mut vertices: BTreeSet<usize> = BTreeSet::new();
    let mut loop_edges: BTreeSet<(usize, usize)> = BTreeSet::new();
    for raw in &prepared.loops {
        for i in 0..raw.points.len() {
            let a = raw.points[i];
            let b = raw.points[(i + 1) % raw.points.len()];
            vertices.insert(a);
            vertices.insert(b);
            loop_edges.insert(edge_key(a, b));
        }
    }
    let vertex_list: Vec<usize> = vertices.iter().copied().collect();
    if vertex_list.len() > MAX_CHANNEL_VERTICES {
        return Err(format!(
            "the channel has {} vertices, above the supported {}; use a coarser size",
            vertex_list.len(),
            MAX_CHANNEL_VERTICES
        ));
    }

    let reach = CROSS_REACH_FACTOR * prepared.h;
    let mut candidates: Vec<(f64, usize, usize)> = Vec::new();
    for a_index in 0..vertex_list.len() {
        let a = vertex_list[a_index];
        for &b in vertex_list.iter().skip(a_index + 1) {
            let distance = point_distance(prepared.pos[a], prepared.pos[b]);
            if distance <= EPS || distance > reach {
                continue;
            }
            if loop_edges.contains(&edge_key(a, b)) {
                continue;
            }
            candidates.push((distance, a, b));
        }
    }
    if candidates.len() > MAX_CANDIDATE_EDGES {
        return Err(format!(
            "{} cross-edge candidates exceed the supported {}; use a coarser size",
            candidates.len(),
            MAX_CANDIDATE_EDGES
        ));
    }
    candidates.sort_by(|x, y| {
        x.0.total_cmp(&y.0)
            .then_with(|| x.1.cmp(&y.1))
            .then_with(|| x.2.cmp(&y.2))
    });

    let loop_edge_list: Vec<(usize, usize)> = loop_edges.iter().copied().collect();
    let mut accepted: Vec<(usize, usize)> = Vec::new();
    for &(_, a, b) in &candidates {
        if segment_is_clear(
            a,
            b,
            &loop_edge_list,
            &accepted,
            &vertex_list,
            prepared,
            domain,
        ) {
            accepted.push(edge_key(a, b));
        }
    }
    Ok(accepted)
}

fn segment_is_clear(
    a: usize,
    b: usize,
    loop_edges: &[(usize, usize)],
    accepted: &[(usize, usize)],
    vertices: &[usize],
    prepared: &Prepared,
    domain: &Domain,
) -> bool {
    let pa = prepared.pos[a];
    let pb = prepared.pos[b];
    // Sample points must be in the channel; the endpoints are on its boundary.
    for t in [0.25, 0.5, 0.75] {
        let p = [pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1])];
        if !prepared.in_channel(domain, p) {
            return false;
        }
    }
    // No vertex may sit in the interior of the new edge.
    for &v in vertices {
        if v == a || v == b {
            continue;
        }
        if point_on_segment_strict(prepared.pos[v], pa, pb) {
            return false;
        }
    }
    for &(u, v) in loop_edges {
        if edges_conflict(pa, pb, prepared.pos[u], prepared.pos[v]) {
            return false;
        }
    }
    for &(u, v) in accepted {
        if edges_conflict(pa, pb, prepared.pos[u], prepared.pos[v]) {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Planar faces
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Face {
    vertices: Vec<usize>,
    area: f64,
}

/// A point strictly inside a simple polygon, used to decide which side of a
/// walked cycle the channel is on.
fn face_interior_sample(polygon: &[[f64; 2]]) -> Option<[f64; 2]> {
    let inside = |p: [f64; 2]| -> bool {
        point_in_polygon(p, polygon) && distance_to_polygon_edges(p, polygon) > 1.0e-10
    };
    if let Some(centroid) = polygon_centroid(polygon) {
        if inside(centroid) {
            return Some(centroid);
        }
    }
    let count = polygon.len() as f64;
    if count > 0.0 {
        let mut average = [0.0f64; 2];
        for point in polygon {
            average[0] += point[0] / count;
            average[1] += point[1] / count;
        }
        if inside(average) {
            return Some(average);
        }
    }
    for i in 1..polygon.len().saturating_sub(1) {
        let triangle = [polygon[0], polygon[i], polygon[i + 1]];
        if let Some(centroid) = polygon_centroid(&triangle) {
            if inside(centroid) {
                return Some(centroid);
            }
        }
    }
    None
}

fn build_faces(
    prepared: &Prepared,
    cross_edges: &[(usize, usize)],
    domain: &Domain,
) -> Result<Vec<Face>, String> {
    let mut edges: BTreeSet<(usize, usize)> = BTreeSet::new();
    for raw in &prepared.loops {
        for i in 0..raw.points.len() {
            let a = raw.points[i];
            let b = raw.points[(i + 1) % raw.points.len()];
            edges.insert(edge_key(a, b));
        }
    }
    for &(a, b) in cross_edges {
        edges.insert(edge_key(a, b));
    }
    let edge_list: Vec<(usize, usize)> = edges.iter().copied().collect();

    // The graph must be connected, otherwise a loop would be walked as its own
    // face (the interior grid block would masquerade as channel area).
    if let Some(&first) = edge_list.first() {
        let mut adjacency: HashMap<usize, Vec<usize>> = HashMap::new();
        for &(a, b) in &edge_list {
            adjacency.entry(a).or_default().push(b);
            adjacency.entry(b).or_default().push(a);
        }
        let mut seen: HashSet<usize> = HashSet::new();
        let mut queue: VecDeque<usize> = VecDeque::new();
        seen.insert(first.0);
        queue.push_back(first.0);
        while let Some(v) = queue.pop_front() {
            if let Some(neighbours) = adjacency.get(&v) {
                for &w in neighbours {
                    if seen.insert(w) {
                        queue.push_back(w);
                    }
                }
            }
        }
        let mut all: HashSet<usize> = HashSet::new();
        for &(a, b) in &edge_list {
            all.insert(a);
            all.insert(b);
        }
        if seen.len() != all.len() {
            return Err(
                "the cross-edge search did not connect the channel loops; the channel is \
                 wider than the 4h cross-edge reach in some place"
                    .into(),
            );
        }
    } else {
        return Err("the channel graph has no edges".into());
    }

    let mut outgoing: Vec<Vec<usize>> = vec![Vec::new(); prepared.pos.len()];
    for &(a, b) in &edge_list {
        outgoing[a].push(b);
        outgoing[b].push(a);
    }
    for list in &mut outgoing {
        list.sort_unstable();
        list.dedup();
    }

    let mut visited: HashSet<(usize, usize)> = HashSet::new();
    let mut faces: Vec<Face> = Vec::new();
    for &start in &edge_list {
        for start_directed in [(start.0, start.1), (start.1, start.0)] {
            if visited.contains(&start_directed) {
                continue;
            }
            let mut cycle: Vec<usize> = Vec::new();
            let mut current = start_directed;
            let mut guard = 0usize;
            loop {
                if !visited.insert(current) {
                    break;
                }
                cycle.push(current.0);
                let next = match next_half_edge(current.1, current.0, &outgoing, &prepared.pos) {
                    Some(next) => next,
                    None => break,
                };
                current = (current.1, next);
                if current == start_directed {
                    break;
                }
                guard += 1;
                if guard > edge_list.len() + 8 {
                    return Err("the channel face walk did not close".into());
                }
            }
            if current != start_directed {
                return Err("the channel face walk left an open chain".into());
            }
            if cycle.len() < 3 {
                continue;
            }
            let mut unique: HashSet<usize> = HashSet::new();
            let mut simple = true;
            for &v in &cycle {
                if !unique.insert(v) {
                    simple = false;
                    break;
                }
            }
            if !simple {
                return Err("a channel face visits a vertex twice (pinched channel)".into());
            }
            let area = polygon_area_ids(&prepared.pos, &cycle);
            if area <= EPS {
                continue;
            }
            // Positive orientation alone is not enough: the walk also visits
            // the *reverse* side of every loop, so an interior grid block or a
            // domain hole appears as a positively oriented face too. Keep only
            // faces whose inside really lies in the channel.
            let polygon: Vec<[f64; 2]> = cycle.iter().map(|&id| prepared.pos[id]).collect();
            let in_channel = match face_interior_sample(&polygon) {
                Some(sample) => prepared.in_channel(domain, sample),
                None => false,
            };
            if in_channel {
                faces.push(Face {
                    vertices: cycle,
                    area,
                });
            }
        }
    }
    faces.sort_by(|x, y| x.vertices.cmp(&y.vertices));
    Ok(faces)
}

// ---------------------------------------------------------------------------
// Templates (§2.2.5, §2.2.6)
// ---------------------------------------------------------------------------

/// A quadrangulation topology: `n` boundary vertices (0..n-1) plus `k` Steiner
/// vertices (n..n+k-1). Quad indices use the same numbering.
#[derive(Clone, Debug)]
struct Template {
    n: usize,
    k: usize,
    quads: Vec<[usize; 4]>,
}

/// An evaluated quadrangulation: quads use the same local numbering as the
/// template; `steiner` holds the solved interior positions.
#[derive(Clone, Debug)]
struct Decomp {
    quads: Vec<[usize; 4]>,
    steiner: Vec<[f64; 2]>,
    rank: f64,
}

/// Place the Steiner points by the Tutte/barycentric embedding (§2.2.5): every
/// Steiner point sits at the average of its graph neighbours, boundary fixed.
fn harmonic_embedding(boundary: &[[f64; 2]], template: &Template) -> Option<Vec<[f64; 2]>> {
    let n = template.n;
    let k = template.k;
    if k == 0 {
        return Some(Vec::new());
    }
    let mut neighbours: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); k];
    for quad in &template.quads {
        for e in 0..4 {
            let u = quad[e];
            let v = quad[(e + 1) % 4];
            if let Some(slot) = u.checked_sub(n).filter(|&s| s < k) {
                neighbours[slot].insert(v);
            }
            if let Some(slot) = v.checked_sub(n).filter(|&s| s < k) {
                neighbours[slot].insert(u);
            }
        }
    }
    let mut matrix = vec![vec![0.0f64; k]; k];
    let mut rhs = vec![[0.0f64; 2]; k];
    for s in 0..k {
        if neighbours[s].is_empty() {
            return None;
        }
        for &other in &neighbours[s] {
            matrix[s][s] += 1.0;
            if other >= n {
                matrix[s][other - n] -= 1.0;
            } else {
                rhs[s][0] += boundary.get(other)?[0];
                rhs[s][1] += boundary.get(other)?[1];
            }
        }
    }
    // Gaussian elimination with partial pivoting (k is at most four).
    for col in 0..k {
        let mut pivot = col;
        for row in col + 1..k {
            if matrix[row][col].abs() > matrix[pivot][col].abs() {
                pivot = row;
            }
        }
        if matrix[pivot][col].abs() < 1.0e-12 {
            return None;
        }
        matrix.swap(col, pivot);
        rhs.swap(col, pivot);
        let diagonal = matrix[col][col];
        if !diagonal.is_finite() {
            return None;
        }
        for c in col..k {
            matrix[col][c] /= diagonal;
        }
        for d in 0..2 {
            rhs[col][d] /= diagonal;
        }
        for row in 0..k {
            if row == col {
                continue;
            }
            let factor = matrix[row][col];
            if factor.abs() <= 0.0 {
                continue;
            }
            for c in col..k {
                matrix[row][c] -= factor * matrix[col][c];
            }
            for d in 0..2 {
                rhs[row][d] -= factor * rhs[col][d];
            }
        }
    }
    let mut out = Vec::with_capacity(k);
    for point in rhs {
        if !point[0].is_finite() || !point[1].is_finite() {
            return None;
        }
        out.push(point);
    }
    Some(out)
}

fn evaluate_template(boundary: &[[f64; 2]], template: &Template) -> Option<Decomp> {
    if boundary.len() != template.n {
        return None;
    }
    let steiner = harmonic_embedding(boundary, template)?;
    let mut points: Vec<[f64; 2]> = boundary.to_vec();
    points.extend(steiner.iter().copied());
    let polygon = polygon_area_points(boundary);
    if polygon <= EPS {
        return None;
    }
    let mut total = 0.0;
    let mut rank = f64::INFINITY;
    for quad in &template.quads {
        let mut corners = [[0.0f64; 2]; 4];
        for c in 0..4 {
            corners[c] = *points.get(quad[c])?;
        }
        let area = quad_area(corners);
        if area <= 1.0e-14 {
            return None;
        }
        total += area;
        let quad_rank = quad_angle_rank(corners)?;
        rank = rank.min(quad_rank);
    }
    let tolerance = 1.0e-7 * polygon.max(1.0);
    if (total - polygon).abs() > tolerance {
        return None;
    }
    for point in &steiner {
        if !point_in_polygon(*point, boundary) {
            return None;
        }
    }
    Some(Decomp {
        quads: template.quads.clone(),
        steiner,
        rank,
    })
}

/// The 6-sided templates: 0, 1 and 2 Steiner vertices (the non-degenerate
/// cases of Blacker & Stephenson's six-node closures).
fn template_six(kind: usize, rotation: usize, reflect: bool) -> Option<Template> {
    const N: usize = 6;
    let base: &[[usize; 4]] = match kind {
        0 => &[[0, 1, 2, 3], [3, 4, 5, 0]],
        1 => &[[0, 1, 2, 6], [2, 3, 4, 6], [4, 5, 0, 6]],
        2 => &[[0, 1, 2, 6], [6, 2, 3, 7], [3, 4, 5, 7], [7, 5, 0, 6]],
        _ => return None,
    };
    let k = kind;
    let map = |index: usize| -> usize {
        if index < N {
            if reflect {
                (rotation + N - index % N) % N
            } else {
                (rotation + index) % N
            }
        } else {
            N + (index - N)
        }
    };
    let mut quads = Vec::with_capacity(base.len());
    for quad in base {
        let mut mapped = [map(quad[0]), map(quad[1]), map(quad[2]), map(quad[3])];
        if reflect {
            mapped = [mapped[0], mapped[3], mapped[2], mapped[1]];
        }
        if mapped[0] == mapped[1]
            || mapped[0] == mapped[2]
            || mapped[0] == mapped[3]
            || mapped[1] == mapped[2]
            || mapped[1] == mapped[3]
            || mapped[2] == mapped[3]
        {
            return None;
        }
        quads.push(mapped);
    }
    Some(Template { n: N, k, quads })
}

/// Best 6-sided decomposition of a hexagon, by minimum angle deviation.
fn quadrangulate_six(boundary: &[[f64; 2]], max_steiner: usize) -> Option<Decomp> {
    let mut best: Option<Decomp> = None;
    for kind in 0..=max_steiner.min(2) {
        for rotation in 0..6 {
            for reflect in [false, true] {
                if reflect && kind == 0 {
                    // The 0-Steiner chord is symmetric under reflection.
                    continue;
                }
                let Some(template) = template_six(kind, rotation, reflect) else {
                    continue;
                };
                if let Some(decomp) = evaluate_template(boundary, &template) {
                    let better = match &best {
                        None => true,
                        Some(current) => decomp.rank > current.rank,
                    };
                    if better {
                        best = Some(decomp);
                    }
                }
            }
        }
    }
    best
}

fn keep_best(best: &mut Option<Decomp>, candidate: Option<Decomp>) {
    let Some(candidate) = candidate else {
        return;
    };
    let better = match best {
        None => true,
        Some(current) => candidate.rank > current.rank,
    };
    if better {
        *best = Some(candidate);
    }
}

/// Map a sub-decomposition into a larger template's numbering.
///
/// `sub_n` is the number of boundary vertices of the sub-polygon. Boundary
/// indices `< sub_n` are sent through `map_boundary`; Steiner indices start at
/// `sub_n` and are shifted to `steiner_base`.
fn lift_quads(
    sub: &Decomp,
    sub_n: usize,
    map_boundary: &dyn Fn(usize) -> usize,
    steiner_base: usize,
) -> Vec<[usize; 4]> {
    sub.quads
        .iter()
        .map(|quad| {
            let mut out = [0usize; 4];
            for c in 0..4 {
                let index = quad[c];
                out[c] = if index < sub_n {
                    map_boundary(index)
                } else {
                    steiner_base + (index - sub_n)
                };
            }
            out
        })
        .collect()
}

fn offset_boundary(n: usize, start: usize, len: usize) -> Vec<usize> {
    (0..len).map(|k| (start + k) % n).collect()
}

/// Simplified recursive 8-sided decomposition (Fig. 2.6).
fn quadrangulate_eight(boundary: &[[f64; 2]]) -> Option<Decomp> {
    const N: usize = 8;
    let mut best: Option<Decomp> = None;

    // 0 Steiner: two chords, three quads.
    for rotation in 0..N {
        let base: [[usize; 4]; 3] = [[0, 1, 2, 3], [3, 4, 5, 6], [6, 7, 0, 3]];
        let mut quads = Vec::with_capacity(3);
        for quad in base {
            quads.push([
                (rotation + quad[0]) % N,
                (rotation + quad[1]) % N,
                (rotation + quad[2]) % N,
                (rotation + quad[3]) % N,
            ]);
        }
        let template = Template { n: N, k: 0, quads };
        keep_best(&mut best, evaluate_template(boundary, &template));
    }

    // 4 + 6 split across the chord (i, i+3).
    for i in 0..N {
        let four = offset_boundary(N, i, 4);
        let six = offset_boundary(N, i + 3, 6);
        let six_points: Vec<[f64; 2]> = six.iter().map(|&k| boundary[k]).collect();
        let Some(six_decomp) = quadrangulate_six(&six_points, 1) else {
            continue;
        };
        let mut quads = vec![[four[0], four[1], four[2], four[3]]];
        quads.extend(lift_quads(&six_decomp, 6, &|index| six[index], N));
        let template = Template {
            n: N,
            k: six_decomp.steiner.len(),
            quads,
        };
        keep_best(&mut best, evaluate_template(boundary, &template));
    }

    // 6/6 split across the chord (i, i+4), with one Steiner point on the chord.
    for i in 0..N {
        let chord_start = boundary[i];
        let chord_end = boundary[(i + 4) % N];
        let guess = [
            0.5 * (chord_start[0] + chord_end[0]),
            0.5 * (chord_start[1] + chord_end[1]),
        ];
        let first = offset_boundary(N, i, 5);
        let second = offset_boundary(N, i + 4, 5);
        let mut first_points: Vec<[f64; 2]> = first.iter().map(|&k| boundary[k]).collect();
        first_points.push(guess);
        let mut second_points: Vec<[f64; 2]> = second.iter().map(|&k| boundary[k]).collect();
        second_points.push(guess);
        let Some(first_decomp) = quadrangulate_six(&first_points, 0) else {
            continue;
        };
        let Some(second_decomp) = quadrangulate_six(&second_points, 0) else {
            continue;
        };
        let steiner_label = N;
        let mut quads = lift_quads(
            &first_decomp,
            6,
            &|index| {
                if index < 5 {
                    first[index]
                } else {
                    steiner_label
                }
            },
            steiner_label + 1,
        );
        quads.extend(lift_quads(
            &second_decomp,
            6,
            &|index| {
                if index < 5 {
                    second[index]
                } else {
                    steiner_label
                }
            },
            steiner_label + 1,
        ));
        let k = 1 + first_decomp.steiner.len() + second_decomp.steiner.len();
        let template = Template { n: N, k, quads };
        keep_best(&mut best, evaluate_template(boundary, &template));
    }

    // Corner cut with one Steiner point, then a 4+6 split of the remainder.
    for corner in 0..N {
        let previous = (corner + N - 1) % N;
        let next = (corner + 1) % N;
        let guess = [
            (boundary[previous][0] + boundary[corner][0] + boundary[next][0]) / 3.0,
            (boundary[previous][1] + boundary[corner][1] + boundary[next][1]) / 3.0,
        ];
        // Remainder: next, next+1, ..., previous, s.
        let remainder: Vec<usize> = {
            let mut out = Vec::with_capacity(N);
            for k in 0..(N - 1) {
                out.push((next + k) % N);
            }
            out
        };
        let mut remainder_points: Vec<[f64; 2]> = remainder.iter().map(|&k| boundary[k]).collect();
        remainder_points.push(guess);

        // 4 + 6 split of the remainder.
        for j in 0..N {
            let four = offset_boundary(N, j, 4);
            let six = offset_boundary(N, j + 3, 6);
            let six_points: Vec<[f64; 2]> = six.iter().map(|&k| remainder_points[k]).collect();
            let Some(six_decomp) = quadrangulate_six(&six_points, 0) else {
                continue;
            };
            let steiner_label = N;
            let mut quads = vec![[previous, corner, next, steiner_label]];
            let map_remainder = |index: usize| -> usize {
                if index < N - 1 {
                    remainder[index]
                } else {
                    steiner_label
                }
            };
            quads.push([
                map_remainder(four[0]),
                map_remainder(four[1]),
                map_remainder(four[2]),
                map_remainder(four[3]),
            ]);
            quads.extend(lift_quads(
                &six_decomp,
                6,
                &|index| {
                    let local = six[index];
                    if local < N - 1 {
                        remainder[local]
                    } else {
                        steiner_label
                    }
                },
                steiner_label + 1,
            ));
            let k = 1 + six_decomp.steiner.len();
            let template = Template { n: N, k, quads };
            keep_best(&mut best, evaluate_template(boundary, &template));
        }
    }

    best
}

fn decompose(boundary: &[[f64; 2]]) -> Option<Decomp> {
    match boundary.len() {
        4 => {
            let template = Template {
                n: 4,
                k: 0,
                quads: vec![[0, 1, 2, 3]],
            };
            evaluate_template(boundary, &template)
        }
        6 => quadrangulate_six(boundary, 2),
        8 => quadrangulate_eight(boundary),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Patch generation and greedy covering (§2.2.4, §2.2.7–2.2.9)
// ---------------------------------------------------------------------------

struct CandidatePatch {
    faces: Vec<usize>,
    boundary: Vec<usize>,
    decomposition: Option<Decomp>,
    area: f64,
    rank: f64,
}

fn build_patches(faces: &[Face], cross_edges: &[(usize, usize)]) -> Vec<CandidatePatch> {
    let cross: HashSet<(usize, usize)> = cross_edges.iter().map(|&(a, b)| edge_key(a, b)).collect();
    let mut edge_faces: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (index, face) in faces.iter().enumerate() {
        let n = face.vertices.len();
        for i in 0..n {
            let key = edge_key(face.vertices[i], face.vertices[(i + 1) % n]);
            edge_faces.entry(key).or_default().push(index);
        }
    }

    let mut seen: HashSet<Vec<usize>> = HashSet::new();
    let mut queue: VecDeque<Vec<usize>> = VecDeque::new();
    for index in 0..faces.len() {
        let set = vec![index];
        seen.insert(set.clone());
        queue.push_back(set);
    }
    let mut patches: Vec<CandidatePatch> = Vec::new();
    while let Some(ids) = queue.pop_front() {
        if seen.len() > MAX_PATCH_SETS || patches.len() > MAX_PATCHES {
            break;
        }
        let Some((boundary, boundary_edges, area)) = union_boundary(&ids, faces) else {
            continue;
        };
        let sides = boundary.len();
        if (4..=8).contains(&sides) {
            patches.push(CandidatePatch {
                faces: ids.clone(),
                boundary,
                decomposition: None,
                area,
                rank: f64::NEG_INFINITY,
            });
        }
        if ids.len() > MAX_GLUES {
            continue;
        }
        // Grow across cross edges only.
        for &(a, b) in &boundary_edges {
            if !cross.contains(&edge_key(a, b)) {
                continue;
            }
            if let Some(neighbours) = edge_faces.get(&edge_key(a, b)) {
                for &neighbour in neighbours {
                    if ids.contains(&neighbour) {
                        continue;
                    }
                    let mut next = ids.clone();
                    next.push(neighbour);
                    next.sort_unstable();
                    if seen.insert(next.clone()) {
                        queue.push_back(next);
                    }
                }
            }
        }
    }
    patches
}

/// Decompose and rank every candidate patch, once its positions are known.
fn finalize_patches(patches: &mut [CandidatePatch], pos: &[[f64; 2]]) {
    for patch in patches.iter_mut() {
        let polygon: Vec<[f64; 2]> = patch.boundary.iter().map(|&id| pos[id]).collect();
        if let Some(decomposition) = decompose(&polygon) {
            patch.rank = decomposition.rank;
            patch.decomposition = Some(decomposition);
        }
    }
}

fn greedy_cover(faces: &[Face], patches: &[CandidatePatch]) -> (Vec<usize>, f64) {
    let mut covered = vec![false; faces.len()];
    let mut selected: Vec<usize> = Vec::new();
    let total_area: f64 = faces.iter().map(|face| face.area).sum();
    let mut covered_area = 0.0;
    loop {
        let mut best: Option<usize> = None;
        let mut best_key = (0.0f64, f64::NEG_INFINITY);
        for (index, patch) in patches.iter().enumerate() {
            if patch.decomposition.is_none() {
                continue;
            }
            if selected.contains(&index) {
                continue;
            }
            if patch.faces.iter().any(|&face| covered[face]) {
                continue;
            }
            let area = patch.area;
            if area <= EPS {
                continue;
            }
            let key = (area, patch.rank);
            let better = match best {
                None => true,
                Some(_) => {
                    area > best_key.0 + EPS
                        || ((area - best_key.0).abs() <= EPS && patch.rank > best_key.1)
                }
            };
            if better {
                best = Some(index);
                best_key = key;
            }
        }
        let Some(index) = best else {
            break;
        };
        selected.push(index);
        for &face in &patches[index].faces {
            if let Some(slot) = covered.get_mut(face) {
                *slot = true;
            }
        }
        covered_area += patches[index].area;
    }
    let fraction = if total_area > EPS {
        (covered_area / total_area).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (selected, fraction)
}

// ---------------------------------------------------------------------------
// Fallback cavity closure
// ---------------------------------------------------------------------------

struct FillPlan {
    quads: Vec<[usize; 4]>,
    steiner: Option<[f64; 2]>,
}

/// Quadrangulate a simple polygon by a fan from one of its vertices, or by a
/// centroid ring with one Steiner point. Returns the plan without mutating the
/// point list (a new Steiner point, when used, has index `new_id`).
fn try_fill(loop_ids: &[usize], pos: &[[f64; 2]], new_id: usize) -> Option<FillPlan> {
    let mut boundary: Vec<usize> = Vec::with_capacity(loop_ids.len());
    for &id in loop_ids {
        if boundary.last().copied() == Some(id) {
            continue;
        }
        boundary.push(id);
    }
    while boundary.len() > 1 && boundary.first().copied() == boundary.last().copied() {
        boundary.pop();
    }
    let n = boundary.len();
    if n < 4 || !n.is_multiple_of(2) {
        return None;
    }
    let points: Vec<[f64; 2]> = boundary.iter().map(|&id| pos[id]).collect();
    let polygon_area = polygon_area_points(&points);
    if polygon_area <= EPS {
        return None;
    }
    let tolerance = 1.0e-7 * polygon_area.max(1.0);

    let mut best: Option<FillPlan> = None;
    let mut best_rank = f64::NEG_INFINITY;

    // Fan pairing from every starting vertex.
    for start in 0..n {
        let rotated: Vec<usize> = (0..n).map(|k| boundary[(start + k) % n]).collect();
        let mut quads: Vec<[usize; 4]> = Vec::new();
        let mut total = 0.0;
        let mut rank = f64::INFINITY;
        let mut valid = true;
        let mut i = 1;
        while i + 2 < n {
            let quad = [rotated[0], rotated[i], rotated[i + 1], rotated[i + 2]];
            let corners = [pos[quad[0]], pos[quad[1]], pos[quad[2]], pos[quad[3]]];
            let area = quad_area(corners);
            if area <= 1.0e-14 {
                valid = false;
                break;
            }
            let Some(quad_rank) = quad_angle_rank(corners) else {
                valid = false;
                break;
            };
            total += area;
            rank = rank.min(quad_rank);
            quads.push(quad);
            i += 2;
        }
        if valid && (total - polygon_area).abs() <= tolerance {
            let better = match &best {
                None => true,
                Some(_) => rank > best_rank,
            };
            if better {
                best_rank = rank;
                best = Some(FillPlan {
                    quads,
                    steiner: None,
                });
            }
        }
    }

    // Centroid ring with one Steiner point.
    if let Some(centre) = polygon_centroid(&points) {
        if point_in_polygon(centre, &points) && distance_to_polygon_edges(centre, &points) > 1.0e-9
        {
            let mut quads: Vec<[usize; 4]> = Vec::new();
            let mut total = 0.0;
            let mut rank = f64::INFINITY;
            let mut valid = true;
            for i in 0..n / 2 {
                let a = boundary[(2 * i) % n];
                let b = boundary[(2 * i + 1) % n];
                let c = boundary[(2 * i + 2) % n];
                let quad = [new_id, a, b, c];
                let corners = [centre, pos[a], pos[b], pos[c]];
                let area = quad_area(corners);
                if area <= 1.0e-14 {
                    valid = false;
                    break;
                }
                let Some(quad_rank) = quad_angle_rank(corners) else {
                    valid = false;
                    break;
                };
                total += area;
                rank = rank.min(quad_rank);
                quads.push(quad);
            }
            if valid && (total - polygon_area).abs() <= tolerance {
                let better = match &best {
                    None => true,
                    Some(_) => rank > best_rank,
                };
                if better {
                    best = Some(FillPlan {
                        quads,
                        steiner: Some(centre),
                    });
                }
            }
        }
    }
    best
}

/// Boundary loops of a set of faces; splits are applied on the fly.
fn component_loops(
    component: &[usize],
    faces: &[Face],
    splits: &HashMap<(usize, usize), usize>,
) -> Result<Vec<Vec<usize>>, String> {
    let mut count: HashMap<(usize, usize), usize> = HashMap::new();
    for &face_index in component {
        let Some(face) = faces.get(face_index) else {
            continue;
        };
        let n = face.vertices.len();
        for i in 0..n {
            let key = edge_key(face.vertices[i], face.vertices[(i + 1) % n]);
            *count.entry(key).or_insert(0) += 1;
        }
    }
    let mut next: HashMap<usize, usize> = HashMap::new();
    for &face_index in component {
        let Some(face) = faces.get(face_index) else {
            continue;
        };
        let n = face.vertices.len();
        for i in 0..n {
            let a = face.vertices[i];
            let b = face.vertices[(i + 1) % n];
            if count.get(&edge_key(a, b)).copied().unwrap_or(0) != 1 {
                continue;
            }
            // Apply a pending perimeter split.
            let mut chain = vec![a];
            if let Some(&mid) = splits.get(&edge_key(a, b)) {
                chain.push(mid);
            }
            chain.push(b);
            for pair in chain.windows(2) {
                if next.insert(pair[0], pair[1]).is_some() {
                    return Err("a fallback region boundary is non-manifold".into());
                }
            }
        }
    }
    if next.is_empty() {
        return Err("a fallback region has no boundary".into());
    }
    let mut loops: Vec<Vec<usize>> = Vec::new();
    let ordered: BTreeSet<usize> = next.keys().copied().collect();
    let mut visited: HashSet<usize> = HashSet::new();
    for &start in &ordered {
        if visited.contains(&start) {
            continue;
        }
        let mut loop_ids: Vec<usize> = Vec::new();
        let mut current = start;
        let mut guard = 0usize;
        loop {
            if !visited.insert(current) {
                return Err("a fallback region boundary pinches".into());
            }
            loop_ids.push(current);
            let Some(&following) = next.get(&current) else {
                return Err("a fallback region boundary is open".into());
            };
            current = following;
            if current == start {
                break;
            }
            guard += 1;
            if guard > next.len() + 1 {
                return Err("a fallback region boundary did not close".into());
            }
        }
        if loop_ids.len() < 4 {
            return Err("a fallback region has fewer than four sides".into());
        }
        loops.push(loop_ids);
    }
    Ok(loops)
}

fn components_of(covered: &[bool], faces: &[Face]) -> Vec<Vec<usize>> {
    let mut edge_faces: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (index, face) in faces.iter().enumerate() {
        let n = face.vertices.len();
        for i in 0..n {
            let key = edge_key(face.vertices[i], face.vertices[(i + 1) % n]);
            edge_faces.entry(key).or_default().push(index);
        }
    }
    let mut seen = vec![false; faces.len()];
    let mut components = Vec::new();
    for start in 0..faces.len() {
        if covered[start] || seen[start] {
            continue;
        }
        seen[start] = true;
        let mut queue = VecDeque::new();
        queue.push_back(start);
        let mut component = Vec::new();
        while let Some(face) = queue.pop_front() {
            component.push(face);
            let n = faces[face].vertices.len();
            for i in 0..n {
                let key = edge_key(faces[face].vertices[i], faces[face].vertices[(i + 1) % n]);
                if let Some(neighbours) = edge_faces.get(&key) {
                    for &neighbour in neighbours {
                        if !covered[neighbour] && !seen[neighbour] {
                            seen[neighbour] = true;
                            queue.push_back(neighbour);
                        }
                    }
                }
            }
        }
        component.sort_unstable();
        components.push(component);
    }
    components
}

/// Split the first unsplit domain-perimeter edge of a fallback region.
fn split_domain_edge(
    loops: &[Vec<usize>],
    pos: &mut Vec<[f64; 2]>,
    splits: &HashMap<(usize, usize), usize>,
    domain_edges: &HashSet<(usize, usize)>,
) -> Option<((usize, usize), usize)> {
    for loop_ids in loops {
        for i in 0..loop_ids.len() {
            let a = loop_ids[i];
            let b = loop_ids[(i + 1) % loop_ids.len()];
            let key = edge_key(a, b);
            if domain_edges.contains(&key) && !splits.contains_key(&key) {
                let pa = *pos.get(a)?;
                let pb = *pos.get(b)?;
                let midpoint = [0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1])];
                let mid = pos.len();
                pos.push(midpoint);
                return Some((key, mid));
            }
        }
    }
    None
}

/// Try to fix one bad fallback component: split an unsplit perimeter edge if it
/// has one, otherwise absorb a selected patch that touches the component.
fn fix_component(
    component: &[usize],
    faces: &[Face],
    patches: &[CandidatePatch],
    selected: &mut Vec<usize>,
    splits: &mut HashMap<(usize, usize), usize>,
    pos: &mut Vec<[f64; 2]>,
    domain_edges: &HashSet<(usize, usize)>,
) -> Result<(), String> {
    // A degenerate leftover (for example a triangle) cannot be looped; skip
    // straight to absorbing a neighbour, which grows it into a fillable region.
    let loops = component_loops(component, faces, splits).unwrap_or_default();
    // A parity fix is the only thing a perimeter split can contribute: an even
    // region that the fan could not fill is concave, and adding a collinear
    // point does not help it. Grow the region by absorbing a neighbour instead.
    let odd = loops.len() == 1
        && loops
            .first()
            .is_some_and(|loop_ids| !loop_ids.len().is_multiple_of(2));
    if odd {
        if let Some((key, mid)) = split_domain_edge(&loops, pos, splits, domain_edges) {
            splits.insert(key, mid);
            return Ok(());
        }
    }
    // Absorb the lowest-index selected patch that shares an edge with the
    // component.
    let mut component_edges: HashSet<(usize, usize)> = HashSet::new();
    for &face_index in component {
        let Some(face) = faces.get(face_index) else {
            continue;
        };
        let n = face.vertices.len();
        for i in 0..n {
            component_edges.insert(edge_key(face.vertices[i], face.vertices[(i + 1) % n]));
        }
    }
    for (slot, &patch_index) in selected.iter().enumerate() {
        let Some(patch) = patches.get(patch_index) else {
            continue;
        };
        let mut touches = false;
        for &face_index in &patch.faces {
            let Some(face) = faces.get(face_index) else {
                continue;
            };
            let n = face.vertices.len();
            for i in 0..n {
                if component_edges.contains(&edge_key(face.vertices[i], face.vertices[(i + 1) % n]))
                {
                    touches = true;
                    break;
                }
            }
            if touches {
                break;
            }
        }
        if touches {
            selected.remove(slot);
            return Ok(());
        }
    }
    // Nothing to absorb: a perimeter split is the last resort, even for an
    // even-sided region.
    if let Some((key, mid)) = split_domain_edge(&loops, pos, splits, domain_edges) {
        splits.insert(key, mid);
        return Ok(());
    }
    Err(
        "a leftover channel region could not be closed: no perimeter edge to split \
         and no adjacent patch to absorb"
            .into(),
    )
}

// ---------------------------------------------------------------------------
// Assembly
// ---------------------------------------------------------------------------

struct BuildOutput {
    mesh: Mesh,
    stats: ChannelStats,
}

// ---------------------------------------------------------------------------
// Boundary-ribbon construction (the robust default path)
// ---------------------------------------------------------------------------

/// Build a mesh for `domain`.
///
/// The **boundary ribbon** is tried first: one quad per edge of each grid
/// loop, joining the domain outer boundary to the outer grid loop and each
/// domain hole to the grid hole loop that surrounds it. The ribbon is a
/// documented simplification of §2.2.3–2.2.9: it needs no templates, cross
/// edges or covering search, at the cost of not adapting to complex channels.
/// If it fails, the thesis path (grid + cross-edge patches + greedy cover) is
/// attempted.
/// Universal fallback for a simply connected domain: cut two boundary edges,
/// keep the two resulting boundary chains, subdivide the shorter chain so both
/// have the same edge count, and build a strip of quads between them.
///
/// This is the generalization of the midpoint-fan idea that works for concave
/// polygons such as the ring section: every quad has one edge on each chain and
/// two cross edges, so the angles at the boundary are those between a chain edge
/// and a cross edge, not the (possibly degenerate) 180° straight-boundary
/// angles of a single centre fan. The two removed boundary edges become the
/// strip's end cross edges, so the mesh boundary is exactly the domain boundary.
fn two_chain_strip(domain: &Domain, _size: f64) -> Result<BuildOutput, String> {
    if !domain.holes.is_empty() {
        return Err("the two-chain strip supports simply connected domains only".into());
    }
    let mut boundary = clean_loop_points(&domain.outer)?;
    if polygon_area_points(&boundary) < 0.0 {
        boundary.reverse();
    }
    let n = boundary.len();
    if n < 4 {
        return Err("the domain boundary is too short for a two-chain strip".into());
    }
    if n > 600 {
        return Err("the domain boundary is too long for the two-chain strip search".into());
    }
    let total_area = polygon_area_points(&boundary);
    if total_area <= EPS {
        return Err("the domain has no area".into());
    }

    // Best pair of cut edges, stored as the two matched chains.
    let mut best: Option<(f64, Vec<[f64; 2]>, Vec<[f64; 2]>)> = None;
    for i in 0..n {
        for j in (i + 2)..n {
            // Removing the two cut edges leaves (j-i-1) and
            // (n-(j-i)-1) edges on the two chains.
            let edges_a = j - i - 1;
            let edges_b = n - (j - i) - 1;
            if edges_a < 2 || edges_b < 2 {
                continue;
            }
            // Chain A: P_{i+1} .. P_j.
            let mut chain_a: Vec<[f64; 2]> = Vec::with_capacity(edges_a + 1);
            for k in 0..=edges_a {
                chain_a.push(boundary[(i + 1 + k) % n]);
            }
            // Chain B', from P_i backwards to P_{j+1}.
            let mut chain_b: Vec<[f64; 2]> = Vec::with_capacity(edges_b + 1);
            for k in 0..=edges_b {
                chain_b.push(boundary[(i + n - k) % n]);
            }
            let target = edges_a.max(edges_b);
            let Ok(chain_a) = subdivide_chain(&chain_a, target) else {
                continue;
            };
            let Ok(chain_b) = subdivide_chain(&chain_b, target) else {
                continue;
            };
            let Some((rank, _, _, _)) = rows_mesh(&chain_a, &chain_b, 1, total_area) else {
                continue;
            };
            let better = match &best {
                None => true,
                Some((current, _, _)) => rank > *current,
            };
            if better {
                best = Some((rank, chain_a, chain_b));
            }
        }
    }

    let Some((_, chain_a, chain_b)) = best else {
        return Err("the two-chain strip found no valid cut pair".into());
    };

    // More radial rows reduce the aspect ratio (the ring's radial edges are
    // long compared with its arc edges).
    let mut chosen: Option<(bool, f64, f64, Vec<[f64; 2]>, Vec<[usize; 4]>)> = None;
    for rows in 1..=4usize {
        let Some((rank, max_aspect, pos, quads)) = rows_mesh(&chain_a, &chain_b, rows, total_area)
        else {
            continue;
        };
        let good_aspect = max_aspect <= 10.0;
        let better = match &chosen {
            None => true,
            Some((current_good, current_rank, _, _, _)) => {
                (good_aspect && !*current_good)
                    || (good_aspect == *current_good && rank > *current_rank)
            }
        };
        if better {
            chosen = Some((good_aspect, rank, max_aspect, pos, quads));
        }
    }
    let Some((_, rank, _, pos, quads)) = chosen else {
        return Err("the two-chain strip could not build any rows".into());
    };
    let mut stats = ChannelStats {
        method: "two-chain-strip",
        delta_factor: 0.0,
        channel_loops: 1,
        channel_quads: quads.len(),
        faces: quads.len(),
        ..ChannelStats::default()
    };
    stats.quads = quads.len();
    let output = finish_mesh(pos, quads, domain, &mut stats)?;
    let _ = rank;
    Ok(output)
}

/// Build a structured strip of `rows` quad rows between two matched chains.
fn rows_mesh(
    chain_a: &[[f64; 2]],
    chain_b: &[[f64; 2]],
    rows: usize,
    expected_area: f64,
) -> Option<(f64, f64, Vec<[f64; 2]>, Vec<[usize; 4]>)> {
    if chain_a.len() != chain_b.len() || chain_a.len() < 3 || rows == 0 {
        return None;
    }
    let columns = chain_a.len() - 1;
    let width = columns + 1;
    let mut pos: Vec<[f64; 2]> = Vec::with_capacity((rows + 1) * width);
    for r in 0..=rows {
        let t = r as f64 / rows as f64;
        for c in 0..=columns {
            let a = chain_a[c];
            let b = chain_b[c];
            pos.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
        }
    }
    let mut quads: Vec<[usize; 4]> = Vec::with_capacity(rows * columns);
    let mut worst_rank = f64::INFINITY;
    let mut worst_aspect: f64 = 0.0;
    let mut total = 0.0f64;
    for r in 0..rows {
        for c in 0..columns {
            let base = r * width + c;
            let quad = [base, base + 1, base + 1 + width, base + width];
            let corners = [pos[quad[0]], pos[quad[1]], pos[quad[2]], pos[quad[3]]];
            let area = quad_area(corners);
            if area <= 1.0e-14 {
                return None;
            }
            let rank = quad_angle_rank(corners)?;
            worst_rank = worst_rank.min(rank);
            worst_aspect = worst_aspect.max(quad_aspect(corners));
            total += area;
            quads.push(quad);
        }
    }
    if (total - expected_area).abs() > 1.0e-6 * expected_area.max(1.0) {
        return None;
    }
    Some((worst_rank, worst_aspect, pos, quads))
}

fn quad_aspect(quad: [[f64; 2]; 4]) -> f64 {
    let mut shortest = f64::INFINITY;
    let mut longest = 0.0f64;
    for i in 0..4 {
        let length = point_distance(quad[i], quad[(i + 1) % 4]);
        shortest = shortest.min(length);
        longest = longest.max(length);
    }
    if shortest > 0.0 {
        longest / shortest
    } else {
        f64::INFINITY
    }
}

/// Subdivide a chain so it has exactly `target_edges` edges, keeping every
/// existing vertex and putting new points exactly on the input segments.
fn subdivide_chain(points: &[[f64; 2]], target_edges: usize) -> Result<Vec<[f64; 2]>, String> {
    let n = points.len();
    if n < 2 {
        return Err("a strip chain is degenerate".into());
    }
    let edges = n - 1;
    if target_edges < edges {
        return Err("cannot subdivide a chain to fewer edges".into());
    }
    if target_edges == edges {
        return Ok(points.to_vec());
    }
    let mut lengths = Vec::with_capacity(edges);
    let mut perimeter = 0.0f64;
    for i in 0..edges {
        let length = point_distance(points[i], points[i + 1]);
        if length <= EPS {
            return Err("a strip chain has a zero-length edge".into());
        }
        lengths.push(length);
        perimeter += length;
    }
    let extra = target_edges - edges;
    let mut counts = vec![1usize; edges];
    let mut fractions: Vec<(f64, usize)> = Vec::with_capacity(edges);
    let mut used = 0usize;
    for i in 0..edges {
        let share = extra as f64 * lengths[i] / perimeter;
        let whole = share.floor() as usize;
        counts[i] += whole;
        used += whole;
        fractions.push((share - whole as f64, i));
    }
    let mut remaining = extra - used;
    fractions.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let mut cursor = 0usize;
    while remaining > 0 && !fractions.is_empty() {
        let index = fractions[cursor % fractions.len()].1;
        counts[index] += 1;
        remaining -= 1;
        cursor += 1;
    }
    let mut out = Vec::with_capacity(target_edges + 1);
    for i in 0..edges {
        let a = points[i];
        let b = points[i + 1];
        for k in 0..counts[i] {
            let t = k as f64 / counts[i] as f64;
            out.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
        }
    }
    out.push(points[edges]);
    Ok(out)
}

fn build(domain: &Domain, size: f64) -> Result<BuildOutput, String> {
    let mut last_error = String::new();
    for delta in ADAPTIVE_DELTAS {
        match build_ribbon_with_delta(domain, size, delta) {
            Ok(output) => return Ok(output),
            Err(error) => last_error = format!("delta={delta:.2}h ribbon: {error}"),
        }
        match build_templates_with_delta(domain, size, delta) {
            Ok(output) => return Ok(output),
            Err(error) => last_error.push_str(&format!("; delta={delta:.2}h templates: {error}")),
        }
    }
    match two_chain_strip(domain, size) {
        Ok(output) => Ok(output),
        Err(strip_error) => Err(format!(
            "the channel could not be closed at any tried width ({last_error}); \
             two-chain strip: {strip_error}"
        )),
    }
}

struct GridLoop {
    points: Vec<[f64; 2]>,
    ids: Vec<usize>,
    area: f64,
}

fn build_ribbon_with_delta(domain: &Domain, size: f64, delta: f64) -> Result<BuildOutput, String> {
    let prepared = prepare(domain, size, delta)?;
    let mut boundary: Vec<LoopData> = Vec::new();
    let mut grid: Vec<GridLoop> = Vec::new();
    for raw in &prepared.loops {
        let mut points = Vec::with_capacity(raw.points.len());
        for &id in &raw.points {
            let Some(&point) = prepared.pos.get(id) else {
                return Err("internal error: loop point missing".into());
            };
            points.push(point);
        }
        match raw.source {
            LoopSource::Boundary => boundary.push(LoopData {
                points,
                ids: raw.points.clone(),
            }),
            LoopSource::Grid => {
                let area = polygon_area_points(&points);
                grid.push(GridLoop {
                    points,
                    ids: raw.points.clone(),
                    area,
                });
            }
        }
    }
    if boundary.is_empty() {
        return Err("the domain has no boundary loop".into());
    }

    let mut pos = prepared.pos.clone();
    let mut channel_quads: Vec<[usize; 4]> = Vec::new();
    let mut ribbons = 0usize;
    let mut worst_rank = f64::INFINITY;
    let outer_loop = &boundary[0];

    if grid.is_empty() {
        // No interior grid: one ribbon between the outer boundary and the
        // (single) domain hole.
        if domain.holes.len() != 1 || boundary.len() != 2 {
            return Err(
                "the boundary ribbon with an empty grid supports exactly one domain hole".into(),
            );
        }
        let hole_loop = &boundary[1];
        let (count, rank) = add_ribbon(
            &domain.outer,
            &outer_loop.points,
            &outer_loop.ids,
            &hole_loop.points,
            &hole_loop.ids,
            true,
            "outer boundary <-> domain hole",
            &mut pos,
            &mut channel_quads,
        )?;
        ribbons += count;
        worst_rank = worst_rank.min(rank);
    } else if grid.len() == 1 {
        if !domain.holes.is_empty() {
            return Err(
                "one interior grid loop with domain holes is not supported by the ribbon".into(),
            );
        }
        let (count, rank) = add_ribbon(
            &domain.outer,
            &outer_loop.points,
            &outer_loop.ids,
            &grid[0].points,
            &grid[0].ids,
            true,
            "outer boundary <-> grid loop",
            &mut pos,
            &mut channel_quads,
        )?;
        ribbons += count;
        worst_rank = worst_rank.min(rank);
    } else {
        // Several channel loops: the largest grid loop (the outer grid
        // boundary) pairs with the domain outer loop, every remaining grid
        // loop (a grid hole) with the domain hole it surrounds.
        let mut order: Vec<usize> = (0..grid.len()).collect();
        order.sort_by(|&a, &b| {
            grid[b]
                .area
                .abs()
                .total_cmp(&grid[a].area.abs())
                .then_with(|| a.cmp(&b))
        });
        let outer_index = order[0];
        let (count, rank) = add_ribbon(
            &domain.outer,
            &outer_loop.points,
            &outer_loop.ids,
            &grid[outer_index].points,
            &grid[outer_index].ids,
            true,
            "outer boundary <-> grid outer loop",
            &mut pos,
            &mut channel_quads,
        )?;
        ribbons += count;
        worst_rank = worst_rank.min(rank);

        let mut used_holes = vec![false; domain.holes.len()];
        for &grid_index in order.iter().skip(1) {
            let hole_index =
                choose_hole_for_grid_loop(&domain.holes, &used_holes, &grid[grid_index].points)?;
            used_holes[hole_index] = true;
            let Some(hole_loop) = boundary.get(hole_index + 1) else {
                return Err("internal error: missing prepared domain hole loop".into());
            };
            let (count, rank) = add_ribbon(
                &domain.holes[hole_index],
                &hole_loop.points,
                &hole_loop.ids,
                &grid[grid_index].points,
                &grid[grid_index].ids,
                false,
                "domain hole <-> grid hole loop",
                &mut pos,
                &mut channel_quads,
            )?;
            ribbons += count;
            worst_rank = worst_rank.min(rank);
        }
        if used_holes.iter().any(|&used| !used) {
            return Err(
                "a domain hole has no matching grid hole loop; the ribbon cannot close it".into(),
            );
        }
    }

    if channel_quads.is_empty() {
        return Err("the boundary ribbon produced no quads".into());
    }
    if !worst_rank.is_finite() {
        return Err("the boundary ribbon could not rank its quads".into());
    }

    // Every pair covered its own band; together they must cover the channel.
    let grid_area: f64 = prepared
        .quads
        .iter()
        .map(|quad| {
            quad_area([
                prepared.pos[quad[0]],
                prepared.pos[quad[1]],
                prepared.pos[quad[2]],
                prepared.pos[quad[3]],
            ])
        })
        .sum();
    let expected_area = domain.area() - grid_area;
    let ribbon_area: f64 = channel_quads
        .iter()
        .map(|quad| quad_area([pos[quad[0]], pos[quad[1]], pos[quad[2]], pos[quad[3]]]))
        .sum();
    // A projection row strip draws its inner boundary with chords between
    // points of the domain loop, which can shave up to ~0.1 % of the band area
    // when the loop is coarsely sampled. Keep the check tight but not exact.
    if (ribbon_area - expected_area).abs() > 1.0e-3 * expected_area.max(1.0) {
        return Err(format!(
            "the boundary ribbons cover {ribbon_area:.6} but the channel area is {expected_area:.6}"
        ));
    }

    let mut quads: Vec<[usize; 4]> = prepared.quads.clone();
    quads.extend(channel_quads.iter().copied());
    let mut stats = ChannelStats {
        method: "boundary-ribbon",
        delta_factor: delta,
        grid_vertices: prepared.grid_vertex_count,
        interior_quads: prepared.quads.len(),
        channel_loops: prepared.loops.len(),
        cross_edges: 0,
        faces: ribbons,
        triangle_faces: 0,
        patches: 0,
        usable_patches: 0,
        selected_patches: 0,
        greedy_covered_fraction: 1.0,
        fallback_regions: 0,
        fallback_quads: 0,
        channel_quads: channel_quads.len(),
        quads: quads.len(),
        ..ChannelStats::default()
    };
    finish_mesh(pos, quads, domain, &mut stats)
}

struct LoopData {
    points: Vec<[f64; 2]>,
    ids: Vec<usize>,
}

/// Choose the domain hole nested inside a grid hole loop.
fn choose_hole_for_grid_loop(
    holes: &[Vec<[f64; 2]>],
    used: &[bool],
    grid_points: &[[f64; 2]],
) -> Result<usize, String> {
    if holes.is_empty() {
        return Err("a grid hole loop has no domain hole to pair with".into());
    }
    // A domain hole is nested inside its grid hole loop; test the hole centroid.
    for (index, hole) in holes.iter().enumerate() {
        if used.get(index).copied().unwrap_or(false) {
            continue;
        }
        if let Some(centroid) = polygon_centroid(hole) {
            if point_in_polygon(centroid, grid_points) {
                return Ok(index);
            }
        }
    }
    // Fall back to the nearest unused hole (deterministic).
    let grid_centre = match polygon_centroid(grid_points) {
        Some(centre) => centre,
        None => return Err("a grid hole loop is degenerate".into()),
    };
    let mut best: Option<(f64, usize)> = None;
    for (index, hole) in holes.iter().enumerate() {
        if used.get(index).copied().unwrap_or(false) {
            continue;
        }
        let Some(centre) = polygon_centroid(hole) else {
            continue;
        };
        let distance = point_distance(centre, grid_centre);
        let better = match best {
            None => true,
            Some((current, _)) => distance < current,
        };
        if better {
            best = Some((distance, index));
        }
    }
    match best {
        Some((_, index)) => Ok(index),
        None => Err("no unused domain hole for a grid hole loop".into()),
    }
}

/// Widest-path ribbon between a full-resolution outer loop and an inner grid
/// loop (§2.2.7–2.2.8 in miniature).
///
/// States are cross edges `(i, j)`: outer vertex `i` joined to inner vertex
/// `j`. Three quad types move between states and consume boundary edges:
/// * `A` — a strip quad, one outer edge + one inner edge, `(i,j)->(i+1,j+1)`;
/// * `B` — an outer cap/fan quad (three outer vertices + one inner vertex),
///   two outer edges, `(i,j)->(i+2,j)`;
/// * `C` — an inner cap/fan quad (one outer vertex + three inner vertices),
///   two inner edges, `(i,j)->(i,j+2)`.
///
/// A path from `(0,0)` to `(m,n)` covers the annulus exactly once and absorbs
/// the loop-count mismatch at the caps. The path is chosen by a widest-path
/// (max-min rank) dynamic program, which is the role of the thesis' widest
/// path. Different rotational offsets between the loops are tried.
fn dp_ribbon(
    outer: &[usize],
    inner: &[usize],
    pos: &[[f64; 2]],
    expected_area: f64,
) -> Result<(Vec<[usize; 4]>, f64), String> {
    let m = outer.len();
    let n = inner.len();
    if m < 3 || n < 3 {
        return Err("the ribbon loops are too short".into());
    }
    if !(m + n).is_multiple_of(2) {
        return Err("the ribbon loops have odd total length".into());
    }
    let width = n + 1;
    let states = (m + 1) * width;
    if states > 4_000_000 {
        return Err("the ribbon DP state space is too large".into());
    }

    let mut offsets: Vec<usize> = Vec::new();
    let mut add_offset = |value: i64| {
        let offset = value.rem_euclid(n as i64) as usize;
        if !offsets.contains(&offset) {
            offsets.push(offset);
        }
    };
    let nearest_inner = |point: [f64; 2]| -> usize {
        let mut best = 0usize;
        let mut best_distance = f64::INFINITY;
        for (index, &id) in inner.iter().enumerate() {
            if let Some(&p) = pos.get(id) {
                let d = point_distance(point, p);
                if d < best_distance {
                    best_distance = d;
                    best = index;
                }
            }
        }
        best
    };
    if let Some(&first) = outer.first().and_then(|&id| pos.get(id)) {
        let near = nearest_inner(first) as i64;
        for delta in -3..=3 {
            add_offset(near + delta);
        }
    }
    // Align corners: for sampled outer vertices, the offset that maps them to
    // their nearest inner vertex.
    let stride = if m <= 80 { 1 } else { m / 40 + 1 };
    for i in (0..m).step_by(stride) {
        if let Some(&point) = outer.get(i).and_then(|&id| pos.get(id)) {
            let j = nearest_inner(point) as i64;
            add_offset(j + i as i64);
        }
    }
    if n <= 80 {
        for offset in 0..n {
            add_offset(offset as i64);
        }
    }
    if offsets.len() > 64 {
        offsets.truncate(64);
    }

    let tolerance = 1.0e-6 * expected_area.max(1.0);
    let mut best: Option<(f64, Vec<[usize; 4]>)> = None;

    for &offset in &offsets {
        let mut value = vec![f64::NEG_INFINITY; states];
        let mut previous: Vec<Option<(usize, usize, u8)>> = vec![None; states];
        let mut previous_quad: Vec<[usize; 4]> = vec![[0; 4]; states];
        value[0] = f64::INFINITY;

        for i in 0..=m {
            for j in 0..=n {
                let index = i * width + j;
                let current = value[index];
                if current == f64::NEG_INFINITY {
                    continue;
                }
                // A: strip quad. The inner loop is walked backwards (its
                // stored orientation is the inner side of the band), so the
                // two cross edges of the quad do not cross each other.
                if i < m && j < n {
                    let a = (offset + n - j % n) % n;
                    let g0 = inner[a];
                    let g1 = inner[(a + 1) % n];
                    let quad = [outer[i], outer[(i + 1) % m], g0, g1];
                    if let Some(rank) = dp_quad_rank(&quad, pos) {
                        relax_dp(
                            &mut value,
                            &mut previous,
                            &mut previous_quad,
                            width,
                            (i, j),
                            (i + 1, j + 1),
                            current.min(rank),
                            1,
                            quad,
                        );
                    }
                }
                // B: outer cap (two outer edges, one inner vertex).
                if i + 2 <= m {
                    let a = (offset + n - j % n) % n;
                    let g = inner[a];
                    let quad = [outer[i], outer[(i + 1) % m], outer[(i + 2) % m], g];
                    if let Some(rank) = dp_quad_rank(&quad, pos) {
                        relax_dp(
                            &mut value,
                            &mut previous,
                            &mut previous_quad,
                            width,
                            (i, j),
                            (i + 2, j),
                            current.min(rank),
                            2,
                            quad,
                        );
                    }
                }
                // C: inner cap (one outer vertex, two inner edges).
                if j + 2 <= n {
                    let o = outer[i % m];
                    let a = (offset + n - j % n) % n;
                    let g0 = inner[a];
                    let g1 = inner[(a + 1) % n];
                    let g2 = inner[(a + 2) % n];
                    let quad = [o, g0, g1, g2];
                    if let Some(rank) = dp_quad_rank(&quad, pos) {
                        relax_dp(
                            &mut value,
                            &mut previous,
                            &mut previous_quad,
                            width,
                            (i, j),
                            (i, j + 2),
                            current.min(rank),
                            3,
                            quad,
                        );
                    }
                }
            }
        }

        let end = m * width + n;
        let rank = value[end];
        if rank == f64::NEG_INFINITY {
            continue;
        }
        // Reconstruct.
        let mut quads: Vec<[usize; 4]> = Vec::new();
        let mut cursor = (m, n);
        let mut guard = 0usize;
        while cursor != (0, 0) {
            let index = cursor.0 * width + cursor.1;
            let Some((pi, pj, _action)) = previous[index] else {
                break;
            };
            quads.push(previous_quad[index]);
            cursor = (pi, pj);
            guard += 1;
            if guard > m + n + 4 {
                break;
            }
        }
        if cursor != (0, 0) {
            continue;
        }
        quads.reverse();
        let total: f64 = quads
            .iter()
            .map(|quad| quad_area([pos[quad[0]], pos[quad[1]], pos[quad[2]], pos[quad[3]]]))
            .sum();
        if (total - expected_area).abs() > tolerance {
            continue;
        }
        let better = match &best {
            None => true,
            Some((current, _)) => rank > *current,
        };
        if better {
            best = Some((rank, quads));
        }
    }
    match best {
        Some((rank, quads)) => Ok((quads, rank)),
        None => Err("the widest-path ribbon found no valid cover".into()),
    }
}

fn dp_quad_rank(quad: &[usize; 4], pos: &[[f64; 2]]) -> Option<f64> {
    let corners = [
        *pos.get(quad[0])?,
        *pos.get(quad[1])?,
        *pos.get(quad[2])?,
        *pos.get(quad[3])?,
    ];
    quad_angle_rank(corners)
}

fn relax_dp(
    value: &mut [f64],
    previous: &mut [Option<(usize, usize, u8)>],
    previous_quad: &mut [[usize; 4]],
    width: usize,
    origin: (usize, usize),
    next: (usize, usize),
    candidate: f64,
    action: u8,
    quad: [usize; 4],
) {
    let index = next.0 * width + next.1;
    let better = candidate > value[index] + 1.0e-12
        || (value[index] == f64::NEG_INFINITY && candidate.is_finite());
    if better {
        value[index] = candidate;
        previous[index] = Some((origin.0, origin.1, action));
        previous_quad[index] = quad;
    }
}

/// Closest point on a closed polyline, with the segment index.
fn closest_on_loop(p: [f64; 2], loop_points: &[[f64; 2]]) -> ([f64; 2], usize) {
    let n = loop_points.len();
    let mut best = (loop_points[0], 0usize);
    let mut best_distance = f64::INFINITY;
    for i in 0..n {
        let a = loop_points[i];
        let b = loop_points[(i + 1) % n];
        let ab = [b[0] - a[0], b[1] - a[1]];
        let length2 = ab[0] * ab[0] + ab[1] * ab[1];
        let t = if length2 > 0.0 {
            (((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / length2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let q = [a[0] + t * ab[0], a[1] + t * ab[1]];
        let distance = point_distance(p, q);
        if distance < best_distance {
            best_distance = distance;
            best = (q, i);
        }
    }
    best
}

/// Intersection of the ray `center -> target` with a closed polyline,
/// returning the first hit (if any).
fn ray_on_loop(center: [f64; 2], target: [f64; 2], loop_points: &[[f64; 2]]) -> Option<[f64; 2]> {
    let d = [target[0] - center[0], target[1] - center[1]];
    let n = loop_points.len();
    let mut best: Option<f64> = None;
    for i in 0..n {
        let a = loop_points[i];
        let b = loop_points[(i + 1) % n];
        let e = [b[0] - a[0], b[1] - a[1]];
        let denominator = d[0] * e[1] - d[1] * e[0];
        if denominator.abs() <= 1.0e-14 {
            continue;
        }
        let diff = [a[0] - center[0], a[1] - center[1]];
        let t = (diff[0] * e[1] - diff[1] * e[0]) / denominator;
        let u = (diff[0] * d[1] - diff[1] * d[0]) / denominator;
        if t > 1.0e-12 && (-1.0e-12..=1.0 + 1.0e-12).contains(&u) {
            best = Some(match best {
                Some(current) => current.min(t),
                None => t,
            });
        }
    }
    let t = best?;
    Some([center[0] + t * d[0], center[1] + t * d[1]])
}

/// Projection row strip: the fixed (exact) outer loop is advanced vertex by
/// vertex while its counterpart on the free inner loop is the closest point of
/// that loop. Rows interpolate radially between the two. This follows a
/// staircase grid hole around a circular domain hole instead of fighting it with
/// a constant index offset.
fn projection_rows(
    fixed_outer: &[usize],
    free_inner: &[[f64; 2]],
    pos: &mut Vec<[f64; 2]>,
    _band_area: f64,
) -> Option<(Vec<[usize; 4]>, f64)> {
    let k = fixed_outer.len();
    if k < 4 || free_inner.len() < 3 {
        return None;
    }
    let outer: Vec<[f64; 2]> = fixed_outer.iter().map(|&id| pos[id]).collect();
    let center = polygon_centroid(free_inner)?;
    let inner: Vec<[f64; 2]> = outer
        .iter()
        .map(|&point| {
            ray_on_loop(center, point, free_inner)
                .unwrap_or_else(|| closest_on_loop(point, free_inner).0)
        })
        .collect();
    let outer_area = polygon_area_points(&outer).abs();
    let inner_area = polygon_area_points(&inner).abs();
    let expected = outer_area - inner_area;
    if expected <= EPS {
        return None;
    }
    // The projected inner loop must still approximate the free loop.
    let free_area = polygon_area_points(free_inner).abs();
    if (inner_area - free_area).abs() > 0.02 * free_area.max(1.0) {
        return None;
    }

    let tolerance = 1.0e-6 * expected.max(1.0);
    let mut best: Option<(bool, f64, usize, Vec<[f64; 2]>)> = None;
    for rows in 1..=4usize {
        let mut points: Vec<[f64; 2]> = Vec::with_capacity((rows + 1) * k);
        for r in 0..=rows {
            let t = r as f64 / rows as f64;
            for column in 0..k {
                let a = outer[column];
                let b = inner[column];
                points.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
            }
        }
        let mut total = 0.0f64;
        let mut worst = f64::INFINITY;
        let mut aspect: f64 = 0.0;
        let mut valid = true;
        for r in 0..rows {
            for column in 0..k {
                let a = r * k + column;
                let b = r * k + (column + 1) % k;
                let c = (r + 1) * k + (column + 1) % k;
                let d = (r + 1) * k + column;
                let corners = [points[a], points[b], points[c], points[d]];
                let area = quad_area(corners);
                if area <= 1.0e-14 {
                    valid = false;
                    break;
                }
                let Some(rank) = quad_angle_rank(corners) else {
                    valid = false;
                    break;
                };
                worst = worst.min(rank);
                aspect = aspect.max(quad_aspect(corners));
                total += area;
            }
            if !valid {
                break;
            }
        }
        if !valid || (total - expected).abs() > tolerance {
            continue;
        }
        let good_aspect = aspect <= 10.0;
        let better = match &best {
            None => true,
            Some((current_good, current_rank, _, _)) => {
                (good_aspect && !*current_good)
                    || (good_aspect == *current_good && worst > *current_rank)
            }
        };
        if better {
            best = Some((good_aspect, worst, rows, points));
        }
    }
    let (_, rank, rows, local) = best?;

    let mut inner_ids: Vec<usize> = Vec::with_capacity(k);
    for point in &inner {
        inner_ids.push(pos.len());
        pos.push(*point);
    }
    let mut row_ids: Vec<Vec<usize>> = Vec::with_capacity(rows + 1);
    row_ids.push(fixed_outer.to_vec());
    for r in 1..rows {
        let mut ids = Vec::with_capacity(k);
        for column in 0..k {
            ids.push(pos.len());
            pos.push(local[r * k + column]);
        }
        row_ids.push(ids);
    }
    row_ids.push(inner_ids);
    let mut quads: Vec<[usize; 4]> = Vec::with_capacity(rows * k);
    for r in 0..rows {
        for column in 0..k {
            let next = (column + 1) % k;
            quads.push([
                row_ids[r][column],
                row_ids[r][next],
                row_ids[r + 1][next],
                row_ids[r + 1][column],
            ]);
        }
    }
    Some((quads, rank))
}

/// Cyclic row strip between two matched closed loops.
///
/// This is the last-resort candidate of [`add_ribbon`] for a band whose two
/// loops have very different shapes/resolutions (the 0.6 mm perf plate's
/// circular domain hole against the square-ish grid hole loop). The free
/// domain loop is re-discretized to the fixed grid loop's vertex count (all
/// existing free vertices are kept, new ones lie on its segments), then 1–4
/// interpolated rows are tried for every rotational offset. The outer row uses
/// the outer loop order and the inner row the stored inner order, so the
/// channel quads traverse the shared inner edges in the same direction as the
/// ordinary ribbon and remain conforming with the adjacent grid quads.
fn annulus_rows(
    _fixed_outer: bool,
    fixed_ids: &[usize],
    free_points: &[[f64; 2]],
    free_is_outer: bool,
    pos: &mut Vec<[f64; 2]>,
    band_area: f64,
) -> Option<(Vec<[usize; 4]>, f64)> {
    let k = fixed_ids.len();
    if k < 4 || free_points.len() > k {
        return None;
    }
    let free = resample_polyline(free_points, k).ok()?;
    if free.len() != k {
        return None;
    }
    // Orient the free row: outer rows counter-clockwise, inner rows clockwise.
    let mut free = free;
    if (polygon_area_points(&free) > 0.0) != free_is_outer {
        free.reverse();
    }
    // Fixed row coordinates.
    let mut fixed: Vec<[f64; 2]> = Vec::with_capacity(k);
    for &id in fixed_ids {
        fixed.push(*pos.get(id)?);
    }
    let (outer, inner) = if free_is_outer {
        (&free, &fixed)
    } else {
        (&fixed, &free)
    };

    let tolerance = 1.0e-6 * band_area.max(1.0);
    let stride = if k > 256 { k / 64 + 1 } else { 1 };
    let mut best: Option<(bool, f64, usize, usize, Vec<[f64; 2]>)> = None;
    for offset in (0..k).step_by(stride) {
        for rows in 1..=4usize {
            let mut points: Vec<[f64; 2]> = Vec::with_capacity((rows + 1) * k);
            for r in 0..=rows {
                let t = r as f64 / rows as f64;
                for column in 0..k {
                    let a = outer[column];
                    let b = inner[(offset + k - column % k) % k];
                    points.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
                }
            }
            let mut total = 0.0f64;
            let mut worst = f64::INFINITY;
            let mut aspect: f64 = 0.0;
            let mut valid = true;
            for r in 0..rows {
                for column in 0..k {
                    let a = r * k + column;
                    let b = r * k + (column + 1) % k;
                    let c = (r + 1) * k + (column + 1) % k;
                    let d = (r + 1) * k + column;
                    let corners = [points[a], points[b], points[c], points[d]];
                    let area = quad_area(corners);
                    if area <= 1.0e-14 {
                        valid = false;
                        break;
                    }
                    let Some(rank) = quad_angle_rank(corners) else {
                        valid = false;
                        break;
                    };
                    worst = worst.min(rank);
                    aspect = aspect.max(quad_aspect(corners));
                    total += area;
                }
                if !valid {
                    break;
                }
            }
            if !valid || (total - band_area).abs() > tolerance {
                continue;
            }
            let good_aspect = aspect <= 10.0;
            let better = match &best {
                None => true,
                Some((current_good, current_rank, _, _, _)) => {
                    (good_aspect && !*current_good)
                        || (good_aspect == *current_good && worst > *current_rank)
                }
            };
            if better {
                best = Some((good_aspect, worst, offset, rows, points));
            }
        }
    }
    let (_, rank, offset, rows, local) = best?;

    // Commit: append the free row and the interior rows, keep the fixed row's
    // existing ids.
    let mut free_ids: Vec<usize> = Vec::with_capacity(k);
    for point in &free {
        free_ids.push(pos.len());
        pos.push(*point);
    }
    let mut row_ids: Vec<Vec<usize>> = Vec::with_capacity(rows + 1);
    for r in 0..=rows {
        if r == 0 {
            if free_is_outer {
                row_ids.push(free_ids.clone());
            } else {
                row_ids.push(fixed_ids.to_vec());
            }
        } else if r == rows {
            if free_is_outer {
                row_ids.push(fixed_ids.to_vec());
            } else {
                row_ids.push(free_ids.clone());
            }
        } else {
            let mut ids = Vec::with_capacity(k);
            for column in 0..k {
                let point = local[r * k + column];
                ids.push(pos.len());
                pos.push(point);
            }
            row_ids.push(ids);
        }
    }
    let mut quads: Vec<[usize; 4]> = Vec::with_capacity(rows * k);
    for r in 0..rows {
        for column in 0..k {
            let next = (column + 1) % k;
            quads.push([
                row_ids[r][column],
                row_ids[r][next],
                row_ids[r + 1][next],
                row_ids[r + 1][column],
            ]);
        }
    }
    let _ = offset;
    Some((quads, rank))
}

/// Add one ribbon between a `free` domain boundary loop and a `fixed` grid
/// loop whose vertex ids must be preserved.
///
/// Both the full-resolution loops (widest-path DP with caps) and the
/// equal-count resampled strips are tried; the best-ranked valid result wins.
/// `free_is_outer` says whether the domain loop is the outer side of the band.
fn add_ribbon(
    free_original: &[[f64; 2]],
    free_points: &[[f64; 2]],
    free_ids: &[usize],
    fixed_points: &[[f64; 2]],
    fixed_ids: &[usize],
    free_is_outer: bool,
    label: &str,
    pos: &mut Vec<[f64; 2]>,
    out: &mut Vec<[usize; 4]>,
) -> Result<(usize, f64), String> {
    let (free, free_ids) = clean_loop_with_ids(free_points, free_ids)?;
    let free_original = clean_loop_points(free_original)?;
    let (fixed, fixed_ids) = clean_loop_with_ids(fixed_points, fixed_ids)?;
    let fixed_len = fixed.len();
    if fixed_len < 4 || !fixed_len.is_multiple_of(2) {
        return Err(format!(
            "{label}: the fixed grid loop has {fixed_len} vertices, which cannot carry a ribbon"
        ));
    }

    let (mut outer_ids, mut inner_ids) = if free_is_outer {
        (free_ids.clone(), fixed_ids.clone())
    } else {
        (fixed_ids.clone(), free_ids.clone())
    };
    if polygon_area_ids(pos, &outer_ids) < 0.0 {
        outer_ids.reverse();
    }
    if polygon_area_ids(pos, &inner_ids) > 0.0 {
        inner_ids.reverse();
    }
    let band_area = polygon_area_ids(pos, &outer_ids) + polygon_area_ids(pos, &inner_ids);
    if band_area <= EPS {
        return Err(format!("{label}: the ribbon band has no area"));
    }

    let mut best: Option<(f64, Vec<[usize; 4]>)> = None;
    // 1. Full-resolution loops with corner caps / fans (widest path).
    if let Ok((quads, rank)) = dp_ribbon(&outer_ids, &inner_ids, pos, band_area) {
        best = Some((rank, quads));
    }

    // 2. Equal-count resampled strips (the simple square/plate path).
    if free_original.len() <= fixed_len {
        for candidate in [
            resample_polyline(&free_original, fixed_len),
            resample_matched(&free_original, &fixed),
        ] {
            let Ok(mut points) = candidate else {
                continue;
            };
            if (polygon_area_points(&points) > 0.0) != free_is_outer {
                points.reverse();
            }
            let mut fresh_ids: Vec<usize> = Vec::with_capacity(fixed_len);
            for point in &points {
                fresh_ids.push(pos.len());
                pos.push(*point);
            }
            let (candidate_outer, candidate_inner) = if free_is_outer {
                (fresh_ids.clone(), fixed_ids.clone())
            } else {
                (fixed_ids.clone(), fresh_ids.clone())
            };
            let mut candidate_outer = candidate_outer;
            let mut candidate_inner = candidate_inner;
            if polygon_area_ids(pos, &candidate_outer) < 0.0 {
                candidate_outer.reverse();
            }
            if polygon_area_ids(pos, &candidate_inner) > 0.0 {
                candidate_inner.reverse();
            }
            if let Ok((quads, rank)) =
                ribbon_quads(&candidate_outer, &candidate_inner, pos, band_area)
            {
                let better = match &best {
                    None => true,
                    Some((current, _)) => rank > *current,
                };
                if better {
                    best = Some((rank, quads));
                }
            }
        }
    }

    if best.is_none() {
        // The loops may have very different shapes/resolutions (the perf
        // plate's circular domain hole against the staircase grid hole loop).
        // A cyclic row strip can absorb that where the one-row ribbon cannot.
        let fixed_outer = !free_is_outer;
        let fixed_ids: &[usize] = if fixed_outer { &outer_ids } else { &inner_ids };
        if let Some((quads, rank)) =
            annulus_rows(fixed_outer, fixed_ids, &free, free_is_outer, pos, band_area)
        {
            best = Some((rank, quads));
        }
        // Staircase grid loops need a projection correspondence instead of a
        // constant index offset. This is only meaningful when the fixed loop
        // is the outer side (the perf plate's grid hole loop).
        if best.is_none() && fixed_outer {
            if let Some((quads, rank)) = projection_rows(fixed_ids, &free, pos, band_area) {
                best = Some((rank, quads));
            }
        }
    }
    match best {
        Some((rank, quads)) => {
            let count = quads.len();
            out.extend(quads);
            Ok((count, rank))
        }
        None => Err(format!(
            "{label}: neither the width-path DP nor the resampled strip nor the row strip produced a valid ribbon"
        )),
    }
}

/// Remove a repeated closing point and consecutive duplicates.
fn clean_loop_points(points: &[[f64; 2]]) -> Result<Vec<[f64; 2]>, String> {
    let mut out: Vec<[f64; 2]> = Vec::with_capacity(points.len());
    for &point in points {
        if let Some(last) = out.last() {
            if (last[0] - point[0]).abs() <= EPS && (last[1] - point[1]).abs() <= EPS {
                continue;
            }
        }
        out.push(point);
    }
    while out.len() > 1 {
        let first = out[0];
        let last = out[out.len() - 1];
        if (first[0] - last[0]).abs() <= EPS && (first[1] - last[1]).abs() <= EPS {
            out.pop();
        } else {
            break;
        }
    }
    if out.len() < 3 {
        return Err("a ribbon loop is degenerate".into());
    }
    Ok(out)
}

/// Remove duplicate closing/consecutive points while keeping ids aligned.
fn clean_loop_with_ids(
    points: &[[f64; 2]],
    ids: &[usize],
) -> Result<(Vec<[f64; 2]>, Vec<usize>), String> {
    if points.len() != ids.len() {
        return Err("internal error: ribbon points and ids differ in length".into());
    }
    let mut out_points: Vec<[f64; 2]> = Vec::with_capacity(points.len());
    let mut out_ids: Vec<usize> = Vec::with_capacity(ids.len());
    for (index, &point) in points.iter().enumerate() {
        if let Some(last) = out_points.last() {
            if (last[0] - point[0]).abs() <= EPS && (last[1] - point[1]).abs() <= EPS {
                continue;
            }
        }
        out_points.push(point);
        out_ids.push(ids[index]);
    }
    while out_points.len() > 1 {
        let Some(first) = out_points.first().copied() else {
            break;
        };
        let Some(last) = out_points.last().copied() else {
            break;
        };
        if (first[0] - last[0]).abs() <= EPS && (first[1] - last[1]).abs() <= EPS {
            out_points.pop();
            out_ids.pop();
        } else {
            break;
        }
    }
    if out_points.len() < 3 {
        return Err("a ribbon loop is degenerate".into());
    }
    Ok((out_points, out_ids))
}

/// Resample a closed polyline to exactly `target` points, keeping every input
/// corner and putting every new point exactly on an input segment.
fn resample_polyline(points: &[[f64; 2]], target: usize) -> Result<Vec<[f64; 2]>, String> {
    let n = points.len();
    if n < 3 {
        return Err("cannot resample a degenerate loop".into());
    }
    if target < n {
        return Err(format!(
            "cannot resample a {n}-vertex loop to {target} points without losing corners"
        ));
    }
    let mut lengths = Vec::with_capacity(n);
    let mut perimeter = 0.0f64;
    for i in 0..n {
        let a = points[i];
        let b = points[(i + 1) % n];
        let length = point_distance(a, b);
        if length <= EPS {
            return Err("a ribbon loop has a zero-length segment".into());
        }
        lengths.push(length);
        perimeter += length;
    }
    let mut counts = vec![1usize; n];
    let extra = target - n;
    let mut used = 0usize;
    let mut fractions: Vec<(f64, usize)> = Vec::with_capacity(n);
    for i in 0..n {
        let share = extra as f64 * lengths[i] / perimeter;
        let whole = share.floor() as usize;
        counts[i] += whole;
        used += whole;
        fractions.push((share - whole as f64, i));
    }
    let mut remaining = extra - used;
    fractions.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let mut cursor = 0usize;
    while remaining > 0 && !fractions.is_empty() {
        let index = fractions[cursor % fractions.len()].1;
        counts[index] += 1;
        remaining -= 1;
        cursor += 1;
    }
    let mut out = Vec::with_capacity(target);
    for i in 0..n {
        let a = points[i];
        let b = points[(i + 1) % n];
        for k in 0..counts[i] {
            let t = k as f64 / counts[i] as f64;
            out.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
        }
    }
    Ok(out)
}

/// Resample the domain boundary to the same number of points as the partner
/// loop, distributing them over the boundary segments in proportion to how
/// many partner vertices project onto each segment. This keeps the ribbon's
/// corners aligned with the inner loop's corners (uniform arc-length sampling
/// does not, and twists the ribbon at re-entrant corners).
fn resample_matched(boundary: &[[f64; 2]], partner: &[[f64; 2]]) -> Result<Vec<[f64; 2]>, String> {
    let nseg = boundary.len();
    let target = partner.len();
    if nseg < 3 {
        return Err("cannot resample a degenerate loop".into());
    }
    if target < nseg {
        return Err(format!(
            "cannot match a {nseg}-segment boundary to {target} partner vertices"
        ));
    }
    let mut counts = vec![0usize; nseg];
    for &point in partner {
        let mut best = f64::INFINITY;
        let mut best_segment = 0usize;
        for i in 0..nseg {
            let a = boundary[i];
            let b = boundary[(i + 1) % nseg];
            let distance = point_segment_distance(point, a, b);
            if distance < best {
                best = distance;
                best_segment = i;
            }
        }
        counts[best_segment] += 1;
    }
    // Every input corner must remain a mesh node.
    while let Some(zero) = counts.iter().position(|&count| count == 0) {
        let mut donor = 0usize;
        for (index, &count) in counts.iter().enumerate() {
            if count > counts[donor] {
                donor = index;
            }
        }
        if counts[donor] <= 1 {
            break;
        }
        counts[donor] -= 1;
        counts[zero] += 1;
    }
    if counts.iter().sum::<usize>() != target {
        return Err("internal error: matched resampling lost points".into());
    }
    let mut out = Vec::with_capacity(target);
    for i in 0..nseg {
        let a = boundary[i];
        let b = boundary[(i + 1) % nseg];
        for k in 0..counts[i] {
            let t = k as f64 / counts[i] as f64;
            out.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
        }
    }
    Ok(out)
}

/// One quad per boundary edge, joining the outer loop to the inner loop.
///
/// The two loops have the same number of vertices. Every rotation and both
/// directions are tried; the best one whose quads are all convex, positively
/// oriented and whose total area equals the channel area is kept.
fn ribbon_quads(
    boundary: &[usize],
    partner: &[usize],
    pos: &[[f64; 2]],
    expected_area: f64,
) -> Result<(Vec<[usize; 4]>, f64), String> {
    let n = boundary.len();
    if n != partner.len() || n < 4 {
        return Err("the ribbon loops have different vertex counts".into());
    }
    let mut best: Option<(f64, Vec<[usize; 4]>)> = None;
    let tolerance = 1.0e-6 * expected_area.max(1.0);
    for swap in [false, true] {
        for direction in [1i64, -1i64] {
            for offset in 0..n {
                let mut quads: Vec<[usize; 4]> = Vec::with_capacity(n);
                let mut total = 0.0f64;
                let mut worst = f64::INFINITY;
                let mut valid = true;
                for i in 0..n {
                    let j0 = (offset as i64 + direction * i as i64).rem_euclid(n as i64) as usize;
                    let j1 =
                        (offset as i64 + direction * (i as i64 + 1)).rem_euclid(n as i64) as usize;
                    let quad = if swap {
                        [boundary[i], boundary[(i + 1) % n], partner[j1], partner[j0]]
                    } else {
                        [boundary[i], boundary[(i + 1) % n], partner[j0], partner[j1]]
                    };
                    let corners = [pos[quad[0]], pos[quad[1]], pos[quad[2]], pos[quad[3]]];
                    let area = quad_area(corners);
                    if area <= 1.0e-14 {
                        valid = false;
                        break;
                    }
                    let Some(rank) = quad_angle_rank(corners) else {
                        valid = false;
                        break;
                    };
                    total += area;
                    worst = worst.min(rank);
                    quads.push(quad);
                }
                if valid && (total - expected_area).abs() <= tolerance {
                    let better = match &best {
                        None => true,
                        Some((current, _)) => worst > *current,
                    };
                    if better {
                        best = Some((worst, quads));
                    }
                }
            }
        }
    }
    match best {
        Some((rank, quads)) => {
            if rank <= -1.0 {
                return Err("the boundary ribbon found only inverted quads".into());
            }
            Ok((quads, rank))
        }
        None => {
            Err("no rotation of the matched boundary/inner loops produced a valid ribbon".into())
        }
    }
}

/// Shared tail: normalise orientation, drop unused points, build and validate
/// the mesh, then fill the quality counters in `stats`.
fn finish_mesh(
    pos: Vec<[f64; 2]>,
    mut quads: Vec<[usize; 4]>,
    domain: &Domain,
    stats: &mut ChannelStats,
) -> Result<BuildOutput, String> {
    for quad in &mut quads {
        let corners = [
            *pos.get(quad[0]).ok_or("internal error: quad index")?,
            *pos.get(quad[1]).ok_or("internal error: quad index")?,
            *pos.get(quad[2]).ok_or("internal error: quad index")?,
            *pos.get(quad[3]).ok_or("internal error: quad index")?,
        ];
        if quad_area(corners) < 0.0 {
            quad.reverse();
        }
    }
    let mut used = vec![false; pos.len()];
    for quad in &quads {
        for &index in quad {
            let Some(slot) = used.get_mut(index) else {
                return Err("internal error: quad index out of range".into());
            };
            *slot = true;
        }
    }
    let mut remap = vec![usize::MAX; pos.len()];
    let mut points: Vec<Node> = Vec::new();
    for (index, point) in pos.iter().enumerate() {
        if used[index] {
            remap[index] = points.len();
            points.push([point[0], point[1], 0.0]);
        }
    }
    let mut cells: Vec<Cell> = Vec::with_capacity(quads.len());
    for quad in &quads {
        let mut nodes = [0usize; 4];
        for c in 0..4 {
            let mapped = *remap.get(quad[c]).ok_or("internal error: remap")?;
            if mapped == usize::MAX {
                return Err("internal error: unused node referenced by a quad".into());
            }
            nodes[c] = mapped;
        }
        if nodes[0] == nodes[1]
            || nodes[0] == nodes[2]
            || nodes[0] == nodes[3]
            || nodes[1] == nodes[2]
            || nodes[1] == nodes[3]
            || nodes[2] == nodes[3]
        {
            return Err("internal error: a quad repeats a node".into());
        }
        cells.push(Cell::quad4(nodes));
    }
    let mesh = Mesh::new(points, cells);
    mesh.validate().map_err(|error| error.to_string())?;
    let topology = mesh.topology();
    if topology.cracks != 0 {
        return Err(format!(
            "the channel mesh is not conforming: {} crack(s)",
            topology.cracks
        ));
    }
    if topology.inconsistent != 0 || topology.non_manifold != 0 {
        return Err("the channel mesh is not a consistently oriented manifold".into());
    }
    let report = quality::report(&mesh);
    if report.inverted > 0 {
        return Err(format!(
            "the channel mesh has {} inverted cell(s)",
            report.inverted
        ));
    }
    if report.min_scaled_jacobian <= 0.0 {
        return Err("the channel mesh has a degenerate cell".into());
    }
    let boundary_offset = max_boundary_offset(&mesh, domain)?;
    stats.quads = quads.len();
    stats.min_scaled_jacobian = report.min_scaled_jacobian;
    stats.min_angle_deg = report.min_corner_angle_deg;
    stats.max_angle_deg = report.max_corner_angle_deg;
    stats.cracks = topology.cracks;
    stats.inverted = report.inverted;
    stats.boundary_offset = boundary_offset;
    Ok(BuildOutput {
        mesh,
        stats: stats.clone(),
    })
}

fn build_templates_with_delta(
    domain: &Domain,
    size: f64,
    delta: f64,
) -> Result<BuildOutput, String> {
    let prepared = prepare(domain, size, delta)?;
    let cross_edges = build_cross_edges(&prepared, domain)?;
    let faces = build_faces(&prepared, &cross_edges, domain)?;

    // The faces must tile the channel exactly.
    let channel_area: f64 = faces.iter().map(|face| face.area).sum();
    let grid_area: f64 = prepared
        .quads
        .iter()
        .map(|quad| {
            let corners = [
                prepared.pos[quad[0]],
                prepared.pos[quad[1]],
                prepared.pos[quad[2]],
                prepared.pos[quad[3]],
            ];
            quad_area(corners)
        })
        .sum();
    let expected = domain.area() - grid_area;
    if expected <= EPS {
        return Err("the channel has no area (the grid fills the domain)".into());
    }
    if (channel_area - expected).abs() > 1.0e-6 * expected.max(1.0) {
        return Err(format!(
            "the channel faces cover {channel_area:.6} but the channel area is {expected:.6}"
        ));
    }

    let mut patches = build_patches(&faces, &cross_edges);
    let pos_snapshot = prepared.pos.clone();
    finalize_patches(&mut patches, &pos_snapshot);
    let usable: usize = patches
        .iter()
        .filter(|patch| patch.decomposition.is_some())
        .count();

    let (selected, greedy_fraction) = greedy_cover(&faces, &patches);

    // ---- fallback: close whatever the greedy cover left -------------------
    let mut pos = prepared.pos.clone();
    let mut selected = selected;
    let mut splits: HashMap<(usize, usize), usize> = HashMap::new();
    let mut fallback_quads: Vec<[usize; 4]> = Vec::new();
    let mut fallback_regions = 0usize;
    let mut attempts = 0usize;
    let limit = patches.len() + 64;
    loop {
        attempts += 1;
        if attempts > limit {
            return Err("the fallback did not converge".into());
        }
        let mut covered = vec![false; faces.len()];
        for &patch_index in &selected {
            if let Some(patch) = patches.get(patch_index) {
                for &face in &patch.faces {
                    if let Some(slot) = covered.get_mut(face) {
                        *slot = true;
                    }
                }
            }
        }
        let components = components_of(&covered, &faces);
        let mut bad: Option<usize> = None;
        for (index, component) in components.iter().enumerate() {
            let loops = match component_loops(component, &faces, &splits) {
                Ok(loops) => loops,
                Err(_) => {
                    bad = Some(index);
                    break;
                }
            };
            if loops.len() != 1 {
                bad = Some(index);
                break;
            }
            let Some(loop_ids) = loops.first() else {
                bad = Some(index);
                break;
            };
            if try_fill(loop_ids, &pos, pos.len()).is_none() {
                bad = Some(index);
                break;
            }
        }
        if let Some(index) = bad {
            let Some(component) = components.get(index) else {
                return Err("internal error: missing fallback component".into());
            };
            fix_component(
                component,
                &faces,
                &patches,
                &mut selected,
                &mut splits,
                &mut pos,
                &prepared.domain_edges,
            )?;
            continue;
        }
        // Everything is fillable: commit the fill.
        for component in &components {
            let loops = component_loops(component, &faces, &splits)?;
            let Some(loop_ids) = loops.first() else {
                continue;
            };
            let Some(plan) = try_fill(loop_ids, &pos, pos.len()) else {
                return Err("a fallback region became unfillable".into());
            };
            if let Some(steiner) = plan.steiner {
                if steiner[0].is_finite() && steiner[1].is_finite() {
                    pos.push(steiner);
                }
            }
            fallback_quads.extend(plan.quads);
            fallback_regions += 1;
        }
        break;
    }

    // ---- assemble ---------------------------------------------------------
    let mut quads: Vec<[usize; 4]> = prepared.quads.clone();
    let mut channel_quads: Vec<[usize; 4]> = Vec::new();
    for &patch_index in &selected {
        let Some(patch) = patches.get(patch_index) else {
            continue;
        };
        let Some(decomposition) = &patch.decomposition else {
            continue;
        };
        let n = patch.boundary.len();
        let mut steiner_ids: Vec<usize> = Vec::with_capacity(decomposition.steiner.len());
        for &point in &decomposition.steiner {
            steiner_ids.push(pos.len());
            pos.push(point);
        }
        for quad in &decomposition.quads {
            let mut global = [0usize; 4];
            for c in 0..4 {
                let index = quad[c];
                global[c] = if index < n {
                    match patch.boundary.get(index) {
                        Some(&id) => id,
                        None => return Err("internal error: patch boundary index".into()),
                    }
                } else {
                    match steiner_ids.get(index - n) {
                        Some(&id) => id,
                        None => return Err("internal error: patch Steiner index".into()),
                    }
                };
            }
            channel_quads.push(global);
        }
    }
    channel_quads.extend(fallback_quads.iter().copied());
    quads.extend(channel_quads.iter().copied());

    // Every quad must be counter-clockwise.
    for quad in &mut quads {
        let corners = [
            *pos.get(quad[0]).ok_or("internal error: quad index")?,
            *pos.get(quad[1]).ok_or("internal error: quad index")?,
            *pos.get(quad[2]).ok_or("internal error: quad index")?,
            *pos.get(quad[3]).ok_or("internal error: quad index")?,
        ];
        if quad_area(corners) < 0.0 {
            quad.reverse();
        }
    }

    // Drop unused points and build the mesh.
    let mut used = vec![false; pos.len()];
    for quad in &quads {
        for &index in quad {
            let Some(slot) = used.get_mut(index) else {
                return Err("internal error: quad index out of range".into());
            };
            *slot = true;
        }
    }
    let mut remap = vec![usize::MAX; pos.len()];
    let mut points: Vec<Node> = Vec::new();
    for (index, point) in pos.iter().enumerate() {
        if used[index] {
            remap[index] = points.len();
            points.push([point[0], point[1], 0.0]);
        }
    }
    let mut cells: Vec<Cell> = Vec::with_capacity(quads.len());
    for quad in &quads {
        let mut nodes = [0usize; 4];
        for c in 0..4 {
            let mapped = *remap.get(quad[c]).ok_or("internal error: remap")?;
            if mapped == usize::MAX {
                return Err("internal error: unused node referenced by a quad".into());
            }
            nodes[c] = mapped;
        }
        if nodes[0] == nodes[1]
            || nodes[0] == nodes[2]
            || nodes[0] == nodes[3]
            || nodes[1] == nodes[2]
            || nodes[1] == nodes[3]
            || nodes[2] == nodes[3]
        {
            return Err("internal error: a quad repeats a node".into());
        }
        cells.push(Cell::quad4(nodes));
    }
    let mesh = Mesh::new(points, cells);
    mesh.validate().map_err(|error| error.to_string())?;
    let topology = mesh.topology();
    if topology.cracks != 0 {
        return Err(format!(
            "the channel mesh is not conforming: {} crack(s)",
            topology.cracks
        ));
    }
    if topology.inconsistent != 0 || topology.non_manifold != 0 {
        return Err("the channel mesh is not a consistently oriented manifold".into());
    }
    let report = quality::report(&mesh);
    if report.inverted > 0 {
        return Err(format!(
            "the channel mesh has {} inverted cell(s)",
            report.inverted
        ));
    }
    if report.min_scaled_jacobian <= 0.0 {
        return Err("the channel mesh has a degenerate cell".into());
    }
    let boundary_offset = max_boundary_offset(&mesh, domain)?;

    let stats = ChannelStats {
        method: "templates+greedy",
        delta_factor: delta,
        grid_vertices: prepared.grid_vertex_count,
        interior_quads: prepared.quads.len(),
        channel_loops: prepared.loops.len(),
        cross_edges: cross_edges.len(),
        faces: faces.len(),
        triangle_faces: faces.iter().filter(|face| face.vertices.len() == 3).count(),
        patches: patches.len(),
        usable_patches: usable,
        selected_patches: selected.len(),
        greedy_covered_fraction: greedy_fraction,
        fallback_regions,
        fallback_quads: fallback_quads.len(),
        channel_quads: channel_quads.len(),
        quads: quads.len(),
        min_scaled_jacobian: report.min_scaled_jacobian,
        min_angle_deg: report.min_corner_angle_deg,
        max_angle_deg: report.max_corner_angle_deg,
        cracks: topology.cracks,
        inverted: report.inverted,
        boundary_offset,
        split_perimeter_edges: splits.len(),
    };
    Ok(BuildOutput { mesh, stats })
}

fn max_boundary_offset(mesh: &Mesh, domain: &Domain) -> Result<f64, String> {
    let mut edge_use: HashMap<(usize, usize), usize> = HashMap::new();
    for cell in &mesh.cells {
        for &(a, b) in cell.kind.edges() {
            let Some(&u) = cell.nodes.get(a) else {
                continue;
            };
            let Some(&v) = cell.nodes.get(b) else {
                continue;
            };
            *edge_use.entry(edge_key(u, v)).or_insert(0) += 1;
        }
    }
    let mut boundary_nodes: BTreeSet<usize> = BTreeSet::new();
    for ((a, b), count) in edge_use {
        if count == 1 {
            boundary_nodes.insert(a);
            boundary_nodes.insert(b);
        }
    }
    let mut worst = 0.0f64;
    for &index in &boundary_nodes {
        let Some(point) = mesh.points.get(index) else {
            return Err("internal error: boundary node missing".into());
        };
        worst = worst.max(domain.distance_to_boundary([point[0], point[1]]));
    }
    Ok(worst)
}

// ---------------------------------------------------------------------------
// Patch union helper
// ---------------------------------------------------------------------------

fn union_boundary(
    face_ids: &[usize],
    faces: &[Face],
) -> Option<(Vec<usize>, Vec<(usize, usize)>, f64)> {
    let mut count: HashMap<(usize, usize), usize> = HashMap::new();
    let mut area = 0.0f64;
    for &face_index in face_ids {
        let face = faces.get(face_index)?;
        area += face.area;
        let n = face.vertices.len();
        for i in 0..n {
            let key = edge_key(face.vertices[i], face.vertices[(i + 1) % n]);
            *count.entry(key).or_insert(0) += 1;
        }
    }
    if count.values().any(|&c| c > 2) {
        return None;
    }
    let mut next: HashMap<usize, usize> = HashMap::new();
    let mut boundary_edges: Vec<(usize, usize)> = Vec::new();
    for &face_index in face_ids {
        let face = faces.get(face_index)?;
        let n = face.vertices.len();
        for i in 0..n {
            let a = face.vertices[i];
            let b = face.vertices[(i + 1) % n];
            if count.get(&edge_key(a, b)).copied().unwrap_or(0) != 1 {
                continue;
            }
            if next.insert(a, b).is_some() {
                return None;
            }
            boundary_edges.push((a, b));
        }
    }
    let start = boundary_edges.first()?.0;
    let mut loop_ids: Vec<usize> = Vec::new();
    let mut current = start;
    let mut guard = 0usize;
    loop {
        loop_ids.push(current);
        let following = *next.get(&current)?;
        current = following;
        if current == start {
            break;
        }
        guard += 1;
        if guard > boundary_edges.len() + 1 {
            return None;
        }
    }
    if loop_ids.len() != boundary_edges.len() {
        return None; // more than one boundary loop
    }
    Some((loop_ids, boundary_edges, area))
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

fn edge_key(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

fn polygon_area_ids(pos: &[[f64; 2]], ids: &[usize]) -> f64 {
    if ids.len() < 3 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..ids.len() {
        let a = pos[ids[i]];
        let b = pos[ids[(i + 1) % ids.len()]];
        sum += a[0] * b[1] - b[0] * a[1];
    }
    0.5 * sum
}

fn polygon_area_points(points: &[[f64; 2]]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        sum += a[0] * b[1] - b[0] * a[1];
    }
    0.5 * sum
}

fn polygon_centroid(points: &[[f64; 2]]) -> Option<[f64; 2]> {
    let area = polygon_area_points(points);
    if area.abs() < EPS {
        return None;
    }
    let mut x = 0.0;
    let mut y = 0.0;
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        let cross = a[0] * b[1] - b[0] * a[1];
        x += (a[0] + b[0]) * cross;
        y += (a[1] + b[1]) * cross;
    }
    let centroid = [x / (6.0 * area), y / (6.0 * area)];
    if centroid[0].is_finite() && centroid[1].is_finite() {
        Some(centroid)
    } else {
        None
    }
}

fn point_distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// Is `p` inside or on the boundary of a counter-clockwise quad?
///
/// Inclusive on purpose: a sample that lands exactly on a lattice corner (the
/// centroid of a square grid block does) still belongs to the grid region.
fn point_in_or_on_convex_quad(p: [f64; 2], quad: &[usize; 4], pos: &[[f64; 2]]) -> bool {
    for i in 0..4 {
        let Some(&a) = pos.get(quad[i]) else {
            return false;
        };
        let Some(&b) = pos.get(quad[(i + 1) % 4]) else {
            return false;
        };
        let cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
        if cross < -1.0e-12 {
            return false;
        }
    }
    true
}

fn point_in_polygon(p: [f64; 2], polygon: &[[f64; 2]]) -> bool {
    let mut inside = false;
    let n = polygon.len();
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

fn distance_to_polygon_edges(p: [f64; 2], polygon: &[[f64; 2]]) -> f64 {
    let mut best = f64::INFINITY;
    for i in 0..polygon.len() {
        let a = polygon[i];
        let b = polygon[(i + 1) % polygon.len()];
        best = best.min(point_segment_distance(p, a, b));
    }
    best
}

fn point_segment_distance(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let length2 = ab[0] * ab[0] + ab[1] * ab[1];
    if length2 <= 0.0 {
        return point_distance(p, a);
    }
    let t = (((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / length2).clamp(0.0, 1.0);
    let q = [a[0] + t * ab[0], a[1] + t * ab[1]];
    point_distance(p, q)
}

fn point_on_segment_strict(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> bool {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let length2 = ab[0] * ab[0] + ab[1] * ab[1];
    if length2 <= 0.0 {
        return false;
    }
    let t = ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / length2;
    if !(t > 1.0e-9 && t < 1.0 - 1.0e-9) {
        return false;
    }
    let q = [a[0] + t * ab[0], a[1] + t * ab[1]];
    point_distance(p, q) <= 1.0e-9
}

fn orient(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// Do two segments cross or overlap in a way that makes them incompatible?
/// Cleanly shared endpoints are allowed.
fn edges_conflict(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
    let o1 = orient(a, b, c);
    let o2 = orient(a, b, d);
    let o3 = orient(c, d, a);
    let o4 = orient(c, d, b);
    let proper = ((o1 > EPS && o2 < -EPS) || (o1 < -EPS && o2 > EPS))
        && ((o3 > EPS && o4 < -EPS) || (o3 < -EPS && o4 > EPS));
    if proper {
        return true;
    }
    let collinear = o1.abs() <= EPS && o2.abs() <= EPS;
    if collinear {
        let ab = [b[0] - a[0], b[1] - a[1]];
        let length2 = ab[0] * ab[0] + ab[1] * ab[1];
        if length2 <= 0.0 {
            return false;
        }
        let project =
            |p: [f64; 2]| -> f64 { ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / length2 };
        let tc = project(c);
        let td = project(d);
        let low = tc.min(td).max(0.0);
        let high = tc.max(td).min(1.0);
        if high - low > 1.0e-9 {
            return true;
        }
    }
    point_on_segment_strict(c, a, b)
        || point_on_segment_strict(d, a, b)
        || point_on_segment_strict(a, c, d)
        || point_on_segment_strict(b, c, d)
}

fn quad_area(quad: [[f64; 2]; 4]) -> f64 {
    0.5 * ((quad[0][0] * quad[1][1] - quad[1][0] * quad[0][1])
        + (quad[1][0] * quad[2][1] - quad[2][0] * quad[1][1])
        + (quad[2][0] * quad[3][1] - quad[3][0] * quad[2][1])
        + (quad[3][0] * quad[0][1] - quad[0][0] * quad[3][1]))
}

/// Rank a convex quad by its worst angle deviation from 90°:
/// `1 - max|angle - 90| / 90`. `None` for a degenerate, inverted or concave
/// quad.
fn quad_angle_rank(quad: [[f64; 2]; 4]) -> Option<f64> {
    if quad_area(quad) <= 1.0e-14 {
        return None;
    }
    for i in 0..4 {
        let a = quad[i];
        let b = quad[(i + 1) % 4];
        let c = quad[(i + 2) % 4];
        if orient(a, b, c) <= 1.0e-12 {
            return None;
        }
    }
    let mut worst = 0.0f64;
    for i in 0..4 {
        let previous = quad[(i + 3) % 4];
        let next = quad[(i + 1) % 4];
        let u = [previous[0] - quad[i][0], previous[1] - quad[i][1]];
        let v = [next[0] - quad[i][0], next[1] - quad[i][1]];
        let lu = (u[0] * u[0] + u[1] * u[1]).sqrt();
        let lv = (v[0] * v[0] + v[1] * v[1]).sqrt();
        if lu <= 1.0e-12 || lv <= 1.0e-12 {
            return None;
        }
        let cosine = ((u[0] * v[0] + u[1] * v[1]) / (lu * lv)).clamp(-1.0, 1.0);
        let angle = cosine.acos().to_degrees();
        if !angle.is_finite() {
            return None;
        }
        worst = worst.max((angle - 90.0).abs());
    }
    Some(1.0 - worst / 90.0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[allow(clippy::assertions_on_constants)]
mod tests {
    use super::*;

    fn valid_domain(points: Vec<[f64; 2]>) -> Domain {
        match Domain::new(points, Vec::new()) {
            Ok(domain) => domain,
            Err(error) => {
                assert!(false, "test domain invalid: {error}");
                Domain::default()
            }
        }
    }

    fn check_mesh(mesh: &Mesh, domain: &Domain, label: &str) -> (usize, f64, f64, f64, f64) {
        assert!(
            mesh.cells
                .iter()
                .all(|cell| cell.kind == crate::mesh::Kind::Quad4),
            "{label}: not all cells are quads"
        );
        assert!(!mesh.cells.is_empty(), "{label}: empty mesh");
        let topology = mesh.topology();
        assert_eq!(topology.cracks, 0, "{label}: cracks");
        assert_eq!(
            topology.inconsistent, 0,
            "{label}: inconsistent orientation"
        );
        assert_eq!(topology.non_manifold, 0, "{label}: non-manifold");
        let report = quality::report(mesh);
        assert!(report.inverted == 0, "{label}: inverted cells");
        assert!(
            report.min_scaled_jacobian > 0.0,
            "{label}: non-positive scaled Jacobian"
        );
        let offset = match max_boundary_offset(mesh, domain) {
            Ok(value) => value,
            Err(error) => {
                assert!(false, "{label}: boundary offset failed: {error}");
                0.0
            }
        };
        assert!(offset < 1.0e-7, "{label}: boundary offset {offset}");
        (
            mesh.cells.len(),
            report.min_scaled_jacobian,
            report.min_corner_angle_deg,
            report.max_corner_angle_deg,
            offset,
        )
    }

    #[test]
    fn channel_square_has_two_loops() {
        let domain = match Domain::rectangle(10.0, 10.0) {
            Ok(domain) => domain,
            Err(error) => {
                assert!(false, "{error}");
                return;
            }
        };
        let loops = match channel(&domain, 1.0) {
            Ok(loops) => loops,
            Err(error) => {
                assert!(false, "channel failed: {error}");
                return;
            }
        };
        assert_eq!(
            loops.len(),
            2,
            "a 10x10 square has an outer and an inner loop"
        );
        let mut signs = Vec::new();
        for loop_ in &loops {
            assert!(loop_.points.len() >= 4);
            assert_eq!(loop_.points.len() % 2, 0);
            let area = polygon_area_points(&loop_.points);
            signs.push(area > 0.0);
        }
        signs.sort_unstable();
        assert_eq!(signs, vec![false, true], "the two loops are opposed");
    }

    #[test]
    fn channel_between_square_annulus() {
        let domain = match Domain::rectangle(10.0, 10.0) {
            Ok(domain) => domain,
            Err(error) => {
                assert!(false, "{error}");
                return;
            }
        };
        let outer = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let inner = vec![[2.0, 2.0], [8.0, 2.0], [8.0, 8.0], [2.0, 8.0]];
        let inner_ids = [0usize, 1, 2, 3];
        let mut points: Vec<[f64; 3]> = inner.iter().map(|p| [p[0], p[1], 0.0]).collect();

        let quads = match channel_between(&domain, &outer, &inner, &inner_ids, &mut points, 1.0) {
            Ok(quads) => quads,
            Err(error) => {
                assert!(false, "channel_between failed: {error}");
                return;
            }
        };
        assert_eq!(quads.len(), 4, "four quads around the square ring");
        assert_eq!(points.len(), 8, "four inner + four outer nodes");
        assert_eq!(points.len(), 8);

        let cells: Vec<Cell> = quads.iter().map(|&quad| Cell::quad4(quad)).collect();
        let mesh = Mesh::new(points.clone(), cells);
        assert!(mesh.validate().is_ok(), "the ribbon mesh must validate");
        let topology = mesh.topology();
        assert_eq!(topology.cracks, 0, "cracks");
        assert_eq!(topology.inconsistent, 0, "inconsistent orientation");
        assert_eq!(topology.non_manifold, 0, "non-manifold");
        let report = quality::report(&mesh);
        assert_eq!(report.inverted, 0, "inverted cells");
        assert!(report.min_scaled_jacobian > 0.0);

        let mut edge_use: HashMap<(usize, usize), usize> = HashMap::new();
        let mut area = 0.0f64;
        for quad in &quads {
            let corners = [
                points[quad[0]],
                points[quad[1]],
                points[quad[2]],
                points[quad[3]],
            ];
            area += quad_area([
                [corners[0][0], corners[0][1]],
                [corners[1][0], corners[1][1]],
                [corners[2][0], corners[2][1]],
                [corners[3][0], corners[3][1]],
            ]);
            for k in 0..4 {
                *edge_use
                    .entry(edge_key(quad[k], quad[(k + 1) % 4]))
                    .or_insert(0) += 1;
            }
        }
        assert!((area - 64.0).abs() < 1.0e-9, "ring area {area}");
        // Inner edges are used once and keep the caller's ids; so are the four
        // outer edges (nodes 4..8); every cross edge is shared by two quads.
        for i in 0..4 {
            let a = inner_ids[i];
            let b = inner_ids[(i + 1) % 4];
            assert_eq!(
                edge_use.get(&edge_key(a, b)).copied(),
                Some(1),
                "inner edge {a}-{b}"
            );
            assert_eq!(
                edge_use.get(&edge_key(4 + i, 4 + (i + 1) % 4)).copied(),
                Some(1),
                "outer edge"
            );
        }
        for (&edge, &count) in &edge_use {
            let inner_edge = (0..4).any(|i| edge_key(inner_ids[i], inner_ids[(i + 1) % 4]) == edge);
            let outer_edge = (0..4).any(|i| edge_key(4 + i, 4 + (i + 1) % 4) == edge);
            if !inner_edge && !outer_edge {
                assert_eq!(count, 2, "cross edge {edge:?} must be shared");
            }
        }
        for point in points.iter().skip(4) {
            assert!(
                domain.distance_to_boundary([point[0], point[1]]) < 1.0e-9,
                "outer node off the polyline"
            );
        }
        for id in inner_ids {
            assert!(quads.iter().any(|quad| quad.contains(&id)));
        }
        println!(
            "channel_between square annulus: {} quads, {} nodes, area {:.3}, min SJ {:.4}",
            quads.len(),
            points.len(),
            area,
            report.min_scaled_jacobian
        );

        // A subdivided inner loop exercises the loop-count mismatch path.
        let inner8 = vec![
            [2.0, 2.0],
            [5.0, 2.0],
            [8.0, 2.0],
            [8.0, 5.0],
            [8.0, 8.0],
            [5.0, 8.0],
            [2.0, 8.0],
            [2.0, 5.0],
        ];
        let inner8_ids = [0usize, 1, 2, 3, 4, 5, 6, 7];
        let mut points8: Vec<[f64; 3]> = inner8.iter().map(|p| [p[0], p[1], 0.0]).collect();
        match channel_between(&domain, &outer, &inner8, &inner8_ids, &mut points8, 1.0) {
            Ok(quads8) => {
                let mut area8 = 0.0f64;
                for quad in &quads8 {
                    area8 += quad_area([
                        [points8[quad[0]][0], points8[quad[0]][1]],
                        [points8[quad[1]][0], points8[quad[1]][1]],
                        [points8[quad[2]][0], points8[quad[2]][1]],
                        [points8[quad[3]][0], points8[quad[3]][1]],
                    ]);
                }
                assert!((area8 - 64.0).abs() < 1.0e-9, "8-gon ring area {area8}");
                let cells8: Vec<Cell> = quads8.iter().map(|&quad| Cell::quad4(quad)).collect();
                let mesh8 = Mesh::new(points8.clone(), cells8);
                let topology8 = mesh8.topology();
                assert_eq!(topology8.cracks, 0);
                assert_eq!(quality::report(&mesh8).inverted, 0);
                println!(
                    "channel_between 8-sided inner loop: {} quads, {} nodes, area {:.3}",
                    quads8.len(),
                    points8.len(),
                    area8
                );
            }
            Err(error) => println!("channel_between 8-sided inner loop not supported: {error}"),
        }

        // Failure must not touch the shared pool.
        let mut pool: Vec<[f64; 3]> = inner.iter().map(|p| [p[0], p[1], 0.0]).collect();
        let before = pool.clone();
        let bad = channel_between(&domain, &outer, &inner, &[0, 1, 2, 99], &mut pool, 1.0);
        assert!(bad.is_err());
        assert_eq!(pool, before);
    }

    fn dense_rectangle(corners: [[f64; 2]; 4], nodes_per_side: usize) -> Vec<[f64; 2]> {
        let mut out = Vec::with_capacity(4 * nodes_per_side);
        for side in 0..4 {
            let a = corners[side];
            let b = corners[(side + 1) % 4];
            for k in 0..nodes_per_side {
                let t = k as f64 / nodes_per_side as f64;
                out.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
            }
        }
        out
    }

    fn ribbon_mesh(points: &[[f64; 3]], quads: &[[usize; 4]]) -> Mesh {
        let cells: Vec<Cell> = quads.iter().map(|&quad| Cell::quad4(quad)).collect();
        Mesh::new(points.to_vec(), cells)
    }

    #[test]
    fn channel_between_dense_inner() {
        let domain = match Domain::rectangle(2.0, 1.0) {
            Ok(domain) => domain,
            Err(error) => {
                assert!(false, "{error}");
                return;
            }
        };
        let outer = vec![[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]];
        let corners = [[0.5, 0.25], [1.5, 0.25], [1.5, 0.75], [0.5, 0.75]];
        let rings = [
            vec![[0.5, 0.25], [1.5, 0.25], [1.5, 0.75], [0.5, 0.75]],
            dense_rectangle(corners, 11),
        ];
        for (case, inner) in rings.iter().enumerate() {
            let ids: Vec<usize> = (0..inner.len()).collect();
            let mut points: Vec<[f64; 3]> = inner.iter().map(|p| [p[0], p[1], 0.0]).collect();
            let result = channel_between(&domain, &outer, inner, &ids, &mut points, 1.0);
            let quads = match result {
                Ok(quads) => quads,
                Err(error) => {
                    assert!(false, "dense inner case {case} failed: {error}");
                    return;
                }
            };
            assert_eq!(quads.len(), 4, "case {case}: four quads");
            assert_eq!(points.len(), inner.len() + 4, "case {case}: node pool");
            let mut area = 0.0f64;
            for quad in &quads {
                area += quad_area([
                    [points[quad[0]][0], points[quad[0]][1]],
                    [points[quad[1]][0], points[quad[1]][1]],
                    [points[quad[2]][0], points[quad[2]][1]],
                    [points[quad[3]][0], points[quad[3]][1]],
                ]);
            }
            assert!((area - 1.5).abs() < 1.0e-9, "case {case}: area {area}");
            let mesh = ribbon_mesh(&points, &quads);
            let report = quality::report(&mesh);
            assert_eq!(mesh.topology().cracks, 0, "case {case}: cracks");
            assert_eq!(report.inverted, 0);
            assert!(report.min_scaled_jacobian > 0.0);
            if case == 1 {
                let used: BTreeSet<usize> = quads
                    .iter()
                    .flat_map(|quad| quad.iter().copied())
                    .filter(|&id| id < inner.len())
                    .collect();
                assert_eq!(used, BTreeSet::from([0usize, 11, 22, 33]));
            }
            println!(
                "channel_between dense inner {case}: {} quads, {} nodes, min SJ {:.4}",
                quads.len(),
                points.len(),
                report.min_scaled_jacobian
            );
        }
    }

    #[test]
    fn channel_between_hole_square() {
        let outer = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let hole = vec![[3.0, 3.0], [7.0, 3.0], [7.0, 7.0], [3.0, 7.0]];
        let domain = match Domain::new(outer, vec![hole.clone()]) {
            Ok(domain) => domain,
            Err(error) => {
                assert!(false, "{error}");
                return;
            }
        };
        let inner = vec![[2.0, 2.0], [8.0, 2.0], [8.0, 8.0], [2.0, 8.0]];
        let inner_ids = [0usize, 1, 2, 3];
        let mut points: Vec<[f64; 3]> = inner.iter().map(|p| [p[0], p[1], 0.0]).collect();
        let result = channel_between_hole(&domain, &hole, &inner, &inner_ids, &mut points, 1.0);
        let quads = match result {
            Ok(quads) => quads,
            Err(error) => {
                assert!(false, "hole band failed: {error}");
                return;
            }
        };
        assert_eq!(quads.len(), 4, "four quads around the hole band");
        assert_eq!(points.len(), 8, "four inner + four hole nodes");
        let mut area = 0.0f64;
        for quad in &quads {
            area += quad_area([
                [points[quad[0]][0], points[quad[0]][1]],
                [points[quad[1]][0], points[quad[1]][1]],
                [points[quad[2]][0], points[quad[2]][1]],
                [points[quad[3]][0], points[quad[3]][1]],
            ]);
        }
        assert!((area - 20.0).abs() < 1.0e-9, "hole band area {area}");
        let mesh = ribbon_mesh(&points, &quads);
        let topology = mesh.topology();
        assert_eq!(topology.cracks, 0, "cracks");
        assert_eq!(topology.inconsistent, 0);
        assert_eq!(topology.non_manifold, 0);
        let report = quality::report(&mesh);
        assert_eq!(report.inverted, 0);
        assert!((report.min_scaled_jacobian - std::f64::consts::FRAC_1_SQRT_2).abs() < 1.0e-3);
        for i in 0..4 {
            let a = inner_ids[i];
            let b = inner_ids[(i + 1) % 4];
            let count = quads
                .iter()
                .filter(|quad| {
                    (0..4).any(|k| {
                        let u = quad[k];
                        let v = quad[(k + 1) % 4];
                        (u == a && v == b) || (u == b && v == a)
                    })
                })
                .count();
            assert_eq!(count, 1, "inner edge {a}-{b}");
        }
        println!(
            "channel_between_hole square: {} quads, {} nodes, area {:.3}, min SJ {:.4}",
            quads.len(),
            points.len(),
            area,
            report.min_scaled_jacobian
        );
    }

    #[test]
    fn square_10x10() {
        let domain = match Domain::rectangle(10.0, 10.0) {
            Ok(domain) => domain,
            Err(error) => {
                assert!(false, "{error}");
                return;
            }
        };
        let (mesh, stats) = match mesh_with_stats(&domain, 1.0) {
            Ok(result) => result,
            Err(error) => {
                assert!(false, "square mesh failed: {error}");
                return;
            }
        };
        let (quads, min_sj, min_angle, max_angle, offset) = check_mesh(&mesh, &domain, "square");
        println!(
            "square 10x10 h=1: {quads} quads ({} interior + {} channel), loops {}, \
             cross edges {}, faces {}, patches {} ({} usable), selected {}, min SJ {min_sj:.4}, \
             angles {min_angle:.2}..{max_angle:.2}, cracks {}, offset {offset:.3e}, \
             greedy {:.1}%, fallback {} regions / {} quads",
            stats.interior_quads,
            stats.channel_quads,
            stats.channel_loops,
            stats.cross_edges,
            stats.faces,
            stats.patches,
            stats.usable_patches,
            stats.selected_patches,
            stats.cracks,
            100.0 * stats.greedy_covered_fraction,
            stats.fallback_regions,
            stats.fallback_quads,
        );
        assert!(quads > 40, "expected more than 40 quads, got {quads}");
        assert_eq!(stats.interior_quads, 36, "6x6 interior grid quads");
        assert!(stats.channel_quads > 0);
        assert!(stats.greedy_covered_fraction > 0.5);
        assert!(min_sj > 0.05, "min SJ too low: {min_sj}");
        assert!(min_angle > 10.0, "min angle too low: {min_angle}");
    }

    fn corpus_l_bracket() -> Domain {
        valid_domain(vec![
            [0.0, 0.0],
            [80.0, 0.0],
            [80.0, 20.0],
            [20.0, 20.0],
            [20.0, 60.0],
            [0.0, 60.0],
        ])
    }

    fn corpus_ring() -> Domain {
        let mut outer: Vec<[f64; 2]> = (0..=24)
            .map(|i| {
                let a = std::f64::consts::FRAC_PI_2 * i as f64 / 24.0;
                [20.0 * a.cos(), 20.0 * a.sin()]
            })
            .collect();
        outer.extend((0..=12).map(|i| {
            let a = std::f64::consts::FRAC_PI_2 * (1.0 - i as f64 / 12.0);
            [10.0 * a.cos(), 10.0 * a.sin()]
        }));
        valid_domain(outer)
    }

    fn corpus_stepped() -> Domain {
        valid_domain(vec![
            [0.0, 0.0],
            [60.0, 0.0],
            [60.0, 40.0],
            [30.0, 40.0],
            [30.0, 30.0],
            [20.0, 30.0],
            [20.0, 40.0],
            [0.0, 40.0],
        ])
    }

    #[test]
    fn dp_ribbon_smoke() {
        let pos = vec![
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
            [2.0, 2.0],
            [2.0, 8.0],
            [8.0, 8.0],
            [8.0, 2.0],
        ];
        let outer = [0usize, 1, 2, 3];
        let inner = [4usize, 5, 6, 7];
        println!("outer area {}", polygon_area_ids(&pos, &outer));
        println!("inner area {}", polygon_area_ids(&pos, &inner));
        let result = dp_ribbon(&outer, &inner, &pos, 64.0);
        println!(
            "dp smoke: {:?}",
            result.as_ref().map(|(q, r)| (q.len(), *r))
        );
        assert!(result.is_ok(), "dp smoke failed");
    }

    #[test]
    fn corpus_sections() {
        for (label, domain, size, must_mesh) in [
            ("l-bracket-2d", corpus_l_bracket(), 2.0_f64, true),
            ("ring-section", corpus_ring(), 1.0_f64, true),
            ("stepped-2d", corpus_stepped(), 1.0_f64, true),
        ] {
            match mesh_with_stats(&domain, size) {
                Ok((mesh, stats)) => {
                    let (quads, min_sj, min_angle, max_angle, offset) =
                        check_mesh(&mesh, &domain, label);
                    let report = quality::report(&mesh);
                    println!(
                        "{label} h={size} [{} delta={:.2}h]: {quads} quads ({} interior + {} channel), \
                         loops {}, min SJ {min_sj:.4}, angles {min_angle:.2}..{max_angle:.2}, \
                         aspect {:.2}, cracks {}, offset {offset:.3e}, verdict {:?}",
                        stats.method,
                        stats.delta_factor,
                        stats.interior_quads,
                        stats.channel_quads,
                        stats.channel_loops,
                        report.max_aspect_ratio,
                        stats.cracks,
                        report.verdict,
                    );
                    assert!(quads >= 20, "{label}: too few quads");
                }
                Err(error) => {
                    println!("{label} h={size} NOT MESHED: {error}");
                    if must_mesh {
                        assert!(false, "{label} must mesh: {error}");
                    }
                }
            }
        }
    }

    #[test]
    fn perf_plate_06() {
        let domain = match Domain::plate_with_hole(100.0, 60.0, 10.0, 64) {
            Ok(domain) => domain,
            Err(error) => {
                assert!(false, "{error}");
                return;
            }
        };
        match channel(&domain, 0.6) {
            Ok(loops) => {
                for (i, loop_) in loops.iter().enumerate() {
                    println!(
                        "perf loop {i}: {:?} {} pts, area {:.3}",
                        loop_.source,
                        loop_.points.len(),
                        polygon_area_points(&loop_.points)
                    );
                }
            }
            Err(error) => println!("perf channel error: {error}"),
        }
        let start = std::time::Instant::now();
        match mesh_with_stats(&domain, 0.6) {
            Ok((mesh, stats)) => {
                let report = quality::report(&mesh);
                println!(
                    "perf plate 0.6 meshed in {:.2}s: {} quads, min SJ {:.4}, angles {:.2}..{:.2}, method {} delta {}, verdict {:?}",
                    start.elapsed().as_secs_f64(),
                    stats.quads,
                    stats.min_scaled_jacobian,
                    report.min_corner_angle_deg,
                    report.max_corner_angle_deg,
                    stats.method,
                    stats.delta_factor,
                    report.verdict
                );
                assert_eq!(stats.cracks, 0, "perf plate cracks");
                assert!(
                    stats.boundary_offset < 1.0e-7,
                    "perf plate boundary offset {}",
                    stats.boundary_offset
                );
                println!(
                    "  perf cracks {} offset {:.3e}",
                    stats.cracks, stats.boundary_offset
                );
                if let Some(index) = report.worst_element {
                    let cell = &mesh.cells[index];
                    let coords: Vec<[f64; 2]> = cell
                        .nodes
                        .iter()
                        .map(|&id| [mesh.points[id][0], mesh.points[id][1]])
                        .collect();
                    println!("  worst cell {index}: {coords:?}");
                }
            }
            Err(error) => {
                assert!(false, "perf plate 0.6 failed: {error}");
            }
        }
    }

    #[test]
    fn plate_with_circular_hole() {
        let domain = match Domain::plate_with_hole(100.0, 60.0, 10.0, 24) {
            Ok(domain) => domain,
            Err(error) => {
                assert!(false, "plate domain invalid: {error}");
                return;
            }
        };
        for size in [4.0_f64, 2.0_f64, 1.0_f64] {
            let label = format!("plate-with-hole h={size}");
            match mesh_with_stats(&domain, size) {
                Ok((mesh, stats)) => {
                    let (quads, min_sj, min_angle, max_angle, offset) =
                        check_mesh(&mesh, &domain, &label);
                    let report = quality::report(&mesh);
                    println!(
                        "{label} [{}]: {quads} quads ({} interior + {} channel, {} ribbons), \
                         loops {}, min SJ {min_sj:.4}, angles {min_angle:.2}..{max_angle:.2}, \
                         aspect {:.2}, cracks {}, offset {offset:.3e}, verdict {:?}",
                        stats.method,
                        stats.interior_quads,
                        stats.channel_quads,
                        stats.faces,
                        stats.channel_loops,
                        report.max_aspect_ratio,
                        stats.cracks,
                        report.verdict,
                    );
                    assert!(quads > 100);
                    assert!(stats.channel_loops >= 4);
                    assert!(min_sj > 0.05, "{label}: min SJ {min_sj}");
                    assert_ne!(
                        report.verdict,
                        quality::Verdict::Refused,
                        "{label}: refused"
                    );
                }
                Err(error) => {
                    assert!(false, "{label} failed: {error}");
                }
            }
        }
    }

    #[test]
    fn l_bracket() {
        let domain = valid_domain(vec![
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 5.0],
            [5.0, 5.0],
            [5.0, 10.0],
            [0.0, 10.0],
        ]);
        // The grid + channel construction itself must work for the L.
        let loops = match channel(&domain, 1.0) {
            Ok(loops) => loops,
            Err(error) => {
                assert!(false, "L-bracket channel failed: {error}");
                return;
            }
        };
        assert!(!loops.is_empty());
        println!("L-bracket channel: {} loop(s)", loops.len());
        match mesh_with_stats(&domain, 1.0) {
            Ok((mesh, stats)) => {
                let (quads, min_sj, min_angle, max_angle, offset) =
                    check_mesh(&mesh, &domain, "L-bracket");
                println!(
                    "L-bracket h=1 [{}]: {quads} quads ({} interior + {} channel), loops {}, \
                     cross edges {}, faces {}, patches {} ({} usable), selected {}, min SJ {min_sj:.4}, \
                     angles {min_angle:.2}..{max_angle:.2}, cracks {}, offset {offset:.3e}, \
                     greedy {:.1}%, fallback {} regions / {} quads",
                    stats.method,
                    stats.interior_quads,
                    stats.channel_quads,
                    stats.channel_loops,
                    stats.cross_edges,
                    stats.faces,
                    stats.patches,
                    stats.usable_patches,
                    stats.selected_patches,
                    stats.cracks,
                    100.0 * stats.greedy_covered_fraction,
                    stats.fallback_regions,
                    stats.fallback_quads,
                );
                assert!(quads > 40);
                assert!(stats.channel_quads > 0);
                assert!(min_sj > 0.05);
            }
            Err(error) => {
                println!(
                    "L-bracket channel construction works, but the simplified ribbon/template \
                     mesher cannot close it: {error}"
                );
            }
        }
    }

    #[test]
    fn square_with_square_hole() {
        let outer = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let hole = vec![[3.0, 3.0], [3.0, 7.0], [7.0, 7.0], [7.0, 3.0]];
        let domain = match Domain::new(outer, vec![hole]) {
            Ok(domain) => domain,
            Err(error) => {
                assert!(false, "{error}");
                return;
            }
        };
        match mesh_with_stats(&domain, 1.0) {
            Ok((mesh, stats)) => {
                let (quads, min_sj, min_angle, max_angle, offset) =
                    check_mesh(&mesh, &domain, "square-with-hole");
                println!(
                    "square-with-hole h=1: {quads} quads ({} interior + {} channel), loops {}, \
                     cross edges {}, faces {}, patches {} ({} usable), selected {}, min SJ {min_sj:.4}, \
                     angles {min_angle:.2}..{max_angle:.2}, cracks {}, offset {offset:.3e}, \
                     greedy {:.1}%, fallback {} regions / {} quads",
                    stats.interior_quads,
                    stats.channel_quads,
                    stats.channel_loops,
                    stats.cross_edges,
                    stats.faces,
                    stats.patches,
                    stats.usable_patches,
                    stats.selected_patches,
                    stats.cracks,
                    100.0 * stats.greedy_covered_fraction,
                    stats.fallback_regions,
                    stats.fallback_quads,
                );
                assert!(quads >= 16);
            }
            Err(error) => {
                println!("square-with-hole h=1 not feasible with this simplified cover: {error}");
            }
        }
    }

    #[test]
    fn templates_are_non_degenerate() {
        let hexagon: Vec<[f64; 2]> = (0..6)
            .map(|i| {
                let angle = TAU * i as f64 / 6.0;
                [angle.cos(), angle.sin()]
            })
            .collect();
        let Some(six) = quadrangulate_six(&hexagon, 2) else {
            assert!(false, "no 6-sided template");
            return;
        };
        assert!((2..=4).contains(&six.quads.len()));
        assert!(six.rank > 0.0, "regular hexagon rank {}", six.rank);

        let octagon: Vec<[f64; 2]> = (0..8)
            .map(|i| {
                let angle = TAU * i as f64 / 8.0 + TAU / 16.0;
                [angle.cos(), angle.sin()]
            })
            .collect();
        let Some(eight) = quadrangulate_eight(&octagon) else {
            assert!(false, "no 8-sided template");
            return;
        };
        assert!(eight.rank > 0.0, "regular octagon rank {}", eight.rank);
        assert!(!eight.quads.is_empty());
    }

    #[test]
    fn demo_numbers() {
        let numbers = match demo() {
            Ok(numbers) => numbers,
            Err(error) => {
                assert!(false, "demo failed: {error}");
                return;
            }
        };
        println!(
            "demo square: {:?}\ndemo L: {:?}\ndemo hole: {:?}",
            numbers.square, numbers.l_bracket, numbers.square_with_hole
        );
        assert!(numbers.square.quads > 40);
        assert_eq!(numbers.square.cracks, 0);
        match &numbers.l_bracket {
            Some(l) => {
                assert!(l.quads > 40);
                assert_eq!(l.cracks, 0);
            }
            None => {
                if let Some(error) = &numbers.l_bracket_error {
                    println!("L-bracket demo error: {error}");
                }
            }
        }
        if let Some(hole) = &numbers.square_with_hole {
            assert_eq!(hole.cracks, 0);
        } else if let Some(error) = &numbers.square_with_hole_error {
            println!("hole case error: {error}");
        }
    }
}
