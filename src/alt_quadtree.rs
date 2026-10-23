//! Alternative all-quad planar route: quadtree–geometry intersection, no cleanup.
//!
//! This is the route of corpus papers **`rushdi-2017`** (Rushdi, Mitchell,
//! Mahmoud, Bajaj and Ebeida, *All-quad meshing without cleanup*,
//! Computer-Aided Design 84 (2017) 1–16) and its IMR 2015 precursor
//! **`rushdi-2015`** (Rushdi et al., *Robust all-quad meshing of domains with
//! connected regions*, Procedia Engineering 124 (2015) 96–108).  The sections
//! implemented here are the grid refinement of §2.1, the repelling of §2.2,
//! polygon formation and midpoint subdivision of §2.3, and the two-refinement
//! parity and template pass of §2.4.  The steps are:
//!
//! 1. **grid refinement to the size field** (§2.1): a quadtree over the
//!    bounding box whose leaves are refined until the leaf diagonal is at most
//!    [`DEFAULT_CELL_FACTOR`] times the requested size (`1.0` is the paper's
//!    density; the factor is the element-count knob).  The default is `1.5`,
//!    which puts the effective edge at `1.06 * size` and matches the paving
//!    element count at the nominal size.  The sweep in
//!    `examples/alt_quadtree.rs` documents `2.0` as the **coarse option**
//!    (largest swept factor that still passes every gate, edge `1.41 * size`,
//!    about 1.7x fewer elements); `1.0` is the conservative paper density.
//!    With the uniform [`Sizing`] of this crate the strongly balanced
//!    quadtree is exactly the uniform grid it refines to, so the
//!    implementation walks that grid directly;
//! 2. **repelling** (§2.2): grid corners closer than `delta = 0.40 s` (the
//!    local square size) to the geometry are moved away from it along a grid
//!    line, so a grid/geometry intersection can never sit next to a corner and
//!    make a flat or tiny angle.  The paper analyses `s/4`; with that value the
//!    measured minimum corner angle over the corpus was 23–25 degrees, below
//!    the 30-degree gate.  The larger `0.40` is a measured tuning choice
//!    ([`DELTA_FRACTION`]): it lifts the minimum above the gate while keeping
//!    the scaled Jacobian above 0.5;
//! 3. **polygon formation** (§2.3): every leaf cut by the boundary is split
//!    into the polygons that conform to the geometry, using
//!    [`Domain::contains`] and [`Domain::segments`].  A smooth run of the
//!    boundary inside a leaf is represented by one straight chord, and the
//!    chord's subdivision midpoint is snapped to the run's arc-length point on
//!    the geometry.  This avoids the flat polygon vertex that a polygonal
//!    approximation of a smooth curve would otherwise produce at every
//!    sampling point, while still putting a boundary node exactly on the
//!    geometry (so `boundary_offset` stays at round-off);
//! 4. **midpoint subdivision** (§2.3): each polygon is split into one quad per
//!    edge by a centre and the edge midpoints;
//! 5. **2-refinement transition** (§2.4): the untouched leaves that receive a
//!    hanging node from a split leaf are marked, a parity/marking pass pairs
//!    every odd leaf with another one through untouched leaves, and each leaf
//!    is meshed with the two-refinement templates: 1 quad (0 hanging nodes),
//!    2 quads (two opposite), 3 quads (two adjacent) or 4 quads (four).  No
//!    hanging node survives, so the output is conforming without pillowing,
//!    edge swapping or smoothing.
//!
//! Measured result (`examples/alt_quadtree.rs`, default options): 7/7 corpus
//! parts are all-quad and gate-passing **raw and after the common post-pass**
//! (post min scaled Jacobian 0.63–1.00, post corner 42.9–90 degrees; 14 900
//! quads on the 0.6 mm performance part in 0.16 s raw).  The raw corner-angle
//! band over the corpus is 35.7–145.0 degrees.
//!
//! Robustness.  The alignment of the grid lines relative to the boundary
//! vertices decides whether a nearly-tangent piece of boundary becomes a thin
//! cap.  The route therefore tries [`ALIGNMENTS`] (= 3) deterministic phase
//! shifts of the grid lines, scores each candidate with [`mesh_score`] and
//! keeps the best.  If every alignment fails to assemble a conforming mesh,
//! the whole grid is rebuilt at half the target spacing, up to
//! [`MAX_ATTEMPTS`] levels.  A sharp
//! re-entrant corner inside a leaf (not reachable on this corpus, where
//! axis-aligned corners sit on grid lines) is split by an interior diagonal.
//!
//! What is **not** implemented: two-sided domains, and the dedicated sharp
//! vertex template of §2.3.1.  Axis-aligned sharp corners are placed on grid
//! lines, where their angle is naturally split between the cells, so a
//! re-entrant corner never becomes one reflex polygon.  A non-axis-aligned
//! sharp corner whose apex falls inside one leaf is *attempted* by the generic
//! diagonal split of [`split_bad_vertices`]; if no visible diagonal exists the
//! attempt fails and another alignment or level is tried, rather than falling
//! back to the paper's corner template.  The quadtree of the paper is strongly
//! balanced and adaptive; the size field of this crate is uniform, so the
//! balanced tree collapses to the uniform grid of item 1 and is walked
//! directly, with no per-leaf refinement pass.
//!
//! **Product limitation: grid-aligned, isotropic mesh.**  This is a
//! Cartesian/quadtree route, so the mesh follows the background grid.  A
//! curved boundary is approximated by axis-aligned cell sides plus short
//! chords whose nodes lie on the input polyline; there is no tangential
//! projection of whole rows and no anisotropic boundary layer.  For CFD
//! boundary layers, strongly curved geometry, or a mesh whose edges must be
//! aligned with a wall normal, an advancing-front/paving or an anisotropic
//! layer route is the right tool — this is a deliberate limitation of the
//! grid-based method, not a conformity or cleanup bug (the boundary nodes are
//! on the geometry and `boundary_offset` is round-off).
//!
//! The module never panics and returns `Err` for configurations it cannot
//! resolve (it retries at half the spacing, up to three levels, before giving
//! up).

use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::mesh::{Cell, Mesh, Node};
use crate::two_d::{Domain, Sizing};

/// How many grid alignments (grid-line phase shifts) are tried per attempt.
///
/// The search is deterministic: for attempt `level`, alignment `k` uses phase
/// seed `level * ALIGNMENTS + k` in [`line_set`].  Three phases were enough for
/// the corpus; this is a tuning choice, not a paper constant.
const ALIGNMENTS: usize = 3;
/// How many times the whole grid is rebuilt at half the spacing before failing.
///
/// Attempt `level` uses target spacing `cell_factor * size / sqrt(2) / 2^level`
/// (see [`mesh_with_options`]), so the effective leaf edge halves per level.
/// Three levels are the work guard; the corpus converges on the first.
const MAX_ATTEMPTS: usize = 3;
/// Coordinates are deduplicated at 10 nm resolution.
const KEY_SCALE: f64 = 1.0e8;
/// Default grid-density factor: the leaf diagonal is at most
/// `DEFAULT_CELL_FACTOR * size`.
///
/// The product contract is "mesh at the requested size", so the default is
/// `1.5`: its effective edge is `1.06 * size` and its element counts match
/// the paving route at the nominal size.  The paper's "diagonal at most the
/// size" is the conservative `1.0` (edge `0.71 * size`).  The sweep in
/// `examples/alt_quadtree.rs` shows that quality does **not** collapse when
/// the grid is coarsened: `2.0` is the largest swept factor that still keeps
/// the whole corpus all-quad, gap-free, `min angle >= 30` and `SJ >= 0.2`,
/// and it removes a further 1.7x elements (e.g. 8 459 quads on the 0.6 mm
/// part instead of 14 900), at an effective edge of about `1.41 * size`.  It
/// remains available as the documented **coarse option** through
/// [`mesh_with_cell_factor`].
pub const DEFAULT_CELL_FACTOR: f64 = 1.5;

/// `delta = repelling fraction * local square side`.  The paper analyses `s/4`
/// (worst corner around 24.3 degrees); with `s/4` the measured minimum over
/// this corpus was 23–25 degrees, below the 30-degree gate.  `0.40` is the
/// measured replacement: it lifts the minimum above the gate while keeping the
/// scaled Jacobian above 0.5 on every part of the corpus.  Tuning choice.
const DELTA_FRACTION: f64 = 0.40;
/// A polygon corner whose interior angle exceeds this is split by a diagonal.
///
/// The paper's sharp-corner discussion is §2.3; the threshold is a tuning
/// choice that catches both a reflex re-entrant corner (> 180 degrees) and the
/// nearly flat vertices a polygonal boundary introduces (160–180 degrees).
const BAD_ANGLE_DEG: f64 = 160.0;
/// The two halves of a split must not become slivers themselves.
///
/// A candidate diagonal is rejected when either corner it creates is below
/// this angle; 12 degrees is a tuning guard, well beyond the 30-degree gate
/// because the quad subdivision that follows can only shrink a corner.
const MIN_SPLIT_ANGLE_DEG: f64 = 12.0;
/// Geometric tolerance for point/segment identities, in model units.
///
/// Used for dimensionless comparisons (coordinate equality, turn tests); the
/// geometric tolerance that scales with the model is `1.0e-7 * self.scale()`
/// in [`Builder::snap_to_cell_boundary`] and friends.
const EPS: f64 = 1.0e-9;
/// Upper bound on diagonal splits per cut cell.
///
/// A guard against a polygon that splits forever; on the corpus one or two
/// splits suffice and the limit is never reached.
const MAX_SPLITS: usize = 32;
/// A chain turn gentler than this is smoothed away.
///
/// The run becomes one chord whose subdivision midpoint is snapped to the run's
/// arc-length point, instead of one flat polygon vertex per sampling point.
/// The value is a tuning choice between the 30-degree gate and the sampling
/// of a curved input (a smooth arc sampled at 5–10 degrees of turn must not
/// keep a vertex per sample, while a real corner of 30 degrees or more must).
const SMOOTH_TURN_DEG: f64 = 25.0;

/// Options for the quadtree route.
///
/// [`Options::default`] uses the constants the route was tuned with, so the
/// default output is the measured table of `examples/alt_quadtree.rs`.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Options {
    /// Grid-density factor: every leaf diagonal is at most
    /// `cell_factor * sizing.size`.  `1.0` is the paper's density and larger
    /// values coarsen the mesh; must lie in `[0.25, 4]`.
    ///
    /// Invariant: the effective leaf edge is `cell_factor * size / sqrt(2)`;
    /// the default 1.5 therefore gives an edge of `1.06 * size` and the
    /// element counts of the paving route at the nominal size.  This is the
    /// knob the `sweep` mode of `examples/alt_quadtree.rs` measures.
    pub cell_factor: f64,

    /// How many deterministic grid alignments are tried per attempt before the
    /// mesh with the best measured gate score is accepted.  Must be at least 1.
    ///
    /// Invariant: `alignments * max_attempts` builds are attempted in the
    /// worst case; `1` disables the search and takes the first conforming
    /// mesh.
    pub alignments: usize,

    /// Repelling distance as a fraction of the local square side; `0.0` turns
    /// repelling off around a grid corner.
    ///
    /// The paper analyses `s/4`; the default is `0.40` because `s/4` leaves
    /// 23–25 degree corners on the corpus (below the 30-degree gate).  Values
    /// are not range-checked: a value large enough to cross a whole cell makes
    /// [`repel`] reject the attempt as folded, which the outer loop retries.
    pub delta_fraction: f64,

    /// How many times the whole grid is rebuilt at half the spacing before the
    /// route fails.  Must lie in `[1, 32]`.
    pub max_attempts: usize,
}

/// The measured configuration: [`DEFAULT_CELL_FACTOR`], [`ALIGNMENTS`],
/// [`DELTA_FRACTION`] and [`MAX_ATTEMPTS`].  Every field is a plain value, so
/// `Options { cell_factor: 2.0, ..Options::default() }` keeps the rest.
impl Default for Options {
    /// The measured defaults; the values are documented on the constants and
    /// on the fields of [`Options`].
    fn default() -> Self {
        Self {
            cell_factor: DEFAULT_CELL_FACTOR,
            alignments: ALIGNMENTS,
            delta_fraction: DELTA_FRACTION,
            max_attempts: MAX_ATTEMPTS,
        }
    }
}

/// Mesh a planar domain with the cleanup-free quadtree route.
///
/// Uses [`Options::default`].  This is the entry point of `Route::Quadtree` in
/// [`crate::config`].  Returns the best gate-scoring candidate mesh of the
/// alignment/attempt search; returns `Err` (never panics) if no candidate is
/// conforming, if the sizing is invalid, or if the domain is degenerate.
pub fn mesh(domain: &Domain, sizing: &Sizing) -> Result<Mesh, String> {
    mesh_with_options(domain, sizing, &Options::default())
}

/// Mesh with an explicit grid-density factor: every leaf diagonal is at most
/// `cell_factor * sizing.size`.  `1.0` is the paper's density; larger values
/// coarsen the mesh.  The factor also scales the repelling distance, so the
/// quality band is preserved as far as the corpus allows.
///
/// This is the documented **coarse option**: the sweep in
/// `examples/alt_quadtree.rs` shows that `2.0` is the largest swept factor
/// that still keeps the corpus 7/7 raw PASS (edge about `1.41 * size`, about
/// 1.7x fewer elements).  Returns the same errors as [`mesh_with_options`].
pub fn mesh_with_cell_factor(
    domain: &Domain,
    sizing: &Sizing,
    cell_factor: f64,
) -> Result<Mesh, String> {
    mesh_with_options(
        domain,
        sizing,
        &Options {
            cell_factor,
            ..Options::default()
        },
    )
}

/// Mesh a planar domain with the cleanup-free quadtree route, using explicit
/// [`Options`].
///
/// Validation is strict and happens before any grid work: `cell_factor` in
/// `[0.25, 4]`, at least one alignment, `max_attempts` in `[1, 32]`, a
/// positive finite size, an outer loop of at least three points and a
/// non-degenerate bounding box.
///
/// Search: for each attempt level (target spacing halved per level) every
/// alignment is built by [`build`]; a conforming candidate is scored by
/// [`mesh_score`] and the best is accepted immediately.  Only if every
/// alignment of every level fails is `Err` returned, carrying the last build
/// error.  Complexity is `O(cells log cells)` for the line sets, plus the
/// per-cell clipping; the memory guard rejects a grid above four million
/// nodes.
pub fn mesh_with_options(
    domain: &Domain,
    sizing: &Sizing,
    options: &Options,
) -> Result<Mesh, String> {
    if !options.cell_factor.is_finite() || !(0.25..=4.0).contains(&options.cell_factor) {
        return Err("the cell factor must be in [0.25, 4]".to_string());
    }
    if options.alignments == 0 {
        return Err("at least one grid alignment is required".to_string());
    }
    if !(1..=32).contains(&options.max_attempts) {
        return Err("the number of attempts must be in [1, 32]".to_string());
    }
    if !sizing.size.is_finite() || sizing.size <= 0.0 {
        return Err("the element size must be positive and finite".to_string());
    }
    if domain.outer.len() < 3 {
        return Err("the outer loop needs at least three points".to_string());
    }
    if domain.area() <= 0.0 {
        return Err("the domain has no area".to_string());
    }
    let bbox = bounding_box(domain)?;
    let width = bbox[2] - bbox[0];
    let height = bbox[3] - bbox[1];
    if !(width.is_finite() && height.is_finite()) || width <= 0.0 || height <= 0.0 {
        return Err("the domain has a degenerate bounding box".to_string());
    }

    let axis_vertices = axis_aligned_corners(domain);
    let vertices = all_vertices(domain);
    let mut last = String::from("no attempt made");
    // The grid alignment (where the lines fall relative to the boundary
    // vertices) decides whether a boundary that is locally tangent to a grid
    // line produces a thin sliver.  Try a few deterministic alignments and
    // keep the best mesh; refine the whole grid only if every alignment fails
    // to assemble a conforming mesh.
    for level in 0..options.max_attempts {
        let target =
            options.cell_factor * sizing.size / std::f64::consts::SQRT_2 / (1u32 << level) as f64;
        let mut best: Option<(f64, Mesh)> = None;
        for variant in 0..options.alignments {
            let variant = level * options.alignments + variant;
            match build(
                domain,
                target,
                &bbox,
                &axis_vertices,
                &vertices,
                options,
                variant,
            ) {
                Ok(candidate) => {
                    let score = mesh_score(&candidate);
                    if best.as_ref().is_none_or(|(s, _)| score > *s) {
                        best = Some((score, candidate));
                    }
                }
                Err(e) => last = e,
            }
        }
        if let Some((_, mesh)) = best {
            return Ok(mesh);
        }
    }
    Err(format!(
        "quadtree route could not build a conforming mesh: {last}"
    ))
}

/// Score a candidate mesh: gate-passing candidates dominate, and among them a
/// balanced minimum Jacobian/angle is preferred.
///
/// `base = min(min scaled Jacobian, min corner angle / 90)` lies in `[0, 1]`
/// for a valid mesh, so adding 10 for a passing gate makes every passing
/// candidate beat every failing one.  Ties keep the earlier alignment because
/// the comparison is strict (`score > best`).  The score is only an ordering,
/// not a quality report.
fn mesh_score(mesh: &Mesh) -> f64 {
    let gate = crate::verify::gate(mesh);
    let base = gate
        .min_scaled_jacobian
        .min(gate.min_corner_angle_deg / 90.0);
    if gate.passes() {
        10.0 + base
    } else {
        base
    }
}

/// The bounding box of the outer loop: `[xmin, ymin, xmax, ymax]`.
///
/// Only the outer loop is scanned (holes are inside it).  Returns `Err` on a
/// non-finite coordinate.  The caller separately rejects a zero width or
/// height.
fn bounding_box(domain: &Domain) -> Result<[f64; 4], String> {
    let mut b = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for p in &domain.outer {
        if !p[0].is_finite() || !p[1].is_finite() {
            return Err("a boundary point is not finite".to_string());
        }
        b[0] = b[0].min(p[0]);
        b[1] = b[1].min(p[1]);
        b[2] = b[2].max(p[0]);
        b[3] = b[3].max(p[1]);
    }
    Ok(b)
}

/// Every boundary vertex of every loop (outer first, then the holes).
///
/// Used by [`line_set`] to keep polygonal input vertices off grid lines.
fn all_vertices(domain: &Domain) -> Vec<[f64; 2]> {
    let mut out = Vec::new();
    out.extend_from_slice(&domain.outer);
    for hole in &domain.holes {
        out.extend_from_slice(hole);
    }
    out
}

/// Vertices whose two incident segments are both axis-aligned and do not form a
/// straight line.  They are put exactly on grid lines, so the boundary follows
/// those lines and the corner angle is split between the adjacent cells.
///
/// This is the implementation of the paper's §2.3 remark that an axis-aligned
/// sharp corner should fall on a grid line rather than inside a leaf: the
/// corner then becomes a grid node whose four incident cells each see a right
/// angle.  `EPS` is used for the axis test and `1 - 1e-9` for the
/// straight-through test, so duplicated or collinear input does not create a
/// spurious line.
fn axis_aligned_corners(domain: &Domain) -> Vec<[f64; 2]> {
    let mut out = Vec::new();
    for loop_points in std::iter::once(&domain.outer).chain(domain.holes.iter()) {
        let n = loop_points.len();
        if n < 3 {
            continue;
        }
        for i in 0..n {
            let prev = loop_points[(i + n - 1) % n];
            let here = loop_points[i];
            let next = loop_points[(i + 1) % n];
            let d1 = sub(here, prev);
            let d2 = sub(next, here);
            let axis1 = d1[0].abs() <= EPS || d1[1].abs() <= EPS;
            let axis2 = d2[0].abs() <= EPS || d2[1].abs() <= EPS;
            if !(axis1 && axis2) {
                continue;
            }
            let l1 = norm(d1);
            let l2 = norm(d2);
            if l1 <= EPS || l2 <= EPS {
                continue;
            }
            let cosine = dot(d1, d2) / (l1 * l2);
            // A straight-through vertex (cos ≈ 1) is not a corner.
            if cosine > 1.0 - 1.0e-9 {
                continue;
            }
            out.push(here);
        }
    }
    out
}

/// Classification of one leaf of the (uniform) quadtree.
///
/// Set in pass 1 of [`build`].  The variant drives pass 2 (hanging nodes),
/// [`Builder::resolve_parity`] and the template choice in
/// [`Builder::full_cell`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CellKind {
    /// No part of the leaf is inside the domain.
    Outside,

    /// The whole leaf is inside and untouched by the boundary; it may receive
    /// hanging nodes and is meshed by the 2-refinement templates.
    Full,

    /// The boundary crosses the leaf; it is meshed by polygon formation and
    /// midpoint subdivision ([`Builder::mesh_cut_cell`]).
    Cut,
}

/// A boundary segment together with its position in the domain loops.
///
/// In [`Builder::segs`] the endpoints follow the stored loop order (`a -> b` is
/// the direction of the input loop).  The segments of [`Builder::clip_cell`]
/// are the same straight pieces clipped to a leaf; they are re-snapped to the
/// leaf boundary so the two leaves sharing a side compute bit-identical
/// endpoints.
#[derive(Clone, Copy, Debug)]
struct DomainSeg {
    /// Start point of the directed segment, in model units.
    a: [f64; 2],

    /// End point of the directed segment, in model units.
    b: [f64; 2],
}

/// The grid plus all accumulated mesh state of one attempt.
///
/// One `Builder` is created per (attempt, alignment) pair by [`build`].  Grid
/// nodes come first in [`Builder::points`] and keep their `gid` for the whole
/// attempt; every other node (boundary point, polygon vertex, edge midpoint,
/// polygon centre) is appended and deduplicated by coordinate through
/// [`Builder::pool`].
struct Builder<'a> {
    /// The input domain; only read (containment, nearest segment, distance).
    domain: &'a Domain,

    /// Sorted x grid lines, length `nx + 1`; strictly increasing.
    xs: Vec<f64>,

    /// Sorted y grid lines, length `ny + 1`; strictly increasing.
    ys: Vec<f64>,

    /// Number of grid cells in x (`xs.len() - 1`).
    nx: usize,

    /// Number of grid cells in y (`ys.len() - 1`).
    ny: usize,

    /// Moved positions of the grid nodes, indexed by `gid`.
    ///
    /// This is the output of [`repel`]; `pos[gid]` differs from
    /// `(xs[i], ys[j])` only where repelling moved the node, and the move is
    /// at most `2 * delta` away from the grid position (Euclidean).
    pos: Vec<[f64; 2]>,

    /// All mesh nodes; the first `(nx + 1) * (ny + 1)` are the grid nodes.
    ///
    /// Node `gid = j * (nx + 1) + i` holds `pos[gid]`; later nodes are added by
    /// [`Builder::add_node`].  Unused nodes are compacted away in
    /// [`Builder::finish`].
    points: Vec<Node>,

    /// Coordinate deduplication map: 10 nm key ([`pack`]) to node id.
    ///
    /// Invariant: two entries with the same key are the same node, so a node
    /// shared by two leaves (a cell side midpoint, a boundary intersection) is
    /// created once and the mesh is conforming by construction.
    pool: HashMap<(i64, i64), usize>,

    /// The domain boundary, one directed segment per input loop edge.
    ///
    /// Includes the closing edge of every loop; holes are included, so a hole
    /// inside the grid is treated exactly like the outer boundary.
    segs: Vec<DomainSeg>,

    /// Grid edges whose midpoint is a mesh node (a hanging node for the leaf
    /// on the other side).
    ///
    /// A key `(min(gid0, gid1), max(...))` is present when the midpoint of that
    /// grid edge is used as a node by a cut leaf or by the parity marking, and
    /// the untouched leaf on the other side must therefore use a 2-refinement
    /// template.  Invariant after [`Builder::resolve_parity`]: every untouched
    /// leaf has an even number of split sides.
    split: HashSet<(usize, usize)>,

    /// Quads of the whole attempt, filled from `cells_quads` in
    /// [`Builder::finish`].
    quads: Vec<[usize; 4]>,

    /// Per-leaf classification, indexed by `j * nx + i`, length `nx * ny`.
    kind: Vec<CellKind>,

    /// Per cut leaf, the quads its polygons produced (indexed like `kind`).
    cut_quads: Vec<Vec<[usize; 4]>>,

    /// Per cut leaf, the cell sides `k` that its polygon uses in full.
    ///
    /// `k` is a side index in the counter-clockwise order `0 = bottom,
    /// 1 = right, 2 = top, 3 = left`.  A side in this list means the leaf
    /// contributes a hanging node to its neighbour; see pass 2 of [`build`].
    cut_sides: Vec<Vec<usize>>,

    /// Per leaf, the final quads selected for the mesh (indexed like `kind`).
    cells_quads: Vec<Vec<[usize; 4]>>,
}

/// Build one attempt: grid lines, repelling, leaf classification, cut-cell
/// meshing, hanging-node marking, parity resolution and template assignment.
///
/// `target` is the requested leaf diagonal (already includes
/// `cell_factor / sqrt(2)` and the halving of the attempt level).  `variant`
/// selects the deterministic grid phase in [`line_set`].  Returns `Err` for a
/// degenerate target, a degenerate grid, a collapsed node (two grid nodes at
/// the same 10 nm key), a cut cell that is not one simply-connected chain, a
/// template failure, or a final T-junction/crack (see [`Builder::finish`]).
/// The caller treats any `Err` as one failed alignment.
fn build(
    domain: &Domain,
    target: f64,
    bbox: &[f64; 4],
    axis_vertices: &[[f64; 2]],
    vertices: &[[f64; 2]],
    options: &Options,
    variant: usize,
) -> Result<Mesh, String> {
    if target <= 0.0 || !target.is_finite() {
        return Err("the target size is not positive".to_string());
    }
    let xs = line_set(
        bbox[0],
        bbox[2],
        target,
        axis_vertices,
        true,
        variant,
        vertices,
    )?;
    let ys = line_set(
        bbox[1],
        bbox[3],
        target,
        axis_vertices,
        false,
        variant,
        vertices,
    )?;
    let nx = xs.len().saturating_sub(1);
    let ny = ys.len().saturating_sub(1);
    if nx == 0 || ny == 0 {
        return Err("the grid has no cells".to_string());
    }
    let n_nodes = (nx + 1) * (ny + 1);
    if n_nodes > 4_000_000 {
        return Err("the requested size needs too many grid nodes".to_string());
    }

    let grid_xy: Vec<[f64; 2]> = (0..n_nodes)
        .map(|id| {
            let i = id % (nx + 1);
            let j = id / (nx + 1);
            [xs[i], ys[j]]
        })
        .collect();
    let pos = repel(domain, &xs, &ys, &grid_xy, nx, ny, options)?;

    let mut points = Vec::with_capacity(n_nodes);
    let mut pool = HashMap::with_capacity(n_nodes * 2);
    for p in &pos {
        let id = points.len();
        let key = pack(*p);
        if pool.insert(key, id).is_some() {
            return Err("two grid nodes collapse to one point".to_string());
        }
        points.push([p[0], p[1], 0.0]);
    }

    let mut builder = Builder {
        domain,
        xs,
        ys,
        nx,
        ny,
        pos,
        points,
        pool,
        segs: domain_segments(domain),
        split: HashSet::new(),
        quads: Vec::new(),
        kind: vec![CellKind::Outside; nx * ny],
        cut_quads: vec![Vec::new(); nx * ny],
        cut_sides: vec![Vec::new(); nx * ny],
        cells_quads: vec![Vec::new(); nx * ny],
    };

    // --- pass 1: classify every leaf and mesh the cut leaves -------------
    for j in 0..ny {
        for i in 0..nx {
            let cell = j * nx + i;
            let quad = builder.cell_quad(i, j);
            let clipped = builder.clip_cell(&quad);
            if clipped.is_empty() {
                // No boundary piece crosses the leaf: a centre test decides
                // Full/Outside (the leaf may still be a hole's inside).
                let centre = average4(&quad);
                if builder.inside(centre, quad[0]) {
                    builder.kind[cell] = CellKind::Full;
                } else {
                    builder.kind[cell] = CellKind::Outside;
                }
            } else {
                let (quads, sides) = builder.mesh_cut_cell(i, j, &quad, &clipped)?;
                if quads.is_empty() {
                    // Degenerate classification: no inside area.
                    builder.kind[cell] = CellKind::Outside;
                } else {
                    builder.kind[cell] = CellKind::Cut;
                    builder.cut_quads[cell] = quads;
                    builder.cut_sides[cell] = sides;
                }
            }
        }
    }

    // --- pass 2: hanging nodes on untouched leaves ------------------------
    for j in 0..ny {
        for i in 0..nx {
            let cell = j * nx + i;
            if builder.kind[cell] != CellKind::Cut {
                continue;
            }
            let gid = builder.cell_gids(i, j);
            for &side in &builder.cut_sides[cell] {
                let (n0, n1) = (gid[side], gid[(side + 1) % 4]);
                if let Some(neighbour) = builder.neighbour(i, j, side) {
                    // Only a Full neighbour needs a 2-ref template; a Cut
                    // neighbour already has the boundary node in its polygon.
                    if builder.kind[neighbour] == CellKind::Full {
                        builder.split.insert(edge_key(n0, n1));
                    }
                }
            }
        }
    }

    // --- pass 3: parity marking and 2-ref templates on untouched leaves ---
    // Cut leaves already carry their quads; Full leaves are meshed only now,
    // after parity has guaranteed an even number of hanging sides.
    builder.resolve_parity()?;
    for j in 0..ny {
        for i in 0..nx {
            let cell = j * nx + i;
            match builder.kind[cell] {
                CellKind::Outside => {}
                CellKind::Cut => builder.cells_quads[cell] = builder.cut_quads[cell].clone(),
                CellKind::Full => builder.cells_quads[cell] = builder.full_cell(i, j)?,
            }
        }
    }
    builder.finish()
}

/// Build the sorted grid lines of one direction.
///
/// The uniform lines are gently displaced (with a sine weight, so the border
/// cells keep their size) to stop a boundary vertex from landing exactly on a
/// line.  Lines through axis-aligned sharp corners are then inserted exactly,
/// dropping uniform lines that would make a sliver next to them.
///
/// `n = ceil(span / target)` and `spacing = span / n`, so `spacing <= target`.
/// The displacement is `phase * spacing * sin(pi * k / n)` with
/// `phase = 0.06 + 0.055 * ((seed + variant) % 8)`: eight phases in `[0.06,
/// 0.445]`, zero at both ends (the border lines are not moved) and maximal at
/// the middle.  A seed is preferred when every polygonal vertex is at least
/// `1e-7 * span` from every interior line and every axis-aligned sharp corner
/// is more than `0.05 * spacing` away; otherwise the seed with the largest
/// corner clearance is kept.  The forced corner lines are added exactly and a
/// uniform line within `0.5 * spacing` of a forced line is dropped (that would
/// be a sliver cell).  The result is strictly increasing, with at least two
/// lines; otherwise `Err`.
fn line_set(
    min: f64,
    max: f64,
    target: f64,
    axis_vertices: &[[f64; 2]],
    is_x: bool,
    variant: usize,
    vertices: &[[f64; 2]],
) -> Result<Vec<f64>, String> {
    let span = max - min;
    if !(span.is_finite() && span > 0.0) {
        return Err("degenerate bounding box".to_string());
    }
    let n = ((span / target).ceil() as usize).max(1);
    let spacing = span / n as f64;
    let mut best: Option<(f64, Vec<f64>)> = None;
    for seed in 0..8 {
        let phase = 0.06 + 0.055 * ((seed + variant) % 8) as f64;
        let mut lines = Vec::with_capacity(n + 8);
        lines.push(min);
        for k in 1..n {
            let w = (std::f64::consts::PI * k as f64 / n as f64).sin();
            lines.push(min + k as f64 * spacing + phase * spacing * w);
        }
        lines.push(max);
        let score = vertex_clearance(&lines, vertices, is_x);
        let sharp_score = vertex_clearance(&lines, axis_vertices, is_x);
        // Take the first seed that clears both clearances; otherwise keep the
        // seed with the largest corner clearance and continue the search.
        if sharp_score > 0.05 * spacing && score > 1.0e-7 * span {
            best = Some((sharp_score, lines));
            break;
        }
        if best.as_ref().is_none_or(|(s, _)| sharp_score > *s) {
            best = Some((sharp_score, lines));
        }
    }
    let (_, mut lines) = best.ok_or_else(|| "could not place the grid lines".to_string())?;

    // Exact lines through axis-aligned sharp corners.
    let mut forced: Vec<f64> = axis_vertices
        .iter()
        .map(|p| if is_x { p[0] } else { p[1] })
        .collect();
    forced.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    forced.dedup_by(|a, b| (*a - *b).abs() <= 1.0e-9);
    if !forced.is_empty() {
        let original = lines.clone();
        let mut kept: Vec<f64> = original
            .iter()
            .copied()
            .filter(|v| {
                (*v - min).abs() <= 1.0e-12
                    || (*v - max).abs() <= 1.0e-12
                    || forced.iter().all(|f| (*v - *f).abs() > 0.5 * spacing)
            })
            .collect();
        kept.extend(forced);
        kept.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        kept.dedup_by(|a, b| (*a - *b).abs() <= 1.0e-9);
        lines = kept;
    }
    if lines.len() < 2 {
        return Err("the grid lines collapsed".to_string());
    }
    for w in lines.windows(2) {
        if w[1] <= w[0] {
            return Err("the grid lines are not strictly increasing".to_string());
        }
    }
    Ok(lines)
}

/// Smallest distance of the given vertices to any interior line, as a share of
/// the grid spacing.  Used to reject a jitter that puts a vertex on a line.
///
/// The end lines are excluded (a vertex on the bounding box is legitimate).
/// Returns 0.0 when there is no vertex or no interior line, and 1.0 when every
/// vertex sits outside the span (the `worst` stays infinite).  The share is
/// dimensionless; the caller compares it with `1e-7 * span` or
/// `0.05 * spacing`.
fn vertex_clearance(lines: &[f64], vertices: &[[f64; 2]], is_x: bool) -> f64 {
    if vertices.is_empty() || lines.len() < 2 {
        return 0.0;
    }
    let spacing = (lines[lines.len() - 1] - lines[0]) / (lines.len() - 1) as f64;
    let mut worst = f64::INFINITY;
    for p in vertices {
        let value = if is_x { p[0] } else { p[1] };
        for &line in &lines[1..lines.len() - 1] {
            worst = worst.min((value - line).abs());
        }
    }
    if worst.is_finite() {
        worst / spacing
    } else {
        1.0
    }
}

/// Domain segments with their loop and segment indices.
///
/// One segment per loop edge, including the closing edge `last -> first`; the
/// outer loop comes first, then each hole.  The direction is the stored loop
/// order, which is what [`Builder::clip_cell`] and the chain walk preserve.
fn domain_segments(domain: &Domain) -> Vec<DomainSeg> {
    let mut out = Vec::new();
    for loop_points in std::iter::once(&domain.outer).chain(domain.holes.iter()) {
        let n = loop_points.len();
        if n < 2 {
            continue;
        }
        for i in 0..n {
            out.push(DomainSeg {
                a: loop_points[i],
                b: loop_points[(i + 1) % n],
            });
        }
    }
    out
}

/// Move grid corners away from the geometry (grid-aligned repelling, §2.2).
///
/// For each grid node the allowed displacement is the **minimum** over its
/// incident leaves of `delta_fraction * min(width, height)` (the local square
/// side), so a node at a size change keeps the smaller allowance.  A node is
/// moved only when its distance to the nearest domain segment is in
/// `(EPS, delta)`; the move is along the grid-aligned direction of
/// [`repel_point`].  The move is accepted only if the distance to that segment
/// strictly increased and the displacement is at most `2 * delta` (a guard
/// against the segment changing under the projection).  Finally every leaf of
/// the moved grid must stay convex; otherwise the attempt is rejected with
/// `Err("repelling folded a grid cell")`, and the caller retries another
/// alignment or a finer grid.
///
/// With the paper's `s/4` the measured 30-degree gate failed at 23–25 degrees;
/// the default `delta_fraction = 0.40` is the measured replacement (see
/// [`DELTA_FRACTION`]).  The function never panics.
fn repel(
    domain: &Domain,
    xs: &[f64],
    ys: &[f64],
    grid_xy: &[[f64; 2]],
    nx: usize,
    ny: usize,
    options: &Options,
) -> Result<Vec<[f64; 2]>, String> {
    let mut delta = vec![0.0f64; grid_xy.len()];
    for j in 0..ny {
        for i in 0..nx {
            let w = xs[i + 1] - xs[i];
            let h = ys[j + 1] - ys[j];
            let local = options.delta_fraction * w.min(h);
            for &(di, dj) in &[(0, 0), (1, 0), (0, 1), (1, 1)] {
                let id = (j + dj) * (nx + 1) + i + di;
                delta[id] = if delta[id] <= 0.0 {
                    local
                } else {
                    delta[id].min(local)
                };
            }
        }
    }
    let mut out = Vec::with_capacity(grid_xy.len());
    for (id, p) in grid_xy.iter().enumerate() {
        let d = delta[id];
        let mut q = *p;
        if d > EPS {
            let (a, b, distance) = domain.nearest_segment(*p);
            if distance > EPS && distance < d {
                if let Some(moved) = repel_point(*p, a, b, d) {
                    let new_distance = point_segment_distance(moved, a, b);
                    if new_distance > distance + EPS && dist(moved, *p) <= 2.0 * d {
                        q = moved;
                    }
                }
            }
        }
        if !q[0].is_finite() || !q[1].is_finite() {
            return Err("repelling produced a non-finite point".to_string());
        }
        out.push(q);
    }
    // Reject moves that would fold a cell.
    for j in 0..ny {
        for i in 0..nx {
            let ids = [
                j * (nx + 1) + i,
                j * (nx + 1) + i + 1,
                (j + 1) * (nx + 1) + i + 1,
                (j + 1) * (nx + 1) + i,
            ];
            let quad = [out[ids[0]], out[ids[1]], out[ids[2]], out[ids[3]]];
            if !quad_convex(&quad) {
                return Err("repelling folded a grid cell".to_string());
            }
        }
    }
    Ok(out)
}

/// One grid-aligned repelling step.  For a horizontal-ish segment the point is
/// moved vertically until its vertical distance to the segment is `delta`, and
/// vice versa.  Returns `None` when the point is not within `delta` of the
/// segment in the selected coordinate.
///
/// "Horizontal-ish" means `|dx| >= |dy|`; the branch is chosen once from the
/// segment direction, never from the point, so the move stays axis-aligned.
/// The projection parameter is clamped to the segment, which keeps the move
/// near an endpoint finite.  The returned point is the input coordinate in the
/// other axis; the caller re-checks the true distance and the fold.
fn repel_point(p: [f64; 2], a: [f64; 2], b: [f64; 2], delta: f64) -> Option<[f64; 2]> {
    let d = sub(b, a);
    if d[0].abs() >= d[1].abs() {
        // Horizontal-ish: move vertically.
        if d[0].abs() <= EPS {
            return None;
        }
        let t = ((p[0] - a[0]) / d[0]).clamp(0.0, 1.0);
        let y_at = a[1] + t * d[1];
        let vertical = (p[1] - y_at).abs();
        if vertical >= delta {
            return None;
        }
        let sign = if p[1] >= y_at { 1.0 } else { -1.0 };
        Some([p[0], y_at + sign * delta])
    } else {
        // Vertical-ish: move horizontally.
        let t = ((p[1] - a[1]) / d[1]).clamp(0.0, 1.0);
        let x_at = a[0] + t * d[0];
        let horizontal = (p[0] - x_at).abs();
        if horizontal >= delta {
            return None;
        }
        let sign = if p[0] >= x_at { 1.0 } else { -1.0 };
        Some([x_at + sign * delta, p[1]])
    }
}

/// Grid queries and mesh construction for one attempt.
impl<'a> Builder<'a> {
    /// Global node id (`gid`) of the grid node `(i, j)`.
    ///
    /// Row-major: `j * (nx + 1) + i`; `i` in `0..=nx`, `j` in `0..=ny`.
    /// Invariant: `gid < (nx + 1) * (ny + 1)`, which is the size of the grid
    /// prefix of [`Builder::points`].
    fn gid(&self, i: usize, j: usize) -> usize {
        j * (self.nx + 1) + i
    }

    /// The four grid node ids of cell `(i, j)`, counter-clockwise from the
    /// bottom-left: `[bl, br, tr, tl]`.
    ///
    /// Side `k` of the leaf is the edge `gid[k] -> gid[(k + 1) % 4]`
    /// (`0 = bottom`, `1 = right`, `2 = top`, `3 = left`).
    fn cell_gids(&self, i: usize, j: usize) -> [usize; 4] {
        [
            self.gid(i, j),
            self.gid(i + 1, j),
            self.gid(i + 1, j + 1),
            self.gid(i, j + 1),
        ]
    }

    /// The **moved** corner coordinates of cell `(i, j)`, counter-clockwise
    /// from the bottom-left.
    ///
    /// Uses [`Builder::pos`], not the uniform lines, so clipping and
    /// containment see the repelled grid that the mesh will use.
    fn cell_quad(&self, i: usize, j: usize) -> [[f64; 2]; 4] {
        let g = self.cell_gids(i, j);
        [
            self.pos[g[0]],
            self.pos[g[1]],
            self.pos[g[2]],
            self.pos[g[3]],
        ]
    }

    /// Flat index (`j * nx + i`) of the leaf across `side`, or `None` on the
    /// grid border.
    ///
    /// `side` uses the counter-clockwise convention of [`Builder::cell_gids`];
    /// the index is in `0..nx * ny`, matching [`Builder::kind`].
    fn neighbour(&self, i: usize, j: usize, side: usize) -> Option<usize> {
        let n = match side {
            0 => {
                if j == 0 {
                    return None;
                }
                (i, j - 1)
            }
            1 => {
                if i + 1 >= self.nx {
                    return None;
                }
                (i + 1, j)
            }
            2 => {
                if j + 1 >= self.ny {
                    return None;
                }
                (i, j + 1)
            }
            _ => {
                if i == 0 {
                    return None;
                }
                (i - 1, j)
            }
        };
        Some(n.1 * self.nx + n.0)
    }

    /// Add a point (deduplicated at 10 nm) and return its node id.
    ///
    /// This is the only way new nodes enter the mesh.  The `pool` key is
    /// [`pack`] (10 nm grid), so a node request that rounds to the same key
    /// returns the existing id; that is what makes shared boundary nodes and
    /// shared edge midpoints conforming.  Coordinates are stored as given (not
    /// rounded); the key only decides identity.
    fn add_node(&mut self, p: [f64; 2]) -> usize {
        let key = pack(p);
        if let Some(&id) = self.pool.get(&key) {
            return id;
        }
        let id = self.points.len();
        self.points.push([p[0], p[1], 0.0]);
        self.pool.insert(key, id);
        id
    }

    /// Is `p` inside the domain?  A point on the geometry is probed a little
    /// toward `toward`, because the even-odd test is ambiguous there.
    ///
    /// `toward` must be a point known to be inside the same leaf (normally the
    /// leaf centre); the probe walks `1e-3` of the way to it if the fixed
    /// offset still lands on the geometry.  The threshold is `1e-9` in model
    /// units; this is a classification test, not a boundary test (the exact
    /// boundary offset is controlled by the snapping of §2.3).
    fn inside(&self, p: [f64; 2], toward: [f64; 2]) -> bool {
        let d = self.domain.distance_to_boundary(p);
        if d > 1.0e-9 {
            return self.domain.contains(p);
        }
        let scale = self.scale();
        let mut probe = [p[0] + 1.0e-6 * scale, p[1] + 1.3e-6 * scale];
        if dist(probe, p) < 1.0e-9 {
            probe = [toward[0] + 1.0e-6 * scale, toward[1] + 1.3e-6 * scale];
        }
        if self.domain.distance_to_boundary(probe) <= 1.0e-9 {
            probe = [
                p[0] + (toward[0] - p[0]) * 1.0e-3,
                p[1] + (toward[1] - p[1]) * 1.0e-3,
            ];
        }
        self.domain.contains(probe)
    }

    /// Characteristic model length: the larger side of the grid, at least 1.
    ///
    /// Used to turn the absolute tolerances (`1e-7`, `1e-9`, area `1e-14`)
    /// into model-relative ones, so a micron part and a kilometre part behave
    /// the same.  The `max(1.0)` keeps a model smaller than one unit from
    /// shrinking the tolerances below round-off.
    fn scale(&self) -> f64 {
        let w = self.xs[self.xs.len() - 1] - self.xs[0];
        let h = self.ys[self.ys.len() - 1] - self.ys[0];
        w.max(h).max(1.0)
    }

    /// The portions of the domain boundary strictly inside a leaf.
    ///
    /// A bounding-box rejection is followed by a convex clip of the segment to
    /// the (convex) leaf; pieces shorter than `1e-12 * scale` and pieces that
    /// only touch the leaf boundary are dropped, because they cannot cut the
    /// leaf.  The clipped endpoints are snapped by
    /// [`Builder::snap_to_cell_boundary`] so that two leaves sharing a side
    /// produce the very same intersection point (otherwise the two halves of a
    /// boundary chain would not meet).
    ///
    /// Returns the chain segments in no guaranteed order; the caller groups
    /// them with [`chain_components`].  An empty result means the leaf is not
    /// cut (or is only touched), which pass 1 resolves by a centre test.
    fn clip_cell(&self, quad: &[[f64; 2]; 4]) -> Vec<DomainSeg> {
        let mut out = Vec::new();
        for seg in &self.segs {
            if !aabb_overlap(
                seg.a[0].min(seg.b[0]),
                seg.a[1].min(seg.b[1]),
                seg.a[0].max(seg.b[0]),
                seg.a[1].max(seg.b[1]),
                quad,
            ) {
                continue;
            }
            let Some((t0, t1)) = clip_segment_to_quad(seg.a, seg.b, quad) else {
                continue;
            };
            let clipped_len = (t1 - t0) * dist(seg.a, seg.b);
            if clipped_len <= 1.0e-12 * self.scale() {
                continue;
            }
            if !segment_has_interior(seg.a, seg.b, t0, t1, quad) {
                continue;
            }
            // Snap the clipped endpoints so two leaves sharing a side compute
            // the very same intersection point.
            let mut a = lerp(seg.a, seg.b, t0);
            let mut b = lerp(seg.a, seg.b, t1);
            a = self.snap_to_cell_boundary(a, seg.a, seg.b, quad, t0 <= 1.0e-12);
            b = self.snap_to_cell_boundary(b, seg.a, seg.b, quad, t1 >= 1.0 - 1.0e-12);
            out.push(DomainSeg { a, b });
        }
        out
    }

    /// Snap a clipping endpoint onto the exact corner/side it belongs to.
    ///
    /// The tolerance is `1e-7 * scale`.  An endpoint that is a domain vertex
    /// (`at_segment_end`) is returned unchanged: it must keep its input
    /// coordinates.  Otherwise the point is snapped to a leaf corner when it is
    /// within tolerance, and to the exact segment/side intersection
    /// ([`side_intersection`]) when it is within `10 * tol`; this is the
    /// "bit-identical on both sides" trick.  If nothing matches, the clipped
    /// point is kept as is.
    fn snap_to_cell_boundary(
        &self,
        p: [f64; 2],
        seg_a: [f64; 2],
        seg_b: [f64; 2],
        quad: &[[f64; 2]; 4],
        at_segment_end: bool,
    ) -> [f64; 2] {
        let tol = 1.0e-7 * self.scale();
        // A domain vertex that is exactly on the cell boundary stays as is.
        if at_segment_end {
            return p;
        }
        for k in 0..4 {
            if dist(p, quad[k]) <= tol {
                return quad[k];
            }
        }
        for k in 0..4 {
            let a = quad[k];
            let b = quad[(k + 1) % 4];
            let len2 = dot(sub(b, a), sub(b, a));
            if len2 <= 0.0 {
                continue;
            }
            let t = dot(sub(p, a), sub(b, a)) / len2;
            if !(-1.0e-6..=1.0 + 1.0e-6).contains(&t) {
                continue;
            }
            let q = lerp(a, b, t.clamp(0.0, 1.0));
            if dist(p, q) <= tol {
                if let Some(x) = side_intersection(seg_a, seg_b, a, b) {
                    if dist(x, p) <= 10.0 * tol {
                        return x;
                    }
                }
            }
        }
        p
    }

    /// Build the polygons, the quads and the split sides of a cut leaf.
    ///
    /// Returns the quads of the leaf and the cell sides whose midpoint became a
    /// mesh node (i.e. the leaf contributes a hanging node to its neighbour).
    ///
    /// Steps: group the clipped pieces into chains (§2.3); require exactly one
    /// chain, because a leaf with two disjoint boundary pieces is not handled
    /// by this route; walk the chain from its entry to its exit on the leaf
    /// boundary ([`walk_chain`]); close it into a polygon with the inside arc
    /// ([`Builder::build_polygon`]); record the leaf sides used in full;
    /// split polygons with a bad corner ([`split_bad_vertices`]); midpoint
    /// subdivide each polygon ([`Builder::subdivide`]).
    ///
    /// `Err` means this alignment/attempt is rejected; the outer loop retries.
    fn mesh_cut_cell(
        &mut self,
        i: usize,
        j: usize,
        quad: &[[f64; 2]; 4],
        clipped: &[DomainSeg],
    ) -> Result<(Vec<[usize; 4]>, Vec<usize>), String> {
        let components = chain_components(clipped, quad, self.scale());
        if components.len() != 1 {
            return Err(format!(
                "a cut cell contains {} boundary chains",
                components.len()
            ));
        }
        let index = 0;
        let component = &components[index];
        let (entry, exit, interior) = walk_chain(component, quad, self.scale())?;
        let polygon = self.build_polygon(quad, entry, exit, &interior)?;
        // Which sides of the cell does the polygon use in full?  Their
        // midpoint is a hanging node for an untouched neighbour.
        let gid = self.cell_gids(i, j);
        let mut sides = Vec::new();
        for k in 0..4 {
            if polygon_has_edge(&polygon.points, gid[k], gid[(k + 1) % 4], &self.points) {
                sides.push(k);
            }
        }
        let sub_polygons = split_bad_vertices(&polygon)?;
        let mut quads = Vec::new();
        for sub in &sub_polygons {
            self.subdivide(sub, &mut quads)?;
        }
        Ok((quads, sides))
    }

    /// Choose the boundary polygon of a cut leaf: the chain plus the arc of the
    /// leaf boundary that lies inside the domain.
    ///
    /// The chain is first reduced to its real corners by
    /// [`simplify_chain`], which also returns the arc-length point to use on
    /// every smooth chord.  Two candidate polygons are formed: the
    /// counter-clockwise leaf arc from the exit back to the entry, and the
    /// complementary arc.  A candidate is accepted when it has positive area
    /// above `1e-14 * scale^2`, is simple, has its centroid strictly inside
    /// itself and that centroid is inside the domain; the first accepted
    /// candidate wins.  The polygon is then cleaned at `1e-7 * scale`
    /// ([`clean_polygon`]) and its edges that coincide with a simplified chain
    /// chord are tagged with the chord's snap point.  `Err` means the leaf
    /// could not be closed into one inside polygon.
    fn build_polygon(
        &self,
        quad: &[[f64; 2]; 4],
        entry: [f64; 2],
        exit: [f64; 2],
        interior: &[[f64; 2]],
    ) -> Result<Poly, String> {
        let Some(ue) = quad_param(quad, entry) else {
            return Err("the entry point is not on the leaf boundary".to_string());
        };
        let Some(ux) = quad_param(quad, exit) else {
            return Err("the exit point is not on the leaf boundary".to_string());
        };
        let centre = average4(quad);
        let mut chain = vec![entry];
        chain.extend_from_slice(interior);
        chain.push(exit);
        let (simplified, run_mids) = simplify_chain(&chain);
        let chain_interior = &simplified[1..simplified.len().saturating_sub(1)];

        // Candidate 1: the CCW arc from exit back to entry.
        let mut first = vec![entry];
        first.extend_from_slice(chain_interior);
        first.push(exit);
        let mut ue2 = ue;
        if ue2 <= ux {
            ue2 += 4.0;
        }
        first.extend(arc_corners(quad, ux, ue2));
        // Candidate 2: the other arc (clockwise from exit to entry).
        let mut second = vec![entry];
        second.extend_from_slice(chain_interior);
        second.push(exit);
        let mut ux2 = ux;
        if ux2 <= ue {
            ux2 += 4.0;
        }
        let mut corners = arc_corners(quad, ue, ux2);
        corners.reverse();
        second.extend(corners);

        let mut chosen: Option<Vec<[f64; 2]>> = None;
        // Use the first valid candidate; the complementary arc is the fallback.
        for mut candidate in [first, second] {
            if polygon_area(&candidate).abs() <= 1.0e-14 * self.scale() * self.scale() {
                continue;
            }
            if polygon_area(&candidate) < 0.0 {
                candidate.reverse();
            }
            if !polygon_simple(&candidate) {
                continue;
            }
            let c = polygon_centroid(&candidate);
            if !point_in_polygon(c, &candidate) {
                continue;
            }
            if self.inside(c, centre) {
                chosen = Some(candidate);
                break;
            }
        }
        let chosen = chosen.ok_or_else(|| "no inside polygon for a cut cell".to_string())?;
        let points = clean_polygon(&chosen, 1.0e-7 * self.scale());
        if points.len() < 3 {
            return Err("a cut polygon collapsed to a segment".to_string());
        }
        if !polygon_simple(&points) {
            return Err("a cut polygon is not simple after cleanup".to_string());
        }
        // Mark the polygon edges that run along the geometry: their midpoints
        // are snapped onto the boundary so that no boundary node leaves it.
        let mut chain_edge = vec![false; points.len()];
        let mut snap = vec![None; points.len()];
        let tol = 1.0e-9;
        for i in 0..points.len() {
            let a = points[i];
            let b = points[(i + 1) % points.len()];
            for j in 0..simplified.len().saturating_sub(1) {
                let p = simplified[j];
                let q = simplified[j + 1];
                let same = (dist(a, p) <= tol && dist(b, q) <= tol)
                    || (dist(a, q) <= tol && dist(b, p) <= tol);
                if same {
                    chain_edge[i] = true;
                    snap[i] = run_mids.get(j).copied();
                    break;
                }
            }
        }
        Ok(Poly {
            points,
            chain_edge,
            snap,
        })
    }

    /// Midpoint subdivision of one polygon into one quad per edge (§2.3).
    ///
    /// For polygon vertices `v_0..v_{n-1}` and edge midpoints `m_k`, the quads
    /// are `[C, m_k, v_{k+1}, m_{k+1}]` with `C` the vertex average; this is
    /// the paper's "one quad per edge" midpoint subdivision.
    ///
    /// The midpoint of a chain edge is snapped onto the geometry; the midpoint
    /// of an interior edge (a cell side or a split diagonal) stays on the
    /// straight edge.  Snapping keeps every boundary node on the domain while
    /// changing the quad by at most the small sagitta of the chord.  A snap is
    /// used only when it is within `0.4` of the chord length; otherwise the
    /// straight midpoint is kept and the polygon is marked unsnapped.
    ///
    /// A degenerate quad after snapping falls back to
    /// [`Builder::subdivide_unsnapped`] for the whole polygon (so a polygon is
    /// never half snapped); if the unsnapped subdivision is also degenerate the
    /// attempt fails with `Err`.
    fn subdivide(&mut self, polygon: &Poly, quads: &mut Vec<[usize; 4]>) -> Result<(), String> {
        let n = polygon.points.len();
        if n < 3 {
            return Err("polygon with fewer than three vertices".to_string());
        }
        let mut vertices = Vec::with_capacity(n);
        for p in &polygon.points {
            vertices.push(self.add_node(*p));
        }
        // First try with the chain midpoints snapped to the boundary.
        let mut mids = Vec::with_capacity(n);
        let mut snapped = true;
        for k in 0..n {
            let a = self.points[vertices[k]];
            let b = self.points[vertices[(k + 1) % n]];
            let mut m = [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])];
            if polygon.chain_edge[k] {
                // Snap the chord midpoint onto the boundary run; a snap farther
                // than 0.4 of the chord would shear the quad too much, so the
                // whole polygon falls back to straight midpoints below.
                let q = match polygon.snap.get(k).copied().flatten() {
                    Some(p) => p,
                    None => nearest_boundary_point(self.domain, m),
                };
                if dist(q, m) <= 0.4 * dist([a[0], a[1]], [b[0], b[1]]) {
                    m = q;
                } else {
                    snapped = false;
                }
            }
            mids.push(self.add_node(m));
        }
        let mut cx = 0.0;
        let mut cy = 0.0;
        for &id in &vertices {
            cx += self.points[id][0];
            cy += self.points[id][1];
        }
        cx /= n as f64;
        cy /= n as f64;
        let centre = self.add_node([cx, cy]);
        for k in 0..n {
            let q = [centre, mids[k], vertices[(k + 1) % n], mids[(k + 1) % n]];
            if !quad_positive(&q, &self.points) || !all_distinct(&q) {
                if snapped {
                    // The snapped midpoint folded a quad; retry on the straight
                    // chord for this polygon.
                    return self.subdivide_unsnapped(polygon, quads);
                }
                return Err(format!(
                    "midpoint subdivision produced a degenerate quad (n={n}, k={k})",
                ));
            }
            quads.push(q);
        }
        Ok(())
    }

    /// Midpoint subdivision without snapping, used as a fallback.
    ///
    /// Identical topology to [`Builder::subdivide`] but every midpoint is the
    /// straight chord midpoint, so the boundary nodes are the unsnapped chord
    /// points (a boundary offset of at most the chord sagitta).  All quads must
    /// be distinct and positively oriented; otherwise `Err`.
    fn subdivide_unsnapped(
        &mut self,
        polygon: &Poly,
        quads: &mut Vec<[usize; 4]>,
    ) -> Result<(), String> {
        let n = polygon.points.len();
        let mut vertices = Vec::with_capacity(n);
        for p in &polygon.points {
            vertices.push(self.add_node(*p));
        }
        let mut mids = Vec::with_capacity(n);
        for k in 0..n {
            let a = self.points[vertices[k]];
            let b = self.points[vertices[(k + 1) % n]];
            mids.push(self.add_node([0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])]));
        }
        let mut cx = 0.0;
        let mut cy = 0.0;
        for &id in &vertices {
            cx += self.points[id][0];
            cy += self.points[id][1];
        }
        cx /= n as f64;
        cy /= n as f64;
        let centre = self.add_node([cx, cy]);
        for k in 0..n {
            let q = [centre, mids[k], vertices[(k + 1) % n], mids[(k + 1) % n]];
            if !quad_positive(&q, &self.points) || !all_distinct(&q) {
                return Err(format!(
                    "midpoint subdivision produced a degenerate quad (n={n}, k={k})",
                ));
            }
            quads.push(q);
        }
        Ok(())
    }

    /// Mark leaves until every untouched leaf has an even number of hanging
    /// sides (the "corner marking" of §2.4), then the 2-ref templates apply.
    ///
    /// An untouched ([`CellKind::Full`]) leaf with an odd number of split sides
    /// cannot be meshed by the 2-refinement templates.  The pass repeatedly
    /// takes the first odd leaf and runs a BFS through untouched leaves over
    /// **unsplit** edges to a second odd leaf; every edge on that path is marked
    /// as split.  Marking an edge toggles the parity of both endpoint leaves, so
    /// each round removes two odd leaves.  The path never crosses a cut leaf,
    /// which would change a cut cell's mesh.
    ///
    /// Invariants after the call: every full leaf has even parity, and the
    /// marked edges are exactly the hanging-node edges.  `Err` if the number of
    /// odd leaves is odd, if an odd leaf cannot reach a partner, or if the BFS
    /// parent chain is cyclic (a guard against an inconsistent `split` set).
    fn resolve_parity(&mut self) -> Result<(), String> {
        let total = self.nx * self.ny;
        loop {
            let mut odd = Vec::new();
            for j in 0..self.ny {
                for i in 0..self.nx {
                    let cell = j * self.nx + i;
                    if self.kind[cell] == CellKind::Full && self.split_count(i, j) % 2 == 1 {
                        odd.push((i, j));
                    }
                }
            }
            if odd.is_empty() {
                return Ok(());
            }
            if odd.len() % 2 == 1 {
                return Err("an odd number of untouched leaves needs marking".to_string());
            }
            let start = odd[0];
            // BFS through untouched leaves over unsplit edges to a second odd
            // leaf; every edge on the path is marked.
            let mut visited = vec![false; total];
            let mut parent: Vec<Option<(usize, (usize, usize))>> = vec![None; total];
            let mut queue = VecDeque::new();
            let start_cell = start.1 * self.nx + start.0;
            visited[start_cell] = true;
            queue.push_back(start);
            let mut found: Option<(usize, usize)> = None;
            'search: while let Some((i, j)) = queue.pop_front() {
                for side in 0..4 {
                    let Some(ncell) = self.neighbour(i, j, side) else {
                        continue;
                    };
                    if self.kind[ncell] != CellKind::Full || visited[ncell] {
                        continue;
                    }
                    let ni = ncell % self.nx;
                    let nj = ncell / self.nx;
                    let gid = self.cell_gids(i, j);
                    let key = edge_key(gid[side], gid[(side + 1) % 4]);
                    if self.split.contains(&key) {
                        continue;
                    }
                    visited[ncell] = true;
                    parent[ncell] = Some((j * self.nx + i, key));
                    if (ni, nj) != start && self.split_count(ni, nj) % 2 == 1 {
                        found = Some((ni, nj));
                        break 'search;
                    }
                    queue.push_back((ni, nj));
                }
            }
            let Some((fi, fj)) = found else {
                return Err("could not pair the hanging nodes of untouched leaves".to_string());
            };
            let mut current = fj * self.nx + fi;
            let mut steps = 0usize;
            while let Some((previous, key)) = parent[current] {
                if key.0 == key.1 {
                    return Err("lost the edge of a parity path".to_string());
                }
                self.split.insert(key);
                current = previous;
                steps += 1;
                if steps > total {
                    return Err("the parity path is cyclic".to_string());
                }
            }
        }
    }

    /// Number of sides of cell `(i, j)` whose midpoint is a split (hanging)
    /// node, in `0..=4`.
    fn split_count(&self, i: usize, j: usize) -> usize {
        let gid = self.cell_gids(i, j);
        (0..4)
            .filter(|&k| self.split.contains(&edge_key(gid[k], gid[(k + 1) % 4])))
            .count()
    }

    /// Mesh one untouched leaf with the 2-ref templates.
    ///
    /// The template depends only on the split sides, which parity guarantees to
    /// be an even count:
    ///
    /// - 0 splits: the cell itself (`1` quad);
    /// - 2 opposite splits: two quads split by the line joining the two edge
    ///   midpoints (`2` quads);
    /// - 2 adjacent splits: three quads around the cell centre, using the two
    ///   adjacent midpoints and the centre (`3` quads);
    /// - 4 splits: four quads around the centre, one per corner (`4` quads).
    ///
    /// An odd count is rejected (`Err`), which means [`Builder::resolve_parity`]
    /// missed something.  Every produced quad is checked for positive
    /// orientation and distinct nodes.
    fn full_cell(&mut self, i: usize, j: usize) -> Result<Vec<[usize; 4]>, String> {
        let gid = self.cell_gids(i, j);
        let mut split = [false; 4];
        for k in 0..4 {
            split[k] = self.split.contains(&edge_key(gid[k], gid[(k + 1) % 4]));
        }
        let count = split.iter().filter(|&&s| s).count();
        let mut out = Vec::new();
        match count {
            0 => out.push(gid),
            2 => {
                let adjacent = (0..4).find(|&k| split[k] && split[(k + 1) % 4]);
                if let Some(k) = adjacent {
                    let a = gid[k];
                    let b = gid[(k + 1) % 4];
                    let c = gid[(k + 2) % 4];
                    let d = gid[(k + 3) % 4];
                    let m0 = self.midpoint_id(a, b);
                    let m1 = self.midpoint_id(b, c);
                    let centre = self.center_id(&gid);
                    out.push([a, m0, centre, d]);
                    out.push([m0, b, m1, centre]);
                    out.push([centre, m1, c, d]);
                } else {
                    let k = (0..4)
                        .find(|&k| split[k])
                        .ok_or_else(|| "two split sides not found".to_string())?;
                    let m0 = self.midpoint_id(gid[k], gid[(k + 1) % 4]);
                    let m2 = self.midpoint_id(gid[(k + 2) % 4], gid[(k + 3) % 4]);
                    out.push([gid[k], m0, m2, gid[(k + 3) % 4]]);
                    out.push([m0, gid[(k + 1) % 4], gid[(k + 2) % 4], m2]);
                }
            }
            4 => {
                let center = self.center_id(&gid);
                for k in 0..4 {
                    let m0 = self.midpoint_id(gid[k], gid[(k + 1) % 4]);
                    let m1 = self.midpoint_id(gid[(k + 1) % 4], gid[(k + 2) % 4]);
                    out.push([center, m0, gid[(k + 1) % 4], m1]);
                }
            }
            _ => return Err("odd number of hanging sides on an untouched leaf".to_string()),
        }
        for q in &out {
            if !quad_positive(q, &self.points) || !all_distinct(q) {
                return Err("a two-refinement template produced a degenerate quad".to_string());
            }
        }
        Ok(out)
    }

    /// Node id of the midpoint of nodes `a` and `b`, creating it on first
    /// use.  Deduplication makes the midpoint shared by both neighbouring
    /// leaves, so no T-junction can remain.
    fn midpoint_id(&mut self, a: usize, b: usize) -> usize {
        let pa = self.points[a];
        let pb = self.points[b];
        self.add_node([0.5 * (pa[0] + pb[0]), 0.5 * (pa[1] + pb[1])])
    }

    /// Node id of the average of a cell's four corners, creating it on first
    /// use.
    ///
    /// The centre is the average of the (possibly repelled) corner positions,
    /// not the average of the uniform cell; that keeps the 2-ref templates
    /// inside the repelled quad.
    fn center_id(&mut self, gid: &[usize; 4]) -> usize {
        let mut x = 0.0;
        let mut y = 0.0;
        for &id in gid {
            x += self.points[id][0];
            y += self.points[id][1];
        }
        self.add_node([x / 4.0, y / 4.0])
    }

    /// Assemble the mesh, compact the point list and check conformity.
    ///
    /// Concatenates `cells_quads` into `quads`, remaps the used nodes to a
    /// compact list, and returns the mesh only after two independent checks:
    /// [`has_t_junction`] (a node strictly inside a single-use edge) and
    /// `Mesh::topology().cracks != 0` (boundary cracks).  These checks are the
    /// "no cleanup" guarantee: a mesh that would need cleanup is rejected and
    /// the outer search tries another alignment/attempt instead.  `Err` on an
    /// unknown node id, an empty mesh, a T-junction or a crack.
    fn finish(mut self) -> Result<Mesh, String> {
        for cell in &self.cells_quads {
            self.quads.extend_from_slice(cell);
        }
        if self.quads.is_empty() {
            return Err("the route produced no elements".to_string());
        }
        // Compact the nodes actually used.
        let mut remap = vec![usize::MAX; self.points.len()];
        let mut points: Vec<Node> = Vec::new();
        let mut cells: Vec<Cell> = Vec::with_capacity(self.quads.len());
        let mut quads: Vec<[usize; 4]> = Vec::with_capacity(self.quads.len());
        for q in &self.quads {
            let mut local = [0usize; 4];
            for (k, &id) in q.iter().enumerate() {
                if id >= self.points.len() {
                    return Err("a quad references an unknown node".to_string());
                }
                if remap[id] == usize::MAX {
                    remap[id] = points.len();
                    points.push(self.points[id]);
                }
                local[k] = remap[id];
            }
            if local.contains(&usize::MAX) {
                return Err("node remapping failed".to_string());
            }
            cells.push(Cell::quad4(local));
            quads.push(local);
        }
        if has_t_junction(&points, &quads) {
            return Err("the assembled mesh has a T-junction".to_string());
        }
        let mesh = Mesh::new(points, cells);
        if mesh.topology().cracks != 0 {
            return Err("the assembled mesh has boundary cracks".to_string());
        }
        Ok(mesh)
    }
}

// ---------------------------------------------------------------------------
// small geometry helpers
// ---------------------------------------------------------------------------

/// Vector `a - b` (the difference is used as a direction, hence the order).
fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

/// Euclidean dot product; the squared length is `dot(a, a)`.
fn dot(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[0] + a[1] * b[1]
}

/// Scalar cross product `a.x * b.y - a.y * b.x`.
///
/// Its sign is the turn direction: positive means `b` is to the left of `a`.
/// Used by [`polygon_area`], [`segments_cross`] and [`quad_convex`].
fn cross(a: [f64; 2], b: [f64; 2]) -> f64 {
    a[0] * b[1] - a[1] * b[0]
}

/// Euclidean length of `a`.
fn norm(a: [f64; 2]) -> f64 {
    dot(a, a).sqrt()
}

/// Euclidean distance between two points, in model units.
fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    norm(sub(a, b))
}

/// Linear interpolation `a + t * (b - a)`; `t` is **not** clamped, so the
/// caller can extrapolate deliberately.
fn lerp(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
}

/// Centroid (vertex average) of four points; the leaf centre used for the
/// inside probe and the 2-ref templates.
fn average4(q: &[[f64; 2]; 4]) -> [f64; 2] {
    [
        0.25 * (q[0][0] + q[1][0] + q[2][0] + q[3][0]),
        0.25 * (q[0][1] + q[1][1] + q[2][1] + q[3][1]),
    ]
}

/// 10 nm deduplication key.
fn pack(p: [f64; 2]) -> (i64, i64) {
    let x = (p[0] * KEY_SCALE).round();
    let y = (p[1] * KEY_SCALE).round();
    let xi = if x >= i64::MAX as f64 {
        i64::MAX
    } else if x <= i64::MIN as f64 {
        i64::MIN
    } else {
        x as i64
    };
    let yi = if y >= i64::MAX as f64 {
        i64::MAX
    } else if y <= i64::MIN as f64 {
        i64::MIN
    } else {
        y as i64
    };
    (xi, yi)
}

/// Order-independent key of an undirected edge, so `(a, b)` and `(b, a)` hit
/// the same entry in [`Builder::split`].
fn edge_key(a: usize, b: usize) -> (usize, usize) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Distance from point `p` to the **closed** segment `a..b`.
///
/// A zero-length segment falls back to the point distance; the projection is
/// clamped, so an endpoint gives the correct distance outside the span.
fn point_segment_distance(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let ab = sub(b, a);
    let len2 = dot(ab, ab);
    if len2 <= 0.0 {
        return dist(p, a);
    }
    let t = (dot(sub(p, a), ab) / len2).clamp(0.0, 1.0);
    dist(p, lerp(a, b, t))
}

/// Signed area of a polygon, positive when counter-clockwise.
fn polygon_area(p: &[[f64; 2]]) -> f64 {
    let n = p.len();
    let mut sum = 0.0;
    for i in 0..n {
        let a = p[i];
        let b = p[(i + 1) % n];
        sum += a[0] * b[1] - b[0] * a[1];
    }
    0.5 * sum
}

/// Area centroid of a polygon (the vertex average when the area is zero).
///
/// Uses the shoelace weights; `a2.abs() <= 1e-30` is a degeneracy guard that
/// returns the vertex average instead of dividing by zero, so the function is
/// total.  Only used for the inside test of a candidate polygon, where a
/// sliver's centroid is not trusted.
fn polygon_centroid(p: &[[f64; 2]]) -> [f64; 2] {
    let n = p.len();
    let mut a2 = 0.0;
    let mut cx = 0.0;
    let mut cy = 0.0;
    for i in 0..n {
        let a = p[i];
        let b = p[(i + 1) % n];
        let w = a[0] * b[1] - b[0] * a[1];
        a2 += w;
        cx += (a[0] + b[0]) * w;
        cy += (a[1] + b[1]) * w;
    }
    if a2.abs() <= 1.0e-30 {
        let mut x = 0.0;
        let mut y = 0.0;
        for q in p {
            x += q[0];
            y += q[1];
        }
        let inv = 1.0 / n.max(1) as f64;
        return [x * inv, y * inv];
    }
    let inv = 1.0 / (3.0 * a2);
    [cx * inv, cy * inv]
}

/// Even-odd (ray-crossing) test: is `p` strictly inside the closed polygon?
///
/// A point exactly on an edge or vertex is ambiguous (the parity can go either
/// way); callers only use it with a centroid or an offset probe, where that
/// case does not arise.
fn point_in_polygon(p: [f64; 2], poly: &[[f64; 2]]) -> bool {
    let n = poly.len();
    let mut inside = false;
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        if (a[1] > p[1]) != (b[1] > p[1]) {
            let t = (p[1] - a[1]) / (b[1] - a[1]);
            if p[0] < a[0] + t * (b[0] - a[0]) {
                inside = !inside;
            }
        }
    }
    inside
}

/// Do the open segments `a..b` and `c..d` cross **properly**?
///
/// Touching at an endpoint (`t` or `u` at 0/1) and collinear overlap return
/// false; the tolerance `1e-9` keeps a shared polygon vertex from counting as
/// a self-intersection in [`polygon_simple`].
fn segments_cross(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
    let d1 = sub(b, a);
    let d2 = sub(d, c);
    let den = cross(d1, d2);
    if den.abs() <= 1.0e-14 {
        return false;
    }
    let t = cross(sub(c, a), d2) / den;
    let u = cross(sub(c, a), d1) / den;
    t > 1.0e-9 && t < 1.0 - 1.0e-9 && u > 1.0e-9 && u < 1.0 - 1.0e-9
}

/// Is the closed polygon simple (no two non-adjacent edges properly cross)?
///
/// Adjacent edges share a vertex by construction and are skipped.  Returns
/// false for fewer than three points.  A tangential self-touch is not
/// detected; the corpus polygons do not produce one.
fn polygon_simple(p: &[[f64; 2]]) -> bool {
    let n = p.len();
    if n < 3 {
        return false;
    }
    for i in 0..n {
        let a = p[i];
        let b = p[(i + 1) % n];
        for j in (i + 1)..n {
            if j == i || (j + 1) % n == i || j == (i + 1) % n {
                continue;
            }
            let c = p[j];
            let d = p[(j + 1) % n];
            if segments_cross(a, b, c, d) {
                return false;
            }
        }
    }
    true
}

/// Interior angles of a CCW polygon, in degrees; reflex vertices exceed 180.
fn polygon_interior_angles(p: &[[f64; 2]]) -> Vec<f64> {
    let n = p.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let prev = p[(i + n - 1) % n];
        let here = p[i];
        let next = p[(i + 1) % n];
        let d1 = sub(here, prev);
        let d2 = sub(next, here);
        if norm(d1) <= 0.0 || norm(d2) <= 0.0 {
            out.push(180.0);
            continue;
        }
        let turn = cross(d1, d2).atan2(dot(d1, d2));
        out.push((180.0 - turn.to_degrees()).rem_euclid(360.0));
    }
    out
}

/// Unoriented angle between two vectors, in degrees, clamped to `[0, 180]`.
///
/// Returns 0.0 for a zero vector instead of NaN, so a degenerate polygon
/// vertex is reported as a bad (flat) corner rather than crashing.
fn angle_between_deg(a: [f64; 2], b: [f64; 2]) -> f64 {
    let na = norm(a);
    let nb = norm(b);
    if na <= 0.0 || nb <= 0.0 {
        return 0.0;
    }
    (dot(a, b) / (na * nb)).clamp(-1.0, 1.0).acos().to_degrees()
}

/// Is the quad convex (in either winding)?
///
/// All four turns must have the same sign; the `1e-14` dead band keeps a
/// nearly straight corner from flipping the test.  Orientation itself is
/// checked separately by [`quad_positive`], which requires counter-clockwise.
fn quad_convex(q: &[[f64; 2]; 4]) -> bool {
    let mut positive = 0;
    let mut negative = 0;
    for k in 0..4 {
        let a = q[k];
        let b = q[(k + 1) % 4];
        let c = q[(k + 2) % 4];
        let turn = cross(sub(b, a), sub(c, b));
        if turn > 1.0e-14 {
            positive += 1;
        } else if turn < -1.0e-14 {
            negative += 1;
        }
    }
    positive == 4 || negative == 4
}

/// Is `p` strictly inside the convex quad `q`, at least `1e-10` from its
/// sides?
///
/// Assumes `q` is counter-clockwise.  The margin is what keeps a clipped
/// boundary endpoint that lies on the leaf boundary from being reported as an
/// interior point of the chain.
fn quad_contains_strict(q: &[[f64; 2]; 4], p: [f64; 2]) -> bool {
    for k in 0..4 {
        let a = q[k];
        let b = q[(k + 1) % 4];
        if cross(sub(b, a), sub(p, a)) <= 1.0e-12 {
            return false;
        }
        if point_segment_distance(p, a, b) <= 1.0e-10 {
            return false;
        }
    }
    true
}

/// Distance from `p` to the nearest side of the quad, in model units.
fn quad_distance_to_boundary(q: &[[f64; 2]; 4], p: [f64; 2]) -> f64 {
    let mut best = f64::INFINITY;
    for k in 0..4 {
        best = best.min(point_segment_distance(p, q[k], q[(k + 1) % 4]));
    }
    best
}

/// Conservative overlap of an axis-aligned box with a (convex) quad.
///
/// Returns true when the boxes overlap, so it only rejects a segment that
/// cannot possibly cut the leaf; the exact test is the clip that follows.
fn aabb_overlap(xmin: f64, ymin: f64, xmax: f64, ymax: f64, q: &[[f64; 2]; 4]) -> bool {
    let mut qxmin = f64::INFINITY;
    let mut qymin = f64::INFINITY;
    let mut qxmax = f64::NEG_INFINITY;
    let mut qymax = f64::NEG_INFINITY;
    for p in q {
        qxmin = qxmin.min(p[0]);
        qymin = qymin.min(p[1]);
        qxmax = qxmax.max(p[0]);
        qymax = qymax.max(p[1]);
    }
    !(xmax < qxmin || xmin > qxmax || ymax < qymin || ymin > qymax)
}

/// Clip the segment `a..b` to a convex CCW quad; returns parameters `(t0, t1)`.
fn clip_segment_to_quad(a: [f64; 2], b: [f64; 2], q: &[[f64; 2]; 4]) -> Option<(f64, f64)> {
    let mut t0 = 0.0f64;
    let mut t1 = 1.0f64;
    for k in 0..4 {
        let p = q[k];
        let r = q[(k + 1) % 4];
        let edge = sub(r, p);
        let fa = cross(edge, sub(a, p));
        let fb = cross(edge, sub(b, p));
        if fa < -1.0e-12 && fb < -1.0e-12 {
            return None;
        }
        if fa < -1.0e-12 {
            t0 = t0.max(fa / (fa - fb));
        } else if fb < -1.0e-12 {
            t1 = t1.min(fa / (fa - fb));
        }
        if t0 > t1 {
            return None;
        }
    }
    Some((t0.clamp(0.0, 1.0), t1.clamp(0.0, 1.0)))
}

/// Does the clipped span `t0..t1` of `a..b` contain a point strictly inside
/// the quad?
///
/// Three sample parameters (1/4, 1/2, 3/4) are enough to reject a segment
/// that only touches the leaf boundary; a genuine cut always contains one of
/// them inside because the clipped span is convex.
fn segment_has_interior(a: [f64; 2], b: [f64; 2], t0: f64, t1: f64, quad: &[[f64; 2]; 4]) -> bool {
    [0.25, 0.5, 0.75].iter().any(|&f| {
        let t = t0 + f * (t1 - t0);
        quad_contains_strict(quad, lerp(a, b, t))
    })
}

/// Exact intersection of a domain segment with a cell side, computed with the
/// side endpoints in a canonical order so that the two cells sharing the side
/// get bit-identical points.
fn side_intersection(
    seg_a: [f64; 2],
    seg_b: [f64; 2],
    side_a: [f64; 2],
    side_b: [f64; 2],
) -> Option<[f64; 2]> {
    let (p, q) = if (side_a[0], side_a[1]) <= (side_b[0], side_b[1]) {
        (side_a, side_b)
    } else {
        (side_b, side_a)
    };
    let d1 = sub(q, p);
    let d2 = sub(seg_b, seg_a);
    let den = cross(d1, d2);
    if den.abs() <= 1.0e-15 {
        return None;
    }
    let t = cross(sub(seg_a, p), d2) / den;
    let u = cross(sub(seg_a, p), d1) / den;
    if !(-1.0e-9..=1.0 + 1.0e-9).contains(&t) || !(-1.0e-9..=1.0 + 1.0e-9).contains(&u) {
        return None;
    }
    Some([p[0] + t * d1[0], p[1] + t * d1[1]])
}

/// Boundary parameter of a point on a quad: side index plus fraction.
fn quad_param(q: &[[f64; 2]; 4], p: [f64; 2]) -> Option<f64> {
    let mut best: Option<(f64, f64)> = None;
    for k in 0..4 {
        let a = q[k];
        let b = q[(k + 1) % 4];
        let ab = sub(b, a);
        let len2 = dot(ab, ab);
        if len2 <= 0.0 {
            continue;
        }
        let t = (dot(sub(p, a), ab) / len2).clamp(0.0, 1.0);
        let d = dist(p, lerp(a, b, t));
        if best.is_none_or(|(bd, _)| d < bd) {
            best = Some((d, k as f64 + t));
        }
    }
    match best {
        Some((d, u)) if d <= 1.0e-7 => Some(u),
        _ => None,
    }
}

/// Corners encountered on the CCW arc from parameter `u0` to `u1` (which must
/// satisfy `u1 > u0`), excluding the endpoints.
fn arc_corners(q: &[[f64; 2]; 4], u0: f64, u1: f64) -> Vec<[f64; 2]> {
    let mut out = Vec::new();
    let mut c = u0.floor() + 1.0;
    while c < u1 - 1.0e-9 {
        let k = (c.round() as i64).rem_euclid(4) as usize;
        out.push(q[k]);
        c += 1.0;
    }
    out
}

/// One connected piece of the boundary inside a leaf.
///
/// Two clipped segments are linked when an endpoint of one is strictly inside
/// the leaf and within `1e-7 * scale` of an endpoint of the other; the
/// equivalence classes are the chains.  Components are sorted by their
/// smallest packed endpoint so the result is deterministic.  `walk_chain`
/// requires exactly one component for a cut cell; a leaf cut by two separate
/// boundary pieces is outside this route's templates.
fn chain_components(segs: &[DomainSeg], quad: &[[f64; 2]; 4], scale: f64) -> Vec<Vec<DomainSeg>> {
    let n = segs.len();
    let mut parent: Vec<usize> = (0..n).collect();
    let tol = 1.0e-7 * scale;
    for i in 0..n {
        for j in (i + 1)..n {
            let mut shared = false;
            for a in [segs[i].a, segs[i].b] {
                if !quad_contains_strict(quad, a) {
                    continue;
                }
                for b in [segs[j].a, segs[j].b] {
                    if !quad_contains_strict(quad, b) {
                        continue;
                    }
                    if dist(a, b) <= tol {
                        shared = true;
                    }
                }
            }
            if shared {
                let (ri, rj) = (find(&mut parent, i), find(&mut parent, j));
                if ri != rj {
                    parent[rj] = ri;
                }
            }
        }
    }
    let mut groups: HashMap<usize, Vec<DomainSeg>> = HashMap::new();
    for i in 0..n {
        let root = find(&mut parent, i);
        groups.entry(root).or_default().push(segs[i]);
    }
    let mut out: Vec<Vec<DomainSeg>> = groups.into_values().collect();
    out.sort_by_key(|g| {
        g.iter()
            .map(|s| (pack(s.a), pack(s.b)))
            .min()
            .unwrap_or(((0, 0), (0, 0)))
    });
    out
}

/// Union-find root of `i` with path halving.
///
/// Used by [`chain_components`]; the parent array is modified in place and the
/// tree stays shallow, so the grouping is near-linear.
fn find(parent: &mut [usize], i: usize) -> usize {
    let mut root = i;
    while parent[root] != root {
        root = parent[root];
    }
    let mut current = i;
    while parent[current] != current {
        let next = parent[current];
        parent[current] = root;
        current = next;
    }
    root
}

/// Follow one chain from an entry point on the leaf boundary to its exit.
///
/// The chain's endpoints are the segment ends that lie on (or within
/// `1e-6 * scale` of) the leaf boundary but not strictly inside it; there must
/// be exactly two, otherwise the chain enters/leaves more than once and the
/// leaf cannot be closed by one polygon (`Err`).  The walk starts at the
/// lexicographically smaller boundary end (deterministic) and returns
/// `(entry, exit, interior_points)`; `interior_points` are the chain vertices
/// strictly inside the leaf, excluding both ends.
fn walk_chain(
    component: &[DomainSeg],
    quad: &[[f64; 2]; 4],
    scale: f64,
) -> Result<([f64; 2], [f64; 2], Vec<[f64; 2]>), String> {
    let tol = 1.0e-6 * scale;
    let mut boundary_ends = Vec::new();
    for seg in component {
        for p in [seg.a, seg.b] {
            if quad_distance_to_boundary(quad, p) <= tol && !quad_contains_strict(quad, p) {
                boundary_ends.push(p);
            }
        }
    }
    boundary_ends.sort_by_key(|p| pack(*p));
    boundary_ends.dedup_by(|a, b| dist(*a, *b) <= tol);
    if boundary_ends.len() != 2 {
        return Err(format!(
            "a boundary chain touches the leaf boundary {} times",
            boundary_ends.len()
        ));
    }
    let entry = boundary_ends[0];
    let mut used = vec![false; component.len()];
    let mut interior = Vec::new();
    let mut current = entry;
    for _ in 0..component.len() {
        let mut next_seg = None;
        for (idx, seg) in component.iter().enumerate() {
            if used[idx] {
                continue;
            }
            if dist(seg.a, current) <= tol || dist(seg.b, current) <= tol {
                next_seg = Some(idx);
                break;
            }
        }
        let Some(idx) = next_seg else {
            return Err("a boundary chain has a loose end".to_string());
        };
        used[idx] = true;
        let seg = &component[idx];
        let other = if dist(seg.a, current) <= tol {
            seg.b
        } else {
            seg.a
        };
        if quad_contains_strict(quad, other) {
            interior.push(other);
            current = other;
        } else {
            if used.iter().any(|&u| !u) {
                return Err("a boundary chain leaves more than once".to_string());
            }
            return Ok((entry, other, interior));
        }
    }
    Err("a boundary chain did not terminate".to_string())
}

/// A planar polygon plus, per edge, whether it runs along the geometry.
///
/// The polygon is the inside region of one cut leaf, counter-clockwise, with
/// its boundary chain edges marked so that midpoint subdivision can snap them
/// onto the geometry (§2.3).  `points`, `chain_edge` and `snap` have the same
/// length; edge `k` runs from `points[k]` to `points[(k + 1) % len]`.
#[derive(Clone, Debug)]
struct Poly {
    /// Polygon vertices, counter-clockwise, in model units.
    points: Vec<[f64; 2]>,

    /// `chain_edge[k]` is true when edge `k` (from `points[k]` to
    /// `points[k+1]`) bridges a smooth run of the domain boundary.
    chain_edge: Vec<bool>,

    /// For a chain edge, the point of the run that should become the boundary
    /// node: the run's arc-length midpoint, on the geometry.
    ///
    /// `None` for an interior edge (a leaf side or a split diagonal), which is
    /// subdivided at its straight midpoint.
    snap: Vec<Option<[f64; 2]>>,
}

/// Reduce a boundary chain to its sharp corners, and return the geometry point
/// that should become the mesh node on each resulting chord.
///
/// A polygonal approximation of a curve introduces a vertex at every sampling
/// point.  Those vertices are nearly flat in the domain polygon, and using
/// them as polygon vertices would give one flat quad per sampling point.  A
/// smooth run between two real corners is therefore replaced by one straight
/// chord; its subdivision midpoint is snapped to the run's arc-length midpoint,
/// which lies on the geometry.  The mesh boundary then consists of two chords
/// per run whose nodes are on the boundary, so `boundary_offset` stays at
/// round-off and no flat polygon vertex is created.
///
/// Returns `(corners, run_midpoints)`: the kept vertices (first and last
/// always kept) and, for each chord between consecutive kept vertices, the
/// arc-length midpoint of the dropped run on the original chain.  A turn is
/// measured as the angle away from straight; `SMOOTH_TURN_DEG` (25 degrees) is
/// the tuning threshold.  A duplicate point keeps its vertex (zero direction
/// is not a smooth run for this purpose).
fn simplify_chain(chain: &[[f64; 2]]) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
    let n = chain.len();
    if n <= 1 {
        return (chain.to_vec(), Vec::new());
    }
    let mut arc = vec![0.0f64; n];
    for i in 1..n {
        arc[i] = arc[i - 1] + dist(chain[i - 1], chain[i]);
    }
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    for i in 1..n - 1 {
        let d1 = sub(chain[i], chain[i - 1]);
        let d2 = sub(chain[i + 1], chain[i]);
        if norm(d1) <= 0.0 || norm(d2) <= 0.0 {
            keep[i] = true;
            continue;
        }
        // `angle` is the turn away from straight: zero for a straight run.
        let angle = angle_between_deg(d1, d2);
        if angle > SMOOTH_TURN_DEG {
            keep[i] = true;
        }
    }
    let mut points = Vec::new();
    let mut mids = Vec::new();
    points.push(chain[0]);
    let mut previous = 0usize;
    for i in 1..n {
        if !keep[i] {
            continue;
        }
        let half = 0.5 * (arc[previous] + arc[i]);
        mids.push(point_at_arclength(chain, &arc, half));
        points.push(chain[i]);
        previous = i;
    }
    (points, mids)
}

/// Point on the polyline `chain` at cumulative arc length `s`.
///
/// `arc` must be the cumulative lengths of `chain` (`arc[0] = 0`), as built by
/// [`simplify_chain`].  `s` is clamped to the polyline ends, so a half-way
/// value outside the chain cannot produce an extrapolated point.
fn point_at_arclength(chain: &[[f64; 2]], arc: &[f64], s: f64) -> [f64; 2] {
    let n = chain.len();
    for i in 1..n {
        if s <= arc[i] || i + 1 == n {
            let span = arc[i] - arc[i - 1];
            if span <= 0.0 {
                return chain[i];
            }
            let t = ((s - arc[i - 1]) / span).clamp(0.0, 1.0);
            return lerp(chain[i - 1], chain[i], t);
        }
    }
    chain[n - 1]
}

/// Closest point of the domain boundary to `p`.
fn nearest_boundary_point(domain: &Domain, p: [f64; 2]) -> [f64; 2] {
    let (a, b, _) = domain.nearest_segment(p);
    let ab = sub(b, a);
    let len2 = dot(ab, ab);
    if len2 <= 0.0 {
        return a;
    }
    let t = (dot(sub(p, a), ab) / len2).clamp(0.0, 1.0);
    lerp(a, b, t)
}

/// Split the polygon at every vertex whose interior angle is bad, using a
/// diagonal to the visible vertex that balances the split best.
///
/// A vertex with interior angle above `BAD_ANGLE_DEG` (160 degrees) is a flat
/// or reflex corner; the worst one is split by [`split_polygon_at`].  The
/// queue is processed depth-first; at most `MAX_SPLITS` splits are performed,
/// after which the remaining polygon is returned unsplit (subdivision may then
/// fail, which rejects the attempt).  A polygon with no bad vertex passes
/// through unchanged.  The order of the output is the split order and is not
/// geometrically meaningful.
fn split_bad_vertices(polygon: &Poly) -> Result<Vec<Poly>, String> {
    let mut queue = vec![polygon.clone()];
    let mut out = Vec::new();
    let mut splits = 0;
    while let Some(poly) = queue.pop() {
        if poly.points.len() < 3 {
            continue;
        }
        let angles = polygon_interior_angles(&poly.points);
        let worst = angles
            .iter()
            .enumerate()
            .filter(|(_, &a)| a > BAD_ANGLE_DEG)
            .max_by(|x, y| x.1.partial_cmp(y.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &a)| (i, a));
        let Some((index, _angle)) = worst else {
            out.push(poly);
            continue;
        };
        if splits >= MAX_SPLITS {
            out.push(poly);
            continue;
        }
        match split_polygon_at(&poly, index) {
            Ok((a, b)) => {
                queue.push(a);
                queue.push(b);
                splits += 1;
            }
            Err(_) => out.push(poly),
        }
    }
    Ok(out)
}

/// Split a polygon along the diagonal from vertex `index` to a visible vertex.
///
/// Candidate targets are the vertices two or more steps away.  A candidate is
/// rejected when the diagonal midpoint is outside, when the diagonal properly
/// crosses an edge, or when either new corner at `index` would be below
/// `MIN_SPLIT_ANGLE_DEG` or above `BAD_ANGLE_DEG`.  Among the visible targets
/// the one maximising the smaller of the two new angles at `index` is taken
/// (the "balanced split").  `Err` when no diagonal is visible.
fn split_polygon_at(poly: &Poly, index: usize) -> Result<(Poly, Poly), String> {
    let points = &poly.points;
    let n = points.len();
    if n < 4 || index >= n {
        return Err("polygon too small to split".to_string());
    }
    let v = points[index];
    let prev = points[(index + n - 1) % n];
    let next = points[(index + 1) % n];
    let mut best: Option<(f64, usize)> = None;
    for offset in 2..n - 1 {
        let j = (index + offset) % n;
        let q = points[j];
        if dist(v, q) <= 1.0e-12 {
            continue;
        }
        let mid = lerp(v, q, 0.5);
        if !point_in_polygon(mid, points) {
            continue;
        }
        let mut crosses = false;
        for e in 0..n {
            let a = points[e];
            let b = points[(e + 1) % n];
            if e == index || e == (index + n - 1) % n || e == j || e == (j + n - 1) % n {
                continue;
            }
            if segments_cross(v, q, a, b) {
                crosses = true;
                break;
            }
        }
        if crosses {
            continue;
        }
        let angle1 = angle_between_deg(sub(next, v), sub(q, v));
        let angle2 = angle_between_deg(sub(prev, v), sub(q, v));
        if angle1 < MIN_SPLIT_ANGLE_DEG
            || angle2 < MIN_SPLIT_ANGLE_DEG
            || angle1 > BAD_ANGLE_DEG
            || angle2 > BAD_ANGLE_DEG
        {
            continue;
        }
        let score = angle1.min(angle2);
        if best.is_none_or(|(s, _)| score > s) {
            best = Some((score, j));
        }
    }
    let Some((_, j)) = best else {
        return Err("no diagonal can split this polygon".to_string());
    };
    let first_indices: Vec<usize> = if j > index {
        (index..=j).collect()
    } else {
        (index..n).chain(0..=j).collect()
    };
    let second_indices: Vec<usize> = if j > index {
        (j..n).chain(0..=index).collect()
    } else {
        (j..=index).collect()
    };
    if first_indices.len() < 3 || second_indices.len() < 3 {
        return Err("the diagonal produced a degenerate polygon".to_string());
    }
    let first = make_sub_polygon(poly, &first_indices);
    let second = make_sub_polygon(poly, &second_indices);
    Ok((first, second))
}

/// Assemble a sub-polygon from a CCW index walk plus the closing diagonal.
fn make_sub_polygon(poly: &Poly, indices: &[usize]) -> Poly {
    let m = indices.len();
    let mut points = Vec::with_capacity(m);
    let mut chain_edge = Vec::with_capacity(m);
    let mut snap: Vec<Option<[f64; 2]>> = Vec::with_capacity(m);
    for (k, &i) in indices.iter().enumerate() {
        points.push(poly.points[i]);
        if k + 1 < m {
            // The edge from i to the next index is the original edge i.
            chain_edge.push(*poly.chain_edge.get(i).unwrap_or(&false));
            snap.push(poly.snap.get(i).copied().flatten());
        } else {
            // Closing diagonal back to the first vertex.
            chain_edge.push(false);
            snap.push(None);
        }
    }
    if polygon_area(&points) < 0.0 {
        points.reverse();
        let mut reversed_edge = vec![false; m];
        let mut reversed_snap: Vec<Option<[f64; 2]>> = vec![None; m];
        for k in 0..m {
            let source = (m + m - 2 - k) % m;
            reversed_edge[k] = chain_edge[source];
            reversed_snap[k] = snap[source];
        }
        chain_edge = reversed_edge;
        snap = reversed_snap;
    }
    Poly {
        points,
        chain_edge,
        snap,
    }
}

/// Remove consecutive duplicates and vertices that lie on the straight line
/// between their neighbours.  Both appear when a chain endpoint coincides with
/// a leaf corner; they would otherwise become flat quads.
///
/// `tol` is an absolute model length (`1e-7 * scale` at the call site).  The
/// loop is bounded by `4 * len + 8` passes as a guard against an input that
/// keeps re-creating a duplicate; it returns a polygon of fewer than three
/// points only if the input degenerates that far.
fn clean_polygon(poly: &[[f64; 2]], tol: f64) -> Vec<[f64; 2]> {
    let mut p = poly.to_vec();
    let mut guard = 0;
    loop {
        guard += 1;
        if guard > 4 * poly.len() + 8 {
            break;
        }
        let n = p.len();
        if n < 3 {
            break;
        }
        let mut changed = false;
        let mut unique: Vec<[f64; 2]> = Vec::with_capacity(n);
        for &q in &p {
            if unique
                .last()
                .is_none_or(|last: &[f64; 2]| dist(*last, q) > tol)
            {
                unique.push(q);
            }
        }
        while unique.len() > 1 && dist(unique[0], unique[unique.len() - 1]) <= tol {
            unique.pop();
            changed = true;
        }
        if unique.len() != n {
            changed = true;
            p = unique;
        }
        let n = p.len();
        if n > 3 {
            let mut keep = vec![true; n];
            for i in 0..n {
                let prev = p[(i + n - 1) % n];
                let here = p[i];
                let next = p[(i + 1) % n];
                if point_segment_distance(here, prev, next) <= tol
                    && dot(sub(here, prev), sub(next, here)) > 0.0
                {
                    keep[i] = false;
                }
            }
            if keep.iter().any(|&k| !k) {
                let mut filtered = Vec::with_capacity(n);
                for (i, &k) in keep.iter().enumerate() {
                    if k {
                        filtered.push(p[i]);
                    }
                }
                if filtered.len() >= 3 {
                    p = filtered;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    p
}

/// Does the polygon contain the leaf side `points[a] -> points[b]` in full?
///
/// The tolerance is `1e-9` in model units; `a` or `b` outside the node list is
/// false.  A true result is what makes the leaf contribute a hanging node to
/// its neighbour (the "split side" of pass 2).
fn polygon_has_edge(poly: &[[f64; 2]], a: usize, b: usize, points: &[Node]) -> bool {
    if a >= points.len() || b >= points.len() {
        return false;
    }
    let pa = [points[a][0], points[a][1]];
    let pb = [points[b][0], points[b][1]];
    let n = poly.len();
    for i in 0..n {
        let p = poly[i];
        let q = poly[(i + 1) % n];
        if (dist(p, pa) <= 1.0e-9 && dist(q, pb) <= 1.0e-9)
            || (dist(p, pb) <= 1.0e-9 && dist(q, pa) <= 1.0e-9)
        {
            return true;
        }
    }
    false
}

/// Is the node quad positively oriented (strictly counter-clockwise) and
/// inside the node list?
///
/// The `1e-15` dead band rejects a collinear or inverted corner.  This is the
/// final geometric guard of every template and subdivision, not a quality
/// metric.
fn quad_positive(q: &[usize; 4], points: &[Node]) -> bool {
    if q.iter().any(|&id| id >= points.len()) {
        return false;
    }
    let p: [[f64; 2]; 4] = [
        [points[q[0]][0], points[q[0]][1]],
        [points[q[1]][0], points[q[1]][1]],
        [points[q[2]][0], points[q[2]][1]],
        [points[q[3]][0], points[q[3]][1]],
    ];
    for k in 0..4 {
        let a = p[k];
        let b = p[(k + 1) % 4];
        let c = p[(k + 2) % 4];
        if cross(sub(b, a), sub(c, b)) <= 1.0e-15 {
            return false;
        }
    }
    true
}

/// Are the four node ids of the quad pairwise distinct?
///
/// A collapsed edge (two equal ids) would pass the cross-product test as zero
/// area only by accident; it is rejected here first so the error message is
/// about distinctness.
fn all_distinct(q: &[usize; 4]) -> bool {
    for i in 0..4 {
        for j in (i + 1)..4 {
            if q[i] == q[j] {
                return false;
            }
        }
    }
    true
}

/// True when a node lies strictly inside an edge that is used by only one
/// cell — the signature of a hanging node that was not resolved.
///
/// Only single-use edges are examined (an interior edge is shared by two
/// cells, a resolved hanging edge by two as well).  Candidate nodes are found
/// through a uniform hash whose cell size is `max(min_edge, 1% of max_edge,
/// 1e-9)`, so a huge model with one tiny edge does not create an enormous
/// table.  A node within `1e-8` of an edge interior, at a parameter strictly
/// inside `(1e-6, 1-1e-6)`, is a T-junction.  This is a final safety net; a
/// true result aborts the attempt, it is never repaired.
fn has_t_junction(points: &[Node], quads: &[[usize; 4]]) -> bool {
    if points.is_empty() || quads.is_empty() {
        return false;
    }
    let mut use_count: HashMap<(usize, usize), usize> = HashMap::new();
    for q in quads {
        for k in 0..4 {
            *use_count.entry(edge_key(q[k], q[(k + 1) % 4])).or_default() += 1;
        }
    }
    let mut lo = f64::INFINITY;
    let mut hi = 0.0f64;
    for q in quads {
        for k in 0..4 {
            let a = [points[q[k]][0], points[q[k]][1]];
            let b = [points[q[(k + 1) % 4]][0], points[q[(k + 1) % 4]][1]];
            let e = dist(a, b);
            if e > 0.0 {
                lo = lo.min(e);
                hi = hi.max(e);
            }
        }
    }
    if !(lo.is_finite() && hi > 0.0) {
        return false;
    }
    let cell = lo.max(hi * 0.01).max(1.0e-9);
    let mut hash: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    let key = |p: [f64; 2]| -> (i64, i64) {
        ((p[0] / cell).floor() as i64, (p[1] / cell).floor() as i64)
    };
    for (i, p) in points.iter().enumerate() {
        hash.entry(key([p[0], p[1]])).or_default().push(i);
    }
    for ((a, b), count) in &use_count {
        if *count != 1 || *a >= points.len() || *b >= points.len() {
            continue;
        }
        let pa = [points[*a][0], points[*a][1]];
        let pb = [points[*b][0], points[*b][1]];
        let len = dist(pa, pb);
        if len <= 0.0 {
            continue;
        }
        let xmin = pa[0].min(pb[0]);
        let xmax = pa[0].max(pb[0]);
        let ymin = pa[1].min(pb[1]);
        let ymax = pa[1].max(pb[1]);
        let i0 = (xmin / cell).floor() as i64 - 1;
        let i1 = (xmax / cell).floor() as i64 + 1;
        let j0 = (ymin / cell).floor() as i64 - 1;
        let j1 = (ymax / cell).floor() as i64 + 1;
        for gi in i0..=i1 {
            for gj in j0..=j1 {
                if let Some(ids) = hash.get(&(gi, gj)) {
                    for &id in ids {
                        if id == *a || id == *b {
                            continue;
                        }
                        let p = [points[id][0], points[id][1]];
                        let ab = sub(pb, pa);
                        let t = dot(sub(p, pa), ab) / (len * len);
                        if !(1.0e-6..=1.0 - 1.0e-6).contains(&t) {
                            continue;
                        }
                        let q = lerp(pa, pb, t);
                        if dist(p, q) <= 1.0e-8 {
                            return true;
                        }
                    }
                }
            }
        }
    }
    false
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
/// Unit tests for the cleanup-free quadtree route.
///
/// They mirror the corpus gates of `examples/alt_quadtree.rs` at a smaller
/// size: all-quad, no cracks, boundary offset at round-off, the analytic area
/// within 2%, and `verify::gate` passing.  The shapes are a square, a plate
/// with a polygonal hole, an L-bracket (re-entrant corner), a convex hexagon,
/// a quarter ring, a stepped part, and the option/error paths.
mod tests {
    use super::*;
    use crate::verify;

    /// Number of quad cells; equal to `mesh.cells.len()` asserts all-quad.
    fn quad_count(mesh: &Mesh) -> usize {
        mesh.cells
            .iter()
            .filter(|c| c.kind == crate::mesh::Kind::Quad4)
            .count()
    }

    /// A 10 x 10 square at size 1 is a pure tensor grid: all-quad, crack-free,
    /// exactly on the boundary and area-exact.
    #[test]
    fn a_square_is_all_quad_and_conforming() {
        let domain = Domain::rectangle(10.0, 10.0).unwrap();
        let sizing = Sizing::uniform(1.0).unwrap();
        let mesh = mesh(&domain, &sizing).unwrap();
        assert_eq!(quad_count(&mesh), mesh.cells.len());
        let gate = verify::gate(&mesh);
        assert_eq!(gate.triangles, 0);
        assert_eq!(gate.cracks, 0, "{gate:?}");
        assert!(verify::boundary_offset(&domain, &mesh) < 1.0e-8);
        assert!((mesh.volume() - 100.0).abs() < 1.0e-6, "{}", mesh.volume());
        assert!(gate.passes(), "{gate:?} {:?}", gate.reason());
    }

    /// The plate with a 16-segment hole is watertight around the hole: the
    /// area is within 2% of `100 * 60 - pi * 10^2`, which also checks that the
    /// hole boundary is followed and not filled.
    #[test]
    fn a_plate_with_a_hole_is_all_quad_and_watertight() {
        let domain = Domain::plate_with_hole(100.0, 60.0, 10.0, 16).unwrap();
        let sizing = Sizing::uniform(4.0).unwrap();
        let mesh = mesh(&domain, &sizing).unwrap();
        assert_eq!(quad_count(&mesh), mesh.cells.len());
        let gate = verify::gate(&mesh);
        assert_eq!(gate.cracks, 0, "{gate:?}");
        assert_eq!(gate.triangles, 0);
        assert!(verify::boundary_offset(&domain, &mesh) < 1.0e-6);
        let expected = 100.0 * 60.0 - std::f64::consts::PI * 100.0;
        let area = mesh.volume();
        assert!(
            (area - expected).abs() / expected < 0.02,
            "area {area} vs {expected}"
        );
        assert!(gate.passes(), "{gate:?} {:?}", gate.reason());
    }

    /// The L-bracket's re-entrant corner lies on grid lines; the mesh must
    /// keep the notch (area `80*20 + 20*40`) and stay conforming.
    #[test]
    fn an_l_bracket_keeps_its_reentrant_corner() {
        let domain = Domain::new(
            vec![
                [0.0, 0.0],
                [80.0, 0.0],
                [80.0, 20.0],
                [20.0, 20.0],
                [20.0, 60.0],
                [0.0, 60.0],
            ],
            vec![],
        )
        .unwrap();
        let sizing = Sizing::uniform(2.0).unwrap();
        let mesh = mesh(&domain, &sizing).unwrap();
        let gate = verify::gate(&mesh);
        assert_eq!(gate.triangles, 0);
        assert_eq!(gate.cracks, 0, "{gate:?}");
        assert!(verify::boundary_offset(&domain, &mesh) < 1.0e-6);
        let expected = 80.0 * 20.0 + 20.0 * 40.0;
        assert!(
            (mesh.volume() - expected).abs() / expected < 0.02,
            "area {} vs {expected}",
            mesh.volume()
        );
        assert!(gate.passes(), "{gate:?} {:?}", gate.reason());
    }

    /// A regular hexagon exercises slanted boundary chords and the snapping
    /// of their subdivision midpoints (`SMOOTH_TURN_DEG`); the area is checked
    /// against `3 * sqrt(3) / 2 * r^2`.
    #[test]
    fn a_convex_hexagon_is_all_quad_and_conforming() {
        let hexagon: Vec<[f64; 2]> = (0..6)
            .map(|i| {
                let a = std::f64::consts::TAU * i as f64 / 6.0;
                [10.0 + 5.0 * a.cos(), 10.0 + 5.0 * a.sin()]
            })
            .collect();
        let domain = Domain::new(hexagon, vec![]).unwrap();
        let mesh = mesh(&domain, &Sizing::uniform(1.0).unwrap()).unwrap();
        assert_eq!(quad_count(&mesh), mesh.cells.len());
        let gate = verify::gate(&mesh);
        assert_eq!(gate.triangles, 0);
        assert_eq!(gate.cracks, 0, "{gate:?}");
        assert!(verify::boundary_offset(&domain, &mesh) < 1.0e-7);
        let expected = 1.5 * 3.0f64.sqrt() * 25.0;
        assert!(
            (mesh.volume() - expected).abs() / expected < 0.02,
            "area {} vs {expected}",
            mesh.volume()
        );
        assert!(gate.passes(), "{gate:?} {:?}", gate.reason());
    }

    /// The documented coarse option: `cell_factor = 2.0` must stay all-quad
    /// and gate-passing while more than halving the element count against
    /// `1.0`, and the out-of-range factors `0.0`/`5.0` must be errors (not
    /// panics).
    #[test]
    fn a_larger_cell_factor_coarsens_the_mesh_and_stays_all_quad() {
        let domain = Domain::plate_with_hole(100.0, 60.0, 10.0, 16).unwrap();
        let sizing = Sizing::uniform(4.0).unwrap();
        let fine = mesh_with_cell_factor(&domain, &sizing, 1.0).unwrap();
        let coarse = mesh_with_cell_factor(&domain, &sizing, 2.0).unwrap();
        assert!(
            coarse.cells.len() * 2 < fine.cells.len(),
            "coarse {} vs fine {}",
            coarse.cells.len(),
            fine.cells.len()
        );
        assert_eq!(quad_count(&coarse), coarse.cells.len());
        let gate = verify::gate(&coarse);
        assert_eq!(gate.triangles, 0);
        assert_eq!(gate.cracks, 0, "{gate:?}");
        assert!(gate.passes(), "{gate:?} {:?}", gate.reason());
        assert!(verify::boundary_offset(&domain, &coarse) < 1.0e-6);
        assert!(mesh_with_cell_factor(&domain, &sizing, 0.0).is_err());
        assert!(mesh_with_cell_factor(&domain, &sizing, 5.0).is_err());
    }

    /// A non-default `Options` (`cell_factor = 2.0`, one alignment) still
    /// produces a conforming, boundary-faithful mesh: the option plumbing does
    /// not silently fall back to the defaults.
    #[test]
    fn non_default_options_still_produce_a_valid_mesh() {
        let domain = Domain::plate_with_hole(100.0, 60.0, 10.0, 16).unwrap();
        let sizing = Sizing::uniform(4.0).unwrap();
        let options = Options {
            cell_factor: 2.0,
            alignments: 1,
            ..Options::default()
        };
        let mesh = mesh_with_options(&domain, &sizing, &options).unwrap();
        assert_eq!(quad_count(&mesh), mesh.cells.len());
        let gate = verify::gate(&mesh);
        assert_eq!(gate.triangles, 0);
        assert_eq!(gate.cracks, 0, "{gate:?}");
        assert!(verify::boundary_offset(&domain, &mesh) < 1.0e-6);
        assert!(gate.passes(), "{gate:?} {:?}", gate.reason());
    }

    /// Invalid sizing and a two-point outer loop are reported as `Err`, never
    /// as a panic (the crate's "never crash" contract).
    #[test]
    fn bad_requests_are_errors_not_panics() {
        let domain = Domain::rectangle(1.0, 1.0).unwrap();
        let bad = Sizing {
            size: 0.0,
            all_quad: false,
        };
        assert!(mesh(&domain, &bad).is_err());
        let short = Domain {
            outer: vec![[0.0, 0.0], [1.0, 0.0]],
            holes: vec![],
        };
        assert!(mesh(&short, &Sizing::uniform(0.5).unwrap()).is_err());
    }

    /// A quarter ring (curved outer and inner arcs, one component with a
    /// concave junction) and a stepped part (three re-entrant corners) both
    /// build, pass the gate and stay on the boundary.
    #[test]
    fn a_ring_and_a_stepped_domain_build() {
        let mut outer: Vec<[f64; 2]> = (0..=24)
            .map(|i| {
                let a = std::f64::consts::FRAC_PI_2 * i as f64 / 24.0;
                [20.0 * a.cos(), 20.0 * a.sin()]
            })
            .collect();
        let inner: Vec<[f64; 2]> = (0..=12)
            .map(|i| {
                let a = std::f64::consts::FRAC_PI_2 * (1.0 - i as f64 / 12.0);
                [10.0 * a.cos(), 10.0 * a.sin()]
            })
            .collect();
        outer.extend(inner);
        let ring = Domain::new(outer, vec![]).unwrap();
        let ring_mesh = mesh(&ring, &Sizing::uniform(1.0).unwrap()).unwrap();
        let gate = verify::gate(&ring_mesh);
        assert_eq!(gate.triangles, 0);
        assert_eq!(gate.cracks, 0, "{gate:?}");
        assert!(verify::boundary_offset(&ring, &ring_mesh) < 1.0e-6);
        assert!(gate.passes(), "{gate:?} {:?}", gate.reason());

        let stepped = Domain::new(
            vec![
                [0.0, 0.0],
                [60.0, 0.0],
                [60.0, 40.0],
                [30.0, 40.0],
                [30.0, 30.0],
                [20.0, 30.0],
                [20.0, 40.0],
                [0.0, 40.0],
            ],
            vec![],
        )
        .unwrap();
        let stepped_mesh = mesh(&stepped, &Sizing::uniform(1.0).unwrap()).unwrap();
        let gate = verify::gate(&stepped_mesh);
        assert_eq!(gate.triangles, 0);
        assert_eq!(gate.cracks, 0, "{gate:?}");
        assert!(gate.passes(), "{gate:?} {:?}", gate.reason());
    }
}
