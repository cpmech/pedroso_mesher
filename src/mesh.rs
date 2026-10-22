//! Mesh data model and topology.
//!
//! A mesh is a point list plus cells that index into it. Only the cell types of
//! the P0 contract exist so far: `Quad4` (a bilinear quadrilateral in 3D space)
//! and `Hex8` (a trilinear hexahedron). `Tri3` is an explicit leftover of a
//! planar conversion, never a promised output of the 3D path.
//!
//! # Coordinates
//!
//! Units are millimetres and Z is up, matching the CAD side. Every node has three
//! coordinates, but a planar (2D) mesh follows the convention that it lies in
//! the plane z = 0; nothing enforces this, the planar algorithms maintain it and
//! the planar quality metrics ([`crate::quality`]) assume it. Node order is the
//! shape-function order: `Quad4` counter-clockwise, so its normal is +Z, and
//! `Hex8` as the picture on the `Kind` impl block shows — bottom face
//! counter-clockwise seen from +Z, then the top face. A positively oriented cell
//! has a positive Jacobian and a positive signed volume/area; a negative one is
//! inverted.
//!
//! # Topology invariants
//!
//! The mesh is conforming: cells are connected by **node index**, so two cells
//! are neighbours only if they index the same nodes; two coincident points with
//! different indices are different nodes and form a seam. In a valid mesh every
//! interior face (3D) or edge (2D) is used by exactly two cells, whose loops run
//! in opposite directions; the boundary is closed (watertight) for a solid.
//! [`Mesh::validate`] checks the structural invariants (finite coordinates,
//! in-range indices, node count, no repeated node) and [`Mesh::topology`]
//! measures the incidence ones. Neither proves geometric validity — a cell can
//! be tagged valid and be tangled, which is [`crate::quality`]'s job — and
//! neither stores adjacency: topology is recomputed from scratch on every call.

use std::collections::HashMap;
use std::fmt;

/// A point in space: millimetres, Z up.
///
/// The alias is deliberately an array, not a struct, so a mesh is cheap to
/// serialize and every geometry helper can index it. A planar node uses
/// `z = 0.0` by convention.
pub type Node = [f64; 3];

/// The element types the mesher produces.
///
/// The node ordering of each variant is fixed by the shape functions and by the
/// metric code; the picture is on the `impl Kind` block below.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// A 4-node bilinear quadrilateral, planar and counter-clockwise in its
    /// plane. Its normal points along +Z for a cell with z = 0.
    Quad4,

    /// An 8-node trilinear hexahedron: bottom face `0 1 2 3`
    /// counter-clockwise seen from +Z, then the top face `4 5 6 7` directly
    /// above it (see the picture on the `impl Kind` block below).
    Hex8,

    /// A 3-node triangle. Only ever produced by a planar mesher as an explicit
    /// leftover; never a promised output of the 3D path. Its node order is
    /// counter-clockwise, which makes the signed area positive.
    Tri3,
}

/// Node ordering of `Hex8`, in the corner order used by the shape functions:
/// the bottom face `0 1 2 3` counter-clockwise seen from +Z, then the top face
/// `4 5 6 7` above it.
///
/// ```text
///       7-------6
///      /|      /|
///     4-------5 |
///     | 3-----|-2
///     |/      |/
///     0-------1
/// ```
///
/// `Quad4` uses the same rule in its plane: counter-clockwise, so the normal
/// points along +Z for a flat cell.
impl Kind {
    /// Spatial dimension of the cell: 2 for a planar cell, 3 for a hex.
    ///
    /// This is the topological dimension, not the ambient one: every cell still
    /// stores three coordinates per node.
    pub fn dim(self) -> usize {
        match self {
            Kind::Quad4 => 2,
            Kind::Hex8 => 3,
            Kind::Tri3 => 2,
        }
    }

    /// Number of nodes a cell of this kind must have.
    ///
    /// [`Mesh::validate`] compares `Cell::len` against this; a `Cell` whose
    /// vector has a different length is malformed.
    pub fn n_nodes(self) -> usize {
        match self {
            Kind::Quad4 => 4,
            Kind::Hex8 => 8,
            Kind::Tri3 => 3,
        }
    }

    /// Lower-case name used in reports and in the JSON input: `quad4`, `hex8`
    /// or `tri3`.
    pub fn name(self) -> &'static str {
        match self {
            Kind::Quad4 => "quad4",
            Kind::Hex8 => "hex8",
            Kind::Tri3 => "tri3",
        }
    }

    /// Local edges as node pairs, in a fixed order. A hex has 12, a quad 4 and
    /// a triangle 3.
    ///
    /// The order matters to the callers: `quality::dihedral_angles` returns one
    /// angle per hex edge in this order, and `quality::aspect_ratio` scans all
    /// of them. The hex list is the bottom ring `0-1-2-3`, the top ring
    /// `4-5-6-7`, then the four verticals `0-4, 1-5, 2-6, 3-7`; the ring order
    /// matches the node loops used by [`Kind::faces`].
    pub fn edges(self) -> &'static [(usize, usize)] {
        match self {
            Kind::Quad4 => &[(0, 1), (1, 2), (2, 3), (3, 0)],
            Kind::Tri3 => &[(0, 1), (1, 2), (2, 0)],
            Kind::Hex8 => &[
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),
            ],
        }
    }

    /// Local faces as node loops, outward-facing, for `Hex8` only. Empty for a
    /// planar cell: its faces are its edges.
    ///
    /// Each loop is counter-clockwise seen from outside the cell, so the Newell
    /// normal of the loop points out (this is what `quality::dihedral_angles`
    /// relies on). The six faces are the bottom and top plates followed by the
    /// four sides, in the order `-Z, +Z, -Y, +X, +Y, -X` for a cell with the
    /// node order of [`Kind`].
    pub fn faces(self) -> &'static [[usize; 4]] {
        match self {
            Kind::Quad4 | Kind::Tri3 => &[],
            Kind::Hex8 => &[
                [0, 3, 2, 1], // z = 0, normal -Z
                [4, 5, 6, 7], // z = 1, normal +Z
                [0, 1, 5, 4], // y = 0, normal -Y
                [1, 2, 6, 5], // x = 1, normal +X
                [2, 3, 7, 6], // y = 1, normal +Y
                [3, 0, 4, 7], // x = 0, normal -X
            ],
        }
    }
}

/// One cell: its kind and its node indices into `Mesh::points`.
///
/// The struct is plain data, and the constructors below are the intended path:
/// the array type fixes the kind and the node count, but they do not validate
/// the indices or the ordering. Run [`Mesh::validate`] on the assembled mesh. Two
/// cells are equal when their kind and their full index vector are equal; the
/// same geometric cell with a rotated start node is a different `Cell`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cell {
    /// Element type; fixes the node count and the ordering of `nodes`.
    pub kind: Kind,

    /// Node indices into `Mesh::points`, in the shape-function order of `kind`
    /// (see [`Kind`]). The length must equal `kind.n_nodes()`, every index must
    /// be in range, and no index may repeat — [`Mesh::validate`] enforces all
    /// three.
    pub nodes: Vec<usize>,
}

impl Cell {
    /// A quadrilateral from four node indices, counter-clockwise.
    pub fn quad4(nodes: [usize; 4]) -> Cell {
        Cell {
            kind: Kind::Quad4,
            nodes: nodes.to_vec(),
        }
    }

    /// A triangle from three node indices, counter-clockwise.
    pub fn tri3(nodes: [usize; 3]) -> Cell {
        Cell {
            kind: Kind::Tri3,
            nodes: nodes.to_vec(),
        }
    }

    /// A hexahedron from eight node indices, bottom ring then top ring.
    pub fn hex8(nodes: [usize; 8]) -> Cell {
        Cell {
            kind: Kind::Hex8,
            nodes: nodes.to_vec(),
        }
    }

    /// Number of node indices stored in this cell.
    ///
    /// This is the stored length, not `kind.n_nodes()`; the two differ only for
    /// a hand-built invalid cell, which [`Mesh::validate`] reports.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// True when the cell stores no node indices.
    ///
    /// Only a hand-built invalid cell is empty; the `quad4`, `tri3` and `hex8`
    /// constructors always produce the full count. The method is kept for API
    /// symmetry with [`Cell::len`] and for Clippy's `len_without_is_empty`
    /// convention.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

/// What is wrong with a mesh, as found by [`Mesh::validate`].
///
/// The first error in validation order is returned, not all of them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MeshError {
    /// A point has a NaN or infinite coordinate; the field is the point index.
    NonFinitePoint(usize),

    /// A cell references a point index that does not exist.
    IndexOutOfRange {
        /// Index of the cell in `Mesh::cells`.
        cell: usize,

        /// Position in the cell's node list (0-based), not the point index.
        node: usize,

        /// The offending point index.
        index: usize,
    },

    /// A cell uses the same point more than once.
    DuplicateNode {
        /// Index of the cell in `Mesh::cells`.
        cell: usize,

        /// The repeated point index.
        node: usize,
    },

    /// A cell's node count does not match its kind.
    WrongNodeCount {
        /// Index of the cell in `Mesh::cells`.
        cell: usize,

        /// `kind.n_nodes()`.
        expected: usize,

        /// `cell.len()`, the stored count.
        found: usize,
    },
}

impl fmt::Display for MeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MeshError::NonFinitePoint(i) => write!(f, "point {i} has a non-finite coordinate"),
            MeshError::IndexOutOfRange { cell, node, index } => write!(
                f,
                "cell {cell} node {node} indexes point {index}, out of range"
            ),
            MeshError::DuplicateNode { cell, node } => {
                write!(f, "cell {cell} uses point {node} more than once")
            }
            MeshError::WrongNodeCount {
                cell,
                expected,
                found,
            } => {
                write!(f, "cell {cell} has {found} nodes, expected {expected}")
            }
        }
    }
}

impl std::error::Error for MeshError {}

/// A point list and the cells that index it.
///
/// The two arrays are independent: `points` may contain unused nodes and two
/// points may share coordinates (they are still distinct nodes, because
/// connectivity is by index). `Default` is the empty mesh. Construction does not
/// validate, so call [`Mesh::validate`] when the mesh comes from outside the
/// crate (the JSON reader, a test fixture, another algorithm).
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    /// All nodes, in no particular order; cells index this array.
    pub points: Vec<Node>,

    /// The cells, each indexing `points` with its `kind`'s node order.
    pub cells: Vec<Cell>,
}

impl Mesh {
    /// Build a mesh from a point list and cells, without validating either.
    ///
    /// This is a plain constructor so that hot paths do not pay for a check they
    /// already know holds; call [`Mesh::validate`] at the boundary where the
    /// data is untrusted.
    pub fn new(points: Vec<Node>, cells: Vec<Cell>) -> Mesh {
        Mesh { points, cells }
    }

    /// Check the structural invariants every later stage assumes: finite
    /// coordinates, in-range indices, the right node count, no repeated node in
    /// a cell.
    ///
    /// The order is points first (a finite coordinate is a precondition for all
    /// geometry), then the cells in index order, and within a cell the node
    /// count before the indices (a wrong count makes the per-node checks
    /// meaningless). Only the first error is returned. Duplicate *points* are
    /// allowed — coincident coordinates are a modelling choice, not a structural
    /// error — and geometric validity (orientation, inversion, overlaps,
    /// z = 0 for planar cells) is deliberately not checked here; see
    /// [`crate::quality`] and [`crate::verify`].
    pub fn validate(&self) -> Result<(), MeshError> {
        for (i, p) in self.points.iter().enumerate() {
            if !p.iter().all(|v| v.is_finite()) {
                return Err(MeshError::NonFinitePoint(i));
            }
        }
        for (c, cell) in self.cells.iter().enumerate() {
            let expected = cell.kind.n_nodes();
            if cell.len() != expected {
                return Err(MeshError::WrongNodeCount {
                    cell: c,
                    expected,
                    found: cell.len(),
                });
            }
            for (n, &index) in cell.nodes.iter().enumerate() {
                if index >= self.points.len() {
                    return Err(MeshError::IndexOutOfRange {
                        cell: c,
                        node: n,
                        index,
                    });
                }
                // Compare with the prefix only: each unordered pair is tested
                // once, and a cell has at most eight nodes, so the scan is
                // cheap.
                if cell.nodes[..n].contains(&index) {
                    return Err(MeshError::DuplicateNode {
                        cell: c,
                        node: index,
                    });
                }
            }
        }
        Ok(())
    }

    /// Sum of |cell volume|, in the mesh unit cubed (a planar cell contributes
    /// its area, unit squared).
    ///
    /// The absolute value is deliberate: signed volumes of cells with opposite
    /// orientation cancel, and a caller asking for "the volume of this mesh"
    /// wants the total material, not the orientation defect. Orientation is
    /// checked separately by [`crate::quality`] and [`crate::verify`]. A
    /// degenerate cell contributes 0.
    pub fn volume(&self) -> f64 {
        self.cells
            .iter()
            .map(|c| crate::quality::volume(c, &self.points).abs())
            .sum()
    }

    /// Face (3D) or edge (2D) sharing, and whether the mesh is closed and
    /// consistently oriented.
    ///
    /// Every cell contributes its faces (a hex) or its edges (a planar cell) as
    /// global node loops. The loops are grouped by the **sorted** node set, so
    /// the two cells on either side of an interface meet on the same key
    /// regardless of the order in which they list the nodes. A group of one is
    /// boundary, two is a conforming interior interface (and the loops must run
    /// in opposite directions, otherwise the cells are inconsistently
    /// oriented), more than two is non-manifold.
    ///
    /// The boundary is then checked for closure: in 3D every edge of a boundary
    /// face must be shared by exactly two boundary faces; in 2D every vertex of
    /// the boundary polyline must have exactly two boundary edges. The global
    /// `dim` is taken from the first cell, so a mesh mixing planar and solid
    /// cells is not supported (and an empty mesh is treated as 3D, giving all
    /// zero counts).
    pub fn topology(&self) -> Topology {
        let mut topo = Topology::default();
        // Mixed-dimension meshes are not produced by any route; the first cell
        // decides which closure rule applies to the whole mesh.
        let dim = self.cells.first().map(|c| c.kind.dim()).unwrap_or(3);
        let mut seen: HashMap<Vec<usize>, Vec<Vec<usize>>> = HashMap::new();
        for cell in &self.cells {
            let loops: Vec<Vec<usize>> = if cell.kind.dim() == 3 {
                cell.kind.faces().iter().map(|f| f.to_vec()).collect()
            } else {
                cell.kind.edges().iter().map(|e| vec![e.0, e.1]).collect()
            };
            for local in loops {
                let global: Vec<usize> = local.iter().map(|&i| cell.nodes[i]).collect();
                // Canonical key: the node set, order discarded. `global` keeps
                // the direction for the `inconsistent` test below.
                let mut key = global.clone();
                key.sort_unstable();
                seen.entry(key).or_default().push(global);
            }
        }
        // Count incidence; a group of 2 is where watertightness is decided.
        for uses in seen.values() {
            match uses.len() {
                0 => {}
                1 => topo.boundary += 1,
                2 => {
                    topo.interior += 1;
                    if same_direction(&uses[0], &uses[1]) {
                        topo.inconsistent += 1;
                    }
                }
                _ => topo.non_manifold += 1,
            }
        }

        // A closed surface: every edge of a boundary face is shared by exactly
        // two boundary faces (3D), or every vertex of the boundary polyline has
        // exactly two boundary edges (2D). This is what "watertight" means —
        // *not* "no boundary", which no solid can satisfy.
        if dim == 3 {
            let mut edges: HashMap<(usize, usize), usize> = HashMap::new();
            for uses in seen.values() {
                if uses.len() != 1 {
                    continue;
                }
                let face = &uses[0];
                for i in 0..face.len() {
                    let (a, b) = (face[i], face[(i + 1) % face.len()]);
                    *edges.entry((a.min(b), a.max(b))).or_insert(0) += 1;
                }
            }
            topo.cracks = edges.values().filter(|&&n| n != 2).count();
        } else {
            let mut vertices: HashMap<usize, usize> = HashMap::new();
            for uses in seen.values() {
                if uses.len() != 1 {
                    continue;
                }
                for &v in &uses[0] {
                    *vertices.entry(v).or_insert(0) += 1;
                }
            }
            topo.cracks = vertices.values().filter(|&&n| n != 2).count();
        }
        topo
    }
}

/// Sharing statistics of a mesh's faces (3D) or edges (2D).
///
/// All fields are counts of distinct entities, not of cells; they have no unit.
/// A conforming solid has `non_manifold == 0`, `inconsistent == 0` and
/// `cracks == 0`; its `boundary` and `interior` counts are then the number of
/// surface entities and of internal interfaces.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Topology {
    /// Faces or edges used by exactly one cell: the outer surface.
    pub boundary: usize,

    /// Used by exactly two cells (a conforming interior interface).
    pub interior: usize,

    /// Used by more than two cells (a non-manifold interface).
    pub non_manifold: usize,

    /// Interior pairs whose loops run the same way, i.e. opposite normals
    /// facing each other — the two cells are inconsistently oriented.
    pub inconsistent: usize,

    /// Boundary entities that do not close the surface: edges of a boundary
    /// face not shared by exactly two boundary faces (3D), or vertices of the
    /// boundary with other than two boundary edges (2D).
    pub cracks: usize,
}

impl Topology {
    /// Closed, manifold and consistently oriented — the boundary is a
    /// watertight surface, which is what a solid mesh must be.
    ///
    /// This is `non_manifold == 0 && inconsistent == 0 && cracks == 0`; it does
    /// not require `boundary > 0`, because a mesh whose topology closes without
    /// a free boundary (a periodic mesh) is also watertight. `has_closed_boundary`
    /// is the stronger "there is a boundary and it closes" test.
    pub fn is_watertight(&self) -> bool {
        self.non_manifold == 0 && self.inconsistent == 0 && self.cracks == 0
    }

    /// A hook for the tests: a boundary that is present and closed.
    ///
    /// Weaker than [`Topology::is_watertight`]: it only looks at `boundary` and
    /// `cracks`, so it can be true while interior faces are non-manifold or
    /// inconsistently oriented.
    pub fn has_closed_boundary(&self) -> bool {
        self.boundary > 0 && self.cracks == 0
    }
}

/// Do two loops traverse their shared nodes in the same direction?
///
/// Rotating to the smallest node keeps the direction for a loop of three or
/// more, but cannot for an edge: `[1, 2]` and `[2, 1]` both rotate to `[1, 2]`.
/// Edges are therefore compared by their own direction. Two adjacent faces with
/// the same direction face the same way, i.e. their outward normals point into
/// each other, which is the inconsistent-orientation case.
fn same_direction(a: &[usize], b: &[usize]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    if a.len() == 2 {
        return (a[0] < a[1]) == (b[0] < b[1]);
    }
    rotate_to_min(a) == rotate_to_min(b)
}

/// Rotate a loop so its smallest node comes first, keeping the direction.
///
/// The rotation is only a canonical start point, not a reversal, so two loops
/// that are rotations of each other compare equal while a reversed loop does
/// not. This assumes the node set is fixed by the caller (it is: the loops being
/// compared came from the same sorted key).
fn rotate_to_min(v: &[usize]) -> Vec<usize> {
    let mut out = v.to_vec();
    let mut min = usize::MAX;
    let mut at = 0;
    for (i, &x) in out.iter().enumerate() {
        if x < min {
            min = x;
            at = i;
        }
    }
    out.rotate_left(at);
    out
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn cube() -> Mesh {
        Mesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
            vec![Cell::hex8([0, 1, 2, 3, 4, 5, 6, 7])],
        )
    }

    #[test]
    fn a_unit_cube_is_valid_and_has_unit_volume() {
        let m = cube();
        assert_eq!(m.validate(), Ok(()));
        assert!((m.volume() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn an_out_of_range_index_is_reported_with_cell_and_node() {
        let m = Mesh::new(vec![[0.0; 3]; 4], vec![Cell::quad4([0, 1, 2, 9])]);
        assert_eq!(
            m.validate(),
            Err(MeshError::IndexOutOfRange {
                cell: 0,
                node: 3,
                index: 9
            })
        );
    }

    #[test]
    fn a_repeated_node_is_reported() {
        let m = Mesh::new(vec![[0.0; 3]; 4], vec![Cell::quad4([0, 1, 2, 2])]);
        assert_eq!(
            m.validate(),
            Err(MeshError::DuplicateNode { cell: 0, node: 2 })
        );
    }

    #[test]
    fn the_wrong_number_of_nodes_is_reported() {
        let m = Mesh::new(
            vec![[0.0; 3]; 4],
            vec![Cell {
                kind: Kind::Hex8,
                nodes: vec![0, 1, 2, 3],
            }],
        );
        assert_eq!(
            m.validate(),
            Err(MeshError::WrongNodeCount {
                cell: 0,
                expected: 8,
                found: 4
            })
        );
    }

    #[test]
    fn a_non_finite_point_is_reported() {
        let mut m = cube();
        m.points[3][1] = f64::NAN;
        assert_eq!(m.validate(), Err(MeshError::NonFinitePoint(3)));
    }

    #[test]
    fn a_single_hex_is_a_closed_solid() {
        let topo = cube().topology();
        assert_eq!(topo.boundary, 6);
        assert_eq!(topo.interior, 0);
        assert_eq!(topo.non_manifold, 0);
        assert_eq!(topo.cracks, 0);
        assert!(topo.is_watertight());
    }

    #[test]
    fn two_stacked_hexes_share_a_face_and_are_watertight_when_oriented() {
        let m = Mesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 2.0],
                [1.0, 0.0, 2.0],
                [1.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
            ],
            vec![
                Cell::hex8([0, 1, 2, 3, 4, 5, 6, 7]),
                Cell::hex8([4, 5, 6, 7, 8, 9, 10, 11]),
            ],
        );
        let topo = m.topology();
        assert_eq!(topo.interior, 1);
        assert_eq!(topo.boundary, 10);
        assert_eq!(topo.inconsistent, 0);
        assert!(topo.is_watertight());
        assert_eq!(topo.cracks, 0);
    }

    #[test]
    fn a_two_by_two_by_two_block_is_watertight() {
        let index = |x: usize, y: usize, z: usize| x + 3 * y + 9 * z;
        let mut points = Vec::new();
        for z in 0..3 {
            for y in 0..3 {
                for x in 0..3 {
                    points.push([x as f64, y as f64, z as f64]);
                }
            }
        }
        let mut cells = Vec::new();
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    cells.push(Cell::hex8([
                        index(x, y, z),
                        index(x + 1, y, z),
                        index(x + 1, y + 1, z),
                        index(x, y + 1, z),
                        index(x, y, z + 1),
                        index(x + 1, y, z + 1),
                        index(x + 1, y + 1, z + 1),
                        index(x, y + 1, z + 1),
                    ]));
                }
            }
        }
        let topo = Mesh::new(points, cells).topology();
        assert_eq!(topo.interior, 12);
        assert_eq!(topo.boundary, 24);
        assert_eq!(topo.inconsistent, 0);
        assert!(topo.is_watertight());
    }

    #[test]
    fn two_hexes_sharing_a_face_with_the_same_orientation_are_flagged() {
        let m = Mesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 2.0],
                [1.0, 0.0, 2.0],
                [1.0, 1.0, 2.0],
                [0.0, 1.0, 2.0],
            ],
            vec![
                Cell::hex8([0, 1, 2, 3, 4, 5, 6, 7]),
                // same bottom/top order: the shared face runs the same way
                Cell::hex8([8, 9, 10, 11, 4, 5, 6, 7]),
            ],
        );
        assert_eq!(m.topology().inconsistent, 1);
    }

    #[test]
    fn two_quads_sharing_an_edge_are_watertight() {
        let m = Mesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
            ],
            vec![Cell::quad4([0, 1, 2, 3]), Cell::quad4([1, 4, 5, 2])],
        );
        let topo = m.topology();
        assert_eq!(topo.interior, 1);
        assert_eq!(topo.boundary, 6);
        assert_eq!(topo.inconsistent, 0);
        assert!(topo.is_watertight());
        assert_eq!(topo.cracks, 0);
    }
}
