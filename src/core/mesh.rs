//! Mesh data model, FEA shape definitions, boundary/region markers, and topology.
//!
//! Supports all eight standard linear and quadratic FEA element shapes used by
//! `gemlab` and `pmsim`, with identical local node, edge, and outward-face
//! ordering:
//! - **2D**: [`Kind::Tri3`], [`Kind::Tri6`], [`Kind::Qua4`], [`Kind::Qua8`]
//! - **3D**: [`Kind::Tet4`], [`Kind::Tet10`], [`Kind::Hex8`], [`Kind::Hex20`]

use std::collections::BTreeMap;
use std::fmt;

use serde::{Deserialize, Serialize};

/// A point in 3D space: millimetres, Z up (`z = 0.0` by convention for planar 2D meshes).
pub type Node = [f64; 3];

/// Finite element shape types supported by `pedroso_mesher`.
///
/// Local node, edge, and outward-face orderings match `gemlab` exactly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Kind {
    /// 3-node linear triangle, counter-clockwise in the XY plane.
    Tri3,
    /// 6-node quadratic triangle: corners `0, 1, 2`, edge midsides `3=(0,1), 4=(1,2), 5=(2,0)`.
    Tri6,
    /// 4-node bilinear quadrilateral, counter-clockwise in the XY plane.
    Qua4,
    /// 8-node serendipity quadrilateral: corners `0..4`, edge midsides `4=(0,1), 5=(1,2), 6=(2,3), 7=(3,0)`.
    Qua8,
    /// 4-node linear tetrahedron: `(0,1,2)` base with `3` on the positive normal side.
    Tet4,
    /// 10-node quadratic tetrahedron: corners `0..4`, midsides `4=(0,1), 5=(1,2), 6=(2,0), 7=(0,3), 8=(1,3), 9=(2,3)`.
    Tet10,
    /// 8-node trilinear hexahedron: bottom ring `0..4` CCW from `+Z`, top ring `4..8`.
    Hex8,
    /// 20-node serendipity hexahedron: corners `0..8`, bottom midsides `8..12`, top midsides `12..16`, vertical midsides `16..20`.
    Hex20,
}

impl Kind {
    /// Topological dimension of the element (`2` for surface/planar, `3` for volume).
    pub fn dim(self) -> usize {
        match self {
            Kind::Tri3 | Kind::Tri6 | Kind::Qua4 | Kind::Qua8 => 2,
            Kind::Tet4 | Kind::Tet10 | Kind::Hex8 | Kind::Hex20 => 3,
        }
    }

    /// Total number of nodes required by this element kind.
    pub fn n_nodes(self) -> usize {
        match self {
            Kind::Tri3 => 3,
            Kind::Tri6 => 6,
            Kind::Qua4 => 4,
            Kind::Qua8 => 8,
            Kind::Tet4 => 4,
            Kind::Tet10 => 10,
            Kind::Hex8 => 8,
            Kind::Hex20 => 20,
        }
    }

    /// Number of corner (vertex) nodes, excluding mid-edge nodes.
    pub fn n_corners(self) -> usize {
        match self {
            Kind::Tri3 | Kind::Tri6 => 3,
            Kind::Qua4 | Kind::Qua8 => 4,
            Kind::Tet4 | Kind::Tet10 => 4,
            Kind::Hex8 | Kind::Hex20 => 8,
        }
    }

    /// True when the element carries quadratic mid-edge nodes.
    pub fn is_quadratic(self) -> bool {
        matches!(self, Kind::Tri6 | Kind::Qua8 | Kind::Tet10 | Kind::Hex20)
    }

    /// Corresponding linear element kind.
    pub fn linear_kind(self) -> Kind {
        match self {
            Kind::Tri3 | Kind::Tri6 => Kind::Tri3,
            Kind::Qua4 | Kind::Qua8 => Kind::Qua4,
            Kind::Tet4 | Kind::Tet10 => Kind::Tet4,
            Kind::Hex8 | Kind::Hex20 => Kind::Hex8,
        }
    }

    /// Corresponding quadratic element kind.
    pub fn quadratic_kind(self) -> Kind {
        match self {
            Kind::Tri3 | Kind::Tri6 => Kind::Tri6,
            Kind::Qua4 | Kind::Qua8 => Kind::Qua8,
            Kind::Tet4 | Kind::Tet10 => Kind::Tet10,
            Kind::Hex8 | Kind::Hex20 => Kind::Hex20,
        }
    }

    /// Lower-case canonical name used in JSON reports and `gemlab` `.msh` files.
    pub fn name(self) -> &'static str {
        match self {
            Kind::Tri3 => "tri3",
            Kind::Tri6 => "tri6",
            Kind::Qua4 => "qua4",
            Kind::Qua8 => "qua8",
            Kind::Tet4 => "tet4",
            Kind::Tet10 => "tet10",
            Kind::Hex8 => "hex8",
            Kind::Hex20 => "hex20",
        }
    }

    /// Lower-case shape key expected by `gemlab` `.msh` files.
    pub fn msh_name(self) -> &'static str {
        self.name()
    }

    /// Parse a shape name (`tri3`, `tri6`, `qua4`, `qua8`, `tet4`, `tet10`, `hex8`, `hex20`).
    pub fn from_name(name: &str) -> Option<Kind> {
        match name.to_ascii_lowercase().as_str() {
            "tri3" => Some(Kind::Tri3),
            "tri6" => Some(Kind::Tri6),
            "qua4" => Some(Kind::Qua4),
            "qua8" => Some(Kind::Qua8),
            "tet4" => Some(Kind::Tet4),
            "tet10" => Some(Kind::Tet10),
            "hex8" => Some(Kind::Hex8),
            "hex20" => Some(Kind::Hex20),
            _ => None,
        }
    }

    /// Local corner-to-corner edges in `gemlab` order.
    ///
    /// For every quadratic kind (`Tri6`, `Qua8`, `Tet10`, `Hex20`), midside node
    /// `n_corners() + e` sits on `edges()[e]`.
    pub fn edges(self) -> &'static [(usize, usize)] {
        match self {
            Kind::Tri3 | Kind::Tri6 => &[(0, 1), (1, 2), (2, 0)],
            Kind::Qua4 | Kind::Qua8 => &[(0, 1), (1, 2), (2, 3), (3, 0)],
            Kind::Tet4 | Kind::Tet10 => &[(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)],
            Kind::Hex8 | Kind::Hex20 => &[
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

    /// Outward-facing 4-node local corner faces for hexahedral kinds (`Hex8`, `Hex20`).
    pub fn faces(self) -> &'static [[usize; 4]] {
        match self {
            Kind::Hex8 | Kind::Hex20 => &[
                [0, 3, 2, 1], // z = -1, normal -Z
                [4, 5, 6, 7], // z = +1, normal +Z
                [0, 1, 5, 4], // y = -1, normal -Y
                [1, 2, 6, 5], // x = +1, normal +X
                [2, 3, 7, 6], // y = +1, normal +Y
                [3, 0, 4, 7], // x = -1, normal -X
            ],
            _ => &[],
        }
    }

    /// Outward-facing local corner loops for any 3D cell (`Tet4`, `Tet10`, `Hex8`, `Hex20`).
    pub fn face_loops(self) -> &'static [&'static [usize]] {
        const TET_FACES: &[&[usize]] = &[&[0, 3, 2], &[0, 1, 3], &[0, 2, 1], &[1, 2, 3]];
        const HEX_FACES: &[&[usize]] = &[
            &[0, 3, 2, 1],
            &[4, 5, 6, 7],
            &[0, 1, 5, 4],
            &[1, 2, 6, 5],
            &[2, 3, 7, 6],
            &[3, 0, 4, 7],
        ];
        match self {
            Kind::Tet4 | Kind::Tet10 => TET_FACES,
            Kind::Hex8 | Kind::Hex20 => HEX_FACES,
            Kind::Tri3 | Kind::Tri6 | Kind::Qua4 | Kind::Qua8 => &[],
        }
    }
}

/// One mesh element: its shape [`Kind`] and its node indices into [`Mesh::points`].
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Cell {
    /// Element shape kind.
    pub kind: Kind,
    /// Node indices into `Mesh::points` in the shape-function order of `kind`.
    pub nodes: Vec<usize>,
}

impl Cell {
    /// Construct a 3-node linear triangle (`Tri3`), counter-clockwise.
    pub fn tri3(nodes: [usize; 3]) -> Cell {
        Cell {
            kind: Kind::Tri3,
            nodes: nodes.to_vec(),
        }
    }

    /// Construct a 6-node quadratic triangle (`Tri6`).
    pub fn tri6(nodes: [usize; 6]) -> Cell {
        Cell {
            kind: Kind::Tri6,
            nodes: nodes.to_vec(),
        }
    }

    /// Construct a 4-node bilinear quadrilateral (`Qua4`), counter-clockwise.
    pub fn qua4(nodes: [usize; 4]) -> Cell {
        Cell {
            kind: Kind::Qua4,
            nodes: nodes.to_vec(),
        }
    }

    /// Construct an 8-node quadratic quadrilateral (`Qua8`).
    pub fn qua8(nodes: [usize; 8]) -> Cell {
        Cell {
            kind: Kind::Qua8,
            nodes: nodes.to_vec(),
        }
    }

    /// Construct a 4-node linear tetrahedron (`Tet4`).
    pub fn tet4(nodes: [usize; 4]) -> Cell {
        Cell {
            kind: Kind::Tet4,
            nodes: nodes.to_vec(),
        }
    }

    /// Construct a 10-node quadratic tetrahedron (`Tet10`).
    pub fn tet10(nodes: [usize; 10]) -> Cell {
        Cell {
            kind: Kind::Tet10,
            nodes: nodes.to_vec(),
        }
    }

    /// Construct an 8-node trilinear hexahedron (`Hex8`).
    pub fn hex8(nodes: [usize; 8]) -> Cell {
        Cell {
            kind: Kind::Hex8,
            nodes: nodes.to_vec(),
        }
    }

    /// Construct a 20-node quadratic hexahedron (`Hex20`).
    pub fn hex20(nodes: [usize; 20]) -> Cell {
        Cell {
            kind: Kind::Hex20,
            nodes: nodes.to_vec(),
        }
    }

    /// Number of node indices stored in this cell.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// True when the cell stores no node indices.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

/// Structural validation errors detected by [`Mesh::validate`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MeshError {
    /// A point has a NaN or infinite coordinate; the field is the point index.
    NonFinitePoint(usize),
    /// A cell references a point index that does not exist.
    IndexOutOfRange {
        cell: usize,
        node: usize,
        index: usize,
    },
    /// A cell uses the same point more than once.
    DuplicateNode { cell: usize, node: usize },
    /// A cell's node count does not match its kind.
    WrongNodeCount {
        cell: usize,
        expected: usize,
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
            } => write!(f, "cell {cell} has {found} nodes, expected {expected}"),
        }
    }
}

impl std::error::Error for MeshError {}

/// Conforming finite-element mesh with optional region and boundary markers.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Mesh {
    /// Vertex coordinates `[x, y, z]` in millimetres.
    pub points: Vec<Node>,
    /// Mesh elements indexing `points`.
    pub cells: Vec<Cell>,
    /// Optional per-point markers (empty or aligned with `points`; default `0`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub point_markers: Vec<i32>,
    /// Optional per-cell region/material markers (empty or aligned with `cells`; default `1`).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cell_markers: Vec<i32>,
    /// Marked boundary/internal edges `(marker, [p0, p1])` using corner vertex indices.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub marked_edges: Vec<(i32, [usize; 2])>,
    /// Marked boundary/internal faces `(marker, corner_nodes)` (`3` nodes for tri, `4` for quad).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub marked_faces: Vec<(i32, Vec<usize>)>,
}

impl Mesh {
    /// Construct a mesh from points and cells with default empty marker tables.
    pub fn new(points: Vec<Node>, cells: Vec<Cell>) -> Mesh {
        Mesh {
            points,
            cells,
            point_markers: Vec::new(),
            cell_markers: Vec::new(),
            marked_edges: Vec::new(),
            marked_faces: Vec::new(),
        }
    }

    /// Return the marker of point `id` (`0` when unmarked).
    pub fn point_marker(&self, id: usize) -> i32 {
        self.point_markers.get(id).copied().unwrap_or(0)
    }

    /// Set the marker of point `id`, allocating `point_markers` if needed.
    pub fn set_point_marker(&mut self, id: usize, marker: i32) {
        if id >= self.points.len() {
            return;
        }
        if self.point_markers.len() < self.points.len() {
            self.point_markers.resize(self.points.len(), 0);
        }
        self.point_markers[id] = marker;
    }

    /// Return the region/material marker of cell `id` (`1` when unmarked).
    pub fn cell_marker(&self, id: usize) -> i32 {
        self.cell_markers.get(id).copied().unwrap_or(1)
    }

    /// Set the region/material marker of cell `id`, allocating `cell_markers` if needed.
    pub fn set_cell_marker(&mut self, id: usize, marker: i32) {
        if id >= self.cells.len() {
            return;
        }
        if self.cell_markers.len() < self.cells.len() {
            self.cell_markers.resize(self.cells.len(), 1);
        }
        self.cell_markers[id] = marker;
    }

    /// Validate structural invariants: finite coordinates, exact node counts,
    /// in-range indices, and no duplicate node within any cell.
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

    /// Sum of `|cell volume|` (or `|cell area|` for 2D elements).
    pub fn volume(&self) -> f64 {
        self.cells
            .iter()
            .map(|c| crate::quality::volume(c, &self.points).abs())
            .sum()
    }

    /// Compute edge (2D) or face (3D) sharing, manifoldness, orientation consistency,
    /// and boundary watertightness (`cracks == 0`).
    pub fn topology(&self) -> Topology {
        let mut topo = Topology::default();
        let dim = self.cells.first().map(|c| c.kind.dim()).unwrap_or(3);
        let mut seen: BTreeMap<Vec<usize>, Vec<Vec<usize>>> = BTreeMap::new();
        for cell in &self.cells {
            if cell.nodes.len() < cell.kind.n_corners() {
                continue;
            }
            if cell.kind.dim() == 3 {
                for face in cell.kind.face_loops() {
                    let global: Vec<usize> = face.iter().map(|&i| cell.nodes[i]).collect();
                    let mut key = global.clone();
                    key.sort_unstable();
                    seen.entry(key).or_default().push(global);
                }
            } else {
                for &(a, b) in cell.kind.edges() {
                    let global = vec![cell.nodes[a], cell.nodes[b]];
                    let mut key = global.clone();
                    key.sort_unstable();
                    seen.entry(key).or_default().push(global);
                }
            }
        }

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

        if dim == 3 {
            let mut edges: BTreeMap<(usize, usize), usize> = BTreeMap::new();
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
            let mut vertices: BTreeMap<usize, usize> = BTreeMap::new();
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Topology {
    /// Entities used by exactly one cell (outer/hole boundary).
    pub boundary: usize,
    /// Entities shared by exactly two cells (conforming interior interface).
    pub interior: usize,
    /// Entities used by three or more cells.
    pub non_manifold: usize,
    /// Shared interior entities traversed in the same direction (inverted relative orientation).
    pub inconsistent: usize,
    /// Unclosed boundary entities (`!= 2` incident boundary faces/edges).
    pub cracks: usize,
}

impl Topology {
    /// True when the mesh is manifold, consistently oriented, and watertight (`cracks == 0`).
    pub fn is_watertight(&self) -> bool {
        self.non_manifold == 0 && self.inconsistent == 0 && self.cracks == 0
    }

    /// True when a non-empty boundary exists and closes without cracks.
    pub fn has_closed_boundary(&self) -> bool {
        self.boundary > 0 && self.cracks == 0
    }
}

fn same_direction(a: &[usize], b: &[usize]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    if a.len() == 2 {
        return (a[0] < a[1]) == (b[0] < b[1]);
    }
    rotate_to_min(a) == rotate_to_min(b)
}

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
    fn all_eight_fea_kinds_have_consistent_metadata() {
        let expected = [
            (Kind::Tri3, 2, 3, 3, false, "tri3"),
            (Kind::Tri6, 2, 6, 3, true, "tri6"),
            (Kind::Qua4, 2, 4, 4, false, "qua4"),
            (Kind::Qua8, 2, 8, 4, true, "qua8"),
            (Kind::Tet4, 3, 4, 4, false, "tet4"),
            (Kind::Tet10, 3, 10, 4, true, "tet10"),
            (Kind::Hex8, 3, 8, 8, false, "hex8"),
            (Kind::Hex20, 3, 20, 8, true, "hex20"),
        ];
        for (kind, dim, nodes, corners, quad, msh) in expected {
            assert_eq!(kind.dim(), dim);
            assert_eq!(kind.n_nodes(), nodes);
            assert_eq!(kind.n_corners(), corners);
            assert_eq!(kind.is_quadratic(), quad);
            assert_eq!(kind.msh_name(), msh);
            assert_eq!(Kind::from_name(msh), Some(kind));
            if quad {
                assert_eq!(kind.n_corners() + kind.edges().len(), kind.n_nodes());
            }
        }
    }

    #[test]
    fn two_tetrahedra_sharing_a_face_are_watertight() {
        let m = Mesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            vec![
                Cell::tet4([0, 1, 2, 3]),
                // Positively oriented partner tet sharing face (1, 2, 3) with outward [1, 3, 2]:
                Cell::tet4([1, 2, 3, 4]),
            ],
        );
        assert_eq!(m.validate(), Ok(()));
        let topo = m.topology();
        assert_eq!(topo.interior, 1);
        assert_eq!(topo.boundary, 6);
        assert_eq!(topo.inconsistent, 0);
        assert!(topo.is_watertight());
    }

    #[test]
    fn an_out_of_range_index_is_reported_with_cell_and_node() {
        let m = Mesh::new(vec![[0.0; 3]; 4], vec![Cell::qua4([0, 1, 2, 9])]);
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
        let m = Mesh::new(vec![[0.0; 3]; 4], vec![Cell::qua4([0, 1, 2, 2])]);
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
            vec![Cell::qua4([0, 1, 2, 3]), Cell::qua4([1, 4, 5, 2])],
        );
        let topo = m.topology();
        assert_eq!(topo.interior, 1);
        assert_eq!(topo.boundary, 6);
        assert_eq!(topo.inconsistent, 0);
        assert!(topo.is_watertight());
        assert_eq!(topo.cracks, 0);
    }
}
