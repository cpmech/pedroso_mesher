//! `pedroso_mesher` — unstructured quad and hex meshing for finite element
//! analysis, MIT licensed.
//!
//! This is **P0**: the data model, the quality contract and the benchmark
//! corpus. The meshing algorithms of P1 onward are built on these, and every
//! one of them is judged by [`quality::report`].
//!
//! ```
//! use pedroso_mesher::{Cell, Mesh, quality};
//!
//! let points = vec![
//!     [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
//!     [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
//! ];
//! let mesh = Mesh::new(points, vec![Cell::hex8([0, 1, 2, 3, 4, 5, 6, 7])]);
//! assert!((mesh.volume() - 1.0).abs() < 1e-12);
//! assert_eq!(quality::report(&mesh).verdict, quality::Verdict::Acceptable);
//! ```

pub mod bench;
pub mod draw;
pub mod grid;
pub mod io;
pub mod mesh;
pub mod msh;
pub mod parity;
pub mod qmorph;
pub mod quad;
pub mod quadify;
pub mod quality;
pub mod recovery;
pub mod smooth;
pub mod stats;
pub mod sun;
pub mod two_d;
pub mod twostage;
pub mod verify;

pub use bench::{Part, PARTS};
pub use io::read_mesh_json;
pub use mesh::{Cell, Kind, Mesh, MeshError, Node, Topology};
pub use msh::{to_msh, write_msh};
pub use quad::{mesh_quads, QuadError, QuadMesh};
pub use quality::{report, report_with, Report, Thresholds, Verdict};
pub use two_d::{mesh_2d, Domain, Sizing, TwoDResult, TwoDStats};

pub mod alt_paving;

pub mod alt_hybrid;

pub mod alt_untangle;

pub mod alt_quadtree;

pub mod alt_bipartite;

pub mod alt_l1;

pub mod alt_decompose;

pub mod alt_crossfield;

pub mod config;

pub mod simons_channel;
