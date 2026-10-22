//! `pedroso_mesher` — clean-room, pure-Rust, MIT-licensed unstructured quad and
//! hex mesh generation for finite element analysis. The module map is: `mesh`
//! is the data model (`Mesh`, `Cell`, `Node`, `Topology`); `two_d` is the
//! planar substrate and the default quad-dominant `mesh_2d`; `quad` is the
//! gate-checked all-quad pipeline (Q-Morph + Sun closure); `config` is the
//! shipped API that selects any of the nine routes and applies the shared
//! smoother; `qmorph`, `parity`, `twostage`, `sun`, `recovery`, `grid`, `smooth`
//! and `quality` are the shared stages; `stats`, `verify` and `bench` are the
//! measurements; `draw`, `io` and `msh` are the I/O and diagnostics;
//! `alt_paving`, `alt_bipartite`, `alt_quadtree`, `alt_decompose`, `alt_hybrid`,
//! `alt_crossfield`, `alt_untangle` and `alt_l1` are the alternative routes.
//! The rules are: every algorithm is implemented from public journal papers and
//! university theses (the clean-room policy in `AGENTS.md` and the corpus
//! `INDEX.md`); no code is ported or copied, BSD code needs prior authorization,
//! non-MIT code is never read, nothing is driven from a commercial product, and
//! the crate must build for `wasm32-unknown-unknown`. The Cargo lints forbid
//! `unsafe`, `unwrap`, `expect`, `panic!`, `todo!` and `unimplemented!` in
//! non-test code, so failures are `Result` values through the crate's error
//! types. The acceptance contract is [`quality::report`] and [`verify::Gate`]
//! (all-quad, scaled Jacobian >= 0.2, corner angle >= 30 degrees, watertight,
//! no overlapping cells); a route that cannot meet it reports the measured
//! failure instead of shipping the mesh.
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

pub mod simons_godf;

pub mod simons_qgp;

pub mod simons_sizefamily;
