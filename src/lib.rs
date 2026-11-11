//! `pedroso_mesher` — clean-room, pure-Rust, MIT-licensed mesh generation for
//! Finite Element Analysis (`Tri3`, `Tri6`, `Qua4`, `Qua8`, `Tet4`, `Tet10`,
//! `Hex8`, `Hex20`).
//!
//! ## Crate Architecture
//! - [`core`] — Shared FEA mesh model, markers, PSLG/PLC domains, exact Edmonds
//!   Blossom graph matching, quality metrics, strict geometric verification,
//!   quadratic mid-edge elevation, and `JSON` / `.msh` / `.vtu` / `SVG` I/O.
//!   All linear algebra uses [`algebra_linear`].
//! - [`papers`] — Dedicated, unmodified clean-room modules organized one-per-paper
//!   (or cohesive same-author paper series) across `foundations`, `tri`,
//!   `quad_direct`, `quad_indirect`, `quad_grid`, `quad_decompose`, `smoothing`,
//!   `tet`, and `hex`.
//! - [`generators`] — Production quality-constrained FEA generators (`tri`,
//!   `quad`, `tet`, `hex`) built atop empirically validated paper methods.
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

pub mod core;
pub mod generators;
pub mod papers;

// Core module aliases at crate root for ergonomic access and test/example continuity.
pub use core::bench;
pub use core::io::json as io;
pub use core::io::msh;
pub use core::io::svg as draw;
pub use core::io::vtu;
pub use core::mesh;
pub use core::quality;
pub use core::stats;
pub use core::verify;

// Paper & legacy compatibility aliases preserving existing tests and examples
// during Phase 2 unbundling.
pub use papers::common::{config, parity, quad, smooth, two_d, twostage};
pub use papers::foundations::delaunay;
pub use papers::quad_decompose::yang_2024 as alt_decompose;
pub use papers::quad_direct::blacker_1991 as alt_paving;
pub use papers::quad_direct::remacle_2013 as alt_l1;
pub use papers::quad_direct::simons_2024::ch2_channel as simons_channel;
pub use papers::quad_direct::simons_2024::ch2_features as simons_features;
pub use papers::quad_direct::simons_2024::ch3_godf as simons_godf;
pub use papers::quad_direct::simons_2024::ch4_limit as simons_limit;
pub use papers::quad_direct::simons_2024::ch4_size_family as simons_sizefamily;
pub use papers::quad_direct::simons_2024::ch5_graded as simons_graded;
pub use papers::quad_direct::simons_2024::ch5_qgp as simons_qgp;
pub use papers::quad_direct::simons_2024::ch5_regions as simons_regions;
pub use papers::quad_direct::simons_2024::ch5_seeded as simons_seeded;
pub use papers::quad_direct::simons_2024::pipeline as simons_pipeline;
pub use papers::quad_direct::simons_2024::sized;
pub use papers::quad_direct::simons_2024::smooth as simons_smooth;
pub use papers::quad_grid::rushdi_2015_2017 as alt_quadtree;
pub use papers::quad_indirect::georgiadis_2022 as alt_bipartite;
pub use papers::quad_indirect::owen_1999 as qmorph;
pub use papers::quad_indirect::pellenard_2014 as alt_crossfield;
pub use papers::quad_indirect::sun_2015_2016 as sun;
pub use papers::quad_indirect::zint_2021 as alt_hybrid;
pub use papers::smoothing::sastry_2014 as alt_untangle;

// Primary top-level type & function re-exports.
pub use bench::{Part, PARTS};
pub use core::error::{Error, Result};
pub use core::high_order::elevate_to_quadratic;
pub use io::{read_mesh_json, write_mesh_json};
pub use mesh::{Cell, Kind, Mesh, MeshError, Node, Topology};
pub use msh::{to_msh, write_msh};
pub use quad::{mesh_quads, QuadError, QuadMesh};
pub use quality::{report, report_with, Report, Thresholds, Verdict};
pub use two_d::{mesh_2d, Domain, Sizing, TwoDResult, TwoDStats};
