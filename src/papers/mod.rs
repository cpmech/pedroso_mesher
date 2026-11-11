//! Clean-room implementations of individual mesh-generation journal papers and
//! university theses.
//!
//! Every submodule corresponds to a single paper (or a cohesive same-author
//! series presenting one unified method), implemented strictly as described by
//! the original authors so their claims and benchmark figures can be verified
//! independently before composing production FEA generators in [`crate::generators`].

pub mod common;
pub mod foundations;
pub mod hex;
pub mod quad_decompose;
pub mod quad_direct;
pub mod quad_grid;
pub mod quad_indirect;
pub mod smoothing;
pub mod tet;
pub mod tri;
