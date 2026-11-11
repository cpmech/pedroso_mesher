//! Production FEA mesh generators for 2D triangles (`Tri3`/`Tri6`),
//! 2D quadrilaterals (`Qua4`/`Qua8`), 3D tetrahedra (`Tet4`/`Tet10`), and
//! 3D hexahedra (`Hex8`/`Hex20`).

pub mod hex;
pub mod quad;
pub mod tet;
pub mod tri;
