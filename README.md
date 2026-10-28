# pedroso_mesher - My Mesh Generator

🚧 Work in progress...

This Rust crate implements several mesh generation algorithms. The goal is to generate unstructured meshes with good quality for Finite Element Analysis (FEA).

The implementation is **MIT-licensed in pure Rust**. No external library is used. Every algorithm is implemented from public journal papers and university theses. No code is ported. See `AGENTS.md` and [`REFERENCES.md`](REFERENCES.md) for further constraints and details. The code also compiles for `wasm32-unknown-unknown` and `wasm32-wasip1`.

This tool will be used by [gemlab](https://github.com/cpmech/gemlab) and will include meshes with the following **shape** (ake, "elements" in the FEA jargon) types:

* Tri3, Tri6 -- Triangles and triangles with "middle" vertices
* Qua4, Qua8 -- Quadrilaterals and quadrilaterals with "middle" vertices
* Tet4, Tet10 -- Tetrahedra and tetrahedra with "middle" vertices along their edges
* Hex8, Hex20 -- Hexahedra and hexahedra with "middle" vertices along their edges

The library includes a set of smoothing algorithms that try to improve the overall quality of the mesh.

As the meshes will be used in FEA, region classification and detection and boundary information are essential to this tool. Specifically, the following are considered:

1. Regions are defined by the user either by selecting a "closure" or a point inside a closed domain
2. Holes are defined by the user in an analogous way as the definition of regions
3. Boundary vertices, edges, and faces can be "marked" (or "tagged") by the user to identify important boundary entities
4. The information about marked boundaries will be carried over to the generated shapes
5. Boundary geometry is preserved exactly (considering finite-precision arithmetic)
6. The desired quality can be controlled by the user by giving target minium areas/volumes and angles  
7. Quality is the primary goal with computational performance being a secondary goal, however, the performance should be "good"
8. TODO
