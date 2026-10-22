# pedroso_mesher

Unstructured **quad** and **hex** mesh generation for finite element analysis, in
Rust, MIT licensed. No GPL, LGPL or AGPL dependency, and no external binary.

> **Status: P0.** The data model, the quality contract and the benchmark corpus
> exist and are tested. The meshing algorithms (P1 onward) are not written yet —
> this crate is the ground they are measured against.

## What P0 provides

- **Mesh model** (`mesh`): `Quad4` and `Hex8`, node ordering documented, topology
  analysis (interior/boundary sharing, orientation consistency, cracks in the
  boundary surface).
- **Quality metrics** (`quality`): scaled Jacobian at every corner from the
  trilinear/bilinear map, signed volume by Gauss integration, hex dihedral
  angles, quad corner angles, aspect ratio, and a `Report` with a `Verdict`
  (`Acceptable` / `BelowTarget` / `Refused`) that names its reasons.
- **Benchmark corpus** (`bench`): twelve parts — four planar, eight solid — each
  with the geometry, the size range and the quality bar it must reach.
- **JSON mesh I/O** (`io`) and a CLI (`parts`, `quality`).

## Quality contract

The numbers a mesh must meet, from Knupp's Jacobian framework:

| Metric | Hard | Target |
|---|---|---|
| Scaled Jacobian (all corners) | > 0 | ≥ 0.2 |
| Hex dihedral angles | (0°, 180°) | 30°–150° |
| Quad corner angles | (0°, 180°) | 45°–135° |
| Aspect ratio | < 100 | ≤ 10 |
| 3D acceptance on the corpus | — | min SJ ≥ 0.1 on 6 of 8 parts |

A mesh is **refused**, with a located reason, rather than shipped when an element
is inverted or a hard limit is broken. `Report::reasons` is the message the user
reads.

## Usage

```bash
cargo test
cargo run -- parts
cargo run -- quality examples/cube.json
cargo run -- quality examples/cube.json --json
```

The mesh JSON is deliberately small until an MSH/VTU writer exists:

```json
{
  "points": [[0,0,0], [1,0,0], [1,1,0], [0,1,0], [0,0,1], [1,0,1], [1,1,1], [0,1,1]],
  "cells": [{"kind": "hex8", "nodes": [0,1,2,3,4,5,6,7]}]
}
```

Exit codes: `0` acceptable or below target, `1` refused, `2` bad input.

## Node ordering

`Hex8`: bottom face `0 1 2 3` counter-clockwise seen from +Z, top face `4 5 6 7`
above it. `Quad4`: counter-clockwise, normal along +Z for a flat cell. The
diagrams are in `src/mesh.rs`; `Kind::faces` lists the outward faces the dihedral
angles are computed from.

## Testing

```bash
cargo test           # unit tests and the doctest
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

The unit cells are hand-checked rather than generated: a unit cube has scaled
Jacobian 1 and volume 1; a sheared cube is exactly `1/√2`; a mirrored cube is
negative; a cube's twelve dihedrals are 90°. A metric that drifts breaks one of
those, not a golden file.

## Licence

MIT. See [LICENSE](LICENSE).
