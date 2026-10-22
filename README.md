# pedroso_mesher

Unstructured **quad** and **hex** mesh generation for finite element analysis, in
Rust, MIT licensed. No GPL, LGPL or AGPL dependency, and no external binary.

> **Status: P1 in progress.** P0 (model, quality contract, corpus) is complete.
> P1 produces **watertight, quad-dominant** planar meshes end to end:
> constrained Delaunay triangulation, Ruppert refinement, matching into quads,
> and smoothing. The P1 gate — all-quad at 45° — is nearly met on three parts and
> **not met** on the rest.
>
> | Part | Quads / triangles | min SJ | min angle | Report | Time |
> |---|---|---|---|---|---|
> | plate-hole 4 mm | 282 / 40 | 0.54 | **43.3°** | Acceptable | 0.04 s |
> | plate-hole 2 mm | 1253 / 130 | 0.53 | 41.1° | Acceptable | 0.1 s |
> | plate-hole 1 mm | 4928 / 642 | 0.50 | 32.9° | Acceptable | 1.5 s |
> | l-bracket 2 mm | 496 / 164 | 0.53 | 42.6° | Acceptable | 0.04 s |
> | ring 1 mm | 249 / 50 | 0.59 | **44.3°** | Acceptable | 0.03 s |
> | stepped 1 mm | 1834 / 320 | 0.44 | 26.1° | Acceptable | 0.2 s |
>
> The scaled Jacobian now clears the 0.2 target everywhere; the 45° angle target
> is met by the ring and nearly by the plate at 4 mm and the l-bracket.
>
> **What made the difference.** Two changes, both measured:
>
> - *Smoothing that protects the Jacobian.* Interior nodes move to their ring
>   centroid and boundary nodes slide **along** their boundary segment — the worst
>   angles are boundary quads. A move is accepted only if it raises the smallest
>   angle of every incident quad **and** does not lower the smallest scaled
>   Jacobian. The first attempt optimised the angle alone and lost 0.37 of
>   Jacobian for one degree; this one lifts the angle and the Jacobian together.
> - *Boundary sampling matched to the size field.* A 64-segment circular hole at
>   a 4 mm size is ten times finer than the interior, and the transition quads are
>   where the worst angles lived. Sampling the circle at 16 segments for 4 mm, 24
>   for 2 mm and 48 for 1 mm lifted that part from 17° to 43°. **A domain's own
>   discretisation has to suit the size field**; the mesher cannot fix a boundary
>   that is already too fine.
>
> **What is still missing: all-quad.** The matching leaves 8–15 % of the
> triangles unpaired; they are kept as `Tri3` elements so the mesh stays
> watertight. Two closure designs were implemented and rejected: a pentagon split
> (triangle + adjacent quad + interior point → two quads) lost elements through a
> bookkeeping error and, once fixed, still left cracks by consuming quads that
> bordered a second leftover triangle. The reliable route is Blossom-Quad's local
> repair — edge flips in the triangulation before matching — not a post-hoc
> polygon split.

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
