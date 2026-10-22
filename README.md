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
> | plate-hole 4 mm | 301 / 2 | 0.71 | 44.9° | Acceptable | 0.04 s |
> | plate-hole 2 mm | 1301 / 34 | 0.59 | 36.2° | BelowTarget | 0.3 s |
> | plate-hole 1 mm | 5132 / 234 | 0.53 | 33.0° | BelowTarget | 2.4 s |
> | l-bracket 2 mm | 561 / 34 | 0.55 | 33.3° | Acceptable | 0.04 s |
> | ring 1 mm | 251 / 46 | 0.66 | 46.8° | Acceptable | 0.03 s |
> | stepped 1 mm | 1910 / 168 | 0.39 | 22.7° | BelowTarget | 0.5 s |
>
> The scaled Jacobian clears the 0.2 target everywhere with a wide margin. The
> 45° angle target is met by the ring and nearly by the plate at 4 mm.
>
> **P2, first two packages (2026-10-22).**
>
> - *W2, optimisation smoother* — the boolean accept/reject was replaced by a
>   single objective in the spirit of Knupp (2000): per corner,
>   `(1 - sin θ)²`, plus a logarithmic barrier that grows as an element collapses.
>   Candidates are the ring centroid, two rings of eight directions, and six
>   positions along a boundary node's segment; a candidate is taken only if it
>   lowers the objective and keeps every incident quad strictly convex. Measured
>   effect: min SJ 0.35 → 0.71 and min angle 28° → 45° on the plate at 2 mm;
>   0.34 → 0.53 and 27° → 33° at 1 mm; the ring went from 29° to 47°.
> - *W1, topological clean-up* — two quads sharing an edge form a hexagon;
>   replacing the shared edge with another diagonal changes four valences. The
>   pass accepts a swap only when it lowers `Σ|valence − 4|` over the interior
>   nodes it touches, keeps both new quads convex with a usable Jacobian, and
>   does not lower the quality of the quads it replaces.
>
>   It barely fires: across the corpus it takes 0–6 swaps and moves the
>   valence-4 share by at most one point (65 → 66 %). The measurement says why —
>   of the edges with two quads, only **11 % admit a geometrically valid swap**
>   (the hexagon is usually non-convex), and of those almost none improve
>   valence without lowering quality. The irregularity is **not** caused by the
>   leftover triangles either: only 2–15 % of irregular nodes touch one. It is
>   inherent to the matching, which is one more reason the all-quad route is
>   Q-Morph's front conversion rather than post-processing.
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
> **All-quad is opt-in, and only partly works.** `Sizing::uniform(size).all_quad()`
> runs a *conforming refinement cascade*: each leftover triangle has its three
> edges split at the midpoints and is fanned into three quads around its
> centroid; every quad that gains a hanging node is split across the midpoint of
> its opposite edge, and that split propagates until it reaches the boundary.
>
> It is correct and safe — every closed component is checked for dangling edges
> and the whole result is verified before it is returned — but it rarely
> finishes:
>
> | Part | default (quads / tri) | `all_quad` |
> |---|---|---|
> | plate-hole 4 mm | 301 / 2 | **340 / 0** |
> | plate-hole 2 mm | 1301 / 34 | 1324 / 33 |
> | plate-hole 1 mm | 5132 / 234 | 5132 / 234 |
> | l-bracket 2 mm | 561 / 34 | 673 / 30 |
> | ring 1 mm | 251 / 46 | 251 / 46 |
> | stepped 1 mm | 1910 / 168 | 2011 / 166 |
>
> **Why the wall, part two.** The leftovers cluster: a cascade from one triangle
> very quickly reaches an edge that belongs to a triangle it is *not* closing,
> and splitting that edge would leave the other triangle with an edge nothing
> else has — a T-junction. Protecting those edges is necessary for validity and
> fatal for coverage. Closing a neighbourhood of leftover triangles together
> would help, but the cascade's chains then grow again (the 4 mm case costs +12 %
> elements; uncapped, the 1 mm case costs +219 %).
>
> The honest conclusion after three designs: **all-quad cannot be reached by
> post-processing the matching result.** It needs the leftovers not to exist —
> edge flips before matching, as Blossom-Quad does — or a quad-first mesher
> (paving). Both are P2 work, and the FEM check says the triangles are not
> costing accuracy, so the default stays quad-dominant.

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
