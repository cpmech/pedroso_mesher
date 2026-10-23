# pedroso_mesher

Unstructured **quad** and **hex** mesh generation for finite element analysis, in
pure Rust, MIT licensed. Every algorithm is implemented from public journal
papers and university theses; no code is ported (see `AGENTS.md` and
[`REFERENCES.md`](REFERENCES.md)).

The crate ships one configuration API, [`config::MeshConfig`](src/config.rs):
choose a route, optionally set that route's options, and call
[`config::generate`](src/config.rs). The generator's mesh is measured, run
through one shared smoother (interior-node relaxation, angle swaps, and an
optional angle+stretch shape pass), and returned with both the generator and the
accepted measurement. The same configuration round-trips through JSON and
builds for `wasm32-unknown-unknown`.

## The twelve routes

| Route | Character | Paper / thesis |
| --- | --- | --- |
| `qmorph` | indirect, gate-checked all-quad (4/7 corpus parts) | Owen et al. 1999, with Sun 2015/2016 closure |
| `paving` | direct, advancing boundary rows (7/7) | Blacker & Stephenson 1991 |
| `bipartite` | indirect, bipartite triangle labelling (7/7) | Georgiadis 2022 |
| `quadtree` (default) | grid, cleanup-free strongly balanced quadtree (7/7) | Rushdi 2017 / 2015 |
| `decompose` | decomposition + T0/T1/T2 templates (all-quad, raw 7/7) | Yang 2024, Miranda 2017, Ma 2017 |
| `hybrid` | fast quad-dominant Blossom band (trades triangles for speed) | Zint 2021, Remacle 2012 |
| `crossfield` | cross-field-aligned paving (7/7) | Pellenard 2011/2014, Remacle 2013 |
| `untangle` | post-pass log-barrier untangling + swaps (4/7) | Sastry 2014, Verma 2017/2018 |
| `l1` | frontal Delaunay in the L-infinity norm (7/7) | Remacle 2013 |
| `sized` | **Simons 2024 thesis** Chapter 2 grid + carved channel (7/7) | Simons 2024 |
| `features` | Simons 2024 interior-feature constraints (needs a feature polyline) | Simons 2024, Ch. 2 |
| `pipeline` | full thesis pipeline: GODF → QGP → channel → smoothing | Simons 2024, Ch. 2–5 |

`statistics.passes` is the gate: all-quad, scaled Jacobian >= 0.2, corner angle
>= 30 degrees, watertight, and no inverted or overlapping cells. A route that
cannot meet the gate reports the measured failure instead of shipping the mesh;
`generate_quad` turns that into `QuadError::Unsupported`.

## Quick start

```bash
# one route on one part (the example lists the parts: plate-hole-4mm/2mm/1mm/
# 0.6mm, l-bracket, ring, stepped)
cargo run --release --offline --example generate -- crossfield plate-hole-0.6mm

# configuration round-trip and machine-readable statistics
cargo run --release --offline --example generate -- sized plate-hole-1mm \
    --write-config sized.json --json stats.json

# the same configuration from JSON
cargo run --release --offline --example generate -- --config sized.json plate-hole-1mm
```

`--hash` prints a stable FNV-1a hash of the returned mesh; it is used by the
performance work to prove that an optimisation did not change the output.
`--no-smooth` disables the shared post-pass for a raw route measurement.

## The Simons 2024 thesis pipeline

The `sized` route is the thesis's Chapter 2 construction: a regular grid
interior, the boundary channel carved out of it, and a quadrangulation of the
remaining band (ribbon, widest-path cycle cover, templates, or the two-chain
strip fallback). The rest of the thesis is implemented as separate modules and
integrated where noted:

| Module | Thesis | State |
| --- | --- | --- |
| `simons_channel` | Ch. 2 channel + patch cover | shipped (`sized`) |
| `simons_features` | Ch. 2 interior features | shipped (`features`) |
| `simons_godf` | Ch. 3 size-aware GODF cross field | shipped (`pipeline`) |
| `simons_sizefamily` | Ch. 4 size functions / anisotropy | implemented, opt-in |
| `simons_limit` | §4.12 limit-cycle detection and removal | implemented, opt-in |
| `simons_regions` | §3.1.6–3.1.12 + §5.1.4 singularities, motorcycle graph, tiles, nautilus | implemented |
| `simons_qgp` | Ch. 5 Quantized Global Parameterization + Coons extraction | shipped (`pipeline`) |
| `simons_graded` | Ch. 5 graded tile-grid QGP | implemented, opt-in |
| `simons_smooth` | §2.2.10 angle + stretch smoothing | shipped (shared shape pass) |
| `simons_pipeline` | Ch. 2–5 integration | shipped (`pipeline`) |

The pipeline is honest about its limits: a constant-size square runs the full
GODF + QGP + stitch path, while a uniaxial size gradient falls back to the
uniform channel (a singularity-free field whose one QGP tile collapses). Those
decisions are recorded in the modules' `//!` documentation and in the measured
tables of the examples.

## Building and testing

```bash
cargo test --offline
cargo clippy --offline --all-targets -- -D warnings
cargo build --target wasm32-unknown-unknown --offline   # wasm must always build
```

No panics in non-test code, no `unsafe`, no new dependencies: failures are
`Result` values. See [`REFERENCES.md`](REFERENCES.md) for the papers and
[`CHANGELOG.md`](CHANGELOG.md) for the measured additions of this wave.
