# Changelog

Notable changes to `pedroso_mesher`. The crate is pre-1.0: the configuration
types are stable, the internal route modules are not. Measurements are on the
seven-part corpus (`plate-hole-4mm/2mm/1mm/0.6mm`, `l-bracket`, `ring`,
`stepped`) with the route's default options and the shared post-pass.

## [Unreleased] — thesis wave (2026-10-23)

### Added

- **`config::MeshConfig`, the twelve-route product API.** One configuration
  selects a route, carries its options and applies one shared post-pass:
  `qmorph`, `paving`, `bipartite`, `quadtree` (default), `decompose`, `hybrid`,
  `crossfield`, `untangle`, `l1`, `sized`, `features`, `pipeline`. The same
  configuration round-trips through JSON and builds for
  `wasm32-unknown-unknown`.
- **The Simons 2024 thesis pipeline**, split into modules with their own
  measured tests: `simons_channel` (Ch. 2 grid + carved channel + patch cover),
  `simons_features` (Ch. 2 interior-feature constraints), `simons_godf`
  (Ch. 3 size-aware Globally-Optimal Direction Field), `simons_sizefamily`
  (Ch. 4 size families and anisotropy), `simons_limit` (§4.12 limit cycles),
  `simons_regions` (§3.1.6–3.1.12 singularities, separatrices and motorcycle
  graph, §5.1.4 tiles), `simons_qgp` (Ch. 5 Quantized Global Parameterization
  and Coons extraction), `simons_graded` (graded tile-grid QGP),
  `simons_smooth` (§2.2.10 smoothing) and `simons_pipeline` (the Ch. 2–5
  integration behind the `pipeline` route).
- **The nautilus ring (Section 5.1.4).** A singularity-free field over a domain
  with a hole is kept as the thesis's self-referential annular tile and meshed
  by angular correspondence about the hole; the stitched circular annulus is
  measured at 192 kept interior quads, coverage 1.000, min scaled Jacobian
  > 0.9 and min angle > 80 degrees (`simons_regions::stitch_demo`).
- **Graded QGP pipeline.** `simons_graded` constructs a simple-tile grid whose
  tile lines follow the cumulative size metric and runs the Chapter 5 integer
  counts and Coons extraction on it, so a size field can be realized without a
  degenerate single-tile solve.
- **Target-area term in the smoother.** `smooth_with_targets` and
  `SmoothOptions::size_weight` add the thesis's size term
  `1 / (1 + w |ln(sqrt(area) / target)|)` to the local objective; the pipeline
  exposes it. It is opt-in because on a uniform nominal size it loses gate
  passes.
- `examples/generate --hash`: a stable FNV-1a hash of the returned mesh, used to
  prove that a performance change did not alter the output.

### Changed

- The shipped route count grew from nine to **twelve**; `sized` is the Chapter 2
  thesis route, `features` adds feature constraints and `pipeline` runs
  GODF → QGP → channel → smoothing. The `README`, crate-level docs and
  `REFERENCES.md` now reflect the twelve routes and map every `simons_*` module
  to its thesis section and reference.
- The `simons_regions` documentation now records that the nautilus tile is
  implemented (singular-free annulus) while the *singular* annulus still needs
  merging; stale "not implemented" notes were removed.

### Performance (thesis/t17)

Measured with `cargo run --release --offline --example generate -- <route>
plate-hole-0.6mm`, isolated target directory, same machine, in-program seconds:

| Case | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| `crossfield` on `plate-hole-0.6mm` | 142.8 s | 93.6 s | **1.53×** |
| `sized` on `plate-hole-0.6mm` | 21.6 s | 16.5 s | **1.30×** |

The hotspots and fixes:

- **Cavity closure (`sun::close_by_cavity`)**: 54 s → 11 s across the four
  crossfield paving depths. Two changes, both output-preserving:
  `compact_region` grows one BFS region incrementally across the twelve cavity
  radii instead of rebuilding it from scratch each time, and the polygon
  quadrangulation DP uses a dense `Vec` memo instead of a `HashMap` in its
  O(n⁴) inner loop.
- **Thesis shape pass (`simons_smooth`)**: 27.8 s → 23.0 s (crossfield) and
  19.3 s → 14.6 s (sized). The per-star quantities that do not depend on the
  candidate node position (the opposite corner's angle, two edge lengths and one
  diagonal) are hoisted out of the Nelder–Mead search, and nodes already at an
  objective value within `ACCEPT_TOLERANCE` of zero are skipped exactly (the
  objective is a sum of squares, so no accepted move can exist there).
- **Smoother relaxation (`smooth::relax_interior`)**: the per-node `Vec` ring and
  candidate buffers are reused across nodes and sweeps.
- **QMorph parity guard (`qmorph`)**: the even-component guard walks an
  append-only live-triangle list instead of scanning every cell id (including
  removed cells) on every candidate placement. The QMorph route on the 0.6 mm
  plate also benefits from the shape-pass work (69.6 s → 53.6 s).
- **Ruled out**: the repeated `MeshStatistics::of` full-mesh scans were
  measured at ~20 ms per call on the 20 k-quad meshes and are not a bottleneck;
  the Blossom matching (~15 s over the three residual components) is dominated
  by the external `mwmatching` solver and was left unchanged.

All changed paths were checked against the pre-optimisation binary with
`--hash` on the seven parts; the meshes are bit-identical (see the verification
section below).

### Verified

- `cargo test --offline`, `cargo fmt --check`, `cargo clippy --offline
  --all-targets -- -D warnings` and `cargo build --target
  wasm32-unknown-unknown --offline` are green.
- Every route keeps its previous seven-part result: the routes that were 7/7
  (`paving`, `bipartite`, `quadtree`, `crossfield`, `l1`, `sized`) are still
  7/7, and the routes that carried failures (`qmorph`, `decompose` after the
  post-pass, `hybrid`, `untangle`) report the same failures. `features` needs a
  feature polyline and `pipeline` is the integrated demo, as documented.
