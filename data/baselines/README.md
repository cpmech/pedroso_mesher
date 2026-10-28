# Phase A baselines

These CSV files freeze the raw and post-smoothing behavior measured on
2026-10-28. Regenerate them from the repository root:

```bash
cargo run --release --offline --example phase_a -- --output data/baselines/phase-a.csv
cargo run --release --offline --example phase_a -- --adversarial --output data/baselines/phase-a-adversarial.csv
```

`phase-a.csv` runs all twelve routes over the seven existing route-comparison
parts (168 records). `phase-a-adversarial.csv` runs Quadtree, L-infinity and
Sized over seven additional cases (42 records): multiple holes, an acute
corner, a small hole, translated/scaled coordinates, a repeated boundary node,
a self-intersecting outer loop, and a hole outside the outer loop.

Each case runs independently with `MeshConfig::new(route)`, once with smoothing
disabled (`raw`) and once with the default smoother (`post`). The runner uses
`generate`, not the strict adapter, so a mesh below the quality gate remains
measurable. Route errors are recorded as rows with `status=error` and do not
abort the matrix. Wall-clock time is excluded.

The mesh hash is FNV-1a over the exact node-coordinate and cell bit patterns.
The remaining columns record quality minima and aspect tails, valence-4 share,
boundary/corner offsets, the existing gate, geometric conformity, and the full
`verify::geometry` result. Floating-point values use 17 significant digits.
These are measurements, not claims that every row is shippable: inspect
`status`, `gate`, `conforming`, `geometry`, and `detail` together.

## Measurements after containment fixes

In the full baseline, 152 of 168 runs return meshes and 16 are explicit route
errors (the Features route has no feature polylines in this generic matrix, and
Pipeline has one error in each stage). Of the 152 meshes, 114 pass the legacy
gate, 66 pass full geometry, and 57 pass both. **57 rows pass the legacy gate
and fail full geometry**, concentrated on the four plate-with-hole resolutions
and the ring. The containment fixes raised full-geometry passes from 44 to 66
and passes of both contracts from 39 to 57, with no legacy-gate regressions.
Twenty-four mesh hashes changed intentionally when boundary smoothing began
preserving every original domain vertex. The remaining failures are evidence
for further investigation, not new quality claims.

In the adversarial matrix, 34 of 42 runs return meshes and eight are explicit
errors; 26 pass both gates. Four outside-hole rows pass the legacy gate but fail
full geometry. Quadtree places cells outside the acute-corner domain, L-infinity
leaves four triangles there, and Sized explicitly rejects the multiple-hole
case. A repeated boundary node is currently accepted by all three routes; this
is recorded behavior, not final input-validation policy. This CSV was unchanged
by the containment fixes.

Reproducibility is checked by generating a second CSV and comparing the files
byte-for-byte. Different legal meshes after an intentional algorithm change
require review of acceptance and quality differences, not artificial hash
identity.

Disconnected regions and internal material interfaces are not represented by
the current single-outer-loop `Domain`; their corpus and baseline coverage
require the unified domain model planned for Phase B. Near-degenerate input
validation is likewise recorded here as current behavior, not declared final.
