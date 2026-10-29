# Work in progress

## Current task

Prototype the smallest whole-pass safety guard in `two_d::smooth_mesh`: preserve
a passing input mesh when the existing smoother returns a failing mesh.

## Implementation and validation plan

1. Measure the input gate and save its node coordinates when it passes.
2. Run the existing smoothing algorithm unchanged.
3. Restore the saved coordinates if the input passed but the result fails.
4. Run `dense_hole_o_grid_survives_common_post_smoothing` without weakening it,
   then run the existing smoothing tests and formatting/diff checks.
5. Record the actual gate metrics and any regressions below. Do not refine this
   into a per-sweep or per-node guard until the whole-pass behavior is measured.

## Constraints

- Pure Rust, no new dependencies, no `unsafe`, and no panics outside tests.
- Do not change the local smoothing objective in this experiment.
- Preserve existing behavior for inputs that already fail the gate.

## Guard result

Implemented the whole-pass guard in `two_d::smooth_mesh`. A passing input has
its coordinates saved and restored if the completed smoothing pass fails the
gate. An input that already fails still runs through the smoother unchanged,
so it retains the chance to be repaired. The local objective was not changed.

The permanent regression now passes without modification:

```text
cargo test dense_hole_o_grid_survives_common_post_smoothing -- --nocapture
```

The full test suite passes: **413 unit tests, 5 CLI tests and 11 documentation
tests**. `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`,
`cargo check --target wasm32-unknown-unknown` and `git diff --check` also pass.
The benchmark command was:

```text
cargo run --example alt_decompose
```

All seven decomposition cases pass raw and post-processed. The guard restores
the raw coordinates on the 2 mm, 1 mm and 0.6 mm hole cases. In particular,
the 1 mm case retains raw SJ `0.405` and corner angle `30.4°` instead of the
former post-pass SJ `0.170` and angle `9.8°`. The 0.6 mm case retains SJ
`0.375` and angle `30.1°`. Boundary offsets remain at floating-point noise and
all rows have zero inverted cells, overlaps and cracks.

Next: keep this minimal guard unless measured quality goals justify a
per-sweep or per-node guard that can retain more safe improvements.

## Smoothing-move result

Temporary instrumentation ran `two_d::smooth_mesh` one sweep at a time and then
stopped at the first node-level gate failure:

```text
cargo test dense_hole_sweep_diagnostic -- --nocapture
```

The **first full sweep fails**: minimum scaled Jacobian falls from
`0.405131722017` to `0.197644640076`, and minimum corner angle falls from
`30.425995029317°` to `11.399257682143°`. The worst quad is cell 2464, corner
3/node 2865, with nodes `[2864, 2883, 2884, 2865]`. All four nodes move during
that sweep, and all are interior nodes.

A second run stopped immediately when an accepted move first lost the gate.
Interior node 458 moved from `[27.41268632381083, 5.506004378814177]` to
`[27.624818358166795, 5.718136413170141]`: the `0.3 * size` candidate at 45
degrees. This move leaves minimum scaled Jacobian at `0.405131722017` but drops
minimum corner angle to `29.924307964860°` in cell 1331, corner 2/node 458.
Both the first gate crossing and the worst full-sweep quad therefore involve
**interior moves, not boundary slides**. Later interior moves degrade quality
further.

The smoother accepts a sum of incident corner objectives, not a global gate
improvement, which explains why a locally improving move can cross the gate.
This diagnostic motivated the whole-pass guard documented above.

## Post-pass stage result

Temporary instrumentation ran each post-pass stage separately with:

```text
cargo test dense_hole_stage_diagnostic -- --nocapture
```

| Stage | Minimum scaled Jacobian | Minimum corner angle | Gate |
| --- | ---: | ---: | --- |
| Raw O-grid | 0.405131722017 | 30.425995029317° | PASS |
| 80 smoothing sweeps | 0.169896689045 | 9.781812395935° | FAIL |
| 12 angle-swap rounds | 0.169896689045 | 9.781812395935° | FAIL |
| 200 perturbation rounds | 0.169896689045 | 9.781812395935° | FAIL |

Every stage reported zero inverted cells, zero overlapping pairs and zero
cracks. Boundary offset remained `7.105e-15`. The **80 smoothing sweeps cause
the complete measured regression**; angle swaps and perturbation make no
further change.

## Completed baseline

Added `dense_hole_o_grid_survives_common_post_smoothing` in
`src/alt_decompose.rs`. Before the guard, the raw mesh passed but the complete
post-pass failed with `min scaled Jacobian 0.170 < 0.2`. The unchanged test now
passes with the whole-pass guard. Its command is:

```text
cargo test dense_hole_o_grid_survives_common_post_smoothing -- --nocapture
```
