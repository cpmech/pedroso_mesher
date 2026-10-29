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
Temporary instrumentation and the early return were removed; production code
and the permanent regression test remain unchanged.

Next: evaluate the smallest acceptance guard that prevents smoothing from
losing a passing mesh, then test it against the permanent regression and the
existing smoothing tests.

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
`src/alt_decompose.rs`. It uses the same case and complete common post-pass.
The raw mesh passes; after post-processing the boundary-offset, zero-crack and
positive-Jacobian checks pass, but the final gate fails with
`min scaled Jacobian 0.170 < 0.2`. This is an intentionally failing regression
test, not a fix. Its command is:

```text
cargo test dense_hole_o_grid_survives_common_post_smoothing -- --nocapture
```
