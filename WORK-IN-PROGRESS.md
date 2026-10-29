# Work in progress

## Current task

Isolate which stage of the common post-pass causes the dense-hole O-grid
quality regression. This is diagnostic work: do not change the meshing or
smoothing implementation.

## Diagnostic plan

Reuse the recorded failing case in
`dense_hole_o_grid_survives_common_post_smoothing`: a 100 x 60 plate,
radius-10 hole with 48 segments, and element size 1. Measure the mesh:

1. after raw O-grid generation;
2. after 80 smoothing sweeps;
3. after the 12 angle-swap rounds; and
4. after up to 200 perturbation rounds.

At every stage record minimum scaled Jacobian, minimum corner angle, inverted
and overlapping counts, cracks, boundary offset and gate verdict. Identify the
first stage that loses the raw pass. Prefer temporary diagnostic instrumentation
over production changes, and record the exact command and results below.

## Constraints

- Pure Rust, no new dependencies, no `unsafe`, and no panics outside tests.
- Do not change the production implementation or weaken the regression test.
- Record the diagnostic command and its actual result below before stopping.

## Completed baseline

Added `dense_hole_o_grid_survives_common_post_smoothing` in
`src/alt_decompose.rs`. It uses the recorded failing case: a 100 x 60 plate,
radius-10 hole with 48 segments, and element size 1. The test runs the real
O-grid route followed by the benchmark's common post-pass (80 smoothing sweeps,
12 angle-swap rounds and up to 200 perturbation rounds).

Command:

```text
cargo test dense_hole_o_grid_survives_common_post_smoothing -- --nocapture
```

Actual result: **the regression is reproduced**. The raw mesh passes the gate.
After smoothing, the boundary-offset, zero-crack and positive-Jacobian checks
pass, but the final gate fails with `min scaled Jacobian 0.170 < 0.2`. This is
an intentionally failing regression test, not a fix.

`cargo fmt --check` and `git diff --check` pass. No implementation code was
changed. Next: diagnose why the post-pass degrades the first hole ring, make
the smallest implementation fix, and rerun this test without weakening it.
