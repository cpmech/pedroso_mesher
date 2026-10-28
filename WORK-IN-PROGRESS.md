# Work in progress

## Current task

Add **one regression test** for the known dense-hole O-grid/post-smoothing
failure. This is a test-only change: do not change the meshing or smoothing
implementation yet.

## Intended checks

Exercise the real generation path for the smallest existing dense-hole case and
check that:

- mesh topology remains valid;
- scaled Jacobians are positive within the project's established tolerance; and
- the domain boundary is preserved.

First locate the existing dense-hole fixtures, tests and recorded failure to
reuse the exact case and test conventions. The expected initial result is a
focused test that reproduces the failure; a green test would mean the case does
not reproduce the reported regression and must be recorded rather than treated
as a fix.

## Constraints

- Pure Rust, no new dependencies, no `unsafe`, and no panics outside tests.
- Keep the change limited to one regression test and this handover file.
- Record the test command and its actual result below before stopping.

## Result

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
