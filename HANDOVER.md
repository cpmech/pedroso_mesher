# Phase D handover — clean-room Delaunay migration

Date: 2026-10-28
Repository: `pedroso_mesher/`
Branch: `main`
State: **work in progress, uncommitted, two tests still failing**

## Governing request and constraints

The current task is Phase D: replace `spade` with the clean-room exact-predicate
Delaunay kernel throughout production. Do not start Phase E. Do not commit or
push without a new explicit user request.

Follow `AGENTS.md`: original clean-room Rust, no copied or ported code, no
GPL/LGPL/AGPL code, no commercial tools, no new dependencies, no `unsafe`, no
panic/unwrap/expect outside tests, errors through `Result`, and preserve
`wasm32-unknown-unknown`. Read each file immediately before editing it. Never
overwrite frozen files under `data/baselines/`.

## Migration implemented

- Added `src/delaunay/adapter.rs` with `CdtAdapter` and `CdtLocation`.
- The adapter supports bulk and incremental insertion, disconnected regions,
  point location, stable point indices, and triangle extraction.
- Extended `src/delaunay/triangulation.rs` with constrained-domain-plus-points
  construction and transactional local insertion.
- Migrated Crossfield, Paving, bipartite, `two_d`, Hybrid indirectly, Q-Morph
  indirectly, Pipeline indirectly, and L-infinity to `CdtAdapter`.
- Removed `spade` from `Cargo.toml` and regenerated `Cargo.lock` offline with
  `cargo metadata --offline --format-version 1 > /dev/null`.
- Updated `REFERENCES.md` for the clean-room triangulator and exact predicates.
- A repository-wide search found no remaining `spade`, `Point2`, or
  `inner_faces` references.
- `cargo test --offline alt_l1` passed all 11 L-infinity tests.

## Baseline results obtained before the latest Q-Morph fixes

Artifacts are under `phase-d/` in temporary storage, including
`crossfield-*.csv`, `paving.csv`, `bipartite.csv`, `qmorph.csv`, `hybrid.csv`,
`l1.csv`, `l1-adversarial.csv`, and `pipeline.csv`.

- Paving and bipartite retained all baseline acceptance classifications.
- Hybrid retained all gate/conformity/geometry classifications. Counts and raw
  quality changed modestly, with several post-pass improvements.
- L-infinity retained every baseline acceptance classification. The 0.6 mm
  case changed from 15,468 to 15,551 quads; post minimum scaled Jacobian
  improved from 0.511961 to 0.538535. Other differences were small.
- L-infinity adversarial cases now explicitly reject self-intersections and
  outside holes rather than returning invalid meshes. Valid cases retained
  their classifications.
- Pipeline ring improved from a cross-field error to a valid 48-quad mesh;
  gate and conformity pass, but strict geometry still fails with 12 outside
  cells. Other Pipeline rows were unchanged.
- Known quality regressions still to document/review:
  - Crossfield ring: 354 to 240 quads, post minimum SJ 0.61575 to 0.28327,
    minimum angle 45.36 to 29.67 degrees, gate pass to fail.
  - Q-Morph ring: 221 to 37 quads, minimum SJ about 0.0654 to 4.52e-15. It was
    already failing gate and geometry.

These affected baselines must be rerun after the new Q-Morph refinement
fallback; the numbers above predate that fallback.

## Q-Morph diagnosis and fixes

The first complete test run after dependency removal had 406 passed and 4
failed:

- `alt_untangle::tests::a_simple_domain_is_all_quad_and_passes_the_gate`
- `quad::tests::a_square_is_supported_and_comes_back_all_quad`
- `simons_features::tests::a_through_and_a_terminating_feature_are_mixed`
- `simons_features::tests::demo_numbers_are_gate_passing`

### Residual refinement fallback

Added a final Q-Morph residual-closure fallback in `src/quad.rs`, after Sun and
cavity closure and before quality passes. It calls the existing
`two_d::close_by_refinement` and accepts the candidate only when topology has
zero cracks, inconsistencies, and non-manifold edges. This fixed the Untangle
square test.

The Q-Morph square initially remained at 60 quads plus two adjacent triangles:

```text
[29, 30, 48]
[29, 30, 31]
```

Both `Route::Front` and `Route::TwoStage` failed, so route fallback was not a
solution. Individual refinement cascades succeeded, but their combined result
was rejected. Duplicate midpoint nodes were being created independently for
the shared residual-triangle edge.

Modified `two_d::close_one` to build an exact `(x_bits, y_bits) -> node_id` map;
`midpoint` now reuses an existing node at identical coordinates as well as
memoising by edge. The combined result then became 85 quads and zero triangles,
but still had six orientation inconsistencies and was correctly rejected.

### Raw-conversion transactional retry

A temporary probe showed the failing post-cleanup square already had three
inconsistent edges before refinement. Its two residual triangles were both
counter-clockwise but lay on the same side of their shared edge, so simply
reorienting or accepting them would be wrong. The raw Q-Morph conversion had
zero cracks, non-manifold edges, and inconsistencies; the cleanup introduced
the defect.

`src/quad.rs` now preserves `converted = mesh.clone()` before cleanup. The
refinement fallback first tries the cleaned mesh and, if no watertight candidate
is produced, retries from the raw conversion. This fixed
`quad::tests::a_square_is_supported_and_comes_back_all_quad` without weakening
the topology check.

The full suite then reached **408 passed, 2 failed** out of 410 library tests.
Only the two mixed-feature tests remained red.

## Mixed through-plus-terminating feature investigation

Failing tests:

- `simons_features::tests::a_through_and_a_terminating_feature_are_mixed`
- `simons_features::tests::demo_numbers_are_gate_passing` (same underlying case)

Case: 10x10 square, through feature `(5,0)-(5,10)`, terminating feature
`(2,0)-(2,5)`, size 1. Historically the sampled-seam bipartite route produced
112 quads. Do not edit the tests to accept failure.

The route list is `sized`, `quadtree`, `bipartite`, `untangle`, `decompose`.
Errors before further experiments:

- sized: channel closure failure
- quadtree: merged mesh has 2 cracks
- bipartite: merged mesh has 2 inverted elements
- untangle: merged mesh has 1 crack
- decompose: merged mesh has 3 inverted elements

### Per-piece evidence

Temporary instrumentation (subsequently removed) showed:

- Quadtree: each piece was valid after smoothing, but the seam chains differed
  drastically (11 nodes on one side, 37 on the other), causing two cracks.
- Bipartite: both pieces had the exact same canonical 11-node sampled seam.
  The left piece already had two inverted quads before welding; the weld was
  not creating them. Raw gate: 59 quads, 2 inverted, minimum SJ about
  -0.195090322, topology otherwise consistent.
- Untangle: the left piece omitted seam nodes at `y=2` and `y=4` while the
  right piece retained all 11 canonical samples, causing the crack.
- Decompose: the left piece already had three inverted quads.

### Experiments that did not fix the mixed case

1. **Incremental bipartite insertion.** Changed the field-aligned triangulation
   from bulk insertion to `from_loops` followed by sequential `insert_point`,
   preserving the old route's insertion order. Results were identical. Reverted
   to `from_loops_and_points` bulk insertion.
2. **Crossfield as an extra feature route.** It dropped seam nodes and the
   merged mesh still had one crack. Removed from `ROUTES`.
3. **L-infinity as an extra feature route.** It preserved the exact 11-node
   seam on both pieces, but returned triangles and `weld` rejected the non-quad
   cell. Removed from `ROUTES`.
4. **L-infinity with `Sizing::all_quad = true`.** Still left triangles. Removed.
5. **Generic `two_d::close_by_refinement` after L-infinity.** Also failed to
   remove every residual triangle, so the weld still rejected the mesh.
   Removed the wrapper and extra route.
6. **`sun::repair_low_quads` in `finish_side`.** Did not remove the bipartite
   inversions. Replaced by the next experiment.
7. **`sun::repair_degenerate_quads` in `finish_side`.** Also did not remove the
   inversions. Replaced by the next experiment.
8. **Log-barrier untangling in `finish_side`.** Made
   `alt_untangle::untangle` `pub(crate)` and called it for raw pieces with
   inversions, accepting only a gate-score improvement. The targeted mixed
   feature test still fails with the same two bipartite inversions. This is the
   current experimental code still present in `src/simons_features.rs` and
   `src/alt_untangle.rs`; decide whether to retain or revert it after further
   diagnosis.

All temporary `eprintln!` seam/piece instrumentation in `simons_features.rs`
was removed. The original five-route `ROUTES` list is restored.

## Current important code state

- `src/two_d.rs`: `close_by_refinement` is temporarily **`pub fn`** for the
  external probe. Restore it to `pub(crate)` before final validation.
- `src/alt_untangle.rs`: `untangle` is currently `pub(crate)` for the
  mixed-feature experiment.
- `src/simons_features.rs`: `finish_side` currently tries that untangler on
  inverted raw pieces, then the existing boundary-preserving smoother.
- `src/quad.rs`: raw-conversion retry is present and fixes the square test.
- `src/alt_bipartite.rs`: bulk `CdtAdapter::from_loops_and_points` is restored.
- No extra Crossfield/L-infinity feature routes remain.
- No debug prints remain in `simons_features.rs`.
- No commit or push has been made for this work.

Current modified/untracked paths from `git status --short`:

```text
 M Cargo.toml
 M REFERENCES.md
 M examples/phase_a.rs
 M src/alt_bipartite.rs
 M src/alt_crossfield.rs
 M src/alt_l1.rs
 M src/alt_paving.rs
 M src/alt_untangle.rs
 M src/config.rs
 M src/delaunay/geometry.rs
 M src/delaunay/mod.rs
 M src/delaunay/triangulation.rs
 M src/quad.rs
 M src/simons_features.rs
 M src/two_d.rs
?? src/delaunay/adapter.rs
?? HANDOVER.md
```

## External probe

`phase-d/probe.rs` in temporary storage extracts the failing Q-Morph square,
prints fan quality, invokes `close_by_refinement`, and reports topology/gate.
It was useful for identifying the raw-versus-cleaned orientation issue. It
requires the temporary public visibility of `close_by_refinement` and is not
part of the repository.

## Recommended next steps

1. Diagnose the two inverted quads at the bipartite conversion stage. The route
   already has `assemble`, bounded single/pair label-flip retries, unfolding,
   repositioning, and Sun repairs, but returns the invalid all-quad topology
   when those fail. Inspect the offending quad coordinates and the pairing that
   created them rather than weakening the gate or feature tests.
2. Decide whether the unsuccessful `finish_side` untangling experiment should
   be reverted. If retained, it must provide a measured benefit and pass all
   quality checks.
3. Restore `two_d::close_by_refinement` to `pub(crate)` after external probing
   is finished.
4. Rerun the two feature tests, then the complete suite.
5. Update affected module comments, especially `quad.rs` stage 4 and stale
   measured claims. Update feature measurements only after the mixed case is
   actually fixed.
6. Rerun Q-Morph and all other affected raw/post baselines because the new
   refinement fallback changes downstream output. Run the complete adversarial
   corpus and review every comparison.
7. Run final validation:

   ```bash
   cargo fmt && cargo test --offline && \
   cargo clippy --offline --all-targets -- -D warnings && \
   cargo build --offline --target wasm32-unknown-unknown --lib && \
   git diff --check
   ```

8. Update `ObsidianVault/projects/Pedroso-Mesher.md`
   with frontmatter `updated: 2026-10-28`, exact migration results, remaining
   quality regressions, and final test counts.
9. Do not declare Phase D complete while tests are red, comparisons are
   unreviewed, or any `spade` dependency remains. Do not begin Phase E.
