# Phase D handover — clean-room Delaunay migration

Date: 2026-10-29
Repository: `pedroso_mesher/`
Branch: `main`
State: **Phase D review complete; final smoothing rescue uncommitted**

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
   feature test still failed with the same two bipartite inversions. The
   experiment was subsequently reverted and `untangle` is private again.

All temporary `eprintln!` seam/piece instrumentation in `simons_features.rs`
was removed. The original five-route `ROUTES` list is restored.

## Mixed-feature resolution after the WIP commit

Temporary coordinate diagnostics found the two bipartite defects in the left
piece:

- A folded quad at the slot tip with corners `(1.5,4.1)`, `(1.5,5)`,
  `(1.538060...,5.191342...)`, `(0.8,5)` and minimum SJ about -0.19509.
- A flat quad at the slot mouth with corners `(2,0)`, `(2.5,0.5)`, `(2,0.5)`,
  `(1.5,0.5)`. The last three nodes are collinear on the slot cap.

`fold_neighbourhood` originally expanded only from free vertices already in a
bad quad. The flat quad had only boundary vertices, so its neighbourhood was
empty. The search now also includes free vertices of every edge-adjacent quad,
then applies the existing one-ring expansion. The bounded single/pair label-flip
retry then removed every inversion. The remaining failure changed to a quality
failure: minimum corner angle 26.565 degrees.

The ordinary smoother could not improve those corners because they occur at
fixed geometric vertices. Added the existing `sun::perturb_low_quads` pass at
the end of bipartite assembly. It slides non-corner boundary samples only along
their original domain edges and leaves original domain vertices fixed. The
mixed feature test and `demo_numbers_are_gate_passing` then passed.

The full suite passed **410 library tests, 5 CLI tests, and 11 doc tests**. The
mixed feature demo now produces 113 quads with minimum SJ 0.2660, minimum angle
32.0 degrees, zero cracks, and a passing gate. This is a quality decrease from
the historical 112-quad SJ 0.6073 / 38.8-degree result and is recorded as such.

## Crossfield ring resolution

The first complete migrated matrix had only one new post-pass gate regression:
Crossfield ring fell from a pass to 240 quads, minimum SJ 0.283266 and minimum
angle 29.671 degrees. Its raw mesh was already failing at 8.730 degrees.

A configuration probe showed that the existing `sun::perturb_low_quads` pass
lifts the corner above 30 degrees without changing connectivity. The shared
smoother now enables at most 64 perturbation rounds by default, but only when
the current quality margin is below the gate. Passing meshes are unchanged;
original domain vertices remain fixed, other boundary nodes can only slide on
the polyline, and the existing `better` acceptance check remains the final veto.
The 30-degree threshold was not changed.

All seven Crossfield corpus parts pass the quality gate after this change. The
ring remains at 240 quads, with minimum SJ 0.283 and minimum angle 30.2 degrees.
The mixed feature demo is unchanged at 113 quads, SJ 0.2660 and 32.0 degrees.
The complete suite, strict Clippy and the Wasm library build pass.

## Current important code state

- `src/two_d.rs`: `close_by_refinement` is restored to `pub(crate)`.
- `src/alt_untangle.rs`: `untangle` is private; the unsuccessful feature-side
  untangling experiment is removed.
- `src/simons_features.rs`: `finish_side` uses only the existing
  boundary-preserving smoother; all diagnostics are removed. The measured mixed
  feature table is updated.
- `src/quad.rs`: raw-conversion retry remains, and stage-4 comments describe the
  watertight residual-refinement fallback.
- `src/alt_bipartite.rs`: bulk `CdtAdapter::from_loops_and_points` remains;
  edge-adjacent retry expansion and final boundary-aware perturbation are the
  new fixes.
- No extra Crossfield/L-infinity feature routes remain.
- `cargo fmt`, strict Clippy, and the Wasm library build pass after the fixes.

## External probe

`phase-d/probe.rs` in temporary storage extracts the failing Q-Morph square,
prints fan quality, invokes `close_by_refinement`, and reports topology/gate.
It was useful for identifying the raw-versus-cleaned orientation issue. It
required temporary public visibility of `close_by_refinement`, which has since
been restored to crate-private; the probe is not part of the repository.

## Floating-point-tolerant outside-cell decision

The 34 Crossfield ring outside-cell reports were investigated with highlighted
figures and independent area measurements. The actual outside area was about
6.4e-14, at floating-point round-off scale; whole-cell highlighting exaggerated
the discrepancy. Exact-zero proper segment intersection tests were classifying
nearly coincident cell/domain edges as crossings.

`cell_extends_outside` now uses signed point-to-line distances with an absolute
1e-13 tolerance for proper segment intersections. This change is local to
outside-cell detection; exact geometric overlap detection is unchanged. A
focused regression proves that a near-coincident crossing is accepted while an
ordinary crossing is still rejected. An end-to-end regression generates the
Crossfield ring and requires zero outside cells and a passing full geometry
report. Both raw and post ring rows now pass geometry with unchanged mesh
hashes. The post ring therefore passes quality, conformity and full geometry.

The adversarial rerun changes only Quadtree acute-corner raw/post: their 22
round-off outside-cell reports become geometry passes, while quality still fails
at minimum SJ 0.091. No mesh hashes change. The full suite now has 412 library
tests, 5 CLI tests and 11 doc tests, all passing with strict Clippy, Wasm and
`git diff --check`.

## Final baseline review and validation

The final 168-row matrix has 154 meshes and 14 explicit errors, 116 quality-gate
passes, 130 full-geometry passes, and 97 rows passing both contracts. The
smoothing rescue changed exactly one row relative to the preceding migrated run:
Crossfield ring post moved from quality-gate fail to pass, with angle 29.671 to
30.191 degrees, unchanged SJ 0.283266, and aspect maximum 29.295 to 24.310. The
subsequent 1e-13 outside-cell tolerance reclassified 64 rows from strict-geometry
fail to pass, including Crossfield and Pipeline ring raw/post, without changing
any mesh hash or quality field. Crossfield ring post now passes quality,
conformity and full geometry.

Relative to the frozen baseline, the only remaining acceptance-classification
changes are Pipeline ring raw/post: explicit errors became valid 48-quad meshes
passing quality, conformity and full geometry. The frozen files under
`data/baselines/` were not modified.

The 42-row adversarial rerun changes only Quadtree acute-corner raw/post from
the preceding migrated run: 22 round-off outside-cell reports become geometry
passes, while quality still fails at minimum SJ 0.091. Relative to the frozen
baseline, its only other changes remain the four intentional L-infinity errors
for invalid self-intersecting and outside-hole inputs.

Final validation passed: **412 library tests, 5 CLI tests and 11 doc tests**,
`cargo fmt`, strict Clippy, `wasm32-unknown-unknown` library build and
`git diff --check`. `ObsidianVault/projects/Pedroso-Mesher.md` contains the final
Phase D record. The smoothing, geometry-tolerance and documentation changes are
uncommitted; Phase E has not started.
