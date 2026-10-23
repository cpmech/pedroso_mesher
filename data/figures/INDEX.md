# Mesh gallery

Every studied mesh, drawn by `examples/draw_gallery.rs`. Cells are
filled by the gate margin `min(angle/30, sj/0.2)` (green clears the
gate, yellow/orange/red fails, magenta is inverted, triangles are dark
red); the title carries the measured numbers.

`raw` disables the shared post-smoother and, for `pipeline`, its
internal Section 2.2.10 pass. `smooth` is the shipping default
(smoothing on). A rejected smoothing candidate is labelled in the
title and the raw mesh is drawn.

## Layout

- `corpus/<route>/<part>-<raw|smooth>.png` - 11 routes x 7 parts
- `features/<case>-<raw|smooth>.png` - through, stub slot, cross, through+terminating, stub channel
- `pipeline/graded-10x10-<raw|smooth>.png` - the graded QGP demo
- `manifest.csv` - one row per case with the statistics and gate verdict

## Corpus routes

| Route | Figures | PASS (smoothed) |
|---|---:|---:|
| qmorph | 14 | 4/7 |
| paving | 14 | 7/7 |
| bipartite | 14 | 7/7 |
| quadtree | 14 | 7/7 |
| decompose | 14 | 7/7 |
| hybrid | 14 | 0/7 |
| crossfield | 14 | 7/7 |
| untangle | 14 | 4/7 |
| l1 | 14 | 7/7 |
| sized | 14 | 7/7 |
| pipeline | 14 | 6/7 |

The `features` route errors on the corpus parts (it needs feature
polylines); its five demo cases are under `features/`. The `pipeline`
route is also drawn on the corpus parts for completeness (it fails on
`ring`, the known cross-field degenerate triangle, so that row has no figure);
its intended case is `pipeline/graded-10x10`.

164 figures, 166 manifest rows (2 `pipeline` x `ring` errors).
