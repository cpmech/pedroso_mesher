# References

Numbered list of the references used by this crate. Corpus ids are the file
stems under `~/03-Writings/articles-and-books/mesh-generation/quad-hex-mesh/`
(`INDEX.md`, 53 entries): `blacker-1991` is `pdf/blacker-1991.pdf` and
`md/blacker-1991.md`. Module paths say where the paper is implemented or
cited. The second block lists the references inside the Simons thesis that
the `simons_*` modules implement.


## Corpus papers used

1. **owen-1999** — Owen S.J., Staten M.L., Canann S.A., Saigal S., 1999. "Q-Morph: an indirect approach to advancing front quad meshing." International Journal for Numerical Methods in Engineering, 44(9): 1317-1340.
   Used in: Q-Morph: the advancing-front indirect conversion (`src/qmorph.rs`).

2. **sun-2015** — Automatic quad meshing and quality improvement by improved combination. 2015. (Full citation in `md/sun-2015.md` and `pdf/sun-2015.pdf`.)
   Used in: The Sun residual walk and templates (`src/sun.rs`).

3. **sun-2016** — Complete triangle-to-quad conversion for multiple-domain geometry. 2016. (Full citation in `md/sun-2016.md` and `pdf/sun-2016.pdf`.)
   Used in: Triangle-to-quad conversion: cavity closure and the Sun templates (`src/sun.rs`).

4. **remacle-2012** — Remacle J.-F., Lambrechts J., Seny B., Marchandise E., Johnen A., Geuzaine C., 2012. "Blossom-Quad: A non-uniform quadrilateral mesh generator using a minimum-cost perfect-matching algorithm." International Journal for Numerical Methods in Engineering, 89(9): 1102-1119.
   Used in: Blossom-Quad minimum-cost perfect matching: the default `two_d::mesh_2d` pairing (`src/two_d.rs`).

5. **shewchuk-2002-delaunay-refinement** — Shewchuk J.R., 2002. "Delaunay refinement algorithms for triangular mesh generation." Computational Geometry, 22(1-3): 21-74.
   Used in: Delaunay refinement used by the clean-room triangulator (`two_d::triangulate`, `src/delaunay`).

6. **shewchuk-1997** — Shewchuk J.R., 1997. "Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates." Discrete & Computational Geometry, 18(3): 305-363.
   Used in: Clean-room adaptive-precision geometric predicates (`src/delaunay/predicates.rs`).

7. **knupp-2000-part1** — Knupp P.M., 2000. "Achieving finite element mesh quality via optimization of the Jacobian matrix norm and associated quantities. Part I: A framework for surface mesh optimization." International Journal for Numerical Methods in Engineering, 48(3): 401-420.
   Used in: Scaled Jacobian / quality metric definitions (`src/quality.rs`).

8. **knupp-2000-part2** — Knupp P.M., 2000. "Achieving finite element mesh quality via optimization of the Jacobian matrix norm and associated quantities. Part II: A framework for volume mesh optimization and the condition number of the Jacobian matrix." International Journal for Numerical Methods in Engineering, 48(8): 1165-1185.
   Used in: 3D Jacobian quality (the hex side of `src/quality.rs`).

9. **blacker-1991** — Blacker T.D., Stephenson M.B., 1991. "Paving: A new approach to automated quadrilateral mesh generation." International Journal for Numerical Methods in Engineering, 32(4): 811-847.
   Used in: Paving: rows, seam (section 2.8), closure (`src/alt_paving.rs`).

10. **georgiadis-2022** — Indirect all-quadrilateral meshing based on bipartite topological labeling. 2022. (Full citation in `md/georgiadis-2022.md` and `pdf/georgiadis-2022.pdf`.)
   Used in: Right-angled triangles + bipartite labelling (`src/alt_bipartite.rs`).

11. **rushdi-2017** — All-quad meshing without cleanup. 2017. (Full citation in `md/rushdi-2017.md` and `pdf/rushdi-2017.pdf`.)
   Used in: Cleanup-free quadtree with 2-ref templates (`src/alt_quadtree.rs`).

12. **rushdi-2015** — Robust all-quad meshing of domains with connected regions. 2015. (Full citation in `md/rushdi-2015.md` and `pdf/rushdi-2015.pdf`.)
   Used in: All-quad meshing of connected regions with the quadtree (`src/alt_quadtree.rs`).

13. **yang-2024** — Boundary-constrained quad meshing by domain decomposition and templates. 2024. (Full citation in `md/yang-2024.md` and `pdf/yang-2024.pdf`.)
   Used in: Boundary-constrained quad meshing by domain decomposition and templates (`src/alt_decompose.rs`).

14. **miranda-2017** — Hierarchical template-based quadrilateral mesh generation. 2017. (Full citation in `md/miranda-2017.md` and `pdf/miranda-2017.pdf`.)
   Used in: Hierarchical template-based quads, the T0/T1/T2 transfinite templates (`src/alt_decompose.rs`).

15. **ma-2017** — Sub-domain decomposition and quadrilateral meshing with line constraints. 2017. (Full citation in `md/ma-2017.md` and `pdf/ma-2017.pdf`.)
   Used in: Sub-domain decomposition with line constraints (`src/alt_decompose.rs`).

16. **zint-2021** — A hybrid approach to fast indirect quadrilateral mesh generation. 2021. (Full citation in `md/zint-2021.md` and `pdf/zint-2021.pdf`.)
   Used in: Hybrid Blossom-band + greedy interior matching (`src/alt_hybrid.rs`).

17. **pellenard-2014** — QMCF: QMorph cross-field-driven quad-dominant meshing. 2014. (Full citation in `md/pellenard-2014.md` and `pdf/pellenard-2014.pdf`.)
   Used in: QMCF: cross-field-driven Q-Morph (`src/alt_crossfield.rs`).

18. **pellenard-2011** — Isotropic 2D quadrangle meshing with size and orientation control. 2011. (Full citation in `md/pellenard-2011.md` and `pdf/pellenard-2011.pdf`.)
   Used in: Isotropic 2D quad meshing with size and orientation control (`src/alt_crossfield.rs`).

19. **remacle-2013** — Remacle J.-F., Henrotte F., Carrier-Baudouin T., Béchet E., Marchandise E., Geuzaine C., Mouton T., 2013. "A frontal Delaunay quad mesh generator using the L-infty norm." International Journal for Numerical Methods in Engineering, 94(5): 494-512.
   Used in: Frontal Delaunay in the L-infinity (Chebyshev) norm (`src/alt_l1.rs`).

20. **sastry-2014** — Sastry S.P., Shontz S.M., Vavasis S.A., 2014 (online 2012). "A log-barrier method for mesh quality improvement and untangling." Engineering with Computers, 30(3): 315-329.
   Used in: Log-barrier untangling / quality improvement (`src/alt_untangle.rs`).

21. **verma-2017** — Combinatorial approach to reduce singularities in quadrilateral meshes. 2017. (Full citation in `md/verma-2017.md` and `pdf/verma-2017.pdf`.)
   Used in: Combinatorial singularity reduction (`src/alt_untangle.rs`).

22. **verma-2018** — αMST: quad mesh adaptation in 2D and 3D. 2018. (Full citation in `md/verma-2018.md` and `pdf/verma-2018.pdf`.)
   Used in: alpha-MST quad mesh adaptation (`src/alt_untangle.rs`).

23. **chi-2025** — Chi B., Yu X., Guo Q., Zhang X., Hao J., Yuan W., 2025. "An efficient and robust unstructured quad-dominated mesh generation method based on the grid-based and Delaunay techniques." Engineering with Computers, 41(5): 3563-3585.
   Used in: Grid + Delaunay + Q-Morph quad-dominant route (surveyed; not implemented).

24. **ebeida-2010** — Q-TRAN: transforming triangular meshes into quadrilaterals locally. 2010. (Full citation in `md/ebeida-2010.md` and `pdf/ebeida-2010.pdf`.)
   Used in: Q-TRAN local triangle-to-quad transformation (surveyed; not implemented).

25. **bommes-2013** — Bommes D., Lévy B., Pietroni N., Puppo E., Silva C., Tarini M., Zorin D., 2013. "Quad-Mesh Generation and Processing: A Survey." Computer Graphics Forum, 32(6): 51-76.
   Used in: Quad-mesh generation and processing survey (general reference).

26. **simons-2024-thesis** — Simons L.C., 2024. "Size-Driven All-Quad Meshing for Geographic Data." PhD dissertation, University of California, Davis.
   Used in: Size-driven all-quad meshing: channel stitching (Ch2), size-aware GODF cross field (Ch3), size families and anisotropy (Ch4), QGP extraction (Ch5) (`src/simons_*.rs`, the `Sized` route).


## References inside the Simons thesis (implemented here)

27. **KCPS13** — Knoppel F., Crane K., Pinkall U., Schroder P., 2013. "Globally optimal direction fields." ACM TOG 32(4):59.
   Used in: The GODF solve of Chapter 3 (`src/simons_godf.rs`).

28. **CBK15** — Campen M., Bommes D., Kobbelt L., 2015. "Quantized global parametrization." ACM TOG 34(6).
   Used in: The QGP counts and Coons extraction of Chapter 5 (`src/simons_qgp.rs`).

29. **BS91** — Blacker T.D., Stephenson M.B., 1991. "Paving: a new approach to automated quadrilateral mesh generation." IJNME 32(4):811-847.
   Used in: The six-sided closure templates of Chapter 2; also `alt_paving`.

30. **Coo67** — Coons S.A., 1967. "Surfaces for computer-aided design of space forms." Technical report, MIT.
   Used in: The bilinear Coons patch of section 5.1.3 (`src/simons_qgp.rs`).

31. **SA17** — Simons L., Amenta N., 2017. "All-quad meshing for geographic data via templated boundary optimization." Procedia Engineering 203:388-400.
   Used in: The channel/patch construction of Chapter 2 (`src/simons_channel.rs`).

32. **FAR15** — Fogg H.J., Armstrong C.G., Robinson T.T., 2015. "Automatic generation of multiblock decompositions of surfaces." IJNME 101(13):965-991.
   Used in: The singularity-pair relaxation of section 5.1.4 (cited, not implemented).

33. **EGKT08** — Eppstein D., Goodrich M.T., Kim E., Tamstorf R., 2008. "Motorcycle graphs: canonical quad mesh partitioning." CGF 27:1477-1486.
   Used in: The motorcycle graph of sections 3.1.9/5 (cited, not implemented).


## Module map (thesis)

Where each `simons_*` module sits in the thesis, and which reference above it
follows. The public route names are in `src/config.rs`.

| Module | Thesis section | Primary reference |
| --- | --- | --- |
| `simons_channel` | Ch. 2 grid interior, boundary channel, patch cover | `simons-2024-thesis`, `SA17`, `BS91` |
| `simons_features` | Ch. 2 interior feature constraints | `simons-2024-thesis` |
| `simons_godf` | Ch. 3 size-aware Globally-Optimal Direction Field | `KCPS13` |
| `simons_sizefamily` | Ch. 4 size families, anisotropy A (4.7), streamline optimisation | `simons-2024-thesis` |
| `simons_limit` | §4.12 removable limit cycles | `simons-2024-thesis` |
| `simons_regions` | §3.1.6–3.1.12 singularities/separatrices/motorcycle graph, §5.1.4 tiles and nautilus | `KCPS13`, `EGKT08` |
| `simons_qgp` | Ch. 5 Quantized Global Parameterization, Coons extraction | `CBK15`, `Coo67` |
| `simons_graded` | Ch. 5 graded tile-grid QGP | `CBK15`, `Coo67` |
| `simons_smooth` | §2.2.10 angle + stretch smoothing | `simons-2024-thesis` |
| `simons_pipeline` | Ch. 2–5 integration (field → QGP → channel → smooth) | all of the above |
| `sized` | product wrapper around `simons_channel` (Chapter 2 route) | `simons-2024-thesis` |

## Notes

- The thesis' own bibliography has about 100 entries; only the ones implemented or
  explicitly approximated by this crate are listed above.

- `papers-metadata.txt` holds Crossref records for only part of the corpus; for
  the rest the title/year come from `INDEX.md` (the full citation is in the
  paper's own `md/` file).

- The corpus paper `remacle-2013` is titled "L 1 norm" in the PDF/markdown, but
  its abstract says "infinity norm" and the equations are `max(|dx|,|dy|)`: the
  route implements the L-infinity (Chebyshev) geometry, not the Manhattan norm.
