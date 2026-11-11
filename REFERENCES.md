# References

Bibliography of the scientific journal papers, conference proceedings, and university theses implemented or referenced in `pedroso_mesher`. All algorithms are implemented clean-room in pure Rust strictly from these public publications.

---

## 1. Foundations (Geometric Predicates, Delaunay Refinement, Quality Metrics & Graph Matching)

1. **shewchuk-1997** — Shewchuk J.R., 1997. "Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates." *Discrete & Computational Geometry*, 18(3): 305–363. DOI: `10.1007/PL00009321`.
   Module: `src/papers/foundations/shewchuk_1997.rs`.

2. **shewchuk-2002-delaunay-refinement** — Shewchuk J.R., 2002. "Delaunay refinement algorithms for triangular mesh generation." *Computational Geometry*, 22(1–3): 21–74. DOI: `10.1016/S0925-7721(01)00047-5`.
   Module: `src/papers/foundations/shewchuk_2002.rs`.

3. **knupp-2000-part1** — Knupp P.M., 2000. "Achieving finite element mesh quality via optimization of the Jacobian matrix norm and associated quantities. Part I—A framework for surface mesh optimization." *International Journal for Numerical Methods in Engineering*, 48(3): 401–420.
   Module: `src/papers/foundations/knupp_2000.rs`, `src/core/quality.rs`.

4. **knupp-2000-part2** — Knupp P.M., 2000. "Achieving finite element mesh quality via optimization of the Jacobian matrix norm and associated quantities. Part II—A framework for volume mesh optimization and the condition number of the Jacobian matrix." *International Journal for Numerical Methods in Engineering*, 48(8): 1165–1185.
   Module: `src/papers/foundations/knupp_2000.rs`, `src/core/quality.rs`.

5. **edmonds-1965** — Edmonds J., 1965. "Paths, trees, and flowers." *Canadian Journal of Mathematics*, 17: 449–467; and Edmonds J., 1965. "Maximum matching and a polyhedron with 0,1-vertices." *Journal of Research of the National Bureau of Standards*, 69B: 125–130. (See also Galil Z., 1986, *ACM Computing Surveys* 18(1): 23–38.)
   Module: `src/core/matching.rs`.

---

## 2. Direct 2D Quadrilateral Generation

6. **blacker-1991** — Blacker T.D., Stephenson M.B., 1991. "Paving: A new approach to automated quadrilateral mesh generation." *International Journal for Numerical Methods in Engineering*, 32(4): 811–847. DOI: `10.1002/nme.1620320410`.
   Module: `src/papers/quad_direct/blacker_1991.rs`.

7. **pellenard-2011** — Pellenard B., Alliez P., Morvan J.-M., 2011. "Isotropic 2D quadrangle meshing with size and orientation control." *Proceedings of the 20th International Meshing Roundtable*, Springer, pp. 81–98.
   Module: `src/papers/quad_direct/pellenard_2011.rs`.

8. **remacle-2013** — Remacle J.-F., Henrotte F., Carrier-Baudouin T., Béchet E., Marchandise E., Geuzaine C., Mouton T., 2013. "A frontal Delaunay quad mesh generator using the $L^\infty$ norm." *International Journal for Numerical Methods in Engineering*, 94(5): 494–512. DOI: `10.1002/nme.4458`.
   Module: `src/papers/quad_direct/remacle_2013.rs`.

9. **simons-2024-thesis** — Simons L.C., 2024. *Size-Driven All-Quad Meshing for Geographic Data*. PhD dissertation, University of California, Davis. (Incorporating Simons L., Amenta N., 2017, *Procedia Engineering* 203: 388–400; Knöppel F., Crane K., Pinkall U., Schröder P., 2013, *ACM TOG* 32(4): 59; Campen M., Bommes D., Kobbelt L., 2015, *ACM TOG* 34(6); Eppstein D. et al., 2008, *CGF* 27: 1477–1486; Coons S.A., 1967, MIT MAC-TR-41.)
   Module: `src/papers/quad_direct/simons_2024/`.

---

## 3. Indirect 2D Quadrilateral Conversion (Triangle-to-Quad)

10. **owen-1999** — Owen S.J., Staten M.L., Canann S.A., Saigal S., 1999. "Q-Morph: an indirect approach to advancing front quad meshing." *International Journal for Numerical Methods in Engineering*, 44(9): 1317–1340.
    Module: `src/papers/quad_indirect/owen_1999.rs`.

11. **lee-2003** — Lee Y.K., Lee C.K., 2003. "A new indirect anisotropic quadrilateral mesh generation scheme with enhanced local mesh smoothing procedures." *International Journal for Numerical Methods in Engineering*, 58(2): 313–339.
    Module: `src/papers/quad_indirect/lee_2003.rs`.

12. **merhof-2007** — Merhof D., Grosso R., Tremel U., Greiner G., 2007. "Anisotropic quadrilateral mesh generation: An indirect approach." *Advances in Engineering Software*, 38(8–9): 560–568.
    Module: `src/papers/quad_indirect/merhof_2007.rs`.

13. **ebeida-2010** — Ebeida M.S., Karamete K., Mestreau E., Dey S., 2010. "Q-TRAN: A new approach to transform triangular meshes into quadrilateral meshes locally." *Proceedings of the 19th International Meshing Roundtable*, Springer, pp. 23–34.
    Module: `src/papers/quad_indirect/ebeida_2010.rs`.

14. **liu-2011** — Liu Y., Xing H.L., Guan Z., 2011. "An indirect approach for automatic generation of quadrilateral meshes with arbitrary line constraints." *International Journal for Numerical Methods in Engineering*, 87(9): 906–924.
    Module: `src/papers/quad_indirect/liu_2011.rs`.

15. **lu-2011** — Lu L., Qian X., Shi X., Liu F., 2011. "Quading triangular meshes with certain topological constraints." *Journal of Computational and Applied Mathematics*, 235(13): 3826–3836.
    Module: `src/papers/quad_indirect/lu_2011.rs`.

16. **remacle-2012** — Remacle J.-F., Lambrechts J., Seny B., Marchandise E., Johnen A., Geuzaine C., 2012. "Blossom-Quad: A non-uniform quadrilateral mesh generator using a minimum-cost perfect-matching algorithm." *International Journal for Numerical Methods in Engineering*, 89(9): 1102–1119. DOI: `10.1002/nme.3279`.
    Module: `src/papers/quad_indirect/remacle_2012.rs`.

17. **liu-2013** — Liu Y., Xing H.L., 2013. "A boundary focused quadrilateral mesh generation algorithm for multi-material structures." *Journal of Computational Physics*, 232(1): 324–342.
    Module: `src/papers/quad_indirect/liu_2013.rs`.

18. **araujo-2014** — Araújo C., Celes W., Gattass M., 2014. "Quadrilateral mesh generation with deferred constraint insertion." *Procedia Engineering*, 82: 78–90.
    Module: `src/papers/quad_indirect/araujo_2014.rs`.

19. **pellenard-2014** — Pellenard B., Orbay G., Chen J., Sohan S., Kwok W., Tristano J.R., 2014. "QMCF: QMorph cross field-driven quad-dominant meshing algorithm." *Procedia Engineering*, 82: 338–350.
    Module: `src/papers/quad_indirect/pellenard_2014.rs`.

20. **johnen-2015** — Johnen A., Ernst D., Geuzaine C., 2015. "Sequential decision-making approach for quadrangular mesh generation." *Engineering with Computers*, 31(4): 719–730.
    Module: `src/papers/quad_indirect/johnen_2015.rs`.

21. **sun-2015** — Sun L., Zhao G., Yeh G.-T., 2015. "Automatic quadrilateral mesh generation and quality improvement based on an improved combination method." *Engineering with Computers*, 31(4): 675–692.
    Module: `src/papers/quad_indirect/sun_2015_2016.rs`.

22. **sun-2016** — Sun L., Yeh G.-T., Lin F.-P., Zhao G., 2016. "An automatic method for complete triangular mesh conversion into quadrilateral mesh for multiple domain geometry." *Terrestrial, Atmospheric and Oceanic Sciences*, 27(2): 139–152. DOI: `10.3319/TAO.2015.09.14.01(Hy)`.
    Module: `src/papers/quad_indirect/sun_2015_2016.rs`.

23. **zint-2021** — Zint D., Grosso R., 2021. "A hybrid approach to fast indirect quadrilateral mesh generation." *Computing and Visualization in Science*, 23.
    Module: `src/papers/quad_indirect/zint_2021.rs`.

24. **georgiadis-2022** — Georgiadis C., Reberol M., Remacle J.-F., 2022. "Indirect all-quadrilateral meshing based on bipartite topological labeling." *Engineering with Computers*, 38: 5365–5385.
    Module: `src/papers/quad_indirect/georgiadis_2022.rs`.

25. **gao-2024** — Gao J., Xiao Z., Xu C., Shen S., Xu G., 2024. "Quadrilateral surface mesh generation with improved quality by combination of triangles." *International Journal for Numerical Methods in Engineering*, 125.
    Module: `src/papers/quad_indirect/gao_2024.rs`.

26. **wang-2025** — Wang H., Yin S., Yan H., Wang F., Yu X., Liu S., Cui X., 2025. "Improved Q-Morph algorithm for quad-dominant hybrid mesh generation with advanced front propagation and topology optimization." *Engineering with Computers*.
    Module: `src/papers/quad_indirect/wang_2025.rs`.

27. **qiu-2026** — Qiu J., Qi L., Zhu X., Tang M., Liu Y., Xu G., Pang Y., 2026. "MSQ-meshing: a multi-constraint size-adaptive quad-dominant mesh generation method for complex surfaces." *Computers & Structures*.
    Module: `src/papers/quad_indirect/qiu_2026.rs`.

28. **wang-2026** — Wang H., Wang F., Yin S., Li S., Wang G., Wang L., Cui X., 2026. "High-quality quad-dominant mesh generation aided by the virtual topological model generated after automatic suppression of small geometric features." *Computers & Structures*.
    Module: `src/papers/quad_indirect/wang_2026.rs`.

---

## 4. Grid & Quadtree 2D Quadrilateral Generation

29. **rushdi-2015** — Rushdi A.A., Mitchell S.A., Mahmoud A.H., Bajaj C.C., Ebeida M.S., 2015. "Robust all-quad meshing of domains with connected regions." *Procedia Engineering*, 124: 96–108.
    Module: `src/papers/quad_grid/rushdi_2015_2017.rs`.

30. **rushdi-2017** — Rushdi A.A., Mitchell S.A., Mahmoud A.H., Bajaj C.C., Ebeida M.S., 2017. "All-quad meshing without cleanup." *Computer-Aided Design*, 84: 1–16.
    Module: `src/papers/quad_grid/rushdi_2015_2017.rs`.

31. **pochet-2017** — Pochet A., Celes W., Lopes H., Gattass M., 2017. "A new quadtree-based approach for automatic quadrilateral mesh generation." *Engineering with Computers*, 33(2): 275–292.
    Module: `src/papers/quad_grid/pochet_2017.rs`.

32. **smith-pierce-2024** — Smith-Pierce M., Ruffin S., Dement D., 2024. "Automated Unstructured Quad/Hex Meshing for High-Order Discontinuous Galerkin CFD." *AIAA SciTech 2024 Forum*, AIAA 2024-0384. DOI: `10.2514/6.2024-0384`.
    Module: `src/papers/quad_grid/smith_pierce_2024.rs`.

33. **chi-2025** — Chi B., Yu X., Guo Q., Zhang X., Hao J., Yuan W., 2025. "An efficient and robust unstructured quad-dominated mesh generation method based on the grid-based and Delaunay techniques." *Engineering with Computers*, 41(5): 3563–3585. DOI: `10.1007/s00366-025-02179-z`.
    Module: `src/papers/quad_grid/chi_2025.rs`.

---

## 5. Domain Decomposition, Templates & Singularity Reduction

34. **nasri-2009** — Nasri A., Sabin M., Yasseen Z., 2009. "Filling $n$-sided regions by quad meshes for subdivision surfaces." *Computer Graphics Forum*, 28(6): 1644–1658.
    Module: `src/papers/quad_decompose/nasri_2009.rs`.

35. **ma-2017** — Ma X., Zhao G., Sun L., 2017. "Sub-domain decomposition and quadrilateral meshing with line constraints." *Advances in Engineering Software*, 109: 26–38.
    Module: `src/papers/quad_decompose/ma_2017.rs`.

36. **miranda-2017** — Miranda A.C.O., Martha L.F., 2017. "Hierarchical template-based quadrilateral mesh generation." *Engineering with Computers*, 33(2): 123–138.
    Module: `src/papers/quad_decompose/miranda_2017.rs`.

37. **verma-2017** — Verma C.S., Suresh K., 2017. "A combinatorial approach to reduce singularities in quadrilateral meshes." *Computer-Aided Design*, 85: 99–110.
    Module: `src/papers/quad_decompose/verma_2017_2018.rs`.

38. **verma-2018** — Verma C.S., Suresh K., 2018. "$\alpha$MST: A unified algorithm for quad mesh adaptation." *International Journal for Numerical Methods in Engineering*, 113: 513–536.
    Module: `src/papers/quad_decompose/verma_2017_2018.rs`.

39. **yang-2024** — Yang S. et al., 2024. "Boundary-constrained quadrilateral mesh generation based on domain decomposition and templates." *Finite Elements in Analysis and Design*.
    Module: `src/papers/quad_decompose/yang_2024.rs`.

---

## 6. Smoothing, Untangling & High-Order Boundary Curving

40. **diaz-2004** — Menéndez-Díaz A., González-Nicieza C., Álvarez-Vigil A.E., 2004. "Improvement of quadrilateral meshes for discretization of tunnels." *Computers and Geotechnics*, 31(1): 47–56.
    Module: `src/papers/smoothing/menendez_diaz_2004.rs`.

41. **diaz-2005** — Menéndez-Díaz A., González-Nicieza C., Álvarez-Vigil A.E., 2005. "Hexahedral mesh smoothing using a direct method." *Computers & Geosciences*, 31(4): 453–463.
    Module: `src/papers/smoothing/menendez_diaz_2005.rs`.

42. **xie-2013** — Xie Z.Q., Sevilla R., Hassan O., Morgan K., 2013. "The generation of arbitrary order curved meshes for 3D finite element analysis." *Computational Mechanics*, 51(3): 361–374. DOI: `10.1007/s00466-012-0736-4`.
    Module: `src/papers/smoothing/xie_2013.rs`.

43. **sastry-2014** — Sastry S.P., Shontz S.M., Vavasis S.A., 2014. "A log-barrier method for mesh quality improvement and untangling." *Engineering with Computers*, 30(3): 315–329.
    Module: `src/papers/smoothing/sastry_2014.rs`.

44. **lin-2014** — Lin T.J., Guan Z.Q., Chang J.H., Lo S.H., 2014. "Vertex-Ball Spring Smoothing: An efficient method for unstructured dynamic hybrid meshes." *Computers & Structures*, 136: 24–33.
    Module: `src/papers/smoothing/lin_2014.rs`.

45. **durand-2019** — Durand R., Pantoja-Rosero B.G., Oliveira V., 2019. "A general mesh smoothing method for finite elements." *Finite Elements in Analysis and Design*, 158: 17–30.
    Module: `src/papers/smoothing/durand_2019.rs`.

46. **sun-2020** — Sun S., Gou Z., Geng M., 2020. "Simultaneous Smoothing and Untangling of 2D Meshes Based on Explicit Element Geometric Transformation and Element Stitching." *Applied Sciences*, 10(14): 4994.
    Module: `src/papers/smoothing/sun_2020.rs`.

47. **yang-2020** — Yang F., Zhang D., Ren H., Xu J., 2020. "2D Mesh smoothing based on Markov chain method." *Engineering with Computers*, 36: 1615–1626. DOI: `10.1007/s00366-019-00786-1`.
    Module: `src/papers/smoothing/yang_2020.rs`.

---

## 7. 3D Hexahedral Generation

48. **yerry-1984** — Yerry M.A., Shephard M.S., 1984. "Automatic three-dimensional mesh generation by the modified-octree technique." *International Journal for Numerical Methods in Engineering*, 20(11): 1965–1990. DOI: `10.1002/nme.1620201103`.
    Module: `src/papers/hex/yerry_1984.rs`.

49. **schneiders-1996** — Schneiders R., 1996. "A grid-based algorithm for the generation of hexahedral element meshes." *Engineering with Computers*, 12(3–4): 168–177.
    Module: `src/papers/hex/schneiders_1996.rs`.

50. **tautges-1996** — Tautges T.J., Blacker T., Mitchell S.A., 1996. "The whisker weaving algorithm: A connectivity-based method for constructing all-hexahedral finite element meshes." *International Journal for Numerical Methods in Engineering*, 39(19): 3327–3349.
    Module: `src/papers/hex/tautges_1996.rs`.

51. **owen-2000** — Owen S.J., Saigal S., 2000. "H-Morph: an indirect approach to advancing front hex meshing." *International Journal for Numerical Methods in Engineering*, 49(1–2): 289–312.
    Module: `src/papers/hex/owen_2000.rs`.

52. **carrier-baudouin-2014** — Carrier-Baudouin T., Remacle J.-F., Marchandise E., Lambrechts J., 2014. "A frontal approach to hex-dominant mesh generation." *Advanced Modeling and Simulation in Engineering Sciences*, 1: 8. DOI: `10.1186/2213-7467-1-8`.
    Module: `src/papers/hex/carrier_baudouin_2014.rs`.

53. **bosnjak-2023** — Bošnjak D., Pepe A., Schussnig R., Schmalstieg D., Fries T.-P., 2023. "Higher-order block-structured hex meshing of tubular structures." *Engineering with Computers*. DOI: `10.1007/s00366-023-01834-7`.
    Module: `src/papers/hex/bosnjak_2023.rs`.

54. **zhang-2023** — Zhang P., Chiang J., Fan X., Mundilova K., 2023. "Local decomposition of hexahedral singular nodes into singular curves." *Computer-Aided Design*, 158: 103484. DOI: `10.1016/j.cad.2023.103484`.
    Module: `src/papers/hex/zhang_2023.rs`.

55. **tong-2024** — Tong H., Halilaj E., Zhang Y.J., 2024. "HybridOctree_Hex: Hybrid octree-based adaptive all-hexahedral mesh generation with Jacobian control." *Journal of Computational Science*, 78: 102278. DOI: `10.1016/j.jocs.2024.102278`.
    Module: `src/papers/hex/tong_2024.rs`.

56. **tong-2026-mchex** — Tong H., Zhang Y.J., 2026. "MCHex: marching cubes based adaptive hexahedral mesh generation with guaranteed positive Jacobian." *ACM Transactions on Graphics*. DOI: `10.1145/3858338`.
    Module: `src/papers/hex/tong_2026_mchex.rs`.

57. **tong-2026-element-saving** — Tong H., Zhang Y.J., 2026. "Element-saving hexahedral 3-refinement templates." *Computer Aided Geometric Design*, 102550. DOI: `10.1016/j.cagd.2026.102550`.
    Module: `src/papers/hex/tong_2026_templates.rs`.

---

## 8. Surveys & Non-Classical Reference Works (Reviewed; No Rust Module)

58. **bommes-2013** — Bommes D., Lévy B., Pietroni N., Puppo E., Silva C., Tarini M., Zorin D., 2013. "Quad-Mesh Generation and Processing: A Survey." *Computer Graphics Forum*, 32(6): 51–76.

59. **pietroni-2022** — Pietroni N., Campen M., Sheffer A., Cherchi G., Bommes D., Gao X., Scateni R., Ledoux F., Remacle J.-F., Livesu M., 2022. "Hex-Mesh Generation and Processing: A Survey." *ACM Transactions on Graphics*, 42(2). DOI: `10.1145/3554920`.

60. **tong-2023** — Tong H., Qian K., Zhang Y.J., 2023. "SRL-assisted AFM: learning-assisted advancing front quad meshing." (Deep neural-network / RL method; excluded from classical library).

61. **xie-2026** — Xie et al., 2026. "Intelligent quad meshing for ship structures by hybrid deep reinforcement learning (S2R-AFM)." (Deep RL policy method; excluded from classical library).
