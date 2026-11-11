//! Paper module reproduction & clean-room SVG figure runner.
//!
//! Usage:
//!   cargo run --release --offline --example reproduce_paper -- list

use pedroso_mesher::core::io::svg::{render_mesh_svg, MeshDrawOptions};
use pedroso_mesher::papers;

const PAPER_MODULES: &[(&str, &str, &str)] = &[
    // Foundations
    (
        "shewchuk_1997",
        "foundations",
        "Shewchuk (1997) DCG — Adaptive precision predicates",
    ),
    (
        "shewchuk_2002",
        "foundations",
        "Shewchuk (2002) Comput. Geom. — Delaunay refinement & Terminator",
    ),
    (
        "knupp_2000",
        "foundations",
        "Knupp (2000) IJNME Parts I & II — Jacobian matrix norms",
    ),
    // Direct 2D Quad
    (
        "blacker_1991",
        "quad_direct",
        "Blacker & Stephenson (1991) IJNME — Paving",
    ),
    (
        "pellenard_2011",
        "quad_direct",
        "Pellenard, Alliez & Morvan (2011) IMR — Tile relaxation",
    ),
    (
        "remacle_2013",
        "quad_direct",
        "Remacle et al. (2013) IJNME — L-infinity frontal Delaunay",
    ),
    (
        "simons_2024",
        "quad_direct",
        "Simons (2024) UC Davis PhD Thesis — Size-driven all-quad",
    ),
    // Indirect 2D Quad
    (
        "owen_1999",
        "quad_indirect",
        "Owen, Staten, Canann & Saigal (1999) IJNME — Q-Morph",
    ),
    (
        "lee_2003",
        "quad_indirect",
        "Lee & Lee (2003) IJNME — Anisotropic SMT & local smoothing",
    ),
    (
        "merhof_2007",
        "quad_indirect",
        "Merhof et al. (2007) AES — Multi-pass anisotropic quads",
    ),
    (
        "ebeida_2010",
        "quad_indirect",
        "Ebeida et al. (2010) IMR — Q-TRAN local transformation",
    ),
    (
        "liu_2011",
        "quad_indirect",
        "Liu, Xing & Guan (2011) IJNME — Line-constrained indirect AFT",
    ),
    (
        "lu_2011",
        "quad_indirect",
        "Lu, Qian, Shi & Liu (2011) JCAM — Topological quading",
    ),
    (
        "remacle_2012",
        "quad_indirect",
        "Remacle et al. (2012) IJNME — Blossom-Quad",
    ),
    (
        "liu_2013",
        "quad_indirect",
        "Liu & Xing (2013) JCP — Multi-material geodesic & Pisces",
    ),
    (
        "araujo_2014",
        "quad_indirect",
        "Araujo, Celes & Gattass (2014) — Deferred constraint insertion",
    ),
    (
        "pellenard_2014",
        "quad_indirect",
        "Pellenard et al. (2014) — QMCF cross-field Q-Morph",
    ),
    (
        "johnen_2015",
        "quad_indirect",
        "Johnen, Ernst & Geuzaine (2015) — Sequential look-ahead",
    ),
    (
        "sun_2015_2016",
        "quad_indirect",
        "Sun, Yeh, Lin & Zhao (2015, 2016) — SJ residual walk & parity",
    ),
    (
        "zint_2021",
        "quad_indirect",
        "Zint & Grosso (2021) — Hybrid Blossom band + greedy interior",
    ),
    (
        "georgiadis_2022",
        "quad_indirect",
        "Georgiadis, Reberol & Remacle (2022) — Bipartite labeling",
    ),
    (
        "gao_2024",
        "quad_indirect",
        "Gao et al. (2024) IJNME — Local-global remeshing & Blossom",
    ),
    (
        "wang_2025",
        "quad_indirect",
        "Wang et al. (2025) — Concave-front Q-Morph & templates",
    ),
    (
        "qiu_2026",
        "quad_indirect",
        "Qiu et al. (2026) — MSQ dual-field & bipartite marking",
    ),
    (
        "wang_2026",
        "quad_indirect",
        "Wang et al. (2026) — Virtual-topology suppression & Q-Morph",
    ),
    // Grid & Quadtree
    (
        "rushdi_2015_2017",
        "quad_grid",
        "Rushdi et al. (2015, 2017) CAD — All-quad without cleanup",
    ),
    (
        "pochet_2017",
        "quad_grid",
        "Pochet, Celes, Lopes & Gattass (2017) — Extended quadtree S1/S2",
    ),
    (
        "smith_pierce_2024",
        "quad_grid",
        "Smith-Pierce, Ruffin & Dement (2024) — Cartesian + wall layers",
    ),
    (
        "chi_2025",
        "quad_grid",
        "Chi et al. (2025) — Binary-tree core + Delaunay/Q-Morph",
    ),
    // Decomposition & Singularity Templates
    (
        "nasri_2009",
        "quad_decompose",
        "Nasri, Sabin & Yasseen (2009) CGF — n-sided Coons & refractors",
    ),
    (
        "ma_2017",
        "quad_decompose",
        "Ma, Zhao & Sun (2017) AES — Line-constrained decomposition & ILP",
    ),
    (
        "miranda_2017",
        "quad_decompose",
        "Miranda & Martha (2017) — Hierarchical 4/3/2-curve templates",
    ),
    (
        "verma_2017_2018",
        "quad_decompose",
        "Verma & Suresh (2017, 2018) — MST & alpha-MST patches",
    ),
    (
        "yang_2024",
        "quad_decompose",
        "Yang et al. (2024) FEAD — Singularity counting & CDT bisection",
    ),
    // Smoothing, Untangling & Curving
    (
        "menendez_diaz_2004",
        "smoothing",
        "Menendez-Diaz et al. (2004) — 2D geometric-stiffness direct",
    ),
    (
        "menendez_diaz_2005",
        "smoothing",
        "Menendez-Diaz et al. (2005) — 3D hex geometric-stiffness direct",
    ),
    (
        "xie_2013",
        "smoothing",
        "Xie, Sevilla, Hassan & Morgan (2013) — High-order curving",
    ),
    (
        "sastry_2014",
        "smoothing",
        "Sastry, Shontz & Vavasis (2014) — Log-barrier untangling",
    ),
    (
        "lin_2014",
        "smoothing",
        "Lin, Guan, Chang & Lo (2014) — VerBSS spring smoothing",
    ),
    (
        "durand_2019",
        "smoothing",
        "Durand, Pantoja-Rosero & Oliveira (2019) — Least-squares smoother",
    ),
    (
        "sun_2020",
        "smoothing",
        "Sun, Gou & Geng (2020) — EEGT + weighted element stitching",
    ),
    (
        "yang_2020",
        "smoothing",
        "Yang, Zhang, Ren & Xu (2020) — Markov-chain smoothing",
    ),
    // 3D Hexahedral
    (
        "yerry_1984",
        "hex",
        "Yerry & Shephard (1984) IJNME — Modified-octree 3D meshing",
    ),
    (
        "schneiders_1996",
        "hex",
        "Schneiders (1996) — Grid-based hex & isomorphic buffer",
    ),
    (
        "tautges_1996",
        "hex",
        "Tautges, Blacker & Mitchell (1996) — Whisker weaving",
    ),
    ("owen_2000", "hex", "Owen & Saigal (2000) IJNME — H-Morph"),
    (
        "carrier_baudouin_2014",
        "hex",
        "Carrier-Baudouin et al. (2014) — Frontal hex-dominant",
    ),
    (
        "bosnjak_2023",
        "hex",
        "Bosnjak et al. (2023) — Higher-order tubular block hex",
    ),
    (
        "zhang_2023",
        "hex",
        "Zhang et al. (2023) CAD — Hex singular node sheet inflation",
    ),
    (
        "tong_2024",
        "hex",
        "Tong, Halilaj & Zhang (2024) — HybridOctree_Hex",
    ),
    (
        "tong_2026_mchex",
        "hex",
        "Tong & Zhang (2026) ACM TOG — MCHex positive-Jacobian hex",
    ),
    (
        "tong_2026_templates",
        "hex",
        "Tong & Zhang (2026) CAGD — Element-saving 3-refinement",
    ),
];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("list");

    if cmd == "list" {
        println!(
            "{:<24} {:<16} Description ({} modules)",
            "Module",
            "Family",
            PAPER_MODULES.len()
        );
        println!("{}", "-".repeat(96));
        for &(module, family, desc) in PAPER_MODULES {
            println!("{module:<24} {family:<16} {desc}");
        }
        return;
    }

    // Smoke-test SVG rendering for an implemented legacy module when requested:
    if cmd == "smoke-svg" {
        let Ok(domain) = papers::common::two_d::Domain::plate_with_hole(100.0, 60.0, 10.0, 24)
        else {
            eprintln!("failed to construct benchmark domain");
            std::process::exit(1);
        };
        let Ok(sizing) = papers::common::two_d::Sizing::uniform(4.0) else {
            eprintln!("failed to construct sizing");
            std::process::exit(1);
        };
        let Ok(mesh) = papers::quad_grid::rushdi_2015_2017::mesh(&domain, &sizing) else {
            eprintln!("failed to generate smoke mesh");
            std::process::exit(1);
        };
        let opts = MeshDrawOptions {
            title: "rushdi_2015_2017 — Plate with Hole (smoke check)".to_string(),
            ..MeshDrawOptions::default()
        };
        match render_mesh_svg(&mesh, Some(&domain), &opts) {
            Ok(svg) => println!(
                "Rendered {} bytes of SVG ({} cells)",
                svg.len(),
                mesh.cells.len()
            ),
            Err(err) => {
                eprintln!("SVG error: {err}");
                std::process::exit(1);
            }
        }
        return;
    }

    eprintln!("unknown subcommand {cmd:?}; expected 'list' or 'smoke-svg'");
    std::process::exit(2);
}
