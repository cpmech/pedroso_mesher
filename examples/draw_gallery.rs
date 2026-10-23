//! Draw every studied mesh: each corpus route on each corpus part, raw and
//! smoothed, plus the feature demos and the graded pipeline.
//!
//! ```bash
//! cargo run --release --offline --example draw_gallery
//! ```
//!
//! Output (all under `data/figures/`):
//!
//! ```text
//! corpus/<route>/<part>-raw.png
//! corpus/<route>/<part>-smooth.png
//! pipeline/<case>-raw.png
//! pipeline/<case>-smooth.png
//! features/<case>-raw.png
//! features/<case>-smooth.png
//! manifest.csv   one row per generated case (or per route error)
//! INDEX.md       what the gallery contains and how to read it
//! ```
//!
//! "raw" disables the shared post-smoother (`SmoothOptions::enabled = false`)
//! and, for the pipeline route, its internal Section 2.2.10 pass as well; the
//! generator is otherwise unchanged. The cell fill is the gate margin
//! `min(angle/30, sj/0.2)` (green clears the gate, yellow/orange/red fails,
//! magenta is inverted, triangles are dark red), so every figure is readable in
//! the gate's own terms.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::fmt::Write as _;
use std::fs;

use pedroso_mesher::config::{generate, MeshConfig, Route};
use pedroso_mesher::draw::{self, MeshDrawOptions};
use pedroso_mesher::simons_features::TerminatingMode;
use pedroso_mesher::two_d::{Domain, Sizing};

const ROOT: &str = "data/figures";

const PARTS: [&str; 7] = [
    "plate-hole-4mm",
    "plate-hole-2mm",
    "plate-hole-1mm",
    "plate-hole-0.6mm",
    "l-bracket",
    "ring",
    "stepped",
];

const ROUTES: [Route; 11] = [
    Route::Qmorph,
    Route::Paving,
    Route::Bipartite,
    Route::Quadtree,
    Route::Decompose,
    Route::Hybrid,
    Route::Crossfield,
    Route::Untangle,
    Route::L1,
    Route::Sized,
    Route::Pipeline,
];

fn part(name: &str) -> (Domain, f64) {
    match name {
        "plate-hole-4mm" => (Domain::plate_with_hole(100.0, 60.0, 10.0, 16).unwrap(), 4.0),
        "plate-hole-2mm" => (Domain::plate_with_hole(100.0, 60.0, 10.0, 24).unwrap(), 2.0),
        "plate-hole-1mm" => (Domain::plate_with_hole(100.0, 60.0, 10.0, 48).unwrap(), 1.0),
        "plate-hole-0.6mm" => (Domain::plate_with_hole(100.0, 60.0, 10.0, 64).unwrap(), 0.6),
        "l-bracket" => (
            Domain::new(
                vec![
                    [0.0, 0.0],
                    [80.0, 0.0],
                    [80.0, 20.0],
                    [20.0, 20.0],
                    [20.0, 60.0],
                    [0.0, 60.0],
                ],
                vec![],
            )
            .unwrap(),
            2.0,
        ),
        "ring" => {
            let arc = |r: f64, a0: f64, a1: f64, n: usize| -> Vec<[f64; 2]> {
                (0..=n)
                    .map(|i| {
                        let a = a0 + (a1 - a0) * i as f64 / n as f64;
                        [r * a.cos(), r * a.sin()]
                    })
                    .collect()
            };
            let mut pts = arc(20.0, 0.0, std::f64::consts::FRAC_PI_2, 24);
            pts.extend(arc(10.0, std::f64::consts::FRAC_PI_2, 0.0, 12));
            (Domain::new(pts, vec![]).unwrap(), 1.0)
        }
        "stepped" => (
            Domain::new(
                vec![
                    [0.0, 0.0],
                    [60.0, 0.0],
                    [60.0, 40.0],
                    [30.0, 40.0],
                    [30.0, 30.0],
                    [20.0, 30.0],
                    [20.0, 40.0],
                    [0.0, 40.0],
                ],
                vec![],
            )
            .unwrap(),
            1.0,
        ),
        other => panic!("unknown part {other}"),
    }
}

/// One generated case, or the error that prevented it.
struct Record {
    group: String,
    route: String,
    mesh: String,
    smoothing: String,
    status: String,
    quads: usize,
    triangles: usize,
    min_sj: f64,
    min_angle: f64,
    passes: bool,
    file: String,
}

impl Record {
    fn error(group: &str, route: &str, mesh: &str, smoothing: &str, message: &str) -> Self {
        Self {
            group: group.to_string(),
            route: route.to_string(),
            mesh: mesh.to_string(),
            smoothing: smoothing.to_string(),
            status: format!("error: {message}"),
            quads: 0,
            triangles: 0,
            min_sj: 0.0,
            min_angle: 0.0,
            passes: false,
            file: String::new(),
        }
    }

    fn csv(&self) -> String {
        let status = self.status.replace(',', ";");
        if self.file.is_empty() {
            return format!(
                "{},{},{},{},{},,,,,,\n",
                self.group, self.route, self.mesh, self.smoothing, status
            );
        }
        format!(
            "{},{},{},{},{},{},{},{:.4},{:.2},{},{}\n",
            self.group,
            self.route,
            self.mesh,
            self.smoothing,
            status,
            self.quads,
            self.triangles,
            self.min_sj,
            self.min_angle,
            if self.passes { "PASS" } else { "FAIL" },
            self.file
        )
    }
}

/// Generate, draw and describe one case.
#[allow(clippy::too_many_arguments)]
fn figure(
    group: &str,
    route: &Route,
    mesh_name: &str,
    smoothing: &str,
    domain: &Domain,
    sizing: &Sizing,
    config: &MeshConfig,
) -> Record {
    let route_name = route.name();
    let mut config = config.clone();
    let smooth = smoothing == "smooth";
    config.smoothing.enabled = smooth;
    if *route == Route::Pipeline {
        config.pipeline.smooth = smooth;
    }
    let out = match generate(domain, sizing, &config) {
        Ok(out) => out,
        Err(error) => {
            println!("{group}/{route_name}/{mesh_name} {smoothing}: error: {error}");
            return Record::error(group, route_name, mesh_name, smoothing, &error);
        }
    };
    let stats = &out.statistics;
    let rejected = smooth && out.smoothing.as_ref().map(|s| !s.accepted).unwrap_or(false);
    let title = format!(
        "{} - {} - {}{}  |  {} q, {} tri, SJ {:.3}, {:.1} deg, {}",
        route_name,
        mesh_name,
        smoothing,
        if rejected {
            " (rejected, raw kept)"
        } else {
            ""
        },
        stats.quads,
        stats.triangles,
        stats.min_scaled_jacobian,
        stats.min_corner_angle_deg,
        if stats.passes { "PASS" } else { "FAIL" }
    );
    let group_dir = if group == "corpus" {
        format!("{group}/{route_name}")
    } else {
        group.to_string()
    };
    let file = format!("{ROOT}/{group_dir}/{mesh_name}-{smoothing}.png");
    if let Some(parent) = std::path::Path::new(&file).parent() {
        fs::create_dir_all(parent).unwrap();
    }
    let options = MeshDrawOptions {
        title,
        figure_inches: (7.0, 5.0),
        ..MeshDrawOptions::default()
    };
    if let Err(error) = draw::draw_domain_with_mesh(domain, &out.mesh, &file, &options) {
        println!("{file}: draw error: {error}");
        return Record::error(group, route_name, mesh_name, smoothing, error);
    }
    // plotpy writes the matplotlib script next to the figure; the gallery keeps
    // only the images.
    let script = file.replace(".png", ".py");
    let _ = fs::remove_file(script);
    println!(
        "{file}: {} q, {} tri, SJ {:.3}, {:.1} deg, {}{}",
        stats.quads,
        stats.triangles,
        stats.min_scaled_jacobian,
        stats.min_corner_angle_deg,
        if stats.passes { "PASS" } else { "FAIL" },
        if rejected {
            " (smoothing rejected)"
        } else {
            ""
        }
    );
    Record {
        group: group.to_string(),
        route: route_name.to_string(),
        mesh: mesh_name.to_string(),
        smoothing: smoothing.to_string(),
        status: if rejected {
            "smoothing rejected".to_string()
        } else {
            "ok".to_string()
        },
        quads: stats.quads,
        triangles: stats.triangles,
        min_sj: stats.min_scaled_jacobian,
        min_angle: stats.min_corner_angle_deg,
        passes: stats.passes,
        file,
    }
}

fn feature_cases() -> Vec<(&'static str, Vec<Vec<[f64; 2]>>, bool)> {
    vec![
        ("through", vec![vec![[5.0, 0.0], [5.0, 10.0]]], false),
        ("stub-slot", vec![vec![[5.0, 0.0], [5.0, 5.0]]], false),
        (
            "cross",
            vec![vec![[5.0, 0.0], [5.0, 10.0]], vec![[0.0, 5.0], [10.0, 5.0]]],
            false,
        ),
        (
            "through-terminating",
            vec![vec![[5.0, 0.0], [5.0, 10.0]], vec![[2.0, 0.0], [2.0, 5.0]]],
            false,
        ),
        ("stub-channel", vec![vec![[5.0, 0.0], [5.0, 5.0]]], true),
    ]
}

fn main() {
    fs::create_dir_all(ROOT).unwrap();
    let mut records = Vec::new();

    for name in PARTS {
        let (domain, size) = part(name);
        let sizing = Sizing::uniform(size).unwrap();
        for route in ROUTES {
            for smoothing in ["raw", "smooth"] {
                let config = MeshConfig::new(route);
                records.push(figure(
                    "corpus", &route, name, smoothing, &domain, &sizing, &config,
                ));
            }
        }
    }

    // The feature route needs interior polylines, so it is drawn on the
    // square-with-features cases of `simons_features::demo()`, not on the
    // corpus parts.
    let square = Domain::rectangle(10.0, 10.0).unwrap();
    let unit = Sizing::uniform(1.0).unwrap();
    for (case, polylines, channel) in feature_cases() {
        let mut config = MeshConfig::new(Route::Features);
        config.features = polylines;
        if channel {
            config.feature_options.terminating = TerminatingMode::Channel;
        }
        for smoothing in ["raw", "smooth"] {
            records.push(figure(
                "features",
                &Route::Features,
                case,
                smoothing,
                &square,
                &unit,
                &config,
            ));
        }
    }

    // The graded pipeline on its intended 10 x 10 domain (sigma = 1 + 0.5 x).
    let pipeline = MeshConfig::new(Route::Pipeline);
    for smoothing in ["raw", "smooth"] {
        records.push(figure(
            "pipeline",
            &Route::Pipeline,
            "graded-10x10",
            smoothing,
            &square,
            &unit,
            &pipeline,
        ));
    }

    let mut manifest = String::from(
        "group,route,mesh,smoothing,status,quads,triangles,min_sj,min_angle,gate,file\n",
    );
    for record in &records {
        manifest.push_str(&record.csv());
    }
    fs::write(format!("{ROOT}/manifest.csv"), manifest).unwrap();

    write_index(&records);
    println!("\n{} cases -> {ROOT}/manifest.csv", records.len());
}

fn write_index(records: &[Record]) {
    let mut text = String::from(
        "# Mesh gallery\n\n\
        Every studied mesh, drawn by `examples/draw_gallery.rs`. Cells are\n\
        filled by the gate margin `min(angle/30, sj/0.2)` (green clears the\n\
        gate, yellow/orange/red fails, magenta is inverted, triangles are dark\n\
        red); the title carries the measured numbers.\n\n\
        `raw` disables the shared post-smoother and, for `pipeline`, its\n\
        internal Section 2.2.10 pass. `smooth` is the shipping default\n\
        (smoothing on). A rejected smoothing candidate is labelled in the\n\
        title and the raw mesh is drawn.\n\n\
        ## Layout\n\n\
        - `corpus/<route>/<part>-<raw|smooth>.png` - 11 routes x 7 parts\n\
        - `features/<case>-<raw|smooth>.png` - through, stub slot, cross, through+terminating, stub channel\n\
        - `pipeline/graded-10x10-<raw|smooth>.png` - the graded QGP demo\n\
        - `manifest.csv` - one row per case with the statistics and gate verdict\n\n\
        ## Corpus routes\n\n\
        | Route | Figures | PASS (smoothed) |\n\
        |---|---:|---:|\n",
    );
    for route in ROUTES {
        let rows: Vec<&Record> = records
            .iter()
            .filter(|r| r.route == route.name() && r.group == "corpus")
            .collect();
        let smooth_pass = rows
            .iter()
            .filter(|r| r.smoothing == "smooth" && r.passes)
            .count();
        let _ = writeln!(
            text,
            "| {} | {} | {}/7 |",
            route.name(),
            rows.len(),
            smooth_pass
        );
    }
    text.push_str(
        "\nThe `features` route errors on the corpus parts (it needs feature\n\
        polylines); its five demo cases are under `features/`. The `pipeline`\n\
        route is also drawn on the corpus parts for completeness (it fails on\n\
        `ring`, the known cross-field degenerate triangle, so that row has no\n\
        figure); its intended case is `pipeline/graded-10x10`.\n",
    );
    fs::write(format!("{ROOT}/INDEX.md"), text).unwrap();
}
