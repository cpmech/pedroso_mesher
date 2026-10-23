//! Draw every studied mesh: each corpus route on each corpus part, raw and
//! smoothed, the feature demos, the graded pipeline, and the Simons 2024
//! thesis meshes (stitched QGP cases, the 2 x 2 QGP layout, graded tiles).
//!
//! ```bash
//! cargo run --release --offline --example draw_gallery            # everything
//! cargo run --release --offline --example draw_gallery -- --only thesis
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
//! thesis/<case>.png
//! manifest.csv   one row per generated case (or per route error)
//! INDEX.md       what the gallery contains and how to read it
//! ```
//!
//! "raw" disables the shared post-smoother (`SmoothOptions::enabled = false`)
//! and, for the pipeline route, its internal Section 2.2.10 pass as well; the
//! generator is otherwise unchanged. The thesis meshes are drawn as built
//! (`as-built` in the manifest): the stitched cases already carry their channel
//! and are not re-smoothed. The cell fill is the gate margin
//! `min(angle/30, sj/0.2)` (green clears the gate, yellow/orange/red fails,
//! magenta is inverted, triangles are dark red), so every figure is readable in
//! the gate's own terms.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::fmt::Write as _;
use std::fs;

use pedroso_mesher::config::{generate, MeshConfig, Route};
use pedroso_mesher::draw::{self, MeshDrawOptions};
use pedroso_mesher::mesh::{Cell, Mesh};
use pedroso_mesher::simons_features::TerminatingMode;
use pedroso_mesher::simons_graded;
use pedroso_mesher::simons_qgp::{self, Region};
use pedroso_mesher::simons_regions;
use pedroso_mesher::two_d::{Domain, Sizing};
use pedroso_mesher::verify;

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

fn slug(text: &str) -> String {
    let mut out = String::new();
    let mut dash = false;
    for character in text.chars() {
        if character.is_ascii_alphanumeric() {
            out.push(character.to_ascii_lowercase());
            dash = false;
        } else if !dash && !out.is_empty() {
            out.push('-');
            dash = true;
        }
    }
    out.trim_end_matches('-').to_string()
}

/// Generate, draw and describe one configured case.
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

/// Draw an already-built mesh (a thesis case) and describe it.
fn draw_mesh_case(
    group: &str,
    route: &str,
    mesh_name: &str,
    note: &str,
    mesh: &Mesh,
    domain: Option<&Domain>,
) -> Record {
    let gate = verify::gate(mesh);
    let title = format!(
        "{} - {}{}  |  {} q, {} tri, SJ {:.3}, {:.1} deg, {}",
        route,
        mesh_name,
        note,
        gate.quads,
        gate.triangles,
        gate.min_scaled_jacobian,
        gate.min_corner_angle_deg,
        if gate.passes() { "PASS" } else { "FAIL" }
    );
    let file = format!("{ROOT}/{group}/{mesh_name}.png");
    fs::create_dir_all(format!("{ROOT}/{group}")).unwrap();
    let options = MeshDrawOptions {
        title,
        figure_inches: (7.0, 5.0),
        ..MeshDrawOptions::default()
    };
    let drawn = match domain {
        Some(domain) => draw::draw_domain_with_mesh(domain, mesh, &file, &options),
        None => draw::draw_mesh(mesh, &file, &options),
    };
    if let Err(error) = drawn {
        println!("{file}: draw error: {error}");
        return Record::error(group, route, mesh_name, "as-built", error);
    }
    let script = file.replace(".png", ".py");
    let _ = fs::remove_file(script);
    println!(
        "{file}: {} q, {} tri, SJ {:.3}, {:.1} deg, {}",
        gate.quads,
        gate.triangles,
        gate.min_scaled_jacobian,
        gate.min_corner_angle_deg,
        if gate.passes() { "PASS" } else { "FAIL" }
    );
    Record {
        group: group.to_string(),
        route: route.to_string(),
        mesh: mesh_name.to_string(),
        smoothing: "as-built".to_string(),
        status: "ok".to_string(),
        quads: gate.quads,
        triangles: gate.triangles,
        min_sj: gate.min_scaled_jacobian,
        min_angle: gate.min_corner_angle_deg,
        passes: gate.passes(),
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

fn sigma_linear(x: f64, _y: f64) -> f64 {
    1.0 + 0.5 * x
}

fn sigma_2d(x: f64, y: f64) -> f64 {
    1.0 + 0.3 * x + 0.3 * y
}

fn sigma_hole(x: f64, y: f64) -> f64 {
    1.0 + 0.2 * x + 0.1 * y
}

/// The 2 x 2 QGP layout of `examples/thesis_demo.rs`: four regions on a 10 x 10
/// square with `sigma = 1 + 0.5 x`, balanced counts and bilinear Coons patches.
fn qgp_tiles_2x2() -> Result<Mesh, String> {
    fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
    }
    fn ideal_count(a: [f64; 2], b: [f64; 2]) -> f64 {
        let n = 16;
        let mut sum = 0.0;
        for i in 0..=n {
            let t = i as f64 / n as f64;
            let p = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
            let w = if i == 0 || i == n { 1.0 } else { 2.0 };
            sum += w / sigma_linear(p[0], p[1]);
        }
        sum * (dist(a, b) / 2.0 / n as f64)
    }
    let corner = |i: usize, j: usize| [5.0 * i as f64, 5.0 * j as f64];
    let nh = 3;
    let nv = 3;
    let h_id = |i: usize, j: usize| j * (nh - 1) + i;
    let v_id = |i: usize, j: usize| (nh * (nv - 1)) + j * nv + i;
    let mut edge_ideal = Vec::new();
    for j in 0..3 {
        for i in 0..(nh - 1) {
            edge_ideal.push(ideal_count(corner(i, j), corner(i + 1, j)));
        }
    }
    for j in 0..(nv - 1) {
        for i in 0..nv {
            edge_ideal.push(ideal_count(corner(i, j), corner(i, j + 1)));
        }
    }
    let mut regions = Vec::new();
    let mut corners = Vec::new();
    for j in 0..2 {
        for i in 0..2 {
            regions.push(Region::new([
                vec![h_id(i, j)],
                vec![v_id(i + 1, j)],
                vec![h_id(i, j + 1)],
                vec![v_id(i, j)],
            ]));
            corners.push([
                corner(i, j),
                corner(i + 1, j),
                corner(i + 1, j + 1),
                corner(i, j + 1),
            ]);
        }
    }
    let counts = simons_qgp::solve_counts(&regions, &edge_ideal)?;
    let mut points: Vec<[f64; 3]> = Vec::new();
    let mut cells: Vec<Cell> = Vec::new();
    for (region, region_corners) in regions.iter().zip(corners.iter()) {
        let sides = [
            region.sides[0][0],
            region.sides[1][0],
            region.sides[2][0],
            region.sides[3][0],
        ];
        let nu = counts[sides[0]].max(1);
        let nv = counts[sides[3]].max(1);
        let quads = simons_qgp::coons(region_corners, nu, nv);
        let mut index = std::collections::HashMap::new();
        for quad in quads {
            let mut ids = [0usize; 4];
            for (k, p) in quad.iter().enumerate() {
                let key = ((p[0] * 1.0e9).round() as i64, (p[1] * 1.0e9).round() as i64);
                let id = *index.entry(key).or_insert_with(|| {
                    points.push([p[0], p[1], 0.0]);
                    points.len() - 1
                });
                ids[k] = id;
            }
            cells.push(Cell::quad4(ids));
        }
    }
    Ok(Mesh::new(points, cells))
}

fn draw_thesis(records: &mut Vec<Record>) {
    // Chapter 5 stitch: the QGP interior plus the Chapter 2 channel. The
    // `StitchCase` carries the stitched mesh itself.
    match simons_regions::stitch_demo() {
        Ok(numbers) => {
            for case in [&numbers.rectangle, &numbers.annulus, &numbers.nautilus] {
                let name = slug(case.name);
                let note = format!(", coverage {:.3}", case.coverage);
                records.push(draw_mesh_case(
                    "thesis", "stitch", &name, &note, &case.mesh, None,
                ));
            }
        }
        Err(error) => {
            println!("thesis/stitch: error: {error}");
            records.push(Record::error(
                "thesis",
                "stitch",
                "stitch-demo",
                "as-built",
                &error,
            ));
        }
    }

    // The 2 x 2 QGP layout of `thesis_demo` (Ch5 counts + Coons).
    match qgp_tiles_2x2() {
        Ok(mesh) => {
            let domain = Domain::rectangle(10.0, 10.0).unwrap();
            records.push(draw_mesh_case(
                "thesis",
                "qgp",
                "qgp-2x2",
                ", sigma = 1 + 0.5x",
                &mesh,
                Some(&domain),
            ));
        }
        Err(error) => {
            println!("thesis/qgp: error: {error}");
            records.push(Record::error(
                "thesis", "qgp", "qgp-2x2", "as-built", &error,
            ));
        }
    }

    // The graded tile-grid demos of `simons_graded`.
    let square = Domain::rectangle(10.0, 10.0).unwrap();
    let plate = Domain::new(
        vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
        vec![vec![[4.0, 4.0], [6.0, 4.0], [6.0, 6.0], [4.0, 6.0]]],
    )
    .unwrap();
    let cases: [(
        &str,
        &Domain,
        simons_graded::Options,
        fn(f64, f64) -> f64,
        &str,
    ); 4] = [
        (
            "graded-linear",
            &square,
            simons_graded::Options::default(),
            sigma_linear,
            ", sigma = 1 + 0.5x",
        ),
        (
            "graded-2d",
            &square,
            simons_graded::Options::default(),
            sigma_2d,
            ", sigma = 1 + 0.3x + 0.3y",
        ),
        (
            "graded-2d-8x8",
            &square,
            simons_graded::Options { tiles: (8, 8) },
            sigma_2d,
            ", requested 8x8",
        ),
        (
            "graded-hole",
            &plate,
            simons_graded::Options::default(),
            sigma_hole,
            ", rectangular hole",
        ),
    ];
    for (name, domain, options, sigma, note) in cases {
        match simons_graded::mesh(domain, &options, &sigma) {
            Ok((mesh, _info)) => records.push(draw_mesh_case(
                "thesis",
                "graded",
                name,
                note,
                &mesh,
                Some(domain),
            )),
            Err(error) => {
                println!("thesis/graded/{name}: error: {error}");
                records.push(Record::error("thesis", "graded", name, "as-built", &error));
            }
        }
    }
}

fn main() {
    fs::create_dir_all(ROOT).unwrap();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let only = args
        .iter()
        .position(|a| a == "--only")
        .and_then(|index| args.get(index + 1))
        .cloned();
    let full = only.is_none();
    let mut records = Vec::new();

    if full {
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
    }

    if full || only.as_deref() == Some("thesis") {
        draw_thesis(&mut records);
    }

    // Keep the other groups' rows when only one group is regenerated.
    let manifest_path = format!("{ROOT}/manifest.csv");
    let existing = if full {
        String::new()
    } else {
        fs::read_to_string(&manifest_path).unwrap_or_default()
    };
    let mut manifest = String::from(
        "group,route,mesh,smoothing,status,quads,triangles,min_sj,min_angle,gate,file\n",
    );
    if !full {
        let regenerated = only.as_deref().unwrap_or("thesis");
        for line in existing.lines().skip(1) {
            if line.trim().is_empty() || line.starts_with(&format!("{regenerated},")) {
                continue;
            }
            manifest.push_str(line);
            manifest.push('\n');
        }
    }
    for record in &records {
        manifest.push_str(&record.csv());
    }
    fs::write(&manifest_path, &manifest).unwrap();
    write_index(&manifest);
    println!("\n{} new case(s) -> {manifest_path}", records.len());
}

/// One parsed manifest row, for the index tables.
struct Row {
    group: String,
    route: String,
    mesh: String,
    smoothing: String,
    quads: usize,
    triangles: usize,
    min_sj: f64,
    min_angle: f64,
    gate: String,
}

fn parse_manifest(manifest: &str) -> Vec<Row> {
    let mut rows = Vec::new();
    for line in manifest.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 10 {
            continue;
        }
        rows.push(Row {
            group: fields[0].to_string(),
            route: fields[1].to_string(),
            mesh: fields[2].to_string(),
            smoothing: fields[3].to_string(),
            quads: fields[5].parse().unwrap_or(0),
            triangles: fields[6].parse().unwrap_or(0),
            min_sj: fields[7].parse().unwrap_or(0.0),
            min_angle: fields[8].parse().unwrap_or(0.0),
            gate: fields[9].to_string(),
        });
    }
    rows
}

fn write_index(manifest: &str) {
    let rows = parse_manifest(manifest);
    let mut text = String::from(
        "# Mesh gallery\n\n\
        Every studied mesh, drawn by `examples/draw_gallery.rs`. Cells are\n\
        filled by the gate margin `min(angle/30, sj/0.2)` (green clears the\n\
        gate, yellow/orange/red fails, magenta is inverted, triangles are dark\n\
        red); the title carries the measured numbers.\n\n\
        `raw` disables the shared post-smoother and, for `pipeline`, its\n\
        internal Section 2.2.10 pass. `smooth` is the shipping default\n\
        (smoothing on). A rejected smoothing candidate is labelled in the\n\
        title and the raw mesh is drawn. Thesis cases are `as-built` (the\n\
        stitched meshes already carry their channel and are not re-smoothed).\n\n\
        ## Layout\n\n\
        - `corpus/<route>/<part>-<raw|smooth>.png` - 11 routes x 7 parts\n\
        - `features/<case>-<raw|smooth>.png` - through, stub slot, cross, through+terminating, stub channel\n\
        - `pipeline/graded-10x10-<raw|smooth>.png` - the graded QGP pipeline\n\
        - `thesis/<case>.png` - stitched rectangle/annulus/nautilus, the 2x2 QGP layout, and the graded tile demos\n\
        - `manifest.csv` - one row per case with the statistics and gate verdict\n\n\
        ## Corpus routes\n\n\
        | Route | Cases | PASS (smoothed) |\n\
        |---|---:|---:|\n",
    );
    for route in ROUTES {
        let route_rows: Vec<&Row> = rows
            .iter()
            .filter(|r| r.group == "corpus" && r.route == route.name())
            .collect();
        let smooth_pass = route_rows
            .iter()
            .filter(|r| r.smoothing == "smooth" && r.gate == "PASS")
            .count();
        let _ = writeln!(
            text,
            "| {} | {} | {}/7 |",
            route.name(),
            route_rows.len(),
            smooth_pass
        );
    }
    text.push_str(
        "\nThe `features` route errors on the corpus parts (it needs feature\n\
        polylines); its five demo cases are under `features/`. The `pipeline`\n\
        route is also drawn on the corpus parts for completeness (it fails on\n\
        `ring`, the known cross-field degenerate triangle, so that row has no\n\
        figure); its intended case is `pipeline/graded-10x10`.\n\n\
        ## Thesis cases\n\n\
        | Route | Case | Quads | Tri | SJ | Angle | Gate |\n\
        |---|---|---:|---:|---:|---:|---|\n",
    );
    for row in rows.iter().filter(|r| r.group == "thesis") {
        let _ = writeln!(
            text,
            "| {} | {} | {} | {} | {:.3} | {:.1} | {} |",
            row.route, row.mesh, row.quads, row.triangles, row.min_sj, row.min_angle, row.gate
        );
    }
    fs::write(format!("{ROOT}/INDEX.md"), text).unwrap();
}
