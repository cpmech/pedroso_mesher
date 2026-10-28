//! Reproduce the Phase A raw/post baselines or run the adversarial corpus.
//!
//! ```text
//! cargo run --release --offline --example phase_a -- --output data/baselines/phase-a.csv
//! cargo run --release --offline --example phase_a -- --adversarial --output data/baselines/phase-a-adversarial.csv
//! ```
//!
//! Timing is deliberately excluded: every other field is deterministic. A route
//! error is a record, not an abort. `generate` is used rather than the strict
//! adapter so raw meshes below the quality gate remain measurable.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::fmt::Write as _;

use pedroso_mesher::config::{generate, MeshConfig, Route};
use pedroso_mesher::mesh::Mesh;
use pedroso_mesher::stats::MeshStatistics;
use pedroso_mesher::two_d::{Domain, Sizing, TwoDError};
use pedroso_mesher::verify;

const BASELINE_PARTS: [&str; 7] = [
    "plate-hole-4mm",
    "plate-hole-2mm",
    "plate-hole-1mm",
    "plate-hole-0.6mm",
    "l-bracket",
    "ring",
    "stepped",
];
const ADVERSARIAL_PARTS: [&str; 7] = [
    "multiple-holes",
    "acute-corner",
    "small-feature",
    "translated-scaled",
    "repeated-boundary-node",
    "self-intersecting",
    "outside-hole",
];
const ADVERSARIAL_ROUTES: [Route; 3] = [Route::Quadtree, Route::L1, Route::Sized];

fn arc(r: f64, a0: f64, a1: f64, n: usize) -> Vec<[f64; 2]> {
    (0..=n)
        .map(|i| {
            let a = a0 + (a1 - a0) * i as f64 / n as f64;
            [r * a.cos(), r * a.sin()]
        })
        .collect()
}

fn transformed_loop(points: &[[f64; 2]], scale: f64, offset: [f64; 2]) -> Vec<[f64; 2]> {
    points
        .iter()
        .map(|p| [offset[0] + scale * p[0], offset[1] + scale * p[1]])
        .collect()
}

fn part(name: &str) -> Result<(Domain, f64), TwoDError> {
    match name {
        "plate-hole-4mm" => Ok((Domain::plate_with_hole(100.0, 60.0, 10.0, 16)?, 4.0)),
        "plate-hole-2mm" => Ok((Domain::plate_with_hole(100.0, 60.0, 10.0, 24)?, 2.0)),
        "plate-hole-1mm" => Ok((Domain::plate_with_hole(100.0, 60.0, 10.0, 48)?, 1.0)),
        "plate-hole-0.6mm" => Ok((Domain::plate_with_hole(100.0, 60.0, 10.0, 64)?, 0.6)),
        "l-bracket" => Ok((
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
            )?,
            2.0,
        )),
        "ring" => {
            let mut points = arc(20.0, 0.0, std::f64::consts::FRAC_PI_2, 24);
            points.extend(arc(10.0, std::f64::consts::FRAC_PI_2, 0.0, 12));
            Ok((Domain::new(points, vec![])?, 1.0))
        }
        "stepped" => Ok((
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
            )?,
            1.0,
        )),
        "multiple-holes" => Ok((
            Domain::new(
                vec![[0.0, 0.0], [30.0, 0.0], [30.0, 16.0], [0.0, 16.0]],
                vec![
                    vec![[4.0, 5.0], [4.0, 9.0], [8.0, 9.0], [8.0, 5.0]],
                    vec![[13.0, 3.0], [13.0, 7.0], [17.0, 7.0], [17.0, 3.0]],
                    vec![[22.0, 8.0], [22.0, 12.0], [26.0, 12.0], [26.0, 8.0]],
                ],
            )?,
            1.0,
        )),
        "acute-corner" => Ok((
            Domain::new(vec![[0.0, 0.0], [12.0, 0.0], [1.0, 1.0]], vec![])?,
            0.5,
        )),
        "small-feature" => Ok((
            Domain::new(
                vec![[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]],
                vec![vec![[9.9, 9.9], [9.9, 10.1], [10.1, 10.1], [10.1, 9.9]]],
            )?,
            1.0,
        )),
        "translated-scaled" => {
            let scale = 1.0e6;
            let offset = [-3.0e7, 2.0e7];
            let outer = transformed_loop(
                &[
                    [0.0, 0.0],
                    [8.0, 0.0],
                    [8.0, 2.0],
                    [2.0, 2.0],
                    [2.0, 6.0],
                    [0.0, 6.0],
                ],
                scale,
                offset,
            );
            Ok((Domain::new(outer, vec![])?, 2.0 * scale))
        }
        "repeated-boundary-node" => Ok((
            Domain::new(
                vec![[0.0, 0.0], [4.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
                vec![],
            )?,
            1.0,
        )),
        "self-intersecting" => Ok((
            Domain::new(vec![[0.0, 0.0], [4.0, 4.0], [0.0, 4.0], [4.0, 0.0]], vec![])?,
            1.0,
        )),
        "outside-hole" => Ok((
            Domain::new(
                vec![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
                vec![vec![[5.0, 5.0], [5.0, 6.0], [6.0, 6.0], [6.0, 5.0]]],
            )?,
            1.0,
        )),
        other => panic!("unknown part {other}"),
    }
}

fn mesh_hash(mesh: &Mesh) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |value: u64| {
        hash ^= value;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    };
    for point in &mesh.points {
        for coordinate in point {
            eat(coordinate.to_bits());
        }
    }
    for cell in &mesh.cells {
        eat(cell.kind as u64);
        eat(cell.nodes.len() as u64);
        for &node in &cell.nodes {
            eat(node as u64);
        }
    }
    format!("{hash:016x}")
}

fn csv_field(text: &str) -> String {
    format!("\"{}\"", text.replace('"', "\"\""))
}

fn error_row(output: &mut String, route: Route, name: &str, stage: &str, error: &str) {
    writeln!(
        output,
        "{},{},{stage},error,,,,,,,,,,,,,,{}",
        route.name(),
        name,
        csv_field(error)
    )
    .unwrap();
}

fn mesh_row(
    output: &mut String,
    route: Route,
    name: &str,
    stage: &str,
    domain: &Domain,
    mesh: &Mesh,
    stats: &MeshStatistics,
) {
    let geometry = verify::geometry(domain, mesh);
    let geometry_status = match &geometry {
        Ok(report) if report.passes() => "pass".to_string(),
        Ok(_) => "fail".to_string(),
        Err(_) => "error".to_string(),
    };
    let detail = match &geometry {
        Ok(report) => report.reason().or_else(|| stats.gate.reason()),
        Err(error) => Some(error.to_string()),
    };
    writeln!(
        output,
        "{},{},{stage},ok,{},{},{},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{},{},{},{}",
        route.name(),
        name,
        mesh_hash(mesh),
        stats.quads,
        stats.triangles,
        stats.min_scaled_jacobian,
        stats.min_corner_angle_deg,
        stats.aspect_ratio.p95,
        stats.aspect_ratio.max,
        stats.interior_valence4_share,
        stats.boundary_offset,
        stats.corner_offset,
        if stats.passes { "pass" } else { "fail" },
        if stats.conforming { "pass" } else { "fail" },
        geometry_status,
        csv_field(detail.as_deref().unwrap_or(""))
    )
    .unwrap();
}

fn main() {
    let mut adversarial = false;
    let mut path = "/tmp/pedroso_mesher/phase-a/baselines.csv".to_string();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--adversarial" => adversarial = true,
            "--output" => {
                i += 1;
                path = args[i].clone();
            }
            other => panic!("unknown argument {other}"),
        }
        i += 1;
    }
    let routes: &[Route] = if adversarial {
        &ADVERSARIAL_ROUTES
    } else {
        &Route::ALL
    };
    let names: &[&str] = if adversarial {
        &ADVERSARIAL_PARTS
    } else {
        &BASELINE_PARTS
    };
    let mut output = "route,part,stage,status,mesh_hash,quads,triangles,min_scaled_jacobian,min_corner_angle_deg,aspect_p95,aspect_max,interior_valence4_share,boundary_offset,corner_offset,gate,conforming,geometry,detail\n".to_string();
    let mut records = 0usize;
    for &route in routes {
        for &name in names {
            let (domain, size) = part(name).unwrap();
            let sizing = Sizing::uniform(size).unwrap();
            for stage in ["raw", "post"] {
                let mut config = MeshConfig::new(route);
                config.smoothing.enabled = stage == "post";
                match generate(&domain, &sizing, &config) {
                    Ok(out) => mesh_row(
                        &mut output,
                        route,
                        name,
                        stage,
                        &domain,
                        &out.mesh,
                        &out.statistics,
                    ),
                    Err(error) => error_row(&mut output, route, name, stage, &error),
                }
                records += 1;
            }
            println!("{} {name}: recorded raw/post", route.name());
        }
    }
    if let Some(parent) = std::path::Path::new(&path).parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(&path, output).unwrap();
    println!("wrote {path} ({records} records)");
}
