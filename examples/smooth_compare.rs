//! Measure every route before and after the shared boundary-preserving smoother,
//! draw both states, and write the statistics as JSON.
//!
//! `cargo run --release --offline --example smooth_compare [part ...]`
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use pedroso_mesher::draw::{self, MeshDrawOptions};
use pedroso_mesher::mesh::Mesh;
use pedroso_mesher::smooth::{self, SmoothOptions};
use pedroso_mesher::stats::MeshStatistics;
use pedroso_mesher::two_d::{Domain, Sizing};

type Route = fn(&Domain, &Sizing) -> Result<Mesh, String>;

fn routes() -> Vec<(&'static str, Route)> {
    vec![
        ("paving", pedroso_mesher::alt_paving::mesh),
        ("bipartite", pedroso_mesher::alt_bipartite::mesh),
        ("quadtree", pedroso_mesher::alt_quadtree::mesh),
        ("decompose", pedroso_mesher::alt_decompose::mesh),
        ("hybrid", pedroso_mesher::alt_hybrid::mesh),
        ("crossfield", pedroso_mesher::alt_crossfield::mesh),
        ("untangle", pedroso_mesher::alt_untangle::mesh),
        ("l1", pedroso_mesher::alt_l1::mesh),
    ]
}

fn part(name: &str) -> (Domain, f64) {
    match name {
        "plate-hole-4mm" => (Domain::plate_with_hole(100.0, 60.0, 10.0, 16).unwrap(), 4.0),
        "plate-hole-2mm" => (Domain::plate_with_hole(100.0, 60.0, 10.0, 24).unwrap(), 2.0),
        "plate-hole-1mm" => (Domain::plate_with_hole(100.0, 60.0, 10.0, 48).unwrap(), 1.0),
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

#[derive(serde::Serialize)]
struct Record {
    route: String,
    part: String,
    before: MeshStatistics,
    after: MeshStatistics,
    accepted: bool,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let parts = if args.is_empty() {
        vec!["plate-hole-2mm", "l-bracket", "ring", "stepped"]
    } else {
        args.iter().map(String::as_str).collect()
    };
    std::fs::create_dir_all("figures/smooth").unwrap();
    let options = SmoothOptions {
        // measure the smoother itself; the route output is the "before" state
        reject_if_worse: false,
        improve_angles: std::env::var("NO_SWAPS").is_err(),
        perturb_rounds: std::env::var("PERTURB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0),
        ..SmoothOptions::default()
    };
    let mut records = Vec::new();
    let mut pass_before = vec![0usize; routes().len()];
    let mut pass_after = vec![0usize; routes().len()];
    for (r, (name, route)) in routes().iter().enumerate() {
        for &part_name in &parts {
            let (domain, size) = part(part_name);
            let sizing = Sizing::uniform(size).unwrap();
            let Ok(mesh) = route(&domain, &sizing) else {
                println!("{name:11} {part_name:15} route returned Err");
                continue;
            };
            let started = std::time::Instant::now();
            let outcome = smooth::smooth(&domain, &mesh, size, &options);
            let secs = started.elapsed().as_secs_f64();
            if outcome.before.passes {
                pass_before[r] += 1;
            }
            if outcome.after.passes {
                pass_after[r] += 1;
            }
            records.push(Record {
                route: name.to_string(),
                part: part_name.to_string(),
                before: outcome.before.clone(),
                after: outcome.after.clone(),
                accepted: outcome.accepted,
            });
            println!(
                "{name:11} {part_name:15} before [{}]  after [{}]  smooth {secs:.2}s",
                outcome.before.line(),
                outcome.after.line()
            );
            for (state, mesh) in [("before", &mesh), ("after", &outcome.smoothed)] {
                let stats = if state == "before" {
                    &outcome.before
                } else {
                    &outcome.after
                };
                let title = format!(
                    "{part_name} — {name} ({state}): {} quads, {} tri, SJ {:.3}, {:.1}°, v4 {:.1}%",
                    stats.quads,
                    stats.triangles,
                    stats.min_scaled_jacobian,
                    stats.min_corner_angle_deg,
                    100.0 * stats.interior_valence4_share
                );
                let opts = MeshDrawOptions {
                    title,
                    color_by_gate_margin: true,
                    mark_nodes: false,
                    figure_inches: (9.0, 6.0),
                };
                draw::draw_domain_with_mesh(
                    &domain,
                    mesh,
                    &format!("figures/smooth/{name}-{part_name}-{state}.png"),
                    &opts,
                )
                .unwrap();
            }
        }
    }
    let path = "figures/smooth/stats.json";
    std::fs::write(path, serde_json::to_string_pretty(&records).unwrap()).unwrap();
    println!("\nwrote {path} ({} records)", records.len());
    println!("\n| route | pass before | pass after |");
    println!("|---|---:|---:|");
    for (r, (name, _)) in routes().iter().enumerate() {
        println!(
            "| {name} | {}/{} | {}/{} |",
            pass_before[r],
            parts.len(),
            pass_after[r],
            parts.len()
        );
    }
}
