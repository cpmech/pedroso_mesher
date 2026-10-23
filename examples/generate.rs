//! Generate a mesh with any route from a `MeshConfig`.
//!
//! ```text
//! cargo run --release --offline --example generate -- [route] [part] [--config in.json] [--write-config out.json] [--json stats.json] [--no-smooth]
//! ```
//!
//! `route` is one of `qmorph, paving, bipartite, quadtree, decompose, hybrid,
//! crossfield, untangle, l1`; the default is `quadtree` on `plate-hole-2mm`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use pedroso_mesher::config::{generate, MeshConfig, Route};
use pedroso_mesher::two_d::{Domain, Sizing};

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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut config = MeshConfig::default();
    let mut part_name = "plate-hole-2mm".to_string();
    let mut json_out: Option<String> = None;
    let mut config_out: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--config" => {
                i += 1;
                let text = std::fs::read_to_string(&args[i]).unwrap();
                config = MeshConfig::from_json(&text).unwrap();
            }
            "--json" => {
                i += 1;
                json_out = Some(args[i].clone());
            }
            "--no-smooth" => config.smoothing.enabled = false,
            "--write-config" => {
                i += 1;
                config_out = Some(args[i].clone());
            }
            other => {
                if let Some(route) = Route::ALL.iter().copied().find(|r| r.name() == other) {
                    config.route = route;
                } else {
                    part_name = other.to_string();
                }
            }
        }
        i += 1;
    }
    if let Some(path) = config_out {
        std::fs::write(&path, config.to_json()).unwrap();
        println!("wrote configuration {path}");
    }
    let (domain, size) = part(&part_name);
    let sizing = Sizing::uniform(size).unwrap();
    let out = match generate(&domain, &sizing, &config) {
        Ok(out) => out,
        Err(e) => {
            println!("{} on {}: error: {e}", config.route.name(), part_name);
            return;
        }
    };
    println!(
        "{} on {}: {} in {:.2}s",
        config.route.name(),
        part_name,
        out.statistics.line(),
        out.seconds
    );
    println!("generator only: {}", out.before.line());
    if let Some(s) = &out.smoothing {
        println!("smoothing accepted: {}", s.accepted);
    }
    if let Some(path) = json_out {
        std::fs::write(&path, out.statistics.to_json()).unwrap();
        println!("wrote {path}");
    }
}
