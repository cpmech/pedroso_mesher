//! Profile the shape of a quad mesh: angles, Jacobians, valences, aspect.
//!
//! The gate only reports minima; this prints the distribution, which is what
//! "the mesh looks terrible" means in numbers.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use pedroso_mesher::mesh::{Kind, Mesh};
use pedroso_mesher::quad::{self, QuadError};
use pedroso_mesher::quality;
use pedroso_mesher::two_d::{mesh_2d, Domain, Sizing};

fn profile(name: &str, mesh: &Mesh) {
    let quads = mesh.cells.iter().filter(|c| c.kind == Kind::Quad4).count();
    let tris = mesh.cells.iter().filter(|c| c.kind == Kind::Tri3).count();
    let mut angles = Vec::new();
    let mut sjs = Vec::new();
    let mut aspect = Vec::new();
    for cell in &mesh.cells {
        if cell.kind != Kind::Quad4 {
            continue;
        }
        let a = quality::quad_angles(cell, &mesh.points);
        angles.push(a.iter().copied().fold(f64::INFINITY, f64::min));
        sjs.push(
            quality::scaled_jacobian(cell, &mesh.points)
                .into_iter()
                .fold(f64::INFINITY, f64::min),
        );
        aspect.push(quality::aspect_ratio(cell, &mesh.points));
    }
    if angles.is_empty() {
        println!("{name:14} no quads");
        return;
    }
    angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sjs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    aspect.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = angles.len();
    let low30 = angles.iter().filter(|a| **a < 30.0).count();
    let low20 = angles.iter().filter(|a| **a < 20.0).count();
    let lowsj = sjs.iter().filter(|s| **s < 0.2).count();
    // interior valence histogram
    let mut edges: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
    for cell in &mesh.cells {
        for i in 0..cell.nodes.len() {
            let a = cell.nodes[i];
            let b = cell.nodes[(i + 1) % cell.nodes.len()];
            *edges.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    let mut on_boundary = vec![false; mesh.points.len()];
    for ((a, b), k) in &edges {
        if *k == 1 {
            on_boundary[*a] = true;
            on_boundary[*b] = true;
        }
    }
    let mut valence = vec![0usize; mesh.points.len()];
    for cell in &mesh.cells {
        if cell.kind == Kind::Quad4 {
            for &v in &cell.nodes {
                valence[v] += 1;
            }
        }
    }
    let interior: Vec<usize> = (0..mesh.points.len())
        .filter(|&v| !on_boundary[v] && valence[v] > 0)
        .collect();
    let v4 = interior.iter().filter(|&&v| valence[v] == 4).count();
    let vshare = 100.0 * v4 as f64 / interior.len().max(1) as f64;
    println!(
        "{name:14} q {:5} tri {:3} | ang min {:5.1} p10 {:5.1} med {:5.1} | <30° {:4} ({:4.1}%) <20° {:3} | SJ min {:.3} med {:.3} <0.2 {:3} | aspect med {:4.1} p95 {:5.1} max {:.0} | v4 {:5.1}%",
        quads,
        tris,
        angles[0],
        angles[n / 10],
        angles[n / 2],
        low30,
        100.0 * low30 as f64 / n as f64,
        low20,
        sjs[0],
        sjs[sjs.len() / 2],
        lowsj,
        aspect[aspect.len() / 2],
        aspect[(aspect.len() * 95) / 100],
        aspect[aspect.len() - 1],
        vshare,
    );
}

fn cases() -> Vec<(&'static str, Domain, f64)> {
    vec![
        (
            "plate-hole 4",
            Domain::plate_with_hole(100.0, 60.0, 10.0, 16).unwrap(),
            4.0,
        ),
        (
            "plate-hole 2",
            Domain::plate_with_hole(100.0, 60.0, 10.0, 24).unwrap(),
            2.0,
        ),
        (
            "plate-hole 1",
            Domain::plate_with_hole(100.0, 60.0, 10.0, 48).unwrap(),
            1.0,
        ),
        (
            "l-bracket",
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
        (
            "ring",
            {
                let arc = |r: f64, a0: f64, a1: f64, n: usize| -> Vec<[f64; 2]> {
                    (0..=n)
                        .map(|i| {
                            let a = a0 + (a1 - a0) * i as f64 / n as f64;
                            [r * a.cos(), r * a.sin()]
                        })
                        .collect()
                };
                let mut outer = arc(20.0, 0.0, std::f64::consts::FRAC_PI_2, 24);
                let inner = arc(10.0, std::f64::consts::FRAC_PI_2, 0.0, 12);
                outer.extend(inner);
                Domain::new(outer, vec![]).unwrap()
            },
            1.0,
        ),
        (
            "stepped",
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
        (
            "perf 0.6",
            Domain::plate_with_hole(100.0, 60.0, 10.0, 64).unwrap(),
            0.6,
        ),
    ]
}

fn main() {
    println!("--- Q-Morph route (product) ---");
    for (name, d, size) in cases() {
        match quad::mesh_quads(&d, &Sizing::uniform(size).unwrap()) {
            Ok(q) => profile(name, &q.mesh),
            Err(QuadError::Unsupported(u)) => profile(name, &u.mesh),
            Err(e) => println!("{name:14} error: {e}"),
        }
    }
    println!("--- matching route (default product) ---");
    for (name, d, size) in cases() {
        let r = mesh_2d(&d, &Sizing::uniform(size).unwrap()).unwrap();
        profile(name, &r.mesh);
    }
}
