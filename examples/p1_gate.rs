//! P1 gate: quad ratio, quality and timing on the four planar parts.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use pedroso_mesher::quality::{self, quad_angles};
use pedroso_mesher::two_d::{mesh_2d, Domain, Sizing};

fn arc(cx: f64, cy: f64, r: f64, a0: f64, a1: f64, n: usize) -> Vec<[f64; 2]> {
    (0..=n)
        .map(|i| {
            let a = a0 + (a1 - a0) * i as f64 / n as f64;
            [cx + r * a.cos(), cy + r * a.sin()]
        })
        .collect()
}

fn l_bracket() -> Domain {
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
    .unwrap()
}

fn ring() -> Domain {
    let mut outer = arc(0.0, 0.0, 20.0, 0.0, std::f64::consts::FRAC_PI_2, 24);
    let inner = arc(0.0, 0.0, 10.0, std::f64::consts::FRAC_PI_2, 0.0, 12);
    outer.extend(inner);
    Domain::new(outer, vec![]).unwrap()
}

fn stepped() -> Domain {
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
    .unwrap()
}

fn run(name: &str, d: &Domain, size: f64) {
    run_with(name, d, size, false);
}

fn run_all_quad(name: &str, d: &Domain, size: f64) {
    run_with(name, d, size, true);
}

fn run_with(name: &str, d: &Domain, size: f64, all_quad: bool) {
    let t = std::time::Instant::now();
    let mut sizing = Sizing::uniform(size).unwrap();
    if all_quad {
        sizing = sizing.all_quad();
    }
    let r = match mesh_2d(d, &sizing) {
        Ok(r) => r,
        Err(e) => {
            println!("{name:16} size {size:4}  ERROR {e}");
            return;
        }
    };
    let rep = quality::report(&r.mesh);
    let mut min_angle = f64::INFINITY;
    for cell in &r.mesh.cells {
        if cell.kind == pedroso_mesher::mesh::Kind::Quad4 {
            for a in quad_angles(cell, &r.mesh.points) {
                min_angle = min_angle.min(a);
            }
        }
    }
    println!(
        "{name:16} size {size:4}  pts {:5}  tri {:5}  quad {:5}  left {:3}  minSJ {:.3}  minAngle {:5.1}°  {:?}  {:.1}s",
        r.stats.points, r.stats.triangles, r.stats.quads, r.stats.unpaired_triangles,
        rep.min_scaled_jacobian, min_angle, rep.verdict, t.elapsed().as_secs_f64()
    );
    for reason in &rep.reasons {
        println!("    reason: {reason}");
    }
}

fn main() {
    // The hole is a polygon: its chord length must match the size field, or the
    // boundary is far finer than the interior and the transition quads suffer.
    for (size, segments) in [(4.0, 16), (2.0, 24), (1.0, 48)] {
        run(
            "plate-hole",
            &Domain::plate_with_hole(100.0, 60.0, 10.0, segments).unwrap(),
            size,
        );
    }
    run("l-bracket-2d", &l_bracket(), 2.0);
    run("ring-section", &ring(), 1.0);
    run("stepped-2d", &stepped(), 1.0);
    println!("--- all-quad mode ---");
    for (size, segments) in [(4.0, 16), (2.0, 24), (1.0, 48)] {
        run_all_quad(
            "plate-hole",
            &Domain::plate_with_hole(100.0, 60.0, 10.0, segments).unwrap(),
            size,
        );
    }
    run_all_quad("l-bracket-2d", &l_bracket(), 2.0);
    run_all_quad("ring-section", &ring(), 1.0);
    run_all_quad("stepped-2d", &stepped(), 1.0);
}
