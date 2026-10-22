#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! W3: Q-Morph conversion measured against the acceptance gate.
//!
//! Every row is the gate applied to the mesh the run produced — quads,
//! triangles, inverted elements, min scaled Jacobian, min corner angle, cracks,
//! pass/fail and the first reason. A "PASS" cannot be read off the intent; it is
//! `verify::gate`.

use pedroso_mesher::mesh::{Cell, Mesh};
use pedroso_mesher::qmorph::convert_with_stats;
use pedroso_mesher::quality;
use pedroso_mesher::two_d::{triangulate, Domain, Sizing};
use pedroso_mesher::verify;

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
    let t0 = std::time::Instant::now();
    let (points, tris) = match triangulate(d, &Sizing::uniform(size).unwrap()) {
        Ok(v) => v,
        Err(e) => {
            println!("{name:12} {size:4}  triangulate error: {e}");
            return;
        }
    };
    let t_tri = t0.elapsed().as_secs_f64();
    let t1 = std::time::Instant::now();
    let ((quads, leftover, points), refusals) = convert_with_stats(&tris, &points, size);
    let t_conv = t1.elapsed().as_secs_f64();

    let mut cells: Vec<Cell> = quads.iter().map(|q| Cell::quad4(*q)).collect();
    cells.extend(leftover.iter().map(|t| Cell::tri3(*t)));
    let mut mesh = Mesh::new(points.clone(), cells);
    let gate_raw = verify::gate(&mesh);

    // The shipped W2 smoother runs on the converted mesh, exactly as it does on
    // the matching path.
    pedroso_mesher::two_d::smooth_mesh(&mut mesh, d, 80, size);
    let gate = verify::gate(&mesh);

    // Quality of the quads alone, so a leftover triangle does not hide whether
    // the converter is producing usable elements.
    let quad_mesh = Mesh::new(
        mesh.points.clone(),
        quads.iter().map(|q| Cell::quad4(*q)).collect::<Vec<_>>(),
    );
    let quad_report = quality::report(&quad_mesh);

    println!(
        "{name:12} {size:4}  W3 raw  quads {:5}  tri {:4}  inv {:3}  SJ {:.3}  ang {:5.1}  cracks {:3}  {:4}  {:.2}s/{:.2}s",
        gate_raw.quads,
        gate_raw.triangles,
        gate_raw.inverted,
        gate_raw.min_scaled_jacobian,
        gate_raw.min_corner_angle_deg,
        gate_raw.cracks,
        if gate_raw.passes() { "PASS" } else { "FAIL" },
        t_tri,
        t_conv,
    );
    if let Some(reason) = gate_raw.reason() {
        println!("      reason: {reason}");
    }
    println!(
        "{:12} {:>4}  +W2 sm  quads {:5}  tri {:4}  inv {:3}  SJ {:.3}  ang {:5.1}  cracks {:3}  {:4}  | quads-only SJ {:.3} ang {:5.1}",
        "",
        "",
        gate.quads,
        gate.triangles,
        gate.inverted,
        gate.min_scaled_jacobian,
        gate.min_corner_angle_deg,
        gate.cracks,
        if gate.passes() { "PASS" } else { "FAIL" },
        quad_report.min_scaled_jacobian,
        quad_report.min_corner_angle_deg,
    );
    if let Some(reason) = gate.reason() {
        println!("      reason: {reason}");
    }
    println!("      refusals: {refusals:?}");
}

fn main() {
    println!(
        "{:12} {:>4}  {:>5}  {:>4}  {:>3}  {:>5}  {:>5}  {:>6}  {:>4}",
        "part", "size", "quads", "tri", "inv", "minSJ", "minAng", "cracks", "gate"
    );
    run(
        "plate-hole",
        &Domain::plate_with_hole(100.0, 60.0, 10.0, 16).unwrap(),
        4.0,
    );
    run(
        "plate-hole",
        &Domain::plate_with_hole(100.0, 60.0, 10.0, 24).unwrap(),
        2.0,
    );
    run(
        "plate-hole",
        &Domain::plate_with_hole(100.0, 60.0, 10.0, 48).unwrap(),
        1.0,
    );
    run("l-bracket", &l_bracket(), 2.0);
    run("ring", &ring(), 1.0);
    run("stepped", &stepped(), 1.0);

    // Performance: the gate wants 10 000 quads converted in under 2 s.
    println!("--- performance ---");
    run(
        "plate-perf",
        &Domain::plate_with_hole(100.0, 60.0, 10.0, 64).unwrap(),
        0.6,
    );
}
