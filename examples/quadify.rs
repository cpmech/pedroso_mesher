#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! W3: front-based conversion, measured against the corpus.

use pedroso_mesher::mesh::{Cell, Mesh};
use pedroso_mesher::quadify::quadify;
use pedroso_mesher::quality;
use pedroso_mesher::two_d::{triangulate, Domain, Sizing};

fn run(name: &str, d: &Domain, size: f64) {
    let _ = name;
    let (points, tris) = triangulate(d, &Sizing::uniform(size).unwrap()).unwrap();
    let (quads, leftover) = quadify(&tris, &points, size);
    let mut cells: Vec<Cell> = quads.iter().map(|q| Cell::quad4(*q)).collect();
    cells.extend(leftover.iter().map(|t| Cell::tri3(*t)));
    let mesh = Mesh::new(points, cells);
    let report = quality::report(&mesh);
    let topo = mesh.topology();
    println!(
        "{name:14} size {size:4}  quads {:5}  leftover {:4} ({:5.1}% quads)  minSJ {:.3}  minAng {:5.1}  aspect {:5.1}  {report_verdict:?}  cracks {}",
        quads.len(),
        leftover.len(),
        100.0 * quads.len() as f64 / (quads.len() + leftover.len()) as f64,
        report.min_scaled_jacobian,
        report.min_corner_angle_deg,
        report.max_aspect_ratio,
        topo.cracks,
        report_verdict = report.verdict
    );
}

fn main() {
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
        "l-bracket",
        &Domain::new(
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
    );
}
