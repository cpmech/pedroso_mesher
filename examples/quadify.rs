#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! W3: front-based conversion, measured against the corpus.

use pedroso_mesher::quadify::quadify;
use pedroso_mesher::two_d::{triangulate, Domain, Sizing};

fn run(name: &str, d: &Domain, size: f64) {
    let _ = name;
    let (points, tris) = triangulate(d, &Sizing::uniform(size).unwrap()).unwrap();
    let (quads, leftover) = quadify(&tris, &points);
    println!(
        "{name:14} size {size:4}  triangles {:5}  quads {:5}  leftover {:4} ({:5.1}% quads)",
        tris.len(),
        quads.len(),
        leftover.len(),
        100.0 * quads.len() as f64 / (quads.len() + leftover.len()) as f64
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
