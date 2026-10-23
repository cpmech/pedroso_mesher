//! Simons 2024 pipeline demo: size-driven QGP tiles + Coons patches (Ch5),
//! with the size-aware cross field (Ch3) and the anisotropy analysis (Ch4) run
//! on the produced mesh as diagnostics.
//!
//! The domain is a 10 x 10 square split into four QGP regions (a 2 x 2 block
//! layout); the target size grows linearly, sigma = 1 + 0.5 x, so the ideal
//! subdivision counts differ between the left and right tiles. QGP balances
//! the shared edges (eq. 5.1), then each region is meshed with bilinear Coons
//! patches (eq. 5.2) at its integer counts.
//!
//! `cargo run --release --offline --example thesis_demo`
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use pedroso_mesher::mesh::{Cell, Mesh};
use pedroso_mesher::quality;
use pedroso_mesher::simons_godf;
use pedroso_mesher::simons_qgp::{self, Region};
use pedroso_mesher::simons_sizefamily;

fn sigma(x: f64, _y: f64) -> f64 {
    1.0 + 0.5 * x
}

/// Integral of ds / sigma along a straight edge, by 16-point trapezoid.
fn ideal_count(a: [f64; 2], b: [f64; 2]) -> f64 {
    let n = 16;
    let mut sum = 0.0;
    for i in 0..=n {
        let t = i as f64 / n as f64;
        let p = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
        let w = if i == 0 || i == n { 1.0 } else { 2.0 };
        sum += w / sigma(p[0], p[1]);
    }
    sum * (dist(a, b) / 2.0 / n as f64)
}

fn dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn main() {
    // 3 x 3 grid of corners: (0,0) ... (10,10) in steps of 5.
    let corner = |i: usize, j: usize| [5.0 * i as f64, 5.0 * j as f64];
    // Edges: horizontal edge h(i,j) from corner(i,j) to corner(i+1,j);
    // vertical edge v(i,j) from corner(i,j) to corner(i,j+1).
    let nh = 3; // horizontal edge ids 0..6
    let nv = 3;
    // id of h(i,j) with i in 0..2, j in 0..3; id of v(i,j) with i in 0..3, j in 0..2
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

    // Four regions, sides (bottom, right, top, left), each side one edge.
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
    let counts = simons_qgp::solve_counts(&regions, &edge_ideal).unwrap();
    println!(
        "ideal counts: {:?}",
        edge_ideal
            .iter()
            .map(|v| (v * 10.0).round() / 10.0)
            .collect::<Vec<_>>()
    );
    println!("integer counts: {counts:?}");

    // Mesh each region with its Coons patch and merge.
    let mut points: Vec<[f64; 3]> = Vec::new();
    let mut cells: Vec<Cell> = Vec::new();
    for (r, c) in regions.iter().zip(corners.iter()) {
        let nu = counts[c_sides(r)[0]].max(1);
        let nv = counts[c_sides(r)[3]].max(1);
        let quads = simons_qgp::coons(c, nu, nv);
        println!(
            "region with sides {:?}: {nu} x {nv} -> {} quads",
            c_sides(r),
            quads.len()
        );
        let mut index = std::collections::HashMap::new();
        for q in quads {
            let mut ids = [0usize; 4];
            for (k, p) in q.iter().enumerate() {
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
    let mesh = Mesh::new(points, cells);
    let gate = pedroso_mesher::verify::gate(&mesh);
    println!(
        "mesh: q={} tri={} inv={} ovl={} cracks={} SJ={:.3} angle={:.1}",
        gate.quads,
        gate.triangles,
        gate.inverted,
        gate.overlapping,
        gate.cracks,
        gate.min_scaled_jacobian,
        gate.min_corner_angle_deg
    );

    // Size accuracy: sqrt(area) / sigma(centroid).
    let mut worst: f64 = 1.0;
    for cell in &mesh.cells {
        let area = quality::volume(cell, &mesh.points).abs();
        let cx = cell.nodes.iter().map(|&v| mesh.points[v][0]).sum::<f64>() / 4.0;
        let cy = cell.nodes.iter().map(|&v| mesh.points[v][1]).sum::<f64>() / 4.0;
        let ratio = (area.sqrt() / sigma(cx, cy)).max(1.0 / (area.sqrt() / sigma(cx, cy)));
        worst = worst.max(ratio);
    }
    println!("size accuracy: worst factor {worst:.3} (1.0 is exact)");

    // Ch3 + Ch4 diagnostics: solve the size-aware cross field on the mesh
    // (each quad split into two triangles) and analyse its anisotropy.
    let mut pts = mesh.points.clone();
    pts.iter_mut().for_each(|p| p[2] = 0.0);
    let mut tris = Vec::new();
    for c in &mesh.cells {
        tris.push([c.nodes[0], c.nodes[1], c.nodes[2]]);
        tris.push([c.nodes[0], c.nodes[2], c.nodes[3]]);
    }
    let size: Vec<f64> = mesh.points.iter().map(|p| sigma(p[0], p[1])).collect();
    match simons_godf::solve(&pts, &tris, &size, 400) {
        Ok(alpha) => {
            let (s3, s5) = simons_godf::count_singularities(&alpha, &tris);
            println!("cross field: singularities valence-3 {s3}, valence-5 {s5}");
            match simons_sizefamily::analyze(&pts, &tris, &alpha, &size) {
                Ok(a) => println!(
                    "anisotropy: max {:.4}, mean {:.4}, samples {}, worst size ratio {:.3}",
                    a.max_anisotropy, a.mean_anisotropy, a.samples, a.max_size_ratio
                ),
                Err(e) => println!("anisotropy analysis failed: {e}"),
            }
        }
        Err(e) => println!("cross field solve failed: {e}"),
    }
}

fn c_sides(r: &Region) -> [usize; 4] {
    [r.sides[0][0], r.sides[1][0], r.sides[2][0], r.sides[3][0]]
}
