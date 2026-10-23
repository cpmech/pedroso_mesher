//! Target-area smoothing: a uniform grid with a graded target size field.
//! `cargo run --release --offline --example smooth_size`
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use pedroso_mesher::mesh::{Cell, Mesh};
use pedroso_mesher::quality;
use pedroso_mesher::smooth::{smooth_with_targets, SmoothOptions};
use pedroso_mesher::two_d::Domain;

fn main() {
    // A compressed grid (rows bunched in the middle) with a uniform target
    // size: the area term should spread the rows towards sqrt(area) = target.
    let rows = [0.0, 0.55, 1.9, 3.1, 4.45, 6.0];
    let mut points = Vec::new();
    for j in 0..rows.len() {
        for i in 0..7 {
            points.push([i as f64, rows[j], 0.0]);
        }
    }
    let mut cells = Vec::new();
    for j in 0..rows.len() - 1 {
        for i in 0..6 {
            let a = j * 7 + i;
            cells.push(Cell::quad4([a, a + 1, a + 8, a + 7]));
        }
    }
    let mesh = Mesh::new(points, cells);
    let domain = Domain::rectangle(6.0, 6.0).unwrap();
    let targets: Vec<f64> = vec![1.0; mesh.points.len()];
    for weight in [0.0, 0.3, 0.6, 1.0] {
        let options = SmoothOptions {
            size_weight: weight,
            shape_pass: false,
            ..SmoothOptions::default()
        };
        let outcome = smooth_with_targets(&domain, &mesh, &targets, &options);
        let mut worst: f64 = 1.0;
        let mut mean = 0.0;
        let mut count = 0usize;
        for cell in &outcome.smoothed.cells {
            let area = quality::volume(cell, &outcome.smoothed.points).abs();
            let target =
                cell.nodes.iter().map(|&v| targets[v]).sum::<f64>() / cell.nodes.len() as f64;
            let r = area.sqrt() / target;
            worst = worst.max(r).max(1.0 / r);
            mean += r;
            count += 1;
        }
        println!(
            "weight {weight:.1}: SJ {:.3} angle {:.1} pass {} size mean {:.3} worst {:.3}",
            outcome.after.min_scaled_jacobian,
            outcome.after.min_corner_angle_deg,
            outcome.after.passes,
            mean / count as f64,
            worst
        );
    }
}
