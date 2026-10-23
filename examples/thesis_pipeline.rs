//! The Simons (2024) end-to-end thesis pipeline demo.
//!
//! Runs `simons_pipeline::demo()` on the 10 x 10 square at `base_size = 1.0`
//! and `growth = 0.5`, then prints the final report. The demo itself prints the
//! route (the graded tile-grid QGP path, the cross-field QGP + stitch path or
//! the channel fallback), the requested/effective graded tile grid, the size
//! accuracy and the smoothing before/after numbers.
//!
//! The second block runs `simons_graded::demo()`, which prints the graded
//! tile-grid numbers for the linear gradient and for the 2D gradient
//! `sigma = 1 + 0.3 x + 0.3 y` (including the explicit `8 x 8` request that is
//! coarsened because the domain only holds about nine `sigma`-sized quads).
//!
//! `cargo run --release --offline --example thesis_pipeline`

use pedroso_mesher::{simons_graded, simons_pipeline};

fn main() -> Result<(), String> {
    let report = simons_pipeline::demo()?;
    println!(
        "report: mesh={} quads={} triangles={} cracks={} SJ={:.4} \
         min_angle={:.2} size_ratio_mean={:.3} size_ratio_worst={:.3} \
         singularities=({}, {}) source={}",
        report.mesh.points.len(),
        report.quads,
        report.triangles,
        report.cracks,
        report.min_scaled_jacobian,
        report.min_angle_deg,
        report.size_ratio_mean,
        report.size_ratio_worst,
        report.singularities.0,
        report.singularities.1,
        report.source,
    );

    println!();
    let numbers = simons_graded::demo()?;
    println!(
        "graded demo summary: linear quads={} worst={:.3}; \
         2D derived quads={} worst={:.3}; 2D requested {:?} -> effective {:?}",
        numbers.linear.quads,
        numbers.linear.stats.worst,
        numbers.two_d_auto.quads,
        numbers.two_d_auto.stats.worst,
        numbers.two_d_eight.requested_tiles,
        numbers.two_d_eight.tiles,
    );
    Ok(())
}
