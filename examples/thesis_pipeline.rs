//! The Simons (2024) end-to-end thesis pipeline demo.
//!
//! Runs `simons_pipeline::demo()` on the 10 x 10 square at `base_size = 1.0`
//! and `growth = 0.5`, then prints the final report. The demo itself prints the
//! route (QGP + stitch or the channel fallback), the size accuracy and the
//! smoothing before/after numbers.
//!
//! `cargo run --release --offline --example thesis_pipeline`

use pedroso_mesher::simons_pipeline;

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
    Ok(())
}
