//! The Ch5 seeded-field grading path on a graded square, with the merged-region
//! block mesher in place. `cargo run --release --offline --example thesis_seeded`
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use pedroso_mesher::simons_pipeline::{self, Options};
use pedroso_mesher::two_d::Domain;

fn main() {
    let domain = Domain::rectangle(10.0, 10.0).unwrap();
    for seeded in [false, true] {
        let options = Options {
            seeded,
            size_weight: 1.0,
            ..Options::default()
        };
        match simons_pipeline::mesh_with_trace(&domain, &options) {
            Ok((report, trace)) => println!(
                "seeded={seeded}: source={} quads={} SJ={:.4} ang={:.1} size mean={:.3} worst={:.3} seeded_coverage={:?} seeded_error={:?}",
                report.source,
                report.quads,
                report.min_scaled_jacobian,
                report.min_angle_deg,
                report.size_ratio_mean,
                report.size_ratio_worst,
                trace.seeded_coverage,
                trace.seeded_error
            ),
            Err(error) => println!("seeded={seeded}: error: {error}"),
        }
    }
}
