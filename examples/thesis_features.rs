//! Interior-feature constraints (Simons 2024, Chapter 2): a through-going
//! polyline splits the domain and both halves are welded along one sampled
//! seam; a terminating polyline becomes a thin slot (a hole).
//!
//! `cargo run --release --offline --example thesis_features`
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use pedroso_mesher::simons_features;

fn main() {
    let numbers = simons_features::demo().unwrap();
    for case in [
        &numbers.through_square,
        &numbers.stub_square,
        &numbers.cross_square,
        &numbers.t_square,
        &numbers.channel_square,
    ] {
        println!(
            "{}: {} quads, SJ {:.4}, angles {:.1}/{:.1}, cracks {}, offset {:.2e}, gate {}",
            case.name,
            case.quads,
            case.min_scaled_jacobian,
            case.min_angle_deg,
            case.max_angle_deg,
            case.cracks,
            case.boundary_offset,
            if case.gate_pass { "PASS" } else { "FAIL" }
        );
    }
}
