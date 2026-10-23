//! Chapter 5 stitch: QGP interior tiles + the Chapter 2 channel to the boundary.
//! `cargo run --release --offline --example thesis_stitch`
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use pedroso_mesher::simons_regions;

fn main() {
    match simons_regions::stitch_demo() {
        Ok(n) => {
            for c in [&n.rectangle, &n.annulus, &n.nautilus] {
                println!(
                    "{}: interior {} carved {} channel {} bands {} -> {} quads, cracks {}, SJ {:.4}, angles {:.1}/{:.1}, coverage {:.3}, offset {:.2e}",
                    c.name, c.interior_quads, c.carved_quads, c.channel_quads, c.bands,
                    c.quads, c.cracks, c.min_scaled_jacobian, c.min_corner_angle_deg,
                    c.max_corner_angle_deg, c.coverage, c.boundary_offset
                );
                for note in &c.notes {
                    println!("  note: {note}");
                }
            }
        }
        Err(e) => println!("stitch demo failed: {e}"),
    }
}
