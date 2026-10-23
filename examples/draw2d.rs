//! Draw the corpus: the domain and the gate-coloured mesh for each part.
//!
//! ```bash
//! cargo run --release --example draw2d            # six planar parts, figures/
//! cargo run --release --example draw2d -- perf    # also the 0.6 mm perf part
//! ```
//!
//! Cells are filled by their **gate margin** `min(angle/30°, sj/0.2)` (see
//! `draw::gate_margin`): green clears the gate, yellow/orange/red fails it,
//! magenta is inverted, and triangles are dark red because they fail the
//! all-quad gate by definition. The title carries the measured numbers.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use pedroso_mesher::draw::{self, MeshDrawOptions};
use pedroso_mesher::quad::{self, QuadError};
use pedroso_mesher::two_d::{Domain, Sizing};

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

struct Part {
    name: &'static str,
    domain: Domain,
    size: f64,
}

fn parts(perf: bool) -> Vec<Part> {
    let mut out = vec![
        Part {
            name: "plate-hole-4mm",
            domain: Domain::plate_with_hole(100.0, 60.0, 10.0, 16).unwrap(),
            size: 4.0,
        },
        Part {
            name: "plate-hole-2mm",
            domain: Domain::plate_with_hole(100.0, 60.0, 10.0, 24).unwrap(),
            size: 2.0,
        },
        Part {
            name: "plate-hole-1mm",
            domain: Domain::plate_with_hole(100.0, 60.0, 10.0, 48).unwrap(),
            size: 1.0,
        },
        Part {
            name: "l-bracket-2mm",
            domain: l_bracket(),
            size: 2.0,
        },
        Part {
            name: "ring-1mm",
            domain: ring(),
            size: 1.0,
        },
        Part {
            name: "stepped-1mm",
            domain: stepped(),
            size: 1.0,
        },
    ];
    if perf {
        out.push(Part {
            name: "plate-perf-0.6mm",
            domain: Domain::plate_with_hole(100.0, 60.0, 10.0, 64).unwrap(),
            size: 0.6,
        });
    }
    out
}

fn main() {
    let perf = std::env::args().any(|a| a == "perf");
    let dir = std::env::args()
        .find(|a| a.starts_with("--out="))
        .map(|a| a["--out=".len()..].to_string())
        .unwrap_or_else(|| "figures".to_string());
    std::fs::create_dir_all(&dir).unwrap();
    let opts = |title: String| MeshDrawOptions {
        title,
        color_by_gate_margin: true,
        mark_nodes: false,
        figure_inches: (9.0, 7.0),
    };

    for part in parts(perf) {
        let base = format!("{dir}/{}", part.name);
        draw::draw_domain(
            &part.domain,
            &format!("{} — domain", part.name),
            &format!("{base}-domain.png"),
        )
        .unwrap();
        let sizing = Sizing::uniform(part.size).unwrap();
        match quad::mesh_quads(&part.domain, &sizing) {
            Ok(q) => {
                let title = format!(
                    "{} at {} mm — {} quads, {} tri, min SJ {:.3}, min angle {:.1}°",
                    part.name,
                    part.size,
                    q.gate.quads,
                    q.gate.triangles,
                    q.gate.min_scaled_jacobian,
                    q.gate.min_corner_angle_deg
                );
                let title = format!("{title}, boundary off {:.1e}", q.stats.boundary_offset);
                draw::draw_domain_with_mesh(
                    &part.domain,
                    &q.mesh,
                    &format!("{base}-mesh.png"),
                    &opts(title),
                )
                .unwrap();
                println!(
                    "{:<18} {:>5} quads  {:>3} tri  SJ {:.3}  {:5.1}°  supported",
                    part.name,
                    q.gate.quads,
                    q.gate.triangles,
                    q.gate.min_scaled_jacobian,
                    q.gate.min_corner_angle_deg
                );
            }
            Err(QuadError::Unsupported(u)) => {
                let title = format!(
                    "{} at {} mm — UNSUPPORTED: {}",
                    part.name, part.size, u.reason
                );
                let title = format!("{title}, boundary off {:.1e}", u.stats.boundary_offset);
                draw::draw_domain_with_mesh(
                    &part.domain,
                    &u.mesh,
                    &format!("{base}-unsupported.png"),
                    &opts(title),
                )
                .unwrap();
                println!(
                    "{:<18} {:>5} quads  {:>3} tri  SJ {:.3}  {:5.1}°  UNSUPPORTED: {}",
                    part.name,
                    u.gate.quads,
                    u.gate.triangles,
                    u.gate.min_scaled_jacobian,
                    u.gate.min_corner_angle_deg,
                    u.reason
                );
            }
            Err(e) => println!("{:<18} error: {e}", part.name),
        }
    }
}
