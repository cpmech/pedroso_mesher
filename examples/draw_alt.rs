//! Draw the two leading routes side by side on one part.
//!
//! `cargo run --release --offline --example draw_alt -- plate-hole-2mm`
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use pedroso_mesher::draw::{self, MeshDrawOptions};
use pedroso_mesher::mesh::{Cell, Mesh};
use pedroso_mesher::quad::{self, QuadError};
use pedroso_mesher::sun;
use pedroso_mesher::two_d::{smooth_mesh, Domain, Sizing};
use pedroso_mesher::verify;

fn quad_cells(mesh: &Mesh) -> Vec<[usize; 4]> {
    mesh.cells
        .iter()
        .filter(|c| c.nodes.len() == 4)
        .map(|c| [c.nodes[0], c.nodes[1], c.nodes[2], c.nodes[3]])
        .collect()
}

fn tri_cells(mesh: &Mesh) -> Vec<[usize; 3]> {
    mesh.cells
        .iter()
        .filter(|c| c.nodes.len() == 3)
        .map(|c| [c.nodes[0], c.nodes[1], c.nodes[2]])
        .collect()
}

fn cells_of(quads: &[[usize; 4]], tris: &[[usize; 3]]) -> Vec<Cell> {
    let mut cells: Vec<Cell> = quads.iter().map(|q| Cell::quad4(*q)).collect();
    cells.extend(tris.iter().map(|t| Cell::tri3(*t)));
    cells
}

/// The same post-pass the agents measure with (W2 smoothing, angle swaps,
/// boundary-aware perturbation), so the figures show what the gate sees.
fn post_pass(domain: &Domain, mesh: &mut Mesh, size: f64) {
    smooth_mesh(mesh, domain, 80, size);
    let (p, q, t, _) = sun::improve_quality(&mesh.points, &quad_cells(mesh), &tri_cells(mesh), 12);
    *mesh = Mesh::new(p, cells_of(&q, &t));
    let (p, q, t, _) = sun::perturb_low_quads(
        &mesh.points,
        &quad_cells(mesh),
        &tri_cells(mesh),
        size,
        200,
        Some(domain),
    );
    *mesh = Mesh::new(p, cells_of(&q, &t));
}

fn part(name: &str) -> (Domain, f64) {
    match name {
        "plate-hole-4mm" => (Domain::plate_with_hole(100.0, 60.0, 10.0, 16).unwrap(), 4.0),
        "plate-hole-2mm" => (Domain::plate_with_hole(100.0, 60.0, 10.0, 24).unwrap(), 2.0),
        "plate-hole-1mm" => (Domain::plate_with_hole(100.0, 60.0, 10.0, 48).unwrap(), 1.0),
        "l-bracket" => (
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
            .unwrap(),
            2.0,
        ),
        "ring" => {
            let arc = |r: f64, a0: f64, a1: f64, n: usize| -> Vec<[f64; 2]> {
                (0..=n)
                    .map(|i| {
                        let a = a0 + (a1 - a0) * i as f64 / n as f64;
                        [r * a.cos(), r * a.sin()]
                    })
                    .collect()
            };
            let mut pts = arc(20.0, 0.0, std::f64::consts::FRAC_PI_2, 24);
            pts.extend(arc(10.0, std::f64::consts::FRAC_PI_2, 0.0, 12));
            (Domain::new(pts, vec![]).unwrap(), 1.0)
        }
        "stepped" => (
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
            .unwrap(),
            1.0,
        ),
        _ => (Domain::plate_with_hole(100.0, 60.0, 10.0, 16).unwrap(), 4.0),
    }
}

fn main() {
    let name = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "plate-hole-2mm".to_string());
    let (domain, size) = part(&name);
    let sizing = Sizing::uniform(size).unwrap();
    std::fs::create_dir_all("figures/alts").unwrap();

    // current product (Q-Morph; the failing mesh is carried by the error)
    let product = match quad::mesh_quads(&domain, &sizing) {
        Ok(q) => Some(q.mesh),
        Err(QuadError::Unsupported(u)) => Some(u.mesh),
        Err(_) => None,
    };
    // paving
    let paving = pedroso_mesher::alt_paving::mesh(&domain, &sizing).ok();
    // quadtree (grid-like all-quad)
    let quadtree = pedroso_mesher::alt_quadtree::mesh(&domain, &sizing).ok();
    // bipartite (right-angled + labelling)
    let bipartite = pedroso_mesher::alt_bipartite::mesh(&domain, &sizing).ok();
    // L-infinity frontal Delaunay (best quality, 2 triangles on some parts)
    let l1 = pedroso_mesher::alt_l1::mesh(&domain, &sizing).ok();
    // decomposition + templates (structured blocks; acute corners on plates)
    let decompose = pedroso_mesher::alt_decompose::mesh(&domain, &sizing).ok();

    for (label, mesh) in [
        ("product", product),
        ("paving", paving),
        ("quadtree", quadtree),
        ("bipartite", bipartite),
        ("l1", l1),
        ("decompose", decompose),
    ] {
        let Some(mut mesh) = mesh else {
            println!("{label}: no mesh");
            continue;
        };
        let corner_before = verify::corner_offset(&domain, &mesh);
        // Show the route's intended output: its raw mesh when that passes the
        // gate, otherwise the shared post-pass (some routes rely on it, and for
        // the structured O-grid the post-pass is harmful).
        let raw_gate = verify::gate(&mesh);
        let applied = if raw_gate.passes() {
            "raw"
        } else {
            post_pass(&domain, &mut mesh, size);
            "post"
        };
        let gate = verify::gate(&mesh);
        let title = format!(
            "{name} — {label} ({applied}): {} quads, {} tri, SJ {:.3}, {:.1}°, off {:.1e}, corner {:.1e}",
            gate.quads,
            gate.triangles,
            gate.min_scaled_jacobian,
            gate.min_corner_angle_deg,
            verify::boundary_offset(&domain, &mesh),
            verify::corner_offset(&domain, &mesh)
        );
        let opts = MeshDrawOptions {
            title,
            color_by_gate_margin: true,
            mark_nodes: false,
            figure_inches: (9.0, 6.0),
        };
        let path = format!("figures/alts/{name}-{label}.png");
        draw::draw_domain_with_mesh(&domain, &mesh, &path, &opts).unwrap();
        println!(
            "{label:8} {applied:4} q {:5} tri {:3} SJ {:.3} ang {:.1} off {:.1e} corner {:.1e} (was {:.1e}) {} -> {path}",
            gate.quads,
            gate.triangles,
            gate.min_scaled_jacobian,
            gate.min_corner_angle_deg,
            verify::boundary_offset(&domain, &mesh),
            verify::corner_offset(&domain, &mesh),
            corner_before,
            if gate.passes() { "PASS" } else { "FAIL" }
        );
    }
}
