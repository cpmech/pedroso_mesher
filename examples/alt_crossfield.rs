#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Measurement of the CROSS-FIELD ALIGNED route on the common corpus.
//!
//! Each part is printed twice: **raw** (as the route produced it) and **post**
//! (after the common post-pass of the brief: `smooth_mesh` 80 iterations,
//! `sun::improve_quality` 12 rounds, `sun::perturb_low_quads` 200 rounds).
//!
//! Columns: quads, triangles, inverted, overlapping, minimum scaled Jacobian,
//! minimum corner angle, cracks, verdict, first reason, seconds, share of
//! interior nodes with valence 4 (`v4`), boundary offset, p95 aspect ratio and
//! maximum aspect ratio (`quality::aspect_ratio`).

use pedroso_mesher::alt_crossfield;
use pedroso_mesher::mesh::{Cell, Kind, Mesh};
use pedroso_mesher::quality;
use pedroso_mesher::sun;
use pedroso_mesher::two_d::{smooth_mesh, Domain, Sizing};
use pedroso_mesher::verify;

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

fn quad_vec(mesh: &Mesh) -> Vec<[usize; 4]> {
    mesh.cells
        .iter()
        .filter(|c| c.kind == Kind::Quad4)
        .map(|c| [c.nodes[0], c.nodes[1], c.nodes[2], c.nodes[3]])
        .collect()
}

fn tri_vec(mesh: &Mesh) -> Vec<[usize; 3]> {
    mesh.cells
        .iter()
        .filter(|c| c.kind == Kind::Tri3)
        .map(|c| [c.nodes[0], c.nodes[1], c.nodes[2]])
        .collect()
}

fn cells_of(quads: &[[usize; 4]], tris: &[[usize; 3]]) -> Vec<Cell> {
    let mut cells: Vec<Cell> = quads.iter().map(|q| Cell::quad4(*q)).collect();
    cells.extend(tris.iter().map(|t| Cell::tri3(*t)));
    cells
}

/// Interior valence-4 share and quad aspect-ratio statistics.
fn structure(mesh: &Mesh) -> (Option<f64>, f64, f64) {
    use std::collections::HashMap;
    let mut incident: HashMap<usize, usize> = HashMap::new();
    let mut boundary = vec![false; mesh.points.len()];
    let mut edge_use: HashMap<(usize, usize), usize> = HashMap::new();
    for cell in &mesh.cells {
        for &v in &cell.nodes {
            *incident.entry(v).or_default() += 1;
        }
        for i in 0..cell.nodes.len() {
            let a = cell.nodes[i];
            let b = cell.nodes[(i + 1) % cell.nodes.len()];
            *edge_use.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    for ((a, b), n) in edge_use {
        if n == 1 {
            if a < boundary.len() {
                boundary[a] = true;
            }
            if b < boundary.len() {
                boundary[b] = true;
            }
        }
    }
    let interior: Vec<usize> = (0..mesh.points.len())
        .filter(|&v| !boundary[v] && incident.contains_key(&v))
        .collect();
    let v4 = if interior.is_empty() {
        None
    } else {
        let four = interior
            .iter()
            .filter(|&&v| incident.get(&v).copied().unwrap_or(0) == 4)
            .count();
        Some(100.0 * four as f64 / interior.len() as f64)
    };
    let mut aspects: Vec<f64> = mesh
        .cells
        .iter()
        .filter(|c| c.kind == Kind::Quad4)
        .map(|c| quality::aspect_ratio(c, &mesh.points))
        .filter(|a| a.is_finite())
        .collect();
    if aspects.is_empty() {
        return (v4, 0.0, 0.0);
    }
    aspects.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let index = ((0.95 * aspects.len() as f64).ceil() as usize).saturating_sub(1);
    let p95 = aspects[index.min(aspects.len() - 1)];
    let max = *aspects.last().unwrap_or(&0.0);
    (v4, p95, max)
}

fn print_row(name: &str, size: f64, kind: &str, mesh: &Mesh, domain: &Domain, secs: f64) {
    let gate = verify::gate(mesh);
    let offset = verify::boundary_offset(domain, mesh);
    let (v4, p95, armax) = structure(mesh);
    let verdict = if gate.passes() { "PASS" } else { "FAIL" };
    let v4_text = match v4 {
        Some(v) => format!("{v:5.1}"),
        None => "  n/a".to_string(),
    };
    let reason = gate.reason().unwrap_or_default();
    println!(
        "{name:12} {size:4} {kind:4} q={:5} tri={:4} inv={:3} ovl={:2} sj={:.3} ang={:5.1} cr={:3} {verdict:4} {reason:28} {secs:5.2}s v4={v4_text} off={offset:.1e} ar={p95:6.2} armax={armax:8.2}",
        gate.quads,
        gate.triangles,
        gate.inverted,
        gate.overlapping,
        gate.min_scaled_jacobian,
        gate.min_corner_angle_deg,
        gate.cracks,
    );
}

fn run(name: &str, domain: &Domain, size: f64) -> f64 {
    let sizing = match Sizing::uniform(size) {
        Ok(sizing) => sizing,
        Err(e) => {
            println!("{name:12} {size:4} ERROR {e}");
            return 0.0;
        }
    };
    let start = std::time::Instant::now();
    let mesh = match alt_crossfield::mesh(domain, &sizing) {
        Ok(mesh) => mesh,
        Err(e) => {
            println!("{name:12} {size:4} ERROR {e}");
            return start.elapsed().as_secs_f64();
        }
    };
    let route_secs = start.elapsed().as_secs_f64();
    print_row(name, size, "raw", &mesh, domain, route_secs);

    // The common post-pass of the brief.
    let start = std::time::Instant::now();
    let mut mesh = mesh;
    smooth_mesh(&mut mesh, domain, 80, size);
    let (p, q, t, _) = sun::improve_quality(&mesh.points, &quad_vec(&mesh), &tri_vec(&mesh), 12);
    mesh = Mesh::new(p, cells_of(&q, &t));
    let (p, q, t, _) = sun::perturb_low_quads(
        &mesh.points,
        &quad_vec(&mesh),
        &tri_vec(&mesh),
        size,
        200,
        Some(domain),
    );
    mesh = Mesh::new(p, cells_of(&q, &t));
    let post_secs = start.elapsed().as_secs_f64();
    print_row(name, size, "post", &mesh, domain, post_secs);
    route_secs
}

fn main() {
    println!(
        "part         size kind   quads  tri inv ovl    sj   ang  cr verdict reason                       secs   v4      off      ar     armax"
    );
    let mut total = 0.0;
    total += run(
        "plate-hole",
        &Domain::plate_with_hole(100.0, 60.0, 10.0, 16).unwrap(),
        4.0,
    );
    total += run(
        "plate-hole",
        &Domain::plate_with_hole(100.0, 60.0, 10.0, 24).unwrap(),
        2.0,
    );
    total += run(
        "plate-hole",
        &Domain::plate_with_hole(100.0, 60.0, 10.0, 48).unwrap(),
        1.0,
    );
    total += run("l-bracket", &l_bracket(), 2.0);
    total += run("ring", &ring(), 1.0);
    total += run("stepped", &stepped(), 1.0);
    total += run(
        "plate-perf",
        &Domain::plate_with_hole(100.0, 60.0, 10.0, 64).unwrap(),
        0.6,
    );
    println!("route total: {total:.2}s");
}
