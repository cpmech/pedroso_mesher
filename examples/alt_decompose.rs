#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Measurement of the domain-decomposition + templates route on the corpus.
//!
//! Every part is printed twice: the route's raw all-quad ouput, and the output
//! of the common post-pass (smooth, edge swaps, perturbation) fixed by the
//! brief.  `v4` is the share of interior nodes with exactly four incident
//! quads; `ar` is the 95th percentile and `armax` the largest aspect ratio.

use std::collections::HashMap;
use std::time::Instant;

use pedroso_mesher::alt_decompose;
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
    let mut cells = Vec::with_capacity(quads.len() + tris.len());
    for q in quads {
        cells.push(Cell::quad4(*q));
    }
    for t in tris {
        cells.push(Cell::tri3(*t));
    }
    cells
}

struct Measured {
    rows: verify::Gate,
    v4: f64,
    interior: usize,
    aspect_p95: f64,
    aspect_max: f64,
    offset: f64,
    secs: f64,
}

fn measure(mesh: &Mesh, domain: &Domain, secs: f64) -> Measured {
    let gate = verify::gate(mesh);
    let mut boundary = vec![false; mesh.points.len()];
    let mut edge_use: HashMap<(usize, usize), usize> = HashMap::new();
    let mut incident = vec![0usize; mesh.points.len()];
    for cell in &mesh.cells {
        for i in 0..cell.nodes.len() {
            let a = cell.nodes[i];
            let b = cell.nodes[(i + 1) % cell.nodes.len()];
            *edge_use.entry((a.min(b), a.max(b))).or_default() += 1;
            incident[a] += 1;
        }
    }
    for ((a, b), n) in edge_use {
        if n == 1 {
            boundary[a] = true;
            boundary[b] = true;
        }
    }
    let mut interior = 0usize;
    let mut regular = 0usize;
    for (v, &is_boundary) in boundary.iter().enumerate() {
        if !is_boundary && incident[v] > 0 {
            interior += 1;
            if incident[v] == 4 {
                regular += 1;
            }
        }
    }
    let v4 = if interior == 0 {
        f64::NAN
    } else {
        100.0 * regular as f64 / interior as f64
    };
    let mut aspects: Vec<f64> = mesh
        .cells
        .iter()
        .filter(|c| c.kind == Kind::Quad4)
        .map(|c| quality::aspect_ratio(c, &mesh.points))
        .filter(|a| a.is_finite())
        .collect();
    aspects.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let aspect_max = aspects.last().copied().unwrap_or(f64::NAN);
    let aspect_p95 = if aspects.is_empty() {
        f64::NAN
    } else {
        aspects[((aspects.len() - 1) as f64 * 0.95).round() as usize]
    };
    Measured {
        rows: gate,
        v4,
        interior,
        aspect_p95,
        aspect_max,
        offset: verify::boundary_offset(domain, mesh),
        secs,
    }
}

fn print_row(part: &str, size: f64, kind: &str, m: &Measured) {
    let verdict = if m.rows.passes() { "PASS" } else { "FAIL" };
    let v4 = if m.interior == 0 {
        "   -".to_string()
    } else {
        format!("{:4.1}", m.v4)
    };
    println!(
        "{part:12} {size:4} {kind:4} q={:6} tri={:4} inv={:3} ovl={:3} sj={:6.3} ang={:5.1} cr={:3} {verdict} v4={v4} ar={:6.2} armax={:8.2} off={:.1e} {:.2}s{}",
        m.rows.quads,
        m.rows.triangles,
        m.rows.inverted,
        m.rows.overlapping,
        m.rows.min_scaled_jacobian,
        m.rows.min_corner_angle_deg,
        m.rows.cracks,
        m.aspect_p95,
        m.aspect_max,
        m.offset,
        m.secs,
        match m.rows.reason() {
            Some(r) => format!(" reason: {r}"),
            None => String::new(),
        }
    );
}

fn run(part: &str, domain: &Domain, size: f64) {
    let sizing = match Sizing::uniform(size) {
        Ok(s) => s,
        Err(e) => {
            println!("{part:12} {size:4} ERROR sizing {e}");
            return;
        }
    };
    let start = Instant::now();
    let result = alt_decompose::mesh(domain, &sizing);
    let route_secs = start.elapsed().as_secs_f64();
    let mut mesh = match result {
        Ok(m) => m,
        Err(e) => {
            println!("{part:12} {size:4} raw  ERROR {e}");
            println!("{part:12} {size:4} post ERROR {e}");
            return;
        }
    };
    let raw = measure(&mesh, domain, route_secs);
    print_row(part, size, "raw", &raw);

    let start = Instant::now();
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
    let post = measure(&mesh, domain, post_secs);
    print_row(part, size, "post", &post);
}

fn main() {
    println!(
        "{:12} {:>4} {:>4} {:>7} {:>4} {:>3} {:>3} {:>6} {:>5} {:>3} {:>4} {:>4} {:>6} {:>8} {:>9} {:>6}",
        "part", "size", "mode", "quads", "tri", "inv", "ovl", "minSJ", "minAng", "cr", "gate", "v4%", "ar95", "armax", "off", "secs"
    );
    run(
        "plate-hole",
        &Domain::plate_with_hole(100.0, 60.0, 10.0, 16).unwrap(),
        4.0,
    );
    run(
        "plate-hole",
        &Domain::plate_with_hole(100.0, 60.0, 10.0, 24).unwrap(),
        2.0,
    );
    run(
        "plate-hole",
        &Domain::plate_with_hole(100.0, 60.0, 10.0, 48).unwrap(),
        1.0,
    );
    run("l-bracket", &l_bracket(), 2.0);
    run("ring", &ring(), 1.0);
    run("stepped", &stepped(), 1.0);
    run(
        "plate-perf",
        &Domain::plate_with_hole(100.0, 60.0, 10.0, 64).unwrap(),
        0.6,
    );
}
