#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! L-norm (infinity-norm) frontal Delaunay route, measured on the corpus.
//!
//! Each part gets a **raw** row (the route's own output) and a **post** row
//! (after the benchmark's common post-pass: 80 smoothing sweeps, 12 quality
//! swap rounds and 200 perturbation rounds).

use pedroso_mesher::alt_l1;
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

fn quads_of(mesh: &Mesh) -> Vec<[usize; 4]> {
    mesh.cells
        .iter()
        .filter(|c| c.kind == Kind::Quad4)
        .map(|c| [c.nodes[0], c.nodes[1], c.nodes[2], c.nodes[3]])
        .collect()
}

fn tris_of(mesh: &Mesh) -> Vec<[usize; 3]> {
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

/// The exact common post-pass of the brief.
fn post(mesh: &Mesh, domain: &Domain, size: f64) -> Mesh {
    let mut m = mesh.clone();
    smooth_mesh(&mut m, domain, 80, size);
    let (p, q, t, _) = sun::improve_quality(&m.points, &quads_of(&m), &tris_of(&m), 12);
    m = Mesh::new(p, cells_of(&q, &t));
    let (p, q, t, _) = sun::perturb_low_quads(
        &m.points,
        &quads_of(&m),
        &tris_of(&m),
        size,
        200,
        Some(domain),
    );
    Mesh::new(p, cells_of(&q, &t))
}

fn valence4(mesh: &Mesh) -> Option<f64> {
    let mut edge_count: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
    for cell in &mesh.cells {
        for i in 0..cell.nodes.len() {
            let a = cell.nodes[i];
            let b = cell.nodes[(i + 1) % cell.nodes.len()];
            *edge_count.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    let mut boundary = vec![false; mesh.points.len()];
    for ((a, b), n) in edge_count {
        if n == 1 {
            if a < boundary.len() {
                boundary[a] = true;
            }
            if b < boundary.len() {
                boundary[b] = true;
            }
        }
    }
    let mut incident = vec![0usize; mesh.points.len()];
    for cell in &mesh.cells {
        if cell.kind == Kind::Quad4 {
            for &v in &cell.nodes {
                incident[v] += 1;
            }
        }
    }
    let interior: Vec<usize> = (0..mesh.points.len())
        .filter(|&v| !boundary[v] && incident[v] > 0)
        .collect();
    if interior.is_empty() {
        return None;
    }
    let four = interior.iter().filter(|&&v| incident[v] == 4).count();
    Some(100.0 * four as f64 / interior.len() as f64)
}

fn aspects(mesh: &Mesh) -> (f64, f64) {
    let mut values: Vec<f64> = mesh
        .cells
        .iter()
        .map(|c| quality::aspect_ratio(c, &mesh.points))
        .filter(|v| v.is_finite())
        .collect();
    if values.is_empty() {
        return (0.0, 0.0);
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let index = ((0.95 * (values.len() - 1) as f64).round() as usize).min(values.len() - 1);
    (values[index], *values.last().unwrap_or(&0.0))
}

fn print_row(name: &str, size: f64, mode: &str, mesh: &Mesh, domain: &Domain, secs: f64) {
    let gate = verify::gate(mesh);
    let offset = verify::boundary_offset(domain, mesh);
    let v4 = valence4(mesh)
        .map(|v| format!("{v:.1}"))
        .unwrap_or_else(|| "--".to_string());
    let (p95, amax) = aspects(mesh);
    let verdict = if gate.passes() { "PASS" } else { "FAIL" };
    let reason = gate
        .reason()
        .map(|r| format!(" {}", r.replace(' ', "_")))
        .unwrap_or_default();
    println!(
        "{name:12} {size:4} {mode:4} q={:5} tri={:4} inv={:3} ovl={:2} sj={:.3} ang={:5.1} cr={:3} {verdict}{reason} {secs:.2} v4={v4} off={offset:.1e} ar={p95:.2} armax={amax:.2}",
        gate.quads, gate.triangles, gate.inverted, gate.overlapping,
        gate.min_scaled_jacobian, gate.min_corner_angle_deg, gate.cracks,
    );
}

struct Row {
    name: String,
    size: f64,
    domain: Domain,
    raw: Option<Mesh>,
    post: Option<Mesh>,
    raw_secs: f64,
    post_secs: f64,
    error: Option<String>,
}

fn measure(name: &str, domain: &Domain, size: f64) -> Row {
    let mut row = Row {
        name: name.to_string(),
        size,
        domain: domain.clone(),
        raw: None,
        post: None,
        raw_secs: 0.0,
        post_secs: 0.0,
        error: None,
    };
    let sizing = match Sizing::uniform(size) {
        Ok(sizing) => sizing,
        Err(e) => {
            row.error = Some(e.to_string());
            return row;
        }
    };
    let start = std::time::Instant::now();
    match alt_l1::mesh(domain, &sizing) {
        Ok(raw) => {
            row.raw_secs = start.elapsed().as_secs_f64();
            let start = std::time::Instant::now();
            let after = post(&raw, domain, size);
            row.post_secs = start.elapsed().as_secs_f64();
            row.raw = Some(raw);
            row.post = Some(after);
        }
        Err(e) => {
            row.raw_secs = start.elapsed().as_secs_f64();
            row.error = Some(e);
        }
    }
    row
}

fn main() {
    let parts: Vec<(&str, Domain, f64)> = vec![
        (
            "plate-hole",
            Domain::plate_with_hole(100.0, 60.0, 10.0, 16).unwrap(),
            4.0,
        ),
        (
            "plate-hole",
            Domain::plate_with_hole(100.0, 60.0, 10.0, 24).unwrap(),
            2.0,
        ),
        (
            "plate-hole",
            Domain::plate_with_hole(100.0, 60.0, 10.0, 48).unwrap(),
            1.0,
        ),
        ("l-bracket", l_bracket(), 2.0),
        ("ring", ring(), 1.0),
        ("stepped", stepped(), 1.0),
        (
            "plate-perf",
            Domain::plate_with_hole(100.0, 60.0, 10.0, 64).unwrap(),
            0.6,
        ),
    ];
    let rows: Vec<Row> = parts
        .iter()
        .map(|(name, domain, size)| measure(name, domain, *size))
        .collect();

    println!(
        "{:12} {:>4} {:4} {:>6} {:>4} {:>3} {:>2} {:>5} {:>5} {:>3} {:>4} {:>5} {:>5} {:>4} {:>7} {:>5} {:>7}",
        "part", "size", "mode", "quads", "tri", "inv", "ovl", "sj", "ang", "cr", "gate", "secs",
        "v4", "off", "ar", "armax", ""
    );
    println!("--- RAW (the route's own output) ---");
    for row in &rows {
        print_raw_post(row, true);
    }
    println!("--- POST (80 smooth + 12 quality swaps + 200 perturb rounds) ---");
    for row in &rows {
        print_raw_post(row, false);
    }
}

fn print_raw_post(row: &Row, raw: bool) {
    let mode = if raw { "raw" } else { "post" };
    let mesh = if raw { &row.raw } else { &row.post };
    let secs = if raw { row.raw_secs } else { row.post_secs };
    match mesh {
        Some(mesh) => print_row(&row.name, row.size, mode, mesh, &row.domain, secs),
        None => {
            let reason = row.error.clone().unwrap_or_else(|| "no mesh".to_string());
            println!(
                "{:12} {:4} {mode:4} q=    0 tri=   0 inv=  0 ovl= 0 sj=0.000 ang=  0.0 cr=  0 ERROR {reason} {secs:.2} v4=-- off=0.0e0 ar=0.00 armax=0.00",
                row.name, row.size
            );
        }
    }
}
