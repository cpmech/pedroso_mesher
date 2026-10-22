#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Measurement table for the cleanup-free quadtree route (`alt_quadtree`).
//!
//! Every part is printed twice: **raw**, the route's output as produced, and
//! **post**, after the exact common post-pass of the brief (80 smoothing
//! iterations, 12 `sun::improve_quality` rounds, 200 `sun::perturb_low_quads`
//! rounds).  The columns are the gate plus `v4` (share of interior nodes with
//! exactly four incident quads), `ar`/`armax` (95th percentile and maximum
//! edge-length aspect ratio) and `off` (`verify::boundary_offset`).

use std::time::Instant;

use pedroso_mesher::mesh::{Cell, Kind, Mesh};
use pedroso_mesher::quality;
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

/// Share of interior nodes with exactly four incident quads, in percent.
fn valence4(mesh: &Mesh) -> Option<f64> {
    let mut edge_uses: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
    for cell in &mesh.cells {
        let n = cell.nodes.len();
        for i in 0..n {
            let a = cell.nodes[i];
            let b = cell.nodes[(i + 1) % n];
            *edge_uses.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    let mut on_boundary = vec![false; mesh.points.len()];
    for ((a, b), uses) in &edge_uses {
        if *uses == 1 {
            on_boundary[*a] = true;
            on_boundary[*b] = true;
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
    let mut interior = 0usize;
    let mut four = 0usize;
    for (i, &flag) in on_boundary.iter().enumerate() {
        if flag {
            continue;
        }
        interior += 1;
        if incident[i] == 4 {
            four += 1;
        }
    }
    if interior == 0 {
        None
    } else {
        Some(100.0 * four as f64 / interior as f64)
    }
}

fn aspect_stats(mesh: &Mesh) -> (f64, f64) {
    let mut values: Vec<f64> = mesh
        .cells
        .iter()
        .map(|c| quality::aspect_ratio(c, &mesh.points))
        .filter(|v| v.is_finite())
        .collect();
    if values.is_empty() {
        return (f64::INFINITY, f64::INFINITY);
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((values.len() - 1) as f64 * 0.95).round() as usize;
    (
        values[idx.min(values.len() - 1)],
        *values.last().unwrap_or(&f64::INFINITY),
    )
}

fn print_row(name: &str, size: f64, tag: &str, mesh: &Mesh, domain: &Domain, secs: f64) {
    let gate = verify::gate(mesh);
    let offset = verify::boundary_offset(domain, mesh);
    let (p95, max) = aspect_stats(mesh);
    let v4 = valence4(mesh);
    let verdict = if gate.passes() { "PASS" } else { "FAIL" };
    let reason = gate.reason().unwrap_or_default();
    let v4_text = match v4 {
        Some(v) => format!("{v:4.1}%"),
        None => "  -- ".to_string(),
    };
    println!(
        "{name:12} {size:4.1} {tag:4} q={:6} tri={:4} inv={:3} ovl={:3} sj={:.3} ang={:5.1} cr={:3} {} {:<28} {:5.2}s v4={} ar={:5.2} armax={:7.2} off={:.1e}",
        gate.quads,
        gate.triangles,
        gate.inverted,
        gate.overlapping,
        gate.min_scaled_jacobian,
        gate.min_corner_angle_deg,
        gate.cracks,
        verdict,
        reason,
        secs,
        v4_text,
        p95,
        max,
        offset,
    );
}

fn run(name: &str, domain: &Domain, size: f64) -> Option<(f64, f64)> {
    let sizing = match Sizing::uniform(size) {
        Ok(s) => s,
        Err(e) => {
            println!("{name:12} {size:4.1} sizing error: {e}");
            return None;
        }
    };
    let start = Instant::now();
    let raw = match pedroso_mesher::alt_quadtree::mesh(domain, &sizing) {
        Ok(m) => m,
        Err(e) => {
            println!("{name:12} {size:4.1} ERROR {e}");
            return None;
        }
    };
    let secs = start.elapsed().as_secs_f64();
    let raw_report = quality::report(&raw);
    let band = (
        raw_report.min_corner_angle_deg,
        raw_report.max_corner_angle_deg,
    );
    print_row(name, size, "raw", &raw, domain, secs);

    // The common post-pass, exactly as specified in the brief.
    let post_start = Instant::now();
    let mut mesh = raw;
    smooth_mesh(&mut mesh, domain, 80, size);
    let (p, q, t, _) =
        pedroso_mesher::sun::improve_quality(&mesh.points, &quad_vec(&mesh), &tri_vec(&mesh), 12);
    mesh = Mesh::new(p, cells_of(&q, &t));
    let (p, q, t, _) = pedroso_mesher::sun::perturb_low_quads(
        &mesh.points,
        &quad_vec(&mesh),
        &tri_vec(&mesh),
        size,
        200,
        Some(domain),
    );
    mesh = Mesh::new(p, cells_of(&q, &t));
    print_row(
        name,
        size,
        "post",
        &mesh,
        domain,
        post_start.elapsed().as_secs_f64(),
    );
    Some(band)
}

fn main() {
    println!(
        "{:12} {:>4} {:<4} {:>8} {:>8} {:>7} {:>7} {:>6} {:>6} {:>6} {:>4} {:<28} {:>6} {:>8} {:>6} {:>8} {:>10}",
        "part", "size", "", "quads", "tri", "inv", "ovl", "minSJ", "minAng", "cracks", "gate", "reason", "secs", "v4", "ar", "armax", "off"
    );
    let mut bands: Vec<(f64, f64)> = Vec::new();
    let mut record = |b: Option<(f64, f64)>| {
        if let Some(v) = b {
            bands.push(v);
        }
    };
    record(run(
        "plate-hole",
        &Domain::plate_with_hole(100.0, 60.0, 10.0, 16).unwrap(),
        4.0,
    ));
    record(run(
        "plate-hole",
        &Domain::plate_with_hole(100.0, 60.0, 10.0, 24).unwrap(),
        2.0,
    ));
    record(run(
        "plate-hole",
        &Domain::plate_with_hole(100.0, 60.0, 10.0, 48).unwrap(),
        1.0,
    ));
    record(run("l-bracket", &l_bracket(), 2.0));
    record(run("ring", &ring(), 1.0));
    record(run("stepped", &stepped(), 1.0));
    println!("--- performance ---");
    record(run(
        "plate-perf",
        &Domain::plate_with_hole(100.0, 60.0, 10.0, 64).unwrap(),
        0.6,
    ));
    if !bands.is_empty() {
        let min = bands.iter().map(|b| b.0).fold(f64::INFINITY, f64::min);
        let max = bands.iter().map(|b| b.1).fold(f64::NEG_INFINITY, f64::max);
        println!("measured raw corner-angle band over the corpus: ({min:.1}°, {max:.1}°)");
    }
}
