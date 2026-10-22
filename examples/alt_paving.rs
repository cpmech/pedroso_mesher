#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! PAVING route: raw and post measurement on the common corpus.
//!
//! The layout is the one in `alts/BRIEF.md`: one line per part and row, with
//! the gate, the boundary offset, valence-4 share and aspect ratios.

use std::time::Instant;

use pedroso_mesher::alt_paving;
use pedroso_mesher::mesh::{Cell, Kind, Mesh};
use pedroso_mesher::quality;
use pedroso_mesher::sun;
use pedroso_mesher::two_d::{smooth_mesh, Domain, Sizing};
use pedroso_mesher::verify::{self, Gate};

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
    let mut uses: std::collections::HashMap<(usize, usize), usize> =
        std::collections::HashMap::new();
    for cell in &mesh.cells {
        let n = cell.nodes.len();
        for i in 0..n {
            let a = cell.nodes[i];
            let b = cell.nodes[(i + 1) % n];
            *uses.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    let mut boundary = vec![false; mesh.points.len()];
    for ((a, b), n) in uses {
        if n == 1 {
            if a < boundary.len() {
                boundary[a] = true;
            }
            if b < boundary.len() {
                boundary[b] = true;
            }
        }
    }
    let mut valence: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for cell in &mesh.cells {
        if cell.kind != Kind::Quad4 {
            continue;
        }
        for &v in &cell.nodes {
            *valence.entry(v).or_default() += 1;
        }
    }
    let mut interior = 0usize;
    let mut regular = 0usize;
    for (v, &n) in &valence {
        if *v < boundary.len() && !boundary[*v] {
            interior += 1;
            if n == 4 {
                regular += 1;
            }
        }
    }
    if interior == 0 {
        None
    } else {
        Some(100.0 * regular as f64 / interior as f64)
    }
}

/// 95th percentile and maximum aspect ratio over the quads.
fn aspects(mesh: &Mesh) -> (f64, f64) {
    let mut values: Vec<f64> = mesh
        .cells
        .iter()
        .filter(|c| c.kind == Kind::Quad4)
        .map(|c| quality::aspect_ratio(c, &mesh.points))
        .filter(|v| v.is_finite())
        .collect();
    if values.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let at = ((values.len() as f64 - 1.0) * 0.95).round() as usize;
    (
        values[at.min(values.len() - 1)],
        *values.last().unwrap_or(&f64::NAN),
    )
}

#[allow(clippy::too_many_arguments)]
fn print_row(
    part: &str,
    size: f64,
    kind: &str,
    gate: &Gate,
    secs: f64,
    domain: &Domain,
    mesh: &Mesh,
    stats: &alt_paving::Stats,
) {
    let verdict = if gate.passes() { "PASS" } else { "FAIL" };
    let reason = gate.reason().unwrap_or_default();
    let v4 = valence4(mesh)
        .map(|v| format!("{v:.1}"))
        .unwrap_or_else(|| "n/a".to_string());
    let (p95, max) = aspects(mesh);
    println!(
        "{part:12} {size:4} {kind:4} q={:6} tri={:4} inv={:3} ovl={:2} sj={:.3} ang={:5.1} cr={:3} {verdict:4} {reason:24} {secs:.2} v4={v4} off={:.1e} ar={p95:.1} armax={max:.1} rows={} core={} rnd={}",
        gate.quads,
        gate.triangles,
        gate.inverted,
        gate.overlapping,
        gate.min_scaled_jacobian,
        gate.min_corner_angle_deg,
        gate.cracks,
        verify::boundary_offset(domain, mesh),
        stats.row_quads,
        stats.core_quads,
        stats.row_rounds,
    );
}

fn run(name: &str, domain: &Domain, size: f64) {
    let sizing = match Sizing::uniform(size) {
        Ok(sizing) => sizing,
        Err(e) => {
            println!("{name:12} {size:4} raw  ERROR {e}");
            println!("{name:12} {size:4} post ERROR {e}");
            return;
        }
    };
    let start = Instant::now();
    let (raw, stats) = match alt_paving::mesh_with_stats(domain, &sizing) {
        Ok(pair) => pair,
        Err(e) => {
            println!("{name:12} {size:4} raw  ERROR {e}");
            println!("{name:12} {size:4} post ERROR {e}");
            return;
        }
    };
    let elapsed = start.elapsed().as_secs_f64();
    let raw_gate = verify::gate(&raw);
    print_row(name, size, "raw", &raw_gate, elapsed, domain, &raw, &stats);

    let post_start = Instant::now();
    let mut mesh = raw;
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
    let post_gate = verify::gate(&mesh);
    print_row(
        name,
        size,
        "post",
        &post_gate,
        post_start.elapsed().as_secs_f64(),
        domain,
        &mesh,
        &stats,
    );
}

fn main() {
    println!(
        "{:12} {:>4} {:4} {:>7} {:>5} {:>4} {:>3} {:>6} {:>6} {:>4} {:4} {:24} {:>4} {:>5} {:>8} {:>5} {:>7} {:>5} {:>6} {:>6}",
        "part", "size", "row", "quads", "tri", "inv", "ovl", "minSJ", "minAng", "crk", "gate", "reason", "secs", "v4", "offset", "ar", "armax", "rows", "core", "rounds"
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
