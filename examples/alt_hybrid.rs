#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Measurement of the hybrid route: the seven corpus parts, raw and after the
//! common post-pass, with the shipped `two_d::mesh_2d` as the baseline.
//!
//! Columns per row: part, size, label, quads, triangles, inverted, overlapping,
//! min scaled Jacobian, min corner angle, cracks, verdict, first reason, wall
//! time, interior valence-4 share, boundary offset, p95 aspect ratio, maximum
//! aspect ratio.

use std::collections::HashMap;
use std::time::Instant;

use pedroso_mesher::alt_hybrid;
use pedroso_mesher::mesh::{Cell, Kind, Mesh};
use pedroso_mesher::quality;
use pedroso_mesher::sun;
use pedroso_mesher::two_d::{mesh_2d, smooth_mesh, Domain, Sizing};
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
        .filter_map(|c| c.nodes.as_slice().try_into().ok())
        .collect()
}

fn tri_vec(mesh: &Mesh) -> Vec<[usize; 3]> {
    mesh.cells
        .iter()
        .filter(|c| c.kind == Kind::Tri3)
        .filter_map(|c| c.nodes.as_slice().try_into().ok())
        .collect()
}

fn cells_of(quads: &[[usize; 4]], tris: &[[usize; 3]]) -> Vec<Cell> {
    let mut cells: Vec<Cell> = quads.iter().map(|q| Cell::quad4(*q)).collect();
    cells.extend(tris.iter().map(|t| Cell::tri3(*t)));
    cells
}

/// The common post-pass, exactly as the brief specifies it.
fn common_post(mesh: &mut Mesh, domain: &Domain, size: f64) {
    smooth_mesh(mesh, domain, 80, size);
    let (p, q, t, _) = sun::improve_quality(&mesh.points, &quad_vec(mesh), &tri_vec(mesh), 12);
    *mesh = Mesh::new(p, cells_of(&q, &t));
    let (p, q, t, _) = sun::perturb_low_quads(
        &mesh.points,
        &quad_vec(mesh),
        &tri_vec(mesh),
        size,
        200,
        Some(domain),
    );
    *mesh = Mesh::new(p, cells_of(&q, &t));
}

/// Boundary nodes (endpoints of edges used by exactly one cell).
fn boundary_nodes(mesh: &Mesh) -> Vec<bool> {
    let mut count: HashMap<(usize, usize), usize> = HashMap::new();
    for cell in &mesh.cells {
        for i in 0..cell.nodes.len() {
            let (a, b) = (cell.nodes[i], cell.nodes[(i + 1) % cell.nodes.len()]);
            *count.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    let mut boundary = vec![false; mesh.points.len()];
    for ((a, b), uses) in count {
        if uses == 1 {
            boundary[a] = true;
            boundary[b] = true;
        }
    }
    boundary
}

/// Share of interior nodes with exactly four incident quads; `None` when the
/// mesh has no interior node.
fn valence4(mesh: &Mesh) -> Option<f64> {
    let boundary = boundary_nodes(mesh);
    let mut quads_at: HashMap<usize, usize> = HashMap::new();
    let mut used = vec![false; mesh.points.len()];
    for cell in &mesh.cells {
        for &v in &cell.nodes {
            used[v] = true;
            if cell.kind == Kind::Quad4 {
                *quads_at.entry(v).or_insert(0) += 1;
            }
        }
    }
    let interior: Vec<usize> = (0..mesh.points.len())
        .filter(|&v| used[v] && !boundary[v])
        .collect();
    if interior.is_empty() {
        return None;
    }
    let four = interior
        .iter()
        .filter(|&&v| quads_at.get(&v).copied().unwrap_or(0) == 4)
        .count();
    Some(100.0 * four as f64 / interior.len() as f64)
}

/// 95th percentile and maximum aspect ratio over all cells.
fn aspects(mesh: &Mesh) -> (f64, f64) {
    let mut values: Vec<f64> = mesh
        .cells
        .iter()
        .map(|cell| quality::aspect_ratio(cell, &mesh.points))
        .filter(|value| value.is_finite())
        .collect();
    if values.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let index = ((values.len() - 1) as f64 * 0.95).round() as usize;
    (values[index], values[values.len() - 1])
}

fn print_row(name: &str, size: f64, label: &str, mesh: &Mesh, domain: &Domain, secs: f64) {
    let gate: Gate = verify::gate(mesh);
    let offset = verify::boundary_offset(domain, mesh);
    let v4 = match valence4(mesh) {
        Some(value) => format!("{value:5.1}"),
        None => "  -- ".to_string(),
    };
    let (p95, max) = aspects(mesh);
    let reason = if gate.passes() {
        String::new()
    } else {
        format!("[{}] ", gate.reason().unwrap_or_default())
    };
    println!(
        "{name:11} {size:4.1} {label:9} q={:5} tri={:5} inv={:3} ovl={:2} sj={:.3} ang={:6.1} cr={:3} {:4} {reason}{:6.2}s v4={} off={:.1e} ar={:.1} armax={:.1}",
        gate.quads,
        gate.triangles,
        gate.inverted,
        gate.overlapping,
        gate.min_scaled_jacobian,
        gate.min_corner_angle_deg,
        gate.cracks,
        if gate.passes() { "PASS" } else { "FAIL" },
        secs,
        v4,
        offset,
        p95,
        max,
    );
}

fn run(name: &str, domain: &Domain, size: f64) {
    let sizing = match Sizing::uniform(size) {
        Ok(sizing) => sizing,
        Err(error) => {
            println!("{name:11} {size:4.1} ERROR sizing: {error}");
            return;
        }
    };

    // ---- hybrid raw and post
    let start = Instant::now();
    let raw = match alt_hybrid::mesh(domain, &sizing) {
        Ok(result) => result,
        Err(error) => {
            println!("{name:11} {size:4.1} raw       ERROR {error}");
            return;
        }
    };
    let raw_secs = start.elapsed().as_secs_f64();
    print_row(name, size, "raw", &raw, domain, raw_secs);
    let mut post = raw.clone();
    let start = Instant::now();
    common_post(&mut post, domain, size);
    let post_secs = start.elapsed().as_secs_f64();
    print_row(name, size, "post", &post, domain, raw_secs + post_secs);

    // ---- shipped matching baseline, raw and after the same post-pass
    let start = Instant::now();
    let baseline = match mesh_2d(domain, &sizing) {
        Ok(result) => result.mesh,
        Err(error) => {
            println!("{name:11} {size:4.1} baseline  ERROR {error}");
            return;
        }
    };
    let baseline_secs = start.elapsed().as_secs_f64();
    print_row(name, size, "baseline", &baseline, domain, baseline_secs);
    let mut baseline_post = baseline.clone();
    let start = Instant::now();
    common_post(&mut baseline_post, domain, size);
    let baseline_post_secs = start.elapsed().as_secs_f64();
    print_row(
        name,
        size,
        "base-post",
        &baseline_post,
        domain,
        baseline_secs + baseline_post_secs,
    );
}

fn main() {
    println!(
        "{:11} {:>4} {:9} {:>7} {:>5} {:>4} {:>4} {:>7} {:>7} {:>4} {:4} {:>8} {:>5} {:>8} {:>5} {:>6}",
        "part", "size", "row", "quads", "tri", "inv", "ovl", "minSJ", "minAng", "cr", "gate", "secs", "v4%", "off", "ar95", "armax"
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
