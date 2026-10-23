#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Measurement table for the cleanup-free quadtree route (`alt_quadtree`).
//!
//! Every part is printed twice: **raw**, the route's output as produced, and
//! **post**, after the exact common post-pass of the brief (80 smoothing
//! iterations, 12 `sun::improve_quality` rounds, 200 `sun::perturb_low_quads`
//! rounds).  The columns are the gate plus `v4` (share of interior nodes with
//! exactly four incident quads), `ar`/`armax` (95th percentile and maximum
//! edge-length aspect ratio) and `off` (`verify::boundary_offset`).
//!
//! `cargo run --release --example alt_quadtree -- sweep` prints the
//! `cell_factor` element-count sweep instead of the corpus table.

use std::time::Instant;

use pedroso_mesher::alt_quadtree::{mesh, mesh_with_cell_factor, DEFAULT_CELL_FACTOR};
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

/// Everything one row of the corpus table prints.
#[derive(Clone, Debug)]
struct Metrics {
    quads: usize,
    triangles: usize,
    inverted: usize,
    overlapping: usize,
    sj: f64,
    angle: f64,
    cracks: usize,
    passes: bool,
    reason: String,
    v4: Option<f64>,
    p95: f64,
    armax: f64,
    offset: f64,
    secs: f64,
}

fn measure(mesh: &Mesh, domain: &Domain, secs: f64) -> Metrics {
    let gate = verify::gate(mesh);
    let (p95, armax) = aspect_stats(mesh);
    Metrics {
        quads: gate.quads,
        triangles: gate.triangles,
        inverted: gate.inverted,
        overlapping: gate.overlapping,
        sj: gate.min_scaled_jacobian,
        angle: gate.min_corner_angle_deg,
        cracks: gate.cracks,
        passes: gate.passes(),
        reason: gate.reason().unwrap_or_default(),
        v4: valence4(mesh),
        p95,
        armax,
        offset: verify::boundary_offset(domain, mesh),
        secs,
    }
}

fn print_row(name: &str, size: f64, tag: &str, m: &Metrics) {
    let verdict = if m.passes { "PASS" } else { "FAIL" };
    let v4_text = match m.v4 {
        Some(v) => format!("{v:4.1}%"),
        None => "  -- ".to_string(),
    };
    println!(
        "{name:12} {size:4.1} {tag:4} q={:6} tri={:4} inv={:3} ovl={:3} sj={:.3} ang={:5.1} cr={:3} {} {:<28} {:5.2}s v4={} ar={:5.2} armax={:7.2} off={:.1e}",
        m.quads,
        m.triangles,
        m.inverted,
        m.overlapping,
        m.sj,
        m.angle,
        m.cracks,
        verdict,
        m.reason,
        m.secs,
        v4_text,
        m.p95,
        m.armax,
        m.offset,
    );
}

/// Boundary-layer statistics of the raw mesh: quads with at least one boundary
/// node, and the corner angles of those quads.
struct BoundaryStats {
    quads: usize,
    share: f64,
    min_angle: f64,
    mean_angle: f64,
    max_angle: f64,
    under_45: f64,
    under_30: f64,
}

fn boundary_stats(mesh: &Mesh) -> BoundaryStats {
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
    let mut boundary_quads = 0usize;
    let mut angles: Vec<f64> = Vec::new();
    let mut under_45 = 0usize;
    let mut under_30 = 0usize;
    let mut total = 0usize;
    for cell in &mesh.cells {
        total += 1;
        if !cell.nodes.iter().any(|&v| on_boundary[v]) {
            continue;
        }
        if cell.kind != Kind::Quad4 {
            continue;
        }
        boundary_quads += 1;
        for a in quality::corner_angles(cell, &mesh.points) {
            if a.is_finite() {
                angles.push(a);
                if a < 45.0 {
                    under_45 += 1;
                }
                if a < 30.0 {
                    under_30 += 1;
                }
            }
        }
    }
    let count = angles.len();
    let sum: f64 = angles.iter().sum();
    let min_angle = angles.iter().copied().fold(f64::INFINITY, f64::min);
    let max_angle = angles.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let share = if total == 0 {
        0.0
    } else {
        100.0 * boundary_quads as f64 / total as f64
    };
    BoundaryStats {
        quads: boundary_quads,
        share,
        min_angle: if count == 0 { f64::NAN } else { min_angle },
        mean_angle: if count == 0 {
            f64::NAN
        } else {
            sum / count as f64
        },
        max_angle: if count == 0 { f64::NAN } else { max_angle },
        under_45: if count == 0 {
            0.0
        } else {
            100.0 * under_45 as f64 / count as f64
        },
        under_30: if count == 0 {
            0.0
        } else {
            100.0 * under_30 as f64 / count as f64
        },
    }
}

struct RunReport {
    band: (f64, f64),
    boundary: BoundaryStats,
}

fn run(name: &str, domain: &Domain, size: f64) -> Option<RunReport> {
    let sizing = match Sizing::uniform(size) {
        Ok(s) => s,
        Err(e) => {
            println!("{name:12} {size:4.1} sizing error: {e}");
            return None;
        }
    };
    let start = Instant::now();
    let raw = match mesh(domain, &sizing) {
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
    let boundary = boundary_stats(&raw);
    let raw_metrics = measure(&raw, domain, secs);
    print_row(name, size, "raw", &raw_metrics);

    // The common post-pass, exactly as specified in the brief.
    let post_start = Instant::now();
    let mut post = raw;
    smooth_mesh(&mut post, domain, 80, size);
    let (p, q, t, _) =
        pedroso_mesher::sun::improve_quality(&post.points, &quad_vec(&post), &tri_vec(&post), 12);
    post = Mesh::new(p, cells_of(&q, &t));
    let (p, q, t, _) = pedroso_mesher::sun::perturb_low_quads(
        &post.points,
        &quad_vec(&post),
        &tri_vec(&post),
        size,
        200,
        Some(domain),
    );
    post = Mesh::new(p, cells_of(&q, &t));
    let post_metrics = measure(&post, domain, post_start.elapsed().as_secs_f64());
    print_row(name, size, "post", &post_metrics);
    Some(RunReport { band, boundary })
}

/// The seven corpus parts, in the brief's order.
fn corpus() -> Vec<(&'static str, Domain, f64)> {
    vec![
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
    ]
}

fn corpus_table() {
    println!(
        "{:12} {:>4} {:<4} {:>8} {:>8} {:>7} {:>7} {:>6} {:>6} {:>6} {:>4} {:<28} {:>6} {:>8} {:>6} {:>8} {:>10}",
        "part", "size", "", "quads", "tri", "inv", "ovl", "minSJ", "minAng", "cracks", "gate", "reason", "secs", "v4", "ar", "armax", "off"
    );
    let mut bands: Vec<(f64, f64)> = Vec::new();
    let mut boundaries: Vec<(&str, BoundaryStats)> = Vec::new();
    for (name, domain, size) in corpus() {
        if let Some(report) = run(name, &domain, size) {
            bands.push(report.band);
            if name != "l-bracket" && name != "stepped" {
                boundaries.push((name, report.boundary));
            }
        }
    }
    if !bands.is_empty() {
        let min = bands.iter().map(|b| b.0).fold(f64::INFINITY, f64::min);
        let max = bands.iter().map(|b| b.1).fold(f64::NEG_INFINITY, f64::max);
        println!("measured raw corner-angle band over the corpus: ({min:.1}°, {max:.1}°)");
    }
    println!("--- boundary layer (raw, default cell_factor) ---");
    println!(
        "{:12} {:>8} {:>7} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "part", "bnd_quad", "share", "ang_min", "ang_mean", "ang_max", "under45", "under30"
    );
    for (name, b) in &boundaries {
        println!(
            "{name:12} {:8} {:6.1}% {:8.1} {:8.1} {:8.1} {:7.1}% {:7.1}%",
            b.quads, b.share, b.min_angle, b.mean_angle, b.max_angle, b.under_45, b.under_30
        );
    }
}

/// The `cell_factor` sweep: element count versus quality, raw meshes only.
fn sweep() {
    let factors = [1.0, 1.25, 1.5, 1.75, 2.0];
    let parts = corpus();
    println!(
        "{:>5} {:>6} {:>9} {:>6} {:>8} {:>8} {:>8} {:>8} {:>9} {:>7}",
        "factor", "pass", "quads", "tri", "minSJ", "minAng", "v4min", "p95max", "armaxmax", "secs"
    );
    let mut best: Option<(f64, usize)> = None;
    let mut per_factor: Vec<(f64, Vec<Metrics>)> = Vec::new();
    for &factor in &factors {
        let mut rows = Vec::new();
        let mut total_secs = 0.0;
        for (name, domain, size) in &parts {
            let sizing = match Sizing::uniform(*size) {
                Ok(s) => s,
                Err(e) => {
                    println!("{name} {size} sizing error: {e}");
                    continue;
                }
            };
            let start = Instant::now();
            match mesh_with_cell_factor(domain, &sizing, factor) {
                Ok(m) => {
                    let secs = start.elapsed().as_secs_f64();
                    total_secs += secs;
                    rows.push(measure(&m, domain, secs));
                }
                Err(e) => {
                    println!("factor {factor:.2} {name} {size} ERROR {e}");
                }
            }
        }
        let pass = rows.iter().filter(|m| m.passes).count();
        let quads: usize = rows.iter().map(|m| m.quads).sum();
        let tris: usize = rows.iter().map(|m| m.triangles).sum();
        let min_sj = rows.iter().map(|m| m.sj).fold(f64::INFINITY, f64::min);
        let min_ang = rows.iter().map(|m| m.angle).fold(f64::INFINITY, f64::min);
        let v4min = rows
            .iter()
            .filter_map(|m| m.v4)
            .fold(f64::INFINITY, f64::min);
        let p95max = rows.iter().map(|m| m.p95).fold(0.0, f64::max);
        let armax = rows.iter().map(|m| m.armax).fold(0.0, f64::max);
        println!(
            "{factor:5.2} {pass:>3}/{} {quads:9} {tris:6} {min_sj:8.3} {min_ang:8.1} {v4min:8.1} {p95max:8.2} {armax:9.2} {total_secs:7.2}",
            rows.len()
        );
        if pass == rows.len() && min_ang >= 30.0 && min_sj >= 0.2 {
            best = Some((factor, quads));
        }
        per_factor.push((factor, rows));
    }
    println!();
    println!("--- quads per part ---");
    print!("{:>5}", "factor");
    for (name, _, size) in &parts {
        print!(" {name}_{size}");
    }
    println!();
    for (factor, rows) in &per_factor {
        print!("{factor:5.2}");
        for m in rows {
            print!(" {:>14}", m.quads);
        }
        println!();
    }
    println!();
    println!("--- minimum corner angle per part (degrees) ---");
    print!("{:>5}", "factor");
    for (name, _, size) in &parts {
        print!(" {name}_{size}");
    }
    println!();
    for (factor, rows) in &per_factor {
        print!("{factor:5.2}");
        for m in rows {
            print!(" {:>14.1}", m.angle);
        }
        println!();
    }
    match best {
        Some((factor, quads)) => println!(
            "largest gate-passing factor (coarse option): cell_factor = {factor:.2} (7/7 raw PASS, min angle >= 30, SJ >= 0.2; {quads} quads on the corpus); committed default = {DEFAULT_CELL_FACTOR:.2}"
        ),
        None => println!("no swept factor keeps 7/7 raw PASS"),
    }
}

fn main() {
    if std::env::args().any(|a| a == "sweep") {
        sweep();
    } else {
        println!("cell_factor = {DEFAULT_CELL_FACTOR:.2}");
        corpus_table();
    }
}
