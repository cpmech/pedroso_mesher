#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Measurement of the untangle + singularity-reduction post-pass on the corpus.
//!
//! Two rows per part, as the brief asks:
//!
//! - **raw** — the mesh the shipped product route (`quad::mesh_quads`) produced
//!   *before* this pass. On the failing configurations that is the mesh carried
//!   by `QuadError::Unsupported`, which is exactly the input of the post-pass.
//! - **post** — this route's mesh (`alt_untangle::mesh`: product pipeline +
//!   untangling + singularity reduction) after the exact common post-pass of
//!   the brief (80 smoothing iterations, `sun::improve_quality` ×12,
//!   `sun::perturb_low_quads` ×200).
//!
//! The columns are the brief's (`q`, `tri`, `inv`, `ovl`, `sj`, `ang`, `cr`,
//! verdict, reason, seconds, `v4`, `off`) plus the orchestrator's `ar` (p95
//! aspect ratio) and `armax`.
//!
//! `v4` is the share of **interior** nodes with exactly four incident quads.
//! `ar`/`armax` use `quality::aspect_ratio` over all cells (longest edge over
//! shortest edge); `ar` is the nearest-rank 95th percentile.

use pedroso_mesher::alt_untangle;
use pedroso_mesher::mesh::{Cell, Kind, Mesh};
use pedroso_mesher::quad::{self, QuadError};
use pedroso_mesher::quality;
use pedroso_mesher::sun;
use pedroso_mesher::two_d::{smooth_mesh, Domain, Sizing};
use pedroso_mesher::verify;
use std::collections::HashMap;

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

/// The exact common post-pass from the brief.
fn common_post(domain: &Domain, size: f64, mesh: &Mesh) -> Mesh {
    let mut mesh = mesh.clone();
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
    Mesh::new(p, cells_of(&q, &t))
}

fn aspect_stats(mesh: &Mesh) -> (f64, f64) {
    if mesh.cells.is_empty() {
        return (0.0, 0.0);
    }
    let mut values: Vec<f64> = mesh
        .cells
        .iter()
        .map(|c| quality::aspect_ratio(c, &mesh.points))
        .filter(|v| v.is_finite())
        .collect();
    if values.is_empty() {
        return (f64::INFINITY, f64::INFINITY);
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let index = ((values.len() as f64 * 0.95).ceil() as usize)
        .saturating_sub(1)
        .min(values.len() - 1);
    (values[index], *values.last().unwrap_or(&0.0))
}

/// Share of interior nodes with exactly four incident quads, in percent.
fn valence4(mesh: &Mesh) -> f64 {
    let quads = quad_vec(mesh);
    let tris = tri_vec(mesh);
    let n = mesh.points.len();
    let mut count: HashMap<(usize, usize), usize> = HashMap::new();
    for c in &mesh.cells {
        for i in 0..c.nodes.len() {
            let a = c.nodes[i];
            let b = c.nodes[(i + 1) % c.nodes.len()];
            *count.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    let mut boundary = vec![false; n];
    for ((a, b), uses) in count {
        if uses == 1 {
            if let Some(f) = boundary.get_mut(a) {
                *f = true;
            }
            if let Some(f) = boundary.get_mut(b) {
                *f = true;
            }
        }
    }
    let mut incident = vec![0usize; n];
    for q in &quads {
        for &v in q {
            if v < n {
                incident[v] += 1;
            }
        }
    }
    let mut interior = 0usize;
    let mut regular = 0usize;
    for v in 0..n {
        if boundary.get(v).copied().unwrap_or(true) {
            continue;
        }
        // A node incident to a triangle has no well-defined quad valence; count
        // it as interior but not regular (the v4 definition is quads only).
        interior += 1;
        if incident[v] == 4 && !tris.iter().any(|t| t.contains(&v)) {
            regular += 1;
        }
    }
    if interior == 0 {
        0.0
    } else {
        100.0 * regular as f64 / interior as f64
    }
}

struct Row {
    gate: verify::Gate,
    v4: f64,
    ar: f64,
    armax: f64,
    offset: f64,
    secs: f64,
}

fn print_row(part: &str, size: f64, stage: &str, row: &Row) {
    let verdict = if row.gate.passes() { "PASS" } else { "FAIL" };
    let reason = match row.gate.reason() {
        Some(r) => r.replace(' ', "_"),
        None => "-".to_string(),
    };
    println!(
        "{part:12} {size:4} {stage:4} q={:5} tri={:3} inv={:2} ovl={:2} sj={:.3} ang={:5.1} cr={:2} {verdict} {reason} {:.2} v4={:.1} off={:.1e} ar={:.1} armax={:.1}",
        row.gate.quads,
        row.gate.triangles,
        row.gate.inverted,
        row.gate.overlapping,
        row.gate.min_scaled_jacobian,
        row.gate.min_corner_angle_deg,
        row.gate.cracks,
        row.secs,
        row.v4,
        row.offset,
        row.ar,
        row.armax,
    );
}

fn measure(domain: &Domain, mesh: &Mesh, secs: f64) -> Row {
    let gate = verify::gate(mesh);
    let (ar, armax) = aspect_stats(mesh);
    Row {
        gate,
        v4: valence4(mesh),
        ar,
        armax,
        offset: verify::boundary_offset(domain, mesh),
        secs,
    }
}

fn run(part: &str, domain: &Domain, size: f64) {
    let sizing = match Sizing::uniform(size) {
        Ok(s) => s,
        Err(e) => {
            println!("{part:12} {size:4} sizing error: {e}");
            return;
        }
    };
    // The raw row: the shipped product mesh, before this post-pass.
    let t0 = std::time::Instant::now();
    let raw = match quad::mesh_quads(domain, &sizing) {
        Ok(q) => q.mesh,
        Err(QuadError::Unsupported(u)) => u.mesh,
        Err(e) => {
            println!("{part:12} {size:4} raw ERROR {e}");
            return;
        }
    };
    let raw_secs = t0.elapsed().as_secs_f64();
    print_row(part, size, "raw", &measure(domain, &raw, raw_secs));

    // The post row: product + untangle + singularity reduction + the common
    // post-pass. `mesh_with_outcome` also gives the improved mesh before the
    // common pass, which is the route's own output.
    let t1 = std::time::Instant::now();
    let outcome = match alt_untangle::mesh_with_outcome(domain, &sizing) {
        Ok(o) => o,
        Err(e) => {
            println!("{part:12} {size:4} post ERROR {e}");
            return;
        }
    };
    let own = measure(
        domain,
        &outcome.improved,
        outcome.base_secs + outcome.improve_secs,
    );
    print_row(part, size, "alt", &own);
    let t2 = std::time::Instant::now();
    let post = common_post(domain, size, &outcome.improved);
    let post_secs = t1.elapsed().as_secs_f64();
    let _ = t2;
    let mut row = measure(domain, &post, post_secs);
    row.secs = post_secs;
    print_row(part, size, "post", &row);
    println!(
        "    swaps={} moves={} base={:.2}s pass={:.2}s",
        outcome.swaps, outcome.moves, outcome.base_secs, outcome.improve_secs
    );
}

fn main() {
    println!(
        "{:12} {:>4} {:>4}  {:>6} {:>5} {:>4} {:>4} {:>5} {:>6} {:>4} {:>5} {:>6} {:>5} {:>6} {:>6} {:>6} {:>6}",
        "part", "size", "row", "quads", "tri", "inv", "ovl", "minSJ", "minAng", "cr", "gate",
        "reason", "secs", "v4", "off", "ar", "armax"
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
    println!("--- performance ---");
    run(
        "plate-perf",
        &Domain::plate_with_hole(100.0, 60.0, 10.0, 64).unwrap(),
        0.6,
    );
}
