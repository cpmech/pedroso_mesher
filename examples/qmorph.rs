#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! W3: Q-Morph conversion measured against the acceptance gate.
//!
//! Every row is the gate applied to the mesh the run produced — quads,
//! triangles, inverted elements, min scaled Jacobian, min corner angle, cracks,
//! pass/fail and the first reason. A "PASS" cannot be read off the intent; it is
//! `verify::gate`.
//!
//! The conversion is followed by an alternating loop: W2 smoothing, then the
//! Sun residual walk again. The smoother moves the nodes that made a pentagon
//! recombination concave, so the last interior slivers can pair; the loop stops
//! when the mesh is all-quad or two consecutive rounds make no progress.

use pedroso_mesher::mesh::{Cell, Kind, Mesh};
use pedroso_mesher::qmorph::convert_with_stats;
use pedroso_mesher::quality;
use pedroso_mesher::sun;
use pedroso_mesher::two_d::{smooth_mesh, triangulate, Domain, Sizing};
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

/// Total order on a float for the best-state comparison (values are finite).
#[derive(Clone, Copy, PartialEq)]
struct OrderedF64(f64);

impl Eq for OrderedF64 {}

impl PartialOrd for OrderedF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .partial_cmp(&other.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

fn quad_cells(mesh: &Mesh) -> Vec<[usize; 4]> {
    mesh.cells
        .iter()
        .filter(|c| c.kind == Kind::Quad4)
        .map(|c| [c.nodes[0], c.nodes[1], c.nodes[2], c.nodes[3]])
        .collect()
}

fn tri_cells(mesh: &Mesh) -> Vec<[usize; 3]> {
    mesh.cells
        .iter()
        .filter(|c| c.kind == Kind::Tri3)
        .map(|c| [c.nodes[0], c.nodes[1], c.nodes[2]])
        .collect()
}

fn run(name: &str, d: &Domain, size: f64) {
    let t0 = std::time::Instant::now();
    let (points, tris) = match triangulate(d, &Sizing::uniform(size).unwrap()) {
        Ok(v) => v,
        Err(e) => {
            println!("{name:12} {size:4}  triangulate error: {e}");
            return;
        }
    };
    let t_tri = t0.elapsed().as_secs_f64();
    let t1 = std::time::Instant::now();
    let ((quads, leftover, points), refusals) = convert_with_stats(&tris, &points, size);
    let t_conv = t1.elapsed().as_secs_f64();

    let mut cells: Vec<Cell> = quads.iter().map(|q| Cell::quad4(*q)).collect();
    cells.extend(leftover.iter().map(|t| Cell::tri3(*t)));
    let mut mesh = Mesh::new(points, cells);
    let gate_raw = verify::gate(&mesh);

    // Alternate smoothing and the residual walk. The first round is the shipped
    // W2 smoother on the conversion output; later rounds widen the smoother's
    // search radius for the last slivers.
    let mut remaining = leftover.len();
    let mut rounds = 0usize;
    let mut stale = 0usize;
    let mut last_remaining = remaining;
    // Each round leaves a valid mesh; keep the best one under the gate's own
    // order (fewest triangles, then highest min angle, then min Jacobian). A
    // wider smoothing pass helps closure but can coarsen a good mesh, and the
    // best state is not always the last one.
    let mut best: Option<(Mesh, verify::Gate)> = None;
    let score = |g: &verify::Gate| {
        (
            g.triangles,
            g.inverted,
            g.cracks,
            std::cmp::Reverse(OrderedF64(g.min_corner_angle_deg)),
            std::cmp::Reverse(OrderedF64(g.min_scaled_jacobian)),
        )
    };
    while remaining > 0 && rounds < 10 && stale < 3 {
        let factor = match rounds {
            0..=1 => 1.0,
            _ => 4.0,
        };
        smooth_mesh(&mut mesh, d, 80, size * factor);
        let q = quad_cells(&mesh);
        let t = tri_cells(&mesh);
        let (p2, q2, t2, _stats) = sun::close_residuals(&mesh.points, &q, &t);
        let mut cells2: Vec<Cell> = q2.iter().map(|q| Cell::quad4(*q)).collect();
        cells2.extend(t2.iter().map(|t| Cell::tri3(*t)));
        mesh = Mesh::new(p2, cells2);
        remaining = t2.len();
        rounds += 1;
        if remaining >= last_remaining {
            stale += 1;
        } else {
            stale = 0;
        }
        last_remaining = remaining;

        let mut candidate = mesh.clone();
        smooth_mesh(&mut candidate, d, 80, size);
        let candidate_gate = verify::gate(&candidate);
        let better = match &best {
            None => true,
            Some((_, current)) => score(&candidate_gate) < score(current),
        };
        if better {
            best = Some((candidate, candidate_gate));
        }
    }
    let t_total = t0.elapsed().as_secs_f64();

    let (mesh, gate) = match best {
        Some(best) => best,
        None => {
            smooth_mesh(&mut mesh, d, 80, size);
            let gate = verify::gate(&mesh);
            (mesh, gate)
        }
    };

    let quad_mesh = Mesh::new(
        mesh.points.clone(),
        mesh.cells
            .iter()
            .filter(|c| c.kind == Kind::Quad4)
            .cloned()
            .collect::<Vec<_>>(),
    );
    let quad_report = quality::report(&quad_mesh);

    println!(
        "{name:12} {size:4}  W3 raw  quads {:5}  tri {:4}  inv {:3}  SJ {:.3}  ang {:5.1}  cracks {:3}  {:4}  {:.2}s/{:.2}s",
        gate_raw.quads,
        gate_raw.triangles,
        gate_raw.inverted,
        gate_raw.min_scaled_jacobian,
        gate_raw.min_corner_angle_deg,
        gate_raw.cracks,
        if gate_raw.passes() { "PASS" } else { "FAIL" },
        t_tri,
        t_conv,
    );
    if let Some(reason) = gate_raw.reason() {
        println!("      reason: {reason}");
    }
    println!(
        "{:12} {:>4}  +W2 sm  quads {:5}  tri {:4}  inv {:3}  SJ {:.3}  ang {:5.1}  cracks {:3}  {:4}  | quads-only SJ {:.3} ang {:5.1} | rounds {rounds} {:.2}s",
        "",
        "",
        gate.quads,
        gate.triangles,
        gate.inverted,
        gate.min_scaled_jacobian,
        gate.min_corner_angle_deg,
        gate.cracks,
        if gate.passes() { "PASS" } else { "FAIL" },
        quad_report.min_scaled_jacobian,
        quad_report.min_corner_angle_deg,
        t_total,
    );
    if let Some(reason) = gate.reason() {
        println!("      reason: {reason}");
    }
    println!("      refusals: {refusals:?}");
}

fn main() {
    println!(
        "{:12} {:>4}  {:>5}  {:>4}  {:>3}  {:>5}  {:>5}  {:>6}  {:>4}",
        "part", "size", "quads", "tri", "inv", "minSJ", "minAng", "cracks", "gate"
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

    // Performance: the gate wants 10 000 quads converted in under 2 s.
    println!("--- performance ---");
    run(
        "plate-perf",
        &Domain::plate_with_hole(100.0, 60.0, 10.0, 64).unwrap(),
        0.6,
    );
}
