#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! W3: the gate-checked quad route, measured on the corpus.
//!
//! Every row is the gate applied to the mesh the run produced: quads, triangles,
//! inverted elements, overlapping pairs, min scaled Jacobian, min corner angle,
//! cracks, verdict and the first reason.
//!
//! The route is exposed as [`quad::mesh_quads`], which returns
//! `Err(QuadError::Unsupported)` when the mesh cannot meet the gate. The rows
//! printed as **UNSUPPORTED** are exactly those configurations: they are not
//! shipped with a failing mesh, they are reported as outside the supported set.

use pedroso_mesher::qmorph::Refusals;
use pedroso_mesher::quad::{self, QuadError, Route, Stats, Timing};
use pedroso_mesher::quality;
use pedroso_mesher::two_d::{Domain, Sizing};
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

/// The parts of a run that the table prints, whether it was accepted or not.
struct Row<'a> {
    gate: &'a verify::Gate,
    raw: &'a verify::Gate,
    stats: &'a Stats,
    timing: &'a Timing,
    refusals: &'a Refusals,
    accepted: bool,
}

fn print_row(name: &str, size: f64, row: &Row) {
    println!(
        "{name:12} {size:4}  W3 raw  quads {:5}  tri {:4}  inv {:3}  ovl {:2}  SJ {:.3}  ang {:5.1}  cracks {:3}  {:4}  {:.2}s/{:.2}s",
        row.raw.quads,
        row.raw.triangles,
        row.raw.inverted,
        row.raw.overlapping,
        row.raw.min_scaled_jacobian,
        row.raw.min_corner_angle_deg,
        row.raw.cracks,
        if row.raw.passes() { "PASS" } else { "FAIL" },
        row.timing.triangulation,
        row.timing.conversion,
    );
    println!(
        "{:12} {:>4}  +W2 sm  quads {:5}  tri {:4}  inv {:3}  ovl {:2}  SJ {:.3}  ang {:5.1}  cracks {:3}  {:4}  | quads-only SJ {:.3} ang {:5.1} | off {:.1e} | rounds {} {:.2}s",
        "",
        "",
        row.gate.quads,
        row.gate.triangles,
        row.gate.inverted,
        row.gate.overlapping,
        row.gate.min_scaled_jacobian,
        row.gate.min_corner_angle_deg,
        row.gate.cracks,
        if row.accepted { "PASS" } else { "UNSUPPORTED" },
        row.stats.quads_only_sj,
        row.stats.quads_only_angle_deg,
        row.stats.boundary_offset,
        row.stats.rounds,
        row.timing.total,
    );
    let mut note = String::new();
    if let Some(two_stage) = &row.stats.two_stage {
        note = format!(
            " | two-stage: front {} core {} ({} comps) matched {} flips {} inverted-core {} rejected {}",
            two_stage.front_quads,
            two_stage.core_triangles,
            two_stage.core_components,
            two_stage.matched_quads,
            two_stage.flips,
            two_stage.inverted_core,
            two_stage.rejected_merge,
        );
    }
    println!(
        "      quality swaps: {} | low repairs: {} | degenerate repairs: {} | perturbations: {} | cavity: {} | parity splits: {}{note}",
        row.stats.quality_swaps,
        row.stats.low_repairs,
        row.stats.degenerate_repairs,
        row.stats.perturbations,
        row.stats.cavity_closures,
        row.stats.parity_splits,
    );
    if let Some(reason) = row.gate.reason() {
        println!("      reason: {reason}");
        if !row.accepted {
            println!("      not supported yet: this configuration is outside the gate");
        }
    }
    println!("      refusals: {:?}", row.refusals);
}

fn run(name: &str, d: &Domain, size: f64, route: Route) {
    let sizing = match Sizing::uniform(size) {
        Ok(sizing) => sizing,
        Err(e) => {
            println!("{name:12} {size:4}  sizing error: {e}");
            return;
        }
    };
    match quad::mesh_quads_with_route(d, &sizing, route) {
        Ok(q) => print_row(
            name,
            size,
            &Row {
                gate: &q.gate,
                raw: &q.raw_gate,
                stats: &q.stats,
                timing: &q.timing,
                refusals: &q.refusals,
                accepted: true,
            },
        ),
        Err(QuadError::Unsupported(u)) => print_row(
            name,
            size,
            &Row {
                gate: &u.gate,
                raw: &u.raw_gate,
                stats: &u.stats,
                timing: &u.timing,
                refusals: &u.refusals,
                accepted: false,
            },
        ),
        Err(e) => println!("{name:12} {size:4}  error: {e}"),
    }
}

fn main() {
    // `cargo run --example qmorph -- two-stage` measures the experimental route;
    // the default is the shipped Q-Morph front + Sun closure.
    let route = if std::env::args().any(|a| a == "two-stage") {
        Route::TwoStage
    } else {
        Route::Front
    };
    println!(
        "{:12} {:>4}  {:>5}  {:>4}  {:>3}  {:>5}  {:>5}  {:>6}  {:>4}",
        "part", "size", "quads", "tri", "inv", "minSJ", "minAng", "cracks", "gate"
    );
    run(
        "plate-hole",
        &Domain::plate_with_hole(100.0, 60.0, 10.0, 16).unwrap(),
        4.0,
        route,
    );
    run(
        "plate-hole",
        &Domain::plate_with_hole(100.0, 60.0, 10.0, 24).unwrap(),
        2.0,
        route,
    );
    run(
        "plate-hole",
        &Domain::plate_with_hole(100.0, 60.0, 10.0, 48).unwrap(),
        1.0,
        route,
    );
    run("l-bracket", &l_bracket(), 2.0, route);
    run("ring", &ring(), 1.0, route);
    run("stepped", &stepped(), 1.0, route);

    println!("--- performance ---");
    run(
        "plate-perf",
        &Domain::plate_with_hole(100.0, 60.0, 10.0, 64).unwrap(),
        0.6,
        route,
    );

    // The whole-mesh quality report of an accepted mesh, as the product uses it.
    if let Ok(q) = quad::mesh_quads(
        &Domain::plate_with_hole(100.0, 60.0, 10.0, 16).unwrap(),
        &Sizing::uniform(4.0).unwrap(),
    ) {
        let report = quality::report(&q.mesh);
        println!(
            "--- accepted: plate-hole 4 mm, {:?}, min SJ {:.3}",
            report.verdict, report.min_scaled_jacobian
        );
    }
}
