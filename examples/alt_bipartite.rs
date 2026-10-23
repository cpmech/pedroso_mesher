#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Measurement of the right-angled + bipartite-labelling route on the corpus.
//!
//! Each part is run once with [`alt_bipartite::mesh_with_stats`] and reported as
//! **raw**; the same mesh is then run through the common post-pass
//! (`smooth_mesh` 80 passes, `sun::improve_quality` 12 rounds,
//! `sun::perturb_low_quads` 200 rounds) and reported as **post**.
//!
//! Row format:
//! `<part> <size> <raw|post> q= tri= inv= ovl= sj= ang= cr= <verdict> <reason?> <secs> v4= off= ar= armax=`
//! where `v4` is the share of interior nodes with exactly four incident quads,
//! `ar` the p95 aspect ratio and `armax` the largest aspect ratio.

use std::time::Instant;

use pedroso_mesher::alt_bipartite::{self, Stats};
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
    let mut cells: Vec<Cell> = quads.iter().map(|&q| Cell::quad4(q)).collect();
    cells.extend(tris.iter().map(|&t| Cell::tri3(t)));
    cells
}

/// Share of interior nodes with exactly four incident quads, or `None` when the
/// mesh has no interior node.
fn valence4(mesh: &Mesh) -> Option<f64> {
    use std::collections::{HashMap, HashSet};

    let mut edge_uses: HashMap<(usize, usize), usize> = HashMap::new();
    for cell in &mesh.cells {
        for i in 0..cell.nodes.len() {
            let (a, b) = (cell.nodes[i], cell.nodes[(i + 1) % cell.nodes.len()]);
            *edge_uses.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    let boundary: HashSet<usize> = edge_uses
        .iter()
        .filter(|(_, &uses)| uses == 1)
        .flat_map(|(&(a, b), _)| [a, b])
        .collect();
    let mut valence: HashMap<usize, usize> = HashMap::new();
    for cell in &mesh.cells {
        if cell.kind != Kind::Quad4 {
            continue;
        }
        for &v in &cell.nodes {
            *valence.entry(v).or_insert(0) += 1;
        }
    }
    let interior: Vec<usize> = (0..mesh.points.len())
        .filter(|v| !boundary.contains(v) && valence.contains_key(v))
        .collect();
    if interior.is_empty() {
        return None;
    }
    let regular = interior
        .iter()
        .filter(|v| valence.get(v).copied() == Some(4))
        .count();
    Some(100.0 * regular as f64 / interior.len() as f64)
}

/// p95 and max aspect ratio over all cells.
fn aspect_stats(mesh: &Mesh) -> (f64, f64) {
    if mesh.cells.is_empty() {
        return (0.0, 0.0);
    }
    let mut ratios: Vec<f64> = mesh
        .cells
        .iter()
        .map(|cell| quality::aspect_ratio(cell, &mesh.points))
        .collect();
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p95 = ratios[((ratios.len() - 1) as f64 * 0.95).round() as usize];
    let max = ratios.last().copied().unwrap_or(0.0);
    (p95, max)
}

struct Row {
    part: &'static str,
    size: f64,
    raw_gate: verify::Gate,
    post_gate: verify::Gate,
    raw_secs: f64,
    post_secs: f64,
    raw_v4: Option<f64>,
    post_v4: Option<f64>,
    raw_aspect: (f64, f64),
    post_aspect: (f64, f64),
    raw_offset: f64,
    post_offset: f64,
    stats: Stats,
    error: Option<String>,
}

fn print_row(row: &Row, stage: &str) {
    let Some(error) = &row.error else {
        let (gate, secs, v4, aspect, offset) = if stage == "raw" {
            (
                &row.raw_gate,
                row.raw_secs,
                row.raw_v4,
                row.raw_aspect,
                row.raw_offset,
            )
        } else {
            (
                &row.post_gate,
                row.post_secs,
                row.post_v4,
                row.post_aspect,
                row.post_offset,
            )
        };
        let verdict = if gate.passes() { "PASS" } else { "FAIL" };
        let reason = match gate.reason() {
            Some(text) => format!(" {text}"),
            None => String::new(),
        };
        let v4 = match v4 {
            Some(value) => format!("{value:.1}"),
            None => "n/a".to_string(),
        };
        println!(
            "{:<12} {:>4.1} {:<4} q={:<6} tri={:<4} inv={:<4} ovl={:<3} sj={:.3} ang={:.1} cr={:<3} {}{} {:.2} v4={} off={:.1e} ar={:.2} armax={:.2}",
            row.part,
            row.size,
            stage,
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
            v4,
            offset,
            aspect.0,
            aspect.1,
        );
        return;
    };
    println!(
        "{:<12} {:>4.1} {:<4} q=0 tri=0 inv=0 ovl=0 sj=0.000 ang=0.0 cr=0 ERROR {error} {:.2} v4=n/a off=0.0e0 ar=0.00 armax=0.00",
        row.part, row.size, stage, 0.0,
    );
}

fn run(name: &'static str, domain: &Domain, size: f64) -> Row {
    let Ok(sizing) = Sizing::uniform(size) else {
        return Row {
            part: name,
            size,
            raw_gate: verify::gate(&Mesh::default()),
            post_gate: verify::gate(&Mesh::default()),
            raw_secs: 0.0,
            post_secs: 0.0,
            raw_v4: None,
            post_v4: None,
            raw_aspect: (0.0, 0.0),
            post_aspect: (0.0, 0.0),
            raw_offset: 0.0,
            post_offset: 0.0,
            stats: Stats::default(),
            error: Some("bad size".to_string()),
        };
    };
    let started = Instant::now();
    let outcome = alt_bipartite::mesh_with_stats(domain, &sizing);
    let raw_secs = started.elapsed().as_secs_f64();
    match outcome {
        Err(error) => Row {
            part: name,
            size,
            raw_gate: verify::gate(&Mesh::default()),
            post_gate: verify::gate(&Mesh::default()),
            raw_secs,
            post_secs: 0.0,
            raw_v4: None,
            post_v4: None,
            raw_aspect: (0.0, 0.0),
            post_aspect: (0.0, 0.0),
            raw_offset: 0.0,
            post_offset: 0.0,
            stats: Stats::default(),
            error: Some(error),
        },
        Ok((raw, stats)) => {
            let raw_gate = verify::gate(&raw);
            let raw_v4 = valence4(&raw);
            let raw_aspect = aspect_stats(&raw);
            let raw_offset = verify::boundary_offset(domain, &raw);

            let post_started = Instant::now();
            let mut post = raw.clone();
            smooth_mesh(&mut post, domain, 80, size);
            let (points, quads, tris, _) =
                sun::improve_quality(&post.points, &quad_vec(&post), &tri_vec(&post), 12);
            post = Mesh::new(points, cells_of(&quads, &tris));
            let (points, quads, tris, _) = sun::perturb_low_quads(
                &post.points,
                &quad_vec(&post),
                &tri_vec(&post),
                size,
                200,
                Some(domain),
            );
            post = Mesh::new(points, cells_of(&quads, &tris));
            let post_secs = post_started.elapsed().as_secs_f64();

            Row {
                part: name,
                size,
                raw_gate,
                post_gate: verify::gate(&post),
                raw_secs,
                post_secs,
                raw_v4,
                post_v4: valence4(&post),
                raw_aspect,
                post_aspect: aspect_stats(&post),
                raw_offset,
                post_offset: verify::boundary_offset(domain, &post),
                stats,
                error: None,
            }
        }
    }
}

fn main() {
    let parts: Vec<Row> = vec![
        run(
            "plate-hole",
            &Domain::plate_with_hole(100.0, 60.0, 10.0, 16).unwrap(),
            4.0,
        ),
        run(
            "plate-hole",
            &Domain::plate_with_hole(100.0, 60.0, 10.0, 24).unwrap(),
            2.0,
        ),
        run(
            "plate-hole",
            &Domain::plate_with_hole(100.0, 60.0, 10.0, 48).unwrap(),
            1.0,
        ),
        run("l-bracket", &l_bracket(), 2.0),
        run("ring", &ring(), 1.0),
        run("stepped", &stepped(), 1.0),
        run(
            "plate-perf",
            &Domain::plate_with_hole(100.0, 60.0, 10.0, 64).unwrap(),
            0.6,
        ),
    ];

    println!("# RAW");
    for row in &parts {
        print_row(row, "raw");
    }
    println!("# POST");
    for row in &parts {
        print_row(row, "post");
    }

    println!("# route stats (raw)");
    println!(
        "{:<12} {:>4} {:>7} {:>5} {:>6} {:>7} {:>7} {:>6} {:>6} {:>7} {:>7} {:>6} {:>6} {:>4} {:>6} {:>3}",
        "part",
        "size",
        "tri_in",
        "loops",
        "flips",
        "ra_pre",
        "ra_post",
        "mono0",
        "lbl",
        "steiner",
        "mono_l",
        "quads",
        "paired",
        "inv",
        "swaps",
        "uf"
    );
    let mut total_raw = 0.0f64;
    let mut total_post = 0.0f64;
    let mut total_flips = 0usize;
    let mut total_mono = 0usize;
    let mut total_steiner = 0usize;
    let mut total_mono_left = 0usize;
    let mut total_quads = 0usize;
    for row in &parts {
        if let Some(error) = &row.error {
            println!("{:<12} {:>4.1} ERROR {error}", row.part, row.size);
            continue;
        }
        total_raw += row.raw_secs;
        total_post += row.post_secs;
        total_flips += row.stats.right_angle_flips;
        total_mono += row.stats.mono_initial;
        total_steiner += row.stats.steiner_points;
        total_mono_left += row.stats.mono_left;
        total_quads += row.stats.quads;
        println!(
            "{:<12} {:>4.1} {:>7} {:>5} {:>6} {:>7} {:>7} {:>6} {:>6} {:>7} {:>7} {:>6} {:>6.3} {:>4} {:>6} {:>3}",
            row.part,
            row.size,
            row.stats.input_triangles,
            row.stats.boundary_loop_splits,
            row.stats.right_angle_flips,
            row.stats.right_angles_before,
            row.stats.right_angles_after,
            row.stats.mono_initial,
            row.stats.label_flips,
            row.stats.steiner_points,
            row.stats.mono_left,
            row.stats.quads,
            row.stats.paired_fraction(),
            row.stats.inverted_left,
            row.stats.quad_swaps,
            row.stats.unfold_rounds,
        );
    }
    println!(
        "# route totals: raw {:.2}s, post {:.2}s, quads {}, flips {}, mono0 {}, steiner {}, mono_left {}",
        total_raw, total_post, total_quads, total_flips, total_mono, total_steiner, total_mono_left
    );
}
