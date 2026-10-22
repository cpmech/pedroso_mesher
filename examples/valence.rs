#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Valence distribution: how far the meshes are from the 4-valent ideal that
//! topological clean-up aims at.

use pedroso_mesher::two_d::{mesh_2d, Domain, Sizing};
use std::collections::HashMap;

fn report(name: &str, d: &Domain, size: f64) {
    let r = mesh_2d(d, &Sizing::uniform(size).unwrap()).unwrap();
    let mut valence: HashMap<usize, usize> = HashMap::new();
    for c in &r.mesh.cells {
        for &v in &c.nodes {
            *valence.entry(v).or_insert(0) += 1;
        }
    }
    // boundary vertices have fewer quads; count interior only
    let boundary: std::collections::HashSet<usize> = {
        let mut edges: HashMap<(usize, usize), usize> = HashMap::new();
        for c in &r.mesh.cells {
            for i in 0..c.nodes.len() {
                let (a, b) = (c.nodes[i], c.nodes[(i + 1) % c.nodes.len()]);
                *edges.entry((a.min(b), a.max(b))).or_insert(0) += 1;
            }
        }
        edges
            .iter()
            .filter(|(_, &n)| n == 1)
            .flat_map(|(&(a, b), _)| [a, b])
            .collect()
    };
    let mut hist: HashMap<usize, usize> = HashMap::new();
    for (&v, &n) in &valence {
        if !boundary.contains(&v) {
            *hist.entry(n).or_insert(0) += 1;
        }
    }
    // how many irregular nodes are touched by a leftover triangle?
    let mut touched: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for c in &r.mesh.cells {
        if c.kind == pedroso_mesher::mesh::Kind::Tri3 {
            touched.extend(c.nodes.iter().copied());
        }
    }
    let irregular: Vec<usize> = valence
        .iter()
        .filter(|(v, &n)| !boundary.contains(v) && n != 4)
        .map(|(&v, _)| v)
        .collect();
    let near_tri = irregular.iter().filter(|v| touched.contains(v)).count();
    let interior: usize = hist.values().sum();
    let regular = hist.get(&4).copied().unwrap_or(0);
    let mut parts: Vec<String> = hist
        .iter()
        .map(|(k, v)| format!("{k}:{v}"))
        .collect::<Vec<_>>();
    parts.sort();
    println!(
        "{name:14} size {size:4}  quads {:5}  interior {interior:5}  valence-4 {:5.1}%  [{}]",
        r.mesh.cells.len(),
        100.0 * regular as f64 / interior.max(1) as f64,
        parts.join(" ")
    );
    println!("               clean-up swaps {}", r.stats.cleanup_swaps);
    println!(
        "               irregular interior nodes {}  of which touched by a triangle {} ({:.0}%)",
        irregular.len(),
        near_tri,
        100.0 * near_tri as f64 / irregular.len().max(1) as f64
    );
}

fn main() {
    report(
        "plate-hole",
        &Domain::plate_with_hole(100.0, 60.0, 10.0, 16).unwrap(),
        4.0,
    );
    report(
        "plate-hole",
        &Domain::plate_with_hole(100.0, 60.0, 10.0, 24).unwrap(),
        2.0,
    );
    report(
        "l-bracket",
        &Domain::new(
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
        .unwrap(),
        2.0,
    );
}
