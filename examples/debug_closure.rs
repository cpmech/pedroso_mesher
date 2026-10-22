#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use pedroso_mesher::two_d::{mesh_2d, Domain, Sizing};
use std::collections::HashMap;

fn main() {
    let d = Domain::plate_with_hole(100.0, 60.0, 10.0, 64).unwrap();
    let r = mesh_2d(&d, &Sizing::uniform(4.0).unwrap()).unwrap();
    println!("stats {:?} topo {:?}", r.stats, r.mesh.topology());
    let mut edges: HashMap<(usize, usize), usize> = HashMap::new();
    for c in &r.mesh.cells {
        for i in 0..c.nodes.len() {
            let a = c.nodes[i];
            let b = c.nodes[(i + 1) % c.nodes.len()];
            *edges.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    let boundary: Vec<(usize, usize)> = edges
        .iter()
        .filter(|(_, &n)| n == 1)
        .map(|(&k, _)| k)
        .collect();
    let mut deg: HashMap<usize, usize> = HashMap::new();
    for (a, b) in &boundary {
        *deg.entry(*a).or_insert(0) += 1;
        *deg.entry(*b).or_insert(0) += 1;
    }
    let bad: Vec<_> = deg.iter().filter(|(_, &n)| n != 2).collect();
    println!(
        "boundary edges {}, vertices with degree != 2: {}",
        boundary.len(),
        bad.len()
    );
    for (v, n) in bad.iter().take(6) {
        println!("   v{v} deg {n} at {:?}", r.mesh.points[**v]);
    }
}
