#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
use pedroso_mesher::two_d::{mesh_2d, Domain, Sizing};
use std::collections::HashMap;

fn main() {
    let d = Domain::plate_with_hole(100.0, 60.0, 10.0, 24).unwrap();
    let r = mesh_2d(&d, &Sizing::uniform(2.0).unwrap().all_quad()).unwrap();
    println!("stats {:?} topo {:?}", r.stats, r.mesh.topology());
    let mut edges: HashMap<(usize, usize), usize> = HashMap::new();
    for c in &r.mesh.cells {
        for i in 0..c.nodes.len() {
            let (a, b) = (c.nodes[i], c.nodes[(i + 1) % c.nodes.len()]);
            *edges.entry((a.min(b), a.max(b))).or_insert(0) += 1;
        }
    }
    let mut deg: HashMap<usize, usize> = HashMap::new();
    for (&(a, b), &n) in &edges {
        if n == 1 {
            *deg.entry(a).or_insert(0) += 1;
            *deg.entry(b).or_insert(0) += 1;
        }
    }
    for (v, n) in deg.iter().filter(|(_, &n)| n != 2) {
        println!("v{v} deg {n} at {:?}", r.mesh.points[*v]);
    }
}
