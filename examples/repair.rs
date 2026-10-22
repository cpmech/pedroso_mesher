#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
//! Spike 1 for P2: can a local repair absorb the leftover triangles?
//!
//! For each leftover triangle we count how many of its edges are shared with
//! another leftover (a cluster) and how many with a quad. A leftover whose three
//! neighbours are quads is *isolated*: no local operation can pair it, because
//! pairing always removes two triangles and the local count is odd.

use pedroso_mesher::mesh::Kind;
use pedroso_mesher::two_d::{mesh_2d, Domain, Sizing};
use std::collections::HashMap;

fn analyse(name: &str, d: &Domain, size: f64) {
    let r = mesh_2d(d, &Sizing::uniform(size).unwrap()).unwrap();
    // edge -> (triangles, quads) incidence, from the mesh
    let mut edge_tris: HashMap<(usize, usize), usize> = HashMap::new();
    let mut edge_quads: HashMap<(usize, usize), usize> = HashMap::new();
    for c in &r.mesh.cells {
        for i in 0..c.nodes.len() {
            let (a, b) = (c.nodes[i], c.nodes[(i + 1) % c.nodes.len()]);
            let key = (a.min(b), a.max(b));
            match c.kind {
                Kind::Tri3 => *edge_tris.entry(key).or_insert(0) += 1,
                Kind::Quad4 => *edge_quads.entry(key).or_insert(0) += 1,
                _ => {}
            }
        }
    }
    let mut isolated = 0usize;
    let mut clustered = 0usize;
    let mut boundary = 0usize;
    for tri in &r.unpaired {
        let mut shared_with_tri = 0;
        let mut free = 0;
        for i in 0..3 {
            let (a, b) = (tri[i], tri[(i + 1) % 3]);
            let key = (a.min(b), a.max(b));
            if edge_tris.get(&key).copied().unwrap_or(0) > 1 {
                shared_with_tri += 1;
            } else if edge_quads.get(&key).copied().unwrap_or(0) == 0 {
                free += 1;
            }
        }
        if shared_with_tri > 0 {
            clustered += 1;
        } else {
            isolated += 1;
        }
        if free > 0 {
            boundary += 1;
        }
    }
    let total = r.unpaired.len();
    println!(
        "{name:14} size {size:4}  leftovers {total:4}  isolated {isolated:4} ({:5.1}%)  clustered {clustered:3}  touching boundary {boundary:3}",
        100.0 * isolated as f64 / total.max(1) as f64
    );
}

fn main() {
    analyse(
        "plate-hole",
        &Domain::plate_with_hole(100.0, 60.0, 10.0, 16).unwrap(),
        4.0,
    );
    analyse(
        "plate-hole",
        &Domain::plate_with_hole(100.0, 60.0, 10.0, 24).unwrap(),
        2.0,
    );
    analyse(
        "plate-hole",
        &Domain::plate_with_hole(100.0, 60.0, 10.0, 48).unwrap(),
        1.0,
    );
    analyse(
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
