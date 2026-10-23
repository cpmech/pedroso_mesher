//! Draw planar domains and meshes to a figure file (PNG/PDF/SVG).
//!
//! The drawing goes through [plotpy](https://docs.rs/plotpy), which writes a
//! Python script and runs `python3` with matplotlib. That is convenient and
//! produces publication-quality figures, but it is a **developer tool**: it needs
//! Python and matplotlib at run time, and it compiles for wasm32 without being
//! usable there (the `python3` call fails and the error is returned).
//!
//! What the figures show:
//!
//! - [`draw_domain`] — the boundary loops of a [`Domain`](crate::two_d::Domain).
//! - [`draw_mesh`] — every cell filled by its **gate margin**
//!   `min(angle/30°, sj/0.2)`, so a cell that clears the gate is green and one
//!   that does not is yellow, orange or red; inverted cells are magenta and
//!   triangles are a separate colour (they fail the all-quad gate by definition).
//!   Boundary edges are drawn thicker.
//! - [`draw_domain_with_mesh`] — the mesh on top of its domain.
//!
//! The margin is the same quantity the acceptance gate measures, so the figure
//! answers "what is going on" in the gate's own terms rather than by eye.

use std::collections::HashMap;

use plotpy::{Plot, StrError};

use crate::mesh::{Cell, Kind, Mesh, Node};
use crate::quality;
use crate::two_d::Domain;

/// How to draw a mesh.
#[derive(Clone, Debug)]
pub struct MeshDrawOptions {
    /// Figure title (avoid `$`, matplotlib reads it as mathtext).
    pub title: String,
    /// Fill each quad by its gate margin. When false every quad is light gray.
    pub color_by_gate_margin: bool,
    /// Draw a dot at every used node (useful for coarse meshes only).
    pub mark_nodes: bool,
    /// Figure size in inches.
    pub figure_inches: (f64, f64),
}

impl Default for MeshDrawOptions {
    fn default() -> Self {
        Self {
            title: String::new(),
            color_by_gate_margin: true,
            mark_nodes: false,
            figure_inches: (9.0, 7.0),
        }
    }
}

/// Draw the boundary loops of a domain.
pub fn draw_domain(domain: &Domain, title: &str, path: &str) -> Result<(), StrError> {
    let segments = domain.segments();
    if segments.is_empty() {
        return Err("the domain has no boundary segments");
    }
    let mut script = String::from("import matplotlib.collections as mc\n");
    script.push_str(&line_collection(&segments, "#111111", 1.6));
    let (xmin, xmax, ymin, ymax) = segment_bounds(&segments);
    let mut plot = Plot::new();
    plot.set_equal_axes(true)
        .set_title(title)
        .set_figure_size_inches(9.0, 7.0)
        .set_range(xmin, xmax, ymin, ymax)
        .extra(&script);
    plot.save(path)
}

/// Draw a mesh, filled by gate margin.
pub fn draw_mesh(mesh: &Mesh, path: &str, opts: &MeshDrawOptions) -> Result<(), StrError> {
    draw_mesh_with_boundary(mesh, None, path, opts)
}

/// Draw a mesh on top of its domain boundary.
pub fn draw_domain_with_mesh(
    domain: &Domain,
    mesh: &Mesh,
    path: &str,
    opts: &MeshDrawOptions,
) -> Result<(), StrError> {
    draw_mesh_with_boundary(mesh, Some(domain), path, opts)
}

fn draw_mesh_with_boundary(
    mesh: &Mesh,
    domain: Option<&Domain>,
    path: &str,
    opts: &MeshDrawOptions,
) -> Result<(), StrError> {
    if mesh.cells.is_empty() {
        return Err("the mesh has no cells");
    }
    let used = used_nodes(mesh);
    let (xmin, xmax, ymin, ymax) = node_bounds(mesh, &used);
    let mut script = String::from("import matplotlib.collections as mc\n");
    script.push_str(&poly_collection(mesh, opts));
    let boundary = boundary_segments(mesh);
    if !boundary.is_empty() {
        script.push_str(&line_collection(&boundary, "#111111", 1.4));
    }
    if let Some(domain) = domain {
        script.push_str(&line_collection(&domain.segments(), "#0055aa", 1.0));
    }
    if opts.mark_nodes && used.len() <= 5000 {
        script.push_str(&node_scatter(mesh, &used));
    }
    let mut plot = Plot::new();
    plot.set_equal_axes(true)
        .set_title(&opts.title)
        .set_figure_size_inches(opts.figure_inches.0, opts.figure_inches.1)
        .set_range(xmin, xmax, ymin, ymax)
        .extra(&script);
    plot.save(path)
}

/// The gate margin of one cell: `min(angle/30°, sj/0.2)`. Negative when the
/// cell is inverted, 0 for a degenerate triangle, and `>= 1` when it clears the
/// gate.
pub fn gate_margin(cell: &Cell, points: &[Node]) -> f64 {
    let sj = quality::scaled_jacobian(cell, points)
        .into_iter()
        .fold(f64::INFINITY, f64::min);
    let angle = quality::corner_angles(cell, points)
        .into_iter()
        .fold(f64::INFINITY, f64::min);
    if !sj.is_finite() || !angle.is_finite() {
        return -1.0;
    }
    (angle / 30.0).min(sj / 0.2)
}

/// Colour for a gate margin. The ramp runs from magenta (inverted) through red,
/// orange and yellow (fails) to two greens (passes).
pub fn margin_color(margin: f64) -> &'static str {
    if margin < 0.0 {
        "#cc00cc"
    } else if margin < 0.5 {
        "#d73027"
    } else if margin < 0.8 {
        "#fc8d59"
    } else if margin < 1.0 {
        "#fee08b"
    } else if margin < 1.5 {
        "#d9ef8b"
    } else {
        "#66bd63"
    }
}

fn poly_collection(mesh: &Mesh, opts: &MeshDrawOptions) -> String {
    let mut verts = String::from("verts = [");
    let mut colors = String::from("face = [");
    for (i, cell) in mesh.cells.iter().enumerate() {
        if cell.kind != Kind::Quad4 && cell.kind != Kind::Tri3 {
            continue;
        }
        let mut v = String::from("[");
        for (k, &node) in cell.nodes.iter().enumerate() {
            let p = mesh.points[node];
            if k > 0 {
                v.push(',');
            }
            v.push_str(&format!("({:.6},{:.6})", p[0], p[1]));
        }
        v.push(']');
        if i > 0 {
            verts.push(',');
            colors.push(',');
        }
        verts.push_str(&v);
        let color = if cell.kind == Kind::Tri3 {
            // a triangle fails the all-quad gate by definition
            "'#b2182b'"
        } else if opts.color_by_gate_margin {
            let c = margin_color(gate_margin(cell, &mesh.points));
            // quote the &'static str literal
            colors.push('\'');
            colors.push_str(c);
            colors.push('\'');
            continue;
        } else {
            "'#dddddd'"
        };
        colors.push_str(color);
    }
    verts.push_str("]\n");
    colors.push_str("]\n");
    let mut script = verts;
    script.push_str(&colors);
    script.push_str(
        "pc = mc.PolyCollection(verts, facecolors=face, edgecolors='#334455', linewidths=0.35)\n\
         plt.gca().add_collection(pc)\n",
    );
    script
}

fn line_collection(segments: &[([f64; 2], [f64; 2])], color: &str, width: f64) -> String {
    let mut script = String::from("segs = [");
    for (i, (a, b)) in segments.iter().enumerate() {
        if i > 0 {
            script.push(',');
        }
        script.push_str(&format!(
            "[({:.6},{:.6}),({:.6},{:.6})]",
            a[0], a[1], b[0], b[1]
        ));
    }
    script.push_str("]\n");
    script.push_str(&format!(
        "lc = mc.LineCollection(segs, colors='{color}', linewidths={width})\n\
         plt.gca().add_collection(lc)\n"
    ));
    script
}

fn node_scatter(mesh: &Mesh, used: &[usize]) -> String {
    let mut script = String::from("nx = [");
    let mut ny = String::from("ny = [");
    for (i, &node) in used.iter().enumerate() {
        let p = mesh.points[node];
        if i > 0 {
            script.push(',');
            ny.push(',');
        }
        script.push_str(&format!("{:.6}", p[0]));
        ny.push_str(&format!("{:.6}", p[1]));
    }
    script.push_str("]\n");
    ny.push_str("]\nplt.plot(nx, ny, '.', color='black', markersize=1.5)\n");
    script.push_str(&ny);
    script
}

fn used_nodes(mesh: &Mesh) -> Vec<usize> {
    let mut used: Vec<usize> = mesh
        .cells
        .iter()
        .flat_map(|c| c.nodes.iter().copied())
        .collect();
    used.sort_unstable();
    used.dedup();
    used
}

fn node_bounds(mesh: &Mesh, used: &[usize]) -> (f64, f64, f64, f64) {
    let (mut xmin, mut xmax, mut ymin, mut ymax) = (
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    );
    for &node in used {
        let p = mesh.points[node];
        xmin = xmin.min(p[0]);
        xmax = xmax.max(p[0]);
        ymin = ymin.min(p[1]);
        ymax = ymax.max(p[1]);
    }
    finish_bounds(xmin, xmax, ymin, ymax)
}

fn segment_bounds(segments: &[([f64; 2], [f64; 2])]) -> (f64, f64, f64, f64) {
    let (mut xmin, mut xmax, mut ymin, mut ymax) = (
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    );
    for (a, b) in segments {
        for p in [a, b] {
            xmin = xmin.min(p[0]);
            xmax = xmax.max(p[0]);
            ymin = ymin.min(p[1]);
            ymax = ymax.max(p[1]);
        }
    }
    finish_bounds(xmin, xmax, ymin, ymax)
}

fn finish_bounds(xmin: f64, xmax: f64, ymin: f64, ymax: f64) -> (f64, f64, f64, f64) {
    let pad = 0.02 * (xmax - xmin).max(ymax - ymin).max(1.0e-12);
    (xmin - pad, xmax + pad, ymin - pad, ymax + pad)
}

fn boundary_segments(mesh: &Mesh) -> Vec<([f64; 2], [f64; 2])> {
    let mut count: HashMap<(usize, usize), usize> = HashMap::new();
    for cell in &mesh.cells {
        for i in 0..cell.nodes.len() {
            let a = cell.nodes[i];
            let b = cell.nodes[(i + 1) % cell.nodes.len()];
            *count.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    let mut edges: Vec<(usize, usize)> = count
        .into_iter()
        .filter(|(_, n)| *n == 1)
        .map(|(e, _)| e)
        .collect();
    edges.sort_unstable();
    edges
        .into_iter()
        .map(|(a, b)| {
            let (p, q) = (mesh.points[a], mesh.points[b]);
            ([p[0], p[1]], [q[0], q[1]])
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;

    fn square() -> Mesh {
        Mesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            vec![Cell::quad4([0, 1, 2, 3])],
        )
    }

    #[test]
    fn a_unit_square_has_a_perfect_margin() {
        let mesh = square();
        let margin = gate_margin(&mesh.cells[0], &mesh.points);
        assert!((margin - 3.0).abs() < 1.0e-12, "{margin}");
        assert_eq!(margin_color(margin), "#66bd63");
    }

    #[test]
    fn a_reversed_quad_is_magenta() {
        let mesh = Mesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            vec![Cell::quad4([1, 0, 3, 2])],
        );
        let margin = gate_margin(&mesh.cells[0], &mesh.points);
        assert!(margin < 0.0, "{margin}");
        assert_eq!(margin_color(margin), "#cc00cc");
    }

    #[test]
    fn the_poly_collection_has_one_entry_per_cell() {
        let mesh = square();
        let opts = MeshDrawOptions::default();
        let script = poly_collection(&mesh, &opts);
        assert!(script.contains("verts = [[(0.000000,0.000000)"));
        assert_eq!(script.matches("(0.000000,0.000000)").count(), 1);
        assert!(script.contains("face = ['#66bd63']"));
    }

    #[test]
    fn the_boundary_of_one_quad_is_four_segments() {
        let mesh = square();
        assert_eq!(boundary_segments(&mesh).len(), 4);
    }

    #[test]
    fn two_quads_share_one_interior_edge() {
        let mesh = Mesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            vec![Cell::quad4([0, 1, 4, 5]), Cell::quad4([1, 2, 3, 4])],
        );
        // 8 cell edges, one shared -> 6 boundary segments
        assert_eq!(boundary_segments(&mesh).len(), 6);
    }
}
