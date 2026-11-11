//! Pure-Rust deterministic SVG renderer for 2D/3D domains and meshes.
//!
//! Replaces external plotting subprocesses so figures render natively in pure
//! Rust (including on `wasm32-unknown-unknown` via [`render_mesh_svg`] and
//! [`render_domain_svg`]).

use std::collections::BTreeMap;
use std::fmt::Write as _;

use crate::mesh::{Cell, Kind, Mesh, Node};
use crate::quality;
use crate::two_d::Domain;

/// Static error message alias for drawing operations.
pub type StrError = &'static str;

/// Options controlling SVG mesh rendering.
#[derive(Clone, Debug)]
pub struct MeshDrawOptions {
    /// Optional title displayed above the figure.
    pub title: String,
    /// Fill cells by their quality gate margin `min(angle / 30°, sj / 0.2)`.
    pub color_by_gate_margin: bool,
    /// Draw a small marker at each referenced vertex.
    pub mark_nodes: bool,
    /// Figure size in inches `(width, height)`, mapped at `96` px/inch.
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

/// Gate margin of one cell: `min(angle / 30°, sj / 0.2)`.
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

/// Color hex code for a gate margin value.
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

/// Render a planar domain boundary to an SVG string.
pub fn render_domain_svg(
    domain: &Domain,
    title: &str,
    figure_inches: (f64, f64),
) -> Result<String, StrError> {
    let segments = domain.segments();
    if segments.is_empty() {
        return Err("the domain has no boundary segments");
    }
    let bounds = segment_bounds(&segments);
    let layout = SvgLayout::new(bounds, figure_inches, !title.is_empty());
    let mut svg = layout.begin(title);
    write_segments(&mut svg, &layout, &segments, "#111111", 1.6);
    layout.finish(&mut svg);
    Ok(svg)
}

/// Render a mesh (with optional reference domain overlay) to an SVG string.
pub fn render_mesh_svg(
    mesh: &Mesh,
    domain: Option<&Domain>,
    opts: &MeshDrawOptions,
) -> Result<String, StrError> {
    if mesh.cells.is_empty() {
        return Err("the mesh has no cells");
    }
    let used = used_nodes(mesh);
    if used.is_empty() {
        return Err("the mesh has no referenced nodes");
    }
    let bounds = node_bounds(mesh, &used);
    let layout = SvgLayout::new(bounds, opts.figure_inches, !opts.title.is_empty());
    let mut svg = layout.begin(&opts.title);

    for cell in &mesh.cells {
        let corner_count = match cell.kind {
            Kind::Tri3 | Kind::Tri6 => 3,
            Kind::Qua4 | Kind::Qua8 => 4,
            _ => continue,
        };
        if cell.nodes.len() < corner_count {
            continue;
        }
        let fill = if matches!(cell.kind, Kind::Tri3 | Kind::Tri6) {
            if opts.color_by_gate_margin {
                "#b2182b"
            } else {
                "#e5e7eb"
            }
        } else if opts.color_by_gate_margin {
            margin_color(gate_margin(cell, &mesh.points))
        } else {
            "#dddddd"
        };

        let _ = write!(
            svg,
            "  <polygon fill=\"{fill}\" stroke=\"#334455\" stroke-width=\"0.6\" points=\""
        );
        for i in 0..corner_count {
            let p = mesh.points[cell.nodes[i]];
            let (sx, sy) = layout.project([p[0], p[1]]);
            if i > 0 {
                svg.push(' ');
            }
            let _ = write!(svg, "{sx:.2},{sy:.2}");
        }
        svg.push_str("\"/>\n");
    }

    let boundary = boundary_segments(mesh);
    if !boundary.is_empty() {
        write_segments(&mut svg, &layout, &boundary, "#111111", 1.4);
    }
    if let Some(dom) = domain {
        write_segments(&mut svg, &layout, &dom.segments(), "#0055aa", 1.0);
    }
    if opts.mark_nodes && used.len() <= 5000 {
        svg.push_str("  <g fill=\"#000000\">\n");
        for &node in &used {
            let p = mesh.points[node];
            let (sx, sy) = layout.project([p[0], p[1]]);
            let _ = writeln!(svg, "    <circle cx=\"{sx:.2}\" cy=\"{sy:.2}\" r=\"1.5\"/>");
        }
        svg.push_str("  </g>\n");
    }

    layout.finish(&mut svg);
    Ok(svg)
}

/// Write a domain SVG figure to `path`.
pub fn draw_domain(domain: &Domain, title: &str, path: &str) -> Result<(), StrError> {
    let svg = render_domain_svg(domain, title, (9.0, 7.0))?;
    std::fs::write(svg_path(path), svg).map_err(|_| "cannot write SVG file")
}

/// Write a mesh SVG figure to `path`.
pub fn draw_mesh(mesh: &Mesh, path: &str, opts: &MeshDrawOptions) -> Result<(), StrError> {
    let svg = render_mesh_svg(mesh, None, opts)?;
    std::fs::write(svg_path(path), svg).map_err(|_| "cannot write SVG file")
}

/// Write a mesh with domain boundary overlay to `path`.
pub fn draw_domain_with_mesh(
    domain: &Domain,
    mesh: &Mesh,
    path: &str,
    opts: &MeshDrawOptions,
) -> Result<(), StrError> {
    let svg = render_mesh_svg(mesh, Some(domain), opts)?;
    std::fs::write(svg_path(path), svg).map_err(|_| "cannot write SVG file")
}

fn svg_path(path: &str) -> String {
    if path.ends_with(".svg") {
        path.to_string()
    } else if let Some(stem) = path.strip_suffix(".png") {
        format!("{stem}.svg")
    } else {
        path.to_string()
    }
}

struct SvgLayout {
    width: f64,
    height: f64,
    xmin: f64,
    ymax: f64,
    scale: f64,
    offset_x: f64,
    offset_y: f64,
}

impl SvgLayout {
    fn new(bounds: (f64, f64, f64, f64), inches: (f64, f64), has_title: bool) -> Self {
        let width = (inches.0.max(2.0)) * 96.0;
        let height = (inches.1.max(2.0)) * 96.0;
        let margin_x = 24.0;
        let top_margin = if has_title { 44.0 } else { 20.0 };
        let bottom_margin = 20.0;
        let avail_w = (width - 2.0 * margin_x).max(10.0);
        let avail_h = (height - top_margin - bottom_margin).max(10.0);

        let (xmin, xmax, ymin, ymax) = bounds;
        let dx = (xmax - xmin).max(1.0e-12);
        let dy = (ymax - ymin).max(1.0e-12);
        let scale = (avail_w / dx).min(avail_h / dy);

        let offset_x = margin_x + 0.5 * (avail_w - dx * scale);
        let offset_y = top_margin + 0.5 * (avail_h - dy * scale);
        Self {
            width,
            height,
            xmin,
            ymax,
            scale,
            offset_x,
            offset_y,
        }
    }

    fn project(&self, p: [f64; 2]) -> (f64, f64) {
        let sx = self.offset_x + (p[0] - self.xmin) * self.scale;
        let sy = self.offset_y + (self.ymax - p[1]) * self.scale;
        (sx, sy)
    }

    fn begin(&self, title: &str) -> String {
        let mut out = String::with_capacity(4096);
        let _ = writeln!(
            out,
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {:.0} {:.0}\" width=\"{:.0}\" height=\"{:.0}\">",
            self.width, self.height, self.width, self.height
        );
        let _ = writeln!(
            out,
            "  <rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>"
        );
        if !title.is_empty() {
            let escaped = escape_xml(title);
            let _ = writeln!(
                out,
                "  <text x=\"{:.1}\" y=\"26\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"14\" fill=\"#111827\">{escaped}</text>",
                0.5 * self.width
            );
        }
        out
    }

    fn finish(&self, out: &mut String) {
        out.push_str("</svg>\n");
    }
}

fn escape_xml(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            _ => out.push(ch),
        }
    }
    out
}

fn write_segments(
    svg: &mut String,
    layout: &SvgLayout,
    segments: &[([f64; 2], [f64; 2])],
    color: &str,
    width: f64,
) {
    if segments.is_empty() {
        return;
    }
    let _ = write!(
        svg,
        "  <path fill=\"none\" stroke=\"{color}\" stroke-width=\"{width:.2}\" stroke-linecap=\"round\" d=\""
    );
    for (i, &(a, b)) in segments.iter().enumerate() {
        let (ax, ay) = layout.project(a);
        let (bx, by) = layout.project(b);
        if i > 0 {
            svg.push(' ');
        }
        let _ = write!(svg, "M{ax:.2},{ay:.2}L{bx:.2},{by:.2}");
    }
    svg.push_str("\"/>\n");
}

fn used_nodes(mesh: &Mesh) -> Vec<usize> {
    let mut used: Vec<usize> = mesh
        .cells
        .iter()
        .flat_map(|c| c.nodes.iter().copied())
        .filter(|&idx| idx < mesh.points.len())
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

/// Extract deterministic boundary segments (edges referenced by exactly one planar cell).
pub fn boundary_segments(mesh: &Mesh) -> Vec<([f64; 2], [f64; 2])> {
    let mut count: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for cell in &mesh.cells {
        let corner_count = match cell.kind {
            Kind::Tri3 | Kind::Tri6 => 3,
            Kind::Qua4 | Kind::Qua8 => 4,
            _ => continue,
        };
        if cell.nodes.len() < corner_count {
            continue;
        }
        for i in 0..corner_count {
            let a = cell.nodes[i];
            let b = cell.nodes[(i + 1) % corner_count];
            *count.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    count
        .into_iter()
        .filter(|(_, n)| *n == 1)
        .filter_map(|((a, b), _)| {
            let p = *mesh.points.get(a)?;
            let q = *mesh.points.get(b)?;
            Some(([p[0], p[1]], [q[0], q[1]]))
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
            vec![Cell::qua4([0, 1, 2, 3])],
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
            vec![Cell::qua4([1, 0, 3, 2])],
        );
        let margin = gate_margin(&mesh.cells[0], &mesh.points);
        assert!(margin < 0.0, "{margin}");
        assert_eq!(margin_color(margin), "#cc00cc");
    }

    #[test]
    fn renders_valid_svg_for_unit_square() {
        let mesh = square();
        let opts = MeshDrawOptions {
            title: "Unit & Square".to_string(),
            mark_nodes: true,
            ..MeshDrawOptions::default()
        };
        let svg = render_mesh_svg(&mesh, None, &opts).unwrap();
        assert!(svg.starts_with("<svg "));
        assert!(svg.contains("fill=\"#66bd63\""));
        assert!(svg.contains("Unit &amp; Square"));
        assert!(svg.ends_with("</svg>\n"));
    }

    #[test]
    fn boundary_segment_counts_are_exact() {
        let mesh = square();
        assert_eq!(boundary_segments(&mesh).len(), 4);

        let two = Mesh::new(
            vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            vec![Cell::qua4([0, 1, 4, 5]), Cell::qua4([1, 2, 3, 4])],
        );
        assert_eq!(boundary_segments(&two).len(), 6);
    }
}
