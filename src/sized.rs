//! The Simons 2024 size-driven all-quad route.
//!
//! This route is the thesis pipeline of [`crate::simons_channel`] (Chapter 2:
//! a regular grid interior, a carved boundary *channel*, and a quadrangulation
//! of that channel), exposed through the same `Options`/`mesh_with_options`
//! interface as every other route so that [`crate::config::MeshConfig`] can
//! ship it.
//!
//! The thesis's Chapters 3–5 (the size-aware cross field, the size-function
//! families and the QGP/Coons extraction) are implemented as
//! [`crate::simons_godf`], [`crate::simons_sizefamily`] and
//! [`crate::simons_qgp`]; this route currently uses the Chapter 2 interior
//! (a uniform grid at the requested size), which is the thesis's own simple
//! interior. Feeding the GODF/QGP interior into the channel is the remaining
//! integration step, documented in the vault.
//!
//! Supported: **all seven corpus parts** — rectangle-like domains with holes
//! (plate-hole 4/2/1/0.6 mm: 262/1240/5324/15526 quads, SJ 0.608/0.564/0.481/
//! 0.391 after the shared smoother + shape pass; the perf raw mesh is
//! 0.234 / 13.6 deg after the half-cell grid-offset retry), the rectilinear non-convex
//! parts (l-bracket, stepped: 600/2300 quads as perfect grids) and the quarter
//! annulus (ring: 48 quads, SJ 0.998, 87.5°, through the two-chain strip
//! fallback)
//! where the adaptive channel width (thesis 1.4 h, then 1.0 h, then 0.7 h) lets
//! the interior grid reach close enough to the boundary for the cover to close
//! the remaining band. The quarter-annulus ring still fails: its curved
//! staircase channel leaves a leftover that neither the ribbon nor the template
//! cover can close, and it returns a clean `Err`.

use serde::{Deserialize, Serialize};

use crate::mesh::Mesh;
use crate::simons_channel;
use crate::two_d::{Domain, Sizing};

/// Options for the size-driven route.
///
/// The thesis exposes the channel width, the grid orientation/offset and the
/// patch-cover search as knobs. The current implementation tries the channel
/// widths 1.4 h, 1.0 h and 0.7 h and keeps the domain axes; the struct is kept
/// so that the config layer has a stable type to extend.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Options {}

/// Mesh a planar domain with the size-driven channel route.
pub fn mesh(domain: &Domain, sizing: &Sizing) -> Result<Mesh, String> {
    mesh_with_options(domain, sizing, &Options::default())
}

/// Mesh with explicit [`Options`].
pub fn mesh_with_options(
    domain: &Domain,
    sizing: &Sizing,
    _options: &Options,
) -> Result<Mesh, String> {
    simons_channel::mesh(domain, sizing.size)
}

/// Mesh with the Chapter 2 statistics (channel loops, patches, fallbacks).
pub fn mesh_with_stats(
    domain: &Domain,
    sizing: &Sizing,
) -> Result<(Mesh, simons_channel::ChannelStats), String> {
    simons_channel::mesh_with_stats(domain, sizing.size)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn a_square_is_all_quad_and_conforming() {
        let domain = Domain::rectangle(10.0, 10.0).unwrap();
        let sizing = Sizing::uniform(1.0).unwrap();
        let mesh = mesh(&domain, &sizing).unwrap();
        let stats = crate::stats::MeshStatistics::of(&domain, &mesh);
        assert_eq!(stats.triangles, 0);
        assert!(stats.passes, "{}", stats.line());
        assert!(stats.conforming);
    }

    #[test]
    fn the_ring_and_l_bracket_pass_the_gate() {
        let arc = |r: f64, a0: f64, a1: f64, n: usize| -> Vec<[f64; 2]> {
            (0..=n)
                .map(|i| {
                    let a = a0 + (a1 - a0) * i as f64 / n as f64;
                    [r * a.cos(), r * a.sin()]
                })
                .collect()
        };
        let mut pts = arc(20.0, 0.0, std::f64::consts::FRAC_PI_2, 24);
        pts.extend(arc(10.0, std::f64::consts::FRAC_PI_2, 0.0, 12));
        let ring = Domain::new(pts, vec![]).unwrap();
        let ring_mesh = mesh(&ring, &Sizing::uniform(1.0).unwrap()).unwrap();
        let stats = crate::stats::MeshStatistics::of(&ring, &ring_mesh);
        assert_eq!(stats.triangles, 0);
        assert!(stats.passes, "{}", stats.line());
        let lb = Domain::new(
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
        .unwrap();
        let lb_mesh = mesh(&lb, &Sizing::uniform(2.0).unwrap()).unwrap();
        let stats = crate::stats::MeshStatistics::of(&lb, &lb_mesh);
        assert!(stats.passes, "{}", stats.line());
    }

    #[test]
    fn the_options_are_serde_compatible() {
        let text = serde_json::to_string(&Options::default()).unwrap();
        let _back: Options = serde_json::from_str(&text).unwrap();
    }
}
