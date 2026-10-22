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
//! Supported: rectangle-like domains, including one or more holes, where the
//! surviving interior grid is large enough for the channel ribbon to span.
//! Non-convex domains whose grid interior degenerates to a thin staircase (the
//! l-bracket is the corpus example) return a clean `Err`.

use serde::{Deserialize, Serialize};

use crate::mesh::Mesh;
use crate::simons_channel;
use crate::two_d::{Domain, Sizing};

/// Options for the size-driven route.
///
/// The thesis exposes the channel width, the grid orientation/offset and the
/// patch-cover search as knobs. The current implementation fixes the channel
/// width at the thesis's 1.4 h and the grid at the domain's axes; the struct is
/// kept so that the config layer has a stable type to extend.
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
    fn the_options_are_serde_compatible() {
        let text = serde_json::to_string(&Options::default()).unwrap();
        let _back: Options = serde_json::from_str(&text).unwrap();
    }
}
