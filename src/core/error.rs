//! Unified error type for `pedroso_mesher` core, paper modules, and FEA generators.

use std::fmt;

use crate::core::mesh::MeshError;

/// Crate-wide error type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// Invalid user input (domain geometry, sizing parameters, or markers).
    InvalidInput(String),
    /// Structural mesh invariant violation.
    Mesh(MeshError),
    /// Algorithmic failure or explicit quality/geometric refusal.
    Algorithm(String),
    /// Feature or paper module scheduled for a subsequent phase.
    NotSupportedYet(String),
}

/// Crate-wide `Result` alias.
pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Error::Mesh(err) => write!(f, "{err}"),
            Error::Algorithm(msg) => write!(f, "{msg}"),
            Error::NotSupportedYet(what) => write!(f, "not supported yet: {what}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<MeshError> for Error {
    fn from(err: MeshError) -> Self {
        Error::Mesh(err)
    }
}
