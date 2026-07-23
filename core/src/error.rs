//! Core error type. Errors are structured data — they ride the bus
//! (spec: Events, queries, and the bus), never log-and-swallowed.
//! TODO(step 5): flesh out variants as subsystems land.

use crate::id::ElementId;

#[derive(Clone, Debug)]
pub enum Error {
    /// Link-time negotiation found no common format between two pads.
    NegotiationFailed,
    /// An element reported a fatal condition.
    Element { element: ElementId, message: String },
    /// A resource (file, socket, device) could not be opened or used.
    Resource(String),
    /// Placeholder until the owning subsystem defines its variants.
    Todo(&'static str),
}

pub type Result<T> = core::result::Result<T, Error>;

/// Failure parsing a launch string (spec: Registry, parse-launch).
#[derive(Clone, Debug)]
pub struct ParseError {
    pub message: String,
}
