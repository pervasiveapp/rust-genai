//! Vertex AI (Anthropic Publisher Model) adapter.
//!
//! Notes:
//! - Expects the base Endpoint to already include `v1/projects/{project}/locations/{location}/`.
//! - Builds URLs of the form `.../publishers/anthropic/models/{model}:rawPredict` (or `:streamRawPredict`).

mod adapter_impl;

pub use adapter_impl::*;
