//! AWS Bedrock (Anthropic Claude) adapter via the `bedrock-runtime` InvokeModel API.
//!
//! Notes:
//! - Uses the Anthropic native request/response schema in the body with
//!   `"anthropic_version": "bedrock-2023-05-31"`, and carries the modelId in the URL path
//!   (`.../model/{modelId}/invoke` or `.../model/{modelId}/invoke-with-response-stream`).
//! - Auth is a Bedrock API key (bearer) taken from `AWS_BEARER_TOKEN_BEDROCK`, sent as
//!   `Authorization: Bearer {token}`. This targets the `bedrock-runtime` endpoint; full SigV4
//!   signing is intentionally not implemented (the bearer-key path covers our use case).
//! - Region comes from `AWS_REGION` (default `us-east-1`); the endpoint host is
//!   `https://bedrock-runtime.{region}.amazonaws.com/`.
//! - Prompt caching uses the native Anthropic `cache_control` form (TTL preserved, and
//!   assistant-block breakpoints honored) — unlike Vertex, Bedrock accepts both.
//! - Streaming responses use the binary `application/vnd.amazon.eventstream` framing; the
//!   inner payloads are standard Anthropic SSE-style event JSON.

mod adapter_impl;

pub use adapter_impl::*;
