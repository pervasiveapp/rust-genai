Title: Add Vertex AI (Anthropic) Adapter with Prompt Caching and TTL

Why Vertex First
- Simpler auth: OAuth2 Bearer vs. AWS SigV4 for Bedrock.
- Easier streaming: avoids Amazon EventStream parsing.
- Payload parity: mirrors Anthropic Messages API you already support.

Scope
- New adapter: VertexAnthropicAdapter for Claude (Opus 4 / Sonnet 4 / Claude 3.x) via Vertex.
- Support chat (non-stream) end-to-end; streaming via Anthropic streamer.
- Prompt caching and extended TTL at content-part level (same as Anthropic). Pass beta headers when present; degrade gracefully if ignored.

Design
- AdapterKind: add `VertexAnthropic` (namespace key `vertex-anthropic`). Usage: `vertex-anthropic::claude-opus-4-20250514`.
- Endpoint: users set full Vertex base via `ServiceTargetResolver` (projects/{id}/locations/{loc}/). Default is generic `https://aiplatform.googleapis.com/v1/`.
- URLs:
  - Non-stream: `{base}publishers/anthropic/models/{model}:rawPredict`
  - Stream: `{base}publishers/anthropic/models/{model}:streamRawPredict`
- Auth: `Authorization: Bearer <token>`; default env `GOOGLE_VERTEX_TOKEN`. Merge `ChatOptions.extra_headers` (e.g., `x-goog-user-project`, `anthropic-beta`).
- Request body: reuse Anthropic message transformation and options; include `anthropic_version` in payload; set `stream` flag.
- Response mapping: reuse Anthropic structure and usage normalization.
- Streaming: wrap `reqwest_eventsource::EventSource` with existing `AnthropicStreamer` for event mapping.
- Embeddings: not supported (returns AdapterNotSupported like Anthropic).

Implementation Steps
1) Add `VertexAnthropic` to `AdapterKind` (strings, env default) and namespacing support.
2) Wire `VertexAnthropicAdapter` into `AdapterDispatcher` switches.
3) Create `src/adapter/adapters/vertex_anthropic/` with `mod.rs` and `adapter_impl.rs`.
4) Implement request building, headers merge, caching TTL logic, and URLs.
5) Implement response parsing and streaming via `AnthropicStreamer`.
6) Document usage via `ServiceTargetResolver` (examples later if desired).

Notes
- Extended TTL: sets cache_control on parts and passes `anthropic-beta` when provided or when TTL is used; Vertex may ignore silently.
- Model list mirrors Anthropic adapter for discovery.

