use crate::adapter::adapters::support::{StreamerCapturedData, StreamerOptions, get_api_key};
use crate::adapter::anthropic::AnthropicStreamer;
use crate::adapter::{Adapter, AdapterKind, ServiceType, WebRequestData};
use crate::chat::{
	Binary, BinarySource, CacheControl, ChatOptionsSet, ChatRequest, ChatResponse, ChatRole, ChatStream,
	ChatStreamResponse, ContentPart, MessageContent, PromptTokensDetails, ToolCall, Usage,
};
use crate::resolver::{AuthData, Endpoint};
use crate::webc::{EventSourceStream, WebResponse, WebStream};
use crate::{Headers, ModelIden, Result, ServiceTarget};
use log::debug;
use reqwest::RequestBuilder;
use serde_json::{Value, json};
use tracing::warn;
use value_ext::JsonValueExt;

pub struct VertexAnthropicAdapter;

// Vertex Anthropic requires its own version identifier
const ANTHROPIC_VERSION: &str = "vertex-2023-10-16";

const MODELS: &[&str] = &[
	// Sonnet 5 (default revision alias on Vertex)
	"claude-sonnet-5@default",
	// GA 4.1 model ID with revision suffix on Vertex
	"claude-opus-4-1@20250805",
	// Commonly available 3.x variants
	"claude-3-7-sonnet-latest",
	"claude-3-5-haiku-20241022",
	"claude-3-5-sonnet-20241022",
	"claude-3-opus-20240229",
	"claude-3-haiku-20240307",
];

// Token maxima (same heuristic as AnthropicAdapter)
const MAX_TOKENS_64K: u32 = 64000; // claude-3-7-sonnet, claude-sonnet-4
const MAX_TOKENS_32K: u32 = 32000; // claude-opus-4
const MAX_TOKENS_8K: u32 = 8192; // claude-3-5-sonnet, claude-3-5-haiku
const MAX_TOKENS_4K: u32 = 4096; // claude-3-opus, claude-3-haiku

impl VertexAnthropicAdapter {
	pub const API_KEY_DEFAULT_ENV_NAME: &str = "GOOGLE_VERTEX_TOKEN";
}

impl Adapter for VertexAnthropicAdapter {
	fn default_endpoint() -> Endpoint {
		// Users should override to include projects/{proj}/locations/{loc}/
		const BASE_URL: &str = "https://aiplatform.googleapis.com/v1/";
		Endpoint::from_static(BASE_URL)
	}

	fn default_auth() -> AuthData {
		AuthData::from_env(Self::API_KEY_DEFAULT_ENV_NAME)
	}

	async fn all_model_names(_kind: AdapterKind) -> Result<Vec<String>> {
		Ok(MODELS.iter().map(|s| s.to_string()).collect())
	}

	fn get_service_url(model: &ModelIden, service_type: ServiceType, endpoint: Endpoint) -> Result<String> {
		let base_url = endpoint.base_url();
		let (_, model_name) = model.model_name.namespace_and_name();
		let suffix = match service_type {
			ServiceType::Chat => ":rawPredict",
			ServiceType::ChatStream => ":streamRawPredict",
			ServiceType::Embed => ":rawPredict", // embeddings unsupported
		};
		// Use publishers/{publisher}/models/{model}:{rawPredict|streamRawPredict}
		// e.g., publishers/anthropic/models/claude-opus-4-1@20250805:rawPredict
		Ok(format!(
			"{base}publishers/anthropic/models/{model}{suffix}",
			base = base_url,
			model = model_name
		))
	}

	fn to_web_request_data(
		target: ServiceTarget,
		service_type: ServiceType,
		chat_req: ChatRequest,
		options_set: ChatOptionsSet<'_, '_>,
	) -> Result<WebRequestData> {
		let ServiceTarget { endpoint, auth, model } = target;

		// -- auth
		let token = get_api_key(auth, &model)?;

		// -- url
		let url = Self::get_service_url(&model, service_type, endpoint.clone())?;

		// -- headers
		let mut headers = Headers::from(("Authorization".to_string(), format!("Bearer {token}")));
		// If we're using end-user OAuth tokens (common with gcloud ADC), Vertex often
		// requires `x-goog-user-project` to attribute billing. Derive it from the endpoint url.
		if let Some(proj) = extract_project_from_endpoint(endpoint.base_url()) {
			headers.merge(("x-goog-user-project".to_string(), proj));
		}
		// Extra headers passthrough (caller-provided overrides)
		if let Some(extra_headers) = options_set.extra_headers() {
			headers.merge_with(extra_headers);
		}

		// Do not send anthropic-beta headers for Vertex. Prompt caching is default and
		// extended TTL is not supported via headers on Vertex Anthropic.

		// -- parts from ChatRequest
		let VertexAnthropicRequestParts {
			system,
			messages,
			tools,
		} = Self::into_vertex_anthropic_request_parts(chat_req)?;

		// -- payload (Anthropic schema in body)
		let (_, model_name) = model.model_name.namespace_and_name();
		let mut payload = json!({
			// NOTE: For Vertex publisher endpoints, the model is specified in the URL path.
			// Do not send a `model` field in the body to avoid 400 "Extra inputs are not permitted".
			"messages": messages,
			"anthropic_version": ANTHROPIC_VERSION,
		});

		// Explicitly request streaming in the payload for streamRawPredict
		if matches!(service_type, ServiceType::ChatStream) {
			let _ = payload.x_insert("stream", true);
		}

		if let Some(system) = system {
			payload.x_insert("system", system)?;
		}
		if let Some(tools) = tools {
			payload.x_insert("/tools", tools)?;
		}

		// -- options
		if let Some(temperature) = options_set.temperature() {
			payload.x_insert("temperature", temperature)?;
		}
		if !options_set.stop_sequences().is_empty() {
			payload.x_insert("stop_sequences", options_set.stop_sequences())?;
		}

		let max_tokens = options_set.max_tokens().unwrap_or_else(|| {
			if model_name.contains("claude-sonnet") || model_name.contains("claude-3-7-sonnet") {
				MAX_TOKENS_64K
			} else if model_name.contains("claude-opus-4") {
				MAX_TOKENS_32K
			} else if model_name.contains("claude-3-5") {
				MAX_TOKENS_8K
			} else if model_name.contains("3-opus") || model_name.contains("3-haiku") {
				MAX_TOKENS_4K
			} else {
				MAX_TOKENS_64K
			}
		});
		payload.x_insert("max_tokens", max_tokens)?;

		if let Some(top_p) = options_set.top_p() {
			payload.x_insert("top_p", top_p)?;
		}

		Ok(WebRequestData { url, headers, payload })
	}

	fn to_chat_response(
		model_iden: ModelIden,
		web_response: WebResponse,
		options_set: ChatOptionsSet<'_, '_>,
	) -> Result<ChatResponse> {
		let WebResponse { mut body, .. } = web_response;
		let captured_raw_body = options_set.capture_raw_body().unwrap_or_default().then(|| body.clone());

		// Provider model iden (if present)
		let provider_model_name: Option<String> = body.x_remove("model").ok();
		let provider_model_iden = model_iden.from_optional_name(provider_model_name);

		// usage normalization (Anthropic style)
		let usage = body.x_take::<Value>("usage");
		let usage = usage.map(Self::into_usage).unwrap_or_default();

		// content parsing (Anthropic schema) - mirror Anthropic adapter behavior
		let mut content = MessageContent::default();
		let json_content_items: Vec<Value> = body.x_take("content")?;
		let mut text_content: Vec<String> = Vec::new();
		let mut tool_calls: Vec<ToolCall> = vec![];
		for mut item in json_content_items {
			let typ: &str = item.x_get_as("type")?;
			if typ == "text" {
				text_content.push(item.x_take("text")?);
			} else if typ == "tool_use" {
				let call_id = item.x_take::<String>("id")?;
				let fn_name = item.x_take::<String>("name")?;
				let fn_arguments = item.x_take::<Value>("input").unwrap_or_default();
				tool_calls.push(ToolCall {
					call_id,
					fn_name,
					fn_arguments,
					thought_signatures: None,
				});
			}
		}
		if !tool_calls.is_empty() {
			content.extend(MessageContent::from(tool_calls));
		}
		if !text_content.is_empty() {
			content.push(text_content.join("\n"));
		}

		Ok(ChatResponse {
			content,
			reasoning_content: None,
			model_iden,
			provider_model_iden,
			usage,
			captured_raw_body,
		})
	}

	fn to_chat_stream(
		model_iden: ModelIden,
		reqwest_builder: RequestBuilder,
		options_set: ChatOptionsSet<'_, '_>,
	) -> Result<ChatStreamResponse> {
		// Vertex streams as SSE (text/event-stream) when "stream": true is provided.
		// Use the Anthropic SSE streamer for event mapping.
		log::debug!("Vertex SSE stream start - using EventSource parser");
		let event_source = EventSourceStream::new(reqwest_builder);
		let stream = AnthropicStreamer::new(event_source, model_iden.clone(), options_set);
		let chat_stream = ChatStream::from_inter_stream(stream);
		Ok(ChatStreamResponse {
			model_iden,
			stream: chat_stream,
		})
	}

	fn to_embed_request_data(
		_service_target: crate::ServiceTarget,
		_embed_req: crate::embed::EmbedRequest,
		_options_set: crate::embed::EmbedOptionsSet<'_, '_>,
	) -> Result<crate::adapter::WebRequestData> {
		Err(crate::Error::AdapterNotSupported {
			adapter_kind: AdapterKind::VertexAnthropic,
			feature: "embeddings".to_string(),
		})
	}

	fn to_embed_response(
		_model_iden: crate::ModelIden,
		_web_response: crate::webc::WebResponse,
		_options_set: crate::embed::EmbedOptionsSet<'_, '_>,
	) -> Result<crate::embed::EmbedResponse> {
		Err(crate::Error::AdapterNotSupported {
			adapter_kind: AdapterKind::VertexAnthropic,
			feature: "embeddings".to_string(),
		})
	}
}

// Attempt to extract the GCP project id from an endpoint base url
// Example: https://us-east5-aiplatform.googleapis.com/v1/projects/my-proj/locations/us-east5/
fn extract_project_from_endpoint(base_url: &str) -> Option<String> {
	let marker = "/projects/";
	let idx = base_url.find(marker)? + marker.len();
	let tail = &base_url[idx..];
	let end = tail.find('/')?;
	let project = &tail[..end];
	if project.is_empty() {
		None
	} else {
		Some(project.to_string())
	}
}

// region: shared helpers (duplicated from Anthropic adapter for now)

impl VertexAnthropicAdapter {
	fn into_usage(mut usage_value: Value) -> Usage {
		let input_tokens: i32 = usage_value.x_take("input_tokens").ok().unwrap_or(0);
		let cache_creation_input_tokens: i32 = usage_value.x_take("cache_creation_input_tokens").unwrap_or(0);
		let cache_read_input_tokens: i32 = usage_value.x_take("cache_read_input_tokens").unwrap_or(0);
		let completion_tokens: i32 = usage_value.x_take("output_tokens").ok().unwrap_or(0);

		let prompt_tokens = input_tokens + cache_creation_input_tokens + cache_read_input_tokens;
		let total_tokens = prompt_tokens + completion_tokens;

		let prompt_tokens_details = if cache_creation_input_tokens > 0 || cache_read_input_tokens > 0 {
			Some(PromptTokensDetails {
				cache_creation_tokens: Some(cache_creation_input_tokens),
				cached_tokens: Some(cache_read_input_tokens),
				audio_tokens: None,
			})
		} else {
			None
		};

		Usage {
			prompt_tokens: Some(prompt_tokens),
			prompt_tokens_details,
			completion_tokens: Some(completion_tokens),
			completion_tokens_details: None,
			total_tokens: Some(total_tokens),
		}
	}

	fn into_vertex_anthropic_request_parts(chat_req: ChatRequest) -> Result<VertexAnthropicRequestParts> {
		let mut messages: Vec<Value> = Vec::new();
		let mut systems: Vec<(String, Option<CacheControl>)> = Vec::new();
		let mut did_strip_ttl: bool = false;

		// NOTE: Vertex Anthropic does not accept TTL in cache_control. Supplying a TTL or sending
		// the Anthropic beta header for extended TTL (e.g., "extended-cache-ttl-2025-04-11")
		// triggers a 400 with "Unexpected value(s) for the 'anthropic-beta' header" or the
		// body field being treated as an extra input. Therefore, we strip TTL and keep only
		// {"type":"ephemeral"} on Vertex. Prompt caching itself is enabled by default on Vertex.
		fn strip_ttl_cache_control(cc: Option<CacheControl>, did: &mut bool) -> Option<CacheControl> {
			match cc {
				Some(CacheControl::EphemeralWithTtl(_)) => {
					*did = true;
					Some(CacheControl::Ephemeral)
				}
				x => x,
			}
		}

		if let Some(system) = chat_req.system {
			systems.push((system, None));
		}

		for msg in chat_req.messages {
			let cache_control = strip_ttl_cache_control(msg.options.and_then(|o| o.cache_control), &mut did_strip_ttl);
			match msg.role {
				ChatRole::System => {
					if let Some(system_text) = msg.content.joined_texts() {
						systems.push((system_text, cache_control));
					}
				}
				ChatRole::User => {
					if msg.content.is_text_only() {
						let text = msg.content.joined_texts().unwrap_or_else(String::new);
						let content = apply_cache_control_to_text(cache_control.as_ref(), text);
						messages.push(json!({"role": "user", "content": content}));
					} else {
						let mut values: Vec<Value> = Vec::new();
						for part in msg.content {
							match part {
								ContentPart::Text(text) => values.push(json!({"type": "text", "text": text})),
								ContentPart::Binary(binary) => {
									let is_image = binary.is_image();
									let Binary {
										content_type, source, ..
									} = binary;
									if is_image {
										match &source {
											BinarySource::Url(_) => {
												warn!(
													"VertexAnthropic: image by URL not supported by Anthropic messages"
												);
											}
											BinarySource::Base64(content) => {
												values.push(json!({
													"type": "image",
													"source": {"type": "base64", "media_type": content_type, "data": content}
												}));
											}
										}
									} else {
										match &source {
											BinarySource::Url(url) => {
												values.push(json!({
													"type": "document",
													"source": {"type": "url", "url": url}
												}));
											}
											BinarySource::Base64(b64) => {
												values.push(json!({
													"type": "document",
													"source": {"type": "base64", "media_type": content_type, "data": b64}
												}));
											}
										}
									}
								}
								ContentPart::ToolCall(_) => {}
								ContentPart::ToolResponse(tool_response) => {
									values.push(
										json!({"type": "tool_result", "content": tool_response.content, "tool_use_id": tool_response.call_id}),
									);
								}
								ContentPart::ThoughtSignature(_) => {}
							}
						}
						let values = apply_cache_control_to_parts(cache_control.as_ref(), values);
						messages.push(json!({"role": "user", "content": values}));
					}
				}
				ChatRole::Assistant => {
					let mut values: Vec<Value> = Vec::new();
					let mut has_tool_use = false;
					let mut has_text = false;
					for part in msg.content {
						match part {
							ContentPart::Text(text) => {
								has_text = true;
								values.push(json!({"type": "text", "text": text}));
							}
							ContentPart::ToolCall(tool_call) => {
								has_tool_use = true;
								values.push(json!({"type": "tool_use", "id": tool_call.call_id, "name": tool_call.fn_name, "input": tool_call.fn_arguments}));
							}
							ContentPart::Binary(_) => {}
							ContentPart::ToolResponse(_) => {}
							ContentPart::ThoughtSignature(_) => {}
						}
					}
					// Vertex Anthropic: do not apply cache_control on assistant messages
					if !has_tool_use && has_text && values.len() == 1 {
						let text = values
							.first()
							.and_then(|v| v.get("text"))
							.and_then(|v| v.as_str())
							.unwrap_or_default()
							.to_string();
						messages.push(json!({"role": "assistant", "content": text}));
					} else {
						messages.push(json!({"role": "assistant", "content": values}));
					}
				}
				ChatRole::Tool => {
					let mut values: Vec<Value> = Vec::new();
					for part in msg.content {
						if let ContentPart::ToolResponse(tool_response) = part {
							values.push(
								json!({"type": "tool_result", "content": tool_response.content, "tool_use_id": tool_response.call_id}),
							);
						}
					}
					if !values.is_empty() {
						let values = apply_cache_control_to_parts(cache_control.as_ref(), values);
						messages.push(json!({"role": "user", "content": values}));
					}
				}
			}
		}

		// Build system as combined or multipart with cache_control on last cached entry
		let system = if !systems.is_empty() {
			let mut last_cache_idx = -1;
			let mut last_cache_control: Option<CacheControl> = None;
			for (idx, (_, cc)) in systems.iter().enumerate() {
				if cc.is_some() {
					last_cache_idx = idx as i32;
					last_cache_control = cc.clone();
				}
			}
			let system: Value = if last_cache_idx >= 0 {
				let mut parts: Vec<Value> = Vec::new();
				for (idx, (content, _)) in systems.iter().enumerate() {
					let idx = idx as i32;
					if idx == last_cache_idx {
						// See note above: TTL is stripped for Vertex to avoid 400s and "extra inputs" errors.
						let cache_control_json = match &last_cache_control {
							Some(CacheControl::Ephemeral) => json!({"type": "ephemeral"}),
							Some(CacheControl::EphemeralWithTtl(_)) => json!({"type": "ephemeral"}),
							None => json!({"type": "ephemeral"}),
						};
						let part = json!({"type": "text", "text": content, "cache_control": cache_control_json});
						parts.push(part);
					} else {
						parts.push(json!({"type": "text", "text": content}));
					}
				}
				json!(parts)
			} else {
				let buff = systems.iter().map(|(c, _)| c.as_str()).collect::<Vec<&str>>();
				json!(buff.join("\n\n"))
			};
			Some(system)
		} else {
			None
		};

		// tools
		let tools = chat_req.tools.map(|tools| {
			tools
				.into_iter()
				.map(|tool| {
					let mut tool_value = json!({"name": tool.name, "input_schema": tool.schema});
					if let Some(description) = tool.description {
						let _ = tool_value.x_insert("description", description);
					}
					tool_value
				})
				.collect::<Vec<Value>>()
		});

		// Ensure first non-system message is a user message (drop leading assistant if necessary)
		let first_user_idx = messages
			.iter()
			.position(|m| m.get("role").and_then(|v| v.as_str()) == Some("user"));
		if let Some(idx) = first_user_idx {
			if idx > 0 {
				let _ = messages.drain(0..idx);
			}
		}

		if did_strip_ttl {
			debug!("VertexAnthropic: TTL not supported on Vertex; stripped ttl from cache_control");
		}
		Ok(VertexAnthropicRequestParts {
			system,
			messages,
			tools,
		})
	}
}

// Helper: apply cache_control to a single text content. On Vertex, TTL is not supported and is stripped.
fn apply_cache_control_to_text(cache_control: Option<&CacheControl>, content: String) -> Value {
	if let Some(cc) = cache_control {
		let cache_control_json = match cc {
			CacheControl::Ephemeral => json!({"type": "ephemeral"}),
			CacheControl::EphemeralWithTtl(_ttl) => json!({"type": "ephemeral"}),
		};
		let value = json!({"type": "text", "text": content, "cache_control": cache_control_json});
		json!(vec![value])
	} else {
		json!(content)
	}
}

// Helper: apply cache_control to the last element of a multipart content array.
// On Vertex, TTL is not supported and is stripped.
fn apply_cache_control_to_parts(cache_control: Option<&CacheControl>, parts: Vec<Value>) -> Vec<Value> {
	let mut parts = parts;
	if let Some(cc) = cache_control {
		if !parts.is_empty() {
			let cache_control_json = match cc {
				CacheControl::Ephemeral => json!({"type": "ephemeral"}),
				CacheControl::EphemeralWithTtl(_ttl) => json!({"type": "ephemeral"}),
			};
			let len = parts.len();
			if let Some(last_value) = parts.get_mut(len - 1) {
				let _ = last_value.x_insert("cache_control", cache_control_json);
			}
		}
	}
	parts
}

struct VertexAnthropicRequestParts {
	system: Option<Value>,
	messages: Vec<Value>,
	tools: Option<Vec<Value>>,
}

// endregion

// region: --- Vertex Anthropic JSON Streamer (newline-delimited JSON)
use crate::adapter::inter_stream::{InterStreamEnd, InterStreamEvent};
use std::pin::Pin;
use std::task::{Context, Poll};

use std::collections::VecDeque;

enum InProgressBlock {
	Text,
	Thinking,
	ToolUse { id: String, name: String, input: String },
}

struct VertexAnthropicJsonStreamer {
	inner: WebStream,
	options: StreamerOptions,
	done: bool,
	captured_data: StreamerCapturedData,
	sent_start: bool,
	pending: VecDeque<InterStreamEvent>,
	in_progress_block: InProgressBlock,
	logged_lines: usize,
}

impl VertexAnthropicJsonStreamer {
	fn new(inner: WebStream, model_iden: ModelIden, options_set: ChatOptionsSet<'_, '_>) -> Self {
		Self {
			inner,
			options: StreamerOptions::new(model_iden, options_set),
			done: false,
			captured_data: Default::default(),
			sent_start: false,
			pending: VecDeque::new(),
			in_progress_block: InProgressBlock::Text,
			logged_lines: 0,
		}
	}
}

impl futures::Stream for VertexAnthropicJsonStreamer {
	type Item = crate::Result<InterStreamEvent>;

	fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
		// Serve any queued events first (even if `done` is true)
		if let Some(ev) = self.pending.pop_front() {
			return Poll::Ready(Some(Ok(ev)));
		}

		if self.done {
			return Poll::Ready(None);
		}

		while let Poll::Ready(item) = Pin::new(&mut self.inner).poll_next(cx) {
			match item {
				Some(Ok(line)) => {
					let trimmed = line.trim();
					if trimmed.is_empty() {
						continue;
					}
					if self.logged_lines < 5 {
						debug!(
							"Vertex stream line[{}]: {}",
							self.logged_lines,
							truncate_str_dbg(trimmed, 600)
						);
						self.logged_lines += 1;
					}
					// Try JSON parse; ignore parse errors (may be keep-alive chunks)
					let parsed = match serde_json::from_str::<Value>(trimmed) {
						Ok(v) => v,
						Err(_) => continue,
					};

					// Ensure Start is queued once we confirm we have a JSON item
					if !self.sent_start {
						self.sent_start = true;
						self.pending.push_back(InterStreamEvent::Start);
					}

					// Capture usage when present on message_start/message_delta shapes
					if self.options.capture_usage {
						if let Some(usage) = parsed
							.get("usage")
							.cloned()
							.or_else(|| parsed.get("message").and_then(|m| m.get("usage")).cloned())
						{
							let usage = VertexAnthropicAdapter::into_usage(usage);
							self.captured_data.usage = Some(usage);
						}
					}

					// Handle stream events (Anthropic schema over NDJSON)
					if let Some(typ) = parsed.get("type").and_then(|v| v.as_str()) {
						debug!("Vertex stream event: {}", typ);
						match typ {
							"message" => {
								// Vertex sometimes sends a single 'message' event with full content.
								// Emit a Chunk with concatenated text, then End.
								let mut text_buf = String::new();
								if let Some(arr) = parsed.get("content").and_then(|v| v.as_array()) {
									for item in arr {
										if let Some(t) = item.get("text").and_then(|v| v.as_str()) {
											text_buf.push_str(t);
										}
									}
								}
								if !text_buf.is_empty() {
									// Always accumulate full text so End has content even when capture_content is off.
									match self.captured_data.content {
										Some(ref mut c) => c.push_str(&text_buf),
										None => self.captured_data.content = Some(text_buf.clone()),
									}
									debug!("Vertex emit chunk (message) len={}", text_buf.len());
									self.pending.push_back(InterStreamEvent::Chunk(text_buf));
								}
								self.done = true;
								let inter_end = InterStreamEnd {
									captured_usage: self.captured_data.usage.take(),
									captured_text_content: self.captured_data.content.take(),
									captured_reasoning_content: self.captured_data.reasoning_content.take(),
									captured_tool_calls: self.captured_data.tool_calls.take(),
									captured_thought_signatures: None,
								};
								self.pending.push_back(InterStreamEvent::End(inter_end));
								if let Some(ev) = self.pending.pop_front() {
									return Poll::Ready(Some(Ok(ev)));
								} else {
									continue;
								}
							}
							"message_start" => {
								// usage captured above if present; nothing to emit
								continue;
							}
							"message_delta" => {
								// usage captured above; nothing to emit
								continue;
							}
							"content_block_start" => {
								// Set in-progress block by type
								if let Some(cb) = parsed.get("content_block") {
									match cb.get("type").and_then(|v| v.as_str()).unwrap_or("") {
										"text" => self.in_progress_block = InProgressBlock::Text,
										"thinking" => self.in_progress_block = InProgressBlock::Thinking,
										"tool_use" => {
											let id = cb.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
											let name =
												cb.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
											self.in_progress_block = InProgressBlock::ToolUse {
												id,
												name,
												input: String::new(),
											};
										}
										_ => {}
									}
								}
								continue;
							}
							"content_block_delta" => {
								match &mut self.in_progress_block {
									InProgressBlock::Text => {
										if let Some(text) =
											parsed.get("delta").and_then(|d| d.get("text")).and_then(|v| v.as_str())
										{
											// Always accumulate text so End has content even when capture_content is off.
											match self.captured_data.content {
												Some(ref mut c) => c.push_str(text),
												None => self.captured_data.content = Some(text.to_string()),
											}
											debug!("Vertex emit chunk len={}", text.len());
											self.pending.push_back(InterStreamEvent::Chunk(text.to_string()));
										}
									}
									InProgressBlock::Thinking => {
										if let Some(thinking) =
											parsed.get("delta").and_then(|d| d.get("thinking")).and_then(|v| v.as_str())
										{
											if self.options.capture_reasoning_content {
												match self.captured_data.reasoning_content {
													Some(ref mut r) => r.push_str(thinking),
													None => {
														self.captured_data.reasoning_content =
															Some(thinking.to_string())
													}
												}
											}
											debug!("Vertex emit reasoning len={}", thinking.len());
											self.pending
												.push_back(InterStreamEvent::ReasoningChunk(thinking.to_string()));
										}
									}
									InProgressBlock::ToolUse { input, .. } => {
										if let Some(part) = parsed
											.get("delta")
											.and_then(|d| d.get("partial_json"))
											.and_then(|v| v.as_str())
										{
											input.push_str(part);
										}
									}
								}
								// After queuing, return the next event if any
								if let Some(ev) = self.pending.pop_front() {
									return Poll::Ready(Some(Ok(ev)));
								} else {
									continue;
								}
							}
							"content_block_stop" => {
								if let InProgressBlock::ToolUse { id, name, input } =
									std::mem::replace(&mut self.in_progress_block, InProgressBlock::Text)
								{
									// Try to parse input JSON; if invalid, default to empty object
									let args = serde_json::from_str::<Value>(&input).unwrap_or_else(|_| json!({}));
									let tc = ToolCall {
										call_id: id,
										fn_name: name,
										fn_arguments: args,
										thought_signatures: None,
									};
									if self.options.capture_tool_calls {
										match self.captured_data.tool_calls {
											Some(ref mut t) => t.push(tc.clone()),
											None => self.captured_data.tool_calls = Some(vec![tc.clone()]),
										}
									}
									debug!("Vertex emit tool_call name={} args_len={}", tc.fn_name, input.len());
									self.pending.push_back(InterStreamEvent::ToolCallChunk(tc));
									if let Some(ev) = self.pending.pop_front() {
										return Poll::Ready(Some(Ok(ev)));
									} else {
										continue;
									}
								}
								continue;
							}
							"message_stop" => {
								self.done = true;
								let inter_end = InterStreamEnd {
									captured_usage: self.captured_data.usage.take(),
									captured_text_content: self.captured_data.content.take(),
									captured_reasoning_content: self.captured_data.reasoning_content.take(),
									captured_tool_calls: self.captured_data.tool_calls.take(),
									captured_thought_signatures: None,
								};
								debug!(
									"Vertex message_stop: text_len={} reasoning_len={} usage_present={}",
									inter_end.captured_text_content.as_ref().map(|s| s.len()).unwrap_or(0),
									inter_end.captured_reasoning_content.as_ref().map(|s| s.len()).unwrap_or(0),
									inter_end.captured_usage.is_some()
								);
								self.pending.push_back(InterStreamEvent::End(inter_end));
								if let Some(ev) = self.pending.pop_front() {
									return Poll::Ready(Some(Ok(ev)));
								} else {
									continue;
								}
							}
							_ => { /* ignore other event types for now */ }
						}
					}
					continue;
				}
				Some(Err(err)) => {
					self.done = true;
					return Poll::Ready(Some(Err(crate::Error::WebStream {
						model_iden: self.options.model_iden.clone(),
						cause: err.to_string(),
					})));
				}
				None => {
					self.done = true;
					debug!(
						"Vertex stream closed: sent_start={} captured_text_len={}",
						self.sent_start,
						self.captured_data.content.as_ref().map(|s| s.len()).unwrap_or(0)
					);
					return Poll::Ready(None);
				}
			}
		}
		Poll::Pending
	}
}

fn truncate_str_dbg(s: &str, max: usize) -> String {
	if s.len() <= max {
		return s.to_string();
	}
	let mut out = s[..max].to_string();
	out.push('…');
	out
}

// endregion

#[cfg(test)]
mod tests {
	use super::*;
	use crate::adapter::ServiceType;
	use crate::chat::{CacheControl, ChatMessage, ChatOptionsSet, ChatRequest};
	use crate::resolver::AuthData;

	fn mk_target(model: &str, base_url: &str, token: &str) -> ServiceTarget {
		let model = ModelIden::new(AdapterKind::VertexAnthropic, model);
		let endpoint = Endpoint::from_owned(base_url.to_string());
		let auth = AuthData::from_single(token);
		ServiceTarget { endpoint, auth, model }
	}

	#[test]
	fn test_vertex_anthropic_url_build() {
		let base = "https://aiplatform.googleapis.com/v1/projects/p/locations/l/";
		let target = mk_target("claude-opus-4-1@20250805", base, "tkn");
		let url =
			VertexAnthropicAdapter::get_service_url(&target.model, ServiceType::Chat, target.endpoint.clone()).unwrap();
		assert_eq!(
			url,
			format!(
				"{}publishers/anthropic/models/claude-opus-4-1@20250805:rawPredict",
				base
			)
		);
		let url_stream =
			VertexAnthropicAdapter::get_service_url(&target.model, ServiceType::ChatStream, target.endpoint).unwrap();
		assert_eq!(
			url_stream,
			format!(
				"{}publishers/anthropic/models/claude-opus-4-1@20250805:streamRawPredict",
				base
			)
		);
	}

	#[test]
	fn test_vertex_anthropic_payload_basic() {
		let base = "https://aiplatform.googleapis.com/v1/projects/p/locations/l/";
		let target = mk_target("claude-opus-4-1@20250805", base, "tkn");

		let chat_req = ChatRequest::new(vec![ChatMessage::system("sys"), ChatMessage::user("hi")]);
		let req =
			VertexAnthropicAdapter::to_web_request_data(target, ServiceType::Chat, chat_req, ChatOptionsSet::default())
				.expect("build ok");

		let auth_header = req.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case("authorization"));
		assert!(auth_header.unwrap().1.starts_with("Bearer "));

		// Vertex carries the model in the URL, not the body, and uses the vertex version string.
		assert!(req.payload.get("model").is_none());
		let version = req.payload.get("anthropic_version").and_then(|v| v.as_str()).unwrap_or("");
		assert_eq!(version, ANTHROPIC_VERSION);
		assert!(req.payload.get("messages").is_some());
	}

	#[test]
	fn test_vertex_anthropic_cache_control_strips_ttl() {
		let base = "https://aiplatform.googleapis.com/v1/projects/p/locations/l/";
		let target = mk_target("claude-opus-4-1@20250805", base, "tkn");

		let msg = ChatMessage::user("cache me").with_options(CacheControl::EphemeralWithTtl("5m".to_string()));
		let chat_req = ChatRequest::new(vec![msg]);
		let req =
			VertexAnthropicAdapter::to_web_request_data(target, ServiceType::Chat, chat_req, ChatOptionsSet::default())
				.expect("build ok");

		let messages = req.payload.get("messages").and_then(|v| v.as_array()).expect("messages array");
		let content = messages
			.first()
			.and_then(|m| m.get("content"))
			.and_then(|v| v.as_array())
			.expect("content array");
		let cc = content
			.last()
			.and_then(|p| p.get("cache_control"))
			.expect("cache_control present");
		// Vertex accepts cache_control but not the ttl field: it must be {"type":"ephemeral"} only.
		assert_eq!(cc.get("type").and_then(|v| v.as_str()), Some("ephemeral"));
		assert!(cc.get("ttl").is_none(), "Vertex must strip ttl from cache_control");
	}
}
