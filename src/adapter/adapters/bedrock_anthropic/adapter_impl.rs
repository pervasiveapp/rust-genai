use crate::adapter::adapters::support::{StreamerCapturedData, StreamerOptions, get_api_key};
use crate::adapter::anthropic::AnthropicAdapter;
use crate::adapter::{Adapter, AdapterKind, ServiceType, WebRequestData};
use crate::chat::{
	ChatOptionsSet, ChatRequest, ChatResponse, ChatStream, ChatStreamResponse, MessageContent, ToolCall,
};
use crate::resolver::{AuthData, Endpoint};
use crate::webc::WebResponse;
use crate::{Headers, ModelIden, Result, ServiceTarget};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;
use bytes::Bytes;
use log::debug;
use reqwest::RequestBuilder;
use serde_json::{Value, json};
use value_ext::JsonValueExt;

pub struct BedrockAnthropicAdapter;

// Bedrock requires the Anthropic body to carry this version string.
const ANTHROPIC_VERSION: &str = "bedrock-2023-05-31";

// Common on-demand model ids (inference-profile ids like `us.anthropic.*` also work). These are
// only used for `all_model_names`; any model id namespaced with `bedrock-anthropic::` is accepted.
const MODELS: &[&str] = &[
	"anthropic.claude-sonnet-4-5-v1:0",
	"anthropic.claude-opus-4-1-v1:0",
	"anthropic.claude-3-7-sonnet-20250219-v1:0",
	"anthropic.claude-3-5-sonnet-20241022-v2:0",
	"anthropic.claude-3-5-haiku-20241022-v1:0",
];

// Token maxima (same heuristic as the Anthropic adapter).
const MAX_TOKENS_64K: u32 = 64000; // claude-sonnet-4+, claude-haiku-4+, claude-3-7-sonnet, opus-4-5
const MAX_TOKENS_32K: u32 = 32000; // claude-opus-4(.0/.1)
const MAX_TOKENS_8K: u32 = 8192; // claude-3-5-sonnet, claude-3-5-haiku
const MAX_TOKENS_4K: u32 = 4096; // claude-3-opus, claude-3-haiku

impl BedrockAnthropicAdapter {
	/// Bedrock API key (bearer). Distinct from AWS SigV4 credentials; targets `bedrock-runtime`.
	pub const API_KEY_DEFAULT_ENV_NAME: &str = "AWS_BEARER_TOKEN_BEDROCK";
}

impl Adapter for BedrockAnthropicAdapter {
	fn default_endpoint() -> Endpoint {
		// Region-specific; the service_target_resolver normally rewrites the host from AWS_REGION.
		const BASE_URL: &str = "https://bedrock-runtime.us-east-1.amazonaws.com/";
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
		// The modelId can contain characters ('.', ':') that are valid in a path segment, so it is
		// embedded directly. Bedrock routes on the raw modelId, e.g.
		// .../model/anthropic.claude-sonnet-4-5-v1:0/invoke
		let suffix = match service_type {
			ServiceType::Chat => "invoke",
			ServiceType::ChatStream => "invoke-with-response-stream",
			ServiceType::Embed => "invoke", // embeddings unsupported
		};
		Ok(format!(
			"{base}model/{model}/{suffix}",
			base = base_url,
			model = model_name,
			suffix = suffix
		))
	}

	fn to_web_request_data(
		target: ServiceTarget,
		service_type: ServiceType,
		chat_req: ChatRequest,
		options_set: ChatOptionsSet<'_, '_>,
	) -> Result<WebRequestData> {
		let ServiceTarget { endpoint, auth, model } = target;

		// -- auth (Bedrock API key as a bearer token)
		let token = get_api_key(auth, &model)?;

		// -- url
		let url = Self::get_service_url(&model, service_type, endpoint)?;

		// -- headers
		let mut headers = Headers::from(("Authorization".to_string(), format!("Bearer {token}")));
		// The streaming endpoint returns application/vnd.amazon.eventstream regardless; the accept
		// header keeps the runtime from downgrading to a buffered response.
		if matches!(service_type, ServiceType::ChatStream) {
			headers.merge(("accept".to_string(), "application/vnd.amazon.eventstream".to_string()));
		}
		if let Some(extra_headers) = options_set.extra_headers() {
			headers.merge_with(extra_headers);
		}

		// -- parts (identical to native Anthropic: native cache_control, TTL preserved, and
		//    assistant-block breakpoints honored — Bedrock accepts all of these).
		let crate::adapter::anthropic::AnthropicRequestParts {
			system,
			messages,
			tools,
		} = AnthropicAdapter::into_anthropic_request_parts(chat_req)?;

		let (_, model_name) = model.model_name.namespace_and_name();

		// -- payload (Anthropic schema in body; modelId lives in the URL, NOT the body)
		let mut payload = json!({
			"anthropic_version": ANTHROPIC_VERSION,
			"messages": messages,
		});

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
		if let Some(top_p) = options_set.top_p() {
			payload.x_insert("top_p", top_p)?;
		}

		let max_tokens = options_set.max_tokens().unwrap_or_else(|| {
			if model_name.contains("claude-sonnet")
				|| model_name.contains("claude-haiku")
				|| model_name.contains("claude-3-7-sonnet")
				|| model_name.contains("claude-opus-4-5")
			{
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
		payload.x_insert("max_tokens", max_tokens)?; // required by Anthropic

		Ok(WebRequestData { url, headers, payload })
	}

	fn to_chat_response(
		model_iden: ModelIden,
		web_response: WebResponse,
		options_set: ChatOptionsSet<'_, '_>,
	) -> Result<ChatResponse> {
		let WebResponse { mut body, .. } = web_response;
		let captured_raw_body = options_set.capture_raw_body().unwrap_or_default().then(|| body.clone());

		let provider_model_name: Option<String> = body.x_remove("model").ok();
		let provider_model_iden = model_iden.from_optional_name(provider_model_name);

		// usage normalization (Anthropic style)
		let usage = body.x_take::<Value>("usage");
		let usage = usage.map(AnthropicAdapter::into_usage).unwrap_or_default();

		// content parsing (Anthropic schema)
		let mut content = MessageContent::default();
		let json_content_items: Vec<Value> = body.x_take("content")?;
		let mut text_content: Vec<String> = Vec::new();
		let mut reasoning_content: Vec<String> = Vec::new();
		let mut tool_calls: Vec<ToolCall> = vec![];
		for mut item in json_content_items {
			let typ: &str = item.x_get_as("type")?;
			match typ {
				"text" => text_content.push(item.x_take("text")?),
				"thinking" => reasoning_content.push(item.x_take("thinking")?),
				"tool_use" => {
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
				_ => {}
			}
		}
		if !tool_calls.is_empty() {
			content.extend(MessageContent::from(tool_calls));
		}
		if !text_content.is_empty() {
			content.push(text_content.join("\n"));
		}
		let reasoning_content = (!reasoning_content.is_empty()).then(|| reasoning_content.join("\n"));

		Ok(ChatResponse {
			content,
			reasoning_content,
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
		debug!("Bedrock eventstream start - using binary frame decoder");
		let streamer = BedrockEventStreamer::new(reqwest_builder, model_iden.clone(), options_set);
		let chat_stream = ChatStream::from_inter_stream(streamer);
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
			adapter_kind: AdapterKind::BedrockAnthropic,
			feature: "embeddings".to_string(),
		})
	}

	fn to_embed_response(
		_model_iden: crate::ModelIden,
		_web_response: crate::webc::WebResponse,
		_options_set: crate::embed::EmbedOptionsSet<'_, '_>,
	) -> Result<crate::embed::EmbedResponse> {
		Err(crate::Error::AdapterNotSupported {
			adapter_kind: AdapterKind::BedrockAnthropic,
			feature: "embeddings".to_string(),
		})
	}
}

// region: --- Bedrock eventstream streamer

use crate::adapter::inter_stream::{InterStreamEnd, InterStreamEvent};
use futures::Stream;
use std::collections::VecDeque;
use std::error::Error as StdError;
use std::pin::Pin;
use std::task::{Context, Poll};

enum InProgressBlock {
	Text,
	Thinking,
	ToolUse { id: String, name: String, input: String },
}

type ResponseFuture =
	Pin<Box<dyn futures::Future<Output = std::result::Result<reqwest::Response, reqwest::Error>> + Send>>;
type ByteStream = Pin<Box<dyn Stream<Item = std::result::Result<Bytes, reqwest::Error>> + Send>>;

/// Streams a Bedrock `invoke-with-response-stream` response.
///
/// The wire format is AWS's `application/vnd.amazon.eventstream` binary framing (prelude +
/// headers + payload + CRCs). Each `chunk` event's payload is `{"bytes": base64(<json>)}`, where
/// the decoded JSON is a standard Anthropic SSE-style event (`message_start`, `content_block_delta`,
/// `message_stop`, ...). We decode the frames off the raw byte stream and map the inner events to
/// `InterStreamEvent`s (mirroring the Anthropic/Vertex event mapping).
struct BedrockEventStreamer {
	reqwest_builder: Option<RequestBuilder>,
	response_future: Option<ResponseFuture>,
	bytes_stream: Option<ByteStream>,
	buf: Vec<u8>,
	options: StreamerOptions,
	done: bool,
	captured_data: StreamerCapturedData,
	sent_start: bool,
	pending: VecDeque<InterStreamEvent>,
	in_progress_block: InProgressBlock,
	// Set when the HTTP response status is not success (e.g. 403/404). The body is not
	// eventstream-framed in that case — it is a plain JSON error — so we accumulate it as text
	// and surface it verbatim instead of feeding it to the frame decoder.
	error_status: Option<u16>,
}

impl BedrockEventStreamer {
	fn new(reqwest_builder: RequestBuilder, model_iden: ModelIden, options_set: ChatOptionsSet<'_, '_>) -> Self {
		Self {
			reqwest_builder: Some(reqwest_builder),
			response_future: None,
			bytes_stream: None,
			buf: Vec::new(),
			options: StreamerOptions::new(model_iden, options_set),
			done: false,
			captured_data: Default::default(),
			sent_start: false,
			pending: VecDeque::new(),
			in_progress_block: InProgressBlock::Text,
			error_status: None,
		}
	}

	/// Try to decode one complete eventstream message from the front of `buf`.
	/// Returns `Some(DecodedFrame::Message { .. })` for a fully-buffered frame (with its parsed
	/// headers and raw payload), `Some(DecodedFrame::Corrupt)` on an impossible length, or `None`
	/// when more bytes are needed. On a decoded frame the consumed bytes are drained from `buf`.
	fn try_take_frame(&mut self) -> Option<DecodedFrame> {
		// Prelude is 12 bytes: total_len(4) + headers_len(4) + prelude_crc(4).
		if self.buf.len() < 12 {
			return None;
		}
		let total_len = u32::from_be_bytes([self.buf[0], self.buf[1], self.buf[2], self.buf[3]]) as usize;
		let headers_len = u32::from_be_bytes([self.buf[4], self.buf[5], self.buf[6], self.buf[7]]) as usize;
		// Guard against a corrupt/absurd length so we never allocate wildly or spin.
		if !(16..=64 * 1024 * 1024).contains(&total_len) {
			// Unrecoverable framing error: drop the connection's buffer.
			self.buf.clear();
			return Some(DecodedFrame::Corrupt);
		}
		if self.buf.len() < total_len {
			return None; // wait for the rest of the message
		}

		// Payload sits after prelude(12) + headers(headers_len), before the trailing message CRC(4).
		// Guard the header/payload bounds before slicing (a malformed headers_len must not panic).
		let payload_start = 12 + headers_len;
		let payload_end = total_len - 4;
		if payload_start > payload_end || payload_end > self.buf.len() {
			self.buf.clear();
			return Some(DecodedFrame::Corrupt);
		}
		// Parse headers to route on :message-type / :event-type / :exception-type.
		let headers = parse_headers(&self.buf[12..payload_start]);
		let payload = self.buf[payload_start..payload_end].to_vec();

		// Drain the consumed message (CRCs are trusted: the response is TLS-protected).
		self.buf.drain(0..total_len);

		Some(DecodedFrame::Message { headers, payload })
	}

	/// Map an inner Anthropic event JSON into queued InterStream events. Returns true if an event
	/// was queued that should be surfaced (the caller pops `pending`).
	fn handle_inner_event(&mut self, parsed: Value) {
		if !self.sent_start {
			self.sent_start = true;
			self.pending.push_back(InterStreamEvent::Start);
		}

		if self.options.capture_usage
			&& let Some(usage) = parsed
				.get("usage")
				.cloned()
				.or_else(|| parsed.get("message").and_then(|m| m.get("usage")).cloned())
		{
			// Anthropic streams usage in pieces: message_start carries input/cache tokens, a later
			// message_delta carries the output tokens. Merge (don't overwrite) so we don't drop the
			// prompt/cache counts — matching the native Anthropic streamer's accumulation.
			merge_usage(&mut self.captured_data.usage, AnthropicAdapter::into_usage(usage));
		}

		let Some(typ) = parsed.get("type").and_then(|v| v.as_str()) else {
			return;
		};
		match typ {
			"content_block_start" => {
				if let Some(cb) = parsed.get("content_block") {
					match cb.get("type").and_then(|v| v.as_str()).unwrap_or("") {
						"thinking" => self.in_progress_block = InProgressBlock::Thinking,
						"tool_use" => {
							let id = cb.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
							let name = cb.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
							self.in_progress_block = InProgressBlock::ToolUse {
								id,
								name,
								input: String::new(),
							};
						}
						_ => self.in_progress_block = InProgressBlock::Text,
					}
				}
			}
			"content_block_delta" => match &mut self.in_progress_block {
				InProgressBlock::Text => {
					if let Some(text) = parsed.get("delta").and_then(|d| d.get("text")).and_then(|v| v.as_str()) {
						match self.captured_data.content {
							Some(ref mut c) => c.push_str(text),
							None => self.captured_data.content = Some(text.to_string()),
						}
						self.pending.push_back(InterStreamEvent::Chunk(text.to_string()));
					}
				}
				InProgressBlock::Thinking => {
					if let Some(thinking) = parsed.get("delta").and_then(|d| d.get("thinking")).and_then(|v| v.as_str())
					{
						if self.options.capture_reasoning_content {
							match self.captured_data.reasoning_content {
								Some(ref mut r) => r.push_str(thinking),
								None => self.captured_data.reasoning_content = Some(thinking.to_string()),
							}
						}
						self.pending.push_back(InterStreamEvent::ReasoningChunk(thinking.to_string()));
					}
				}
				InProgressBlock::ToolUse { input, .. } => {
					if let Some(part) = parsed.get("delta").and_then(|d| d.get("partial_json")).and_then(|v| v.as_str())
					{
						input.push_str(part);
					}
				}
			},
			"content_block_stop" => {
				if let InProgressBlock::ToolUse { id, name, input } =
					std::mem::replace(&mut self.in_progress_block, InProgressBlock::Text)
				{
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
					self.pending.push_back(InterStreamEvent::ToolCallChunk(tc));
				}
			}
			"message_stop" => {
				self.done = true;
				self.pending.push_back(InterStreamEvent::End(InterStreamEnd {
					captured_usage: self.captured_data.usage.take(),
					captured_text_content: self.captured_data.content.take(),
					captured_reasoning_content: self.captured_data.reasoning_content.take(),
					captured_tool_calls: self.captured_data.tool_calls.take(),
					captured_thought_signatures: None,
				}));
			}
			// message_start / message_delta / ping: usage already captured above; nothing to emit.
			_ => {}
		}
	}
}

enum DecodedFrame {
	Message { headers: FrameHeaders, payload: Vec<u8> },
	Corrupt,
}

/// Merge a per-event `Usage` (as produced by `AnthropicAdapter::into_usage`) into the running
/// total. Anthropic splits usage across events (input/cache on `message_start`, output on
/// `message_delta`), and each event reports 0 for the fields it does not carry. Taking the
/// per-field max therefore keeps the input/cache counts from `message_start` and the output count
/// from `message_delta` — and, unlike a blind add, it will not double-count a field that a later
/// event redundantly echoes. `total_tokens` is recomputed from the merged prompt/completion.
fn merge_usage(acc: &mut Option<crate::chat::Usage>, incoming: crate::chat::Usage) {
	let slot = acc.get_or_insert_with(Default::default);
	let max_opt = |a: Option<i32>, b: Option<i32>| match (a, b) {
		(Some(a), Some(b)) => Some(a.max(b)),
		(x, None) => x,
		(None, y) => y,
	};
	slot.prompt_tokens = max_opt(slot.prompt_tokens, incoming.prompt_tokens);
	slot.completion_tokens = max_opt(slot.completion_tokens, incoming.completion_tokens);
	// Keep whichever detail breakdown is present (input/cache details arrive with message_start).
	if incoming.prompt_tokens_details.is_some() {
		slot.prompt_tokens_details = incoming.prompt_tokens_details;
	}
	if incoming.completion_tokens_details.is_some() {
		slot.completion_tokens_details = incoming.completion_tokens_details;
	}
	if slot.prompt_tokens.is_some() || slot.completion_tokens.is_some() {
		slot.total_tokens = Some(slot.prompt_tokens.unwrap_or(0) + slot.completion_tokens.unwrap_or(0));
	}
}

/// The eventstream header values genai cares about. AWS marks normal model output with
/// `:message-type=event` + `:event-type=chunk`, and modeled errors with `:message-type=exception`
/// + `:exception-type=<Name>` (no `:event-type`), so both must be inspected.
#[derive(Default)]
struct FrameHeaders {
	message_type: Option<String>,
	event_type: Option<String>,
	exception_type: Option<String>,
}

/// Parse the eventstream header block, extracting the string headers genai routes on.
/// Header layout: name_len(u8) | name | value_type(u8) | [value_len(u16) | value] (string=7).
fn parse_headers(headers: &[u8]) -> FrameHeaders {
	let mut out = FrameHeaders::default();
	// Best-effort: a malformed header block simply yields whatever was parsed so far.
	let _ = (|| -> Option<()> {
		let mut i = 0usize;
		while i < headers.len() {
			let name_len = *headers.get(i)? as usize;
			i += 1;
			let name = headers.get(i..i + name_len)?;
			i += name_len;
			let value_type = *headers.get(i)?;
			i += 1;
			match value_type {
				// string (7) and byte_array (6): 2-byte length prefix + value.
				6 | 7 => {
					let hi = *headers.get(i)? as usize;
					let lo = *headers.get(i + 1)? as usize;
					let val_len = (hi << 8) | lo;
					i += 2;
					let value = headers.get(i..i + val_len)?;
					i += val_len;
					match name {
						b":message-type" => out.message_type = Some(String::from_utf8_lossy(value).into_owned()),
						b":event-type" => out.event_type = Some(String::from_utf8_lossy(value).into_owned()),
						b":exception-type" => out.exception_type = Some(String::from_utf8_lossy(value).into_owned()),
						_ => {}
					}
				}
				0 | 1 => {} // bool true/false: no value bytes
				2 => i += 1,
				3 => i += 2,
				4 => i += 4,
				5 | 8 => i += 8,  // long / timestamp
				9 => i += 16,     // uuid
				_ => return None, // unknown type: cannot compute remaining offsets, stop
			}
		}
		Some(())
	})();
	out
}

impl Stream for BedrockEventStreamer {
	type Item = crate::Result<InterStreamEvent>;

	fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
		// Serve queued events first (even after `done`, to flush End).
		if let Some(ev) = self.pending.pop_front() {
			return Poll::Ready(Some(Ok(ev)));
		}
		if self.done {
			return Poll::Ready(None);
		}

		loop {
			// On an HTTP error the body is plain JSON, not eventstream framing — skip frame
			// decoding entirely and let the byte-stream arm accumulate the body, then surface it
			// on close.
			if self.error_status.is_none() {
				// Drain any complete frames already buffered.
				match self.try_take_frame() {
					Some(DecodedFrame::Corrupt) => {
						self.done = true;
						return Poll::Ready(Some(Err(crate::Error::WebStream {
							model_iden: self.options.model_iden.clone(),
							cause: "Bedrock eventstream: corrupt frame length".to_string(),
						})));
					}
					Some(DecodedFrame::Message { headers, payload }) => {
						// AWS marks modeled errors (throttling, modelStreamError, timeout, ...) with
						// :message-type=exception|error and/or an :exception-type, and they carry no
						// :event-type. Surface those as a stream error rather than parsing them as chunks.
						let is_error = headers.exception_type.is_some()
							|| matches!(headers.message_type.as_deref(), Some("exception") | Some("error"));
						if is_error {
							self.done = true;
							let kind = headers
								.exception_type
								.as_deref()
								.or(headers.message_type.as_deref())
								.unwrap_or("exception");
							let msg = String::from_utf8_lossy(&payload).into_owned();
							return Poll::Ready(Some(Err(crate::Error::WebStream {
								model_iden: self.options.model_iden.clone(),
								cause: format!("Bedrock eventstream {kind}: {msg}"),
							})));
						}

						// Normal output: :message-type=event, :event-type=chunk. Be lenient about the
						// exact event-type (only errors are special-cased above).
						if let Ok(outer) = serde_json::from_slice::<Value>(&payload)
							&& let Some(inner_b64) = outer.get("bytes").and_then(|v| v.as_str())
							&& let Ok(inner_bytes) = B64.decode(inner_b64)
							&& let Ok(parsed) = serde_json::from_slice::<Value>(&inner_bytes)
						{
							self.handle_inner_event(parsed);
						}
						if let Some(ev) = self.pending.pop_front() {
							return Poll::Ready(Some(Ok(ev)));
						}
						continue;
					}
					None => {} // need more bytes
				}
			}

			// Establish the connection lazily on first poll.
			if let Some(reqwest_builder) = self.reqwest_builder.take() {
				let fut = async move { reqwest_builder.send().await };
				self.response_future = Some(Box::pin(fut));
			}

			if let Some(fut) = self.response_future.as_mut() {
				match fut.as_mut().poll(cx) {
					Poll::Ready(Ok(response)) => {
						let status = response.status();
						debug!("Bedrock eventstream connected: status={}", status.as_u16());
						// On a non-success status the body is a plain JSON error, not eventstream
						// framing. Remember that so we surface the body verbatim instead of trying
						// to decode frames out of it (which otherwise reads as "corrupt frame").
						if !status.is_success() {
							self.error_status = Some(status.as_u16());
						}
						let stream = response.bytes_stream();
						self.bytes_stream = Some(Box::pin(stream));
						self.response_future = None;
					}
					Poll::Ready(Err(e)) => {
						self.done = true;
						self.response_future = None;
						return Poll::Ready(Some(Err(crate::Error::WebStream {
							model_iden: self.options.model_iden.clone(),
							cause: e.to_string(),
						})));
					}
					Poll::Pending => return Poll::Pending,
				}
			}

			if let Some(stream) = self.bytes_stream.as_mut() {
				match stream.as_mut().poll_next(cx) {
					Poll::Ready(Some(Ok(bytes))) => {
						self.buf.extend_from_slice(&bytes);
						continue; // try to decode
					}
					Poll::Ready(Some(Err(e))) => {
						self.done = true;
						return Poll::Ready(Some(Err(crate::Error::WebStream {
							model_iden: self.options.model_iden.clone(),
							cause: (&e as &dyn StdError).to_string(),
						})));
					}
					Poll::Ready(None) => {
						// Stream ended.
						self.done = true;
						// Non-success HTTP: the accumulated body is a plain JSON error; surface it.
						if let Some(code) = self.error_status {
							let body = String::from_utf8_lossy(&self.buf).into_owned();
							return Poll::Ready(Some(Err(crate::Error::WebStream {
								model_iden: self.options.model_iden.clone(),
								cause: format!("Bedrock HTTP {code}: {}", body.trim()),
							})));
						}
						// Leftover bytes that never formed a complete frame mean the response was
						// truncated mid-frame: report an error rather than a clean End.
						if !self.buf.is_empty() {
							return Poll::Ready(Some(Err(crate::Error::WebStream {
								model_iden: self.options.model_iden.clone(),
								cause: format!(
									"Bedrock eventstream: connection closed with {} buffered bytes (incomplete frame)",
									self.buf.len()
								),
							})));
						}
						// A clean close before any event is just an empty stream.
						if !self.sent_start {
							return Poll::Ready(None);
						}
						// Flush End if the model never sent an explicit message_stop.
						return Poll::Ready(Some(Ok(InterStreamEvent::End(InterStreamEnd {
							captured_usage: self.captured_data.usage.take(),
							captured_text_content: self.captured_data.content.take(),
							captured_reasoning_content: self.captured_data.reasoning_content.take(),
							captured_tool_calls: self.captured_data.tool_calls.take(),
							captured_thought_signatures: None,
						}))));
					}
					Poll::Pending => return Poll::Pending,
				}
			}

			return Poll::Ready(None);
		}
	}
}

// endregion: --- Bedrock eventstream streamer

#[cfg(test)]
mod tests {
	use super::*;

	/// Build one AWS eventstream message from `(name, value)` string headers and a payload.
	/// CRCs are written as zero (the decoder trusts TLS and does not validate them).
	fn make_frame_with_headers(hdrs: &[(&str, &str)], payload: &[u8]) -> Vec<u8> {
		// each header: name_len(u8) | name | value_type(u8=7) | value_len(u16) | value
		let mut headers = Vec::new();
		for (name, value) in hdrs {
			headers.push(name.len() as u8);
			headers.extend_from_slice(name.as_bytes());
			headers.push(7u8); // string
			headers.extend_from_slice(&(value.len() as u16).to_be_bytes());
			headers.extend_from_slice(value.as_bytes());
		}

		let total_len = 12 + headers.len() + payload.len() + 4;
		let mut msg = Vec::with_capacity(total_len);
		msg.extend_from_slice(&(total_len as u32).to_be_bytes());
		msg.extend_from_slice(&(headers.len() as u32).to_be_bytes());
		msg.extend_from_slice(&0u32.to_be_bytes()); // prelude crc (unchecked)
		msg.extend_from_slice(&headers);
		msg.extend_from_slice(payload);
		msg.extend_from_slice(&0u32.to_be_bytes()); // message crc (unchecked)
		msg
	}

	/// Convenience: a normal model-output frame (:message-type=event, :event-type=chunk).
	fn make_frame(event_type: &str, payload: &[u8]) -> Vec<u8> {
		make_frame_with_headers(&[(":message-type", "event"), (":event-type", event_type)], payload)
	}

	fn streamer() -> BedrockEventStreamer {
		// options_set is only used for capture flags; default (all false) is fine here.
		let model_iden = ModelIden::new(AdapterKind::BedrockAnthropic, "test-model");
		BedrockEventStreamer::new(
			reqwest::Client::new().post("http://localhost/"),
			model_iden,
			ChatOptionsSet::default(),
		)
	}

	#[test]
	fn parses_multiple_headers() {
		let frame = make_frame("chunk", b"{}");
		let headers_len = u32::from_be_bytes([frame[4], frame[5], frame[6], frame[7]]) as usize;
		let h = parse_headers(&frame[12..12 + headers_len]);
		assert_eq!(h.message_type.as_deref(), Some("event"));
		assert_eq!(h.event_type.as_deref(), Some("chunk"));
		assert!(h.exception_type.is_none());
	}

	#[test]
	fn parses_exception_headers() {
		let frame = make_frame_with_headers(
			&[(":message-type", "exception"), (":exception-type", "throttlingException")],
			br#"{"message":"slow down"}"#,
		);
		let headers_len = u32::from_be_bytes([frame[4], frame[5], frame[6], frame[7]]) as usize;
		let h = parse_headers(&frame[12..12 + headers_len]);
		assert_eq!(h.message_type.as_deref(), Some("exception"));
		assert_eq!(h.exception_type.as_deref(), Some("throttlingException"));
		assert!(h.event_type.is_none());
	}

	#[test]
	fn decodes_single_chunk_frame() {
		let mut s = streamer();
		let frame = make_frame("chunk", br#"{"bytes":"abc"}"#);
		s.buf.extend_from_slice(&frame);
		match s.try_take_frame() {
			Some(DecodedFrame::Message { headers, payload }) => {
				assert_eq!(headers.event_type.as_deref(), Some("chunk"));
				assert_eq!(payload, br#"{"bytes":"abc"}"#);
			}
			_ => panic!("expected a decoded message frame"),
		}
		assert!(s.buf.is_empty(), "consumed frame should be drained");
	}

	#[test]
	fn merges_usage_across_events() {
		// message_start: input/cache only. message_delta: output only. Merge must keep both.
		let mut acc = None;
		merge_usage(
			&mut acc,
			AnthropicAdapter::into_usage(json!({
				"input_tokens": 100,
				"cache_read_input_tokens": 20,
				"output_tokens": 0
			})),
		);
		merge_usage(
			&mut acc,
			AnthropicAdapter::into_usage(json!({ "input_tokens": 0, "output_tokens": 42 })),
		);
		let u = acc.expect("usage");
		// prompt = input(100) + cache(20) folded by into_usage; output from the delta.
		assert_eq!(u.prompt_tokens, Some(120));
		assert_eq!(u.completion_tokens, Some(42));
		assert_eq!(u.total_tokens, Some(162));
	}

	#[test]
	fn waits_for_partial_frame() {
		let mut s = streamer();
		let frame = make_frame("chunk", b"{}");
		// Feed all but the last byte: decoder must ask for more (None) and keep the buffer.
		s.buf.extend_from_slice(&frame[..frame.len() - 1]);
		assert!(s.try_take_frame().is_none());
		assert_eq!(s.buf.len(), frame.len() - 1, "partial frame must be retained");
		// Feed the final byte: now it decodes.
		s.buf.push(*frame.last().unwrap());
		assert!(matches!(s.try_take_frame(), Some(DecodedFrame::Message { .. })));
	}

	#[test]
	fn decodes_two_frames_back_to_back() {
		let mut s = streamer();
		s.buf.extend_from_slice(&make_frame("chunk", b"A"));
		s.buf.extend_from_slice(&make_frame("chunk", b"B"));
		let f1 = s.try_take_frame();
		let f2 = s.try_take_frame();
		assert!(matches!(f1, Some(DecodedFrame::Message { .. })));
		assert!(matches!(f2, Some(DecodedFrame::Message { .. })));
		assert!(s.buf.is_empty());
		assert!(s.try_take_frame().is_none());
	}

	#[test]
	fn end_to_end_inner_event_produces_chunk() {
		// A chunk frame whose payload is {"bytes": base64(<anthropic content_block_delta>)}.
		let inner = br#"{"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}"#;
		let outer = json!({ "bytes": B64.encode(inner) });
		let frame = make_frame("chunk", outer.to_string().as_bytes());

		let mut s = streamer();
		s.buf.extend_from_slice(&frame);
		let decoded = s.try_take_frame();
		let DecodedFrame::Message { payload, .. } = decoded.unwrap() else {
			panic!("expected message");
		};
		let outer_val: Value = serde_json::from_slice(&payload).unwrap();
		let inner_b64 = outer_val.get("bytes").unwrap().as_str().unwrap();
		let inner_bytes = B64.decode(inner_b64).unwrap();
		let parsed: Value = serde_json::from_slice(&inner_bytes).unwrap();
		s.handle_inner_event(parsed);

		// Start is queued first, then the text chunk.
		assert!(matches!(s.pending.pop_front(), Some(InterStreamEvent::Start)));
		match s.pending.pop_front() {
			Some(InterStreamEvent::Chunk(t)) => assert_eq!(t, "hi"),
			other => panic!("expected text chunk, got {other:?}"),
		}
	}
}
