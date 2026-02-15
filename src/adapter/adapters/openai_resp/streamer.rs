use crate::adapter::adapters::support::{StreamerCapturedData, StreamerOptions};
use crate::adapter::inter_stream::{InterStreamEnd, InterStreamEvent};
use crate::adapter::openai_resp::resp_types::RespResponse;
use crate::chat::{ChatOptionsSet, ToolCall, Usage};
use crate::webc::{Event, EventSourceStream};
use crate::{Error, ModelIden, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::pin::Pin;
use std::task::{Context, Poll};
use value_ext::JsonValueExt;

pub struct OpenAIRespStreamer {
	inner: EventSourceStream,
	options: StreamerOptions,
	done: bool,
	captured_data: StreamerCapturedData,
	fn_args_by_item_id: HashMap<String, String>,
	fn_name_by_item_id: HashMap<String, String>,
}

impl OpenAIRespStreamer {
	pub fn new(inner: EventSourceStream, model_iden: ModelIden, options_set: ChatOptionsSet<'_, '_>) -> Self {
		Self {
			inner,
			done: false,
			options: StreamerOptions::new(model_iden, options_set),
			captured_data: Default::default(),
			fn_args_by_item_id: HashMap::new(),
			fn_name_by_item_id: HashMap::new(),
		}
	}

	fn end_event(&mut self) -> InterStreamEvent {
		let captured_usage = if self.options.capture_usage {
			self.captured_data.usage.take()
		} else {
			None
		};
		let captured_tool_calls = if self.options.capture_tool_calls {
			self.captured_data.tool_calls.take()
		} else {
			None
		};
		let captured_text_content = if self.options.capture_content {
			self.captured_data.content.take()
		} else {
			None
		};
		let captured_reasoning_content = if self.options.capture_reasoning_content {
			self.captured_data.reasoning_content.take()
		} else {
			None
		};

		InterStreamEvent::End(InterStreamEnd {
			captured_usage,
			captured_text_content,
			captured_reasoning_content,
			captured_tool_calls,
			captured_thought_signatures: self.captured_data.thought_signatures.take(),
		})
	}

	fn capture_text_delta(&mut self, delta: &str) {
		if !self.options.capture_content {
			return;
		}
		match self.captured_data.content {
			Some(ref mut c) => c.push_str(delta),
			None => self.captured_data.content = Some(delta.to_string()),
		}
	}

	fn capture_reasoning_delta(&mut self, delta: &str) {
		if !self.options.capture_reasoning_content {
			return;
		}
		match self.captured_data.reasoning_content {
			Some(ref mut c) => c.push_str(delta),
			None => self.captured_data.reasoning_content = Some(delta.to_string()),
		}
	}

	fn capture_tool_call(&mut self, tool_call: ToolCall) {
		if !self.options.capture_tool_calls {
			return;
		}
		self.captured_data.tool_calls.get_or_insert_with(Vec::new).push(tool_call);
	}

	fn handle_event_value(&mut self, mut v: Value) -> Result<Option<InterStreamEvent>> {
		let typ = v.x_get_str("type").unwrap_or_default().to_string();

		match typ.as_str() {
			"response.output_text.delta" => {
				let delta: String = v.x_take("delta").unwrap_or_default();
				if !delta.is_empty() {
					self.capture_text_delta(&delta);
					return Ok(Some(InterStreamEvent::Chunk(delta)));
				}
			}
			"response.reasoning_text.delta" => {
				let delta: String = v.x_take("delta").unwrap_or_default();
				if !delta.is_empty() {
					self.capture_reasoning_delta(&delta);
					return Ok(Some(InterStreamEvent::ReasoningChunk(delta)));
				}
			}
			"response.function_call_arguments.delta" => {
				let item_id: String = v.x_take("item_id").unwrap_or_default();
				let delta: String = v.x_take("delta").unwrap_or_default();
				if !item_id.is_empty() && !delta.is_empty() {
					self.fn_args_by_item_id
						.entry(item_id)
						.and_modify(|s| s.push_str(&delta))
						.or_insert(delta);
				}
			}
			"response.function_call_arguments.done" => {
				let item_id: String = v.x_take("item_id").unwrap_or_default();
				let name: String = v.x_take("name").unwrap_or_default();
				let arguments: String = v.x_take("arguments").unwrap_or_default();

				if !item_id.is_empty() {
					self.fn_name_by_item_id.insert(item_id.clone(), name.clone());
				}

				let fn_arguments: Value = match serde_json::from_str(&arguments) {
					Ok(v) => v,
					Err(_) => Value::String(arguments),
				};

				let tool_call = ToolCall {
					call_id: item_id,
					fn_name: name,
					fn_arguments,
					thought_signatures: None,
				};
				self.capture_tool_call(tool_call);
			}

			"response.web_search_call.in_progress"
			| "response.web_search_call.searching"
			| "response.web_search_call.completed" => {
				let item_id: String = v.x_take("item_id").unwrap_or_default();
				if !item_id.is_empty() {
					let status = typ.split('.').last().unwrap_or("in_progress").to_string();
					let tool_call = ToolCall {
						call_id: item_id,
						fn_name: "web_search".to_string(),
						fn_arguments: serde_json::json!({"status": status}),
						thought_signatures: None,
					};
					self.capture_tool_call(tool_call);
				}
			}

			"response.file_search_call.in_progress"
			| "response.file_search_call.searching"
			| "response.file_search_call.completed" => {
				let item_id: String = v.x_take("item_id").unwrap_or_default();
				if !item_id.is_empty() {
					let status = typ.split('.').last().unwrap_or("in_progress").to_string();
					let tool_call = ToolCall {
						call_id: item_id,
						fn_name: "file_search".to_string(),
						fn_arguments: serde_json::json!({"status": status}),
						thought_signatures: None,
					};
					self.capture_tool_call(tool_call);
				}
			}

			"response.completed" | "response.failed" | "response.incomplete" => {
				if self.options.capture_usage {
					if let Ok(resp_val) = v.x_take::<Value>("response") {
						if let Ok(resp) = serde_json::from_value::<RespResponse>(resp_val) {
							let usage = resp.usage.map(Usage::from).unwrap_or_default();
							self.captured_data.usage = Some(usage);
						}
					}
				}

				self.done = true;
				return Ok(Some(self.end_event()));
			}
			"error" => {
				return Err(Error::ChatResponse {
					model_iden: self.options.model_iden.clone(),
					body: v,
				});
			}
			_ => {}
		}

		Ok(None)
	}
}

impl futures::Stream for OpenAIRespStreamer {
	type Item = Result<InterStreamEvent>;

	fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
		if self.done {
			return Poll::Ready(None);
		}

		while let Poll::Ready(event) = Pin::new(&mut self.inner).poll_next(cx) {
			match event {
				Some(Ok(Event::Open)) => return Poll::Ready(Some(Ok(InterStreamEvent::Start))),
				Some(Ok(Event::Message(message))) => {
					if message.data == "[DONE]" {
						self.done = true;
						return Poll::Ready(Some(Ok(self.end_event())));
					}

					let v: Value = serde_json::from_str(&message.data).map_err(|serde_error| Error::StreamParse {
						model_iden: self.options.model_iden.clone(),
						serde_error,
					})?;

					if let Some(out) = self.handle_event_value(v)? {
						return Poll::Ready(Some(Ok(out)));
					}

					continue;
				}
				Some(Err(err)) => {
					return Poll::Ready(Some(Err(Error::WebStream {
						model_iden: self.options.model_iden.clone(),
						cause: err.to_string(),
					})));
				}
				None => {
					// Stream ended without a terminal event; finalize what we have.
					self.done = true;
					return Poll::Ready(Some(Ok(self.end_event())));
				}
			}
		}

		Poll::Pending
	}
}
