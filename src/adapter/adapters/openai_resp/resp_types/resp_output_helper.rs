use crate::chat::{ContentPart, ToolCall};
use crate::{Error, Result};
use serde_json::Value;
use value_ext::JsonValueExt;

/// Convert a OpenAI response output Item to a ContentPart
///
/// NOTE: At this point this is infallible, will ignore item that cannot be transformed
impl ContentPart {
	pub fn from_resp_output_item(mut item_value: Value) -> Result<Vec<Self>> {
		let mut parts = Vec::new();
		let Some(item_type) = ItemType::from_item_value(&item_value) else {
			return Ok(parts);
		};

		match item_type {
			ItemType::Message => {
				if let Ok(content) = item_value.x_remove::<Vec<Value>>("content") {
					// each content item {}
					for mut content_item in content {
						if let Ok("output_text") = content_item.x_get_str("type")
							&& let Ok(text) = content_item.x_remove::<String>("text")
						{
							parts.push(text.into())
						}
					}
				}
			}
			ItemType::FunctionCall => {
				let fn_name = item_value.x_remove::<String>("name")?;
				let call_id = item_value.x_remove::<String>("call_id")?;
				let arguments = item_value.x_remove::<String>("arguments")?;
				let fn_arguments: Value =
					serde_json::from_str(&arguments).map_err(|_| Error::InvalidJsonResponseElement {
						info: "tool call arguments is not an object.\nCause",
					})?;

				let tool_call = ToolCall {
					call_id,
					fn_name,
					fn_arguments,
					thought_signatures: None,
				};

				parts.push(tool_call.into());
			}
			ItemType::BuiltinToolCall(tool_type) => {
				// Example: {"type":"web_search_call","id":"ws_...","status":"completed","action":{...}}
				let call_id = item_value
					.x_take::<String>("id")
					.or_else(|_| item_value.x_take::<String>("call_id"))
					.or_else(|_| item_value.x_take::<String>("item_id"))?;

				// Remove type so arguments only contains tool-specific fields.
				let _ = item_value.x_remove::<String>("type");

				let fn_name = tool_type.strip_suffix("_call").unwrap_or(&tool_type).to_string();

				let tool_call = ToolCall {
					call_id,
					fn_name,
					fn_arguments: item_value,
					thought_signatures: None,
				};
				parts.push(tool_call.into());
			}
		}

		Ok(parts)
	}
}

// region:    --- Support Type

/// The managed
enum ItemType {
	Message,
	FunctionCall,
	BuiltinToolCall(String),
}

impl ItemType {
	fn from_item_value(item_value: &Value) -> Option<Self> {
		let typ = item_value.x_get_str("type").ok()?;
		match typ {
			"message" => Some(ItemType::Message),
			"function_call" => Some(ItemType::FunctionCall),
			other if other.ends_with("_call") => Some(ItemType::BuiltinToolCall(other.to_string())),
			_ => None,
		}
	}
}

// endregion: --- Support Type

#[cfg(test)]
mod tests {
	use super::*;
	use serde_json::json;

	#[test]
	fn parses_web_search_call_as_tool_call() {
		let item = json!({
			"type": "web_search_call",
			"id": "ws_123",
			"status": "completed",
			"action": {"type": "search", "queries": ["fennel seeds nutrition"]}
		});

		let parts = ContentPart::from_resp_output_item(item).unwrap();
		assert_eq!(parts.len(), 1);

		let Some(tool_call) = parts[0].as_tool_call() else {
			panic!("expected tool call");
		};

		assert_eq!(tool_call.call_id, "ws_123");
		assert_eq!(tool_call.fn_name, "web_search");
		assert_eq!(tool_call.fn_arguments.x_get_str("status").unwrap(), "completed");
		assert_eq!(tool_call.fn_arguments.x_get_str("/action/type").unwrap(), "search");
	}
}
