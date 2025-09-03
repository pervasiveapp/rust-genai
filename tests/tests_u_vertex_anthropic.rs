use genai::adapter::{AdapterDispatcher, AdapterKind, ServiceType};
use genai::chat::{CacheControl, ChatMessage, ChatOptionsSet, ChatRequest};
use genai::resolver::{AuthData, Endpoint};
use genai::{ModelIden, ServiceTarget};
use serde_json::Value;

fn mk_target(model: &str, base_url: &str, token: &str) -> ServiceTarget {
	let model = ModelIden::new(AdapterKind::VertexAnthropic, model);
	let endpoint = Endpoint::from_static(base_url);
	let auth = AuthData::from_single(token);
	ServiceTarget { endpoint, auth, model }
}

#[test]
fn test_vertex_anthropic_url_build() {
	let base = "https://aiplatform.googleapis.com/v1/projects/p/locations/l/";
	let target = mk_target("claude-opus-4-20250514", base, "tkn");
	let url = AdapterDispatcher::get_service_url(&target.model, ServiceType::Chat, target.endpoint.clone());
	assert_eq!(
		url,
		format!("{}publishers/anthropic/models/claude-opus-4-20250514:rawPredict", base)
	);
	let url_stream = AdapterDispatcher::get_service_url(&target.model, ServiceType::ChatStream, target.endpoint);
	assert_eq!(
		url_stream,
		format!(
			"{}publishers/anthropic/models/claude-opus-4-20250514:streamRawPredict",
			base
		)
	);
}

#[test]
fn test_vertex_anthropic_payload_basic() {
	let base = "https://aiplatform.googleapis.com/v1/projects/p/locations/l/";
	let target = mk_target("claude-opus-4-20250514", base, "tkn");

	let chat_req = ChatRequest::new(vec![ChatMessage::system("sys"), ChatMessage::user("hi")]);
	let options_set = ChatOptionsSet::default();

	let req =
		AdapterDispatcher::to_web_request_data(target, ServiceType::Chat, chat_req, options_set).expect("build ok");

	// header contains Authorization: Bearer
	let auth_header = req.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case("authorization"));
	assert!(auth_header.is_some());
	assert!(auth_header.unwrap().1.starts_with("Bearer "));

	// payload assertions
	let model = req.payload.get("model").and_then(|v| v.as_str()).unwrap_or("");
	assert_eq!(model, "claude-opus-4-20250514");
	let version = req.payload.get("anthropic_version").and_then(|v| v.as_str()).unwrap_or("");
	assert_eq!(version, "2023-06-01");
	assert!(req.payload.get("messages").is_some());
}

#[test]
fn test_vertex_anthropic_cache_control_ttl_on_user() {
	let base = "https://aiplatform.googleapis.com/v1/projects/p/locations/l/";
	let target = mk_target("claude-opus-4-20250514", base, "tkn");

	let msg = ChatMessage::user("cache me").with_options(CacheControl::EphemeralWithTtl("5m".to_string()));
	let chat_req = ChatRequest::new(vec![msg]);
	let options_set = ChatOptionsSet::default();

	let req =
		AdapterDispatcher::to_web_request_data(target, ServiceType::Chat, chat_req, options_set).expect("build ok");

	let messages = req.payload.get("messages").and_then(|v| v.as_array()).expect("messages array");
	let first = messages.first().cloned().unwrap_or(Value::Null);
	// For text-only, adapter collapses to { role: user, content: [ { type: text, text: ..., cache_control: { type: "ephemeral", ttl: "5m" } } ] }
	let content = first.get("content").and_then(|v| v.as_array()).expect("content array");
	let part = content.last().expect("last part with cache_control");
	let cc = part.get("cache_control").and_then(|v| v.get("ttl")).and_then(|v| v.as_str());
	assert_eq!(cc, Some("5m"));
}
