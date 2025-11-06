use crate::adapter::{AdapterDispatcher, AdapterKind, ServiceType, WebRequestData};
use crate::chat::{ChatOptions, ChatOptionsSet, ChatRequest, ChatResponse, ChatStreamResponse};
use crate::client::Headers;
use crate::embed::{EmbedOptions, EmbedOptionsSet, EmbedRequest, EmbedResponse};
use crate::resolver::AuthData;
use crate::{Client, Error, ModelIden, Result, ServiceTarget};

/// High-level client APIs.
impl Client {
	/// Lists model names for the given adapter.
	///
	/// Notes:
	///
	/// - Non-Ollama adapters use a static list.
	///
	/// - Ollama queries the default host (http://localhost:11434/v1/).
	///
	/// - May evolve to accept a custom endpoint.
	///
	/// - For most adapters, names also drive AdapterKind detection (see [`AdapterKind`]).
	///
	/// - Adapters should filter non-chat models until more skills are supported.
	///   Future: `model_names(adapter_kind, Option<&[Skill]>)`.
	pub async fn all_model_names(&self, adapter_kind: AdapterKind) -> Result<Vec<String>> {
		let models = AdapterDispatcher::all_model_names(adapter_kind).await?;
		Ok(models)
	}

	/// Builds a ModelIden by inferring AdapterKind from the model name.
	pub fn default_model(&self, model_name: &str) -> Result<ModelIden> {
		// -- First get the default ModelInfo
		let adapter_kind = AdapterKind::from_model(model_name)?;
		let model_iden = ModelIden::new(adapter_kind, model_name);
		Ok(model_iden)
	}

	/// Deprecated: use `Client::resolve_service_target`.
	#[deprecated(note = "use `client.resolve_service_target(model_name)`")]
	pub async fn resolve_model_iden(&self, model_name: &str) -> Result<ModelIden> {
		let model = self.default_model(model_name)?;
		let target = self.config().resolve_service_target(model).await?;
		Ok(target.model)
	}

	/// Resolves the service target (endpoint, auth, and model) for the given model name.
	pub async fn resolve_service_target(&self, model_name: &str) -> Result<ServiceTarget> {
		let model = self.default_model(model_name)?;
		self.config().resolve_service_target(model).await
	}

	/// Executes a chat.
	pub async fn exec_chat(
		&self,
		model: &str,
		chat_req: ChatRequest,
		// options not implemented yet
		options: Option<&ChatOptions>,
	) -> Result<ChatResponse> {
		let options_set = ChatOptionsSet::default()
			.with_chat_options(options)
			.with_client_options(self.config().chat_options());

		let model = self.default_model(model)?;
		let target = self.config().resolve_service_target(model).await?;
		let endpoint_base = target.endpoint.base_url().to_string();
		let model = target.model.clone();
		let auth_data = target.auth.clone();

		let WebRequestData {
			mut url,
			mut headers,
			payload,
		} = AdapterDispatcher::to_web_request_data(target, ServiceType::Chat, chat_req, options_set.clone())?;

		if let Some(extra_headers) = options.and_then(|o| o.extra_headers.as_ref()) {
			headers.merge_with(&extra_headers);
		}

		if let AuthData::RequestOverride {
			url: override_url,
			headers: override_headers,
		} = auth_data
		{
			url = override_url;
			headers = override_headers;
		};

		let web_res = self.web_client().do_post(&url, &headers, payload.clone()).await.map_err(|webc_error| {
            // On failure, emit a concise diagnostic with sanitized request context and response body snippet
            let sanitized = sanitize_headers(&headers);
            let payload_snip = truncate_json(&payload, 1200);
            let (status_str, resp_body_snip) = match &webc_error {
                crate::webc::Error::ResponseFailedStatus { status, body, .. } => {
                    (status.as_str().to_string(), truncate_str(body, 2000))
                }
                _ => ("".to_string(), String::new()),
            };
            if status_str.is_empty() {
                log::warn!(
                    "GENAI HTTP error - adapter={:?} endpoint_base={} url={} headers={:?} payload_snip={}",
                    model.adapter_kind, endpoint_base, url, sanitized, payload_snip
                );
            } else {
                log::warn!(
                    "GENAI HTTP error - adapter={:?} endpoint_base={} url={} status={} headers={:?} payload_snip={} resp_body_snip={}",
                    model.adapter_kind, endpoint_base, url, status_str, sanitized, payload_snip, resp_body_snip
                );
            }
            Error::WebModelCall { model_iden: model.clone(), webc_error }
        })?;

		let chat_res = AdapterDispatcher::to_chat_response(model, web_res, options_set)?;

		Ok(chat_res)
	}

	/// Executes a chat stream response.
	pub async fn exec_chat_stream(
		&self,
		model: &str,
		chat_req: ChatRequest, // options not implemented yet
		options: Option<&ChatOptions>,
	) -> Result<ChatStreamResponse> {
		let options_set = ChatOptionsSet::default()
			.with_chat_options(options)
			.with_client_options(self.config().chat_options());

		let model = self.default_model(model)?;
		let target = self.config().resolve_service_target(model).await?;
		let endpoint_base = target.endpoint.base_url().to_string();
		let model = target.model.clone();
		let auth_data = target.auth.clone();

		let WebRequestData {
			mut url,
			mut headers,
			payload,
		} = AdapterDispatcher::to_web_request_data(target, ServiceType::ChatStream, chat_req, options_set.clone())?;

		if let Some(extra_headers) = options.and_then(|o| o.extra_headers.as_ref()) {
			headers.merge_with(&extra_headers);
		}

		// TODO: Need to check this.
		//       This was part of the 429c5cee2241dbef9f33699b9c91202233c22816 commit
		//       But now it is missing in the the exec_chat(..) above, which is probably an issue.
		if let AuthData::RequestOverride {
			url: override_url,
			headers: override_headers,
		} = auth_data
		{
			url = override_url;
			headers = override_headers;
		};

		// Log the resolved request context at debug level (only once for stream start)
		let payload_snip = truncate_json(&payload, 1200);
		log::debug!(
			"GENAI Stream start - adapter={:?} endpoint_base={} url={} headers={:?} payload_snip={}",
			model.adapter_kind,
			endpoint_base,
			url,
			sanitize_headers(&headers),
			payload_snip
		);

		let reqwest_builder = self
			.web_client()
			.new_req_builder(&url, &headers, payload)
			.map_err(|webc_error| Error::WebModelCall {
				model_iden: model.clone(),
				webc_error,
			})?;

		let res = AdapterDispatcher::to_chat_stream(model, reqwest_builder, options_set)?;

		Ok(res)
	}

	/// Creates embeddings for a single input string.
	pub async fn embed(
		&self,
		model: &str,
		input: impl Into<String>,
		options: Option<&EmbedOptions>,
	) -> Result<EmbedResponse> {
		let embed_req = EmbedRequest::new(input);
		self.exec_embed(model, embed_req, options).await
	}

	/// Creates embeddings for multiple input strings.
	pub async fn embed_batch(
		&self,
		model: &str,
		inputs: Vec<String>,
		options: Option<&EmbedOptions>,
	) -> Result<EmbedResponse> {
		let embed_req = EmbedRequest::new_batch(inputs);
		self.exec_embed(model, embed_req, options).await
	}

	/// Sends an embedding request and returns the response.
	pub async fn exec_embed(
		&self,
		model: &str,
		embed_req: EmbedRequest,
		options: Option<&EmbedOptions>,
	) -> Result<EmbedResponse> {
		let options_set = EmbedOptionsSet::new()
			.with_request_options(options)
			.with_client_options(self.config().embed_options());

		let model = self.default_model(model)?;
		let target = self.config().resolve_service_target(model).await?;
		let model = target.model.clone();

		let WebRequestData { headers, payload, url } =
			AdapterDispatcher::to_embed_request_data(target, embed_req, options_set.clone())?;

		let web_res =
			self.web_client()
				.do_post(&url, &headers, payload)
				.await
				.map_err(|webc_error| Error::WebModelCall {
					model_iden: model.clone(),
					webc_error,
				})?;

		let res = AdapterDispatcher::to_embed_response(model, web_res, options_set)?;

		Ok(res)
	}
}

// region: --- Logging helpers
fn sanitize_headers(headers: &Headers) -> Vec<(String, String)> {
	headers
		.iter()
		.map(|(k, v)| {
			let kl = k.to_ascii_lowercase();
			if kl == "authorization" {
				let masked = if let Some(rest) = v.strip_prefix("Bearer ") {
					format!("Bearer ***len={}***", rest.len())
				} else {
					"***".to_string()
				};
				(k.clone(), masked)
			} else if kl.contains("api-key") || kl == "x-goog-api-key" || kl == "x-api-key" {
				(k.clone(), "***".to_string())
			} else {
				(k.clone(), v.clone())
			}
		})
		.collect()
}

fn truncate_json(val: &serde_json::Value, max: usize) -> String {
	let s = serde_json::to_string(val).unwrap_or_else(|_| "<serde_json_error>".to_string());
	truncate_with_ellipsis(s, max)
}

fn truncate_str(s: &str, max: usize) -> String {
	truncate_with_ellipsis(s.to_owned(), max)
}

fn truncate_with_ellipsis(mut s: String, max: usize) -> String {
	if s.len() <= max {
		return s;
	}
	let mut boundary = max;
	while boundary > 0 && !s.is_char_boundary(boundary) {
		boundary -= 1;
	}

	s.truncate(boundary);
	s.push('…');
	s
}
// endregion
