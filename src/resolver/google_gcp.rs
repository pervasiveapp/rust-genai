//! Google Cloud auth helpers (feature `gcp-auth`).
//!
//! Provides an AuthResolver that returns a short‑lived OAuth2 access token (Bearer) suitable for
//! calling Google APIs (e.g., Vertex AI). It supports two sources:
//! - A base64‑encoded service account JSON from an env var (default: `GCP_SERVICE_ACCOUNT_B64`).
//! - Application Default Credentials (ADC) via `gcp_auth::Authenticator` (metadata server, gcloud, etc.).
//!
//! The resolver fetches a fresh token on each use; the underlying gcp_auth providers cache tokens until expiry.
//!
//! Enable with Cargo feature: `gcp-auth`.

use crate::ModelIden;
use crate::resolver::{AuthData, AuthResolver, Error};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as B64;
use gcp_auth::TokenProvider;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

pub const GCP_SA_B64_ENV: &str = "GCP_SERVICE_ACCOUNT_B64";
pub const GCP_SCOPE_CLOUD_PLATFORM: &str = "https://www.googleapis.com/auth/cloud-platform";

/// Create an AuthResolver that obtains a Bearer token from either a base64‑encoded
/// service account JSON (env var) or Application Default Credentials (ADC).
///
/// - `env_var_name`: the env var name containing base64 of the service account JSON. Default is `GCP_SERVICE_ACCOUNT_B64`.
/// - `scopes`: OAuth scopes to request; defaults to cloud‑platform.
pub fn google_adc_or_sa_b64_auth_resolver(env_var_name: Option<&str>, scopes: Option<&[&str]>) -> AuthResolver {
	let env_name: Arc<str> = env_var_name.unwrap_or(GCP_SA_B64_ENV).into();
	let scopes: Vec<String> = scopes
		.map(|s| s.iter().map(|v| v.to_string()).collect())
		.unwrap_or_else(|| vec![GCP_SCOPE_CLOUD_PLATFORM.to_string()]);

	AuthResolver::from_resolver_async_fn(move |_model: ModelIden| {
		let env_name = env_name.clone();
		let scopes = scopes.clone();
		Box::pin(async move {
			// 0) If caller provided a ready Bearer token (must be an OAuth 2 access token), use it
			if let Ok(tok) = std::env::var("GOOGLE_VERTEX_TOKEN") {
				if !tok.trim().is_empty() {
					return Ok(Some(AuthData::Key(tok)));
				}
			}

			// 1) Try SA JSON from base64 env var
			if let Ok(b64) = std::env::var(&*env_name) {
				// Allow both base64 or raw JSON in the env var
				let json = match B64.decode(b64.as_bytes()) {
					Ok(bytes) => String::from_utf8(bytes).unwrap_or_default(),
					Err(_) => b64, // treat as raw JSON
				};

				let account = gcp_auth::CustomServiceAccount::from_json(&json)
					.map_err(|e| Error::Custom(format!("GCP SA JSON parse: {e}")))?;
				let scope_refs: Vec<&str> = scopes.iter().map(|s| s.as_str()).collect();
				let token = account
					.token(&scope_refs)
					.await
					.map_err(|e| Error::Custom(format!("GCP SA token: {e}")))?;
				return Ok(Some(AuthData::Key(token.as_str().to_string())));
			}

			// 2) Fallback to ADC via GOOGLE_APPLICATION_CREDENTIALS (service account file)
			if let Ok(path) = std::env::var("GOOGLE_APPLICATION_CREDENTIALS") {
				let json =
					std::fs::read_to_string(path).map_err(|e| Error::Custom(format!("GCP ADC read file: {e}")))?;
				let account = gcp_auth::CustomServiceAccount::from_json(&json)
					.map_err(|e| Error::Custom(format!("GCP ADC SA JSON parse: {e}")))?;
				let scope_refs: Vec<&str> = scopes.iter().map(|s| s.as_str()).collect();
				let token = account
					.token(&scope_refs)
					.await
					.map_err(|e| Error::Custom(format!("GCP ADC token: {e}")))?;
				return Ok(Some(AuthData::Key(token.as_str().to_string())));
			}

			// 3) Last resort: try gcloud application-default token
			if let Ok(out) = std::process::Command::new("gcloud")
				.args(["auth", "application-default", "print-access-token"])
				.output()
			{
				if out.status.success() {
					let tok = String::from_utf8_lossy(&out.stdout).trim().to_string();
					if !tok.is_empty() {
						return Ok(Some(AuthData::Key(tok)));
					}
				}
			}

			// Not available; let adapter defaults handle auth (may be OPENROUTER_API_KEY or provider-specific)
			Ok(None)
		}) as Pin<Box<dyn Future<Output = Result<_, _>> + Send>>
	})
}
