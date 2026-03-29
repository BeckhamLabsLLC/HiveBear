//! CloudProtocol trait — abstracts over different API formats (OpenAI, Anthropic, Gemini, etc.)

use async_trait::async_trait;

use super::registry::ProviderDef;
use crate::error::Result;
use crate::types::{GenerateRequest, GenerateResponse, Token};

/// Trait implemented by each API protocol family.
///
/// A single protocol implementation (e.g. `OpenAiCompatProtocol`) can serve
/// many providers that share the same API shape.
#[async_trait]
pub trait CloudProtocol: Send + Sync {
    /// Build the HTTP request body from a `GenerateRequest`.
    ///
    /// When `stream` is true, the body should include `"stream": true` or
    /// the protocol's equivalent.
    fn build_request_body(
        &self,
        provider: &ProviderDef,
        model_id: &str,
        req: &GenerateRequest,
        stream: bool,
    ) -> serde_json::Value;

    /// Parse a non-streaming response body into a `GenerateResponse`.
    fn parse_response(&self, body: &serde_json::Value) -> Result<GenerateResponse>;

    /// Parse a single SSE `data:` payload during streaming.
    ///
    /// Returns `Ok(Some(token))` for normal content, `Ok(None)` for events
    /// that should be skipped (e.g. role-only deltas), and `Err` for parse failures.
    ///
    /// The caller handles the `[DONE]` sentinel before calling this method.
    fn parse_stream_chunk(&self, data: &str) -> Result<Option<Token>>;

    /// Build the full URL for the chat endpoint.
    ///
    /// Most protocols just concatenate `base_url + chat_endpoint`, but some
    /// (like Gemini) need to interpolate the model ID into the URL.
    fn build_url(&self, provider: &ProviderDef, _model_id: &str) -> String {
        format!("{}{}", provider.base_url, provider.chat_endpoint)
    }

    /// Build the authentication and extra headers for a request.
    fn build_headers(
        &self,
        provider: &ProviderDef,
        api_key: Option<&str>,
    ) -> Vec<(String, String)> {
        let mut headers = Vec::new();

        // Authentication
        if let Some(key) = api_key {
            match &provider.auth_style {
                super::registry::AuthStyle::BearerToken => {
                    headers.push(("Authorization".into(), format!("Bearer {key}")));
                }
                super::registry::AuthStyle::CustomHeader(name) => {
                    headers.push((name.to_string(), key.to_string()));
                }
                super::registry::AuthStyle::ApiKeyInUrl | super::registry::AuthStyle::None => {
                    // Handled elsewhere (URL param or no auth)
                }
            }
        }

        // Extra provider headers
        for (name, value) in provider.extra_headers {
            headers.push((name.to_string(), value.to_string()));
        }

        // Content-Type
        headers.push(("Content-Type".into(), "application/json".into()));

        headers
    }

    /// Whether this protocol supports a streaming "done" sentinel.
    /// OpenAI uses `data: [DONE]`, Anthropic uses event types, etc.
    fn done_sentinel(&self) -> Option<&str> {
        Some("[DONE]")
    }
}
