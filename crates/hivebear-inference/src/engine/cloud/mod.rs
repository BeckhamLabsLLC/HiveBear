//! Cloud provider backend — routes inference to 35+ cloud API providers.
//!
//! Architecture:
//! - `registry.rs` — declarative provider catalog (`ProviderDef`, `BUILTIN_PROVIDERS`)
//! - `protocol.rs` — `CloudProtocol` trait for API format abstraction
//! - `sse.rs` — real SSE stream parser
//! - `protocols/` — concrete protocol handlers (OpenAI-compat, Anthropic, Gemini, Cohere)

pub mod protocol;
pub mod protocols;
pub mod registry;
pub mod sse;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;

use crate::engine::{InferenceBackend, TokenStream};
use crate::error::{InferenceError, Result};
use crate::types::{GenerateRequest, GenerateResponse, LoadConfig, ModelHandle, Token};
use hivebear_core::types::{InferenceEngine, ModelFormat};

use protocol::CloudProtocol;
use registry::{ApiProtocol, ProviderDef, BUILTIN_PROVIDERS};

// ── CloudBackend ─────────────────────────────────────────────────────

/// Cloud inference backend supporting 35+ providers via a registry + protocol
/// architecture.
///
/// Adding a new OpenAI-compatible provider requires only a new `ProviderDef`
/// entry in `registry::BUILTIN_PROVIDERS` — no new code.
pub struct CloudBackend {
    /// Resolved API keys, keyed by provider prefix.
    api_keys: HashMap<String, String>,
    /// Protocol handlers, keyed by `ApiProtocol` variant.
    protocol_handlers: HashMap<ApiProtocol, Arc<dyn CloudProtocol>>,
    /// Shared HTTP client.
    client: reqwest::Client,
}

impl Default for CloudBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CloudBackend {
    /// Create a new backend resolving API keys from environment variables only.
    pub fn new() -> Self {
        let api_keys = registry::resolve_all_keys(&HashMap::new(), None, None);
        Self::with_keys(api_keys)
    }

    /// Create a backend with config-driven API keys + legacy migration.
    pub fn with_config(config: &hivebear_core::config::CloudConfig) -> Self {
        let api_keys = registry::resolve_all_keys(
            &config.api_keys,
            config.openai_api_key.as_deref(),
            config.anthropic_api_key.as_deref(),
        );
        Self::with_keys(api_keys)
    }

    fn with_keys(api_keys: HashMap<String, String>) -> Self {
        let mut protocol_handlers: HashMap<ApiProtocol, Arc<dyn CloudProtocol>> = HashMap::new();
        protocol_handlers.insert(
            ApiProtocol::OpenAiCompat,
            Arc::new(protocols::openai_compat::OpenAiCompatProtocol::new()),
        );
        protocol_handlers.insert(
            ApiProtocol::Anthropic,
            Arc::new(protocols::anthropic::AnthropicProtocol::new()),
        );
        protocol_handlers.insert(
            ApiProtocol::Gemini,
            Arc::new(protocols::gemini::GeminiProtocol::new()),
        );
        protocol_handlers.insert(
            ApiProtocol::Cohere,
            Arc::new(protocols::cohere::CohereProtocol::new()),
        );

        Self {
            api_keys,
            protocol_handlers,
            client: reqwest::Client::new(),
        }
    }

    /// Resolve a `"prefix/model_id"` string to the provider definition,
    /// API key, and model ID.
    fn resolve_provider<'a>(
        &'a self,
        model_name: &'a str,
    ) -> Result<(&'static ProviderDef, Option<&'a str>, &'a str)> {
        let (prefix, model_id) = model_name.split_once('/').ok_or_else(|| {
            InferenceError::GenerationError(format!(
                "Cloud model name must be 'provider/model', got '{model_name}'"
            ))
        })?;

        let provider = registry::find_builtin(prefix).ok_or_else(|| {
            InferenceError::GenerationError(format!(
                "Unknown cloud provider '{prefix}'. Available providers: {}",
                self.available_provider_list()
            ))
        })?;

        let api_key = self.api_keys.get(prefix).map(|s| s.as_str());

        if provider.requires_api_key && api_key.is_none() {
            return Err(InferenceError::GenerationError(format!(
                "{} API key not set. Set {} env var or add to config [cloud.api_keys.{}].",
                provider.display_name, provider.env_var, provider.prefix
            )));
        }

        Ok((provider, api_key, model_id))
    }

    /// Get the protocol handler for a provider.
    fn get_protocol(&self, provider: &ProviderDef) -> Result<&Arc<dyn CloudProtocol>> {
        self.protocol_handlers
            .get(&provider.protocol)
            .ok_or_else(|| {
                InferenceError::GenerationError(format!(
                    "No protocol handler for {:?} (provider '{}')",
                    provider.protocol, provider.display_name
                ))
            })
    }

    /// List available providers that have API keys configured.
    pub fn available_providers(&self) -> Vec<&'static ProviderDef> {
        BUILTIN_PROVIDERS
            .iter()
            .filter(|p| !p.requires_api_key || self.api_keys.contains_key(p.prefix))
            .collect()
    }

    /// Check if a specific provider prefix is known.
    pub fn has_provider(prefix: &str) -> bool {
        registry::find_builtin(prefix).is_some()
    }

    /// Comma-separated list of all known provider prefixes.
    fn available_provider_list(&self) -> String {
        BUILTIN_PROVIDERS
            .iter()
            .map(|p| p.prefix)
            .collect::<Vec<_>>()
            .join(", ")
    }
}

#[async_trait]
impl InferenceBackend for CloudBackend {
    fn engine_id(&self) -> InferenceEngine {
        InferenceEngine::Cloud
    }

    fn name(&self) -> &str {
        "Cloud"
    }

    fn supported_formats(&self) -> &[ModelFormat] {
        &[] // Cloud doesn't load local files
    }

    fn is_available(&self) -> bool {
        // Available if any provider has a key, or any local provider exists
        BUILTIN_PROVIDERS
            .iter()
            .any(|p| !p.requires_api_key || self.api_keys.contains_key(p.prefix))
    }

    fn supports_grammar(&self) -> bool {
        false
    }

    async fn load_model(&self, path: &Path, _config: &LoadConfig) -> Result<ModelHandle> {
        let model_name = path.to_str().unwrap_or("unknown");
        // Validate that the provider is accessible
        self.resolve_provider(model_name)?;
        Ok(ModelHandle::new(path.to_path_buf(), InferenceEngine::Cloud))
    }

    async fn generate(
        &self,
        handle: &ModelHandle,
        req: &GenerateRequest,
    ) -> Result<GenerateResponse> {
        let model_name = handle.model_path.to_str().unwrap_or("unknown");
        let (provider, api_key, model_id) = self.resolve_provider(model_name)?;
        let protocol = self.get_protocol(provider)?;

        let url = protocol.build_url(provider, model_id);
        let headers = protocol.build_headers(provider, api_key);
        let body = protocol.build_request_body(provider, model_id, req, false);

        // Build the request
        let mut request = self.client.post(&url);
        for (name, value) in &headers {
            request = request.header(name, value);
        }

        // For Gemini: append API key as URL query param
        if let registry::AuthStyle::ApiKeyInUrl = &provider.auth_style {
            if let Some(key) = api_key {
                let separator = if url.contains('?') { "&" } else { "?" };
                let url_with_key = format!("{url}{separator}key={key}");
                request = self.client.post(&url_with_key);
                for (name, value) in &headers {
                    request = request.header(name, value);
                }
                request = request.json(&body);
            } else {
                request = request.json(&body);
            }
        } else {
            request = request.json(&body);
        }

        let resp = request
            .send()
            .await
            .map_err(|e| InferenceError::GenerationError(format!("HTTP error: {e}")))?;

        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| InferenceError::GenerationError(format!("Read error: {e}")))?;

        if !status.is_success() {
            return Err(InferenceError::GenerationError(format!(
                "{} API error ({}): {}",
                provider.display_name, status, text
            )));
        }

        let json: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| InferenceError::GenerationError(format!("Parse error: {e}")))?;

        protocol.parse_response(&json)
    }

    fn stream(&self, handle: &ModelHandle, req: &GenerateRequest) -> TokenStream {
        let handle = handle.clone();
        let req = req.clone();
        let api_keys = self.api_keys.clone();
        let protocol_handlers = self.protocol_handlers.clone();
        let client = self.client.clone();

        let stream = async_stream::try_stream! {
            let model_name = handle.model_path.to_str().unwrap_or("unknown");

            let (prefix, model_id) = model_name.split_once('/').ok_or_else(|| {
                InferenceError::GenerationError(format!(
                    "Cloud model name must be 'provider/model', got '{model_name}'"
                ))
            })?;

            let provider = registry::find_builtin(prefix).ok_or_else(|| {
                InferenceError::GenerationError(format!("Unknown cloud provider '{prefix}'"))
            })?;

            let api_key = api_keys.get(prefix).map(|s| s.as_str());

            if provider.requires_api_key && api_key.is_none() {
                Err(InferenceError::GenerationError(format!(
                    "{} API key not set. Set {} env var.",
                    provider.display_name, provider.env_var
                )))?;
            }

            let protocol = protocol_handlers
                .get(&provider.protocol)
                .ok_or_else(|| {
                    InferenceError::GenerationError(format!(
                        "No protocol handler for {:?}",
                        provider.protocol
                    ))
                })?
                .clone();

            let mut url = protocol.build_url(provider, model_id);
            let headers = protocol.build_headers(provider, api_key);
            let body = protocol.build_request_body(provider, model_id, &req, true);

            // For Gemini: append API key as URL query param
            if let registry::AuthStyle::ApiKeyInUrl = &provider.auth_style {
                if let Some(key) = api_key {
                    let separator = if url.contains('?') { "&" } else { "?" };
                    url = format!("{url}{separator}key={key}");
                }
            }

            let mut request = client.post(&url);
            for (name, value) in &headers {
                request = request.header(name, value);
            }
            request = request.json(&body);

            let resp = request
                .send()
                .await
                .map_err(|e| InferenceError::GenerationError(format!("HTTP error: {e}")))?;

            let status = resp.status();
            let resp = if !status.is_success() {
                let text = resp.text().await.unwrap_or_default();
                Err(InferenceError::GenerationError(format!(
                    "{} API error ({}): {}",
                    provider.display_name, status, text
                )))?;
                // unreachable, but needed for type inference
                return;
            } else {
                resp
            };

            // Set up the SSE parser with the protocol's chunk parser
            let done_sentinel = protocol.done_sentinel().map(|s| s.to_string());
            let protocol_clone = protocol.clone();
            let parse_fn = move |data: &str| -> Result<Option<Token>> {
                protocol_clone.parse_stream_chunk(data)
            };

            // Use the real SSE stream parser
            let mut token_stream = sse::stream_sse_with_parser(resp, done_sentinel, parse_fn);

            use futures::StreamExt;
            while let Some(token_result) = token_stream.next().await {
                match token_result {
                    Ok(token) => yield token,
                    Err(e) => {
                        tracing::debug!("Stream token error: {e}");
                        Err(e)?;
                    }
                }
            }
        };

        Box::pin(stream)
    }

    async fn unload(&self, _handle: &ModelHandle) -> Result<()> {
        Ok(()) // Nothing to unload for cloud
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cloud_backend_creates() {
        let backend = CloudBackend::new();
        assert_eq!(backend.engine_id(), InferenceEngine::Cloud);
        assert_eq!(backend.name(), "Cloud");
        assert!(backend.supported_formats().is_empty());
    }

    #[test]
    fn test_has_provider() {
        assert!(CloudBackend::has_provider("openai"));
        assert!(CloudBackend::has_provider("groq"));
        assert!(CloudBackend::has_provider("anthropic"));
        assert!(CloudBackend::has_provider("gemini"));
        assert!(CloudBackend::has_provider("ollama"));
        assert!(!CloudBackend::has_provider("nonexistent"));
    }

    #[test]
    fn test_local_providers_always_available() {
        let backend = CloudBackend::with_keys(HashMap::new());
        // Even with no API keys, local providers (ollama, lmstudio, etc.) make it available
        assert!(backend.is_available());
        let available = backend.available_providers();
        assert!(available.iter().any(|p| p.prefix == "ollama"));
        assert!(available.iter().any(|p| p.prefix == "lmstudio"));
        assert!(available.iter().any(|p| p.prefix == "vllm"));
    }

    #[test]
    fn test_resolve_provider_no_slash() {
        let backend = CloudBackend::with_keys(HashMap::new());
        let result = backend.resolve_provider("gpt-4o");
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_provider_unknown_prefix() {
        let backend = CloudBackend::with_keys(HashMap::new());
        let result = backend.resolve_provider("fakeprovider/model");
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_provider_missing_key() {
        let backend = CloudBackend::with_keys(HashMap::new());
        let result = backend.resolve_provider("openai/gpt-4o");
        assert!(result.is_err()); // No API key set
    }

    #[test]
    fn test_resolve_provider_with_key() {
        let mut keys = HashMap::new();
        keys.insert("openai".to_string(), "sk-test".to_string());
        let backend = CloudBackend::with_keys(keys);
        let (provider, api_key, model_id) = backend.resolve_provider("openai/gpt-4o").unwrap();
        assert_eq!(provider.prefix, "openai");
        assert_eq!(api_key, Some("sk-test"));
        assert_eq!(model_id, "gpt-4o");
    }

    #[test]
    fn test_resolve_local_provider_no_key_needed() {
        let backend = CloudBackend::with_keys(HashMap::new());
        let (provider, api_key, model_id) = backend.resolve_provider("ollama/llama3").unwrap();
        assert_eq!(provider.prefix, "ollama");
        assert!(api_key.is_none());
        assert_eq!(model_id, "llama3");
    }

    #[test]
    fn test_all_protocols_registered() {
        let backend = CloudBackend::new();
        assert!(backend
            .protocol_handlers
            .contains_key(&ApiProtocol::OpenAiCompat));
        assert!(backend
            .protocol_handlers
            .contains_key(&ApiProtocol::Anthropic));
        assert!(backend.protocol_handlers.contains_key(&ApiProtocol::Gemini));
        assert!(backend.protocol_handlers.contains_key(&ApiProtocol::Cohere));
    }
}
