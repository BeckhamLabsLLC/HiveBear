//! Google Gemini protocol handler implementing `CloudProtocol`.
//!
//! Maps `GenerateRequest` to Gemini's `generateContent` / `streamGenerateContent`
//! API format and parses responses back into the common types.

use super::super::protocol::CloudProtocol;
use super::super::registry::ProviderDef;
use crate::error::{InferenceError, Result};
use crate::types::*;

/// Protocol handler for the Google Gemini `generateContent` API.
#[derive(Default)]
pub struct GeminiProtocol;

impl GeminiProtocol {
    pub fn new() -> Self {
        Self
    }

    /// Convert a `ChatMessage` to Gemini's `{"role": ..., "parts": [...]}` format.
    ///
    /// System messages are handled separately (via `systemInstruction`), so this
    /// only processes user and assistant (model) messages.
    fn convert_message(msg: &ChatMessage) -> Option<serde_json::Value> {
        match msg {
            ChatMessage::System(_) => None, // handled separately
            ChatMessage::User(parts) => {
                let gemini_parts: Vec<serde_json::Value> = parts
                    .iter()
                    .map(|p| match p {
                        ContentPart::Text { text } => {
                            serde_json::json!({ "text": text })
                        }
                        ContentPart::Image { data, media_type } => {
                            use base64::Engine;
                            let b64 = base64::engine::general_purpose::STANDARD.encode(data);
                            serde_json::json!({
                                "inlineData": {
                                    "mimeType": media_type,
                                    "data": b64
                                }
                            })
                        }
                        ContentPart::ImageUrl { url } => {
                            // Gemini supports fileData for URLs, but inline text
                            // fallback for simplicity; real impl would use fileData.
                            serde_json::json!({ "text": format!("[image: {url}]") })
                        }
                    })
                    .collect();
                Some(serde_json::json!({
                    "role": "user",
                    "parts": gemini_parts
                }))
            }
            ChatMessage::Assistant { content, tool_calls } => {
                let mut parts = Vec::new();
                if let Some(text) = content {
                    if !text.is_empty() {
                        parts.push(serde_json::json!({ "text": text }));
                    }
                }
                for tc in tool_calls {
                    parts.push(serde_json::json!({
                        "functionCall": {
                            "name": tc.tool_name,
                            "args": tc.arguments,
                        }
                    }));
                }
                if parts.is_empty() {
                    parts.push(serde_json::json!({ "text": "" }));
                }
                Some(serde_json::json!({
                    "role": "model",
                    "parts": parts,
                }))
            }
            ChatMessage::ToolResult { content, .. } => Some(serde_json::json!({
                "role": "user",
                "parts": [{ "text": content }]
            })),
        }
    }

    /// Extract the system instruction text from the messages list.
    fn extract_system_instruction(messages: &[ChatMessage]) -> Option<String> {
        messages.iter().find_map(|m| match m {
            ChatMessage::System(text) => Some(text.clone()),
            _ => None,
        })
    }
}

impl CloudProtocol for GeminiProtocol {
    fn build_request_body(
        &self,
        _provider: &ProviderDef,
        _model_id: &str,
        req: &GenerateRequest,
        _stream: bool,
    ) -> serde_json::Value {
        // Build contents array (skip system messages — they go in systemInstruction)
        let contents: Vec<serde_json::Value> = req
            .messages
            .iter()
            .filter_map(Self::convert_message)
            .collect();

        // Build generationConfig
        let mut gen_config = serde_json::json!({
            "temperature": req.sampling.temperature,
            "maxOutputTokens": req.max_tokens,
            "topP": req.sampling.top_p,
        });

        if !req.stop_sequences.is_empty() {
            gen_config["stopSequences"] = serde_json::json!(req.stop_sequences);
        }

        if req.sampling.top_k > 0 {
            gen_config["topK"] = serde_json::json!(req.sampling.top_k);
        }

        let mut body = serde_json::json!({
            "contents": contents,
            "generationConfig": gen_config,
        });

        // System instruction as a top-level field
        if let Some(system_text) = Self::extract_system_instruction(&req.messages) {
            body["systemInstruction"] = serde_json::json!({
                "parts": [{ "text": system_text }]
            });
        }

        body
    }

    fn parse_response(&self, body: &serde_json::Value) -> Result<GenerateResponse> {
        // Gemini response: {"candidates": [{"content": {"parts": [{"text": "..."}]}}]}
        let text = body
            .get("candidates")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("content"))
            .and_then(|c| c.get("parts"))
            .and_then(|p| p.get(0))
            .and_then(|p| p.get("text"))
            .and_then(|t| t.as_str())
            .ok_or_else(|| {
                InferenceError::GenerationError(format!(
                    "Failed to parse Gemini response: missing candidates[0].content.parts[0].text in {body}"
                ))
            })?;

        Ok(GenerateResponse::Text(text.to_string()))
    }

    fn parse_stream_chunk(&self, data: &str) -> Result<Option<Token>> {
        // Gemini streaming returns JSON objects (when using alt=sse, each data: line
        // is a full candidate JSON object).
        // Extract text from candidates[0].content.parts[0].text
        let json: serde_json::Value = serde_json::from_str(data).map_err(|e| {
            InferenceError::GenerationError(format!("Gemini stream JSON parse error: {e}"))
        })?;

        let text = json
            .get("candidates")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("content"))
            .and_then(|c| c.get("parts"))
            .and_then(|p| p.get(0))
            .and_then(|p| p.get("text"))
            .and_then(|t| t.as_str());

        match text {
            Some(t) if !t.is_empty() => Ok(Some(Token {
                text: t.to_string(),
                id: 0,
                logprob: None,
                is_special: false,
            })),
            _ => Ok(None),
        }
    }

    fn build_url(&self, provider: &ProviderDef, model_id: &str) -> String {
        // Gemini URLs embed the model ID in the path.
        // Non-streaming: {base_url}/v1beta/models/{model_id}:generateContent
        // The streaming variant is handled by the caller appending ?alt=sse
        // and changing the endpoint to :streamGenerateContent.
        //
        // For the non-streaming case the caller uses this URL directly.
        // For streaming, the caller should call build_stream_url or adjust.
        format!(
            "{}/v1beta/models/{}:generateContent",
            provider.base_url, model_id
        )
    }

    fn build_headers(
        &self,
        provider: &ProviderDef,
        _api_key: Option<&str>,
    ) -> Vec<(String, String)> {
        // Gemini uses API key in URL query param, NOT in headers.
        let mut headers = Vec::new();

        // Extra provider headers (if any)
        for (name, value) in provider.extra_headers {
            headers.push((name.to_string(), value.to_string()));
        }

        headers.push(("Content-Type".into(), "application/json".into()));
        headers
    }

    fn done_sentinel(&self) -> Option<&str> {
        // Gemini does not use a [DONE] sentinel.
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn protocol() -> GeminiProtocol {
        GeminiProtocol::new()
    }

    #[test]
    fn test_parse_response_basic() {
        let body = serde_json::json!({
            "candidates": [{
                "content": {
                    "parts": [{ "text": "Hello from Gemini!" }],
                    "role": "model"
                },
                "finishReason": "STOP",
                "index": 0
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15
            }
        });

        let resp = protocol().parse_response(&body).unwrap();
        match resp {
            GenerateResponse::Text(t) => assert_eq!(t, "Hello from Gemini!"),
            other => panic!("Expected Text response, got: {other:?}"),
        }
    }

    #[test]
    fn test_parse_response_missing_candidates() {
        let body = serde_json::json!({ "error": { "message": "bad request" } });
        let result = protocol().parse_response(&body);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_stream_chunk_text() {
        let chunk =
            r#"{"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"},"index":0}]}"#;
        let token = protocol().parse_stream_chunk(chunk).unwrap();
        assert!(token.is_some());
        assert_eq!(token.unwrap().text, "Hello");
    }

    #[test]
    fn test_parse_stream_chunk_empty_text() {
        let chunk =
            r#"{"candidates":[{"content":{"parts":[{"text":""}],"role":"model"},"index":0}]}"#;
        let token = protocol().parse_stream_chunk(chunk).unwrap();
        assert!(token.is_none());
    }

    #[test]
    fn test_parse_stream_chunk_no_text_field() {
        // A chunk that has candidates but no text part (e.g. safety metadata only)
        let chunk = r#"{"candidates":[{"content":{"parts":[]},"index":0,"finishReason":"STOP"}]}"#;
        let token = protocol().parse_stream_chunk(chunk).unwrap();
        assert!(token.is_none());
    }

    #[test]
    fn test_parse_stream_chunk_invalid_json() {
        let result = protocol().parse_stream_chunk("not json at all");
        assert!(result.is_err());
    }

    #[test]
    fn test_build_url_non_streaming() {
        let provider = ProviderDef {
            prefix: "gemini",
            display_name: "Google Gemini",
            base_url: "https://generativelanguage.googleapis.com",
            protocol: super::super::super::registry::ApiProtocol::Gemini,
            auth_style: super::super::super::registry::AuthStyle::ApiKeyInUrl,
            env_var: "GEMINI_API_KEY",
            extra_headers: &[],
            chat_endpoint: "/v1beta/models/{model}:generateContent",
            requires_api_key: true,
            capabilities: super::super::super::registry::ProviderCapabilities::default(),
        };

        let url = protocol().build_url(&provider, "gemini-pro");
        assert_eq!(
            url,
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        );
    }

    #[test]
    fn test_build_headers_no_auth() {
        let provider = ProviderDef {
            prefix: "gemini",
            display_name: "Google Gemini",
            base_url: "https://generativelanguage.googleapis.com",
            protocol: super::super::super::registry::ApiProtocol::Gemini,
            auth_style: super::super::super::registry::AuthStyle::ApiKeyInUrl,
            env_var: "GEMINI_API_KEY",
            extra_headers: &[],
            chat_endpoint: "/v1beta/models/{model}:generateContent",
            requires_api_key: true,
            capabilities: super::super::super::registry::ProviderCapabilities::default(),
        };

        let headers = protocol().build_headers(&provider, Some("test-key"));
        // Should NOT contain an Authorization header
        assert!(
            !headers.iter().any(|(k, _)| k == "Authorization"),
            "Gemini should not put API key in headers"
        );
        // Should have Content-Type
        assert!(headers
            .iter()
            .any(|(k, v)| k == "Content-Type" && v == "application/json"));
    }

    #[test]
    fn test_done_sentinel_is_none() {
        assert!(protocol().done_sentinel().is_none());
    }

    #[test]
    fn test_build_request_body_with_system() {
        let provider = ProviderDef {
            prefix: "gemini",
            display_name: "Google Gemini",
            base_url: "https://generativelanguage.googleapis.com",
            protocol: super::super::super::registry::ApiProtocol::Gemini,
            auth_style: super::super::super::registry::AuthStyle::ApiKeyInUrl,
            env_var: "GEMINI_API_KEY",
            extra_headers: &[],
            chat_endpoint: "/v1beta/models/{model}:generateContent",
            requires_api_key: true,
            capabilities: super::super::super::registry::ProviderCapabilities::default(),
        };

        let req = GenerateRequest {
            messages: vec![
                ChatMessage::System("You are helpful.".into()),
                ChatMessage::user_text("Hello"),
            ],
            max_tokens: 100,
            ..Default::default()
        };

        let body = protocol().build_request_body(&provider, "gemini-pro", &req, false);

        // System instruction should be a top-level field
        assert!(body.get("systemInstruction").is_some());
        let sys_text = body["systemInstruction"]["parts"][0]["text"]
            .as_str()
            .unwrap();
        assert_eq!(sys_text, "You are helpful.");

        // Contents should only have the user message (no system)
        let contents = body["contents"].as_array().unwrap();
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0]["role"], "user");
    }
}
