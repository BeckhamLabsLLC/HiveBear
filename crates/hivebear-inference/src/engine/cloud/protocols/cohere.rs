//! Cohere v2 chat protocol handler implementing `CloudProtocol`.
//!
//! Maps `GenerateRequest` to Cohere's `/v2/chat` API format and parses
//! responses back into the common types.

use super::super::protocol::CloudProtocol;
use super::super::registry::ProviderDef;
use crate::error::{InferenceError, Result};
use crate::types::*;

/// Protocol handler for the Cohere v2 `/v2/chat` API.
#[derive(Default)]
pub struct CohereProtocol;

impl CohereProtocol {
    pub fn new() -> Self {
        Self
    }

    /// Convert a `ChatMessage` to Cohere's `{"role": ..., "content": ...}` format.
    fn convert_message(msg: &ChatMessage) -> serde_json::Value {
        match msg {
            ChatMessage::System(text) => serde_json::json!({
                "role": "system",
                "content": text
            }),
            ChatMessage::User(parts) => {
                // Cohere v2 expects plain text content
                let text: String = parts
                    .iter()
                    .filter_map(|p| match p {
                        ContentPart::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                serde_json::json!({
                    "role": "user",
                    "content": text
                })
            }
            ChatMessage::Assistant { content, tool_calls } => {
                let mut msg = serde_json::json!({
                    "role": "assistant",
                    "content": content.as_deref().unwrap_or(""),
                });
                if !tool_calls.is_empty() {
                    msg["tool_calls"] = serde_json::json!(tool_calls.iter().map(|tc| serde_json::json!({
                        "id": tc.call_id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": tc.arguments.to_string(),
                        }
                    })).collect::<Vec<_>>());
                }
                msg
            }
            ChatMessage::ToolResult {
                content,
                tool_call_id,
            } => serde_json::json!({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": content
            }),
        }
    }
}

impl CloudProtocol for CohereProtocol {
    fn build_request_body(
        &self,
        _provider: &ProviderDef,
        model_id: &str,
        req: &GenerateRequest,
        stream: bool,
    ) -> serde_json::Value {
        let messages: Vec<serde_json::Value> =
            req.messages.iter().map(Self::convert_message).collect();

        let mut body = serde_json::json!({
            "model": model_id,
            "messages": messages,
            "max_tokens": req.max_tokens,
            "temperature": req.sampling.temperature,
            "stream": stream,
        });

        if req.sampling.top_p < 1.0 {
            body["p"] = serde_json::json!(req.sampling.top_p);
        }

        if req.sampling.top_k > 0 {
            body["k"] = serde_json::json!(req.sampling.top_k);
        }

        if req.sampling.frequency_penalty != 0.0 {
            body["frequency_penalty"] = serde_json::json!(req.sampling.frequency_penalty);
        }

        if req.sampling.presence_penalty != 0.0 {
            body["presence_penalty"] = serde_json::json!(req.sampling.presence_penalty);
        }

        if !req.stop_sequences.is_empty() {
            body["stop_sequences"] = serde_json::json!(req.stop_sequences);
        }

        body
    }

    fn parse_response(&self, body: &serde_json::Value) -> Result<GenerateResponse> {
        // Cohere v2 response format:
        //   {"message": {"content": [{"type": "text", "text": "..."}]}, ...}
        // or in some cases a flat {"text": "..."} field.

        // Try message.content[0].text first (v2 format)
        if let Some(text) = body
            .get("message")
            .and_then(|m| m.get("content"))
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("text"))
            .and_then(|t| t.as_str())
        {
            return Ok(GenerateResponse::Text(text.to_string()));
        }

        // Fallback: top-level "text" field
        if let Some(text) = body.get("text").and_then(|t| t.as_str()) {
            return Ok(GenerateResponse::Text(text.to_string()));
        }

        Err(InferenceError::GenerationError(format!(
            "Failed to parse Cohere response: expected message.content[0].text or text field in {body}"
        )))
    }

    fn parse_stream_chunk(&self, data: &str) -> Result<Option<Token>> {
        // Cohere streaming sends NDJSON (one JSON object per line, not SSE data: prefix).
        // Each line: {"event_type": "text-generation", "text": "..."}
        // End:       {"event_type": "stream-end", ...}

        let json: serde_json::Value = serde_json::from_str(data).map_err(|e| {
            InferenceError::GenerationError(format!("Cohere stream JSON parse error: {e}"))
        })?;

        let event_type = json.get("event_type").and_then(|e| e.as_str());

        match event_type {
            Some("stream-end") => {
                // Signal end of stream by returning None
                Ok(None)
            }
            Some("text-generation") => {
                let text = json.get("text").and_then(|t| t.as_str()).unwrap_or("");
                if text.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(Token {
                        text: text.to_string(),
                        id: 0,
                        logprob: None,
                        is_special: false,
                    }))
                }
            }
            _ => {
                // Other event types (stream-start, search-queries-generation, etc.)
                // are metadata — skip them.
                Ok(None)
            }
        }
    }

    fn done_sentinel(&self) -> Option<&str> {
        // Cohere uses event_type "stream-end" instead of a [DONE] sentinel.
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn protocol() -> CohereProtocol {
        CohereProtocol::new()
    }

    #[test]
    fn test_parse_response_v2_format() {
        let body = serde_json::json!({
            "id": "abc-123",
            "message": {
                "role": "assistant",
                "content": [
                    { "type": "text", "text": "Hello from Cohere!" }
                ]
            },
            "finish_reason": "COMPLETE",
            "usage": {
                "billed_units": { "input_tokens": 5, "output_tokens": 4 },
                "tokens": { "input_tokens": 5, "output_tokens": 4 }
            }
        });

        let resp = protocol().parse_response(&body).unwrap();
        match resp {
            GenerateResponse::Text(t) => assert_eq!(t, "Hello from Cohere!"),
            other => panic!("Expected Text response, got: {other:?}"),
        }
    }

    #[test]
    fn test_parse_response_flat_text_fallback() {
        let body = serde_json::json!({
            "text": "Fallback response text",
            "generation_id": "gen-123"
        });

        let resp = protocol().parse_response(&body).unwrap();
        match resp {
            GenerateResponse::Text(t) => assert_eq!(t, "Fallback response text"),
            other => panic!("Expected Text response, got: {other:?}"),
        }
    }

    #[test]
    fn test_parse_response_missing_fields() {
        let body = serde_json::json!({ "error": "invalid request" });
        let result = protocol().parse_response(&body);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_stream_chunk_text_generation() {
        let chunk = r#"{"event_type":"text-generation","text":"Hello"}"#;
        let token = protocol().parse_stream_chunk(chunk).unwrap();
        assert!(token.is_some());
        assert_eq!(token.unwrap().text, "Hello");
    }

    #[test]
    fn test_parse_stream_chunk_stream_end() {
        let chunk = r#"{"event_type":"stream-end","response":{"text":"full response"},"finish_reason":"COMPLETE"}"#;
        let token = protocol().parse_stream_chunk(chunk).unwrap();
        assert!(token.is_none());
    }

    #[test]
    fn test_parse_stream_chunk_stream_start() {
        let chunk = r#"{"event_type":"stream-start","generation_id":"gen-123"}"#;
        let token = protocol().parse_stream_chunk(chunk).unwrap();
        assert!(token.is_none());
    }

    #[test]
    fn test_parse_stream_chunk_empty_text() {
        let chunk = r#"{"event_type":"text-generation","text":""}"#;
        let token = protocol().parse_stream_chunk(chunk).unwrap();
        assert!(token.is_none());
    }

    #[test]
    fn test_parse_stream_chunk_invalid_json() {
        let result = protocol().parse_stream_chunk("not json");
        assert!(result.is_err());
    }

    #[test]
    fn test_done_sentinel_is_none() {
        assert!(protocol().done_sentinel().is_none());
    }

    #[test]
    fn test_build_request_body_basic() {
        let provider = ProviderDef {
            prefix: "cohere",
            display_name: "Cohere",
            base_url: "https://api.cohere.com",
            protocol: super::super::super::registry::ApiProtocol::Cohere,
            auth_style: super::super::super::registry::AuthStyle::BearerToken,
            env_var: "COHERE_API_KEY",
            extra_headers: &[],
            chat_endpoint: "/v2/chat",
            requires_api_key: true,
            capabilities: super::super::super::registry::ProviderCapabilities::default(),
        };

        let req = GenerateRequest {
            messages: vec![
                ChatMessage::System("Be concise.".into()),
                ChatMessage::user_text("What is Rust?"),
            ],
            max_tokens: 256,
            ..Default::default()
        };

        let body = protocol().build_request_body(&provider, "command-r-plus", &req, true);

        assert_eq!(body["model"], "command-r-plus");
        assert_eq!(body["stream"], true);
        assert_eq!(body["max_tokens"], 256);

        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "Be concise.");
        assert_eq!(messages[1]["role"], "user");
        assert_eq!(messages[1]["content"], "What is Rust?");
    }

    #[test]
    fn test_build_request_body_non_streaming() {
        let provider = ProviderDef {
            prefix: "cohere",
            display_name: "Cohere",
            base_url: "https://api.cohere.com",
            protocol: super::super::super::registry::ApiProtocol::Cohere,
            auth_style: super::super::super::registry::AuthStyle::BearerToken,
            env_var: "COHERE_API_KEY",
            extra_headers: &[],
            chat_endpoint: "/v2/chat",
            requires_api_key: true,
            capabilities: super::super::super::registry::ProviderCapabilities::default(),
        };

        let req = GenerateRequest {
            messages: vec![ChatMessage::user_text("Hi")],
            max_tokens: 100,
            ..Default::default()
        };

        let body = protocol().build_request_body(&provider, "command-r", &req, false);
        assert_eq!(body["stream"], false);
    }
}
