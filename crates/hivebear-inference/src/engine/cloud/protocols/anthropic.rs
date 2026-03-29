//! Anthropic Messages API protocol implementation.
//!
//! Maps HiveBear's unified types to Anthropic's `/v1/messages` endpoint format,
//! handling system message extraction, multimodal content blocks, tool use, and
//! event-based SSE streaming.

use base64::Engine as _;

use super::super::protocol::CloudProtocol;
use super::super::registry::ProviderDef;
use crate::error::{InferenceError, Result};
use crate::types::*;

/// Protocol handler for the Anthropic Messages API.
#[derive(Default)]
pub struct AnthropicProtocol;

impl AnthropicProtocol {
    pub fn new() -> Self {
        Self
    }

    /// Convert a `ContentPart` into the Anthropic JSON content block format.
    fn content_part_to_json(part: &ContentPart) -> serde_json::Value {
        match part {
            ContentPart::Text { text } => {
                serde_json::json!({ "type": "text", "text": text })
            }
            ContentPart::Image { data, media_type } => {
                let b64 = base64::engine::general_purpose::STANDARD.encode(data);
                serde_json::json!({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": b64,
                    }
                })
            }
            ContentPart::ImageUrl { url } => {
                serde_json::json!({
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": url,
                    }
                })
            }
        }
    }

    /// Convert a `ToolChoice` to the Anthropic API `tool_choice` value.
    fn tool_choice_to_json(tc: &ToolChoice) -> serde_json::Value {
        match tc {
            ToolChoice::Auto => serde_json::json!({ "type": "auto" }),
            ToolChoice::Required => serde_json::json!({ "type": "any" }),
            ToolChoice::None => serde_json::json!({ "type": "none" }),
            ToolChoice::Specific(name) => {
                serde_json::json!({ "type": "tool", "name": name })
            }
        }
    }
}

impl CloudProtocol for AnthropicProtocol {
    fn build_request_body(
        &self,
        _provider: &ProviderDef,
        model_id: &str,
        req: &GenerateRequest,
        stream: bool,
    ) -> serde_json::Value {
        // Extract system message (Anthropic takes it as a top-level field).
        let system_text: Option<String> = req.messages.iter().find_map(|m| {
            if let ChatMessage::System(s) = m {
                Some(s.clone())
            } else {
                None
            }
        });

        // Build non-system messages.
        let messages: Vec<serde_json::Value> = req
            .messages
            .iter()
            .filter_map(|m| match m {
                ChatMessage::System(_) => None,
                ChatMessage::User(parts) => {
                    let content: Vec<serde_json::Value> =
                        parts.iter().map(Self::content_part_to_json).collect();
                    Some(serde_json::json!({
                        "role": "user",
                        "content": content,
                    }))
                }
                ChatMessage::Assistant(text) => Some(serde_json::json!({
                    "role": "assistant",
                    "content": text,
                })),
                ChatMessage::ToolResult {
                    tool_call_id,
                    content,
                } => Some(serde_json::json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": content,
                    }],
                })),
            })
            .collect();

        let mut body = serde_json::json!({
            "model": model_id,
            "messages": messages,
            "max_tokens": req.max_tokens,
        });

        // System message as top-level field.
        if let Some(sys) = system_text {
            body["system"] = serde_json::json!(sys);
        }

        // Sampling parameters.
        let sampling = &req.sampling;
        if (sampling.temperature - 0.7).abs() > f32::EPSILON {
            body["temperature"] = serde_json::json!(sampling.temperature);
        }
        if (sampling.top_p - 0.9).abs() > f32::EPSILON {
            body["top_p"] = serde_json::json!(sampling.top_p);
        }

        // Stop sequences.
        if !req.stop_sequences.is_empty() {
            body["stop_sequences"] = serde_json::json!(req.stop_sequences);
        }

        // Tools.
        if !req.tools.is_empty() {
            let tools: Vec<serde_json::Value> = req
                .tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.input_schema,
                    })
                })
                .collect();
            body["tools"] = serde_json::json!(tools);

            // Only include tool_choice when tools are present.
            body["tool_choice"] = Self::tool_choice_to_json(&req.tool_choice);
        }

        // Streaming flag.
        if stream {
            body["stream"] = serde_json::json!(true);
        }

        body
    }

    fn parse_response(&self, body: &serde_json::Value) -> Result<GenerateResponse> {
        let content = body
            .get("content")
            .and_then(|c| c.as_array())
            .ok_or_else(|| {
                InferenceError::GenerationError(format!(
                    "Anthropic response missing 'content' array: {}",
                    body
                ))
            })?;

        if content.is_empty() {
            return Err(InferenceError::GenerationError(
                "Anthropic response has empty content array".into(),
            ));
        }

        let mut text_blocks: Vec<String> = Vec::new();
        let mut tool_calls: Vec<ToolCallResponse> = Vec::new();

        for block in content {
            match block.get("type").and_then(|t| t.as_str()) {
                Some("text") => {
                    if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                        text_blocks.push(text.to_string());
                    }
                }
                Some("tool_use") => {
                    let name = block
                        .get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("")
                        .to_string();
                    let id = block
                        .get("id")
                        .and_then(|i| i.as_str())
                        .unwrap_or("")
                        .to_string();
                    let input = block
                        .get("input")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null);
                    tool_calls.push(ToolCallResponse {
                        tool_name: name,
                        arguments: input,
                        call_id: id,
                    });
                }
                _ => {} // Ignore unknown block types.
            }
        }

        // Single text block, no tool calls -> simple text response.
        if tool_calls.is_empty() {
            return Ok(GenerateResponse::Text(text_blocks.join("")));
        }

        // Single tool call, no text -> simple tool call response.
        if text_blocks.is_empty() && tool_calls.len() == 1 {
            return Ok(GenerateResponse::ToolCall(tool_calls.remove(0)));
        }

        // Mixed content: build a vector of content blocks.
        let mut blocks: Vec<ContentBlock> = Vec::new();
        for block in content {
            match block.get("type").and_then(|t| t.as_str()) {
                Some("text") => {
                    if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                        blocks.push(ContentBlock::Text(text.to_string()));
                    }
                }
                Some("tool_use") => {
                    let name = block
                        .get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or("")
                        .to_string();
                    let id = block
                        .get("id")
                        .and_then(|i| i.as_str())
                        .unwrap_or("")
                        .to_string();
                    let input = block
                        .get("input")
                        .cloned()
                        .unwrap_or(serde_json::Value::Null);
                    blocks.push(ContentBlock::ToolCall(ToolCallResponse {
                        tool_name: name,
                        arguments: input,
                        call_id: id,
                    }));
                }
                _ => {}
            }
        }
        Ok(GenerateResponse::Mixed(blocks))
    }

    fn parse_stream_chunk(&self, data: &str) -> Result<Option<Token>> {
        let parsed: serde_json::Value = serde_json::from_str(data).map_err(|e| {
            InferenceError::GenerationError(format!("Failed to parse Anthropic stream chunk: {e}"))
        })?;

        let event_type = parsed.get("type").and_then(|t| t.as_str()).unwrap_or("");

        match event_type {
            "content_block_delta" => {
                let delta = parsed.get("delta").ok_or_else(|| {
                    InferenceError::GenerationError(
                        "content_block_delta missing 'delta' field".into(),
                    )
                })?;

                let delta_type = delta.get("type").and_then(|t| t.as_str()).unwrap_or("");

                match delta_type {
                    "text_delta" => {
                        let text = delta
                            .get("text")
                            .and_then(|t| t.as_str())
                            .unwrap_or("")
                            .to_string();
                        Ok(Some(Token {
                            text,
                            id: 0,
                            logprob: None,
                            is_special: false,
                        }))
                    }
                    // input_json_delta for tool use streaming -- skip for now.
                    _ => Ok(None),
                }
            }
            // Non-content events: skip them.
            "message_start"
            | "content_block_start"
            | "content_block_stop"
            | "message_delta"
            | "message_stop"
            | "ping" => Ok(None),
            _ => {
                // Unknown event type -- silently skip rather than error,
                // since Anthropic may add new event types over time.
                Ok(None)
            }
        }
    }

    fn build_url(&self, provider: &ProviderDef, _model_id: &str) -> String {
        format!("{}{}", provider.base_url, provider.chat_endpoint)
    }

    fn done_sentinel(&self) -> Option<&str> {
        // Anthropic does not use a `[DONE]` sentinel. Instead, the stream ends
        // with a `message_stop` event, which we handle in `parse_stream_chunk`
        // by returning `None`.
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a minimal Anthropic provider definition for tests.
    fn test_provider() -> ProviderDef {
        ProviderDef {
            prefix: "anthropic",
            display_name: "Anthropic",
            base_url: "https://api.anthropic.com",
            protocol: super::super::super::registry::ApiProtocol::Anthropic,
            auth_style: super::super::super::registry::AuthStyle::CustomHeader("x-api-key"),
            env_var: "ANTHROPIC_API_KEY",
            extra_headers: &[("anthropic-version", "2023-06-01")],
            chat_endpoint: "/v1/messages",
            requires_api_key: true,
            capabilities: super::super::super::registry::ProviderCapabilities {
                supports_streaming: true,
                supports_tool_calling: true,
                supports_vision: true,
                supports_structured_output: true,
                max_context_length: Some(200_000),
            },
        }
    }

    /// Helper: build a basic generate request.
    fn basic_request() -> GenerateRequest {
        GenerateRequest {
            messages: vec![
                ChatMessage::System("You are helpful.".into()),
                ChatMessage::user_text("Hello"),
            ],
            max_tokens: 1024,
            sampling: SamplingParams::default(),
            ..GenerateRequest::default()
        }
    }

    // ── build_request_body ──────────────────────────────────────────

    #[test]
    fn test_build_request_body_basic() {
        let proto = AnthropicProtocol::new();
        let provider = test_provider();
        let req = basic_request();

        let body = proto.build_request_body(&provider, "claude-sonnet-4-20250514", &req, false);

        assert_eq!(body["model"], "claude-sonnet-4-20250514");
        assert_eq!(body["max_tokens"], 1024);
        assert_eq!(body["system"], "You are helpful.");

        // System message must NOT appear in the messages array.
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");

        // stream key should be absent when not streaming.
        assert!(body.get("stream").is_none());
    }

    #[test]
    fn test_build_request_body_streaming() {
        let proto = AnthropicProtocol::new();
        let provider = test_provider();
        let req = basic_request();

        let body = proto.build_request_body(&provider, "claude-sonnet-4-20250514", &req, true);
        assert_eq!(body["stream"], true);
    }

    #[test]
    fn test_build_request_body_with_tools() {
        let proto = AnthropicProtocol::new();
        let provider = test_provider();
        let mut req = basic_request();
        req.tools = vec![ToolDefinition {
            name: "get_weather".into(),
            description: "Get weather for a location".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "location": { "type": "string" }
                },
                "required": ["location"]
            }),
        }];
        req.tool_choice = ToolChoice::Auto;

        let body = proto.build_request_body(&provider, "claude-sonnet-4-20250514", &req, false);

        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "get_weather");
        assert_eq!(body["tool_choice"]["type"], "auto");
    }

    #[test]
    fn test_build_request_body_with_images() {
        let proto = AnthropicProtocol::new();
        let provider = test_provider();

        let req = GenerateRequest {
            messages: vec![ChatMessage::User(vec![
                ContentPart::Text {
                    text: "What is in this image?".into(),
                },
                ContentPart::Image {
                    data: vec![0xFF, 0xD8, 0xFF],
                    media_type: "image/jpeg".into(),
                },
            ])],
            max_tokens: 512,
            ..GenerateRequest::default()
        };

        let body = proto.build_request_body(&provider, "claude-sonnet-4-20250514", &req, false);

        let messages = body["messages"].as_array().unwrap();
        let content = messages[0]["content"].as_array().unwrap();
        assert_eq!(content.len(), 2);
        assert_eq!(content[0]["type"], "text");
        assert_eq!(content[1]["type"], "image");
        assert_eq!(content[1]["source"]["type"], "base64");
        assert_eq!(content[1]["source"]["media_type"], "image/jpeg");
    }

    #[test]
    fn test_build_request_body_tool_result() {
        let proto = AnthropicProtocol::new();
        let provider = test_provider();

        let req = GenerateRequest {
            messages: vec![
                ChatMessage::user_text("What is the weather?"),
                ChatMessage::Assistant("Let me check.".into()),
                ChatMessage::ToolResult {
                    tool_call_id: "toolu_123".into(),
                    content: "72F and sunny".into(),
                },
            ],
            max_tokens: 512,
            ..GenerateRequest::default()
        };

        let body = proto.build_request_body(&provider, "claude-sonnet-4-20250514", &req, false);

        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[2]["role"], "user");
        let tool_result_content = messages[2]["content"].as_array().unwrap();
        assert_eq!(tool_result_content[0]["type"], "tool_result");
        assert_eq!(tool_result_content[0]["tool_use_id"], "toolu_123");
    }

    #[test]
    fn test_build_request_body_stop_sequences() {
        let proto = AnthropicProtocol::new();
        let provider = test_provider();

        let mut req = basic_request();
        req.stop_sequences = vec!["Human:".into(), "###".into()];

        let body = proto.build_request_body(&provider, "claude-sonnet-4-20250514", &req, false);

        let stops = body["stop_sequences"].as_array().unwrap();
        assert_eq!(stops.len(), 2);
        assert_eq!(stops[0], "Human:");
        assert_eq!(stops[1], "###");
    }

    // ── parse_response ──────────────────────────────────────────────

    #[test]
    fn test_parse_response_text() {
        let proto = AnthropicProtocol::new();
        let body = serde_json::json!({
            "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Hello! How can I help you today?"
                }
            ],
            "model": "claude-sonnet-4-20250514",
            "stop_reason": "end_turn",
            "usage": { "input_tokens": 12, "output_tokens": 10 }
        });

        let resp = proto.parse_response(&body).unwrap();
        match resp {
            GenerateResponse::Text(text) => {
                assert_eq!(text, "Hello! How can I help you today?");
            }
            _ => panic!("Expected Text response"),
        }
    }

    #[test]
    fn test_parse_response_tool_use() {
        let proto = AnthropicProtocol::new();
        let body = serde_json::json!({
            "id": "msg_abc",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01A09q90qw90lq917835lq9",
                    "name": "get_weather",
                    "input": { "location": "San Francisco, CA" }
                }
            ],
            "stop_reason": "tool_use",
            "usage": { "input_tokens": 50, "output_tokens": 30 }
        });

        let resp = proto.parse_response(&body).unwrap();
        match resp {
            GenerateResponse::ToolCall(tc) => {
                assert_eq!(tc.tool_name, "get_weather");
                assert_eq!(tc.call_id, "toolu_01A09q90qw90lq917835lq9");
                assert_eq!(tc.arguments["location"], "San Francisco, CA");
            }
            _ => panic!("Expected ToolCall response"),
        }
    }

    #[test]
    fn test_parse_response_mixed() {
        let proto = AnthropicProtocol::new();
        let body = serde_json::json!({
            "id": "msg_mixed",
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "I'll check the weather for you."
                },
                {
                    "type": "tool_use",
                    "id": "toolu_xyz",
                    "name": "get_weather",
                    "input": { "location": "NYC" }
                }
            ],
            "stop_reason": "tool_use",
            "usage": { "input_tokens": 50, "output_tokens": 40 }
        });

        let resp = proto.parse_response(&body).unwrap();
        match resp {
            GenerateResponse::Mixed(blocks) => {
                assert_eq!(blocks.len(), 2);
                match &blocks[0] {
                    ContentBlock::Text(t) => {
                        assert_eq!(t, "I'll check the weather for you.");
                    }
                    _ => panic!("Expected text block at index 0"),
                }
                match &blocks[1] {
                    ContentBlock::ToolCall(tc) => {
                        assert_eq!(tc.tool_name, "get_weather");
                        assert_eq!(tc.call_id, "toolu_xyz");
                    }
                    _ => panic!("Expected tool call block at index 1"),
                }
            }
            _ => panic!("Expected Mixed response"),
        }
    }

    #[test]
    fn test_parse_response_empty_content() {
        let proto = AnthropicProtocol::new();
        let body = serde_json::json!({
            "id": "msg_empty",
            "type": "message",
            "content": []
        });

        let result = proto.parse_response(&body);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_response_missing_content() {
        let proto = AnthropicProtocol::new();
        let body = serde_json::json!({
            "id": "msg_bad",
            "type": "error",
            "error": { "type": "invalid_request_error", "message": "bad request" }
        });

        let result = proto.parse_response(&body);
        assert!(result.is_err());
    }

    // ── parse_stream_chunk ──────────────────────────────────────────

    #[test]
    fn test_parse_stream_chunk_text_delta() {
        let proto = AnthropicProtocol::new();
        let data = r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}"#;

        let result = proto.parse_stream_chunk(data).unwrap();
        let token = result.unwrap();
        assert_eq!(token.text, "Hello");
        assert!(!token.is_special);
    }

    #[test]
    fn test_parse_stream_chunk_message_start() {
        let proto = AnthropicProtocol::new();
        let data = r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-sonnet-4-20250514","stop_reason":null,"usage":{"input_tokens":25,"output_tokens":1}}}"#;

        let result = proto.parse_stream_chunk(data).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_stream_chunk_content_block_start() {
        let proto = AnthropicProtocol::new();
        let data =
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#;

        let result = proto.parse_stream_chunk(data).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_stream_chunk_message_delta() {
        let proto = AnthropicProtocol::new();
        let data = r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":15}}"#;

        let result = proto.parse_stream_chunk(data).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_stream_chunk_message_stop() {
        let proto = AnthropicProtocol::new();
        let data = r#"{"type":"message_stop"}"#;

        let result = proto.parse_stream_chunk(data).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_stream_chunk_ping() {
        let proto = AnthropicProtocol::new();
        let data = r#"{"type":"ping"}"#;

        let result = proto.parse_stream_chunk(data).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_stream_chunk_input_json_delta() {
        let proto = AnthropicProtocol::new();
        let data = r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"location\":\"SF\"}"}}"#;

        let result = proto.parse_stream_chunk(data).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_stream_chunk_invalid_json() {
        let proto = AnthropicProtocol::new();
        let result = proto.parse_stream_chunk("not json at all");
        assert!(result.is_err());
    }

    // ── done_sentinel / build_url ───────────────────────────────────

    #[test]
    fn test_done_sentinel_is_none() {
        let proto = AnthropicProtocol::new();
        assert!(proto.done_sentinel().is_none());
    }

    #[test]
    fn test_build_url() {
        let proto = AnthropicProtocol::new();
        let provider = test_provider();
        let url = proto.build_url(&provider, "claude-sonnet-4-20250514");
        assert_eq!(url, "https://api.anthropic.com/v1/messages");
    }
}
