//! OpenAI-compatible protocol implementation.
//!
//! Covers ~25 of the 35 builtin providers that share the standard
//! `/v1/chat/completions` request/response shape (OpenAI, Groq, Together,
//! Fireworks, DeepSeek, Mistral, OpenRouter, Ollama, vLLM, etc.).

use serde_json::{json, Value};

use super::super::protocol::CloudProtocol;
use super::super::registry::ProviderDef;
use crate::error::{InferenceError, Result};
use crate::types::*;

// ── Helpers ──────────────────────────────────────────────────────────

/// Extract concatenated text from multimodal content parts, ignoring images.
fn extract_text_from_parts(parts: &[ContentPart]) -> String {
    parts
        .iter()
        .filter_map(|p| match p {
            ContentPart::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Returns `true` if any part is an image (inline data or URL).
fn has_images(parts: &[ContentPart]) -> bool {
    parts
        .iter()
        .any(|p| matches!(p, ContentPart::Image { .. } | ContentPart::ImageUrl { .. }))
}

/// Convert `ContentPart` slices into the OpenAI multimodal content array.
fn parts_to_content_array(parts: &[ContentPart]) -> Value {
    let items: Vec<Value> = parts
        .iter()
        .map(|p| match p {
            ContentPart::Text { text } => json!({
                "type": "text",
                "text": text,
            }),
            ContentPart::Image { data, media_type } => {
                use base64::Engine;
                let b64 = base64::engine::general_purpose::STANDARD.encode(data);
                json!({
                    "type": "image_url",
                    "image_url": {
                        "url": format!("data:{media_type};base64,{b64}"),
                    },
                })
            }
            ContentPart::ImageUrl { url } => json!({
                "type": "image_url",
                "image_url": {
                    "url": url,
                },
            }),
        })
        .collect();
    Value::Array(items)
}

/// Map `ToolChoice` to the OpenAI `tool_choice` JSON value.
fn tool_choice_to_value(choice: &ToolChoice) -> Value {
    match choice {
        ToolChoice::Auto => json!("auto"),
        ToolChoice::Required => json!("required"),
        ToolChoice::None => json!("none"),
        ToolChoice::Specific(name) => json!({
            "type": "function",
            "function": { "name": name },
        }),
    }
}

// ── Protocol Implementation ──────────────────────────────────────────

/// Protocol handler for all providers that speak the OpenAI
/// `/v1/chat/completions` format.
#[derive(Default)]
pub struct OpenAiCompatProtocol;

impl OpenAiCompatProtocol {
    pub fn new() -> Self {
        Self
    }
}

impl CloudProtocol for OpenAiCompatProtocol {
    fn build_request_body(
        &self,
        _provider: &ProviderDef,
        model_id: &str,
        req: &GenerateRequest,
        stream: bool,
    ) -> Value {
        // ── Messages ────────────────────────────────────────────────
        let messages: Vec<Value> = req
            .messages
            .iter()
            .map(|m| match m {
                ChatMessage::System(s) => json!({
                    "role": "system",
                    "content": s,
                }),
                ChatMessage::User(parts) => {
                    if has_images(parts) {
                        json!({
                            "role": "user",
                            "content": parts_to_content_array(parts),
                        })
                    } else {
                        json!({
                            "role": "user",
                            "content": extract_text_from_parts(parts),
                        })
                    }
                }
                ChatMessage::Assistant { content, tool_calls } => {
                    let mut msg = json!({
                        "role": "assistant",
                    });
                    if let Some(text) = content {
                        msg["content"] = json!(text);
                    }
                    if !tool_calls.is_empty() {
                        msg["tool_calls"] = json!(tool_calls.iter().map(|tc| json!({
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
                    tool_call_id,
                    content,
                } => json!({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": content,
                }),
            })
            .collect();

        // ── Core body ───────────────────────────────────────────────
        let mut body = json!({
            "model": model_id,
            "messages": messages,
            "max_tokens": req.max_tokens,
            "temperature": req.sampling.temperature,
            "top_p": req.sampling.top_p,
            "stream": stream,
        });

        // Frequency / presence penalty (skip defaults to keep payloads small)
        if req.sampling.frequency_penalty != 0.0 {
            body["frequency_penalty"] = json!(req.sampling.frequency_penalty);
        }
        if req.sampling.presence_penalty != 0.0 {
            body["presence_penalty"] = json!(req.sampling.presence_penalty);
        }

        // Stop sequences
        if !req.stop_sequences.is_empty() {
            body["stop"] = json!(req.stop_sequences);
        }

        // ── Tools ───────────────────────────────────────────────────
        if !req.tools.is_empty() {
            let tools: Vec<Value> = req
                .tools
                .iter()
                .map(|t| {
                    json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.input_schema,
                        },
                    })
                })
                .collect();
            body["tools"] = Value::Array(tools);
            body["tool_choice"] = tool_choice_to_value(&req.tool_choice);
        }

        // ── Structured output (response_format) ─────────────────────
        if let Some(schema) = &req.output_schema {
            body["response_format"] = json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "strict": true,
                    "schema": schema,
                },
            });
        }

        body
    }

    fn parse_response(&self, body: &Value) -> Result<GenerateResponse> {
        let choice = body.get("choices").and_then(|c| c.get(0)).ok_or_else(|| {
            InferenceError::GenerationError("OpenAI response missing choices[0]".into())
        })?;

        let message = choice.get("message").ok_or_else(|| {
            InferenceError::GenerationError("OpenAI response missing choices[0].message".into())
        })?;

        // ── Tool calls ──────────────────────────────────────────────
        if let Some(tool_calls) = message.get("tool_calls") {
            if let Some(arr) = tool_calls.as_array() {
                if !arr.is_empty() {
                    // If there is text AND tool calls, return Mixed.
                    let has_text = message
                        .get("content")
                        .and_then(|v| v.as_str())
                        .map(|s| !s.is_empty())
                        .unwrap_or(false);

                    let mut blocks: Vec<ContentBlock> = Vec::new();

                    if has_text {
                        let text = message["content"].as_str().unwrap_or("").to_string();
                        blocks.push(ContentBlock::Text(text));
                    }

                    for tc in arr {
                        let func = &tc["function"];
                        let name = func["name"].as_str().unwrap_or("").to_string();
                        let args_str = func["arguments"].as_str().unwrap_or("{}");
                        let arguments: Value =
                            serde_json::from_str(args_str).unwrap_or_else(|_| json!({}));
                        let call_id = tc["id"].as_str().unwrap_or("").to_string();

                        let tcr = ToolCallResponse {
                            tool_name: name,
                            arguments,
                            call_id,
                        };

                        if !has_text && arr.len() == 1 {
                            return Ok(GenerateResponse::ToolCall(tcr));
                        }
                        blocks.push(ContentBlock::ToolCall(tcr));
                    }

                    return Ok(GenerateResponse::Mixed(blocks));
                }
            }
        }

        // ── Plain text ──────────────────────────────────────────────
        let content = message
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        Ok(GenerateResponse::Text(content))
    }

    fn parse_stream_chunk(&self, data: &str) -> Result<Option<Token>> {
        let parsed: Value = serde_json::from_str(data).map_err(|e| {
            InferenceError::GenerationError(format!("Failed to parse SSE chunk: {e}"))
        })?;

        let delta = parsed
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("delta"));

        let delta = match delta {
            Some(d) => d,
            None => return Ok(None),
        };

        // Role-only delta (e.g. {"role":"assistant"}) — skip.
        if delta.get("content").is_none() {
            return Ok(None);
        }

        let text = delta["content"].as_str().unwrap_or("").to_string();

        // Empty content string — skip (some providers send this initially).
        if text.is_empty() {
            return Ok(None);
        }

        Ok(Some(Token {
            text,
            id: 0,
            logprob: None,
            is_special: false,
        }))
    }

    // `build_url`, `build_headers`, and `done_sentinel` use the trait defaults.
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::cloud::registry::{ApiProtocol, AuthStyle, ProviderCapabilities};

    fn test_provider() -> ProviderDef {
        ProviderDef {
            prefix: "test",
            display_name: "Test",
            base_url: "https://api.test.com/v1",
            protocol: ApiProtocol::OpenAiCompat,
            auth_style: AuthStyle::BearerToken,
            env_var: "TEST_API_KEY",
            extra_headers: &[],
            chat_endpoint: "/chat/completions",
            requires_api_key: true,
            capabilities: ProviderCapabilities {
                supports_streaming: true,
                supports_tool_calling: true,
                supports_vision: true,
                supports_structured_output: true,
                max_context_length: Some(128_000),
            },
        }
    }

    fn simple_request() -> GenerateRequest {
        GenerateRequest {
            messages: vec![
                ChatMessage::System("You are a helpful assistant.".into()),
                ChatMessage::user_text("Hello!"),
            ],
            max_tokens: 100,
            ..Default::default()
        }
    }

    // ── build_request_body ───────────────────────────────────────

    #[test]
    fn test_build_basic_request() {
        let proto = OpenAiCompatProtocol::new();
        let provider = test_provider();
        let req = simple_request();
        let body = proto.build_request_body(&provider, "gpt-4o", &req, false);

        assert_eq!(body["model"], "gpt-4o");
        assert_eq!(body["stream"], false);
        assert_eq!(body["max_tokens"], 100);

        let msgs = body["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "system");
        assert_eq!(msgs[0]["content"], "You are a helpful assistant.");
        assert_eq!(msgs[1]["role"], "user");
        assert_eq!(msgs[1]["content"], "Hello!");
    }

    #[test]
    fn test_build_streaming_request() {
        let proto = OpenAiCompatProtocol::new();
        let provider = test_provider();
        let req = simple_request();
        let body = proto.build_request_body(&provider, "gpt-4o", &req, true);
        assert_eq!(body["stream"], true);
    }

    #[test]
    fn test_build_request_with_images() {
        let proto = OpenAiCompatProtocol::new();
        let provider = test_provider();
        let req = GenerateRequest {
            messages: vec![ChatMessage::User(vec![
                ContentPart::Text {
                    text: "What is this?".into(),
                },
                ContentPart::ImageUrl {
                    url: "https://example.com/cat.jpg".into(),
                },
            ])],
            ..Default::default()
        };
        let body = proto.build_request_body(&provider, "gpt-4o", &req, false);

        let content = &body["messages"][0]["content"];
        assert!(content.is_array());
        let parts = content.as_array().unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "text");
        assert_eq!(parts[0]["text"], "What is this?");
        assert_eq!(parts[1]["type"], "image_url");
        assert_eq!(parts[1]["image_url"]["url"], "https://example.com/cat.jpg");
    }

    #[test]
    fn test_build_request_with_inline_image() {
        let proto = OpenAiCompatProtocol::new();
        let provider = test_provider();
        let req = GenerateRequest {
            messages: vec![ChatMessage::User(vec![
                ContentPart::Text {
                    text: "Describe this.".into(),
                },
                ContentPart::Image {
                    data: vec![0x89, 0x50, 0x4E, 0x47],
                    media_type: "image/png".into(),
                },
            ])],
            ..Default::default()
        };
        let body = proto.build_request_body(&provider, "gpt-4o", &req, false);

        let parts = body["messages"][0]["content"].as_array().unwrap();
        assert_eq!(parts[1]["type"], "image_url");
        let url = parts[1]["image_url"]["url"].as_str().unwrap();
        assert!(url.starts_with("data:image/png;base64,"));
    }

    #[test]
    fn test_build_request_with_tools() {
        let proto = OpenAiCompatProtocol::new();
        let provider = test_provider();
        let req = GenerateRequest {
            messages: vec![ChatMessage::user_text("What is the weather?")],
            tools: vec![ToolDefinition {
                name: "get_weather".into(),
                description: "Get the current weather.".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "location": { "type": "string" }
                    },
                    "required": ["location"],
                }),
            }],
            tool_choice: ToolChoice::Auto,
            ..Default::default()
        };
        let body = proto.build_request_body(&provider, "gpt-4o", &req, false);

        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["type"], "function");
        assert_eq!(tools[0]["function"]["name"], "get_weather");
        assert_eq!(body["tool_choice"], "auto");
    }

    #[test]
    fn test_build_request_specific_tool_choice() {
        let proto = OpenAiCompatProtocol::new();
        let provider = test_provider();
        let req = GenerateRequest {
            messages: vec![ChatMessage::user_text("Do it")],
            tools: vec![ToolDefinition {
                name: "my_func".into(),
                description: "Does something.".into(),
                input_schema: json!({}),
            }],
            tool_choice: ToolChoice::Specific("my_func".into()),
            ..Default::default()
        };
        let body = proto.build_request_body(&provider, "gpt-4o", &req, false);

        assert_eq!(body["tool_choice"]["type"], "function");
        assert_eq!(body["tool_choice"]["function"]["name"], "my_func");
    }

    #[test]
    fn test_build_request_with_structured_output() {
        let proto = OpenAiCompatProtocol::new();
        let provider = test_provider();
        let schema = json!({
            "type": "object",
            "properties": {
                "answer": { "type": "string" }
            },
            "required": ["answer"],
        });
        let req = GenerateRequest {
            messages: vec![ChatMessage::user_text("Answer")],
            output_schema: Some(schema.clone()),
            ..Default::default()
        };
        let body = proto.build_request_body(&provider, "gpt-4o", &req, false);

        assert_eq!(body["response_format"]["type"], "json_schema");
        assert_eq!(body["response_format"]["json_schema"]["schema"], schema);
        assert_eq!(body["response_format"]["json_schema"]["strict"], true);
    }

    #[test]
    fn test_build_request_with_stop_sequences() {
        let proto = OpenAiCompatProtocol::new();
        let provider = test_provider();
        let req = GenerateRequest {
            messages: vec![ChatMessage::user_text("Go")],
            stop_sequences: vec!["STOP".into(), "END".into()],
            ..Default::default()
        };
        let body = proto.build_request_body(&provider, "gpt-4o", &req, false);

        let stops = body["stop"].as_array().unwrap();
        assert_eq!(stops.len(), 2);
        assert_eq!(stops[0], "STOP");
        assert_eq!(stops[1], "END");
    }

    #[test]
    fn test_build_request_tool_result_message() {
        let proto = OpenAiCompatProtocol::new();
        let provider = test_provider();
        let req = GenerateRequest {
            messages: vec![ChatMessage::ToolResult {
                tool_call_id: "call_abc123".into(),
                content: r#"{"temperature": 72}"#.into(),
            }],
            ..Default::default()
        };
        let body = proto.build_request_body(&provider, "gpt-4o", &req, false);

        let msg = &body["messages"][0];
        assert_eq!(msg["role"], "tool");
        assert_eq!(msg["tool_call_id"], "call_abc123");
        assert_eq!(msg["content"], r#"{"temperature": 72}"#);
    }

    #[test]
    fn test_build_request_omits_zero_penalties() {
        let proto = OpenAiCompatProtocol::new();
        let provider = test_provider();
        let req = simple_request();
        let body = proto.build_request_body(&provider, "gpt-4o", &req, false);

        // Default penalties are 0.0, so they should be omitted.
        assert!(body.get("frequency_penalty").is_none());
        assert!(body.get("presence_penalty").is_none());
    }

    #[test]
    fn test_build_request_includes_nonzero_penalties() {
        let proto = OpenAiCompatProtocol::new();
        let provider = test_provider();
        let req = GenerateRequest {
            messages: vec![ChatMessage::user_text("Hi")],
            sampling: SamplingParams {
                frequency_penalty: 0.5,
                presence_penalty: 0.3,
                ..Default::default()
            },
            ..Default::default()
        };
        let body = proto.build_request_body(&provider, "gpt-4o", &req, false);

        assert!((body["frequency_penalty"].as_f64().unwrap() - 0.5).abs() < 0.01);
        assert!((body["presence_penalty"].as_f64().unwrap() - 0.3).abs() < 0.01);
    }

    // ── parse_response ──────────────────────────────────────────

    #[test]
    fn test_parse_text_response() {
        let proto = OpenAiCompatProtocol::new();
        let body = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Hello there!"
                },
                "finish_reason": "stop",
            }]
        });

        let resp = proto.parse_response(&body).unwrap();
        match resp {
            GenerateResponse::Text(t) => assert_eq!(t, "Hello there!"),
            _ => panic!("Expected Text response"),
        }
    }

    #[test]
    fn test_parse_tool_call_response() {
        let proto = OpenAiCompatProtocol::new();
        let body = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_xyz",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\":\"London\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls",
            }]
        });

        let resp = proto.parse_response(&body).unwrap();
        match resp {
            GenerateResponse::ToolCall(tc) => {
                assert_eq!(tc.tool_name, "get_weather");
                assert_eq!(tc.call_id, "call_xyz");
                assert_eq!(tc.arguments, json!({"location": "London"}));
            }
            _ => panic!("Expected ToolCall response"),
        }
    }

    #[test]
    fn test_parse_mixed_response() {
        let proto = OpenAiCompatProtocol::new();
        let body = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Let me check that for you.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{\"location\":\"NYC\"}"
                            }
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "get_time",
                                "arguments": "{\"timezone\":\"EST\"}"
                            }
                        }
                    ]
                },
                "finish_reason": "tool_calls",
            }]
        });

        let resp = proto.parse_response(&body).unwrap();
        match resp {
            GenerateResponse::Mixed(blocks) => {
                assert_eq!(blocks.len(), 3);
                match &blocks[0] {
                    ContentBlock::Text(t) => {
                        assert_eq!(t, "Let me check that for you.")
                    }
                    _ => panic!("Expected text block first"),
                }
                match &blocks[1] {
                    ContentBlock::ToolCall(tc) => {
                        assert_eq!(tc.tool_name, "get_weather");
                        assert_eq!(tc.call_id, "call_1");
                    }
                    _ => panic!("Expected tool call block"),
                }
                match &blocks[2] {
                    ContentBlock::ToolCall(tc) => {
                        assert_eq!(tc.tool_name, "get_time");
                        assert_eq!(tc.call_id, "call_2");
                    }
                    _ => panic!("Expected tool call block"),
                }
            }
            _ => panic!("Expected Mixed response"),
        }
    }

    #[test]
    fn test_parse_response_missing_choices() {
        let proto = OpenAiCompatProtocol::new();
        let body = json!({"error": "something went wrong"});
        assert!(proto.parse_response(&body).is_err());
    }

    #[test]
    fn test_parse_response_empty_content() {
        let proto = OpenAiCompatProtocol::new();
        let body = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": ""
                },
                "finish_reason": "stop",
            }]
        });

        let resp = proto.parse_response(&body).unwrap();
        match resp {
            GenerateResponse::Text(t) => assert_eq!(t, ""),
            _ => panic!("Expected Text response"),
        }
    }

    #[test]
    fn test_parse_response_null_content_no_tools() {
        let proto = OpenAiCompatProtocol::new();
        let body = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null
                },
                "finish_reason": "stop",
            }]
        });

        let resp = proto.parse_response(&body).unwrap();
        match resp {
            GenerateResponse::Text(t) => assert_eq!(t, ""),
            _ => panic!("Expected Text response"),
        }
    }

    // ── parse_stream_chunk ──────────────────────────────────────

    #[test]
    fn test_parse_stream_content_chunk() {
        let proto = OpenAiCompatProtocol::new();
        let data = r#"{"choices":[{"delta":{"content":"hello"}}]}"#;
        let result = proto.parse_stream_chunk(data).unwrap();
        let token = result.expect("Expected Some(token)");
        assert_eq!(token.text, "hello");
        assert_eq!(token.id, 0);
        assert!(token.logprob.is_none());
        assert!(!token.is_special);
    }

    #[test]
    fn test_parse_stream_role_only_delta() {
        let proto = OpenAiCompatProtocol::new();
        let data = r#"{"choices":[{"delta":{"role":"assistant"}}]}"#;
        let result = proto.parse_stream_chunk(data).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_stream_empty_content() {
        let proto = OpenAiCompatProtocol::new();
        let data = r#"{"choices":[{"delta":{"content":""}}]}"#;
        let result = proto.parse_stream_chunk(data).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_stream_no_delta() {
        let proto = OpenAiCompatProtocol::new();
        let data = r#"{"choices":[{"finish_reason":"stop"}]}"#;
        let result = proto.parse_stream_chunk(data).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_stream_invalid_json() {
        let proto = OpenAiCompatProtocol::new();
        let result = proto.parse_stream_chunk("not json at all");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_stream_multi_token_sequence() {
        let proto = OpenAiCompatProtocol::new();
        let chunks = [
            r#"{"choices":[{"delta":{"role":"assistant"}}]}"#,
            r#"{"choices":[{"delta":{"content":"Hello"}}]}"#,
            r#"{"choices":[{"delta":{"content":" world"}}]}"#,
            r#"{"choices":[{"delta":{"content":"!"}}]}"#,
        ];

        let mut assembled = String::new();
        for chunk in &chunks {
            if let Some(token) = proto.parse_stream_chunk(chunk).unwrap() {
                assembled.push_str(&token.text);
            }
        }
        assert_eq!(assembled, "Hello world!");
    }

    // ── done_sentinel ───────────────────────────────────────────

    #[test]
    fn test_done_sentinel() {
        let proto = OpenAiCompatProtocol::new();
        assert_eq!(proto.done_sentinel(), Some("[DONE]"));
    }

    // ── extract_text_from_parts helper ──────────────────────────

    #[test]
    fn test_extract_text_from_parts_text_only() {
        let parts = vec![
            ContentPart::Text {
                text: "Hello".into(),
            },
            ContentPart::Text {
                text: "World".into(),
            },
        ];
        assert_eq!(extract_text_from_parts(&parts), "Hello\nWorld");
    }

    #[test]
    fn test_extract_text_from_parts_mixed() {
        let parts = vec![
            ContentPart::Text {
                text: "Describe this:".into(),
            },
            ContentPart::ImageUrl {
                url: "https://example.com/img.png".into(),
            },
        ];
        assert_eq!(extract_text_from_parts(&parts), "Describe this:");
    }

    #[test]
    fn test_extract_text_from_parts_no_text() {
        let parts = vec![ContentPart::ImageUrl {
            url: "https://example.com/img.png".into(),
        }];
        assert_eq!(extract_text_from_parts(&parts), "");
    }
}
