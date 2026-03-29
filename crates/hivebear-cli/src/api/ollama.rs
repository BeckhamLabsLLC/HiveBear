//! Ollama-compatible API endpoints.
//!
//! Implements the core Ollama API so that tools like Open WebUI,
//! Continue.dev, and other Ollama-compatible clients work with HiveBear.
//!
//! Key difference from OpenAI: Ollama uses NDJSON (newline-delimited JSON)
//! for streaming instead of SSE.

use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::response::{IntoResponse, Json};
use axum::routing::{get, post};
use axum::Router;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use hivebear_inference::{ChatMessage, GenerateRequest, SamplingParams};

use super::AppState;

/// Build the Ollama-compatible route group.
pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/chat", post(ollama_chat))
        .route("/api/generate", post(ollama_generate))
        .route("/api/tags", get(ollama_tags))
        .route("/api/show", post(ollama_show))
}

// ── Ollama request/response types ────────────────────────────────────

#[derive(Deserialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    #[serde(default = "default_true")]
    stream: bool,
    #[serde(default)]
    options: OllamaOptions,
}

#[derive(Deserialize)]
struct OllamaGenerateRequest {
    model: String,
    prompt: String,
    #[serde(default = "default_true")]
    stream: bool,
    #[serde(default)]
    options: OllamaOptions,
}

#[derive(Deserialize, Default)]
struct OllamaOptions {
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    num_predict: Option<u32>,
}

#[derive(Deserialize)]
struct OllamaMessage {
    role: String,
    content: String,
}

fn default_true() -> bool {
    true
}
fn default_temperature() -> f32 {
    0.7
}

// Response types

#[derive(Serialize)]
struct OllamaChatResponse {
    model: String,
    message: OllamaResponseMessage,
    done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    eval_count: Option<u32>,
}

#[derive(Serialize)]
struct OllamaGenerateResponse {
    model: String,
    response: String,
    done: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_duration: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    eval_count: Option<u32>,
}

#[derive(Serialize)]
struct OllamaResponseMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct OllamaTagsResponse {
    models: Vec<OllamaModelInfo>,
}

#[derive(Serialize)]
struct OllamaModelInfo {
    name: String,
    model: String,
    size: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    details: Option<OllamaModelDetails>,
}

#[derive(Serialize)]
struct OllamaModelDetails {
    format: String,
    family: String,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct OllamaShowRequest {
    model: String,
}

#[derive(Serialize)]
struct OllamaShowResponse {
    modelfile: String,
    details: OllamaModelDetails,
}

// ── Route handlers ───────────────────────────────────────────────────

/// POST /api/chat — Ollama chat completion
async fn ollama_chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OllamaChatRequest>,
) -> axum::response::Response {
    let messages: Vec<ChatMessage> = req
        .messages
        .into_iter()
        .map(|m| match m.role.as_str() {
            "system" => ChatMessage::System(m.content),
            "assistant" => ChatMessage::Assistant(m.content),
            _ => ChatMessage::user_text(&m.content),
        })
        .collect();

    let gen_req = GenerateRequest {
        messages,
        max_tokens: req.options.num_predict.unwrap_or(2048),
        sampling: SamplingParams {
            temperature: req.options.temperature,
            top_p: req.options.top_p.unwrap_or(0.9),
            ..Default::default()
        },
        model_name: Some(state.model_name.clone()),
        ..Default::default()
    };

    let model_name = state.model_name.clone();

    if req.stream {
        // NDJSON streaming
        let stream = match state.orchestrator.stream(&state.handle, &gen_req) {
            Ok(s) => s,
            Err(e) => return error_response(&e.to_string()),
        };

        let ndjson_stream = stream
            .map(move |result| {
                let line = match result {
                    Ok(token) => serde_json::to_string(&OllamaChatResponse {
                        model: model_name.clone(),
                        message: OllamaResponseMessage {
                            role: "assistant".into(),
                            content: token.text,
                        },
                        done: false,
                        total_duration: None,
                        eval_count: None,
                    })
                    .unwrap_or_default(),
                    Err(_) => serde_json::to_string(&OllamaChatResponse {
                        model: model_name.clone(),
                        message: OllamaResponseMessage {
                            role: "assistant".into(),
                            content: String::new(),
                        },
                        done: true,
                        total_duration: None,
                        eval_count: None,
                    })
                    .unwrap_or_default(),
                };
                Ok::<_, std::convert::Infallible>(format!("{line}\n"))
            })
            .chain(futures::stream::once({
                let model_name2 = req.model.clone();
                async move {
                    let final_line = serde_json::to_string(&OllamaChatResponse {
                        model: model_name2,
                        message: OllamaResponseMessage {
                            role: "assistant".into(),
                            content: String::new(),
                        },
                        done: true,
                        total_duration: None,
                        eval_count: None,
                    })
                    .unwrap_or_default();
                    Ok::<_, std::convert::Infallible>(format!("{final_line}\n"))
                }
            }));

        let body = Body::from_stream(ndjson_stream);
        axum::response::Response::builder()
            .header("Content-Type", "application/x-ndjson")
            .body(body)
            .unwrap()
            .into_response()
    } else {
        // Non-streaming: collect full response
        match state.orchestrator.generate(&state.handle, &gen_req).await {
            Ok(response) => {
                let text = match response {
                    hivebear_inference::GenerateResponse::Text(t) => t,
                    _ => String::new(),
                };
                Json(OllamaChatResponse {
                    model: model_name,
                    message: OllamaResponseMessage {
                        role: "assistant".into(),
                        content: text,
                    },
                    done: true,
                    total_duration: None,
                    eval_count: None,
                })
                .into_response()
            }
            Err(e) => error_response(&e.to_string()),
        }
    }
}

/// POST /api/generate — Ollama raw text generation
async fn ollama_generate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OllamaGenerateRequest>,
) -> axum::response::Response {
    let gen_req = GenerateRequest {
        messages: vec![ChatMessage::user_text(&req.prompt)],
        max_tokens: req.options.num_predict.unwrap_or(2048),
        sampling: SamplingParams {
            temperature: req.options.temperature,
            top_p: req.options.top_p.unwrap_or(0.9),
            ..Default::default()
        },
        model_name: Some(state.model_name.clone()),
        ..Default::default()
    };

    let model_name = state.model_name.clone();

    if req.stream {
        let stream = match state.orchestrator.stream(&state.handle, &gen_req) {
            Ok(s) => s,
            Err(e) => return error_response(&e.to_string()),
        };

        let ndjson_stream = stream
            .map(move |result| {
                let line = match result {
                    Ok(token) => serde_json::to_string(&OllamaGenerateResponse {
                        model: model_name.clone(),
                        response: token.text,
                        done: false,
                        total_duration: None,
                        eval_count: None,
                    })
                    .unwrap_or_default(),
                    Err(_) => serde_json::to_string(&OllamaGenerateResponse {
                        model: model_name.clone(),
                        response: String::new(),
                        done: true,
                        total_duration: None,
                        eval_count: None,
                    })
                    .unwrap_or_default(),
                };
                Ok::<_, std::convert::Infallible>(format!("{line}\n"))
            })
            .chain(futures::stream::once({
                let model_name2 = req.model.clone();
                async move {
                    let final_line = serde_json::to_string(&OllamaGenerateResponse {
                        model: model_name2,
                        response: String::new(),
                        done: true,
                        total_duration: None,
                        eval_count: None,
                    })
                    .unwrap_or_default();
                    Ok::<_, std::convert::Infallible>(format!("{final_line}\n"))
                }
            }));

        let body = Body::from_stream(ndjson_stream);
        axum::response::Response::builder()
            .header("Content-Type", "application/x-ndjson")
            .body(body)
            .unwrap()
            .into_response()
    } else {
        match state.orchestrator.generate(&state.handle, &gen_req).await {
            Ok(response) => {
                let text = match response {
                    hivebear_inference::GenerateResponse::Text(t) => t,
                    _ => String::new(),
                };
                Json(OllamaGenerateResponse {
                    model: model_name,
                    response: text,
                    done: true,
                    total_duration: None,
                    eval_count: None,
                })
                .into_response()
            }
            Err(e) => error_response(&e.to_string()),
        }
    }
}

/// GET /api/tags — List available models (Ollama format)
async fn ollama_tags(State(state): State<Arc<AppState>>) -> Json<OllamaTagsResponse> {
    Json(OllamaTagsResponse {
        models: vec![OllamaModelInfo {
            name: state.model_name.clone(),
            model: state.model_name.clone(),
            size: 0,
            details: Some(OllamaModelDetails {
                format: "gguf".into(),
                family: "hivebear".into(),
            }),
        }],
    })
}

/// POST /api/show — Show model info (Ollama format)
async fn ollama_show(
    State(state): State<Arc<AppState>>,
    Json(_req): Json<OllamaShowRequest>,
) -> Json<OllamaShowResponse> {
    Json(OllamaShowResponse {
        modelfile: format!("FROM {}", state.model_name),
        details: OllamaModelDetails {
            format: "gguf".into(),
            family: "hivebear".into(),
        },
    })
}

fn error_response(message: &str) -> axum::response::Response {
    let body = serde_json::json!({ "error": message });
    (axum::http::StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
}
