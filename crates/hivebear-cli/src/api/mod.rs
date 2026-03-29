mod ollama;

use std::sync::Arc;

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json};
use axum::routing::{get, post};
use axum::Router;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tower_http::cors::{Any, CorsLayer};

use hivebear_inference::{ChatMessage, GenerateRequest, ModelHandle, Orchestrator, SamplingParams};

pub(crate) struct AppState {
    pub orchestrator: Orchestrator,
    pub handle: ModelHandle,
    pub model_name: String,
    pub api_key: Option<String>,
}

// ── Auth middleware ───────────────────────────────────────────────────

async fn auth_middleware(
    State(state): State<Arc<AppState>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    // If no API key configured, allow all requests (--no-auth mode)
    let Some(expected_key) = &state.api_key else {
        return next.run(req).await;
    };

    // Skip auth for health endpoint
    if req.uri().path() == "/health" {
        return next.run(req).await;
    }

    // Check Authorization header
    if let Some(auth_header) = req.headers().get("authorization") {
        if let Ok(auth_str) = auth_header.to_str() {
            if let Some(token) = auth_str.strip_prefix("Bearer ") {
                if token == expected_key {
                    return next.run(req).await;
                }
            }
        }
    }

    // Return 401
    let body = serde_json::json!({
        "error": {
            "message": "Invalid or missing API key. Use Authorization: Bearer <key>",
            "type": "authentication_error",
            "code": "invalid_api_key"
        }
    });
    (axum::http::StatusCode::UNAUTHORIZED, Json(body)).into_response()
}

/// Start the OpenAI-compatible API server.
pub async fn start_server(
    orchestrator: Orchestrator,
    handle: ModelHandle,
    model_name: String,
    port: u16,
    api_key: Option<String>,
    bind_address: &str,
    cors_origins: &[String],
) {
    let state = Arc::new(AppState {
        orchestrator,
        handle,
        model_name,
        api_key: api_key.clone(),
    });

    // Security warnings
    if api_key.is_none() && bind_address != "127.0.0.1" && bind_address != "localhost" {
        eprintln!("⚠️  WARNING: API server is network-accessible ({bind_address}) with NO authentication!");
        eprintln!("   Anyone on your network can send inference requests.");
        eprintln!(
            "   Use --api-key <key> to require authentication, or --bind 127.0.0.1 for local-only."
        );
    }

    if cors_origins.iter().any(|o| o == "*") && api_key.is_some() {
        eprintln!("⚠️  WARNING: Wildcard CORS ('*') is enabled with authentication.");
        eprintln!("   Any website can make authenticated API requests if the key is exposed.");
        eprintln!("   Consider restricting CORS origins to specific domains.");
    }

    // Build CORS layer — default to localhost if no origins specified
    let effective_origins: Vec<String> = if cors_origins.is_empty() {
        vec![
            "http://localhost:*".to_string(),
            "http://127.0.0.1:*".to_string(),
        ]
    } else {
        cors_origins.to_vec()
    };

    let cors = if effective_origins.iter().any(|o| o == "*") {
        CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any)
    } else {
        let origins: Vec<_> = effective_origins
            .iter()
            .filter_map(|o| o.parse().ok())
            .collect();
        CorsLayer::new()
            .allow_origin(origins)
            .allow_methods(Any)
            .allow_headers(Any)
    };

    let app = Router::new()
        // OpenAI-compatible endpoints
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        // Ollama-compatible endpoints
        .merge(ollama::routes())
        // Common
        .route("/health", get(health))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            auth_middleware,
        ))
        .layer(cors)
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(format!("{bind_address}:{port}"))
        .await
        .expect("Failed to bind to port");

    println!("API server listening on http://{}:{}", bind_address, port);
    println!("  OpenAI:  POST /v1/chat/completions");
    println!("  OpenAI:  GET  /v1/models");
    println!("  Ollama:  POST /api/chat");
    println!("  Ollama:  POST /api/generate");
    println!("  Ollama:  GET  /api/tags");
    println!("  Ollama:  POST /api/show");
    println!("  Health:  GET  /health");
    if let Some(key) = &api_key {
        let masked = if key.len() > 8 {
            format!("{}...{}", &key[..4], &key[key.len() - 4..])
        } else {
            "*".repeat(key.len())
        };
        println!("  API key: {}", masked);
        println!("  Use: curl -H 'Authorization: Bearer <your-key>' ...");
    } else {
        println!("  Auth: disabled (--no-auth)");
    }

    axum::serve(listener, app).await.expect("Server error");
}

// ── OpenAI-compatible request/response types ──────────────────────────

#[derive(Deserialize)]
struct ChatCompletionRequest {
    messages: Vec<ApiMessage>,
    #[serde(default = "default_max_tokens")]
    max_tokens: u32,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    top_p: Option<f32>,
}

fn default_max_tokens() -> u32 {
    2048
}
fn default_temperature() -> f32 {
    0.7
}

#[derive(Deserialize)]
struct ApiMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    model: String,
    choices: Vec<Choice>,
}

#[derive(Serialize)]
struct Choice {
    index: u32,
    message: ChoiceMessage,
    finish_reason: String,
}

#[derive(Serialize)]
struct ChoiceMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct StreamChunk {
    id: String,
    object: String,
    model: String,
    choices: Vec<StreamChoice>,
}

#[derive(Serialize)]
struct StreamChoice {
    index: u32,
    delta: Delta,
    finish_reason: Option<String>,
}

#[derive(Serialize)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Serialize)]
struct ModelListResponse {
    object: String,
    data: Vec<ModelEntry>,
}

#[derive(Serialize)]
struct ModelEntry {
    id: String,
    object: String,
    owned_by: String,
}

// ── Route handlers ────────────────────────────────────────────────────

async fn health() -> &'static str {
    "ok"
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelListResponse> {
    Json(ModelListResponse {
        object: "list".into(),
        data: vec![ModelEntry {
            id: state.model_name.clone(),
            object: "model".into(),
            owned_by: "hivebear".into(),
        }],
    })
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
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
        max_tokens: req.max_tokens,
        sampling: SamplingParams {
            temperature: req.temperature,
            top_p: req.top_p.unwrap_or(0.9),
            ..Default::default()
        },
        model_name: Some(state.model_name.clone()),
        ..Default::default()
    };

    if req.stream {
        stream_response(state, gen_req).await
    } else {
        non_stream_response(state, gen_req).await
    }
}

async fn non_stream_response(
    state: Arc<AppState>,
    req: GenerateRequest,
) -> axum::response::Response {
    match state.orchestrator.generate(&state.handle, &req).await {
        Ok(response) => {
            let text = match response {
                hivebear_inference::GenerateResponse::Text(t) => t,
                _ => "".into(),
            };

            let resp = ChatCompletionResponse {
                id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                object: "chat.completion".into(),
                model: state.model_name.clone(),
                choices: vec![Choice {
                    index: 0,
                    message: ChoiceMessage {
                        role: "assistant".into(),
                        content: text,
                    },
                    finish_reason: "stop".into(),
                }],
            };

            Json(resp).into_response()
        }
        Err(e) => {
            let body = serde_json::json!({
                "error": { "message": e.to_string(), "type": "server_error" }
            });
            (axum::http::StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
        }
    }
}

async fn stream_response(state: Arc<AppState>, req: GenerateRequest) -> axum::response::Response {
    let model_name = state.model_name.clone();
    let chat_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

    let stream = match state.orchestrator.stream(&state.handle, &req) {
        Ok(s) => s,
        Err(e) => {
            let body = serde_json::json!({
                "error": { "message": e.to_string(), "type": "server_error" }
            });
            return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response();
        }
    };

    let mut first = true;
    let chat_id_clone = chat_id.clone();
    let model_name_clone = model_name.clone();

    let event_stream = stream.map(move |result| {
        let chunk = match result {
            Ok(token) => {
                let mut delta = Delta {
                    role: None,
                    content: Some(token.text),
                };
                if first {
                    delta.role = Some("assistant".into());
                    first = false;
                }
                StreamChunk {
                    id: chat_id_clone.clone(),
                    object: "chat.completion.chunk".into(),
                    model: model_name_clone.clone(),
                    choices: vec![StreamChoice {
                        index: 0,
                        delta,
                        finish_reason: None,
                    }],
                }
            }
            Err(_) => StreamChunk {
                id: chat_id_clone.clone(),
                object: "chat.completion.chunk".into(),
                model: model_name_clone.clone(),
                choices: vec![StreamChoice {
                    index: 0,
                    delta: Delta {
                        role: None,
                        content: None,
                    },
                    finish_reason: Some("stop".into()),
                }],
            },
        };

        let data = serde_json::to_string(&chunk).unwrap_or_default();
        Ok::<_, std::convert::Infallible>(Event::default().data(data))
    });

    Sse::new(event_stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}
