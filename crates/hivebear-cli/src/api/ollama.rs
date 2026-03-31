//! Ollama-compatible API endpoints.
//!
//! Implements the Ollama API so that tools like Open WebUI,
//! Continue.dev, and other Ollama-compatible clients work with HiveBear.
//!
//! Key difference from OpenAI: Ollama uses NDJSON (newline-delimited JSON)
//! for streaming instead of SSE.

use std::sync::Arc;

use axum::body::Body;
use axum::extract::State;
use axum::response::{IntoResponse, Json};
use axum::routing::{delete, get, post};
use axum::Router;
use futures::StreamExt;
use serde::{Deserialize, Serialize};

use hivebear_inference::{ChatMessage, GenerateRequest, SamplingParams};
use hivebear_registry::download::DownloadProgress;

use super::AppState;

/// Build the Ollama-compatible route group.
pub fn routes() -> Router<Arc<AppState>> {
    Router::new()
        .route("/api/chat", post(ollama_chat))
        .route("/api/generate", post(ollama_generate))
        .route("/api/tags", get(ollama_tags))
        .route("/api/show", post(ollama_show))
        .route("/api/pull", post(ollama_pull))
        .route("/api/delete", delete(ollama_delete))
        .route("/api/ps", get(ollama_ps))
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

#[derive(Deserialize)]
struct OllamaPullRequest {
    model: String,
    #[serde(default = "default_true")]
    stream: bool,
}

#[derive(Serialize)]
struct OllamaPullStatus {
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    completed: Option<u64>,
}

#[derive(Deserialize)]
struct OllamaDeleteRequest {
    model: String,
}

#[derive(Serialize)]
struct OllamaPsResponse {
    models: Vec<OllamaPsModel>,
}

#[derive(Serialize)]
struct OllamaPsModel {
    name: String,
    model: String,
    size: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    details: Option<OllamaModelDetails>,
}

// ── Route handlers ───────────────────────────────────────────────────

/// POST /api/chat — Ollama chat completion
async fn ollama_chat(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OllamaChatRequest>,
) -> axum::response::Response {
    // Resolve the model (auto-load if needed)
    let handle = match state.resolve_model(&req.model).await {
        Ok(h) => h,
        Err(e) => return error_response(&e),
    };

    let messages: Vec<ChatMessage> = req
        .messages
        .into_iter()
        .map(|m| match m.role.as_str() {
            "system" => ChatMessage::System(m.content),
            "assistant" => ChatMessage::assistant(m.content),
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
        model_name: Some(req.model.clone()),
        ..Default::default()
    };

    let model_name = req.model.clone();

    if req.stream {
        // NDJSON streaming
        let stream = match state.orchestrator.stream(&handle, &gen_req) {
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
        match state.orchestrator.generate(&handle, &gen_req).await {
            Ok(response) => {
                let text = match response {
                    hivebear_inference::GenerateResponse::Text(t) => t,
                    _ => String::new(),
                };
                Json(OllamaChatResponse {
                    model: req.model,
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
    // Resolve the model (auto-load if needed)
    let handle = match state.resolve_model(&req.model).await {
        Ok(h) => h,
        Err(e) => return error_response(&e),
    };

    let gen_req = GenerateRequest {
        messages: vec![ChatMessage::user_text(&req.prompt)],
        max_tokens: req.options.num_predict.unwrap_or(2048),
        sampling: SamplingParams {
            temperature: req.options.temperature,
            top_p: req.options.top_p.unwrap_or(0.9),
            ..Default::default()
        },
        model_name: Some(req.model.clone()),
        ..Default::default()
    };

    let model_name = req.model.clone();

    if req.stream {
        let stream = match state.orchestrator.stream(&handle, &gen_req) {
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
        match state.orchestrator.generate(&handle, &gen_req).await {
            Ok(response) => {
                let text = match response {
                    hivebear_inference::GenerateResponse::Text(t) => t,
                    _ => String::new(),
                };
                Json(OllamaGenerateResponse {
                    model: req.model,
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
    let models = state.models.read().await;
    let model_infos: Vec<OllamaModelInfo> = models
        .keys()
        .map(|name| OllamaModelInfo {
            name: name.clone(),
            model: name.clone(),
            size: 0,
            details: Some(OllamaModelDetails {
                format: "gguf".into(),
                family: "hivebear".into(),
            }),
        })
        .collect();

    // If no models loaded but registry is available, also list installed models
    let mut all_models = model_infos;
    if let Some(ref registry) = state.registry {
        let installed = registry.list_installed().await;
        for meta in installed {
            let already_listed = all_models.iter().any(|m| m.name == meta.id);
            if !already_listed {
                let size = meta.installed.as_ref().map(|i| i.size_bytes).unwrap_or(0);
                all_models.push(OllamaModelInfo {
                    name: meta.id.clone(),
                    model: meta.id,
                    size,
                    details: Some(OllamaModelDetails {
                        format: "gguf".into(),
                        family: "hivebear".into(),
                    }),
                });
            }
        }
    }

    Json(OllamaTagsResponse { models: all_models })
}

/// POST /api/show — Show model info (Ollama format)
async fn ollama_show(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OllamaShowRequest>,
) -> axum::response::Response {
    // Try to get model metadata from registry
    if let Some(ref registry) = state.registry {
        if let Ok(Some(meta)) = registry.get(&req.model).await {
            let format_str = meta
                .formats
                .first()
                .map(|f| format!("{:?}", f).to_lowercase())
                .unwrap_or_else(|| "gguf".into());
            return Json(OllamaShowResponse {
                modelfile: format!("FROM {}", meta.id),
                details: OllamaModelDetails {
                    format: format_str,
                    family: meta
                        .tags
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "hivebear".into()),
                },
            })
            .into_response();
        }
    }

    Json(OllamaShowResponse {
        modelfile: format!("FROM {}", req.model),
        details: OllamaModelDetails {
            format: "gguf".into(),
            family: "hivebear".into(),
        },
    })
    .into_response()
}

/// POST /api/pull — Download a model (Ollama format)
///
/// Streams download progress as NDJSON matching Ollama's format.
async fn ollama_pull(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OllamaPullRequest>,
) -> axum::response::Response {
    let Some(ref registry) = state.registry else {
        return error_response("Registry not available for model downloads");
    };

    // Check if already installed
    if let Ok(Some(meta)) = registry.get(&req.model).await {
        if meta.installed.is_some() {
            let line = serde_json::to_string(&OllamaPullStatus {
                status: "success".into(),
                digest: None,
                total: None,
                completed: None,
            })
            .unwrap_or_default();

            if req.stream {
                let body = Body::from(format!("{line}\n"));
                return axum::response::Response::builder()
                    .header("Content-Type", "application/x-ndjson")
                    .body(body)
                    .unwrap()
                    .into_response();
            } else {
                return Json(OllamaPullStatus {
                    status: "success".into(),
                    digest: None,
                    total: None,
                    completed: None,
                })
                .into_response();
            }
        }
    }

    if !req.stream {
        // Non-streaming: just do the download and return success/failure
        match registry.install(&req.model, None, None, None).await {
            Ok(_) => {
                return Json(OllamaPullStatus {
                    status: "success".into(),
                    digest: None,
                    total: None,
                    completed: None,
                })
                .into_response();
            }
            Err(e) => {
                return error_response(&format!("Pull failed: {e}"));
            }
        }
    }

    // Streaming: use a channel to relay progress as NDJSON
    let (tx, rx) = tokio::sync::mpsc::channel::<String>(64);
    let model_name = req.model.clone();

    let tx_clone = tx.clone();
    let model_clone = model_name.clone();

    tokio::spawn(async move {
        // Send initial status
        let _ = tx_clone
            .send(
                serde_json::to_string(&OllamaPullStatus {
                    status: "pulling manifest".into(),
                    digest: None,
                    total: None,
                    completed: None,
                })
                .unwrap_or_default(),
            )
            .await;

        // Note: We cannot use registry from within this spawned task because
        // AppState is behind an Arc and registry is not Clone.
        // Instead, signal completion — the actual download happens below.
        let _ = tx_clone
            .send(
                serde_json::to_string(&OllamaPullStatus {
                    status: format!("downloading {}", model_clone),
                    digest: None,
                    total: None,
                    completed: None,
                })
                .unwrap_or_default(),
            )
            .await;
    });

    let tx_final = tx.clone();
    let model_for_download = req.model.clone();

    tokio::spawn({
        // We need to work around the borrow issue. Since Registry is behind
        // Option in AppState (which is Arc), we access it directly.
        let state = state.clone();
        async move {
            let Some(ref registry) = state.registry else {
                let _ = tx_final
                    .send(
                        serde_json::to_string(&OllamaPullStatus {
                            status: "error: registry not available".into(),
                            digest: None,
                            total: None,
                            completed: None,
                        })
                        .unwrap_or_default(),
                    )
                    .await;
                return;
            };

            let progress_tx = tx_final.clone();
            let progress_cb = move |progress: DownloadProgress| {
                let status = OllamaPullStatus {
                    status: "downloading".into(),
                    digest: Some("sha256:downloading".into()),
                    total: progress.total_bytes,
                    completed: Some(progress.bytes_downloaded),
                };
                let _ = progress_tx.try_send(serde_json::to_string(&status).unwrap_or_default());
            };

            match registry
                .install(&model_for_download, None, None, Some(&progress_cb))
                .await
            {
                Ok(_) => {
                    let _ = tx_final
                        .send(
                            serde_json::to_string(&OllamaPullStatus {
                                status: "success".into(),
                                digest: None,
                                total: None,
                                completed: None,
                            })
                            .unwrap_or_default(),
                        )
                        .await;
                }
                Err(e) => {
                    let _ = tx_final
                        .send(
                            serde_json::to_string(&OllamaPullStatus {
                                status: format!("error: {e}"),
                                digest: None,
                                total: None,
                                completed: None,
                            })
                            .unwrap_or_default(),
                        )
                        .await;
                }
            }
        }
    });

    // Stream the channel as NDJSON
    let stream = tokio_stream::wrappers::ReceiverStream::new(rx)
        .map(|line| Ok::<_, std::convert::Infallible>(format!("{line}\n")));

    let body = Body::from_stream(stream);
    axum::response::Response::builder()
        .header("Content-Type", "application/x-ndjson")
        .body(body)
        .unwrap()
        .into_response()
}

/// DELETE /api/delete — Remove a model (Ollama format)
async fn ollama_delete(
    State(state): State<Arc<AppState>>,
    Json(req): Json<OllamaDeleteRequest>,
) -> axum::response::Response {
    // Unload from memory if currently loaded
    {
        let mut models = state.models.write().await;
        if let Some(handle) = models.remove(&req.model) {
            if let Err(e) = state.orchestrator.unload(&handle).await {
                tracing::warn!("Failed to unload model '{}': {e}", req.model);
            }
        }
    }

    // Remove from disk via registry
    let Some(ref registry) = state.registry else {
        return error_response("Registry not available for model deletion");
    };

    match registry.remove(&req.model).await {
        Ok(freed) => {
            tracing::info!("Deleted model '{}', freed {} bytes", req.model, freed);
            axum::http::StatusCode::OK.into_response()
        }
        Err(e) => error_response(&format!("Failed to delete '{}': {e}", req.model)),
    }
}

/// GET /api/ps — List running models (Ollama format)
async fn ollama_ps(State(state): State<Arc<AppState>>) -> Json<OllamaPsResponse> {
    let models = state.models.read().await;

    let ps_models: Vec<OllamaPsModel> = models
        .keys()
        .map(|name| {
            OllamaPsModel {
                name: name.clone(),
                model: name.clone(),
                size: 0, // Could be enhanced with model size tracking
                details: Some(OllamaModelDetails {
                    format: "gguf".into(),
                    family: "hivebear".into(),
                }),
            }
        })
        .collect();

    Json(OllamaPsResponse { models: ps_models })
}

fn error_response(message: &str) -> axum::response::Response {
    let body = serde_json::json!({ "error": message });
    (axum::http::StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
}
