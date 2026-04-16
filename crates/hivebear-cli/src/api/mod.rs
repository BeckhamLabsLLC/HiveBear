mod ollama;

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Json};
use axum::routing::{get, post};
use axum::Router;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tower_http::cors::{Any, CorsLayer};

use hivebear_core::config::paths::AppPaths;
use hivebear_core::types::HardwareProfile;
use hivebear_core::Config;
use hivebear_inference::{
    ChatMessage, ContentPart, GenerateRequest, LoadConfig, ModelHandle, Orchestrator,
    SamplingParams,
};
use hivebear_registry::Registry;

/// Shared application state for the API server.
///
/// Supports multiple concurrently loaded models. When a request arrives for
/// a model that is not yet loaded, the server resolves it through the
/// registry (downloading if necessary) and loads it into the orchestrator.
#[allow(dead_code)]
pub(crate) struct AppState {
    pub orchestrator: Orchestrator,
    pub models: RwLock<HashMap<String, ModelHandle>>,
    pub registry: Option<Registry>,
    pub config: Config,
    pub hw: HardwareProfile,
    pub api_key: Option<String>,
    pub default_model: Option<String>,
}

impl AppState {
    /// Resolve a model name to a loaded `ModelHandle`.
    ///
    /// If the model is already loaded, returns its handle immediately.
    /// Otherwise, resolves the model through the registry (downloading if
    /// needed) and loads it into the orchestrator.
    pub async fn resolve_model(&self, model_name: &str) -> Result<ModelHandle, String> {
        // Check if already loaded
        {
            let models = self.models.read().await;
            if let Some(handle) = models.get(model_name) {
                return Ok(handle.clone());
            }
        }

        // Check if it's a cloud model (e.g., "openai/gpt-4o", "anthropic/claude-sonnet-4-20250514")
        #[cfg(feature = "cloud")]
        if hivebear_inference::engine::cloud::registry::is_cloud_model(model_name) {
            return match self.orchestrator.load_cloud(model_name).await {
                Ok(handle) => {
                    let mut models = self.models.write().await;
                    models.insert(model_name.to_string(), handle.clone());
                    tracing::info!("Cloud model '{model_name}' loaded successfully");
                    Ok(handle)
                }
                Err(e) => Err(format!("Cloud model '{}': {}", model_name, e)),
            };
        }

        // Try to resolve and load via local registry
        let Some(ref registry) = self.registry else {
            return Err(format!(
                "Model '{}' is not loaded and no registry is available for on-demand loading",
                model_name
            ));
        };

        // Try resolving through registry (checks local index + file paths)
        let model_path = match registry.resolve(model_name).await {
            Ok(path) => path,
            Err(_) => {
                // Model not installed — try to download it
                tracing::info!("Model '{model_name}' not found locally, downloading...");
                match registry.install(model_name, None, None, None).await {
                    Ok(installed) => installed.path.join(&installed.filename),
                    Err(e) => {
                        return Err(format!(
                            "Model '{}' not found and download failed: {}",
                            model_name, e
                        ));
                    }
                }
            }
        };

        let load_config = LoadConfig {
            context_length: 4096,
            offload: hivebear_inference::OffloadConfig {
                auto: true,
                ..Default::default()
            },
            ..Default::default()
        };

        match self.orchestrator.load(&model_path, &load_config).await {
            Ok(handle) => {
                let mut models = self.models.write().await;
                models.insert(model_name.to_string(), handle.clone());
                tracing::info!("Model '{model_name}' loaded successfully");
                Ok(handle)
            }
            Err(e) => Err(format!("Failed to load model '{}': {}", model_name, e)),
        }
    }

    /// Get a handle for the default model, or the only loaded model.
    pub async fn default_handle(&self) -> Result<(String, ModelHandle), String> {
        if let Some(ref name) = self.default_model {
            let handle = self.resolve_model(name).await?;
            return Ok((name.clone(), handle));
        }

        let models = self.models.read().await;
        if models.len() == 1 {
            let (name, handle) = models.iter().next().unwrap();
            return Ok((name.clone(), handle.clone()));
        }

        if models.is_empty() {
            return Err("No models loaded. Specify a model in your request.".to_string());
        }

        Err(format!(
            "Multiple models loaded ({}). Specify which model to use in your request.",
            models.keys().cloned().collect::<Vec<_>>().join(", ")
        ))
    }
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

/// Configuration for starting the API server.
pub struct ServerConfig {
    pub orchestrator: Orchestrator,
    pub models: HashMap<String, ModelHandle>,
    pub default_model: Option<String>,
    pub port: u16,
    pub api_key: Option<String>,
    pub bind_address: String,
    pub cors_origins: Vec<String>,
    pub with_registry: bool,
}

/// Start the API server (supports both single-model and multi-model modes).
pub async fn start_server(config: ServerConfig) {
    let hw = config.orchestrator.profile().clone();
    let app_config = Config::load();

    let registry = if config.with_registry {
        let paths = AppPaths::new();
        match Registry::new(&app_config, &paths).await {
            Ok(r) => Some(r),
            Err(e) => {
                eprintln!("Warning: could not initialize registry for on-demand loading: {e}");
                None
            }
        }
    } else {
        None
    };

    let state = Arc::new(AppState {
        orchestrator: config.orchestrator,
        models: RwLock::new(config.models),
        registry,
        config: app_config,
        hw,
        api_key: config.api_key.clone(),
        default_model: config.default_model,
    });

    // Security warnings
    if config.api_key.is_none()
        && config.bind_address != "127.0.0.1"
        && config.bind_address != "localhost"
    {
        eprintln!(
            "⚠️  WARNING: API server is network-accessible ({}) with NO authentication!",
            config.bind_address
        );
        eprintln!("   Anyone on your network can send inference requests.");
        eprintln!(
            "   Use --api-key <key> to require authentication, or --bind 127.0.0.1 for local-only."
        );
    }

    if config.cors_origins.iter().any(|o| o == "*") && config.api_key.is_some() {
        eprintln!("⚠️  WARNING: Wildcard CORS ('*') is enabled with authentication.");
        eprintln!("   Any website can make authenticated API requests if the key is exposed.");
        eprintln!("   Consider restricting CORS origins to specific domains.");
    }

    // Build CORS layer — default to localhost if no origins specified
    let effective_origins: Vec<String> = if config.cors_origins.is_empty() {
        vec![
            "http://localhost:*".to_string(),
            "http://127.0.0.1:*".to_string(),
        ]
    } else {
        config.cors_origins.clone()
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

    let (listener, bound_port) =
        match bind_with_fallback(&config.bind_address, config.port, 10).await {
            Ok(pair) => pair,
            Err(e) => {
                eprintln!(
                    "\n❌ Could not start HiveBear API server: {e}\n\
                     \n\
                     Port {port} (and the next 10 after it) are all in use on {host}.\n\
                     Another process is probably holding the port — common culprits:\n\
                       • Another HiveBear or Ollama instance is already running\n\
                       • A previous run didn't exit cleanly (try again in a few seconds)\n\
                       • A different app is bound to the port\n\
                     \n\
                     You can pick a different port with `--port <N>` or stop the\n\
                     conflicting process and try again.\n",
                    port = config.port,
                    host = config.bind_address,
                );
                return;
            }
        };

    if bound_port != config.port {
        println!(
            "Port {} was busy — bound to {} instead.",
            config.port, bound_port
        );
    }

    println!(
        "API server listening on http://{}:{}",
        config.bind_address, bound_port
    );
    println!("  OpenAI:  POST /v1/chat/completions");
    println!("  OpenAI:  GET  /v1/models");
    println!("  Ollama:  POST /api/chat");
    println!("  Ollama:  POST /api/generate");
    println!("  Ollama:  GET  /api/tags");
    println!("  Ollama:  POST /api/show");
    println!("  Ollama:  POST /api/pull");
    println!("  Ollama:  DELETE /api/delete");
    println!("  Ollama:  GET  /api/ps");
    println!("  Health:  GET  /health");
    if let Some(key) = &config.api_key {
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

    if let Err(e) = axum::serve(listener, app).await {
        tracing::error!(error = %e, "API server exited with error");
        eprintln!("API server exited with error: {e}");
    }
}

/// Try to bind to the requested (host, port); if the port is taken, walk
/// forward up to `max_attempts - 1` additional ports before giving up.
/// Returns the bound listener and the actual port used.
async fn bind_with_fallback(
    host: &str,
    port: u16,
    max_attempts: u16,
) -> Result<(tokio::net::TcpListener, u16), std::io::Error> {
    let mut last_err: Option<std::io::Error> = None;
    for offset in 0..max_attempts {
        let try_port = match port.checked_add(offset) {
            Some(p) => p,
            None => break,
        };
        match tokio::net::TcpListener::bind(format!("{host}:{try_port}")).await {
            Ok(listener) => return Ok((listener, try_port)),
            Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
                last_err = Some(e);
                continue;
            }
            Err(e) => return Err(e),
        }
    }
    Err(last_err.unwrap_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::AddrInUse, "no free port in range")
    }))
}

// ── Legacy single-model start_server (used by `hivebear run --api`) ──

/// Start the API server with a single pre-loaded model.
/// Wraps the new multi-model `start_server` for backwards compatibility.
pub async fn start_server_single(
    orchestrator: Orchestrator,
    handle: ModelHandle,
    model_name: String,
    port: u16,
    api_key: Option<String>,
    bind_address: &str,
    cors_origins: &[String],
) {
    let mut models = HashMap::new();
    models.insert(model_name.clone(), handle);

    start_server(ServerConfig {
        orchestrator,
        models,
        default_model: Some(model_name),
        port,
        api_key,
        bind_address: bind_address.to_string(),
        cors_origins: cors_origins.to_vec(),
        with_registry: false,
    })
    .await;
}

// ── OpenAI-compatible request/response types ──────────────────────────

#[derive(Deserialize)]
struct ChatCompletionRequest {
    #[serde(default)]
    model: Option<String>,
    messages: Vec<ApiMessage>,
    #[serde(default = "default_max_tokens")]
    max_tokens: u32,
    #[serde(default = "default_temperature")]
    temperature: f32,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    top_p: Option<f32>,
    // Tool calling (critical for Cursor)
    #[serde(default)]
    tools: Vec<ApiToolDefinition>,
    #[serde(default)]
    tool_choice: Option<serde_json::Value>,
    // Structured output
    #[serde(default)]
    response_format: Option<ApiResponseFormat>,
    // Additional sampling parameters
    #[serde(default)]
    stop: Option<StopSequences>,
    #[serde(default)]
    frequency_penalty: Option<f32>,
    #[serde(default)]
    presence_penalty: Option<f32>,
}

fn default_max_tokens() -> u32 {
    2048
}
fn default_temperature() -> f32 {
    0.7
}

// ── Message types (multimodal + tool calls) ─────────────────────────

#[derive(Deserialize)]
struct ApiMessage {
    role: String,
    #[serde(default)]
    content: Option<ApiMessageContent>,
    #[serde(default)]
    tool_calls: Option<Vec<ApiToolCall>>,
    #[serde(default)]
    tool_call_id: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ApiMessageContent {
    Text(String),
    Parts(Vec<ApiContentPart>),
}

#[derive(Deserialize)]
#[serde(tag = "type")]
enum ApiContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ApiImageUrl },
}

#[derive(Deserialize)]
struct ApiImageUrl {
    url: String,
}

// ── Tool calling types ──────────────────────────────────────────────

#[derive(Deserialize)]
struct ApiToolDefinition {
    #[serde(rename = "type")]
    _type: String,
    function: ApiFunctionDef,
}

#[derive(Deserialize)]
struct ApiFunctionDef {
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    parameters: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct ApiToolCall {
    id: String,
    #[serde(rename = "type")]
    _type: String,
    function: ApiToolCallFunction,
}

#[derive(Deserialize)]
struct ApiToolCallFunction {
    name: String,
    arguments: String,
}

// ── Response format / structured output ─────────────────────────────

#[derive(Deserialize)]
struct ApiResponseFormat {
    #[serde(rename = "type")]
    _type: String,
    #[serde(default)]
    json_schema: Option<serde_json::Value>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum StopSequences {
    Single(String),
    Multiple(Vec<String>),
}

// ── Response types ──────────────────────────────────────────────────

#[derive(Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: String,
    model: String,
    choices: Vec<Choice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Usage>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ApiToolCallResponse>>,
}

#[derive(Serialize)]
struct ApiToolCallResponse {
    id: String,
    #[serde(rename = "type")]
    _type: String,
    function: ApiToolCallFunctionResponse,
}

#[derive(Serialize)]
struct ApiToolCallFunctionResponse {
    name: String,
    arguments: String,
}

#[derive(Serialize, Clone)]
struct Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Serialize)]
struct StreamChunk {
    id: String,
    object: String,
    model: String,
    choices: Vec<StreamChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Usage>,
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

// ── Conversion helpers ──────────────────────────────────────────────

fn convert_api_messages(messages: Vec<ApiMessage>) -> Vec<ChatMessage> {
    use hivebear_inference::ToolCallResponse;

    messages
        .into_iter()
        .map(|m| match m.role.as_str() {
            "system" => {
                let text = match m.content {
                    Some(ApiMessageContent::Text(s)) => s,
                    _ => String::new(),
                };
                ChatMessage::System(text)
            }
            "assistant" => {
                let content = match m.content {
                    Some(ApiMessageContent::Text(s)) => Some(s),
                    _ => None,
                };
                let tool_calls = m
                    .tool_calls
                    .unwrap_or_default()
                    .into_iter()
                    .map(|tc| ToolCallResponse {
                        tool_name: tc.function.name,
                        arguments: serde_json::from_str(&tc.function.arguments)
                            .unwrap_or(serde_json::Value::Null),
                        call_id: tc.id,
                    })
                    .collect();
                ChatMessage::assistant_with_tool_calls(content, tool_calls)
            }
            "tool" => ChatMessage::ToolResult {
                tool_call_id: m.tool_call_id.unwrap_or_default(),
                content: match m.content {
                    Some(ApiMessageContent::Text(s)) => s,
                    _ => String::new(),
                },
            },
            _ => {
                // user (or unknown role)
                match m.content {
                    Some(ApiMessageContent::Parts(parts)) => {
                        let content_parts: Vec<ContentPart> = parts
                            .into_iter()
                            .map(|p| match p {
                                ApiContentPart::Text { text } => ContentPart::Text { text },
                                ApiContentPart::ImageUrl { image_url } => {
                                    ContentPart::ImageUrl { url: image_url.url }
                                }
                            })
                            .collect();
                        ChatMessage::User(content_parts)
                    }
                    Some(ApiMessageContent::Text(s)) => ChatMessage::user_text(&s),
                    None => ChatMessage::user_text(""),
                }
            }
        })
        .collect()
}

fn convert_tools(tools: &[ApiToolDefinition]) -> Vec<hivebear_inference::ToolDefinition> {
    tools
        .iter()
        .map(|t| hivebear_inference::ToolDefinition {
            name: t.function.name.clone(),
            description: t.function.description.clone().unwrap_or_default(),
            input_schema: t
                .function
                .parameters
                .clone()
                .unwrap_or(serde_json::json!({})),
        })
        .collect()
}

fn convert_tool_choice(choice: &Option<serde_json::Value>) -> hivebear_inference::ToolChoice {
    use hivebear_inference::ToolChoice;
    match choice {
        None => ToolChoice::Auto,
        Some(v) => {
            if let Some(s) = v.as_str() {
                match s {
                    "none" => ToolChoice::None,
                    "required" => ToolChoice::Required,
                    _ => ToolChoice::Auto,
                }
            } else if let Some(name) = v
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
            {
                ToolChoice::Specific(name.to_string())
            } else {
                ToolChoice::Auto
            }
        }
    }
}

fn convert_stop(stop: &Option<StopSequences>) -> Vec<String> {
    match stop {
        None => vec![],
        Some(StopSequences::Single(s)) => vec![s.clone()],
        Some(StopSequences::Multiple(v)) => v.clone(),
    }
}

fn convert_tool_call_response(tc: hivebear_inference::ToolCallResponse) -> ApiToolCallResponse {
    ApiToolCallResponse {
        id: tc.call_id,
        _type: "function".into(),
        function: ApiToolCallFunctionResponse {
            name: tc.tool_name,
            arguments: tc.arguments.to_string(),
        },
    }
}

// ── Route handlers ────────────────────────────────────────────────────

async fn health() -> &'static str {
    "ok"
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelListResponse> {
    let models = state.models.read().await;
    let mut data: Vec<ModelEntry> = models
        .keys()
        .map(|name| ModelEntry {
            id: name.clone(),
            object: "model".into(),
            owned_by: "hivebear-local".into(),
        })
        .collect();

    // List cloud providers that have API keys configured
    #[cfg(feature = "cloud")]
    {
        use hivebear_inference::engine::cloud::registry::BUILTIN_PROVIDERS;

        for provider in BUILTIN_PROVIDERS {
            if !provider.requires_api_key
                || std::env::var(provider.env_var)
                    .map(|v| !v.is_empty())
                    .unwrap_or(false)
            {
                // Skip providers already listed as loaded models
                let prefix = provider.prefix;
                let already_listed = data.iter().any(|m| m.id.starts_with(prefix));
                if !already_listed {
                    data.push(ModelEntry {
                        id: format!("{}/*", prefix),
                        object: "model".into(),
                        owned_by: provider.display_name.to_string(),
                    });
                }
            }
        }
    }

    Json(ModelListResponse {
        object: "list".into(),
        data,
    })
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> axum::response::Response {
    // Resolve the model handle
    let (model_name, handle) = if let Some(ref requested) = req.model {
        match state.resolve_model(requested).await {
            Ok(h) => (requested.clone(), h),
            Err(e) => return error_response(&e),
        }
    } else {
        match state.default_handle().await {
            Ok(pair) => pair,
            Err(e) => return error_response(&e),
        }
    };

    let messages = convert_api_messages(req.messages);

    let gen_req = GenerateRequest {
        messages,
        max_tokens: req.max_tokens,
        sampling: SamplingParams {
            temperature: req.temperature,
            top_p: req.top_p.unwrap_or(0.9),
            frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
            presence_penalty: req.presence_penalty.unwrap_or(0.0),
            ..Default::default()
        },
        tools: convert_tools(&req.tools),
        tool_choice: convert_tool_choice(&req.tool_choice),
        stop_sequences: convert_stop(&req.stop),
        output_schema: req.response_format.and_then(|rf| rf.json_schema),
        model_name: Some(model_name.clone()),
        ..Default::default()
    };

    if req.stream {
        stream_response(state, handle, model_name, gen_req).await
    } else {
        non_stream_response(state, handle, model_name, gen_req).await
    }
}

async fn non_stream_response(
    state: Arc<AppState>,
    handle: ModelHandle,
    model_name: String,
    req: GenerateRequest,
) -> axum::response::Response {
    match state.orchestrator.generate(&handle, &req).await {
        Ok(response) => {
            let (content, tool_calls, finish_reason) = match response {
                hivebear_inference::GenerateResponse::Text(t) => (Some(t), None, "stop"),
                hivebear_inference::GenerateResponse::ToolCall(tc) => (
                    None,
                    Some(vec![convert_tool_call_response(tc)]),
                    "tool_calls",
                ),
                hivebear_inference::GenerateResponse::Mixed(blocks) => {
                    let mut text_parts = Vec::new();
                    let mut calls = Vec::new();
                    for block in blocks {
                        match block {
                            hivebear_inference::ContentBlock::Text(t) => text_parts.push(t),
                            hivebear_inference::ContentBlock::ToolCall(tc) => {
                                calls.push(convert_tool_call_response(tc))
                            }
                        }
                    }
                    let text = if text_parts.is_empty() {
                        None
                    } else {
                        Some(text_parts.join(""))
                    };
                    let tool_calls = if calls.is_empty() { None } else { Some(calls) };
                    let reason = if tool_calls.is_some() {
                        "tool_calls"
                    } else {
                        "stop"
                    };
                    (text, tool_calls, reason)
                }
            };

            let resp = ChatCompletionResponse {
                id: format!("chatcmpl-{}", uuid::Uuid::new_v4()),
                object: "chat.completion".into(),
                model: model_name,
                choices: vec![Choice {
                    index: 0,
                    message: ChoiceMessage {
                        role: "assistant".into(),
                        content,
                        tool_calls,
                    },
                    finish_reason: finish_reason.into(),
                }],
                usage: None,
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

async fn stream_response(
    state: Arc<AppState>,
    handle: ModelHandle,
    model_name: String,
    req: GenerateRequest,
) -> axum::response::Response {
    let chat_id = format!("chatcmpl-{}", uuid::Uuid::new_v4());

    let stream = match state.orchestrator.stream(&handle, &req) {
        Ok(s) => s,
        Err(e) => {
            let body = serde_json::json!({
                "error": { "message": e.to_string(), "type": "server_error" }
            });
            return (axum::http::StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response();
        }
    };

    let mut first = true;
    let mut token_count: u32 = 0;
    let chat_id_clone = chat_id.clone();
    let model_name_clone = model_name.clone();

    let event_stream = stream
        .map(move |result| {
            let chunk = match result {
                Ok(token) => {
                    token_count += 1;
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
                        usage: None,
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
                    usage: Some(Usage {
                        prompt_tokens: 0,
                        completion_tokens: token_count,
                        total_tokens: token_count,
                    }),
                },
            };

            let data = serde_json::to_string(&chunk).unwrap_or_default();
            Ok::<_, std::convert::Infallible>(Event::default().data(data))
        })
        // Final chunk with finish_reason (in case stream ends without error)
        .chain(futures::stream::once({
            let chat_id2 = chat_id.clone();
            let model_name2 = model_name.clone();
            async move {
                let chunk = StreamChunk {
                    id: chat_id2,
                    object: "chat.completion.chunk".into(),
                    model: model_name2,
                    choices: vec![StreamChoice {
                        index: 0,
                        delta: Delta {
                            role: None,
                            content: None,
                        },
                        finish_reason: Some("stop".into()),
                    }],
                    usage: None,
                };
                let data = serde_json::to_string(&chunk).unwrap_or_default();
                Ok::<_, std::convert::Infallible>(Event::default().data(data))
            }
        }))
        // [DONE] sentinel — required by OpenAI spec, Cursor depends on this
        .chain(futures::stream::once(async {
            Ok::<_, std::convert::Infallible>(Event::default().data("[DONE]"))
        }));

    Sse::new(event_stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn error_response(message: &str) -> axum::response::Response {
    let body = serde_json::json!({
        "error": { "message": message, "type": "server_error" }
    });
    (axum::http::StatusCode::INTERNAL_SERVER_ERROR, Json(body)).into_response()
}
