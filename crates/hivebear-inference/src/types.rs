use hivebear_core::types::InferenceEngine;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

// ── Model Handle ──────────────────────────────────────────────────────

/// Opaque handle to a loaded model.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelHandle {
    pub id: u64,
    pub model_path: PathBuf,
    pub engine: InferenceEngine,
}

static HANDLE_COUNTER: AtomicU64 = AtomicU64::new(1);

impl ModelHandle {
    pub fn new(path: PathBuf, engine: InferenceEngine) -> Self {
        Self {
            id: HANDLE_COUNTER.fetch_add(1, Ordering::Relaxed),
            model_path: path,
            engine,
        }
    }
}

// ── Load Configuration ────────────────────────────────────────────────

/// Configuration for loading a model into an engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadConfig {
    pub context_length: u32,
    pub offload: OffloadConfig,
    pub batch_size: u32,
    /// Number of threads for CPU inference. None = auto-detect.
    pub threads: Option<u32>,
    pub use_mmap: bool,
    pub use_mlock: bool,
    pub seed: Option<u64>,
}

impl Default for LoadConfig {
    fn default() -> Self {
        Self {
            context_length: 4096,
            offload: OffloadConfig::default(),
            batch_size: 512,
            threads: None,
            use_mmap: true,
            use_mlock: false,
            seed: None,
        }
    }
}

/// How to distribute model layers across GPU, CPU, disk, and mesh peers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OffloadConfig {
    /// Number of layers to offload to GPU. None = auto-calculate.
    pub gpu_layers: Option<u32>,
    pub cpu_layers: Option<u32>,
    pub disk_mmap: bool,
    /// Number of layers to offload to P2P mesh peers. None = no mesh.
    pub mesh_layers: Option<u32>,
    /// Let HiveBear calculate the optimal split based on hardware.
    pub auto: bool,
}

impl Default for OffloadConfig {
    fn default() -> Self {
        Self {
            gpu_layers: None,
            cpu_layers: None,
            disk_mmap: false,
            mesh_layers: None,
            auto: true,
        }
    }
}

// ── Chat Messages ─────────────────────────────────────────────────────

/// A part of a multimodal user message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image {
        #[serde(with = "base64_bytes")]
        data: Vec<u8>,
        media_type: String,
    },
    #[serde(rename = "image_url")]
    ImageUrl { url: String },
}

/// Serde helper for base64 encoding of image bytes.
mod base64_bytes {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(bytes: &[u8], s: S) -> Result<S::Ok, S::Error> {
        use base64::Engine;
        let encoded = base64::engine::general_purpose::STANDARD.encode(bytes);
        encoded.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        use base64::Engine;
        let s = String::deserialize(d)?;
        base64::engine::general_purpose::STANDARD
            .decode(&s)
            .map_err(serde::de::Error::custom)
    }
}

/// A message in a chat conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role", content = "content")]
pub enum ChatMessage {
    #[serde(rename = "system")]
    System(String),
    #[serde(rename = "user")]
    User(Vec<ContentPart>),
    #[serde(rename = "assistant")]
    Assistant {
        content: Option<String>,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        tool_calls: Vec<ToolCallResponse>,
    },
    #[serde(rename = "tool")]
    ToolResult {
        tool_call_id: String,
        content: String,
    },
}

impl ChatMessage {
    /// Create a text-only assistant message (convenience for the common case).
    pub fn assistant(text: impl Into<String>) -> Self {
        Self::Assistant {
            content: Some(text.into()),
            tool_calls: vec![],
        }
    }

    /// Create an assistant message with tool calls.
    pub fn assistant_with_tool_calls(
        content: Option<String>,
        tool_calls: Vec<ToolCallResponse>,
    ) -> Self {
        Self::Assistant {
            content,
            tool_calls,
        }
    }

    /// Create a text-only user message (convenience for the common case).
    pub fn user_text(text: &str) -> Self {
        Self::User(vec![ContentPart::Text {
            text: text.to_string(),
        }])
    }

    /// Extract the text content from a user message, ignoring images.
    pub fn user_text_content(&self) -> Option<String> {
        match self {
            Self::User(parts) => {
                let texts: Vec<&str> = parts
                    .iter()
                    .filter_map(|p| match p {
                        ContentPart::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect();
                if texts.is_empty() {
                    None
                } else {
                    Some(texts.join("\n"))
                }
            }
            _ => None,
        }
    }

    /// Check if this user message contains any images.
    pub fn has_images(&self) -> bool {
        match self {
            Self::User(parts) => parts
                .iter()
                .any(|p| matches!(p, ContentPart::Image { .. } | ContentPart::ImageUrl { .. })),
            _ => false,
        }
    }

    /// Extract image data from a user message.
    pub fn images(&self) -> Vec<&ContentPart> {
        match self {
            Self::User(parts) => parts
                .iter()
                .filter(|p| matches!(p, ContentPart::Image { .. } | ContentPart::ImageUrl { .. }))
                .collect(),
            _ => vec![],
        }
    }
}

// ── Generation Request/Response ───────────────────────────────────────

/// A request to generate text from a loaded model.
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub messages: Vec<ChatMessage>,
    pub max_tokens: u32,
    pub sampling: SamplingParams,
    /// JSON Schema for structured output (forces conforming JSON).
    pub output_schema: Option<serde_json::Value>,
    pub tools: Vec<ToolDefinition>,
    pub tool_choice: ToolChoice,
    pub stop_sequences: Vec<String>,
    /// Raw GBNF grammar override.
    pub grammar: Option<String>,
    /// Model name or path hint for chat template detection (e.g., "llama-3.1-8b").
    pub model_name: Option<String>,
}

impl Default for GenerateRequest {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            max_tokens: 2048,
            sampling: SamplingParams::default(),
            output_schema: None,
            tools: Vec::new(),
            tool_choice: ToolChoice::Auto,
            stop_sequences: Vec::new(),
            grammar: None,
            model_name: None,
        }
    }
}

/// Sampling parameters for token generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub repeat_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }
}

// ── Tool Calling ──────────────────────────────────────────────────────

/// Definition of a tool the model can call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

/// Controls whether/how the model should use tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolChoice {
    Auto,
    Required,
    None,
    Specific(String),
}

/// A tool call produced by the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallResponse {
    pub tool_name: String,
    pub arguments: serde_json::Value,
    pub call_id: String,
}

// ── Generation Response ───────────────────────────────────────────────

/// Response from a non-streaming generation request.
#[derive(Debug, Clone)]
pub enum GenerateResponse {
    Text(String),
    ToolCall(ToolCallResponse),
    Mixed(Vec<ContentBlock>),
}

/// A block within a mixed response.
#[derive(Debug, Clone)]
pub enum ContentBlock {
    Text(String),
    ToolCall(ToolCallResponse),
}

// ── Streaming Token ───────────────────────────────────────────────────

/// A single token emitted during streaming generation.
#[derive(Debug, Clone)]
pub struct Token {
    pub text: String,
    pub id: u32,
    pub logprob: Option<f32>,
    pub is_special: bool,
}

// ── Model Info ────────────────────────────────────────────────────────

/// Summary information about a loaded model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub handle_id: u64,
    pub model_path: String,
    pub engine: InferenceEngine,
    pub context_length: u32,
    pub gpu_layers: Option<u32>,
}

// ── Pipeline Parallelism ─────────────────────────────────────────────

/// Configuration for a pipeline stage (partial model execution).
#[derive(Debug, Clone)]
pub struct PipelineStageConfig {
    /// Layer range this node is responsible for.
    pub layer_range: std::ops::Range<u32>,
    /// Total layers in the model.
    pub total_layers: u32,
}

impl PipelineStageConfig {
    pub fn is_first_stage(&self) -> bool {
        self.layer_range.start == 0
    }
    pub fn is_last_stage(&self) -> bool {
        self.layer_range.end == self.total_layers
    }
}

/// Serializable activation tensor for inter-node transfer.
#[derive(Debug, Clone)]
pub struct ActivationData {
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub dtype: ActivationDtype,
}

/// Data type of activation tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationDtype {
    F32,
    F16,
    BF16,
}

impl ActivationDtype {
    pub fn byte_size(&self) -> usize {
        match self {
            ActivationDtype::F32 => 4,
            ActivationDtype::F16 | ActivationDtype::BF16 => 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_handle_unique_ids() {
        let h1 = ModelHandle::new(PathBuf::from("/a"), InferenceEngine::LlamaCpp);
        let h2 = ModelHandle::new(PathBuf::from("/b"), InferenceEngine::Candle);
        assert_ne!(h1.id, h2.id);
    }

    #[test]
    fn test_load_config_defaults() {
        let cfg = LoadConfig::default();
        assert_eq!(cfg.context_length, 4096);
        assert_eq!(cfg.batch_size, 512);
        assert!(cfg.use_mmap);
        assert!(!cfg.use_mlock);
        assert!(cfg.offload.auto);
    }

    #[test]
    fn test_sampling_params_defaults() {
        let s = SamplingParams::default();
        assert!((s.temperature - 0.7).abs() < f32::EPSILON);
        assert!((s.top_p - 0.9).abs() < f32::EPSILON);
        assert_eq!(s.top_k, 40);
    }

    #[test]
    fn test_chat_message_serde() {
        let msg = ChatMessage::user_text("Hello");
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("user"));
        assert!(json.contains("Hello"));

        let round_trip: ChatMessage = serde_json::from_str(&json).unwrap();
        match &round_trip {
            ChatMessage::User(_) => {
                let text = round_trip.user_text_content().unwrap();
                assert_eq!(text, "Hello");
            }
            _ => panic!("Expected User message"),
        }
    }

    #[test]
    fn test_generate_request_defaults() {
        let req = GenerateRequest::default();
        assert_eq!(req.max_tokens, 2048);
        assert!(req.messages.is_empty());
        assert!(req.tools.is_empty());
        assert!(req.output_schema.is_none());
    }
}
