pub mod benchmark;
pub mod chat_template;
pub mod engine;
pub mod error;
pub mod loader;
pub mod offload;
pub mod orchestrator;
pub mod selector;
pub mod structured_output;
pub mod tool_calling;
pub mod types;

// Re-export key types for convenience
pub use engine::{EngineRegistry, InferenceBackend};
pub use error::{InferenceError, Result};
pub use orchestrator::Orchestrator;
pub use types::{
    ChatMessage, ContentBlock, GenerateRequest, GenerateResponse, LoadConfig, ModelHandle,
    ModelInfo, OffloadConfig, SamplingParams, Token, ToolCallResponse, ToolChoice, ToolDefinition,
};
