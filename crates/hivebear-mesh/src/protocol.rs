//! Wire protocol types and handler trait for mesh inference.
//!
//! This module defines:
//! - `MeshInferenceMessage`: a high-level envelope that maps to the
//!   `InferenceRequest` / `InferenceToken` / `InferenceComplete` variants
//!   already present in `MeshMessage`.
//! - `MeshInferenceHandler`: a trait that worker nodes implement to serve
//!   inference requests without coupling the mesh crate to a specific backend.

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// High-level inference message envelope
// ---------------------------------------------------------------------------

/// Wire-level message for full-model replication inference.
///
/// These map 1:1 to the corresponding `MeshMessage` variants, but are
/// provided as a standalone enum so that higher-level code can work with
/// inference semantics without importing the full transport protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeshInferenceMessage {
    /// Request inference from a peer.
    InferenceRequest {
        request_id: String,
        model_id: String,
        /// Serialized `Vec<ChatMessage>` as JSON.
        messages_json: String,
        max_tokens: u32,
        temperature: f32,
    },
    /// A generated token from the peer.
    InferenceToken {
        request_id: String,
        text: String,
        is_final: bool,
    },
    /// An error from the peer.
    InferenceError { request_id: String, message: String },
}

// ---------------------------------------------------------------------------
// Inference handler trait (implemented by the CLI / host application)
// ---------------------------------------------------------------------------

/// Trait for handling inference requests on a mesh worker node.
///
/// The mesh crate cannot depend on `hivebear-inference` directly (to avoid
/// circular dependencies), so the host application (e.g. the CLI) implements
/// this trait using its own `Orchestrator`.
///
/// The returned stream yields token text strings. Returning `Err(String)` at
/// any point signals a fatal error to the requesting peer.
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
pub trait MeshInferenceHandler: Send + Sync {
    /// Handle an inference request and return a stream of token strings.
    async fn handle_inference(
        &self,
        model_id: &str,
        messages_json: &str,
        max_tokens: u32,
        temperature: f32,
    ) -> std::result::Result<
        Pin<Box<dyn Stream<Item = std::result::Result<String, String>> + Send>>,
        String,
    >;
}

// ---------------------------------------------------------------------------
// Pipeline-parallel inference handler
// ---------------------------------------------------------------------------

/// Handler for pipeline-parallel inference on a worker node.
/// Processes activation tensors through assigned layers and produces output activations.
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
pub trait MeshPipelineHandler: Send + Sync {
    /// Load only the specified layer range from a model.
    async fn load_layers(
        &self,
        model_source: &str,
        layer_range: std::ops::Range<u32>,
        total_layers: u32,
    ) -> std::result::Result<(), String>;

    /// Run a forward pass through loaded layers.
    /// Returns (output_data, output_shape, output_dtype).
    async fn forward_layers(
        &self,
        activation_data: Vec<u8>,
        shape: Vec<usize>,
        dtype: u8,
        index_pos: usize,
    ) -> std::result::Result<(Vec<u8>, Vec<usize>, u8), String>;

    /// Unload layers and free resources.
    async fn unload_layers(&self) -> std::result::Result<(), String>;
}
