pub mod dummy;

#[cfg(feature = "llamacpp")]
pub mod llamacpp;

#[cfg(all(any(feature = "candle", feature = "wasm"), not(target_arch = "wasm32")))]
pub mod candle_backend;

#[cfg(all(any(feature = "candle", feature = "wasm"), not(target_arch = "wasm32")))]
pub(crate) mod candle_pipeline;

#[cfg(all(any(feature = "candle", feature = "wasm"), target_arch = "wasm32"))]
pub mod candle_wasm;

#[cfg(feature = "onnx")]
pub mod onnx;

#[cfg(feature = "cloud")]
pub mod cloud;

#[cfg(feature = "mlx")]
pub mod mlx;

use async_trait::async_trait;
use futures::Stream;
use std::path::Path;
use std::pin::Pin;

use crate::error::Result;
use crate::types::{
    ActivationData, GenerateRequest, GenerateResponse, LoadConfig, ModelHandle,
    PipelineStageConfig, Token,
};
use hivebear_core::types::{InferenceEngine, ModelFormat};

/// Token stream type — `Send` on native, not on WASM (single-threaded).
#[cfg(not(target_arch = "wasm32"))]
pub type TokenStream = Pin<Box<dyn Stream<Item = Result<Token>> + Send>>;

#[cfg(target_arch = "wasm32")]
pub type TokenStream = Pin<Box<dyn Stream<Item = Result<Token>>>>;

/// Trait implemented by each inference engine backend.
///
/// `InferenceBackend` is the behavioral counterpart to the `InferenceEngine` enum
/// (which is a data-level discriminator in hivebear-core). The `engine_id()` method
/// bridges the two.
#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
pub trait InferenceBackend: Send + Sync {
    /// Which engine variant this backend represents.
    fn engine_id(&self) -> InferenceEngine;

    /// Human-readable name.
    fn name(&self) -> &str;

    /// Model formats this backend can load.
    fn supported_formats(&self) -> &[ModelFormat];

    /// Whether this backend is available on the current platform
    /// (compiled in, libraries found, etc.)
    fn is_available(&self) -> bool;

    /// Whether this backend supports GBNF grammar-constrained decoding.
    fn supports_grammar(&self) -> bool {
        false
    }

    /// Load a model from a file path.
    async fn load_model(&self, path: &Path, config: &LoadConfig) -> Result<ModelHandle>;

    /// Non-streaming generation.
    async fn generate(
        &self,
        handle: &ModelHandle,
        req: &GenerateRequest,
    ) -> Result<GenerateResponse>;

    /// Streaming generation returning a token stream.
    fn stream(&self, handle: &ModelHandle, req: &GenerateRequest) -> TokenStream;

    /// Unload a model, freeing resources.
    async fn unload(&self, handle: &ModelHandle) -> Result<()>;

    /// Run a partial forward pass through assigned layers only.
    /// Returns the intermediate activation tensor for forwarding to the next stage,
    /// or final logits if this is the last stage.
    async fn forward_partial(
        &self,
        _handle: &ModelHandle,
        _input: &ActivationData,
        _stage: &PipelineStageConfig,
        _index_pos: usize,
    ) -> Result<ActivationData> {
        Err(crate::error::InferenceError::GenerationError(
            "Pipeline parallelism not supported by this backend".into(),
        ))
    }

    /// Tokenize input and run the embedding layer, returning the initial activation.
    /// Only called on the first pipeline stage.
    async fn embed_tokens(
        &self,
        _handle: &ModelHandle,
        _token_ids: &[u32],
    ) -> Result<ActivationData> {
        Err(crate::error::InferenceError::GenerationError(
            "Pipeline parallelism not supported by this backend".into(),
        ))
    }

    /// Sample a token from logits returned by the final stage.
    async fn sample_from_logits(
        &self,
        _handle: &ModelHandle,
        _logits: &ActivationData,
        _temperature: f32,
        _top_p: f32,
    ) -> Result<(u32, String)> {
        Err(crate::error::InferenceError::GenerationError(
            "Pipeline parallelism not supported by this backend".into(),
        ))
    }
}

/// Registry of all compiled-in inference backends.
pub struct EngineRegistry {
    backends: Vec<Box<dyn InferenceBackend>>,
}

impl EngineRegistry {
    /// Create a new registry with all compiled-in backends.
    /// API keys are resolved from environment variables only.
    pub fn new() -> Self {
        Self::new_with_cloud_config(None)
    }

    /// Create a registry with optional cloud config for API key resolution.
    pub fn new_with_cloud_config(
        #[allow(unused)] cloud_config: Option<&hivebear_core::config::CloudConfig>,
    ) -> Self {
        #[allow(unused_mut)]
        let mut backends: Vec<Box<dyn InferenceBackend>> = vec![
            #[cfg(feature = "llamacpp")]
            Box::new(llamacpp::LlamaCppBackend::new()),
            #[cfg(all(any(feature = "candle", feature = "wasm"), not(target_arch = "wasm32")))]
            Box::new(candle_backend::CandleBackend::new()),
            #[cfg(all(any(feature = "candle", feature = "wasm"), target_arch = "wasm32"))]
            Box::new(candle_wasm::CandleWasmBackend::new()),
            #[cfg(feature = "onnx")]
            Box::new(onnx::OnnxBackend::new()),
            // Dummy backend is always available (for testing)
            Box::new(dummy::DummyBackend),
        ];

        #[cfg(feature = "cloud")]
        {
            let cloud_backend = if let Some(cc) = cloud_config {
                cloud::CloudBackend::with_config(cc)
            } else {
                cloud::CloudBackend::new()
            };
            backends.push(Box::new(cloud_backend));
        }

        #[cfg(feature = "mlx")]
        backends.push(Box::new(mlx::MlxBackend::new()));

        Self { backends }
    }

    /// Get all backends that are currently available.
    pub fn available_engines(&self) -> Vec<&dyn InferenceBackend> {
        self.backends
            .iter()
            .filter(|b| b.is_available())
            .map(|b| b.as_ref())
            .collect()
    }

    /// Find a specific backend by engine ID.
    pub fn get(&self, engine: InferenceEngine) -> Option<&dyn InferenceBackend> {
        self.backends
            .iter()
            .find(|b| b.engine_id() == engine && b.is_available())
            .map(|b| b.as_ref())
    }

    /// Find the best available backend for a given model format.
    pub fn find_for_format(&self, format: ModelFormat) -> Option<&dyn InferenceBackend> {
        // Priority order: LlamaCpp > Candle > others
        let priority = [
            InferenceEngine::LlamaCpp,
            InferenceEngine::Candle,
            InferenceEngine::OnnxRuntime,
            InferenceEngine::Mlx,
        ];

        for engine in priority {
            if let Some(backend) = self.backends.iter().find(|b| {
                b.engine_id() == engine
                    && b.is_available()
                    && b.supported_formats().contains(&format)
            }) {
                return Some(backend.as_ref());
            }
        }
        None
    }

    /// Register an external backend at runtime.
    ///
    /// This allows external crates (like `hivebear-mesh`) to add backends
    /// without creating circular dependencies.
    pub fn register(&mut self, backend: Box<dyn InferenceBackend>) {
        self.backends.push(backend);
    }

    /// List all compiled-in backends (available or not).
    pub fn all_backends(&self) -> &[Box<dyn InferenceBackend>] {
        &self.backends
    }
}

impl Default for EngineRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_has_backends() {
        let registry = EngineRegistry::new();
        assert!(
            !registry.all_backends().is_empty(),
            "Registry should have at least the dummy backend"
        );
    }

    #[test]
    fn test_registry_has_dummy() {
        let registry = EngineRegistry::new();
        // The dummy backend should always be present but not "available" for real use
        let all = registry.all_backends();
        assert!(all.iter().any(|b| b.name() == "Dummy"));
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_find_for_format_gguf() {
        let registry = EngineRegistry::new();
        let backend = registry.find_for_format(ModelFormat::Gguf);
        assert!(backend.is_some(), "Should find a backend for GGUF format");
    }
}
