use std::path::Path;

use async_trait::async_trait;
use tracing::{info, warn};

use crate::error::{InferenceError, Result};
use crate::types::*;
use hivebear_core::types::{InferenceEngine, ModelFormat};

use super::{InferenceBackend, TokenStream};

/// ONNX Runtime inference backend.
///
/// Uses the `ort` crate (Rust bindings for ONNX Runtime) to load and
/// execute ONNX-format models. Supports CPU execution by default, with
/// optional CUDA/CoreML/TensorRT execution providers when available.
///
/// Note: ONNX Runtime requires the shared library (`libonnxruntime`) to
/// be available at runtime. Set `ORT_DYLIB_PATH` to the library location.
/// If the library is not found, `is_available()` returns false.
pub struct OnnxBackend {
    available: bool,
}

impl OnnxBackend {
    pub fn new() -> Self {
        // Try to initialize ONNX Runtime
        let available = ort::init().commit();
        if available {
            info!("ONNX Runtime initialized successfully");
        } else {
            warn!(
                "ONNX Runtime not available. Set ORT_DYLIB_PATH to the \
                 libonnxruntime shared library path."
            );
        }
        Self { available }
    }
}

impl Default for OnnxBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl InferenceBackend for OnnxBackend {
    fn engine_id(&self) -> InferenceEngine {
        InferenceEngine::OnnxRuntime
    }

    fn name(&self) -> &str {
        "ONNX Runtime"
    }

    fn supported_formats(&self) -> &[ModelFormat] {
        &[ModelFormat::Onnx]
    }

    fn is_available(&self) -> bool {
        self.available
    }

    fn supports_grammar(&self) -> bool {
        false
    }

    async fn load_model(&self, path: &Path, _config: &LoadConfig) -> Result<ModelHandle> {
        if !self.available {
            return Err(InferenceError::EngineNotCompiled {
                engine: "ONNX Runtime — set ORT_DYLIB_PATH to libonnxruntime".into(),
            });
        }

        if !path.exists() {
            return Err(InferenceError::ModelNotFound(path.display().to_string()));
        }

        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if ext != "onnx" {
            return Err(InferenceError::UnsupportedFormat(format!(
                "Expected .onnx file, got .{ext}"
            )));
        }

        // Validate the model can be loaded
        match ort::session::Session::builder().and_then(|b| b.commit_from_file(path)) {
            Ok(_session) => {
                info!("ONNX model loaded successfully: {}", path.display());
            }
            Err(e) => {
                return Err(InferenceError::LoadError(format!(
                    "Failed to load ONNX model: {e}"
                )));
            }
        }

        Ok(ModelHandle::new(
            path.to_path_buf(),
            InferenceEngine::OnnxRuntime,
        ))
    }

    async fn generate(
        &self,
        _handle: &ModelHandle,
        _req: &GenerateRequest,
    ) -> Result<GenerateResponse> {
        // ONNX model execution requires model-specific input/output handling.
        // A full implementation would tokenize, create tensors, run session,
        // and decode outputs. This varies by model architecture.
        Err(InferenceError::GenerationError(
            "ONNX inference requires a model-specific adapter. \
             Use `hivebear convert <model> --to gguf` to convert to GGUF format \
             for full inference support via llama.cpp or Candle."
                .into(),
        ))
    }

    fn stream(&self, _handle: &ModelHandle, _req: &GenerateRequest) -> TokenStream {
        let (tx, rx) = tokio::sync::mpsc::channel(1);
        tokio::spawn(async move {
            let _ = tx
                .send(Err(InferenceError::GenerationError(
                    "ONNX streaming not yet implemented. \
                     Convert to GGUF for streaming inference."
                        .into(),
                )))
                .await;
        });
        Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
    }

    async fn unload(&self, _handle: &ModelHandle) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_backend_metadata() {
        let backend = OnnxBackend { available: false };
        assert_eq!(backend.engine_id(), InferenceEngine::OnnxRuntime);
        assert_eq!(backend.name(), "ONNX Runtime");
        assert!(!backend.supports_grammar());
        assert!(backend.supported_formats().contains(&ModelFormat::Onnx));
    }

    #[tokio::test]
    async fn test_onnx_unavailable_load_error() {
        let backend = OnnxBackend { available: false };
        let result = backend
            .load_model(Path::new("/tmp/model.onnx"), &LoadConfig::default())
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("ORT_DYLIB_PATH"));
    }
}
