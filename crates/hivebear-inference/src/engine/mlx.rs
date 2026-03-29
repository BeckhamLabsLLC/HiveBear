//! MLX backend for Apple Silicon Macs.
//!
//! Uses a subprocess wrapper around `mlx-lm` (Python) for inference.
//! Requires: `pip install mlx-lm` on macOS with Apple Silicon.

use async_trait::async_trait;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::engine::{InferenceBackend, TokenStream};
use crate::error::{InferenceError, Result};
use crate::types::*;
use hivebear_core::types::{InferenceEngine, ModelFormat};

static NEXT_HANDLE_ID: AtomicU64 = AtomicU64::new(800_000);

pub struct MlxBackend;

impl MlxBackend {
    pub fn new() -> Self {
        Self
    }

    fn check_mlx_available() -> bool {
        #[cfg(target_os = "macos")]
        {
            if std::env::consts::ARCH != "aarch64" {
                return false;
            }
            std::process::Command::new("python3")
                .args(["-c", "import mlx_lm"])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }

    fn run_mlx_generate(
        model_path: &str,
        prompt: &str,
        max_tokens: u32,
        temperature: f32,
    ) -> std::result::Result<String, String> {
        let script = format!(
            r#"
import sys
from mlx_lm import load, generate
model, tokenizer = load("{}")
response = generate(model, tokenizer, prompt="{}", max_tokens={}, temp={})
sys.stdout.write(response)
"#,
            model_path.replace('\\', "\\\\").replace('"', "\\\""),
            prompt.replace('\\', "\\\\").replace('"', "\\\""),
            max_tokens,
            temperature,
        );

        let output = std::process::Command::new("python3")
            .args(["-c", &script])
            .output()
            .map_err(|e| format!("Failed to run mlx-lm: {e}"))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("mlx-lm error: {stderr}"));
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }
}

#[async_trait]
impl InferenceBackend for MlxBackend {
    fn engine_id(&self) -> InferenceEngine {
        InferenceEngine::Mlx
    }

    fn name(&self) -> &str {
        "MLX"
    }

    fn supported_formats(&self) -> &[ModelFormat] {
        &[ModelFormat::Mlx, ModelFormat::SafeTensors]
    }

    fn is_available(&self) -> bool {
        Self::check_mlx_available()
    }

    fn supports_grammar(&self) -> bool {
        false
    }

    async fn load_model(&self, path: &Path, _config: &LoadConfig) -> Result<ModelHandle> {
        if !self.is_available() {
            return Err(InferenceError::EngineNotCompiled {
                engine: "MLX".into(),
            });
        }

        if !path.exists() {
            return Err(InferenceError::LoadError(format!(
                "Model path does not exist: {}",
                path.display()
            )));
        }

        Ok(ModelHandle {
            id: NEXT_HANDLE_ID.fetch_add(1, Ordering::Relaxed),
            model_path: path.to_path_buf(),
            engine: InferenceEngine::Mlx,
        })
    }

    async fn generate(
        &self,
        handle: &ModelHandle,
        req: &GenerateRequest,
    ) -> Result<GenerateResponse> {
        let prompt = req
            .messages
            .last()
            .and_then(|m| m.user_text_content())
            .unwrap_or_default();

        let model_path = handle.model_path.display().to_string();
        let max_tokens = req.max_tokens;
        let temperature = req.sampling.temperature;

        let text = tokio::task::spawn_blocking(move || {
            Self::run_mlx_generate(&model_path, &prompt, max_tokens, temperature)
        })
        .await
        .map_err(|e| InferenceError::GenerationError(format!("Task join error: {e}")))?
        .map_err(|e| InferenceError::GenerationError(e))?;

        Ok(GenerateResponse::Text(text))
    }

    fn stream(&self, handle: &ModelHandle, req: &GenerateRequest) -> TokenStream {
        let handle = handle.clone();
        let req = req.clone();

        let stream = async_stream::try_stream! {
            let prompt = req
                .messages
                .last()
                .and_then(|m| m.user_text_content())
                .unwrap_or_default();

            let model_path = handle.model_path.display().to_string();
            let max_tokens = req.max_tokens;
            let temperature = req.sampling.temperature;

            let text = tokio::task::spawn_blocking(move || {
                MlxBackend::run_mlx_generate(&model_path, &prompt, max_tokens, temperature)
            })
            .await
            .map_err(|e| InferenceError::GenerationError(format!("Task join error: {e}")))?
            .map_err(|e| InferenceError::GenerationError(e))?;

            for word in text.split_inclusive(' ') {
                yield Token {
                    text: word.to_string(),
                    id: 0,
                    logprob: None,
                    is_special: false,
                };
            }
        };

        Box::pin(stream)
    }

    async fn unload(&self, _handle: &ModelHandle) -> Result<()> {
        Ok(())
    }
}
