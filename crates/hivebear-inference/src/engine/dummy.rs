use async_trait::async_trait;
use std::path::Path;

use crate::error::{InferenceError, Result};
use crate::types::*;
use hivebear_core::types::{InferenceEngine, ModelFormat};

use super::{InferenceBackend, TokenStream};

/// A no-op backend that returns canned responses. Used for testing the
/// orchestrator and CLI without requiring real models or native dependencies.
pub struct DummyBackend;

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl InferenceBackend for DummyBackend {
    fn engine_id(&self) -> InferenceEngine {
        // Re-use Candle as the "identity" since there's no Dummy variant in the enum
        InferenceEngine::Candle
    }

    fn name(&self) -> &str {
        "Dummy"
    }

    fn supported_formats(&self) -> &[ModelFormat] {
        &[ModelFormat::Gguf, ModelFormat::SafeTensors]
    }

    fn is_available(&self) -> bool {
        // Dummy is always available but should rank last in selection
        true
    }

    async fn load_model(&self, path: &Path, _config: &LoadConfig) -> Result<ModelHandle> {
        if !path.exists() && path.to_string_lossy() != "dummy://test" {
            return Err(InferenceError::ModelNotFound(path.display().to_string()));
        }
        Ok(ModelHandle::new(path.to_path_buf(), self.engine_id()))
    }

    async fn generate(
        &self,
        _handle: &ModelHandle,
        req: &GenerateRequest,
    ) -> Result<GenerateResponse> {
        // Return a canned response based on the last user message
        let user_msg = req
            .messages
            .iter()
            .rev()
            .find_map(|m| match m {
                ChatMessage::User(_) => Some(m.user_text_content().unwrap_or_default()),
                _ => None,
            })
            .unwrap_or_default();

        let response = format!(
            "This is a dummy response to: \"{}\". \
             The dummy backend does not perform real inference.",
            user_msg
        );

        Ok(GenerateResponse::Text(response))
    }

    fn stream(&self, _handle: &ModelHandle, req: &GenerateRequest) -> TokenStream {
        let user_msg = req
            .messages
            .iter()
            .rev()
            .find_map(|m| match m {
                ChatMessage::User(_) => Some(m.user_text_content().unwrap_or_default()),
                _ => None,
            })
            .unwrap_or_else(|| "hello".into());

        let response = format!("Dummy response to: {user_msg}");
        let tokens: Vec<Result<Token>> = response
            .split_whitespace()
            .enumerate()
            .map(|(i, w)| {
                Ok(Token {
                    text: format!("{w} "),
                    id: i as u32,
                    logprob: None,
                    is_special: false,
                })
            })
            .collect();

        #[cfg(not(target_arch = "wasm32"))]
        {
            let (tx, rx) = tokio::sync::mpsc::channel(32);
            tokio::spawn(async move {
                for tok in tokens {
                    if tx.send(tok).await.is_err() {
                        break;
                    }
                }
            });
            Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
        }

        #[cfg(target_arch = "wasm32")]
        {
            Box::pin(futures::stream::iter(tokens))
        }
    }

    async fn unload(&self, _handle: &ModelHandle) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[tokio::test]
    async fn test_dummy_load_model() {
        let backend = DummyBackend;
        let handle = backend
            .load_model(Path::new("dummy://test"), &LoadConfig::default())
            .await
            .unwrap();
        assert_eq!(handle.engine, InferenceEngine::Candle);
    }

    #[tokio::test]
    async fn test_dummy_generate() {
        let backend = DummyBackend;
        let handle = ModelHandle::new(PathBuf::from("dummy://test"), InferenceEngine::Candle);
        let req = GenerateRequest {
            messages: vec![ChatMessage::user_text("What is 2+2?")],
            ..Default::default()
        };

        let response = backend.generate(&handle, &req).await.unwrap();
        match response {
            GenerateResponse::Text(text) => {
                assert!(text.contains("2+2"));
                assert!(text.contains("dummy"));
            }
            _ => panic!("Expected text response"),
        }
    }

    #[tokio::test]
    async fn test_dummy_stream() {
        use futures::StreamExt;

        let backend = DummyBackend;
        let handle = ModelHandle::new(PathBuf::from("dummy://test"), InferenceEngine::Candle);
        let req = GenerateRequest {
            messages: vec![ChatMessage::user_text("Hello")],
            ..Default::default()
        };

        let mut stream = backend.stream(&handle, &req);
        let mut tokens = Vec::new();
        while let Some(result) = stream.next().await {
            tokens.push(result.unwrap());
        }
        assert!(!tokens.is_empty(), "Stream should produce tokens");

        let full_text: String = tokens.iter().map(|t| t.text.as_str()).collect();
        assert!(full_text.contains("Hello"));
    }

    #[test]
    fn test_dummy_metadata() {
        let backend = DummyBackend;
        assert_eq!(backend.name(), "Dummy");
        assert!(backend.is_available());
        assert!(!backend.supports_grammar());
        assert!(backend.supported_formats().contains(&ModelFormat::Gguf));
    }
}
