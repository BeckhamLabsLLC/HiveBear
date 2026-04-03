use async_trait::async_trait;
use std::collections::HashMap;
use std::io::Cursor;
use std::path::Path;
use std::sync::Mutex;

use candle_core::{quantized::gguf_file, Device, IndexOp, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;

use crate::chat_template;
use crate::error::{InferenceError, Result};
use crate::types::*;
use hivebear_core::types::{InferenceEngine, ModelFormat};

use super::{InferenceBackend, TokenStream};

/// Internal state for a loaded Candle model in WASM.
struct LoadedModel {
    weights: ModelWeights,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
}

/// Pure Rust inference backend for WASM using HuggingFace Candle.
///
/// Adapted from `CandleBackend` for single-threaded WASM:
/// - No `tokio::task::spawn_blocking` (no threads in WASM)
/// - Models loaded from in-memory bytes via `load_model_from_bytes`
/// - RNG uses `getrandom` instead of `SystemTime`
pub struct CandleWasmBackend {
    loaded_models: Mutex<HashMap<u64, LoadedModel>>,
}

impl CandleWasmBackend {
    pub fn new() -> Self {
        Self {
            loaded_models: Mutex::new(HashMap::new()),
        }
    }

    /// Load a model from in-memory bytes (the primary loading path for WASM).
    ///
    /// In the browser, models are fetched via HTTP into an ArrayBuffer,
    /// then passed to this function as `&[u8]`.
    pub fn load_model_from_bytes(
        &self,
        model_bytes: &[u8],
        tokenizer_bytes: &[u8],
        model_name: &str,
    ) -> Result<ModelHandle> {
        let device = Device::Cpu;

        let mut cursor = Cursor::new(model_bytes);
        let content = gguf_file::Content::read(&mut cursor)
            .map_err(|e| InferenceError::LoadError(format!("Failed to read GGUF content: {e}")))?;

        let weights = ModelWeights::from_gguf(content, &mut cursor, &device)
            .map_err(|e| InferenceError::LoadError(format!("Failed to load model weights: {e}")))?;

        let tokenizer = tokenizers::Tokenizer::from_bytes(tokenizer_bytes)
            .map_err(|e| InferenceError::LoadError(format!("Failed to load tokenizer: {e}")))?;

        tracing::info!(model_name, "Model loaded via Candle WASM");

        let handle = ModelHandle::new(
            std::path::PathBuf::from(format!("memory://{model_name}")),
            InferenceEngine::Candle,
        );

        let loaded = LoadedModel {
            weights,
            tokenizer,
            device,
        };

        self.loaded_models
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(handle.id, loaded);

        Ok(handle)
    }

    fn get_loaded_mut(&self) -> std::sync::MutexGuard<'_, HashMap<u64, LoadedModel>> {
        self.loaded_models.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Default for CandleWasmBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait(?Send)]
impl InferenceBackend for CandleWasmBackend {
    fn engine_id(&self) -> InferenceEngine {
        InferenceEngine::Candle
    }

    fn name(&self) -> &str {
        "Candle (WASM)"
    }

    fn supported_formats(&self) -> &[ModelFormat] {
        &[ModelFormat::Gguf]
    }

    fn is_available(&self) -> bool {
        true
    }

    fn supports_grammar(&self) -> bool {
        false
    }

    async fn load_model(&self, path: &Path, _config: &LoadConfig) -> Result<ModelHandle> {
        // In WASM, load_model via path is not supported.
        // Use load_model_from_bytes() instead.
        Err(InferenceError::LoadError(format!(
            "Filesystem loading not available in WASM. \
             Use load_model_from_bytes() instead. Path: {}",
            path.display()
        )))
    }

    async fn generate(
        &self,
        handle: &ModelHandle,
        req: &GenerateRequest,
    ) -> Result<GenerateResponse> {
        let text = {
            let mut models = self.get_loaded_mut();
            let loaded = models
                .get_mut(&handle.id)
                .ok_or(InferenceError::InvalidHandle)?;
            generate_blocking(loaded, req)?
        };
        Ok(GenerateResponse::Text(text))
    }

    fn stream(&self, handle: &ModelHandle, req: &GenerateRequest) -> TokenStream {
        // In WASM (single-threaded), we collect all tokens synchronously
        // and wrap them in a stream. True streaming with Web Workers
        // can be added as a future enhancement.
        let tokens: Vec<Result<Token>> = {
            let mut models = self.get_loaded_mut();
            match models.get_mut(&handle.id) {
                Some(loaded) => collect_tokens_blocking(loaded, req),
                None => vec![Err(InferenceError::InvalidHandle)],
            }
        };

        Box::pin(futures::stream::iter(tokens))
    }

    async fn unload(&self, handle: &ModelHandle) -> Result<()> {
        self.loaded_models
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&handle.id);
        Ok(())
    }
}

/// Build the prompt string from chat messages using the correct per-model chat template.
fn build_prompt(req: &GenerateRequest) -> String {
    let model_name = req.model_name.as_deref().unwrap_or("");
    let format = chat_template::detect_template(model_name);
    chat_template::render(format, &req.messages, &req.tools)
}

/// Sample the next token from logits.
fn sample_token(logits: &Tensor, temperature: f32, top_p: f32) -> Result<u32> {
    let logits = logits
        .squeeze(0)
        .map_err(|e| InferenceError::GenerationError(format!("Squeeze failed: {e}")))?;

    let logits = if temperature > 0.0 {
        let logits = (&logits / temperature as f64)
            .map_err(|e| InferenceError::GenerationError(format!("Temp scaling failed: {e}")))?;
        let exp = logits
            .exp()
            .map_err(|e| InferenceError::GenerationError(format!("Exp failed: {e}")))?;
        let sum = exp
            .sum_all()
            .map_err(|e| InferenceError::GenerationError(format!("Sum failed: {e}")))?;
        let probs = exp
            .broadcast_div(&sum)
            .map_err(|e| InferenceError::GenerationError(format!("Div failed: {e}")))?;

        let probs_vec: Vec<f32> = probs
            .to_vec1()
            .map_err(|e| InferenceError::GenerationError(format!("To vec failed: {e}")))?;

        let mut indexed: Vec<(usize, f32)> = probs_vec.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumsum = 0.0f32;
        let mut candidates: Vec<(usize, f32)> = Vec::new();
        for (idx, prob) in indexed {
            cumsum += prob;
            candidates.push((idx, prob));
            if cumsum >= top_p {
                break;
            }
        }

        let total: f32 = candidates.iter().map(|(_, p)| p).sum();
        let mut rng_val: f32 = rand_f32() * total;
        for (idx, prob) in &candidates {
            rng_val -= prob;
            if rng_val <= 0.0 {
                return Ok(*idx as u32);
            }
        }
        return Ok(candidates[0].0 as u32);
    } else {
        logits
    };

    let token_id = logits
        .argmax(0)
        .map_err(|e| InferenceError::GenerationError(format!("Argmax failed: {e}")))?
        .to_scalar::<u32>()
        .map_err(|e| InferenceError::GenerationError(format!("Scalar conversion failed: {e}")))?;
    Ok(token_id)
}

/// Random float [0, 1) using getrandom (works in WASM).
fn rand_f32() -> f32 {
    let mut bytes = [0u8; 4];
    getrandom::getrandom(&mut bytes).unwrap_or_default();
    let val = u32::from_le_bytes(bytes);
    (val as f32) / (u32::MAX as f32)
}

/// Blocking generation returning the complete text.
fn generate_blocking(loaded: &mut LoadedModel, req: &GenerateRequest) -> Result<String> {
    let prompt = build_prompt(req);
    let encoding = loaded
        .tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| InferenceError::GenerationError(format!("Tokenization failed: {e}")))?;

    let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
    let input = Tensor::new(prompt_tokens.as_slice(), &loaded.device)
        .map_err(|e| InferenceError::GenerationError(format!("Tensor creation failed: {e}")))?
        .unsqueeze(0)
        .map_err(|e| InferenceError::GenerationError(format!("Unsqueeze failed: {e}")))?;

    let logits = loaded
        .weights
        .forward(&input, 0)
        .map_err(|e| InferenceError::GenerationError(format!("Forward pass failed: {e}")))?;

    let last_logits = logits
        .i((.., logits.dim(1).unwrap_or(1) - 1, ..))
        .map_err(|e| InferenceError::GenerationError(format!("Index failed: {e}")))?;

    let mut next_token = sample_token(&last_logits, req.sampling.temperature, req.sampling.top_p)?;
    let mut output = String::new();
    let seq_len = prompt_tokens.len();

    let eos_token = loaded
        .tokenizer
        .token_to_id("</s>")
        .or_else(|| loaded.tokenizer.token_to_id("<|endoftext|>"))
        .or_else(|| loaded.tokenizer.token_to_id("<|end|>"))
        .unwrap_or(u32::MAX);

    for i in 0..req.max_tokens {
        if next_token == eos_token {
            break;
        }

        if let Ok(text) = loaded.tokenizer.decode(&[next_token], true) {
            output.push_str(&text);
        }

        if req.stop_sequences.iter().any(|stop| output.ends_with(stop)) {
            for stop in &req.stop_sequences {
                if output.ends_with(stop) {
                    output.truncate(output.len() - stop.len());
                    break;
                }
            }
            break;
        }

        let input = Tensor::new(&[next_token], &loaded.device)
            .map_err(|e| InferenceError::GenerationError(format!("Tensor failed: {e}")))?
            .unsqueeze(0)
            .map_err(|e| InferenceError::GenerationError(format!("Unsqueeze failed: {e}")))?;

        let logits = loaded
            .weights
            .forward(&input, seq_len + i as usize)
            .map_err(|e| InferenceError::GenerationError(format!("Forward failed: {e}")))?;

        let last_logits = logits
            .i((.., logits.dim(1).unwrap_or(1) - 1, ..))
            .map_err(|e| InferenceError::GenerationError(format!("Index failed: {e}")))?;

        next_token = sample_token(&last_logits, req.sampling.temperature, req.sampling.top_p)?;
    }

    Ok(output)
}

/// Collect all tokens synchronously into a Vec for the stream() method.
fn collect_tokens_blocking(loaded: &mut LoadedModel, req: &GenerateRequest) -> Vec<Result<Token>> {
    let prompt = match build_and_encode(loaded, req) {
        Ok(v) => v,
        Err(e) => return vec![Err(e)],
    };

    let (prompt_tokens, input) = prompt;
    let mut tokens = Vec::new();

    let logits = match loaded.weights.forward(&input, 0) {
        Ok(l) => l,
        Err(e) => {
            return vec![Err(InferenceError::GenerationError(format!(
                "Forward pass failed: {e}"
            )))]
        }
    };

    let last_logits = match logits.i((.., logits.dim(1).unwrap_or(1) - 1, ..)) {
        Ok(l) => l,
        Err(e) => {
            return vec![Err(InferenceError::GenerationError(format!(
                "Index failed: {e}"
            )))]
        }
    };

    let mut next_token =
        match sample_token(&last_logits, req.sampling.temperature, req.sampling.top_p) {
            Ok(t) => t,
            Err(e) => return vec![Err(e)],
        };

    let seq_len = prompt_tokens.len();
    let mut accumulated = String::new();

    let eos_token = loaded
        .tokenizer
        .token_to_id("</s>")
        .or_else(|| loaded.tokenizer.token_to_id("<|endoftext|>"))
        .or_else(|| loaded.tokenizer.token_to_id("<|end|>"))
        .unwrap_or(u32::MAX);

    for i in 0..req.max_tokens {
        if next_token == eos_token {
            break;
        }

        let piece = loaded
            .tokenizer
            .decode(&[next_token], true)
            .unwrap_or_default();

        accumulated.push_str(&piece);

        if req
            .stop_sequences
            .iter()
            .any(|stop| accumulated.ends_with(stop))
        {
            break;
        }

        tokens.push(Ok(Token {
            text: piece,
            id: next_token,
            logprob: None,
            is_special: false,
        }));

        let input = match Tensor::new(&[next_token], &loaded.device) {
            Ok(t) => match t.unsqueeze(0) {
                Ok(t) => t,
                Err(e) => {
                    tokens.push(Err(InferenceError::GenerationError(format!(
                        "Unsqueeze failed: {e}"
                    ))));
                    break;
                }
            },
            Err(e) => {
                tokens.push(Err(InferenceError::GenerationError(format!(
                    "Tensor failed: {e}"
                ))));
                break;
            }
        };

        let logits = match loaded.weights.forward(&input, seq_len + i as usize) {
            Ok(l) => l,
            Err(e) => {
                tokens.push(Err(InferenceError::GenerationError(format!(
                    "Forward failed: {e}"
                ))));
                break;
            }
        };

        let last_logits = match logits.i((.., logits.dim(1).unwrap_or(1) - 1, ..)) {
            Ok(l) => l,
            Err(e) => {
                tokens.push(Err(InferenceError::GenerationError(format!(
                    "Index failed: {e}"
                ))));
                break;
            }
        };

        next_token = match sample_token(&last_logits, req.sampling.temperature, req.sampling.top_p)
        {
            Ok(t) => t,
            Err(e) => {
                tokens.push(Err(e));
                break;
            }
        };
    }

    tokens
}

/// Helper to build prompt and encode into tensor.
fn build_and_encode(loaded: &LoadedModel, req: &GenerateRequest) -> Result<(Vec<u32>, Tensor)> {
    let prompt = build_prompt(req);
    let encoding = loaded
        .tokenizer
        .encode(prompt.as_str(), true)
        .map_err(|e| InferenceError::GenerationError(format!("Tokenization failed: {e}")))?;

    let prompt_tokens: Vec<u32> = encoding.get_ids().to_vec();
    let input = Tensor::new(prompt_tokens.as_slice(), &loaded.device)
        .map_err(|e| InferenceError::GenerationError(format!("Tensor creation failed: {e}")))?
        .unsqueeze(0)
        .map_err(|e| InferenceError::GenerationError(format!("Unsqueeze failed: {e}")))?;

    Ok((prompt_tokens, input))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_wasm_backend_metadata() {
        let backend = CandleWasmBackend::new();
        assert_eq!(backend.engine_id(), InferenceEngine::Candle);
        assert_eq!(backend.name(), "Candle (WASM)");
        assert!(backend.is_available());
        assert!(!backend.supports_grammar());
        assert!(backend.supported_formats().contains(&ModelFormat::Gguf));
    }

    #[test]
    fn test_build_prompt() {
        let req = GenerateRequest {
            messages: vec![
                ChatMessage::System("You are a helpful AI.".into()),
                ChatMessage::user_text("What is Rust?"),
            ],
            ..Default::default()
        };
        let prompt = build_prompt(&req);
        assert!(prompt.contains("You are a helpful AI."));
        assert!(prompt.contains("What is Rust?"));
        assert!(prompt.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn test_rand_f32_range() {
        for _ in 0..100 {
            let val = rand_f32();
            assert!((0.0..1.0).contains(&val), "rand_f32 out of range: {val}");
        }
    }
}
