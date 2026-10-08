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

    /// Generate with a callback per token, for callers that can surface tokens
    /// as they arrive (the browser playground, running in a Web Worker).
    /// `stream()` can't do this on wasm32: with no threads it has to finish
    /// generating before the stream yields anything.
    pub fn generate_streaming(
        &self,
        handle: &ModelHandle,
        req: &GenerateRequest,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<String> {
        let mut models = self.get_loaded_mut();
        let loaded = models
            .get_mut(&handle.id)
            .ok_or(InferenceError::InvalidHandle)?;
        let mut full_text = String::new();
        let mut failure = None;
        for_each_token(loaded, req, &mut |t| match t {
            Ok(token) => {
                full_text.push_str(&token.text);
                on_token(&token.text);
            }
            Err(e) => failure = Some(e),
        });
        match failure {
            Some(e) => Err(e),
            None => Ok(full_text),
        }
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

/// Logits for the last position. candle's quantized_llama already returns only
/// the last position (`[batch, vocab]`); other models return
/// `[batch, seq, vocab]`. Indexing the 2-D case as 3-D panicked on the first
/// token, so browser generation never produced output.
fn last_position_logits(logits: &Tensor) -> Result<Tensor> {
    match logits.rank() {
        2 => Ok(logits.clone()),
        3 => {
            let last = logits
                .dim(1)
                .map_err(|e| InferenceError::GenerationError(format!("Index failed: {e}")))?
                .saturating_sub(1);
            logits
                .i((.., last, ..))
                .map_err(|e| InferenceError::GenerationError(format!("Index failed: {e}")))
        }
        r => Err(InferenceError::GenerationError(format!(
            "Unexpected logits rank {r}"
        ))),
    }
}

/// Every end-of-turn / end-of-text token the tokenizer knows. Chat models stop
/// on their template's end-of-turn token (`<|im_end|>` for ChatML models such
/// as SmolLM2, `<|eot_id|>` for Llama 3), not only on the base EOS token.
fn stop_token_ids(tokenizer: &tokenizers::Tokenizer) -> Vec<u32> {
    [
        "</s>",
        "<|endoftext|>",
        "<|end|>",
        "<|im_end|>",
        "<|eot_id|>",
        "<|end_of_text|>",
    ]
    .iter()
    .filter_map(|t| tokenizer.token_to_id(t))
    .collect()
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
    let mut output = String::new();
    let mut failure = None;
    for_each_token(loaded, req, &mut |t| match t {
        Ok(token) => output.push_str(&token.text),
        Err(e) => failure = Some(e),
    });
    match failure {
        Some(e) => Err(e),
        None => Ok(output),
    }
}

/// Collect all tokens synchronously into a Vec for the stream() method.
fn collect_tokens_blocking(loaded: &mut LoadedModel, req: &GenerateRequest) -> Vec<Result<Token>> {
    let mut tokens = Vec::new();
    for_each_token(loaded, req, &mut |t| tokens.push(t));
    tokens
}

/// Run generation, handing each token (or the error that ended it) to `emit`
/// as soon as it is sampled. This is what lets the browser show tokens while
/// the model is still generating instead of all at once at the end.
fn for_each_token(
    loaded: &mut LoadedModel,
    req: &GenerateRequest,
    emit: &mut dyn FnMut(Result<Token>),
) {
    let (prompt_tokens, input) = match build_and_encode(loaded, req) {
        Ok(v) => v,
        Err(e) => return emit(Err(e)),
    };

    let first = loaded
        .weights
        .forward(&input, 0)
        .map_err(|e| InferenceError::GenerationError(format!("Forward pass failed: {e}")))
        .and_then(|logits| last_position_logits(&logits))
        .and_then(|logits| sample_token(&logits, req.sampling.temperature, req.sampling.top_p));
    let mut next_token = match first {
        Ok(t) => t,
        Err(e) => return emit(Err(e)),
    };

    let seq_len = prompt_tokens.len();
    let mut accumulated = String::new();
    let stop_tokens = stop_token_ids(&loaded.tokenizer);

    for i in 0..req.max_tokens {
        if stop_tokens.contains(&next_token) {
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

        emit(Ok(Token {
            text: piece,
            id: next_token,
            logprob: None,
            is_special: false,
        }));

        let step = Tensor::new(&[next_token], &loaded.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| InferenceError::GenerationError(format!("Tensor failed: {e}")))
            .and_then(|input| {
                loaded
                    .weights
                    .forward(&input, seq_len + i as usize)
                    .map_err(|e| InferenceError::GenerationError(format!("Forward failed: {e}")))
            })
            .and_then(|logits| last_position_logits(&logits))
            .and_then(|logits| sample_token(&logits, req.sampling.temperature, req.sampling.top_p));

        next_token = match step {
            Ok(t) => t,
            Err(e) => return emit(Err(e)),
        };
    }
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
    fn test_last_position_logits_accepts_both_shapes() {
        let device = Device::Cpu;
        // quantized_llama returns [batch, vocab] for the last position only.
        let two_d = Tensor::new(&[[0.1f32, 0.2, 0.3]], &device).unwrap();
        let out = last_position_logits(&two_d).unwrap();
        assert_eq!(out.dims(), &[1, 3]);

        // Full-sequence models return [batch, seq, vocab]; take the last row.
        let three_d = Tensor::new(&[[[0.0f32, 0.0], [1.0, 2.0]]], &device).unwrap();
        let out = last_position_logits(&three_d).unwrap();
        assert_eq!(out.to_vec2::<f32>().unwrap(), vec![vec![1.0, 2.0]]);
    }

    #[test]
    fn test_rand_f32_range() {
        for _ in 0..100 {
            let val = rand_f32();
            assert!((0.0..1.0).contains(&val), "rand_f32 out of range: {val}");
        }
    }
}
