use async_trait::async_trait;
use std::collections::HashMap;
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::{Arc, Mutex};

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;

use crate::chat_template;
use crate::error::{InferenceError, Result};
use crate::types::*;
use hivebear_core::types::{InferenceEngine, ModelFormat};

use super::{InferenceBackend, TokenStream};

/// Internal state for a loaded llama.cpp model + context.
struct LoadedModel {
    model: LlamaModel,
    backend: LlamaBackend,
}

// SAFETY: LlamaModel and LlamaBackend are thread-safe in practice —
// llama.cpp uses internal locks. We wrap in Arc<Mutex<>> for Rust's Send/Sync.
unsafe impl Send for LoadedModel {}
unsafe impl Sync for LoadedModel {}

/// llama.cpp inference backend via the `llama-cpp-2` crate.
///
/// Supports GGUF models with GPU offloading, grammar-constrained decoding,
/// and streaming token generation.
pub struct LlamaCppBackend {
    loaded_models: Arc<Mutex<HashMap<u64, Arc<LoadedModel>>>>,
}

impl LlamaCppBackend {
    pub fn new() -> Self {
        Self {
            loaded_models: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl Default for LlamaCppBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl InferenceBackend for LlamaCppBackend {
    fn engine_id(&self) -> InferenceEngine {
        InferenceEngine::LlamaCpp
    }

    fn name(&self) -> &str {
        "llama.cpp"
    }

    fn supported_formats(&self) -> &[ModelFormat] {
        &[ModelFormat::Gguf]
    }

    fn is_available(&self) -> bool {
        true
    }

    fn supports_grammar(&self) -> bool {
        true
    }

    async fn load_model(&self, path: &Path, config: &LoadConfig) -> Result<ModelHandle> {
        let path = path.to_path_buf();
        let config = config.clone();
        let loaded_models = self.loaded_models.clone();

        // Model loading is blocking — run on a blocking thread
        tokio::task::spawn_blocking(move || {
            if !path.exists() {
                return Err(InferenceError::ModelNotFound(path.display().to_string()));
            }

            let backend = LlamaBackend::init().map_err(|e| {
                InferenceError::LoadError(format!("Failed to init llama backend: {e}"))
            })?;

            // Redirect llama.cpp logs to tracing
            llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default());

            // Set up model params
            let mut model_params = LlamaModelParams::default();
            if config.use_mmap {
                model_params = model_params.with_use_mmap(true);
            }
            if config.use_mlock {
                model_params = model_params.with_use_mlock(true);
            }

            // GPU layer offloading
            let gpu_layers = config.offload.gpu_layers.unwrap_or(if config.offload.auto {
                // Auto: offload as many layers as possible (use 999 to signal "all")
                999
            } else {
                0
            });
            model_params = model_params.with_n_gpu_layers(gpu_layers);

            let model = LlamaModel::load_from_file(&backend, &path, &model_params)
                .map_err(|e| InferenceError::LoadError(format!("Failed to load model: {e}")))?;

            tracing::info!(
                path = %path.display(),
                gpu_layers = gpu_layers,
                "Model loaded via llama.cpp"
            );

            let handle = ModelHandle::new(path, InferenceEngine::LlamaCpp);
            let loaded = Arc::new(LoadedModel { model, backend });

            loaded_models
                .lock()
                .expect("lock poisoned")
                .insert(handle.id, loaded);

            Ok(handle)
        })
        .await
        .map_err(|e| InferenceError::LoadError(format!("Task join error: {e}")))?
    }

    async fn generate(
        &self,
        handle: &ModelHandle,
        req: &GenerateRequest,
    ) -> Result<GenerateResponse> {
        let loaded = self.get_loaded(handle.id)?;
        let config = LoadConfig::default(); // TODO: store per-handle config
        let req = req.clone();

        tokio::task::spawn_blocking(move || {
            let text = generate_blocking(&loaded, &config, &req)?;
            Ok(GenerateResponse::Text(text))
        })
        .await
        .map_err(|e| InferenceError::GenerationError(format!("Task join error: {e}")))?
    }

    fn stream(&self, handle: &ModelHandle, req: &GenerateRequest) -> TokenStream {
        let loaded = match self.get_loaded(handle.id) {
            Ok(l) => l,
            Err(e) => {
                let (tx, rx) = tokio::sync::mpsc::channel(1);
                tokio::spawn(async move {
                    let _ = tx.send(Err(e)).await;
                });
                return Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx));
            }
        };

        let req = req.clone();
        let (tx, rx) = tokio::sync::mpsc::channel(64);

        tokio::task::spawn_blocking(move || {
            let config = LoadConfig::default();
            if let Err(e) = stream_blocking(&loaded, &config, &req, &tx) {
                let _ = tx.blocking_send(Err(e));
            }
        });

        Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
    }

    async fn unload(&self, handle: &ModelHandle) -> Result<()> {
        self.loaded_models
            .lock()
            .expect("lock poisoned")
            .remove(&handle.id);
        Ok(())
    }
}

impl LlamaCppBackend {
    fn get_loaded(&self, id: u64) -> Result<Arc<LoadedModel>> {
        self.loaded_models
            .lock()
            .expect("lock poisoned")
            .get(&id)
            .cloned()
            .ok_or(InferenceError::InvalidHandle)
    }
}

/// Build the prompt string from chat messages using the correct per-model chat template.
fn build_prompt(req: &GenerateRequest) -> String {
    let model_name = req.model_name.as_deref().unwrap_or("");
    let format = chat_template::detect_template(model_name);
    chat_template::render(format, &req.messages, &req.tools)
}

/// Build the sampler chain from the request's sampling params.
fn build_sampler(model: &LlamaModel, req: &GenerateRequest, seed: u32) -> LlamaSampler {
    let s = &req.sampling;

    let mut samplers: Vec<LlamaSampler> = Vec::new();

    // Penalties (repeat, frequency, presence)
    if s.repeat_penalty != 1.0 || s.frequency_penalty != 0.0 || s.presence_penalty != 0.0 {
        samplers.push(LlamaSampler::penalties(
            64, // look-back window
            s.repeat_penalty,
            s.frequency_penalty,
            s.presence_penalty,
        ));
    }

    // Top-K
    if s.top_k > 0 {
        samplers.push(LlamaSampler::top_k(s.top_k as i32));
    }

    // Top-P (nucleus)
    if s.top_p < 1.0 {
        samplers.push(LlamaSampler::top_p(s.top_p, 1));
    }

    // Temperature
    if s.temperature > 0.0 {
        samplers.push(LlamaSampler::temp(s.temperature));
        samplers.push(LlamaSampler::dist(seed));
    } else {
        samplers.push(LlamaSampler::greedy());
    }

    // Grammar constraint (if provided)
    if let Some(grammar_str) = &req.grammar {
        if let Ok(grammar_sampler) = LlamaSampler::grammar(model, grammar_str, "root") {
            // Insert grammar before the final selection sampler
            samplers.insert(samplers.len().saturating_sub(1), grammar_sampler);
        } else {
            tracing::warn!("Failed to parse GBNF grammar; ignoring grammar constraint");
        }
    }

    LlamaSampler::chain_simple(samplers)
}

/// Blocking generation — runs the full decode loop and returns the complete text.
fn generate_blocking(
    loaded: &LoadedModel,
    config: &LoadConfig,
    req: &GenerateRequest,
) -> Result<String> {
    let prompt = build_prompt(req);
    let tokens = loaded
        .model
        .str_to_token(&prompt, AddBos::Always)
        .map_err(|e| InferenceError::GenerationError(format!("Tokenization failed: {e}")))?;

    let n_ctx = config.context_length;
    let n_threads = config.threads.unwrap_or(num_cpus() as u32);

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(config.batch_size)
        .with_n_threads(n_threads as i32)
        .with_n_threads_batch(n_threads as i32);

    let mut ctx = loaded
        .model
        .new_context(&loaded.backend, ctx_params)
        .map_err(|e| InferenceError::GenerationError(format!("Context creation failed: {e}")))?;

    let seed = config.seed.unwrap_or(42) as u32;
    let mut sampler = build_sampler(&loaded.model, req, seed);

    // Process prompt tokens
    let mut batch = LlamaBatch::new(tokens.len(), 1);
    batch
        .add_sequence(&tokens, 0, true)
        .map_err(|e| InferenceError::GenerationError(format!("Batch add failed: {e}")))?;

    ctx.decode(&mut batch)
        .map_err(|e| InferenceError::GenerationError(format!("Prompt decode failed: {e}")))?;

    let mut output = String::new();
    let mut n_decoded = tokens.len() as i32;
    let eos = loaded.model.token_eos();

    for _ in 0..req.max_tokens {
        let token = sampler.sample(&ctx, -1);
        sampler.accept(token);

        if token == eos || loaded.model.is_eog_token(token) {
            break;
        }

        let piece = token_to_string(&loaded.model, token);
        output.push_str(&piece);

        // Check stop sequences
        if req.stop_sequences.iter().any(|stop| output.ends_with(stop)) {
            for stop in &req.stop_sequences {
                if output.ends_with(stop) {
                    output.truncate(output.len() - stop.len());
                    break;
                }
            }
            break;
        }

        // Prepare next token for decoding
        batch.clear();
        batch
            .add(token, n_decoded as i32, &[0], true)
            .map_err(|e| InferenceError::GenerationError(format!("Batch add failed: {e}")))?;

        ctx.decode(&mut batch)
            .map_err(|e| InferenceError::GenerationError(format!("Decode failed: {e}")))?;

        n_decoded += 1;
    }

    Ok(output)
}

/// Blocking streaming generation — sends tokens through a channel as they're generated.
fn stream_blocking(
    loaded: &LoadedModel,
    config: &LoadConfig,
    req: &GenerateRequest,
    tx: &tokio::sync::mpsc::Sender<Result<Token>>,
) -> Result<()> {
    let prompt = build_prompt(req);
    let tokens = loaded
        .model
        .str_to_token(&prompt, AddBos::Always)
        .map_err(|e| InferenceError::GenerationError(format!("Tokenization failed: {e}")))?;

    let n_ctx = config.context_length;
    let n_threads = config.threads.unwrap_or(num_cpus() as u32);

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(config.batch_size)
        .with_n_threads(n_threads as i32)
        .with_n_threads_batch(n_threads as i32);

    let mut ctx = loaded
        .model
        .new_context(&loaded.backend, ctx_params)
        .map_err(|e| InferenceError::GenerationError(format!("Context creation failed: {e}")))?;

    let seed = config.seed.unwrap_or(42) as u32;
    let mut sampler = build_sampler(&loaded.model, req, seed);

    // Process prompt
    let mut batch = LlamaBatch::new(tokens.len(), 1);
    batch
        .add_sequence(&tokens, 0, true)
        .map_err(|e| InferenceError::GenerationError(format!("Batch add failed: {e}")))?;

    ctx.decode(&mut batch)
        .map_err(|e| InferenceError::GenerationError(format!("Prompt decode failed: {e}")))?;

    let mut n_decoded = tokens.len() as i32;
    let eos = loaded.model.token_eos();
    let mut accumulated = String::new();

    for _ in 0..req.max_tokens {
        let token = sampler.sample(&ctx, -1);
        sampler.accept(token);

        if token == eos || loaded.model.is_eog_token(token) {
            break;
        }

        let piece = token_to_string(&loaded.model, token);

        accumulated.push_str(&piece);

        // Check stop sequences
        let should_stop = req
            .stop_sequences
            .iter()
            .any(|stop| accumulated.ends_with(stop));

        if should_stop {
            break;
        }

        let tok = Token {
            text: piece,
            id: token.0 as u32,
            logprob: None,
            is_special: false,
        };

        if tx.blocking_send(Ok(tok)).is_err() {
            break; // Receiver dropped
        }

        // Prepare next token
        batch.clear();
        batch
            .add(token, n_decoded as i32, &[0], true)
            .map_err(|e| InferenceError::GenerationError(format!("Batch add failed: {e}")))?;

        ctx.decode(&mut batch)
            .map_err(|e| InferenceError::GenerationError(format!("Decode failed: {e}")))?;

        n_decoded += 1;
    }

    Ok(())
}

/// Convert a token to a string using the non-deprecated `token_to_piece` API.
fn token_to_string(model: &LlamaModel, token: llama_cpp_2::token::LlamaToken) -> String {
    let mut decoder = encoding_rs::UTF_8.new_decoder();
    model
        .token_to_piece(token, &mut decoder, true, None)
        .unwrap_or_default()
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llamacpp_backend_metadata() {
        let backend = LlamaCppBackend::new();
        assert_eq!(backend.engine_id(), InferenceEngine::LlamaCpp);
        assert_eq!(backend.name(), "llama.cpp");
        assert!(backend.is_available());
        assert!(backend.supports_grammar());
        assert!(backend.supported_formats().contains(&ModelFormat::Gguf));
    }

    #[test]
    fn test_build_prompt_default_chatml() {
        let req = GenerateRequest {
            messages: vec![
                ChatMessage::System("You are helpful.".into()),
                ChatMessage::user_text("Hi"),
            ],
            ..Default::default()
        };
        let prompt = build_prompt(&req);
        assert!(prompt.contains("You are helpful."));
        assert!(prompt.contains("Hi"));
        // Default (no model_name) uses ChatML format
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_build_prompt_llama3() {
        let req = GenerateRequest {
            messages: vec![
                ChatMessage::System("You are helpful.".into()),
                ChatMessage::user_text("Hi"),
            ],
            model_name: Some("llama-3.1-8b".into()),
            ..Default::default()
        };
        let prompt = build_prompt(&req);
        assert!(prompt.contains("<|begin_of_text|>"));
        assert!(prompt.contains("You are helpful."));
        assert!(prompt.ends_with("assistant<|end_header_id|>\n\n"));
    }

    #[tokio::test]
    async fn test_load_nonexistent_model() {
        let backend = LlamaCppBackend::new();
        let result = backend
            .load_model(Path::new("/nonexistent/model.gguf"), &LoadConfig::default())
            .await;
        assert!(result.is_err());
        match result.unwrap_err() {
            InferenceError::ModelNotFound(path) => {
                assert!(path.contains("nonexistent"));
            }
            other => panic!("Expected ModelNotFound, got: {other}"),
        }
    }
}
