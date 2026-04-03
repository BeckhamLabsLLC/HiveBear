use async_trait::async_trait;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use candle_core::{quantized::gguf_file, DType, Device, IndexOp, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;

use crate::chat_template;
use crate::engine::candle_pipeline::SplittableLlama;
use crate::error::{InferenceError, Result};
use crate::types::*;
use hivebear_core::types::{InferenceEngine, ModelFormat};

use super::{InferenceBackend, TokenStream};

/// Internal state for a loaded Candle model.
struct LoadedModel {
    weights: ModelWeights,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
}

// Candle models are Send+Sync when using CPU device
unsafe impl Send for LoadedModel {}
unsafe impl Sync for LoadedModel {}

/// State for a pipeline-loaded model (partial layer range).
struct PipelineModel {
    llama: SplittableLlama,
    tokenizer: tokenizers::Tokenizer,
    #[allow(dead_code)]
    stage_config: PipelineStageConfig,
}

unsafe impl Send for PipelineModel {}
unsafe impl Sync for PipelineModel {}

/// Pure Rust inference backend using HuggingFace Candle.
///
/// Supports GGUF and SafeTensors models with no native dependencies.
/// This is the default/fallback backend that compiles everywhere.
pub struct CandleBackend {
    loaded_models: Arc<Mutex<HashMap<u64, Arc<LoadedModel>>>>,
    /// Models loaded for pipeline-parallel inference (partial layer ranges).
    pipeline_models: Arc<Mutex<HashMap<u64, Arc<Mutex<PipelineModel>>>>>,
}

impl CandleBackend {
    pub fn new() -> Self {
        Self {
            loaded_models: Arc::new(Mutex::new(HashMap::new())),
            pipeline_models: Arc::new(Mutex::new(HashMap::new())),
        }
    }
}

impl Default for CandleBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait(?Send))]
impl InferenceBackend for CandleBackend {
    fn engine_id(&self) -> InferenceEngine {
        InferenceEngine::Candle
    }

    fn name(&self) -> &str {
        "Candle"
    }

    fn supported_formats(&self) -> &[ModelFormat] {
        &[ModelFormat::Gguf, ModelFormat::SafeTensors]
    }

    fn is_available(&self) -> bool {
        true
    }

    fn supports_grammar(&self) -> bool {
        false
    }

    async fn load_model(&self, path: &Path, config: &LoadConfig) -> Result<ModelHandle> {
        let path = path.to_path_buf();
        let loaded_models = self.loaded_models.clone();
        let pipeline_models = self.pipeline_models.clone();
        let pipeline_stage = config.pipeline_stage.clone();

        tokio::task::spawn_blocking(move || {
            if !path.exists() {
                return Err(InferenceError::ModelNotFound(path.display().to_string()));
            }

            let device = Device::Cpu;

            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if ext != "gguf" {
                return Err(InferenceError::UnsupportedFormat(format!(
                    "Candle backend currently supports GGUF files only, got: .{ext}"
                )));
            }

            let mut file = std::fs::File::open(&path).map_err(|e| {
                InferenceError::LoadError(format!("Failed to open model file: {e}"))
            })?;

            let content = gguf_file::Content::read(&mut file).map_err(|e| {
                InferenceError::LoadError(format!("Failed to read GGUF content: {e}"))
            })?;

            // Load tokenizer from same directory
            let model_dir = path.parent().unwrap_or(Path::new("."));
            let tokenizer_path = model_dir.join("tokenizer.json");
            let tokenizer = if tokenizer_path.exists() {
                tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| {
                    InferenceError::LoadError(format!("Failed to load tokenizer: {e}"))
                })?
            } else {
                return Err(InferenceError::LoadError(format!(
                    "tokenizer.json not found in {}. Place the HuggingFace tokenizer.json \
                     alongside the GGUF file.",
                    model_dir.display()
                )));
            };

            let handle = ModelHandle::new(path.clone(), InferenceEngine::Candle);

            // Pipeline-parallel load path: load only the assigned layer range
            if let Some(stage) = pipeline_stage {
                let llama = SplittableLlama::load_partial(
                    &content,
                    &mut file,
                    &device,
                    stage.layer_range.clone(),
                    stage.total_layers,
                )
                .map_err(|e| {
                    InferenceError::LoadError(format!("Failed to load partial model: {e}"))
                })?;

                tracing::info!(
                    path = %path.display(),
                    layers = ?stage.layer_range,
                    total = stage.total_layers,
                    "Pipeline model loaded via Candle"
                );

                let pipeline_model = PipelineModel {
                    llama,
                    tokenizer,
                    stage_config: stage,
                };

                pipeline_models
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert(handle.id, Arc::new(Mutex::new(pipeline_model)));
            } else {
                // Standard full-model load path
                let weights =
                    ModelWeights::from_gguf(content, &mut file, &device).map_err(|e| {
                        InferenceError::LoadError(format!("Failed to load model weights: {e}"))
                    })?;

                tracing::info!(path = %path.display(), "Model loaded via Candle");

                let loaded = Arc::new(LoadedModel {
                    weights,
                    tokenizer,
                    device,
                });

                loaded_models
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert(handle.id, loaded);
            }

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
        let req = req.clone();

        tokio::task::spawn_blocking(move || {
            let text = generate_blocking(&loaded, &req)?;
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
            if let Err(e) = stream_blocking(&loaded, &req, &tx) {
                let _ = tx.blocking_send(Err(e));
            }
        });

        Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
    }

    async fn unload(&self, handle: &ModelHandle) -> Result<()> {
        self.loaded_models
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&handle.id);
        self.pipeline_models
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&handle.id);
        Ok(())
    }

    async fn forward_partial(
        &self,
        handle: &ModelHandle,
        input: &ActivationData,
        stage: &PipelineStageConfig,
        index_pos: usize,
    ) -> Result<ActivationData> {
        let pm = self.get_pipeline_model(handle.id)?;
        let input_clone = input.clone();
        let stage_clone = stage.clone();

        tokio::task::spawn_blocking(move || {
            let mut pm = pm.lock().unwrap_or_else(|e| e.into_inner());
            let device = &Device::Cpu;

            // Deserialize input activation → Candle Tensor
            let mut x = activation_to_tensor(&input_clone, device)?;

            // First stage with token-ID-shaped input: embed first
            let dims = x.dims();
            if stage_clone.is_first_stage() && dims.len() == 2 && dims[1] == 1 {
                let token_ids = tensor_to_u32_ids(&input_clone)?;
                let ids_tensor = Tensor::new(token_ids.as_slice(), device)
                    .map_err(|e| ie(format!("Tensor from token IDs: {e}")))?
                    .unsqueeze(0)
                    .map_err(|e| ie(format!("Unsqueeze: {e}")))?;
                x = pm
                    .llama
                    .embed(&ids_tensor)
                    .map_err(|e| ie(format!("Embed: {e}")))?;
            }

            // Forward through our assigned blocks
            x = pm
                .llama
                .forward_blocks(x, index_pos)
                .map_err(|e| ie(format!("Forward blocks: {e}")))?;

            // Last stage: project to logits
            if stage_clone.is_last_stage() {
                x = pm
                    .llama
                    .project_to_logits(&x)
                    .map_err(|e| ie(format!("Project to logits: {e}")))?;
            }

            tensor_to_activation(&x)
        })
        .await
        .map_err(|e| ie(format!("Task join error: {e}")))?
    }

    async fn embed_tokens(
        &self,
        handle: &ModelHandle,
        token_ids: &[u32],
    ) -> Result<ActivationData> {
        let pm = self.get_pipeline_model(handle.id)?;
        let ids = token_ids.to_vec();

        tokio::task::spawn_blocking(move || {
            let pm = pm.lock().unwrap_or_else(|e| e.into_inner());
            let device = &Device::Cpu;

            let ids_tensor = Tensor::new(ids.as_slice(), device)
                .map_err(|e| ie(format!("Tensor from token IDs: {e}")))?
                .unsqueeze(0)
                .map_err(|e| ie(format!("Unsqueeze: {e}")))?;

            let embedded = pm
                .llama
                .embed(&ids_tensor)
                .map_err(|e| ie(format!("Embed: {e}")))?;

            tensor_to_activation(&embedded)
        })
        .await
        .map_err(|e| ie(format!("Task join error: {e}")))?
    }

    async fn sample_from_logits(
        &self,
        handle: &ModelHandle,
        logits: &ActivationData,
        temperature: f32,
        top_p: f32,
    ) -> Result<(u32, String)> {
        let pm = self.get_pipeline_model(handle.id)?;
        let logits_clone = logits.clone();

        tokio::task::spawn_blocking(move || {
            let pm = pm.lock().unwrap_or_else(|e| e.into_inner());
            let device = &Device::Cpu;

            let logits_tensor = activation_to_tensor(&logits_clone, device)?;
            let token_id = sample_token(&logits_tensor, temperature, top_p)?;
            let text = pm.tokenizer.decode(&[token_id], true).unwrap_or_default();

            Ok((token_id, text))
        })
        .await
        .map_err(|e| ie(format!("Task join error: {e}")))?
    }
}

impl CandleBackend {
    fn get_loaded(&self, id: u64) -> Result<Arc<LoadedModel>> {
        self.loaded_models
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&id)
            .cloned()
            .ok_or(InferenceError::InvalidHandle)
    }

    fn get_pipeline_model(&self, id: u64) -> Result<Arc<Mutex<PipelineModel>>> {
        self.pipeline_models
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&id)
            .cloned()
            .ok_or(InferenceError::InvalidHandle)
    }
}

/// Shorthand for creating an InferenceError::GenerationError.
fn ie(msg: String) -> InferenceError {
    InferenceError::GenerationError(msg)
}

/// Serialize a Candle Tensor to ActivationData bytes (F32).
fn tensor_to_activation(t: &Tensor) -> Result<ActivationData> {
    let t = t
        .to_dtype(DType::F32)
        .map_err(|e| ie(format!("to_dtype: {e}")))?;
    let shape = t.dims().to_vec();
    let data: Vec<f32> = t
        .flatten_all()
        .map_err(|e| ie(format!("flatten: {e}")))?
        .to_vec1()
        .map_err(|e| ie(format!("to_vec1: {e}")))?;
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    Ok(ActivationData {
        data: bytes,
        shape,
        dtype: ActivationDtype::F32,
    })
}

/// Deserialize ActivationData bytes back into a Candle Tensor.
fn activation_to_tensor(act: &ActivationData, device: &Device) -> Result<Tensor> {
    let floats: Vec<f32> = match act.dtype {
        ActivationDtype::F32 => act
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        ActivationDtype::F16 => act
            .data
            .chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                f16_to_f32(bits)
            })
            .collect(),
        ActivationDtype::BF16 => act
            .data
            .chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                f32::from_bits((bits as u32) << 16)
            })
            .collect(),
    };
    Tensor::from_vec(floats, act.shape.as_slice(), device).map_err(|e| ie(format!("from_vec: {e}")))
}

/// IEEE 754 half-precision (F16) to single-precision (F32) conversion.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;

    let f32_bits = if exp == 0 {
        if mant == 0 {
            sign << 31 // ±0
        } else {
            // Denormalized: convert to normalized f32
            let mut m = mant;
            let mut e = 0u32;
            while (m & 0x400) == 0 {
                m <<= 1;
                e += 1;
            }
            let m = (m & 0x3ff) << 13;
            let e = (127 - 15 - e + 1) << 23;
            (sign << 31) | e | m
        }
    } else if exp == 31 {
        // Inf or NaN
        (sign << 31) | (0xff << 23) | (mant << 13)
    } else {
        // Normalized
        let e = (exp as i32 - 15 + 127) as u32;
        (sign << 31) | (e << 23) | (mant << 13)
    };
    f32::from_bits(f32_bits)
}

/// Extract u32 token IDs from an ActivationData that contains packed f32 values.
fn tensor_to_u32_ids(act: &ActivationData) -> Result<Vec<u32>> {
    match act.dtype {
        ActivationDtype::F32 => {
            let ids: Vec<u32> = act
                .data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32)
                .collect();
            Ok(ids)
        }
        _ => Err(ie("Token IDs must be F32-encoded".into())),
    }
}

/// Build the prompt string from chat messages using the correct per-model chat template.
fn build_prompt(req: &GenerateRequest) -> String {
    let model_name = req.model_name.as_deref().unwrap_or("");
    let format = chat_template::detect_template(model_name);
    chat_template::render(format, &req.messages, &req.tools)
}

/// Sample the next token from logits using temperature sampling.
fn sample_token(logits: &Tensor, temperature: f32, top_p: f32) -> Result<u32> {
    let logits = logits
        .squeeze(0)
        .map_err(|e| InferenceError::GenerationError(format!("Squeeze failed: {e}")))?;

    let logits = if temperature > 0.0 {
        let logits = (&logits / temperature as f64)
            .map_err(|e| InferenceError::GenerationError(format!("Temp scaling failed: {e}")))?;
        // Apply softmax
        let exp = logits
            .exp()
            .map_err(|e| InferenceError::GenerationError(format!("Exp failed: {e}")))?;
        let sum = exp
            .sum_all()
            .map_err(|e| InferenceError::GenerationError(format!("Sum failed: {e}")))?;
        let probs = exp
            .broadcast_div(&sum)
            .map_err(|e| InferenceError::GenerationError(format!("Div failed: {e}")))?;

        // Top-p (nucleus) sampling
        let probs_vec: Vec<f32> = probs
            .to_vec1()
            .map_err(|e| InferenceError::GenerationError(format!("To vec failed: {e}")))?;

        // Sort by probability descending
        let mut indexed: Vec<(usize, f32)> = probs_vec.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Cumulative sum for top-p
        let mut cumsum = 0.0f32;
        let mut candidates: Vec<(usize, f32)> = Vec::new();
        for (idx, prob) in indexed {
            cumsum += prob;
            candidates.push((idx, prob));
            if cumsum >= top_p {
                break;
            }
        }

        // Re-normalize and sample
        let total: f32 = candidates.iter().map(|(_, p)| p).sum();
        let mut rng_val: f32 = rand_f32() * total;
        for (idx, prob) in &candidates {
            rng_val -= prob;
            if rng_val <= 0.0 {
                return Ok(*idx as u32);
            }
        }
        // Fallback to most likely
        return Ok(candidates[0].0 as u32);
    } else {
        // Greedy: argmax
        logits
    };

    let token_id = logits
        .argmax(0)
        .map_err(|e| InferenceError::GenerationError(format!("Argmax failed: {e}")))?
        .to_scalar::<u32>()
        .map_err(|e| InferenceError::GenerationError(format!("Scalar conversion failed: {e}")))?;
    Ok(token_id)
}

/// Simple random float [0, 1) using thread-local state.
fn rand_f32() -> f32 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;

    thread_local! {
        static STATE: std::cell::Cell<u64> = std::cell::Cell::new({
            let mut hasher = DefaultHasher::new();
            SystemTime::now().hash(&mut hasher);
            std::thread::current().id().hash(&mut hasher);
            hasher.finish()
        });
    }

    STATE.with(|s| {
        // xorshift64
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        (x as f32) / (u64::MAX as f32)
    })
}

/// Blocking generation returning the complete text.
fn generate_blocking(loaded: &LoadedModel, req: &GenerateRequest) -> Result<String> {
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

    // Clone weights for mutable forward pass
    let mut weights = loaded.weights.clone();

    // Forward pass on prompt
    let logits = weights
        .forward(&input, 0)
        .map_err(|e| InferenceError::GenerationError(format!("Forward pass failed: {e}")))?;

    let last_logits = logits
        .i((.., logits.dim(1).unwrap_or(1) - 1, ..))
        .map_err(|e| InferenceError::GenerationError(format!("Index failed: {e}")))?;

    let mut next_token = sample_token(&last_logits, req.sampling.temperature, req.sampling.top_p)?;
    let mut output = String::new();
    let seq_len = prompt_tokens.len();

    let eos_token = find_eos_token(&loaded.tokenizer, req.model_name.as_deref());

    for i in 0..req.max_tokens {
        if next_token == eos_token {
            break;
        }

        if let Ok(text) = loaded.tokenizer.decode(&[next_token], true) {
            output.push_str(&text);
        }

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

        // Next forward pass (single token)
        let input = Tensor::new(&[next_token], &loaded.device)
            .map_err(|e| InferenceError::GenerationError(format!("Tensor failed: {e}")))?
            .unsqueeze(0)
            .map_err(|e| InferenceError::GenerationError(format!("Unsqueeze failed: {e}")))?;

        let logits = weights
            .forward(&input, seq_len + i as usize)
            .map_err(|e| InferenceError::GenerationError(format!("Forward failed: {e}")))?;

        let last_logits = logits
            .i((.., logits.dim(1).unwrap_or(1) - 1, ..))
            .map_err(|e| InferenceError::GenerationError(format!("Index failed: {e}")))?;

        next_token = sample_token(&last_logits, req.sampling.temperature, req.sampling.top_p)?;
    }

    Ok(output)
}

/// Blocking streaming — sends tokens through a channel.
fn stream_blocking(
    loaded: &LoadedModel,
    req: &GenerateRequest,
    tx: &tokio::sync::mpsc::Sender<Result<Token>>,
) -> Result<()> {
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

    let mut weights = loaded.weights.clone();

    let logits = weights
        .forward(&input, 0)
        .map_err(|e| InferenceError::GenerationError(format!("Forward pass failed: {e}")))?;

    let last_logits = logits
        .i((.., logits.dim(1).unwrap_or(1) - 1, ..))
        .map_err(|e| InferenceError::GenerationError(format!("Index failed: {e}")))?;

    let mut next_token = sample_token(&last_logits, req.sampling.temperature, req.sampling.top_p)?;
    let seq_len = prompt_tokens.len();
    let mut accumulated = String::new();

    let eos_token = find_eos_token(&loaded.tokenizer, req.model_name.as_deref());

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

        let tok = Token {
            text: piece,
            id: next_token,
            logprob: None,
            is_special: false,
        };

        if tx.blocking_send(Ok(tok)).is_err() {
            break;
        }

        let input = Tensor::new(&[next_token], &loaded.device)
            .map_err(|e| InferenceError::GenerationError(format!("Tensor failed: {e}")))?
            .unsqueeze(0)
            .map_err(|e| InferenceError::GenerationError(format!("Unsqueeze failed: {e}")))?;

        let logits = weights
            .forward(&input, seq_len + i as usize)
            .map_err(|e| InferenceError::GenerationError(format!("Forward failed: {e}")))?;

        let last_logits = logits
            .i((.., logits.dim(1).unwrap_or(1) - 1, ..))
            .map_err(|e| InferenceError::GenerationError(format!("Index failed: {e}")))?;

        next_token = sample_token(&last_logits, req.sampling.temperature, req.sampling.top_p)?;
    }

    Ok(())
}

/// Find the EOS token ID by checking model-family-specific tokens and common fallbacks.
fn find_eos_token(tokenizer: &tokenizers::Tokenizer, model_name: Option<&str>) -> u32 {
    let format = chat_template::detect_template(model_name.unwrap_or(""));

    // Check the model-family-specific EOS token first
    let primary = match format {
        chat_template::TemplateFormat::Llama3 => "<|eot_id|>",
        chat_template::TemplateFormat::ChatML => "<|im_end|>",
        chat_template::TemplateFormat::Gemma => "<end_of_turn>",
        chat_template::TemplateFormat::Phi3 => "<|end|>",
        chat_template::TemplateFormat::Mistral => "</s>",
        chat_template::TemplateFormat::Generic => "</s>",
    };

    tokenizer
        .token_to_id(primary)
        // Fallback chain for models whose tokenizer uses a different EOS spelling
        .or_else(|| tokenizer.token_to_id("</s>"))
        .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
        .or_else(|| tokenizer.token_to_id("<|eot_id|>"))
        .or_else(|| tokenizer.token_to_id("<|im_end|>"))
        .or_else(|| tokenizer.token_to_id("<|end|>"))
        .or_else(|| tokenizer.token_to_id("<end_of_turn>"))
        .unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_backend_metadata() {
        let backend = CandleBackend::new();
        assert_eq!(backend.engine_id(), InferenceEngine::Candle);
        assert_eq!(backend.name(), "Candle");
        assert!(backend.is_available());
        assert!(!backend.supports_grammar());
        assert!(backend.supported_formats().contains(&ModelFormat::Gguf));
        assert!(backend
            .supported_formats()
            .contains(&ModelFormat::SafeTensors));
    }

    #[test]
    fn test_build_prompt_default_chatml() {
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
        // Default (no model_name) uses ChatML format
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn test_build_prompt_llama3() {
        let req = GenerateRequest {
            messages: vec![
                ChatMessage::System("You are a helpful AI.".into()),
                ChatMessage::user_text("What is Rust?"),
            ],
            model_name: Some("llama-3.1-8b".into()),
            ..Default::default()
        };
        let prompt = build_prompt(&req);
        assert!(prompt.contains("<|begin_of_text|>"));
        assert!(prompt.contains("You are a helpful AI."));
        assert!(prompt.ends_with("assistant<|end_header_id|>\n\n"));
    }

    #[tokio::test]
    async fn test_load_nonexistent_model() {
        let backend = CandleBackend::new();
        let result = backend
            .load_model(Path::new("/nonexistent/model.gguf"), &LoadConfig::default())
            .await;
        assert!(result.is_err());
    }

    #[test]
    fn test_rand_f32_range() {
        for _ in 0..100 {
            let val = rand_f32();
            assert!((0.0..1.0).contains(&val), "rand_f32 out of range: {val}");
        }
    }
}
