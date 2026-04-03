//! Splittable quantized Llama model for pipeline-parallel inference.
//!
//! Reimplements the forward pass from `candle_transformers::models::quantized_llama`
//! with support for:
//! - Selective layer loading (load only layers N..M from a GGUF file)
//! - Partial forward passes (process only assigned layers)
//! - Memory efficiency (only the assigned layer weights are loaded)
//!
//! This enables pipeline-parallel inference across mesh peers, where each peer
//! loads and processes a subset of the model's transformer blocks.

use std::collections::HashMap;
use std::io::{Read, Seek};
use std::ops::Range;

use candle_core::quantized::gguf_file;
use candle_core::quantized::QTensor;
use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{Embedding, Module};
use candle_transformers::quantized_nn::RmsNorm;

const MAX_SEQ_LEN: usize = 4096;

// ── QMatMul wrapper (matches upstream, adds tracing) ────────────────────

#[derive(Debug, Clone)]
struct QMatMul {
    inner: candle_core::quantized::QMatMul,
    span: tracing::Span,
}

impl QMatMul {
    fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        let inner = candle_core::quantized::QMatMul::from_qtensor(qtensor)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

// ── MLP / MoE ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_w1: QMatMul,
    feed_forward_w2: QMatMul,
    feed_forward_w3: QMatMul,
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.feed_forward_w1.forward(xs)?;
        let w3 = self.feed_forward_w3.forward(xs)?;
        self.feed_forward_w2
            .forward(&(candle_nn::ops::silu(&w1)? * w3)?)
    }
}

#[derive(Debug, Clone)]
enum MlpOrMoe {
    Mlp(Mlp),
    MoE {
        n_expert_used: usize,
        feed_forward_gate_inp: QMatMul,
        experts: Vec<Mlp>,
    },
}

impl Module for MlpOrMoe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::MoE {
                feed_forward_gate_inp,
                experts,
                n_expert_used,
            } => {
                let (b_size, seq_len, hidden_dim) = xs.dims3()?;
                let xs = xs.reshape(((), hidden_dim))?;
                let router_logits = feed_forward_gate_inp.forward(&xs)?;
                let routing_weights = candle_nn::ops::softmax_last_dim(&router_logits)?;

                let routing_weights = routing_weights.to_dtype(DType::F32)?.to_vec2::<f32>()?;

                let mut top_x = vec![vec![]; experts.len()];
                let mut selected_rws = vec![vec![]; experts.len()];
                for (row_idx, rw) in routing_weights.iter().enumerate() {
                    let mut dst = (0..rw.len() as u32).collect::<Vec<u32>>();
                    dst.sort_by(|&i, &j| rw[j as usize].total_cmp(&rw[i as usize]));
                    let mut sum_routing_weights = 0f32;
                    for &expert_idx in dst.iter().take(*n_expert_used) {
                        let expert_idx = expert_idx as usize;
                        let routing_weight = rw[expert_idx];
                        sum_routing_weights += routing_weight;
                        top_x[expert_idx].push(row_idx as u32);
                    }
                    for &expert_idx in dst.iter().take(*n_expert_used) {
                        let expert_idx = expert_idx as usize;
                        let routing_weight = rw[expert_idx];
                        selected_rws[expert_idx].push(routing_weight / sum_routing_weights);
                    }
                }

                let mut ys = xs.zeros_like()?;
                for (expert_idx, expert_layer) in experts.iter().enumerate() {
                    let top_x_slice = &top_x[expert_idx];
                    if top_x_slice.is_empty() {
                        continue;
                    }
                    let top_x_t = Tensor::new(top_x_slice.as_slice(), xs.device())?;
                    let selected_rws_t =
                        Tensor::new(selected_rws[expert_idx].as_slice(), xs.device())?
                            .reshape(((), 1))?;
                    let current_state = xs.index_select(&top_x_t, 0)?.reshape(((), hidden_dim))?;
                    let current_hidden_states = expert_layer.forward(&current_state)?;
                    let current_hidden_states =
                        current_hidden_states.broadcast_mul(&selected_rws_t)?;
                    ys = ys.index_add(&top_x_t, &current_hidden_states, 0)?;
                }

                let ys = ys.reshape((b_size, seq_len, hidden_dim))?;
                Ok(ys)
            }
            Self::Mlp(mlp) => mlp.forward(xs),
        }
    }
}

// ── Transformer block ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct LayerWeights {
    attention_wq: QMatMul,
    attention_wk: QMatMul,
    attention_wv: QMatMul,
    attention_wo: QMatMul,
    attention_norm: RmsNorm,
    mlp_or_moe: MlpOrMoe,
    ffn_norm: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    neg_inf: Tensor,
    kv_cache: Option<(Tensor, Tensor)>,
    span_attn: tracing::Span,
    span_rot: tracing::Span,
    span_mlp: tracing::Span,
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: &Tensor) -> Result<Tensor> {
    let shape = mask.shape();
    let m = mask.where_cond(&on_true.broadcast_as(shape.dims())?, on_false)?;
    Ok(m)
}

/// Repeat KV heads for grouped query attention (GQA/MQA).
fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
        Tensor::cat(&vec![&xs; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

impl LayerWeights {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (_b_sz, _n_head, seq_len, _n_embd) = x.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        candle_nn::rotary_emb::rope_i(&x.contiguous()?, &cos, &sin)
    }

    fn forward_attn(
        &mut self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b_sz, seq_len, n_embd) = x.dims3()?;
        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        let q = q
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    let k = Tensor::cat(&[k_cache, &k], 2)?;
                    let v = Tensor::cat(&[v_cache, &v], 2)?;
                    (k, v)
                }
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        let y = if q.device().is_metal() && seq_len == 1 {
            candle_nn::ops::sdpa(&q, &k, &v, 1. / (self.head_dim as f32).sqrt(), 1.)?
        } else {
            let k = repeat_kv(k, self.n_head / self.n_kv_head)?;
            let v = repeat_kv(v, self.n_head / self.n_kv_head)?;

            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let att = match mask {
                None => att,
                Some(mask) => {
                    let mask = mask.broadcast_as(att.shape())?;
                    masked_fill(&att, &mask, &self.neg_inf)?
                }
            };
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            att.matmul(&v.contiguous()?)?
        };

        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        self.attention_wo.forward(&y)
    }
}

// ── RoPE frequency precomputation ───────────────────────────────────────

fn precompute_freqs_cis(
    head_dim: usize,
    freq_base: f32,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), device)?;
    let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
        .to_dtype(DType::F32)?
        .reshape((MAX_SEQ_LEN, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
}

// ── Config ──────────────────────────────────────────────────────────────

/// Model configuration extracted from GGUF metadata.
#[derive(Debug, Clone)]
pub(crate) struct LlamaConfig {
    pub head_count: usize,
    pub head_count_kv: usize,
    pub embedding_length: usize,
    #[allow(dead_code)]
    pub block_count: usize,
    pub rope_dim: usize,
    pub rope_freq_base: f32,
    pub rms_norm_eps: f64,
    pub n_expert: usize,
    pub n_expert_used: usize,
}

impl LlamaConfig {
    /// Read model config from GGUF metadata without loading any tensor data.
    pub fn from_gguf(content: &gguf_file::Content) -> Result<Self> {
        let md_get = |s: &str| match content.metadata.get(s) {
            None => candle_core::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        let n_expert = md_get("llama.expert_count")
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;
        let n_expert_used = md_get("llama.expert_used_count")
            .and_then(|v| v.to_u32())
            .unwrap_or(0) as usize;

        Ok(Self {
            head_count: md_get("llama.attention.head_count")?.to_u32()? as usize,
            head_count_kv: md_get("llama.attention.head_count_kv")?.to_u32()? as usize,
            block_count: md_get("llama.block_count")?.to_u32()? as usize,
            embedding_length: md_get("llama.embedding_length")?.to_u32()? as usize,
            rope_dim: md_get("llama.rope.dimension_count")?.to_u32()? as usize,
            rms_norm_eps: md_get("llama.attention.layer_norm_rms_epsilon")?.to_f32()? as f64,
            rope_freq_base: md_get("llama.rope.freq_base")
                .and_then(|m| m.to_f32())
                .unwrap_or(10000f32),
            n_expert,
            n_expert_used,
        })
    }
}

// ── SplittableLlama ─────────────────────────────────────────────────────

/// A partially-loaded Llama model that can execute any layer range.
///
/// Unlike `ModelWeights` from candle-transformers (which loads all layers and
/// runs a monolithic forward pass), `SplittableLlama`:
/// - Loads only the tensors for the assigned layer range (memory efficient)
/// - Can run embedding (first stage), transformer blocks (any stage),
///   and final projection (last stage) independently
///
/// This enables pipeline-parallel inference across mesh peers.
pub(crate) struct SplittableLlama {
    #[allow(dead_code)]
    config: LlamaConfig,
    device: Device,
    /// Token embedding layer. Only loaded for the first pipeline stage.
    tok_embeddings: Option<Embedding>,
    /// Transformer blocks for our assigned layer range only.
    blocks: Vec<LayerWeights>,
    /// The global layer index of our first block.
    #[allow(dead_code)]
    layer_offset: u32,
    /// Final RMS norm. Only loaded for the last pipeline stage.
    final_norm: Option<RmsNorm>,
    /// Output projection (lm_head). Only loaded for the last pipeline stage.
    output_proj: Option<QMatMul>,
    /// Cached causal attention masks keyed by sequence length.
    masks: HashMap<usize, Tensor>,
}

unsafe impl Send for SplittableLlama {}
unsafe impl Sync for SplittableLlama {}

impl SplittableLlama {
    /// Load only the weights for the specified layer range from a GGUF file.
    ///
    /// - First stage (`layer_range.start == 0`): also loads token embeddings
    /// - Last stage (`layer_range.end == total_layers`): also loads output norm + lm_head
    /// - Memory: loads approximately `(range_len / total_layers)` of the full weights
    pub fn load_partial<R: Read + Seek>(
        content: &gguf_file::Content,
        reader: &mut R,
        device: &Device,
        layer_range: Range<u32>,
        total_layers: u32,
    ) -> Result<Self> {
        let config = LlamaConfig::from_gguf(content)?;

        if layer_range.start >= layer_range.end || layer_range.end > total_layers {
            candle_core::bail!(
                "invalid layer range {}..{} for model with {} layers",
                layer_range.start,
                layer_range.end,
                total_layers
            );
        }

        let (cos, sin) = precompute_freqs_cis(config.rope_dim, config.rope_freq_base, device)?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        // First stage: load token embeddings
        let tok_embeddings = if layer_range.start == 0 {
            let q = content.tensor(reader, "token_embd.weight", device)?;
            Some(Embedding::new(
                q.dequantize(device)?,
                config.embedding_length,
            ))
        } else {
            None
        };

        // Load only the transformer blocks in our assigned range
        let range_len = (layer_range.end - layer_range.start) as usize;
        let mut blocks = Vec::with_capacity(range_len);
        for i in layer_range.start..layer_range.end {
            let prefix = format!("blk.{i}");

            let attention_wq =
                content.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk =
                content.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv =
                content.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;
            let attention_wo =
                content.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;

            let mlp_or_moe = if config.n_expert <= 1 {
                let w1 = content.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?;
                let w2 = content.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;
                let w3 = content.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
                MlpOrMoe::Mlp(Mlp {
                    feed_forward_w1: QMatMul::from_qtensor(w1)?,
                    feed_forward_w2: QMatMul::from_qtensor(w2)?,
                    feed_forward_w3: QMatMul::from_qtensor(w3)?,
                })
            } else {
                let gate_inp =
                    content.tensor(reader, &format!("{prefix}.ffn_gate_inp.weight"), device)?;
                let mut experts = Vec::with_capacity(config.n_expert);
                for j in 0..config.n_expert {
                    let w1 =
                        content.tensor(reader, &format!("{prefix}.ffn_gate.{j}.weight"), device)?;
                    let w2 =
                        content.tensor(reader, &format!("{prefix}.ffn_down.{j}.weight"), device)?;
                    let w3 =
                        content.tensor(reader, &format!("{prefix}.ffn_up.{j}.weight"), device)?;
                    experts.push(Mlp {
                        feed_forward_w1: QMatMul::from_qtensor(w1)?,
                        feed_forward_w2: QMatMul::from_qtensor(w2)?,
                        feed_forward_w3: QMatMul::from_qtensor(w3)?,
                    });
                }
                MlpOrMoe::MoE {
                    n_expert_used: config.n_expert_used,
                    feed_forward_gate_inp: QMatMul::from_qtensor(gate_inp)?,
                    experts,
                }
            };

            let attention_norm =
                content.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?;
            let ffn_norm = content.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?;

            blocks.push(LayerWeights {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_norm: RmsNorm::from_qtensor(attention_norm, config.rms_norm_eps)?,
                mlp_or_moe,
                ffn_norm: RmsNorm::from_qtensor(ffn_norm, config.rms_norm_eps)?,
                n_head: config.head_count,
                n_kv_head: config.head_count_kv,
                head_dim: config.embedding_length / config.head_count,
                cos: cos.clone(),
                sin: sin.clone(),
                neg_inf: neg_inf.clone(),
                kv_cache: None,
                span_attn: tracing::span!(tracing::Level::TRACE, "attn"),
                span_rot: tracing::span!(tracing::Level::TRACE, "attn-rot"),
                span_mlp: tracing::span!(tracing::Level::TRACE, "attn-mlp"),
            });
        }

        // Last stage: load output norm + lm_head
        let (final_norm, output_proj) = if layer_range.end == total_layers {
            let norm = RmsNorm::from_qtensor(
                content.tensor(reader, "output_norm.weight", device)?,
                config.rms_norm_eps,
            )?;
            let output = match content.tensor(reader, "output.weight", device) {
                Ok(tensor) => tensor,
                // Some models share embedding weights with the output projection
                Err(_) => content.tensor(reader, "token_embd.weight", device)?,
            };
            (Some(norm), Some(QMatMul::from_qtensor(output)?))
        } else {
            (None, None)
        };

        tracing::info!(
            layers = ?layer_range,
            total = total_layers,
            has_embed = tok_embeddings.is_some(),
            has_output = output_proj.is_some(),
            "Loaded partial model"
        );

        Ok(Self {
            config,
            device: device.clone(),
            tok_embeddings,
            blocks,
            layer_offset: layer_range.start,
            final_norm,
            output_proj,
            masks: HashMap::new(),
        })
    }

    /// Compute or retrieve a cached causal attention mask.
    fn mask(&mut self, t: usize) -> Result<Tensor> {
        if let Some(mask) = self.masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..t)
                .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (t, t), &self.device)?;
            self.masks.insert(t, mask.clone());
            Ok(mask)
        }
    }

    /// Run token embedding (first stage only).
    ///
    /// Input: `[batch, seq_len]` tensor of u32 token IDs.
    /// Output: `[batch, seq_len, hidden_dim]` activation tensor.
    pub fn embed(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.tok_embeddings
            .as_ref()
            .ok_or_else(|| {
                candle_core::Error::Msg("embed() called on non-first pipeline stage".into())
            })?
            .forward(token_ids)
    }

    /// Forward pass through our loaded transformer blocks only.
    ///
    /// Input: `[batch, seq_len, hidden_dim]` activation tensor.
    /// Output: `[batch, seq_len, hidden_dim]` activation tensor.
    ///
    /// `index_pos` is the token position for RoPE — must be correct
    /// regardless of which pipeline stage this is.
    pub fn forward_blocks(&mut self, mut x: Tensor, index_pos: usize) -> Result<Tensor> {
        let seq_len = x.dim(1)?;
        let mask = if seq_len == 1 {
            None
        } else {
            Some(self.mask(seq_len)?)
        };

        for layer in self.blocks.iter_mut() {
            let residual = &x;
            let normed = layer.attention_norm.forward(&x)?;
            let attn = layer.forward_attn(&normed, mask.as_ref(), index_pos)?;
            let x2 = (attn + residual)?;

            let _enter = layer.span_mlp.enter();
            let residual = &x2;
            let normed = layer.ffn_norm.forward(&x2)?;
            let mlp_out = layer.mlp_or_moe.forward(&normed)?;
            x = (mlp_out + residual)?;
        }
        Ok(x)
    }

    /// Apply final norm and output projection (last stage only).
    ///
    /// Input: `[batch, seq_len, hidden_dim]`.
    /// Output: `[batch, vocab_size]` logits for the last token position.
    pub fn project_to_logits(&self, x: &Tensor) -> Result<Tensor> {
        let norm = self.final_norm.as_ref().ok_or_else(|| {
            candle_core::Error::Msg("project_to_logits() called on non-last pipeline stage".into())
        })?;
        let output = self
            .output_proj
            .as_ref()
            .ok_or_else(|| candle_core::Error::Msg("no output projection loaded".into()))?;
        let seq_len = x.dim(1)?;
        let x = norm.forward(x)?;
        let x = x.i((.., seq_len - 1, ..))?;
        output.forward(&x)
    }

    /// Clear all KV caches (call when starting a new generation session).
    #[allow(dead_code)]
    pub fn clear_kv_cache(&mut self) {
        for block in &mut self.blocks {
            block.kv_cache = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_fields() {
        let config = LlamaConfig {
            head_count: 32,
            head_count_kv: 8,
            embedding_length: 4096,
            block_count: 32,
            rope_dim: 128,
            rope_freq_base: 10000.0,
            rms_norm_eps: 1e-6,
            n_expert: 0,
            n_expert_used: 0,
        };
        assert_eq!(config.head_count, 32);
        assert_eq!(config.block_count, 32);
        assert_eq!(config.embedding_length / config.head_count, 128);
    }

    #[test]
    fn test_precompute_freqs_cis() {
        let device = Device::Cpu;
        let (cos, sin) = precompute_freqs_cis(128, 10000.0, &device).unwrap();
        assert_eq!(cos.dims(), &[MAX_SEQ_LEN, 64]);
        assert_eq!(sin.dims(), &[MAX_SEQ_LEN, 64]);
    }

    #[test]
    fn test_repeat_kv_identity() {
        let device = Device::Cpu;
        let xs = Tensor::zeros((1, 8, 10, 64), DType::F32, &device).unwrap();
        let result = repeat_kv(xs.clone(), 1).unwrap();
        assert_eq!(result.dims(), xs.dims());
    }

    #[test]
    fn test_repeat_kv_expand() {
        let device = Device::Cpu;
        let xs = Tensor::zeros((1, 8, 10, 64), DType::F32, &device).unwrap();
        let result = repeat_kv(xs, 4).unwrap();
        assert_eq!(result.dims(), &[1, 32, 10, 64]);
    }
}
