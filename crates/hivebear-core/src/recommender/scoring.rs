use crate::recommender::model_db::ModelEntry;
use crate::types::{
    ComputeApi, HardwareProfile, InferenceEngine, ModelRecommendation, Quantization,
};

/// All quantization levels to consider, ordered from smallest to largest.
const QUANT_LEVELS: &[Quantization] = &[
    Quantization::Q4KM,
    Quantization::Q4KS,
    Quantization::Q5KM,
    Quantization::Q5KS,
    Quantization::Q6K,
    Quantization::Q8_0,
    Quantization::Q3KM,
    Quantization::Q3KL,
    Quantization::Q2K,
];

/// Maximum model size in billions of parameters by platform.
fn max_params_for_platform(profile: &HardwareProfile) -> f64 {
    if profile.platform.os == "browser" || profile.platform.arch == "wasm32" {
        // Browser: limited by available memory, cap at 8B (realistically 1-3B)
        8.0
    } else if profile.platform.is_mobile {
        // Mobile: 3B max to keep memory usage low
        3.0
    } else if profile.platform.arch == "aarch64"
        && profile.memory.total_bytes <= 9 * 1024 * 1024 * 1024
    {
        // ARM embedded (e.g., Raspberry Pi 5 with 8GB): cap at 7B
        7.0
    } else {
        // Desktop: no limit
        f64::MAX
    }
}

/// Get memory usage fraction based on platform.
fn memory_fraction_for_platform(profile: &HardwareProfile, requested: f64) -> f64 {
    if profile.platform.is_mobile {
        // Mobile OS kills background apps aggressively — use at most 50%
        requested.min(0.50)
    } else {
        requested
    }
}

/// Generate recommendations for a given hardware profile and model database.
pub fn recommend(
    profile: &HardwareProfile,
    models: &[ModelEntry],
    max_memory_fraction: f64,
    min_tok_s: f32,
    top_n: usize,
) -> Vec<ModelRecommendation> {
    let adjusted_memory_fraction = memory_fraction_for_platform(profile, max_memory_fraction);
    let effective_memory = calculate_effective_memory(profile, adjusted_memory_fraction);
    let bandwidth_gbps = effective_bandwidth(profile);
    let best_engine = select_engine(profile);
    let max_params = max_params_for_platform(profile);

    let mut recommendations: Vec<ModelRecommendation> = Vec::new();

    for model in models {
        // Skip models too large for this platform
        if model.params_billions > max_params {
            continue;
        }
        // For each model, find the best quantization that fits
        let mut best_for_model: Option<ModelRecommendation> = None;

        for &quant in QUANT_LEVELS {
            let memory_bytes = estimate_memory(model, quant, 4096);

            if memory_bytes > effective_memory {
                continue;
            }

            let tok_s = estimate_tokens_per_sec(model, quant, bandwidth_gbps);

            if tok_s < min_tok_s {
                continue;
            }

            let quality = model.quality_score * quant.quality_retention();
            let speed_normalized = (tok_s as f64 / 100.0).min(1.0); // Normalize to 0-1
            let score = quality * 0.6 + speed_normalized * 0.4;

            let mut warnings = Vec::new();

            // Warning if we're using a very aggressive quantization
            if quant.bits_per_weight() < 4.0 {
                warnings.push(format!(
                    "Using aggressive {} quantization — expect some quality loss",
                    quant
                ));
            }

            // Warning if we need disk offloading
            let ram_only = profile.memory.available_bytes as f64 * adjusted_memory_fraction;
            if memory_bytes > ram_only as u64 && !profile.gpus.is_empty() {
                warnings.push("Will split layers between CPU and GPU".to_string());
            }

            // Warning for very large models on limited hardware
            if tok_s < 10.0 {
                warnings.push(format!(
                    "Estimated {:.1} tok/s — may feel slow for interactive use",
                    tok_s
                ));
            }

            let confidence = calculate_confidence(model, quant, profile);

            let rec = ModelRecommendation {
                model_id: model.id.clone(),
                model_name: model.name.clone(),
                quantization: quant,
                engine: best_engine,
                estimated_tokens_per_sec: tok_s,
                estimated_memory_usage_bytes: memory_bytes,
                confidence,
                warnings,
                score,
            };

            // Keep the best quantization for this model (highest score)
            match &best_for_model {
                Some(existing) if existing.score >= score => {}
                _ => best_for_model = Some(rec),
            }
        }

        if let Some(rec) = best_for_model {
            recommendations.push(rec);
        }
    }

    // Sort by score descending
    recommendations.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Return top N
    recommendations.truncate(top_n);
    recommendations
}

/// Calculate total usable memory (RAM + VRAM).
fn calculate_effective_memory(profile: &HardwareProfile, max_fraction: f64) -> u64 {
    let ram = (profile.memory.available_bytes as f64 * max_fraction) as u64;
    let vram: u64 = profile.gpus.iter().map(|g| g.vram_bytes).sum();
    ram + vram
}

/// Get the effective memory bandwidth for estimation.
fn effective_bandwidth(profile: &HardwareProfile) -> f64 {
    // If we have GPU with known high bandwidth, use that
    // Otherwise use the measured RAM bandwidth
    let ram_bw = profile.memory.estimated_bandwidth_gbps;

    // GPU memory bandwidth is typically much higher than RAM
    // but we don't measure it directly. Use conservative estimates.
    if !profile.gpus.is_empty() {
        // Assume at least some layers on GPU; blend bandwidths
        // This is a rough heuristic — real performance depends on layer split
        let gpu_bw_estimate = estimate_gpu_bandwidth(&profile.gpus[0]);
        if gpu_bw_estimate > ram_bw {
            // Weighted average: assume 60% GPU, 40% CPU for typical offloading
            gpu_bw_estimate * 0.6 + ram_bw * 0.4
        } else {
            ram_bw
        }
    } else {
        ram_bw
    }
}

/// Rough GPU memory bandwidth estimate based on known GPU classes.
fn estimate_gpu_bandwidth(gpu: &crate::types::GpuInfo) -> f64 {
    let name = gpu.name.to_lowercase();

    // NVIDIA RTX 40 series
    if name.contains("4090") {
        return 1008.0;
    }
    if name.contains("4080") {
        return 717.0;
    }
    if name.contains("4070") {
        return 504.0;
    }
    if name.contains("4060") {
        return 288.0;
    }

    // NVIDIA RTX 30 series
    if name.contains("3090") {
        return 936.0;
    }
    if name.contains("3080") {
        return 760.0;
    }
    if name.contains("3070") {
        return 448.0;
    }
    if name.contains("3060") {
        return 360.0;
    }

    // AMD RX 7000 series
    if name.contains("7900 xtx") {
        return 960.0;
    }
    if name.contains("7900 xt") {
        return 800.0;
    }
    if name.contains("7800 xt") {
        return 624.0;
    }

    // Apple Silicon (unified memory)
    if name.contains("apple") {
        return 200.0; // Conservative for M1/M2
    }

    // Unknown GPU: conservative estimate
    100.0
}

/// Select the best inference engine for this hardware.
fn select_engine(profile: &HardwareProfile) -> InferenceEngine {
    // Check for Apple Silicon
    if profile.platform.os == "macos" && profile.platform.arch == "aarch64" {
        return InferenceEngine::Mlx;
    }

    // Check for GPU with CUDA
    for gpu in &profile.gpus {
        if gpu.compute_api == ComputeApi::Cuda {
            return InferenceEngine::LlamaCpp; // llama.cpp with CUDA
        }
    }

    // Check for any GPU (Vulkan, Metal, etc.)
    if !profile.gpus.is_empty() {
        return InferenceEngine::LlamaCpp; // llama.cpp with Vulkan/Metal
    }

    // CPU only — llama.cpp is still the best option
    InferenceEngine::LlamaCpp
}

/// Estimate memory usage for a model at a given quantization.
fn estimate_memory(model: &ModelEntry, quant: Quantization, context_length: u32) -> u64 {
    let params = model.params_billions * 1_000_000_000.0;
    let bits = quant.bits_per_weight();

    // Model weights
    let weight_bytes = (params * bits / 8.0) as u64;

    // KV cache estimate: 2 (key+value) * 2 bytes (FP16) * num_kv_heads * head_dim * ctx_len * num_layers
    // Simplified: ~0.125 bytes per parameter per 1K tokens of context
    // A 7B model at 4K context uses ~250MB of KV cache
    let kv_cache_bytes = (params * 0.125 * context_length as f64 / 1000.0) as u64;

    // Overhead (tokenizer, buffers, etc.): ~10%
    let overhead = (weight_bytes + kv_cache_bytes) / 10;

    weight_bytes + kv_cache_bytes + overhead
}

/// Estimate tokens per second based on memory bandwidth.
/// LLM inference during generation is memory-bandwidth-bound.
fn estimate_tokens_per_sec(model: &ModelEntry, quant: Quantization, bandwidth_gbps: f64) -> f32 {
    let params = model.params_billions * 1_000_000_000.0;
    let bits = quant.bits_per_weight();

    // Bytes that must be read per token (approximately all weights)
    let bytes_per_token = params * bits / 8.0;

    // Bandwidth in bytes/sec
    let bandwidth_bytes_per_sec = bandwidth_gbps * 1_000_000_000.0;

    if bytes_per_token > 0.0 {
        (bandwidth_bytes_per_sec / bytes_per_token) as f32
    } else {
        0.0
    }
}

/// Calculate confidence in our estimate (0.0 - 1.0).
fn calculate_confidence(
    model: &ModelEntry,
    _quant: Quantization,
    profile: &HardwareProfile,
) -> f32 {
    let mut confidence: f32 = 0.7; // Base confidence for heuristic estimates

    // Higher confidence for well-known model sizes
    if [1.0, 3.0, 7.0, 8.0, 13.0, 14.0, 32.0, 70.0].contains(&model.params_billions) {
        confidence += 0.1;
    }

    // Higher confidence if we measured bandwidth (vs guessing)
    if profile.memory.estimated_bandwidth_gbps > 0.0 {
        confidence += 0.1;
    }

    // Lower confidence for very large models (more offloading uncertainty)
    if model.params_billions > 30.0 {
        confidence -= 0.1;
    }

    confidence.clamp(0.1, 0.95)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recommender::model_db::builtin_models;
    use crate::types::*;

    fn test_profile(ram_gb: u64, gpu_vram_gb: u64, bandwidth_gbps: f64) -> HardwareProfile {
        let gb = 1024 * 1024 * 1024;
        HardwareProfile {
            cpu: CpuInfo {
                model_name: "Test CPU".into(),
                physical_cores: 8,
                logical_cores: 16,
                isa_extensions: vec!["AVX2".into()],
                cache_size_bytes: 16 * 1024 * 1024,
            },
            memory: MemoryInfo {
                total_bytes: ram_gb * gb,
                available_bytes: (ram_gb as f64 * 0.8) as u64 * gb,
                estimated_bandwidth_gbps: bandwidth_gbps,
            },
            gpus: if gpu_vram_gb > 0 {
                vec![GpuInfo {
                    name: "Test GPU".into(),
                    vram_bytes: gpu_vram_gb * gb,
                    compute_api: ComputeApi::Vulkan,
                    driver_version: None,
                }]
            } else {
                vec![]
            },
            storage: StorageInfo {
                available_bytes: 100 * gb,
                estimated_read_speed_mbps: 500.0,
            },
            platform: PlatformInfo {
                os: "linux".into(),
                arch: "x86_64".into(),
                is_mobile: false,
                power_source: PowerSource::Ac,
            },
        }
    }

    #[test]
    fn test_recommend_4gb_ram() {
        let profile = test_profile(4, 0, 20.0);
        let models = builtin_models();
        let recs = recommend(&profile, &models, 0.85, 5.0, 10);

        // Should only recommend small models
        for rec in &recs {
            assert!(
                rec.estimated_memory_usage_bytes < 4 * 1024 * 1024 * 1024,
                "Recommended model {} uses too much memory for 4GB system",
                rec.model_name
            );
        }
    }

    #[test]
    fn test_recommend_16gb_ram_8gb_vram() {
        let profile = test_profile(16, 8, 30.0);
        let models = builtin_models();
        let recs = recommend(&profile, &models, 0.85, 5.0, 10);

        // Should recommend more and larger models
        assert!(!recs.is_empty());
        // Should have some 7-8B models
        assert!(recs
            .iter()
            .any(|r| r.model_name.contains("8B") || r.model_name.contains("7B")));
    }

    #[test]
    fn test_recommend_sorted_by_score() {
        let profile = test_profile(16, 0, 30.0);
        let models = builtin_models();
        let recs = recommend(&profile, &models, 0.85, 5.0, 10);

        for window in recs.windows(2) {
            assert!(
                window[0].score >= window[1].score,
                "Recommendations not sorted by score"
            );
        }
    }

    #[test]
    fn test_estimate_memory() {
        let model = ModelEntry {
            id: "test-7b".into(),
            name: "Test 7B".into(),
            params_billions: 7.0,
            formats: vec![],
            context_length: 4096,
            quality_score: 0.7,
            category: crate::recommender::model_db::ModelCategory::General,
            huggingface_id: None,
        };

        let mem_q4 = estimate_memory(&model, Quantization::Q4KM, 4096);
        let mem_f16 = estimate_memory(&model, Quantization::F16, 4096);

        // Q4 should use significantly less memory than F16
        assert!(mem_q4 < mem_f16);
        // Q4_K_M 7B weights are ~4GB, plus KV cache and overhead = ~5-9 GB total
        let gb = 1024 * 1024 * 1024;
        assert!(
            mem_q4 > 3 * gb && mem_q4 < 10 * gb,
            "Q4_K_M 7B memory estimate: {} GB",
            mem_q4 as f64 / gb as f64
        );
    }
}
