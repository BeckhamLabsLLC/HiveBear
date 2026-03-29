use crate::types::BenchmarkResult;

/// Given a benchmark result from a small model, extrapolate expected
/// performance for larger models.
///
/// LLM inference during token generation is memory-bandwidth-bound.
/// Performance scales roughly inversely with model size (in bytes):
///   tok/s ∝ bandwidth / model_size_bytes
///
/// If we benchmark a 1B model and get X tok/s, a 7B model at the same
/// quantization should get approximately X/7 tok/s.
pub fn extrapolate(
    benchmark: &BenchmarkResult,
    benchmark_model_params_b: f64,
    target_model_params_b: f64,
    target_quant_bits: f64,
    benchmark_quant_bits: f64,
) -> ExtrapolatedResult {
    let size_ratio = (target_model_params_b * target_quant_bits)
        / (benchmark_model_params_b * benchmark_quant_bits);

    let estimated_tok_s = benchmark.tokens_per_sec / size_ratio as f32;

    // TTFT scales roughly with model size (prefill is compute-bound)
    let estimated_ttft_ms = (benchmark.time_to_first_token_ms as f64 * size_ratio.sqrt()) as u64;

    // Memory scales linearly with model size and quant bits
    let estimated_memory_bytes = (target_model_params_b * 1e9 * target_quant_bits / 8.0) as u64;

    ExtrapolatedResult {
        estimated_tokens_per_sec: estimated_tok_s,
        estimated_ttft_ms,
        estimated_memory_bytes,
        confidence: calculate_extrapolation_confidence(size_ratio),
    }
}

/// Confidence decreases as the extrapolation distance increases.
fn calculate_extrapolation_confidence(size_ratio: f64) -> f32 {
    if size_ratio <= 2.0 {
        0.85
    } else if size_ratio <= 5.0 {
        0.70
    } else if size_ratio <= 10.0 {
        0.55
    } else if size_ratio <= 50.0 {
        0.40
    } else {
        0.25
    }
}

pub struct ExtrapolatedResult {
    pub estimated_tokens_per_sec: f32,
    pub estimated_ttft_ms: u64,
    pub estimated_memory_bytes: u64,
    pub confidence: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::BenchmarkResult;

    fn sample_benchmark() -> BenchmarkResult {
        BenchmarkResult {
            model_used: "test-1b-q4km".into(),
            tokens_generated: 100,
            total_duration_ms: 5000,
            tokens_per_sec: 20.0,
            time_to_first_token_ms: 200,
            peak_memory_bytes: 700_000_000,
            cpu_utilization: 80.0,
            gpu_utilization: None,
            benchmark_type: "synthetic".to_string(),
            prompt_eval_tokens_per_sec: None,
        }
    }

    #[test]
    fn test_extrapolate_7b() {
        let bench = sample_benchmark();
        let result = extrapolate(&bench, 1.0, 7.0, 4.5, 4.5);

        // 7x larger model should be ~7x slower
        assert!(result.estimated_tokens_per_sec < 5.0);
        assert!(result.estimated_tokens_per_sec > 1.0);
    }

    #[test]
    fn test_extrapolate_same_model() {
        let bench = sample_benchmark();
        let result = extrapolate(&bench, 1.0, 1.0, 4.5, 4.5);

        // Same model should give same tok/s
        assert!((result.estimated_tokens_per_sec - bench.tokens_per_sec).abs() < 0.1);
        assert!(result.confidence > 0.8);
    }

    #[test]
    fn test_confidence_decreases_with_distance() {
        let bench = sample_benchmark();
        let close = extrapolate(&bench, 1.0, 1.5, 4.5, 4.5);
        let far = extrapolate(&bench, 1.0, 70.0, 4.5, 4.5);

        assert!(close.confidence > far.confidence);
    }
}
