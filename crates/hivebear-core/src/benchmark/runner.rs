use crate::types::BenchmarkResult;

/// Run inference benchmark using a bundled micro-model.
///
/// Currently returns a placeholder. Full implementation requires the
/// `llama-cpp-rs` dependency and a bundled model file.
///
/// TODO(Phase 2): Integrate with the inference orchestrator for real benchmarks.
pub fn run(duration_secs: u32) -> Result<BenchmarkResult, BenchmarkError> {
    tracing::warn!(
        "Full inference benchmark not yet available. \
         This requires a bundled model and inference engine (Phase 2). \
         Running CPU throughput estimation instead."
    );

    // For now, run a synthetic compute benchmark that estimates
    // matrix multiplication throughput as a proxy for inference speed.
    let result = synthetic_benchmark(duration_secs)?;
    Ok(result)
}

/// Synthetic benchmark: estimate compute throughput via matrix operations.
/// This provides a rough proxy for inference performance on CPU.
fn synthetic_benchmark(duration_secs: u32) -> Result<BenchmarkResult, BenchmarkError> {
    const MATRIX_SIZE: usize = 512;

    let a: Vec<f32> = (0..MATRIX_SIZE * MATRIX_SIZE)
        .map(|i| (i as f32) * 0.001)
        .collect();
    let b: Vec<f32> = (0..MATRIX_SIZE * MATRIX_SIZE)
        .map(|i| (i as f32) * 0.002)
        .collect();
    let mut c = vec![0.0f32; MATRIX_SIZE * MATRIX_SIZE];

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(duration_secs as u64);
    let start = std::time::Instant::now();
    let mut iterations = 0u32;

    while std::time::Instant::now() < deadline {
        // Naive matmul — tests raw CPU throughput
        for i in 0..MATRIX_SIZE {
            for j in 0..MATRIX_SIZE {
                let mut sum = 0.0f32;
                for k in 0..MATRIX_SIZE {
                    sum += a[i * MATRIX_SIZE + k] * b[k * MATRIX_SIZE + j];
                }
                c[i * MATRIX_SIZE + j] = sum;
            }
        }
        iterations += 1;
    }

    let elapsed = start.elapsed();
    std::hint::black_box(&c);

    // Estimate "equivalent tokens per second" from compute throughput.
    // A rough heuristic: each iteration is ~MATRIX_SIZE^3 * 2 FLOPs.
    // A 7B model at Q4_K_M needs ~7 GFLOPs per token on CPU.
    let flops_per_iter = (MATRIX_SIZE as f64).powi(3) * 2.0;
    let total_flops = flops_per_iter * iterations as f64;
    let gflops = total_flops / elapsed.as_secs_f64() / 1e9;

    // Very rough conversion: ~1 GFLOP/s ≈ 1 tok/s for a 7B Q4 model
    let estimated_tok_s = (gflops / 7.0) as f32;

    Ok(BenchmarkResult {
        model_used: format!("synthetic-matmul-{MATRIX_SIZE}x{MATRIX_SIZE}"),
        tokens_generated: iterations,
        total_duration_ms: elapsed.as_millis() as u64,
        tokens_per_sec: estimated_tok_s,
        time_to_first_token_ms: 0,
        peak_memory_bytes: (MATRIX_SIZE * MATRIX_SIZE * 4 * 3) as u64,
        cpu_utilization: 100.0,
        gpu_utilization: None,
        benchmark_type: "synthetic".to_string(),
        prompt_eval_tokens_per_sec: None,
    })
}

#[derive(Debug, thiserror::Error)]
pub enum BenchmarkError {
    #[error("Benchmark timed out")]
    Timeout,
    #[error("Model not found: {0}")]
    ModelNotFound(String),
    #[error("Inference engine error: {0}")]
    EngineError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_benchmark() {
        let result = synthetic_benchmark(1).unwrap();
        assert!(result.tokens_per_sec > 0.0);
        assert!(result.total_duration_ms >= 900); // At least ~1 second
    }
}
