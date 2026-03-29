use futures::StreamExt;
use std::time::Instant;

use crate::error::Result;
use crate::types::{ChatMessage, GenerateRequest, ModelHandle, SamplingParams};
use crate::Orchestrator;
use hivebear_core::types::BenchmarkResult;

/// Configuration for an inference benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of tokens in the prefill prompt.
    pub prefill_tokens: u32,
    /// Number of tokens to generate.
    pub generate_tokens: u32,
    /// Number of warmup iterations (not counted).
    pub warmup_runs: u32,
    /// Number of benchmark iterations to average.
    pub iterations: u32,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            prefill_tokens: 128,
            generate_tokens: 256,
            warmup_runs: 1,
            iterations: 3,
        }
    }
}

/// Standard prefill prompts of varying lengths for reproducible benchmarks.
fn prefill_prompt(target_tokens: u32) -> String {
    // ~4 chars per token on average. Build a prompt that approximates the target length.
    let base = "The following is a detailed analysis of modern computer architecture, \
                covering topics from CPU design to memory hierarchies, cache coherence \
                protocols, branch prediction, out-of-order execution, and the interaction \
                between hardware and software optimization techniques. ";
    let repetitions = (target_tokens as usize * 4) / base.len() + 1;
    base.repeat(repetitions)
}

/// Run a real inference benchmark using the orchestrator and a loaded model.
pub async fn run_inference_benchmark(
    orchestrator: &Orchestrator,
    handle: &ModelHandle,
    model_name: &str,
    config: &BenchmarkConfig,
) -> Result<BenchmarkResult> {
    let prompt = prefill_prompt(config.prefill_tokens);
    let messages = vec![ChatMessage::user_text(&prompt)];

    let req = GenerateRequest {
        messages: messages.clone(),
        max_tokens: config.generate_tokens,
        sampling: SamplingParams {
            temperature: 0.0, // Deterministic for reproducibility
            top_p: 1.0,
            ..Default::default()
        },
        ..Default::default()
    };

    // Warmup runs
    for _ in 0..config.warmup_runs {
        let mut stream = orchestrator.stream(handle, &req)?;
        while stream.next().await.is_some() {}
    }

    // Measure memory before
    let mem_before = get_process_memory();

    let mut total_ttft_ms = 0u128;
    let mut total_gen_duration_ms = 0u128;
    let mut total_tokens = 0u32;
    let mut total_prefill_ms = 0u128;

    for _ in 0..config.iterations {
        let start = Instant::now();
        let mut first_token_time: Option<Instant> = None;
        let mut tokens_this_run = 0u32;

        let mut stream = orchestrator.stream(handle, &req)?;
        while let Some(result) = stream.next().await {
            if result.is_ok() {
                if first_token_time.is_none() {
                    first_token_time = Some(Instant::now());
                }
                tokens_this_run += 1;
            }
        }

        let end = Instant::now();
        let run_duration = end.duration_since(start);

        if let Some(ftt) = first_token_time {
            let ttft = ftt.duration_since(start);
            total_ttft_ms += ttft.as_millis();
            total_prefill_ms += ttft.as_millis();

            // Generation time is from first token to end
            let gen_time = end.duration_since(ftt);
            total_gen_duration_ms += gen_time.as_millis();
        } else {
            total_gen_duration_ms += run_duration.as_millis();
        }

        total_tokens += tokens_this_run;
    }

    let mem_after = get_process_memory();
    let peak_memory = mem_after.max(mem_before);

    let avg_ttft_ms = total_ttft_ms / config.iterations as u128;
    let avg_tokens = total_tokens / config.iterations;
    let avg_gen_duration_ms = total_gen_duration_ms / config.iterations as u128;

    // Generate tokens per second (excludes prefill)
    let gen_tok_s = if avg_gen_duration_ms > 0 {
        (avg_tokens as f64 / avg_gen_duration_ms as f64) * 1000.0
    } else {
        0.0
    };

    // Prompt eval tokens per second
    let prompt_eval_tok_s = if total_prefill_ms > 0 {
        let avg_prefill_ms = total_prefill_ms / config.iterations as u128;
        if avg_prefill_ms > 0 {
            Some((config.prefill_tokens as f64 / avg_prefill_ms as f64 * 1000.0) as f32)
        } else {
            None
        }
    } else {
        None
    };

    let total_avg_duration_ms = avg_ttft_ms + avg_gen_duration_ms;

    Ok(BenchmarkResult {
        model_used: model_name.to_string(),
        tokens_generated: avg_tokens,
        total_duration_ms: total_avg_duration_ms as u64,
        tokens_per_sec: gen_tok_s as f32,
        time_to_first_token_ms: avg_ttft_ms as u64,
        peak_memory_bytes: peak_memory,
        cpu_utilization: 0.0, // Not easily measurable per-run
        gpu_utilization: None,
        benchmark_type: "inference".to_string(),
        prompt_eval_tokens_per_sec: prompt_eval_tok_s,
    })
}

/// Get current process RSS memory in bytes.
fn get_process_memory() -> u64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    if let Some(kb_str) = line.split_whitespace().nth(1) {
                        if let Ok(kb) = kb_str.parse::<u64>() {
                            return kb * 1024;
                        }
                    }
                }
            }
        }
        0
    }
    #[cfg(not(target_os = "linux"))]
    {
        0 // TODO: macOS/Windows memory measurement
    }
}
