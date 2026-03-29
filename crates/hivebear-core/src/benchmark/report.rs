use crate::types::{format_bytes, BenchmarkResult};

/// Format a benchmark result for terminal display.
pub fn format_report(result: &BenchmarkResult) -> String {
    let mut report = String::new();

    report.push_str(&format!("Model: {}\n", result.model_used));
    report.push_str(&format!("Tokens generated: {}\n", result.tokens_generated));
    report.push_str(&format!(
        "Total duration: {:.1}s\n",
        result.total_duration_ms as f64 / 1000.0
    ));
    report.push_str(&format!("Tokens/sec: {:.1}\n", result.tokens_per_sec));

    if result.time_to_first_token_ms > 0 {
        report.push_str(&format!(
            "Time to first token: {}ms\n",
            result.time_to_first_token_ms
        ));
    }

    report.push_str(&format!(
        "Peak memory: {}\n",
        format_bytes(result.peak_memory_bytes)
    ));
    report.push_str(&format!(
        "CPU utilization: {:.0}%\n",
        result.cpu_utilization
    ));

    if let Some(gpu_util) = result.gpu_utilization {
        report.push_str(&format!("GPU utilization: {:.0}%\n", gpu_util));
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_report() {
        let result = BenchmarkResult {
            model_used: "test-model".into(),
            tokens_generated: 100,
            total_duration_ms: 5000,
            tokens_per_sec: 20.0,
            time_to_first_token_ms: 200,
            peak_memory_bytes: 1024 * 1024 * 1024,
            cpu_utilization: 75.0,
            gpu_utilization: Some(85.0),
            benchmark_type: "synthetic".to_string(),
            prompt_eval_tokens_per_sec: None,
        };

        let report = format_report(&result);
        assert!(report.contains("test-model"));
        assert!(report.contains("20.0"));
        assert!(report.contains("1.0 GB"));
        assert!(report.contains("85%"));
    }
}
