use crate::error::CmdResult;
use hivebear_core::types::{BenchmarkResult, ProfileMode};

#[tauri::command]
pub async fn run_benchmark(duration_secs: Option<u32>) -> CmdResult<Option<BenchmarkResult>> {
    let mode = ProfileMode::Benchmark {
        duration_secs: duration_secs.unwrap_or(30),
    };
    tokio::task::spawn_blocking(move || hivebear_core::benchmark::run_benchmark(mode))
        .await
        .map_err(|e| format!("Benchmark task failed: {e}"))
}
