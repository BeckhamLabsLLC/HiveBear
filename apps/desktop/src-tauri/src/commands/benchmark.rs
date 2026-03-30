use crate::error::CmdResult;
use crate::state::AppState;
use hivebear_core::types::{BenchmarkResult, CommunityBenchmarkSummary, ProfileMode};
use hivebear_core::{CommunityBenchmarkSubmission, HardwareFingerprint};
use tauri::State;

#[tauri::command]
pub async fn run_benchmark(duration_secs: Option<u32>) -> CmdResult<Option<BenchmarkResult>> {
    let mode = ProfileMode::Benchmark {
        duration_secs: duration_secs.unwrap_or(30),
    };
    tokio::task::spawn_blocking(move || hivebear_core::benchmark::run_benchmark(mode))
        .await
        .map_err(|e| format!("Benchmark task failed: {e}"))
}

/// Share an anonymized benchmark result with the community.
#[tauri::command]
pub async fn share_benchmark(
    state: State<'_, AppState>,
    result: BenchmarkResult,
    model_id: String,
    quantization: String,
    engine: String,
) -> CmdResult<bool> {
    let config = state
        .config
        .lock()
        .map_err(|_| String::from("Config lock poisoned"))?
        .clone();

    if !config.share_benchmarks {
        return Ok(false);
    }

    let fp = HardwareFingerprint::from_profile(&state.profile);
    let submission = CommunityBenchmarkSubmission {
        hardware_fingerprint: fp,
        model_id,
        quantization,
        engine,
        context_length: 4096,
        benchmark_type: result.benchmark_type,
        tokens_per_sec: result.tokens_per_sec,
        time_to_first_token_ms: Some(result.time_to_first_token_ms),
        prompt_eval_tokens_per_sec: result.prompt_eval_tokens_per_sec,
        peak_memory_bytes: result.peak_memory_bytes,
        client_version: env!("CARGO_PKG_VERSION").to_string(),
    };

    let url = format!("{}/benchmarks", config.mesh.coordination_server);
    let mut req = state.http_client.post(&url).json(&submission);
    if let Some(ref token) = config.account.jwt_token {
        req = req.header("Authorization", format!("Bearer {token}"));
    }

    match req.send().await {
        Ok(resp) if resp.status().is_success() => Ok(true),
        _ => Ok(false),
    }
}

/// Fetch community benchmark data for the user's hardware profile.
#[tauri::command]
pub async fn get_community_benchmarks(
    state: State<'_, AppState>,
    model_id: Option<String>,
) -> CmdResult<Vec<CommunityBenchmarkSummary>> {
    let config = state
        .config
        .lock()
        .map_err(|_| String::from("Config lock poisoned"))?
        .clone();

    let fp = HardwareFingerprint::from_profile(&state.profile);
    let mut url = format!(
        "{}/benchmarks?gpu_class={}&ram_gb_bucket={}&platform_arch={}",
        config.mesh.coordination_server, fp.gpu_class, fp.ram_gb_bucket, fp.platform_arch
    );
    if let Some(ref mid) = model_id {
        url.push_str(&format!("&model_id={mid}"));
    }

    #[derive(serde::Deserialize)]
    struct QueryResponse {
        results: Vec<CommunityBenchmarkSummary>,
    }

    match state.http_client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => match resp.json::<QueryResponse>().await {
            Ok(data) => Ok(data.results),
            Err(_) => Ok(vec![]),
        },
        _ => Ok(vec![]),
    }
}
