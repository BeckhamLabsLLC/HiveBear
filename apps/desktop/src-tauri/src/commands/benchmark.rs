use crate::error::CmdResult;
use crate::state::AppState;
use hivebear_core::benchmark::submit;
use hivebear_core::types::{BenchmarkResult, CommunityBenchmarkSummary, ProfileMode};
use hivebear_core::HardwareFingerprint;
use hivebear_inference::benchmark::BenchmarkConfig;
use serde::Serialize;
use tauri::State;

/// Synthetic CPU estimate, for when no model is installed. Never shareable:
/// it measures a matmul loop, not a model.
#[tauri::command]
pub async fn run_benchmark(duration_secs: Option<u32>) -> CmdResult<Option<BenchmarkResult>> {
    let mode = ProfileMode::Benchmark {
        duration_secs: duration_secs.unwrap_or(30),
    };
    tokio::task::spawn_blocking(move || hivebear_core::benchmark::run_benchmark(mode))
        .await
        .map_err(|e| format!("Benchmark task failed: {e}"))
}

/// A real benchmark of an installed model, with what sharing it needs.
#[derive(Serialize)]
pub struct ModelBenchmarkResult {
    pub result: BenchmarkResult,
    pub model_id: String,
    /// The real quantization, or "unknown". Never "auto".
    pub quantization: String,
    pub engine: String,
    pub context_length: u32,
}

/// Load an installed model, benchmark it, and unload it.
///
/// Same measurement as `hivebear benchmark --model`, via the shared
/// `hivebear_inference::benchmark::benchmark_model_file`.
#[tauri::command]
pub async fn run_model_benchmark(
    state: State<'_, AppState>,
    model_id: String,
) -> CmdResult<ModelBenchmarkResult> {
    let path = state
        .registry
        .resolve(&model_id)
        .await
        .map_err(|e| format!("Could not find installed model '{model_id}': {e}"))?;

    // Prefer what the registry recorded at install time; fall back to the
    // filename, which is where that came from in the first place.
    let recorded_quant = state
        .registry
        .list_installed()
        .await
        .into_iter()
        .find(|m| m.id == model_id)
        .and_then(|m| m.installed)
        .and_then(|i| i.quantization)
        .map(|q| q.to_string());

    // Fewer tokens than the CLI default so the desktop run is ~30s, not minutes.
    let config = BenchmarkConfig {
        prefill_tokens: 128,
        generate_tokens: 128,
        warmup_runs: 1,
        iterations: 2,
    };

    let bench = hivebear_inference::benchmark::benchmark_model_file(
        &state.orchestrator,
        &path,
        &model_id,
        &config,
    )
    .await
    .map_err(|e| String::from(crate::error::CommandError::from(e)))?;

    Ok(ModelBenchmarkResult {
        quantization: recorded_quant
            .or(bench.quantization)
            .unwrap_or_else(|| "unknown".to_string()),
        engine: bench.engine,
        context_length: bench.context_length,
        model_id,
        result: bench.result,
    })
}

/// Share a real benchmark result to the community leaderboard.
///
/// Clicking Share is the consent, so this does not consult
/// `share_benchmarks` (that setting only controls automatic sharing).
/// Anonymous unless signed in. Errors are returned, not swallowed: this used
/// to answer `Ok(false)` for every failure, and the UI could not say why.
#[tauri::command]
pub async fn share_benchmark(
    state: State<'_, AppState>,
    result: BenchmarkResult,
    model_id: String,
    quantization: String,
    engine: String,
    context_length: Option<u32>,
) -> CmdResult<bool> {
    let config = state
        .config
        .lock()
        .map_err(|_| String::from("Config lock poisoned"))?
        .clone();

    let submission = submit::build_submission(
        &result,
        &state.profile,
        &model_id,
        &quantization,
        &engine,
        context_length.unwrap_or(hivebear_inference::benchmark::BENCHMARK_CONTEXT_LENGTH),
    );

    submit::submit_benchmark(&config, &submission)
        .await
        .map_err(|e| format!("Could not share benchmark: {e}"))?;
    Ok(true)
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
