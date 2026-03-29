pub mod extrapolator;
pub mod report;
pub mod runner;

use crate::types::{BenchmarkResult, ProfileMode};

/// Run a benchmark based on the profile mode.
///
/// In `Estimate` mode, returns `None` (use the recommender's estimates instead).
/// In `Benchmark` mode, runs real inference on a bundled micro-model.
pub fn run_benchmark(mode: ProfileMode) -> Option<BenchmarkResult> {
    match mode {
        ProfileMode::Estimate => None,
        ProfileMode::Benchmark { duration_secs } => {
            tracing::info!("Running inference benchmark ({duration_secs}s)...");
            match runner::run(duration_secs) {
                Ok(result) => {
                    tracing::info!("Benchmark complete: {:.1} tok/s", result.tokens_per_sec);
                    Some(result)
                }
                Err(e) => {
                    tracing::error!("Benchmark failed: {e}");
                    None
                }
            }
        }
    }
}
