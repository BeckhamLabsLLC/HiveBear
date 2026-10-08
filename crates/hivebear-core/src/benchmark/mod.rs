pub mod extrapolator;
pub mod report;
pub mod runner;
#[cfg(not(target_arch = "wasm32"))]
pub mod submit;

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

/// Quantization tags as they appear in GGUF filenames, most specific first so
/// `Q4_K_M` is not reported as `Q4_K`, and `BF16` is not reported as `F16`.
const QUANT_TAGS: &[&str] = &[
    "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M",
    "IQ4_XS", "IQ4_NL", "Q2_K_S", "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_K_S", "Q4_K_M", "Q4_0",
    "Q4_1", "Q5_K_S", "Q5_K_M", "Q5_0", "Q5_1", "Q6_K", "Q8_0", "BF16", "F16", "F32",
];

/// Best-effort quantization of a model file, read from its filename.
///
/// GGUF files from HuggingFace are named after their quantization
/// (`qwen2.5-0.5b-instruct-q4_k_m.gguf`), and that is the string the
/// community leaderboard groups by. Returns `None` when nothing matches, so
/// callers can send an honest "unknown" rather than a guess.
pub fn detect_quantization(path: &std::path::Path) -> Option<String> {
    let name = path.file_name()?.to_string_lossy().to_ascii_uppercase();
    QUANT_TAGS
        .iter()
        .find(|tag| {
            name.match_indices(*tag).any(|(idx, _)| {
                // Require a separator (or the ends of the name) on both sides,
                // so "F16" inside "BF16" or "Q4_0" inside "Q4_0_4_4" is skipped.
                let before = name[..idx].chars().last();
                let after = name[idx + tag.len()..].chars().next();
                let is_sep =
                    |c: Option<char>| c.is_none_or(|c| !c.is_ascii_alphanumeric() && c != '_');
                is_sep(before) && is_sep(after)
            })
        })
        .map(|tag| tag.to_string())
}

#[cfg(test)]
mod tests {
    use super::detect_quantization;
    use std::path::Path;

    #[test]
    fn detects_common_gguf_names() {
        let cases = [
            ("qwen2.5-0.5b-instruct-q4_k_m.gguf", Some("Q4_K_M")),
            ("Llama-3.2-1B-Instruct-Q8_0.gguf", Some("Q8_0")),
            (
                "/models/x/Meta-Llama-3.1-8B-Instruct-Q5_K_S.gguf",
                Some("Q5_K_S"),
            ),
            ("model-bf16.gguf", Some("BF16")),
            ("model-f16.gguf", Some("F16")),
            ("model.IQ4_XS.gguf", Some("IQ4_XS")),
            ("model.gguf", None),
        ];
        for (name, expected) in cases {
            assert_eq!(
                detect_quantization(Path::new(name)).as_deref(),
                expected,
                "{name}"
            );
        }
    }
}
