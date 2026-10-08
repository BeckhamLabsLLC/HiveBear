//! Submitting a benchmark result to the community leaderboard.
//!
//! Shared by the CLI and the desktop app so the two cannot drift on what they
//! send. Submissions are anonymous by default: the coordinator accepts them
//! without an account, keyed by the random per-install ID. A bearer token is
//! attached only when the user is signed in, which lets the coordinator link
//! the result to their account.

use std::time::Duration;

use crate::config::Config;
use crate::fingerprint::HardwareFingerprint;
use crate::types::{BenchmarkResult, CommunityBenchmarkSubmission, HardwareProfile};

/// Where people can see shared results.
pub const LEADERBOARD_URL: &str = "https://hivebear.com/benchmarks";

/// Long enough for a slow uplink, short enough that a CLI user is not left
/// staring at a hung prompt when the coordinator is down.
const SUBMIT_TIMEOUT: Duration = Duration::from_secs(15);

/// Why a submission did not go through, worded for the person who asked.
#[derive(Debug, thiserror::Error)]
pub enum SubmitError {
    #[error("only real model benchmarks can be shared, not the synthetic estimate")]
    NotShareable,
    #[error("could not reach {server}: {source}")]
    Network {
        server: String,
        #[source]
        source: reqwest::Error,
    },
    #[error("the leaderboard rejected the result ({status}){}", fmt_body(.body))]
    Rejected { status: u16, body: String },
}

fn fmt_body(body: &str) -> String {
    let body = body.trim();
    if body.is_empty() {
        String::new()
    } else {
        format!(": {body}")
    }
}

/// Build the anonymized submission for a real inference benchmark.
///
/// `quantization` should be the model's actual quantization (for example from
/// [`super::detect_quantization`]); pass `"unknown"` rather than guessing.
pub fn build_submission(
    result: &BenchmarkResult,
    hw: &HardwareProfile,
    model_id: &str,
    quantization: &str,
    engine: &str,
    context_length: u32,
) -> CommunityBenchmarkSubmission {
    CommunityBenchmarkSubmission {
        hardware_fingerprint: HardwareFingerprint::from_profile(hw),
        model_id: model_id.to_string(),
        quantization: quantization.to_string(),
        engine: engine.to_string(),
        context_length,
        benchmark_type: result.benchmark_type.clone(),
        tokens_per_sec: result.tokens_per_sec,
        time_to_first_token_ms: Some(result.time_to_first_token_ms),
        prompt_eval_tokens_per_sec: result.prompt_eval_tokens_per_sec,
        peak_memory_bytes: result.peak_memory_bytes,
        client_version: env!("CARGO_PKG_VERSION").to_string(),
        install_id: Some(crate::usage::install_id()),
    }
}

/// POST a submission to `{coordination_server}/benchmarks`.
///
/// Fills in `install_id` if the caller left it empty. On success, also sends
/// the `benchmark_shared` usage count (subject to the user's setting).
pub async fn submit_benchmark(
    config: &Config,
    submission: &CommunityBenchmarkSubmission,
) -> Result<(), SubmitError> {
    if submission.benchmark_type != "inference" {
        return Err(SubmitError::NotShareable);
    }

    let mut submission = submission.clone();
    if submission.install_id.is_none() {
        submission.install_id = Some(crate::usage::install_id());
    }

    let server = config.mesh.coordination_server.trim_end_matches('/');
    let url = format!("{server}/benchmarks");
    let client = reqwest::Client::builder()
        .timeout(SUBMIT_TIMEOUT)
        .build()
        .map_err(|source| SubmitError::Network {
            server: server.to_string(),
            source,
        })?;

    let mut req = client.post(&url).json(&submission);
    if let Some(token) = config.account.jwt_token.as_deref() {
        if may_send_token(&url) {
            req = req.bearer_auth(token);
        } else {
            // Still submitted, just anonymously.
            tracing::warn!(
                "Not sending account token to {server}: it is not HTTPS. Submitting anonymously."
            );
        }
    }

    let resp = req.send().await.map_err(|source| SubmitError::Network {
        server: server.to_string(),
        source,
    })?;

    let status = resp.status();
    if !status.is_success() {
        let mut body = resp.text().await.unwrap_or_default();
        body.truncate(300);
        return Err(SubmitError::Rejected {
            status: status.as_u16(),
            body,
        });
    }

    crate::usage::send_usage_event(crate::usage::EVENT_BENCHMARK_SHARED).await;
    Ok(())
}

/// Whether the account token may be attached to a request to `url`.
///
/// Only over HTTPS, or plain HTTP to the local machine for development. A
/// misconfigured or hostile `coordination_server` must not receive the
/// user's JWT in cleartext.
fn may_send_token(url: &str) -> bool {
    let Ok(parsed) = reqwest::Url::parse(url) else {
        return false;
    };
    match parsed.scheme() {
        "https" => true,
        "http" => matches!(parsed.host_str(), Some("localhost" | "127.0.0.1" | "[::1]")),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_only_goes_over_https_or_to_localhost() {
        assert!(may_send_token("https://mesh.hivebear.com/benchmarks"));
        assert!(may_send_token("http://localhost:7879/benchmarks"));
        assert!(may_send_token("http://127.0.0.1:7879/benchmarks"));
        assert!(may_send_token("http://[::1]:7879/benchmarks"));
        assert!(!may_send_token("http://mesh.hivebear.com/benchmarks"));
        assert!(!may_send_token("http://localhost.evil.example/benchmarks"));
        assert!(!may_send_token("http://10.0.0.5:7879/benchmarks"));
        assert!(!may_send_token("ftp://mesh.hivebear.com/benchmarks"));
        assert!(!may_send_token("not a url"));
    }

    #[test]
    fn rejected_error_includes_body_only_when_present() {
        let with = SubmitError::Rejected {
            status: 400,
            body: "bad quantization".into(),
        };
        assert_eq!(
            with.to_string(),
            "the leaderboard rejected the result (400): bad quantization"
        );
        let without = SubmitError::Rejected {
            status: 503,
            body: String::new(),
        };
        assert_eq!(
            without.to_string(),
            "the leaderboard rejected the result (503)"
        );
    }
}
