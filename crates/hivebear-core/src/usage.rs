//! Anonymous usage counts: did people get as far as a first chat?
//!
//! Crash reports (see [`crate::telemetry`]) say what broke. They cannot say
//! that 200 people installed HiveBear and nobody got a reply out of a model.
//! These counts can, and they are deliberately the least that answers it:
//!
//! * One request per event, `{event, version, os, install_id}`. No prompts, no
//!   model names, no hardware, no account. The coordinator does not store IPs.
//! * Only a fixed set of events, most of them sent at most once per install.
//! * Off when `telemetry.usage_events = false`, `HIVEBEAR_TELEMETRY=0` or
//!   `DO_NOT_TRACK=1`.
//! * Fire-and-forget with a short timeout. A failure is logged at debug level
//!   and otherwise ignored; nothing here can slow down or break the app.

use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::config::paths::AppPaths;
use crate::config::Config;
use crate::telemetry::{env_flag, DO_NOT_TRACK_ENV, TELEMETRY_ENV};

/// The desktop app opened for the first time on this install.
pub const EVENT_FIRST_LAUNCH: &str = "first_launch";
/// A model finished downloading.
pub const EVENT_MODEL_INSTALLED: &str = "model_installed";
/// The first successful local response on this install.
pub const EVENT_FIRST_INFERENCE: &str = "first_inference";
/// A benchmark was shared to the leaderboard.
pub const EVENT_BENCHMARK_SHARED: &str = "benchmark_shared";

const SEND_TIMEOUT: Duration = Duration::from_secs(3);
const INSTALL_ID_FILE: &str = "install_id";
const MARKERS_FILE: &str = "usage_events_sent";

#[derive(serde::Serialize)]
struct UsageEvent<'a> {
    event: &'a str,
    version: &'a str,
    os: &'a str,
    install_id: String,
}

/// The random, stable ID for this install.
///
/// Stored in its own file in the config directory rather than in
/// `config.toml`: the desktop app keeps a copy of the config in memory and
/// writes the whole thing back from Settings, which would silently replace an
/// ID minted behind its back. Seeded from `telemetry.install_id` when that
/// already exists, so crash reports and usage counts agree on who is who.
pub fn install_id() -> String {
    let paths = AppPaths::new();
    let seed = Config::load().telemetry.install_id;
    install_id_in(&paths.config_dir, seed.as_deref())
}

fn install_id_in(dir: &Path, seed: Option<&str>) -> String {
    let file = dir.join(INSTALL_ID_FILE);
    if let Ok(existing) = std::fs::read_to_string(&file) {
        let existing = existing.trim();
        if !existing.is_empty() {
            return existing.to_string();
        }
    }

    let id = seed
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    // Best-effort. If the write fails the next call mints another ID, which
    // costs some accuracy in the counts and nothing else.
    if let Err(e) = std::fs::create_dir_all(dir).and_then(|_| std::fs::write(&file, &id)) {
        tracing::debug!("Could not persist install ID at {}: {e}", file.display());
    }
    id
}

/// Whether usage counts may be sent, given this config and the environment.
pub fn usage_events_enabled(config: &Config) -> bool {
    if env_flag(TELEMETRY_ENV) == Some(false) {
        return false;
    }
    if env_flag(TELEMETRY_ENV).is_none() && env_flag(DO_NOT_TRACK_ENV).unwrap_or(false) {
        return false;
    }
    config.telemetry.usage_events
}

/// Send one usage count to `{coordination_server}/telemetry/event`.
///
/// Returns once the request finishes or times out (3s). Callers that must
/// not wait should spawn it.
pub async fn send_usage_event(event: &str) {
    let config = Config::load();
    if !usage_events_enabled(&config) {
        return;
    }
    send(&config, event).await;
}

/// Like [`send_usage_event`], but at most once per install.
///
/// The marker is written only once the coordinator has answered. It used to
/// be written first, so one offline first run, or a send that timed out while
/// the coordinator woke from suspension, lost the event for good. Now a
/// failed send is retried the next time the event fires; the coordinator
/// dedupes these events per install, so a double send is harmless.
pub async fn send_usage_event_once(event: &str) {
    let config = Config::load();
    if !usage_events_enabled(&config) {
        return;
    }
    let dir = AppPaths::new().config_dir;
    if already_sent_in(&dir, event) {
        return;
    }
    if send(&config, event).await {
        mark_sent_in(&dir, event);
    }
}

/// Returns `true` when the event needs no retry: the coordinator accepted it,
/// or rejected it in a way a retry can't fix (any 4xx but 429).
async fn send(config: &Config, event: &str) -> bool {
    let server = config.mesh.coordination_server.trim_end_matches('/');
    let url = format!("{server}/telemetry/event");
    let body = UsageEvent {
        event,
        version: env!("CARGO_PKG_VERSION"),
        os: std::env::consts::OS,
        install_id: install_id(),
    };

    let client = match reqwest::Client::builder().timeout(SEND_TIMEOUT).build() {
        Ok(c) => c,
        Err(e) => {
            tracing::debug!("Usage event '{event}' not sent: {e}");
            return false;
        }
    };
    match client.post(&url).json(&body).send().await {
        Ok(resp) if resp.status().is_success() => true,
        Ok(resp) => {
            let status = resp.status();
            tracing::debug!("Usage event '{event}' returned {status}");
            status.is_client_error() && status != reqwest::StatusCode::TOO_MANY_REQUESTS
        }
        Err(e) => {
            tracing::debug!("Usage event '{event}' not sent: {e}");
            false
        }
    }
}

/// Whether `event` is already in the marker file.
fn already_sent_in(dir: &Path, event: &str) -> bool {
    std::fs::read_to_string(dir.join(MARKERS_FILE))
        .unwrap_or_default()
        .lines()
        .any(|line| line.trim() == event)
}

/// Record `event` in the marker file (no-op if it's already there).
fn mark_sent_in(dir: &Path, event: &str) {
    let file: PathBuf = dir.join(MARKERS_FILE);
    let existing = std::fs::read_to_string(&file).unwrap_or_default();
    if existing.lines().any(|line| line.trim() == event) {
        return;
    }
    let mut updated = existing;
    if !updated.is_empty() && !updated.ends_with('\n') {
        updated.push('\n');
    }
    updated.push_str(event);
    updated.push('\n');
    if let Err(e) = std::fs::create_dir_all(dir).and_then(|_| std::fs::write(&file, updated)) {
        tracing::debug!("Could not record usage marker at {}: {e}", file.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hivebear-usage-test-{name}-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn install_id_is_stable_once_written() {
        let dir = scratch_dir("stable");
        let first = install_id_in(&dir, None);
        assert!(!first.is_empty());
        assert_eq!(install_id_in(&dir, None), first);
        // A seed only matters before the file exists.
        assert_eq!(install_id_in(&dir, Some("other")), first);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn install_id_reuses_the_crash_reporting_id() {
        let dir = scratch_dir("seed");
        assert_eq!(install_id_in(&dir, Some("abc-123")), "abc-123");
        assert_eq!(install_id_in(&dir, None), "abc-123");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn once_markers_fire_once_per_event() {
        let dir = scratch_dir("markers");
        assert!(!already_sent_in(&dir, EVENT_FIRST_LAUNCH));
        mark_sent_in(&dir, EVENT_FIRST_LAUNCH);
        assert!(already_sent_in(&dir, EVENT_FIRST_LAUNCH));
        assert!(!already_sent_in(&dir, EVENT_FIRST_INFERENCE));
        mark_sent_in(&dir, EVENT_FIRST_INFERENCE);
        mark_sent_in(&dir, EVENT_FIRST_INFERENCE);
        assert!(already_sent_in(&dir, EVENT_FIRST_INFERENCE));
        assert!(already_sent_in(&dir, EVENT_FIRST_LAUNCH));
        let markers = std::fs::read_to_string(dir.join(MARKERS_FILE)).unwrap();
        assert_eq!(markers.lines().count(), 2);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn config_opt_out_disables_usage_events() {
        let mut config = Config::default();
        config.telemetry.usage_events = false;
        assert!(!usage_events_enabled(&config));
    }
}
