use serde::Serialize;

use crate::telemetry;
use hivebear_core::telemetry::{resolve, TelemetryDecision};
use hivebear_core::Config;

/// What the webview needs to configure its own reporting.
///
/// The webview cannot read the compile-time DSN or the config file, and it must
/// not make its own decision about consent — otherwise turning telemetry off
/// would silence the Rust half of the app while the JavaScript half kept
/// reporting. Rust decides; the webview is told.
#[derive(Debug, Serialize)]
pub struct TelemetryStatus {
    /// Whether reporting is on. False for an opt-out *or* a source build.
    pub enabled: bool,
    /// The DSN to report to, or `None` when disabled. DSNs are not secret.
    pub dsn: Option<String>,
    /// `desktop` or `android` — the same web bundle ships as both.
    pub platform: String,
    /// The pseudonymous per-install ID, so both halves report as one user.
    pub install_id: Option<String>,
    /// Whether the first-run notice still needs showing.
    pub notice_pending: bool,
}

/// Report the resolved telemetry state to the webview.
#[tauri::command]
pub fn telemetry_status() -> TelemetryStatus {
    let config = Config::load();
    let decision = resolve(&config.telemetry);

    TelemetryStatus {
        enabled: decision.is_enabled(),
        dsn: match &decision {
            TelemetryDecision::Enabled { dsn } => Some(dsn.clone()),
            TelemetryDecision::Disabled(_) => None,
        },
        platform: telemetry::platform().to_string(),
        install_id: config.telemetry.install_id.clone(),
        notice_pending: decision.is_enabled() && !config.telemetry.notice_shown,
    }
}

/// Record that the first-run telemetry notice has been shown.
///
/// Opt-out reporting is only defensible if people are actually told, and this is
/// a GUI app with no console — so the notice lives in the UI and this is how the
/// UI says it has done its job.
#[tauri::command]
pub fn acknowledge_telemetry_notice() -> Result<(), String> {
    let mut config = Config::load();
    config.telemetry.notice_shown = true;
    config.save().map_err(|e| e.to_string())
}

// Turning reporting on and off deliberately does NOT have its own command. It
// is an ordinary config field, so it goes through `save_config` like every other
// setting — one write path, no chance of the two disagreeing about which is
// authoritative.
