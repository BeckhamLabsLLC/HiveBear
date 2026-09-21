//! Anonymous crash and error reporting settings, shared by every HiveBear binary.
//!
//! HiveBear runs on other people's machines, so the rules here are deliberately
//! strict and deliberately boring:
//!
//! * Reporting is **opt-out**. It is on by default in the binaries we ship, and a
//!   single environment variable or one line of config turns it off for good.
//! * The DSN is baked in at compile time. Anyone who builds from source — every
//!   contributor, every distro packager — gets a binary with no DSN, which cannot
//!   report anything even if the setting says otherwise.
//! * `DO_NOT_TRACK` is honoured, because it costs nothing and it is the
//!   convention people already expect.
//! * The only identifier attached to a report is a random per-install UUID. It is
//!   not derived from hardware, from the mesh identity, or from anything about the
//!   person. It exists so we can tell "one person hitting this 400 times" apart
//!   from "400 people hitting it once", and for nothing else.

use serde::{Deserialize, Serialize};

/// Set to `0`, `false`, `no` or `off` to disable reporting entirely.
pub const TELEMETRY_ENV: &str = "HIVEBEAR_TELEMETRY";

/// Cross-vendor opt-out convention (<https://consoledonottrack.com>).
pub const DO_NOT_TRACK_ENV: &str = "DO_NOT_TRACK";

/// Overrides the compile-time DSN. Mostly useful for self-hosted Sentry.
pub const DSN_ENV: &str = "HIVEBEAR_SENTRY_DSN";

/// Crash/error reporting settings, persisted in the HiveBear config file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    /// Whether to send anonymous crash and error reports.
    ///
    /// Defaults to `true`. See the module docs for why this is opt-out rather
    /// than opt-in, and for everything that constrains what may be sent.
    #[serde(default = "crate::config::default_true")]
    pub enabled: bool,

    /// Random, stable identifier for this install, generated on first use.
    ///
    /// `None` until [`TelemetryConfig::ensure_install_id`] is called.
    #[serde(default)]
    pub install_id: Option<String>,

    /// Whether the first-run notice has been shown.
    ///
    /// Opt-out reporting is only honest if people are told it is happening, so
    /// each binary shows a one-time notice and records it here.
    #[serde(default)]
    pub notice_shown: bool,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            install_id: None,
            notice_shown: false,
        }
    }
}

/// The DSN compiled into this binary, if any.
///
/// Set `HIVEBEAR_SENTRY_DSN` at build time (the release workflows do) to bake one
/// in. A source build leaves this `None`, so it reports nothing.
pub const COMPILED_DSN: Option<&str> = option_env!("HIVEBEAR_SENTRY_DSN");

/// Read a boolean-ish environment variable.
///
/// Returns `None` when unset or empty, so that an unset variable is distinct
/// from one explicitly set to `0`.
fn env_flag(name: &str) -> Option<bool> {
    let raw = std::env::var(name).ok()?;
    let value = raw.trim().to_ascii_lowercase();
    if value.is_empty() {
        return None;
    }
    Some(!matches!(value.as_str(), "0" | "false" | "no" | "off"))
}

/// Why reporting is off, for a helpful message rather than silence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisabledReason {
    /// `HIVEBEAR_TELEMETRY` is set to a falsey value.
    EnvOptOut,
    /// `DO_NOT_TRACK` is set.
    DoNotTrack,
    /// `telemetry.enabled = false` in the config file.
    ConfigOptOut,
    /// No DSN was compiled in and none was supplied — e.g. a source build.
    NoDsn,
}

/// The outcome of resolving whether this process should report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TelemetryDecision {
    /// Report to this DSN.
    Enabled { dsn: String },
    /// Do not report.
    Disabled(DisabledReason),
}

impl TelemetryDecision {
    pub fn dsn(&self) -> Option<&str> {
        match self {
            Self::Enabled { dsn } => Some(dsn),
            Self::Disabled(_) => None,
        }
    }

    pub fn is_enabled(&self) -> bool {
        matches!(self, Self::Enabled { .. })
    }
}

/// Decide whether to report, and to where.
///
/// Precedence, highest first:
/// 1. `HIVEBEAR_TELEMETRY=0` — an explicit opt-out always wins.
/// 2. `DO_NOT_TRACK=1`.
/// 3. `telemetry.enabled` from the config file.
/// 4. Availability of a DSN.
///
/// `HIVEBEAR_TELEMETRY=1` deliberately does *not* override the config file: it
/// only stops step 2 from disabling things, so someone who turned reporting off
/// in their config stays off.
pub fn resolve(config: &TelemetryConfig) -> TelemetryDecision {
    match env_flag(TELEMETRY_ENV) {
        Some(false) => return TelemetryDecision::Disabled(DisabledReason::EnvOptOut),
        Some(true) => {}
        None => {
            if env_flag(DO_NOT_TRACK_ENV).unwrap_or(false) {
                return TelemetryDecision::Disabled(DisabledReason::DoNotTrack);
            }
        }
    }

    if !config.enabled {
        return TelemetryDecision::Disabled(DisabledReason::ConfigOptOut);
    }

    let dsn = std::env::var(DSN_ENV)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| COMPILED_DSN.map(str::to_owned))
        .filter(|value| !value.trim().is_empty());

    match dsn {
        Some(dsn) => TelemetryDecision::Enabled { dsn },
        None => TelemetryDecision::Disabled(DisabledReason::NoDsn),
    }
}

/// The one-time notice shown on first run when reporting is active.
pub fn first_run_notice() -> String {
    format!(
        "HiveBear sends anonymous crash reports so we can fix what breaks.\n\
         No prompts, no file contents, no account details — just the error and \n\
         a random install ID. Turn it off any time with {TELEMETRY_ENV}=0."
    )
}

impl TelemetryConfig {
    /// Return the install ID, generating one if this is the first time.
    ///
    /// Returns `true` in the second slot when a new ID was generated, so the
    /// caller knows the config needs saving.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn ensure_install_id(&mut self) -> (String, bool) {
        match &self.install_id {
            Some(existing) if !existing.trim().is_empty() => (existing.clone(), false),
            _ => {
                let generated = uuid::Uuid::new_v4().to_string();
                self.install_id = Some(generated.clone());
                (generated, true)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Env vars are process-global, so these run under one lock and always
    /// restore what they changed.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct EnvGuard(&'static str, Option<String>);

    impl EnvGuard {
        fn set(key: &'static str, value: Option<&str>) -> Self {
            let previous = std::env::var(key).ok();
            match value {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
            Self(key, previous)
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.1 {
                Some(v) => std::env::set_var(self.0, v),
                None => std::env::remove_var(self.0),
            }
        }
    }

    fn enabled_config() -> TelemetryConfig {
        TelemetryConfig {
            enabled: true,
            ..Default::default()
        }
    }

    #[test]
    fn defaults_to_opt_out_not_opt_in() {
        assert!(TelemetryConfig::default().enabled);
    }

    #[test]
    fn env_opt_out_wins_over_everything() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _dnt = EnvGuard::set(DO_NOT_TRACK_ENV, None);
        let _dsn = EnvGuard::set(DSN_ENV, Some("https://key@example.invalid/1"));

        for value in ["0", "false", "no", "off", "OFF"] {
            let _flag = EnvGuard::set(TELEMETRY_ENV, Some(value));
            assert_eq!(
                resolve(&enabled_config()),
                TelemetryDecision::Disabled(DisabledReason::EnvOptOut),
                "{value} should disable telemetry"
            );
        }
    }

    #[test]
    fn do_not_track_is_honoured() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _flag = EnvGuard::set(TELEMETRY_ENV, None);
        let _dsn = EnvGuard::set(DSN_ENV, Some("https://key@example.invalid/1"));
        let _dnt = EnvGuard::set(DO_NOT_TRACK_ENV, Some("1"));

        assert_eq!(
            resolve(&enabled_config()),
            TelemetryDecision::Disabled(DisabledReason::DoNotTrack)
        );
    }

    #[test]
    fn config_opt_out_is_respected() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _flag = EnvGuard::set(TELEMETRY_ENV, None);
        let _dnt = EnvGuard::set(DO_NOT_TRACK_ENV, None);
        let _dsn = EnvGuard::set(DSN_ENV, Some("https://key@example.invalid/1"));

        let config = TelemetryConfig {
            enabled: false,
            ..Default::default()
        };
        assert_eq!(
            resolve(&config),
            TelemetryDecision::Disabled(DisabledReason::ConfigOptOut)
        );
    }

    /// Turning the env flag on must not resurrect reporting for someone who
    /// switched it off in their config.
    #[test]
    fn env_opt_in_does_not_override_config_opt_out() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _dnt = EnvGuard::set(DO_NOT_TRACK_ENV, None);
        let _dsn = EnvGuard::set(DSN_ENV, Some("https://key@example.invalid/1"));
        let _flag = EnvGuard::set(TELEMETRY_ENV, Some("1"));

        let config = TelemetryConfig {
            enabled: false,
            ..Default::default()
        };
        assert_eq!(
            resolve(&config),
            TelemetryDecision::Disabled(DisabledReason::ConfigOptOut)
        );
    }

    /// A source build has no compiled DSN, so it must stay silent.
    #[test]
    fn no_dsn_means_disabled() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _flag = EnvGuard::set(TELEMETRY_ENV, None);
        let _dnt = EnvGuard::set(DO_NOT_TRACK_ENV, None);
        let _dsn = EnvGuard::set(DSN_ENV, Some(""));

        if COMPILED_DSN.is_none() {
            assert_eq!(
                resolve(&enabled_config()),
                TelemetryDecision::Disabled(DisabledReason::NoDsn)
            );
        }
    }

    #[test]
    fn dsn_from_env_is_used() {
        let _lock = ENV_LOCK.lock().unwrap();
        let _flag = EnvGuard::set(TELEMETRY_ENV, None);
        let _dnt = EnvGuard::set(DO_NOT_TRACK_ENV, None);
        let _dsn = EnvGuard::set(DSN_ENV, Some("https://key@example.invalid/1"));

        assert_eq!(
            resolve(&enabled_config()),
            TelemetryDecision::Enabled {
                dsn: "https://key@example.invalid/1".to_string()
            }
        );
    }

    #[test]
    fn install_id_is_generated_once_and_then_reused() {
        let mut config = TelemetryConfig::default();
        assert!(config.install_id.is_none());

        let (first, generated) = config.ensure_install_id();
        assert!(generated, "first call should generate");
        assert!(!first.is_empty());

        let (second, generated_again) = config.ensure_install_id();
        assert!(!generated_again, "second call should reuse");
        assert_eq!(first, second);
    }

    #[test]
    fn install_ids_differ_between_installs() {
        let (a, _) = TelemetryConfig::default().ensure_install_id();
        let (b, _) = TelemetryConfig::default().ensure_install_id();
        assert_ne!(a, b);
    }
}
