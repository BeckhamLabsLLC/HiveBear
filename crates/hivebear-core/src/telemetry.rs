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

// ── Redaction ────────────────────────────────────────────────────────
//
// Shared by every HiveBear binary. Error strings here are built with `format!`
// from whatever failed, so they routinely end up carrying a model path under the
// user's home directory, a cloud provider key, or the text of a prompt. The
// binaries apply this in their Sentry `before_send`.

/// Replace the username inside a home-directory path with `<user>`.
///
/// `/home/alice/.config/...` is not obviously personal data until you notice it
/// contains a real person's name, and these paths appear in almost every I/O
/// error the CLI produces.
fn redact_home_paths(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;

    // Windows uses backslashes; everything else uses forward slashes.
    const PREFIXES: [(&str, char); 4] = [
        ("/home/", '/'),
        ("/Users/", '/'),
        ("C:\\Users\\", '\\'),
        ("\\Users\\", '\\'),
    ];

    'outer: loop {
        for (prefix, sep) in PREFIXES {
            if let Some(idx) = rest.find(prefix) {
                let after = idx + prefix.len();
                out.push_str(&rest[..after]);
                let tail = &rest[after..];
                let end = tail.find(sep).unwrap_or(tail.len());
                if end > 0 {
                    out.push_str("<user>");
                }
                rest = &tail[end..];
                continue 'outer;
            }
        }
        break;
    }

    out.push_str(rest);
    out
}

/// Known cloud-provider key shapes, longest prefix first so `sk-ant-` is not
/// half-matched by `sk-`.
const KEY_PREFIXES: [&str; 6] = ["sk-ant-", "sk-", "gsk_", "hf_", "xai-", "AIza"];

fn redact_api_keys(input: &str) -> String {
    let mut out = input.to_string();
    for prefix in KEY_PREFIXES {
        while let Some(idx) = out.find(prefix) {
            let after = idx + prefix.len();
            let tail = &out[after..];
            let end = tail
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '-'))
                .unwrap_or(tail.len());
            // A bare prefix with nothing after it is not a key. Break rather
            // than continue, or this spins forever on the same match.
            if end == 0 {
                break;
            }
            out.replace_range(idx..after + end, "[redacted-key]");
        }
    }
    out
}

fn redact_emails(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for token in input.split_inclusive(char::is_whitespace) {
        let trimmed = token.trim_end();
        let looks_like_email = trimmed.contains('@')
            && trimmed.split('@').count() == 2
            && trimmed.split('@').nth(1).is_some_and(|d| d.contains('.'));
        if looks_like_email {
            out.push_str("[email]");
            out.push_str(&token[trimmed.len()..]);
        } else {
            out.push_str(token);
        }
    }
    out
}

/// Strip the things a HiveBear error message should never carry off-device.
///
/// Applied to exception values, log messages and breadcrumbs before they leave
/// the process. It is a backstop, not the primary defence — the primary defence
/// is not putting secrets in error strings in the first place.
pub fn redact_sensitive(input: &str) -> String {
    redact_emails(&redact_api_keys(&redact_home_paths(input)))
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

    // ── Redaction ────────────────────────────────────────────────────

    #[test]
    fn strips_the_username_from_unix_home_paths() {
        let out = redact_sensitive("failed to open /home/alice/.hivebear/models/q4.gguf");
        assert!(!out.contains("alice"), "{out}");
        assert!(
            out.contains("/home/<user>/.hivebear/models/q4.gguf"),
            "{out}"
        );
    }

    #[test]
    fn strips_the_username_from_macos_and_windows_home_paths() {
        let mac = redact_sensitive("no such file: /Users/bob/Library/App/hivebear.toml");
        assert!(!mac.contains("bob"), "{mac}");
        assert!(mac.contains("/Users/<user>/Library"), "{mac}");

        let win = redact_sensitive(r"cannot write C:\Users\carol\AppData\hivebear.db");
        assert!(!win.contains("carol"), "{win}");
        assert!(win.contains(r"C:\Users\<user>\AppData"), "{win}");
    }

    #[test]
    fn strips_every_home_path_in_a_message_not_just_the_first() {
        let out = redact_sensitive("copy /home/dave/a.gguf to /home/dave/b.gguf");
        assert!(!out.contains("dave"), "{out}");
        assert_eq!(out.matches("<user>").count(), 2, "{out}");
    }

    #[test]
    fn strips_cloud_provider_api_keys() {
        for key in [
            "sk-abcdef1234567890",
            "sk-ant-api03-abcdef123456",
            "gsk_abcdef1234567890",
            "hf_abcdefABCDEF123456",
            "xai-abcdef1234567890",
        ] {
            let out = redact_sensitive(&format!("auth failed with key {key} for provider"));
            assert!(!out.contains(key), "leaked {key}: {out}");
            assert!(out.contains("[redacted-key]"), "{out}");
        }
    }

    /// `sk-ant-` must not be half-matched by the shorter `sk-` prefix, leaving
    /// the distinctive part of the key behind.
    #[test]
    fn longer_key_prefixes_win_over_shorter_ones() {
        let out = redact_sensitive("key sk-ant-api03-SECRETVALUE end");
        assert!(!out.contains("SECRETVALUE"), "{out}");
        assert!(!out.contains("api03"), "{out}");
    }

    #[test]
    fn strips_email_addresses() {
        let out = redact_sensitive("login rejected for someone@example.com (401)");
        assert!(!out.contains("someone@example.com"), "{out}");
        assert!(out.contains("[email]"), "{out}");
    }

    /// Over-redaction makes reports useless, so ordinary text must survive.
    #[test]
    fn leaves_ordinary_diagnostics_alone() {
        let input = "connection refused to mesh.hivebear.com:7878 after 3 retries";
        assert_eq!(redact_sensitive(input), input);

        let input = "model llama-3-8b failed to load: out of memory (8192 MB required)";
        assert_eq!(redact_sensitive(input), input);
    }

    /// A relative or system path has no username in it and should not be touched.
    #[test]
    fn leaves_non_home_paths_alone() {
        let input = "failed to read /usr/share/hivebear/default.toml";
        assert_eq!(redact_sensitive(input), input);
    }

    #[test]
    fn redaction_terminates_on_pathological_input() {
        // A bare prefix with no username after it must not loop forever.
        assert_eq!(redact_sensitive("/home/"), "/home/");
        assert_eq!(redact_sensitive("sk-"), "sk-");
        assert_eq!(redact_sensitive("/home//home//home/"), "/home//home//home/");
    }
}
