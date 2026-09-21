//! Crash reporting for the `hivebear` CLI.
//!
//! The consent rules, the DSN resolution and the redaction all live in
//! `hivebear_core::telemetry` so the CLI and the desktop app cannot drift apart
//! on any of it. What is here is the CLI-specific part: which tags to attach,
//! when to print the first-run notice, and making sure events are flushed before
//! a short-lived command exits.

use std::borrow::Cow;

use hivebear_core::telemetry::{
    self, first_run_notice, redact_sensitive, DisabledReason, TelemetryDecision,
};
use hivebear_core::Config;

/// A one-shot `hivebear profile` can be over in well under a second, so the
/// flush has to be bounded tightly enough not to be annoying but long enough to
/// actually get the event out on a slow connection.
const FLUSH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

/// Initialise crash reporting.
///
/// Returns the guard, which must be held for the lifetime of the process.
/// `None` means reporting is off — because the user opted out, or because this
/// is a source build with no DSN compiled in.
///
/// Mutates `config` when it has to mint an install ID or record that the
/// first-run notice has been shown, and saves it if so.
pub fn init(config: &mut Config) -> Option<sentry::ClientInitGuard> {
    let decision = telemetry::resolve(&config.telemetry);

    let dsn = match &decision {
        TelemetryDecision::Enabled { dsn } => dsn.clone(),
        TelemetryDecision::Disabled(reason) => {
            // Only worth a line when the user might be surprised. A source build
            // having no DSN is expected and silent.
            if !matches!(reason, DisabledReason::NoDsn) {
                tracing::debug!("Crash reporting disabled: {reason:?}");
            }
            return None;
        }
    };

    let (install_id, generated) = config.telemetry.ensure_install_id();
    let show_notice = !config.telemetry.notice_shown;

    if generated || show_notice {
        config.telemetry.notice_shown = true;
        if let Err(e) = config.save() {
            // Not fatal: worst case the notice is shown again next run.
            tracing::debug!("Could not persist telemetry settings: {e}");
        }
    }

    if show_notice {
        eprintln!("{}", first_run_notice());
        eprintln!();
    }

    let mut options = sentry::ClientOptions::default();
    options.release = Some(Cow::Borrowed(env!("CARGO_PKG_VERSION")));
    options.environment = Some(Cow::Borrowed(if cfg!(debug_assertions) {
        "development"
    } else {
        "production"
    }));
    options.attach_stacktrace = true;

    // This is software on other people's machines. Nothing is collected
    // automatically that could identify them.
    options.send_default_pii = false;

    // Sessions power the crash-free-rate metric, which is the clearest signal
    // that a release broke something for new users.
    options.auto_session_tracking = true;
    options.session_mode = sentry::SessionMode::Application;

    options.before_send = Some(std::sync::Arc::new(|mut event| {
        scrub_event(&mut event);
        Some(event)
    }));

    let guard = sentry::init((dsn, options));

    sentry::configure_scope(|scope| {
        // Pseudonymous: a random UUID minted on this machine, unconnected to the
        // hardware fingerprint or the mesh identity.
        scope.set_user(Some(sentry::User {
            id: Some(install_id),
            ..Default::default()
        }));
        scope.set_tag("platform", "cli");
        scope.set_tag("os", std::env::consts::OS);
        scope.set_tag("arch", std::env::consts::ARCH);
    });

    Some(guard)
}

/// Strip anything personal from an event before it leaves the machine.
///
/// The CLI builds its error strings with `format!`, so they routinely contain a
/// model path under the user's home directory or a cloud provider key. The
/// redaction itself is shared with the desktop app and tested in
/// `hivebear_core::telemetry`.
fn scrub_event(event: &mut sentry::protocol::Event<'static>) {
    if let Some(message) = &event.message {
        event.message = Some(redact_sensitive(message));
    }

    for exception in &mut event.exception.values {
        if let Some(value) = &exception.value {
            exception.value = Some(redact_sensitive(value));
        }
    }

    for breadcrumb in &mut event.breadcrumbs.values {
        if let Some(message) = &breadcrumb.message {
            breadcrumb.message = Some(redact_sensitive(message));
        }
    }

    for entry in event.logentry.iter_mut() {
        entry.message = redact_sensitive(&entry.message);
    }
}

/// Flush queued events.
///
/// Most CLI subcommands finish in well under a second, and the guard's Drop is
/// not always reached (`std::process::exit`, and the `unreachable` arm of a
/// diverging command). Call this before any deliberate exit.
pub fn flush() {
    if let Some(client) = sentry::Hub::current().client() {
        client.flush(Some(FLUSH_TIMEOUT));
    }
}

/// Send a test event, for `hivebear sentry-check`.
///
/// Returns false when reporting is disabled, so the command can say so rather
/// than silently appearing to succeed.
pub fn send_test_event() -> bool {
    if sentry::Hub::current().client().is_none() {
        return false;
    }
    sentry::capture_message("HiveBear CLI telemetry verification", sentry::Level::Info);
    flush();
    true
}
