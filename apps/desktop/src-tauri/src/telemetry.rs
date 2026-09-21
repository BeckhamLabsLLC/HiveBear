//! Crash reporting for the desktop and Android app.
//!
//! Consent rules, DSN resolution and redaction are shared with the CLI via
//! `hivebear_core::telemetry`. What differs here:
//!
//! * There is no console to print a first-run notice to. The Windows build is
//!   `windows_subsystem = "windows"` and Android has no terminal at all, so the
//!   notice is shown in the UI instead and this module only records that it is
//!   owed.
//! * The same binary and the same web bundle ship as desktop and as Android, so
//!   everything is tagged with which one it is. Without that the two are
//!   indistinguishable in a single project.

use std::borrow::Cow;

use hivebear_core::telemetry::{self, redact_sensitive, DisabledReason, TelemetryDecision};
use hivebear_core::Config;

const FLUSH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(3);

/// `desktop` or `android`, as a tag value.
pub fn platform() -> &'static str {
    if cfg!(target_os = "android") {
        "android"
    } else if cfg!(target_os = "ios") {
        "ios"
    } else {
        "desktop"
    }
}

/// Initialise crash reporting. The guard must be held for the process lifetime.
///
/// Takes the config by value and hands it back, because this runs before
/// `AppState` exists and may need to mint an install ID.
pub fn init(config: &mut Config) -> Option<sentry::ClientInitGuard> {
    let dsn = match telemetry::resolve(&config.telemetry) {
        TelemetryDecision::Enabled { dsn } => dsn,
        TelemetryDecision::Disabled(reason) => {
            if !matches!(reason, DisabledReason::NoDsn) {
                tracing::debug!("Crash reporting disabled: {reason:?}");
            }
            return None;
        }
    };

    let (install_id, generated) = config.telemetry.ensure_install_id();
    if generated {
        if let Err(e) = config.save() {
            tracing::debug!("Could not persist telemetry settings: {e}");
        }
    }

    let mut options = sentry::ClientOptions::default();
    options.release = Some(Cow::Borrowed(env!("CARGO_PKG_VERSION")));
    options.environment = Some(Cow::Borrowed(if cfg!(debug_assertions) {
        "development"
    } else {
        "production"
    }));
    options.attach_stacktrace = true;
    options.send_default_pii = false;

    // Crash-free session rate is the clearest signal that a release broke
    // something for new users, which is the population that matters most here.
    options.auto_session_tracking = true;
    options.session_mode = sentry::SessionMode::Application;

    options.before_send = Some(std::sync::Arc::new(|mut event| {
        scrub_event(&mut event);
        Some(event)
    }));

    let guard = sentry::init((dsn, options));

    sentry::configure_scope(|scope| {
        scope.set_user(Some(sentry::User {
            id: Some(install_id),
            ..Default::default()
        }));
        scope.set_tag("platform", platform());
        scope.set_tag("os", std::env::consts::OS);
        scope.set_tag("arch", std::env::consts::ARCH);
    });

    Some(guard)
}

/// Strip anything personal before an event leaves the device.
///
/// Redaction is shared with the CLI and tested in `hivebear_core::telemetry`.
/// It matters more here than on the server: these errors carry model paths under
/// the user's home directory, and the secrets and account commands handle cloud
/// provider keys and email addresses.
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

/// Report a fatal startup failure and wait for it to be sent.
///
/// Both callers follow this with `std::process::exit(1)`, which does not run
/// destructors — so without the explicit flush the guard never gets a chance to
/// send, and the failures we would most want to see are precisely the ones that
/// would never arrive.
pub fn capture_fatal(message: &str) {
    sentry::configure_scope(|scope| scope.set_tag("fatal", "startup"));
    sentry::capture_message(&redact_sensitive(message), sentry::Level::Fatal);
    flush();
}

pub fn flush() {
    if let Some(client) = sentry::Hub::current().client() {
        client.flush(Some(FLUSH_TIMEOUT));
    }
}
