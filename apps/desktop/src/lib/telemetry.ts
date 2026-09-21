import * as Sentry from "@sentry/react";
import { invoke } from "@tauri-apps/api/core";

/**
 * Crash reporting for the desktop and Android webview.
 *
 * The Rust side of this app reports through the Sentry Rust SDK; this covers the
 * half that runs in the webview, where a React render error or a broken IPC
 * bridge would otherwise only reach a `console.error` that, in a packaged app on
 * Windows or Android, nothing is attached to.
 *
 * Rust owns the decision. The DSN, the opt-out state and the install ID all come
 * from `telemetry_status`, rather than being baked in here separately — if the
 * webview made its own call, turning telemetry off would silence Rust while
 * JavaScript carried on reporting, which is worse than not offering the switch.
 */

declare const __APP_VERSION__: string;

interface TelemetryStatus {
  enabled: boolean;
  dsn: string | null;
  platform: string;
  install_id: string | null;
  notice_pending: boolean;
}

/**
 * Paths are the main way personal data leaks out of this app: an error like
 * "failed to load /home/alice/models/q4.gguf" carries the user's real name.
 * Mirrors `redact_sensitive` in hivebear-core, which is the tested copy — keep
 * the two in step.
 */
export function redact(input: string): string {
  return input
    .replace(/(\/home\/|\/Users\/)[^/\\]+/g, "$1<user>")
    .replace(/([A-Z]:\\Users\\)[^\\]+/gi, "$1<user>")
    .replace(/\b(sk-ant-|sk-|gsk_|hf_|xai-)[A-Za-z0-9_-]+/g, "[redacted-key]")
    .replace(/[\w.+-]+@[\w-]+\.[\w.-]+/g, "[email]");
}

let status: TelemetryStatus | null = null;

/** The resolved status, once `initTelemetry` has run. */
export function telemetryStatus(): TelemetryStatus | null {
  return status;
}

/**
 * Start reporting, if Rust says we should.
 *
 * Deliberately async and deliberately not awaited before the first render: it
 * costs one IPC round trip, and blocking paint on it would trade a real
 * user-visible delay for a marginal gain in coverage. Errors thrown in that
 * window are missed, which is the right way round — the alternative is
 * initialising before we know whether the user opted out.
 */
export async function initTelemetry(): Promise<void> {
  try {
    status = await invoke<TelemetryStatus>("telemetry_status");
  } catch {
    // If we cannot ask, do not report. Failing closed is the only safe default
    // for something the user is entitled to switch off.
    return;
  }

  if (!status.enabled || !status.dsn) return;

  Sentry.init({
    dsn: status.dsn,
    release: `hivebear@${__APP_VERSION__}`,
    environment: import.meta.env.DEV ? "development" : "production",

    tracesSampleRate: import.meta.env.DEV ? 1.0 : 0.1,

    // Never. This is someone else's computer.
    sendDefaultPii: false,

    beforeSend(event) {
      if (event.message) event.message = redact(event.message);
      for (const exception of event.exception?.values ?? []) {
        if (exception.value) exception.value = redact(exception.value);
      }
      for (const crumb of event.breadcrumbs ?? []) {
        if (crumb.message) crumb.message = redact(crumb.message);
      }
      return event;
    },
  });

  Sentry.setUser(status.install_id ? { id: status.install_id } : null);
  // The same bundle ships as desktop and as the Android webview; without this
  // the two are indistinguishable in one project.
  Sentry.setTag("platform", status.platform);
  Sentry.setTag("surface", "webview");
}

/** Report an error we are handling rather than rethrowing. */
export function reportError(
  error: unknown,
  context: Record<string, string> = {},
): void {
  Sentry.captureException(error, { tags: context });
}
