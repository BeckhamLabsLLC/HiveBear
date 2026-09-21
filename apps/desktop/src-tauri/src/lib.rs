mod commands;
mod error;
mod state;
mod telemetry;
mod validation;

use state::AppState;
use tauri::Manager;
use tracing::{error, warn};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Before anything that can fail. Config is read directly rather than through
    // AppState because AppState::init is itself one of the things that fails.
    let mut config = hivebear_core::Config::load();
    let _sentry = telemetry::init(&mut config);

    // Composed as a registry so the Sentry layer sits beside the formatter.
    // Worth noting what this replaces: stdout at `warn` level, on a binary built
    // with `windows_subsystem = "windows"` and shipped to Android — neither of
    // which has a console. For real users these logs went nowhere at all.
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")))
        .with(tracing_subscriber::fmt::layer())
        .with(sentry::integrations::tracing::layer())
        .init();

    tauri::Builder::default()
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            // On mobile, use Tauri's app data dir (Android internal storage).
            // On desktop, use the default ProjectDirs-based paths.
            let init_result = if cfg!(target_os = "android") || cfg!(target_os = "ios") {
                match app.path().app_data_dir() {
                    Ok(base) => AppState::init_with_paths(AppState::paths_from_base(base)),
                    Err(e) => Err(format!("Could not resolve the app data directory.\n\n{e}")),
                }
            } else {
                AppState::init()
            };

            // Startup used to panic here. That happens inside setup(), before
            // any window exists, so the process just vanished — no window, no
            // dialog, nothing an ordinary user could find. Report it somewhere
            // retrievable and exit deliberately instead.
            let app_state = match init_result {
                Ok(state) => state,
                Err(message) => {
                    error!("HiveBear could not start: {message}");
                    eprintln!("HiveBear could not start:\n{message}");
                    write_startup_failure(&message);
                    std::process::exit(1);
                }
            };
            // Auto-start mesh if enabled and auto_join is configured
            {
                let should_start = {
                    let config = app_state.config.lock().unwrap_or_else(|e| e.into_inner());
                    config.mesh.enabled && config.mesh.auto_join
                };
                if should_start {
                    if let Err(e) = app_state.start_mesh() {
                        warn!("Failed to auto-start mesh (non-fatal): {e}");
                    }
                }
            }

            app.manage(app_state);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::profile::get_hardware_profile,
            commands::profile::get_recommendations,
            commands::registry::search_models,
            commands::registry::install_model,
            commands::registry::list_installed,
            commands::registry::remove_model,
            commands::registry::get_storage_report,
            commands::inference::load_model,
            commands::inference::stream_chat,
            commands::inference::unload_model,
            commands::inference::list_loaded_models,
            commands::benchmark::run_benchmark,
            commands::benchmark::share_benchmark,
            commands::benchmark::get_community_benchmarks,
            commands::config::get_config,
            commands::config::save_config,
            commands::mesh::get_mesh_status,
            commands::mesh::get_mesh_config,
            commands::mesh::save_mesh_config,
            commands::mesh::join_mesh,
            commands::mesh::leave_mesh,
            commands::mesh::get_mesh_connection_status,
            commands::chat::list_conversations,
            commands::chat::create_conversation,
            commands::chat::get_conversation_messages,
            commands::chat::add_message,
            commands::chat::delete_conversation,
            commands::chat::rename_conversation,
            commands::chat::search_conversations,
            commands::secrets::set_cloud_api_key,
            commands::secrets::get_cloud_api_keys,
            commands::secrets::delete_cloud_api_key,
            commands::secrets::migrate_api_keys_to_keychain,
            commands::secrets::is_keychain_available,
            commands::account::login,
            commands::account::register,
            commands::account::activate_device,
            commands::account::logout,
            commands::account::get_account,
            commands::account::get_usage_summary,
            commands::account::create_checkout,
            commands::account::list_api_keys,
            commands::account::create_api_key,
            commands::account::revoke_api_key,
            commands::device::get_device_status,
            commands::device::can_contribute_to_mesh,
            commands::telemetry::telemetry_status,
            commands::telemetry::acknowledge_telemetry_notice,
        ])
        .run(tauri::generate_context!())
        .unwrap_or_else(|e| {
            error!("HiveBear exited with an error: {e}");
            eprintln!("HiveBear exited with an error: {e}");
            write_startup_failure(&e.to_string());
            std::process::exit(1);
        });
}

/// Record a startup failure where a user can actually be pointed at it.
/// The data directory may itself be the thing that is broken, so fall back to
/// the temp directory.
///
/// Also reports it. A local log file only helps a user who knows to look for it
/// and knows how to send it to us; for a startup crash — where there is no
/// window and nothing on screen — that is close to nobody.
fn write_startup_failure(message: &str) {
    telemetry::capture_fatal(message);
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let body = format!(
        "HiveBear {} startup failure\n\n{message}\n",
        env!("CARGO_PKG_VERSION")
    );

    let candidates = [
        hivebear_core::AppPaths::new()
            .data_dir
            .join("startup-error.log"),
        std::env::temp_dir().join(format!("hivebear-startup-error-{stamp}.log")),
    ];
    for path in candidates {
        if std::fs::write(&path, &body).is_ok() {
            eprintln!("Details written to {}", path.display());
            return;
        }
    }
}
