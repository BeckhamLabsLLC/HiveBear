mod commands;
mod error;
mod state;
mod validation;

use state::AppState;
use tauri::Manager;
use tracing::warn;
use tracing_subscriber::EnvFilter;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .init();

    tauri::Builder::default()
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_shell::init())
        .setup(|app| {
            // On mobile, use Tauri's app data dir (Android internal storage).
            // On desktop, use the default ProjectDirs-based paths.
            let app_state = if cfg!(target_os = "android") || cfg!(target_os = "ios") {
                let base = app
                    .path()
                    .app_data_dir()
                    .expect("Failed to resolve app data directory");
                let paths = AppState::paths_from_base(base);
                AppState::init_with_paths(paths)
            } else {
                AppState::init()
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
        ])
        .run(tauri::generate_context!())
        .expect("error while running HiveBear");
}
