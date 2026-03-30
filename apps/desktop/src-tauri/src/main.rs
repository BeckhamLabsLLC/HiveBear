#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod error;
mod state;
mod validation;

use state::AppState;
use tracing_subscriber::EnvFilter;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .init();

    let app_state = AppState::init();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(app_state)
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
        ])
        .run(tauri::generate_context!())
        .expect("error while running HiveBear");
}
