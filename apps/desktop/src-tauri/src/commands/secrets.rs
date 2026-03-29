use crate::error::CmdResult;
use crate::state::AppState;
use std::collections::HashMap;
use tauri::State;

/// Set a cloud provider API key in the OS keychain.
#[tauri::command]
pub fn set_cloud_api_key(provider: String, key: String) -> CmdResult<()> {
    if provider.is_empty() || provider.len() > 64 {
        return Err("Invalid provider name".into());
    }
    if key.is_empty() || key.len() > 4096 {
        return Err("Invalid API key".into());
    }
    hivebear_core::secrets::set_api_key(&provider, &key)
}

/// Get all stored cloud provider API keys from the OS keychain.
///
/// Returns a map of provider → key for all providers that have keys stored.
#[tauri::command]
pub fn get_cloud_api_keys() -> CmdResult<HashMap<String, String>> {
    Ok(hivebear_core::secrets::get_all_api_keys(
        hivebear_core::secrets::KNOWN_PROVIDERS,
    ))
}

/// Delete a cloud provider API key from the OS keychain.
#[tauri::command]
pub fn delete_cloud_api_key(provider: String) -> CmdResult<()> {
    hivebear_core::secrets::delete_api_key(&provider)
}

/// Migrate plaintext API keys from config to OS keychain.
///
/// Reads keys from the current config's `cloud.api_keys` map,
/// stores them in the OS keychain, and clears them from the config.
#[tauri::command]
pub fn migrate_api_keys_to_keychain(state: State<'_, AppState>) -> CmdResult<Vec<String>> {
    let mut config = state
        .config
        .lock()
        .map_err(|_| String::from("Config lock poisoned"))?;

    if config.cloud.api_keys.is_empty() {
        return Ok(Vec::new());
    }

    let migrated = hivebear_core::secrets::migrate_plaintext_keys(&config.cloud.api_keys);

    // Remove migrated keys from the config file
    for provider in &migrated {
        config.cloud.api_keys.remove(provider);
    }

    // Save config without the migrated keys
    if !migrated.is_empty() {
        config
            .save()
            .map_err(|e| format!("Failed to save config after migration: {e}"))?;
    }

    Ok(migrated)
}

/// Check if the OS keychain is available on this system.
#[tauri::command]
pub fn is_keychain_available() -> bool {
    hivebear_core::secrets::is_keychain_available()
}
