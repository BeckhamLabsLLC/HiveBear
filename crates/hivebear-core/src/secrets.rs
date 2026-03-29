//! Secure secret storage via OS keychain (macOS Keychain, Windows Credential
//! Manager, Linux Secret Service / libsecret).
//!
//! API keys are stored in the OS keychain instead of plaintext config files.
//! This module provides a unified interface for get/set/delete operations.

use std::collections::HashMap;
use tracing::{debug, warn};

/// Service name used in the OS keychain.
const SERVICE_NAME: &str = "hivebear";

/// Prefix for cloud provider API key entries in the keychain.
const CLOUD_KEY_PREFIX: &str = "cloud::";

/// Store a cloud provider API key in the OS keychain.
///
/// The key is stored under the service "hivebear" with user "cloud::{provider}".
pub fn set_api_key(provider: &str, key: &str) -> Result<(), String> {
    let user = format!("{CLOUD_KEY_PREFIX}{provider}");
    let entry = keyring::Entry::new(SERVICE_NAME, &user)
        .map_err(|e| format!("Failed to create keyring entry for {provider}: {e}"))?;
    entry
        .set_password(key)
        .map_err(|e| format!("Failed to store API key for {provider}: {e}"))?;
    debug!("Stored API key for {provider} in OS keychain");
    Ok(())
}

/// Retrieve a cloud provider API key from the OS keychain.
///
/// Returns `None` if the key is not found.
pub fn get_api_key(provider: &str) -> Option<String> {
    let user = format!("{CLOUD_KEY_PREFIX}{provider}");
    let entry = keyring::Entry::new(SERVICE_NAME, &user).ok()?;
    match entry.get_password() {
        Ok(key) => Some(key),
        Err(keyring::Error::NoEntry) => None,
        Err(e) => {
            warn!("Failed to read API key for {provider} from keychain: {e}");
            None
        }
    }
}

/// Delete a cloud provider API key from the OS keychain.
pub fn delete_api_key(provider: &str) -> Result<(), String> {
    let user = format!("{CLOUD_KEY_PREFIX}{provider}");
    let entry = keyring::Entry::new(SERVICE_NAME, &user)
        .map_err(|e| format!("Failed to create keyring entry for {provider}: {e}"))?;
    match entry.delete_credential() {
        Ok(()) => {
            debug!("Deleted API key for {provider} from OS keychain");
            Ok(())
        }
        Err(keyring::Error::NoEntry) => Ok(()), // Already gone
        Err(e) => Err(format!("Failed to delete API key for {provider}: {e}")),
    }
}

/// Get all stored cloud provider API keys from the OS keychain.
///
/// This retrieves keys for a known set of providers. Since OS keychains
/// don't support enumeration by prefix, we try each provider in the list.
pub fn get_all_api_keys(providers: &[&str]) -> HashMap<String, String> {
    let mut keys = HashMap::new();
    for provider in providers {
        if let Some(key) = get_api_key(provider) {
            keys.insert(provider.to_string(), key);
        }
    }
    keys
}

/// Known cloud providers that may have API keys stored.
pub const KNOWN_PROVIDERS: &[&str] = &[
    "openai",
    "anthropic",
    "google",
    "mistral",
    "groq",
    "together",
    "fireworks",
    "deepseek",
    "xai",
    "cohere",
    "perplexity",
    "openrouter",
    "lepton",
    "novita",
    "cerebras",
    "sambanova",
    "hyperbolic",
    "nebius",
    "kluster",
    "inferencenet",
    "nineteen",
    "chutes",
    "targon",
    "centauri",
    "ncompass",
    "atoma",
    "venice",
    "akash",
    "tensoropera",
    "featherless",
    "ollama",
];

/// Migrate plaintext API keys from a config HashMap to the OS keychain.
///
/// Returns the list of providers that were successfully migrated.
pub fn migrate_plaintext_keys(plaintext_keys: &HashMap<String, String>) -> Vec<String> {
    let mut migrated = Vec::new();
    for (provider, key) in plaintext_keys {
        if key.is_empty() {
            continue;
        }
        match set_api_key(provider, key) {
            Ok(()) => {
                debug!("Migrated {provider} API key to OS keychain");
                migrated.push(provider.clone());
            }
            Err(e) => {
                warn!("Failed to migrate {provider} API key to keychain: {e}");
            }
        }
    }
    if !migrated.is_empty() {
        tracing::info!(
            "Migrated {} API key(s) from config file to OS keychain: {}",
            migrated.len(),
            migrated.join(", ")
        );
    }
    migrated
}

/// Check if the OS keychain service is available.
///
/// Returns true if we can create and delete a test entry.
pub fn is_keychain_available() -> bool {
    let entry = match keyring::Entry::new(SERVICE_NAME, "__keychain_test__") {
        Ok(e) => e,
        Err(_) => return false,
    };
    // Try setting and immediately deleting a test value
    if entry.set_password("test").is_err() {
        return false;
    }
    let _ = entry.delete_credential();
    true
}
