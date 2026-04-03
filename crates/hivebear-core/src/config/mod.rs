pub mod paths;

use paths::AppPaths;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// HiveBear configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Directory where models are stored.
    pub models_dir: PathBuf,

    /// Maximum percentage of available RAM to use for inference (0.0 - 1.0).
    pub max_memory_usage: f64,

    /// Minimum acceptable tokens per second for recommendations.
    pub min_tokens_per_sec: f32,

    /// Number of top recommendations to show.
    pub top_n_recommendations: usize,

    /// Whether to share anonymous benchmark data.
    pub share_benchmarks: bool,

    /// Default context length for recommendations.
    pub default_context_length: u32,

    /// P2P mesh configuration.
    #[serde(default)]
    pub mesh: MeshConfig,

    /// Mobile-specific configuration.
    #[serde(default)]
    pub mobile: MobileConfig,

    /// API server configuration.
    #[serde(default)]
    pub api: ApiConfig,

    /// Cloud provider configuration.
    #[serde(default)]
    pub cloud: CloudConfig,

    /// Account and subscription configuration.
    #[serde(default)]
    pub account: AccountConfig,
}

/// Configuration for the P2P inference mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshConfig {
    /// Whether mesh mode is enabled.
    pub enabled: bool,
    /// Automatically join the mesh when running inference.
    #[serde(default = "default_true")]
    pub auto_join: bool,
    /// Port for QUIC transport.
    pub port: u16,
    /// URL of the coordination server.
    pub coordination_server: String,
    /// Bootstrap coordination servers for initial discovery.
    #[serde(default = "default_bootstrap_servers")]
    pub bootstrap_servers: Vec<String>,
    /// STUN servers for NAT traversal.
    #[serde(default = "default_stun_servers")]
    pub stun_servers: Vec<String>,
    /// Relay servers for symmetric NAT fallback.
    #[serde(default = "default_relay_servers")]
    pub relay_servers: Vec<String>,
    /// Whether to use relay servers when direct/hole-punch connections fail.
    #[serde(default = "default_true")]
    pub relay_enabled: bool,
    /// This node's tier ("free" or "paid").
    pub tier: String,
    /// Geographic region hint (auto-detected if None).
    #[serde(default)]
    pub region: Option<String>,
    /// Maximum percentage of local resources to share with the mesh (0.0 - 1.0).
    pub max_contribution_percent: f64,
    /// Minimum reputation score to trust a peer (0.0 - 1.0).
    pub min_reputation: f64,
    /// Fraction of tokens to verify (0.0 = never, 1.0 = every token).
    pub verification_rate: f64,
    /// Enable mDNS for LAN peer discovery.
    #[serde(default = "default_true")]
    pub mdns_enabled: bool,
    /// Enable peer exchange (PEX) gossip protocol.
    #[serde(default = "default_true")]
    pub pex_enabled: bool,
    /// Compression method for tensor transfer: "auto", "none", "lz4", "zstd".
    #[serde(default = "default_compression")]
    pub compression: String,
    /// Data type for tensor transfer: "auto", "f16", "f32".
    #[serde(default = "default_transfer_dtype")]
    pub transfer_dtype: String,
}

fn default_true() -> bool {
    true
}

fn default_bootstrap_servers() -> Vec<String> {
    vec!["https://mesh.hivebear.com".into()]
}

fn default_stun_servers() -> Vec<String> {
    vec!["stun.l.google.com:19302".into()]
}

fn default_relay_servers() -> Vec<String> {
    vec!["relay.hivebear.com:3478".into()]
}

fn default_compression() -> String {
    "auto".into()
}

fn default_transfer_dtype() -> String {
    "auto".into()
}

/// API server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    /// API key for Bearer token auth. Auto-generated on first server start if None.
    pub api_key: Option<String>,
    /// Bind address for the API server (default: 127.0.0.1).
    pub bind_address: String,
    /// Allowed CORS origins (e.g., ["http://localhost:3000"]).
    pub cors_origins: Vec<String>,
}

/// Cloud provider configuration.
///
/// Supports 35+ providers via a unified `api_keys` map keyed by provider prefix
/// (e.g. `"openai"`, `"groq"`, `"anthropic"`). Legacy `openai_api_key` and
/// `anthropic_api_key` fields are read for backward compatibility but not written.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct CloudConfig {
    /// Provider API keys, keyed by provider prefix.
    /// Example: `{ "openai": "sk-...", "groq": "gsk_...", "anthropic": "sk-ant-..." }`
    #[serde(default)]
    pub api_keys: std::collections::HashMap<String, String>,

    /// Custom OpenAI-compatible endpoints (user-defined providers).
    #[serde(default)]
    pub custom_endpoints: Vec<CustomEndpoint>,

    /// Whether to fall back to cloud when local inference fails.
    #[serde(default)]
    pub cloud_fallback: bool,

    /// Default provider to use when no prefix is given.
    #[serde(default)]
    pub default_provider: Option<String>,

    // ── Backward compatibility (deprecated, read-only) ───────────
    /// Deprecated: use `api_keys.openai` instead. Read on load, not written on save.
    #[serde(default, skip_serializing)]
    pub openai_api_key: Option<String>,
    /// Deprecated: use `api_keys.anthropic` instead. Read on load, not written on save.
    #[serde(default, skip_serializing)]
    pub anthropic_api_key: Option<String>,
}

impl std::fmt::Debug for CloudConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CloudConfig")
            .field("api_keys", &format!("[{} keys]", self.api_keys.len()))
            .field("custom_endpoints", &self.custom_endpoints)
            .field("cloud_fallback", &self.cloud_fallback)
            .field("default_provider", &self.default_provider)
            .field(
                "openai_api_key",
                &self.openai_api_key.as_ref().map(|_| "[REDACTED]"),
            )
            .field(
                "anthropic_api_key",
                &self.anthropic_api_key.as_ref().map(|_| "[REDACTED]"),
            )
            .finish()
    }
}

impl CloudConfig {
    /// Migrate legacy fields into the `api_keys` map.
    /// Call this after deserialization to ensure backward compatibility.
    pub fn migrate_legacy(&mut self) {
        if let Some(key) = self.openai_api_key.take() {
            self.api_keys.entry("openai".into()).or_insert(key);
        }
        if let Some(key) = self.anthropic_api_key.take() {
            self.api_keys.entry("anthropic".into()).or_insert(key);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomEndpoint {
    pub name: String,
    pub base_url: String,
    pub api_key: Option<String>,
    /// Protocol to use: "openai" (default), "anthropic", "gemini", "cohere".
    #[serde(default = "default_protocol")]
    pub protocol: String,
}

fn default_protocol() -> String {
    "openai".into()
}

// Default is derived — all fields use #[serde(default)]

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            bind_address: "127.0.0.1".into(),
            cors_origins: vec!["*".into()],
        }
    }
}

/// Configuration for mobile platforms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MobileConfig {
    /// Maximum model size in billions of parameters.
    pub max_model_params_b: f64,
    /// Maximum percentage of available RAM to use (more conservative than desktop).
    pub max_memory_usage: f64,
    /// Whether to contribute idle compute to the mesh when conditions are met.
    pub background_mesh_enabled: bool,
    /// Only contribute to mesh when on Wi-Fi (not cellular).
    pub mesh_wifi_only: bool,
    /// Only contribute to mesh when charging.
    pub mesh_charging_only: bool,
}

impl Default for MobileConfig {
    fn default() -> Self {
        Self {
            max_model_params_b: 3.0,
            max_memory_usage: 0.50,
            background_mesh_enabled: true,
            mesh_wifi_only: true,
            mesh_charging_only: true,
        }
    }
}

impl Default for MeshConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            auto_join: true,
            port: 7878,
            coordination_server: "https://mesh.hivebear.com".into(),
            bootstrap_servers: default_bootstrap_servers(),
            stun_servers: default_stun_servers(),
            relay_servers: default_relay_servers(),
            relay_enabled: true,
            tier: "free".into(),
            region: None,
            max_contribution_percent: 0.8,
            min_reputation: 0.5,
            verification_rate: 0.05,
            mdns_enabled: true,
            pex_enabled: true,
            compression: "auto".into(),
            transfer_dtype: "auto".into(),
        }
    }
}

/// Account and subscription configuration.
///
/// Supports two auth paths:
/// - Path A (email): traditional email/password account
/// - Path B (device): anonymous device-key identity (Mullvad-style)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AccountConfig {
    /// Auth mode: "email" or "device" (None = not authenticated).
    #[serde(default)]
    pub auth_mode: Option<String>,

    /// JWT token for the coordination server (works for both paths).
    #[serde(default)]
    pub jwt_token: Option<String>,

    /// Refresh token for silent re-auth.
    #[serde(default)]
    pub refresh_token: Option<String>,

    /// Signed license token for offline tier verification (Path B).
    #[serde(default)]
    pub license_token: Option<String>,

    /// Cached subscription tier from JWT/license.
    #[serde(default)]
    pub tier: Option<String>,

    /// User ID (Path A only).
    #[serde(default)]
    pub user_id: Option<String>,

    /// Display name (Path A only).
    #[serde(default)]
    pub display_name: Option<String>,

    /// Optional recovery email (Path B only, user's choice).
    #[serde(default)]
    pub recovery_email: Option<String>,

    /// Coordination server URL for account operations.
    #[serde(default)]
    pub server_url: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        let paths = AppPaths::new();
        Self {
            models_dir: paths.models_dir,
            max_memory_usage: 0.85,
            min_tokens_per_sec: 5.0,
            top_n_recommendations: 10,
            share_benchmarks: false,
            default_context_length: 4096,
            mesh: MeshConfig::default(),
            mobile: MobileConfig::default(),
            api: ApiConfig::default(),
            cloud: CloudConfig::default(),
            account: AccountConfig::default(),
        }
    }
}

impl Config {
    /// Load config from the default location, or return defaults.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load() -> Self {
        let paths = AppPaths::new();
        Self::load_from(&paths.config_file)
    }

    /// In WASM, always return defaults (no filesystem).
    #[cfg(target_arch = "wasm32")]
    pub fn load() -> Self {
        Self::default()
    }

    /// Load config from a specific path.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load_from(path: &std::path::Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(contents) => match toml::from_str::<Config>(&contents) {
                Ok(mut config) => {
                    config.cloud.migrate_legacy();
                    config
                }
                Err(e) => {
                    tracing::warn!("Failed to parse config at {}: {e}", path.display());
                    Self::default()
                }
            },
            Err(_) => Self::default(),
        }
    }

    /// Save config to the default location.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn save(&self) -> Result<(), ConfigError> {
        let paths = AppPaths::new();
        self.save_to(&paths.config_file)
    }

    /// No-op on WASM — no filesystem.
    #[cfg(target_arch = "wasm32")]
    pub fn save(&self) -> Result<(), ConfigError> {
        Ok(())
    }

    /// Save config to a specific path.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn save_to(&self, path: &std::path::Path) -> Result<(), ConfigError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(ConfigError::Io)?;
        }
        let contents = toml::to_string_pretty(self).map_err(ConfigError::Serialize)?;
        std::fs::write(path, contents).map_err(ConfigError::Io)?;
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialize(#[from] toml::ser::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.max_memory_usage, 0.85);
        assert_eq!(config.min_tokens_per_sec, 5.0);
        assert_eq!(config.top_n_recommendations, 10);
    }

    #[test]
    fn test_config_roundtrip() {
        let config = Config::default();
        let serialized = toml::to_string_pretty(&config).unwrap();
        let deserialized: Config = toml::from_str(&serialized).unwrap();
        assert_eq!(config.max_memory_usage, deserialized.max_memory_usage);
        assert_eq!(config.min_tokens_per_sec, deserialized.min_tokens_per_sec);
    }
}
