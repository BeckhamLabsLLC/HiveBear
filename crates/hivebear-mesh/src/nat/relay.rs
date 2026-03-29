use std::net::SocketAddr;

use tracing::{info, warn};

use crate::error::{MeshError, Result};
use crate::peer::PeerInfo;

/// Relay client for peers behind symmetric NATs.
///
/// When both direct connections and hole-punching fail (typically for
/// symmetric NATs, ~15% of home networks), the relay server acts as a
/// UDP proxy between the two peers.
pub struct RelayClient {
    relay_servers: Vec<String>,
}

impl RelayClient {
    pub fn new(relay_servers: Vec<String>) -> Self {
        Self { relay_servers }
    }

    /// Request a relay allocation for connecting to a target peer.
    ///
    /// Returns the relay address that should be used instead of the
    /// peer's direct address.
    pub async fn allocate_relay(&self, target: &PeerInfo) -> Result<SocketAddr> {
        if self.relay_servers.is_empty() {
            return Err(MeshError::Relay("No relay servers configured".into()));
        }

        let relay_server = &self.relay_servers[0];
        info!(
            "Requesting relay allocation via {} for peer {}",
            relay_server, target.node_id
        );

        // Build relay allocation request
        let client = reqwest::Client::new();
        let url = if relay_server.starts_with("http") {
            format!("{}/allocate", relay_server)
        } else {
            format!("https://{}/allocate", relay_server)
        };

        let resp = client
            .post(&url)
            .json(&serde_json::json!({
                "target_node_id": target.node_id.to_hex(),
                "target_addr": target.addr.to_string(),
            }))
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await
            .map_err(|e| MeshError::Relay(format!("Relay allocation request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(MeshError::Relay(format!(
                "Relay allocation failed ({}): {}",
                status, body
            )));
        }

        #[derive(serde::Deserialize)]
        struct AllocateResponse {
            relay_addr: String,
        }

        let alloc: AllocateResponse = resp
            .json()
            .await
            .map_err(|e| MeshError::Relay(format!("Failed to parse relay response: {e}")))?;

        let relay_addr: SocketAddr = alloc
            .relay_addr
            .parse()
            .map_err(|e| MeshError::Relay(format!("Invalid relay address: {e}")))?;

        info!(
            "Relay allocated at {} for peer {}",
            relay_addr, target.node_id
        );
        Ok(relay_addr)
    }

    /// Check if any relay servers are available.
    pub async fn is_available(&self) -> bool {
        if self.relay_servers.is_empty() {
            return false;
        }

        let relay_server = &self.relay_servers[0];
        let url = if relay_server.starts_with("http") {
            format!("{}/health", relay_server)
        } else {
            format!("https://{}/health", relay_server)
        };

        let client = reqwest::Client::new();
        match client
            .get(&url)
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await
        {
            Ok(resp) => resp.status().is_success(),
            Err(e) => {
                warn!("Relay server health check failed: {e}");
                false
            }
        }
    }
}
