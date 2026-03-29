use async_trait::async_trait;
use tracing::{debug, warn};

use super::PeerDiscovery;
use crate::error::{MeshError, Result};
use crate::peer::PeerInfo;

/// Client for the centralized coordination server.
///
/// The coordination server is a lightweight HTTP service that maintains
/// a registry of mesh peers. It provides:
/// - POST /register — register a node
/// - POST /heartbeat — maintain registration
/// - GET /peers?model={id}&min_memory={bytes} — find peers
/// - DELETE /deregister — remove registration
pub struct CoordinationServerClient {
    base_url: String,
    http: reqwest::Client,
    node_info: tokio::sync::Mutex<Option<PeerInfo>>,
}

impl CoordinationServerClient {
    pub fn new(base_url: String) -> Self {
        if !base_url.starts_with("https://") {
            tracing::warn!(
                "Coordination server URL uses HTTP ({}). Credentials will be sent in cleartext. \
                 Use https:// in production.",
                base_url
            );
        }
        Self {
            base_url,
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .build()
                .unwrap_or_default(),
            node_info: tokio::sync::Mutex::new(None),
        }
    }
}

#[async_trait]
impl PeerDiscovery for CoordinationServerClient {
    async fn register(&self, info: &PeerInfo) -> Result<()> {
        let url = format!("{}/register", self.base_url);
        debug!("Registering with coordination server at {url}");

        // Store info for heartbeats
        *self.node_info.lock().await = Some(info.clone());

        match self.http.post(&url).json(info).send().await {
            Ok(resp) if resp.status().is_success() => {
                debug!("Registered successfully with coordination server");
                Ok(())
            }
            Ok(resp) => {
                let status = resp.status();
                let body = resp.text().await.unwrap_or_default();
                warn!("Registration failed: {status} — {body}");
                Err(MeshError::Discovery(format!(
                    "Coordination server returned {status}"
                )))
            }
            Err(e) if e.is_connect() || e.is_timeout() => {
                // Server not reachable — degrade gracefully
                warn!(
                    "Coordination server at {} not reachable: {e}. Running in local-only mode.",
                    self.base_url
                );
                Ok(())
            }
            Err(e) => Err(MeshError::Discovery(format!(
                "Failed to contact coordination server: {e}"
            ))),
        }
    }

    async fn find_peers(&self, model_id: &str, min_memory_bytes: u64) -> Result<Vec<PeerInfo>> {
        let url = format!(
            "{}/peers?model={}&min_memory={}",
            self.base_url, model_id, min_memory_bytes
        );
        debug!("Finding peers via {url}");

        match self.http.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => {
                let peers: Vec<PeerInfo> = resp
                    .json()
                    .await
                    .map_err(|e| MeshError::Discovery(format!("Failed to parse peer list: {e}")))?;
                debug!("Found {} peers from coordination server", peers.len());
                Ok(peers)
            }
            Ok(resp) => {
                let status = resp.status();
                warn!("Peer discovery returned {status}");
                Ok(Vec::new())
            }
            Err(e) if e.is_connect() || e.is_timeout() => {
                warn!("Coordination server not reachable for peer discovery: {e}");
                Ok(Vec::new())
            }
            Err(e) => Err(MeshError::Discovery(format!("Peer discovery failed: {e}"))),
        }
    }

    async fn heartbeat(&self) -> Result<()> {
        let info = self.node_info.lock().await;
        let info = match info.as_ref() {
            Some(i) => i,
            None => return Err(MeshError::Discovery("Not registered".into())),
        };

        let url = format!("{}/heartbeat", self.base_url);
        match self.http.post(&url).json(info).send().await {
            Ok(resp) if resp.status().is_success() => Ok(()),
            Ok(resp) => {
                let status = resp.status();
                warn!("Heartbeat returned {status}");
                Ok(()) // Non-fatal
            }
            Err(e) if e.is_connect() || e.is_timeout() => {
                debug!("Heartbeat skipped (server unreachable): {e}");
                Ok(()) // Non-fatal
            }
            Err(e) => Err(MeshError::Discovery(format!("Heartbeat failed: {e}"))),
        }
    }

    async fn deregister(&self) -> Result<()> {
        let info = self.node_info.lock().await;
        if info.is_none() {
            return Ok(());
        }

        let url = format!("{}/deregister", self.base_url);
        match self.http.delete(&url).send().await {
            Ok(_) => {
                debug!("Deregistered from coordination server");
            }
            Err(e) => {
                // Non-fatal — node will eventually time out
                debug!("Deregister failed (non-fatal): {e}");
            }
        }

        drop(info);
        *self.node_info.lock().await = None;
        Ok(())
    }
}

// ── Extended coordination client methods (swarm management) ──────────

impl CoordinationServerClient {
    /// POST /matchmake — find or create the best swarm for this peer.
    pub async fn matchmake(
        &self,
        node_id: &str,
        total_vram_bytes: i64,
        total_ram_bytes: i64,
        preferred_model: Option<&str>,
    ) -> Result<serde_json::Value> {
        let url = format!("{}/matchmake", self.base_url);
        let mut body = serde_json::json!({
            "node_id": node_id,
            "total_vram_bytes": total_vram_bytes,
            "total_ram_bytes": total_ram_bytes,
        });
        if let Some(model) = preferred_model {
            body["preferred_model"] = serde_json::json!(model);
        }

        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| MeshError::Discovery(format!("Matchmake request failed: {e}")))?;

        if resp.status().is_success() {
            resp.json().await.map_err(|e| {
                MeshError::Discovery(format!("Failed to parse matchmake response: {e}"))
            })
        } else {
            let status = resp.status();
            Err(MeshError::Discovery(format!("Matchmake returned {status}")))
        }
    }

    /// POST /swarms/:id/join — join a swarm.
    pub async fn join_swarm(
        &self,
        swarm_id: &str,
        node_id: &str,
        role: &str,
        layers_from: Option<i64>,
        layers_to: Option<i64>,
    ) -> Result<()> {
        let url = format!("{}/swarms/{}/join", self.base_url, swarm_id);
        let body = serde_json::json!({
            "node_id": node_id,
            "role": role,
            "layers_from": layers_from,
            "layers_to": layers_to,
        });

        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| MeshError::Discovery(format!("Join swarm failed: {e}")))?;

        if resp.status().is_success() {
            Ok(())
        } else {
            let status = resp.status();
            Err(MeshError::Discovery(format!(
                "Join swarm returned {status}"
            )))
        }
    }

    /// POST /swarms/:id/leave — leave a swarm.
    pub async fn leave_swarm(&self, swarm_id: &str, node_id: &str) -> Result<()> {
        let url = format!("{}/swarms/{}/leave", self.base_url, swarm_id);
        let body = serde_json::json!({ "node_id": node_id });

        let resp = self
            .http
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| MeshError::Discovery(format!("Leave swarm failed: {e}")))?;

        if resp.status().is_success() {
            Ok(())
        } else {
            let status = resp.status();
            Err(MeshError::Discovery(format!(
                "Leave swarm returned {status}"
            )))
        }
    }

    /// GET /models — list all models the network can serve.
    pub async fn get_models(&self) -> Result<Vec<serde_json::Value>> {
        let url = format!("{}/models", self.base_url);
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|e| MeshError::Discovery(format!("Get models failed: {e}")))?;

        if resp.status().is_success() {
            resp.json()
                .await
                .map_err(|e| MeshError::Discovery(format!("Failed to parse models: {e}")))
        } else {
            Ok(Vec::new())
        }
    }

    /// GET /dashboard — aggregate network stats.
    pub async fn get_dashboard(&self) -> Result<serde_json::Value> {
        let url = format!("{}/dashboard", self.base_url);
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|e| MeshError::Discovery(format!("Get dashboard failed: {e}")))?;

        resp.json()
            .await
            .map_err(|e| MeshError::Discovery(format!("Failed to parse dashboard: {e}")))
    }

    /// GET /me — node's own profile and contribution status.
    pub async fn get_me(&self) -> Result<serde_json::Value> {
        let url = format!("{}/me", self.base_url);
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .map_err(|e| MeshError::Discovery(format!("Get me failed: {e}")))?;

        if resp.status().is_success() {
            resp.json()
                .await
                .map_err(|e| MeshError::Discovery(format!("Failed to parse me: {e}")))
        } else {
            Err(MeshError::Discovery("Not authenticated".into()))
        }
    }
}
