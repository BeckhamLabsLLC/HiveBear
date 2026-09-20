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
    /// Bearer token issued by POST /register.
    ///
    /// The response body was previously discarded, so no request ever carried
    /// an Authorization header. /signal validates one and returns 401 without
    /// it — and send_signal treats a non-success status as non-fatal, so NAT
    /// signalling failed completely silently.
    auth_token: tokio::sync::Mutex<Option<String>>,
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
            auth_token: tokio::sync::Mutex::new(None),
        }
    }

    /// Token issued at registration, if we have one.
    pub async fn auth_token(&self) -> Option<String> {
        self.auth_token.lock().await.clone()
    }

    /// Attach the registration bearer token, when we have one.
    async fn authed(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match self.auth_token.lock().await.as_ref() {
            Some(token) => req.bearer_auth(token),
            None => req,
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
                // Keep the bearer token; /signal and the other authenticated
                // endpoints are unusable without it.
                match resp.json::<serde_json::Value>().await {
                    Ok(body) => {
                        if let Some(token) = body.get("token").and_then(|t| t.as_str()) {
                            *self.auth_token.lock().await = Some(token.to_string());
                            debug!("Registered; received coordination token");
                        } else {
                            warn!("Register response carried no token; signalling will fail");
                        }
                    }
                    Err(e) => warn!("Could not parse register response: {e}"),
                }
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
                // Unreachable is still a failure to register. This used to
                // return Ok(()), so MeshNode::start "succeeded", `running`
                // flipped to true, and the CLI and desktop both told the user
                // they were "Connected to Hive" while connected to nothing.
                // Degrading gracefully is the caller's decision to make, and
                // it cannot make it if we lie here.
                warn!(
                    "Coordination server at {} not reachable: {e}",
                    self.base_url
                );
                Err(MeshError::Discovery(format!(
                    "Coordination server at {} not reachable: {e}",
                    self.base_url
                )))
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
        let req = self.authed(self.http.post(&url)).await.json(info);

        // Report failures as failures. This returned Ok(()) for a rejected
        // heartbeat *and* for an unreachable server, so a caller could not
        // tell a live registration from a dead one — which would have made
        // MeshNode's `registered` flag report success against a coordinator
        // that had rejected us outright. Callers decide whether it is fatal;
        // the maintenance loop treats it as "not registered, keep trying".
        match req.send().await {
            Ok(resp) if resp.status().is_success() => Ok(()),
            Ok(resp) => {
                let status = resp.status();
                warn!("Heartbeat rejected with {status}");
                Err(MeshError::Discovery(format!(
                    "Coordination server rejected heartbeat: {status}"
                )))
            }
            Err(e) if e.is_connect() || e.is_timeout() => {
                debug!("Heartbeat skipped (server unreachable): {e}");
                Err(MeshError::Discovery(format!(
                    "Coordination server unreachable: {e}"
                )))
            }
            Err(e) => Err(MeshError::Discovery(format!("Heartbeat failed: {e}"))),
        }
    }

    async fn send_signal(
        &self,
        to_node: &str,
        action: &str,
        from_addr: Option<&str>,
    ) -> Result<()> {
        let from = match self.node_info.lock().await.as_ref() {
            Some(i) => i.node_id.to_hex(),
            None => return Err(MeshError::Discovery("Not registered".into())),
        };
        CoordinationServerClient::send_signal(self, &from, to_node, action, from_addr).await
    }

    async fn poll_signals(&self) -> Result<Vec<serde_json::Value>> {
        let node_id = match self.node_info.lock().await.as_ref() {
            Some(i) => i.node_id.to_hex(),
            None => return Ok(Vec::new()),
        };
        CoordinationServerClient::poll_signals(self, &node_id).await
    }

    async fn deregister(&self) -> Result<()> {
        let info = self.node_info.lock().await;
        if info.is_none() {
            return Ok(());
        }

        // The server takes {"node_id": ...} plus a bearer token. This sent
        // neither, so it could not identify the caller and every deregister
        // was rejected — departing peers lingered in the registry until they
        // timed out, and other nodes kept trying to dial them.
        let node_id = info.as_ref().map(|i| i.node_id.to_hex());
        let url = format!("{}/deregister", self.base_url);
        if let Some(node_id) = node_id {
            let req = self
                .authed(self.http.delete(&url))
                .await
                .json(&serde_json::json!({ "node_id": node_id }));
            match req.send().await {
                Ok(resp) if resp.status().is_success() => {
                    debug!("Deregistered from coordination server");
                }
                Ok(resp) => {
                    warn!("Deregister returned {}", resp.status());
                }
                Err(e) => {
                    // Non-fatal — the node will eventually time out.
                    debug!("Deregister failed (non-fatal): {e}");
                }
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

    // ── Signal relay methods (NAT traversal coordination) ────────────

    /// POST /signal — send a signaling message to another peer via the coordinator.
    ///
    /// Used for NAT traversal coordination: exchanging external addresses,
    /// coordinating simultaneous QUIC handshakes for hole-punching, etc.
    /// The body must match the coordinator's `Signal` type exactly:
    /// `{from_node, to_node, from_addr, action}`. It previously sent
    /// `signal_type` and `payload`, which that struct does not declare — so
    /// serde dropped both on the way in and every relayed signal arrived
    /// empty. Nothing surfaced, because a non-2xx response here is treated
    /// as non-fatal.
    pub async fn send_signal(
        &self,
        from_node: &str,
        to_node: &str,
        action: &str,
        from_addr: Option<&str>,
    ) -> Result<()> {
        let url = format!("{}/signal", self.base_url);
        let body = serde_json::json!({
            "from_node": from_node,
            "to_node": to_node,
            "from_addr": from_addr,
            "action": action,
        });

        let req = self.authed(self.http.post(&url)).await.json(&body);
        match req.send().await {
            Ok(resp) if resp.status().is_success() => Ok(()),
            Ok(resp) => {
                let status = resp.status();
                // Louder than debug!: a 401 here means NAT traversal is dead,
                // and it used to disappear without trace.
                warn!("Signal to {to_node} rejected with {status}");
                Ok(()) // Non-fatal — a direct connection may still work
            }
            Err(e) if e.is_connect() || e.is_timeout() => {
                debug!("Signal relay unreachable (non-fatal): {e}");
                Ok(())
            }
            Err(e) => Err(MeshError::Discovery(format!("Signal send failed: {e}"))),
        }
    }

    /// GET /signals?node_id=... — retrieve pending signaling messages for this node.
    ///
    /// Returns signals from other peers (connection requests, NAT info exchange).
    /// The coordinator clears retrieved signals automatically.
    pub async fn poll_signals(&self, node_id: &str) -> Result<Vec<serde_json::Value>> {
        let url = format!("{}/signals?node_id={}", self.base_url, node_id);

        let req = self.authed(self.http.get(&url)).await;
        match req.send().await {
            Ok(resp) if resp.status().is_success() => {
                let signals: Vec<serde_json::Value> = resp
                    .json()
                    .await
                    .map_err(|e| MeshError::Discovery(format!("Failed to parse signals: {e}")))?;
                Ok(signals)
            }
            Ok(_) => Ok(Vec::new()),
            Err(e) if e.is_connect() || e.is_timeout() => {
                debug!("Signal poll failed (non-fatal): {e}");
                Ok(Vec::new())
            }
            Err(e) => Err(MeshError::Discovery(format!("Signal poll failed: {e}"))),
        }
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
