pub mod composite;
pub mod pex;
pub mod server;

use async_trait::async_trait;

use crate::error::Result;
use crate::peer::PeerInfo;

/// Trait for peer discovery mechanisms.
///
/// Implemented by the coordination server client (primary)
/// and libp2p DHT (decentralized fallback).
#[async_trait]
pub trait PeerDiscovery: Send + Sync {
    /// Register this node with the discovery service.
    async fn register(&self, info: &PeerInfo) -> Result<()>;

    /// Discover peers that can contribute to running a model.
    async fn find_peers(&self, model_id: &str, min_memory_bytes: u64) -> Result<Vec<PeerInfo>>;

    /// Send a heartbeat to maintain registration.
    async fn heartbeat(&self) -> Result<()>;

    /// Deregister this node from the discovery service.
    async fn deregister(&self) -> Result<()>;

    /// Relay a signalling message to another peer.
    ///
    /// Used to coordinate NAT hole punching: both sides must start dialling
    /// at roughly the same moment, and only the coordinator can reach a peer
    /// that is not yet reachable directly. Defaults to a no-op so discovery
    /// backends without a relay (mDNS, PEX) need not implement it.
    /// `action` and `from_addr` are the coordinator's `Signal` fields; do not
    /// invent others, as anything it does not declare is dropped by serde.
    async fn send_signal(
        &self,
        _to_node: &str,
        _action: &str,
        _from_addr: Option<&str>,
    ) -> Result<()> {
        Ok(())
    }

    /// Collect signalling messages addressed to this node.
    async fn poll_signals(&self) -> Result<Vec<serde_json::Value>> {
        Ok(Vec::new())
    }
}

/// Mock discovery for testing: holds peers in memory.
pub struct MockDiscovery {
    peers: tokio::sync::Mutex<Vec<PeerInfo>>,
}

impl MockDiscovery {
    pub fn new() -> Self {
        Self {
            peers: tokio::sync::Mutex::new(Vec::new()),
        }
    }

    pub fn with_peers(peers: Vec<PeerInfo>) -> Self {
        Self {
            peers: tokio::sync::Mutex::new(peers),
        }
    }
}

impl Default for MockDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl PeerDiscovery for MockDiscovery {
    async fn register(&self, info: &PeerInfo) -> Result<()> {
        let mut peers = self.peers.lock().await;
        // Replace if already registered (same node_id)
        peers.retain(|p| p.node_id != info.node_id);
        peers.push(info.clone());
        Ok(())
    }

    async fn find_peers(&self, _model_id: &str, min_memory_bytes: u64) -> Result<Vec<PeerInfo>> {
        let peers = self.peers.lock().await;
        Ok(peers
            .iter()
            .filter(|p| p.available_memory_bytes >= min_memory_bytes)
            .cloned()
            .collect())
    }

    async fn heartbeat(&self) -> Result<()> {
        Ok(())
    }

    async fn deregister(&self) -> Result<()> {
        self.peers.lock().await.clear();
        Ok(())
    }
}
