use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;
use tracing::debug;

use crate::discovery::PeerDiscovery;
use crate::error::Result;
use crate::peer::{NodeId, PeerInfo};
use crate::transport::MeshTransport;

/// Maximum number of peers to track in the PEX cache.
const MAX_PEERS: usize = 200;

/// Minimum reputation to share a peer via PEX.
const MIN_REPUTATION_TO_SHARE: f64 = 0.3;

/// Peer Exchange (PEX) discovery.
///
/// Peers gossip their known peer lists to reduce dependency on the
/// coordination server. Periodically sends PeerExchangeRequest to
/// random connected peers and merges the responses.
pub struct PexDiscovery {
    transport: Arc<dyn MeshTransport>,
    known_peers: DashMap<Vec<u8>, PeerInfo>,
}

impl PexDiscovery {
    pub fn new(transport: Arc<dyn MeshTransport>) -> Self {
        Self {
            transport,
            known_peers: DashMap::new(),
        }
    }

    /// Add a peer to the known peers cache.
    pub fn add_peer(&self, peer: PeerInfo) {
        if self.known_peers.len() >= MAX_PEERS {
            return; // Cache full
        }
        let key = peer.node_id.0.to_bytes().to_vec();
        self.known_peers.insert(key, peer);
    }

    /// Remove a peer from the cache.
    pub fn remove_peer(&self, node_id: &NodeId) {
        let key = node_id.0.to_bytes().to_vec();
        self.known_peers.remove(&key);
    }

    /// Get all known peers with reputation above the sharing threshold.
    pub fn shareable_peers(&self) -> Vec<PeerInfo> {
        self.known_peers
            .iter()
            .filter(|entry| entry.value().reputation_score >= MIN_REPUTATION_TO_SHARE)
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// Handle an incoming PeerExchangeRequest by returning our known peers.
    pub fn handle_exchange_request(&self, max_peers: u32) -> Vec<PeerInfo> {
        self.shareable_peers()
            .into_iter()
            .take(max_peers as usize)
            .collect()
    }

    /// Merge peers received from a PeerExchangeResponse into our cache.
    pub fn merge_peers(&self, peers: Vec<PeerInfo>) {
        for peer in peers {
            if self.known_peers.len() >= MAX_PEERS {
                break;
            }
            let key = peer.node_id.0.to_bytes().to_vec();
            // Only add if we don't already know this peer
            self.known_peers.entry(key).or_insert(peer);
        }
    }

    /// Number of known peers in the cache.
    pub fn peer_count(&self) -> usize {
        self.known_peers.len()
    }

    /// Get a reference to the underlying transport.
    pub fn transport(&self) -> &Arc<dyn MeshTransport> {
        &self.transport
    }
}

#[async_trait]
impl PeerDiscovery for PexDiscovery {
    async fn register(&self, info: &PeerInfo) -> Result<()> {
        self.add_peer(info.clone());
        debug!("PEX: registered peer {}", info.node_id);
        Ok(())
    }

    async fn find_peers(&self, _model_id: &str, min_memory_bytes: u64) -> Result<Vec<PeerInfo>> {
        Ok(self
            .known_peers
            .iter()
            .map(|entry| entry.value().clone())
            .filter(|p| p.available_memory_bytes >= min_memory_bytes)
            .collect())
    }

    async fn heartbeat(&self) -> Result<()> {
        // PEX doesn't need heartbeats — gossip is opportunistic
        Ok(())
    }

    async fn deregister(&self) -> Result<()> {
        self.known_peers.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MeshTier;
    use crate::transport::mock::{MockRegistry, MockTransport};

    fn make_peer(reputation: f64) -> PeerInfo {
        let (node_id, _) = NodeId::generate();
        PeerInfo {
            node_id,
            hardware: hivebear_core::profile(),
            available_memory_bytes: 8 * 1024 * 1024 * 1024,
            available_vram_bytes: 0,
            network_bandwidth_mbps: 100.0,
            latency_ms: None,
            tier: MeshTier::Free,
            reputation_score: reputation,
            addr: "127.0.0.1:7878".parse().unwrap(),
            external_addr: None,
            nat_type: crate::nat::NatType::Unknown,
            latency_map: std::collections::HashMap::new(),
            serving_model_id: None,
            swarm_id: None,
            draft_capability: None,
        }
    }

    fn make_pex() -> PexDiscovery {
        let (id, _) = NodeId::generate();
        let registry = MockRegistry::new();
        let transport = Arc::new(MockTransport::new(id, registry));
        PexDiscovery::new(transport)
    }

    #[test]
    fn test_add_and_find_peers() {
        let pex = make_pex();
        let peer = make_peer(0.8);
        pex.add_peer(peer);
        assert_eq!(pex.peer_count(), 1);
    }

    #[test]
    fn test_reputation_filter() {
        let pex = make_pex();
        pex.add_peer(make_peer(0.8));
        pex.add_peer(make_peer(0.1));

        let shareable = pex.shareable_peers();
        assert_eq!(shareable.len(), 1);
        assert!(shareable[0].reputation_score >= MIN_REPUTATION_TO_SHARE);
    }

    #[test]
    fn test_merge_peers() {
        let pex = make_pex();
        let peers = vec![make_peer(0.5), make_peer(0.9)];
        pex.merge_peers(peers);
        assert_eq!(pex.peer_count(), 2);
    }

    #[test]
    fn test_remove_peer() {
        let pex = make_pex();
        let peer = make_peer(0.7);
        let id = peer.node_id.clone();
        pex.add_peer(peer);
        assert_eq!(pex.peer_count(), 1);
        pex.remove_peer(&id);
        assert_eq!(pex.peer_count(), 0);
    }

    #[test]
    fn test_handle_exchange_request() {
        let pex = make_pex();
        pex.add_peer(make_peer(0.8));
        pex.add_peer(make_peer(0.9));
        pex.add_peer(make_peer(0.1)); // Below threshold

        let result = pex.handle_exchange_request(10);
        assert_eq!(result.len(), 2); // Only the two above threshold
    }

    #[test]
    fn test_max_peers_limit() {
        let pex = make_pex();
        for _ in 0..MAX_PEERS + 10 {
            pex.add_peer(make_peer(0.5));
        }
        assert_eq!(pex.peer_count(), MAX_PEERS);
    }

    #[tokio::test]
    async fn test_discovery_trait_find_peers() {
        let pex = make_pex();
        pex.add_peer(make_peer(0.8));

        let found = pex.find_peers("any-model", 0).await.unwrap();
        assert_eq!(found.len(), 1);
    }

    #[tokio::test]
    async fn test_discovery_trait_find_peers_memory_filter() {
        let pex = make_pex();
        pex.add_peer(make_peer(0.8)); // 8GB

        // Require more memory than available
        let found = pex
            .find_peers("any-model", 16 * 1024 * 1024 * 1024)
            .await
            .unwrap();
        assert_eq!(found.len(), 0);
    }

    #[tokio::test]
    async fn test_deregister_clears_cache() {
        let pex = make_pex();
        pex.add_peer(make_peer(0.8));
        assert_eq!(pex.peer_count(), 1);

        pex.deregister().await.unwrap();
        assert_eq!(pex.peer_count(), 0);
    }
}
