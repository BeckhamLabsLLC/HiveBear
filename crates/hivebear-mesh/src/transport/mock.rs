use std::net::SocketAddr;
use std::sync::Arc;

use super::inbox::Inbox;
use super::protocol::MeshMessage;
use super::MeshTransport;
use crate::error::{MeshError, Result};
use crate::peer::NodeId;
use async_trait::async_trait;
use dashmap::DashMap;

/// In-process mock transport for testing.
///
/// Uses tokio mpsc channels to simulate network communication between
/// nodes within the same process. No real networking involved.
pub struct MockTransport {
    local_id: NodeId,
    /// Outbound routes: NodeId -> that peer's inbox.
    peers: DashMap<Vec<u8>, Arc<Inbox>>,
    /// Our own inbox. Shares the session-routing logic with QuicTransport so
    /// tests exercise the same demultiplexing as production.
    inbox: Arc<Inbox>,
    /// Shared registry for mock peer discovery (addr -> inbox sender).
    registry: Arc<MockRegistry>,
}

/// Shared registry allowing mock nodes to find each other by address.
pub struct MockRegistry {
    listeners: DashMap<String, (NodeId, Arc<Inbox>)>,
}

impl MockRegistry {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            listeners: DashMap::new(),
        })
    }
}

impl Default for MockRegistry {
    fn default() -> Self {
        Self {
            listeners: DashMap::new(),
        }
    }
}

impl MockTransport {
    pub fn new(local_id: NodeId, registry: Arc<MockRegistry>) -> Self {
        Self {
            local_id,
            peers: DashMap::new(),
            inbox: Arc::new(Inbox::new()),
            registry,
        }
    }

    fn node_key(id: &NodeId) -> Vec<u8> {
        id.0.to_bytes().to_vec()
    }

    /// This transport's node id.
    pub fn local_id(&self) -> &NodeId {
        &self.local_id
    }

    /// Give this transport a one-way route to `other`, without going through
    /// listen/connect. Test helper for wiring up more than two nodes, which
    /// `create_linked_pair` cannot express.
    pub fn connect_for_test(&self, other: &MockTransport) {
        self.peers
            .insert(Self::node_key(&other.local_id), Arc::clone(&other.inbox));
    }
}

#[async_trait]
impl MeshTransport for MockTransport {
    async fn send(&self, peer: &NodeId, msg: MeshMessage) -> Result<()> {
        let key = Self::node_key(peer);
        let target = self
            .peers
            .get(&key)
            .ok_or_else(|| MeshError::PeerDisconnected(peer.to_string()))?;
        if !target.deliver(self.local_id.clone(), msg) {
            return Err(MeshError::PeerDisconnected(peer.to_string()));
        }
        Ok(())
    }

    async fn recv(&self) -> Result<(NodeId, MeshMessage)> {
        self.inbox.recv().await
    }

    async fn connect(&self, addr: SocketAddr) -> Result<NodeId> {
        let addr_key = addr.to_string();
        let entry = self
            .registry
            .listeners
            .get(&addr_key)
            .ok_or_else(|| MeshError::Transport(format!("No listener at {addr}")))?;
        let (peer_id, peer_inbox) = entry.value().clone();

        // Give the remote peer a channel to send to us
        let key = Self::node_key(&peer_id);
        self.peers.insert(key, peer_inbox);

        // Also register ourselves in the remote's perspective
        // (the remote needs our inbox_tx to send back)
        // In tests, both sides are typically linked via create_linked_pair().
        // The real integration happens when both MockTransports share the same registry.

        Ok(peer_id)
    }

    async fn disconnect(&self, peer: &NodeId) -> Result<()> {
        let key = Self::node_key(peer);
        self.peers.remove(&key);
        Ok(())
    }

    async fn listen(&self, addr: SocketAddr) -> Result<()> {
        let addr_key = addr.to_string();
        self.registry
            .listeners
            .insert(addr_key, (self.local_id.clone(), Arc::clone(&self.inbox)));
        Ok(())
    }

    fn is_connected(&self, peer: &NodeId) -> bool {
        let key = Self::node_key(peer);
        self.peers.contains_key(&key)
    }

    fn peer_count(&self) -> usize {
        self.peers.len()
    }

    fn subscribe_session(&self, session_id: uuid::Uuid) -> super::inbox::SessionReceiver {
        self.inbox.subscribe(session_id)
    }

    fn unsubscribe_session(&self, session_id: &uuid::Uuid) {
        self.inbox.unsubscribe(session_id);
    }
}

/// Create a directly-linked pair of MockTransports for testing.
///
/// Messages sent from A to B's NodeId arrive in B's inbox, and vice versa.
pub fn create_linked_pair() -> (MockTransport, MockTransport) {
    let registry = MockRegistry::new();
    let (id_a, _key_a) = NodeId::generate();
    let (id_b, _key_b) = NodeId::generate();

    let transport_a = MockTransport {
        local_id: id_a.clone(),
        peers: DashMap::new(),
        inbox: Arc::new(Inbox::new()),
        registry: registry.clone(),
    };

    let transport_b = MockTransport {
        local_id: id_b.clone(),
        peers: DashMap::new(),
        inbox: Arc::new(Inbox::new()),
        registry,
    };

    // Cross-link: A can send to B's inbox, B can send to A's inbox
    let key_b = MockTransport::node_key(&id_b);
    transport_a
        .peers
        .insert(key_b, Arc::clone(&transport_b.inbox));

    let key_a = MockTransport::node_key(&id_a);
    transport_b
        .peers
        .insert(key_a, Arc::clone(&transport_a.inbox));

    (transport_a, transport_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_linked_pair_send_recv() {
        let (a, b) = create_linked_pair();

        let b_id = b.local_id.clone();
        let a_id = a.local_id.clone();

        // A sends to B
        a.send(&b_id, MeshMessage::Ping { timestamp_ms: 1000 })
            .await
            .unwrap();

        // B receives from A
        let (sender_id, msg) = b.recv().await.unwrap();
        assert_eq!(sender_id, a_id);
        match msg {
            MeshMessage::Ping { timestamp_ms } => assert_eq!(timestamp_ms, 1000),
            _ => panic!("Expected Ping"),
        }
    }

    #[tokio::test]
    async fn test_bidirectional_communication() {
        let (a, b) = create_linked_pair();
        let b_id = b.local_id.clone();
        let a_id = a.local_id.clone();

        // A -> B
        a.send(&b_id, MeshMessage::Ping { timestamp_ms: 100 })
            .await
            .unwrap();

        // B -> A
        b.send(&a_id, MeshMessage::Pong { timestamp_ms: 100 })
            .await
            .unwrap();

        // B reads A's message
        let (_, msg) = b.recv().await.unwrap();
        assert!(matches!(msg, MeshMessage::Ping { .. }));

        // A reads B's message
        let (_, msg) = a.recv().await.unwrap();
        assert!(matches!(msg, MeshMessage::Pong { .. }));
    }

    #[tokio::test]
    async fn test_disconnect_prevents_send() {
        let (a, b) = create_linked_pair();
        let b_id = b.local_id.clone();

        a.disconnect(&b_id).await.unwrap();
        assert!(!a.is_connected(&b_id));

        let result = a.send(&b_id, MeshMessage::Ping { timestamp_ms: 0 }).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_peer_count() {
        let (a, _b) = create_linked_pair();
        assert_eq!(a.peer_count(), 1);
    }
}
