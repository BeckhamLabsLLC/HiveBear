pub mod compression;
pub mod mock;
pub mod protocol;
pub mod quic;
pub mod tensor_transfer;

use std::net::SocketAddr;

use async_trait::async_trait;

use crate::error::Result;
use crate::peer::NodeId;
use protocol::MeshMessage;

/// Abstraction over the data transport layer.
///
/// Implemented by `QuicTransport` (production) and `MockTransport` (testing).
#[async_trait]
pub trait MeshTransport: Send + Sync {
    /// Send a message to a specific peer.
    async fn send(&self, peer: &NodeId, msg: MeshMessage) -> Result<()>;

    /// Receive the next message from any connected peer.
    async fn recv(&self) -> Result<(NodeId, MeshMessage)>;

    /// Connect to a peer at the given address. Returns their NodeId after handshake.
    async fn connect(&self, addr: SocketAddr) -> Result<NodeId>;

    /// Disconnect from a peer.
    async fn disconnect(&self, peer: &NodeId) -> Result<()>;

    /// Start listening for incoming connections on the given address.
    async fn listen(&self, addr: SocketAddr) -> Result<()>;

    /// Check if a peer is currently connected.
    fn is_connected(&self, peer: &NodeId) -> bool;

    /// Number of currently connected peers.
    fn peer_count(&self) -> usize;
}
