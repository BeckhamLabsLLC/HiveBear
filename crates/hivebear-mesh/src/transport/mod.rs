pub mod compression;
pub mod inbox;
pub mod mock;
pub mod protocol;
pub mod quic;
pub mod tensor_transfer;

use std::net::SocketAddr;

use async_trait::async_trait;

use crate::error::Result;
use crate::peer::NodeId;
use inbox::SessionReceiver;
use protocol::MeshMessage;
use uuid::Uuid;

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

    /// Claim a session's inbound messages.
    ///
    /// Messages carrying `session_id` are delivered to the returned queue
    /// rather than the shared one that [`Self::recv`] drains. Without this,
    /// concurrent consumers steal each other's messages — see
    /// [`inbox::Inbox`]. Callers must [`Self::unsubscribe_session`] when the
    /// session ends.
    fn subscribe_session(&self, session_id: Uuid) -> SessionReceiver;

    /// Release a session claimed with [`Self::subscribe_session`].
    fn unsubscribe_session(&self, session_id: &Uuid);

    /// The NAT mapping discovered for the listening socket, if the transport
    /// probed for one. Only meaningful after `listen`.
    async fn discovered_external_addr(&self) -> Option<SocketAddr> {
        None
    }

    /// Permit `peer` to reach us through our TURN allocation.
    ///
    /// Only the peer behind a symmetric NAT needs an allocation: everyone
    /// else simply dials the relayed address it advertises. What the
    /// allocating node must do is authorise each peer, or the relay drops
    /// their traffic. Defaults to an error for transports with no relay.
    async fn relay_to(&self, _peer: SocketAddr) -> Result<()> {
        Err(crate::error::MeshError::Relay(
            "this transport has no relay".into(),
        ))
    }
}
