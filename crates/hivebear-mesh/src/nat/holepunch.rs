use std::net::SocketAddr;
use std::time::Duration;

use tracing::{debug, info, warn};

use crate::error::{MeshError, Result};
use crate::peer::PeerInfo;
use crate::transport::MeshTransport;

/// Attempt UDP hole punching to connect to a peer behind NAT.
///
/// Both peers must simultaneously send packets to each other's external
/// addresses. The coordination server signals both sides to begin.
pub async fn attempt_holepunch(
    transport: &dyn MeshTransport,
    peer: &PeerInfo,
    _local_external_addr: Option<SocketAddr>,
    timeout: Duration,
) -> Result<()> {
    let target_addr = peer.addr;

    info!(
        "Attempting UDP hole punch to {} at {}",
        peer.node_id, target_addr
    );

    // Try to connect with the given timeout.
    // The QUIC connect will send initial handshake packets which act
    // as the hole-punch packets for the NAT.
    let result = tokio::time::timeout(timeout, transport.connect(target_addr)).await;

    match result {
        Ok(Ok(peer_id)) => {
            info!("Hole punch succeeded: connected to {}", peer_id);
            Ok(())
        }
        Ok(Err(e)) => {
            debug!("Hole punch connection failed: {e}");
            Err(MeshError::NatTraversal(format!(
                "Hole punch to {} failed: {e}",
                peer.node_id
            )))
        }
        Err(_) => {
            warn!("Hole punch timed out after {:?}", timeout);
            Err(MeshError::NatTraversal(format!(
                "Hole punch to {} timed out",
                peer.node_id
            )))
        }
    }
}
