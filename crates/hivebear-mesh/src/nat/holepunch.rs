use std::net::SocketAddr;
use std::time::Duration;

use tracing::{debug, info};

use crate::error::{MeshError, Result};
use crate::peer::{NodeId, PeerInfo};
use crate::transport::MeshTransport;

/// Signal type used to ask a peer to start dialling us back.
pub const PUNCH_REQUEST: &str = "punch-request";

/// How often to re-dial while punching. NAT mappings are created by the
/// outbound packet, so both sides have to be dialling at roughly the same
/// time; repeating narrows the window that has to line up.
const RETRY_INTERVAL: Duration = Duration::from_millis(250);

/// Per-attempt connect timeout. Short, because we want many attempts inside
/// the overall budget rather than one long one.
const ATTEMPT_TIMEOUT: Duration = Duration::from_millis(750);

/// Read a punch request out of a relayed coordinator signal.
///
/// The coordinator's `Signal` carries exactly `{from_node, to_node,
/// from_addr, action}` — anything else is dropped by serde on the way in, so
/// the address to dial back must travel in `from_addr`.
///
/// Returns the address to dial, or `None` if this is not a usable punch
/// request.
pub fn punch_target_from_signal(signal: &serde_json::Value) -> Option<SocketAddr> {
    let action = signal.get("action").and_then(|v| v.as_str())?;
    if action != PUNCH_REQUEST {
        return None;
    }
    signal
        .get("from_addr")
        .and_then(|v| v.as_str())
        .and_then(|a| a.parse().ok())
}

/// Repeatedly dial `target` until it answers or `budget` runs out.
///
/// QUIC's own handshake packets are the punch packets: sending them creates
/// the outbound NAT mapping that lets the peer's packets back in. This used
/// to make a single `connect` to `peer.addr` — the peer's *LAN* address —
/// ignoring both the caller's carefully computed target and the local
/// external address entirely, so it was just a slower retry of the direct
/// connection that had already failed.
pub async fn attempt_holepunch(
    transport: &dyn MeshTransport,
    peer: &PeerInfo,
    target: SocketAddr,
    budget: Duration,
) -> Result<NodeId> {
    info!(
        "Hole punching to {} at {} for up to {:?}",
        peer.node_id, target, budget
    );

    let deadline = tokio::time::Instant::now() + budget;
    let mut attempts = 0u32;

    while tokio::time::Instant::now() < deadline {
        attempts += 1;
        match tokio::time::timeout(ATTEMPT_TIMEOUT, transport.connect(target)).await {
            Ok(Ok(peer_id)) => {
                info!("Hole punch to {target} succeeded after {attempts} attempt(s)");
                return Ok(peer_id);
            }
            Ok(Err(e)) => debug!("Punch attempt {attempts} to {target} failed: {e}"),
            Err(_) => debug!("Punch attempt {attempts} to {target} timed out"),
        }

        // Keep sending: the peer may not have started dialling yet.
        tokio::time::sleep(RETRY_INTERVAL).await;
    }

    Err(MeshError::NatTraversal(format!(
        "Hole punch to {} at {target} gave up after {attempts} attempt(s)",
        peer.node_id
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn reads_the_dial_target_from_a_relayed_signal() {
        let signal = json!({
            "from_node": "abc",
            "to_node": "def",
            "from_addr": "203.0.113.9:7878",
            "action": PUNCH_REQUEST,
        });
        assert_eq!(
            punch_target_from_signal(&signal),
            Some("203.0.113.9:7878".parse().unwrap())
        );
    }

    #[test]
    fn ignores_other_actions() {
        let signal = json!({
            "from_node": "abc",
            "to_node": "def",
            "from_addr": "203.0.113.9:7878",
            "action": "something-else",
        });
        assert_eq!(punch_target_from_signal(&signal), None);
    }

    /// The coordinator declares from_addr and action as optional, so a peer
    /// on an older build can relay a signal with neither.
    #[test]
    fn tolerates_missing_or_unparseable_fields() {
        assert_eq!(
            punch_target_from_signal(&json!({"from_node": "a", "to_node": "b"})),
            None
        );
        assert_eq!(
            punch_target_from_signal(&json!({"action": PUNCH_REQUEST})),
            None
        );
        assert_eq!(
            punch_target_from_signal(&json!({
                "action": PUNCH_REQUEST,
                "from_addr": "not-an-address",
            })),
            None
        );
    }
}
