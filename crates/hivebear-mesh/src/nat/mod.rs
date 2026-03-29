pub mod holepunch;
pub mod relay;
pub mod stun;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

use crate::error::Result;
use crate::peer::{NodeId, PeerInfo};

/// Detected NAT type for this node.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum NatType {
    /// No NAT, directly reachable from the internet.
    Open,
    /// Full cone NAT: any external host can send to the mapped port.
    FullCone,
    /// Restricted cone: only hosts we've sent to can reply.
    RestrictedCone,
    /// Port-restricted cone: same as above, port-specific.
    PortRestricted,
    /// Symmetric NAT: different mapping per destination. Hole-punch fails.
    Symmetric,
    /// Could not determine NAT type.
    #[default]
    Unknown,
}

/// Trait for signaling between peers via the coordination server.
#[async_trait]
pub trait NatSignaler: Send + Sync {
    /// Send a connection signal to a remote peer via the coordination server.
    async fn send_signal(&self, target: &NodeId, signal: ConnectionSignal) -> Result<()>;
    /// Receive pending signals for this node.
    async fn receive_signals(&self) -> Result<Vec<ConnectionSignal>>;
}

/// A signal relayed between peers for NAT traversal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionSignal {
    pub from_node: NodeId,
    pub from_addr: SocketAddr,
    pub action: SignalAction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalAction {
    /// Request the target to initiate a simultaneous connection.
    Connect,
    /// Acknowledge a connection request.
    ConnectAck,
}

/// NAT traversal coordinator that manages the connection strategy.
pub struct NatTraversal {
    pub external_addr: Option<SocketAddr>,
    pub nat_type: NatType,
    pub stun_servers: Vec<String>,
    pub relay_servers: Vec<String>,
    pub relay_enabled: bool,
}

impl NatTraversal {
    pub fn new(stun_servers: Vec<String>, relay_servers: Vec<String>, relay_enabled: bool) -> Self {
        Self {
            external_addr: None,
            nat_type: NatType::Unknown,
            stun_servers,
            relay_servers,
            relay_enabled,
        }
    }

    /// Discover external address using STUN.
    pub async fn discover_external_addr(&mut self) -> Result<Option<SocketAddr>> {
        if self.stun_servers.is_empty() {
            return Ok(None);
        }

        match stun::discover_external_addr(&self.stun_servers[0]).await {
            Ok(addr) => {
                self.external_addr = Some(addr);
                self.nat_type = NatType::Unknown; // Full detection requires multiple probes
                Ok(Some(addr))
            }
            Err(e) => {
                tracing::warn!("STUN discovery failed: {e}");
                Ok(None)
            }
        }
    }

    /// Attempt to connect to a peer, trying strategies in order:
    /// 1. Direct connect (if external_addr is known)
    /// 2. Hole punch (via signaling)
    /// 3. Relay (fallback)
    pub async fn connect_strategy(&self, _peer: &PeerInfo) -> ConnectionStrategy {
        if self.nat_type == NatType::Open {
            return ConnectionStrategy::Direct;
        }

        if self.nat_type != NatType::Symmetric {
            return ConnectionStrategy::HolePunch;
        }

        if self.relay_enabled && !self.relay_servers.is_empty() {
            return ConnectionStrategy::Relay;
        }

        ConnectionStrategy::Direct // Best effort
    }
}

/// Strategy selected for connecting to a peer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionStrategy {
    Direct,
    HolePunch,
    Relay,
}
