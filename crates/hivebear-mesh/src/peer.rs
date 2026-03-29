use std::collections::HashMap;
use std::fmt;
use std::net::SocketAddr;
use std::ops::Range;

use ed25519_dalek::VerifyingKey;
use hivebear_core::types::HardwareProfile;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::config::MeshTier;
use crate::swarm::DraftCapability;

/// Unique cryptographic identity for a mesh node.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct NodeId(pub VerifyingKey);

impl NodeId {
    /// Generate a new random node identity.
    pub fn generate() -> (Self, ed25519_dalek::SigningKey) {
        let signing_key = ed25519_dalek::SigningKey::generate(&mut rand::thread_rng());
        let verifying_key = signing_key.verifying_key();
        (Self(verifying_key), signing_key)
    }

    /// Hex-encoded short ID (first 8 bytes) for display.
    pub fn short_id(&self) -> String {
        hex::encode(&self.0.to_bytes()[..8])
    }

    /// Full hex-encoded public key.
    pub fn to_hex(&self) -> String {
        hex::encode(self.0.to_bytes())
    }
}

impl fmt::Debug for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "NodeId({}..)", self.short_id())
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.short_id())
    }
}

impl Serialize for NodeId {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(&self.0.to_bytes())
    }
}

impl<'de> Deserialize<'de> for NodeId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bytes: Vec<u8> = Vec::deserialize(deserializer)?;
        let key = VerifyingKey::from_bytes(
            bytes
                .as_slice()
                .try_into()
                .map_err(serde::de::Error::custom)?,
        )
        .map_err(serde::de::Error::custom)?;
        Ok(Self(key))
    }
}

/// Information about a peer node in the mesh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeerInfo {
    pub node_id: NodeId,
    pub hardware: HardwareProfile,
    pub available_memory_bytes: u64,
    pub available_vram_bytes: u64,
    pub network_bandwidth_mbps: f64,
    pub latency_ms: Option<f64>,
    pub tier: MeshTier,
    pub reputation_score: f64,
    pub addr: SocketAddr,
    /// External (NAT-mapped) address, if known via STUN.
    pub external_addr: Option<SocketAddr>,
    /// Detected NAT type.
    pub nat_type: crate::nat::NatType,

    // ── Swarm-related fields ─────────────────────────────────────
    /// Measured RTT (ms) to known peers, used for swarm formation clustering.
    #[serde(default)]
    pub latency_map: HashMap<NodeId, f64>,
    /// Model this peer is currently serving (if in a swarm).
    #[serde(default)]
    pub serving_model_id: Option<String>,
    /// Swarm this peer currently belongs to.
    #[serde(default)]
    pub swarm_id: Option<Uuid>,
    /// If set, this peer can serve as a speculative decoding drafter.
    #[serde(default)]
    pub draft_capability: Option<DraftCapability>,
}

/// Lifecycle state of a peer connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PeerState {
    /// Discovered via DHT or coordination server, not yet connected.
    Discovered,
    /// QUIC connection established, ready for work.
    Connected,
    /// Actively processing layers for a session.
    Active {
        session_id: uuid::Uuid,
        assigned_layers: Range<u32>,
    },
    /// Connection lost.
    Disconnected,
    /// Reputation too low, blocked from mesh.
    Banned,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node_id_generation() {
        let (id1, _key1) = NodeId::generate();
        let (id2, _key2) = NodeId::generate();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_node_id_display() {
        let (id, _) = NodeId::generate();
        let short = id.short_id();
        assert_eq!(short.len(), 16); // 8 bytes = 16 hex chars
        let full = id.to_hex();
        assert_eq!(full.len(), 64); // 32 bytes = 64 hex chars
    }

    #[test]
    fn test_node_id_serde_roundtrip() {
        let (id, _) = NodeId::generate();
        let serialized = bincode::serialize(&id).unwrap();
        let deserialized: NodeId = bincode::deserialize(&serialized).unwrap();
        assert_eq!(id, deserialized);
    }
}
