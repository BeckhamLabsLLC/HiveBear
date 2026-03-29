pub mod formation;
pub mod rebalance;
pub mod router;
pub mod speculative;

use std::ops::Range;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::peer::NodeId;
use crate::scheduler::plan::InferencePlan;

/// Unique identifier for a swarm.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SwarmId(pub Uuid);

impl SwarmId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for SwarmId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for SwarmId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", &self.0.to_string()[..8])
    }
}

/// A swarm is a group of 2-4 low-latency peers that collectively serve one
/// model instance via a short pipeline. Multiple swarms can serve the same
/// model for data-parallel throughput scaling.
#[derive(Debug, Clone)]
pub struct SwarmDefinition {
    pub swarm_id: SwarmId,
    pub model_id: String,
    pub members: Vec<SwarmMember>,
    pub pipeline_plan: Option<InferencePlan>,
    pub aggregate_capacity_bytes: u64,
    pub estimated_tok_s: f64,
    pub status: SwarmStatus,
    pub created_at: Instant,
}

impl SwarmDefinition {
    pub fn new(model_id: String) -> Self {
        Self {
            swarm_id: SwarmId::new(),
            model_id,
            members: Vec::new(),
            pipeline_plan: None,
            aggregate_capacity_bytes: 0,
            estimated_tok_s: 0.0,
            status: SwarmStatus::Forming,
            created_at: Instant::now(),
        }
    }

    /// Number of active (non-standby) members.
    pub fn active_member_count(&self) -> usize {
        self.members
            .iter()
            .filter(|m| !matches!(m.role, SwarmRole::Standby))
            .count()
    }

    /// Total member count including standby.
    pub fn member_count(&self) -> usize {
        self.members.len()
    }

    /// Get the swarm leader, if any.
    pub fn leader(&self) -> Option<&SwarmMember> {
        self.members
            .iter()
            .find(|m| matches!(m.role, SwarmRole::Leader))
    }

    /// Check if a peer is a member of this swarm.
    pub fn has_member(&self, peer_id: &NodeId) -> bool {
        self.members.iter().any(|m| m.peer_id == *peer_id)
    }

    /// Recalculate aggregate capacity from members.
    pub fn recalculate_capacity(&mut self) {
        self.aggregate_capacity_bytes = self
            .members
            .iter()
            .filter(|m| !matches!(m.role, SwarmRole::Standby))
            .map(|m| m.capacity_bytes)
            .sum();
    }

    /// Check if the swarm can accept another member.
    pub fn can_accept_member(&self) -> bool {
        self.member_count() < MAX_SWARM_SIZE
            && matches!(
                self.status,
                SwarmStatus::Forming | SwarmStatus::Ready | SwarmStatus::Degraded
            )
    }
}

/// A peer participating in a swarm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmMember {
    pub peer_id: NodeId,
    pub role: SwarmRole,
    pub assigned_layers: Range<u32>,
    /// Total usable memory (VRAM + RAM) this peer contributes.
    pub capacity_bytes: u64,
    /// Measured RTT to the swarm leader in milliseconds.
    pub latency_to_leader_ms: f64,
}

/// Role of a peer within a swarm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SwarmRole {
    /// Manages the pipeline, receives and routes inference requests.
    Leader,
    /// Holds assigned layers and processes activations.
    Worker,
    /// Ready to replace a failed member (does not hold layers until promoted).
    Standby,
}

impl std::fmt::Display for SwarmRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SwarmRole::Leader => write!(f, "leader"),
            SwarmRole::Worker => write!(f, "worker"),
            SwarmRole::Standby => write!(f, "standby"),
        }
    }
}

/// Lifecycle status of a swarm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SwarmStatus {
    /// Gathering members, not yet serving.
    Forming,
    /// Model loaded across members, accepting inference requests.
    Ready,
    /// All pipeline slots are occupied processing requests.
    Busy,
    /// Lost a member but still operational with reduced capacity.
    Degraded,
    /// Shut down, no longer active.
    Disbanded,
}

impl std::fmt::Display for SwarmStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SwarmStatus::Forming => write!(f, "forming"),
            SwarmStatus::Ready => write!(f, "ready"),
            SwarmStatus::Busy => write!(f, "busy"),
            SwarmStatus::Degraded => write!(f, "degraded"),
            SwarmStatus::Disbanded => write!(f, "disbanded"),
        }
    }
}

/// Describes a peer's ability to run a small draft model for speculative decoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DraftCapability {
    /// Model ID of the draft model (e.g. "tinyllama-1.1b-q4_k_m").
    pub draft_model_id: String,
    /// Size of the draft model in bytes.
    pub draft_model_size_bytes: u64,
    /// Estimated tokens/sec the draft model can produce on this peer.
    pub estimated_draft_tok_s: f64,
}

// ── Constants ────────────────────────────────────────────────────────

/// Maximum peers per swarm (keeps pipeline depth manageable).
pub const MAX_SWARM_SIZE: usize = 4;

/// Minimum peers for a swarm to be functional.
pub const MIN_SWARM_SIZE: usize = 2;

/// Maximum latency (ms) between swarm members on a LAN.
pub const MAX_LAN_LATENCY_MS: f64 = 15.0;

/// Maximum latency (ms) between swarm members on a WAN.
pub const MAX_WAN_LATENCY_MS: f64 = 50.0;

/// Minimum combined capacity (bytes) for a swarm to serve any model.
/// ~3 GB — enough for a 3B model at Q4.
pub const MIN_SWARM_CAPACITY_BYTES: u64 = 3 * 1024 * 1024 * 1024;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swarm_id_display() {
        let id = SwarmId::new();
        let s = format!("{id}");
        assert_eq!(s.len(), 8);
    }

    #[test]
    fn test_swarm_definition_basics() {
        let mut swarm = SwarmDefinition::new("llama-3.1-8b".into());
        assert_eq!(swarm.member_count(), 0);
        assert_eq!(swarm.active_member_count(), 0);
        assert!(swarm.leader().is_none());
        assert!(swarm.can_accept_member());

        let (peer_id, _) = NodeId::generate();
        swarm.members.push(SwarmMember {
            peer_id: peer_id.clone(),
            role: SwarmRole::Leader,
            assigned_layers: 0..16,
            capacity_bytes: 8 * 1024 * 1024 * 1024,
            latency_to_leader_ms: 0.0,
        });

        assert_eq!(swarm.member_count(), 1);
        assert_eq!(swarm.active_member_count(), 1);
        assert!(swarm.leader().is_some());
        assert!(swarm.has_member(&peer_id));
    }

    #[test]
    fn test_swarm_capacity_recalculation() {
        let mut swarm = SwarmDefinition::new("test-model".into());
        let gb = 1024 * 1024 * 1024;

        let (id1, _) = NodeId::generate();
        let (id2, _) = NodeId::generate();
        let (id3, _) = NodeId::generate();

        swarm.members.push(SwarmMember {
            peer_id: id1,
            role: SwarmRole::Leader,
            assigned_layers: 0..10,
            capacity_bytes: 8 * gb,
            latency_to_leader_ms: 0.0,
        });
        swarm.members.push(SwarmMember {
            peer_id: id2,
            role: SwarmRole::Worker,
            assigned_layers: 10..20,
            capacity_bytes: 4 * gb,
            latency_to_leader_ms: 5.0,
        });
        swarm.members.push(SwarmMember {
            peer_id: id3,
            role: SwarmRole::Standby,
            assigned_layers: 0..0,
            capacity_bytes: 4 * gb,
            latency_to_leader_ms: 8.0,
        });

        swarm.recalculate_capacity();
        // Standby doesn't count toward aggregate
        assert_eq!(swarm.aggregate_capacity_bytes, 12 * gb);
    }

    #[test]
    fn test_max_swarm_size() {
        let mut swarm = SwarmDefinition::new("test".into());
        for _ in 0..MAX_SWARM_SIZE {
            let (id, _) = NodeId::generate();
            swarm.members.push(SwarmMember {
                peer_id: id,
                role: SwarmRole::Worker,
                assigned_layers: 0..1,
                capacity_bytes: 1024,
                latency_to_leader_ms: 1.0,
            });
        }
        assert!(!swarm.can_accept_member());
    }
}
