use tracing::{info, warn};

use crate::error::{MeshError, Result};
use crate::peer::NodeId;
use crate::swarm::{SwarmDefinition, SwarmRole, SwarmStatus, MIN_SWARM_SIZE};

/// Reason for a rebalance operation.
#[derive(Debug, Clone)]
pub enum RebalanceReason {
    /// A member left gracefully.
    MemberLeft { peer_id: NodeId },
    /// A member was detected as failed (health check failure).
    MemberFailed { peer_id: NodeId },
    /// A new member joined the swarm.
    MemberJoined { peer_id: NodeId },
}

impl std::fmt::Display for RebalanceReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RebalanceReason::MemberLeft { peer_id } => {
                write!(f, "member {peer_id} left")
            }
            RebalanceReason::MemberFailed { peer_id } => {
                write!(f, "member {peer_id} failed")
            }
            RebalanceReason::MemberJoined { peer_id } => {
                write!(f, "member {peer_id} joined")
            }
        }
    }
}

/// Result of a rebalance operation.
#[derive(Debug)]
pub enum RebalanceAction {
    /// Layers were redistributed among remaining members.
    Redistributed {
        new_assignments: Vec<(NodeId, std::ops::Range<u32>)>,
    },
    /// A standby member was promoted to replace the departing one.
    StandbyPromoted {
        promoted_peer: NodeId,
        new_assignments: Vec<(NodeId, std::ops::Range<u32>)>,
    },
    /// Swarm is disbanded because too few members remain.
    Disbanded,
}

/// Attempt to rebalance a swarm after a membership change.
///
/// Returns the action taken, or an error if rebalancing is not possible.
pub fn rebalance(
    swarm: &mut SwarmDefinition,
    reason: &RebalanceReason,
    total_layers: u32,
) -> Result<RebalanceAction> {
    match reason {
        RebalanceReason::MemberLeft { peer_id } | RebalanceReason::MemberFailed { peer_id } => {
            handle_member_departure(swarm, peer_id, total_layers)
        }
        RebalanceReason::MemberJoined { peer_id: _ } => {
            // Redistribute layers to include the new member
            redistribute_layers(swarm, total_layers)
        }
    }
}

fn handle_member_departure(
    swarm: &mut SwarmDefinition,
    departing_peer: &NodeId,
    total_layers: u32,
) -> Result<RebalanceAction> {
    // Check for a standby member to promote
    let standby_idx = swarm
        .members
        .iter()
        .position(|m| matches!(m.role, SwarmRole::Standby));

    // Remove the departing member
    let departed = swarm
        .members
        .iter()
        .position(|m| m.peer_id == *departing_peer);

    if let Some(idx) = departed {
        let removed = swarm.members.remove(idx);
        info!(
            "Removed member {} from swarm {} (layers {}..{})",
            departing_peer,
            swarm.swarm_id,
            removed.assigned_layers.start,
            removed.assigned_layers.end
        );
    }

    // Check if the departing peer was the leader
    let needs_new_leader = swarm.leader().is_none();

    // If we have a standby, promote them
    if let Some(standby_idx) = standby_idx {
        // Adjust index if it shifted after removal
        let adjusted_idx = if departed.is_some() && departed.unwrap() < standby_idx {
            standby_idx - 1
        } else {
            standby_idx
        };

        if adjusted_idx < swarm.members.len() {
            let promoted_peer = swarm.members[adjusted_idx].peer_id.clone();
            swarm.members[adjusted_idx].role = if needs_new_leader {
                SwarmRole::Leader
            } else {
                SwarmRole::Worker
            };

            info!(
                "Promoted standby {} in swarm {}",
                promoted_peer, swarm.swarm_id
            );

            // Redistribute layers across all active members
            let action = redistribute_layers(swarm, total_layers)?;
            match action {
                RebalanceAction::Redistributed { new_assignments } => {
                    return Ok(RebalanceAction::StandbyPromoted {
                        promoted_peer,
                        new_assignments,
                    });
                }
                other => return Ok(other),
            }
        }
    }

    // If the leader left and no standby, promote the first worker
    if needs_new_leader && !swarm.members.is_empty() {
        swarm.members[0].role = SwarmRole::Leader;
    }

    // Check if we still have enough active members
    let active_count = swarm.active_member_count();
    if active_count < MIN_SWARM_SIZE {
        warn!(
            "Swarm {} has only {} active members, disbanding",
            swarm.swarm_id, active_count
        );
        swarm.status = SwarmStatus::Disbanded;
        return Ok(RebalanceAction::Disbanded);
    }

    // Redistribute layers among remaining members
    swarm.status = SwarmStatus::Degraded;
    redistribute_layers(swarm, total_layers)
}

/// Redistribute layers proportionally across all active swarm members.
fn redistribute_layers(swarm: &mut SwarmDefinition, total_layers: u32) -> Result<RebalanceAction> {
    let active_members: Vec<usize> = swarm
        .members
        .iter()
        .enumerate()
        .filter(|(_, m)| !matches!(m.role, SwarmRole::Standby))
        .map(|(i, _)| i)
        .collect();

    if active_members.is_empty() {
        swarm.status = SwarmStatus::Disbanded;
        return Ok(RebalanceAction::Disbanded);
    }

    let total_capacity: u64 = active_members
        .iter()
        .map(|&i| swarm.members[i].capacity_bytes)
        .sum();

    if total_capacity == 0 {
        return Err(MeshError::Scheduling("No capacity available".into()));
    }

    let mut layer_cursor = 0u32;
    let mut new_assignments = Vec::new();

    for (idx, &member_idx) in active_members.iter().enumerate() {
        if layer_cursor >= total_layers {
            break;
        }

        let share = if idx == active_members.len() - 1 {
            total_layers - layer_cursor
        } else {
            let fraction = swarm.members[member_idx].capacity_bytes as f64 / total_capacity as f64;
            let raw = (fraction * total_layers as f64).round() as u32;
            raw.max(1).min(total_layers - layer_cursor)
        };

        let range = layer_cursor..layer_cursor + share;
        swarm.members[member_idx].assigned_layers = range.clone();
        new_assignments.push((swarm.members[member_idx].peer_id.clone(), range));
        layer_cursor += share;
    }

    swarm.recalculate_capacity();

    Ok(RebalanceAction::Redistributed { new_assignments })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swarm::{SwarmDefinition, SwarmMember};

    fn make_test_swarm() -> SwarmDefinition {
        let gb = 1024u64 * 1024 * 1024;
        let (id1, _) = NodeId::generate();
        let (id2, _) = NodeId::generate();
        let (id3, _) = NodeId::generate();

        let mut swarm = SwarmDefinition::new("test-model".into());
        swarm.members = vec![
            SwarmMember {
                peer_id: id1,
                role: SwarmRole::Leader,
                assigned_layers: 0..16,
                capacity_bytes: 8 * gb,
                latency_to_leader_ms: 0.0,
            },
            SwarmMember {
                peer_id: id2,
                role: SwarmRole::Worker,
                assigned_layers: 16..32,
                capacity_bytes: 8 * gb,
                latency_to_leader_ms: 5.0,
            },
            SwarmMember {
                peer_id: id3,
                role: SwarmRole::Standby,
                assigned_layers: 0..0,
                capacity_bytes: 8 * gb,
                latency_to_leader_ms: 7.0,
            },
        ];
        swarm.status = SwarmStatus::Ready;
        swarm.recalculate_capacity();
        swarm
    }

    #[test]
    fn test_member_departure_promotes_standby() {
        let mut swarm = make_test_swarm();
        let departing = swarm.members[1].peer_id.clone(); // Worker leaves

        let action = rebalance(
            &mut swarm,
            &RebalanceReason::MemberLeft { peer_id: departing },
            32,
        )
        .unwrap();

        match action {
            RebalanceAction::StandbyPromoted {
                promoted_peer: _,
                new_assignments,
            } => {
                assert!(!new_assignments.is_empty());
            }
            other => panic!("Expected StandbyPromoted, got {:?}", other),
        }
    }

    #[test]
    fn test_disband_when_too_few_members() {
        let mut swarm = make_test_swarm();
        // Remove standby first
        swarm
            .members
            .retain(|m| !matches!(m.role, SwarmRole::Standby));

        let departing = swarm.members[1].peer_id.clone();
        let action = rebalance(
            &mut swarm,
            &RebalanceReason::MemberFailed { peer_id: departing },
            32,
        )
        .unwrap();

        match action {
            RebalanceAction::Disbanded => {
                assert_eq!(swarm.status, SwarmStatus::Disbanded);
            }
            other => panic!("Expected Disbanded, got {:?}", other),
        }
    }

    #[test]
    fn test_redistribute_on_join() {
        let mut swarm = make_test_swarm();
        swarm
            .members
            .retain(|m| !matches!(m.role, SwarmRole::Standby));

        let (new_id, _) = NodeId::generate();
        let gb = 1024u64 * 1024 * 1024;
        swarm.members.push(SwarmMember {
            peer_id: new_id.clone(),
            role: SwarmRole::Worker,
            assigned_layers: 0..0,
            capacity_bytes: 4 * gb,
            latency_to_leader_ms: 8.0,
        });

        let action = rebalance(
            &mut swarm,
            &RebalanceReason::MemberJoined { peer_id: new_id },
            32,
        )
        .unwrap();

        match action {
            RebalanceAction::Redistributed { new_assignments } => {
                assert_eq!(new_assignments.len(), 3);
                // All layers should be covered
                let total: u32 = new_assignments.iter().map(|(_, r)| r.end - r.start).sum();
                assert_eq!(total, 32);
            }
            other => panic!("Expected Redistributed, got {:?}", other),
        }
    }
}
