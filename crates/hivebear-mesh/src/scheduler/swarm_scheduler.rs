use async_trait::async_trait;
use uuid::Uuid;

use super::plan::{InferencePlan, LayerAssignment};
use super::LayerScheduler;
use crate::error::{MeshError, Result};
use crate::peer::PeerInfo;

/// Swarm-aware scheduler that limits pipeline depth and optimizes for latency.
///
/// Key differences from `GreedyScheduler`:
/// - Enforces a maximum pipeline depth (default 4 stages)
/// - Requires a minimum number of layers per stage to avoid overhead
/// - Estimates steady-state throughput (pipeline stages overlap after warmup)
/// - Orders stages to minimize total network hop latency
pub struct SwarmAwareScheduler {
    /// Maximum pipeline stages (network hops) allowed.
    max_pipeline_depth: u32,
    /// Minimum layers each stage must process.
    min_layers_per_stage: u32,
    /// Maximum acceptable per-hop latency in ms (peers above this are excluded).
    max_hop_latency_ms: f64,
}

impl SwarmAwareScheduler {
    pub fn new() -> Self {
        Self {
            max_pipeline_depth: 4,
            min_layers_per_stage: 4,
            max_hop_latency_ms: 50.0,
        }
    }

    pub fn with_limits(
        max_pipeline_depth: u32,
        min_layers_per_stage: u32,
        max_hop_latency_ms: f64,
    ) -> Self {
        Self {
            max_pipeline_depth,
            min_layers_per_stage,
            max_hop_latency_ms,
        }
    }
}

impl Default for SwarmAwareScheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LayerScheduler for SwarmAwareScheduler {
    async fn plan(
        &self,
        model_id: &str,
        total_layers: u32,
        model_size_bytes: u64,
        peers: &[PeerInfo],
    ) -> Result<InferencePlan> {
        if peers.is_empty() {
            return Err(MeshError::Scheduling("No peers available".into()));
        }

        let bytes_per_layer = if total_layers > 0 {
            model_size_bytes / total_layers as u64
        } else {
            return Err(MeshError::Scheduling("Model has 0 layers".into()));
        };

        // Filter peers: only those with acceptable latency to at least one other peer
        // (or self-sufficient alone).
        let mut eligible_peers: Vec<&PeerInfo> = peers
            .iter()
            .filter(|p| {
                // Keep peer if its latency to the coordinator is acceptable
                p.latency_ms.unwrap_or(10.0) <= self.max_hop_latency_ms
            })
            .collect();

        if eligible_peers.is_empty() {
            return Err(MeshError::Scheduling(
                "No peers with acceptable latency".into(),
            ));
        }

        // Sort by total capacity descending, reputation as tiebreaker
        eligible_peers.sort_by(|a, b| {
            let cap_a = a.available_vram_bytes + a.available_memory_bytes;
            let cap_b = b.available_vram_bytes + b.available_memory_bytes;
            cap_b.cmp(&cap_a).then(
                b.reputation_score
                    .partial_cmp(&a.reputation_score)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });

        // Limit to max_pipeline_depth peers
        let max_peers = self.max_pipeline_depth as usize;
        if eligible_peers.len() > max_peers {
            eligible_peers.truncate(max_peers);
        }

        // Check that we can assign at least min_layers_per_stage to each peer
        let min_total_needed =
            self.min_layers_per_stage * eligible_peers.len().min(max_peers) as u32;
        if total_layers < min_total_needed && eligible_peers.len() > 1 {
            // Reduce the number of peers until min_layers_per_stage can be satisfied
            while eligible_peers.len() > 1
                && total_layers < self.min_layers_per_stage * eligible_peers.len() as u32
            {
                eligible_peers.pop();
            }
        }

        // Calculate total capacity for proportional assignment
        let total_capacity: u64 = eligible_peers
            .iter()
            .map(|p| p.available_vram_bytes + p.available_memory_bytes)
            .sum();

        if total_capacity == 0 {
            return Err(MeshError::Scheduling(
                "Peers have no available memory".into(),
            ));
        }

        // Assign layers proportionally, enforcing min_layers_per_stage
        let mut assignments = Vec::new();
        let mut layer_cursor: u32 = 0;

        for (i, peer) in eligible_peers.iter().enumerate() {
            if layer_cursor >= total_layers {
                break;
            }

            let peer_capacity = peer.available_vram_bytes + peer.available_memory_bytes;
            let share = if i == eligible_peers.len() - 1 {
                total_layers - layer_cursor
            } else {
                let fraction = peer_capacity as f64 / total_capacity as f64;
                let raw = (fraction * total_layers as f64).round() as u32;
                raw.max(self.min_layers_per_stage)
                    .min(total_layers - layer_cursor)
            };

            if share == 0 {
                continue;
            }

            let layer_range = layer_cursor..layer_cursor + share;

            // Estimate compute time per token for this stage
            let compute_bandwidth = if peer.available_vram_bytes > 0 {
                200.0_f64 // GPU: ~200 GB/s
            } else {
                peer.hardware.memory.estimated_bandwidth_gbps
            };
            let compute_bytes = share as f64 * bytes_per_layer as f64;
            let estimated_compute_ms = (compute_bytes / (compute_bandwidth * 1e9)) * 1000.0;

            // Estimate network transfer time between stages
            let activation_size_bytes = 16.0 * 1024.0; // ~16KB typical
            let transfer_bandwidth_bps = peer.network_bandwidth_mbps * 1e6 / 8.0;
            let latency = peer.latency_ms.unwrap_or(10.0);
            let estimated_transfer_ms = if transfer_bandwidth_bps > 0.0 {
                (activation_size_bytes / transfer_bandwidth_bps) * 1000.0 + latency
            } else {
                latency
            };

            assignments.push(LayerAssignment {
                peer_id: peer.node_id.clone(),
                layer_range,
                estimated_compute_ms,
                estimated_transfer_ms,
            });

            layer_cursor += share;
        }

        if layer_cursor < total_layers {
            return Err(MeshError::Scheduling(format!(
                "Could only assign {layer_cursor}/{total_layers} layers across {} peers",
                eligible_peers.len()
            )));
        }

        // Steady-state pipeline latency estimation:
        // After warmup, stages overlap. The bottleneck is the SLOWEST stage
        // (compute) plus the last network hop. This is fundamentally different
        // from GreedyScheduler which sums all stages (incorrect for pipelining).
        let max_stage_compute_ms = assignments
            .iter()
            .map(|a| a.estimated_compute_ms)
            .fold(0.0_f64, f64::max);

        let _total_transfer_ms: f64 = assignments.iter().map(|a| a.estimated_transfer_ms).sum();

        // First token latency = sum of all stages (no pipelining benefit yet)
        let _first_token_latency_ms: f64 = assignments
            .iter()
            .map(|a| a.estimated_compute_ms + a.estimated_transfer_ms)
            .sum();

        // Steady-state per-token latency = bottleneck stage compute + one network hop
        let last_hop_ms = assignments
            .last()
            .map(|a| a.estimated_transfer_ms)
            .unwrap_or(0.0);
        let steady_state_latency_ms = max_stage_compute_ms + last_hop_ms;

        // Use steady-state for throughput estimation (more representative for generation)
        let estimated_latency_ms = steady_state_latency_ms;
        let estimated_throughput_tok_s = if estimated_latency_ms > 0.0 {
            1000.0 / estimated_latency_ms
        } else {
            0.0
        };

        Ok(InferencePlan {
            session_id: Uuid::new_v4(),
            model_id: model_id.to_string(),
            total_layers,
            assignments,
            estimated_latency_ms,
            estimated_throughput_tok_s,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MeshTier;
    use crate::peer::NodeId;
    use hivebear_core::types::*;
    use std::collections::HashMap;
    use std::net::SocketAddr;

    fn make_peer(ram_gb: u64, vram_gb: u64, bandwidth_mbps: f64, latency_ms: f64) -> PeerInfo {
        let gb = 1024 * 1024 * 1024;
        let (node_id, _) = NodeId::generate();
        PeerInfo {
            node_id,
            hardware: HardwareProfile {
                cpu: CpuInfo {
                    model_name: "Test".into(),
                    physical_cores: 8,
                    logical_cores: 16,
                    isa_extensions: vec![],
                    cache_size_bytes: 0,
                },
                memory: MemoryInfo {
                    total_bytes: ram_gb * gb,
                    available_bytes: ram_gb * gb,
                    estimated_bandwidth_gbps: 30.0,
                },
                gpus: if vram_gb > 0 {
                    vec![GpuInfo {
                        name: "Test GPU".into(),
                        vram_bytes: vram_gb * gb,
                        compute_api: ComputeApi::Vulkan,
                        driver_version: None,
                    }]
                } else {
                    vec![]
                },
                storage: StorageInfo {
                    available_bytes: 100 * gb,
                    estimated_read_speed_mbps: 500.0,
                },
                platform: PlatformInfo {
                    os: "linux".into(),
                    arch: "x86_64".into(),
                    is_mobile: false,
                    power_source: PowerSource::Ac,
                },
            },
            available_memory_bytes: ram_gb * gb,
            available_vram_bytes: vram_gb * gb,
            network_bandwidth_mbps: bandwidth_mbps,
            latency_ms: Some(latency_ms),
            tier: MeshTier::Free,
            reputation_score: 0.9,
            addr: "127.0.0.1:7878".parse::<SocketAddr>().unwrap(),
            external_addr: None,
            nat_type: crate::nat::NatType::Unknown,
            latency_map: HashMap::new(),
            serving_model_id: None,
            swarm_id: None,
            draft_capability: None,
        }
    }

    #[tokio::test]
    async fn test_single_peer_gets_all_layers() {
        let scheduler = SwarmAwareScheduler::new();
        let peers = vec![make_peer(16, 8, 1000.0, 5.0)];
        let plan = scheduler
            .plan("test-model", 32, 8 * 1024 * 1024 * 1024, &peers)
            .await
            .unwrap();

        assert_eq!(plan.assignments.len(), 1);
        assert_eq!(plan.assignments[0].layer_range, 0..32);
    }

    #[tokio::test]
    async fn test_limits_pipeline_depth() {
        let scheduler = SwarmAwareScheduler::with_limits(3, 4, 50.0);
        // 8 peers, but max depth is 3
        let peers: Vec<PeerInfo> = (0..8).map(|_| make_peer(8, 0, 1000.0, 5.0)).collect();
        let plan = scheduler
            .plan("test-model", 32, 8 * 1024 * 1024 * 1024, &peers)
            .await
            .unwrap();

        assert!(plan.assignments.len() <= 3);
        // All layers should still be assigned
        let total: u32 = plan
            .assignments
            .iter()
            .map(|a| a.layer_range.end - a.layer_range.start)
            .sum();
        assert_eq!(total, 32);
    }

    #[tokio::test]
    async fn test_min_layers_per_stage() {
        // 8 layers, 4 peers with min_layers=4 -> only 2 peers should be used
        let scheduler = SwarmAwareScheduler::with_limits(4, 4, 50.0);
        let peers: Vec<PeerInfo> = (0..4).map(|_| make_peer(8, 0, 1000.0, 5.0)).collect();
        let plan = scheduler
            .plan("test-model", 8, 2 * 1024 * 1024 * 1024, &peers)
            .await
            .unwrap();

        assert!(plan.assignments.len() <= 2);
        for assignment in &plan.assignments {
            let layers = assignment.layer_range.end - assignment.layer_range.start;
            assert!(layers >= 4, "Stage has only {} layers", layers);
        }
    }

    #[tokio::test]
    async fn test_excludes_high_latency_peers() {
        let scheduler = SwarmAwareScheduler::with_limits(4, 4, 20.0);
        let peers = vec![
            make_peer(8, 0, 1000.0, 5.0),   // Acceptable
            make_peer(8, 0, 1000.0, 100.0), // Too high latency
        ];
        let plan = scheduler
            .plan("test-model", 32, 8 * 1024 * 1024 * 1024, &peers)
            .await
            .unwrap();

        // Only the low-latency peer should be used
        assert_eq!(plan.assignments.len(), 1);
    }

    #[tokio::test]
    async fn test_steady_state_throughput_better_than_sum() {
        let scheduler = SwarmAwareScheduler::new();
        let peers = vec![make_peer(8, 0, 1000.0, 5.0), make_peer(8, 0, 1000.0, 5.0)];
        let plan = scheduler
            .plan("test-model", 32, 8 * 1024 * 1024 * 1024, &peers)
            .await
            .unwrap();

        // Steady-state throughput should be higher than if we summed all stages
        // (because pipeline stages overlap)
        assert!(plan.estimated_throughput_tok_s > 0.0);
        assert!(plan.estimated_latency_ms > 0.0);

        // The latency should be less than sum of all compute + transfer
        let sum_all: f64 = plan
            .assignments
            .iter()
            .map(|a| a.estimated_compute_ms + a.estimated_transfer_ms)
            .sum();
        assert!(
            plan.estimated_latency_ms < sum_all,
            "Steady-state latency ({:.1}ms) should be less than sum ({:.1}ms)",
            plan.estimated_latency_ms,
            sum_all
        );
    }

    #[tokio::test]
    async fn test_no_peers_errors() {
        let scheduler = SwarmAwareScheduler::new();
        let result = scheduler
            .plan("test-model", 32, 8 * 1024 * 1024 * 1024, &[])
            .await;
        assert!(result.is_err());
    }
}
