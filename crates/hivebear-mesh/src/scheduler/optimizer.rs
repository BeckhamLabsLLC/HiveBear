use async_trait::async_trait;
use uuid::Uuid;

use super::plan::{InferencePlan, LayerAssignment};
use super::LayerScheduler;
use crate::error::{MeshError, Result};
use crate::peer::PeerInfo;

/// Greedy layer scheduler that assigns layers proportional to each peer's
/// available memory, then estimates pipeline latency.
pub struct GreedyScheduler;

impl GreedyScheduler {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GreedyScheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LayerScheduler for GreedyScheduler {
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

        // Sort peers by total capacity (VRAM + RAM) descending, with
        // reputation as tiebreaker.
        let mut ranked_peers: Vec<&PeerInfo> = peers.iter().collect();
        ranked_peers.sort_by(|a, b| {
            let cap_a = a.available_vram_bytes + a.available_memory_bytes;
            let cap_b = b.available_vram_bytes + b.available_memory_bytes;
            cap_b.cmp(&cap_a).then(
                b.reputation_score
                    .partial_cmp(&a.reputation_score)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });

        // Assign layers proportional to capacity
        let total_capacity: u64 = ranked_peers
            .iter()
            .map(|p| p.available_vram_bytes + p.available_memory_bytes)
            .sum();

        if total_capacity == 0 {
            return Err(MeshError::Scheduling(
                "Peers have no available memory".into(),
            ));
        }

        let mut assignments = Vec::new();
        let mut layer_cursor: u32 = 0;

        for (i, peer) in ranked_peers.iter().enumerate() {
            if layer_cursor >= total_layers {
                break;
            }

            let peer_capacity = peer.available_vram_bytes + peer.available_memory_bytes;
            let share = if i == ranked_peers.len() - 1 {
                // Last peer gets all remaining layers
                total_layers - layer_cursor
            } else {
                let fraction = peer_capacity as f64 / total_capacity as f64;
                let layers = (fraction * total_layers as f64).round() as u32;
                layers.max(1).min(total_layers - layer_cursor)
            };

            if share == 0 {
                continue;
            }

            let layer_range = layer_cursor..layer_cursor + share;

            // Estimate compute time: layers * bytes_per_layer / bandwidth
            let compute_bandwidth = if peer.available_vram_bytes > 0 {
                // GPU: assume ~200 GB/s for modern GPUs
                200.0_f64
            } else {
                // CPU: use reported memory bandwidth
                peer.hardware.memory.estimated_bandwidth_gbps
            };
            let compute_bytes = share as f64 * bytes_per_layer as f64;
            let estimated_compute_ms = (compute_bytes / (compute_bandwidth * 1e9)) * 1000.0;

            // Estimate transfer time: activation tensor (~16KB) over network
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
                peers.len()
            )));
        }

        // Pipeline latency = sum of all stage costs (compute + transfer)
        let estimated_latency_ms: f64 = assignments
            .iter()
            .map(|a| a.estimated_compute_ms + a.estimated_transfer_ms)
            .sum();

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
    use std::net::SocketAddr;

    fn make_peer(ram_gb: u64, vram_gb: u64, bandwidth_mbps: f64) -> PeerInfo {
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
            latency_ms: Some(5.0),
            tier: MeshTier::Free,
            reputation_score: 0.9,
            addr: "127.0.0.1:7878".parse::<SocketAddr>().unwrap(),
            external_addr: None,
            nat_type: crate::nat::NatType::Unknown,
            latency_map: std::collections::HashMap::new(),
            serving_model_id: None,
            swarm_id: None,
            draft_capability: None,
        }
    }

    #[tokio::test]
    async fn test_single_peer_gets_all_layers() {
        let scheduler = GreedyScheduler::new();
        let peers = vec![make_peer(16, 8, 1000.0)];
        let plan = scheduler
            .plan("test-model", 40, 8 * 1024 * 1024 * 1024, &peers)
            .await
            .unwrap();

        assert_eq!(plan.assignments.len(), 1);
        assert_eq!(plan.assignments[0].layer_range, 0..40);
        assert_eq!(plan.total_layers, 40);
    }

    #[tokio::test]
    async fn test_two_peers_split_layers() {
        let scheduler = GreedyScheduler::new();
        let peers = vec![
            make_peer(16, 8, 1000.0), // 24GB total
            make_peer(8, 0, 1000.0),  // 8GB total
        ];
        let plan = scheduler
            .plan("test-model", 32, 8 * 1024 * 1024 * 1024, &peers)
            .await
            .unwrap();

        assert_eq!(plan.assignments.len(), 2);
        // First peer should get more layers (larger capacity)
        let first_layers =
            plan.assignments[0].layer_range.end - plan.assignments[0].layer_range.start;
        let second_layers =
            plan.assignments[1].layer_range.end - plan.assignments[1].layer_range.start;
        assert!(first_layers > second_layers);
        assert_eq!(first_layers + second_layers, 32);
    }

    #[tokio::test]
    async fn test_no_peers_errors() {
        let scheduler = GreedyScheduler::new();
        let result = scheduler
            .plan("test-model", 40, 8 * 1024 * 1024 * 1024, &[])
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_plan_has_positive_estimates() {
        let scheduler = GreedyScheduler::new();
        let peers = vec![make_peer(16, 8, 1000.0)];
        let plan = scheduler
            .plan("test-model", 40, 8 * 1024 * 1024 * 1024, &peers)
            .await
            .unwrap();

        assert!(plan.estimated_latency_ms > 0.0);
        assert!(plan.estimated_throughput_tok_s > 0.0);
    }
}
