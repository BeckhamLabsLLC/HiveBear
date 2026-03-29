use std::collections::HashMap;

use tracing::{debug, info};

use crate::error::Result;
use crate::peer::{NodeId, PeerInfo};
use crate::swarm::{
    DraftCapability, SwarmDefinition, SwarmMember, SwarmRole, SwarmStatus, MAX_SWARM_SIZE,
    MAX_WAN_LATENCY_MS, MIN_SWARM_CAPACITY_BYTES, MIN_SWARM_SIZE,
};

/// Model size class for matching swarm capacity to appropriate models.
#[derive(Debug, Clone)]
pub struct ModelSizeClass {
    pub model_id: String,
    pub display_name: String,
    /// Total model weight size in bytes (for the recommended quantization).
    pub size_bytes: u64,
    /// Number of transformer layers.
    pub total_layers: u32,
    /// Parameter count in billions.
    pub params_billions: f64,
}

/// Result of the swarm formation process.
#[derive(Debug)]
pub struct FormationResult {
    /// Swarms that were successfully formed.
    pub swarms: Vec<SwarmDefinition>,
    /// Peers assigned as speculative decoding drafters (too weak for any swarm).
    pub drafters: Vec<(NodeId, DraftCapability)>,
    /// Peers that could not be placed in any swarm or drafter role.
    pub unplaced: Vec<NodeId>,
}

/// Configuration for swarm formation.
#[derive(Debug, Clone)]
pub struct FormationConfig {
    /// Maximum pairwise latency (ms) between swarm members.
    pub max_latency_ms: f64,
    /// Maximum pipeline stages per swarm.
    pub max_pipeline_depth: usize,
    /// Minimum layers each stage must process.
    pub min_layers_per_stage: u32,
    /// Minimum memory (bytes) for a peer to join a swarm (not as drafter).
    pub min_peer_capacity_bytes: u64,
    /// Minimum memory (bytes) for a peer to serve as a drafter.
    pub min_drafter_capacity_bytes: u64,
}

impl Default for FormationConfig {
    fn default() -> Self {
        let gb = 1024 * 1024 * 1024;
        Self {
            max_latency_ms: MAX_WAN_LATENCY_MS,
            max_pipeline_depth: MAX_SWARM_SIZE,
            min_layers_per_stage: 4,
            min_peer_capacity_bytes: 2 * gb,
            min_drafter_capacity_bytes: 512 * 1024 * 1024, // 512 MB
        }
    }
}

/// Well-known model size classes for automatic model matching.
pub fn builtin_model_classes() -> Vec<ModelSizeClass> {
    let gb: u64 = 1024 * 1024 * 1024;
    vec![
        ModelSizeClass {
            model_id: "tinyllama-1.1b-q4_k_m".into(),
            display_name: "TinyLlama 1.1B".into(),
            size_bytes: (0.7 * gb as f64) as u64,
            total_layers: 22,
            params_billions: 1.1,
        },
        ModelSizeClass {
            model_id: "phi-3-mini-3.8b-q4_k_m".into(),
            display_name: "Phi-3 Mini 3.8B".into(),
            size_bytes: (2.3 * gb as f64) as u64,
            total_layers: 32,
            params_billions: 3.8,
        },
        ModelSizeClass {
            model_id: "llama-3.2-3b-q4_k_m".into(),
            display_name: "Llama 3.2 3B".into(),
            size_bytes: 2 * gb,
            total_layers: 28,
            params_billions: 3.0,
        },
        ModelSizeClass {
            model_id: "mistral-7b-q4_k_m".into(),
            display_name: "Mistral 7B".into(),
            size_bytes: 4 * gb,
            total_layers: 32,
            params_billions: 7.0,
        },
        ModelSizeClass {
            model_id: "llama-3.1-8b-q4_k_m".into(),
            display_name: "Llama 3.1 8B".into(),
            size_bytes: 5 * gb,
            total_layers: 32,
            params_billions: 8.0,
        },
        ModelSizeClass {
            model_id: "llama-3.1-8b-q6_k".into(),
            display_name: "Llama 3.1 8B (Q6)".into(),
            size_bytes: 7 * gb,
            total_layers: 32,
            params_billions: 8.0,
        },
        ModelSizeClass {
            model_id: "codellama-13b-q4_k_m".into(),
            display_name: "CodeLlama 13B".into(),
            size_bytes: 8 * gb,
            total_layers: 40,
            params_billions: 13.0,
        },
        ModelSizeClass {
            model_id: "yi-34b-q4_k_m".into(),
            display_name: "Yi 34B".into(),
            size_bytes: 20 * gb,
            total_layers: 60,
            params_billions: 34.0,
        },
        ModelSizeClass {
            model_id: "llama-3.1-70b-q4_k_m".into(),
            display_name: "Llama 3.1 70B".into(),
            size_bytes: 40 * gb,
            total_layers: 80,
            params_billions: 70.0,
        },
    ]
}

/// Find the largest model that fits within the given combined capacity.
pub fn best_model_for_capacity(
    capacity_bytes: u64,
    models: &[ModelSizeClass],
) -> Option<&ModelSizeClass> {
    // Reserve ~15% overhead for KV cache and runtime allocations.
    let usable = (capacity_bytes as f64 * 0.85) as u64;
    models
        .iter()
        .filter(|m| m.size_bytes <= usable)
        .max_by_key(|m| m.size_bytes)
}

/// Determine if a peer is too weak for swarm membership but can serve as a drafter.
fn classify_as_drafter(peer: &PeerInfo, config: &FormationConfig) -> Option<DraftCapability> {
    let capacity = peer.available_vram_bytes + peer.available_memory_bytes;
    if capacity < config.min_drafter_capacity_bytes {
        return None; // Too weak even for drafting
    }
    if capacity >= config.min_peer_capacity_bytes {
        return None; // Strong enough for a swarm, not a drafter
    }

    let gb = 1024u64 * 1024 * 1024;
    // Pick the largest draft model that fits
    if capacity >= (2 * gb) {
        Some(DraftCapability {
            draft_model_id: "phi-3-mini-3.8b-q4_k_m".into(),
            draft_model_size_bytes: (2.3 * gb as f64) as u64,
            estimated_draft_tok_s: 5.0, // conservative estimate for CPU
        })
    } else {
        Some(DraftCapability {
            draft_model_id: "tinyllama-1.1b-q4_k_m".into(),
            draft_model_size_bytes: (0.7 * gb as f64) as u64,
            estimated_draft_tok_s: 10.0,
        })
    }
}

/// Build a pairwise latency matrix from PeerInfo latency maps.
fn build_latency_matrix(peers: &[&PeerInfo]) -> HashMap<(usize, usize), f64> {
    let mut matrix = HashMap::new();
    for (i, a) in peers.iter().enumerate() {
        for (j, b) in peers.iter().enumerate() {
            if i >= j {
                continue;
            }
            // Check both directions for measured latency
            let latency = a
                .latency_map
                .get(&b.node_id)
                .or_else(|| b.latency_map.get(&a.node_id))
                .copied()
                // Fall back to the average of their reported latencies to the coordinator
                .unwrap_or_else(|| {
                    let a_lat = a.latency_ms.unwrap_or(50.0);
                    let b_lat = b.latency_ms.unwrap_or(50.0);
                    (a_lat + b_lat) / 2.0
                });
            matrix.insert((i, j), latency);
            matrix.insert((j, i), latency);
        }
    }
    matrix
}

/// Form swarms from a set of available peers.
///
/// Uses a greedy graph-based approach:
/// 1. Sort peers by capacity descending
/// 2. For each unassigned peer, find nearby unassigned peers (latency < threshold)
/// 3. Group them into a swarm of 2-4 peers
/// 4. Match the swarm to the best model for its combined capacity
pub fn form_swarms(
    peers: &[PeerInfo],
    config: &FormationConfig,
    models: &[ModelSizeClass],
) -> Result<FormationResult> {
    if peers.is_empty() {
        return Ok(FormationResult {
            swarms: vec![],
            drafters: vec![],
            unplaced: vec![],
        });
    }

    let mut drafters = Vec::new();
    let mut swarm_eligible: Vec<&PeerInfo> = Vec::new();
    let mut too_weak: Vec<NodeId> = Vec::new();

    // Phase 1: Classify peers
    for peer in peers {
        let capacity = peer.available_vram_bytes + peer.available_memory_bytes;
        if capacity >= config.min_peer_capacity_bytes {
            swarm_eligible.push(peer);
        } else if let Some(draft_cap) = classify_as_drafter(peer, config) {
            drafters.push((peer.node_id.clone(), draft_cap));
        } else {
            too_weak.push(peer.node_id.clone());
        }
    }

    // Sort eligible peers by capacity descending (strongest first)
    swarm_eligible.sort_by(|a, b| {
        let cap_a = a.available_vram_bytes + a.available_memory_bytes;
        let cap_b = b.available_vram_bytes + b.available_memory_bytes;
        cap_b.cmp(&cap_a)
    });

    let latency_matrix = build_latency_matrix(&swarm_eligible);
    let mut assigned = vec![false; swarm_eligible.len()];
    let mut swarms = Vec::new();

    // Phase 2: Greedy swarm formation
    for i in 0..swarm_eligible.len() {
        if assigned[i] {
            continue;
        }

        // Find nearby unassigned peers
        let mut candidates: Vec<usize> = Vec::new();
        candidates.push(i);

        #[allow(clippy::needless_range_loop)]
        for j in 0..swarm_eligible.len() {
            if i == j || assigned[j] {
                continue;
            }

            // Check that this candidate has acceptable latency to ALL current members
            let all_close = candidates.iter().all(|&member_idx| {
                let key = if member_idx < j {
                    (member_idx, j)
                } else {
                    (j, member_idx)
                };
                latency_matrix
                    .get(&key)
                    .map(|&lat| lat <= config.max_latency_ms)
                    .unwrap_or(false)
            });

            if all_close && candidates.len() < config.max_pipeline_depth {
                candidates.push(j);
            }
        }

        // Need at least MIN_SWARM_SIZE peers
        if candidates.len() < MIN_SWARM_SIZE {
            continue;
        }

        // Calculate combined capacity
        let combined_capacity: u64 = candidates
            .iter()
            .map(|&idx| {
                swarm_eligible[idx].available_vram_bytes
                    + swarm_eligible[idx].available_memory_bytes
            })
            .sum();

        if combined_capacity < MIN_SWARM_CAPACITY_BYTES {
            continue;
        }

        // Find the best model for this capacity
        let model = match best_model_for_capacity(combined_capacity, models) {
            Some(m) => m,
            None => continue, // No model fits
        };

        // Check that the model can be split into enough stages with min layers each
        let max_stages = candidates.len() as u32;
        if model.total_layers < config.min_layers_per_stage * max_stages.min(MIN_SWARM_SIZE as u32)
        {
            continue;
        }

        // Form the swarm
        let mut swarm = SwarmDefinition::new(model.model_id.clone());

        // Sort candidates by capacity descending for layer assignment
        let mut sorted_candidates = candidates.clone();
        sorted_candidates.sort_by(|&a, &b| {
            let cap_a =
                swarm_eligible[a].available_vram_bytes + swarm_eligible[a].available_memory_bytes;
            let cap_b =
                swarm_eligible[b].available_vram_bytes + swarm_eligible[b].available_memory_bytes;
            cap_b.cmp(&cap_a)
        });

        // Assign layers proportional to capacity
        let total_layers = model.total_layers;
        let total_cap = combined_capacity as f64;
        let mut layer_cursor = 0u32;

        for (member_idx, &peer_idx) in sorted_candidates.iter().enumerate() {
            let peer = swarm_eligible[peer_idx];
            let peer_cap = (peer.available_vram_bytes + peer.available_memory_bytes) as f64;

            let share = if member_idx == sorted_candidates.len() - 1 {
                total_layers - layer_cursor
            } else {
                let fraction = peer_cap / total_cap;
                let raw = (fraction * total_layers as f64).round() as u32;
                raw.max(config.min_layers_per_stage)
                    .min(total_layers - layer_cursor)
            };

            if share == 0 {
                continue;
            }

            let role = if member_idx == 0 {
                SwarmRole::Leader
            } else {
                SwarmRole::Worker
            };

            let latency_to_leader = if member_idx == 0 {
                0.0
            } else {
                let leader_idx = sorted_candidates[0];
                let key = if leader_idx < peer_idx {
                    (leader_idx, peer_idx)
                } else {
                    (peer_idx, leader_idx)
                };
                latency_matrix.get(&key).copied().unwrap_or(10.0)
            };

            swarm.members.push(SwarmMember {
                peer_id: peer.node_id.clone(),
                role,
                assigned_layers: layer_cursor..layer_cursor + share,
                capacity_bytes: peer.available_vram_bytes + peer.available_memory_bytes,
                latency_to_leader_ms: latency_to_leader,
            });

            layer_cursor += share;
            assigned[peer_idx] = true;
        }

        swarm.recalculate_capacity();
        swarm.status = SwarmStatus::Forming;

        info!(
            "Formed swarm {} for model '{}': {} members, {:.1} GB capacity",
            swarm.swarm_id,
            model.display_name,
            swarm.member_count(),
            swarm.aggregate_capacity_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        );

        swarms.push(swarm);
    }

    // Collect unplaced peers (eligible but couldn't form a swarm)
    let mut unplaced = too_weak;
    for (i, &is_assigned) in assigned.iter().enumerate() {
        if !is_assigned {
            unplaced.push(swarm_eligible[i].node_id.clone());
        }
    }

    debug!(
        "Formation complete: {} swarms, {} drafters, {} unplaced",
        swarms.len(),
        drafters.len(),
        unplaced.len()
    );

    Ok(FormationResult {
        swarms,
        drafters,
        unplaced,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::MeshTier;
    use crate::nat::NatType;
    use hivebear_core::types::*;
    use std::collections::HashMap;
    use std::net::SocketAddr;

    fn make_peer(ram_gb: u64, vram_gb: u64, latency_map: HashMap<NodeId, f64>) -> PeerInfo {
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
            network_bandwidth_mbps: 1000.0,
            latency_ms: Some(5.0),
            tier: MeshTier::Free,
            reputation_score: 0.9,
            addr: "127.0.0.1:7878".parse::<SocketAddr>().unwrap(),
            external_addr: None,
            nat_type: NatType::Unknown,
            latency_map,
            serving_model_id: None,
            swarm_id: None,
            draft_capability: None,
        }
    }

    #[test]
    fn test_best_model_for_capacity() {
        let models = builtin_model_classes();
        let gb = 1024u64 * 1024 * 1024;

        // 6 GB -> should fit Mistral 7B (4 GB)
        let model = best_model_for_capacity(6 * gb, &models).unwrap();
        assert!(model.params_billions >= 3.0);

        // 12 GB -> should fit Llama 3.1 8B Q6 (7 GB)
        let model = best_model_for_capacity(12 * gb, &models).unwrap();
        assert!(model.params_billions >= 7.0);

        // 500 MB -> nothing fits
        assert!(best_model_for_capacity(500 * 1024 * 1024, &models).is_none());
    }

    #[test]
    fn test_form_swarms_basic() {
        let models = builtin_model_classes();
        let config = FormationConfig::default();

        // Create 3 peers with 8GB each, all with low latency to each other
        let (id1, _) = NodeId::generate();
        let (id2, _) = NodeId::generate();
        let (id3, _) = NodeId::generate();

        let mut lat1 = HashMap::new();
        lat1.insert(id2.clone(), 5.0);
        lat1.insert(id3.clone(), 5.0);

        let mut lat2 = HashMap::new();
        lat2.insert(id1.clone(), 5.0);
        lat2.insert(id3.clone(), 5.0);

        let mut lat3 = HashMap::new();
        lat3.insert(id1.clone(), 5.0);
        lat3.insert(id2.clone(), 5.0);

        let mut peers = vec![
            make_peer(8, 0, lat1),
            make_peer(8, 0, lat2),
            make_peer(8, 0, lat3),
        ];
        // Override node IDs
        peers[0].node_id = id1;
        peers[1].node_id = id2;
        peers[2].node_id = id3;

        let result = form_swarms(&peers, &config, &models).unwrap();

        assert_eq!(result.swarms.len(), 1);
        assert!(result.swarms[0].member_count() >= 2);
        assert!(result.unplaced.is_empty() || result.unplaced.len() <= 1);
    }

    #[test]
    fn test_weak_peers_become_drafters() {
        let models = builtin_model_classes();
        let config = FormationConfig::default();

        // 1 GB RAM, no GPU -> too weak for swarm, should become drafter
        let peer = make_peer(1, 0, HashMap::new());
        let result = form_swarms(&[peer], &config, &models).unwrap();

        assert_eq!(result.swarms.len(), 0);
        assert_eq!(result.drafters.len(), 1);
        assert_eq!(result.drafters[0].1.draft_model_id, "tinyllama-1.1b-q4_k_m");
    }

    #[test]
    fn test_high_latency_prevents_swarm() {
        let models = builtin_model_classes();
        let config = FormationConfig::default();

        let (id1, _) = NodeId::generate();
        let (id2, _) = NodeId::generate();

        // 200ms latency -- far too high
        let mut lat1 = HashMap::new();
        lat1.insert(id2.clone(), 200.0);
        let mut lat2 = HashMap::new();
        lat2.insert(id1.clone(), 200.0);

        let mut peers = vec![make_peer(8, 0, lat1), make_peer(8, 0, lat2)];
        peers[0].node_id = id1;
        peers[1].node_id = id2;

        let result = form_swarms(&peers, &config, &models).unwrap();
        assert_eq!(result.swarms.len(), 0);
    }

    #[test]
    fn test_empty_peers() {
        let models = builtin_model_classes();
        let config = FormationConfig::default();
        let result = form_swarms(&[], &config, &models).unwrap();
        assert!(result.swarms.is_empty());
        assert!(result.drafters.is_empty());
        assert!(result.unplaced.is_empty());
    }
}
