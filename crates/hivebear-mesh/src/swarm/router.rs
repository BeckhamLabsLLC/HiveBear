use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use dashmap::DashMap;
use tokio::sync::{oneshot, Mutex};
use tracing::info;
use uuid::Uuid;

use crate::error::{MeshError, Result};
use crate::peer::NodeId;
use crate::swarm::{SwarmDefinition, SwarmId, SwarmStatus};

/// Tracks per-swarm load metrics.
pub struct SwarmLoad {
    pub active_requests: AtomicU32,
    pub tokens_generated_total: AtomicU64,
    pub last_request_at: Mutex<Instant>,
}

impl Default for SwarmLoad {
    fn default() -> Self {
        Self::new()
    }
}

impl SwarmLoad {
    pub fn new() -> Self {
        Self {
            active_requests: AtomicU32::new(0),
            tokens_generated_total: AtomicU64::new(0),
            last_request_at: Mutex::new(Instant::now()),
        }
    }
}

/// A queued inference request waiting for an available swarm.
pub struct QueuedRequest {
    pub request_id: Uuid,
    pub model_id: String,
    pub queued_at: Instant,
    pub respond_to: oneshot::Sender<SwarmId>,
}

/// Routes inference requests to the best available swarm.
///
/// Enforces the contribution-required access model: only peers that are
/// actively contributing (members of an active swarm with current heartbeat)
/// can make inference requests.
pub struct SwarmRouter {
    /// Active swarms indexed by model_id.
    swarms: DashMap<String, Vec<Arc<SwarmDefinition>>>,
    /// Per-swarm load counters.
    load: DashMap<SwarmId, Arc<SwarmLoad>>,
    /// Overflow queue for when all swarms are busy.
    queue: Mutex<VecDeque<QueuedRequest>>,
    /// Maximum queue depth before rejecting requests.
    #[allow(dead_code)]
    max_queue_depth: usize,
    /// Maximum concurrent requests per swarm.
    max_requests_per_swarm: u32,
    /// Set of currently contributing peer IDs.
    contributing_peers: DashMap<Vec<u8>, bool>,
}

impl SwarmRouter {
    pub fn new() -> Self {
        Self {
            swarms: DashMap::new(),
            load: DashMap::new(),
            queue: Mutex::new(VecDeque::new()),
            max_queue_depth: 100,
            max_requests_per_swarm: 2,
            contributing_peers: DashMap::new(),
        }
    }

    pub fn with_limits(max_queue_depth: usize, max_requests_per_swarm: u32) -> Self {
        Self {
            max_queue_depth,
            max_requests_per_swarm,
            ..Self::new()
        }
    }

    /// Register a swarm with the router.
    pub fn register_swarm(&self, swarm: Arc<SwarmDefinition>) {
        let model_id = swarm.model_id.clone();
        let swarm_id = swarm.swarm_id.clone();

        self.load
            .entry(swarm_id)
            .or_insert_with(|| Arc::new(SwarmLoad::new()));

        self.swarms.entry(model_id.clone()).or_default().push(swarm);

        info!("Registered swarm for model '{model_id}'");
    }

    /// Unregister a swarm (on disband).
    pub fn unregister_swarm(&self, swarm_id: &SwarmId) {
        self.load.remove(swarm_id);
        // Remove from all model entries
        for mut entry in self.swarms.iter_mut() {
            entry.value_mut().retain(|s| s.swarm_id != *swarm_id);
        }
        // Clean up empty model entries
        self.swarms.retain(|_, v| !v.is_empty());
    }

    /// Mark a peer as contributing (they can make inference requests).
    pub fn set_contributing(&self, peer_id: &NodeId, contributing: bool) {
        let key = peer_id.0.to_bytes().to_vec();
        if contributing {
            self.contributing_peers.insert(key, true);
        } else {
            self.contributing_peers.remove(&key);
        }
    }

    /// Check if a peer is currently contributing.
    pub fn is_contributing(&self, peer_id: &NodeId) -> bool {
        let key = peer_id.0.to_bytes().to_vec();
        self.contributing_peers.contains_key(&key)
    }

    /// Select the best swarm for a given model and requesting peer.
    ///
    /// Returns an error if the peer is not contributing or no swarm is available.
    pub fn select_swarm(&self, model_id: &str, requester: &NodeId) -> Result<Arc<SwarmDefinition>> {
        // Gate: contribution required
        if !self.is_contributing(requester) {
            return Err(MeshError::Auth(
                "You must contribute to use the network. Run `hivebear contribute` to join.".into(),
            ));
        }

        let swarms = self
            .swarms
            .get(model_id)
            .ok_or_else(|| MeshError::NoPeersAvailable(model_id.into()))?;

        let mut best_swarm: Option<(Arc<SwarmDefinition>, f64)> = None;

        for swarm in swarms.value().iter() {
            if !matches!(swarm.status, SwarmStatus::Ready | SwarmStatus::Busy) {
                continue;
            }

            let load = self
                .load
                .get(&swarm.swarm_id)
                .map(|l| l.active_requests.load(Ordering::Relaxed))
                .unwrap_or(0);

            if load >= self.max_requests_per_swarm {
                continue; // At capacity
            }

            let health_factor = match swarm.status {
                SwarmStatus::Ready => 1.0,
                SwarmStatus::Degraded => 0.5,
                _ => 0.0,
            };

            let throughput = swarm.estimated_tok_s.max(1.0);
            let score = (1.0 / (load as f64 + 1.0)) * throughput * health_factor;

            match &best_swarm {
                Some((_, best_score)) if score <= *best_score => {}
                _ => best_swarm = Some((swarm.clone(), score)),
            }
        }

        best_swarm
            .map(|(swarm, _)| swarm)
            .ok_or_else(|| MeshError::NoPeersAvailable(model_id.into()))
    }

    /// Record that a request started on a swarm.
    pub fn request_started(&self, swarm_id: &SwarmId) {
        if let Some(load) = self.load.get(swarm_id) {
            load.active_requests.fetch_add(1, Ordering::Relaxed);
            if let Ok(mut last) = load.last_request_at.try_lock() {
                *last = Instant::now();
            }
        }
    }

    /// Record that a request finished on a swarm.
    pub fn request_finished(&self, swarm_id: &SwarmId, tokens_generated: u32) {
        if let Some(load) = self.load.get(swarm_id) {
            load.active_requests.fetch_sub(1, Ordering::Relaxed);
            load.tokens_generated_total
                .fetch_add(tokens_generated as u64, Ordering::Relaxed);
        }

        // Try to drain the queue
        self.try_drain_queue();
    }

    /// Attempt to route queued requests to newly available swarms.
    fn try_drain_queue(&self) {
        let mut queue = match self.queue.try_lock() {
            Ok(q) => q,
            Err(_) => return,
        };

        while let Some(queued) = queue.front() {
            // Check for expired requests (30 second timeout)
            if queued.queued_at.elapsed().as_secs() > 30 {
                queue.pop_front(); // Drop expired
                continue;
            }

            // Try to find a swarm (use a dummy peer ID -- queue doesn't re-check contribution)
            if let Some(swarms) = self.swarms.get(&queued.model_id) {
                let available = swarms.value().iter().find(|s| {
                    let load = self
                        .load
                        .get(&s.swarm_id)
                        .map(|l| l.active_requests.load(Ordering::Relaxed))
                        .unwrap_or(0);
                    load < self.max_requests_per_swarm
                        && matches!(s.status, SwarmStatus::Ready | SwarmStatus::Busy)
                });

                if let Some(swarm) = available {
                    let req = queue.pop_front().unwrap();
                    let _ = req.respond_to.send(swarm.swarm_id.clone());
                } else {
                    break; // No swarm available, stop draining
                }
            } else {
                queue.pop_front(); // Model doesn't exist, drop
            }
        }
    }

    /// List all models currently available on the network.
    pub fn available_models(&self) -> Vec<(String, usize)> {
        self.swarms
            .iter()
            .map(|entry| {
                let model_id = entry.key().clone();
                let count = entry
                    .value()
                    .iter()
                    .filter(|s| matches!(s.status, SwarmStatus::Ready | SwarmStatus::Busy))
                    .count();
                (model_id, count)
            })
            .filter(|(_, count)| *count > 0)
            .collect()
    }

    /// Total number of registered swarms.
    pub fn swarm_count(&self) -> usize {
        self.swarms.iter().map(|e| e.value().len()).sum()
    }

    /// Get load info for a specific swarm.
    pub fn get_load(&self, swarm_id: &SwarmId) -> Option<u32> {
        self.load
            .get(swarm_id)
            .map(|l| l.active_requests.load(Ordering::Relaxed))
    }
}

impl Default for SwarmRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::swarm::{SwarmMember, SwarmRole};

    fn make_swarm(model_id: &str, status: SwarmStatus, tok_s: f64) -> SwarmDefinition {
        let (peer_id, _) = NodeId::generate();
        SwarmDefinition {
            swarm_id: SwarmId::new(),
            model_id: model_id.into(),
            members: vec![SwarmMember {
                peer_id,
                role: SwarmRole::Leader,
                assigned_layers: 0..32,
                capacity_bytes: 8 * 1024 * 1024 * 1024,
                latency_to_leader_ms: 0.0,
            }],
            pipeline_plan: None,
            aggregate_capacity_bytes: 8 * 1024 * 1024 * 1024,
            estimated_tok_s: tok_s,
            status,
            created_at: Instant::now(),
        }
    }

    #[test]
    fn test_contribution_gate() {
        let router = SwarmRouter::new();
        let swarm = make_swarm("test-model", SwarmStatus::Ready, 10.0);
        router.register_swarm(Arc::new(swarm));

        let (requester, _) = NodeId::generate();

        // Not contributing -> rejected
        let result = router.select_swarm("test-model", &requester);
        assert!(result.is_err());

        // Mark as contributing -> accepted
        router.set_contributing(&requester, true);
        let result = router.select_swarm("test-model", &requester);
        assert!(result.is_ok());
    }

    #[test]
    fn test_load_balancing() {
        let router = SwarmRouter::new();

        let swarm1 = Arc::new(make_swarm("model-a", SwarmStatus::Ready, 10.0));
        let swarm2 = Arc::new(make_swarm("model-a", SwarmStatus::Ready, 10.0));
        let id1 = swarm1.swarm_id.clone();
        let id2 = swarm2.swarm_id.clone();

        router.register_swarm(swarm1);
        router.register_swarm(swarm2);

        let (requester, _) = NodeId::generate();
        router.set_contributing(&requester, true);

        // Add load to swarm1
        router.request_started(&id1);

        // Should prefer swarm2 (lower load)
        let selected = router.select_swarm("model-a", &requester).unwrap();
        assert_eq!(selected.swarm_id, id2);
    }

    #[test]
    fn test_available_models() {
        let router = SwarmRouter::new();
        router.register_swarm(Arc::new(make_swarm("model-a", SwarmStatus::Ready, 10.0)));
        router.register_swarm(Arc::new(make_swarm("model-b", SwarmStatus::Ready, 5.0)));
        router.register_swarm(Arc::new(make_swarm("model-b", SwarmStatus::Forming, 5.0)));

        let models = router.available_models();
        assert_eq!(models.len(), 2);

        let model_b = models.iter().find(|(id, _)| id == "model-b").unwrap();
        assert_eq!(model_b.1, 1); // Only the Ready one counts
    }

    #[test]
    fn test_unregister_swarm() {
        let router = SwarmRouter::new();
        let swarm = Arc::new(make_swarm("test", SwarmStatus::Ready, 10.0));
        let id = swarm.swarm_id.clone();
        router.register_swarm(swarm);

        assert_eq!(router.swarm_count(), 1);
        router.unregister_swarm(&id);
        assert_eq!(router.swarm_count(), 0);
    }
}
