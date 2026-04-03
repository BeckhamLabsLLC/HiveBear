use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

use tracing::warn;

use crate::peer::NodeId;

/// Default number of consecutive ping failures before declaring a peer dead.
const DEFAULT_FAILURE_THRESHOLD: u32 = 3;

/// Monitors the health of peers in an active pipeline.
pub struct PeerHealthMonitor {
    states: Mutex<HashMap<Vec<u8>, PeerHealthState>>,
    failure_threshold: u32,
}

/// Health state tracking for a single peer.
#[derive(Debug, Clone)]
pub struct PeerHealthState {
    pub last_ping_sent: Option<Instant>,
    pub last_pong_received: Option<Instant>,
    pub consecutive_failures: u32,
    pub avg_latency_ms: f64,
    latency_samples: Vec<f64>,
}

impl PeerHealthState {
    fn new() -> Self {
        Self {
            last_ping_sent: None,
            last_pong_received: None,
            consecutive_failures: 0,
            avg_latency_ms: 0.0,
            latency_samples: Vec::new(),
        }
    }
}

impl PeerHealthMonitor {
    pub fn new() -> Self {
        Self {
            states: Mutex::new(HashMap::new()),
            failure_threshold: DEFAULT_FAILURE_THRESHOLD,
        }
    }

    pub fn with_threshold(failure_threshold: u32) -> Self {
        Self {
            states: Mutex::new(HashMap::new()),
            failure_threshold,
        }
    }

    /// Register a peer to monitor.
    pub fn track_peer(&self, peer_id: &NodeId) {
        let key = peer_id.0.to_bytes().to_vec();
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        states.entry(key).or_insert_with(PeerHealthState::new);
    }

    /// Record that a ping was sent to a peer.
    pub fn record_ping_sent(&self, peer_id: &NodeId) {
        let key = peer_id.0.to_bytes().to_vec();
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(state) = states.get_mut(&key) {
            state.last_ping_sent = Some(Instant::now());
        }
    }

    /// Record that a pong was received from a peer.
    pub fn record_pong_received(&self, peer_id: &NodeId) {
        let key = peer_id.0.to_bytes().to_vec();
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(state) = states.get_mut(&key) {
            let now = Instant::now();
            if let Some(ping_time) = state.last_ping_sent {
                let latency = now.duration_since(ping_time).as_millis() as f64;
                state.latency_samples.push(latency);
                if state.latency_samples.len() > 10 {
                    state.latency_samples.remove(0);
                }
                state.avg_latency_ms =
                    state.latency_samples.iter().sum::<f64>() / state.latency_samples.len() as f64;
            }
            state.last_pong_received = Some(now);
            state.consecutive_failures = 0;
        }
    }

    /// Record a ping failure (no pong received within timeout).
    pub fn record_ping_failure(&self, peer_id: &NodeId) {
        let key = peer_id.0.to_bytes().to_vec();
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(state) = states.get_mut(&key) {
            state.consecutive_failures += 1;
            if state.consecutive_failures >= self.failure_threshold {
                warn!(
                    "Peer {} has {} consecutive ping failures (threshold: {})",
                    peer_id, state.consecutive_failures, self.failure_threshold
                );
            }
        }
    }

    /// Check if a peer is considered dead (exceeded failure threshold).
    pub fn is_dead(&self, peer_id: &NodeId) -> bool {
        let key = peer_id.0.to_bytes().to_vec();
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        states
            .get(&key)
            .map(|s| s.consecutive_failures >= self.failure_threshold)
            .unwrap_or(false)
    }

    /// Get all peers that are considered dead.
    pub fn dead_peers(&self) -> Vec<Vec<u8>> {
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        states
            .iter()
            .filter(|(_, state)| state.consecutive_failures >= self.failure_threshold)
            .map(|(key, _)| key.clone())
            .collect()
    }

    /// Remove a peer from monitoring.
    pub fn remove_peer(&self, peer_id: &NodeId) {
        let key = peer_id.0.to_bytes().to_vec();
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        states.remove(&key);
    }

    /// Get the average latency for a peer in milliseconds.
    pub fn get_latency(&self, peer_id: &NodeId) -> Option<f64> {
        let key = peer_id.0.to_bytes().to_vec();
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        states.get(&key).map(|s| s.avg_latency_ms)
    }
}

impl Default for PeerHealthMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_track_and_ping() {
        let monitor = PeerHealthMonitor::new();
        let (peer_id, _) = NodeId::generate();

        monitor.track_peer(&peer_id);
        assert!(!monitor.is_dead(&peer_id));

        monitor.record_ping_sent(&peer_id);
        monitor.record_pong_received(&peer_id);
        assert!(!monitor.is_dead(&peer_id));
    }

    #[test]
    fn test_failure_threshold() {
        let monitor = PeerHealthMonitor::with_threshold(3);
        let (peer_id, _) = NodeId::generate();

        monitor.track_peer(&peer_id);

        monitor.record_ping_failure(&peer_id);
        assert!(!monitor.is_dead(&peer_id));
        monitor.record_ping_failure(&peer_id);
        assert!(!monitor.is_dead(&peer_id));
        monitor.record_ping_failure(&peer_id);
        assert!(monitor.is_dead(&peer_id));
    }

    #[test]
    fn test_pong_resets_failures() {
        let monitor = PeerHealthMonitor::with_threshold(3);
        let (peer_id, _) = NodeId::generate();

        monitor.track_peer(&peer_id);

        monitor.record_ping_failure(&peer_id);
        monitor.record_ping_failure(&peer_id);
        monitor.record_ping_sent(&peer_id);
        monitor.record_pong_received(&peer_id); // Resets failures
        assert!(!monitor.is_dead(&peer_id));
    }

    #[test]
    fn test_dead_peers() {
        let monitor = PeerHealthMonitor::with_threshold(1);
        let (peer1, _) = NodeId::generate();
        let (peer2, _) = NodeId::generate();

        monitor.track_peer(&peer1);
        monitor.track_peer(&peer2);

        monitor.record_ping_failure(&peer1);
        assert_eq!(monitor.dead_peers().len(), 1);
    }
}
