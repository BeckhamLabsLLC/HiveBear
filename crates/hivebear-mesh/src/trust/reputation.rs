use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::peer::NodeId;

/// Tracks and persists peer reputation scores.
pub struct ReputationManager {
    scores: HashMap<Vec<u8>, ReputationRecord>,
    storage_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReputationRecord {
    /// Overall score (0.0 = banned, 1.0 = perfect).
    pub score: f64,
    /// Number of verification checks passed.
    pub verifications_passed: u64,
    /// Number of verification checks failed.
    pub verifications_failed: u64,
    /// Total uptime in seconds contributed.
    pub uptime_seconds: u64,
    /// Average latency consistency (lower is better).
    pub avg_latency_variance_ms: f64,
    /// Unix timestamp of last interaction.
    pub last_seen: u64,
}

impl Default for ReputationRecord {
    fn default() -> Self {
        Self {
            score: 0.5, // Neutral starting score
            verifications_passed: 0,
            verifications_failed: 0,
            uptime_seconds: 0,
            avg_latency_variance_ms: 0.0,
            last_seen: 0,
        }
    }
}

/// Threshold below which a peer is banned.
const BAN_THRESHOLD: f64 = 0.2;

/// Time decay factor: reputation decays toward neutral over time.
const DECAY_HALF_LIFE_SECS: f64 = 7.0 * 24.0 * 3600.0; // 1 week

impl ReputationManager {
    pub fn new(storage_path: Option<PathBuf>) -> Self {
        let mut mgr = Self {
            scores: HashMap::new(),
            storage_path,
        };
        mgr.load();
        mgr
    }

    fn node_key(id: &NodeId) -> Vec<u8> {
        id.0.to_bytes().to_vec()
    }

    /// Get the reputation score for a peer.
    pub fn score(&self, node_id: &NodeId) -> f64 {
        self.scores
            .get(&Self::node_key(node_id))
            .map(|r| r.score)
            .unwrap_or(0.5)
    }

    /// Check if a peer is banned.
    pub fn is_banned(&self, node_id: &NodeId) -> bool {
        self.score(node_id) < BAN_THRESHOLD
    }

    /// Record a verification result.
    pub fn record_verification(&mut self, node_id: &NodeId, passed: bool) {
        let key = Self::node_key(node_id);
        let record = self.scores.entry(key).or_default();

        if passed {
            record.verifications_passed += 1;
        } else {
            record.verifications_failed += 1;
        }

        record.last_seen = now_secs();
        self.recalculate_score(node_id);
        self.save();
    }

    /// Record uptime contribution.
    pub fn record_uptime(&mut self, node_id: &NodeId, seconds: u64) {
        let key = Self::node_key(node_id);
        let record = self.scores.entry(key).or_default();
        record.uptime_seconds += seconds;
        record.last_seen = now_secs();
        self.recalculate_score(node_id);
        self.save();
    }

    /// Recalculate score based on all factors.
    fn recalculate_score(&mut self, node_id: &NodeId) {
        let key = Self::node_key(node_id);
        let record = match self.scores.get_mut(&key) {
            Some(r) => r,
            None => return,
        };

        let total_verifications = record.verifications_passed + record.verifications_failed;
        let verification_rate = if total_verifications > 0 {
            record.verifications_passed as f64 / total_verifications as f64
        } else {
            0.5 // Neutral when no data
        };

        // Uptime bonus: up to +0.1 for 24h+ uptime
        let uptime_bonus = (record.uptime_seconds as f64 / 86400.0).min(1.0) * 0.1;

        // Base score from verification rate (weight: 0.8)
        let base = verification_rate * 0.8 + uptime_bonus + 0.1; // 0.1 floor

        // Time decay toward 0.5
        let age_secs = now_secs().saturating_sub(record.last_seen) as f64;
        let decay = (-age_secs / DECAY_HALF_LIFE_SECS * std::f64::consts::LN_2).exp();
        record.score = base * decay + 0.5 * (1.0 - decay);
        record.score = record.score.clamp(0.0, 1.0);
    }

    /// Get all reputation records (for status display).
    pub fn all_records(&self) -> Vec<(Vec<u8>, &ReputationRecord)> {
        self.scores.iter().map(|(k, v)| (k.clone(), v)).collect()
    }

    fn load(&mut self) {
        if let Some(path) = &self.storage_path {
            if let Ok(data) = std::fs::read_to_string(path) {
                match serde_json::from_str(&data) {
                    Ok(scores) => self.scores = scores,
                    Err(e) => warn!("Failed to parse reputation data: {e}"),
                }
            }
        }
    }

    fn save(&self) {
        if let Some(path) = &self.storage_path {
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            match serde_json::to_string_pretty(&self.scores) {
                Ok(data) => {
                    if let Err(e) = std::fs::write(path, data) {
                        warn!("Failed to save reputation data: {e}");
                    }
                }
                Err(e) => warn!("Failed to serialize reputation data: {e}"),
            }
        }
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_score() {
        let mgr = ReputationManager::new(None);
        let (id, _) = NodeId::generate();
        assert!((mgr.score(&id) - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_passed_verification_increases_score() {
        let mut mgr = ReputationManager::new(None);
        let (id, _) = NodeId::generate();

        for _ in 0..10 {
            mgr.record_verification(&id, true);
        }

        assert!(mgr.score(&id) > 0.5);
    }

    #[test]
    fn test_failed_verification_decreases_score() {
        let mut mgr = ReputationManager::new(None);
        let (id, _) = NodeId::generate();

        for _ in 0..20 {
            mgr.record_verification(&id, false);
        }

        assert!(mgr.score(&id) < 0.5);
    }

    #[test]
    fn test_ban_threshold() {
        let mut mgr = ReputationManager::new(None);
        let (id, _) = NodeId::generate();

        // Many failures should trigger ban
        for _ in 0..50 {
            mgr.record_verification(&id, false);
        }

        assert!(mgr.is_banned(&id));
    }

    #[test]
    fn test_score_clamped() {
        let mut mgr = ReputationManager::new(None);
        let (id, _) = NodeId::generate();

        for _ in 0..1000 {
            mgr.record_verification(&id, true);
        }

        assert!(mgr.score(&id) <= 1.0);
        assert!(mgr.score(&id) >= 0.0);
    }
}
