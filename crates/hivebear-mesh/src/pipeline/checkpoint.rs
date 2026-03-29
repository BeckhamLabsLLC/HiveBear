use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

use bytes::Bytes;
use uuid::Uuid;

use crate::transport::protocol::TensorDtype;

/// Stores activation tensor checkpoints at pipeline stage boundaries.
///
/// When a peer drops mid-inference, the system can replay from the last
/// checkpoint rather than restarting from token 0.
pub struct CheckpointStore {
    /// session_id -> (token_position -> checkpoint)
    checkpoints: Mutex<HashMap<Uuid, HashMap<u32, ActivationCheckpoint>>>,
    /// Maximum checkpoints to keep per session (older ones are evicted).
    max_per_session: usize,
}

/// A snapshot of the activation tensor at a pipeline stage boundary.
#[derive(Debug, Clone)]
pub struct ActivationCheckpoint {
    pub token_position: u32,
    pub activation_data: Bytes,
    pub shape: Vec<usize>,
    pub dtype: TensorDtype,
    /// Which layer boundary this checkpoint is from.
    pub source_layer: u32,
    pub timestamp: Instant,
}

impl CheckpointStore {
    pub fn new(max_per_session: usize) -> Self {
        Self {
            checkpoints: Mutex::new(HashMap::new()),
            max_per_session,
        }
    }

    /// Save a checkpoint for a given session and token position.
    pub fn save(
        &self,
        session_id: Uuid,
        token_position: u32,
        activation_data: Bytes,
        shape: Vec<usize>,
        dtype: TensorDtype,
        source_layer: u32,
    ) {
        let checkpoint = ActivationCheckpoint {
            token_position,
            activation_data,
            shape,
            dtype,
            source_layer,
            timestamp: Instant::now(),
        };

        let mut store = self.checkpoints.lock().unwrap();
        let session_map = store.entry(session_id).or_default();

        // Evict oldest if at capacity
        if session_map.len() >= self.max_per_session {
            if let Some(oldest_pos) = session_map
                .iter()
                .min_by_key(|(_, cp)| cp.timestamp)
                .map(|(pos, _)| *pos)
            {
                session_map.remove(&oldest_pos);
            }
        }

        session_map.insert(token_position, checkpoint);
    }

    /// Get the most recent checkpoint for a session.
    pub fn latest(&self, session_id: &Uuid) -> Option<ActivationCheckpoint> {
        let store = self.checkpoints.lock().unwrap();
        store.get(session_id).and_then(|session_map| {
            session_map
                .values()
                .max_by_key(|cp| cp.token_position)
                .cloned()
        })
    }

    /// Get a specific checkpoint by session and position.
    pub fn get(&self, session_id: &Uuid, token_position: u32) -> Option<ActivationCheckpoint> {
        let store = self.checkpoints.lock().unwrap();
        store
            .get(session_id)
            .and_then(|session_map| session_map.get(&token_position).cloned())
    }

    /// Remove all checkpoints for a session (on teardown).
    pub fn clear_session(&self, session_id: &Uuid) {
        let mut store = self.checkpoints.lock().unwrap();
        store.remove(session_id);
    }

    /// Total number of checkpoints across all sessions.
    pub fn total_checkpoints(&self) -> usize {
        let store = self.checkpoints.lock().unwrap();
        store.values().map(|m| m.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_and_retrieve() {
        let store = CheckpointStore::new(10);
        let session = Uuid::new_v4();

        store.save(
            session,
            0,
            Bytes::from(vec![1, 2, 3]),
            vec![1, 3],
            TensorDtype::F16,
            0,
        );
        store.save(
            session,
            1,
            Bytes::from(vec![4, 5, 6]),
            vec![1, 3],
            TensorDtype::F16,
            0,
        );

        let latest = store.latest(&session).unwrap();
        assert_eq!(latest.token_position, 1);
        assert_eq!(latest.activation_data.as_ref(), &[4, 5, 6]);
    }

    #[test]
    fn test_eviction() {
        let store = CheckpointStore::new(2);
        let session = Uuid::new_v4();

        store.save(
            session,
            0,
            Bytes::from(vec![1]),
            vec![1],
            TensorDtype::F16,
            0,
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
        store.save(
            session,
            1,
            Bytes::from(vec![2]),
            vec![1],
            TensorDtype::F16,
            0,
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
        store.save(
            session,
            2,
            Bytes::from(vec![3]),
            vec![1],
            TensorDtype::F16,
            0,
        );

        // Position 0 should have been evicted (oldest)
        assert!(store.get(&session, 0).is_none());
        assert!(store.get(&session, 1).is_some());
        assert!(store.get(&session, 2).is_some());
    }

    #[test]
    fn test_clear_session() {
        let store = CheckpointStore::new(10);
        let session = Uuid::new_v4();

        store.save(
            session,
            0,
            Bytes::from(vec![1]),
            vec![1],
            TensorDtype::F16,
            0,
        );
        assert_eq!(store.total_checkpoints(), 1);

        store.clear_session(&session);
        assert_eq!(store.total_checkpoints(), 0);
    }
}
