use std::ops::Range;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::peer::NodeId;

/// A plan describing how to distribute a model across mesh peers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferencePlan {
    pub session_id: Uuid,
    pub model_id: String,
    pub total_layers: u32,
    pub assignments: Vec<LayerAssignment>,
    pub estimated_latency_ms: f64,
    pub estimated_throughput_tok_s: f64,
}

/// Assignment of a range of model layers to a specific peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerAssignment {
    pub peer_id: NodeId,
    pub layer_range: Range<u32>,
    pub estimated_compute_ms: f64,
    pub estimated_transfer_ms: f64,
}

impl InferencePlan {
    /// Get the pipeline order: list of peer IDs in layer-order.
    pub fn pipeline_order(&self) -> Vec<NodeId> {
        let mut assignments = self.assignments.clone();
        assignments.sort_by_key(|a| a.layer_range.start);
        assignments.into_iter().map(|a| a.peer_id).collect()
    }

    /// Find the assignment for a given peer.
    pub fn assignment_for(&self, peer_id: &NodeId) -> Option<&LayerAssignment> {
        self.assignments.iter().find(|a| &a.peer_id == peer_id)
    }

    /// Total number of peers involved.
    pub fn peer_count(&self) -> usize {
        self.assignments.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node_id() -> NodeId {
        let (id, _) = NodeId::generate();
        id
    }

    #[test]
    fn test_pipeline_order() {
        let id_a = make_node_id();
        let id_b = make_node_id();

        let plan = InferencePlan {
            session_id: Uuid::new_v4(),
            model_id: "test-model".into(),
            total_layers: 40,
            assignments: vec![
                LayerAssignment {
                    peer_id: id_b.clone(),
                    layer_range: 20..40,
                    estimated_compute_ms: 10.0,
                    estimated_transfer_ms: 5.0,
                },
                LayerAssignment {
                    peer_id: id_a.clone(),
                    layer_range: 0..20,
                    estimated_compute_ms: 10.0,
                    estimated_transfer_ms: 5.0,
                },
            ],
            estimated_latency_ms: 30.0,
            estimated_throughput_tok_s: 10.0,
        };

        let order = plan.pipeline_order();
        assert_eq!(order.len(), 2);
        assert_eq!(order[0], id_a);
        assert_eq!(order[1], id_b);
    }
}
