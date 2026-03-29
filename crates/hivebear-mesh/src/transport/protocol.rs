use std::ops::Range;

use bincode::Options;
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::peer::NodeId;
use hivebear_core::types::HardwareProfile;

/// Data type of tensor elements transferred between nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorDtype {
    F16,
    F32,
    BF16,
}

impl TensorDtype {
    pub fn byte_size(&self) -> usize {
        match self {
            TensorDtype::F16 | TensorDtype::BF16 => 2,
            TensorDtype::F32 => 4,
        }
    }

    /// Encode as a `u8` tag for cross-boundary transfer (e.g. pipeline handler).
    pub fn to_u8(self) -> u8 {
        match self {
            TensorDtype::F16 => 0,
            TensorDtype::F32 => 1,
            TensorDtype::BF16 => 2,
        }
    }

    /// Decode from a `u8` tag. Returns `None` for unknown values.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(TensorDtype::F16),
            1 => Some(TensorDtype::F32),
            2 => Some(TensorDtype::BF16),
            _ => None,
        }
    }
}

/// Compression method for tensor data transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionMethod {
    None,
    Lz4,
    Zstd { level: u8 },
}

/// Messages exchanged between mesh nodes over QUIC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeshMessage {
    /// Handshake: exchange node identity and capabilities.
    Hello {
        node_id: NodeId,
        hardware: HardwareProfile,
        protocol_version: u32,
    },

    /// Acknowledge a Hello message.
    HelloAck { node_id: NodeId, accepted: bool },

    /// Layer assignment from the initiator to a worker.
    AssignLayers {
        session_id: Uuid,
        model_id: String,
        layer_range: Range<u32>,
        total_layers: u32,
        /// Path or URL where the worker can find the model weights.
        model_source: String,
        /// Next peer in the pipeline (None if this is the final stage).
        next_peer: Option<crate::peer::NodeId>,
        /// The initiator peer that receives final logits.
        initiator_peer: crate::peer::NodeId,
    },

    /// Worker acknowledges layer assignment and readiness.
    AssignLayersAck {
        session_id: Uuid,
        ready: bool,
        error: Option<String>,
    },

    /// Activation tensor flowing through the pipeline.
    ActivationTensor {
        session_id: Uuid,
        token_position: u32,
        data: Bytes,
        shape: Vec<usize>,
        dtype: TensorDtype,
    },

    /// Final logits returned to the initiator from the last worker.
    Logits {
        session_id: Uuid,
        token_position: u32,
        data: Bytes,
        vocab_size: u32,
    },

    /// Request to verify a specific layer's computation.
    VerifyChallenge {
        session_id: Uuid,
        layer_index: u32,
        input_hash: [u8; 32],
    },

    /// Response to a verification challenge.
    VerifyResponse {
        session_id: Uuid,
        output_hash: [u8; 32],
        passed: bool,
    },

    /// Session teardown: release model and resources.
    ReleaseSession { session_id: Uuid },

    /// Heartbeat / keep-alive.
    Ping { timestamp_ms: u64 },

    /// Heartbeat response.
    Pong { timestamp_ms: u64 },

    /// Error from a peer.
    Error {
        session_id: Option<Uuid>,
        message: String,
    },

    // ── Full-model replication messages ─────────────────────────
    /// Request a peer to run full-model inference (replication mode).
    InferenceRequest {
        session_id: Uuid,
        model_id: String,
        /// Serialized chat messages (JSON).
        messages_json: String,
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
    },

    /// Streamed token from a peer running inference (replication mode).
    InferenceToken {
        session_id: Uuid,
        text: String,
        token_id: u32,
        is_done: bool,
    },

    /// Final response or error from replication inference.
    InferenceComplete {
        session_id: Uuid,
        full_text: String,
        tokens_generated: u32,
        error: Option<String>,
    },

    /// Heartbeat within an active pipeline session.
    PipelineHeartbeat {
        session_id: Uuid,
        stage_index: u32,
        tokens_processed: u32,
    },

    /// Notification that a peer is leaving the pipeline gracefully.
    PeerLeaving {
        session_id: Uuid,
        reason: String,
        /// Seconds until departure (gives time to rebalance).
        departure_in_secs: u32,
    },

    /// Compression method used for activation tensor data.
    CompressedActivationTensor {
        session_id: Uuid,
        token_position: u32,
        data: Bytes,
        shape: Vec<usize>,
        dtype: TensorDtype,
        compression: CompressionMethod,
    },

    // ── Swarm management messages ────────────────────────────────
    /// Invite a peer to join a swarm with a specific role and layer assignment.
    SwarmInvite {
        swarm_id: Uuid,
        model_id: String,
        /// 0 = Leader, 1 = Worker, 2 = Standby
        role: u8,
        layer_range: Range<u32>,
        total_layers: u32,
        /// Path or URL where the worker can find the model weights.
        model_source: String,
        /// All members and their layer assignments: (node_id, layer_start, layer_end).
        members: Vec<(NodeId, u32, u32)>,
    },

    /// Acknowledge a swarm invitation.
    SwarmInviteAck {
        swarm_id: Uuid,
        accepted: bool,
        error: Option<String>,
    },

    /// Periodic swarm health status from the leader.
    SwarmHeartbeat {
        swarm_id: Uuid,
        active_requests: u32,
        avg_tok_s: f64,
    },

    /// Notify all swarm members of a layer reassignment (rebalance).
    SwarmRebalance {
        swarm_id: Uuid,
        /// New layer assignments: (node_id, layer_start, layer_end).
        new_assignments: Vec<(NodeId, u32, u32)>,
        reason: String,
    },

    // ── Speculative decoding messages ────────────────────────────
    /// Draft tokens generated by a speculative decoding drafter.
    DraftTokens {
        session_id: Uuid,
        /// Token IDs generated by the draft model.
        tokens: Vec<u32>,
        /// Which draft model was used.
        draft_model_id: String,
    },

    /// Verification result from the target model swarm.
    VerifyDraft {
        session_id: Uuid,
        /// How many of the draft tokens were accepted (0..=tokens.len()).
        accepted_count: u32,
        /// Correction token at the first rejection point (if any).
        correction_token: Option<u32>,
    },
}

/// Current protocol version.
pub const PROTOCOL_VERSION: u32 = 2;

/// Maximum serialized message size (16 MiB).
const MAX_MESSAGE_SIZE: u64 = 16 * 1024 * 1024;

/// Serialize a MeshMessage to bytes using bincode.
///
/// Uses `DefaultOptions` with a size limit for consistency with `decode`.
pub fn encode(msg: &MeshMessage) -> crate::error::Result<Vec<u8>> {
    bincode::DefaultOptions::new()
        .with_limit(MAX_MESSAGE_SIZE)
        .serialize(msg)
        .map_err(|e| crate::error::MeshError::Serialization(e.to_string()))
}

/// Deserialize a MeshMessage from bytes.
///
/// Applies a 16 MiB size limit to prevent memory-bomb attacks from
/// oversized payloads.
pub fn decode(data: &[u8]) -> crate::error::Result<MeshMessage> {
    bincode::DefaultOptions::new()
        .with_limit(MAX_MESSAGE_SIZE)
        .deserialize(data)
        .map_err(|e| crate::error::MeshError::Protocol(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ping_pong_roundtrip() {
        let msg = MeshMessage::Ping {
            timestamp_ms: 1234567890,
        };
        let encoded = encode(&msg).unwrap();
        let decoded = decode(&encoded).unwrap();
        match decoded {
            MeshMessage::Ping { timestamp_ms } => assert_eq!(timestamp_ms, 1234567890),
            _ => panic!("Expected Ping"),
        }
    }

    #[test]
    fn test_activation_tensor_roundtrip() {
        let data = Bytes::from(vec![0u8; 1024]);
        let msg = MeshMessage::ActivationTensor {
            session_id: Uuid::new_v4(),
            token_position: 42,
            data: data.clone(),
            shape: vec![1, 4096],
            dtype: TensorDtype::F16,
        };
        let encoded = encode(&msg).unwrap();
        let decoded = decode(&encoded).unwrap();
        match decoded {
            MeshMessage::ActivationTensor {
                token_position,
                shape,
                dtype,
                ..
            } => {
                assert_eq!(token_position, 42);
                assert_eq!(shape, vec![1, 4096]);
                assert_eq!(dtype, TensorDtype::F16);
            }
            _ => panic!("Expected ActivationTensor"),
        }
    }

    #[test]
    fn test_error_message_roundtrip() {
        let msg = MeshMessage::Error {
            session_id: Some(Uuid::new_v4()),
            message: "something went wrong".into(),
        };
        let encoded = encode(&msg).unwrap();
        let decoded = decode(&encoded).unwrap();
        match decoded {
            MeshMessage::Error { message, .. } => {
                assert_eq!(message, "something went wrong");
            }
            _ => panic!("Expected Error"),
        }
    }

    #[test]
    fn test_tensor_dtype_byte_size() {
        assert_eq!(TensorDtype::F16.byte_size(), 2);
        assert_eq!(TensorDtype::F32.byte_size(), 4);
        assert_eq!(TensorDtype::BF16.byte_size(), 2);
    }
}
