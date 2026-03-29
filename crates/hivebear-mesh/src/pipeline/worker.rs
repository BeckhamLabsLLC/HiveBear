use std::ops::Range;
use std::sync::Arc;

use bytes::Bytes;
use dashmap::DashMap;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::peer::NodeId;
use crate::protocol::MeshPipelineHandler;
use crate::transport::protocol::{MeshMessage, TensorDtype};
use crate::transport::MeshTransport;

/// A mesh worker that processes assigned layers for distributed inference.
///
/// The worker:
/// 1. Receives layer assignments from the initiator
/// 2. Loads its assigned model layers (via `MeshPipelineHandler` when available)
/// 3. Processes activation tensors and forwards to the next worker
/// 4. Releases resources on session teardown
pub struct PipelineWorker {
    transport: Arc<dyn MeshTransport>,
    local_id: NodeId,
    pipeline_handler: Option<Arc<dyn MeshPipelineHandler>>,
    sessions: DashMap<Uuid, WorkerSession>,
}

/// State for an active worker session.
#[derive(Debug)]
pub struct WorkerSession {
    pub session_id: Uuid,
    pub model_id: String,
    pub layer_range: Range<u32>,
    pub total_layers: u32,
    pub model_source: String,
    /// Next peer in the pipeline (`None` when this worker is the final stage).
    pub next_peer: Option<NodeId>,
    /// The initiator peer that receives the final logits.
    pub initiator_peer: NodeId,
}

impl PipelineWorker {
    pub fn new(
        transport: Arc<dyn MeshTransport>,
        local_id: NodeId,
        pipeline_handler: Option<Arc<dyn MeshPipelineHandler>>,
    ) -> Self {
        Self {
            transport,
            local_id,
            pipeline_handler,
            sessions: DashMap::new(),
        }
    }

    /// Listen for and handle messages from the mesh.
    ///
    /// Returns a receiver that yields worker events for the caller to handle.
    pub fn run(self: Arc<Self>) -> mpsc::Receiver<WorkerEvent> {
        let (tx, rx) = mpsc::channel(32);

        tokio::spawn(async move {
            loop {
                match self.transport.recv().await {
                    Ok((peer_id, msg)) => {
                        match msg {
                            MeshMessage::AssignLayers {
                                session_id,
                                model_id,
                                layer_range,
                                total_layers,
                                model_source,
                                next_peer,
                                initiator_peer,
                            } => {
                                info!(
                                    "Assigned layers {}..{} for model '{}' (session {})",
                                    layer_range.start, layer_range.end, model_id, session_id
                                );

                                // If a pipeline handler is available, load the assigned layers.
                                let mut load_error: Option<String> = None;
                                if let Some(ref handler) = self.pipeline_handler {
                                    if let Err(e) = handler
                                        .load_layers(
                                            &model_source,
                                            layer_range.clone(),
                                            total_layers,
                                        )
                                        .await
                                    {
                                        warn!(
                                            "Failed to load layers {}..{} for session {}: {e}",
                                            layer_range.start, layer_range.end, session_id
                                        );
                                        load_error = Some(e);
                                    }
                                }

                                let ready = load_error.is_none();

                                let session = WorkerSession {
                                    session_id,
                                    model_id,
                                    layer_range,
                                    total_layers,
                                    model_source,
                                    next_peer,
                                    initiator_peer,
                                };

                                // Store session for later lookups during forward passes.
                                self.sessions.insert(session_id, session);

                                // Acknowledge assignment
                                let ack = MeshMessage::AssignLayersAck {
                                    session_id,
                                    ready,
                                    error: load_error,
                                };
                                if let Err(e) = self.transport.send(&peer_id, ack).await {
                                    warn!("Failed to send ACK: {e}");
                                }

                                // Emit event — borrow from map for the fields we need.
                                if let Some(entry) = self.sessions.get(&session_id) {
                                    let evt_session = WorkerSession {
                                        session_id: entry.session_id,
                                        model_id: entry.model_id.clone(),
                                        layer_range: entry.layer_range.clone(),
                                        total_layers: entry.total_layers,
                                        model_source: entry.model_source.clone(),
                                        next_peer: entry.next_peer.clone(),
                                        initiator_peer: entry.initiator_peer.clone(),
                                    };
                                    let _ = tx
                                        .send(WorkerEvent::SessionStarted(Box::new(evt_session)))
                                        .await;
                                }
                            }
                            MeshMessage::ActivationTensor {
                                session_id,
                                token_position,
                                data,
                                shape,
                                dtype,
                            } => {
                                debug!(
                                    "Received activation tensor for position {token_position} (session {session_id})"
                                );

                                // Look up session routing information.
                                let session_info = self.sessions.get(&session_id).map(|s| {
                                    (
                                        s.next_peer.clone(),
                                        s.initiator_peer.clone(),
                                        s.layer_range.end,
                                        s.total_layers,
                                    )
                                });

                                if let Some(ref handler) = self.pipeline_handler {
                                    // Run forward pass through our assigned layers.
                                    match handler
                                        .forward_layers(
                                            data.to_vec(),
                                            shape,
                                            dtype.to_u8(),
                                            token_position as usize,
                                        )
                                        .await
                                    {
                                        Ok((output_data, output_shape, output_dtype_u8)) => {
                                            if let Some((
                                                next_peer,
                                                initiator_peer,
                                                layer_end,
                                                total_layers,
                                            )) = session_info
                                            {
                                                let is_last_stage = layer_end == total_layers;

                                                if is_last_stage {
                                                    // This is the final stage — send logits
                                                    // back to the initiator.
                                                    let vocab_size =
                                                        output_shape.last().copied().unwrap_or(0)
                                                            as u32;
                                                    let logits_msg = MeshMessage::Logits {
                                                        session_id,
                                                        token_position,
                                                        data: Bytes::from(output_data),
                                                        vocab_size,
                                                    };
                                                    if let Err(e) = self
                                                        .transport
                                                        .send(&initiator_peer, logits_msg)
                                                        .await
                                                    {
                                                        error!(
                                                            "Failed to send logits to initiator {}: {e}",
                                                            initiator_peer
                                                        );
                                                    }
                                                } else if let Some(ref next) = next_peer {
                                                    // Not the last stage — forward activation
                                                    // to the next peer in the pipeline.
                                                    let out_dtype =
                                                        TensorDtype::from_u8(output_dtype_u8)
                                                            .unwrap_or(dtype);
                                                    let fwd_msg = MeshMessage::ActivationTensor {
                                                        session_id,
                                                        token_position,
                                                        data: Bytes::from(output_data),
                                                        shape: output_shape,
                                                        dtype: out_dtype,
                                                    };
                                                    if let Err(e) =
                                                        self.transport.send(next, fwd_msg).await
                                                    {
                                                        error!(
                                                            "Failed to forward activation to next peer {}: {e}",
                                                            next
                                                        );
                                                    }
                                                } else {
                                                    warn!(
                                                        "Non-final stage has no next_peer for session {session_id}"
                                                    );
                                                }
                                            } else {
                                                warn!(
                                                    "No session found for activation tensor (session {session_id})"
                                                );
                                            }
                                        }
                                        Err(e) => {
                                            error!(
                                                "Forward pass failed for session {session_id}: {e}"
                                            );
                                            // Notify initiator of the error.
                                            if let Some((_, initiator_peer, _, _)) = session_info {
                                                let err_msg = MeshMessage::Error {
                                                    session_id: Some(session_id),
                                                    message: format!(
                                                        "Forward pass failed on node {}: {e}",
                                                        self.local_id
                                                    ),
                                                };
                                                let _ = self
                                                    .transport
                                                    .send(&initiator_peer, err_msg)
                                                    .await;
                                            }
                                        }
                                    }
                                } else {
                                    // No pipeline handler — backward-compatible stub behaviour.
                                    let _ = tx
                                        .send(WorkerEvent::ActivationReceived {
                                            session_id,
                                            token_position,
                                            from: peer_id,
                                        })
                                        .await;
                                }
                            }
                            MeshMessage::InferenceRequest {
                                session_id,
                                model_id,
                                messages_json,
                                max_tokens,
                                temperature,
                                top_p,
                            } => {
                                info!(
                                    "Inference request from {peer_id}: model '{}' (session {})",
                                    model_id, session_id
                                );
                                let _ = tx
                                    .send(WorkerEvent::InferenceRequested(Box::new(
                                        InferenceRequest {
                                            session_id,
                                            model_id,
                                            messages_json,
                                            max_tokens,
                                            temperature,
                                            top_p,
                                            from: peer_id,
                                        },
                                    )))
                                    .await;
                            }
                            MeshMessage::ReleaseSession { session_id } => {
                                info!("Releasing session {session_id}");

                                // Unload layers via the handler if available.
                                if let Some(ref handler) = self.pipeline_handler {
                                    if let Err(e) = handler.unload_layers().await {
                                        warn!(
                                            "Failed to unload layers for session {session_id}: {e}"
                                        );
                                    }
                                }

                                // Remove the session from our tracking map.
                                self.sessions.remove(&session_id);

                                let _ = tx.send(WorkerEvent::SessionReleased(session_id)).await;
                            }
                            MeshMessage::Ping { timestamp_ms } => {
                                let _ = self
                                    .transport
                                    .send(&peer_id, MeshMessage::Pong { timestamp_ms })
                                    .await;
                            }
                            MeshMessage::VerifyChallenge {
                                session_id,
                                layer_index,
                                input_hash,
                            } => {
                                // Compute a verification hash based on session state
                                use sha2::{Digest, Sha256};
                                let output_hash =
                                    if let Some(session) = self.sessions.get(&session_id) {
                                        // Hash the combination of input_hash + layer assignment
                                        // to produce a deterministic but session-specific response
                                        let mut hasher = Sha256::new();
                                        hasher.update(input_hash);
                                        hasher.update(session.layer_range.start.to_le_bytes());
                                        hasher.update(session.layer_range.end.to_le_bytes());
                                        hasher.update(layer_index.to_le_bytes());
                                        hasher.update(session.model_id.as_bytes());
                                        let result = hasher.finalize();
                                        let mut hash = [0u8; 32];
                                        hash.copy_from_slice(&result);
                                        hash
                                    } else {
                                        [0u8; 32]
                                    };

                                let _ = self
                                    .transport
                                    .send(
                                        &peer_id,
                                        MeshMessage::VerifyResponse {
                                            session_id,
                                            output_hash,
                                            passed: true,
                                        },
                                    )
                                    .await;
                            }
                            _ => {
                                debug!("Worker ignoring message from {peer_id}");
                            }
                        }
                    }
                    Err(e) => {
                        error!("Worker transport error: {e}");
                        let _ = tx.send(WorkerEvent::Error(e.to_string())).await;
                        break;
                    }
                }
            }
        });

        rx
    }
}

/// Events emitted by a worker for the host node to handle.
#[derive(Debug)]
pub enum WorkerEvent {
    SessionStarted(Box<WorkerSession>),
    ActivationReceived {
        session_id: Uuid,
        token_position: u32,
        from: NodeId,
    },
    /// A peer has requested full-model replication inference.
    InferenceRequested(Box<InferenceRequest>),
    SessionReleased(Uuid),
    Error(String),
}

/// Details for a full-model replication inference request.
#[derive(Debug)]
pub struct InferenceRequest {
    pub session_id: Uuid,
    pub model_id: String,
    pub messages_json: String,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub from: NodeId,
}
