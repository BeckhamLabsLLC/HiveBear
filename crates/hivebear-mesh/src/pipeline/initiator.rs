use std::sync::Arc;

use bytes::Bytes;
use tokio::sync::mpsc;
use tracing::{debug, info};
use uuid::Uuid;

use crate::error::{MeshError, Result};
use crate::peer::NodeId;
use crate::pipeline::checkpoint::CheckpointStore;
use crate::protocol::MeshPipelineHandler;
use crate::scheduler::plan::InferencePlan;
use crate::transport::protocol::MeshMessage;
use crate::transport::MeshTransport;
use hivebear_inference::Token;

/// How often to save activation checkpoints (every N tokens).
const CHECKPOINT_INTERVAL: u32 = 10;

/// The initiator orchestrates a distributed inference pipeline.
///
/// It assigns layers to workers, sends activation tensors through the
/// pipeline, receives logits from the final worker, samples the next
/// token, and streams tokens to the caller.
///
/// Includes checkpoint support: every `CHECKPOINT_INTERVAL` tokens, the
/// activation tensor is saved to the `CheckpointStore` so that recovery
/// can resume from the latest checkpoint instead of restarting.
pub struct PipelineInitiator {
    transport: Arc<dyn MeshTransport>,
    plan: InferencePlan,
    local_id: NodeId,
    checkpoints: Arc<CheckpointStore>,
}

impl PipelineInitiator {
    pub fn new(transport: Arc<dyn MeshTransport>, plan: InferencePlan, local_id: NodeId) -> Self {
        Self {
            transport,
            plan,
            local_id,
            checkpoints: Arc::new(CheckpointStore::new(20)),
        }
    }

    /// Create an initiator with an existing checkpoint store (for shared recovery).
    pub fn with_checkpoints(
        transport: Arc<dyn MeshTransport>,
        plan: InferencePlan,
        local_id: NodeId,
        checkpoints: Arc<CheckpointStore>,
    ) -> Self {
        Self {
            transport,
            plan,
            local_id,
            checkpoints,
        }
    }

    /// Get a reference to the checkpoint store.
    pub fn checkpoints(&self) -> &Arc<CheckpointStore> {
        &self.checkpoints
    }

    /// Set up the pipeline by assigning layers to all workers.
    pub async fn setup(&self, model_source: &str) -> Result<()> {
        info!(
            "Setting up pipeline for model '{}' across {} peers",
            self.plan.model_id,
            self.plan.peer_count()
        );

        for (i, assignment) in self.plan.assignments.iter().enumerate() {
            let next_peer = self.plan.assignments.get(i + 1).map(|a| a.peer_id.clone());
            let msg = MeshMessage::AssignLayers {
                session_id: self.plan.session_id,
                model_id: self.plan.model_id.clone(),
                layer_range: assignment.layer_range.clone(),
                total_layers: self.plan.total_layers,
                model_source: model_source.to_string(),
                next_peer,
                initiator_peer: self.local_id.clone(),
            };

            self.transport.send(&assignment.peer_id, msg).await?;
        }

        // Wait for all workers to acknowledge
        let mut acks_received = 0;
        let expected = self.plan.peer_count();

        while acks_received < expected {
            let (peer_id, msg) = self.transport.recv().await?;
            match msg {
                MeshMessage::AssignLayersAck {
                    session_id,
                    ready,
                    error,
                } if session_id == self.plan.session_id => {
                    if ready {
                        debug!("Peer {peer_id} ready");
                        acks_received += 1;
                    } else {
                        let err_msg = error.unwrap_or_else(|| "unknown".into());
                        return Err(MeshError::Pipeline(format!(
                            "Peer {peer_id} failed to set up: {err_msg}"
                        )));
                    }
                }
                MeshMessage::Error {
                    session_id,
                    message,
                } if session_id == Some(self.plan.session_id) => {
                    return Err(MeshError::Pipeline(format!(
                        "Peer {peer_id} error: {message}"
                    )));
                }
                _ => {
                    // Ignore unexpected messages during setup
                    debug!("Ignoring unexpected message from {peer_id} during setup");
                }
            }
        }

        info!("All {} workers ready", expected);
        Ok(())
    }

    /// Run the distributed pipeline, yielding tokens through the returned channel.
    ///
    /// Orchestrates auto-regressive token generation by:
    /// 1. Embedding the prompt tokens via the local `pipeline_handler`.
    /// 2. Sending the resulting activation tensor to the first worker.
    /// 3. Waiting for logits from the final worker, sampling the next token,
    ///    embedding it, and feeding the activation back into the pipeline.
    /// 4. Repeating until EOS or `max_tokens` is reached.
    pub fn stream_tokens(
        self: Arc<Self>,
        prompt_tokens: Vec<u32>,
        max_tokens: u32,
        _temperature: f32,
        _top_p: f32,
        pipeline_handler: Arc<dyn MeshPipelineHandler>,
    ) -> mpsc::Receiver<Result<Token>> {
        let (tx, rx) = mpsc::channel(32);
        let session_id = self.plan.session_id;

        // Determine the first worker in the pipeline.
        let first_worker = match self.plan.assignments.first() {
            Some(a) => a.peer_id.clone(),
            None => {
                let tx = tx.clone();
                tokio::spawn(async move {
                    let _ = tx
                        .send(Err(MeshError::Pipeline("No peers in plan".into())))
                        .await;
                });
                return rx;
            }
        };

        tokio::spawn(async move {
            // ------------------------------------------------------------------
            // Phase 1: Process prompt tokens
            // ------------------------------------------------------------------
            // Encode the prompt token ids as raw little-endian bytes and pass
            // them through the local pipeline handler (embedding + any layers
            // the initiator owns).
            let embed_result = pipeline_handler
                .forward_layers(
                    prompt_tokens.iter().flat_map(|t| t.to_le_bytes()).collect(),
                    vec![prompt_tokens.len(), 1], // shape: [seq_len, 1]
                    0,                            // dtype marker for "prompt tokens"
                    0,                            // start position
                )
                .await;

            let activation = match embed_result {
                Ok(a) => a,
                Err(e) => {
                    let _ = tx
                        .send(Err(MeshError::Pipeline(format!("Embedding failed: {e}"))))
                        .await;
                    return;
                }
            };

            // Send the initial activation through the pipeline.
            let msg = MeshMessage::ActivationTensor {
                session_id,
                token_position: 0,
                data: Bytes::from(activation.0.clone()),
                shape: activation.1.clone(),
                dtype: crate::transport::protocol::TensorDtype::F16,
            };
            if let Err(e) = self.transport.send(&first_worker, msg).await {
                let _ = tx.send(Err(e)).await;
                return;
            }

            // ------------------------------------------------------------------
            // Phase 2: Auto-regressive generation loop
            // ------------------------------------------------------------------
            for position in 0..max_tokens {
                // Wait for logits from the final pipeline worker.
                let logits_data = loop {
                    match self.transport.recv().await {
                        Ok((
                            _,
                            MeshMessage::Logits {
                                session_id: sid,
                                data,
                                ..
                            },
                        )) if sid == session_id => {
                            break data;
                        }
                        Ok((_, MeshMessage::Error { message, .. })) => {
                            let _ = tx.send(Err(MeshError::Pipeline(message))).await;
                            // Teardown on error.
                            for a in &self.plan.assignments {
                                let _ = self
                                    .transport
                                    .send(&a.peer_id, MeshMessage::ReleaseSession { session_id })
                                    .await;
                            }
                            return;
                        }
                        Err(e) => {
                            let _ = tx.send(Err(e)).await;
                            return;
                        }
                        _ => continue, // Ignore unrelated messages.
                    }
                };

                // Sample the next token from the received logits.
                let sample_result = pipeline_handler
                    .forward_layers(
                        logits_data.to_vec(),
                        vec![1], // shape marker for "sample from logits"
                        255,     // special dtype marker meaning "sample"
                        position as usize,
                    )
                    .await;

                let (token_bytes, _token_text_shape, _) = match sample_result {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = tx
                            .send(Err(MeshError::Pipeline(format!("Sampling failed: {e}"))))
                            .await;
                        break;
                    }
                };

                // Decode the token: first 4 bytes are the token id (u32 LE),
                // the rest is the UTF-8 text.
                let token_id = if token_bytes.len() >= 4 {
                    u32::from_le_bytes(token_bytes[..4].try_into().unwrap_or([0; 4]))
                } else {
                    0
                };
                let token_text = String::from_utf8_lossy(&token_bytes[4..]).to_string();
                let is_eos = token_text.is_empty() || token_id == 0;

                let token = Token {
                    text: token_text,
                    id: token_id,
                    logprob: None,
                    is_special: is_eos,
                };

                if tx.send(Ok(token)).await.is_err() {
                    break; // Receiver dropped.
                }
                if is_eos {
                    break;
                }

                // Embed the newly generated token and send the activation back
                // into the pipeline for the next position.
                let next_activation = pipeline_handler
                    .forward_layers(
                        token_id.to_le_bytes().to_vec(),
                        vec![1, 1], // shape: [1, 1]
                        0,          // dtype marker for single token
                        (position + 1) as usize,
                    )
                    .await;

                let activation = match next_activation {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = tx
                            .send(Err(MeshError::Pipeline(format!(
                                "Next embedding failed: {e}"
                            ))))
                            .await;
                        break;
                    }
                };

                // Save checkpoint periodically for recovery
                if (position + 1) % CHECKPOINT_INTERVAL == 0 {
                    self.checkpoints.save(
                        session_id,
                        position + 1,
                        Bytes::from(activation.0.clone()),
                        activation.1.clone(),
                        crate::transport::protocol::TensorDtype::F16,
                        0, // source layer (initiator-side)
                    );
                    debug!("Saved checkpoint at token position {}", position + 1);
                }

                let msg = MeshMessage::ActivationTensor {
                    session_id,
                    token_position: position + 1,
                    data: Bytes::from(activation.0),
                    shape: activation.1,
                    dtype: crate::transport::protocol::TensorDtype::F16,
                };
                if let Err(e) = self.transport.send(&first_worker, msg).await {
                    let _ = tx.send(Err(e)).await;
                    break;
                }
            }

            // Clean up checkpoints for this session.
            self.checkpoints.clear_session(&session_id);

            // Teardown: release session on all workers.
            for assignment in &self.plan.assignments {
                let _ = self
                    .transport
                    .send(
                        &assignment.peer_id,
                        MeshMessage::ReleaseSession { session_id },
                    )
                    .await;
            }
        });

        rx
    }

    /// Stream tokens using full-model replication.
    ///
    /// Instead of distributing layers, this sends the entire inference
    /// request to a single peer that has the full model loaded. The peer
    /// streams tokens back via `InferenceToken` messages.
    pub fn stream_tokens_replicated(
        self: Arc<Self>,
        model_id: String,
        messages_json: String,
        max_tokens: u32,
        temperature: f32,
        top_p: f32,
    ) -> mpsc::Receiver<Result<Token>> {
        let (tx, rx) = mpsc::channel(32);
        let session_id = Uuid::new_v4();

        // Pick the first peer (scheduler already ranked by capability)
        let target_peer = match self.plan.assignments.first() {
            Some(a) => a.peer_id.clone(),
            None => {
                let tx = tx.clone();
                tokio::spawn(async move {
                    let _ = tx
                        .send(Err(MeshError::Pipeline("No peers in plan".into())))
                        .await;
                });
                return rx;
            }
        };

        tokio::spawn(async move {
            info!("Sending inference request to peer {target_peer} (session {session_id})");

            // Send inference request
            let req = MeshMessage::InferenceRequest {
                session_id,
                model_id,
                messages_json,
                max_tokens,
                temperature,
                top_p,
            };

            if let Err(e) = self.transport.send(&target_peer, req).await {
                let _ = tx
                    .send(Err(MeshError::Transport(format!(
                        "Failed to send inference request: {e}"
                    ))))
                    .await;
                return;
            }

            // Receive streamed tokens
            loop {
                match self.transport.recv().await {
                    Ok((
                        peer_id,
                        MeshMessage::InferenceToken {
                            session_id: sid,
                            text,
                            token_id,
                            is_done,
                        },
                    )) if sid == session_id => {
                        debug!("Token from {peer_id}: {text:?}");
                        let token = Token {
                            text,
                            id: token_id,
                            logprob: None,
                            is_special: is_done,
                        };
                        if tx.send(Ok(token)).await.is_err() {
                            break; // Receiver dropped
                        }
                        if is_done {
                            break;
                        }
                    }
                    Ok((
                        _,
                        MeshMessage::InferenceComplete {
                            session_id: sid,
                            error: Some(err),
                            ..
                        },
                    )) if sid == session_id => {
                        let _ = tx.send(Err(MeshError::Pipeline(err))).await;
                        break;
                    }
                    Ok((
                        _,
                        MeshMessage::InferenceComplete {
                            session_id: sid, ..
                        },
                    )) if sid == session_id => {
                        break; // Done
                    }
                    Ok((_, MeshMessage::Error { message, .. })) => {
                        let _ = tx.send(Err(MeshError::Pipeline(message))).await;
                        break;
                    }
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        break;
                    }
                    _ => continue, // Ignore unrelated messages
                }
            }

            // Cleanup
            let _ = self
                .transport
                .send(&target_peer, MeshMessage::ReleaseSession { session_id })
                .await;
        });

        rx
    }
}
