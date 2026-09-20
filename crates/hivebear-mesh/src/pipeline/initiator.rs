use std::sync::Arc;

use bytes::Bytes;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::error::{MeshError, Result};
use crate::peer::NodeId;
use crate::pipeline::checkpoint::CheckpointStore;
use crate::protocol::MeshPipelineHandler;
use crate::scheduler::plan::InferencePlan;
use crate::transport::protocol::MeshMessage;
use crate::transport::MeshTransport;
use crate::trust::{ReputationManager, TrustVerifier};
use hivebear_inference::Token;

/// How long to wait for the next message in a replication session before
/// declaring the peer dead. Generous, because the first token can be behind a
/// cold model load on the peer's side — but finite, because the previous
/// behaviour was to block forever with no output.
const REPLICATION_IDLE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// How long to wait for every worker to acknowledge a layer assignment.
/// Covers a cold model load on the slowest peer.
const SETUP_ACK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);

/// How long to wait for one token's logits to come back round the pipeline.
const PIPELINE_STAGE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// Releases a session subscription when it goes out of scope, including on
/// the early-return paths.
struct SessionGuard {
    transport: Arc<dyn MeshTransport>,
    session_id: Uuid,
}

impl SessionGuard {
    fn new(transport: Arc<dyn MeshTransport>, session_id: Uuid) -> Self {
        Self {
            transport,
            session_id,
        }
    }
}

impl Drop for SessionGuard {
    fn drop(&mut self) {
        self.transport.unsubscribe_session(&self.session_id);
    }
}

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
    /// Optional trust verifier for probabilistic output verification.
    verifier: Option<Arc<TrustVerifier>>,
    /// Optional reputation manager for recording verification results.
    reputation: Option<Arc<tokio::sync::Mutex<ReputationManager>>>,
}

impl PipelineInitiator {
    pub fn new(transport: Arc<dyn MeshTransport>, plan: InferencePlan, local_id: NodeId) -> Self {
        Self {
            transport,
            plan,
            local_id,
            checkpoints: Arc::new(CheckpointStore::new(20)),
            verifier: None,
            reputation: None,
        }
    }

    /// Attach trust verification to this initiator.
    ///
    /// When set, the initiator will probabilistically verify peer outputs
    /// and update reputation scores. Peers that fall below the ban threshold
    /// will be disconnected.
    pub fn with_trust(
        mut self,
        verifier: Arc<TrustVerifier>,
        reputation: Arc<tokio::sync::Mutex<ReputationManager>>,
    ) -> Self {
        self.verifier = Some(verifier);
        self.reputation = Some(reputation);
        self
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
            verifier: None,
            reputation: None,
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

        // Wait for all workers to acknowledge.
        //
        // Subscribed rather than sharing the general queue: acks used to be
        // taken by whatever else was calling recv(), and the `_ =>` arm below
        // silently discarded anything that arrived here instead. Bounded,
        // because a worker that never acks used to hang setup forever.
        let mut acks_received = 0;
        let expected = self.plan.peer_count();
        let mut session_rx = self.transport.subscribe_session(self.plan.session_id);
        let _guard = SessionGuard::new(self.transport.clone(), self.plan.session_id);

        while acks_received < expected {
            let (peer_id, msg) =
                match tokio::time::timeout(SETUP_ACK_TIMEOUT, session_rx.recv()).await {
                    Ok(Some(pair)) => pair,
                    Ok(None) => {
                        return Err(MeshError::Pipeline(
                            "Session queue closed while waiting for worker acknowledgements".into(),
                        ))
                    }
                    Err(_) => {
                        return Err(MeshError::Pipeline(format!(
                            "Only {acks_received} of {expected} workers acknowledged within {}s",
                            SETUP_ACK_TIMEOUT.as_secs()
                        )))
                    }
                };
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

    /// Tell every assigned worker to drop this session's resources.
    ///
    /// Without this, each mesh load left a model resident on every peer for
    /// the rest of their process lifetime — `MeshBackend::unload` was a
    /// no-op, so nothing ever released them.
    pub async fn teardown(&self) {
        for assignment in &self.plan.assignments {
            let msg = MeshMessage::ReleaseSession {
                session_id: self.plan.session_id,
            };
            if let Err(e) = self.transport.send(&assignment.peer_id, msg).await {
                debug!(
                    "Could not release session {} on peer {}: {e}",
                    self.plan.session_id, assignment.peer_id
                );
            }
        }
    }

    /// Session this initiator's plan runs under.
    pub fn session_id(&self) -> Uuid {
        self.plan.session_id
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
        // These were `_temperature` / `_top_p`: the sampling call went
        // through forward_layers, which has nowhere to put them, so the
        // caller's sampling settings were silently discarded.
        temperature: f32,
        top_p: f32,
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

        // Claim the session before anything is sent, so logits cannot be
        // taken off the shared queue by another task.
        let mut session_rx = self.transport.subscribe_session(session_id);

        tokio::spawn(async move {
            let _guard = SessionGuard::new(self.transport.clone(), session_id);
            // ------------------------------------------------------------------
            // Phase 1: Process prompt tokens
            // ------------------------------------------------------------------
            // Encode the prompt token ids as raw little-endian bytes and pass
            // them through the local pipeline handler (embedding + any layers
            // the initiator owns).
            let embed_result = pipeline_handler.embed_prompt(&prompt_tokens).await;

            let activation = match embed_result {
                Ok(a) => a,
                Err(e) => {
                    let _ = tx
                        .send(Err(MeshError::Pipeline(format!("Embedding failed: {e}"))))
                        .await;
                    return;
                }
            };

            // Keep the first activation: for a single-stage pipeline the
            // initiator observes both this input and the logits that come
            // back, which is the one place it holds a trustworthy reference
            // to re-challenge the worker against.
            let spot_check_input = crate::trust::verification::ActivationBytes {
                data: activation.0.clone(),
                shape: activation.1.clone(),
                dtype: crate::transport::protocol::TensorDtype::from_u8(activation.2)
                    .unwrap_or(crate::transport::protocol::TensorDtype::F32),
            };

            // Send the initial activation through the pipeline.
            // The dtype MUST come from what the engine actually produced
            // (`activation.2`); hardcoding it mislabels the tensor and the receiving
            // stage then decodes it at the wrong element width.
            let msg = MeshMessage::ActivationTensor {
                session_id,
                token_position: 0,
                data: Bytes::from(activation.0.clone()),
                shape: activation.1.clone(),
                dtype: crate::transport::protocol::TensorDtype::from_u8(activation.2)
                    .unwrap_or(crate::transport::protocol::TensorDtype::F32),
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
                let (logits_data, vocab_size) = loop {
                    let next =
                        tokio::time::timeout(PIPELINE_STAGE_TIMEOUT, session_rx.recv()).await;
                    match next {
                        // A stage that stops responding used to hang this
                        // loop forever with no output.
                        Err(_) => {
                            let _ = tx
                                .send(Err(MeshError::Pipeline(format!(
                                    "No logits within {}s; a pipeline stage is unresponsive",
                                    PIPELINE_STAGE_TIMEOUT.as_secs()
                                ))))
                                .await;
                            self.teardown().await;
                            return;
                        }
                        Ok(None) => {
                            let _ = tx
                                .send(Err(MeshError::Transport(
                                    "Session queue closed mid-generation".into(),
                                )))
                                .await;
                            return;
                        }
                        Ok(Some((
                            _,
                            MeshMessage::Logits {
                                session_id: sid,
                                data,
                                vocab_size,
                                ..
                            },
                        ))) if sid == session_id => {
                            break (data, vocab_size);
                        }
                        Ok(Some((_, MeshMessage::Error { message, .. }))) => {
                            let _ = tx.send(Err(MeshError::Pipeline(message))).await;
                            self.teardown().await;
                            return;
                        }
                        Ok(Some((from, other))) => {
                            debug!("Ignoring {other:?} from {from} while awaiting logits");
                        }
                    }
                };

                // Spot-check the worker, once, on the first token.
                //
                // Only sound with a single stage: then this input and these
                // logits are the same peer's input and output, so re-running
                // the input must reproduce the same hash. With several stages
                // the initiator never sees any individual stage's output and
                // has no reference to compare against, so it does not guess.
                if position == 0 && self.plan.assignments.len() == 1 {
                    if let Some(verifier) = self.verifier.clone() {
                        if verifier.should_verify() {
                            let expected = crate::trust::verification::hash_bytes(&logits_data);
                            let peer = first_worker.clone();
                            let input = spot_check_input.clone();
                            let reputation = self.reputation.clone();
                            // A fresh session id: the generation loop owns
                            // `session_id`'s queue, and re-subscribing would
                            // steal its messages.
                            let check_session = Uuid::new_v4();
                            tokio::spawn(async move {
                                match verifier
                                    .verify_consistent_with(
                                        &peer,
                                        check_session,
                                        0..u32::MAX,
                                        0,
                                        &input,
                                        expected,
                                    )
                                    .await
                                {
                                    Ok(passed) => {
                                        if let Some(rep) = reputation {
                                            rep.lock().await.record_verification(&peer, passed);
                                        }
                                        if !passed {
                                            warn!(
                                                "Spot check failed for {peer}: it did not \
                                                 reproduce its own output"
                                            );
                                        }
                                    }
                                    Err(e) => debug!("Spot check against {peer} failed: {e}"),
                                }
                            });
                        }
                    }
                }

                // Sample the next token from the received logits.
                let sample_result = pipeline_handler
                    .sample_token(
                        logits_data.to_vec(),
                        vec![vocab_size as usize],
                        // Logits come off the final stage as F32.
                        0,
                        temperature,
                        top_p,
                    )
                    .await;

                let (token_id, token_text) = match sample_result {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = tx
                            .send(Err(MeshError::Pipeline(format!("Sampling failed: {e}"))))
                            .await;
                        break;
                    }
                };

                let is_eos = token_text.is_empty();

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
                let next_activation = pipeline_handler.embed_prompt(&[token_id]).await;

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

                // Carry the engine's real dtype, not a hardcoded guess — a checkpoint
                // restored with the wrong dtype resumes the session on garbage.
                let activation_dtype =
                    crate::transport::protocol::TensorDtype::from_u8(activation.2)
                        .unwrap_or(crate::transport::protocol::TensorDtype::F32);

                // Save checkpoint periodically for recovery
                if (position + 1) % CHECKPOINT_INTERVAL == 0 {
                    self.checkpoints.save(
                        session_id,
                        position + 1,
                        Bytes::from(activation.0.clone()),
                        activation.1.clone(),
                        activation_dtype,
                        0, // source layer (initiator-side)
                    );
                    debug!("Saved checkpoint at token position {}", position + 1);
                }

                let msg = MeshMessage::ActivationTensor {
                    session_id,
                    token_position: position + 1,
                    data: Bytes::from(activation.0),
                    shape: activation.1,
                    dtype: activation_dtype,
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

        // Claim this session's inbound traffic *before* sending, so a fast
        // reply cannot land on the general queue and be consumed by another
        // task. Previously this loop shared one queue with the worker daemon
        // and the CLI, so they stole each other's messages and `_ => continue`
        // dropped the stolen ones silently.
        let mut session_rx = self.transport.subscribe_session(session_id);

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
                self.transport.unsubscribe_session(&session_id);
                return;
            }

            // Receive streamed tokens. Everything arriving here belongs to
            // this session, so there is nothing to filter out.
            loop {
                let next = tokio::time::timeout(REPLICATION_IDLE_TIMEOUT, session_rx.recv()).await;

                let (peer_id, msg) = match next {
                    // An unresponsive peer used to hang this loop forever with
                    // no output; the UI just sat there.
                    Err(_) => {
                        let _ = tx
                            .send(Err(MeshError::Pipeline(format!(
                                "Peer {target_peer} sent nothing for {}s; giving up on this session",
                                REPLICATION_IDLE_TIMEOUT.as_secs()
                            ))))
                            .await;
                        break;
                    }
                    Ok(None) => {
                        let _ = tx
                            .send(Err(MeshError::Transport(
                                "Session queue closed before completion".into(),
                            )))
                            .await;
                        break;
                    }
                    Ok(Some(pair)) => pair,
                };

                match (peer_id, msg) {
                    (
                        peer_id,
                        MeshMessage::InferenceToken {
                            session_id: sid,
                            text,
                            token_id,
                            is_done,
                        },
                    ) if sid == session_id => {
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
                    (
                        _,
                        MeshMessage::InferenceComplete {
                            session_id: sid,
                            error: Some(err),
                            ..
                        },
                    ) if sid == session_id => {
                        let _ = tx.send(Err(MeshError::Pipeline(err))).await;
                        break;
                    }
                    (
                        _,
                        MeshMessage::InferenceComplete {
                            session_id: sid, ..
                        },
                    ) if sid == session_id => {
                        break; // Done
                    }
                    (_, MeshMessage::Error { message, .. }) => {
                        let _ = tx.send(Err(MeshError::Pipeline(message))).await;
                        break;
                    }
                    (peer_id, other) => {
                        // Routed to this session but not part of the
                        // replication protocol. Worth knowing about rather
                        // than discarding in silence.
                        debug!("Ignoring {other:?} from {peer_id} on session {session_id}");
                    }
                }
            }

            // Cleanup
            let _ = self
                .transport
                .send(&target_peer, MeshMessage::ReleaseSession { session_id })
                .await;
            self.transport.unsubscribe_session(&session_id);
        });

        rx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::daemon::MeshWorkerDaemon;
    use crate::protocol::{MeshInferenceHandler, MeshPipelineHandler};
    use crate::scheduler::plan::{InferencePlan, LayerAssignment};
    use crate::transport::mock::{MockRegistry, MockTransport};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// Records which trait methods the initiator actually reaches for.
    ///
    /// The old code funnelled embedding and sampling through
    /// `forward_layers` with out-of-band dtype markers (0 and 255), which
    /// `CliPipelineHandler` mapped to F32 like everything else — so it never
    /// embedded or sampled, and the initiator decoded whatever came back as
    /// `[token_id_le_bytes][utf8_text]`. This handler fails the test if
    /// anything routes through forward_layers again.
    struct RecordingHandler {
        embeds: AtomicUsize,
        samples: AtomicUsize,
        forwards: AtomicUsize,
    }

    impl RecordingHandler {
        fn new() -> Self {
            Self {
                embeds: AtomicUsize::new(0),
                samples: AtomicUsize::new(0),
                forwards: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait::async_trait]
    impl MeshPipelineHandler for RecordingHandler {
        async fn load_layers(
            &self,
            _model_source: &str,
            _layer_range: std::ops::Range<u32>,
            _total_layers: u32,
        ) -> std::result::Result<(), String> {
            Ok(())
        }

        async fn embed_prompt(
            &self,
            token_ids: &[u32],
        ) -> std::result::Result<(Vec<u8>, Vec<usize>, u8), String> {
            self.embeds.fetch_add(1, Ordering::SeqCst);
            // One f32 per token, F32 tag.
            let data: Vec<u8> = token_ids
                .iter()
                .flat_map(|t| (*t as f32).to_le_bytes())
                .collect();
            Ok((data, vec![token_ids.len(), 1], 0))
        }

        async fn sample_token(
            &self,
            _logits: Vec<u8>,
            _shape: Vec<usize>,
            _dtype: u8,
            _temperature: f32,
            _top_p: f32,
        ) -> std::result::Result<(u32, String), String> {
            let n = self.samples.fetch_add(1, Ordering::SeqCst);
            // Emit two real tokens, then an empty one to signal EOS.
            if n < 2 {
                Ok((100 + n as u32, format!("tok{n}")))
            } else {
                Ok((0, String::new()))
            }
        }

        async fn forward_layers(
            &self,
            data: Vec<u8>,
            shape: Vec<usize>,
            dtype: u8,
            _index_pos: usize,
        ) -> std::result::Result<(Vec<u8>, Vec<usize>, u8), String> {
            self.forwards.fetch_add(1, Ordering::SeqCst);
            Ok((data, shape, dtype))
        }

        async fn unload_layers(&self) -> std::result::Result<(), String> {
            Ok(())
        }
    }

    /// Worker side: echoes the activation, so the single stage is also the
    /// final stage and returns it as logits.
    struct EchoPipeline;

    #[async_trait::async_trait]
    impl MeshPipelineHandler for EchoPipeline {
        async fn load_layers(
            &self,
            _m: &str,
            _r: std::ops::Range<u32>,
            _t: u32,
        ) -> std::result::Result<(), String> {
            Ok(())
        }
        async fn forward_layers(
            &self,
            data: Vec<u8>,
            shape: Vec<usize>,
            dtype: u8,
            _i: usize,
        ) -> std::result::Result<(Vec<u8>, Vec<usize>, u8), String> {
            Ok((data, shape, dtype))
        }
        async fn unload_layers(&self) -> std::result::Result<(), String> {
            Ok(())
        }
    }

    struct NoInference;

    #[async_trait::async_trait]
    impl MeshInferenceHandler for NoInference {
        async fn handle_inference(
            &self,
            _m: &str,
            _j: &str,
            _mt: u32,
            _t: f32,
        ) -> std::result::Result<
            std::pin::Pin<
                Box<dyn futures::Stream<Item = std::result::Result<String, String>> + Send>,
            >,
            String,
        > {
            Ok(Box::pin(futures::stream::empty()))
        }
    }

    /// End to end over a one-stage pipeline: embed the prompt, ship the
    /// activation to a worker, get logits back, sample, repeat. Before the
    /// routing and protocol fixes this could not complete at all — the worker
    /// echoed activations instead of emitting Logits, and the initiator
    /// blocked forever waiting for them.
    #[tokio::test]
    async fn pipeline_generates_tokens_end_to_end() {
        let registry = MockRegistry::new();
        let initiator_t = Arc::new(MockTransport::new(NodeId::generate().0, registry.clone()));
        let worker_t = Arc::new(MockTransport::new(NodeId::generate().0, registry.clone()));
        initiator_t.connect_for_test(&worker_t);
        worker_t.connect_for_test(&initiator_t);

        let daemon = Arc::new(MeshWorkerDaemon::with_pipeline(
            Arc::new(NoInference),
            worker_t.clone(),
            Arc::new(EchoPipeline),
        ));
        let d = daemon.clone();
        tokio::spawn(async move { d.run().await });

        let plan = InferencePlan {
            session_id: Uuid::new_v4(),
            model_id: "test-model".into(),
            total_layers: 4,
            assignments: vec![LayerAssignment {
                peer_id: worker_t.local_id().clone(),
                layer_range: 0..4,
                estimated_compute_ms: 1.0,
                estimated_transfer_ms: 1.0,
            }],
            estimated_latency_ms: 1.0,
            estimated_throughput_tok_s: 1.0,
        };

        let initiator = Arc::new(PipelineInitiator::new(
            initiator_t.clone(),
            plan,
            initiator_t.local_id().clone(),
        ));

        // The worker must know where to send its output.
        initiator.setup("test-model").await.expect("setup");

        let handler = Arc::new(RecordingHandler::new());
        let mut rx = Arc::clone(&initiator).stream_tokens(
            vec![1, 2, 3],
            8,
            0.7,
            0.9,
            handler.clone() as Arc<dyn MeshPipelineHandler>,
        );

        let mut texts = Vec::new();
        while let Ok(Some(item)) = tokio::time::timeout(Duration::from_secs(5), rx.recv()).await {
            match item {
                Ok(tok) => {
                    if tok.text.is_empty() {
                        break;
                    }
                    texts.push(tok.text);
                }
                Err(e) => panic!("pipeline returned an error: {e}"),
            }
        }

        assert_eq!(
            texts,
            vec!["tok0".to_string(), "tok1".to_string()],
            "the generation loop should complete and yield the sampled tokens"
        );
        assert!(
            handler.embeds.load(Ordering::SeqCst) >= 1,
            "the prompt must go through embed_prompt"
        );
        assert!(
            handler.samples.load(Ordering::SeqCst) >= 2,
            "each token must go through sample_token"
        );
        assert_eq!(
            handler.forwards.load(Ordering::SeqCst),
            0,
            "embedding and sampling must not be smuggled through forward_layers"
        );
    }
}
