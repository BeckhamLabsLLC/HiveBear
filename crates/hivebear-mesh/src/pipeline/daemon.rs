//! Mesh worker daemon: listens for inference requests and serves them
//! using a caller-provided `MeshInferenceHandler`.

use std::sync::Arc;

use futures::StreamExt;
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::peer::NodeId;
use crate::protocol::{MeshInferenceHandler, MeshPipelineHandler};
use crate::transport::protocol::MeshMessage;
use crate::transport::MeshTransport;

/// A background daemon that listens on the mesh transport for incoming
/// `InferenceRequest` messages, runs inference via the supplied handler,
/// and streams tokens back to the requesting peer.
///
/// Optionally also handles pipeline-parallel messages (`AssignLayers`,
/// `ActivationTensor`) when a `MeshPipelineHandler` is provided.
pub struct MeshWorkerDaemon<H: MeshInferenceHandler> {
    handler: Arc<H>,
    transport: Arc<dyn MeshTransport>,
    pipeline_handler: Option<Arc<dyn MeshPipelineHandler>>,
    /// Routing and ownership for pipeline sessions assigned to this node.
    active_pipeline_session: dashmap::DashMap<Uuid, PipelineSession>,
}

/// What this worker needs to know to place itself in a pipeline.
#[derive(Debug, Clone)]
struct PipelineSession {
    /// Peer that assigned these layers; only they may release the session.
    owner: Vec<u8>,
    /// Next stage, or `None` when this worker is the final stage.
    next_peer: Option<NodeId>,
    /// Peer that receives the final logits.
    initiator_peer: NodeId,
}

impl<H: MeshInferenceHandler + 'static> MeshWorkerDaemon<H> {
    pub fn new(handler: Arc<H>, transport: Arc<dyn MeshTransport>) -> Self {
        Self {
            handler,
            transport,
            pipeline_handler: None,
            active_pipeline_session: dashmap::DashMap::new(),
        }
    }

    /// Create a daemon with pipeline-parallel support.
    pub fn with_pipeline(
        handler: Arc<H>,
        transport: Arc<dyn MeshTransport>,
        pipeline_handler: Arc<dyn MeshPipelineHandler>,
    ) -> Self {
        Self {
            handler,
            transport,
            pipeline_handler: Some(pipeline_handler),
            active_pipeline_session: dashmap::DashMap::new(),
        }
    }

    /// Run the daemon loop. This will block (async) until the transport
    /// is shut down or an unrecoverable error occurs.
    pub async fn run(&self) {
        info!("MeshWorkerDaemon: listening for inference requests");

        loop {
            let (peer_id, msg) = match self.transport.recv().await {
                Ok(pair) => pair,
                Err(e) => {
                    error!("MeshWorkerDaemon: transport recv error: {e}");
                    break;
                }
            };

            match msg {
                MeshMessage::InferenceRequest {
                    session_id,
                    model_id,
                    messages_json,
                    max_tokens,
                    temperature,
                    top_p: _,
                } => {
                    info!(
                        "MeshWorkerDaemon: inference request from {peer_id} \
                         (session {session_id}, model '{model_id}')"
                    );
                    let handler = self.handler.clone();
                    let transport = self.transport.clone();
                    tokio::spawn(async move {
                        Self::handle_request(
                            handler,
                            transport,
                            peer_id,
                            session_id,
                            model_id,
                            messages_json,
                            max_tokens,
                            temperature,
                        )
                        .await;
                    });
                }
                MeshMessage::Ping { timestamp_ms } => {
                    let _ = self
                        .transport
                        .send(&peer_id, MeshMessage::Pong { timestamp_ms })
                        .await;
                }
                MeshMessage::ReleaseSession { session_id } => {
                    // Only the peer that opened the session may close it.
                    let entry = self.active_pipeline_session.get(&session_id);
                    let owned_by_sender = entry
                        .as_ref()
                        .is_some_and(|e| e.value().owner == peer_id.0.to_bytes().to_vec());
                    drop(entry);

                    if !owned_by_sender {
                        warn!(
                            "MeshWorkerDaemon: ignoring ReleaseSession for {session_id} from \
                             {peer_id}, which does not own it"
                        );
                        continue;
                    }

                    info!("MeshWorkerDaemon: session {session_id} released by {peer_id}");
                    self.active_pipeline_session.remove(&session_id);
                    if let Some(ref ph) = self.pipeline_handler {
                        let ph = ph.clone();
                        tokio::spawn(async move {
                            let _ = ph.unload_layers().await;
                        });
                    }
                }
                MeshMessage::AssignLayers {
                    session_id,
                    model_id: _,
                    layer_range,
                    total_layers,
                    model_source,
                    next_peer,
                    initiator_peer,
                } => {
                    if let Some(ref ph) = self.pipeline_handler {
                        info!(
                            "MeshWorkerDaemon: layer assignment from {peer_id}: \
                             layers {}..{} of {total_layers} (session {session_id})",
                            layer_range.start, layer_range.end
                        );
                        // Record where this stage's output goes. The daemon
                        // used to discard next_peer and initiator_peer here,
                        // which is why activations went nowhere useful.
                        self.active_pipeline_session.insert(
                            session_id,
                            PipelineSession {
                                owner: peer_id.0.to_bytes().to_vec(),
                                next_peer,
                                initiator_peer,
                            },
                        );
                        let ph = ph.clone();
                        let transport = self.transport.clone();
                        tokio::spawn(async move {
                            let result = ph
                                .load_layers(&model_source, layer_range, total_layers)
                                .await;
                            let ack = MeshMessage::AssignLayersAck {
                                session_id,
                                ready: result.is_ok(),
                                error: result.err(),
                            };
                            let _ = transport.send(&peer_id, ack).await;
                        });
                    } else {
                        warn!(
                            "MeshWorkerDaemon: received AssignLayers but no pipeline handler set"
                        );
                    }
                }
                MeshMessage::ActivationTensor {
                    session_id,
                    token_position,
                    data,
                    shape,
                    dtype,
                } => {
                    // Where does this stage's output go? Without a recorded
                    // session we cannot know, and echoing it back to the
                    // sender (the old behaviour) deadlocks the pipeline: the
                    // initiator waits for Logits that never come.
                    let Some(route) = self
                        .active_pipeline_session
                        .get(&session_id)
                        .map(|e| e.value().clone())
                    else {
                        warn!(
                            "MeshWorkerDaemon: activation for unknown session {session_id} \
                             from {peer_id}; dropping"
                        );
                        continue;
                    };

                    if let Some(ref ph) = self.pipeline_handler {
                        let ph = ph.clone();
                        let transport = self.transport.clone();
                        let dtype_u8 = match dtype {
                            crate::transport::protocol::TensorDtype::F32 => 0u8,
                            crate::transport::protocol::TensorDtype::F16 => 1u8,
                            crate::transport::protocol::TensorDtype::BF16 => 2u8,
                        };
                        tokio::spawn(async move {
                            match ph
                                .forward_layers(
                                    data.to_vec(),
                                    shape.clone(),
                                    dtype_u8,
                                    token_position as usize,
                                )
                                .await
                            {
                                Ok((out_data, out_shape, out_dtype)) => {
                                    let out_tensor_dtype = match out_dtype {
                                        1 => crate::transport::protocol::TensorDtype::F16,
                                        2 => crate::transport::protocol::TensorDtype::BF16,
                                        _ => crate::transport::protocol::TensorDtype::F32,
                                    };

                                    let (target, msg) = match route.next_peer {
                                        // Intermediate stage: hand the
                                        // activation to the next worker.
                                        Some(next) => (
                                            next,
                                            MeshMessage::ActivationTensor {
                                                session_id,
                                                token_position,
                                                data: bytes::Bytes::from(out_data),
                                                shape: out_shape,
                                                dtype: out_tensor_dtype,
                                            },
                                        ),
                                        // Final stage: the output *is* the
                                        // logits, and they go to the
                                        // initiator, not back up the chain.
                                        None => {
                                            let vocab_size =
                                                out_shape.last().copied().unwrap_or(0) as u32;
                                            (
                                                route.initiator_peer,
                                                MeshMessage::Logits {
                                                    session_id,
                                                    token_position,
                                                    data: bytes::Bytes::from(out_data),
                                                    vocab_size,
                                                },
                                            )
                                        }
                                    };

                                    if let Err(e) = transport.send(&target, msg).await {
                                        error!(
                                            "MeshWorkerDaemon: could not forward stage output \
                                             for session {session_id} to {target}: {e}"
                                        );
                                    }
                                }
                                Err(e) => {
                                    error!("MeshWorkerDaemon: forward_layers failed: {e}");
                                    let err_msg = MeshMessage::Error {
                                        session_id: Some(session_id),
                                        message: format!("forward_layers failed: {e}"),
                                    };
                                    // Tell the initiator rather than leaving
                                    // it blocked on a tensor that will never
                                    // arrive.
                                    let _ = transport.send(&route.initiator_peer, err_msg).await;
                                }
                            }
                        });
                    }
                }
                _ => {
                    // Ignore messages we don't handle (Hello, etc.)
                }
            }
        }
    }

    /// Serve a single inference request: call the handler, stream tokens
    /// back through the transport, and send an `InferenceComplete` at the end.
    #[allow(clippy::too_many_arguments)]
    async fn handle_request(
        handler: Arc<H>,
        transport: Arc<dyn MeshTransport>,
        peer_id: NodeId,
        session_id: Uuid,
        model_id: String,
        messages_json: String,
        max_tokens: u32,
        temperature: f32,
    ) {
        let stream_result = handler
            .handle_inference(&model_id, &messages_json, max_tokens, temperature)
            .await;

        let mut token_stream = match stream_result {
            Ok(s) => s,
            Err(err) => {
                warn!("MeshWorkerDaemon: handler error for session {session_id}: {err}");
                let _ = transport
                    .send(
                        &peer_id,
                        MeshMessage::InferenceComplete {
                            session_id,
                            full_text: String::new(),
                            tokens_generated: 0,
                            error: Some(err),
                        },
                    )
                    .await;
                return;
            }
        };

        let mut full_text = String::new();
        let mut token_count: u32 = 0;

        while let Some(result) = token_stream.next().await {
            match result {
                Ok(text) => {
                    full_text.push_str(&text);
                    token_count += 1;

                    let msg = MeshMessage::InferenceToken {
                        session_id,
                        text,
                        token_id: token_count,
                        is_done: false,
                    };
                    if let Err(e) = transport.send(&peer_id, msg).await {
                        warn!("MeshWorkerDaemon: failed to send token to {peer_id}: {e}");
                        return;
                    }
                }
                Err(err) => {
                    let _ = transport
                        .send(
                            &peer_id,
                            MeshMessage::InferenceComplete {
                                session_id,
                                full_text,
                                tokens_generated: token_count,
                                error: Some(err),
                            },
                        )
                        .await;
                    return;
                }
            }
        }

        // Send the final "done" token so the initiator knows streaming is over
        let _ = transport
            .send(
                &peer_id,
                MeshMessage::InferenceToken {
                    session_id,
                    text: String::new(),
                    token_id: token_count,
                    is_done: true,
                },
            )
            .await;

        // Then a complete message for good measure
        let _ = transport
            .send(
                &peer_id,
                MeshMessage::InferenceComplete {
                    session_id,
                    full_text,
                    tokens_generated: token_count,
                    error: None,
                },
            )
            .await;

        info!("MeshWorkerDaemon: completed session {session_id} — {token_count} tokens");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::mock::{MockRegistry, MockTransport};
    use crate::transport::protocol::TensorDtype;
    use futures::stream;
    use std::pin::Pin;
    use std::time::Duration;

    /// Passes activations through unchanged so the test observes routing,
    /// not arithmetic.
    struct EchoPipeline;

    #[async_trait::async_trait]
    impl MeshPipelineHandler for EchoPipeline {
        async fn load_layers(
            &self,
            _model_source: &str,
            _layer_range: std::ops::Range<u32>,
            _total_layers: u32,
        ) -> std::result::Result<(), String> {
            Ok(())
        }

        async fn forward_layers(
            &self,
            activation_data: Vec<u8>,
            shape: Vec<usize>,
            dtype: u8,
            _index_pos: usize,
        ) -> std::result::Result<(Vec<u8>, Vec<usize>, u8), String> {
            Ok((activation_data, shape, dtype))
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
            _model_id: &str,
            _messages_json: &str,
            _max_tokens: u32,
            _temperature: f32,
        ) -> std::result::Result<
            Pin<Box<dyn futures::Stream<Item = std::result::Result<String, String>> + Send>>,
            String,
        > {
            Ok(Box::pin(stream::empty()))
        }
    }

    /// Wire three mock transports into a line: initiator -> mid -> last.
    fn link(a: &MockTransport, b: &MockTransport) {
        a.connect_for_test(b);
    }

    async fn recv_soon(t: &MockTransport) -> Option<(NodeId, MeshMessage)> {
        tokio::time::timeout(Duration::from_secs(2), t.recv())
            .await
            .ok()
            .and_then(|r| r.ok())
    }

    /// The regression: the daemon discarded next_peer and initiator_peer and
    /// sent every stage's output straight back to whoever sent it, never
    /// forwarding along the pipeline and never emitting Logits. The
    /// initiator then blocked forever waiting for logits that could not
    /// arrive.
    #[tokio::test]
    async fn middle_stage_forwards_and_last_stage_returns_logits() {
        let registry = MockRegistry::new();
        let initiator = MockTransport::new(NodeId::generate().0, registry.clone());
        let mid = Arc::new(MockTransport::new(NodeId::generate().0, registry.clone()));
        let last = Arc::new(MockTransport::new(NodeId::generate().0, registry.clone()));

        // Everyone needs a route to everyone they will talk to.
        link(&initiator, &mid);
        link(&initiator, &last);
        link(&mid, &last);
        link(&mid, &initiator);
        link(&last, &initiator);

        let mid_daemon = Arc::new(MeshWorkerDaemon::with_pipeline(
            Arc::new(NoInference),
            mid.clone(),
            Arc::new(EchoPipeline),
        ));
        let last_daemon = Arc::new(MeshWorkerDaemon::with_pipeline(
            Arc::new(NoInference),
            last.clone(),
            Arc::new(EchoPipeline),
        ));
        let m = mid_daemon.clone();
        tokio::spawn(async move { m.run().await });
        let l = last_daemon.clone();
        tokio::spawn(async move { l.run().await });

        let session_id = Uuid::new_v4();
        let initiator_id = initiator.local_id().clone();

        // Stage 1 forwards to stage 2; stage 2 is final.
        initiator
            .send(
                mid.local_id(),
                MeshMessage::AssignLayers {
                    session_id,
                    model_id: "m".into(),
                    layer_range: 0..8,
                    total_layers: 16,
                    model_source: "m".into(),
                    next_peer: Some(last.local_id().clone()),
                    initiator_peer: initiator_id.clone(),
                },
            )
            .await
            .unwrap();
        initiator
            .send(
                last.local_id(),
                MeshMessage::AssignLayers {
                    session_id,
                    model_id: "m".into(),
                    layer_range: 8..16,
                    total_layers: 16,
                    model_source: "m".into(),
                    next_peer: None,
                    initiator_peer: initiator_id.clone(),
                },
            )
            .await
            .unwrap();

        // Drain both acks.
        recv_soon(&initiator).await.expect("ack 1");
        recv_soon(&initiator).await.expect("ack 2");

        // Kick the pipeline.
        initiator
            .send(
                mid.local_id(),
                MeshMessage::ActivationTensor {
                    session_id,
                    token_position: 0,
                    data: bytes::Bytes::from_static(&[1, 2, 3, 4]),
                    shape: vec![1, 4],
                    dtype: TensorDtype::F32,
                },
            )
            .await
            .unwrap();

        // The initiator must receive Logits — not an ActivationTensor bounced
        // straight back from the middle stage.
        let (from, msg) = recv_soon(&initiator)
            .await
            .expect("initiator should receive the pipeline result");
        match msg {
            MeshMessage::Logits {
                session_id: sid,
                vocab_size,
                ..
            } => {
                assert_eq!(sid, session_id);
                assert_eq!(vocab_size, 4, "vocab size comes from the last shape dim");
                assert_eq!(
                    from.0.to_bytes(),
                    last.local_id().0.to_bytes(),
                    "logits must come from the final stage, not the middle one"
                );
            }
            other => panic!("expected Logits from the final stage, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn release_session_from_a_stranger_is_ignored() {
        let registry = MockRegistry::new();
        let owner = MockTransport::new(NodeId::generate().0, registry.clone());
        let stranger = MockTransport::new(NodeId::generate().0, registry.clone());
        let worker = Arc::new(MockTransport::new(NodeId::generate().0, registry.clone()));

        link(&owner, &worker);
        link(&stranger, &worker);
        link(&worker, &owner);

        let daemon = Arc::new(MeshWorkerDaemon::with_pipeline(
            Arc::new(NoInference),
            worker.clone(),
            Arc::new(EchoPipeline),
        ));
        let d = daemon.clone();
        tokio::spawn(async move { d.run().await });

        let session_id = Uuid::new_v4();
        owner
            .send(
                worker.local_id(),
                MeshMessage::AssignLayers {
                    session_id,
                    model_id: "m".into(),
                    layer_range: 0..4,
                    total_layers: 4,
                    model_source: "m".into(),
                    next_peer: None,
                    initiator_peer: owner.local_id().clone(),
                },
            )
            .await
            .unwrap();
        recv_soon(&owner).await.expect("ack");

        // A peer that does not own the session tries to tear it down.
        stranger
            .send(
                worker.local_id(),
                MeshMessage::ReleaseSession { session_id },
            )
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;

        assert_eq!(
            daemon.active_pipeline_session.len(),
            1,
            "a stranger must not be able to release someone else's session"
        );

        // The real owner can.
        owner
            .send(
                worker.local_id(),
                MeshMessage::ReleaseSession { session_id },
            )
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(daemon.active_pipeline_session.len(), 0);
    }
}
