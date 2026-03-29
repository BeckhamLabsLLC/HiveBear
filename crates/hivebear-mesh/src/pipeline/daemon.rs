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
}

impl<H: MeshInferenceHandler + 'static> MeshWorkerDaemon<H> {
    pub fn new(handler: Arc<H>, transport: Arc<dyn MeshTransport>) -> Self {
        Self {
            handler,
            transport,
            pipeline_handler: None,
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
                    info!("MeshWorkerDaemon: session {session_id} released by {peer_id}");
                    // Also unload pipeline layers if any were loaded
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
                    next_peer: _,
                    initiator_peer: _,
                } => {
                    if let Some(ref ph) = self.pipeline_handler {
                        info!(
                            "MeshWorkerDaemon: layer assignment from {peer_id}: \
                             layers {}..{} of {total_layers} (session {session_id})",
                            layer_range.start, layer_range.end
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
                                    // Send output back to the initiator (peer who sent it)
                                    let msg = MeshMessage::ActivationTensor {
                                        session_id,
                                        token_position,
                                        data: bytes::Bytes::from(out_data),
                                        shape: out_shape,
                                        dtype: out_tensor_dtype,
                                    };
                                    let _ = transport.send(&peer_id, msg).await;
                                }
                                Err(e) => {
                                    error!("MeshWorkerDaemon: forward_layers failed: {e}");
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
