//! Mesh worker daemon: listens for inference requests and serves them
//! using a caller-provided `MeshInferenceHandler`.

use std::sync::Arc;

use futures::StreamExt;
use tracing::{error, info, warn};
use uuid::Uuid;

use crate::peer::NodeId;
use crate::protocol::MeshInferenceHandler;
use crate::transport::protocol::MeshMessage;
use crate::transport::MeshTransport;

/// A background daemon that listens on the mesh transport for incoming
/// `InferenceRequest` messages, runs inference via the supplied handler,
/// and streams tokens back to the requesting peer.
pub struct MeshWorkerDaemon<H: MeshInferenceHandler> {
    handler: Arc<H>,
    transport: Arc<dyn MeshTransport>,
}

impl<H: MeshInferenceHandler + 'static> MeshWorkerDaemon<H> {
    pub fn new(handler: Arc<H>, transport: Arc<dyn MeshTransport>) -> Self {
        Self { handler, transport }
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
                }
                _ => {
                    // Ignore messages we don't handle (Hello, ActivationTensor, etc.)
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
