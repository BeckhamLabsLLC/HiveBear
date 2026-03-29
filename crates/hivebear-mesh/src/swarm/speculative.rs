use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::error::{MeshError, Result};
use crate::peer::NodeId;
use crate::transport::protocol::MeshMessage;
use crate::transport::MeshTransport;
use hivebear_inference::Token;

/// Configuration for speculative decoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeConfig {
    /// Whether speculative decoding is enabled.
    pub enabled: bool,
    /// Number of draft tokens to generate per speculation round (K).
    pub draft_length: u32,
    /// Override draft model (auto-selected if None).
    pub draft_model: Option<String>,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            draft_length: 5,
            draft_model: None,
        }
    }
}

/// Result of a speculative decoding verification round.
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// How many of the K draft tokens were accepted.
    pub accepted_count: u32,
    /// Correction token sampled at the first rejection point (if any).
    pub correction_token: Option<u32>,
}

/// Orchestrates speculative decoding between a drafter peer and a verifier swarm.
///
/// Flow per round:
/// 1. Drafter generates K candidate tokens using a small draft model
/// 2. Sends `DraftTokens` to the verifier (swarm leader)
/// 3. Verifier batch-verifies all K positions in one forward pass
/// 4. Sends `VerifyDraft` back with accepted count + correction token
/// 5. Accepted tokens are emitted; loop continues from the correction point
pub struct SpeculativeSession {
    transport: Arc<dyn MeshTransport>,
    session_id: Uuid,
    drafter_peer: NodeId,
    verifier_peer: NodeId,
    config: SpeculativeConfig,
}

impl SpeculativeSession {
    pub fn new(
        transport: Arc<dyn MeshTransport>,
        drafter_peer: NodeId,
        verifier_peer: NodeId,
        config: SpeculativeConfig,
    ) -> Self {
        Self {
            transport,
            session_id: Uuid::new_v4(),
            drafter_peer,
            verifier_peer,
            config,
        }
    }

    /// Run the speculative decoding loop, yielding verified tokens.
    ///
    /// The drafter generates candidate tokens and sends them to the verifier.
    /// The verifier batch-checks them and returns how many were accepted.
    /// This method coordinates both sides via mesh messages.
    pub fn stream_tokens(
        self: Arc<Self>,
        draft_model_id: String,
        initial_prompt_tokens: Vec<u32>,
        max_tokens: u32,
    ) -> mpsc::Receiver<Result<Token>> {
        let (tx, rx) = mpsc::channel(32);
        let session_id = self.session_id;
        let draft_length = self.config.draft_length;

        tokio::spawn(async move {
            let mut total_generated = 0u32;
            let mut context_tokens = initial_prompt_tokens;

            info!(
                "Starting speculative decoding session {} (K={}, drafter={}, verifier={})",
                session_id, draft_length, self.drafter_peer, self.verifier_peer
            );

            while total_generated < max_tokens {
                // Phase 1: Request draft tokens from the drafter peer.
                // In a full implementation, the drafter would run its local model
                // and generate K tokens. For now, we send a DraftTokens request
                // and wait for the drafter to respond with candidate tokens.
                //
                // The drafter receives InferenceRequest and responds with DraftTokens.
                let remaining = max_tokens - total_generated;
                let k = draft_length.min(remaining);

                // Send draft request to drafter
                let draft_req = MeshMessage::InferenceRequest {
                    session_id,
                    model_id: draft_model_id.clone(),
                    messages_json: serde_json::to_string(&context_tokens).unwrap_or_default(),
                    max_tokens: k,
                    temperature: 0.0, // Greedy for drafting
                    top_p: 1.0,
                };

                if let Err(e) = self.transport.send(&self.drafter_peer, draft_req).await {
                    let _ = tx.send(Err(e)).await;
                    break;
                }

                // Wait for DraftTokens response
                let draft_tokens = loop {
                    match self.transport.recv().await {
                        Ok((
                            _,
                            MeshMessage::DraftTokens {
                                session_id: sid,
                                tokens,
                                ..
                            },
                        )) if sid == session_id => {
                            break tokens;
                        }
                        Ok((_, MeshMessage::Error { message, .. })) => {
                            let _ = tx
                                .send(Err(MeshError::Pipeline(format!(
                                    "Drafter error: {message}"
                                ))))
                                .await;
                            return;
                        }
                        Err(e) => {
                            let _ = tx.send(Err(e)).await;
                            return;
                        }
                        _ => continue,
                    }
                };

                if draft_tokens.is_empty() {
                    debug!("Drafter produced no tokens, ending session");
                    break;
                }

                debug!(
                    "Received {} draft tokens, sending to verifier",
                    draft_tokens.len()
                );

                // Phase 2: Send draft tokens to verifier for batch verification
                let verify_msg = MeshMessage::DraftTokens {
                    session_id,
                    tokens: draft_tokens.clone(),
                    draft_model_id: draft_model_id.clone(),
                };

                if let Err(e) = self.transport.send(&self.verifier_peer, verify_msg).await {
                    let _ = tx.send(Err(e)).await;
                    break;
                }

                // Wait for VerifyDraft response
                let verification = loop {
                    match self.transport.recv().await {
                        Ok((
                            _,
                            MeshMessage::VerifyDraft {
                                session_id: sid,
                                accepted_count,
                                correction_token,
                            },
                        )) if sid == session_id => {
                            break VerificationResult {
                                accepted_count,
                                correction_token,
                            };
                        }
                        Ok((_, MeshMessage::Error { message, .. })) => {
                            let _ = tx
                                .send(Err(MeshError::Pipeline(format!(
                                    "Verifier error: {message}"
                                ))))
                                .await;
                            return;
                        }
                        Err(e) => {
                            let _ = tx.send(Err(e)).await;
                            return;
                        }
                        _ => continue,
                    }
                };

                debug!(
                    "Verification: {}/{} accepted, correction: {:?}",
                    verification.accepted_count,
                    draft_tokens.len(),
                    verification.correction_token
                );

                // Phase 3: Emit accepted tokens
                let accepted = verification.accepted_count as usize;
                for &token_id in draft_tokens.iter().take(accepted) {
                    let token = Token {
                        text: format!("<token_{token_id}>"), // Placeholder — real impl uses tokenizer
                        id: token_id,
                        logprob: None,
                        is_special: false,
                    };
                    context_tokens.push(token_id);
                    total_generated += 1;

                    if tx.send(Ok(token)).await.is_err() {
                        return; // Receiver dropped
                    }
                }

                // Emit correction token if present
                if let Some(correction_id) = verification.correction_token {
                    let token = Token {
                        text: format!("<token_{correction_id}>"),
                        id: correction_id,
                        logprob: None,
                        is_special: false,
                    };
                    context_tokens.push(correction_id);
                    total_generated += 1;

                    if tx.send(Ok(token)).await.is_err() {
                        return;
                    }
                }

                // If no tokens were accepted and no correction, something is wrong
                if accepted == 0 && verification.correction_token.is_none() {
                    warn!("No tokens accepted and no correction — ending session");
                    break;
                }
            }

            // Cleanup
            let _ = self
                .transport
                .send(
                    &self.drafter_peer,
                    MeshMessage::ReleaseSession { session_id },
                )
                .await;
            let _ = self
                .transport
                .send(
                    &self.verifier_peer,
                    MeshMessage::ReleaseSession { session_id },
                )
                .await;

            info!(
                "Speculative session {} complete: {} tokens generated",
                session_id, total_generated
            );
        });

        rx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speculative_config_defaults() {
        let config = SpeculativeConfig::default();
        assert!(config.enabled);
        assert_eq!(config.draft_length, 5);
        assert!(config.draft_model.is_none());
    }

    #[test]
    fn test_verification_result() {
        let result = VerificationResult {
            accepted_count: 3,
            correction_token: Some(42),
        };
        assert_eq!(result.accepted_count, 3);
        assert_eq!(result.correction_token, Some(42));
    }
}
