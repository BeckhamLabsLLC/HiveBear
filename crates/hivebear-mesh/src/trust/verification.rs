use std::sync::Arc;

use sha2::{Digest, Sha256};
use tracing::{debug, warn};
use uuid::Uuid;

use crate::error::{MeshError, Result};
use crate::peer::NodeId;
use crate::transport::protocol::MeshMessage;
use crate::transport::MeshTransport;

/// Minimum verification rate floor — cannot be configured below this in production.
const MIN_VERIFICATION_RATE: f64 = 0.01;

/// How long to wait for a peer to answer a challenge.
const CHALLENGE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Probabilistic trust verifier that spot-checks peer computations.
///
/// Uses random sampling instead of deterministic intervals to prevent
/// malicious peers from predicting which tokens will be verified.
pub struct TrustVerifier {
    transport: Arc<dyn MeshTransport>,
    /// Fraction of tokens to verify (0.0 to 1.0).
    verification_rate: f64,
}

impl TrustVerifier {
    pub fn new(transport: Arc<dyn MeshTransport>, verification_rate: f64) -> Self {
        // Enforce minimum floor to prevent disabling verification entirely
        let rate = verification_rate.clamp(MIN_VERIFICATION_RATE, 1.0);
        Self {
            transport,
            verification_rate: rate,
        }
    }

    /// Decide whether to verify this token using random sampling.
    ///
    /// Random sampling prevents malicious peers from predicting which
    /// tokens will be verified and only returning correct results for those.
    pub fn should_verify(&self) -> bool {
        if self.verification_rate >= 1.0 {
            return true;
        }

        use rand::Rng;
        rand::thread_rng().gen_bool(self.verification_rate)
    }

    /// Ask `peer` to re-run `layer_range` over `input` and report the hash of
    /// what it produced.
    ///
    /// Returns the hash. It does **not** decide anything: comparing against a
    /// reference is the caller's job, because only the caller knows where a
    /// trustworthy reference came from.
    pub async fn challenge(
        &self,
        peer: &NodeId,
        session_id: Uuid,
        layer_range: std::ops::Range<u32>,
        token_position: u32,
        input: &ActivationBytes,
    ) -> Result<[u8; 32]> {
        // Claim the session first: the reply is routed by session id, and
        // this used to sit on the shared queue where any other task could
        // take it.
        let mut rx = self.transport.subscribe_session(session_id);

        let challenge = MeshMessage::VerifyChallenge {
            session_id,
            layer_range,
            token_position,
            data: bytes::Bytes::from(input.data.clone()),
            shape: input.shape.clone(),
            dtype: input.dtype,
        };

        let result = async {
            self.transport.send(peer, challenge).await?;

            loop {
                // A peer that simply never answers must not stall the caller
                // forever; the old code awaited recv() with no bound at all.
                let next = tokio::time::timeout(CHALLENGE_TIMEOUT, rx.recv())
                    .await
                    .map_err(|_| {
                        MeshError::NatTraversal(format!(
                            "Verification challenge to {peer} timed out"
                        ))
                    })?;

                match next {
                    None => {
                        return Err(MeshError::Transport(
                            "Session queue closed during verification".into(),
                        ))
                    }
                    Some((
                        resp_peer,
                        MeshMessage::VerifyResponse {
                            session_id: sid,
                            output_hash,
                            error,
                        },
                    )) if sid == session_id && resp_peer == *peer => {
                        if let Some(err) = error {
                            return Err(MeshError::Pipeline(format!(
                                "Peer {peer} could not answer the challenge: {err}"
                            )));
                        }
                        debug!("Challenge answered by {peer}");
                        return Ok(output_hash);
                    }
                    Some((from, other)) => {
                        debug!("Ignoring {other:?} from {from} while awaiting a challenge reply");
                    }
                }
            }
        }
        .await;

        self.transport.unsubscribe_session(&session_id);
        result
    }

    /// Check that `peer` still produces the same output it produced before
    /// for the same input.
    ///
    /// This is a *consistency* check, and that is all it claims to be: it
    /// catches a peer whose answers change — returning good results while
    /// being watched and garbage otherwise — but it cannot catch one that is
    /// wrong in the same way every time. Detecting that needs an independent
    /// reference, either a second peer holding the same layers or a local
    /// recomputation; `challenge` returns the raw hash so a caller with
    /// either can compare directly.
    ///
    /// The previous implementation read `passed` straight off the response,
    /// so a dishonest peer verified itself by answering "yes".
    pub async fn verify_consistent_with(
        &self,
        peer: &NodeId,
        session_id: Uuid,
        layer_range: std::ops::Range<u32>,
        token_position: u32,
        input: &ActivationBytes,
        expected_output_hash: [u8; 32],
    ) -> Result<bool> {
        let actual = self
            .challenge(peer, session_id, layer_range, token_position, input)
            .await?;

        let matches = actual == expected_output_hash;
        if !matches {
            warn!(
                "Peer {peer} returned a different result for the same input \
                 (expected {}, got {})",
                hex::encode(&expected_output_hash[..8]),
                hex::encode(&actual[..8])
            );
        }
        Ok(matches)
    }
}

/// An activation tensor in wire form, used to pose a challenge.
#[derive(Debug, Clone)]
pub struct ActivationBytes {
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub dtype: crate::transport::protocol::TensorDtype,
}

/// Hash arbitrary bytes with SHA-256.
pub fn hash_bytes(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

/// Answer a [`MeshMessage::VerifyChallenge`] by running the challenged input
/// through the loaded layers and hashing the real output.
///
/// Shared by every responder so there is exactly one place where "what does
/// an honest answer look like" is defined. A node with no pipeline handler
/// says so rather than inventing a hash.
pub async fn answer_challenge(
    handler: Option<&dyn crate::protocol::MeshPipelineHandler>,
    session_id: Uuid,
    token_position: u32,
    data: Vec<u8>,
    shape: Vec<usize>,
    dtype: crate::transport::protocol::TensorDtype,
) -> MeshMessage {
    let Some(handler) = handler else {
        return MeshMessage::VerifyResponse {
            session_id,
            output_hash: [0u8; 32],
            error: Some("no pipeline handler loaded on this node".into()),
        };
    };

    match handler
        .forward_layers(data, shape, dtype.to_u8(), token_position as usize)
        .await
    {
        Ok((out_data, _shape, _dtype)) => MeshMessage::VerifyResponse {
            session_id,
            output_hash: hash_bytes(&out_data),
            error: None,
        },
        Err(e) => MeshMessage::VerifyResponse {
            session_id,
            output_hash: [0u8; 32],
            error: Some(e),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::protocol::MeshPipelineHandler;
    use crate::transport::protocol::TensorDtype;

    /// Computes honestly: output = input with every byte incremented.
    struct HonestLayers;

    #[async_trait::async_trait]
    impl MeshPipelineHandler for HonestLayers {
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
            Ok((
                data.iter().map(|b| b.wrapping_add(1)).collect(),
                shape,
                dtype,
            ))
        }
        async fn unload_layers(&self) -> std::result::Result<(), String> {
            Ok(())
        }
    }

    /// Returns something other than the real computation.
    struct CheatingLayers;

    #[async_trait::async_trait]
    impl MeshPipelineHandler for CheatingLayers {
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
            _data: Vec<u8>,
            shape: Vec<usize>,
            dtype: u8,
            _i: usize,
        ) -> std::result::Result<(Vec<u8>, Vec<usize>, u8), String> {
            Ok((vec![0u8; 4], shape, dtype))
        }
        async fn unload_layers(&self) -> std::result::Result<(), String> {
            Ok(())
        }
    }

    fn unwrap_response(msg: MeshMessage) -> ([u8; 32], Option<String>) {
        match msg {
            MeshMessage::VerifyResponse {
                output_hash, error, ..
            } => (output_hash, error),
            other => panic!("expected VerifyResponse, got {other:?}"),
        }
    }

    /// The regression: the responder hashed session metadata — layer range
    /// and model id — and always answered `passed: true`. It therefore
    /// returned the same hash whatever the input, and graded itself.
    #[tokio::test]
    async fn an_honest_answer_depends_on_the_input() {
        let handler = HonestLayers;
        let session = Uuid::new_v4();

        let (h1, e1) = unwrap_response(
            answer_challenge(
                Some(&handler),
                session,
                0,
                vec![1, 2, 3, 4],
                vec![1, 4],
                TensorDtype::F32,
            )
            .await,
        );
        let (h2, e2) = unwrap_response(
            answer_challenge(
                Some(&handler),
                session,
                0,
                vec![9, 9, 9, 9],
                vec![1, 4],
                TensorDtype::F32,
            )
            .await,
        );

        assert!(e1.is_none() && e2.is_none());
        assert_ne!(
            h1, h2,
            "a hash that does not change with the input proves nothing"
        );

        // And it is the hash of the real output, not of anything else.
        assert_eq!(h1, hash_bytes(&[2, 3, 4, 5]));
    }

    #[tokio::test]
    async fn a_cheating_peer_produces_a_different_hash() {
        let session = Uuid::new_v4();
        let input = vec![1, 2, 3, 4];

        let (honest, _) = unwrap_response(
            answer_challenge(
                Some(&HonestLayers),
                session,
                0,
                input.clone(),
                vec![1, 4],
                TensorDtype::F32,
            )
            .await,
        );
        let (cheat, _) = unwrap_response(
            answer_challenge(
                Some(&CheatingLayers),
                session,
                0,
                input,
                vec![1, 4],
                TensorDtype::F32,
            )
            .await,
        );

        assert_ne!(
            honest, cheat,
            "a peer that does not do the work must not be able to match the honest hash"
        );
    }

    #[tokio::test]
    async fn a_node_with_no_layers_says_so_rather_than_inventing_a_hash() {
        let (hash, error) = unwrap_response(
            answer_challenge(None, Uuid::new_v4(), 0, vec![1], vec![1], TensorDtype::F32).await,
        );
        assert_eq!(hash, [0u8; 32]);
        assert!(error.is_some(), "silence would look like a passing result");
    }

    #[test]
    fn test_hash_deterministic() {
        let data = b"hello world";
        let h1 = hash_bytes(data);
        let h2 = hash_bytes(data);
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_hash_different_inputs() {
        let h1 = hash_bytes(b"hello");
        let h2 = hash_bytes(b"world");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_should_verify_rate_minimum_floor() {
        let transport = Arc::new(crate::transport::mock::MockTransport::new(
            NodeId::generate().0,
            crate::transport::mock::MockRegistry::new(),
        ));
        // Rate of 0.0 should be clamped to MIN_VERIFICATION_RATE
        let verifier = TrustVerifier::new(transport, 0.0);
        // With the minimum floor, at least some verifications should happen
        // over a large sample
        let mut verified = 0;
        for _ in 0..1000 {
            if verifier.should_verify() {
                verified += 1;
            }
        }
        assert!(
            verified > 0,
            "Minimum floor should prevent zero verifications"
        );
    }

    #[test]
    fn test_should_verify_rate_one() {
        let transport = Arc::new(crate::transport::mock::MockTransport::new(
            NodeId::generate().0,
            crate::transport::mock::MockRegistry::new(),
        ));
        let verifier = TrustVerifier::new(transport, 1.0);
        for _ in 0..100 {
            assert!(verifier.should_verify());
        }
    }

    #[test]
    fn test_should_verify_rate_partial() {
        let transport = Arc::new(crate::transport::mock::MockTransport::new(
            NodeId::generate().0,
            crate::transport::mock::MockRegistry::new(),
        ));
        let verifier = TrustVerifier::new(transport, 0.1); // 10%
        let mut verified = 0;
        for _ in 0..100 {
            if verifier.should_verify() {
                verified += 1;
            }
        }
        // Should be approximately 10 out of 100
        assert!(verified > 0);
        assert!(verified < 50);
    }
}
