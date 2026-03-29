use std::sync::Arc;

use sha2::{Digest, Sha256};
use tracing::{debug, warn};
use uuid::Uuid;

use crate::error::Result;
use crate::peer::NodeId;
use crate::transport::protocol::MeshMessage;
use crate::transport::MeshTransport;

/// Minimum verification rate floor — cannot be configured below this in production.
const MIN_VERIFICATION_RATE: f64 = 0.01;

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

    /// Send a verification challenge to a peer and check the response.
    pub async fn verify_layer(
        &self,
        peer: &NodeId,
        session_id: Uuid,
        layer_index: u32,
        expected_input: &[u8],
    ) -> Result<bool> {
        let input_hash = hash_bytes(expected_input);

        let challenge = MeshMessage::VerifyChallenge {
            session_id,
            layer_index,
            input_hash,
        };

        self.transport.send(peer, challenge).await?;

        // Wait for response (with timeout handled by caller)
        let (resp_peer, resp_msg) = self.transport.recv().await?;

        match resp_msg {
            MeshMessage::VerifyResponse {
                session_id: sid,
                output_hash,
                passed,
            } if sid == session_id && resp_peer == *peer => {
                debug!(
                    "Verification result from {peer}: passed={passed}, hash={:?}",
                    &output_hash[..4]
                );
                Ok(passed)
            }
            _ => {
                warn!("Unexpected response to verification challenge");
                Ok(false)
            }
        }
    }
}

/// Hash arbitrary bytes with SHA-256.
pub fn hash_bytes(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

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
