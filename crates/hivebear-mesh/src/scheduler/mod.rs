pub mod optimizer;
pub mod plan;
pub mod swarm_scheduler;

use async_trait::async_trait;

use crate::error::Result;
use crate::peer::PeerInfo;
use plan::InferencePlan;

/// Scheduler that computes optimal layer distribution across peers.
#[async_trait]
pub trait LayerScheduler: Send + Sync {
    /// Given a model and available peers, produce an inference plan.
    async fn plan(
        &self,
        model_id: &str,
        total_layers: u32,
        model_size_bytes: u64,
        peers: &[PeerInfo],
    ) -> Result<InferencePlan>;
}
