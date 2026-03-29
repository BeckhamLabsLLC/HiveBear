pub mod checkpoint;
pub mod daemon;
pub mod health;
pub mod initiator;
pub mod worker;

use crate::scheduler::plan::InferencePlan;

/// Pipeline status for a distributed inference session.
#[derive(Debug, Clone)]
pub enum PipelineStatus {
    /// Setting up: assigning layers to peers.
    Initializing,
    /// All peers ready, running inference.
    Running {
        tokens_generated: u32,
        current_token_position: u32,
    },
    /// Inference complete.
    Completed { total_tokens: u32 },
    /// An error occurred.
    Failed { error: String },
}

/// Metadata about a running pipeline.
#[derive(Debug, Clone)]
pub struct PipelineSession {
    pub plan: InferencePlan,
    pub status: PipelineStatus,
}
