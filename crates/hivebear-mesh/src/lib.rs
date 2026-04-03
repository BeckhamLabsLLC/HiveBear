pub mod backend;
pub mod config;
pub mod discovery;
pub mod error;
pub mod identity;
pub mod nat;
pub mod node;
pub mod peer;
pub mod pipeline;
pub mod protocol;
pub mod scheduler;
pub mod swarm;
pub mod transport;
pub mod trust;

#[cfg(all(feature = "insecure-dev", not(debug_assertions)))]
compile_error!("The `insecure-dev` feature must not be used in release builds");

// Re-export key types for convenience
pub use backend::MeshBackend;
pub use config::{MeshSecurityMode, MeshTier};
pub use error::{MeshError, Result};
pub use identity::NodeIdentity;
pub use node::MeshNode;
pub use peer::{NodeId, PeerInfo, PeerState};
pub use pipeline::daemon::MeshWorkerDaemon;
pub use protocol::{MeshInferenceHandler, MeshInferenceMessage};
pub use scheduler::plan::InferencePlan;
pub use transport::MeshTransport;

pub use nat::NatType;
pub use protocol::MeshPipelineHandler;
pub use transport::compression;
pub use transport::tensor_transfer;
