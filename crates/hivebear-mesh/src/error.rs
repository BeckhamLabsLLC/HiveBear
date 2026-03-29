use thiserror::Error;

/// Errors that can occur during mesh operations.
#[derive(Debug, Error)]
pub enum MeshError {
    #[error("Transport error: {0}")]
    Transport(String),

    #[error("Discovery error: {0}")]
    Discovery(String),

    #[error("Scheduling error: {0}")]
    Scheduling(String),

    #[error("Pipeline error: {0}")]
    Pipeline(String),

    #[error("Trust verification failed: {0}")]
    Trust(String),

    #[error("Protocol error: {0}")]
    Protocol(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Peer disconnected: {0}")]
    PeerDisconnected(String),

    #[error("Operation timed out: {0}")]
    Timeout(String),

    #[error("No peers available for model '{0}'")]
    NoPeersAvailable(String),

    #[error("Inference error: {0}")]
    Inference(#[from] hivebear_inference::InferenceError),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("NAT traversal failed: {0}")]
    NatTraversal(String),

    #[error("Relay connection failed: {0}")]
    Relay(String),

    #[error("Peer exchange error: {0}")]
    PeerExchange(String),

    #[error("Authentication failed: {0}")]
    Auth(String),

    #[error("Pipeline recovery failed: {0}")]
    Recovery(String),

    #[error("Compression error: {0}")]
    Compression(String),
}

pub type Result<T> = std::result::Result<T, MeshError>;
