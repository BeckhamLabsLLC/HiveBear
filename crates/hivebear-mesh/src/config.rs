use serde::{Deserialize, Serialize};

/// Mesh tier: free (volunteer) or paid (priority).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeshTier {
    Free,
    Paid,
}

impl MeshTier {
    pub fn from_str_lossy(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "paid" => MeshTier::Paid,
            _ => MeshTier::Free,
        }
    }
}

impl std::fmt::Display for MeshTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MeshTier::Free => write!(f, "free"),
            MeshTier::Paid => write!(f, "paid"),
        }
    }
}

/// Security mode for mesh TLS connections.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MeshSecurityMode {
    /// Trust on first use: accept and pin certificate on first connection.
    /// Subsequent connections to the same peer must present the same certificate.
    #[default]
    Pinned,
    /// Skip certificate verification entirely. For development/testing only.
    /// WARNING: Vulnerable to MITM attacks. Only available with the `insecure-dev` feature.
    #[cfg(feature = "insecure-dev")]
    Insecure,
}

impl std::fmt::Display for MeshSecurityMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MeshSecurityMode::Pinned => write!(f, "pinned"),
            #[cfg(feature = "insecure-dev")]
            MeshSecurityMode::Insecure => write!(f, "insecure"),
        }
    }
}

impl MeshSecurityMode {
    pub fn from_str_lossy(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            #[cfg(feature = "insecure-dev")]
            "insecure" => {
                tracing::warn!(
                    "⚠️  Insecure mesh mode enabled via config. \
                     Do NOT use in production — all TLS verification is disabled."
                );
                MeshSecurityMode::Insecure
            }
            #[cfg(not(feature = "insecure-dev"))]
            "insecure" => {
                tracing::error!(
                    "Insecure mesh mode requested but the 'insecure-dev' feature is not enabled. \
                     Falling back to Pinned (TOFU) mode. To use insecure mode, rebuild with: \
                     cargo build --features hivebear-mesh/insecure-dev"
                );
                MeshSecurityMode::Pinned
            }
            _ => MeshSecurityMode::Pinned,
        }
    }
}
