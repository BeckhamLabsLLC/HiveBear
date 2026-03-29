use std::path::Path;

use ed25519_dalek::SigningKey;
use tracing::{info, warn};

use crate::error::{MeshError, Result};
use crate::peer::NodeId;

/// Persistent node identity backed by an Ed25519 keypair.
///
/// The identity is loaded from disk if it exists, or generated fresh
/// and saved for future runs. This ensures TOFU certificate pinning
/// and reputation scores persist across restarts.
pub struct NodeIdentity {
    pub node_id: NodeId,
    pub signing_key: SigningKey,
}

impl NodeIdentity {
    /// Load an existing identity from `path`, or generate a new one and save it.
    pub fn load_or_generate(path: &Path) -> Result<Self> {
        if path.exists() {
            match Self::load(path) {
                Ok(identity) => {
                    info!(
                        "Loaded node identity {} from {}",
                        identity.node_id,
                        path.display()
                    );
                    return Ok(identity);
                }
                Err(e) => {
                    warn!(
                        "Failed to load identity from {}: {e}. Generating new identity.",
                        path.display()
                    );
                }
            }
        }

        let identity = Self::generate();
        if let Err(e) = identity.save(path) {
            warn!("Failed to save identity to {}: {e}", path.display());
        } else {
            info!(
                "Generated and saved new node identity {} to {}",
                identity.node_id,
                path.display()
            );
        }
        Ok(identity)
    }

    /// Generate a fresh random identity.
    pub fn generate() -> Self {
        let (node_id, signing_key) = NodeId::generate();
        Self {
            node_id,
            signing_key,
        }
    }

    /// Load identity from a file containing raw Ed25519 keypair bytes (64 bytes).
    fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|e| MeshError::Config(format!("Failed to read identity file: {e}")))?;

        if bytes.len() != 64 {
            return Err(MeshError::Config(format!(
                "Identity file has invalid size: expected 64 bytes, got {}",
                bytes.len()
            )));
        }

        let secret_bytes: [u8; 32] = bytes[..32]
            .try_into()
            .map_err(|_| MeshError::Config("Invalid secret key bytes".into()))?;

        let signing_key = SigningKey::from_bytes(&secret_bytes);
        let verifying_key = signing_key.verifying_key();

        // Verify the stored public key matches
        let stored_public: [u8; 32] = bytes[32..64]
            .try_into()
            .map_err(|_| MeshError::Config("Invalid public key bytes".into()))?;

        if verifying_key.to_bytes() != stored_public {
            return Err(MeshError::Config(
                "Identity file corrupted: public key does not match secret key".into(),
            ));
        }

        Ok(Self {
            node_id: NodeId(verifying_key),
            signing_key,
        })
    }

    /// Save identity to a file as raw Ed25519 keypair bytes (64 bytes: 32 secret + 32 public).
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                MeshError::Config(format!("Failed to create identity directory: {e}"))
            })?;
        }

        let mut bytes = Vec::with_capacity(64);
        bytes.extend_from_slice(&self.signing_key.to_bytes());
        bytes.extend_from_slice(&self.signing_key.verifying_key().to_bytes());

        // Set restrictive permissions before writing
        std::fs::write(path, &bytes)
            .map_err(|e| MeshError::Config(format!("Failed to write identity file: {e}")))?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o600);
            std::fs::set_permissions(path, perms).map_err(|e| {
                MeshError::Config(format!("Failed to set identity file permissions: {e}"))
            })?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_generate_and_save_load() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("node_identity.key");

        let identity = NodeIdentity::load_or_generate(&path).unwrap();
        assert!(path.exists());

        let loaded = NodeIdentity::load_or_generate(&path).unwrap();
        assert_eq!(identity.node_id, loaded.node_id);
        assert_eq!(
            identity.signing_key.to_bytes(),
            loaded.signing_key.to_bytes()
        );
    }

    #[test]
    fn test_corrupted_identity_regenerates() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("node_identity.key");

        // Write garbage
        std::fs::write(&path, b"not a valid key").unwrap();

        // Should generate a new identity
        let identity = NodeIdentity::load_or_generate(&path).unwrap();
        assert!(!identity.node_id.to_hex().is_empty());
    }

    #[test]
    fn test_file_size_validation() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("node_identity.key");

        std::fs::write(&path, vec![0u8; 32]).unwrap();
        assert!(NodeIdentity::load(&path).is_err());
    }
}
