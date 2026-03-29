use std::path::PathBuf;

/// Errors from the model registry.
#[derive(Debug, thiserror::Error)]
pub enum RegistryError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Download failed: {0}")]
    DownloadError(String),

    #[error("Download interrupted — resume with `hivebear install {model_id}`")]
    DownloadInterrupted { model_id: String },

    #[error("Integrity check failed for {path}: expected SHA-256 {expected}, got {actual}")]
    IntegrityError {
        path: PathBuf,
        expected: String,
        actual: String,
    },

    #[error("Model already installed: {0}")]
    AlreadyInstalled(String),

    #[error("No downloadable files found for model: {0}")]
    NoFilesFound(String),

    #[error("Conversion not available: {0}")]
    ConversionUnavailable(String),

    #[error("HuggingFace API error: {0}")]
    HuggingFaceError(String),

    #[error("Local index error: {0}")]
    IndexError(String),

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, RegistryError>;
