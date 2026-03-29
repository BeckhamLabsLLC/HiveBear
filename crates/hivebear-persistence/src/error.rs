#[derive(Debug, thiserror::Error)]
pub enum PersistenceError {
    #[error("Database error: {0}")]
    Database(#[from] rusqlite::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("Conversation not found: {0}")]
    NotFound(String),
}

pub type Result<T> = std::result::Result<T, PersistenceError>;
