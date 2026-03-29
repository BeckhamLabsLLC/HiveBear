mod db;
mod error;
mod models;

pub use db::ChatDatabase;
pub use error::{PersistenceError, Result};
pub use models::{Conversation, MessageRole, PersistedMessage};
