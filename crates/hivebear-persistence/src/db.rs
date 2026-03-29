use crate::error::{PersistenceError, Result};
use crate::models::{Conversation, MessageRole, PersistedMessage};
use chrono::Utc;
use rusqlite::{params, Connection};
use std::path::Path;
use std::sync::Mutex;

pub struct ChatDatabase {
    conn: Mutex<Connection>,
}

impl ChatDatabase {
    /// Open (or create) the database at the given path.
    pub fn open(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        let conn = Connection::open(path)?;
        let db = Self {
            conn: Mutex::new(conn),
        };
        db.migrate()?;
        Ok(db)
    }

    /// Open an in-memory database (for testing).
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        let db = Self {
            conn: Mutex::new(conn),
        };
        db.migrate()?;
        Ok(db)
    }

    fn migrate(&self) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                model_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tool_call_json TEXT,
                created_at TEXT NOT NULL,
                sequence_num INTEGER NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_messages_conversation
                ON messages(conversation_id, sequence_num);
            PRAGMA foreign_keys = ON;",
        )?;
        Ok(())
    }

    /// Create a new conversation. Returns the conversation.
    pub fn create_conversation(&self, title: &str, model_id: &str) -> Result<Conversation> {
        let conn = self.conn.lock().unwrap();
        let id = uuid::Uuid::new_v4().to_string();
        let now = Utc::now();
        let now_str = now.to_rfc3339();
        conn.execute(
            "INSERT INTO conversations (id, title, model_id, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id, title, model_id, now_str, now_str],
        )?;
        Ok(Conversation {
            id,
            title: title.to_string(),
            model_id: model_id.to_string(),
            created_at: now,
            updated_at: now,
            message_count: 0,
        })
    }

    /// List all conversations, most recent first.
    pub fn list_conversations(&self) -> Result<Vec<Conversation>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT c.id, c.title, c.model_id, c.created_at, c.updated_at,
                    (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) as msg_count
             FROM conversations c
             ORDER BY c.updated_at DESC",
        )?;
        let rows = stmt.query_map([], |row| {
            let created_str: String = row.get(3)?;
            let updated_str: String = row.get(4)?;
            Ok(Conversation {
                id: row.get(0)?,
                title: row.get(1)?,
                model_id: row.get(2)?,
                created_at: chrono::DateTime::parse_from_rfc3339(&created_str)
                    .unwrap_or_default()
                    .with_timezone(&Utc),
                updated_at: chrono::DateTime::parse_from_rfc3339(&updated_str)
                    .unwrap_or_default()
                    .with_timezone(&Utc),
                message_count: row.get(5)?,
            })
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    /// Get messages for a conversation, ordered by sequence number.
    pub fn get_messages(&self, conversation_id: &str) -> Result<Vec<PersistedMessage>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, conversation_id, role, content, tool_call_json, created_at, sequence_num
             FROM messages
             WHERE conversation_id = ?1
             ORDER BY sequence_num ASC",
        )?;
        let rows = stmt.query_map(params![conversation_id], |row| {
            let created_str: String = row.get(5)?;
            let role_str: String = row.get(2)?;
            Ok(PersistedMessage {
                id: row.get(0)?,
                conversation_id: row.get(1)?,
                role: MessageRole::parse(&role_str),
                content: row.get(3)?,
                tool_call_json: row.get(4)?,
                created_at: chrono::DateTime::parse_from_rfc3339(&created_str)
                    .unwrap_or_default()
                    .with_timezone(&Utc),
                sequence_num: row.get(6)?,
            })
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    /// Add a message to a conversation. Updates the conversation's updated_at.
    pub fn add_message(
        &self,
        conversation_id: &str,
        role: MessageRole,
        content: &str,
        tool_call_json: Option<&str>,
    ) -> Result<PersistedMessage> {
        let conn = self.conn.lock().unwrap();
        let id = uuid::Uuid::new_v4().to_string();
        let now = Utc::now();
        let now_str = now.to_rfc3339();

        // Get next sequence number
        let seq: i64 = conn.query_row(
            "SELECT COALESCE(MAX(sequence_num), 0) + 1 FROM messages WHERE conversation_id = ?1",
            params![conversation_id],
            |row| row.get(0),
        )?;

        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, tool_call_json, created_at, sequence_num)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![id, conversation_id, role.as_str(), content, tool_call_json, now_str, seq],
        )?;

        // Update conversation timestamp
        conn.execute(
            "UPDATE conversations SET updated_at = ?1 WHERE id = ?2",
            params![now_str, conversation_id],
        )?;

        Ok(PersistedMessage {
            id,
            conversation_id: conversation_id.to_string(),
            role,
            content: content.to_string(),
            tool_call_json: tool_call_json.map(|s| s.to_string()),
            created_at: now,
            sequence_num: seq,
        })
    }

    /// Delete a conversation and all its messages.
    pub fn delete_conversation(&self, conversation_id: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        // Messages are cascade-deleted via FK
        let changes = conn.execute(
            "DELETE FROM conversations WHERE id = ?1",
            params![conversation_id],
        )?;
        if changes == 0 {
            return Err(PersistenceError::NotFound(conversation_id.to_string()));
        }
        Ok(())
    }

    /// Rename a conversation.
    pub fn rename_conversation(&self, conversation_id: &str, new_title: &str) -> Result<()> {
        let conn = self.conn.lock().unwrap();
        let now_str = Utc::now().to_rfc3339();
        let changes = conn.execute(
            "UPDATE conversations SET title = ?1, updated_at = ?2 WHERE id = ?3",
            params![new_title, now_str, conversation_id],
        )?;
        if changes == 0 {
            return Err(PersistenceError::NotFound(conversation_id.to_string()));
        }
        Ok(())
    }

    /// Search conversations by title.
    pub fn search_conversations(&self, query: &str) -> Result<Vec<Conversation>> {
        let conn = self.conn.lock().unwrap();
        let pattern = format!("%{query}%");
        let mut stmt = conn.prepare(
            "SELECT c.id, c.title, c.model_id, c.created_at, c.updated_at,
                    (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) as msg_count
             FROM conversations c
             WHERE c.title LIKE ?1
             ORDER BY c.updated_at DESC",
        )?;
        let rows = stmt.query_map(params![pattern], |row| {
            let created_str: String = row.get(3)?;
            let updated_str: String = row.get(4)?;
            Ok(Conversation {
                id: row.get(0)?,
                title: row.get(1)?,
                model_id: row.get(2)?,
                created_at: chrono::DateTime::parse_from_rfc3339(&created_str)
                    .unwrap_or_default()
                    .with_timezone(&Utc),
                updated_at: chrono::DateTime::parse_from_rfc3339(&updated_str)
                    .unwrap_or_default()
                    .with_timezone(&Utc),
                message_count: row.get(5)?,
            })
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_and_list_conversations() {
        let db = ChatDatabase::open_in_memory().unwrap();

        let conv = db.create_conversation("Test Chat", "llama-3.1-8b").unwrap();
        assert_eq!(conv.title, "Test Chat");
        assert_eq!(conv.model_id, "llama-3.1-8b");

        let convs = db.list_conversations().unwrap();
        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].id, conv.id);
    }

    #[test]
    fn test_add_and_get_messages() {
        let db = ChatDatabase::open_in_memory().unwrap();
        let conv = db.create_conversation("Test", "model").unwrap();

        db.add_message(&conv.id, MessageRole::User, "Hello", None)
            .unwrap();
        db.add_message(&conv.id, MessageRole::Assistant, "Hi there!", None)
            .unwrap();

        let msgs = db.get_messages(&conv.id).unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].content, "Hello");
        assert_eq!(msgs[1].content, "Hi there!");
        assert_eq!(msgs[0].sequence_num, 1);
        assert_eq!(msgs[1].sequence_num, 2);
    }

    #[test]
    fn test_delete_conversation() {
        let db = ChatDatabase::open_in_memory().unwrap();
        let conv = db.create_conversation("Test", "model").unwrap();
        db.add_message(&conv.id, MessageRole::User, "Hello", None)
            .unwrap();

        db.delete_conversation(&conv.id).unwrap();

        let convs = db.list_conversations().unwrap();
        assert!(convs.is_empty());
    }

    #[test]
    fn test_rename_conversation() {
        let db = ChatDatabase::open_in_memory().unwrap();
        let conv = db.create_conversation("Old Title", "model").unwrap();

        db.rename_conversation(&conv.id, "New Title").unwrap();

        let convs = db.list_conversations().unwrap();
        assert_eq!(convs[0].title, "New Title");
    }

    #[test]
    fn test_search_conversations() {
        let db = ChatDatabase::open_in_memory().unwrap();
        db.create_conversation("Rust Programming", "model").unwrap();
        db.create_conversation("Python Scripts", "model").unwrap();

        let results = db.search_conversations("Rust").unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].title, "Rust Programming");
    }
}
