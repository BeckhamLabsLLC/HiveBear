use crate::error::CmdResult;

/// Maximum message content size (1 MB).
const MAX_MESSAGE_BYTES: usize = 1_000_000;
/// Maximum title length.
const MAX_TITLE_LEN: usize = 500;
/// Maximum search query length.
const MAX_SEARCH_QUERY_LEN: usize = 200;

/// Validate a conversation ID looks like a valid UUID (8-4-4-4-12 hex format).
pub fn validate_conversation_id(id: &str) -> CmdResult<()> {
    // UUID format: 8-4-4-4-12 hex chars = 36 total with dashes
    if id.len() != 36 {
        return Err("Invalid conversation ID: must be a valid UUID".into());
    }
    let parts: Vec<&str> = id.split('-').collect();
    if parts.len() != 5
        || parts[0].len() != 8
        || parts[1].len() != 4
        || parts[2].len() != 4
        || parts[3].len() != 4
        || parts[4].len() != 12
    {
        return Err("Invalid conversation ID: must be a valid UUID".into());
    }
    if !id.chars().all(|c| c.is_ascii_hexdigit() || c == '-') {
        return Err("Invalid conversation ID: must be a valid UUID".into());
    }
    Ok(())
}

/// Validate a conversation or model title.
pub fn validate_title(title: &str) -> CmdResult<()> {
    if title.is_empty() || title.len() > MAX_TITLE_LEN {
        return Err(format!(
            "Title must be between 1 and {MAX_TITLE_LEN} characters"
        ));
    }
    Ok(())
}

/// Validate message content has reasonable size.
pub fn validate_message_content(content: &str) -> CmdResult<()> {
    if content.is_empty() {
        return Err("Message content cannot be empty".into());
    }
    if content.len() > MAX_MESSAGE_BYTES {
        return Err(format!(
            "Message content exceeds maximum size of {MAX_MESSAGE_BYTES} bytes"
        ));
    }
    Ok(())
}

/// Validate message role is one of the expected values.
pub fn validate_role(role: &str) -> CmdResult<()> {
    match role {
        "system" | "user" | "assistant" => Ok(()),
        _ => Err(format!(
            "Invalid role '{role}': must be 'system', 'user', or 'assistant'"
        )),
    }
}

/// Validate a search query string.
pub fn validate_search_query(query: &str) -> CmdResult<()> {
    if query.len() > MAX_SEARCH_QUERY_LEN {
        return Err(format!(
            "Search query exceeds maximum length of {MAX_SEARCH_QUERY_LEN} characters"
        ));
    }
    Ok(())
}
