use crate::error::CmdResult;
use crate::state::AppState;
use crate::validation;
use hivebear_persistence::{Conversation, MessageRole, PersistedMessage};
use tauri::State;

#[tauri::command]
pub fn list_conversations(state: State<'_, AppState>) -> CmdResult<Vec<Conversation>> {
    state
        .chat_db
        .list_conversations()
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn create_conversation(
    state: State<'_, AppState>,
    title: String,
    model_id: String,
) -> CmdResult<Conversation> {
    validation::validate_title(&title)?;
    validation::validate_title(&model_id)?;
    state
        .chat_db
        .create_conversation(&title, &model_id)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn get_conversation_messages(
    state: State<'_, AppState>,
    conversation_id: String,
) -> CmdResult<Vec<PersistedMessage>> {
    validation::validate_conversation_id(&conversation_id)?;
    state
        .chat_db
        .get_messages(&conversation_id)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn add_message(
    state: State<'_, AppState>,
    conversation_id: String,
    role: String,
    content: String,
) -> CmdResult<PersistedMessage> {
    validation::validate_conversation_id(&conversation_id)?;
    validation::validate_role(&role)?;
    validation::validate_message_content(&content)?;
    let msg_role = MessageRole::parse(&role);
    state
        .chat_db
        .add_message(&conversation_id, msg_role, &content, None)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn delete_conversation(state: State<'_, AppState>, conversation_id: String) -> CmdResult<()> {
    validation::validate_conversation_id(&conversation_id)?;
    state
        .chat_db
        .delete_conversation(&conversation_id)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn rename_conversation(
    state: State<'_, AppState>,
    conversation_id: String,
    new_title: String,
) -> CmdResult<()> {
    validation::validate_conversation_id(&conversation_id)?;
    validation::validate_title(&new_title)?;
    state
        .chat_db
        .rename_conversation(&conversation_id, &new_title)
        .map_err(|e| e.to_string())
}

#[tauri::command]
pub fn search_conversations(
    state: State<'_, AppState>,
    query: String,
) -> CmdResult<Vec<Conversation>> {
    validation::validate_search_query(&query)?;
    state
        .chat_db
        .search_conversations(&query)
        .map_err(|e| e.to_string())
}
