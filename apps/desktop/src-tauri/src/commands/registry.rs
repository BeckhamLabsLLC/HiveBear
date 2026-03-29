use crate::error::CmdResult;
use crate::state::AppState;
use hivebear_registry::{metadata::SearchResult, ModelMetadata, StorageReport};
use tauri::State;

#[tauri::command]
pub async fn search_models(
    state: State<'_, AppState>,
    query: String,
    limit: Option<usize>,
) -> CmdResult<Vec<SearchResult>> {
    state
        .registry
        .search(&query, limit.unwrap_or(20), Some(&state.profile))
        .await
        .map_err(|e| crate::error::CommandError::from(e).into())
}

#[tauri::command]
pub async fn install_model(
    state: State<'_, AppState>,
    window: tauri::WebviewWindow,
    model_id: String,
    quant: Option<String>,
) -> CmdResult<hivebear_registry::InstalledInfo> {
    use tauri::Emitter;
    let progress_cb = move |p: hivebear_registry::download::DownloadProgress| {
        let _ = window.emit("download-progress", &p);
    };
    state
        .registry
        .install(&model_id, quant.as_deref(), None, Some(&progress_cb))
        .await
        .map_err(|e| crate::error::CommandError::from(e).into())
}

#[tauri::command]
pub async fn list_installed(state: State<'_, AppState>) -> CmdResult<Vec<ModelMetadata>> {
    Ok(state.registry.list_installed().await)
}

#[tauri::command]
pub async fn remove_model(state: State<'_, AppState>, model_id: String) -> CmdResult<u64> {
    state
        .registry
        .remove(&model_id)
        .await
        .map_err(|e| crate::error::CommandError::from(e).into())
}

#[tauri::command]
pub async fn get_storage_report(state: State<'_, AppState>) -> CmdResult<StorageReport> {
    state
        .registry
        .storage_manager()
        .await
        .report()
        .await
        .map_err(|e| e.to_string())
}
