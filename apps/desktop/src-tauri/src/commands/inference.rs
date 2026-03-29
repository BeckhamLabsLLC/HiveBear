use crate::error::CmdResult;
use crate::state::AppState;
use hivebear_inference::types::{ChatMessage, ModelInfo};
use serde::Serialize;
use tauri::State;

#[derive(Serialize)]
pub struct LoadedModel {
    pub handle_id: u64,
    pub model_path: String,
    pub engine: String,
}

#[tauri::command]
pub async fn load_model(
    state: State<'_, AppState>,
    model_path: String,
    context_length: Option<u32>,
    gpu_layers: Option<u32>,
) -> CmdResult<LoadedModel> {
    let mut load_config = hivebear_inference::LoadConfig::default();
    if let Some(ctx) = context_length {
        load_config.context_length = ctx;
    }
    if let Some(layers) = gpu_layers {
        load_config.offload.gpu_layers = Some(layers);
        load_config.offload.auto = false;
    }

    let resolved = state
        .registry
        .resolve(&model_path)
        .await
        .unwrap_or_else(|_| model_path.clone().into());

    let handle = state
        .orchestrator
        .load(&resolved, &load_config)
        .await
        .map_err(|e| String::from(crate::error::CommandError::from(e)))?;

    Ok(LoadedModel {
        handle_id: handle.id,
        model_path: handle.model_path.display().to_string(),
        engine: format!("{:?}", handle.engine),
    })
}

#[tauri::command]
pub async fn stream_chat(
    state: State<'_, AppState>,
    window: tauri::WebviewWindow,
    handle_id: u64,
    messages: Vec<ChatMessage>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
) -> CmdResult<String> {
    use futures::StreamExt;
    use tauri::Emitter;

    let models = state.orchestrator.loaded_models();
    let info = models
        .iter()
        .find(|m| m.handle_id == handle_id)
        .ok_or_else(|| String::from("Model not loaded"))?;

    let handle = hivebear_inference::ModelHandle {
        id: handle_id,
        model_path: info.model_path.clone().into(),
        engine: info.engine,
    };

    let mut request = hivebear_inference::GenerateRequest {
        messages,
        model_name: Some(info.model_path.clone()),
        ..Default::default()
    };
    if let Some(t) = temperature {
        request.sampling.temperature = t;
    }
    if let Some(m) = max_tokens {
        request.max_tokens = m;
    }

    let mut stream = state
        .orchestrator
        .stream(&handle, &request)
        .map_err(|e| String::from(crate::error::CommandError::from(e)))?;

    let mut full_text = String::new();
    while let Some(result) = stream.next().await {
        match result {
            Ok(token) => {
                full_text.push_str(&token.text);
                let _ = window.emit("chat-token", &token.text);
            }
            Err(e) => {
                let _ = window.emit("chat-error", e.to_string());
                break;
            }
        }
    }

    let _ = window.emit("chat-done", ());
    Ok(full_text)
}

#[tauri::command]
pub async fn unload_model(state: State<'_, AppState>, handle_id: u64) -> CmdResult<()> {
    let models = state.orchestrator.loaded_models();
    let info = models
        .iter()
        .find(|m| m.handle_id == handle_id)
        .ok_or_else(|| String::from("Model not loaded"))?;

    let handle = hivebear_inference::ModelHandle {
        id: handle_id,
        model_path: info.model_path.clone().into(),
        engine: info.engine,
    };

    state
        .orchestrator
        .unload(&handle)
        .await
        .map_err(|e| String::from(crate::error::CommandError::from(e)))
}

#[tauri::command]
pub fn list_loaded_models(state: State<'_, AppState>) -> CmdResult<Vec<ModelInfo>> {
    Ok(state.orchestrator.loaded_models())
}
