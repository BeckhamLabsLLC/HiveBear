use crate::error::CmdResult;
use crate::state::AppState;
use hivebear_core::{HardwareProfile, ModelRecommendation};
use tauri::State;

#[tauri::command]
pub fn get_hardware_profile(state: State<'_, AppState>) -> CmdResult<HardwareProfile> {
    Ok(state.profile.clone())
}

#[tauri::command]
pub fn get_recommendations(state: State<'_, AppState>) -> CmdResult<Vec<ModelRecommendation>> {
    let config = state
        .config
        .lock()
        .map_err(|_| String::from("Config lock poisoned"))?;
    let recs = hivebear_core::recommender::recommend(&state.profile, &config);
    Ok(recs)
}
