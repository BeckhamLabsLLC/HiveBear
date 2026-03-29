use crate::error::CmdResult;
use crate::state::AppState;
use hivebear_core::config::MeshConfig;
use serde::Serialize;
use tauri::State;

#[derive(Serialize)]
pub struct MeshStatus {
    pub enabled: bool,
    pub tier: String,
    pub port: u16,
    pub coordination_server: String,
    pub max_contribution_percent: f64,
    pub min_reputation: f64,
    pub verification_rate: f64,
}

#[tauri::command]
pub fn get_mesh_status(state: State<'_, AppState>) -> CmdResult<MeshStatus> {
    let config = state
        .config
        .lock()
        .map_err(|_| String::from("Config lock poisoned"))?;
    Ok(MeshStatus {
        enabled: config.mesh.enabled,
        tier: config.mesh.tier.clone(),
        port: config.mesh.port,
        coordination_server: config.mesh.coordination_server.clone(),
        max_contribution_percent: config.mesh.max_contribution_percent,
        min_reputation: config.mesh.min_reputation,
        verification_rate: config.mesh.verification_rate,
    })
}

#[tauri::command]
pub fn get_mesh_config(state: State<'_, AppState>) -> CmdResult<MeshConfig> {
    let config = state
        .config
        .lock()
        .map_err(|_| String::from("Config lock poisoned"))?;
    Ok(config.mesh.clone())
}

#[tauri::command]
pub fn save_mesh_config(state: State<'_, AppState>, mesh_config: MeshConfig) -> CmdResult<()> {
    let mut config = state
        .config
        .lock()
        .map_err(|_| String::from("Config lock poisoned"))?;
    config.mesh = mesh_config;
    config
        .save()
        .map_err(|e| format!("Failed to save config: {e}"))
}
