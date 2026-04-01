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

// ── Mesh lifecycle commands ────────────────────────────────────────

/// Live connection status (vs MeshStatus which is just config).
#[derive(Serialize)]
pub struct MeshConnectionStatus {
    /// Whether the mesh node process is running.
    pub running: bool,
    /// Number of directly connected peers.
    pub peer_count: usize,
    /// The node's hex-encoded public key (if running).
    pub node_id: Option<String>,
}

/// Start the mesh node: register with the coordination server and begin heartbeats.
#[tauri::command]
pub fn join_mesh(state: State<'_, AppState>) -> CmdResult<MeshConnectionStatus> {
    state.start_mesh()?;
    get_mesh_connection_status(state)
}

/// Stop the mesh node: deregister and disconnect.
#[tauri::command]
pub async fn leave_mesh(state: State<'_, AppState>) -> CmdResult<()> {
    state.stop_mesh().await
}

/// Get the live mesh connection status.
#[tauri::command]
pub fn get_mesh_connection_status(state: State<'_, AppState>) -> CmdResult<MeshConnectionStatus> {
    let node = state.mesh_node.lock().unwrap();
    match node.as_ref() {
        Some(n) => Ok(MeshConnectionStatus {
            running: n.is_running(),
            peer_count: n.peer_count(),
            node_id: Some(n.local_id.to_hex()),
        }),
        None => Ok(MeshConnectionStatus {
            running: false,
            peer_count: 0,
            node_id: None,
        }),
    }
}
