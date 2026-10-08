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
    /// Whether the mesh node is listening locally.
    ///
    /// This is NOT the same as being on the hive — see `registered`. The UI
    /// used to render "Connected to Hive" off this flag alone, which stayed
    /// true while the coordination server was unreachable.
    pub running: bool,
    /// Whether the coordination server has acknowledged this node.
    pub registered: bool,
    /// Number of directly connected peers.
    pub peer_count: usize,
    /// The node's hex-encoded public key (if running).
    pub node_id: Option<String>,
    /// Why the last start attempt failed, if it did.
    pub last_error: Option<String>,
}

/// Start the mesh node: register with the coordination server and begin heartbeats.
///
/// Clicking Join is the opt-in. The mesh is off by default (so a first
/// launch never binds a socket or raises a firewall prompt), so joining also
/// turns it on and remembers to rejoin at the next launch.
#[tauri::command]
pub fn join_mesh(state: State<'_, AppState>) -> CmdResult<MeshConnectionStatus> {
    set_mesh_opt_in(&state, true)?;
    state.start_mesh()?;
    get_mesh_connection_status(state)
}

/// Stop the mesh node: deregister and disconnect, and stop rejoining at launch.
#[tauri::command]
pub async fn leave_mesh(state: State<'_, AppState>) -> CmdResult<()> {
    state.stop_mesh().await?;
    set_mesh_opt_in(&state, false)
}

/// Persist whether the user wants to be on the mesh.
///
/// Leaving keeps `enabled` as it is (that is the Settings switch) and only
/// clears `auto_join`; joining sets both.
fn set_mesh_opt_in(state: &AppState, joined: bool) -> CmdResult<()> {
    let mut config = state
        .config
        .lock()
        .map_err(|_| String::from("Config lock poisoned"))?;
    let enabled = config.mesh.enabled || joined;
    if config.mesh.enabled == enabled && config.mesh.auto_join == joined {
        return Ok(());
    }
    config.mesh.enabled = enabled;
    config.mesh.auto_join = joined;
    config
        .save()
        .map_err(|e| format!("Failed to save config: {e}"))
}

/// Get the live mesh connection status.
#[tauri::command]
pub fn get_mesh_connection_status(state: State<'_, AppState>) -> CmdResult<MeshConnectionStatus> {
    let last_error = state
        .mesh_error
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone();
    let node = state.mesh_node.lock().unwrap_or_else(|e| e.into_inner());
    match node.as_ref() {
        Some(n) => Ok(MeshConnectionStatus {
            running: n.is_running(),
            registered: n.is_registered(),
            peer_count: n.peer_count(),
            node_id: Some(n.local_id.to_hex()),
            last_error,
        }),
        None => Ok(MeshConnectionStatus {
            running: false,
            registered: false,
            peer_count: 0,
            node_id: None,
            last_error,
        }),
    }
}
