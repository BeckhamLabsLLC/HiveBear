use crate::error::CmdResult;
use crate::state::AppState;
use hivebear_core::Config;
use tauri::State;

#[tauri::command]
pub fn get_config(state: State<'_, AppState>) -> CmdResult<Config> {
    let config = state
        .config
        .lock()
        .map_err(|_| String::from("Config lock poisoned"))?;
    Ok(config.clone())
}

#[tauri::command]
pub fn save_config(state: State<'_, AppState>, new_config: Config) -> CmdResult<()> {
    validate_config(&new_config)?;
    let mut config = state
        .config
        .lock()
        .map_err(|_| String::from("Config lock poisoned"))?;
    *config = new_config;
    config
        .save()
        .map_err(|e| format!("Failed to save config: {e}"))
}

/// Validate configuration values before saving.
fn validate_config(c: &Config) -> CmdResult<()> {
    if c.max_memory_usage < 0.1 || c.max_memory_usage > 1.0 {
        return Err("max_memory_usage must be between 0.1 and 1.0".into());
    }
    if c.min_tokens_per_sec < 0.0 || c.min_tokens_per_sec > 1000.0 {
        return Err("min_tokens_per_sec must be between 0 and 1000".into());
    }
    if c.default_context_length < 512 || c.default_context_length > 131_072 {
        return Err("default_context_length must be between 512 and 131072".into());
    }
    if c.top_n_recommendations == 0 || c.top_n_recommendations > 100 {
        return Err("top_n_recommendations must be between 1 and 100".into());
    }

    // Mesh config validation
    if c.mesh.port > 0 && c.mesh.port < 1024 {
        return Err("mesh port must be 0 (auto) or >= 1024".into());
    }
    if c.mesh.max_contribution_percent < 0.0 || c.mesh.max_contribution_percent > 1.0 {
        return Err("mesh max_contribution_percent must be between 0.0 and 1.0".into());
    }
    if c.mesh.min_reputation < 0.0 || c.mesh.min_reputation > 1.0 {
        return Err("mesh min_reputation must be between 0.0 and 1.0".into());
    }
    if c.mesh.verification_rate < 0.0 || c.mesh.verification_rate > 1.0 {
        return Err("mesh verification_rate must be between 0.0 and 1.0".into());
    }

    Ok(())
}
