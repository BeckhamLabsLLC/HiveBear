use crate::error::CmdResult;
use crate::state::AppState;
use serde::Serialize;
use tauri::State;

/// Device status for mobile mesh participation decisions.
#[derive(Debug, Clone, Serialize)]
pub struct DeviceStatus {
    /// Whether the device is currently charging.
    pub is_charging: bool,
    /// Battery level 0-100, or None on desktop.
    pub battery_percent: Option<u8>,
    /// Whether the device is on WiFi (vs cellular/ethernet).
    pub is_wifi: bool,
    /// Whether the device is a mobile device.
    pub is_mobile: bool,
    /// Thermal state: "nominal", "fair", "serious", "critical", or "unknown".
    pub thermal_state: String,
}

/// Returns the current device status for mesh participation gating.
///
/// On desktop, this returns sensible defaults (charging=true, wifi=true).
/// On mobile, this reads actual device state.
#[tauri::command]
pub fn get_device_status(state: State<'_, AppState>) -> CmdResult<DeviceStatus> {
    let profile = &state.profile;

    let (is_charging, battery_percent) = match &profile.platform.power_source {
        hivebear_core::types::PowerSource::Ac => (true, None),
        hivebear_core::types::PowerSource::Battery { charge_percent } => {
            (false, Some(*charge_percent))
        }
        hivebear_core::types::PowerSource::Unknown => (true, None), // Assume AC on unknown
    };

    Ok(DeviceStatus {
        is_charging,
        battery_percent,
        is_wifi: true, // TODO: Detect via tauri-plugin-network when available
        is_mobile: profile.platform.is_mobile,
        thermal_state: "nominal".to_string(), // TODO: Platform-specific thermal APIs
    })
}

/// Check whether mesh contribution is allowed given current device state
/// and user's MobileConfig preferences.
#[tauri::command]
pub fn can_contribute_to_mesh(state: State<'_, AppState>) -> CmdResult<MeshEligibility> {
    let config = state
        .config
        .lock()
        .map_err(|_| String::from("Config lock poisoned"))?;
    let profile = &state.profile;

    // Desktop devices are always eligible (unless mesh is disabled)
    if !profile.platform.is_mobile {
        return Ok(MeshEligibility {
            eligible: config.mesh.enabled,
            reasons: if config.mesh.enabled {
                vec![]
            } else {
                vec!["Mesh is disabled in settings".to_string()]
            },
        });
    }

    // Mobile: check MobileConfig constraints
    let mobile = &config.mobile;
    let mut reasons = Vec::new();

    if !mobile.background_mesh_enabled {
        reasons.push("Background mesh is disabled in settings".to_string());
    }

    if mobile.mesh_charging_only {
        if let hivebear_core::types::PowerSource::Battery { .. } = &profile.platform.power_source {
            reasons.push("Device is on battery (mesh requires charging)".to_string());
        }
    }

    if mobile.mesh_wifi_only {
        // TODO: actual WiFi detection
        // For now, assume WiFi on mobile unless we can detect otherwise
    }

    Ok(MeshEligibility {
        eligible: reasons.is_empty() && config.mesh.enabled,
        reasons,
    })
}

/// Whether the device is eligible for mesh contribution.
#[derive(Debug, Clone, Serialize)]
pub struct MeshEligibility {
    /// Whether the device is eligible to contribute.
    pub eligible: bool,
    /// Reasons why mesh contribution is blocked (empty if eligible).
    pub reasons: Vec<String>,
}
