use crate::types::{PlatformInfo, PowerSource};

#[cfg(not(target_arch = "wasm32"))]
pub fn detect_platform() -> PlatformInfo {
    let os = detect_os();
    let arch = std::env::consts::ARCH.to_string();
    let is_mobile = cfg!(target_os = "android") || cfg!(target_os = "ios");
    let power_source = detect_power_source();

    PlatformInfo {
        os,
        arch,
        is_mobile,
        power_source,
    }
}

#[cfg(target_arch = "wasm32")]
pub fn detect_platform() -> PlatformInfo {
    // Detect mobile via user agent
    let is_mobile = web_sys::window()
        .and_then(|w| w.navigator().user_agent().ok())
        .map(|ua| {
            let ua = ua.to_lowercase();
            ua.contains("mobile") || ua.contains("android") || ua.contains("iphone")
        })
        .unwrap_or(false);

    PlatformInfo {
        os: "browser".to_string(),
        arch: "wasm32".to_string(),
        is_mobile,
        power_source: PowerSource::Unknown,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn detect_os() -> String {
    #[cfg(target_os = "linux")]
    {
        "linux".to_string()
    }
    #[cfg(target_os = "macos")]
    {
        "macos".to_string()
    }
    #[cfg(target_os = "windows")]
    {
        "windows".to_string()
    }
    #[cfg(target_os = "android")]
    {
        "android".to_string()
    }
    #[cfg(target_os = "ios")]
    {
        "ios".to_string()
    }
    #[cfg(not(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "windows",
        target_os = "android",
        target_os = "ios"
    )))]
    {
        std::env::consts::OS.to_string()
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn detect_power_source() -> PowerSource {
    #[cfg(target_os = "linux")]
    {
        // Try reading from /sys/class/power_supply/
        if let Ok(entries) = std::fs::read_dir("/sys/class/power_supply") {
            for entry in entries.flatten() {
                let type_path = entry.path().join("type");
                if let Ok(psu_type) = std::fs::read_to_string(type_path) {
                    if psu_type.trim() == "Battery" {
                        let status_path = entry.path().join("status");
                        let capacity_path = entry.path().join("capacity");

                        let charge_percent = std::fs::read_to_string(capacity_path)
                            .ok()
                            .and_then(|s| s.trim().parse::<u8>().ok())
                            .unwrap_or(0);

                        if let Ok(status) = std::fs::read_to_string(status_path) {
                            let status = status.trim();
                            if status == "Discharging" {
                                return PowerSource::Battery { charge_percent };
                            } else {
                                // Charging or Full — on AC power
                                return PowerSource::Ac;
                            }
                        }

                        return PowerSource::Battery { charge_percent };
                    }
                }
            }
        }
        // No battery found — desktop, likely on AC
        PowerSource::Ac
    }

    #[cfg(not(target_os = "linux"))]
    {
        PowerSource::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_platform() {
        let platform = detect_platform();
        assert!(!platform.os.is_empty());
        assert!(!platform.arch.is_empty());
    }
}
