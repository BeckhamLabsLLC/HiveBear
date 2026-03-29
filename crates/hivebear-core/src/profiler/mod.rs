mod cpu;
mod gpu;
mod memory;
mod platform;
mod storage;

use crate::types::HardwareProfile;

pub use cpu::detect_cpu;
pub use gpu::detect_gpus;
pub use memory::detect_memory;
pub use platform::detect_platform;
pub use storage::detect_storage;

/// Run a complete hardware profile of the current device.
///
/// On native platforms, this performs full hardware detection including GPU enumeration.
/// On WASM, this performs synchronous detection only (limited GPU info).
/// Use [`profile_async`] on WASM for WebGPU detection.
pub fn profile() -> HardwareProfile {
    tracing::info!("Starting hardware profiling...");

    let cpu = detect_cpu();
    tracing::debug!("CPU: {} ({} cores)", cpu.model_name, cpu.physical_cores);

    let memory = detect_memory();
    tracing::debug!(
        "Memory: {:.1} GB total, {:.1} GB available",
        memory.total_bytes as f64 / 1_073_741_824.0,
        memory.available_bytes as f64 / 1_073_741_824.0
    );

    let gpus = detect_gpus();
    tracing::debug!("GPUs found: {}", gpus.len());

    let storage = detect_storage();
    let platform = detect_platform();

    tracing::info!("Hardware profiling complete");

    HardwareProfile {
        cpu,
        memory,
        gpus,
        storage,
        platform,
    }
}

/// Async hardware profile that includes WebGPU detection.
///
/// On WASM, this performs async WebGPU adapter enumeration for accurate GPU info.
/// On native platforms, this is equivalent to [`profile`].
#[cfg(target_arch = "wasm32")]
pub async fn profile_async() -> HardwareProfile {
    let cpu = detect_cpu();
    let memory = detect_memory();
    let gpus = gpu::detect_gpus_browser().await;
    let storage = detect_storage();
    let platform = detect_platform();

    HardwareProfile {
        cpu,
        memory,
        gpus,
        storage,
        platform,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_profile() {
        let hw = profile();
        assert!(!hw.cpu.model_name.is_empty());
        assert!(hw.memory.total_bytes > 0);
        assert!(!hw.platform.os.is_empty());
    }
}
