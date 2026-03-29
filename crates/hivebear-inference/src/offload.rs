use hivebear_core::types::HardwareProfile;

use crate::types::OffloadConfig;

/// Calculate the optimal layer offloading configuration based on:
/// - Model size in bytes
/// - Number of model layers
/// - Available GPU VRAM
/// - Available system RAM
/// - Optional mesh peer capacity
///
/// Strategy: fill GPU VRAM first (fastest), overflow to CPU RAM,
/// overflow to mesh peers if available, flag disk mmap as last resort.
pub fn calculate_offload(
    model_size_bytes: u64,
    num_layers: u32,
    profile: &HardwareProfile,
    context_length: u32,
) -> OffloadConfig {
    calculate_offload_with_mesh(model_size_bytes, num_layers, profile, context_length, None)
}

/// Extended offload calculation that accounts for mesh peer capacity.
///
/// When `mesh_capacity_bytes` is `Some(n)`, layers that don't fit locally
/// can be offloaded to mesh peers before falling back to disk mmap.
pub fn calculate_offload_with_mesh(
    model_size_bytes: u64,
    num_layers: u32,
    profile: &HardwareProfile,
    context_length: u32,
    mesh_capacity_bytes: Option<u64>,
) -> OffloadConfig {
    let total_vram: u64 = profile.gpus.iter().map(|g| g.vram_bytes).sum();
    let available_ram = (profile.memory.available_bytes as f64 * 0.85) as u64;

    if total_vram == 0 && mesh_capacity_bytes.is_none() {
        // No GPU, no mesh — everything on CPU
        return OffloadConfig {
            gpu_layers: Some(0),
            cpu_layers: Some(num_layers),
            disk_mmap: false,
            mesh_layers: None,
            auto: false,
        };
    }

    // Estimate memory overhead for KV cache
    let kv_overhead = (context_length as u64) * 256 * 1024;
    let usable_vram = total_vram.saturating_sub(kv_overhead);

    // Bytes per layer (approximate)
    let bytes_per_layer = if num_layers > 0 {
        model_size_bytes / num_layers as u64
    } else {
        model_size_bytes
    };

    // How many layers fit in GPU VRAM?
    let gpu_layers = if bytes_per_layer > 0 {
        (usable_vram / bytes_per_layer).min(num_layers as u64) as u32
    } else {
        num_layers
    };

    let mut remaining_layers = num_layers.saturating_sub(gpu_layers);

    // How many remaining layers fit in CPU RAM?
    let cpu_layers = if bytes_per_layer > 0 {
        let cpu_fit = (available_ram / bytes_per_layer) as u32;
        cpu_fit.min(remaining_layers)
    } else {
        remaining_layers
    };
    remaining_layers = remaining_layers.saturating_sub(cpu_layers);

    // How many remaining layers can be offloaded to mesh peers?
    let mesh_layers = if remaining_layers > 0 {
        if let Some(mesh_cap) = mesh_capacity_bytes {
            let mesh_fit = if bytes_per_layer > 0 {
                (mesh_cap / bytes_per_layer) as u32
            } else {
                remaining_layers
            };
            let assigned = mesh_fit.min(remaining_layers);
            remaining_layers = remaining_layers.saturating_sub(assigned);
            if assigned > 0 {
                Some(assigned)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    // If layers still remain, fall back to disk mmap
    let disk_mmap = remaining_layers > 0;

    OffloadConfig {
        gpu_layers: Some(gpu_layers),
        cpu_layers: Some(cpu_layers),
        disk_mmap,
        mesh_layers,
        auto: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hivebear_core::types::*;

    fn profile_with_vram(ram_gb: u64, vram_gb: u64) -> HardwareProfile {
        let gb = 1024 * 1024 * 1024;
        HardwareProfile {
            cpu: CpuInfo {
                model_name: "Test".into(),
                physical_cores: 8,
                logical_cores: 16,
                isa_extensions: vec![],
                cache_size_bytes: 0,
            },
            memory: MemoryInfo {
                total_bytes: ram_gb * gb,
                available_bytes: (ram_gb as f64 * 0.8) as u64 * gb,
                estimated_bandwidth_gbps: 30.0,
            },
            gpus: if vram_gb > 0 {
                vec![GpuInfo {
                    name: "Test GPU".into(),
                    vram_bytes: vram_gb * gb,
                    compute_api: ComputeApi::Vulkan,
                    driver_version: None,
                }]
            } else {
                vec![]
            },
            storage: StorageInfo {
                available_bytes: 100 * gb,
                estimated_read_speed_mbps: 500.0,
            },
            platform: PlatformInfo {
                os: "linux".into(),
                arch: "x86_64".into(),
                is_mobile: false,
                power_source: PowerSource::Ac,
            },
        }
    }

    #[test]
    fn test_no_gpu_all_cpu() {
        let profile = profile_with_vram(16, 0);
        let config = calculate_offload(4 * 1024 * 1024 * 1024, 32, &profile, 4096);
        assert_eq!(config.gpu_layers, Some(0));
        assert_eq!(config.cpu_layers, Some(32));
        assert!(!config.disk_mmap);
    }

    #[test]
    fn test_model_fits_in_vram() {
        let profile = profile_with_vram(16, 8);
        // 4GB model, 32 layers, 8GB VRAM — should fit entirely on GPU
        let config = calculate_offload(4 * 1024 * 1024 * 1024, 32, &profile, 4096);
        assert_eq!(config.gpu_layers, Some(32));
        assert_eq!(config.cpu_layers, Some(0));
        assert!(!config.disk_mmap);
    }

    #[test]
    fn test_model_split_gpu_cpu() {
        let profile = profile_with_vram(16, 4);
        // 8GB model, 32 layers, 4GB VRAM — should split
        let config = calculate_offload(8 * 1024 * 1024 * 1024, 32, &profile, 4096);
        assert!(config.gpu_layers.unwrap() > 0);
        assert!(config.gpu_layers.unwrap() < 32);
        assert!(config.cpu_layers.unwrap() > 0);
        assert!(!config.disk_mmap);
    }

    #[test]
    fn test_huge_model_needs_disk() {
        let profile = profile_with_vram(8, 4);
        // 100GB model, 80 layers, 4GB VRAM + 8GB RAM — needs disk mmap
        let config = calculate_offload(100 * 1024 * 1024 * 1024, 80, &profile, 4096);
        assert!(config.disk_mmap);
    }
}
