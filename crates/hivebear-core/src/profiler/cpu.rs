use crate::types::CpuInfo;

#[cfg(not(target_arch = "wasm32"))]
use sysinfo::System;

#[cfg(not(target_arch = "wasm32"))]
pub fn detect_cpu() -> CpuInfo {
    let sys = System::new_all();
    let cpus = sys.cpus();

    let model_name = cpus
        .first()
        .map(|c| c.brand().to_string())
        .unwrap_or_else(|| "Unknown CPU".to_string());

    let physical_cores = sys.physical_core_count().unwrap_or(1) as u32;
    let logical_cores = cpus.len() as u32;

    let isa_extensions = detect_isa_extensions();
    let cache_size_bytes = detect_cache_size();

    CpuInfo {
        model_name,
        physical_cores,
        logical_cores,
        isa_extensions,
        cache_size_bytes,
    }
}

#[cfg(target_arch = "wasm32")]
pub fn detect_cpu() -> CpuInfo {
    let logical_cores = web_sys::window()
        .map(|w| w.navigator().hardware_concurrency() as u32)
        .unwrap_or(1);

    CpuInfo {
        model_name: "Browser".to_string(),
        physical_cores: logical_cores,
        logical_cores,
        isa_extensions: Vec::new(),
        cache_size_bytes: 0,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn detect_isa_extensions() -> Vec<String> {
    let mut extensions = Vec::new();

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse4.1") {
            extensions.push("SSE4.1".to_string());
        }
        if std::arch::is_x86_feature_detected!("sse4.2") {
            extensions.push("SSE4.2".to_string());
        }
        if std::arch::is_x86_feature_detected!("avx") {
            extensions.push("AVX".to_string());
        }
        if std::arch::is_x86_feature_detected!("avx2") {
            extensions.push("AVX2".to_string());
        }
        if std::arch::is_x86_feature_detected!("fma") {
            extensions.push("FMA".to_string());
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            extensions.push("AVX-512".to_string());
        }
        if std::arch::is_x86_feature_detected!("f16c") {
            extensions.push("F16C".to_string());
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is mandatory on AArch64
        extensions.push("NEON".to_string());

        #[cfg(target_feature = "dotprod")]
        extensions.push("DOTPROD".to_string());

        #[cfg(target_feature = "fp16")]
        extensions.push("FP16".to_string());
    }

    extensions
}

#[cfg(not(target_arch = "wasm32"))]
fn detect_cache_size() -> u64 {
    // Try to read L3 cache size from sysfs on Linux
    #[cfg(target_os = "linux")]
    {
        if let Ok(entries) = std::fs::read_dir("/sys/devices/system/cpu/cpu0/cache") {
            let mut max_cache = 0u64;
            for entry in entries.flatten() {
                let size_path = entry.path().join("size");
                if let Ok(size_str) = std::fs::read_to_string(size_path) {
                    let size_str = size_str.trim();
                    if let Some(kb_str) = size_str.strip_suffix('K') {
                        if let Ok(kb) = kb_str.parse::<u64>() {
                            max_cache = max_cache.max(kb * 1024);
                        }
                    } else if let Some(mb_str) = size_str.strip_suffix('M') {
                        if let Ok(mb) = mb_str.parse::<u64>() {
                            max_cache = max_cache.max(mb * 1024 * 1024);
                        }
                    }
                }
            }
            if max_cache > 0 {
                return max_cache;
            }
        }
    }

    // Fallback: unknown
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_cpu() {
        let cpu = detect_cpu();
        assert!(!cpu.model_name.is_empty());
        assert!(cpu.physical_cores >= 1);
        assert!(cpu.logical_cores >= cpu.physical_cores);
    }

    #[test]
    fn test_detect_isa_extensions() {
        let extensions = detect_isa_extensions();
        // On x86_64, we should have at least SSE4
        #[cfg(target_arch = "x86_64")]
        assert!(!extensions.is_empty());
        // On aarch64, NEON is mandatory
        #[cfg(target_arch = "aarch64")]
        assert!(extensions.contains(&"NEON".to_string()));
    }
}
