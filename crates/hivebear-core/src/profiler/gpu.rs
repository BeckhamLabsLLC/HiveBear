use crate::types::{ComputeApi, GpuInfo};

#[cfg(not(target_arch = "wasm32"))]
pub fn detect_gpus() -> Vec<GpuInfo> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        flags: wgpu::InstanceFlags::empty(), // Disable validation layer warnings
        ..Default::default()
    });

    let adapters = instance.enumerate_adapters(wgpu::Backends::all());

    adapters
        .into_iter()
        .filter_map(|adapter| {
            let info = adapter.get_info();

            // Skip software/CPU renderers
            if info.device_type == wgpu::DeviceType::Cpu {
                return None;
            }

            let compute_api = match info.backend {
                wgpu::Backend::Vulkan => ComputeApi::Vulkan,
                wgpu::Backend::Metal => ComputeApi::Metal,
                wgpu::Backend::Dx12 => ComputeApi::DirectX12,
                wgpu::Backend::BrowserWebGpu => ComputeApi::WebGpu,
                _ => ComputeApi::None,
            };

            // wgpu doesn't directly expose VRAM, but we can get the max buffer size
            // as a rough proxy. For accurate VRAM, we'd need platform-specific queries.
            let limits = adapter.limits();
            let vram_bytes = estimate_vram(&info, &limits);

            Some(GpuInfo {
                name: info.name.clone(),
                vram_bytes,
                compute_api,
                driver_version: Some(info.driver.clone()),
            })
        })
        .collect()
}

#[cfg(not(target_arch = "wasm32"))]
fn estimate_vram(info: &wgpu::AdapterInfo, limits: &wgpu::Limits) -> u64 {
    // Try to use dedicated video memory reported by the adapter
    // wgpu doesn't expose this directly, so we use max_buffer_size as a lower bound
    // and known GPU models for better estimates.
    let max_buffer = limits.max_buffer_size;

    // For known NVIDIA GPUs, try to extract VRAM from the name
    let name = info.name.to_lowercase();
    if let Some(vram) = estimate_vram_from_name(&name) {
        return vram;
    }

    // Fallback: max_buffer_size is typically ~25% of VRAM
    // Use it as a rough estimate, with a minimum of the buffer size itself
    max_buffer.max(1024 * 1024 * 1024) // At least 1GB assumption for discrete GPUs
}

#[cfg(not(target_arch = "wasm32"))]
fn estimate_vram_from_name(name: &str) -> Option<u64> {
    const GB: u64 = 1024 * 1024 * 1024;

    // Common NVIDIA GPUs
    let nvidia_patterns: &[(&str, u64)] = &[
        ("4090", 24 * GB),
        ("4080", 16 * GB),
        ("4070 ti super", 16 * GB),
        ("4070 ti", 12 * GB),
        ("4070 super", 12 * GB),
        ("4070", 12 * GB),
        ("4060 ti", 16 * GB), // 16GB variant exists
        ("4060", 8 * GB),
        ("3090", 24 * GB),
        ("3080 ti", 12 * GB),
        ("3080", 10 * GB),
        ("3070 ti", 8 * GB),
        ("3070", 8 * GB),
        ("3060 ti", 8 * GB),
        ("3060", 12 * GB),
        ("2080 ti", 11 * GB),
        ("2080 super", 8 * GB),
        ("2080", 8 * GB),
        ("2070 super", 8 * GB),
        ("2070", 8 * GB),
        ("2060 super", 8 * GB),
        ("2060", 6 * GB),
        ("1080 ti", 11 * GB),
        ("1080", 8 * GB),
        ("1070 ti", 8 * GB),
        ("1070", 8 * GB),
        ("1060", 6 * GB),
        ("1050 ti", 4 * GB),
        ("1050", 2 * GB),
    ];

    for (pattern, vram) in nvidia_patterns {
        if name.contains(pattern) {
            return Some(*vram);
        }
    }

    // Common AMD GPUs
    let amd_patterns: &[(&str, u64)] = &[
        ("7900 xtx", 24 * GB),
        ("7900 xt", 20 * GB),
        ("7800 xt", 16 * GB),
        ("7700 xt", 12 * GB),
        ("7600", 8 * GB),
        ("6900 xt", 16 * GB),
        ("6800 xt", 16 * GB),
        ("6800", 16 * GB),
        ("6700 xt", 12 * GB),
        ("6600 xt", 8 * GB),
        ("6600", 8 * GB),
    ];

    for (pattern, vram) in amd_patterns {
        if name.contains(pattern) {
            return Some(*vram);
        }
    }

    None
}

#[cfg(target_arch = "wasm32")]
pub fn detect_gpus() -> Vec<GpuInfo> {
    // In WASM, we can't enumerate GPUs synchronously.
    // Return empty; use detect_gpus_browser() for async WebGPU detection.
    Vec::new()
}

/// Async WebGPU detection for browsers. Returns GPU info if WebGPU is available.
///
/// Uses JS interop to check `navigator.gpu` and request an adapter.
/// The web-sys `Gpu` / `GpuAdapter` types require many feature flags,
/// so we use raw `JsValue` reflection instead.
#[cfg(target_arch = "wasm32")]
pub async fn detect_gpus_browser() -> Vec<GpuInfo> {
    use wasm_bindgen_futures::JsFuture;

    let window = match web_sys::window() {
        Some(w) => w,
        None => return Vec::new(),
    };
    let navigator = window.navigator();

    // Check if navigator.gpu exists
    let gpu = match js_sys::Reflect::get(&navigator, &"gpu".into()) {
        Ok(v) if !v.is_undefined() && !v.is_null() => v,
        _ => return Vec::new(),
    };

    // Call gpu.requestAdapter()
    let request_adapter = match js_sys::Reflect::get(&gpu, &"requestAdapter".into()) {
        Ok(f) if f.is_function() => js_sys::Function::from(f),
        _ => return Vec::new(),
    };

    let promise = match request_adapter.call0(&gpu) {
        Ok(p) => js_sys::Promise::from(p),
        Err(_) => return Vec::new(),
    };

    let adapter = match JsFuture::from(promise).await {
        Ok(v) if !v.is_null() && !v.is_undefined() => v,
        _ => return Vec::new(),
    };

    // Read adapter info via reflection
    let info = js_sys::Reflect::get(&adapter, &"info".into())
        .ok()
        .and_then(|info_fn| {
            if info_fn.is_function() {
                // adapter.info is a getter in some implementations
                js_sys::Function::from(info_fn).call0(&adapter).ok()
            } else if !info_fn.is_undefined() {
                Some(info_fn)
            } else {
                None
            }
        });

    let name = info
        .as_ref()
        .and_then(|i| js_sys::Reflect::get(i, &"device".into()).ok())
        .and_then(|v| v.as_string())
        .unwrap_or_else(|| "WebGPU Device".to_string());

    // Get max buffer size from adapter.limits
    let max_buffer = js_sys::Reflect::get(&adapter, &"limits".into())
        .ok()
        .and_then(|limits| js_sys::Reflect::get(&limits, &"maxBufferSize".into()).ok())
        .and_then(|v| v.as_f64())
        .unwrap_or(256.0 * 1024.0 * 1024.0) as u64;

    vec![GpuInfo {
        name,
        vram_bytes: max_buffer,
        compute_api: ComputeApi::WebGpu,
        driver_version: None,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_gpus_does_not_crash() {
        // This may return empty on CI or headless servers
        let gpus = detect_gpus();
        for gpu in &gpus {
            assert!(!gpu.name.is_empty());
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_vram_estimation_known_gpus() {
        const GB: u64 = 1024 * 1024 * 1024;
        assert_eq!(
            estimate_vram_from_name("nvidia geforce rtx 4090"),
            Some(24 * GB)
        );
        assert_eq!(
            estimate_vram_from_name("nvidia geforce rtx 3060"),
            Some(12 * GB)
        );
        assert_eq!(
            estimate_vram_from_name("amd radeon rx 7900 xtx"),
            Some(24 * GB)
        );
        assert_eq!(estimate_vram_from_name("some unknown gpu"), None);
    }
}
