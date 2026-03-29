use crate::types::MemoryInfo;

#[cfg(not(target_arch = "wasm32"))]
use sysinfo::System;

#[cfg(not(target_arch = "wasm32"))]
pub fn detect_memory() -> MemoryInfo {
    let sys = System::new_all();

    let total_bytes = sys.total_memory();
    let available_bytes = sys.available_memory();
    let estimated_bandwidth_gbps = estimate_bandwidth();

    MemoryInfo {
        total_bytes,
        available_bytes,
        estimated_bandwidth_gbps,
    }
}

#[cfg(target_arch = "wasm32")]
pub fn detect_memory() -> MemoryInfo {
    // navigator.deviceMemory returns memory in GB (Chrome/Edge only, returns 0 elsewhere)
    let device_memory_gb = js_sys::Reflect::get(
        &web_sys::window().unwrap().navigator(),
        &wasm_bindgen::JsValue::from_str("deviceMemory"),
    )
    .ok()
    .and_then(|v| v.as_f64())
    .unwrap_or(4.0); // Conservative default

    let total_bytes = (device_memory_gb * 1_073_741_824.0) as u64;
    // Assume 60% available in browser context
    let available_bytes = (total_bytes as f64 * 0.6) as u64;
    // Estimate bandwidth by device class: desktop ~25 GB/s, mobile ~10 GB/s
    let estimated_bandwidth_gbps = if device_memory_gb >= 8.0 { 25.0 } else { 10.0 };

    MemoryInfo {
        total_bytes,
        available_bytes,
        estimated_bandwidth_gbps,
    }
}

/// Estimate memory bandwidth by performing a sequential read benchmark.
/// Allocates a buffer and measures throughput reading it sequentially.
#[cfg(not(target_arch = "wasm32"))]
fn estimate_bandwidth() -> f64 {
    const BUFFER_SIZE: usize = 64 * 1024 * 1024; // 64 MB
    const ITERATIONS: usize = 4;

    let buffer: Vec<u8> = vec![1u8; BUFFER_SIZE];
    let mut sink: u64 = 0;

    // Warmup
    for chunk in buffer.chunks(64) {
        sink = sink.wrapping_add(chunk[0] as u64);
    }

    let start = std::time::Instant::now();
    for _ in 0..ITERATIONS {
        for chunk in buffer.chunks(64) {
            sink = sink.wrapping_add(chunk[0] as u64);
        }
    }
    let elapsed = start.elapsed();

    // Prevent optimization of sink
    std::hint::black_box(sink);

    let bytes_read = (BUFFER_SIZE * ITERATIONS) as f64;
    let seconds = elapsed.as_secs_f64();

    if seconds > 0.0 {
        let bytes_per_sec = bytes_read / seconds;
        bytes_per_sec / 1_000_000_000.0 // Convert to GB/s
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_memory() {
        let mem = detect_memory();
        assert!(mem.total_bytes > 0);
        assert!(mem.available_bytes > 0);
        assert!(mem.available_bytes <= mem.total_bytes);
    }

    #[test]
    fn test_bandwidth_estimate() {
        let bw = estimate_bandwidth();
        // Should return some positive value on any real hardware
        assert!(bw > 0.0, "Bandwidth estimate was {bw}");
        // Sanity: should be less than 1 TB/s
        assert!(
            bw < 1000.0,
            "Bandwidth estimate was unrealistically high: {bw}"
        );
    }
}
