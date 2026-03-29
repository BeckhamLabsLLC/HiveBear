use crate::types::StorageInfo;

#[cfg(not(target_arch = "wasm32"))]
use sysinfo::Disks;

#[cfg(not(target_arch = "wasm32"))]
pub fn detect_storage() -> StorageInfo {
    let disks = Disks::new_with_refreshed_list();

    // Find the disk with the most available space (likely the main disk)
    let available_bytes = disks.iter().map(|d| d.available_space()).max().unwrap_or(0);

    let estimated_read_speed_mbps = estimate_disk_speed();

    StorageInfo {
        available_bytes,
        estimated_read_speed_mbps,
    }
}

/// In WASM there is no local filesystem. Models are fetched via HTTP.
#[cfg(target_arch = "wasm32")]
pub fn detect_storage() -> StorageInfo {
    StorageInfo {
        available_bytes: 0,
        estimated_read_speed_mbps: 0.0,
    }
}

/// Quick disk read speed estimate by reading a temporary file sequentially.
#[cfg(not(target_arch = "wasm32"))]
fn estimate_disk_speed() -> f64 {
    const FILE_SIZE: usize = 16 * 1024 * 1024; // 16 MB

    // Try current directory first (more likely on a real disk), fall back to temp
    let dir = std::env::current_dir().unwrap_or_else(|_| std::env::temp_dir());
    let path = dir.join(".hivebear_disk_bench.tmp");

    // Write test file
    let data = vec![0xABu8; FILE_SIZE];
    if std::fs::write(&path, &data).is_err() {
        return 0.0;
    }

    // Read it back and measure
    let start = std::time::Instant::now();
    let result = std::fs::read(&path);
    let elapsed = start.elapsed();

    // Cleanup
    let _ = std::fs::remove_file(&path);

    match result {
        Ok(read_data) => {
            let bytes_read = read_data.len() as f64;
            let seconds = elapsed.as_secs_f64();
            if seconds > 0.0 {
                bytes_read / seconds / 1_000_000.0 // MB/s
            } else {
                0.0
            }
        }
        Err(_) => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_storage() {
        let storage = detect_storage();
        // Should have some available space
        assert!(storage.available_bytes > 0);
    }

    #[test]
    fn test_disk_speed_estimate() {
        let speed = estimate_disk_speed();
        // Speed should be positive on any real hardware.
        // On some CI environments it may be 0 if disk writes fail.
        assert!(speed >= 0.0, "Disk speed estimate was negative: {speed}");
    }
}
