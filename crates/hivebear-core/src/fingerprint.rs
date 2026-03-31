use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

use crate::types::{ComputeApi, GpuInfo, HardwareProfile};

/// Privacy-preserving GPU capability classification based on VRAM.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum GpuClass {
    None,
    Low,
    Mid,
    High,
    Ultra,
}

impl GpuClass {
    /// Ordinal position for similarity distance calculation.
    fn ordinal(self) -> i8 {
        match self {
            GpuClass::None => 0,
            GpuClass::Low => 1,
            GpuClass::Mid => 2,
            GpuClass::High => 3,
            GpuClass::Ultra => 4,
        }
    }
}

impl fmt::Display for GpuClass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuClass::None => write!(f, "none"),
            GpuClass::Low => write!(f, "low"),
            GpuClass::Mid => write!(f, "mid"),
            GpuClass::High => write!(f, "high"),
            GpuClass::Ultra => write!(f, "ultra"),
        }
    }
}

/// Anonymized hardware fingerprint using bucketed attributes.
///
/// All values are coarsened into broad categories so that individual devices
/// cannot be identified, while still allowing meaningful hardware similarity
/// comparisons across community benchmark submissions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HardwareFingerprint {
    /// SHA-256 hash of the concatenated bucketed attributes.
    pub hash: String,
    /// CPU physical cores bucketed: {2, 4, 8, 16, 32, 64}.
    pub cpu_core_bucket: u8,
    /// Total RAM bucketed to nearest power-of-2 in GB: {2, 4, 8, 16, 32, 64, 128, 256}.
    pub ram_gb_bucket: u16,
    /// GPU capability class derived from total VRAM.
    pub gpu_class: GpuClass,
    /// GPU VRAM bucketed in GB: {0, 4, 6, 8, 12, 16, 24, 48}.
    pub gpu_vram_gb_bucket: u8,
    /// Primary compute API (Cuda, Metal, Vulkan, etc.).
    pub compute_api: ComputeApi,
    /// Operating system (lowercase): "linux", "macos", "windows", "browser".
    pub platform_os: String,
    /// CPU architecture: "x86_64", "aarch64", "wasm32".
    pub platform_arch: String,
}

impl HardwareFingerprint {
    /// Create an anonymized fingerprint from a full hardware profile.
    pub fn from_profile(profile: &HardwareProfile) -> Self {
        let cpu_core_bucket = bucket_cores(profile.cpu.physical_cores);
        let ram_gb_bucket = bucket_ram(profile.memory.total_bytes);
        let (gpu_class, gpu_vram_gb_bucket) = classify_gpu(&profile.gpus);
        let compute_api = profile
            .gpus
            .first()
            .map(|g| g.compute_api)
            .unwrap_or(ComputeApi::None);
        let platform_os = profile.platform.os.to_lowercase();
        let platform_arch = profile.platform.arch.to_lowercase();

        let hash_input = format!(
            "{}:{}:{}:{}:{}:{}:{}",
            cpu_core_bucket,
            ram_gb_bucket,
            gpu_class,
            gpu_vram_gb_bucket,
            compute_api,
            platform_os,
            platform_arch
        );

        let mut hasher = Sha256::new();
        hasher.update(hash_input.as_bytes());
        let hash = format!("{:x}", hasher.finalize());

        Self {
            hash,
            cpu_core_bucket,
            ram_gb_bucket,
            gpu_class,
            gpu_vram_gb_bucket,
            compute_api,
            platform_os,
            platform_arch,
        }
    }
}

/// Compute similarity between two hardware fingerprints (0.0 = no match, 1.0 = identical).
///
/// Uses weighted scoring across bucketed dimensions.
pub fn similarity(a: &HardwareFingerprint, b: &HardwareFingerprint) -> f64 {
    // GPU class: 0.30 weight
    let gpu_score = {
        let dist = (a.gpu_class.ordinal() - b.gpu_class.ordinal()).unsigned_abs();
        match dist {
            0 => 1.0,
            1 => 0.5,
            _ => 0.0,
        }
    };

    // Compute API: 0.20 weight
    let api_score = if a.compute_api == b.compute_api {
        1.0
    } else {
        0.0
    };

    // RAM bucket: 0.20 weight (log2 distance)
    let ram_score = {
        let a_log = (a.ram_gb_bucket as f64).log2();
        let b_log = (b.ram_gb_bucket as f64).log2();
        let dist = (a_log - b_log).abs();
        if dist < 0.01 {
            1.0
        } else if dist <= 1.0 {
            0.7
        } else if dist <= 2.0 {
            0.3
        } else {
            0.0
        }
    };

    // CPU core bucket: 0.15 weight (log2 distance)
    let cpu_score = {
        let a_log = (a.cpu_core_bucket as f64).log2();
        let b_log = (b.cpu_core_bucket as f64).log2();
        let dist = (a_log - b_log).abs();
        if dist < 0.01 {
            1.0
        } else if dist <= 1.0 {
            0.5
        } else {
            0.0
        }
    };

    // Platform arch: 0.10 weight
    let arch_score = if a.platform_arch == b.platform_arch {
        1.0
    } else {
        0.0
    };

    // Platform OS: 0.05 weight
    let os_score = if a.platform_os == b.platform_os {
        1.0
    } else {
        0.5
    };

    let total: f64 = gpu_score * 0.30
        + api_score * 0.20
        + ram_score * 0.20
        + cpu_score * 0.15
        + arch_score * 0.10
        + os_score * 0.05;

    total.clamp(0.0, 1.0)
}

/// Bucket physical core count into broad categories.
fn bucket_cores(cores: u32) -> u8 {
    match cores {
        0..=2 => 2,
        3..=4 => 4,
        5..=8 => 8,
        9..=16 => 16,
        17..=32 => 32,
        _ => 64,
    }
}

/// Bucket total RAM (bytes) to nearest power-of-2 in GB.
fn bucket_ram(bytes: u64) -> u16 {
    let gb = bytes / (1024 * 1024 * 1024);
    match gb {
        0..=3 => 2,
        4..=6 => 4,
        7..=12 => 8,
        13..=24 => 16,
        25..=48 => 32,
        49..=96 => 64,
        97..=192 => 128,
        _ => 256,
    }
}

/// Classify GPU capability from VRAM and return (class, vram_gb_bucket).
fn classify_gpu(gpus: &[GpuInfo]) -> (GpuClass, u8) {
    if gpus.is_empty() {
        return (GpuClass::None, 0);
    }

    // Use the GPU with the most VRAM.
    let max_vram_bytes = gpus.iter().map(|g| g.vram_bytes).max().unwrap_or(0);
    let vram_gb = max_vram_bytes / (1024 * 1024 * 1024);

    let class = match vram_gb {
        0 => GpuClass::None,
        1..=5 => GpuClass::Low,
        6..=11 => GpuClass::Mid,
        12..=23 => GpuClass::High,
        _ => GpuClass::Ultra,
    };

    let bucket = match vram_gb {
        0 => 0,
        1..=5 => 4,
        6..=7 => 6,
        8..=11 => 8,
        12..=15 => 12,
        16..=23 => 16,
        24..=47 => 24,
        _ => 48,
    };

    (class, bucket)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    fn make_profile(
        cores: u32,
        ram_gb: u64,
        gpu_vram_gb: u64,
        compute: ComputeApi,
        os: &str,
        arch: &str,
    ) -> HardwareProfile {
        let gb = 1024 * 1024 * 1024;
        HardwareProfile {
            cpu: CpuInfo {
                model_name: "Test CPU".into(),
                physical_cores: cores,
                logical_cores: cores * 2,
                isa_extensions: vec![],
                cache_size_bytes: 8 * 1024 * 1024,
            },
            memory: MemoryInfo {
                total_bytes: ram_gb * gb,
                available_bytes: (ram_gb as f64 * 0.8) as u64 * gb,
                estimated_bandwidth_gbps: 30.0,
            },
            gpus: if gpu_vram_gb > 0 {
                vec![GpuInfo {
                    name: "Test GPU".into(),
                    vram_bytes: gpu_vram_gb * gb,
                    compute_api: compute,
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
                os: os.into(),
                arch: arch.into(),
                is_mobile: false,
                power_source: PowerSource::Ac,
            },
        }
    }

    #[test]
    fn test_bucketing_cores() {
        assert_eq!(bucket_cores(1), 2);
        assert_eq!(bucket_cores(2), 2);
        assert_eq!(bucket_cores(3), 4);
        assert_eq!(bucket_cores(4), 4);
        assert_eq!(bucket_cores(6), 8);
        assert_eq!(bucket_cores(8), 8);
        assert_eq!(bucket_cores(12), 16);
        assert_eq!(bucket_cores(16), 16);
        assert_eq!(bucket_cores(24), 32);
        assert_eq!(bucket_cores(32), 32);
        assert_eq!(bucket_cores(64), 64);
        assert_eq!(bucket_cores(128), 64);
    }

    #[test]
    fn test_bucketing_ram() {
        assert_eq!(bucket_ram(2 * 1024 * 1024 * 1024), 2);
        assert_eq!(bucket_ram(4 * 1024 * 1024 * 1024), 4);
        assert_eq!(bucket_ram(8 * 1024 * 1024 * 1024), 8);
        assert_eq!(bucket_ram(16 * 1024 * 1024 * 1024), 16);
        assert_eq!(bucket_ram(32 * 1024 * 1024 * 1024), 32);
        assert_eq!(bucket_ram(64 * 1024 * 1024 * 1024), 64);
        assert_eq!(bucket_ram(128 * 1024 * 1024 * 1024), 128);
        assert_eq!(bucket_ram(512 * 1024 * 1024 * 1024), 256);
    }

    #[test]
    fn test_gpu_classification() {
        assert_eq!(classify_gpu(&[]), (GpuClass::None, 0));

        let gb = 1024 * 1024 * 1024;
        let make_gpu = |vram_gb: u64| GpuInfo {
            name: "GPU".into(),
            vram_bytes: vram_gb * gb,
            compute_api: ComputeApi::Cuda,
            driver_version: None,
        };

        assert_eq!(classify_gpu(&[make_gpu(4)]), (GpuClass::Low, 4));
        assert_eq!(classify_gpu(&[make_gpu(8)]), (GpuClass::Mid, 8));
        assert_eq!(classify_gpu(&[make_gpu(12)]), (GpuClass::High, 12));
        assert_eq!(classify_gpu(&[make_gpu(16)]), (GpuClass::High, 16));
        assert_eq!(classify_gpu(&[make_gpu(24)]), (GpuClass::Ultra, 24));
        assert_eq!(classify_gpu(&[make_gpu(48)]), (GpuClass::Ultra, 48));
    }

    #[test]
    fn test_fingerprint_deterministic() {
        let profile = make_profile(8, 16, 8, ComputeApi::Cuda, "linux", "x86_64");
        let fp1 = HardwareFingerprint::from_profile(&profile);
        let fp2 = HardwareFingerprint::from_profile(&profile);
        assert_eq!(fp1, fp2);
        assert_eq!(fp1.hash, fp2.hash);
    }

    #[test]
    fn test_fingerprint_differs() {
        let profile_a = make_profile(8, 16, 8, ComputeApi::Cuda, "linux", "x86_64");
        let profile_b = make_profile(4, 32, 24, ComputeApi::Metal, "macos", "aarch64");
        let fp_a = HardwareFingerprint::from_profile(&profile_a);
        let fp_b = HardwareFingerprint::from_profile(&profile_b);
        assert_ne!(fp_a.hash, fp_b.hash);
        assert_ne!(fp_a.gpu_class, fp_b.gpu_class);
    }

    #[test]
    fn test_similarity_exact_match() {
        let profile = make_profile(8, 16, 8, ComputeApi::Cuda, "linux", "x86_64");
        let fp = HardwareFingerprint::from_profile(&profile);
        let score = similarity(&fp, &fp);
        assert!(
            (score - 1.0).abs() < f64::EPSILON,
            "Exact match should be 1.0, got {score}"
        );
    }

    #[test]
    fn test_similarity_no_match() {
        let fp_a = HardwareFingerprint::from_profile(&make_profile(
            2,
            4,
            0,
            ComputeApi::None,
            "linux",
            "x86_64",
        ));
        let fp_b = HardwareFingerprint::from_profile(&make_profile(
            64,
            256,
            48,
            ComputeApi::Cuda,
            "windows",
            "aarch64",
        ));
        let score = similarity(&fp_a, &fp_b);
        assert!(
            score < 0.3,
            "Completely different hardware should score low, got {score}"
        );
    }

    #[test]
    fn test_similarity_partial() {
        // Same GPU class and compute API, different RAM and cores
        let fp_a = HardwareFingerprint::from_profile(&make_profile(
            8,
            16,
            8,
            ComputeApi::Cuda,
            "linux",
            "x86_64",
        ));
        let fp_b = HardwareFingerprint::from_profile(&make_profile(
            16,
            32,
            12,
            ComputeApi::Cuda,
            "linux",
            "x86_64",
        ));
        let score = similarity(&fp_a, &fp_b);
        assert!(
            score > 0.5 && score < 1.0,
            "Partially similar hardware should score between 0.5 and 1.0, got {score}"
        );
    }
}
