use std::path::Path;

use hivebear_core::types::{HardwareProfile, InferenceEngine, ModelFormat};

use crate::engine::{EngineRegistry, InferenceBackend};
use crate::error::{InferenceError, Result};

/// Detect model format from file extension and magic bytes.
pub fn detect_format(path: &Path) -> Result<ModelFormat> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "gguf" => Ok(ModelFormat::Gguf),
        "onnx" => Ok(ModelFormat::Onnx),
        "safetensors" => Ok(ModelFormat::SafeTensors),
        "mlx" | "npz" => Ok(ModelFormat::Mlx),
        _ => {
            // Try reading magic bytes for GGUF
            if let Ok(bytes) = std::fs::read(path).map(|b| b[..4].to_vec()) {
                if &bytes == b"GGUF" {
                    return Ok(ModelFormat::Gguf);
                }
            }
            Err(InferenceError::UnsupportedFormat(format!(
                "Cannot determine format for: {}",
                path.display()
            )))
        }
    }
}

/// Select the best available engine for a given model format and hardware profile.
///
/// Priority order:
/// 1. GPU-accelerated llama.cpp (for GGUF with GPU)
/// 2. CPU llama.cpp (for GGUF without GPU)
/// 3. Candle (pure Rust fallback for GGUF/SafeTensors)
/// 4. Error if no engine supports the format
pub fn select_engine<'a>(
    registry: &'a EngineRegistry,
    format: ModelFormat,
    profile: &HardwareProfile,
) -> Result<&'a dyn InferenceBackend> {
    let has_gpu = !profile.gpus.is_empty();
    let is_apple_silicon = profile.platform.os == "macos" && profile.platform.arch == "aarch64";

    // Build priority order based on hardware
    let priority: Vec<InferenceEngine> = match format {
        ModelFormat::Gguf => {
            if is_apple_silicon {
                // Apple Silicon: prefer llama.cpp (Metal), then Candle
                vec![InferenceEngine::LlamaCpp, InferenceEngine::Candle]
            } else if has_gpu {
                // GPU available: prefer llama.cpp (CUDA/Vulkan)
                vec![InferenceEngine::LlamaCpp, InferenceEngine::Candle]
            } else {
                // CPU only: llama.cpp still generally faster, Candle as fallback
                vec![InferenceEngine::LlamaCpp, InferenceEngine::Candle]
            }
        }
        ModelFormat::SafeTensors => {
            vec![InferenceEngine::Candle]
        }
        ModelFormat::Onnx => {
            vec![InferenceEngine::OnnxRuntime]
        }
        ModelFormat::Mlx => {
            vec![InferenceEngine::Mlx]
        }
    };

    for engine_id in &priority {
        if let Some(backend) = registry.get(*engine_id) {
            if backend.supported_formats().contains(&format) {
                tracing::info!(
                    engine = backend.name(),
                    format = %format,
                    has_gpu = has_gpu,
                    "Selected inference engine"
                );
                return Ok(backend);
            }
        }
    }

    // Last resort: find any backend that supports this format
    if let Some(backend) = registry.find_for_format(format) {
        return Ok(backend);
    }

    Err(InferenceError::NoEngineAvailable {
        format: format.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use hivebear_core::types::*;

    fn test_profile(os: &str, arch: &str, has_gpu: bool) -> HardwareProfile {
        let gb = 1024 * 1024 * 1024;
        HardwareProfile {
            cpu: CpuInfo {
                model_name: "Test CPU".into(),
                physical_cores: 8,
                logical_cores: 16,
                isa_extensions: vec!["AVX2".into()],
                cache_size_bytes: 16 * 1024 * 1024,
            },
            memory: MemoryInfo {
                total_bytes: 16 * gb,
                available_bytes: 12 * gb,
                estimated_bandwidth_gbps: 30.0,
            },
            gpus: if has_gpu {
                vec![GpuInfo {
                    name: "Test GPU".into(),
                    vram_bytes: 8 * gb,
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
                os: os.into(),
                arch: arch.into(),
                is_mobile: false,
                power_source: PowerSource::Ac,
            },
        }
    }

    #[test]
    fn test_detect_format_gguf() {
        assert_eq!(
            detect_format(Path::new("model.gguf")).unwrap(),
            ModelFormat::Gguf
        );
    }

    #[test]
    fn test_detect_format_safetensors() {
        assert_eq!(
            detect_format(Path::new("model.safetensors")).unwrap(),
            ModelFormat::SafeTensors
        );
    }

    #[test]
    fn test_detect_format_onnx() {
        assert_eq!(
            detect_format(Path::new("model.onnx")).unwrap(),
            ModelFormat::Onnx
        );
    }

    #[test]
    fn test_detect_format_unknown() {
        assert!(detect_format(Path::new("model.xyz")).is_err());
    }

    #[test]
    fn test_select_engine_gguf_with_gpu() {
        let registry = EngineRegistry::new();
        let profile = test_profile("linux", "x86_64", true);
        let result = select_engine(&registry, ModelFormat::Gguf, &profile);
        // Should find some engine (either llama.cpp or candle depending on features)
        assert!(result.is_ok());
    }

    #[test]
    fn test_select_engine_gguf_cpu_only() {
        let registry = EngineRegistry::new();
        let profile = test_profile("linux", "x86_64", false);
        let result = select_engine(&registry, ModelFormat::Gguf, &profile);
        assert!(result.is_ok());
    }

    #[test]
    fn test_select_engine_unsupported_format() {
        let registry = EngineRegistry::new();
        let profile = test_profile("linux", "x86_64", false);
        // MLX format is not supported by any compiled-in backend
        let result = select_engine(&registry, ModelFormat::Mlx, &profile);
        assert!(result.is_err());
    }
}
