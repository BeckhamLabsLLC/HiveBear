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
            // Read just the magic bytes.
            //
            // This used to be `std::fs::read(path).map(|b| b[..4].to_vec())`,
            // which pulls the *entire* file into memory to look at four bytes
            // — an OOM on a 40 GB model with an unrecognised extension — and
            // panics with index-out-of-bounds on anything shorter than four
            // bytes, such as a truncated download or a zero-byte placeholder.
            if read_magic(path) == Some(*b"GGUF") {
                return Ok(ModelFormat::Gguf);
            }
            Err(InferenceError::UnsupportedFormat(format!(
                "Cannot determine format for: {}",
                path.display()
            )))
        }
    }
}

/// First four bytes of `path`, or `None` if it cannot be read or is shorter.
fn read_magic(path: &Path) -> Option<[u8; 4]> {
    use std::io::Read;

    let mut file = std::fs::File::open(path).ok()?;
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic).ok()?;
    Some(magic)
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
    select_engine_for(registry, format, profile, false)
}

/// Pick a backend, optionally restricted to ones that can serve a pipeline
/// stage.
///
/// When `needs_pipeline` is set, backends that cannot honour
/// `LoadConfig::pipeline_stage` are skipped entirely rather than selected and
/// then failing on the first `forward_partial`. On this path llama.cpp is not
/// a candidate at all: it ignores the stage config and loads the whole model.
pub fn select_engine_for<'a>(
    registry: &'a EngineRegistry,
    format: ModelFormat,
    profile: &HardwareProfile,
    needs_pipeline: bool,
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
            if !backend.supported_formats().contains(&format) {
                continue;
            }
            if needs_pipeline && !backend.supports_pipeline() {
                tracing::debug!(
                    engine = backend.name(),
                    "Skipping: cannot serve a pipeline stage"
                );
                continue;
            }
            tracing::info!(
                engine = backend.name(),
                format = %format,
                has_gpu = has_gpu,
                pipeline = needs_pipeline,
                "Selected inference engine"
            );
            return Ok(backend);
        }
    }

    // A pipeline stage has a hard requirement; do not fall back to a backend
    // that would silently load the whole model, and never to the mesh (this
    // node *is* the mesh worker).
    if needs_pipeline {
        return Err(InferenceError::NoEngineAvailable {
            format: format!(
                "{format} with pipeline-parallel support (only Candle implements                  forward_partial; enable the `candle` feature)"
            ),
        });
    }

    // Last resort: find any backend that supports this format
    if let Some(backend) = registry.find_for_format(format) {
        return Ok(backend);
    }

    // Final fallback: mesh distributed inference (if registered and available)
    if let Some(mesh) = registry.get(InferenceEngine::Mesh) {
        if mesh.supported_formats().contains(&format) {
            tracing::info!(
                format = %format,
                "No local engine available, routing to mesh"
            );
            return Ok(mesh);
        }
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

    /// A backend that serves GGUF but cannot run a pipeline stage — the
    /// shape llama.cpp has.
    struct NoPipelineBackend;

    #[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
    #[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
    impl crate::engine::InferenceBackend for NoPipelineBackend {
        fn engine_id(&self) -> InferenceEngine {
            InferenceEngine::LlamaCpp
        }
        fn name(&self) -> &str {
            "NoPipeline"
        }
        fn supported_formats(&self) -> &[ModelFormat] {
            &[ModelFormat::Gguf]
        }
        fn is_available(&self) -> bool {
            true
        }
        async fn load_model(
            &self,
            path: &std::path::Path,
            _config: &crate::types::LoadConfig,
        ) -> crate::error::Result<crate::types::ModelHandle> {
            Ok(crate::types::ModelHandle::new(
                path.to_path_buf(),
                self.engine_id(),
            ))
        }
        async fn generate(
            &self,
            _handle: &crate::types::ModelHandle,
            _req: &crate::types::GenerateRequest,
        ) -> crate::error::Result<crate::types::GenerateResponse> {
            unimplemented!("not exercised by selection tests")
        }
        fn stream(
            &self,
            _handle: &crate::types::ModelHandle,
            _req: &crate::types::GenerateRequest,
        ) -> crate::engine::TokenStream {
            unimplemented!("not exercised by selection tests")
        }
        async fn unload(&self, _handle: &crate::types::ModelHandle) -> crate::error::Result<()> {
            Ok(())
        }
    }

    /// The regression: a pipeline stage was handed to llama.cpp, which
    /// ignores `pipeline_stage` entirely. The worker loaded the whole model,
    /// reported `ready: true`, and only failed later on the first
    /// forward_partial — so a node assigned 8 of 80 layers still needed
    /// memory for all 80.
    #[test]
    fn pipeline_stage_refuses_a_backend_that_cannot_serve_one() {
        let mut registry = EngineRegistry::empty();
        registry.register(Box::new(NoPipelineBackend));
        let profile = test_profile("linux", "x86_64", false);

        // Fine for ordinary inference.
        assert!(
            select_engine_for(&registry, ModelFormat::Gguf, &profile, false).is_ok(),
            "a non-pipeline backend is still valid for normal loads"
        );

        // Not acceptable for a pipeline stage.
        match select_engine_for(&registry, ModelFormat::Gguf, &profile, true) {
            Ok(backend) => panic!(
                "selected {} for a pipeline stage; it cannot serve one",
                backend.name()
            ),
            Err(err) => assert!(
                format!("{err}").contains("pipeline"),
                "error should say why: {err}"
            ),
        }
    }

    /// Whatever the feature set, a pipeline selection must never return a
    /// backend that cannot actually serve a stage.
    #[test]
    fn pipeline_selection_only_ever_returns_capable_backends() {
        let registry = EngineRegistry::new();
        let profile = test_profile("linux", "x86_64", false);
        for format in [ModelFormat::Gguf, ModelFormat::SafeTensors] {
            if let Ok(backend) = select_engine_for(&registry, format, &profile, true) {
                assert!(
                    backend.supports_pipeline(),
                    "{} was selected for a pipeline stage but cannot serve one",
                    backend.name()
                );
            }
        }
    }

    #[test]
    fn detect_format_survives_a_short_file() {
        let dir = std::env::temp_dir().join(format!("hb-detect-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // Two bytes, no recognised extension. The old code sliced [..4] on
        // this and panicked.
        let short = dir.join("truncated.partial");
        std::fs::write(&short, b"GG").unwrap();
        assert!(detect_format(&short).is_err());

        let empty = dir.join("empty.partial");
        std::fs::write(&empty, b"").unwrap();
        assert!(detect_format(&empty).is_err());

        // Magic bytes still work without an extension.
        let magic = dir.join("headless");
        std::fs::write(&magic, b"GGUF and then some more content").unwrap();
        assert_eq!(detect_format(&magic).unwrap(), ModelFormat::Gguf);

        let _ = std::fs::remove_dir_all(&dir);
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
