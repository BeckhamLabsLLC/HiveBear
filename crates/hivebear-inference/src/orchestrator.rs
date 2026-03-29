use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use hivebear_core::types::HardwareProfile;

use crate::engine::{EngineRegistry, InferenceBackend, TokenStream};
use crate::error::{InferenceError, Result};
use crate::selector;
use crate::types::*;

/// High-level inference orchestrator that manages engine selection,
/// model loading, and generation across multiple backends.
pub struct Orchestrator {
    registry: EngineRegistry,
    profile: HardwareProfile,
    /// Maps handle IDs to their backend engine ID for dispatch.
    handle_backends: Arc<Mutex<HashMap<u64, hivebear_core::types::InferenceEngine>>>,
    /// Track loaded model info.
    model_info: Arc<Mutex<HashMap<u64, ModelInfo>>>,
}

impl Orchestrator {
    /// Create a new orchestrator with the given hardware profile.
    pub fn new(profile: HardwareProfile) -> Self {
        Self {
            registry: EngineRegistry::new(),
            profile,
            handle_backends: Arc::new(Mutex::new(HashMap::new())),
            model_info: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Create an orchestrator with cloud config for API key resolution.
    pub fn with_config(profile: HardwareProfile, config: &hivebear_core::config::Config) -> Self {
        Self {
            registry: EngineRegistry::new_with_cloud_config(Some(&config.cloud)),
            profile,
            handle_backends: Arc::new(Mutex::new(HashMap::new())),
            model_info: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Get a reference to the engine registry.
    pub fn registry(&self) -> &EngineRegistry {
        &self.registry
    }

    /// Get a reference to the hardware profile.
    pub fn profile(&self) -> &HardwareProfile {
        &self.profile
    }

    /// Load a model, automatically selecting the best engine.
    pub async fn load(&self, path: &Path, config: &LoadConfig) -> Result<ModelHandle> {
        let format = selector::detect_format(path)?;
        let backend = selector::select_engine(&self.registry, format, &self.profile)?;

        let handle = backend.load_model(path, config).await?;

        // Track the mapping
        self.handle_backends
            .lock()
            .expect("lock poisoned")
            .insert(handle.id, backend.engine_id());

        self.model_info.lock().expect("lock poisoned").insert(
            handle.id,
            ModelInfo {
                handle_id: handle.id,
                model_path: path.display().to_string(),
                engine: backend.engine_id(),
                context_length: config.context_length,
                gpu_layers: config.offload.gpu_layers,
            },
        );

        Ok(handle)
    }

    /// Load a cloud model by name (e.g., "openai/gpt-4o").
    #[cfg(feature = "cloud")]
    pub async fn load_cloud(&self, model_name: &str) -> Result<ModelHandle> {
        use std::path::Path;
        let backend = self
            .registry
            .get(hivebear_core::types::InferenceEngine::Cloud)
            .ok_or(InferenceError::NoEngineAvailable {
                format: "cloud".to_string(),
            })?;

        let handle = backend
            .load_model(Path::new(model_name), &LoadConfig::default())
            .await?;

        self.handle_backends
            .lock()
            .expect("lock poisoned")
            .insert(handle.id, backend.engine_id());

        self.model_info.lock().expect("lock poisoned").insert(
            handle.id,
            ModelInfo {
                handle_id: handle.id,
                model_path: model_name.to_string(),
                engine: backend.engine_id(),
                context_length: 128_000,
                gpu_layers: None,
            },
        );

        Ok(handle)
    }

    /// Non-streaming generation.
    pub async fn generate(
        &self,
        handle: &ModelHandle,
        req: &GenerateRequest,
    ) -> Result<GenerateResponse> {
        let backend = self.get_backend(handle)?;
        backend.generate(handle, req).await
    }

    /// Streaming generation.
    pub fn stream(&self, handle: &ModelHandle, req: &GenerateRequest) -> Result<TokenStream> {
        let backend = self.get_backend(handle)?;
        Ok(backend.stream(handle, req))
    }

    /// Unload a model.
    pub async fn unload(&self, handle: &ModelHandle) -> Result<()> {
        let backend = self.get_backend(handle)?;
        backend.unload(handle).await?;

        self.handle_backends
            .lock()
            .expect("lock poisoned")
            .remove(&handle.id);
        self.model_info
            .lock()
            .expect("lock poisoned")
            .remove(&handle.id);

        Ok(())
    }

    /// List all currently loaded models.
    pub fn loaded_models(&self) -> Vec<ModelInfo> {
        self.model_info
            .lock()
            .expect("lock poisoned")
            .values()
            .cloned()
            .collect()
    }

    fn get_backend(&self, handle: &ModelHandle) -> Result<&dyn InferenceBackend> {
        let engine_id = self
            .handle_backends
            .lock()
            .expect("lock poisoned")
            .get(&handle.id)
            .copied()
            .ok_or(InferenceError::InvalidHandle)?;

        self.registry
            .get(engine_id)
            .ok_or(InferenceError::InvalidHandle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hivebear_core::types::*;

    fn test_profile() -> HardwareProfile {
        let gb = 1024 * 1024 * 1024;
        HardwareProfile {
            cpu: CpuInfo {
                model_name: "Test CPU".into(),
                physical_cores: 8,
                logical_cores: 16,
                isa_extensions: vec![],
                cache_size_bytes: 0,
            },
            memory: MemoryInfo {
                total_bytes: 16 * gb,
                available_bytes: 12 * gb,
                estimated_bandwidth_gbps: 30.0,
            },
            gpus: vec![],
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
    fn test_orchestrator_new() {
        let orchestrator = Orchestrator::new(test_profile());
        assert!(orchestrator.loaded_models().is_empty());
    }

    #[test]
    fn test_registry_accessible() {
        let orchestrator = Orchestrator::new(test_profile());
        assert!(!orchestrator.registry().all_backends().is_empty());
    }

    #[tokio::test]
    async fn test_load_nonexistent() {
        let orchestrator = Orchestrator::new(test_profile());
        let result = orchestrator
            .load(Path::new("/nonexistent.gguf"), &LoadConfig::default())
            .await;
        assert!(result.is_err());
    }
}
