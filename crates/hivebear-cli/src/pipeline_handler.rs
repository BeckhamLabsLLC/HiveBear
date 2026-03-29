//! Implementation of `MeshPipelineHandler` for the CLI.
//!
//! Bridges the mesh pipeline-parallel protocol to the local inference engine,
//! allowing this node to serve as a worker in a distributed inference pipeline.
//!
//! Current approach: loads the full model (llama.cpp handles memory mapping
//! efficiently) and processes activations through the assigned layer range.
//! Future optimization: true partial model loading when llama.cpp exposes
//! layer-range APIs.

use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use hivebear_inference::types::{ModelHandle, OffloadConfig};
use hivebear_inference::{LoadConfig, Orchestrator};
use hivebear_mesh::protocol::MeshPipelineHandler;

/// State for a loaded pipeline stage.
struct PipelineState {
    handle: ModelHandle,
    layer_range: Range<u32>,
    total_layers: u32,
}

/// CLI implementation of `MeshPipelineHandler` that wraps the local
/// `Orchestrator` to run partial inference for pipeline parallelism.
///
/// When a pipeline initiator assigns layers to this worker, it:
/// 1. Loads the model (memory-mapped, so only accessed layers are paged in)
/// 2. Processes activation tensors through the assigned layer range
/// 3. Returns output activations to the next stage
pub struct CliPipelineHandler {
    orchestrator: Arc<Orchestrator>,
    state: Mutex<Option<PipelineState>>,
}

impl CliPipelineHandler {
    pub fn new(orchestrator: Arc<Orchestrator>) -> Self {
        Self {
            orchestrator,
            state: Mutex::new(None),
        }
    }
}

#[async_trait]
impl MeshPipelineHandler for CliPipelineHandler {
    async fn load_layers(
        &self,
        model_source: &str,
        layer_range: Range<u32>,
        total_layers: u32,
    ) -> Result<(), String> {
        let model_path = Path::new(model_source);

        if !model_path.exists() {
            return Err(format!("Model not found: {model_source}"));
        }

        info!(
            "Loading layers {}..{} of {} (total: {total_layers})",
            layer_range.start, layer_range.end, model_source
        );

        // Load with auto offload — mmap ensures only accessed layers are paged in.
        // GPU offload is limited to the assigned layer range.
        let gpu_layers = layer_range.end - layer_range.start;
        let load_config = LoadConfig {
            context_length: 4096,
            offload: OffloadConfig {
                gpu_layers: Some(gpu_layers),
                auto: false,
                ..Default::default()
            },
            use_mmap: true,
            ..Default::default()
        };

        let handle = self
            .orchestrator
            .load(model_path, &load_config)
            .await
            .map_err(|e| format!("Failed to load layers: {e}"))?;

        info!(
            "Pipeline stage ready: layers {}..{} on {}",
            layer_range.start, layer_range.end, handle.engine
        );

        *self.state.lock().await = Some(PipelineState {
            handle,
            layer_range,
            total_layers,
        });

        Ok(())
    }

    async fn forward_layers(
        &self,
        activation_data: Vec<u8>,
        shape: Vec<usize>,
        dtype: u8,
        index_pos: usize,
    ) -> Result<(Vec<u8>, Vec<usize>, u8), String> {
        let state = self.state.lock().await;
        let state = state
            .as_ref()
            .ok_or("No layers loaded — call load_layers first")?;

        debug!(
            "Forward pass: layers {}..{}, activation size: {} bytes, pos: {}",
            state.layer_range.start,
            state.layer_range.end,
            activation_data.len(),
            index_pos
        );

        // Build ActivationData from the wire format
        let input = hivebear_inference::types::ActivationData {
            data: activation_data,
            shape: shape.clone(),
            dtype: match dtype {
                0 => hivebear_inference::types::ActivationDtype::F32,
                1 => hivebear_inference::types::ActivationDtype::F16,
                2 => hivebear_inference::types::ActivationDtype::BF16,
                _ => hivebear_inference::types::ActivationDtype::F32,
            },
        };

        let stage_config = hivebear_inference::types::PipelineStageConfig {
            layer_range: state.layer_range.clone(),
            total_layers: state.total_layers,
        };

        // Call the backend's forward_partial method
        let output = self
            .orchestrator
            .registry()
            .get(state.handle.engine)
            .ok_or("Backend not found for loaded model")?
            .forward_partial(&state.handle, &input, &stage_config, index_pos)
            .await
            .map_err(|e| format!("Forward pass failed: {e}"))?;

        let out_dtype = match output.dtype {
            hivebear_inference::types::ActivationDtype::F32 => 0u8,
            hivebear_inference::types::ActivationDtype::F16 => 1u8,
            hivebear_inference::types::ActivationDtype::BF16 => 2u8,
        };

        Ok((output.data, output.shape, out_dtype))
    }

    async fn unload_layers(&self) -> Result<(), String> {
        let mut state = self.state.lock().await;
        if let Some(pipeline_state) = state.take() {
            info!(
                "Unloading pipeline stage: layers {}..{}",
                pipeline_state.layer_range.start, pipeline_state.layer_range.end
            );
            self.orchestrator
                .unload(&pipeline_state.handle)
                .await
                .map_err(|e| format!("Unload failed: {e}"))?;
        } else {
            warn!("unload_layers called but no layers were loaded");
        }
        Ok(())
    }
}
