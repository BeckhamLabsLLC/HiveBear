use std::path::Path;

use hivebear_core::types::HardwareProfile;

use crate::engine::{EngineRegistry, InferenceBackend};
use crate::error::Result;
use crate::selector;
use crate::types::{LoadConfig, ModelHandle};

/// Load a model by detecting its format, selecting the best engine,
/// and configuring offloading based on hardware.
pub async fn load_model<'a>(
    registry: &'a EngineRegistry,
    path: &Path,
    profile: &HardwareProfile,
    config: &LoadConfig,
) -> Result<(ModelHandle, &'a dyn InferenceBackend)> {
    let format = selector::detect_format(path)?;
    let backend = selector::select_engine(registry, format, profile)?;

    tracing::info!(
        path = %path.display(),
        format = %format,
        engine = backend.name(),
        "Loading model"
    );

    let handle = backend.load_model(path, config).await?;
    Ok((handle, backend))
}
