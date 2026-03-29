use serde::Serialize;

/// Error type that serializes cleanly over Tauri IPC.
#[derive(Debug, Serialize)]
pub struct CommandError {
    pub message: String,
    pub code: ErrorCode,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    HardwareDetection,
    ModelNotFound,
    EngineUnavailable,
    LoadFailed,
    GenerationFailed,
    DownloadFailed,
    ConfigError,
    RegistryError,
    Internal,
}

impl std::fmt::Display for CommandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl From<hivebear_inference::InferenceError> for CommandError {
    fn from(e: hivebear_inference::InferenceError) -> Self {
        let code = match &e {
            hivebear_inference::InferenceError::ModelNotFound(_) => ErrorCode::ModelNotFound,
            hivebear_inference::InferenceError::NoEngineAvailable { .. } => {
                ErrorCode::EngineUnavailable
            }
            hivebear_inference::InferenceError::LoadError(_) => ErrorCode::LoadFailed,
            hivebear_inference::InferenceError::GenerationError(_) => ErrorCode::GenerationFailed,
            _ => ErrorCode::Internal,
        };
        CommandError {
            message: e.to_string(),
            code,
        }
    }
}

impl From<hivebear_registry::RegistryError> for CommandError {
    fn from(e: hivebear_registry::RegistryError) -> Self {
        let code = match &e {
            hivebear_registry::RegistryError::ModelNotFound(_) => ErrorCode::ModelNotFound,
            hivebear_registry::RegistryError::DownloadError(_)
            | hivebear_registry::RegistryError::DownloadInterrupted { .. } => {
                ErrorCode::DownloadFailed
            }
            _ => ErrorCode::RegistryError,
        };
        CommandError {
            message: e.to_string(),
            code,
        }
    }
}

/// Convert CommandError to String for Tauri IPC.
impl From<CommandError> for String {
    fn from(e: CommandError) -> Self {
        serde_json::to_string(&e).unwrap_or(e.message)
    }
}

pub type CmdResult<T> = std::result::Result<T, String>;
