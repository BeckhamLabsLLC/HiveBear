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

impl ErrorCode {
    /// Whether a failure with this code is worth reporting.
    ///
    /// `ModelNotFound` is excluded because it is an ordinary outcome of looking
    /// up a name that does not exist — reporting it would bury the real faults
    /// under routine misses. Everything else is a genuine failure: a download
    /// that broke, an engine that would not start, a config that would not
    /// parse. Those are exactly the onboarding failures that are currently
    /// invisible.
    fn should_report(&self) -> bool {
        !matches!(self, ErrorCode::ModelNotFound)
    }
}

/// Convert CommandError to String for Tauri IPC.
///
/// Every one of the 46 `#[tauri::command]` functions returns `CmdResult<T>`,
/// which is `Result<T, String>`, and every error path goes through this
/// conversion — so this is the one place that sees every backend failure the
/// frontend is ever told about.
impl From<CommandError> for String {
    fn from(e: CommandError) -> Self {
        if e.code.should_report() {
            // The message is redacted in `before_send`; command errors routinely
            // interpolate model paths under the user's home directory.
            sentry::with_scope(
                |scope| scope.set_tag("error.code", format!("{:?}", e.code)),
                || sentry::capture_message(&e.message, sentry::Level::Error),
            );
        }
        serde_json::to_string(&e).unwrap_or(e.message)
    }
}

pub type CmdResult<T> = std::result::Result<T, String>;
