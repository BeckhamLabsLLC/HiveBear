pub mod benchmark;
pub mod config;
pub mod contribution;
pub mod profiler;
pub mod recommender;
#[cfg(all(not(target_arch = "wasm32"), feature = "keychain"))]
pub mod secrets;
pub mod types;

// Re-export key types for convenience
#[cfg(not(target_arch = "wasm32"))]
pub use config::paths::AppPaths;
pub use config::Config;
pub use profiler::profile;
#[cfg(target_arch = "wasm32")]
pub use profiler::profile_async;
pub use recommender::model_db::ModelCategory;
pub use types::{
    BenchmarkResult, ComputeApi, HardwareProfile, InferenceEngine, ModelRecommendation,
    Quantization,
};
