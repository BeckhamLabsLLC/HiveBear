pub mod model_db;
pub mod scoring;

use crate::config::Config;
use crate::types::{HardwareProfile, ModelRecommendation};
use model_db::builtin_models;

/// Generate model recommendations based on hardware profile and config.
pub fn recommend(profile: &HardwareProfile, config: &Config) -> Vec<ModelRecommendation> {
    let models = builtin_models();

    scoring::recommend(
        profile,
        &models,
        config.max_memory_usage,
        config.min_tokens_per_sec,
        config.top_n_recommendations,
    )
}
