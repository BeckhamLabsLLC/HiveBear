use crate::error::CmdResult;
use crate::state::AppState;
use hivebear_core::types::CommunityBenchmarkSummary;
use hivebear_core::{HardwareFingerprint, HardwareProfile, ModelRecommendation};
use tauri::State;

#[tauri::command]
pub fn get_hardware_profile(state: State<'_, AppState>) -> CmdResult<HardwareProfile> {
    Ok(state.profile.clone())
}

#[tauri::command]
pub async fn get_recommendations(
    state: State<'_, AppState>,
    include_community: Option<bool>,
) -> CmdResult<Vec<ModelRecommendation>> {
    let config = state
        .config
        .lock()
        .map_err(|_| String::from("Config lock poisoned"))?
        .clone();

    let use_community = include_community.unwrap_or(config.share_benchmarks);

    if use_community {
        let fp = HardwareFingerprint::from_profile(&state.profile);
        let url = format!(
            "{}/benchmarks?gpu_class={}&ram_gb_bucket={}&platform_arch={}",
            config.mesh.coordination_server, fp.gpu_class, fp.ram_gb_bucket, fp.platform_arch
        );

        #[derive(serde::Deserialize)]
        struct QueryResponse {
            results: Vec<CommunityBenchmarkSummary>,
        }

        let community_data = match state.http_client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => resp
                .json::<QueryResponse>()
                .await
                .map(|d| d.results)
                .unwrap_or_default(),
            _ => vec![],
        };

        let recs = hivebear_core::recommender::recommend_with_community(
            &state.profile,
            &config,
            &community_data,
        );
        Ok(recs)
    } else {
        let recs = hivebear_core::recommender::recommend(&state.profile, &config);
        Ok(recs)
    }
}
