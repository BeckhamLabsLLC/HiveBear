use crate::types::{CommunityBenchmarkSummary, ModelRecommendation};

/// Merge community benchmark data into recommendation results.
///
/// For each recommendation, finds matching community summaries (by model_id +
/// quantization) and blends the community-observed performance into the local
/// heuristic estimate. The blend weight depends on hardware similarity and
/// sample count — more data from closer hardware gets more influence.
pub fn merge_community_data(
    recs: &mut [ModelRecommendation],
    community: &[CommunityBenchmarkSummary],
) {
    for rec in recs.iter_mut() {
        let quant_str = rec.quantization.to_string();

        // Find the best-matching community summary for this model + quantization.
        let best = community
            .iter()
            .filter(|s| s.model_id == rec.model_id && s.quantization == quant_str)
            .max_by(|a, b| {
                a.hardware_similarity
                    .partial_cmp(&b.hardware_similarity)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

        if let Some(summary) = best {
            rec.community_tokens_per_sec = Some(summary.tokens_per_sec_p50);
            rec.community_sample_count = Some(summary.sample_count);

            // Blend estimate toward community value.
            // Weight increases with hardware similarity and sample count.
            let sample_factor = (summary.sample_count.min(20) as f64) / 20.0;
            let blend_weight = summary.hardware_similarity * sample_factor * 0.6;

            rec.estimated_tokens_per_sec = (rec.estimated_tokens_per_sec as f64
                * (1.0 - blend_weight)
                + summary.tokens_per_sec_p50 as f64 * blend_weight)
                as f32;

            // Boost confidence when community data confirms the estimate (within 30%).
            let ratio = if summary.tokens_per_sec_p50 > 0.0 {
                rec.estimated_tokens_per_sec / summary.tokens_per_sec_p50
            } else {
                1.0
            };
            if (0.7..=1.3).contains(&ratio) {
                rec.confidence = (rec.confidence + 0.1 * summary.hardware_similarity as f32)
                    .clamp(0.1, 0.99);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{InferenceEngine, Quantization};

    fn make_rec(model_id: &str, quant: Quantization, est_tok_s: f32) -> ModelRecommendation {
        ModelRecommendation {
            model_id: model_id.into(),
            model_name: model_id.into(),
            quantization: quant,
            engine: InferenceEngine::LlamaCpp,
            estimated_tokens_per_sec: est_tok_s,
            estimated_memory_usage_bytes: 4_000_000_000,
            confidence: 0.7,
            warnings: vec![],
            score: 0.8,
            community_tokens_per_sec: None,
            community_sample_count: None,
        }
    }

    fn make_summary(
        model_id: &str,
        quant: &str,
        p50: f32,
        samples: u32,
        similarity: f64,
    ) -> CommunityBenchmarkSummary {
        CommunityBenchmarkSummary {
            model_id: model_id.into(),
            quantization: quant.into(),
            engine: "llama.cpp".into(),
            sample_count: samples,
            tokens_per_sec_p50: p50,
            tokens_per_sec_p25: p50 * 0.8,
            tokens_per_sec_p75: p50 * 1.2,
            ttft_ms_p50: Some(200),
            peak_memory_mb_p50: 4000,
            hardware_similarity: similarity,
        }
    }

    #[test]
    fn test_merge_empty_community() {
        let mut recs = vec![make_rec("llama-3.1-8b", Quantization::Q4KM, 25.0)];
        merge_community_data(&mut recs, &[]);
        assert!(recs[0].community_tokens_per_sec.is_none());
        assert!(recs[0].community_sample_count.is_none());
        assert!((recs[0].estimated_tokens_per_sec - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_merge_populates_fields() {
        let mut recs = vec![make_rec("llama-3.1-8b", Quantization::Q4KM, 25.0)];
        let community = vec![make_summary("llama-3.1-8b", "Q4_K_M", 22.0, 15, 0.9)];
        merge_community_data(&mut recs, &community);
        assert_eq!(recs[0].community_tokens_per_sec, Some(22.0));
        assert_eq!(recs[0].community_sample_count, Some(15));
    }

    #[test]
    fn test_merge_blends_estimate() {
        let mut recs = vec![make_rec("llama-3.1-8b", Quantization::Q4KM, 30.0)];
        let community = vec![make_summary("llama-3.1-8b", "Q4_K_M", 20.0, 20, 1.0)];
        merge_community_data(&mut recs, &community);

        // With 20 samples at similarity 1.0, blend_weight = 1.0 * 1.0 * 0.6 = 0.6
        // blended = 30 * 0.4 + 20 * 0.6 = 12 + 12 = 24
        let blended = recs[0].estimated_tokens_per_sec;
        assert!(
            blended > 20.0 && blended < 30.0,
            "Blended value should be between estimate and community, got {blended}"
        );
        assert!(
            (blended - 24.0).abs() < 0.1,
            "Expected ~24.0 with full blend, got {blended}"
        );
    }
}
