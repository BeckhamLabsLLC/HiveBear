import type { ModelRecommendation } from "../types";

/**
 * What to offer when the recommender returns nothing (unusual hardware, a
 * very small machine, or a profile that failed). An empty first-run screen is
 * where new users give up, so there is always something to install.
 *
 * Qwen 2.5 0.5B is in the built-in model database (`model_db.rs`) and runs on
 * almost anything. The size is the Q4_K_M weights estimate the recommender
 * would give (0.5B params × 4.5 bits).
 */
export const FALLBACK_MODEL: ModelRecommendation = {
  model_id: "qwen-2.5-0.5b",
  model_name: "Qwen 2.5 0.5B",
  quantization: "Q4KM",
  engine: "LlamaCpp",
  estimated_tokens_per_sec: 0,
  estimated_memory_usage_bytes: 600_000_000,
  confidence: 0,
  warnings: [],
  score: 0,
  estimated_download_bytes: 281_250_000,
};

/** The model to offer a first-run user: the top recommendation, or the fallback. */
export function firstModel(recommendations: ModelRecommendation[]): {
  model: ModelRecommendation;
  isFallback: boolean;
} {
  const top = recommendations[0];
  return top ? { model: top, isFallback: false } : { model: FALLBACK_MODEL, isFallback: true };
}
