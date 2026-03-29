use crate::error::{RegistryError, Result};
use crate::metadata::{ModelMetadata, ModelSource, RemoteFile};
use hivebear_core::recommender::model_db::ModelCategory;
use hivebear_core::types::{ModelFormat, Quantization};
use reqwest::header;
use serde::Deserialize;

/// HuggingFace Hub model source.
pub struct HuggingFaceSource {
    client: reqwest::Client,
}

/// HuggingFace API model response (subset of fields).
#[derive(Debug, Deserialize)]
struct HfModelResponse {
    #[serde(rename = "modelId", alias = "id")]
    model_id: String,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    likes: u64,
    #[serde(default)]
    siblings: Vec<HfSibling>,
    #[serde(rename = "lastModified", default)]
    last_modified: Option<String>,
}

#[derive(Debug, Deserialize)]
struct HfSibling {
    #[serde(rename = "rfilename")]
    filename: String,
    #[serde(default)]
    size: Option<u64>,
}

impl HuggingFaceSource {
    pub fn new() -> Self {
        let mut headers = header::HeaderMap::new();
        // Use HF_TOKEN if available for higher rate limits
        if let Ok(token) = std::env::var("HF_TOKEN") {
            if let Ok(val) = header::HeaderValue::from_str(&format!("Bearer {token}")) {
                headers.insert(header::AUTHORIZATION, val);
            }
        }

        let client = reqwest::Client::builder()
            .default_headers(headers)
            .user_agent("hivebear/0.1.0")
            .build()
            .unwrap_or_default();

        Self { client }
    }

    /// Search HuggingFace Hub for models.
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<ModelMetadata>> {
        let url = format!(
            "https://huggingface.co/api/models?search={}&filter=gguf&sort=downloads&direction=-1&limit={}",
            urlencoding::encode(query),
            limit
        );

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| RegistryError::HuggingFaceError(e.to_string()))?;

        if response.status() == 429 {
            return Err(RegistryError::HuggingFaceError(
                "Rate limited by HuggingFace API. Set HF_TOKEN for higher limits.".into(),
            ));
        }

        if !response.status().is_success() {
            return Err(RegistryError::HuggingFaceError(format!(
                "HTTP {}",
                response.status()
            )));
        }

        let models: Vec<HfModelResponse> = response
            .json()
            .await
            .map_err(|e| RegistryError::HuggingFaceError(e.to_string()))?;

        Ok(models.into_iter().map(hf_to_metadata).collect())
    }

    /// Get metadata for a specific model.
    pub async fn get(&self, model_id: &str) -> Result<Option<ModelMetadata>> {
        let url = format!("https://huggingface.co/api/models/{model_id}");

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| RegistryError::HuggingFaceError(e.to_string()))?;

        if response.status() == 404 {
            return Ok(None);
        }

        if !response.status().is_success() {
            return Err(RegistryError::HuggingFaceError(format!(
                "HTTP {}",
                response.status()
            )));
        }

        let model: HfModelResponse = response
            .json()
            .await
            .map_err(|e| RegistryError::HuggingFaceError(e.to_string()))?;

        Ok(Some(hf_to_metadata(model)))
    }

    /// List downloadable GGUF files for a model repo.
    pub async fn list_files(&self, repo_id: &str) -> Result<Vec<RemoteFile>> {
        let url = format!("https://huggingface.co/api/models/{repo_id}");

        let response = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| RegistryError::HuggingFaceError(e.to_string()))?;

        if !response.status().is_success() {
            return Err(RegistryError::HuggingFaceError(format!(
                "HTTP {} for {repo_id}",
                response.status()
            )));
        }

        let model: HfModelResponse = response
            .json()
            .await
            .map_err(|e| RegistryError::HuggingFaceError(e.to_string()))?;

        let files: Vec<RemoteFile> = model
            .siblings
            .into_iter()
            .filter(|s| s.filename.ends_with(".gguf"))
            .map(|s| {
                let quantization = parse_quantization_from_filename(&s.filename);
                let download_url = format!(
                    "https://huggingface.co/{}/resolve/main/{}",
                    repo_id, s.filename
                );
                RemoteFile {
                    filename: s.filename,
                    size_bytes: s.size.unwrap_or(0),
                    sha256: None,
                    quantization,
                    download_url,
                }
            })
            .collect();

        Ok(files)
    }
}

/// Convert a HuggingFace API response into our ModelMetadata.
fn hf_to_metadata(model: HfModelResponse) -> ModelMetadata {
    let name = model
        .model_id
        .rsplit('/')
        .next()
        .unwrap_or(&model.model_id)
        .to_string();

    // Try to extract param count from tags or name
    let params = extract_param_count(&name, &model.tags);

    // Guess category from tags
    let category = guess_category(&model.tags);

    let last_modified = model
        .last_modified
        .and_then(|s| chrono::DateTime::parse_from_rfc3339(&s).ok())
        .map(|dt| dt.with_timezone(&chrono::Utc));

    ModelMetadata {
        id: name.to_lowercase().replace(' ', "-"),
        name,
        params_billions: params,
        formats: vec![ModelFormat::Gguf],
        context_length: 4096,
        quality_score: 0.0,
        category,
        source: ModelSource::HuggingFace {
            repo_id: model.model_id.clone(),
            revision: None,
        },
        huggingface_id: Some(model.model_id),
        installed: None,
        description: None,
        tags: model.tags,
        downloads_count: Some(model.downloads),
        likes_count: Some(model.likes),
        last_modified,
    }
}

/// Extract parameter count (in billions) from model name or tags.
fn extract_param_count(name: &str, tags: &[String]) -> f64 {
    let name_lower = name.to_lowercase();

    // Decimal patterns MUST come before their integer suffixes
    // (e.g., "3.8b" before "8b") to avoid partial matches in "phi-3-mini-3.8b"
    let patterns = [
        ("405b", 405.0),
        ("236b", 236.0),
        ("180b", 180.0),
        ("72b", 72.0),
        ("70b", 70.0),
        ("65b", 65.0),
        ("46.7b", 46.7),
        ("34b", 34.0),
        ("33b", 33.0),
        ("32b", 32.0),
        ("27b", 27.0),
        ("22b", 22.0),
        ("14b", 14.0),
        ("13b", 13.0),
        ("12b", 12.0),
        ("11b", 11.0),
        ("3.8b", 3.8),
        ("9b", 9.0),
        ("8b", 8.0),
        ("7b", 7.0),
        ("2.7b", 2.7),
        ("3b", 3.0),
        ("1.5b", 1.5),
        ("2b", 2.0),
        ("0.5b", 0.5),
        ("1b", 1.0),
    ];

    if let Some(size) = match_param_pattern(&name_lower, &patterns) {
        return size;
    }

    // Check tags
    for tag in tags {
        let tag_lower = tag.to_lowercase();
        if let Some(size) = match_param_pattern(&tag_lower, &patterns) {
            return size;
        }
    }

    0.0
}

/// Match a parameter pattern ensuring it's at a word boundary.
/// "3.8b" in "phi-3-mini-3.8b" should match 3.8, not 8.
fn match_param_pattern(text: &str, patterns: &[(&str, f64)]) -> Option<f64> {
    for (pattern, size) in patterns {
        if let Some(pos) = text.find(pattern) {
            // Check that the character before the match is a separator or start-of-string
            let is_boundary = pos == 0 || {
                let prev = text.as_bytes()[pos - 1];
                prev == b'-' || prev == b'_' || prev == b'.' || prev == b' ' || prev == b'/'
            };
            if is_boundary {
                return Some(*size);
            }
        }
    }
    None
}

/// Guess model category from HuggingFace tags.
fn guess_category(tags: &[String]) -> ModelCategory {
    for tag in tags {
        let t = tag.to_lowercase();
        if t.contains("code") || t.contains("coder") || t.contains("starcoder") {
            return ModelCategory::Code;
        }
        if t.contains("chat") || t.contains("conversational") {
            return ModelCategory::Chat;
        }
        if t.contains("instruct") {
            return ModelCategory::Instruction;
        }
        if t.contains("math") {
            return ModelCategory::Math;
        }
        if t.contains("multilingual") {
            return ModelCategory::Multilingual;
        }
    }
    ModelCategory::General
}

/// Parse quantization level from a GGUF filename.
///
/// Handles common patterns like:
/// - `Model-Q4_K_M.gguf`
/// - `model.Q4_K_M.gguf`
/// - `model-q4_k_m.gguf`
/// - `model-IQ4_XS.gguf`
pub fn parse_quantization_from_filename(filename: &str) -> Option<Quantization> {
    let upper = filename.to_uppercase();

    // Check from highest to lowest specificity to avoid partial matches
    let mappings = [
        ("Q3_K_S", Quantization::Q3KS),
        ("Q3_K_M", Quantization::Q3KM),
        ("Q3_K_L", Quantization::Q3KL),
        ("Q4_K_S", Quantization::Q4KS),
        ("Q4_K_M", Quantization::Q4KM),
        ("Q5_K_S", Quantization::Q5KS),
        ("Q5_K_M", Quantization::Q5KM),
        ("Q2_K", Quantization::Q2K),
        ("Q6_K", Quantization::Q6K),
        ("Q8_0", Quantization::Q8_0),
        ("Q4_0", Quantization::Q4_0),
        ("Q4_1", Quantization::Q4_1),
        ("Q5_0", Quantization::Q5_0),
        ("Q5_1", Quantization::Q5_1),
        ("F16", Quantization::F16),
        ("F32", Quantization::F32),
    ];

    for (pattern, quant) in &mappings {
        if upper.contains(pattern) {
            return Some(*quant);
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_quantization_standard() {
        assert_eq!(
            parse_quantization_from_filename("model-Q4_K_M.gguf"),
            Some(Quantization::Q4KM)
        );
        assert_eq!(
            parse_quantization_from_filename("model-Q5_K_S.gguf"),
            Some(Quantization::Q5KS)
        );
        assert_eq!(
            parse_quantization_from_filename("model-Q8_0.gguf"),
            Some(Quantization::Q8_0)
        );
        assert_eq!(
            parse_quantization_from_filename("model-F16.gguf"),
            Some(Quantization::F16)
        );
    }

    #[test]
    fn test_parse_quantization_lowercase() {
        assert_eq!(
            parse_quantization_from_filename("model-q4_k_m.gguf"),
            Some(Quantization::Q4KM)
        );
    }

    #[test]
    fn test_parse_quantization_dot_separator() {
        assert_eq!(
            parse_quantization_from_filename("model.Q4_K_M.gguf"),
            Some(Quantization::Q4KM)
        );
    }

    #[test]
    fn test_parse_quantization_none() {
        assert_eq!(parse_quantization_from_filename("model.gguf"), None);
        assert_eq!(parse_quantization_from_filename("tokenizer.json"), None);
    }

    #[test]
    fn test_extract_param_count() {
        assert_eq!(extract_param_count("Llama-3.1-8B-Instruct", &[]), 8.0);
        assert_eq!(extract_param_count("Qwen2.5-72B-GGUF", &[]), 72.0);
        assert_eq!(extract_param_count("phi-3-mini-3.8b", &[]), 3.8);
        assert_eq!(extract_param_count("tiny-model-0.5b", &[]), 0.5);
        assert_eq!(extract_param_count("unknown-model", &[]), 0.0);
    }

    #[test]
    fn test_guess_category() {
        assert_eq!(
            guess_category(&["code".into(), "python".into()]),
            ModelCategory::Code
        );
        assert_eq!(
            guess_category(&["conversational".into()]),
            ModelCategory::Chat
        );
        assert_eq!(
            guess_category(&["text-generation".into()]),
            ModelCategory::General
        );
    }

    #[test]
    fn test_hf_to_metadata() {
        let response = HfModelResponse {
            model_id: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF".into(),
            tags: vec!["gguf".into(), "llama".into()],
            downloads: 50000,
            likes: 200,
            siblings: vec![HfSibling {
                filename: "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf".into(),
                size: Some(4_500_000_000),
            }],
            last_modified: None,
        };

        let meta = hf_to_metadata(response);
        assert_eq!(
            meta.huggingface_id.as_deref(),
            Some("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF")
        );
        assert_eq!(meta.downloads_count, Some(50000));
        assert_eq!(meta.params_billions, 8.0);
    }
}
