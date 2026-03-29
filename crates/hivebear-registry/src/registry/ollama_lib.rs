use std::path::{Path, PathBuf};

use tracing::{debug, warn};

use crate::error::{RegistryError, Result};
use crate::metadata::{InstalledInfo, ModelMetadata, ModelSource};
use hivebear_core::recommender::model_db::ModelCategory;
use hivebear_core::types::{ModelFormat, Quantization};

/// Ollama model source — detects and imports models from a local Ollama
/// installation.
///
/// Ollama stores models as blobs under `~/.ollama/models/`. The directory
/// structure is:
///   manifests/registry.ollama.ai/library/<model>/<tag>  (JSON manifest)
///   blobs/sha256-<hash>                                  (model weight files)
pub struct OllamaSource {
    ollama_dir: Option<PathBuf>,
}

impl Default for OllamaSource {
    fn default() -> Self {
        Self::new()
    }
}

impl OllamaSource {
    pub fn new() -> Self {
        let ollama_dir = detect_ollama_dir();
        if let Some(ref dir) = ollama_dir {
            debug!("Found Ollama installation at {}", dir.display());
        }
        Self { ollama_dir }
    }

    /// Check whether Ollama is installed and accessible.
    pub fn is_available(&self) -> bool {
        self.ollama_dir.is_some()
    }

    /// List all models installed in Ollama.
    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<ModelMetadata>> {
        let ollama_dir = match &self.ollama_dir {
            Some(d) => d,
            None => {
                return Err(RegistryError::ConversionUnavailable(
                    "Ollama is not installed. Install it from https://ollama.com".into(),
                ))
            }
        };

        let manifests_dir = ollama_dir
            .join("models")
            .join("manifests")
            .join("registry.ollama.ai")
            .join("library");

        if !manifests_dir.exists() {
            return Ok(Vec::new());
        }

        let mut results = Vec::new();

        let entries = match std::fs::read_dir(&manifests_dir) {
            Ok(e) => e,
            Err(_) => return Ok(Vec::new()),
        };

        for entry in entries.flatten() {
            let model_name = entry.file_name().to_string_lossy().to_string();

            // Filter by query (case-insensitive substring match)
            if !query.is_empty() && !model_name.to_lowercase().contains(&query.to_lowercase()) {
                continue;
            }

            // List tags for this model
            let tag_dir = entry.path();
            let tags = match std::fs::read_dir(&tag_dir) {
                Ok(t) => t,
                Err(_) => continue,
            };

            for tag_entry in tags.flatten() {
                let tag = tag_entry.file_name().to_string_lossy().to_string();
                let manifest_path = tag_entry.path();

                if let Some(metadata) =
                    parse_ollama_manifest(&model_name, &tag, &manifest_path, ollama_dir)
                {
                    results.push(metadata);
                    if results.len() >= limit {
                        return Ok(results);
                    }
                }
            }
        }

        Ok(results)
    }

    /// Get metadata for a specific Ollama model by name (e.g., "llama3:8b").
    pub async fn get(&self, model_id: &str) -> Result<Option<ModelMetadata>> {
        let ollama_dir = match &self.ollama_dir {
            Some(d) => d,
            None => return Ok(None),
        };

        let (model_name, tag) = parse_model_tag(model_id);

        let manifest_path = ollama_dir
            .join("models")
            .join("manifests")
            .join("registry.ollama.ai")
            .join("library")
            .join(&model_name)
            .join(&tag);

        if !manifest_path.exists() {
            return Ok(None);
        }

        Ok(parse_ollama_manifest(
            &model_name,
            &tag,
            &manifest_path,
            ollama_dir,
        ))
    }
}

/// Detect the Ollama models directory.
fn detect_ollama_dir() -> Option<PathBuf> {
    // Check OLLAMA_MODELS env var first
    if let Ok(dir) = std::env::var("OLLAMA_MODELS") {
        let path = PathBuf::from(dir);
        if path.exists() {
            return Some(path);
        }
    }

    // Default location: ~/.ollama
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .ok()?;
    let default = PathBuf::from(home).join(".ollama");

    if default.exists() {
        Some(default)
    } else {
        None
    }
}

/// Parse "model:tag" into (model, tag), defaulting tag to "latest".
fn parse_model_tag(model_id: &str) -> (String, String) {
    if let Some((name, tag)) = model_id.split_once(':') {
        (name.to_string(), tag.to_string())
    } else {
        (model_id.to_string(), "latest".to_string())
    }
}

/// Ollama manifest JSON structure (simplified).
#[derive(serde::Deserialize)]
struct OllamaManifest {
    #[serde(default)]
    layers: Vec<OllamaLayer>,
}

#[derive(serde::Deserialize)]
struct OllamaLayer {
    digest: String,
    size: u64,
    #[serde(rename = "mediaType")]
    media_type: String,
}

/// Parse an Ollama manifest file and construct ModelMetadata.
fn parse_ollama_manifest(
    model_name: &str,
    tag: &str,
    manifest_path: &Path,
    ollama_dir: &Path,
) -> Option<ModelMetadata> {
    let content = std::fs::read_to_string(manifest_path).ok()?;
    let manifest: OllamaManifest = serde_json::from_str(&content).ok()?;

    // Find the model weights layer (application/vnd.ollama.image.model)
    let model_layer = manifest
        .layers
        .iter()
        .find(|l| l.media_type == "application/vnd.ollama.image.model")?;

    // Resolve the blob path (sha256-<hash> format)
    let digest = model_layer.digest.replace(':', "-");
    let blob_path = ollama_dir.join("models").join("blobs").join(&digest);

    let installed = if blob_path.exists() {
        let quant = detect_quantization_from_tag(tag);

        Some(InstalledInfo {
            path: blob_path,
            format: ModelFormat::Gguf,
            quantization: quant,
            size_bytes: model_layer.size,
            sha256: Some(model_layer.digest.clone()),
            installed_at: chrono::Utc::now(),
            last_used: None,
            filename: digest.clone(),
        })
    } else {
        warn!("Ollama blob missing: {}", blob_path.display());
        None
    };

    // Guess parameter count from model name (e.g., "llama3:8b" -> 8.0)
    let params = guess_params_from_tag(tag)
        .or_else(|| guess_params_from_name(model_name))
        .unwrap_or(0.0);

    Some(ModelMetadata {
        id: format!("ollama/{model_name}:{tag}"),
        name: format!("{model_name}:{tag}"),
        params_billions: params,
        formats: vec![ModelFormat::Gguf],
        context_length: 4096, // Default; Ollama doesn't expose this in manifests
        quality_score: 0.5,   // Unknown quality; neutral default
        category: ModelCategory::General,
        source: ModelSource::Ollama {
            tag: tag.to_string(),
        },
        huggingface_id: None,
        installed,
        description: Some(format!("Imported from Ollama — {model_name}:{tag}")),
        tags: vec!["ollama".into(), model_name.into()],
        downloads_count: None,
        likes_count: None,
        last_modified: None,
    })
}

/// Try to detect quantization level from an Ollama tag name.
fn detect_quantization_from_tag(tag: &str) -> Option<Quantization> {
    let tag_upper = tag.to_uppercase();
    if tag_upper.contains("Q2_K") {
        Some(Quantization::Q2K)
    } else if tag_upper.contains("Q3_K_S") {
        Some(Quantization::Q3KS)
    } else if tag_upper.contains("Q3_K_M") {
        Some(Quantization::Q3KM)
    } else if tag_upper.contains("Q3_K_L") {
        Some(Quantization::Q3KL)
    } else if tag_upper.contains("Q4_0") {
        Some(Quantization::Q4_0)
    } else if tag_upper.contains("Q4_1") {
        Some(Quantization::Q4_1)
    } else if tag_upper.contains("Q4_K_S") {
        Some(Quantization::Q4KS)
    } else if tag_upper.contains("Q4_K_M") {
        Some(Quantization::Q4KM)
    } else if tag_upper.contains("Q5_0") {
        Some(Quantization::Q5_0)
    } else if tag_upper.contains("Q5_1") {
        Some(Quantization::Q5_1)
    } else if tag_upper.contains("Q5_K_S") {
        Some(Quantization::Q5KS)
    } else if tag_upper.contains("Q5_K_M") {
        Some(Quantization::Q5KM)
    } else if tag_upper.contains("Q6_K") {
        Some(Quantization::Q6K)
    } else if tag_upper.contains("Q8_0") {
        Some(Quantization::Q8_0)
    } else if tag_upper.contains("F16") || tag_upper.contains("FP16") {
        Some(Quantization::F16)
    } else if tag_upper.contains("F32") || tag_upper.contains("FP32") {
        Some(Quantization::F32)
    } else {
        None
    }
}

/// Guess parameter count from a tag like "8b", "70b", "3.8b".
fn guess_params_from_tag(tag: &str) -> Option<f64> {
    let tag_lower = tag.to_lowercase();
    // Match patterns like "8b", "70b", "3.8b" in the tag
    for part in tag_lower.split(&['-', '_', ':'][..]) {
        if let Some(num_str) = part.strip_suffix('b') {
            if let Ok(val) = num_str.parse::<f64>() {
                return Some(val);
            }
        }
    }
    None
}

/// Guess parameter count from a model name like "llama3", "phi3".
fn guess_params_from_name(name: &str) -> Option<f64> {
    let name_lower = name.to_lowercase();
    // Common model families with default sizes
    if name_lower.contains("llama3") || name_lower.contains("llama-3") {
        Some(8.0)
    } else if name_lower.contains("mistral") {
        Some(7.0)
    } else if name_lower.contains("phi3") || name_lower.contains("phi-3") {
        Some(3.8)
    } else if name_lower.contains("gemma") {
        Some(7.0)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_model_tag() {
        assert_eq!(parse_model_tag("llama3:8b"), ("llama3".into(), "8b".into()));
        assert_eq!(
            parse_model_tag("mistral"),
            ("mistral".into(), "latest".into())
        );
        assert_eq!(
            parse_model_tag("codellama:7b-instruct-q4_0"),
            ("codellama".into(), "7b-instruct-q4_0".into())
        );
    }

    #[test]
    fn test_detect_quantization_from_tag() {
        assert_eq!(
            detect_quantization_from_tag("7b-q4_k_m"),
            Some(Quantization::Q4KM)
        );
        assert_eq!(
            detect_quantization_from_tag("13b-Q8_0"),
            Some(Quantization::Q8_0)
        );
        assert_eq!(detect_quantization_from_tag("latest"), None);
        assert_eq!(
            detect_quantization_from_tag("fp16"),
            Some(Quantization::F16)
        );
    }

    #[test]
    fn test_guess_params_from_tag() {
        assert_eq!(guess_params_from_tag("8b"), Some(8.0));
        assert_eq!(guess_params_from_tag("70b-q4_k_m"), Some(70.0));
        assert_eq!(guess_params_from_tag("3.8b-instruct"), Some(3.8));
        assert_eq!(guess_params_from_tag("latest"), None);
    }

    #[test]
    fn test_ollama_source_creation() {
        let source = OllamaSource::new();
        // May or may not have Ollama installed — just verify it doesn't panic
        let _ = source.is_available();
    }
}
