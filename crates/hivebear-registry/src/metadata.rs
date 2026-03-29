use chrono::{DateTime, Utc};
use hivebear_core::recommender::model_db::{ModelCategory, ModelEntry};
use hivebear_core::types::{ModelFormat, Quantization};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Where a model was sourced from.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ModelSource {
    HuggingFace {
        repo_id: String,
        revision: Option<String>,
    },
    Ollama {
        tag: String,
    },
    Local {
        imported: bool,
    },
}

/// Unified metadata for a model in the registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Unique identifier (e.g., "llama-3.1-8b")
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Number of parameters in billions
    pub params_billions: f64,
    /// Supported formats
    pub formats: Vec<ModelFormat>,
    /// Default context length
    pub context_length: u32,
    /// Relative quality score (0.0 - 1.0)
    pub quality_score: f64,
    /// Model category
    pub category: ModelCategory,

    /// Where this model comes from
    pub source: ModelSource,
    /// HuggingFace repo ID (e.g., "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF")
    pub huggingface_id: Option<String>,

    /// Installation details (populated when installed locally)
    pub installed: Option<InstalledInfo>,

    /// Model description
    pub description: Option<String>,
    /// Tags for search
    pub tags: Vec<String>,
    /// HuggingFace download count
    pub downloads_count: Option<u64>,
    /// HuggingFace likes count
    pub likes_count: Option<u64>,
    /// Last modified timestamp
    pub last_modified: Option<DateTime<Utc>>,
}

/// Information about an installed model on disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstalledInfo {
    /// Directory containing the model files
    pub path: PathBuf,
    /// Primary format on disk
    pub format: ModelFormat,
    /// Quantization level (if known)
    pub quantization: Option<Quantization>,
    /// File size in bytes
    pub size_bytes: u64,
    /// SHA-256 hash of the model file
    pub sha256: Option<String>,
    /// When the model was installed
    pub installed_at: DateTime<Utc>,
    /// When the model was last used for inference
    pub last_used: Option<DateTime<Utc>>,
    /// The specific model filename (e.g., "model-Q4_K_M.gguf")
    pub filename: String,
}

/// A search result combining metadata with local state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Whether this model is already installed locally
    pub is_installed: bool,
    /// Hardware compatibility score (0.0-1.0), if hardware profile available
    pub compatibility_score: Option<f64>,
}

/// A remote file available for download.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RemoteFile {
    /// Filename (e.g., "model-Q4_K_M.gguf")
    pub filename: String,
    /// File size in bytes
    pub size_bytes: u64,
    /// SHA-256 hash if known
    pub sha256: Option<String>,
    /// Detected quantization from filename
    pub quantization: Option<Quantization>,
    /// Direct download URL
    pub download_url: String,
}

impl From<ModelEntry> for ModelMetadata {
    fn from(entry: ModelEntry) -> Self {
        let source = match &entry.huggingface_id {
            Some(repo_id) => ModelSource::HuggingFace {
                repo_id: repo_id.clone(),
                revision: None,
            },
            None => ModelSource::Local { imported: false },
        };

        Self {
            id: entry.id,
            name: entry.name,
            params_billions: entry.params_billions,
            formats: entry.formats,
            context_length: entry.context_length,
            quality_score: entry.quality_score,
            category: entry.category,
            source,
            huggingface_id: entry.huggingface_id,
            installed: None,
            description: None,
            tags: Vec::new(),
            downloads_count: None,
            likes_count: None,
            last_modified: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_json_roundtrip() {
        let meta = ModelMetadata {
            id: "test-model".into(),
            name: "Test Model".into(),
            params_billions: 7.0,
            formats: vec![ModelFormat::Gguf],
            context_length: 4096,
            quality_score: 0.75,
            category: ModelCategory::General,
            source: ModelSource::HuggingFace {
                repo_id: "test/repo".into(),
                revision: None,
            },
            huggingface_id: Some("test/repo".into()),
            installed: None,
            description: Some("A test model".into()),
            tags: vec!["test".into()],
            downloads_count: Some(1000),
            likes_count: Some(50),
            last_modified: None,
        };

        let json = serde_json::to_string(&meta).unwrap();
        let deserialized: ModelMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, "test-model");
        assert_eq!(deserialized.params_billions, 7.0);
        assert_eq!(deserialized.formats, vec![ModelFormat::Gguf]);
    }

    #[test]
    fn test_installed_info_json_roundtrip() {
        let info = InstalledInfo {
            path: PathBuf::from("/models/test"),
            format: ModelFormat::Gguf,
            quantization: Some(Quantization::Q4KM),
            size_bytes: 4_000_000_000,
            sha256: Some("abc123".into()),
            installed_at: Utc::now(),
            last_used: None,
            filename: "model-Q4_K_M.gguf".into(),
        };

        let json = serde_json::to_string(&info).unwrap();
        let deserialized: InstalledInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.filename, "model-Q4_K_M.gguf");
        assert_eq!(deserialized.quantization, Some(Quantization::Q4KM));
    }

    #[test]
    fn test_from_model_entry() {
        let entry = ModelEntry {
            id: "llama-3.1-8b".into(),
            name: "Llama 3.1 8B".into(),
            params_billions: 8.0,
            formats: vec![ModelFormat::Gguf, ModelFormat::SafeTensors],
            context_length: 128_000,
            quality_score: 0.72,
            category: ModelCategory::General,
            huggingface_id: Some("bartowski/Meta-Llama-3.1-8B-Instruct-GGUF".into()),
        };

        let meta: ModelMetadata = entry.into();
        assert_eq!(meta.id, "llama-3.1-8b");
        assert_eq!(meta.params_billions, 8.0);
        assert!(matches!(meta.source, ModelSource::HuggingFace { .. }));
        assert!(meta.installed.is_none());
    }

    #[test]
    fn test_from_model_entry_no_hf() {
        let entry = ModelEntry {
            id: "custom-model".into(),
            name: "Custom Model".into(),
            params_billions: 1.0,
            formats: vec![ModelFormat::Gguf],
            context_length: 4096,
            quality_score: 0.5,
            category: ModelCategory::General,
            huggingface_id: None,
        };

        let meta: ModelMetadata = entry.into();
        assert!(matches!(
            meta.source,
            ModelSource::Local { imported: false }
        ));
    }
}
