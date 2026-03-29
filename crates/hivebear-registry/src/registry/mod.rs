mod huggingface;
pub mod local;
pub mod ollama_lib;

use crate::download::DownloadManager;
use crate::error::{RegistryError, Result};
use crate::metadata::{InstalledInfo, ModelMetadata, ModelSource, RemoteFile, SearchResult};
use crate::storage::StorageManager;
use chrono::Utc;
use hivebear_core::config::paths::AppPaths;
use hivebear_core::types::{HardwareProfile, ModelFormat};
use hivebear_core::Config;
use huggingface::HuggingFaceSource;
use local::LocalIndex;
use ollama_lib::OllamaSource;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;

/// Unified model registry combining all sources.
pub struct Registry {
    hf: HuggingFaceSource,
    ollama: OllamaSource,
    index: Arc<Mutex<LocalIndex>>,
    downloader: DownloadManager,
    models_dir: PathBuf,
}

impl Registry {
    /// Create a new registry from config.
    pub async fn new(config: &Config, paths: &AppPaths) -> Result<Self> {
        paths
            .ensure_dirs()
            .map_err(|e| RegistryError::IndexError(format!("Failed to create directories: {e}")))?;

        let index_path = paths.data_dir.join("registry.json");
        let index = LocalIndex::load(&index_path)?;

        Ok(Self {
            hf: HuggingFaceSource::new(),
            ollama: OllamaSource::new(),
            index: Arc::new(Mutex::new(index)),
            downloader: DownloadManager::new(config.models_dir.clone()),
            models_dir: config.models_dir.clone(),
        })
    }

    /// Search for models across all sources.
    pub async fn search(
        &self,
        query: &str,
        limit: usize,
        hw_profile: Option<&HardwareProfile>,
    ) -> Result<Vec<SearchResult>> {
        let index = self.index.lock().await;
        let local_results = index.search(query);

        // Search HuggingFace
        let hf_results = match self.hf.search(query, limit).await {
            Ok(results) => results,
            Err(e) => {
                tracing::warn!("HuggingFace search failed: {e}");
                Vec::new()
            }
        };

        drop(index);

        let mut results: Vec<SearchResult> = Vec::new();
        let index = self.index.lock().await;

        // Add local results first
        for meta in local_results {
            let compatibility = hw_profile.map(|hw| compute_compatibility(hw, &meta));
            results.push(SearchResult {
                is_installed: meta.installed.is_some(),
                compatibility_score: compatibility,
                metadata: meta,
            });
        }

        // Add HF results, avoiding duplicates
        for meta in hf_results {
            let already_listed = results.iter().any(|r| r.metadata.id == meta.id);
            if already_listed {
                continue;
            }
            let is_installed = index.find(&meta.id).is_some();
            let compatibility = hw_profile.map(|hw| compute_compatibility(hw, &meta));
            results.push(SearchResult {
                is_installed,
                compatibility_score: compatibility,
                metadata: meta,
            });
        }

        // Sort: installed first, then by compatibility score, then by downloads
        results.sort_by(|a, b| {
            b.is_installed
                .cmp(&a.is_installed)
                .then_with(|| {
                    b.compatibility_score
                        .unwrap_or(0.0)
                        .partial_cmp(&a.compatibility_score.unwrap_or(0.0))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| {
                    b.metadata
                        .downloads_count
                        .unwrap_or(0)
                        .cmp(&a.metadata.downloads_count.unwrap_or(0))
                })
        });

        results.truncate(limit);
        Ok(results)
    }

    /// Install a model by ID.
    pub async fn install(
        &self,
        model_id: &str,
        quant_preference: Option<&str>,
        file_preference: Option<&str>,
        progress_cb: Option<&(dyn Fn(crate::download::DownloadProgress) + Send + Sync)>,
    ) -> Result<InstalledInfo> {
        // Check if already installed
        {
            let index = self.index.lock().await;
            if let Some(meta) = index.find(model_id) {
                if meta.installed.is_some() {
                    return Err(RegistryError::AlreadyInstalled(model_id.to_string()));
                }
            }
        }

        // Resolve HuggingFace repo ID
        let repo_id = self.resolve_hf_repo(model_id).await?;

        // List available files
        let files = self.hf.list_files(&repo_id).await?;
        if files.is_empty() {
            return Err(RegistryError::NoFilesFound(model_id.to_string()));
        }

        // Pick the best file
        let file = pick_file(&files, quant_preference, file_preference)?;

        // Create model subdirectory
        let model_dir = self.models_dir.join(model_id);
        tokio::fs::create_dir_all(&model_dir).await?;

        // Download the model file
        let model_path = self
            .downloader
            .download(
                &file.download_url,
                &model_dir,
                &file.filename,
                file.sha256.as_deref(),
                model_id,
                progress_cb,
            )
            .await?;

        let size_bytes = tokio::fs::metadata(&model_path).await?.len();

        // Also download tokenizer.json if available (needed by Candle backend)
        let tokenizer_url = format!(
            "https://huggingface.co/{}/resolve/main/tokenizer.json",
            repo_id
        );
        if let Err(e) = self
            .downloader
            .download(
                &tokenizer_url,
                &model_dir,
                "tokenizer.json",
                None,
                model_id,
                None,
            )
            .await
        {
            tracing::debug!("No tokenizer.json available: {e}");
        }

        // Build installed info
        let installed = InstalledInfo {
            path: model_dir,
            format: ModelFormat::Gguf,
            quantization: file.quantization,
            size_bytes,
            sha256: file.sha256.clone(),
            installed_at: Utc::now(),
            last_used: None,
            filename: file.filename.clone(),
        };

        // Update local index
        let mut index = self.index.lock().await;
        let mut meta = match index.find(model_id) {
            Some(existing) => existing.clone(),
            None => {
                // Create a new metadata entry from what we know
                ModelMetadata {
                    id: model_id.to_string(),
                    name: model_id.to_string(),
                    params_billions: 0.0,
                    formats: vec![ModelFormat::Gguf],
                    context_length: 4096,
                    quality_score: 0.0,
                    category: hivebear_core::recommender::model_db::ModelCategory::General,
                    source: ModelSource::HuggingFace {
                        repo_id: repo_id.clone(),
                        revision: None,
                    },
                    huggingface_id: Some(repo_id),
                    installed: None,
                    description: None,
                    tags: Vec::new(),
                    downloads_count: None,
                    likes_count: None,
                    last_modified: None,
                }
            }
        };
        meta.installed = Some(installed.clone());
        index.add(meta)?;

        Ok(installed)
    }

    /// Remove an installed model.
    pub async fn remove(&self, model_id: &str) -> Result<u64> {
        let mut index = self.index.lock().await;
        let meta = index
            .find(model_id)
            .ok_or_else(|| RegistryError::ModelNotFound(model_id.to_string()))?
            .clone();

        let freed_bytes = if let Some(ref installed) = meta.installed {
            let size = installed.size_bytes;
            // Remove the model directory
            if installed.path.exists() {
                tokio::fs::remove_dir_all(&installed.path).await?;
            }
            size
        } else {
            0
        };

        index.remove(model_id)?;
        Ok(freed_bytes)
    }

    /// List all installed models.
    pub async fn list_installed(&self) -> Vec<ModelMetadata> {
        let index = self.index.lock().await;
        index
            .all()
            .iter()
            .filter(|m| m.installed.is_some())
            .cloned()
            .collect()
    }

    /// Resolve a model ID or file path to a local file path.
    pub async fn resolve(&self, model_id_or_path: &str) -> Result<PathBuf> {
        // If it's an existing file path, return it directly
        let path = Path::new(model_id_or_path);
        if path.exists() {
            return Ok(path.to_path_buf());
        }

        // Try to resolve through the local index
        let index = self.index.lock().await;
        if let Some(meta) = index.find(model_id_or_path) {
            if let Some(ref installed) = meta.installed {
                let model_file = installed.path.join(&installed.filename);
                if model_file.exists() {
                    return Ok(model_file);
                }
            }
        }

        Err(RegistryError::ModelNotFound(format!(
            "'{}' is not a file path or installed model. Use `hivebear install {}` first.",
            model_id_or_path, model_id_or_path
        )))
    }

    /// Get metadata for a specific model.
    pub async fn get(&self, model_id: &str) -> Result<Option<ModelMetadata>> {
        // Check local first
        let index = self.index.lock().await;
        if let Some(meta) = index.find(model_id) {
            return Ok(Some(meta.clone()));
        }
        drop(index);

        // Try HuggingFace
        self.hf.get(model_id).await
    }

    /// List available files for a model (for choosing quantization).
    pub async fn list_files(&self, model_id: &str) -> Result<Vec<RemoteFile>> {
        let repo_id = self.resolve_hf_repo(model_id).await?;
        self.hf.list_files(&repo_id).await
    }

    /// Get a reference to the storage manager.
    pub async fn storage_manager(&self) -> StorageManager {
        StorageManager::new(self.models_dir.clone(), self.index.clone())
    }

    /// Check whether Ollama is installed locally.
    pub fn ollama_available(&self) -> bool {
        self.ollama.is_available()
    }

    /// List all models found in the local Ollama installation.
    pub async fn list_ollama(&self) -> Result<Vec<ModelMetadata>> {
        self.ollama.search("", 100).await
    }

    /// Import a model from the local Ollama installation into HiveBear's index.
    pub async fn import_ollama(&self, model_id: &str) -> Result<InstalledInfo> {
        let meta = self
            .ollama
            .get(model_id)
            .await?
            .ok_or_else(|| {
                RegistryError::ModelNotFound(format!(
                    "Ollama model '{}' not found. Run `hivebear import-ollama` to list available models.",
                    model_id
                ))
            })?;

        let installed = meta.installed.clone().ok_or_else(|| {
            RegistryError::ModelNotFound(format!(
                "Ollama model '{}' found in manifests but blob file is missing.",
                model_id
            ))
        })?;

        let mut index = self.index.lock().await;
        index.add(meta)?;

        Ok(installed)
    }

    /// Resolve a model ID to a HuggingFace repo ID.
    async fn resolve_hf_repo(&self, model_id: &str) -> Result<String> {
        // If it already looks like a HF repo (contains '/'), use directly
        if model_id.contains('/') {
            return Ok(model_id.to_string());
        }

        // Check if we have a known HF ID from the builtin DB or local index
        let index = self.index.lock().await;
        if let Some(meta) = index.find(model_id) {
            if let Some(ref hf_id) = meta.huggingface_id {
                return Ok(hf_id.clone());
            }
        }
        drop(index);

        // Check builtin model database
        let builtins = hivebear_core::recommender::model_db::builtin_models();
        if let Some(entry) = builtins.iter().find(|e| e.id == model_id) {
            if let Some(ref hf_id) = entry.huggingface_id {
                return Ok(hf_id.clone());
            }
        }

        // Try searching HuggingFace
        let results = self.hf.search(model_id, 1).await?;
        if let Some(first) = results.first() {
            if let Some(ref hf_id) = first.huggingface_id {
                return Ok(hf_id.clone());
            }
        }

        Err(RegistryError::ModelNotFound(format!(
            "Could not resolve '{}' to a HuggingFace repository",
            model_id
        )))
    }
}

/// Compute a simple compatibility score based on hardware and model size.
fn compute_compatibility(hw: &HardwareProfile, meta: &ModelMetadata) -> f64 {
    let total_memory =
        hw.memory.available_bytes as f64 + hw.gpus.iter().map(|g| g.vram_bytes as f64).sum::<f64>();

    // Rough estimate: Q4_K_M at ~4.5 bits per weight
    let estimated_size = meta.params_billions * 1e9 * 4.5 / 8.0;

    if estimated_size > total_memory {
        0.0
    } else {
        let headroom = 1.0 - (estimated_size / total_memory);
        (headroom * 2.0).clamp(0.0, 1.0)
    }
}

/// Pick the best file from available options based on user preferences.
fn pick_file(
    files: &[RemoteFile],
    quant_preference: Option<&str>,
    file_preference: Option<&str>,
) -> Result<RemoteFile> {
    // If user specified a specific file, find it
    if let Some(filename) = file_preference {
        return files
            .iter()
            .find(|f| f.filename == filename)
            .cloned()
            .ok_or_else(|| {
                RegistryError::NoFilesFound(format!(
                    "File '{}' not found. Available: {}",
                    filename,
                    files
                        .iter()
                        .map(|f| f.filename.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            });
    }

    // If user specified a quantization preference, filter by it
    if let Some(quant_str) = quant_preference {
        let quant_upper = quant_str.to_uppercase().replace('-', "_");
        if let Some(file) = files
            .iter()
            .find(|f| f.filename.to_uppercase().contains(&quant_upper))
        {
            return Ok(file.clone());
        }
        tracing::warn!("Quantization '{quant_str}' not found, using best available");
    }

    // Default: prefer Q4_K_M as a good balance, then Q5_K_M, then largest available
    let preferred_quants = ["Q4_K_M", "Q5_K_M", "Q4_K_S", "Q5_K_S", "Q6_K", "Q8_0"];
    for pref in &preferred_quants {
        if let Some(file) = files
            .iter()
            .find(|f| f.filename.to_uppercase().contains(pref))
        {
            return Ok(file.clone());
        }
    }

    // Fall back to the smallest GGUF file
    files
        .iter()
        .min_by_key(|f| f.size_bytes)
        .cloned()
        .ok_or_else(|| RegistryError::NoFilesFound("No files available".to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::RemoteFile;

    #[test]
    fn test_pick_file_by_name() {
        let files = vec![
            RemoteFile {
                filename: "model-Q4_K_M.gguf".into(),
                size_bytes: 4_000_000_000,
                sha256: None,
                quantization: None,
                download_url: "https://example.com/q4km".into(),
            },
            RemoteFile {
                filename: "model-Q8_0.gguf".into(),
                size_bytes: 8_000_000_000,
                sha256: None,
                quantization: None,
                download_url: "https://example.com/q80".into(),
            },
        ];

        let result = pick_file(&files, None, Some("model-Q8_0.gguf")).unwrap();
        assert_eq!(result.filename, "model-Q8_0.gguf");
    }

    #[test]
    fn test_pick_file_by_quant() {
        let files = vec![
            RemoteFile {
                filename: "model-Q4_K_M.gguf".into(),
                size_bytes: 4_000_000_000,
                sha256: None,
                quantization: None,
                download_url: "https://example.com/q4km".into(),
            },
            RemoteFile {
                filename: "model-Q8_0.gguf".into(),
                size_bytes: 8_000_000_000,
                sha256: None,
                quantization: None,
                download_url: "https://example.com/q80".into(),
            },
        ];

        let result = pick_file(&files, Some("Q8_0"), None).unwrap();
        assert_eq!(result.filename, "model-Q8_0.gguf");
    }

    #[test]
    fn test_pick_file_default_prefers_q4km() {
        let files = vec![
            RemoteFile {
                filename: "model-Q2_K.gguf".into(),
                size_bytes: 2_000_000_000,
                sha256: None,
                quantization: None,
                download_url: "https://example.com/q2k".into(),
            },
            RemoteFile {
                filename: "model-Q4_K_M.gguf".into(),
                size_bytes: 4_000_000_000,
                sha256: None,
                quantization: None,
                download_url: "https://example.com/q4km".into(),
            },
            RemoteFile {
                filename: "model-Q8_0.gguf".into(),
                size_bytes: 8_000_000_000,
                sha256: None,
                quantization: None,
                download_url: "https://example.com/q80".into(),
            },
        ];

        let result = pick_file(&files, None, None).unwrap();
        assert_eq!(result.filename, "model-Q4_K_M.gguf");
    }
}
