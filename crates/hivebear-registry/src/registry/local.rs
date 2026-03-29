use crate::error::{RegistryError, Result};
use crate::metadata::ModelMetadata;
use std::path::{Path, PathBuf};

/// JSON-backed local model index.
pub struct LocalIndex {
    path: PathBuf,
    entries: Vec<ModelMetadata>,
}

impl LocalIndex {
    /// Load the index from disk, or create an empty one.
    pub fn load(path: &Path) -> Result<Self> {
        let entries = if path.exists() {
            let contents = std::fs::read_to_string(path).map_err(|e| {
                RegistryError::IndexError(format!(
                    "Failed to read index at {}: {e}",
                    path.display()
                ))
            })?;
            serde_json::from_str(&contents)
                .map_err(|e| {
                    tracing::warn!("Corrupt index at {}, starting fresh: {e}", path.display());
                    RegistryError::IndexError(format!("Corrupt index: {e}"))
                })
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        Ok(Self {
            path: path.to_path_buf(),
            entries,
        })
    }

    /// Save the index to disk atomically.
    pub fn save(&self) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let tmp_path = self.path.with_extension("json.tmp");
        let contents = serde_json::to_string_pretty(&self.entries)?;
        std::fs::write(&tmp_path, &contents)?;
        std::fs::rename(&tmp_path, &self.path)?;

        Ok(())
    }

    /// Add or update an entry. Saves automatically.
    pub fn add(&mut self, metadata: ModelMetadata) -> Result<()> {
        if let Some(existing) = self.entries.iter_mut().find(|e| e.id == metadata.id) {
            *existing = metadata;
        } else {
            self.entries.push(metadata);
        }
        self.save()
    }

    /// Remove an entry by ID. Saves automatically.
    pub fn remove(&mut self, model_id: &str) -> Result<()> {
        self.entries.retain(|e| e.id != model_id);
        self.save()
    }

    /// Find an entry by ID.
    pub fn find(&self, model_id: &str) -> Option<&ModelMetadata> {
        self.entries.iter().find(|e| e.id == model_id)
    }

    /// Search entries by substring match on id, name, tags, and description.
    pub fn search(&self, query: &str) -> Vec<ModelMetadata> {
        let query_lower = query.to_lowercase();
        self.entries
            .iter()
            .filter(|e| {
                e.id.to_lowercase().contains(&query_lower)
                    || e.name.to_lowercase().contains(&query_lower)
                    || e.tags
                        .iter()
                        .any(|t| t.to_lowercase().contains(&query_lower))
                    || e.description
                        .as_deref()
                        .map(|d| d.to_lowercase().contains(&query_lower))
                        .unwrap_or(false)
            })
            .cloned()
            .collect()
    }

    /// Get all entries.
    pub fn all(&self) -> &[ModelMetadata] {
        &self.entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::ModelSource;
    use hivebear_core::recommender::model_db::ModelCategory;
    use hivebear_core::types::ModelFormat;
    use tempfile::TempDir;

    fn make_meta(id: &str, name: &str) -> ModelMetadata {
        ModelMetadata {
            id: id.into(),
            name: name.into(),
            params_billions: 7.0,
            formats: vec![ModelFormat::Gguf],
            context_length: 4096,
            quality_score: 0.7,
            category: ModelCategory::General,
            source: ModelSource::Local { imported: false },
            huggingface_id: None,
            installed: None,
            description: None,
            tags: Vec::new(),
            downloads_count: None,
            likes_count: None,
            last_modified: None,
        }
    }

    #[test]
    fn test_empty_index() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("registry.json");
        let index = LocalIndex::load(&path).unwrap();
        assert!(index.all().is_empty());
    }

    #[test]
    fn test_add_and_find() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("registry.json");
        let mut index = LocalIndex::load(&path).unwrap();

        index.add(make_meta("test-model", "Test Model")).unwrap();
        assert_eq!(index.all().len(), 1);
        assert!(index.find("test-model").is_some());
        assert!(index.find("nonexistent").is_none());
    }

    #[test]
    fn test_add_updates_existing() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("registry.json");
        let mut index = LocalIndex::load(&path).unwrap();

        index.add(make_meta("test-model", "Original")).unwrap();
        index.add(make_meta("test-model", "Updated")).unwrap();

        assert_eq!(index.all().len(), 1);
        assert_eq!(index.find("test-model").unwrap().name, "Updated");
    }

    #[test]
    fn test_remove() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("registry.json");
        let mut index = LocalIndex::load(&path).unwrap();

        index.add(make_meta("model-a", "Model A")).unwrap();
        index.add(make_meta("model-b", "Model B")).unwrap();
        assert_eq!(index.all().len(), 2);

        index.remove("model-a").unwrap();
        assert_eq!(index.all().len(), 1);
        assert!(index.find("model-a").is_none());
        assert!(index.find("model-b").is_some());
    }

    #[test]
    fn test_persist_and_reload() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("registry.json");

        {
            let mut index = LocalIndex::load(&path).unwrap();
            index
                .add(make_meta("persist-test", "Persist Test"))
                .unwrap();
        }

        let index = LocalIndex::load(&path).unwrap();
        assert_eq!(index.all().len(), 1);
        assert_eq!(index.find("persist-test").unwrap().name, "Persist Test");
    }

    #[test]
    fn test_search() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("registry.json");
        let mut index = LocalIndex::load(&path).unwrap();

        index.add(make_meta("llama-8b", "Llama 8B")).unwrap();
        index.add(make_meta("qwen-7b", "Qwen 7B")).unwrap();
        index
            .add(make_meta("codellama-7b", "CodeLlama 7B"))
            .unwrap();

        let results = index.search("llama");
        assert_eq!(results.len(), 2);

        let results = index.search("qwen");
        assert_eq!(results.len(), 1);

        let results = index.search("nonexistent");
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_case_insensitive() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("registry.json");
        let mut index = LocalIndex::load(&path).unwrap();

        index.add(make_meta("llama-8b", "Llama 8B")).unwrap();

        assert_eq!(index.search("LLAMA").len(), 1);
        assert_eq!(index.search("Llama").len(), 1);
        assert_eq!(index.search("llama").len(), 1);
    }
}
