use crate::error::Result;
use crate::registry::local::LocalIndex;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Disk usage information for a single model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStorageInfo {
    pub model_id: String,
    pub model_name: String,
    pub size_bytes: u64,
    pub path: PathBuf,
    pub last_used: Option<chrono::DateTime<chrono::Utc>>,
}

/// Information about a partial download.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialDownloadInfo {
    pub model_id: String,
    pub filename: String,
    pub bytes_downloaded: u64,
    pub path: PathBuf,
}

/// Overall storage report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageReport {
    pub total_bytes: u64,
    pub models: Vec<ModelStorageInfo>,
    pub partial_downloads: Vec<PartialDownloadInfo>,
    pub orphaned_files: Vec<PathBuf>,
}

/// Manages disk usage for installed models.
pub struct StorageManager {
    models_dir: PathBuf,
    index: Arc<Mutex<LocalIndex>>,
}

impl StorageManager {
    pub fn new(models_dir: PathBuf, index: Arc<Mutex<LocalIndex>>) -> Self {
        Self { models_dir, index }
    }

    /// Generate a full storage report.
    pub async fn report(&self) -> Result<StorageReport> {
        let index = self.index.lock().await;
        let mut models = Vec::new();
        let mut total_bytes: u64 = 0;
        let mut partial_downloads = Vec::new();
        let mut orphaned_files = Vec::new();
        let mut tracked_dirs: Vec<PathBuf> = Vec::new();

        // Gather info from index
        for meta in index.all() {
            if let Some(ref installed) = meta.installed {
                let size = installed.size_bytes;
                total_bytes += size;
                tracked_dirs.push(installed.path.clone());
                models.push(ModelStorageInfo {
                    model_id: meta.id.clone(),
                    model_name: meta.name.clone(),
                    size_bytes: size,
                    path: installed.path.clone(),
                    last_used: installed.last_used,
                });
            }
        }

        // Sort by size descending
        models.sort_by(|a, b| b.size_bytes.cmp(&a.size_bytes));

        // Scan for partial downloads and orphaned files
        if self.models_dir.exists() {
            if let Ok(mut entries) = tokio::fs::read_dir(&self.models_dir).await {
                while let Ok(Some(entry)) = entries.next_entry().await {
                    let path = entry.path();
                    if path.is_dir() {
                        let is_tracked = tracked_dirs.iter().any(|t| t == &path);
                        if !is_tracked {
                            // Check for partial downloads
                            let mut has_partial = false;
                            if let Ok(mut sub) = tokio::fs::read_dir(&path).await {
                                while let Ok(Some(sub_entry)) = sub.next_entry().await {
                                    let name = sub_entry.file_name().to_string_lossy().to_string();
                                    if name.ends_with(".partial") {
                                        has_partial = true;
                                        let size = sub_entry
                                            .metadata()
                                            .await
                                            .map(|m| m.len())
                                            .unwrap_or(0);
                                        partial_downloads.push(PartialDownloadInfo {
                                            model_id: path
                                                .file_name()
                                                .unwrap_or_default()
                                                .to_string_lossy()
                                                .to_string(),
                                            filename: name,
                                            bytes_downloaded: size,
                                            path: sub_entry.path(),
                                        });
                                    }
                                }
                            }
                            if !has_partial {
                                orphaned_files.push(path);
                            }
                        }
                    }
                }
            }
        }

        Ok(StorageReport {
            total_bytes,
            models,
            partial_downloads,
            orphaned_files,
        })
    }

    /// Remove stale partial downloads (older than `max_age`).
    pub async fn cleanup_partial(&self, max_age: std::time::Duration) -> Result<Vec<PathBuf>> {
        let mut cleaned = Vec::new();

        if !self.models_dir.exists() {
            return Ok(cleaned);
        }

        let cutoff = std::time::SystemTime::now() - max_age;

        let mut entries = tokio::fs::read_dir(&self.models_dir).await?;
        while let Ok(Some(entry)) = entries.next_entry().await {
            if entry.file_type().await?.is_dir() {
                let mut sub = tokio::fs::read_dir(entry.path()).await?;
                while let Ok(Some(sub_entry)) = sub.next_entry().await {
                    let name = sub_entry.file_name().to_string_lossy().to_string();
                    if name.ends_with(".partial") || name.ends_with(".partial.meta") {
                        if let Ok(meta) = sub_entry.metadata().await {
                            if let Ok(modified) = meta.modified() {
                                if modified < cutoff {
                                    let path = sub_entry.path();
                                    tokio::fs::remove_file(&path).await.ok();
                                    cleaned.push(path);
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(cleaned)
    }

    /// Find files in models_dir not tracked by the index.
    pub async fn find_orphans(&self) -> Result<Vec<PathBuf>> {
        let index = self.index.lock().await;
        let tracked: Vec<PathBuf> = index
            .all()
            .iter()
            .filter_map(|m| m.installed.as_ref().map(|i| i.path.clone()))
            .collect();

        let mut orphans = Vec::new();

        if !self.models_dir.exists() {
            return Ok(orphans);
        }

        let mut entries = tokio::fs::read_dir(&self.models_dir).await?;
        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            if path.is_dir() && !tracked.contains(&path) {
                orphans.push(path);
            }
        }

        Ok(orphans)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::local::LocalIndex;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_empty_storage_report() {
        let tmp = TempDir::new().unwrap();
        let index_path = tmp.path().join("registry.json");
        let index = LocalIndex::load(&index_path).unwrap();
        let manager = StorageManager::new(tmp.path().join("models"), Arc::new(Mutex::new(index)));

        let report = manager.report().await.unwrap();
        assert_eq!(report.total_bytes, 0);
        assert!(report.models.is_empty());
        assert!(report.partial_downloads.is_empty());
        assert!(report.orphaned_files.is_empty());
    }

    #[tokio::test]
    async fn test_find_orphans_empty() {
        let tmp = TempDir::new().unwrap();
        let index_path = tmp.path().join("registry.json");
        let index = LocalIndex::load(&index_path).unwrap();
        let manager = StorageManager::new(tmp.path().join("models"), Arc::new(Mutex::new(index)));

        let orphans = manager.find_orphans().await.unwrap();
        assert!(orphans.is_empty());
    }

    #[tokio::test]
    async fn test_find_orphans_with_untracked_dir() {
        let tmp = TempDir::new().unwrap();
        let models_dir = tmp.path().join("models");
        tokio::fs::create_dir_all(models_dir.join("orphan-model"))
            .await
            .unwrap();

        let index_path = tmp.path().join("registry.json");
        let index = LocalIndex::load(&index_path).unwrap();
        let manager = StorageManager::new(models_dir, Arc::new(Mutex::new(index)));

        let orphans = manager.find_orphans().await.unwrap();
        assert_eq!(orphans.len(), 1);
        assert!(orphans[0].ends_with("orphan-model"));
    }
}
