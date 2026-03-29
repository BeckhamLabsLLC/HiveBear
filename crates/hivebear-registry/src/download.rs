use crate::error::{RegistryError, Result};
use chrono::{DateTime, Utc};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use tokio::io::AsyncWriteExt;

/// Progress information for download callbacks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    pub bytes_downloaded: u64,
    pub total_bytes: Option<u64>,
    pub bytes_per_sec: f64,
}

/// Metadata stored alongside partial downloads for resume support.
#[derive(Debug, Serialize, Deserialize)]
pub struct PartialMeta {
    url: String,
    total_bytes: Option<u64>,
    sha256_expected: Option<String>,
    model_id: String,
    filename: String,
    started_at: DateTime<Utc>,
}

/// Manages model file downloads with resume support.
pub struct DownloadManager {
    client: reqwest::Client,
    models_dir: PathBuf,
}

impl DownloadManager {
    pub fn new(models_dir: PathBuf) -> Self {
        let client = reqwest::Client::builder()
            .user_agent("hivebear/0.1.0")
            .build()
            .unwrap_or_default();

        Self { client, models_dir }
    }

    /// Send an HTTP GET with retry logic for transient errors (429, 5xx).
    async fn send_with_retry(
        &self,
        url: &str,
        bytes_downloaded: u64,
        max_retries: u32,
    ) -> Result<reqwest::Response> {
        let mut delay = std::time::Duration::from_secs(2);

        for attempt in 0..=max_retries {
            let mut request = self.client.get(url);
            if bytes_downloaded > 0 {
                request = request.header("Range", format!("bytes={bytes_downloaded}-"));
            }

            match request.send().await {
                Ok(response) => {
                    let status = response.status();
                    if status.is_success() || status.as_u16() == 206 {
                        return Ok(response);
                    }

                    // Transient errors: retry
                    if (status.as_u16() == 429 || status.is_server_error()) && attempt < max_retries
                    {
                        let msg = match status.as_u16() {
                            429 => "Rate limited by server".to_string(),
                            _ => format!("Server error (HTTP {})", status),
                        };
                        tracing::warn!(
                            "{msg}, retrying in {}s (attempt {}/{})",
                            delay.as_secs(),
                            attempt + 1,
                            max_retries
                        );
                        tokio::time::sleep(delay).await;
                        delay *= 2; // Exponential backoff
                        continue;
                    }

                    // Non-retryable error
                    let msg = match status.as_u16() {
                        401 | 403 => format!(
                            "Access denied (HTTP {status}). You may need to set HF_TOKEN for gated models."
                        ),
                        404 => "File not found (HTTP 404). The model file may have been moved or renamed.".to_string(),
                        429 => "Rate limited (HTTP 429). Try again in a few minutes, or set HF_TOKEN for higher limits.".to_string(),
                        _ => format!("HTTP {status} for {url}"),
                    };
                    return Err(RegistryError::DownloadError(msg));
                }
                Err(e) if attempt < max_retries => {
                    tracing::warn!(
                        "Network error: {e}, retrying in {}s (attempt {}/{})",
                        delay.as_secs(),
                        attempt + 1,
                        max_retries
                    );
                    tokio::time::sleep(delay).await;
                    delay *= 2;
                }
                Err(e) => {
                    return Err(RegistryError::DownloadError(format!(
                        "Download failed after {} attempts: {e}",
                        max_retries + 1
                    )));
                }
            }
        }

        unreachable!()
    }

    /// Download a file with resume support and optional SHA-256 verification.
    pub async fn download(
        &self,
        url: &str,
        dest_dir: &Path,
        filename: &str,
        expected_sha256: Option<&str>,
        model_id: &str,
        progress_cb: Option<&(dyn Fn(DownloadProgress) + Send + Sync)>,
    ) -> Result<PathBuf> {
        tokio::fs::create_dir_all(dest_dir).await?;

        let final_path = dest_dir.join(filename);
        let partial_path = dest_dir.join(format!("{filename}.partial"));
        let meta_path = dest_dir.join(format!("{filename}.partial.meta"));

        // Check for existing partial download
        let mut bytes_downloaded: u64 = 0;
        let mut hasher = Sha256::new();

        if partial_path.exists() && meta_path.exists() {
            // Read partial metadata
            let meta_contents = tokio::fs::read_to_string(&meta_path).await?;
            if let Ok(meta) = serde_json::from_str::<PartialMeta>(&meta_contents) {
                if meta.url == url {
                    bytes_downloaded = tokio::fs::metadata(&partial_path).await?.len();
                    tracing::info!("Resuming download from {} bytes", bytes_downloaded);

                    // Re-hash the existing partial file for integrity
                    if expected_sha256.is_some() {
                        let existing = tokio::fs::read(&partial_path).await?;
                        hasher.update(&existing);
                    }
                } else {
                    // URL changed, start fresh
                    tokio::fs::remove_file(&partial_path).await.ok();
                    tokio::fs::remove_file(&meta_path).await.ok();
                }
            }
        }

        // Save partial metadata
        let meta = PartialMeta {
            url: url.to_string(),
            total_bytes: None,
            sha256_expected: expected_sha256.map(String::from),
            model_id: model_id.to_string(),
            filename: filename.to_string(),
            started_at: Utc::now(),
        };
        let meta_json = serde_json::to_string(&meta)?;
        tokio::fs::write(&meta_path, &meta_json).await?;

        // Build request with Range header for resume, with retry on transient errors
        let response = self.send_with_retry(url, bytes_downloaded, 3).await?;

        let total_bytes = response.content_length().map(|cl| cl + bytes_downloaded);

        // Open file for appending (resume) or creating
        let mut file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&partial_path)
            .await?;

        let mut stream = response.bytes_stream();
        let start_time = std::time::Instant::now();
        let bytes_at_start = bytes_downloaded;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| RegistryError::DownloadError(e.to_string()))?;
            file.write_all(&chunk).await?;

            if expected_sha256.is_some() {
                hasher.update(&chunk);
            }

            bytes_downloaded += chunk.len() as u64;

            if let Some(cb) = progress_cb {
                let elapsed = start_time.elapsed().as_secs_f64();
                // Calculate speed based on bytes downloaded in this session (not total)
                let session_bytes = bytes_downloaded - bytes_at_start;
                let bps = if elapsed > 0.0 {
                    session_bytes as f64 / elapsed
                } else {
                    0.0
                };
                cb(DownloadProgress {
                    bytes_downloaded,
                    total_bytes,
                    bytes_per_sec: bps,
                });
            }
        }

        file.flush().await?;
        drop(file);

        // Verify SHA-256 if expected
        if let Some(expected) = expected_sha256 {
            let actual = format!("{:x}", hasher.finalize());
            if actual != expected {
                return Err(RegistryError::IntegrityError {
                    path: partial_path,
                    expected: expected.to_string(),
                    actual,
                });
            }
        }

        // Move partial to final
        tokio::fs::rename(&partial_path, &final_path).await?;
        tokio::fs::remove_file(&meta_path).await.ok();

        Ok(final_path)
    }

    /// List partial downloads that could be resumed.
    pub async fn list_partial(&self) -> Result<Vec<PartialMeta>> {
        let mut partials = Vec::new();

        if !self.models_dir.exists() {
            return Ok(partials);
        }

        let mut entries = tokio::fs::read_dir(&self.models_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            if entry.file_type().await?.is_dir() {
                let mut sub = tokio::fs::read_dir(entry.path()).await?;
                while let Some(sub_entry) = sub.next_entry().await? {
                    let name = sub_entry.file_name().to_string_lossy().to_string();
                    if name.ends_with(".partial.meta") {
                        if let Ok(contents) = tokio::fs::read_to_string(sub_entry.path()).await {
                            if let Ok(meta) = serde_json::from_str::<PartialMeta>(&contents) {
                                partials.push(meta);
                            }
                        }
                    }
                }
            }
        }

        Ok(partials)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_download_manager_creation() {
        let tmp = TempDir::new().unwrap();
        let dm = DownloadManager::new(tmp.path().to_path_buf());
        assert_eq!(dm.models_dir, tmp.path());
    }

    #[tokio::test]
    async fn test_list_partial_empty() {
        let tmp = TempDir::new().unwrap();
        let dm = DownloadManager::new(tmp.path().to_path_buf());
        let partials = dm.list_partial().await.unwrap();
        assert!(partials.is_empty());
    }
}
