#[cfg(not(target_arch = "wasm32"))]
use directories::ProjectDirs;
use std::path::PathBuf;

#[cfg(not(target_arch = "wasm32"))]
const QUALIFIER: &str = "com";
#[cfg(not(target_arch = "wasm32"))]
const ORGANIZATION: &str = "HiveBear";
#[cfg(not(target_arch = "wasm32"))]
const APPLICATION: &str = "hivebear";

/// Paths for HiveBear data, config, and cache.
pub struct AppPaths {
    pub config_dir: PathBuf,
    pub config_file: PathBuf,
    pub data_dir: PathBuf,
    pub models_dir: PathBuf,
    pub cache_dir: PathBuf,
    pub benchmark_cache: PathBuf,
    pub db_file: PathBuf,
}

impl AppPaths {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Self {
        let dirs = ProjectDirs::from(QUALIFIER, ORGANIZATION, APPLICATION);

        match dirs {
            Some(dirs) => {
                let config_dir = dirs.config_dir().to_path_buf();
                let data_dir = dirs.data_dir().to_path_buf();
                let cache_dir = dirs.cache_dir().to_path_buf();

                Self {
                    config_file: config_dir.join("config.toml"),
                    config_dir,
                    models_dir: data_dir.join("models"),
                    db_file: data_dir.join("hivebear.db"),
                    data_dir,
                    benchmark_cache: cache_dir.join("benchmark.json"),
                    cache_dir,
                }
            }
            None => {
                // Fallback to home directory
                let home = std::env::var("HOME")
                    .map(PathBuf::from)
                    .unwrap_or_else(|_| PathBuf::from("."));
                let base = home.join(".hivebear");

                Self {
                    config_dir: base.join("config"),
                    config_file: base.join("config").join("config.toml"),
                    data_dir: base.join("data"),
                    models_dir: base.join("data").join("models"),
                    db_file: base.join("data").join("hivebear.db"),
                    cache_dir: base.join("cache"),
                    benchmark_cache: base.join("cache").join("benchmark.json"),
                }
            }
        }
    }

    /// In WASM there is no filesystem. Return dummy paths.
    #[cfg(target_arch = "wasm32")]
    pub fn new() -> Self {
        Self {
            config_dir: PathBuf::from("/hivebear/config"),
            config_file: PathBuf::from("/hivebear/config/config.toml"),
            data_dir: PathBuf::from("/hivebear/data"),
            models_dir: PathBuf::from("/hivebear/data/models"),
            db_file: PathBuf::from("/hivebear/data/hivebear.db"),
            cache_dir: PathBuf::from("/hivebear/cache"),
            benchmark_cache: PathBuf::from("/hivebear/cache/benchmark.json"),
        }
    }

    /// Ensure all directories exist.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn ensure_dirs(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.config_dir)?;
        std::fs::create_dir_all(&self.data_dir)?;
        std::fs::create_dir_all(&self.models_dir)?;
        std::fs::create_dir_all(&self.cache_dir)?;
        Ok(())
    }

    /// No-op on WASM — no filesystem to create directories in.
    #[cfg(target_arch = "wasm32")]
    pub fn ensure_dirs(&self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Default for AppPaths {
    fn default() -> Self {
        Self::new()
    }
}
