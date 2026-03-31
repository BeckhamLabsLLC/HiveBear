use hivebear_core::{AppPaths, Config, HardwareProfile};
use hivebear_inference::Orchestrator;
use hivebear_persistence::ChatDatabase;
use hivebear_registry::Registry;
use std::path::PathBuf;
use std::sync::Mutex;

/// Shared application state managed by Tauri.
pub struct AppState {
    pub config: Mutex<Config>,
    pub profile: HardwareProfile,
    pub orchestrator: Orchestrator,
    pub registry: Registry,
    pub chat_db: ChatDatabase,
    #[allow(dead_code)]
    pub paths: AppPaths,
    pub http_client: reqwest::Client,
}

impl AppState {
    pub fn init() -> Self {
        Self::init_with_paths(AppPaths::new())
    }

    /// Initialize with explicit paths — used on Android where the default
    /// `ProjectDirs` paths point to read-only locations.
    pub fn init_with_paths(paths: AppPaths) -> Self {
        paths
            .ensure_dirs()
            .expect("Failed to create app directories");

        let config = Config::load();
        let profile = hivebear_core::profile();
        let orchestrator = Orchestrator::with_config(profile.clone(), &config);
        let registry = tauri::async_runtime::block_on(Registry::new(&config, &paths))
            .expect("Failed to initialize model registry");
        let chat_db = ChatDatabase::open(&paths.db_file).expect("Failed to open chat database");

        Self {
            config: Mutex::new(config),
            profile,
            orchestrator,
            registry,
            chat_db,
            paths,
            http_client: reqwest::Client::new(),
        }
    }

    /// Create AppPaths from a base directory (e.g., Tauri's app_data_dir).
    pub fn paths_from_base(base: PathBuf) -> AppPaths {
        AppPaths {
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
