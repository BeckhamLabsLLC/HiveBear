use hivebear_core::{AppPaths, Config, HardwareProfile};
use hivebear_inference::Orchestrator;
use hivebear_persistence::ChatDatabase;
use hivebear_registry::Registry;
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
        let paths = AppPaths::new();
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
}
