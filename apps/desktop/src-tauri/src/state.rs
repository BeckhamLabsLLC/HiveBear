use hivebear_core::{AppPaths, Config, HardwareProfile};
use hivebear_inference::Orchestrator;
use hivebear_mesh::MeshNode;
use hivebear_persistence::ChatDatabase;
use hivebear_registry::Registry;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tracing::info;

/// Shared application state managed by Tauri.
pub struct AppState {
    pub config: Mutex<Config>,
    pub profile: HardwareProfile,
    pub orchestrator: Orchestrator,
    pub registry: Registry,
    pub chat_db: ChatDatabase,
    pub paths: AppPaths,
    pub http_client: reqwest::Client,
    pub mesh_node: Mutex<Option<Arc<MeshNode>>>,
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
            mesh_node: Mutex::new(None),
        }
    }

    /// Start the mesh node: register with the coordination server and begin heartbeats.
    ///
    /// Uses the same persistent Ed25519 identity as device-key auth.
    /// Safe to call multiple times — no-ops if already running.
    pub fn start_mesh(&self) -> Result<(), String> {
        {
            let existing = self.mesh_node.lock().unwrap();
            if existing.as_ref().is_some_and(|n| n.is_running()) {
                return Ok(());
            }
        }

        let config = self.config.lock().unwrap();
        if !config.mesh.enabled {
            return Err("Mesh is disabled in settings".into());
        }

        let tier = hivebear_mesh::MeshTier::from_str_lossy(&config.mesh.tier);
        let identity_path = self.paths.data_dir.join("node_identity.key");
        let identity = hivebear_mesh::NodeIdentity::load_or_generate(&identity_path)
            .map_err(|e| format!("Failed to load identity: {e}"))?;

        let security_mode = hivebear_mesh::MeshSecurityMode::default();
        let transport: Arc<dyn hivebear_mesh::MeshTransport> =
            Arc::new(hivebear_mesh::transport::quic::QuicTransport::new(
                identity.node_id.clone(),
                security_mode,
                None,
            ));
        let discovery: Arc<dyn hivebear_mesh::discovery::PeerDiscovery> = Arc::new(
            hivebear_mesh::discovery::server::CoordinationServerClient::new(
                config.mesh.coordination_server.clone(),
            ),
        );

        let reputation_path = Some(self.paths.data_dir.join("reputation.json"));
        let node = Arc::new(MeshNode::with_identity(
            identity,
            transport,
            discovery,
            tier,
            reputation_path,
        ));

        let listen_addr: std::net::SocketAddr =
            format!("0.0.0.0:{}", config.mesh.port).parse().unwrap();
        let total_vram: u64 = self.profile.gpus.iter().map(|g| g.vram_bytes).sum();

        let local_info = hivebear_mesh::PeerInfo {
            node_id: node.local_id.clone(),
            hardware: self.profile.clone(),
            available_memory_bytes: self.profile.memory.available_bytes,
            available_vram_bytes: total_vram,
            network_bandwidth_mbps: 100.0,
            latency_ms: None,
            tier,
            reputation_score: 1.0,
            addr: listen_addr,
            external_addr: None,
            nat_type: hivebear_mesh::NatType::Unknown,
            latency_map: std::collections::HashMap::new(),
            serving_model_id: None,
            swarm_id: None,
            draft_capability: None,
        };

        // Start in background — never blocks the caller
        node.start_background(listen_addr, local_info);
        info!("Mesh node starting in background");

        *self.mesh_node.lock().unwrap() = Some(node);
        Ok(())
    }

    /// Stop the mesh node gracefully: deregister and disconnect all peers.
    pub async fn stop_mesh(&self) -> Result<(), String> {
        let node = self.mesh_node.lock().unwrap().take();
        if let Some(node) = node {
            node.stop()
                .await
                .map_err(|e| format!("Failed to stop mesh: {e}"))?;
            info!("Mesh node stopped");
        }
        Ok(())
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
