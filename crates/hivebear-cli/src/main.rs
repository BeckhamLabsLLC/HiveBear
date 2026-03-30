mod account_commands;
#[cfg(feature = "api")]
mod api;
mod pipeline_handler;
mod registry_commands;

use clap::{Parser, Subcommand};
use colored::Colorize;
use futures::StreamExt;
use hivebear_core::types::format_bytes;
use hivebear_core::{Config, HardwareProfile, ModelRecommendation};
use hivebear_inference::{ChatMessage, GenerateRequest, LoadConfig, Orchestrator, SamplingParams};
use hivebear_mesh::discovery::PeerDiscovery;

#[derive(Parser)]
#[command(
    name = "hivebear",
    about = "Run open-source AI models on any device",
    version,
    long_about = "HiveBear makes open-source LLMs accessible on any hardware.\n\
                   It profiles your device, recommends the best models,\n\
                   and optimizes inference for your specific hardware."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose logging
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Show hardware profile for this device
    Profile,

    /// Get model recommendations for this device
    Recommend {
        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Maximum number of recommendations
        #[arg(short, long)]
        top: Option<usize>,
    },

    /// Run inference benchmark
    Benchmark {
        /// Duration in seconds (for synthetic benchmark)
        #[arg(short, long, default_value = "5")]
        duration: u32,

        /// Model to benchmark (runs real inference benchmark instead of synthetic)
        #[arg(short, long)]
        model: Option<String>,

        /// Number of tokens to generate per iteration
        #[arg(long, default_value = "256")]
        generate_tokens: u32,

        /// Number of benchmark iterations
        #[arg(long, default_value = "3")]
        iterations: u32,
    },

    /// View or manage configuration
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },

    /// Run inference on a model (file path or installed model ID)
    Run {
        /// Model file path or installed model ID
        model: String,

        /// Single prompt (non-interactive mode)
        #[arg(long)]
        prompt: Option<String>,

        /// Start as OpenAI-compatible API server
        #[arg(long)]
        api: bool,

        /// API server port
        #[arg(long, default_value = "8080")]
        port: u16,

        /// Context length
        #[arg(long, default_value = "4096")]
        context_length: u32,

        /// Number of GPU layers to offload (default: auto)
        #[arg(long)]
        gpu_layers: Option<u32>,

        /// Temperature for sampling
        #[arg(long, default_value = "0.7")]
        temperature: f32,

        /// Disable API authentication
        #[arg(long)]
        no_auth: bool,

        /// API key (overrides config; auto-generated if not set)
        #[arg(long)]
        api_key: Option<String>,

        /// Bind address for API server
        #[arg(long)]
        bind: Option<String>,
    },

    /// List available inference engines
    Engines,

    /// Search for models on HuggingFace
    Search {
        /// Search query (e.g., "code generation", "llama 8b")
        query: String,

        /// Maximum results to show
        #[arg(short, long, default_value = "10")]
        limit: usize,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Download and install a model
    Install {
        /// Model ID (e.g., "llama-3.1-8b") or HuggingFace repo (e.g., "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF")
        model: String,

        /// Specific quantization to download (e.g., "Q4_K_M")
        #[arg(short, long)]
        quant: Option<String>,

        /// Specific file to download from the repo
        #[arg(long)]
        file: Option<String>,
    },

    /// List installed models
    List {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Remove an installed model
    Remove {
        /// Model ID to remove
        model: String,

        /// Skip confirmation prompt
        #[arg(short = 'y', long)]
        yes: bool,
    },

    /// Convert a model to a different format
    Convert {
        /// Model ID or path to convert
        model: String,

        /// Target format (e.g., "gguf")
        #[arg(long)]
        to: String,

        /// Target quantization (e.g., "Q4_K_M")
        #[arg(long)]
        quant: Option<String>,
    },

    /// Show disk usage for installed models
    Storage {
        /// Clean up stale partial downloads
        #[arg(long)]
        cleanup: bool,
    },

    /// Import models from a local Ollama installation
    ImportOllama {
        /// Specific model to import (e.g., "llama3:8b"). Lists all if omitted.
        model: Option<String>,
    },

    /// P2P mesh distributed inference
    Mesh {
        #[command(subcommand)]
        action: MeshAction,
    },

    /// Contribute your compute to the network and earn the right to use it
    Contribute {
        /// QUIC listen port
        #[arg(long, default_value = "7878")]
        port: u16,

        /// Override model to serve (auto-selected if omitted)
        #[arg(long)]
        model: Option<String>,

        /// Coordinator server URL
        #[arg(long, default_value = "http://localhost:7879")]
        coordinator: String,
    },

    /// Profile hardware, recommend a model, install it, and start chatting — all in one command
    Quickstart {
        /// Temperature for sampling
        #[arg(long, default_value = "0.7")]
        temperature: f32,

        /// Context length
        #[arg(long, default_value = "4096")]
        context_length: u32,
    },

    /// Manage your account (login, device activation, billing, API keys)
    Account {
        #[command(subcommand)]
        action: account_commands::AccountAction,
    },

    /// Update HiveBear to the latest version
    Update {
        /// Check for updates without installing
        #[arg(long)]
        check: bool,
    },

    /// Uninstall HiveBear from this system
    Uninstall {
        /// Also remove downloaded models, config, and all data
        #[arg(long)]
        purge: bool,

        /// Skip confirmation prompt
        #[arg(short = 'y', long)]
        yes: bool,
    },
}

#[derive(Subcommand)]
enum MeshAction {
    /// Join the mesh as a contributor
    Start {
        /// QUIC listen port
        #[arg(long, default_value = "7878")]
        port: u16,
    },

    /// Show mesh status (connected peers, contribution stats)
    Status,

    /// Run a model across the mesh
    Run {
        /// Model file path or installed model ID
        model: String,

        /// Single prompt (non-interactive mode)
        #[arg(long)]
        prompt: Option<String>,

        /// Context length
        #[arg(long, default_value = "4096")]
        context_length: u32,

        /// Temperature for sampling
        #[arg(long, default_value = "0.7")]
        temperature: f32,
    },

    /// Leave the mesh
    Stop,
}

#[derive(Subcommand)]
enum ConfigAction {
    /// Show current configuration
    Show,
    /// Show configuration file path
    Path,
    /// Reset configuration to defaults
    Reset,
}

/// Auto-start a background mesh node if mesh is enabled and auto_join is true.
///
/// Returns `Some(Arc<MeshNode>)` if the node was started (or is starting),
/// `None` if mesh is disabled or the command doesn't benefit from it.
fn maybe_start_mesh(
    config: &hivebear_core::Config,
    hw: &HardwareProfile,
) -> Option<std::sync::Arc<hivebear_mesh::MeshNode>> {
    use std::sync::Arc;

    if !config.mesh.enabled || !config.mesh.auto_join {
        return None;
    }

    let tier = hivebear_mesh::MeshTier::from_str_lossy(&config.mesh.tier);
    let paths = hivebear_core::config::paths::AppPaths::new();
    let identity_path = paths.data_dir.join("node_identity.key");
    let identity = hivebear_mesh::NodeIdentity::load_or_generate(&identity_path)
        .unwrap_or_else(|_| hivebear_mesh::NodeIdentity::generate());

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

    let reputation_path = Some(paths.data_dir.join("reputation.json"));
    let node = Arc::new(hivebear_mesh::MeshNode::with_identity(
        identity,
        transport,
        discovery,
        tier,
        reputation_path,
    ));

    let listen_addr: std::net::SocketAddr =
        format!("0.0.0.0:{}", config.mesh.port).parse().unwrap();
    let total_vram: u64 = hw.gpus.iter().map(|g| g.vram_bytes).sum();

    let local_info = hivebear_mesh::PeerInfo {
        node_id: node.local_id.clone(),
        hardware: hw.clone(),
        available_memory_bytes: hw.memory.available_bytes,
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

    // Start in background — never blocks the CLI
    node.start_background(listen_addr, local_info);

    Some(node)
}

/// Global mesh node reference, set during startup if auto-join is enabled.
/// Accessed by command functions to register the mesh backend with their Orchestrator.
static MESH_NODE: std::sync::OnceLock<std::sync::Arc<hivebear_mesh::MeshNode>> =
    std::sync::OnceLock::new();

/// Create an Orchestrator with mesh backend auto-registered if a mesh node is active.
fn create_orchestrator(hw: HardwareProfile) -> Orchestrator {
    let mut orchestrator = Orchestrator::new(hw);
    if let Some(node) = MESH_NODE.get() {
        let mesh_backend = hivebear_mesh::MeshBackend::new(std::sync::Arc::clone(node));
        orchestrator.register_backend(Box::new(mesh_backend));
    }
    orchestrator
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();

    // Set up logging
    let filter = if cli.verbose { "debug" } else { "warn" };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(filter)),
        )
        .with_target(false)
        .init();

    // Auto-start mesh for commands that benefit from it
    let needs_mesh = matches!(
        cli.command,
        Commands::Run { .. }
            | Commands::Mesh { .. }
            | Commands::Contribute { .. }
            | Commands::Quickstart { .. }
    );

    let config = hivebear_core::Config::load();
    if needs_mesh {
        let hw = hivebear_core::profile();
        if let Some(node) = maybe_start_mesh(&config, &hw) {
            let _ = MESH_NODE.set(node);
        }
    }

    match cli.command {
        Commands::Profile => cmd_profile(),
        Commands::Recommend { json, top } => cmd_recommend(json, top),
        Commands::Benchmark {
            duration,
            model,
            generate_tokens,
            iterations,
        } => cmd_benchmark(duration, model, generate_tokens, iterations).await,
        Commands::Config { action } => cmd_config(action),
        Commands::Run {
            model,
            prompt,
            api,
            port,
            context_length,
            gpu_layers,
            temperature,
            no_auth,
            api_key,
            bind,
        } => {
            cmd_run(
                model,
                prompt,
                api,
                port,
                context_length,
                gpu_layers,
                temperature,
                no_auth,
                api_key,
                bind,
            )
            .await
        }
        Commands::Engines => cmd_engines(),
        Commands::Search { query, limit, json } => {
            registry_commands::cmd_search(query, limit, json).await
        }
        Commands::Install { model, quant, file } => {
            registry_commands::cmd_install(model, quant, file).await
        }
        Commands::List { json } => registry_commands::cmd_list(json).await,
        Commands::Remove { model, yes } => registry_commands::cmd_remove(model, yes).await,
        Commands::Convert { model, to, quant } => {
            registry_commands::cmd_convert(model, to, quant).await
        }
        Commands::Storage { cleanup } => registry_commands::cmd_storage(cleanup).await,
        Commands::ImportOllama { model } => registry_commands::cmd_import_ollama(model).await,
        Commands::Mesh { action } => cmd_mesh(action).await,
        Commands::Contribute {
            port,
            model,
            coordinator,
        } => cmd_contribute(port, model, coordinator).await,
        Commands::Quickstart {
            temperature,
            context_length,
        } => cmd_quickstart(temperature, context_length).await,
        Commands::Account { action } => account_commands::cmd_account(action).await,
        Commands::Update { check } => cmd_update(check).await,
        Commands::Uninstall { purge, yes } => cmd_uninstall(purge, yes).await,
    }

    // Graceful mesh shutdown
    if let Some(node) = MESH_NODE.get() {
        let _ = node.stop().await;
    }
}

fn cmd_profile() {
    println!(
        "\n{}",
        "  HiveBear Hardware Profile  ".bold().white().on_blue()
    );
    println!();

    let hw = hivebear_core::profile();
    print_profile(&hw);
}

fn print_profile(hw: &HardwareProfile) {
    // CPU
    println!("{}", "CPU".bold().cyan());
    println!("  Model:       {}", hw.cpu.model_name);
    println!(
        "  Cores:       {} physical, {} logical",
        hw.cpu.physical_cores, hw.cpu.logical_cores
    );
    if !hw.cpu.isa_extensions.is_empty() {
        println!("  Extensions:  {}", hw.cpu.isa_extensions.join(", "));
    }
    if hw.cpu.cache_size_bytes > 0 {
        println!("  Cache:       {}", format_bytes(hw.cpu.cache_size_bytes));
    }
    println!();

    // Memory
    println!("{}", "Memory".bold().cyan());
    println!("  Total:       {}", format_bytes(hw.memory.total_bytes));
    println!("  Available:   {}", format_bytes(hw.memory.available_bytes));
    println!(
        "  Bandwidth:   {:.1} GB/s (estimated)",
        hw.memory.estimated_bandwidth_gbps
    );
    println!();

    // GPUs
    if hw.gpus.is_empty() {
        println!("{}", "GPU".bold().cyan());
        println!("  {}", "No discrete GPU detected".yellow());
    } else {
        for (i, gpu) in hw.gpus.iter().enumerate() {
            println!("{}", format!("GPU {}", i).bold().cyan());
            println!("  Name:        {}", gpu.name);
            println!("  VRAM:        {}", format_bytes(gpu.vram_bytes));
            println!("  API:         {}", gpu.compute_api);
            if let Some(driver) = &gpu.driver_version {
                println!("  Driver:      {}", driver);
            }
        }
    }
    println!();

    // Storage
    println!("{}", "Storage".bold().cyan());
    println!(
        "  Available:   {}",
        format_bytes(hw.storage.available_bytes)
    );
    println!(
        "  Read speed:  {:.0} MB/s (estimated)",
        hw.storage.estimated_read_speed_mbps
    );
    println!();

    // Platform
    println!("{}", "Platform".bold().cyan());
    println!("  OS:          {}", hw.platform.os);
    println!("  Arch:        {}", hw.platform.arch);
    println!(
        "  Power:       {}",
        match &hw.platform.power_source {
            hivebear_core::types::PowerSource::Ac => "AC power".to_string(),
            hivebear_core::types::PowerSource::Battery { charge_percent } =>
                format!("Battery ({}%)", charge_percent),
            hivebear_core::types::PowerSource::Unknown => "Unknown".to_string(),
        }
    );
}

fn cmd_recommend(json: bool, top: Option<usize>) {
    let hw = hivebear_core::profile();
    let mut config = Config::load();

    if let Some(n) = top {
        config.top_n_recommendations = n;
    }

    let recs = hivebear_core::recommender::recommend(&hw, &config);

    if json {
        println!("{}", serde_json::to_string_pretty(&recs).unwrap());
        return;
    }

    println!(
        "\n{}",
        "  HiveBear Model Recommendations  "
            .bold()
            .white()
            .on_green()
    );
    println!();

    if recs.is_empty() {
        println!(
            "{}",
            "No models found that can run on this hardware at acceptable speed."
                .yellow()
                .bold()
        );
        println!("Try lowering the minimum tok/s threshold in your config.");
        return;
    }

    println!(
        "Based on: {} | {} RAM | {} available",
        hw.cpu.model_name.bold(),
        format_bytes(hw.memory.total_bytes),
        format_bytes(hw.memory.available_bytes)
    );

    if !hw.gpus.is_empty() {
        for gpu in &hw.gpus {
            println!(
                "          {} | {} VRAM",
                gpu.name.bold(),
                format_bytes(gpu.vram_bytes)
            );
        }
    }

    println!();
    print_recommendations(&recs);
}

fn print_recommendations(recs: &[ModelRecommendation]) {
    // Table header
    println!(
        "  {:<4} {:<25} {:<10} {:<12} {:<10} {:<10} {:<6}",
        "#".bold(),
        "Model".bold(),
        "Quant".bold(),
        "Engine".bold(),
        "Est. tok/s".bold(),
        "Memory".bold(),
        "Conf.".bold()
    );
    println!("  {}", "-".repeat(80));

    for (i, rec) in recs.iter().enumerate() {
        let rank = format!("{}", i + 1);
        let tok_s = format!("{:.1}", rec.estimated_tokens_per_sec);
        let memory = format_bytes(rec.estimated_memory_usage_bytes);
        let confidence = format!("{:.0}%", rec.confidence * 100.0);

        let tok_s_colored = if rec.estimated_tokens_per_sec >= 20.0 {
            tok_s.green()
        } else if rec.estimated_tokens_per_sec >= 10.0 {
            tok_s.yellow()
        } else {
            tok_s.red()
        };

        println!(
            "  {:<4} {:<25} {:<10} {:<12} {:<10} {:<10} {:<6}",
            rank,
            rec.model_name,
            rec.quantization.to_string(),
            rec.engine.to_string(),
            tok_s_colored,
            memory,
            confidence
        );

        // Print warnings indented
        for warning in &rec.warnings {
            println!("       {}", format!("  {warning}").yellow());
        }
    }

    println!();
    println!(
        "{}",
        "Tip: Use 'hivebear benchmark' for more accurate performance estimates.".dimmed()
    );
}

async fn cmd_benchmark(
    duration: u32,
    model: Option<String>,
    generate_tokens: u32,
    iterations: u32,
) {
    println!("\n{}", "  HiveBear Benchmark  ".bold().white().on_magenta());
    println!();

    if let Some(model_id) = model {
        // Real inference benchmark
        use std::path::Path;

        println!(
            "Running {} real inference benchmark(s)...",
            "real".bold().green()
        );
        println!();

        let hw = hivebear_core::profile();
        let model_path_str = registry_commands::resolve_model(&model_id, &hw).await;
        let orchestrator = create_orchestrator(hw);

        let load_config = hivebear_inference::LoadConfig {
            context_length: 4096,
            offload: hivebear_inference::OffloadConfig {
                auto: true,
                ..Default::default()
            },
            ..Default::default()
        };

        println!("Loading model: {}", model_id.bold());
        let handle = match orchestrator
            .load(Path::new(&model_path_str), &load_config)
            .await
        {
            Ok(h) => {
                println!("{}", format!("  Engine: {}", h.engine).dimmed());
                h
            }
            Err(e) => {
                eprintln!("{}: {e}", "Failed to load model".red().bold());
                return;
            }
        };

        let bench_config = hivebear_inference::benchmark::BenchmarkConfig {
            prefill_tokens: 128,
            generate_tokens,
            warmup_runs: 1,
            iterations,
        };

        println!();
        println!(
            "Config: {} prefill tokens, {} generate tokens, {} warmup, {} iteration(s)",
            bench_config.prefill_tokens,
            bench_config.generate_tokens,
            bench_config.warmup_runs,
            bench_config.iterations
        );
        println!();

        match hivebear_inference::benchmark::run_inference_benchmark(
            &orchestrator,
            &handle,
            &model_id,
            &bench_config,
        )
        .await
        {
            Ok(result) => {
                println!("{}", "Results".bold().cyan());
                println!("  Model:            {}", result.model_used);
                println!(
                    "  Type:             {}",
                    result.benchmark_type.bold().green()
                );
                println!("  Tokens generated: {}", result.tokens_generated);
                println!("  Time to 1st tok:  {} ms", result.time_to_first_token_ms);
                println!(
                    "  Generate tok/s:   {}",
                    format!("{:.1}", result.tokens_per_sec).bold()
                );
                if let Some(prefill) = result.prompt_eval_tokens_per_sec {
                    println!("  Prefill tok/s:    {:.1}", prefill);
                }
                println!("  Total duration:   {} ms", result.total_duration_ms);
                if result.peak_memory_bytes > 0 {
                    println!(
                        "  Peak memory:      {}",
                        hivebear_core::types::format_bytes(result.peak_memory_bytes)
                    );
                }
            }
            Err(e) => {
                eprintln!("{}: {e}", "Benchmark failed".red().bold());
            }
        }

        if let Err(e) = orchestrator.unload(&handle).await {
            tracing::warn!("Failed to unload model: {e}");
        }
    } else {
        // Synthetic benchmark (fallback)
        println!("Running {duration}s {} benchmark...", "synthetic".yellow());
        println!(
            "{}",
            "Tip: Use 'hivebear benchmark --model <name>' for real inference benchmarks.".dimmed()
        );
        println!();

        let result =
            hivebear_core::benchmark::run_benchmark(hivebear_core::types::ProfileMode::Benchmark {
                duration_secs: duration,
            });

        match result {
            Some(result) => {
                let report = hivebear_core::benchmark::report::format_report(&result);
                println!("{report}");
            }
            None => {
                println!("{}", "Benchmark could not be completed.".red());
            }
        }
    }
}

fn cmd_config(action: ConfigAction) {
    let paths = hivebear_core::config::paths::AppPaths::new();

    match action {
        ConfigAction::Show => {
            let config = Config::load();
            println!("{}", toml::to_string_pretty(&config).unwrap());
        }
        ConfigAction::Path => {
            println!("{}", paths.config_file.display());
        }
        ConfigAction::Reset => {
            let config = Config::default();
            match config.save() {
                Ok(()) => println!("{}", "Configuration reset to defaults.".green()),
                Err(e) => eprintln!("{}: {e}", "Failed to save config".red()),
            }
        }
    }
}

fn generate_api_key() -> String {
    format!(
        "op-{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    )
}

#[allow(clippy::too_many_arguments)]
async fn cmd_run(
    model: String,
    prompt: Option<String>,
    api: bool,
    port: u16,
    context_length: u32,
    gpu_layers: Option<u32>,
    temperature: f32,
    no_auth: bool,
    cli_api_key: Option<String>,
    bind: Option<String>,
) {
    use std::io::Write;
    use std::path::Path;

    println!("\n{}", "  HiveBear Inference  ".bold().white().on_blue());
    println!();

    let hw = hivebear_core::profile();

    // Check if this is a cloud model (e.g., "openai/gpt-4o", "groq/llama-3.1-70b", etc.)
    #[cfg(feature = "cloud")]
    let is_cloud = hivebear_inference::engine::cloud::registry::is_cloud_model(&model);
    #[cfg(not(feature = "cloud"))]
    let is_cloud = false;

    let config = hivebear_core::config::Config::load();
    let orchestrator = Orchestrator::with_config(hw.clone(), &config);

    println!("Loading model: {}", model.bold());

    let handle = if is_cloud {
        #[cfg(feature = "cloud")]
        {
            match orchestrator.load_cloud(&model).await {
                Ok(h) => {
                    println!("{}", format!("  Engine: Cloud | Model: {}", model).dimmed());
                    println!();
                    h
                }
                Err(e) => {
                    eprintln!("{}: {e}", "Failed to load cloud model".red().bold());
                    return;
                }
            }
        }
        #[cfg(not(feature = "cloud"))]
        {
            eprintln!(
                "{}",
                "Cloud inference not available: build with --features cloud"
                    .red()
                    .bold()
            );
            return;
        }
    } else {
        // Resolve model ID through registry if it's not a file path
        let model = registry_commands::resolve_model(&model, &hw).await;

        let config = LoadConfig {
            context_length,
            offload: hivebear_inference::OffloadConfig {
                gpu_layers,
                auto: gpu_layers.is_none(),
                ..Default::default()
            },
            ..Default::default()
        };

        let model_path = Path::new(&model);

        match orchestrator.load(model_path, &config).await {
            Ok(h) => {
                println!(
                    "{}",
                    format!(
                        "  Engine: {} | Context: {} tokens",
                        h.engine, context_length
                    )
                    .dimmed()
                );
                if let Some(layers) = gpu_layers {
                    println!("{}", format!("  GPU layers: {layers}").dimmed());
                }
                println!();
                h
            }
            Err(e) => {
                eprintln!("{}: {e}", "Failed to load model".red().bold());
                return;
            }
        }
    };

    if api {
        #[cfg(feature = "api")]
        {
            let mut config = Config::load();

            let resolved_key = if no_auth {
                None
            } else if let Some(key) = cli_api_key {
                Some(key)
            } else if let Some(key) = config.api.api_key.clone() {
                Some(key)
            } else {
                // Auto-generate and save
                let key = generate_api_key();
                config.api.api_key = Some(key.clone());
                if let Err(e) = config.save() {
                    eprintln!("Warning: could not save generated API key: {e}");
                }
                Some(key)
            };

            let bind_addr = bind.as_deref().unwrap_or(&config.api.bind_address);

            // API server mode
            let model_name = if is_cloud {
                model.clone()
            } else {
                Path::new(&model)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string()
            };
            println!(
                "{}",
                format!("Starting API server on {bind_addr}:{port}...").green()
            );
            api::start_server(
                orchestrator,
                handle,
                model_name,
                port,
                resolved_key,
                bind_addr,
                &config.api.cors_origins,
            )
            .await;
            return;
        }
        #[cfg(not(feature = "api"))]
        {
            eprintln!(
                "{}",
                "API server not available: build with --features api"
                    .red()
                    .bold()
            );
            return;
        }
    }

    if let Some(prompt_text) = prompt {
        // Single prompt mode
        let req = GenerateRequest {
            messages: vec![ChatMessage::user_text(&prompt_text)],
            sampling: SamplingParams {
                temperature,
                ..Default::default()
            },
            model_name: Some(model.clone()),
            ..Default::default()
        };

        match orchestrator.stream(&handle, &req) {
            Ok(mut stream) => {
                while let Some(result) = stream.next().await {
                    match result {
                        Ok(token) => {
                            print!("{}", token.text);
                            std::io::stdout().flush().ok();
                        }
                        Err(e) => {
                            eprintln!("\n{}: {e}", "Generation error".red());
                            break;
                        }
                    }
                }
                println!();
            }
            Err(e) => {
                eprintln!("{}: {e}", "Stream error".red());
            }
        }
    } else {
        // Interactive chat mode
        println!("{}", "Ready. Type your message, or /quit to exit.".green());
        println!();

        let mut history: Vec<ChatMessage> = Vec::new();

        loop {
            print!("{} ", ">".bold().cyan());
            std::io::stdout().flush().ok();

            let mut input = String::new();
            if std::io::stdin().read_line(&mut input).is_err() {
                break;
            }

            let input = input.trim();
            if input.is_empty() {
                continue;
            }
            if input == "/quit" || input == "/exit" || input == "/q" {
                println!("Goodbye!");
                break;
            }

            history.push(ChatMessage::user_text(input));

            let req = GenerateRequest {
                messages: history.clone(),
                sampling: SamplingParams {
                    temperature,
                    ..Default::default()
                },
                model_name: Some(model.clone()),
                ..Default::default()
            };

            let mut response_text = String::new();
            match orchestrator.stream(&handle, &req) {
                Ok(mut stream) => {
                    while let Some(result) = stream.next().await {
                        match result {
                            Ok(token) => {
                                print!("{}", token.text);
                                std::io::stdout().flush().ok();
                                response_text.push_str(&token.text);
                            }
                            Err(e) => {
                                eprintln!("\n{}: {e}", "Generation error".red());
                                break;
                            }
                        }
                    }
                    println!("\n");
                }
                Err(e) => {
                    eprintln!("{}: {e}", "Stream error".red());
                    continue;
                }
            }

            history.push(ChatMessage::Assistant(response_text));
        }
    }

    // Cleanup
    if let Err(e) = orchestrator.unload(&handle).await {
        tracing::warn!("Failed to unload model: {e}");
    }
}

fn cmd_engines() {
    println!(
        "\n{}",
        "  HiveBear Inference Engines  ".bold().white().on_cyan()
    );
    println!();

    let registry = hivebear_inference::EngineRegistry::new();

    println!(
        "  {:<15} {:<12} {:<30}",
        "Engine".bold(),
        "Available".bold(),
        "Formats".bold()
    );
    println!("  {}", "-".repeat(60));

    for backend in registry.all_backends() {
        let name = backend.name();
        if name == "Dummy" {
            continue; // Skip test-only backend
        }

        let available = if backend.is_available() {
            "yes".green().to_string()
        } else {
            "no".red().to_string()
        };

        let formats: Vec<String> = backend
            .supported_formats()
            .iter()
            .map(|f| f.to_string())
            .collect();

        println!(
            "  {:<15} {:<12} {:<30}",
            name,
            available,
            formats.join(", ")
        );

        if backend.supports_grammar() {
            println!("  {}", "  Supports grammar-constrained decoding".dimmed());
        }
    }
    println!();
}

async fn cmd_quickstart(temperature: f32, context_length: u32) {
    use std::io::Write;
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    // ── Banner ──────────────────────────────────────────────────────
    println!();
    let banner = r#"
  _   _ _          ____
 | | | (_)_   ____|  _ \ ___  __ _ _ __
 | |_| | \ \ / / _ \ |_) / _ \/ _` | '__|
 |  _  | |\ V /  __/  _ <  __/ (_| | |
 |_| |_|_| \_/ \___|_| \_\___|\__,_|_|

        ▓█▓   ▓█▓   ▓█▓   ▓█▓
       ▓███▓ ▓███▓ ▓███▓ ▓███▓
        ▓█▓   ▓█▓   ▓█▓   ▓█▓

          ▓███████████████▓
        ▓███████████████████▓
       ▓█████████████████████▓
      ▓███████████████████████▓
      ▓███████████████████████▓
      ▓███████████████████████▓
       ▓█████████████████████▓
        ▓███████████████████▓
          ▓███████████████▓
"#;
    for line in banner.lines() {
        println!("{}", line.bold().truecolor(232, 141, 42)); // #E88D2A
    }
    println!(
        "  {}",
        "AI that fits your machine.".bold().truecolor(232, 141, 42)
    );
    println!();

    // ── Step 1: Profile hardware ────────────────────────────────────
    let sp = indicatif::ProgressBar::new_spinner();
    sp.set_style(indicatif::ProgressStyle::with_template("{spinner:.green} {msg}").unwrap());
    sp.set_message(format!(
        "{} Sniffing your hardware...",
        "Step 1/4".bold().cyan()
    ));
    sp.enable_steady_tick(std::time::Duration::from_millis(80));
    let hw = hivebear_core::profile();
    sp.finish_and_clear();
    println!(
        "  {}  {} ({} cores)",
        "CPU".dimmed(),
        hw.cpu.model_name.bold(),
        hw.cpu.physical_cores
    );
    println!(
        "  {}  {} available",
        "RAM".dimmed(),
        format_bytes(hw.memory.available_bytes)
    );
    if !hw.gpus.is_empty() {
        for gpu in &hw.gpus {
            println!(
                "  {}  {} ({} VRAM)",
                "GPU".dimmed(),
                gpu.name.bold(),
                format_bytes(gpu.vram_bytes)
            );
        }
    } else {
        println!(
            "  {}  {}",
            "GPU".dimmed(),
            "None (CPU-only inference)".yellow()
        );
        println!(
            "  {}",
            "Tip: A GPU with 4+ GB VRAM would significantly speed up inference.".dimmed()
        );
    }

    // Detect Ollama models
    let ollama = hivebear_registry::registry::ollama_lib::OllamaSource::new();
    if ollama.is_available() {
        if let Ok(ollama_models) = ollama.search("", 50).await {
            if !ollama_models.is_empty() {
                println!(
                    "  {}  Found {} model(s) in your Ollama installation.",
                    "🦙".dimmed(),
                    ollama_models.len()
                );
                println!(
                    "  {}",
                    "  Import them with: hivebear import-ollama".dimmed()
                );
            }
        }
    }
    println!();

    // ── Step 2: Get recommendations ─────────────────────────────────
    let sp = indicatif::ProgressBar::new_spinner();
    sp.set_style(indicatif::ProgressStyle::with_template("{spinner:.green} {msg}").unwrap());
    sp.set_message(format!(
        "{} Finding the best model for your device...",
        "Step 2/4".bold().cyan()
    ));
    sp.enable_steady_tick(std::time::Duration::from_millis(80));
    let config = Config::load();
    let recs = hivebear_core::recommender::recommend(&hw, &config);
    sp.finish_and_clear();

    if recs.is_empty() {
        eprintln!(
            "{}",
            "No models found that can run on this hardware at acceptable speed."
                .red()
                .bold()
        );
        eprintln!(
            "{}",
            "Try lowering the minimum tok/s threshold in your config.".dimmed()
        );
        return;
    }

    let top = &recs[0];
    println!(
        "\n  {} {} ({}) via {}",
        "Beary best match!".bold().green(),
        top.model_name.bold(),
        top.quantization,
        top.engine,
    );
    println!(
        "  {} est. {:.1} tok/s, {}",
        "  ".dimmed(),
        top.estimated_tokens_per_sec,
        format_bytes(top.estimated_memory_usage_bytes)
    );
    if !top.warnings.is_empty() {
        for w in &top.warnings {
            println!("  {}", format!("  ⚠ {w}").yellow());
        }
    }
    println!();

    // ── Step 3: Install model if needed ──────────────────────────────
    println!("{} Preparing model...", "Step 3/4".bold().cyan());

    let paths = hivebear_core::config::paths::AppPaths::new();
    let registry = match hivebear_registry::Registry::new(&config, &paths).await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{}: {e}", "Failed to initialize registry".red().bold());
            return;
        }
    };

    // Check if already installed — skip download entirely
    let already_installed = if let Ok(Some(meta)) = registry.get(&top.model_id).await {
        meta.installed.is_some()
    } else {
        false
    };

    if already_installed {
        println!("  {}", "Already installed — skipping download.".green());
    } else {
        println!("  Downloading {}...", top.model_name.bold());

        let pb = Arc::new(Mutex::new(indicatif::ProgressBar::new(0)));
        pb.lock().unwrap().set_style(
            indicatif::ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({bytes_per_sec}, {eta})")
                .unwrap()
                .progress_chars("=>-"),
        );

        let pb_clone = Arc::clone(&pb);
        let progress_cb = move |progress: hivebear_registry::download::DownloadProgress| {
            let bar = pb_clone.lock().unwrap();
            if let Some(total) = progress.total_bytes {
                bar.set_length(total);
            }
            bar.set_position(progress.bytes_downloaded);
        };

        match registry
            .install(&top.model_id, None, None, Some(&progress_cb))
            .await
        {
            Ok(installed) => {
                pb.lock().unwrap().finish_and_clear();
                println!(
                    "  {} {} ({})",
                    "Installed!".green(),
                    installed.filename,
                    format_bytes(installed.size_bytes)
                );
            }
            Err(hivebear_registry::RegistryError::AlreadyInstalled(_)) => {
                pb.lock().unwrap().finish_and_clear();
                println!("  {}", "Already installed.".green());
            }
            Err(e) => {
                pb.lock().unwrap().finish_and_clear();
                eprintln!("{}: {e}", "Installation failed".red().bold());
                return;
            }
        }
    }

    let model_path = registry_commands::resolve_model(&top.model_id, &hw).await;
    println!();

    // ── Step 4: Load and chat ────────────────────────────────────────
    let sp = indicatif::ProgressBar::new_spinner();
    sp.set_style(indicatif::ProgressStyle::with_template("{spinner:.green} {msg}").unwrap());
    sp.set_message(format!("{} Loading model...", "Step 4/4".bold().cyan()));
    sp.enable_steady_tick(std::time::Duration::from_millis(80));

    let orchestrator = create_orchestrator(hw.clone());
    let load_config = LoadConfig {
        context_length,
        offload: hivebear_inference::OffloadConfig {
            auto: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let handle = match orchestrator
        .load(Path::new(&model_path), &load_config)
        .await
    {
        Ok(h) => {
            sp.finish_and_clear();
            println!(
                "{}",
                format!(
                    "  Engine: {} | Context: {} tokens",
                    h.engine, context_length
                )
                .dimmed()
            );
            h
        }
        Err(e) => {
            sp.finish_and_clear();
            eprintln!("{}: {e}", "Failed to load model".red().bold());
            return;
        }
    };

    println!();
    println!("  {}", "Your hive is ready!".bold().green());
    println!("  {}", "Type your message, or /quit to exit.".dimmed());
    println!();

    let mut history: Vec<ChatMessage> = Vec::new();
    let mut first_reply = true;

    loop {
        print!("{} ", "🐻>".bold().truecolor(232, 141, 42));
        std::io::stdout().flush().ok();

        let mut input = String::new();
        if std::io::stdin().read_line(&mut input).is_err() {
            break;
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input == "/quit" || input == "/exit" || input == "/q" {
            println!("Goodbye!");
            break;
        }

        history.push(ChatMessage::user_text(input));

        let req = GenerateRequest {
            messages: history.clone(),
            sampling: SamplingParams {
                temperature,
                ..Default::default()
            },
            model_name: Some(top.model_id.clone()),
            ..Default::default()
        };

        let mut response_text = String::new();
        match orchestrator.stream(&handle, &req) {
            Ok(mut stream) => {
                while let Some(result) = stream.next().await {
                    match result {
                        Ok(token) => {
                            print!("{}", token.text);
                            std::io::stdout().flush().ok();
                            response_text.push_str(&token.text);
                        }
                        Err(e) => {
                            eprintln!("\n{}: {e}", "Generation error".red());
                            break;
                        }
                    }
                }
                println!("\n");
            }
            Err(e) => {
                eprintln!("{}: {e}", "Stream error".red());
                continue;
            }
        }

        history.push(ChatMessage::Assistant(response_text));

        if first_reply {
            first_reply = false;
            println!(
                "  {}",
                "Tip: Run `hivebear contribute` to share your idle compute with the hive.".dimmed()
            );
            println!();
        }
    }

    if let Err(e) = orchestrator.unload(&handle).await {
        tracing::warn!("Failed to unload model: {e}");
    }
}

async fn cmd_mesh(action: MeshAction) {
    use std::sync::Arc;

    println!("\n{}", "  HiveBear P2P Mesh  ".bold().white().on_magenta());
    println!();

    let config = hivebear_core::Config::load();
    let hw = hivebear_core::profile();

    match action {
        MeshAction::Start { port } => {
            let tier = hivebear_mesh::MeshTier::from_str_lossy(&config.mesh.tier);

            // Load or generate persistent identity
            let paths = hivebear_core::config::paths::AppPaths::new();
            let identity_path = paths.data_dir.join("node_identity.key");
            let identity = match hivebear_mesh::NodeIdentity::load_or_generate(&identity_path) {
                Ok(id) => id,
                Err(e) => {
                    eprintln!("{}: {e}", "Failed to load node identity".red().bold());
                    return;
                }
            };

            let security_mode = hivebear_mesh::MeshSecurityMode::default();
            let transport = Arc::new(hivebear_mesh::transport::quic::QuicTransport::new(
                identity.node_id.clone(),
                security_mode,
                None,
            ));
            let discovery = Arc::new(
                hivebear_mesh::discovery::server::CoordinationServerClient::new(
                    config.mesh.coordination_server.clone(),
                ),
            );

            let reputation_path = Some(paths.data_dir.join("reputation.json"));
            let node = Arc::new(hivebear_mesh::MeshNode::with_identity(
                identity,
                transport.clone(),
                discovery,
                tier,
                reputation_path,
            ));

            let listen_addr: std::net::SocketAddr = format!("0.0.0.0:{port}").parse().unwrap();

            let local_info = hivebear_mesh::PeerInfo {
                node_id: node.local_id.clone(),
                hardware: hw.clone(),
                available_memory_bytes: hw.memory.available_bytes,
                available_vram_bytes: hw.gpus.iter().map(|g| g.vram_bytes).sum(),
                network_bandwidth_mbps: 100.0, // Default estimate
                latency_ms: None,
                tier,
                reputation_score: 1.0,
                addr: listen_addr,
                external_addr: None,
                nat_type: hivebear_mesh::nat::NatType::Unknown,
                latency_map: std::collections::HashMap::new(),
                serving_model_id: None,
                swarm_id: None,
                draft_capability: None,
            };

            match node.start(listen_addr, local_info).await {
                Ok(()) => {
                    // Start background maintenance (heartbeat, discovery, health checks)
                    node.start_maintenance();

                    println!(
                        "{}",
                        format!(
                            "Mesh node started. Peer ID: {} | Listening on port {}",
                            node.local_id, port
                        )
                        .green()
                    );
                    println!(
                        "{}",
                        format!(
                            "Tier: {} | Hardware: {} RAM",
                            tier,
                            hivebear_core::types::format_bytes(hw.memory.total_bytes)
                        )
                        .dimmed()
                    );
                    println!(
                        "{}",
                        format!("Coordination server: {}", config.mesh.coordination_server)
                            .dimmed()
                    );
                    println!();
                    println!("Press Ctrl+C to stop.");

                    // Keep running until interrupted
                    tokio::signal::ctrl_c().await.ok();
                    println!("\nShutting down...");
                    if let Err(e) = node.stop().await {
                        eprintln!("{}: {e}", "Failed to stop cleanly".red());
                    }
                    println!("{}", "Mesh node stopped.".green());
                }
                Err(e) => {
                    eprintln!("{}: {e}", "Failed to start mesh node".red().bold());
                }
            }
        }
        MeshAction::Status => {
            // Show mesh configuration
            println!("{}", "Mesh Configuration".bold().cyan());
            println!(
                "  Enabled:      {}",
                if config.mesh.enabled {
                    "yes".green()
                } else {
                    "no".yellow()
                }
            );
            println!("  Port:         {}", config.mesh.port);
            println!("  Tier:         {}", config.mesh.tier);
            println!(
                "  Contribution: {:.0}% of resources",
                config.mesh.max_contribution_percent * 100.0
            );
            println!(
                "  Verification: {:.0}% of tokens",
                config.mesh.verification_rate * 100.0
            );
            println!();

            // Show local hardware summary
            println!("{}", "Local Hardware".bold().cyan());
            println!(
                "  CPU:    {} ({} cores)",
                hw.cpu.model_name, hw.cpu.physical_cores
            );
            println!(
                "  RAM:    {} available",
                hivebear_core::types::format_bytes(hw.memory.available_bytes)
            );
            if !hw.gpus.is_empty() {
                for gpu in &hw.gpus {
                    println!(
                        "  GPU:    {} ({} VRAM)",
                        gpu.name,
                        hivebear_core::types::format_bytes(gpu.vram_bytes)
                    );
                }
            } else {
                println!("  GPU:    {}", "None detected".yellow());
            }
            println!();

            println!(
                "{}",
                "Tip: Run 'hivebear mesh start' to join the mesh network.".dimmed()
            );
        }
        MeshAction::Run {
            model,
            prompt,
            context_length,
            temperature,
        } => {
            cmd_mesh_run(model, prompt, context_length, temperature, hw).await;
        }
        MeshAction::Stop => {
            println!("Sending stop signal to running mesh node...");
            println!();
            println!(
                "{}",
                "Note: The mesh node must be running in another terminal.\n\
                 Use Ctrl+C in that terminal, or implement the daemon + signal \n\
                 protocol for remote stop."
                    .dimmed()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Mesh inference handler: bridges mesh protocol to local Orchestrator
// ---------------------------------------------------------------------------

/// Implementation of `MeshInferenceHandler` that uses the local
/// `Orchestrator` to run inference. This is what a worker node uses
/// to actually generate tokens when a peer sends an inference request.
struct CliInferenceHandler {
    orchestrator: std::sync::Arc<Orchestrator>,
    handle: std::sync::Arc<hivebear_inference::ModelHandle>,
}

#[cfg_attr(not(target_arch = "wasm32"), async_trait::async_trait)]
#[cfg_attr(target_arch = "wasm32", async_trait::async_trait(?Send))]
impl hivebear_mesh::MeshInferenceHandler for CliInferenceHandler {
    async fn handle_inference(
        &self,
        model_id: &str,
        messages_json: &str,
        max_tokens: u32,
        temperature: f32,
    ) -> std::result::Result<
        std::pin::Pin<Box<dyn futures::Stream<Item = std::result::Result<String, String>> + Send>>,
        String,
    > {
        let messages: Vec<ChatMessage> =
            serde_json::from_str(messages_json).map_err(|e| format!("Bad messages JSON: {e}"))?;

        let req = GenerateRequest {
            messages,
            max_tokens,
            sampling: SamplingParams {
                temperature,
                ..Default::default()
            },
            model_name: Some(model_id.to_string()),
            ..Default::default()
        };

        let stream = self
            .orchestrator
            .stream(&self.handle, &req)
            .map_err(|e| format!("Stream error: {e}"))?;

        // Map Token stream -> String stream
        let string_stream = stream.map(|result| match result {
            Ok(token) => Ok(token.text),
            Err(e) => Err(format!("Generation error: {e}")),
        });

        Ok(Box::pin(string_stream))
    }
}

// ---------------------------------------------------------------------------
// `hivebear mesh run` implementation
// ---------------------------------------------------------------------------

async fn cmd_mesh_run(
    model: String,
    prompt: Option<String>,
    context_length: u32,
    temperature: f32,
    hw: hivebear_core::HardwareProfile,
) {
    use std::io::Write;
    use std::path::Path;
    use std::sync::Arc;

    // Resolve the model (file path or registry ID)
    let model_path_str = registry_commands::resolve_model(&model, &hw).await;
    let model_path = Path::new(&model_path_str);

    if !model_path.exists() {
        eprintln!(
            "{}: Model not found at '{}'",
            "Error".red().bold(),
            model_path_str
        );
        eprintln!(
            "{}",
            "Install a model first with 'hivebear install <model>'".dimmed()
        );
        return;
    }

    // ── Attempt peer discovery ──────────────────────────────────────

    let config = hivebear_core::Config::load();
    let discovery = Arc::new(
        hivebear_mesh::discovery::server::CoordinationServerClient::new(
            config.mesh.coordination_server.clone(),
        ),
    );
    let peers = discovery.find_peers("", 0).await.unwrap_or_default();

    if !peers.is_empty() {
        println!(
            "{} {} peer(s) available — routing inference to best peer",
            "Mesh:".bold().magenta(),
            peers.len()
        );

        // Pick the best peer (first available with the model)
        let best_peer = &peers[0];
        let peer_addr = best_peer.addr;

        // Create identity and QUIC transport for the client side
        let paths = hivebear_core::config::paths::AppPaths::new();
        let identity_path = paths.data_dir.join("node_identity.key");
        let identity = hivebear_mesh::NodeIdentity::load_or_generate(&identity_path)
            .unwrap_or_else(|_| hivebear_mesh::NodeIdentity::generate());
        let security_mode = hivebear_mesh::MeshSecurityMode::default();
        let transport: Arc<dyn hivebear_mesh::transport::MeshTransport> =
            Arc::new(hivebear_mesh::transport::quic::QuicTransport::new(
                identity.node_id.clone(),
                security_mode,
                None,
            ));

        // Connect to the peer
        match transport.connect(peer_addr).await {
            Ok(peer_id) => {
                println!(
                    "{} Connected to peer {} at {}",
                    "Mesh:".bold().magenta(),
                    &peer_id.to_hex()[..12],
                    peer_addr
                );

                // Build messages JSON
                let messages = if let Some(ref prompt_text) = prompt {
                    vec![ChatMessage::user_text(prompt_text)]
                } else {
                    println!(
                        "{}",
                        "Mesh interactive mode: type your message, or /quit to exit.".green()
                    );
                    println!();

                    // Interactive loop over the mesh
                    let mut history: Vec<ChatMessage> = Vec::new();
                    loop {
                        print!("{} ", "mesh>".bold().magenta());
                        std::io::stdout().flush().ok();

                        let mut input = String::new();
                        if std::io::stdin().read_line(&mut input).is_err() {
                            break;
                        }
                        let input = input.trim();
                        if input.is_empty() {
                            continue;
                        }
                        if input == "/quit" || input == "/exit" || input == "/q" {
                            println!("Goodbye!");
                            break;
                        }

                        history.push(ChatMessage::user_text(input));

                        let session_id = uuid::Uuid::new_v4();
                        let messages_json = serde_json::to_string(&history).unwrap_or_default();
                        let req_msg =
                            hivebear_mesh::transport::protocol::MeshMessage::InferenceRequest {
                                session_id,
                                model_id: model.clone(),
                                messages_json,
                                max_tokens: 2048,
                                temperature,
                                top_p: 0.9,
                            };

                        if let Err(e) = transport.send(&peer_id, req_msg).await {
                            eprintln!("{}: {e}", "Failed to send request".red());
                            continue;
                        }

                        // Stream tokens back
                        let mut response_text = String::new();
                        loop {
                            match transport.recv().await {
                                Ok((_from, hivebear_mesh::transport::protocol::MeshMessage::InferenceToken {
                                    text, is_done, ..
                                })) => {
                                    if is_done { break; }
                                    print!("{}", text);
                                    std::io::stdout().flush().ok();
                                    response_text.push_str(&text);
                                }
                                Ok((_from, hivebear_mesh::transport::protocol::MeshMessage::InferenceComplete {
                                    error, ..
                                })) => {
                                    if let Some(err) = error {
                                        eprintln!("\n{}: {err}", "Remote error".red());
                                    }
                                    break;
                                }
                                Ok((_from, hivebear_mesh::transport::protocol::MeshMessage::Error {
                                    message, ..
                                })) => {
                                    eprintln!("\n{}: {message}", "Peer error".red());
                                    break;
                                }
                                Ok(_) => continue, // Ignore other messages
                                Err(e) => {
                                    eprintln!("\n{}: {e}", "Transport error".red());
                                    break;
                                }
                            }
                        }
                        println!();
                        println!();

                        if !response_text.is_empty() {
                            history.push(ChatMessage::Assistant(response_text));
                        }
                    }
                    return;
                };

                // Single prompt mode over mesh
                let session_id = uuid::Uuid::new_v4();
                let messages_json = serde_json::to_string(&messages).unwrap_or_default();
                let req_msg = hivebear_mesh::transport::protocol::MeshMessage::InferenceRequest {
                    session_id,
                    model_id: model.clone(),
                    messages_json,
                    max_tokens: 2048,
                    temperature,
                    top_p: 0.9,
                };

                if let Err(e) = transport.send(&peer_id, req_msg).await {
                    eprintln!("{}: {e}", "Failed to send request".red());
                } else {
                    // Stream tokens
                    loop {
                        match transport.recv().await {
                            Ok((_from, hivebear_mesh::transport::protocol::MeshMessage::InferenceToken {
                                text, is_done, ..
                            })) => {
                                if is_done { break; }
                                print!("{}", text);
                                std::io::stdout().flush().ok();
                            }
                            Ok((_from, hivebear_mesh::transport::protocol::MeshMessage::InferenceComplete {
                                error, ..
                            })) => {
                                if let Some(err) = error {
                                    eprintln!("\n{}: {err}", "Remote error".red());
                                }
                                break;
                            }
                            Ok(_) => continue,
                            Err(e) => {
                                eprintln!("\n{}: {e}", "Transport error".red());
                                break;
                            }
                        }
                    }
                    println!();
                }

                return;
            }
            Err(e) => {
                println!(
                    "{} Failed to connect to peer: {e}. Falling back to local inference.",
                    "Mesh:".yellow()
                );
            }
        }
    }

    // ── Fall back to local inference ────────────────────────────────

    println!("{}", "Running locally (no mesh peers available).".yellow());
    println!();

    let orchestrator = create_orchestrator(hw.clone());

    let load_config = LoadConfig {
        context_length,
        offload: hivebear_inference::OffloadConfig {
            auto: true,
            ..Default::default()
        },
        ..Default::default()
    };

    println!("Loading model: {}", model.bold());
    let handle = match orchestrator.load(model_path, &load_config).await {
        Ok(h) => {
            println!(
                "{}",
                format!(
                    "  Engine: {} | Context: {} tokens",
                    h.engine, context_length
                )
                .dimmed()
            );
            println!();
            h
        }
        Err(e) => {
            eprintln!("{}: {e}", "Failed to load model".red().bold());
            return;
        }
    };

    // Build the inference handler (same one the daemon would use)
    let _handler = Arc::new(CliInferenceHandler {
        orchestrator: Arc::new(create_orchestrator(hw)),
        handle: Arc::new(handle.clone()),
    });

    if let Some(prompt_text) = prompt {
        // Single prompt mode
        let req = GenerateRequest {
            messages: vec![ChatMessage::user_text(&prompt_text)],
            sampling: SamplingParams {
                temperature,
                ..Default::default()
            },
            model_name: Some(model.clone()),
            ..Default::default()
        };

        match orchestrator.stream(&handle, &req) {
            Ok(mut stream) => {
                while let Some(result) = stream.next().await {
                    match result {
                        Ok(token) => {
                            print!("{}", token.text);
                            std::io::stdout().flush().ok();
                        }
                        Err(e) => {
                            eprintln!("\n{}: {e}", "Generation error".red());
                            break;
                        }
                    }
                }
                println!();
            }
            Err(e) => {
                eprintln!("{}: {e}", "Stream error".red());
            }
        }
    } else {
        // Interactive chat mode
        println!("{}", "Ready. Type your message, or /quit to exit.".green());
        println!(
            "{}",
            "(Running in mesh local-fallback mode — connect peers with 'hivebear mesh start')"
                .dimmed()
        );
        println!();

        let mut history: Vec<ChatMessage> = Vec::new();

        loop {
            print!("{} ", "mesh>".bold().magenta());
            std::io::stdout().flush().ok();

            let mut input = String::new();
            if std::io::stdin().read_line(&mut input).is_err() {
                break;
            }

            let input = input.trim();
            if input.is_empty() {
                continue;
            }
            if input == "/quit" || input == "/exit" || input == "/q" {
                println!("Goodbye!");
                break;
            }

            history.push(ChatMessage::user_text(input));

            let req = GenerateRequest {
                messages: history.clone(),
                sampling: SamplingParams {
                    temperature,
                    ..Default::default()
                },
                model_name: Some(model.clone()),
                ..Default::default()
            };

            let mut response_text = String::new();
            match orchestrator.stream(&handle, &req) {
                Ok(mut stream) => {
                    while let Some(result) = stream.next().await {
                        match result {
                            Ok(token) => {
                                print!("{}", token.text);
                                std::io::stdout().flush().ok();
                                response_text.push_str(&token.text);
                            }
                            Err(e) => {
                                eprintln!("\n{}: {e}", "Generation error".red());
                                break;
                            }
                        }
                    }
                    println!();
                    println!();

                    if !response_text.is_empty() {
                        history.push(ChatMessage::Assistant(response_text));
                    }
                }
                Err(e) => {
                    eprintln!("{}: {e}", "Stream error".red());
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Contribute command: one-click swarm join + contribution dashboard
// ---------------------------------------------------------------------------

async fn cmd_contribute(port: u16, model_override: Option<String>, coordinator_url: String) {
    use hivebear_core::contribution::{determine_tier, plan_contribution};

    println!("\n{}", "  HiveBear Contributor  ".bold().white().on_green());
    println!();

    // Step 1: Profile hardware
    println!("{}", "Step 1/4: Profiling hardware...".cyan());
    let hw = hivebear_core::profile();
    let tier = determine_tier(&hw);
    let plan = plan_contribution(&hw);
    let total_vram: u64 = hw.gpus.iter().map(|g| g.vram_bytes).sum();

    println!(
        "  {} {} | {} RAM | {} VRAM",
        "Hardware:".bold(),
        hw.cpu.model_name,
        format_bytes(hw.memory.total_bytes),
        if total_vram > 0 {
            format_bytes(total_vram)
        } else {
            "none".to_string()
        }
    );
    println!("  {} {}", "Tier:".bold(), tier.description());
    println!(
        "  {} ~{:.1} TFLOPS",
        "Estimated compute:".bold(),
        plan.estimated_tflops
    );
    println!();

    // Step 2: Determine model
    let model_id = model_override.unwrap_or_else(|| plan.recommended_model.clone());
    println!("{}", "Step 2/4: Selecting model...".cyan());
    println!(
        "  {} {} ({})",
        "Model:".bold(),
        plan.recommended_model_name,
        model_id
    );
    println!(
        "  {} {} | {} layers serviceable locally",
        "Size:".bold(),
        format_bytes(plan.model_size_bytes),
        plan.max_layers_serviceable
    );
    println!();

    // Step 3: Connect to coordinator and matchmake
    println!("{}", "Step 3/4: Connecting to network...".cyan());
    let coordinator =
        hivebear_mesh::discovery::server::CoordinationServerClient::new(coordinator_url.clone());

    let identity_path = hivebear_core::AppPaths::new()
        .data_dir
        .join("node_identity.key");
    let identity = hivebear_mesh::NodeIdentity::load_or_generate(&identity_path)
        .unwrap_or_else(|_| hivebear_mesh::NodeIdentity::generate());
    let node_id_hex = identity.node_id.to_hex();
    let listen_addr: std::net::SocketAddr = format!("0.0.0.0:{port}").parse().unwrap();
    let local_info = hivebear_mesh::PeerInfo {
        node_id: identity.node_id.clone(),
        hardware: hw.clone(),
        available_memory_bytes: hw.memory.available_bytes,
        available_vram_bytes: total_vram,
        network_bandwidth_mbps: 100.0,
        latency_ms: None,
        tier: hivebear_mesh::MeshTier::Free,
        reputation_score: 1.0,
        addr: listen_addr,
        external_addr: None,
        nat_type: hivebear_mesh::NatType::Unknown,
        latency_map: std::collections::HashMap::new(),
        serving_model_id: Some(model_id.clone()),
        swarm_id: None,
        draft_capability: None,
    };

    if let Err(e) = coordinator.register(&local_info).await {
        eprintln!(
            "{}: {e}",
            "Warning: Could not register with coordinator".yellow()
        );
        eprintln!("{}", "Running in standalone mode.".dimmed());
    }

    // Try matchmaking
    let matchmake_result = coordinator
        .matchmake(
            &node_id_hex,
            total_vram as i64,
            hw.memory.available_bytes as i64,
            Some(&model_id),
        )
        .await;

    match &matchmake_result {
        Ok(result) => {
            let swarm_id = result
                .get("swarm_id")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let role = result
                .get("role")
                .and_then(|v| v.as_str())
                .unwrap_or("worker");
            let existing = result
                .get("existing_members")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            let layers_from = result
                .get("layers_from")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);
            let layers_to = result
                .get("layers_to")
                .and_then(|v| v.as_i64())
                .unwrap_or(0);

            println!(
                "  {} {} ({})",
                "Swarm:".bold(),
                &swarm_id[..8.min(swarm_id.len())],
                role
            );
            println!("  {} {} existing members", "Network:".bold(), existing);
            println!(
                "  {} layers {}..{}",
                "Assignment:".bold(),
                layers_from,
                layers_to
            );

            // Join the swarm
            if let Err(e) = coordinator
                .join_swarm(
                    swarm_id,
                    &node_id_hex,
                    role,
                    Some(layers_from),
                    Some(layers_to),
                )
                .await
            {
                eprintln!("{}: {e}", "Warning: Could not join swarm".yellow());
            }
        }
        Err(e) => {
            eprintln!("{}: {e}", "Warning: Matchmaking failed".yellow());
            eprintln!("{}", "Will retry when peers are available.".dimmed());
        }
    }

    println!();

    // Step 4: Load model and start serving
    println!(
        "{}",
        "Step 4/4: Loading model and starting inference daemon...".cyan()
    );

    // Resolve and load the model locally
    let model_path_str = registry_commands::resolve_model(&model_id, &hw).await;
    let model_path = std::path::Path::new(&model_path_str);

    if !model_path.exists() {
        eprintln!(
            "{}: Model not found at '{}'. Install it first with 'hivebear install {}'",
            "Error".red().bold(),
            model_path_str,
            model_id,
        );
        return;
    }

    let orchestrator = std::sync::Arc::new(create_orchestrator(hw.clone()));
    let load_config = LoadConfig {
        context_length: 4096,
        offload: hivebear_inference::OffloadConfig {
            auto: true,
            ..Default::default()
        },
        ..Default::default()
    };

    let handle = match orchestrator.load(model_path, &load_config).await {
        Ok(h) => std::sync::Arc::new(h),
        Err(e) => {
            eprintln!("{}: {e}", "Failed to load model".red().bold());
            return;
        }
    };

    // Create inference handler
    let handler = std::sync::Arc::new(CliInferenceHandler {
        orchestrator: orchestrator.clone(),
        handle: handle.clone(),
    });

    // Create QUIC transport and start listening
    let security_mode = hivebear_mesh::MeshSecurityMode::default();
    let transport: std::sync::Arc<dyn hivebear_mesh::transport::MeshTransport> =
        std::sync::Arc::new(hivebear_mesh::transport::quic::QuicTransport::new(
            identity.node_id.clone(),
            security_mode,
            None,
        ));

    if let Err(e) = transport.listen(listen_addr).await {
        eprintln!("{}: {e}", "Failed to start QUIC listener".red().bold());
        return;
    }

    // Create pipeline handler for distributed layer serving
    let pipeline_h = std::sync::Arc::new(pipeline_handler::CliPipelineHandler::new(
        orchestrator.clone(),
    ));

    // Start the worker daemon with pipeline support
    let daemon = hivebear_mesh::pipeline::daemon::MeshWorkerDaemon::with_pipeline(
        handler,
        transport.clone(),
        pipeline_h,
    );
    tokio::spawn(async move {
        daemon.run().await;
    });

    println!(
        "  {} Listening for inference requests on port {}",
        "Serving:".bold(),
        port
    );
    println!();

    // Print the live dashboard
    println!("{}", "━".repeat(52).dimmed());
    println!("{}", "  HiveBear Contributor Dashboard".bold().green());
    println!("{}", "━".repeat(52).dimmed());
    println!();
    println!("  {} {}", "Node:".bold(), &node_id_hex[..16]);
    println!("  {} {} (active)", "Status:".bold(), "Contributing".green());
    println!(
        "  {} {} — {}",
        "Model:".bold(),
        plan.recommended_model_name,
        model_id
    );
    println!(
        "  {} {:.1} TFLOPS",
        "Contributing:".bold(),
        plan.estimated_tflops
    );
    println!();
    println!(
        "  {}",
        "You're contributing — you can use any model on the network.".green()
    );
    println!();
    println!("  {}", "Press Ctrl+C to stop contributing.".dimmed());
    println!("{}", "━".repeat(52).dimmed());

    // Keep running until Ctrl+C
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to listen for ctrl+c");

    println!();
    println!("{}", "Stopping contribution...".yellow());

    // Leave swarm and deregister
    if let Ok(result) = &matchmake_result {
        if let Some(swarm_id) = result.get("swarm_id").and_then(|v| v.as_str()) {
            let _ = coordinator.leave_swarm(swarm_id, &node_id_hex).await;
        }
    }
    let _ = coordinator.deregister().await;
    println!("{}", "Contribution stopped. Thank you!".green());
}

async fn cmd_update(check_only: bool) {
    use self_update::backends::github::Update;

    let current_version = env!("CARGO_PKG_VERSION");

    println!("\n{}", "  HiveBear Update  ".bold().white().on_blue());
    println!();
    println!("Current version: {}", current_version.bold());
    println!("Checking for updates...");

    let target = self_update::get_target();
    let ext = if cfg!(target_os = "windows") {
        "zip"
    } else {
        "tar.gz"
    };
    let asset_name = format!("hivebear-{}.{}", target, ext);

    let updater = match Update::configure()
        .repo_owner("BeckhamLabsLLC")
        .repo_name("HiveBear")
        .bin_name("hivebear")
        .target(&asset_name)
        .current_version(current_version)
        .build()
    {
        Ok(u) => u,
        Err(e) => {
            eprintln!(
                "{} Failed to check for updates: {}",
                "Error:".red().bold(),
                e
            );
            std::process::exit(1);
        }
    };

    let latest = match updater.get_latest_release() {
        Ok(r) => r,
        Err(e) => {
            eprintln!(
                "{} Failed to fetch latest release: {}",
                "Error:".red().bold(),
                e
            );
            std::process::exit(1);
        }
    };

    let latest_version = latest.version.trim_start_matches('v');

    if latest_version == current_version {
        println!(
            "\n{} You are already on the latest version ({}).",
            "✓".green().bold(),
            current_version
        );
        return;
    }

    println!(
        "\n{} New version available: {} → {}",
        "→".cyan().bold(),
        current_version,
        latest_version.green().bold()
    );

    if check_only {
        println!("\nRun {} to install the update.", "hivebear update".bold());
        return;
    }

    println!("Downloading and installing {}...", latest_version);
    match updater.update() {
        Ok(status) => {
            println!(
                "\n{} Updated to version {}!",
                "✓".green().bold(),
                status.version()
            );
        }
        Err(e) => {
            eprintln!("{} Update failed: {}", "Error:".red().bold(), e);
            eprintln!("\nYou can manually download the latest version from:");
            eprintln!("  https://github.com/BeckhamLabsLLC/HiveBear/releases/latest");
            std::process::exit(1);
        }
    }
}

async fn cmd_uninstall(purge: bool, yes: bool) {
    println!("\n{}", "  HiveBear Uninstall  ".bold().white().on_red());
    println!();

    let paths = hivebear_core::config::paths::AppPaths::new();
    let binary_path = std::env::current_exe().unwrap_or_default();

    println!("This will remove:");
    println!("  Binary:  {}", binary_path.display());
    if purge {
        println!("  Config:  {}", paths.config_dir.display());
        println!(
            "  Data:    {} (including downloaded models)",
            paths.data_dir.display()
        );
    }

    if !yes {
        println!();
        print!("Are you sure? [y/N] ");
        use std::io::Write;
        std::io::stdout().flush().unwrap();
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        if !input.trim().eq_ignore_ascii_case("y") {
            println!("Uninstall cancelled.");
            return;
        }
    }

    if purge {
        // Remove config directory
        if paths.config_dir.exists() {
            match std::fs::remove_dir_all(&paths.config_dir) {
                Ok(_) => println!(
                    "{} Removed config: {}",
                    "✓".green(),
                    paths.config_dir.display()
                ),
                Err(e) => eprintln!("{} Failed to remove config: {}", "✗".red(), e),
            }
        }

        // Remove data directory (models, etc.)
        if paths.data_dir.exists() {
            match std::fs::remove_dir_all(&paths.data_dir) {
                Ok(_) => println!("{} Removed data: {}", "✓".green(), paths.data_dir.display()),
                Err(e) => eprintln!("{} Failed to remove data: {}", "✗".red(), e),
            }
        }
    }

    // Remove the binary itself (on Unix, a running binary can delete itself)
    #[cfg(unix)]
    {
        match std::fs::remove_file(&binary_path) {
            Ok(_) => println!("{} Removed binary: {}", "✓".green(), binary_path.display()),
            Err(e) => {
                eprintln!("{} Failed to remove binary: {}", "✗".red(), e);
                eprintln!("  You can manually delete it: rm {}", binary_path.display());
            }
        }
    }

    // On Windows, rename the binary so it's cleaned up on next reboot
    #[cfg(windows)]
    {
        let trash_path = binary_path.with_extension("old");
        match std::fs::rename(&binary_path, &trash_path) {
            Ok(_) => {
                println!(
                    "{} Marked binary for removal: {}",
                    "✓".green(),
                    binary_path.display()
                );
                println!("  The file will be fully removed on next reboot.");
            }
            Err(e) => {
                eprintln!("{} Failed to remove binary: {}", "✗".red(), e);
                eprintln!("  You can manually delete it after closing this terminal.");
            }
        }
    }

    println!();
    println!("{}", "HiveBear has been uninstalled.".bold());
    if !purge {
        println!(
            "Your config and models are still at {}",
            paths.data_dir.display()
        );
        println!("Run with {} to remove everything.", "--purge".bold());
    }
}
