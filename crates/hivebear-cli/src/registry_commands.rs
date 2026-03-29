use colored::Colorize;
use hivebear_core::config::paths::AppPaths;
use hivebear_core::types::format_bytes;
use hivebear_core::{Config, HardwareProfile};
use hivebear_registry::{Registry, SearchResult};
use indicatif::{ProgressBar, ProgressStyle};
use std::sync::{Arc, Mutex};

async fn make_registry() -> Registry {
    let config = Config::load();
    let paths = AppPaths::new();
    match Registry::new(&config, &paths).await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{}: {e}", "Failed to initialize registry".red().bold());
            std::process::exit(1);
        }
    }
}

pub async fn cmd_search(query: String, limit: usize, json: bool) {
    let registry = make_registry().await;
    let hw = hivebear_core::profile();

    println!(
        "\n{}",
        "  HiveBear Model Search  ".bold().white().on_magenta()
    );
    println!();
    println!("Searching for: {}", query.bold());
    println!();

    let results = match registry.search(&query, limit, Some(&hw)).await {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{}: {e}", "Search failed".red().bold());
            return;
        }
    };

    if json {
        println!("{}", serde_json::to_string_pretty(&results).unwrap());
        return;
    }

    if results.is_empty() {
        println!("{}", "No models found matching your query.".yellow());
        println!(
            "{}",
            "Try a broader search term or check your network connection.".dimmed()
        );
        return;
    }

    print_search_results(&results);

    println!();
    println!(
        "{}",
        "Tip: Use 'hivebear install <model>' to download a model.".dimmed()
    );
}

fn print_search_results(results: &[SearchResult]) {
    println!(
        "  {:<4} {:<35} {:<8} {:<10} {:<12} {:<10} {}",
        "#".bold(),
        "Model".bold(),
        "Params".bold(),
        "Compat.".bold(),
        "Downloads".bold(),
        "Status".bold(),
        "Source".bold(),
    );
    println!("  {}", "-".repeat(95));

    for (i, result) in results.iter().enumerate() {
        let rank = format!("{}", i + 1);
        let params = if result.metadata.params_billions > 0.0 {
            format!("{:.1}B", result.metadata.params_billions)
        } else {
            "?".into()
        };

        let compat = match result.compatibility_score {
            Some(s) if s >= 0.7 => format!("{:.0}%", s * 100.0).green().to_string(),
            Some(s) if s >= 0.3 => format!("{:.0}%", s * 100.0).yellow().to_string(),
            Some(s) => format!("{:.0}%", s * 100.0).red().to_string(),
            None => "-".into(),
        };

        let downloads = match result.metadata.downloads_count {
            Some(d) if d >= 1_000_000 => format!("{:.1}M", d as f64 / 1_000_000.0),
            Some(d) if d >= 1_000 => format!("{:.1}K", d as f64 / 1_000.0),
            Some(d) => format!("{d}"),
            None => "-".into(),
        };

        let status = if result.is_installed {
            "installed".green().to_string()
        } else {
            "available".dimmed().to_string()
        };

        let source = result.metadata.huggingface_id.as_deref().unwrap_or("-");

        println!(
            "  {:<4} {:<35} {:<8} {:<10} {:<12} {:<10} {}",
            rank,
            &result.metadata.name[..result.metadata.name.len().min(34)],
            params,
            compat,
            downloads,
            status,
            source.dimmed(),
        );
    }
}

pub async fn cmd_install(model: String, quant: Option<String>, file: Option<String>) {
    let registry = make_registry().await;

    println!(
        "\n{}",
        "  HiveBear Model Install  ".bold().white().on_green()
    );
    println!();

    // Show available files if no preference given
    if quant.is_none() && file.is_none() {
        match registry.list_files(&model).await {
            Ok(files) if !files.is_empty() => {
                println!("Available files for {}:", model.bold());
                for f in &files {
                    let quant_str = f
                        .quantization
                        .map(|q| q.to_string())
                        .unwrap_or_else(|| "?".into());
                    let size = if f.size_bytes > 0 {
                        format_bytes(f.size_bytes)
                    } else {
                        "?".into()
                    };
                    println!("  {} ({}, {})", f.filename, quant_str.cyan(), size);
                }
                println!();
                println!(
                    "{}",
                    "Selecting best quantization (Q4_K_M preferred)...".dimmed()
                );
                println!();
            }
            _ => {}
        }
    }

    // Set up progress bar
    let pb = Arc::new(Mutex::new(ProgressBar::new(0)));
    pb.lock().unwrap().set_style(
        ProgressStyle::default_bar()
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

    println!("Installing {}...", model.bold());

    match registry
        .install(
            &model,
            quant.as_deref(),
            file.as_deref(),
            Some(&progress_cb),
        )
        .await
    {
        Ok(installed) => {
            pb.lock().unwrap().finish_and_clear();
            println!();
            println!("{}", "Model installed successfully!".green().bold());
            println!("  File:     {}", installed.filename);
            println!("  Path:     {}", installed.path.display());
            println!("  Size:     {}", format_bytes(installed.size_bytes));
            if let Some(q) = installed.quantization {
                println!("  Quant:    {q}");
            }
            println!();
            println!("{}", format!("Run it with: hivebear run {model}").dimmed());
        }
        Err(hivebear_registry::RegistryError::AlreadyInstalled(id)) => {
            println!(
                "{}",
                format!("Model '{id}' is already installed. Use 'hivebear remove {id}' first.")
                    .yellow()
            );
        }
        Err(e) => {
            pb.lock().unwrap().finish_and_clear();
            eprintln!("{}: {e}", "Installation failed".red().bold());
        }
    }
}

pub async fn cmd_list(json: bool) {
    let registry = make_registry().await;

    let models = registry.list_installed().await;

    if json {
        println!("{}", serde_json::to_string_pretty(&models).unwrap());
        return;
    }

    println!(
        "\n{}",
        "  HiveBear Installed Models  ".bold().white().on_blue()
    );
    println!();

    if models.is_empty() {
        println!("{}", "No models installed.".yellow());
        println!(
            "{}",
            "Use 'hivebear search <query>' to find models, then 'hivebear install <model>' to download."
                .dimmed()
        );
        return;
    }

    println!(
        "  {:<25} {:<12} {:<10} {:<12} {}",
        "Model".bold(),
        "Format".bold(),
        "Quant".bold(),
        "Size".bold(),
        "Path".bold(),
    );
    println!("  {}", "-".repeat(80));

    for model in &models {
        if let Some(ref installed) = model.installed {
            let quant = installed
                .quantization
                .map(|q| q.to_string())
                .unwrap_or_else(|| "-".into());
            println!(
                "  {:<25} {:<12} {:<10} {:<12} {}",
                &model.name[..model.name.len().min(24)],
                installed.format.to_string(),
                quant,
                format_bytes(installed.size_bytes),
                installed.path.display(),
            );
        }
    }

    println!();
    println!(
        "{}",
        format!("{} model(s) installed", models.len()).dimmed()
    );
}

pub async fn cmd_remove(model: String, yes: bool) {
    let registry = make_registry().await;

    if !yes {
        println!(
            "Remove model '{}'? This will delete the model files from disk.",
            model.bold()
        );
        print!("Continue? [y/N] ");
        std::io::Write::flush(&mut std::io::stdout()).ok();

        let mut input = String::new();
        if std::io::stdin().read_line(&mut input).is_err() {
            return;
        }
        if !input.trim().eq_ignore_ascii_case("y") {
            println!("Cancelled.");
            return;
        }
    }

    match registry.remove(&model).await {
        Ok(freed) => {
            println!(
                "{}",
                format!("Removed '{}'. Freed {}.", model, format_bytes(freed))
                    .green()
                    .bold()
            );
        }
        Err(e) => {
            eprintln!("{}: {e}", "Failed to remove model".red().bold());
        }
    }
}

pub async fn cmd_convert(model: String, to: String, _quant: Option<String>) {
    println!(
        "\n{}",
        "  HiveBear Model Convert  ".bold().white().on_yellow()
    );
    println!();

    // Check for available converters
    let converters = hivebear_registry::conversion::available_converters().await;
    if converters.is_empty() {
        eprintln!(
            "{}",
            "No conversion tools found on this system.".red().bold()
        );
        println!();
        println!("To convert models, install one of:");
        println!("  {} — for GGUF quantization", "llama-quantize".bold());
        println!(
            "  {} — for format conversion",
            "python3 + llama.cpp convert scripts".bold()
        );
        println!();
        println!(
            "{}",
            "See: https://github.com/ggerganov/llama.cpp#prepare-and-quantize".dimmed()
        );
        return;
    }

    println!("Converting {} to {}...", model.bold(), to.bold());
    println!(
        "{}",
        "Full native conversion coming in a future release.".yellow()
    );
}

pub async fn cmd_storage(cleanup: bool) {
    let registry = make_registry().await;
    let manager = registry.storage_manager().await;

    println!("\n{}", "  HiveBear Storage  ".bold().white().on_cyan());
    println!();

    match manager.report().await {
        Ok(report) => {
            println!(
                "Total model storage: {}",
                format_bytes(report.total_bytes).bold()
            );
            println!();

            if !report.models.is_empty() {
                println!("{}", "Installed Models:".bold());
                for m in &report.models {
                    println!("  {:<25} {}", m.model_name, format_bytes(m.size_bytes));
                }
                println!();
            }

            if !report.partial_downloads.is_empty() {
                println!(
                    "{} ({}):",
                    "Partial Downloads".yellow().bold(),
                    report.partial_downloads.len()
                );
                for p in &report.partial_downloads {
                    println!(
                        "  {} — {} downloaded",
                        p.filename,
                        format_bytes(p.bytes_downloaded)
                    );
                }
                println!();
            }

            if !report.orphaned_files.is_empty() {
                println!(
                    "{} ({}):",
                    "Orphaned Directories".yellow().bold(),
                    report.orphaned_files.len()
                );
                for o in &report.orphaned_files {
                    println!("  {}", o.display());
                }
                println!();
            }

            if cleanup {
                println!("{}", "Cleaning up...".dimmed());
                match manager
                    .cleanup_partial(std::time::Duration::from_secs(7 * 24 * 3600))
                    .await
                {
                    Ok(cleaned) if !cleaned.is_empty() => {
                        println!(
                            "{}",
                            format!("Removed {} stale partial download(s).", cleaned.len()).green()
                        );
                    }
                    Ok(_) => {
                        println!("{}", "No stale partial downloads to clean.".dimmed());
                    }
                    Err(e) => {
                        eprintln!("{}: {e}", "Cleanup failed".red());
                    }
                }
            }
        }
        Err(e) => {
            eprintln!("{}: {e}", "Failed to generate storage report".red());
        }
    }
}

pub async fn cmd_import_ollama(model: Option<String>) {
    let registry = make_registry().await;

    println!(
        "\n{}",
        "  HiveBear Ollama Import  ".bold().white().on_magenta()
    );
    println!();

    if !registry.ollama_available() {
        eprintln!(
            "{}",
            "Ollama is not installed or not found on this system."
                .red()
                .bold()
        );
        println!();
        println!(
            "{}",
            "Install Ollama from https://ollama.com, then try again.".dimmed()
        );
        return;
    }

    match model {
        Some(model_id) => {
            // Import a specific model
            println!("Importing {}...", model_id.bold());
            match registry.import_ollama(&model_id).await {
                Ok(installed) => {
                    println!();
                    println!("{}", "Model imported successfully!".green().bold());
                    println!("  Source:   Ollama ({})", model_id);
                    println!("  Path:     {}", installed.path.display());
                    println!("  Size:     {}", format_bytes(installed.size_bytes));
                    if let Some(q) = installed.quantization {
                        println!("  Quant:    {q}");
                    }
                    println!();
                    println!(
                        "{}",
                        format!("Run it with: hivebear run ollama/{model_id}").dimmed()
                    );
                }
                Err(e) => {
                    eprintln!("{}: {e}", "Import failed".red().bold());
                }
            }
        }
        None => {
            // List all available Ollama models
            match registry.list_ollama().await {
                Ok(models) if models.is_empty() => {
                    println!(
                        "{}",
                        "No models found in your Ollama installation.".yellow()
                    );
                    println!("{}", "Pull a model with: ollama pull llama3".dimmed());
                }
                Ok(models) => {
                    println!(
                        "Found {} model(s) in your Ollama installation:\n",
                        models.len().to_string().bold()
                    );
                    println!(
                        "  {:<30} {:<8} {:<10} {:<12}",
                        "Model".bold(),
                        "Params".bold(),
                        "Quant".bold(),
                        "Size".bold(),
                    );
                    println!("  {}", "-".repeat(65));

                    for model in &models {
                        let params = if model.params_billions > 0.0 {
                            format!("{:.1}B", model.params_billions)
                        } else {
                            "?".into()
                        };
                        let quant = model
                            .installed
                            .as_ref()
                            .and_then(|i| i.quantization)
                            .map(|q| q.to_string())
                            .unwrap_or_else(|| "-".into());
                        let size = model
                            .installed
                            .as_ref()
                            .map(|i| format_bytes(i.size_bytes))
                            .unwrap_or_else(|| "?".into());

                        println!(
                            "  {:<30} {:<8} {:<10} {:<12}",
                            &model.name[..model.name.len().min(29)],
                            params,
                            quant,
                            size,
                        );
                    }

                    println!();
                    println!(
                        "{}",
                        "Import a model with: hivebear import-ollama <model>".dimmed()
                    );
                    println!("{}", "Example: hivebear import-ollama llama3:8b".dimmed());
                }
                Err(e) => {
                    eprintln!("{}: {e}", "Failed to scan Ollama models".red().bold());
                }
            }
        }
    }
}

/// Resolve a model ID through the registry, or use the path directly if it exists.
pub async fn resolve_model(model: &str, hw: &HardwareProfile) -> String {
    // If it's already an existing path, return as-is
    if std::path::Path::new(model).exists() {
        return model.to_string();
    }

    let config = Config::load();
    let paths = AppPaths::new();
    let registry = match Registry::new(&config, &paths).await {
        Ok(r) => r,
        Err(_) => return model.to_string(),
    };

    match registry.resolve(model).await {
        Ok(path) => {
            println!(
                "{}",
                format!("Resolved '{}' → {}", model, path.display()).dimmed()
            );
            path.to_string_lossy().to_string()
        }
        Err(_) => {
            // Try searching for a recommendation
            if let Ok(results) = registry.search(model, 1, Some(hw)).await {
                if let Some(first) = results.first() {
                    if first.is_installed {
                        if let Some(ref installed) = first.metadata.installed {
                            let path = installed.path.join(&installed.filename);
                            println!(
                                "{}",
                                format!("Resolved '{}' → {}", model, path.display()).dimmed()
                            );
                            return path.to_string_lossy().to_string();
                        }
                    }
                }
            }
            model.to_string()
        }
    }
}
