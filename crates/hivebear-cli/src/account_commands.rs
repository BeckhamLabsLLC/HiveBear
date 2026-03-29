use clap::Subcommand;
use colored::Colorize;
use hivebear_core::Config;

#[derive(Subcommand)]
pub enum AccountAction {
    /// Log in with email and password
    Login {
        /// Email address
        #[arg(long)]
        email: Option<String>,
    },

    /// Activate anonymous device-key authentication (no email required)
    Activate,

    /// Show current account status
    Status,

    /// Log out and clear stored credentials
    Logout,

    /// Show current period usage statistics
    Usage,

    /// Create a checkout URL to upgrade your plan
    Upgrade {
        /// Plan to upgrade to: "pro" or "team"
        #[arg(long, default_value = "pro")]
        plan: String,
    },

    /// Link this device to an email account
    Link {
        /// Email address for the account to link to
        #[arg(long)]
        email: Option<String>,
    },

    /// Export device identity key for backup
    ExportKey {
        /// Output file path
        #[arg(long)]
        output: Option<String>,
    },

    /// Manage API keys
    ApiKeys {
        #[command(subcommand)]
        action: ApiKeyAction,
    },
}

#[derive(Subcommand)]
pub enum ApiKeyAction {
    /// List your API keys
    List,
    /// Create a new API key
    Create {
        /// Label for the key
        #[arg(long)]
        label: Option<String>,
    },
    /// Revoke an API key
    Revoke {
        /// API key ID to revoke
        id: String,
    },
}

fn get_server_url(config: &Config) -> String {
    config
        .account
        .server_url
        .clone()
        .unwrap_or_else(|| config.mesh.coordination_server.clone())
}

pub async fn cmd_account(action: AccountAction) {
    match action {
        AccountAction::Login { email } => cmd_login(email).await,
        AccountAction::Activate => cmd_activate().await,
        AccountAction::Status => cmd_status().await,
        AccountAction::Logout => cmd_logout(),
        AccountAction::Usage => cmd_usage().await,
        AccountAction::Upgrade { plan } => cmd_upgrade(plan).await,
        AccountAction::Link { email } => cmd_link(email).await,
        AccountAction::ExportKey { output } => cmd_export_key(output),
        AccountAction::ApiKeys { action } => cmd_api_keys(action).await,
    }
}

async fn cmd_login(email_arg: Option<String>) {
    let email = email_arg.unwrap_or_else(|| {
        eprint!("Email: ");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        input.trim().to_string()
    });

    let password = rpassword::prompt_password("Password: ").unwrap_or_else(|_| String::new());
    if password.is_empty() {
        eprintln!("{}", "Password cannot be empty".red());
        return;
    }

    let config = Config::load();
    let server = get_server_url(&config);
    let client = reqwest::Client::new();

    match client
        .post(format!("{server}/auth/login"))
        .json(&serde_json::json!({ "email": email, "password": password }))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            let data: serde_json::Value = resp.json().await.unwrap_or(serde_json::Value::Null);
            let jwt = data["jwt"].as_str().unwrap_or("");
            let refresh = data["refresh_token"].as_str().unwrap_or("");
            let tier = data["tier"].as_str().unwrap_or("community");

            let mut config = Config::load();
            config.account.auth_mode = Some("email".into());
            config.account.jwt_token = Some(jwt.to_string());
            config.account.refresh_token = Some(refresh.to_string());
            config.account.tier = Some(tier.to_string());
            let _ = config.save();

            println!(
                "{} Logged in as {} (tier: {})",
                "OK".green(),
                email.cyan(),
                tier.yellow()
            );
        }
        Ok(resp) => {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_else(|_| String::new());
            eprintln!("{} Login failed ({}): {}", "ERR".red(), status, body);
        }
        Err(e) => {
            eprintln!("{} Connection failed: {}", "ERR".red(), e);
        }
    }
}

async fn cmd_activate() {
    println!("{}", "Activating device key authentication...".dimmed());

    let config = Config::load();
    let paths = hivebear_core::AppPaths::new();
    let identity_path = paths.data_dir.join("node_identity.key");

    let identity = load_device_identity(&identity_path);
    let pubkey_hex = hex::encode(identity.verifying_key().to_bytes());

    println!("Device fingerprint: {}", pubkey_hex[..16].cyan());

    let server = get_server_url(&config);
    let client = reqwest::Client::new();

    // Step 1: Challenge
    let challenge_resp = match client
        .post(format!("{server}/auth/challenge"))
        .json(&serde_json::json!({ "pubkey": pubkey_hex }))
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => r
            .json::<serde_json::Value>()
            .await
            .unwrap_or(serde_json::Value::Null),
        Ok(r) => {
            eprintln!("{} Challenge failed: {}", "ERR".red(), r.status());
            return;
        }
        Err(e) => {
            eprintln!("{} Connection failed: {}", "ERR".red(), e);
            return;
        }
    };

    let nonce = match challenge_resp["nonce"].as_str() {
        Some(n) => n.to_string(),
        None => {
            eprintln!("{} Invalid challenge response", "ERR".red());
            return;
        }
    };

    // Step 2: Sign
    let nonce_bytes = hex::decode(&nonce).unwrap_or_else(|_| Vec::new());
    use ed25519_dalek::Signer;
    let signature = identity.sign(&nonce_bytes);
    let sig_hex = hex::encode(signature.to_bytes());

    // Step 3: Verify
    match client
        .post(format!("{server}/auth/verify"))
        .json(&serde_json::json!({
            "pubkey": pubkey_hex,
            "nonce": nonce,
            "signature": sig_hex,
        }))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            let data: serde_json::Value = resp.json().await.unwrap_or(serde_json::Value::Null);
            let jwt = data["jwt"].as_str().unwrap_or("");
            let refresh = data["refresh_token"].as_str().unwrap_or("");
            let tier = data["tier"].as_str().unwrap_or("community");
            let license = data["license_token"].as_str().map(|s| s.to_string());

            let mut config = Config::load();
            config.account.auth_mode = Some("device".into());
            config.account.jwt_token = Some(jwt.to_string());
            config.account.refresh_token = Some(refresh.to_string());
            config.account.tier = Some(tier.to_string());
            config.account.license_token = license;
            let _ = config.save();

            println!(
                "{} Device activated (tier: {})",
                "OK".green(),
                tier.yellow()
            );
            println!(
                "{}",
                "Your device key is your identity. Back it up with: hivebear account export-key"
                    .dimmed()
            );
        }
        Ok(resp) => {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_else(|_| String::new());
            eprintln!("{} Verification failed ({}): {}", "ERR".red(), status, body);
        }
        Err(e) => {
            eprintln!("{} Connection failed: {}", "ERR".red(), e);
        }
    }
}

async fn cmd_status() {
    let config = Config::load();

    match config.account.auth_mode.as_deref() {
        Some("email") => {
            println!("{} Email account", "Auth:".dimmed());
            if let Some(ref uid) = config.account.user_id {
                println!("{} {}", "User:".dimmed(), uid);
            }
        }
        Some("device") => {
            println!("{} Anonymous device key", "Auth:".dimmed());
            let paths = hivebear_core::AppPaths::new();
            let identity_path = paths.data_dir.join("node_identity.key");
            if identity_path.exists() {
                let identity = load_device_identity(&identity_path);
                let pk = hex::encode(identity.verifying_key().to_bytes());
                println!("{} {}...", "Key: ".dimmed(), &pk[..16]);
            }
        }
        _ => {
            println!(
                "{}",
                "Not authenticated. Use `hivebear account login` or `hivebear account activate`."
                    .yellow()
            );
            return;
        }
    }

    println!(
        "{} {}",
        "Tier:".dimmed(),
        config
            .account
            .tier
            .as_deref()
            .unwrap_or("community")
            .yellow()
    );

    if let Some(ref jwt) = config.account.jwt_token {
        let server = get_server_url(&config);
        let client = reqwest::Client::new();
        if let Ok(resp) = client
            .get(format!("{server}/billing/status"))
            .header("Authorization", format!("Bearer {jwt}"))
            .send()
            .await
        {
            if resp.status().is_success() {
                let data: serde_json::Value = resp.json().await.unwrap_or(serde_json::Value::Null);
                if let Some(status) = data["status"].as_str() {
                    println!("{} {}", "Status:".dimmed(), status);
                }
                if let Some(end) = data["current_period_end"].as_str() {
                    println!("{} {}", "Renews:".dimmed(), end);
                }
            }
        }
    }
}

fn cmd_logout() {
    let mut config = Config::load();
    config.account = hivebear_core::config::AccountConfig::default();
    let _ = config.save();
    println!("{} Logged out", "OK".green());
}

async fn cmd_usage() {
    let config = Config::load();
    let jwt = match config.account.jwt_token.as_deref() {
        Some(jwt) => jwt,
        None => {
            eprintln!("{}", "Not authenticated.".yellow());
            return;
        }
    };

    let server = get_server_url(&config);
    let client = reqwest::Client::new();

    match client
        .get(format!("{server}/usage/summary"))
        .header("Authorization", format!("Bearer {jwt}"))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            let data: serde_json::Value = resp.json().await.unwrap_or(serde_json::Value::Null);
            println!("\n{}", "  Usage Summary  ".bold().white().on_blue());
            println!();
            let cloud_today = data["cloud_requests_today"].as_i64().unwrap_or(0);
            let cloud_limit = data["cloud_requests_limit"].as_i64();
            let limit_str = match cloud_limit {
                Some(l) => format!(" / {l}"),
                None => " (unlimited)".into(),
            };
            println!("  Cloud requests today:  {cloud_today}{limit_str}");
            println!(
                "  Mesh contributed:      {}m",
                data["mesh_seconds_contributed"].as_i64().unwrap_or(0) / 60
            );
            println!(
                "  Downloads this month:  {}",
                data["model_downloads_this_month"].as_i64().unwrap_or(0)
            );
            println!();
        }
        Ok(resp) => {
            eprintln!("{} Failed ({})", "ERR".red(), resp.status());
        }
        Err(e) => {
            eprintln!("{} Connection failed: {}", "ERR".red(), e);
        }
    }
}

async fn cmd_upgrade(plan: String) {
    let config = Config::load();
    let jwt = match config.account.jwt_token.as_deref() {
        Some(jwt) => jwt,
        None => {
            eprintln!(
                "{}",
                "Not authenticated. Please login or activate first.".yellow()
            );
            return;
        }
    };

    let server = get_server_url(&config);
    let client = reqwest::Client::new();

    match client
        .post(format!("{server}/billing/create-checkout"))
        .header("Authorization", format!("Bearer {jwt}"))
        .json(&serde_json::json!({ "plan": plan }))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            let data: serde_json::Value = resp.json().await.unwrap_or(serde_json::Value::Null);
            if let Some(url) = data["checkout_url"].as_str() {
                println!("Open this URL to complete your upgrade:");
                println!("\n  {}\n", url.cyan().underline());
            }
        }
        Ok(resp) => {
            let body = resp.text().await.unwrap_or_else(|_| String::new());
            eprintln!("{} Checkout failed: {}", "ERR".red(), body);
        }
        Err(e) => {
            eprintln!("{} Connection failed: {}", "ERR".red(), e);
        }
    }
}

async fn cmd_link(_email: Option<String>) {
    println!(
        "{}",
        "Device-to-account linking is not yet implemented in the CLI.".yellow()
    );
    println!("Use the desktop app for this feature.");
}

fn cmd_export_key(output: Option<String>) {
    let paths = hivebear_core::AppPaths::new();
    let identity_path = paths.data_dir.join("node_identity.key");

    if !identity_path.exists() {
        eprintln!("{} No device identity found.", "ERR".red());
        return;
    }

    let dest = output.unwrap_or_else(|| "hivebear-identity-backup.key".into());

    match std::fs::copy(&identity_path, &dest) {
        Ok(_) => {
            println!("{} Identity exported to {}", "OK".green(), dest.cyan());
            println!(
                "{}",
                "Keep this file safe! Anyone with it can authenticate as your device.".yellow()
            );
        }
        Err(e) => {
            eprintln!("{} Export failed: {}", "ERR".red(), e);
        }
    }
}

async fn cmd_api_keys(action: ApiKeyAction) {
    let config = Config::load();
    let jwt = match config.account.jwt_token.as_deref() {
        Some(jwt) => jwt,
        None => {
            eprintln!("{}", "Not authenticated.".yellow());
            return;
        }
    };

    let server = get_server_url(&config);
    let client = reqwest::Client::new();

    match action {
        ApiKeyAction::List => {
            match client
                .get(format!("{server}/api-keys"))
                .header("Authorization", format!("Bearer {jwt}"))
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {
                    let keys: Vec<serde_json::Value> =
                        resp.json().await.unwrap_or_else(|_| Vec::new());
                    if keys.is_empty() {
                        println!("No API keys.");
                    } else {
                        println!("\n{}", "  API Keys  ".bold().white().on_blue());
                        println!();
                        for key in &keys {
                            println!(
                                "  {}  {}  {}",
                                key["key_prefix"].as_str().unwrap_or("?").cyan(),
                                key["label"].as_str().unwrap_or("").dimmed(),
                                key["created_at"].as_str().unwrap_or("").dimmed(),
                            );
                        }
                        println!();
                    }
                }
                Ok(resp) => eprintln!("{} Failed ({})", "ERR".red(), resp.status()),
                Err(e) => eprintln!("{} Connection failed: {}", "ERR".red(), e),
            }
        }
        ApiKeyAction::Create { label } => {
            let label = label.unwrap_or_else(|| "cli".into());
            match client
                .post(format!("{server}/api-keys"))
                .header("Authorization", format!("Bearer {jwt}"))
                .json(&serde_json::json!({ "label": label }))
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {
                    let data: serde_json::Value =
                        resp.json().await.unwrap_or(serde_json::Value::Null);
                    println!("{} API key created:", "OK".green());
                    println!();
                    println!("  {}", data["key"].as_str().unwrap_or("?").cyan());
                    println!();
                    println!(
                        "{}",
                        "Copy this key now - it will not be shown again.".yellow()
                    );
                }
                Ok(resp) => {
                    let body = resp.text().await.unwrap_or_else(|_| String::new());
                    eprintln!("{} Failed: {}", "ERR".red(), body);
                }
                Err(e) => eprintln!("{} Connection failed: {}", "ERR".red(), e),
            }
        }
        ApiKeyAction::Revoke { id } => {
            match client
                .delete(format!("{server}/api-keys/{id}"))
                .header("Authorization", format!("Bearer {jwt}"))
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => {
                    println!("{} API key revoked", "OK".green());
                }
                Ok(resp) => eprintln!("{} Failed ({})", "ERR".red(), resp.status()),
                Err(e) => eprintln!("{} Connection failed: {}", "ERR".red(), e),
            }
        }
    }
}

/// Load the device Ed25519 signing key, or generate a new one.
fn load_device_identity(path: &std::path::Path) -> ed25519_dalek::SigningKey {
    if path.exists() {
        if let Ok(bytes) = std::fs::read(path) {
            if bytes.len() == 64 {
                let seed: [u8; 32] = bytes[..32].try_into().unwrap();
                return ed25519_dalek::SigningKey::from_bytes(&seed);
            }
        }
    }

    use rand::rngs::OsRng;
    let key = ed25519_dalek::SigningKey::generate(&mut OsRng);

    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut data = Vec::with_capacity(64);
    data.extend_from_slice(&key.to_bytes());
    data.extend_from_slice(&key.verifying_key().to_bytes());
    let _ = std::fs::write(path, &data);

    key
}
