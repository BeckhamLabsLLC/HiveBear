use serde::{Deserialize, Serialize};
use tauri::State;

use crate::error::CmdResult;
use crate::state::AppState;

// ── Response types ──────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct AuthResult {
    pub auth_mode: String,
    pub tier: String,
    pub jwt: String,
    pub license_token: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AccountInfo {
    pub auth_mode: String,
    pub tier: String,
    pub status: String,
    pub user_id: Option<String>,
    pub email: Option<String>,
    pub display_name: Option<String>,
    pub pubkey: Option<String>,
    pub current_period_end: Option<String>,
    pub limits: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UsageSummary {
    pub cloud_requests_today: i64,
    pub cloud_requests_limit: Option<i64>,
    pub mesh_seconds_contributed: i64,
    pub mesh_seconds_consumed: i64,
    pub model_downloads_this_month: i64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ApiKeyInfo {
    pub id: String,
    pub key_prefix: String,
    pub label: String,
    pub created_at: String,
    pub last_used_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NewApiKey {
    pub id: String,
    pub key: String,
    pub key_prefix: String,
    pub label: String,
}

// ── Helpers ─────────────────────────────────────────────────────────

fn get_server_url(state: &AppState) -> String {
    let config = state.config.lock().unwrap();
    config
        .account
        .server_url
        .clone()
        .unwrap_or_else(|| config.mesh.coordination_server.clone())
}

fn get_jwt(state: &AppState) -> Option<String> {
    state.config.lock().unwrap().account.jwt_token.clone()
}

fn save_auth(
    state: &AppState,
    auth_mode: &str,
    jwt: &str,
    refresh: &str,
    tier: &str,
    license: Option<&str>,
) {
    let mut config = state.config.lock().unwrap();
    config.account.auth_mode = Some(auth_mode.to_string());
    config.account.jwt_token = Some(jwt.to_string());
    config.account.refresh_token = Some(refresh.to_string());
    config.account.tier = Some(tier.to_string());
    config.account.license_token = license.map(|s| s.to_string());
    let _ = config.save();
}

fn clear_auth(state: &AppState) {
    let mut config = state.config.lock().unwrap();
    config.account = Default::default();
    let _ = config.save();
}

// ── Path A: Email login/register ────────────────────────────────────

/// Log in with email and password.
#[tauri::command]
pub async fn login(
    state: State<'_, AppState>,
    email: String,
    password: String,
) -> CmdResult<AuthResult> {
    let server = get_server_url(&state);
    let resp = state
        .http_client
        .post(format!("{server}/auth/login"))
        .json(&serde_json::json!({ "email": email, "password": password }))
        .send()
        .await
        .map_err(|e| format!("Connection failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("Login failed ({status}): {body}"));
    }

    let data: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;

    let jwt = data["jwt"].as_str().unwrap_or("").to_string();
    let refresh = data["refresh_token"].as_str().unwrap_or("").to_string();
    let tier = data["tier"].as_str().unwrap_or("community").to_string();

    save_auth(&state, "email", &jwt, &refresh, &tier, None);

    Ok(AuthResult {
        auth_mode: "email".into(),
        tier,
        jwt,
        license_token: None,
    })
}

/// Register a new email account.
#[tauri::command]
pub async fn register(
    state: State<'_, AppState>,
    email: String,
    password: String,
    display_name: String,
) -> CmdResult<AuthResult> {
    let server = get_server_url(&state);
    let resp = state
        .http_client
        .post(format!("{server}/auth/register"))
        .json(&serde_json::json!({
            "email": email,
            "password": password,
            "display_name": display_name,
        }))
        .send()
        .await
        .map_err(|e| format!("Connection failed: {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("Registration failed ({status}): {body}"));
    }

    let data: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;

    let jwt = data["jwt"].as_str().unwrap_or("").to_string();
    let refresh = data["refresh_token"].as_str().unwrap_or("").to_string();
    let tier = data["tier"].as_str().unwrap_or("community").to_string();

    save_auth(&state, "email", &jwt, &refresh, &tier, None);

    Ok(AuthResult {
        auth_mode: "email".into(),
        tier,
        jwt,
        license_token: None,
    })
}

// ── Path B: Anonymous device-key auth ───────────────────────────────

/// Activate device-key authentication. Uses the local Ed25519 identity.
/// Zero user input required — one-click anonymous auth.
#[tauri::command]
pub async fn activate_device(state: State<'_, AppState>) -> CmdResult<AuthResult> {
    // Load device identity (same format as hivebear-mesh: 32-byte secret + 32-byte public)
    let identity_path = state.paths.data_dir.join("node_identity.key");
    let identity = load_device_identity(&identity_path)?;
    let pubkey_hex = hex::encode(identity.verifying_key().to_bytes());

    let server = get_server_url(&state);

    // Step 1: Request challenge
    let challenge_resp = state
        .http_client
        .post(format!("{server}/auth/challenge"))
        .json(&serde_json::json!({ "pubkey": pubkey_hex }))
        .send()
        .await
        .map_err(|e| format!("Challenge request failed: {e}"))?;

    if !challenge_resp.status().is_success() {
        return Err("Failed to get auth challenge".into());
    }

    let challenge: serde_json::Value = challenge_resp.json().await.map_err(|e| e.to_string())?;
    let nonce = challenge["nonce"].as_str().ok_or("Missing nonce")?;

    // Step 2: Sign the nonce
    let nonce_bytes = hex::decode(nonce).map_err(|_| "Invalid nonce")?;
    use ed25519_dalek::Signer;
    let signature = identity.sign(&nonce_bytes);
    let sig_hex = hex::encode(signature.to_bytes());

    // Step 3: Verify
    let verify_resp = state
        .http_client
        .post(format!("{server}/auth/verify"))
        .json(&serde_json::json!({
            "pubkey": pubkey_hex,
            "nonce": nonce,
            "signature": sig_hex,
        }))
        .send()
        .await
        .map_err(|e| format!("Verify request failed: {e}"))?;

    if !verify_resp.status().is_success() {
        let status = verify_resp.status();
        let body = verify_resp.text().await.unwrap_or_default();
        return Err(format!("Device verification failed ({status}): {body}"));
    }

    let data: serde_json::Value = verify_resp.json().await.map_err(|e| e.to_string())?;

    let jwt = data["jwt"].as_str().unwrap_or("").to_string();
    let refresh = data["refresh_token"].as_str().unwrap_or("").to_string();
    let tier = data["tier"].as_str().unwrap_or("community").to_string();
    let license = data["license_token"].as_str().map(|s| s.to_string());

    save_auth(&state, "device", &jwt, &refresh, &tier, license.as_deref());

    Ok(AuthResult {
        auth_mode: "device".into(),
        tier,
        jwt,
        license_token: license,
    })
}

// ── Shared commands ─────────────────────────────────────────────────

/// Log out and clear stored credentials.
#[tauri::command]
pub async fn logout(state: State<'_, AppState>) -> CmdResult<()> {
    // Try to notify server (best-effort)
    if let Some(jwt) = get_jwt(&state) {
        let server = get_server_url(&state);
        let _ = state
            .http_client
            .post(format!("{server}/auth/logout"))
            .header("Authorization", format!("Bearer {jwt}"))
            .send()
            .await;
    }

    clear_auth(&state);
    Ok(())
}

/// Get current account info. Returns None if not authenticated.
#[tauri::command]
pub async fn get_account(state: State<'_, AppState>) -> CmdResult<Option<AccountInfo>> {
    let jwt = match get_jwt(&state) {
        Some(jwt) => jwt,
        None => return Ok(None),
    };

    let server = get_server_url(&state);
    let resp = state
        .http_client
        .get(format!("{server}/auth/me"))
        .header("Authorization", format!("Bearer {jwt}"))
        .send()
        .await;

    match resp {
        Ok(r) if r.status().is_success() => {
            let info: AccountInfo = r.json().await.map_err(|e| e.to_string())?;
            Ok(Some(info))
        }
        Ok(r) if r.status().as_u16() == 401 => {
            // JWT expired — try refresh
            match try_refresh(&state).await {
                Ok(new_jwt) => {
                    let retry = state
                        .http_client
                        .get(format!("{server}/auth/me"))
                        .header("Authorization", format!("Bearer {new_jwt}"))
                        .send()
                        .await
                        .map_err(|e| e.to_string())?;
                    if retry.status().is_success() {
                        let info: AccountInfo = retry.json().await.map_err(|e| e.to_string())?;
                        Ok(Some(info))
                    } else {
                        clear_auth(&state);
                        Ok(None)
                    }
                }
                Err(_) => {
                    // Check cached license token for offline mode
                    let config = state.config.lock().unwrap();
                    if let Some(ref _license) = config.account.license_token {
                        // Return cached info from config
                        Ok(Some(AccountInfo {
                            auth_mode: config.account.auth_mode.clone().unwrap_or_default(),
                            tier: config
                                .account
                                .tier
                                .clone()
                                .unwrap_or_else(|| "community".into()),
                            status: "active".into(),
                            user_id: config.account.user_id.clone(),
                            email: None,
                            display_name: config.account.display_name.clone(),
                            pubkey: None,
                            current_period_end: None,
                            limits: serde_json::json!({}),
                        }))
                    } else {
                        Ok(None)
                    }
                }
            }
        }
        _ => {
            // Network error — return cached info if available
            let config = state.config.lock().unwrap();
            if config.account.jwt_token.is_some() {
                Ok(Some(AccountInfo {
                    auth_mode: config.account.auth_mode.clone().unwrap_or_default(),
                    tier: config
                        .account
                        .tier
                        .clone()
                        .unwrap_or_else(|| "community".into()),
                    status: "active".into(),
                    user_id: config.account.user_id.clone(),
                    email: None,
                    display_name: config.account.display_name.clone(),
                    pubkey: None,
                    current_period_end: None,
                    limits: serde_json::json!({}),
                }))
            } else {
                Ok(None)
            }
        }
    }
}

/// Get usage summary.
#[tauri::command]
pub async fn get_usage_summary(state: State<'_, AppState>) -> CmdResult<UsageSummary> {
    let jwt = get_jwt(&state).ok_or("Not authenticated")?;
    let server = get_server_url(&state);

    let resp = state
        .http_client
        .get(format!("{server}/usage/summary"))
        .header("Authorization", format!("Bearer {jwt}"))
        .send()
        .await
        .map_err(|e| format!("Request failed: {e}"))?;

    if !resp.status().is_success() {
        return Err("Failed to fetch usage".into());
    }

    resp.json::<UsageSummary>().await.map_err(|e| e.to_string())
}

/// Create a checkout URL for upgrading.
#[tauri::command]
pub async fn create_checkout(state: State<'_, AppState>, plan: String) -> CmdResult<String> {
    let jwt = get_jwt(&state).ok_or("Not authenticated")?;
    let server = get_server_url(&state);

    let resp = state
        .http_client
        .post(format!("{server}/billing/create-checkout"))
        .header("Authorization", format!("Bearer {jwt}"))
        .json(&serde_json::json!({ "plan": plan }))
        .send()
        .await
        .map_err(|e| format!("Request failed: {e}"))?;

    if !resp.status().is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("Checkout failed: {body}"));
    }

    let data: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;
    data["checkout_url"]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| "No checkout URL returned".into())
}

/// List API keys.
#[tauri::command]
pub async fn list_api_keys(state: State<'_, AppState>) -> CmdResult<Vec<ApiKeyInfo>> {
    let jwt = get_jwt(&state).ok_or("Not authenticated")?;
    let server = get_server_url(&state);

    let resp = state
        .http_client
        .get(format!("{server}/api-keys"))
        .header("Authorization", format!("Bearer {jwt}"))
        .send()
        .await
        .map_err(|e| format!("Request failed: {e}"))?;

    if !resp.status().is_success() {
        return Err("Failed to list API keys".into());
    }

    resp.json::<Vec<ApiKeyInfo>>()
        .await
        .map_err(|e| e.to_string())
}

/// Create a new API key.
#[tauri::command]
pub async fn create_api_key(state: State<'_, AppState>, label: String) -> CmdResult<NewApiKey> {
    let jwt = get_jwt(&state).ok_or("Not authenticated")?;
    let server = get_server_url(&state);

    let resp = state
        .http_client
        .post(format!("{server}/api-keys"))
        .header("Authorization", format!("Bearer {jwt}"))
        .json(&serde_json::json!({ "label": label }))
        .send()
        .await
        .map_err(|e| format!("Request failed: {e}"))?;

    if !resp.status().is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("Failed to create API key: {body}"));
    }

    resp.json::<NewApiKey>().await.map_err(|e| e.to_string())
}

/// Revoke an API key.
#[tauri::command]
pub async fn revoke_api_key(state: State<'_, AppState>, key_id: String) -> CmdResult<()> {
    let jwt = get_jwt(&state).ok_or("Not authenticated")?;
    let server = get_server_url(&state);

    let resp = state
        .http_client
        .delete(format!("{server}/api-keys/{key_id}"))
        .header("Authorization", format!("Bearer {jwt}"))
        .send()
        .await
        .map_err(|e| format!("Request failed: {e}"))?;

    if !resp.status().is_success() {
        return Err("Failed to revoke API key".into());
    }

    Ok(())
}

// ── Internal helpers ────────────────────────────────────────────────

/// Load the device Ed25519 signing key from the identity file, or generate a new one.
fn load_device_identity(path: &std::path::Path) -> Result<ed25519_dalek::SigningKey, String> {
    if path.exists() {
        let bytes = std::fs::read(path).map_err(|e| format!("Failed to read identity: {e}"))?;
        if bytes.len() == 64 {
            let seed: [u8; 32] = bytes[..32].try_into().unwrap();
            return Ok(ed25519_dalek::SigningKey::from_bytes(&seed));
        }
    }

    // Generate new identity
    use rand::rngs::OsRng;
    let key = ed25519_dalek::SigningKey::generate(&mut OsRng);

    // Save: 32 bytes secret + 32 bytes public
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut data = Vec::with_capacity(64);
    data.extend_from_slice(&key.to_bytes());
    data.extend_from_slice(&key.verifying_key().to_bytes());
    std::fs::write(path, &data).map_err(|e| format!("Failed to save identity: {e}"))?;

    Ok(key)
}

async fn try_refresh(state: &AppState) -> Result<String, String> {
    let refresh_token = {
        let config = state.config.lock().unwrap();
        config.account.refresh_token.clone()
    };

    let refresh_token = refresh_token.ok_or("No refresh token")?;
    let server = get_server_url(state);

    let resp = state
        .http_client
        .post(format!("{server}/auth/refresh"))
        .json(&serde_json::json!({ "refresh_token": refresh_token }))
        .send()
        .await
        .map_err(|e| e.to_string())?;

    if !resp.status().is_success() {
        return Err("Refresh failed".into());
    }

    let data: serde_json::Value = resp.json().await.map_err(|e| e.to_string())?;

    let jwt = data["jwt"].as_str().unwrap_or("").to_string();
    let new_refresh = data["refresh_token"].as_str().unwrap_or("").to_string();
    let tier = data["tier"].as_str().unwrap_or("community").to_string();
    let auth_mode = data["auth_mode"].as_str().unwrap_or("email").to_string();
    let license = data["license_token"].as_str().map(|s| s.to_string());

    save_auth(
        state,
        &auth_mode,
        &jwt,
        &new_refresh,
        &tier,
        license.as_deref(),
    );

    Ok(jwt)
}
