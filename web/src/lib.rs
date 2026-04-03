// This crate targets wasm32; suppress dead_code warnings for non-wasm builds.
#![allow(dead_code, unused_imports)]

mod model_fetch;
mod webgpu;

use wasm_bindgen::prelude::*;

use hivebear_core::profiler;
use hivebear_inference::engine::InferenceBackend;
use hivebear_inference::error::InferenceError;
use hivebear_inference::types::*;

use std::sync::Mutex;

#[cfg(target_arch = "wasm32")]
use hivebear_inference::engine::candle_wasm::CandleWasmBackend;

#[cfg(target_arch = "wasm32")]
static BACKEND: Mutex<Option<CandleWasmBackend>> = Mutex::new(None);

#[cfg(not(target_arch = "wasm32"))]
static BACKEND: Mutex<Option<()>> = Mutex::new(None);

static CURRENT_HANDLE: Mutex<Option<ModelHandle>> = Mutex::new(None);

fn err_to_js(e: InferenceError) -> JsValue {
    JsValue::from_str(&e.to_string())
}

fn text_content(s: &str) -> Vec<ContentPart> {
    vec![ContentPart::Text {
        text: s.to_string(),
    }]
}

#[wasm_bindgen]
#[cfg(target_arch = "wasm32")]
pub async fn init() -> Result<JsValue, JsValue> {
    console_error_panic_hook::set_once();

    let profile = profiler::profile();

    let backend = CandleWasmBackend::new();
    *BACKEND.lock().unwrap_or_else(|e| e.into_inner()) = Some(backend);

    serde_wasm_bindgen::to_value(&profile).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
#[cfg(target_arch = "wasm32")]
pub async fn get_hardware_profile() -> Result<JsValue, JsValue> {
    let profile = profiler::profile_async().await;
    serde_wasm_bindgen::to_value(&profile).map_err(|e| JsValue::from_str(&e.to_string()))
}

#[wasm_bindgen]
#[cfg(target_arch = "wasm32")]
pub fn load_model_from_bytes(
    model_bytes: &[u8],
    tokenizer_bytes: &[u8],
    model_name: &str,
) -> Result<f64, JsValue> {
    let mut backend_guard = BACKEND.lock().unwrap_or_else(|e| e.into_inner());
    let backend = backend_guard
        .as_mut()
        .ok_or_else(|| JsValue::from_str("Not initialized. Call init() first."))?;

    let handle = backend
        .load_model_from_bytes(model_bytes, tokenizer_bytes, model_name)
        .map_err(err_to_js)?;

    let id = handle.id as f64;
    *CURRENT_HANDLE.lock().unwrap_or_else(|e| e.into_inner()) = Some(handle);
    Ok(id)
}

#[wasm_bindgen]
#[cfg(target_arch = "wasm32")]
pub async fn load_model_from_url(
    model_url: &str,
    tokenizer_url: &str,
    model_name: &str,
    on_progress: Option<js_sys::Function>,
) -> Result<f64, JsValue> {
    let model_bytes = model_fetch::fetch_with_progress(model_url, on_progress.as_ref()).await?;
    let tokenizer_bytes = model_fetch::fetch_bytes(tokenizer_url).await?;
    load_model_from_bytes(&model_bytes, &tokenizer_bytes, model_name)
}

#[wasm_bindgen]
#[cfg(target_arch = "wasm32")]
pub async fn generate(
    prompt: &str,
    system_prompt: Option<String>,
    max_tokens: u32,
) -> Result<String, JsValue> {
    let backend_guard = BACKEND.lock().unwrap_or_else(|e| e.into_inner());
    let backend = backend_guard
        .as_ref()
        .ok_or_else(|| JsValue::from_str("Not initialized."))?;

    let handle_guard = CURRENT_HANDLE.lock().unwrap_or_else(|e| e.into_inner());
    let handle = handle_guard
        .as_ref()
        .ok_or_else(|| JsValue::from_str("No model loaded."))?;

    let mut messages = Vec::new();
    if let Some(sys) = system_prompt {
        messages.push(ChatMessage::System(sys));
    }
    messages.push(ChatMessage::User(text_content(prompt)));

    let req = GenerateRequest {
        messages,
        max_tokens,
        ..Default::default()
    };

    let gen_result: hivebear_inference::error::Result<GenerateResponse> =
        backend.generate(handle, &req).await;
    let response = gen_result.map_err(err_to_js)?;

    match response {
        GenerateResponse::Text(text) => Ok(text),
        GenerateResponse::ToolCall(tc) => Ok(serde_json::to_string(&tc).unwrap_or_default()),
        GenerateResponse::Mixed(blocks) => {
            let mut result = String::new();
            for block in blocks {
                match block {
                    ContentBlock::Text(t) => result.push_str(&t),
                    ContentBlock::ToolCall(tc) => {
                        result.push_str(&serde_json::to_string(&tc).unwrap_or_default())
                    }
                }
            }
            Ok(result)
        }
    }
}

#[wasm_bindgen]
#[cfg(target_arch = "wasm32")]
pub async fn stream_generate(
    prompt: &str,
    system_prompt: Option<String>,
    max_tokens: u32,
    on_token: &js_sys::Function,
) -> Result<String, JsValue> {
    use futures::StreamExt;

    let backend_guard = BACKEND.lock().unwrap_or_else(|e| e.into_inner());
    let backend = backend_guard
        .as_ref()
        .ok_or_else(|| JsValue::from_str("Not initialized."))?;

    let handle_guard = CURRENT_HANDLE.lock().unwrap_or_else(|e| e.into_inner());
    let handle = handle_guard
        .as_ref()
        .ok_or_else(|| JsValue::from_str("No model loaded."))?;

    let mut messages = Vec::new();
    if let Some(sys) = system_prompt {
        messages.push(ChatMessage::System(sys));
    }
    messages.push(ChatMessage::User(text_content(prompt)));

    let req = GenerateRequest {
        messages,
        max_tokens,
        ..Default::default()
    };

    let mut stream = backend.stream(handle, &req);
    let mut full_text = String::new();

    while let Some(result) = stream.next().await {
        let token_result: Result<Token, InferenceError> = result;
        match token_result {
            Ok(token) => {
                full_text.push_str(&token.text);
                let _ = on_token.call1(&JsValue::NULL, &JsValue::from_str(&token.text));
            }
            Err(e) => {
                return Err(err_to_js(e));
            }
        }
    }

    Ok(full_text)
}

#[wasm_bindgen]
#[cfg(target_arch = "wasm32")]
pub fn unload_model() -> Result<(), JsValue> {
    let mut handle_guard = CURRENT_HANDLE.lock().unwrap_or_else(|e| e.into_inner());
    if handle_guard.is_none() {
        return Ok(());
    }
    *handle_guard = None;

    let mut backend_guard = BACKEND.lock().unwrap_or_else(|e| e.into_inner());
    *backend_guard = Some(CandleWasmBackend::new());
    Ok(())
}

#[wasm_bindgen]
pub fn is_webgpu_available() -> bool {
    webgpu::is_webgpu_available()
}
