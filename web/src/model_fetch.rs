use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, RequestMode, Response};

/// Fetch raw bytes from a URL.
pub async fn fetch_bytes(url: &str) -> Result<Vec<u8>, JsValue> {
    let opts = RequestInit::new();
    opts.set_method("GET");
    opts.set_mode(RequestMode::Cors);

    let request = Request::new_with_str_and_init(url, &opts)?;

    let window = web_sys::window().ok_or_else(|| JsValue::from_str("No window"))?;
    let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;
    let resp: Response = resp_value.dyn_into()?;

    if !resp.ok() {
        return Err(JsValue::from_str(&format!(
            "HTTP error: {} {}",
            resp.status(),
            resp.status_text()
        )));
    }

    let array_buffer = JsFuture::from(resp.array_buffer()?).await?;
    let uint8_array = js_sys::Uint8Array::new(&array_buffer);
    Ok(uint8_array.to_vec())
}

/// Fetch raw bytes from a URL with progress reporting.
///
/// Calls `on_progress(downloaded_bytes, total_bytes)` periodically.
/// `total_bytes` is 0 if Content-Length is not available.
pub async fn fetch_with_progress(
    url: &str,
    on_progress: Option<&js_sys::Function>,
) -> Result<Vec<u8>, JsValue> {
    let opts = RequestInit::new();
    opts.set_method("GET");
    opts.set_mode(RequestMode::Cors);

    let request = Request::new_with_str_and_init(url, &opts)?;

    let window = web_sys::window().ok_or_else(|| JsValue::from_str("No window"))?;
    let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;
    let resp: Response = resp_value.dyn_into()?;

    if !resp.ok() {
        return Err(JsValue::from_str(&format!(
            "HTTP error: {} {}",
            resp.status(),
            resp.status_text()
        )));
    }

    // Get total size from Content-Length header
    let total_bytes: f64 = resp
        .headers()
        .get("Content-Length")
        .ok()
        .flatten()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0);

    // If no progress callback or no streaming body, use simple fetch
    let on_progress = match on_progress {
        Some(cb) => cb,
        None => {
            let array_buffer = JsFuture::from(resp.array_buffer()?).await?;
            let uint8_array = js_sys::Uint8Array::new(&array_buffer);
            return Ok(uint8_array.to_vec());
        }
    };

    // Stream the response body for progress reporting
    let body = resp
        .body()
        .ok_or_else(|| JsValue::from_str("Response has no body"))?;

    let reader = body
        .get_reader()
        .dyn_into::<web_sys::ReadableStreamDefaultReader>()?;

    let mut downloaded: f64 = 0.0;
    let mut chunks: Vec<Vec<u8>> = Vec::new();

    loop {
        let result = JsFuture::from(reader.read()).await?;
        let done = js_sys::Reflect::get(&result, &JsValue::from_str("done"))?
            .as_bool()
            .unwrap_or(true);

        if done {
            break;
        }

        let value = js_sys::Reflect::get(&result, &JsValue::from_str("value"))?;
        let chunk = js_sys::Uint8Array::new(&value);
        let chunk_vec = chunk.to_vec();

        downloaded += chunk_vec.len() as f64;
        chunks.push(chunk_vec);

        // Report progress
        let _ = on_progress.call2(
            &JsValue::NULL,
            &JsValue::from_f64(downloaded),
            &JsValue::from_f64(total_bytes),
        );
    }

    // Concatenate all chunks
    let total_len: usize = chunks.iter().map(|c| c.len()).sum();
    let mut result = Vec::with_capacity(total_len);
    for chunk in chunks {
        result.extend_from_slice(&chunk);
    }

    Ok(result)
}
