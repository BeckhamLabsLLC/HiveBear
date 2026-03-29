use wasm_bindgen::prelude::*;

/// Check if WebGPU is available in the current browser.
pub fn is_webgpu_available() -> bool {
    web_sys::window()
        .map(|w| {
            let gpu =
                js_sys::Reflect::get(&w.navigator(), &JsValue::from_str("gpu")).unwrap_or_default();
            !gpu.is_undefined() && !gpu.is_null()
        })
        .unwrap_or(false)
}
