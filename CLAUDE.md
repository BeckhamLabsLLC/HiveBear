# HiveBear Public — Client, Desktop App, WASM Bridge

## Quick Commands
```sh
cargo build                                                            # Build default members (excludes web)
cargo test                                                             # Run all tests
cargo clippy --workspace --exclude hivebear-web -- -D warnings         # Lint — MUST exclude web
cargo fmt --all -- --check                                             # Format check
cd web && wasm-pack build --target web                                 # Build WASM (needs wasm32 target)
cd apps/desktop && npm ci && npm run dev                               # Tauri desktop dev
cargo deny check advisories && cargo deny check licenses               # Security audit
cd apps/desktop && npm audit --audit-level=critical                    # NPM audit
cargo check -p hivebear-core -p hivebear-inference --target aarch64-unknown-linux-gnu  # ARM cross-check
```

## Workspace (8 crates)
| Crate | Purpose |
|-------|---------|
| `hivebear-core` | Hardware profiling, config, types. Foundation — no deps on other crates |
| `hivebear-inference` | Multi-engine orchestrator. Feature-gated: llamacpp, candle, onnx, mlx, cloud, wasm |
| `hivebear-registry` | Model discovery, HuggingFace downloads, format conversion |
| `hivebear-mesh` | P2P mesh: QUIC transport (quinn/rustls), swarm scheduling, reputation/trust |
| `hivebear-persistence` | Chat history in SQLite (rusqlite bundled) |
| `hivebear-cli` | CLI binary (`hivebear`) + Ollama-compatible API server (axum + SSE) |
| `hivebear-web` | WASM bridge (cdylib). NOT in default-members — build via wasm-pack only |
| `apps/desktop/src-tauri` | Tauri 2 desktop app (React 19 + Vite 6 + TailwindCSS 4) |

## Critical Gotchas
- `hivebear-web` is excluded from default-members — `cargo build` skips it intentionally
- Clippy MUST use `--exclude hivebear-web` or it fails (requires wasm32 target)
- CI also tests with `--no-default-features` on hivebear-inference — do the same locally
- `insecure-dev` feature on hivebear-mesh disables TLS verification — NEVER in release builds
- Linux build deps: `sudo apt-get install -y libwebkit2gtk-4.1-dev libappindicator3-dev librsvg2-dev patchelf`
- Desktop `mobile` feature disables local inference engines (ARM FP16 build issues)

## Feature Flags (hivebear-inference)
- default = `["candle", "llamacpp"]`
- GPU: `llamacpp-cuda`, `llamacpp-metal`, `llamacpp-vulkan`
- Other engines: `onnx`, `mlx`, `cloud` (OpenAI/Anthropic passthrough)
- WASM: `wasm` (Candle + WebGPU, separate target)
- All: `all-engines` = llamacpp + candle + onnx

## Desktop App (`apps/desktop/`)
- Tauri 2 + React 19 + Vite 6 + TailwindCSS 4 + react-router-dom 7
- Tauri Rust source: `apps/desktop/src-tauri/`
- Node 22+ required

## CI Checks (all must pass)
1. `cargo build` + `cargo test` (Linux, macOS, Windows)
2. `cargo build -p hivebear-inference --no-default-features` + test
3. `cargo clippy --workspace --exclude hivebear-web -- -D warnings`
4. `cargo fmt --all -- --check`
5. WASM: `cd web && wasm-pack build --target web`
6. ARM: `cargo check -p hivebear-core -p hivebear-inference --target aarch64-unknown-linux-gnu`
7. `cargo deny check advisories` + `cargo deny check licenses`
8. `cd apps/desktop && npm ci && npm audit --audit-level=critical`

## GitHub: BeckhamLabsLLC/HiveBear
- Version: 0.1.5, Edition 2021, MIT license
- Branch: main
