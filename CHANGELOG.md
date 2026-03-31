# Changelog

All notable changes to HiveBear are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.5] - 2026-03-30

### Added
- Android/mobile support via Tauri
- Community benchmark sharing and hardware-matched model recommendations
- Ollama-compatible `serve` mode with overflow-to-mesh
- Universal AI backend for IDE extensions (OpenAI-compatible API server)
- Comprehensive security hardening across all crates

### Fixed
- Tauri bundle artifact paths (workspace root, not src-tauri)
- macOS minimum system version set to 10.15
- `MACOSX_DEPLOYMENT_TARGET=10.15` for llama.cpp `std::filesystem` compatibility
- Corrected default URLs from hivebear.dev to hivebear.com
- Clippy `single_match` lint and `cargo fmt` diffs

## [0.1.2] - 2026-03-30

### Fixed
- Release workflow tolerates matrix failures
- Docker Rust updated from 1.83 to 1.88 (time-0.3.47 compatibility)
- OpenSSL added for cross-compile builds
- Release pipeline reliability improvements
- Added `update` and `uninstall` CLI commands

### Changed
- cargo-deny v2 configuration updated (removed deprecated keys)
- License allowlist expanded: MPL-2.0, Apache-2.0 WITH LLVM-exception, CDLA-Permissive-2.0
- Relaxed npm audit to critical-level only

## [0.1.0] - 2026-03-29

### Added
- Initial open-source release
- Hardware profiling and smart model recommendations
- Multi-engine inference orchestration (llama.cpp, Candle)
- P2P mesh distributed inference via QUIC transport
- OpenAI-compatible API server
- Model registry with HuggingFace integration
- SQLite conversation persistence
- Tauri desktop application (Linux, macOS, Windows)
- WASM bridge for browser-based inference
- CLI with quickstart, profile, recommend, search, install, run, mesh commands
- Docker images (CPU and CUDA variants)
- Cross-platform install script with SHA256 verification
- Homebrew tap and Scoop bucket

[0.1.5]: https://github.com/BeckhamLabsLLC/HiveBear/releases/tag/v0.1.5
[0.1.2]: https://github.com/BeckhamLabsLLC/HiveBear/releases/tag/v0.1.2
[0.1.0]: https://github.com/BeckhamLabsLLC/HiveBear/releases/tag/v0.1.0
