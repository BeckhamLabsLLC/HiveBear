# Contributing to HiveBear

Thank you for your interest in contributing to HiveBear! This guide will help you get started.

## Prerequisites

- **Rust 1.75+** - Install via [rustup](https://rustup.rs/)
- **Node.js 22+** - Required for the desktop app frontend
- **npm** - Comes with Node.js

## Building

### Full workspace

```bash
cargo build --workspace
```

### Individual crates

```bash
cargo build -p hivebear-core
cargo build -p hivebear-inference
cargo build -p hivebear-registry
cargo build -p hivebear-mesh
cargo build -p hivebear-persistence
cargo build -p hivebear-cli
```

### Desktop app

The desktop app is built with [Tauri](https://tauri.app/) and requires the Tauri CLI.

```bash
cd apps/desktop
npm install
npm run dev
```

### WASM module

Requires [wasm-pack](https://rustwasm.github.io/wasm-pack/).

```bash
cd web
./build.sh
```

## Testing

```bash
cargo test --workspace
```

## Linting

Clippy warnings are treated as errors in CI. Run both checks before submitting a PR:

```bash
cargo clippy --workspace --exclude hivebear-web -- -D warnings
cargo fmt --all -- --check
```

## Architecture Overview

HiveBear is organized as a Cargo workspace with eight crates:

| Crate | Description |
|---|---|
| **hivebear-core** | Shared types, configuration, hardware profiling, and model recommendation logic. The foundation all other crates build on. |
| **hivebear-inference** | Multi-engine inference orchestrator supporting llama.cpp, Candle, ONNX, MLX, and cloud backends. |
| **hivebear-registry** | Model discovery and management. HuggingFace integration, downloads, storage, format conversion. |
| **hivebear-mesh** | P2P distributed inference. QUIC transport, peer discovery, swarm scheduling, trust and reputation. |
| **hivebear-persistence** | Conversation persistence via SQLite. Stores chat history and session metadata. |
| **hivebear-cli** | Command-line interface and OpenAI-compatible API server (`hivebear profile`, `hivebear run`, etc.). |
| **hivebear-web** | WASM bridge for browser-based LLM inference via WebGPU and Candle. |
| **desktop** | Tauri 2.x + React desktop application. Hardware dashboard, model browser, chat interface, benchmarks. |

## Pull Request Guidelines

- **One PR per feature or fix.** Keep changes focused and reviewable.
- **Write descriptive commit messages.** Explain *what* changed and *why*.
- **Run tests before submitting.** All of `cargo test --workspace`, `cargo clippy --workspace --exclude hivebear-web -- -D warnings`, and `cargo fmt --all -- --check` must pass.
- **Add tests** for new functionality when possible.
- **Update documentation** if your change affects public APIs or user-facing behavior.

## Code Style

- **rustfmt is enforced.** Run `cargo fmt --all` before committing.
- **Clippy warnings are errors in CI.** Fix all warnings before submitting.
- **Follow existing patterns.** Look at surrounding code for conventions on error handling, module structure, and naming.
- **Keep dependencies minimal.** Propose new dependencies in the PR description with a justification.

## Getting Help

If you have questions about contributing, open a [Discussion](https://github.com/BeckhamLabsLLC/HiveBear/discussions) or ask in an issue. We're happy to help you get started.
