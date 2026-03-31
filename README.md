<p align="center">
  <img src="assets/logo-readme.png" alt="HiveBear" width="120" />
</p>

<h1 align="center">HiveBear</h1>

<p align="center">
  <strong>Run AI models on the hardware you already own.</strong>
</p>

<p align="center">
  <a href="https://github.com/BeckhamLabsLLC/HiveBear/actions"><img src="https://github.com/BeckhamLabsLLC/HiveBear/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://github.com/BeckhamLabsLLC/HiveBear/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License" /></a>
  <a href="https://github.com/BeckhamLabsLLC/HiveBear/releases/latest"><img src="https://img.shields.io/github/v/release/BeckhamLabsLLC/HiveBear?label=release" alt="Latest Release" /></a>
</p>

---

## The Problem

Running open-source AI models locally is harder than it should be. You have to figure out which models fit your hardware, which quantizations to use, which inference engine works with your GPU, and how to configure it all. Get it wrong and you're staring at an out-of-memory crash or waiting 30 seconds per token.

Cloud APIs solve the complexity problem but create new ones: costs scale with usage, your data leaves your machine, and you're locked into someone else's model lineup.

## What HiveBear Does

HiveBear is a local AI runtime that automatically matches models to your hardware. One command profiles your device, picks the best model and quantization for your specific CPU/RAM/GPU, downloads it, and starts a chat:

```bash
hivebear quickstart
```

It works on everything from a Raspberry Pi to a multi-GPU workstation, and when a model is too large for one device, HiveBear can distribute inference across multiple machines over the network.

### Core Capabilities

**Automatic hardware profiling and model selection.** HiveBear benchmarks your CPU, RAM, GPU (CUDA/Metal/Vulkan), and storage, then recommends models that will actually run well. It picks the right quantization level (Q4, Q5, Q8, F16) based on your available memory and compute, so you get the best quality your hardware can deliver without manual tuning.

**Multi-engine inference.** Instead of being locked to one backend, HiveBear selects the best inference engine for the job. llama.cpp for GGUF models with GPU offloading, Candle for pure-Rust execution when native compilation matters. The engine selection is automatic based on model format and hardware.

**Ollama-compatible API server.** `hivebear serve` is a drop-in replacement for `ollama serve`. Point your existing tools, IDE extensions, and scripts at `localhost:11434` and they work without code changes. It also exposes a full OpenAI-compatible API at the same endpoint.

**P2P distributed inference.** Connect multiple devices into a mesh network over QUIC/TLS. HiveBear splits model layers across machines so you can run a 70B model on three 8GB laptops. Each device contributes what it can. No central server required.

**Cross-platform.** Native binaries for Linux, macOS, and Windows. ARM support for Raspberry Pi and Apple Silicon. A Tauri desktop app with hardware dashboard and chat UI. A WASM build for browser-based inference. Android APK.

## Install

```bash
# One-line install (Linux/macOS)
curl -fsSL https://raw.githubusercontent.com/BeckhamLabsLLC/HiveBear/main/install.sh | bash

# Homebrew
brew install BeckhamLabsLLC/hivebear/hivebear

# Scoop (Windows)
scoop bucket add hivebear https://github.com/BeckhamLabsLLC/scoop-hivebear
scoop install hivebear

# Docker
docker run -it --rm -p 11434:11434 ghcr.io/beckhamlabsllc/hivebear quickstart

# Docker with NVIDIA GPU
docker run -it --rm --gpus all -p 11434:11434 ghcr.io/beckhamlabsllc/hivebear:latest-cuda quickstart

# Build from source
cargo install --git https://github.com/BeckhamLabsLLC/HiveBear hivebear-cli
```

## Usage

```bash
# Auto-detect hardware, pick best model, download, and chat
hivebear quickstart

# Step by step
hivebear profile              # What can my hardware handle?
hivebear recommend            # What models should I run?
hivebear install llama-3.1-8b # Download a model
hivebear run llama-3.1-8b     # Chat with it

# Persistent API server (Ollama + OpenAI compatible)
hivebear serve

# Search and manage models
hivebear search "code assistant"
hivebear list
hivebear remove llama-3.1-8b

# Distributed inference across devices
hivebear mesh start --port 7878
hivebear mesh run llama-3.1-70b --prompt "Explain quantum computing"
```

### API Server

`hivebear serve` starts an API server compatible with both Ollama and OpenAI clients:

```bash
hivebear serve

# Works with any OpenAI-compatible client
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.1-8b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

This means HiveBear works with VS Code extensions (Continue, Cody), JetBrains AI, Open WebUI, and anything else that speaks the OpenAI or Ollama protocol.

### Desktop App

Download from [Releases](https://github.com/BeckhamLabsLLC/HiveBear/releases) or build from source:

```bash
cd apps/desktop && npm install && cargo tauri dev
```

## Hardware Support

| Device | RAM | What Runs Well |
|--------|-----|----------------|
| Raspberry Pi 5 | 8 GB | TinyLlama 1.1B, Phi-2 2.7B |
| Older laptop | 8 GB | Llama 3.1 8B (Q4), Mistral 7B (Q4) |
| Gaming PC | 16 GB | Llama 3.1 8B (Q8), CodeLlama 13B (Q4) |
| Workstation | 32+ GB | Llama 3.1 70B (Q4), Mixtral 8x7B |
| Multi-device mesh | Any | Models too large for any single machine |

GPU acceleration is automatic when available (CUDA, Metal, Vulkan, WebGPU).

## Architecture

Rust workspace with 8 crates:

```
hivebear-core          Hardware profiling, model recommendations, config
hivebear-inference     Multi-engine inference orchestrator (llama.cpp, Candle)
hivebear-registry      Model search, download, conversion, storage
hivebear-mesh          P2P distributed inference over QUIC
hivebear-persistence   Conversation history (SQLite)
hivebear-cli           CLI + API server (Ollama + OpenAI compatible)
hivebear-web           WASM bridge for browser inference
apps/desktop           Tauri desktop app (Rust backend + React frontend)
```

## Feature Comparison

| | HiveBear | Ollama | LM Studio | Jan.ai |
|--|----------|--------|-----------|--------|
| Auto hardware profiling | Yes | -- | -- | -- |
| Smart model recommendation | Yes | -- | -- | -- |
| Multi-engine (llama.cpp + Candle) | Yes | llama.cpp | llama.cpp | llama.cpp |
| P2P distributed inference | Yes | -- | -- | -- |
| Ollama-compatible API | Yes | Yes | -- | -- |
| OpenAI-compatible API | Yes | Yes | Yes | Yes |
| Native desktop app | Yes | -- | Yes | Yes |
| Browser inference (WASM) | Yes | -- | -- | -- |
| Mobile (Android) | Yes | -- | -- | -- |
| License | MIT | MIT | Proprietary | AGPL |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Good first issues:
- Add GPU bandwidth entries to the [hardware database](crates/hivebear-core/src/recommender/scoring.rs)
- Add models to the [recommendation database](crates/hivebear-core/src/recommender/model_db.rs)
- Improve CLI output formatting

## License

MIT. See [LICENSE](LICENSE).
