<p align="center">
  <img src="assets/logo-readme.png" alt="HiveBear" width="120" />
</p>

<h1 align="center">HiveBear</h1>

<p align="center">
  <strong>The world's largest peer-to-peer AI network.</strong><br>
  Every device is a node. Every node makes the network smarter.
</p>

<p align="center">
  <a href="https://github.com/BeckhamLabsLLC/HiveBear/actions"><img src="https://github.com/BeckhamLabsLLC/HiveBear/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://github.com/BeckhamLabsLLC/HiveBear/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License" /></a>
  <a href="https://github.com/BeckhamLabsLLC/HiveBear/releases/latest"><img src="https://img.shields.io/github/v/release/BeckhamLabsLLC/HiveBear?label=release" alt="Latest Release" /></a>
</p>

---

## Why HiveBear Exists

There are billions of devices sitting idle right now — laptops, desktops, gaming PCs, workstations, even Raspberry Pis — with CPUs and GPUs doing nothing. Meanwhile, running AI costs a fortune in cloud compute, and access is controlled by a handful of companies.

HiveBear connects these idle devices into a single distributed AI network. When you join the mesh, your hardware contributes to a collective compute pool. When you need to run a model that's too large for your machine, the mesh splits it across multiple devices automatically. No central server. No cloud bill. No data leaving the network.

The goal is simple: **build a global P2P mesh where anyone can run any AI model, regardless of what hardware they own, by pooling compute with everyone else.**

## How It Works

```
   You (8GB laptop)          Friend (16GB desktop)        Mesh peer (GPU workstation)
        |                           |                              |
        +------------- QUIC/TLS encrypted mesh ---------------+
                                    |
                          HiveBear Mesh Network
                                    |
                     Distributed inference: 70B model
                     split across all three devices
```

1. **Install HiveBear** on any device
2. **Join the mesh** — your device auto-profiles its hardware and advertises its capabilities
3. **Run any model** — if it fits locally, it runs locally. If it doesn't, HiveBear distributes the model layers across mesh peers automatically
4. **Contribute idle compute** — when you're not using your device, it helps others run their models

```bash
# Join the global mesh
hivebear mesh start

# Run a 70B model you couldn't run alone
hivebear mesh run llama-3.1-70b --prompt "Explain quantum computing"

# See who's connected
hivebear mesh status
```

The mesh uses QUIC transport with TLS encryption. Inference is distributed using pipeline parallelism — each device holds a subset of model layers and forwards activations to the next peer. No raw model weights or user prompts are exposed to other nodes.

## It Also Works Standalone

Even without the mesh, HiveBear is a complete local AI runtime. It profiles your hardware, picks the best model and quantization automatically, and runs it:

```bash
# One command: profile hardware, pick best model, download, chat
hivebear quickstart

# Or use it as an Ollama-compatible API server
hivebear serve
```

The `serve` command is a drop-in replacement for `ollama serve` — same port (11434), same API. Your existing tools, IDE extensions (Continue, Cody), and scripts work without changes. When a model is too large for your hardware, it automatically overflows to the mesh.

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

## What Your Hardware Can Run

HiveBear auto-detects and adapts to whatever you have:

| Device | RAM | Solo | With Mesh |
|--------|-----|------|-----------|
| Raspberry Pi 5 | 8 GB | TinyLlama 1.1B, Phi-2 2.7B | Contribute layers to larger models |
| Old laptop | 8 GB | Llama 3.1 8B (Q4), Mistral 7B | Help run 13B-30B models |
| Gaming PC | 16 GB | Llama 3.1 8B (Q8), CodeLlama 13B | Help run 70B+ models |
| Workstation | 32+ GB | Llama 3.1 70B (Q4), Mixtral 8x7B | Run anything |

GPU acceleration is automatic (CUDA, Metal, Vulkan, WebGPU).

## Architecture

Rust workspace, 8 crates:

```
hivebear-core          Hardware profiling, model recommendations
hivebear-inference     Multi-engine inference (llama.cpp, Candle)
hivebear-mesh          P2P distributed inference over QUIC/TLS
hivebear-registry      Model search, download, conversion (HuggingFace)
hivebear-persistence   Conversation history (SQLite)
hivebear-cli           CLI + API server (Ollama + OpenAI compatible)
hivebear-web           WASM bridge for browser inference
apps/desktop           Tauri desktop app (Rust + React)
```

## CLI Reference

```
hivebear quickstart                    Profile -> recommend -> install -> chat
hivebear serve                         Start Ollama + OpenAI compatible API server
hivebear profile                       Show hardware capabilities
hivebear recommend                     Get model recommendations for your hardware

hivebear mesh start [--port 7878]      Join the P2P mesh network
hivebear mesh status                   Show connected peers and network capacity
hivebear mesh run <model>              Distributed inference across the mesh
hivebear mesh stop                     Leave the mesh

hivebear search <query>                Search models on HuggingFace
hivebear install <model>               Download a model
hivebear run <model>                   Local inference (chat, --api, or --prompt)
hivebear list / remove / storage       Manage installed models
```

## Platforms

- **CLI**: Linux, macOS, Windows, ARM (Raspberry Pi, Apple Silicon)
- **Desktop app**: Linux (.deb, .AppImage), macOS (.dmg), Windows (.msi, .exe)
- **Mobile**: Android (.apk)
- **Browser**: WASM + WebGPU
- **Docker**: CPU and CUDA images

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The most impactful contributions right now are around the mesh networking layer and hardware profiling coverage.

## License

MIT. See [LICENSE](LICENSE).
