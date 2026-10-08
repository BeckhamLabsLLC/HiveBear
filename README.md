<p align="center">
  <img src="assets/logo-readme.png" alt="HiveBear" width="120" />
</p>

<h1 align="center">HiveBear</h1>

<p align="center">
  <strong>Run local AI models on your own machine in one click.</strong><br>
  HiveBear picks the right model for your hardware, and a community speed leaderboard shows what every machine can do.<br>
  As the hive grows, pool machines with others over a P2P mesh.
</p>

<p align="center">
  <a href="https://github.com/BeckhamLabsLLC/HiveBear/actions"><img src="https://github.com/BeckhamLabsLLC/HiveBear/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://github.com/BeckhamLabsLLC/HiveBear/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License" /></a>
  <a href="https://github.com/BeckhamLabsLLC/HiveBear/releases/latest"><img src="https://img.shields.io/github/v/release/BeckhamLabsLLC/HiveBear?label=release" alt="Latest Release" /></a>
</p>

<p align="center">
  <a href="https://hivebear.com/download"><img src="https://img.shields.io/badge/macOS-Download-111111?style=for-the-badge&logo=apple&logoColor=white" alt="Download for macOS" /></a>
  <a href="https://hivebear.com/download"><img src="https://img.shields.io/badge/Windows-Download-0078D4?style=for-the-badge&logo=windows&logoColor=white" alt="Download for Windows" /></a>
  <a href="https://hivebear.com/download"><img src="https://img.shields.io/badge/Linux-Download-E95420?style=for-the-badge&logo=linux&logoColor=white" alt="Download for Linux" /></a>
</p>

<!-- TODO: add a screenshot or short GIF of the desktop app here (chat + model picker).
     Nothing in assets/ shows the app yet; the pawpaw_* images are mascot art. -->

---

## What it does

- **Picks a model that fits.** HiveBear profiles your CPU, GPU and memory, then recommends a model and quantization that will actually run well on that machine, and downloads it.
- **Runs it locally.** Chat in the desktop app or the terminal. Your prompts and chats stay on your machine.
- **Drops in for Ollama.** `hivebear serve` speaks the Ollama and OpenAI APIs on port 11434, so Continue, Open WebUI and your own scripts work unchanged.
- **Shows how fast your machine really is.** Benchmark a model and share the result to the community leaderboard on [hivebear.com](https://hivebear.com), so you can see what a given GPU or laptop does before you buy or download.
- **Pools machines (experimental).** A P2P mesh that splits a model too big for one machine across several. It works, but it needs peers, and the network is young; see [Mesh](#mesh-experimental).

## Quickstart

**Desktop app:** download it from [hivebear.com/download](https://hivebear.com/download) (macOS, Windows, Linux). It walks you through picking and installing a model on first launch.

**CLI**, in two commands:

```bash
curl -fsSL https://hivebear.com/install.sh | sh
hivebear quickstart     # profile hardware -> pick a model -> download -> chat
```

On Windows (PowerShell):

```powershell
irm https://hivebear.com/install.ps1 | iex
hivebear quickstart
```

Then, if you want an API server:

```bash
hivebear serve          # Ollama + OpenAI compatible, http://localhost:11434
```

## Install

```bash
# CLI (Linux, macOS)
curl -fsSL https://hivebear.com/install.sh | sh

# Desktop app without any Gatekeeper prompt (macOS), or .deb/AppImage (Linux)
curl -fsSL https://hivebear.com/install.sh | sh -s -- --desktop

# Homebrew (macOS, Linux)
brew install BeckhamLabsLLC/hivebear/hivebear

# Scoop (Windows)
scoop bucket add hivebear https://github.com/BeckhamLabsLLC/scoop-hivebear
scoop install hivebear

# Build from source
cargo install --git https://github.com/BeckhamLabsLLC/HiveBear hivebear-cli --locked
# --locked builds the dependency versions this release was tested with.
# Without it cargo re-resolves every dependency to the newest compatible
# release, which is not what we build or test.
```

```powershell
# Windows: CLI, plus the desktop app with -Desktop
irm https://hivebear.com/install.ps1 | iex
& ([scriptblock]::Create((irm https://hivebear.com/install.ps1))) -Desktop
```

The Linux binaries need glibc 2.35 or newer (Ubuntu 22.04+, Debian 12+, Fedora 36+).

### Installing unsigned builds

HiveBear is not signed with a paid Apple or Microsoft certificate, so the
operating system warns the first time you open a downloaded installer. The
builds are the ones in [GitHub Releases](https://github.com/BeckhamLabsLLC/HiveBear/releases),
with SHA-256 checksums alongside.

**macOS.** The app is ad-hoc signed but not notarized. Any one of these works:

- Open it once, then go to **System Settings → Privacy & Security** and click **Open Anyway**.
- Or clear the quarantine flag after dragging it to Applications:
  `xattr -dr com.apple.quarantine /Applications/HiveBear.app`
- Or skip the prompt entirely by installing with the script, which downloads
  with `curl` (no quarantine flag) and copies HiveBear.app into Applications:
  `curl -fsSL https://hivebear.com/install.sh | sh -s -- --desktop`

**Windows.** SmartScreen shows "Windows protected your PC". Click
**More info → Run anyway**. Installing with `install.ps1` avoids the prompt.

> **Docker images are not published yet.** The `build-docker` jobs failed on
> every release up to 0.1.6, because the build images lacked the libclang that
> llama-cpp-sys-2's bindgen step needs. That is fixed, but a package pushed to
> ghcr for the first time is private, so `ghcr.io/beckhamlabsllc/hivebear` is
> not pullable until it is made public. These instructions will return then.

## What Your Hardware Can Run

HiveBear auto-detects and adapts to whatever you have:

| Device | RAM | Runs well locally |
|--------|-----|-------------------|
| Raspberry Pi 5 | 8 GB | TinyLlama 1.1B, Phi-2 2.7B |
| Old laptop | 8 GB | Llama 3.1 8B (Q4), Mistral 7B |
| Gaming PC | 16 GB | Llama 3.1 8B (Q8), CodeLlama 13B |
| Workstation | 32+ GB | Llama 3.1 70B (Q4), Mixtral 8x7B |

**GPU acceleration:** the prebuilt downloads use **Metal on Apple Silicon**
Macs. On Intel Macs, Windows and Linux they run on the **CPU** (llama.cpp, with
AVX/AVX2 where available). CUDA and Vulkan are supported when you build from
source with `--features llamacpp-cuda` or `--features llamacpp-vulkan`; prebuilt
GPU builds for Windows and Linux are planned.

## Mesh (experimental)

The long-term idea: idle laptops, desktops and GPUs pooled into one network,
so a model too large for any one machine is split across several. Each device
holds a slice of the model's layers and forwards activations to the next peer
over QUIC with TLS (pipeline parallelism).

```
   You (8GB laptop)          Friend (16GB desktop)        Mesh peer (GPU workstation)
        |                           |                              |
        +------------- QUIC/TLS encrypted mesh ---------------+
                                    |
                     Distributed inference: 70B model
                     split across all three devices
```

```bash
hivebear mesh start      # join the mesh and advertise this machine's capacity
hivebear mesh status     # peers and network capacity
hivebear mesh stop
```

Honest status: the network is young, so most of the time there are few or no
peers online, and `hivebear mesh run` still downloads the full model locally
before distributing it. Joining is opt-in; nothing is shared unless you start
the mesh. If you want to help make it real, the mesh layer is where
contributions matter most.

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

hivebear search <query>                Search models on HuggingFace
hivebear install <model>               Download a model
hivebear run <model>                   Local inference (chat, --api, or --prompt)
hivebear list / remove / storage       Manage installed models
hivebear benchmark --model <model>     Measure tokens/sec, optionally share to the leaderboard

hivebear mesh start [--port 7878]      Join the P2P mesh (experimental)
hivebear mesh status                   Show connected peers and network capacity
hivebear mesh run <model>              Distributed inference across the mesh (experimental)
hivebear mesh stop                     Leave the mesh
```

## Platforms

- **CLI**: Linux x86_64 and ARM64 (Raspberry Pi 5), macOS (Apple Silicon and Intel), Windows x64
- **Desktop app**: macOS (Apple Silicon and Intel .dmg), Windows (.msi, .exe), Linux x86_64 (.deb, .AppImage)
- **Mobile**: Android (.apk), early
- **Browser**: WASM build (`hivebear-web-wasm.tar.gz` on each release), CPU only for now
- **Docker**: Dockerfiles for CPU and CUDA are in the repo; images are not published yet (see above)

## Crash Reports

Official HiveBear builds send anonymous crash reports, so we hear about the
things that break on hardware we do not have. It is on by default and takes one
command to turn off:

```bash
# Any of these disables it permanently
export HIVEBEAR_TELEMETRY=0
export DO_NOT_TRACK=1            # honoured, as a matter of course
hivebear sentry-check            # show whether it is on, and why
```

In the desktop app it is a toggle under **Settings → Crash Reports**.

**What is sent:** the error and its stack trace, the HiveBear version, your OS
and CPU architecture, and a random identifier generated on your machine.

**What is never sent:** your prompts, chat history, model files, file contents,
email address, IP address, API keys, or mesh identity. File paths are stripped
of your username before they leave the machine.

The random identifier exists only so we can tell "one person hit this 400 times"
apart from "400 people hit it once". It is not derived from your hardware or
your mesh keypair, and it is not linked to anything else.

If you build HiveBear from source, there is no reporting endpoint compiled in at
all — a source build cannot report anything even if the setting says it should.
That includes every distro package built from this repository.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The most impactful contributions right now are around the mesh networking layer and hardware profiling coverage.

## License

MIT. See [LICENSE](LICENSE).
