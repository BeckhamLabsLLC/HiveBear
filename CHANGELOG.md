# Changelog

All notable changes to HiveBear are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.6] - 2026-09-20

The mesh did not work before this release. Several independent defects each
prevented it on their own; they are all fixed, and there are tests covering
the specific failures.

### Fixed — the mesh

- **`hivebear contribute` failed on every machine.** It asked for model ids
  with a quantisation suffix (`phi-3-mini-3.8b-q4_k_m`), and three that are
  not in the catalogue at all. Model resolution is exact, so every tier
  failed, nobody could become a peer, and the coordinator reported zero
  peers. This was the root cause of "the mesh is empty".
- **A node could hold exactly one peer connection.** Every TLS connection
  used the same server name, and certificate pinning keys on that name, so
  the second distinct peer was rejected as a possible MITM.
- **`hivebear mesh run` never reached the mesh.** It dialled a transport that
  had never called `listen()`, so it failed every time and silently fell back
  to local inference.
- **Port 7878 was bound twice.** The background node grabbed it before
  `mesh start` and `contribute` tried to, so both failed with "address
  already in use" against a node the same process had just started.
- **Registration reported success when the coordinator was unreachable**, so
  the CLI and desktop both said "Connected to Hive" while connected to
  nothing. Registration state is now tracked separately and retried.
- **Pipeline-parallel inference could not complete.** Workers sent their
  output back to the sender instead of forwarding it, and never emitted
  logits, so the initiator waited forever. Embedding and sampling were also
  smuggled through the wrong method using out-of-band type tags, so neither
  actually happened.
- **Concurrent tasks stole each other's messages** from one shared queue, and
  discarded what they took. Inbound messages are now routed per session.
- **NAT discovery measured the wrong socket**, advertising a port nothing was
  listening on, and hole punching never told the other side to dial.

### Added

- **Layer splitting.** A model too large for one machine now runs across
  peers, with the initiator serving the first stage. Layer counts come from
  the model file instead of a hardcoded guess.
- **TURN relay support** for peers behind symmetric NATs, which hole punching
  cannot reach. Requires a relay to be deployed; see `deploy/turn`.
- **Verification that means something.** Workers now recompute a challenged
  input and report the hash of what they produced. Previously a challenge
  carried only a hash of the input — unanswerable — so the responder simply
  asserted it had passed, and peer reputation never moved.

### Fixed — desktop and install

- The **auto-updater never worked**: the app was not built with updater
  artifacts, so the update manifest was empty and every install was frozen at
  the version it shipped with. If you are on 0.1.5, this is the last update
  you will need to install by hand.
- The app could **die during startup with no window and no error**. A corrupt
  model registry or chat database now repairs itself instead.
- `uninstall.sh` **left everything behind on macOS** — it looked in Linux
  locations, so config and the whole model cache survived.
- `install.sh` could **reject a valid download** through an over-broad
  checksum match.
- The Docker instructions were removed: those images have never been
  published, because the build has failed on every release.

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
