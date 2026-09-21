# Changelog

All notable changes to HiveBear are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.8] - 2026-09-21

A node can now register with the coordinator. None ever could before, so the
mesh had no members at all — the peer count has been zero since March. Android
also runs models on the device instead of only through a remote peer.

### Fixed — the mesh

- **No client could register.** `node_id` was serialized with
  `serialize_bytes`, and JSON has no byte type, so it went out as an array of
  32 numbers while the coordinator declares `node_id` as a string and
  validates it with `hex::decode`. Every `/register` and `/heartbeat` came
  back `422 Unprocessable Entity`. It now encodes as hex for human-readable
  formats and keeps the compact byte form for bincode, which is what the mesh
  wire protocol uses.
- **Registration carried no proof of key.** `/register` requires a `signature`
  over `register:{node_id}:{timestamp}` and rejects anything without one, but
  the discovery client held no signing key, so it could not have produced one
  even with the encoding right. It now signs with the node identity.
- **Registered nodes still saw an empty mesh.** `GET /peers` authenticates the
  caller, and peer discovery was the one request that did not send the
  registration token. The `401` was folded into an empty result, so a broken
  mesh was indistinguishable from an idle one.

### Added

- **On-device inference on Android.** llama.cpp now compiles into the APK for
  arm64 and x86_64, so a phone runs models locally rather than only relaying
  to a peer. Candle stays out on Android — its ARM FP16 build issues are
  unchanged — so it is excluded by target rather than by feature.
- **Crash reporting**, off unless you turn it on, for the CLI, the desktop app,
  its webview, and Android. The DSN is compiled in only for official release
  builds, so anything built from source cannot report regardless of config.
  See `docs/telemetry.md` for exactly what is sent.

### Fixed — Android

- **The app died moments after launch.** Mesh auto-start called `tokio::spawn`
  from a thread with no runtime entered — desktop happens to have one there,
  Android does not — and the panic unwound across the FFI boundary into
  `SIGABRT`. The guard around it tested for `Err`, which a panic never
  produces. It now spawns through Tauri's runtime, and the mesh returns an
  error instead of panicking, so a library can no longer abort its host.
- **The APK was never installable.** The release build had no signing config,
  so it came out unsigned and Android refused it. Release builds are signed
  when a keystore is configured, and fall back to the debug key otherwise
  rather than producing something unusable.
- **The Android build had never once succeeded.** The SDK setup asked for the
  obsolete `tools` package, and the Rust targets were added to the wrong
  toolchain, so it failed before reaching a build.

### Fixed — build and packaging

- **The Docker images build.** With libclang added, the build reached the
  runtime stage and failed there instead: `chown: cannot access '/data'`.
  `VOLUME /data` does not create the directory, and anything written to a
  volume path after `VOLUME` is declared is discarded, so the image had never
  got this far to show it. The user and data directory are now created before
  the volume is declared. Verified locally: the image runs as the non-root
  `hivebear` user with `/data` owned by it and writable. A package pushed to
  ghcr for the first time is private, so it must be made public before
  `docker pull` works for anyone.
- Docker images are built on every change to `main` now, not only when a
  release is tagged. Both faults above had accumulated behind that blind spot.
- `cargo install tauri-cli` uses `--force`. The cargo cache restores
  `~/.cargo/bin`, so every run after a successful one failed with "binary
  `cargo-tauri` already exists in destination" — including the release job.
- The Homebrew formula carries real checksums instead of placeholders, and a
  version check fails the build when the four files that declare the version
  disagree. The formula had been stuck at 0.1.3 while everything else moved.
- All 11 npm advisories in the desktop app are cleared, including a
  react-router RCE.

## [0.1.7] - 2026-09-20

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
- **The Docker build gets further, but still does not publish an image.** The
  first cause is fixed: `llama-cpp-sys-2` generates its bindings with bindgen,
  which needs libclang, and the slim Rust and CUDA base images do not ship it
  (the GitHub runners do, which is why ordinary CI stayed green). The Rust
  version was bumped three times chasing this and never touched the cause.
  A second fault was hiding behind it and is fixed for the next release.
- **`cargo install` did not compile.** It ignores `Cargo.lock`, so it picked
  up llama-cpp-2 0.1.156, which added a parameter to `LlamaSampler::penalties`
  in a patch release. Both llama-cpp crates are now pinned, and the README's
  install command passes `--locked`.

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
