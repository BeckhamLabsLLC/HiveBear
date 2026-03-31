# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

If you discover a security vulnerability in HiveBear, please report it responsibly.

**Do not open a public issue.** Instead, email **security@beckhamlabs.com** with:

- A description of the vulnerability
- Steps to reproduce
- Affected versions
- Any potential impact

We will acknowledge your report within **48 hours** and aim to provide a fix or mitigation within **7 days** for critical issues.

## Scope

This policy covers:

- The `hivebear-cli` binary and all workspace crates
- The Tauri desktop application
- The WASM web bridge
- Docker images published to `ghcr.io/beckhamlabsllc/hivebear`
- The install script (`install.sh`)

Out of scope:

- Third-party models downloaded via the registry
- The HiveBear P2P mesh network traffic between peers (encrypted via QUIC/TLS, but peer-contributed content is untrusted)

## Security Practices

- Dependencies are audited via [cargo-deny](https://github.com/EmbarkStudios/cargo-deny) on every CI run
- NPM dependencies are audited at the critical level
- Docker images run as a non-root user
- The install script verifies SHA256 checksums before executing downloaded binaries
