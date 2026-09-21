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

## Crash Reporting and Data Handling

Official builds send anonymous crash reports (see
[README](README.md#crash-reports) for how to turn this off). What leaves the
machine is deliberately narrow, and we treat a leak here as a security bug:

- **Sent:** the error and stack trace, HiveBear version, OS and CPU
  architecture, and a random per-install identifier.
- **Never sent:** prompts, chat history, model or file contents, email
  addresses, passwords, IP addresses, cloud provider API keys, or the mesh
  Ed25519 identity.

Two layers enforce this. Errors are redacted before they are sent —
home-directory paths have the username replaced, and known API-key and email
patterns are stripped (`redact_sensitive` in `hivebear-core`, with tests). The
reporting client is additionally configured never to collect request bodies,
headers, or local variables.

Builds from source have no reporting endpoint compiled in and cannot report at
all, regardless of configuration.

**If you find a way to make HiveBear transmit data from the lists above, please
report it under this policy.** We consider that a vulnerability, not a bug.

## Security Practices

- Dependencies are audited via [cargo-deny](https://github.com/EmbarkStudios/cargo-deny) on every CI run
- NPM dependencies are audited at the critical level
- Docker images run as a non-root user
- The install script verifies SHA256 checksums before executing downloaded binaries
