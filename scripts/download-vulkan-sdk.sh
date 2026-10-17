#!/bin/bash
# Fail if the version has drifted between the places that declare it.
#
# The workspace Cargo.toml is the source of truth. Three other files repeat it by
# hand, and there was no check, so they drifted — the Homebrew formula sat at
# 0.1.3 while everything else was 0.1.7.
#
# This matters more now than it did: Sentry keys release health and "which
# release introduced this" off these strings. If the Rust binary reports
# 0.1.7 and the webview reports 0.1.6, the two halves of the same app look like
# different releases and every crash-free-rate number is wrong.
set -euo pipefail

VERSION=$1
echo "DOWNLOAD : vulkansdk-linux-x86_64-${VERSION}.tar.xz"
wget "https://sdk.lunarg.com/sdk/download/${VERSION}/linux/vulkansdk-linux-x86_64-${VERSION}.tar.xz" -O vulkansdk-linux.tar.xz
