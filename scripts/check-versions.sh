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

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Source of truth: [workspace.package] version in the root Cargo.toml.
expected="$(awk '/^\[workspace\.package\]/{f=1} f && /^version[[:space:]]*=/{gsub(/[",]/,"",$3); print $3; exit}' Cargo.toml)"

if [[ -z "$expected" ]]; then
    echo "could not read version from [workspace.package] in Cargo.toml" >&2
    exit 1
fi

echo "workspace version: $expected"

status=0

check() {
    local label="$1" file="$2" found="$3"
    if [[ "$found" != "$expected" ]]; then
        echo "MISMATCH  $label ($file): '$found' != '$expected'" >&2
        status=1
    else
        echo "ok        $label"
    fi
}

check "tauri.conf.json" "apps/desktop/src-tauri/tauri.conf.json" \
    "$(grep -m1 '"version"' apps/desktop/src-tauri/tauri.conf.json | sed -E 's/.*"version"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/')"

check "desktop package.json" "apps/desktop/package.json" \
    "$(grep -m1 '"version"' apps/desktop/package.json | sed -E 's/.*"version"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/')"

check "homebrew formula" "packaging/homebrew/hivebear.rb" \
    "$(grep -m1 '^  version ' packaging/homebrew/hivebear.rb | sed -E 's/.*version "([^"]+)".*/\1/')"

if [[ $status -ne 0 ]]; then
    echo >&2
    echo "Bump every file above to $expected before releasing." >&2
fi

exit $status
