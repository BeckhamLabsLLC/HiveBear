#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Building HiveBear WASM module ==="
wasm-pack build --target web

WASM_SIZE=$(du -sh pkg/hivebear_web_bg.wasm | cut -f1)
echo ""
echo "=== Build complete ==="
echo "  WASM binary: ${WASM_SIZE}"
echo "  Output: web/pkg/"
