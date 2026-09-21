#!/bin/bash
# Regenerate packaging/homebrew/hivebear.rb from a published release.
#
# The formula shipped with PLACEHOLDER_* checksums for four releases, which meant
# `brew install` could never have worked — the version being stale was the more
# visible problem but the less serious one. This pulls the real values from the
# release's own SHA256SUMS.txt so the formula cannot silently rot again.
#
# Usage:
#   scripts/update-homebrew-formula.sh            # uses the workspace version
#   scripts/update-homebrew-formula.sh v0.1.7     # or an explicit tag
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

FORMULA="packaging/homebrew/hivebear.rb"

if [[ $# -ge 1 ]]; then
    TAG="$1"
else
    version="$(awk '/^\[workspace\.package\]/{f=1} f && /^version[[:space:]]*=/{gsub(/[",]/,"",$3); print $3; exit}' Cargo.toml)"
    TAG="v${version}"
fi
VERSION="${TAG#v}"

command -v gh >/dev/null || { echo "gh CLI is required" >&2; exit 1; }

workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

echo "Fetching checksums for ${TAG}"
gh release download "$TAG" --pattern "SHA256SUMS.txt" --dir "$workdir" --clobber

sums="$workdir/SHA256SUMS.txt"

# Look up one artifact's checksum, failing loudly rather than writing a
# placeholder — a wrong or missing sha256 makes `brew install` fail for everyone.
lookup() {
    local artifact="$1" sum
    sum="$(awk -v a="$artifact" '$2 == a { print $1 }' "$sums")"
    if [[ -z "$sum" ]]; then
        echo "no checksum for ${artifact} in ${TAG}'s SHA256SUMS.txt" >&2
        exit 1
    fi
    printf '%s' "$sum"
}

ARM_MAC="$(lookup hivebear-aarch64-apple-darwin.tar.gz)"
X86_MAC="$(lookup hivebear-x86_64-apple-darwin.tar.gz)"
ARM_LINUX="$(lookup hivebear-aarch64-unknown-linux-gnu.tar.gz)"
X86_LINUX="$(lookup hivebear-x86_64-unknown-linux-gnu.tar.gz)"

cat > "$FORMULA" <<EOF
class Hivebear < Formula
  desc "AI that fits your machine — run LLMs on any device regardless of GPU"
  homepage "https://github.com/BeckhamLabsLLC/HiveBear"
  version "${VERSION}"
  license "MIT"

  on_macos do
    if Hardware::CPU.arm?
      url "https://github.com/BeckhamLabsLLC/HiveBear/releases/download/v#{version}/hivebear-aarch64-apple-darwin.tar.gz"
      sha256 "${ARM_MAC}"
    else
      url "https://github.com/BeckhamLabsLLC/HiveBear/releases/download/v#{version}/hivebear-x86_64-apple-darwin.tar.gz"
      sha256 "${X86_MAC}"
    end
  end

  on_linux do
    if Hardware::CPU.arm?
      url "https://github.com/BeckhamLabsLLC/HiveBear/releases/download/v#{version}/hivebear-aarch64-unknown-linux-gnu.tar.gz"
      sha256 "${ARM_LINUX}"
    else
      url "https://github.com/BeckhamLabsLLC/HiveBear/releases/download/v#{version}/hivebear-x86_64-unknown-linux-gnu.tar.gz"
      sha256 "${X86_LINUX}"
    end
  end

  def install
    bin.install "hivebear"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/hivebear --version")
  end

  def caveats
    <<~EOS
      Get started with HiveBear:

        hivebear quickstart

      This will profile your hardware, recommend the best model,
      download it, and start an interactive chat session.
    EOS
  end
end
EOF

echo "Updated ${FORMULA} to ${VERSION}"
