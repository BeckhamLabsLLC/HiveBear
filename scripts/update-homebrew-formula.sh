#!/bin/bash
# Regenerate the package-manager files from a published release:
#   packaging/homebrew/hivebear.rb   (tap: BeckhamLabsLLC/homebrew-hivebear, Formula/hivebear.rb)
#   packaging/scoop/hivebear.json    (bucket: BeckhamLabsLLC/scoop-hivebear, bucket/hivebear.json)
#   packaging/aur/PKGBUILD
#
# release.yml runs this on every stable tag and pushes the first two to the tap
# and bucket repos. Run it by hand when that job was skipped (no
# TAP_GITHUB_TOKEN) and copy the files over.
#
# The formula shipped with PLACEHOLDER_* checksums for four releases, which meant
# `brew install` could never have worked — the version being stale was the more
# visible problem but the less serious one. This pulls the real values from the
# release's own SHA256SUMS.txt so the formula cannot silently rot again.
#
# Usage:
#   scripts/update-homebrew-formula.sh            # uses the workspace version
#   scripts/update-homebrew-formula.sh v0.1.7     # or an explicit tag
#   SHA256SUMS_FILE=path/SHA256SUMS.txt scripts/update-homebrew-formula.sh v0.1.7
#                                                 # use a local copy, no gh needed
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

FORMULA="packaging/homebrew/hivebear.rb"
SCOOP="packaging/scoop/hivebear.json"
PKGBUILD="packaging/aur/PKGBUILD"

if [[ $# -ge 1 ]]; then
    TAG="$1"
else
    version="$(awk '/^\[workspace\.package\]/{f=1} f && /^version[[:space:]]*=/{gsub(/[",]/,"",$3); print $3; exit}' Cargo.toml)"
    TAG="v${version}"
fi
VERSION="${TAG#v}"

if [[ -n "${SHA256SUMS_FILE:-}" ]]; then
    sums="$SHA256SUMS_FILE"
else
    command -v gh >/dev/null || { echo "gh CLI is required" >&2; exit 1; }

    workdir="$(mktemp -d)"
    trap 'rm -rf "$workdir"' EXIT

    echo "Fetching checksums for ${TAG}"
    gh release download "$TAG" --repo BeckhamLabsLLC/HiveBear --pattern "SHA256SUMS.txt" --dir "$workdir" --clobber
    sums="$workdir/SHA256SUMS.txt"
fi

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
X86_WINDOWS="$(lookup hivebear-x86_64-pc-windows-msvc.zip)"

mkdir -p "$(dirname "$FORMULA")" "$(dirname "$SCOOP")" "$(dirname "$PKGBUILD")"

cat > "$FORMULA" <<EOF
class Hivebear < Formula
  desc "Run local AI models, with picks matched to your hardware"
  homepage "https://hivebear.com"
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

# Scoop's autoupdate block lets `scoop` users (and Scoop's own excavator) pick
# up a release even if this file falls behind.
cat > "$SCOOP" <<EOF
{
    "version": "${VERSION}",
    "description": "Run local AI models, with picks matched to your hardware",
    "homepage": "https://hivebear.com",
    "license": "MIT",
    "architecture": {
        "64bit": {
            "url": "https://github.com/BeckhamLabsLLC/HiveBear/releases/download/v${VERSION}/hivebear-x86_64-pc-windows-msvc.zip",
            "hash": "${X86_WINDOWS}"
        }
    },
    "bin": "hivebear.exe",
    "checkver": {
        "github": "https://github.com/BeckhamLabsLLC/HiveBear"
    },
    "autoupdate": {
        "architecture": {
            "64bit": {
                "url": "https://github.com/BeckhamLabsLLC/HiveBear/releases/download/v\$version/hivebear-x86_64-pc-windows-msvc.zip"
            }
        },
        "hash": {
            "url": "https://github.com/BeckhamLabsLLC/HiveBear/releases/download/v\$version/SHA256SUMS.txt"
        }
    }
}
EOF

cat > "$PKGBUILD" <<EOF
# Maintainer: BeckhamLabs <hello@beckhamlabs.com>
pkgname=hivebear-bin
pkgver=${VERSION}
pkgrel=1
pkgdesc="Run local AI models, with picks matched to your hardware"
arch=('x86_64' 'aarch64')
url="https://hivebear.com"
license=('MIT')
provides=('hivebear')
conflicts=('hivebear')

source_x86_64=("https://github.com/BeckhamLabsLLC/HiveBear/releases/download/v\${pkgver}/hivebear-x86_64-unknown-linux-gnu.tar.gz")
source_aarch64=("https://github.com/BeckhamLabsLLC/HiveBear/releases/download/v\${pkgver}/hivebear-aarch64-unknown-linux-gnu.tar.gz")

sha256sums_x86_64=('${X86_LINUX}')
sha256sums_aarch64=('${ARM_LINUX}')

package() {
    install -Dm755 hivebear "\${pkgdir}/usr/bin/hivebear"
}
EOF

echo "Updated ${FORMULA}, ${SCOOP} and ${PKGBUILD} to ${VERSION}"
