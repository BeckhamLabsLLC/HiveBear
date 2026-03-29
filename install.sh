#!/bin/bash
set -euo pipefail

REPO="BeckhamLabsLLC/HiveBear"
INSTALL_DIR="$HOME/.local/bin"

main() {
    echo "HiveBear Installer"
    echo "================="
    echo ""

    # Detect OS
    OS="$(uname -s)"
    case "$OS" in
        Linux)  OS_NAME="linux" ;;
        Darwin) OS_NAME="macos" ;;
        CYGWIN*|MINGW*|MSYS*)
            echo "Error: Windows is not supported by this installer."
            echo "Please download the .msi installer from:"
            echo "  https://github.com/$REPO/releases/latest"
            exit 1
            ;;
        *)
            echo "Error: Unsupported operating system: $OS"
            exit 1
            ;;
    esac

    # Detect architecture
    ARCH="$(uname -m)"
    case "$ARCH" in
        x86_64|amd64)   ARCH_NAME="x86_64" ;;
        aarch64|arm64)   ARCH_NAME="aarch64" ;;
        *)
            echo "Error: Unsupported architecture: $ARCH"
            exit 1
            ;;
    esac

    # Map to Rust target triple
    case "${OS_NAME}-${ARCH_NAME}" in
        linux-x86_64)   TARGET="x86_64-unknown-linux-gnu" ;;
        linux-aarch64)  TARGET="aarch64-unknown-linux-gnu" ;;
        macos-x86_64)   TARGET="x86_64-apple-darwin" ;;
        macos-aarch64)  TARGET="aarch64-apple-darwin" ;;
        *)
            echo "Error: Unsupported platform: ${OS_NAME}-${ARCH_NAME}"
            exit 1
            ;;
    esac

    # Version: use HIVEBEAR_VERSION env var, or default to "latest"
    VERSION="${HIVEBEAR_VERSION:-latest}"
    if [ "$VERSION" = "latest" ]; then
        BASE_URL="https://github.com/$REPO/releases/latest/download"
    else
        BASE_URL="https://github.com/$REPO/releases/download/${VERSION}"
    fi

    DOWNLOAD_URL="${BASE_URL}/hivebear-${TARGET}.tar.gz"
    CHECKSUMS_URL="${BASE_URL}/SHA256SUMS.txt"
    FILENAME="hivebear-${TARGET}.tar.gz"

    echo "Detected:  $OS ($ARCH_NAME)"
    echo "Target:    $TARGET"
    echo "Version:   $VERSION"
    echo ""

    # Create temp directory
    TMPDIR="$(mktemp -d)"
    trap 'rm -rf "$TMPDIR"' EXIT

    # Download binary
    echo "Downloading hivebear..."
    download "$DOWNLOAD_URL" "$TMPDIR/$FILENAME"

    # Download checksums and verify
    echo "Verifying checksum..."
    download "$CHECKSUMS_URL" "$TMPDIR/SHA256SUMS.txt"
    verify_checksum "$TMPDIR/$FILENAME" "$TMPDIR/SHA256SUMS.txt" "$FILENAME"

    # Extract
    echo "Extracting..."
    tar xzf "$TMPDIR/$FILENAME" -C "$TMPDIR"

    # Install
    mkdir -p "$INSTALL_DIR"
    cp "$TMPDIR/hivebear" "$INSTALL_DIR/hivebear"
    chmod +x "$INSTALL_DIR/hivebear"
    echo "Installed hivebear to $INSTALL_DIR/hivebear"

    # Check if INSTALL_DIR is in PATH
    case ":$PATH:" in
        *":$INSTALL_DIR:"*)
            ;;
        *)
            echo ""
            echo "NOTE: $INSTALL_DIR is not in your PATH."
            echo "Add it by appending this to your shell profile (~/.bashrc, ~/.zshrc, etc.):"
            echo ""
            echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
            echo ""
            ;;
    esac

    # Verify installation
    if "$INSTALL_DIR/hivebear" --version >/dev/null 2>&1; then
        VERSION="$("$INSTALL_DIR/hivebear" --version)"
        echo "Verified: $VERSION"
    else
        echo "Warning: Could not verify installation. The binary may require additional dependencies."
    fi

    echo ""
    echo "Welcome to HiveBear! Get started by running:"
    echo ""
    echo "  hivebear quickstart"
    echo ""
}

download() {
    local url="$1"
    local output="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$url" -o "$output"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO "$output" "$url"
    else
        echo "Error: Neither curl nor wget found. Please install one and try again."
        exit 1
    fi
}

verify_checksum() {
    local file="$1"
    local checksums_file="$2"
    local filename="$3"

    # Extract expected hash for our file from SHA256SUMS.txt
    local expected
    expected="$(grep "$filename" "$checksums_file" | awk '{print $1}')"

    if [ -z "$expected" ]; then
        echo "Error: No checksum found for $filename in SHA256SUMS.txt. Aborting."
        exit 1
    fi

    # Compute actual hash (sha256sum on Linux, shasum on macOS)
    local actual
    if command -v sha256sum >/dev/null 2>&1; then
        actual="$(sha256sum "$file" | awk '{print $1}')"
    elif command -v shasum >/dev/null 2>&1; then
        actual="$(shasum -a 256 "$file" | awk '{print $1}')"
    else
        echo "Error: No sha256sum or shasum found. Cannot verify checksum. Aborting."
        exit 1
    fi

    if [ "$expected" != "$actual" ]; then
        echo "Error: Checksum verification failed!"
        echo "  Expected: $expected"
        echo "  Actual:   $actual"
        echo ""
        echo "The downloaded file may be corrupted or tampered with."
        echo "Please try again or download manually from:"
        echo "  https://github.com/$REPO/releases/latest"
        exit 1
    fi

    echo "Checksum OK"
}

main
