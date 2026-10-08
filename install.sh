#!/bin/sh
# HiveBear installer for Linux and macOS.
#
#   curl -fsSL https://hivebear.com/install.sh | sh                  # CLI
#   curl -fsSL https://hivebear.com/install.sh | sh -s -- --desktop   # desktop app
#   curl -fsSL https://hivebear.com/install.sh | sh -s -- --cli --desktop
#
# Environment:
#   HIVEBEAR_VERSION=v0.1.9         install a specific release (default: latest)
#   HIVEBEAR_INSTALL_DIR=DIR        where the CLI goes (default: ~/.local/bin)
#   HIVEBEAR_DESKTOP_FORMAT=appimage|deb   Linux desktop package (default: deb
#                                   when apt and root/sudo are available)
#
# Written for POSIX sh, not bash: hivebear.com tells people to pipe this into
# `sh`, which is dash on Debian and Ubuntu. dash before 0.5.12 rejects
# `set -o pipefail`, and the bash version of this script died on that line.
#
# Windows: use install.ps1 instead (irm https://hivebear.com/install.ps1 | iex).
set -eu

REPO="BeckhamLabsLLC/HiveBear"
INSTALL_DIR="${HIVEBEAR_INSTALL_DIR:-$HOME/.local/bin}"
# Linux release binaries are built on Ubuntu 22.04, so they need glibc 2.35+.
# Keep in step with the runner in .github/workflows/release.yml.
MIN_GLIBC="2.35"

WANT_CLI=""
WANT_DESKTOP=""

usage() {
    cat <<EOF
HiveBear installer

Usage: install.sh [--cli] [--desktop]

  --cli        Install the hivebear command-line tool (the default)
  --desktop    Install the HiveBear desktop app
               macOS: HiveBear.app into /Applications (or ~/Applications)
               Linux: the .deb (with apt) or the AppImage
  -h, --help   Show this help

Set HIVEBEAR_VERSION=vX.Y.Z to install a specific release.
EOF
}

main() {
    for arg in "$@"; do
        case "$arg" in
            --cli) WANT_CLI=1 ;;
            --desktop) WANT_DESKTOP=1 ;;
            -h|--help) usage; exit 0 ;;
            *) echo "Unknown option: $arg" >&2; usage >&2; exit 1 ;;
        esac
    done
    if [ -z "$WANT_CLI" ] && [ -z "$WANT_DESKTOP" ]; then
        WANT_CLI=1
    fi

    echo "HiveBear Installer"
    echo "=================="
    echo ""

    detect_platform

    VERSION="${HIVEBEAR_VERSION:-latest}"
    if [ "$VERSION" = "latest" ]; then
        BASE_URL="https://github.com/$REPO/releases/latest/download"
    else
        case "$VERSION" in v*) ;; *) VERSION="v$VERSION" ;; esac
        BASE_URL="https://github.com/$REPO/releases/download/${VERSION}"
    fi

    echo "Detected:  $OS ($ARCH_NAME)"
    echo "Version:   $VERSION"
    echo ""

    WORKDIR="$(mktemp -d)"
    trap 'rm -rf "$WORKDIR"' EXIT INT TERM

    if [ "$OS_NAME" = "linux" ]; then
        check_glibc
    fi

    if [ -n "$WANT_CLI" ]; then
        install_cli
    fi
    if [ -n "$WANT_DESKTOP" ]; then
        [ -n "$WANT_CLI" ] && echo ""
        case "$OS_NAME" in
            macos) install_desktop_macos ;;
            linux) install_desktop_linux ;;
        esac
    fi
}

detect_platform() {
    OS="$(uname -s)"
    case "$OS" in
        Linux)  OS_NAME="linux" ;;
        Darwin) OS_NAME="macos" ;;
        CYGWIN*|MINGW*|MSYS*)
            echo "Error: this installer is for Linux and macOS."
            echo "On Windows, run this in PowerShell instead:"
            echo ""
            echo "  irm https://hivebear.com/install.ps1 | iex"
            echo ""
            echo "or download the installer from https://hivebear.com/download"
            exit 1
            ;;
        *)
            echo "Error: Unsupported operating system: $OS"
            exit 1
            ;;
    esac

    ARCH="$(uname -m)"
    case "$ARCH" in
        x86_64|amd64)   ARCH_NAME="x86_64" ;;
        aarch64|arm64)  ARCH_NAME="aarch64" ;;
        *)
            echo "Error: Unsupported architecture: $ARCH"
            exit 1
            ;;
    esac

    # A shell running under Rosetta reports x86_64 on an Apple Silicon Mac.
    # The native build is faster and is the only one with Metal, so use it.
    if [ "$OS_NAME" = "macos" ] && [ "$ARCH_NAME" = "x86_64" ] \
        && [ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" = "1" ]; then
        ARCH_NAME="aarch64"
    fi

    case "${OS_NAME}-${ARCH_NAME}" in
        linux-x86_64)   TARGET="x86_64-unknown-linux-gnu" ;;
        linux-aarch64)  TARGET="aarch64-unknown-linux-gnu" ;;
        macos-x86_64)   TARGET="x86_64-apple-darwin" ;;
        macos-aarch64)  TARGET="aarch64-apple-darwin" ;;
    esac
}

# Print the system glibc version (e.g. "2.35"), or nothing if it is not glibc.
glibc_version() {
    if command -v getconf >/dev/null 2>&1; then
        v="$(getconf GNU_LIBC_VERSION 2>/dev/null | awk '{print $2}')" || v=""
        if [ -n "$v" ]; then echo "$v"; return 0; fi
    fi
    if command -v ldd >/dev/null 2>&1; then
        ldd --version 2>&1 | head -n1 | grep -i -e glibc -e 'gnu libc' \
            | grep -o '[0-9][0-9]*\.[0-9][0-9]*' | tail -n1
    fi
    return 0
}

# True if version $1 >= version $2 (dotted, numeric).
version_ge() {
    [ "$(printf '%s\n%s\n' "$1" "$2" | sort -t. -k1,1n -k2,2n -k3,3n | head -n1)" = "$2" ]
}

check_glibc() {
    GLIBC="$(glibc_version)"
    if [ -z "$GLIBC" ]; then
        if ldd --version 2>&1 | grep -qi musl; then
            echo "Error: this system uses musl libc (Alpine or similar)."
            echo "HiveBear's Linux builds need glibc ${MIN_GLIBC} or newer."
            echo "Build from source instead:"
            echo "  cargo install --git https://github.com/$REPO hivebear-cli --locked"
            exit 1
        fi
        echo "Warning: could not determine the glibc version. Continuing anyway."
        return 0
    fi
    if ! version_ge "$GLIBC" "$MIN_GLIBC"; then
        echo "Error: this system has glibc $GLIBC, but HiveBear's Linux builds need"
        echo "glibc $MIN_GLIBC or newer (Ubuntu 22.04+, Debian 12+, Fedora 36+)."
        echo ""
        echo "Options:"
        echo "  - Upgrade the distribution, or"
        echo "  - Build from source, which links against your own glibc:"
        echo "      cargo install --git https://github.com/$REPO hivebear-cli --locked"
        exit 1
    fi
}

install_cli() {
    FILENAME="hivebear-${TARGET}.tar.gz"
    echo "Installing the HiveBear CLI ($TARGET)"

    echo "Downloading $FILENAME..."
    download "${BASE_URL}/${FILENAME}" "$WORKDIR/$FILENAME"

    echo "Verifying checksum..."
    download "${BASE_URL}/SHA256SUMS.txt" "$WORKDIR/SHA256SUMS.txt"
    verify_checksum "$WORKDIR/$FILENAME" "$WORKDIR/SHA256SUMS.txt" "$FILENAME"

    echo "Extracting..."
    tar xzf "$WORKDIR/$FILENAME" -C "$WORKDIR"

    mkdir -p "$INSTALL_DIR"
    cp "$WORKDIR/hivebear" "$INSTALL_DIR/hivebear"
    chmod +x "$INSTALL_DIR/hivebear"
    echo "Installed hivebear to $INSTALL_DIR/hivebear"

    if out="$("$INSTALL_DIR/hivebear" --version 2>&1)"; then
        echo "Verified: $out"
    else
        explain_run_failure "$out"
    fi

    path_hint

    echo ""
    echo "Get started by running:"
    echo ""
    echo "  hivebear quickstart"
    echo ""
}

# The binary installed but would not start. Say why, as precisely as we can.
explain_run_failure() {
    out="$1"
    needed="$(printf '%s\n' "$out" | grep -o 'GLIBC_[0-9][0-9.]*' | sed 's/GLIBC_//' | sort -t. -k1,1n -k2,2n | tail -n1)"
    if [ -n "$needed" ]; then
        echo ""
        echo "Error: hivebear needs glibc $needed, but this system has glibc ${GLIBC:-unknown}."
        echo "Releases up to v0.1.8 were built against glibc 2.39; later ones need only $MIN_GLIBC."
        echo "Install a newer release, upgrade to a distribution with glibc $needed+, or build from source:"
        echo "  cargo install --git https://github.com/$REPO hivebear-cli --locked"
        exit 1
    fi
    # llama.cpp is built with OpenMP, so the binary links libgomp. Desktop
    # installs nearly always have it; minimal servers and containers do not.
    lib="$(printf '%s\n' "$out" | sed -n 's/.*error while loading shared libraries: \([^:]*\):.*/\1/p' | head -n1)"
    if [ -n "$lib" ]; then
        echo ""
        echo "Error: hivebear is installed but cannot start: the system library $lib is missing."
        case "$lib" in
            libgomp.so*)
                echo "Install it with one of:"
                echo "  sudo apt install libgomp1        # Debian, Ubuntu"
                echo "  sudo dnf install libgomp         # Fedora, RHEL"
                echo "  sudo pacman -S gcc-libs          # Arch"
                ;;
            *)
                echo "Install the package that provides $lib with your package manager."
                ;;
        esac
        echo "Then run: $INSTALL_DIR/hivebear --version"
        exit 1
    fi
    echo "Warning: the installed binary did not run. Its output was:"
    printf '%s\n' "$out" | sed 's/^/  /'
}

# If the CLI's directory is not on PATH, print the exact line for this shell.
path_hint() {
    case ":$PATH:" in
        *":$INSTALL_DIR:"*) return 0 ;;
    esac

    # Prefer $HOME-relative so the line still works if the profile is shared.
    case "$INSTALL_DIR" in
        "$HOME"/*) dir_expr="\$HOME/${INSTALL_DIR#"$HOME"/}" ;;
        *) dir_expr="$INSTALL_DIR" ;;
    esac

    shell_name="$(basename "${SHELL:-sh}")"
    echo ""
    echo "NOTE: $INSTALL_DIR is not on your PATH. To fix that, run:"
    echo ""
    case "$shell_name" in
        zsh)
            echo "  echo 'export PATH=\"$dir_expr:\$PATH\"' >> ~/.zshrc && source ~/.zshrc"
            ;;
        bash)
            # Literal "~" on purpose: this is text for the user to paste.
            # shellcheck disable=SC2088
            if [ "$OS_NAME" = "macos" ]; then rc="~/.bash_profile"; else rc="~/.bashrc"; fi
            echo "  echo 'export PATH=\"$dir_expr:\$PATH\"' >> $rc && source $rc"
            ;;
        fish)
            echo "  fish_add_path $INSTALL_DIR"
            ;;
        *)
            echo "  echo 'export PATH=\"$dir_expr:\$PATH\"' >> ~/.profile"
            echo ""
            echo "then log out and back in."
            ;;
    esac
    echo ""
    echo "Until then, run it as: $INSTALL_DIR/hivebear"
}

# ---------------------------------------------------------------------------
# Desktop app
# ---------------------------------------------------------------------------

# macOS: the updater bundle (HiveBear.app inside a .tar.gz) rather than the
# DMG, because there is nothing to mount. Files fetched with curl or wget get
# no com.apple.quarantine attribute, so Gatekeeper never asks: this is the
# no-warning route for an app without an Apple Developer ID.
install_desktop_macos() {
    if [ "$ARCH_NAME" = "aarch64" ]; then
        # HiveBear.app.tar.gz is the name releases before v0.1.9 used, when
        # only Apple Silicon had a desktop build.
        candidates="HiveBear_aarch64.app.tar.gz HiveBear.app.tar.gz"
    else
        candidates="HiveBear_x64.app.tar.gz"
    fi

    echo "Installing the HiveBear desktop app (macOS, $ARCH_NAME)"
    download "${BASE_URL}/SHA256SUMS-desktop.txt" "$WORKDIR/SHA256SUMS-desktop.txt"

    asset=""
    for c in $candidates; do
        if awk -v f="$c" '$2 == f || $2 == "*" f { found=1 } END { exit !found }' "$WORKDIR/SHA256SUMS-desktop.txt"; then
            asset="$c"
            break
        fi
    done
    if [ -z "$asset" ]; then
        echo "Error: release $VERSION has no desktop build for this Mac ($ARCH_NAME)."
        if [ "$ARCH_NAME" = "x86_64" ]; then
            echo "Intel Mac desktop builds start with v0.1.9. The CLI works on Intel Macs:"
            echo "  curl -fsSL https://hivebear.com/install.sh | sh"
        fi
        exit 1
    fi

    echo "Downloading $asset..."
    download "${BASE_URL}/${asset}" "$WORKDIR/$asset"
    echo "Verifying checksum..."
    verify_checksum "$WORKDIR/$asset" "$WORKDIR/SHA256SUMS-desktop.txt" "$asset"

    mkdir -p "$WORKDIR/app"
    tar xzf "$WORKDIR/$asset" -C "$WORKDIR/app"
    if [ ! -d "$WORKDIR/app/HiveBear.app" ]; then
        echo "Error: $asset did not contain HiveBear.app"
        exit 1
    fi

    if [ -w /Applications ]; then
        APP_DIR="/Applications"
    else
        APP_DIR="$HOME/Applications"
        mkdir -p "$APP_DIR"
    fi
    dest="$APP_DIR/HiveBear.app"

    if pgrep -x HiveBear >/dev/null 2>&1; then
        echo "Note: HiveBear is running. Quit and reopen it after this to use the new version."
    fi
    rm -rf "$dest"
    # ditto preserves the code signature and extended attributes that a plain
    # cp -R can drop.
    ditto "$WORKDIR/app/HiveBear.app" "$dest"
    # Belt and braces: never set by curl/wget, but harmless to clear.
    xattr -dr com.apple.quarantine "$dest" 2>/dev/null || true

    echo "Installed HiveBear to $dest"
    echo ""
    echo "Open it from Launchpad, or run:"
    echo ""
    echo "  open \"$dest\""
    echo ""
}

install_desktop_linux() {
    if [ "$ARCH_NAME" != "x86_64" ]; then
        echo "Error: the desktop app is only built for x86_64 Linux."
        echo "The CLI works on $ARCH_NAME:"
        echo "  curl -fsSL https://hivebear.com/install.sh | sh"
        exit 1
    fi

    format="${HIVEBEAR_DESKTOP_FORMAT:-}"
    if [ -z "$format" ]; then
        if command -v apt-get >/dev/null 2>&1 && command -v dpkg >/dev/null 2>&1 \
            && { [ "$(id -u)" -eq 0 ] || command -v sudo >/dev/null 2>&1; }; then
            format="deb"
        else
            format="appimage"
        fi
    fi

    echo "Installing the HiveBear desktop app (Linux, $format)"
    download "${BASE_URL}/SHA256SUMS-desktop.txt" "$WORKDIR/SHA256SUMS-desktop.txt"

    case "$format" in
        deb)
            asset="HiveBear-x86_64.deb"
            echo "Downloading $asset..."
            download "${BASE_URL}/${asset}" "$WORKDIR/$asset"
            verify_checksum "$WORKDIR/$asset" "$WORKDIR/SHA256SUMS-desktop.txt" "$asset"
            # apt-get (not dpkg -i) so missing webkit2gtk etc. are pulled in.
            # It must be given a path, and the _apt user must be able to read it.
            chmod 644 "$WORKDIR/$asset"
            chmod 755 "$WORKDIR"
            if [ "$(id -u)" -eq 0 ]; then
                apt-get install -y "$WORKDIR/$asset"
            else
                echo "Installing the package needs administrator rights (sudo)."
                sudo apt-get install -y "$WORKDIR/$asset"
            fi
            echo ""
            echo "Installed. Launch HiveBear from your applications menu."
            echo ""
            ;;
        appimage)
            asset="HiveBear-x86_64.AppImage"
            echo "Downloading $asset..."
            download "${BASE_URL}/${asset}" "$WORKDIR/$asset"
            verify_checksum "$WORKDIR/$asset" "$WORKDIR/SHA256SUMS-desktop.txt" "$asset"

            mkdir -p "$INSTALL_DIR"
            app="$INSTALL_DIR/HiveBear.AppImage"
            cp "$WORKDIR/$asset" "$app"
            chmod +x "$app"

            data_home="${XDG_DATA_HOME:-$HOME/.local/share}"
            icon="$data_home/icons/hicolor/512x512/apps/hivebear.png"
            mkdir -p "$data_home/applications" "$(dirname "$icon")"
            download "https://raw.githubusercontent.com/$REPO/main/assets/icon-512.png" "$icon" 2>/dev/null \
                || rm -f "$icon"
            cat > "$data_home/applications/hivebear.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=HiveBear
Comment=Run local AI models on your own machine
Exec="$app" %U
Icon=hivebear
Terminal=false
Categories=Utility;Development;
EOF
            if command -v update-desktop-database >/dev/null 2>&1; then
                update-desktop-database "$data_home/applications" >/dev/null 2>&1 || true
            fi

            echo "Installed HiveBear to $app"
            echo "Added HiveBear to your applications menu."

            # AppImages mount themselves with FUSE 2, which newer distros no
            # longer install by default. Without it the app fails to start
            # with a message most people will not decode.
            if ! { ldconfig -p 2>/dev/null | grep -q 'libfuse\.so\.2'; }; then
                echo ""
                echo "NOTE: AppImages need FUSE 2 (libfuse.so.2), which was not found."
                echo "  Ubuntu 24.04+:     sudo apt install libfuse2t64"
                echo "  Ubuntu/Debian:     sudo apt install libfuse2"
                echo "  Fedora:            sudo dnf install fuse-libs"
                echo "Or run it without FUSE: $app --appimage-extract-and-run"
            fi
            echo ""
            ;;
        *)
            echo "Error: HIVEBEAR_DESKTOP_FORMAT must be 'deb' or 'appimage', not '$format'"
            exit 1
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

download() {
    url="$1"
    output="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$url" -o "$output" || { echo "Error: download failed: $url" >&2; return 1; }
    elif command -v wget >/dev/null 2>&1; then
        wget -qO "$output" "$url" || { echo "Error: download failed: $url" >&2; return 1; }
    else
        echo "Error: Neither curl nor wget found. Please install one and try again."
        exit 1
    fi
}

verify_checksum() {
    file="$1"
    checksums_file="$2"
    filename="$3"

    # Exact filename match on field 2 -- sha256sum writes "<hash>  <name>" (or
    # "<hash> *<name>" in binary mode). A substring grep would also match a
    # sibling line such as "<name>.sig" and return two hashes.
    expected="$(awk -v f="$filename" '$2 == f || $2 == "*" f { print $1 }' "$checksums_file")"

    if [ -z "$expected" ]; then
        echo "Error: No checksum found for $filename in $(basename "$checksums_file"). Aborting."
        exit 1
    fi

    if command -v sha256sum >/dev/null 2>&1; then
        actual="$(sha256sum "$file" | awk '{print $1}')"
    elif command -v shasum >/dev/null 2>&1; then
        actual="$(shasum -a 256 "$file" | awk '{print $1}')"
    else
        echo "Error: No sha256sum or shasum found. Cannot verify checksum. Aborting."
        exit 1
    fi

    if [ "$expected" != "$actual" ]; then
        echo "Error: Checksum verification failed for $filename!"
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

main "$@"
