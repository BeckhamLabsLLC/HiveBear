#!/bin/bash
set -euo pipefail

BINARY_NAME="hivebear"
INSTALL_DIR="$HOME/.local/bin"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/hivebear"
DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/hivebear"

main() {
    echo "HiveBear Uninstaller"
    echo "==================="
    echo ""

    local found=false

    # Check for binary in common locations
    local binary_path=""
    if command -v "$BINARY_NAME" >/dev/null 2>&1; then
        binary_path="$(command -v "$BINARY_NAME")"
        echo "Found binary: $binary_path"
        found=true
    elif [ -f "$INSTALL_DIR/$BINARY_NAME" ]; then
        binary_path="$INSTALL_DIR/$BINARY_NAME"
        echo "Found binary: $binary_path"
        found=true
    fi

    if [ -d "$CONFIG_DIR" ]; then
        echo "Found config: $CONFIG_DIR"
        found=true
    fi

    if [ -d "$DATA_DIR" ]; then
        local data_size
        data_size="$(du -sh "$DATA_DIR" 2>/dev/null | cut -f1)"
        echo "Found data:   $DATA_DIR ($data_size)"
        found=true
    fi

    if [ "$found" = false ]; then
        echo "HiveBear does not appear to be installed."
        exit 0
    fi

    echo ""

    # Remove binary
    if [ -n "$binary_path" ] && [ -f "$binary_path" ]; then
        rm -f "$binary_path"
        echo "Removed binary: $binary_path"
    fi

    # Ask about data removal
    if [ -d "$DATA_DIR" ] || [ -d "$CONFIG_DIR" ]; then
        echo ""
        echo "Remove configuration and downloaded models?"
        printf "[y/N] "
        read -r reply
        if [ "$reply" = "y" ] || [ "$reply" = "Y" ]; then
            [ -d "$CONFIG_DIR" ] && rm -rf "$CONFIG_DIR" && echo "Removed config: $CONFIG_DIR"
            [ -d "$DATA_DIR" ] && rm -rf "$DATA_DIR" && echo "Removed data:   $DATA_DIR"
        else
            echo "Keeping config and data."
        fi
    fi

    echo ""
    echo "HiveBear has been uninstalled."
}

main
