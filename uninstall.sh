#!/bin/bash
set -euo pipefail

BINARY_NAME="hivebear"
INSTALL_DIR="$HOME/.local/bin"

# Must match AppPaths::new(), which uses
# ProjectDirs::from("com", "HiveBear", "hivebear") -- the layout differs per
# platform, and hardcoding the XDG paths meant macOS uninstalls silently left
# the config and the whole model cache on disk.
case "$(uname -s)" in
    Darwin)
        CONFIG_DIR="$HOME/Library/Application Support/com.HiveBear.hivebear"
        DATA_DIR="$HOME/Library/Application Support/com.HiveBear.hivebear"
        CACHE_DIR="$HOME/Library/Caches/com.HiveBear.hivebear"
        ;;
    *)
        CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/hivebear"
        DATA_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/hivebear"
        CACHE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/hivebear"
        ;;
esac

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

    # De-duplicate: on macOS ProjectDirs maps config and data to one directory.
    # An array, not a string -- these paths contain spaces on macOS.
    local -a dirs=()
    local d existing seen dir_size
    for d in "$CONFIG_DIR" "$DATA_DIR" "$CACHE_DIR"; do
        [ -d "$d" ] || continue
        seen=false
        for existing in "${dirs[@]+"${dirs[@]}"}"; do
            [ "$existing" = "$d" ] && seen=true && break
        done
        [ "$seen" = true ] && continue
        dirs+=("$d")
        dir_size="$(du -sh "$d" 2>/dev/null | cut -f1)"
        echo "Found data:   $d ($dir_size)"
        found=true
    done

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
    if [ "${#dirs[@]}" -gt 0 ]; then
        echo ""
        echo "Remove configuration and downloaded models?"
        printf "[y/N] "
        read -r reply
        if [ "$reply" = "y" ] || [ "$reply" = "Y" ]; then
            for d in "${dirs[@]}"; do
                rm -rf "$d" && echo "Removed: $d"
            done
        else
            echo "Keeping config and data."
        fi
    fi

    echo ""
    echo "HiveBear has been uninstalled."
}

main
