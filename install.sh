#!/bin/sh
# WrenCode installer — downloads the latest standalone binary from GitHub Releases.
#
#   curl -fsSL https://raw.githubusercontent.com/almostly/wrencode/main/install.sh | sh
#
# Environment overrides:
#   WRENCODE_INSTALL_DIR   where to install     (default: ~/.local/bin)
#   WRENCODE_VERSION       release tag to fetch (default: latest)
set -eu

REPO="almostly/wrencode"
BIN_NAME="wrencode"
INSTALL_DIR="${WRENCODE_INSTALL_DIR:-$HOME/.local/bin}"
VERSION="${WRENCODE_VERSION:-latest}"

say() { printf '%s\n' "$*"; }
err() { printf 'error: %s\n' "$*" >&2; exit 1; }

# -----------------------------------------------------------------------------------------------
# Detect platform
# -----------------------------------------------------------------------------------------------
os="$(uname -s)"
arch="$(uname -m)"
case "$os" in
  Linux)
    case "$arch" in
      x86_64 | amd64) asset="wrencode-linux-x64" ;;
      *) err "no prebuilt binary for Linux/$arch — install from source: https://github.com/$REPO" ;;
    esac
    ;;
  Darwin)
    case "$arch" in
      arm64 | aarch64) asset="wrencode-macos-arm64" ;;
      x86_64) asset="wrencode-macos-x64" ;;
      *) err "no prebuilt binary for macOS/$arch" ;;
    esac
    ;;
  *) err "unsupported OS: $os — install from source: https://github.com/$REPO" ;;
esac

# -----------------------------------------------------------------------------------------------
# Pick a downloader
# -----------------------------------------------------------------------------------------------
if command -v curl >/dev/null 2>&1; then
  dl() { curl -fsSL "$1" -o "$2"; }
elif command -v wget >/dev/null 2>&1; then
  dl() { wget -qO "$2" "$1"; }
else
  err "need curl or wget on PATH"
fi

if [ "$VERSION" = "latest" ]; then
  url="https://github.com/$REPO/releases/latest/download/$asset"
else
  url="https://github.com/$REPO/releases/download/$VERSION/$asset"
fi

# -----------------------------------------------------------------------------------------------
# Download
# -----------------------------------------------------------------------------------------------
say "Downloading $asset ($VERSION)..."
tmp="$(mktemp)"
trap 'rm -f "$tmp" "$tmp.sha256"' EXIT INT TERM
dl "$url" "$tmp" || err "download failed: $url"

# GitHub serves an HTML page when an asset is missing; reject that.
if head -c 64 "$tmp" | grep -qi '<!doctype\|<html'; then
  err "release asset not found ($asset @ $VERSION). Has a release been published?"
fi

# -----------------------------------------------------------------------------------------------
# Verify the SHA-256 published with the release (releases before 0.4 have none)
# -----------------------------------------------------------------------------------------------
if dl "$url.sha256" "$tmp.sha256" 2>/dev/null; then
  want="$(cut -d' ' -f1 "$tmp.sha256" | tr -d '[:space:]')"
  if command -v sha256sum >/dev/null 2>&1; then
    got="$(sha256sum "$tmp" | cut -d' ' -f1)"
  elif command -v shasum >/dev/null 2>&1; then
    got="$(shasum -a 256 "$tmp" | cut -d' ' -f1)"
  else
    got=""
  fi
  if [ -z "$got" ]; then
    say "⚠  Neither sha256sum nor shasum found; skipping checksum verification."
  elif [ "$got" != "$want" ]; then
    err "checksum mismatch for $asset: expected $want, got $got"
  else
    say "✓ Checksum verified"
  fi
else
  say "⚠  No checksum published for $asset @ $VERSION; skipping verification."
fi

# -----------------------------------------------------------------------------------------------
# Install
# -----------------------------------------------------------------------------------------------
mkdir -p "$INSTALL_DIR"
chmod +x "$tmp"
mv "$tmp" "$INSTALL_DIR/$BIN_NAME"
trap - EXIT INT TERM

say ""
say "✓ Installed $BIN_NAME to $INSTALL_DIR/$BIN_NAME"

# -----------------------------------------------------------------------------------------------
# PATH hint
# -----------------------------------------------------------------------------------------------
case ":$PATH:" in
  *":$INSTALL_DIR:"*) ;;
  *)
    say ""
    say "⚠  $INSTALL_DIR is not on your PATH. Add it, then restart your shell:"
    case "${SHELL:-}" in
      */zsh)  say "    echo 'export PATH=\"$INSTALL_DIR:\$PATH\"' >> ~/.zshrc" ;;
      */bash) say "    echo 'export PATH=\"$INSTALL_DIR:\$PATH\"' >> ~/.bashrc" ;;
      *)      say "    export PATH=\"$INSTALL_DIR:\$PATH\"" ;;
    esac
    ;;
esac

say ""
say "Run '$BIN_NAME' to start — you'll choose a backend on first run."
