#!/usr/bin/env bash
# install-cli.sh — symlink engram_cli.py to /usr/local/bin/engram-cli (idempotent).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/engram_cli.py"
DEST="${ENGRAM_CLI_DEST:-/usr/local/bin/engram-cli}"

chmod +x "$SRC"

# ln -sf is idempotent: replaces an existing symlink pointing anywhere.
ln -sf "$SRC" "$DEST"

echo "installed: $DEST -> $SRC"
"$DEST" --help >/dev/null && echo "ok: engram-cli runs"
