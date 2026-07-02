#!/usr/bin/env bash
# 00-control-venv.sh — create the Ansible control venv used to drive the cluster.
# Run once on the control node (node0). The venv lives OUTSIDE the repo at
# $OPS_VENV_DIR so it is never committed; only this script is tracked.
set -euo pipefail

VENV_DIR="${OPS_VENV_DIR:-$HOME/venvs/ops}"
PYTHON="${PYTHON:-python3}"

if [[ "${1:-}" == "--force" && -d "$VENV_DIR" ]]; then
  echo "[setup] removing existing venv at $VENV_DIR"; rm -rf "$VENV_DIR"
fi

if [[ -d "$VENV_DIR" ]]; then
  echo "[setup] venv exists at $VENV_DIR (use --force to recreate)"
else
  echo "[setup] creating venv at $VENV_DIR"; "$PYTHON" -m venv "$VENV_DIR"
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip >/dev/null
# Pinned deps live in requirements.txt (single source of truth): ansible-core to
# drive the cluster, modal for the `modal` CLI (Modal backend auth) on this node.
pip install -r "$HERE/requirements.txt"

echo "[setup] done. $(ansible --version | head -1)"
echo "[setup] activate with: source $VENV_DIR/bin/activate"
