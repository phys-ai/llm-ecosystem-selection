#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export MPLBACKEND=Agg
PY="${PYTHON_BIN:-python3}"
$PY figures.py all
$PY tables.py all
$PY check.py
echo "figures -> figures/ (or PAPER_DIR/figures), derived tables -> data/outputs"
