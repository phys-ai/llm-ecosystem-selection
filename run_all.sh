#!/usr/bin/env bash
# Reproduce the paper from the stored analysis outputs in data/.
#   bash run_all.sh            figures, tables and the number check (~2 min)
# Recomputing data/ itself from the raw logs is the pipeline (see pipeline/README.md).
set -euo pipefail
cd "$(dirname "$0")"
export MPLBACKEND=Agg
PY="${PYTHON_BIN:-python3}"
$PY figures.py all
$PY tables.py all
$PY check.py
echo "figures -> figures/ (or PAPER_DIR/figures), derived tables -> data/outputs"
