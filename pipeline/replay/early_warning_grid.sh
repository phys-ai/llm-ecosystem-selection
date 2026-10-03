#!/usr/bin/env bash
set -euo pipefail

SCRIPT="${SCRIPT:-replay/engine.py}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNS_BASE="${RUNS_BASE:-../runs}"  # run from pipeline/
OUT_EW="${OUT_EW:-results/early_warning_validation_grid}"
MAX_JOBS="${MAX_JOBS:-5}"
EARLY_FRACS="${EARLY_FRACS:-0.05,0.1,0.2,0.3,0.4}"

RUNS=()
while IFS= read -r run; do
  RUNS+=("$run")
done < <(
  find "$RUNS_BASE" -maxdepth 1 -type d -name '*.checkpoints' \
    ! -name '*reuse_together_evalaudit_compact*' \
    -exec basename {} \; | sort
)

if (( ${#RUNS[@]} == 0 )); then
  echo "No checkpoint runs found under $RUNS_BASE" >&2
  exit 1
fi

wait_for_slot() {
  while (( $(jobs -rp | wc -l | tr -d ' ') >= MAX_JOBS )); do
    sleep 5
  done
}

mkdir -p "$OUT_EW"

for run in "${RUNS[@]}"; do
  wait_for_slot

  echo "Launching EW: $run"
  "$PYTHON_BIN" "$SCRIPT" \
    --runs_dir "$RUNS_BASE/$run" \
    --out_dir "$OUT_EW/$run" \
    --estimate_stability \
    --evaluate_early_warning \
    --early_fracs "$EARLY_FRACS" \
    --risk_lambdas 0 \
    --route_strengths 2 \
    --context_trait_strengths 0 \
    --polarization_strengths 0 \
    --route_powers 1 \
    --route_zscore \
    --endpoint_last_k 100 \
    --active_eps 1e-4 \
    --fd_eps 1e-5 \
    --invasion_eps 1e-5 \
    --stability_context_stride 4 \
    --stability_max_active 96 \
    --trajectory_stride 1 \
    --etas 0.4,0.8,1.2,2.0,3.0,4.0,5.0,7.0 \
    --alphas 0.5,0.75,1.0 \
    --support_strengths 0,0.05,0.1,0.2,0.5,1.0,1.5,2.0 \
    > "$OUT_EW/${run}.log" 2>&1 &
done

wait

"$PYTHON_BIN" - "$OUT_EW" <<'PY'
from pathlib import Path
import sys
import pandas as pd

root = Path(sys.argv[1])
outputs = [
    ("early_warning_features.csv", "early_warning_features.csv", {}),
    ("early_warning_evaluation.csv", "early_warning_evaluation.csv", {}),
    ("early_warning_predictions.csv.gz", "early_warning_predictions.csv.gz", {"compression": "gzip"}),
]
for pattern, out_name, write_kwargs in outputs:
    files = sorted(p for p in root.glob(f"**/{pattern}") if p.parent != root)
    if not files:
        continue
    parts = []
    for path in files:
        df = pd.read_csv(path)
        df["source_result_dir"] = str(path.parent)
        df["source_result_path"] = str(path)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True, sort=False)
    out_path = root / out_name
    out.to_csv(out_path, index=False, **write_kwargs)
    print(f"merged {len(files)} files -> {out_path} ({len(out)} rows)")
PY

echo "Finished early-warning validation grid under $OUT_EW"
