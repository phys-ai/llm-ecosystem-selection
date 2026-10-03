#!/usr/bin/env bash
set -euo pipefail

SCRIPT="${SCRIPT:-replay/engine.py}"
RUNS_BASE="${RUNS_BASE:-../runs}"  # run from pipeline/
OUT_STAB_ROBUST="${OUT_STAB_ROBUST:-results/stability_threshold_robustness}"
MAX_JOBS="${MAX_JOBS:-5}"

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

STAB_ROBUST_SETTINGS=(
  "active=1e-5 fd=1e-5 inv=1e-5 stride=2 K=100"
  "active=1e-3 fd=1e-5 inv=1e-5 stride=2 K=100"
  "active=1e-4 fd=1e-6 inv=1e-5 stride=2 K=100"
  "active=1e-4 fd=1e-4 inv=1e-5 stride=2 K=100"
  "active=1e-4 fd=1e-5 inv=1e-6 stride=2 K=100"
  "active=1e-4 fd=1e-5 inv=1e-4 stride=2 K=100"
  "active=1e-4 fd=1e-5 inv=1e-5 stride=1 K=100"
  "active=1e-4 fd=1e-5 inv=1e-5 stride=4 K=100"
  "active=1e-4 fd=1e-5 inv=1e-5 stride=2 K=50"
  "active=1e-4 fd=1e-5 inv=1e-5 stride=2 K=150"
)

mkdir -p "$OUT_STAB_ROBUST"

for setting in "${STAB_ROBUST_SETTINGS[@]}"; do
  active_eps=""
  fd_eps=""
  invasion_eps=""
  stride=""
  last_k=""

  for token in $setting; do
    key="${token%%=*}"
    value="${token#*=}"
    case "$key" in
      active) active_eps="$value" ;;
      fd) fd_eps="$value" ;;
      inv) invasion_eps="$value" ;;
      stride) stride="$value" ;;
      K) last_k="$value" ;;
      *) echo "Unknown stability robustness token: $token" >&2; exit 1 ;;
    esac
  done

  label="active=${active_eps}_fd=${fd_eps}_inv=${invasion_eps}_stride=${stride}_K=${last_k}"
  mkdir -p "$OUT_STAB_ROBUST/$label"

  for run in "${RUNS[@]}"; do
    wait_for_slot

    echo "Launching stability robustness [$label] / $run"
    python "$SCRIPT" \
      --runs_dir "$RUNS_BASE/$run" \
      --out_dir "$OUT_STAB_ROBUST/$label/$run" \
      --estimate_stability \
      --risk_lambdas 0 \
      --support_strengths 0 \
      --route_strengths 2 \
      --context_trait_strengths 0 \
      --polarization_strengths 0 \
      --route_powers 1 \
      --route_zscore \
      --active_eps "$active_eps" \
      --fd_eps "$fd_eps" \
      --invasion_eps "$invasion_eps" \
      --stability_context_stride "$stride" \
      --endpoint_last_k "$last_k" \
      --stability_max_active 96 \
      --no_trajectories \
      --etas 0.8,1.2,2.0,3.0,4.0,5.0 \
      --alphas 0.5,0.75,1.0 \
      > "$OUT_STAB_ROBUST/$label/${run}.log" 2>&1 &
  done
done

wait

echo "Finished stability robustness reruns under $OUT_STAB_ROBUST"
