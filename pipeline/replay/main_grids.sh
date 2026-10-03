#!/usr/bin/env bash
set -euo pipefail

# Run all grids for checkpoint runs under ../runs.
#
# Usage:
#   bash replay_main_grids.sh
#
# To specify only particular runs:
#   NEW_RUNS="run_popseed_66.checkpoints,run_popseed_77.checkpoints" bash replay_main_grids.sh
#
# To include old seeds too:
#   INCLUDE_EXISTING=1 bash replay_main_grids.sh


SCRIPT="${SCRIPT:-replay/engine.py}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNS_BASE="${RUNS_BASE:-../runs}"  # run from pipeline/
MAX_JOBS="${MAX_JOBS:-5}"

# Output roots
OUT_FIG04="${OUT_FIG04:-results/fig0_4_full_grid}"
OUT_FIG5="${OUT_FIG5:-results/fig5_full_grid}"
OUT_FIG10="${OUT_FIG10:-results/fig10_full_grid}"
OUT_FIGS="${OUT_FIGS:-results/specialization_wide_dense_figS}"
OUT_ROUTE_APPENDIX="${OUT_ROUTE_APPENDIX:-results/appendix_route_sweep_parallel}"
OUT_STAB_ROBUST="${OUT_STAB_ROBUST:-results/stability_threshold_robustness}"
EARLY_FRACS="${EARLY_FRACS:-0.05,0.1,0.2,0.3,0.4}"

# Known old runs to skip by default.
#OLD_RUNS=(
  #run_popseed_11.checkpoints
  #run_popseed_22.checkpoints
  #run_popseed_33.checkpoints
  #run_popseed_44.checkpoints
  #run_popseed_55.checkpoints
#)

# Known old runs to skip by default.
# Empty string means skip nothing.
OLD_RUNS=""

is_old_run () {
  local r="$1"
  local old

  if [[ -z "${OLD_RUNS:-}" ]]; then
    return 1
  fi

  for old in $OLD_RUNS; do
    if [[ "$r" == "$old" ]]; then
      return 0
    fi
  done

  return 1
}

# Determine runs.
if [[ -n "${NEW_RUNS:-}" ]]; then
  IFS=',' read -r -a RUNS <<< "$NEW_RUNS"
else
  ALL_RUNS=()
  while IFS= read -r r; do
    ALL_RUNS+=("$r")
  done < <(find "$RUNS_BASE" -maxdepth 1 -type d -name "*.checkpoints" -exec basename {} \; | sort)

  RUNS=()
  for r in "${ALL_RUNS[@]}"; do
    if [[ "${INCLUDE_EXISTING:-0}" == "1" ]]; then
      RUNS+=("$r")
    else
      if ! is_old_run "$r"; then
        RUNS+=("$r")
      fi
    fi
  done
fi


if (( ${#RUNS[@]} == 0 )); then
  echo "No runs selected."
  echo "Either add new *.checkpoints under $RUNS_BASE, or run with:"
  echo '  INCLUDE_EXISTING=1 bash replay_main_grids.sh'
  echo '  NEW_RUNS="run_popseed_66.checkpoints" bash replay_main_grids.sh'
  exit 1
fi

echo "Selected runs:"
printf '  %s\n' "${RUNS[@]}"
echo

wait_for_slot () {
  while (( $(jobs -rp | wc -l | tr -d ' ') >= MAX_JOBS )); do
    sleep 5
  done
}

run_one () {
  local label="$1"
  local out_dir="$2"
  local run="$3"
  shift 3

  mkdir -p "$out_dir"
  wait_for_slot

  echo "Launching [$label] / $run"
  "$PYTHON_BIN" "$SCRIPT" \
    --runs_dir "$RUNS_BASE/$run" \
    --out_dir "$out_dir/$run" \
    "$@" \
    > "$out_dir/${run}.log" 2>&1 &
}

check_outputs () {
  local label="$1"
  local out_dir="$2"

  echo
  echo "Checking $label outputs under $out_dir"

  "$PYTHON_BIN" - "$out_dir" "$label" <<'PY'
from pathlib import Path
import sys
import pandas as pd

out_dir = Path(sys.argv[1])
label = sys.argv[2]

files = sorted(out_dir.glob("**/phase_summary.csv"))
print(f"{label}: phase_summary files = {len(files)}")

if not files:
    logs = sorted(out_dir.glob("**/*.log"))
    print(f"{label}: no phase_summary.csv found. Showing log tails.")
    for log in logs[:20]:
        print(f"\n--- {log} ---")
        print("\n".join(log.read_text(errors="replace").splitlines()[-40:]))
    raise SystemExit(1)

df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True, sort=False)

for c in ["eta", "alpha", "beta_sup", "beta_route", "context_trait_strength", "route_power"]:
    if c in df.columns:
        vals = sorted(pd.to_numeric(df[c], errors="coerce").dropna().unique())
        shown = vals[:40]
        suffix = " ..." if len(vals) > 40 else ""
        print(f"{c}: n={len(vals)} vals={shown}{suffix}")

if {"eta", "alpha"}.issubset(df.columns):
    print("eta-alpha cells:", df[["eta", "alpha"]].drop_duplicates().shape[0])

if {"eta", "beta_sup"}.issubset(df.columns):
    print("eta-beta_sup cells:", df[["eta", "beta_sup"]].drop_duplicates().shape[0])

if {"beta_sup", "context_trait_strength"}.issubset(df.columns):
    print("beta_sup-context_trait_strength cells:", df[["beta_sup", "context_trait_strength"]].drop_duplicates().shape[0])

print("rows:", len(df))
PY
}

merge_early_warning_outputs () {
  local out_dir="$1"
  "$PYTHON_BIN" - "$out_dir" <<'PY'
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
}

###############################################################################
# 1. Fig0–4 full dense grid
###############################################################################

COMMON_FIG04=(
  --estimate_stability
  --evaluate_early_warning
  --early_fracs "$EARLY_FRACS"
  --support_strengths 0
  --route_strengths 2
  --risk_lambdas 0
  --context_trait_strengths 0
  --polarization_strengths 0
  --route_powers 1
  --route_zscore
  --endpoint_last_k 100
  --active_eps 1e-4
  --fd_eps 1e-5
  --invasion_eps 1e-5
  --stability_context_stride 2
  --stability_max_active 96
  --trajectory_stride 1
  --etas 0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.35,1.5,1.75,2.0,2.25,2.5,2.75,3.0,3.25,3.5,3.75,4.0,4.5,5.0,6.0,7.0,8.0
  --alphas 0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0
)

mkdir -p "$OUT_FIG04"
for run in "${RUNS[@]}"; do
  run_one "Fig0-4 full grid" "$OUT_FIG04" "$run" "${COMMON_FIG04[@]}"
done
wait
echo "Finished Fig0-4 full grid."
check_outputs "Fig0-4" "$OUT_FIG04"

###############################################################################
# 2. Fig5 support full grid
###############################################################################

COMMON_FIG5=(
  --estimate_stability
  --risk_lambdas 0
  --alphas 1
  --route_strengths 2
  --context_trait_strengths 0
  --polarization_strengths 0
  --route_powers 1
  --route_zscore
  --endpoint_last_k 100
  --active_eps 1e-4
  --fd_eps 1e-5
  --invasion_eps 1e-5
  --stability_context_stride 4
  --stability_max_active 96
  --no_trajectories
  --etas 0.25,0.3,0.35,0.4,0.45,0.475,0.5,0.525,0.55,0.625,0.7,0.8,0.9,1.0,1.1,1.15,1.2,1.275,1.35,1.55,1.75,1.875,2.0,2.125,2.25,2.5,2.75,2.875,3.0,3.125,3.25,3.5,3.75,3.875,4.0,4.25,4.5,4.75,5.0,5.5,6.0,6.5,7.0,7.5,8.0
  --support_strengths 0,0.01,0.02,0.025,0.03,0.035,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.135,0.15,0.165,0.18,0.19,0.2,0.225,0.25,0.275,0.3,0.35,0.4,0.45,0.5,0.575,0.65,0.725,0.8,0.9,1.0,1.1,1.2,1.35,1.5,1.75,2.0
)

mkdir -p "$OUT_FIG5"
for run in "${RUNS[@]}"; do
  run_one "Fig5 support full grid" "$OUT_FIG5" "$run" "${COMMON_FIG5[@]}"
done
wait
echo "Finished Fig5 support full grid."
check_outputs "Fig5" "$OUT_FIG5"

###############################################################################
# 3. Fig10 specialization heatmap grid
###############################################################################

COMMON_FIG10=(
  --estimate_stability
  --risk_lambdas 0
  --alphas 1
  --etas 1.0
  --route_strengths 3
  --route_powers 2
  --polarization_strengths 0
  --route_zscore
  --endpoint_last_k 100
  --active_eps 1e-4
  --fd_eps 1e-5
  --invasion_eps 1e-5
  --stability_context_stride 5
  --stability_max_active 96
  --no_trajectories
  --support_strengths 0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9,1.0,1.2,1.5
  --context_trait_strengths 0.8,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5,2.7,3.0
)

mkdir -p "$OUT_FIG10/specialization"
for run in "${RUNS[@]}"; do
  run_one "Fig10 specialization grid" "$OUT_FIG10/specialization" "$run" "${COMMON_FIG10[@]}"
done
wait
echo "Finished Fig10 specialization grid."
check_outputs "Fig10" "$OUT_FIG10"

###############################################################################
# 4. FigS dense stable-specialization heatmap grid
###############################################################################
# This replaces the older coarse specialization_wide_6h block.
# It targets the exact default FigS slice in appendix_figures.py:
#   alpha = 1
#   beta_route = 2
#   route_power = 1.5
#   context_trait_strength = 1.0
#   polarization_strength = 0
# and densifies eta × beta_sup.

COMMON_FIGS=(
  --estimate_stability
  --risk_lambdas 0
  --alphas 1
  --route_strengths 2
  --route_powers 1.5
  --context_trait_strengths 1.0
  --polarization_strengths 0
  --route_zscore
  --endpoint_last_k 100
  --active_eps 1e-4
  --fd_eps 1e-5
  --invasion_eps 1e-5
  --stability_context_stride 8
  --stability_max_active 96
  --no_trajectories
  --etas 0.1,0.125,0.15,0.175,0.2,0.225,0.25,0.275,0.3,0.325,0.35,0.375,0.4,0.425,0.45,0.475,0.5,0.525,0.55,0.575,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.275,1.35,1.425,1.5,1.625,1.75,1.875,2.0
  --support_strengths 0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.2,1.275,1.35,1.425,1.5,1.625,1.75,1.875,2.0,2.125,2.25,2.375,2.5,2.75,3.0
)

mkdir -p "$OUT_FIGS"
for run in "${RUNS[@]}"; do
  run_one "FigS dense eta-beta grid" "$OUT_FIGS" "$run" "${COMMON_FIGS[@]}"
done
wait
echo "Finished FigS dense eta-beta grid."
check_outputs "FigS" "$OUT_FIGS"

###############################################################################
# 5. Appendix route-strength sweep
###############################################################################

COMMON_ROUTE_APPENDIX=(
  --estimate_stability
  --context_key topic_category
  --endpoint_last_k 50
  --active_eps 1e-4
  --fd_eps 1e-5
  --invasion_eps 1e-5
  --stability_context_stride 1
  --stability_max_active 160
  --trajectory_stride 5
  --progress_every 20
  --etas 0.25,0.4,0.5,0.7,0.9,1.0,1.2,1.5,2.0,2.5,2.75,3.25,4,5,6,8
  --alphas 1.0
  --support_strengths 0
  --route_strengths 0,0.25,0.5,0.75,1,1.25,1.5,2,2.5,3,4,5,6
  --risk_lambdas 0
  --hard_safety_thresholds -1
)

mkdir -p "$OUT_ROUTE_APPENDIX"
for run in "${RUNS[@]}"; do
  run_one "Appendix route-strength sweep" "$OUT_ROUTE_APPENDIX" "$run" "${COMMON_ROUTE_APPENDIX[@]}"
done
wait
echo "Finished appendix route-strength sweep."
check_outputs "Appendix route sweep" "$OUT_ROUTE_APPENDIX"

###############################################################################
# 6. Numerical stability threshold robustness (optional compact sweep)
###############################################################################
# This block reruns a small representative grid while perturbing the numerical
# knobs that affect kappa_map estimation itself. It is opt-in because these are
# extra stability-estimation jobs, not required for the main figure grids.
#
# Enable with:
#   RUN_STABILITY_ROBUSTNESS=1 bash replay_main_grids.sh
#
# Optional output override:
#   OUT_STAB_ROBUST=results/my_stability_robustness RUN_STABILITY_ROBUSTNESS=1 bash replay_main_grids.sh

if [[ "${RUN_STABILITY_ROBUSTNESS:-0}" == "1" ]]; then
  mkdir -p "$OUT_STAB_ROBUST"

  # One-at-a-time perturbations around the main Fig0-4 stability baseline:
  #   active_eps=1e-4, fd_eps=1e-5, invasion_eps=1e-5,
  #   stability_context_stride=2, endpoint_last_k=100.
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

    if [[ -z "$active_eps" || -z "$fd_eps" || -z "$invasion_eps" || -z "$stride" || -z "$last_k" ]]; then
      echo "Malformed stability robustness setting: $setting" >&2
      exit 1
    fi

    label="active=${active_eps}_fd=${fd_eps}_inv=${invasion_eps}_stride=${stride}_K=${last_k}"
    safe_label="${label//[^A-Za-z0-9_.=-]/_}"

    COMMON_STAB_ROBUST=(
      --estimate_stability
      --risk_lambdas 0
      --support_strengths 0
      --route_strengths 2
      --context_trait_strengths 0
      --polarization_strengths 0
      --route_powers 1
      --route_zscore
      --active_eps "$active_eps"
      --fd_eps "$fd_eps"
      --invasion_eps "$invasion_eps"
      --stability_context_stride "$stride"
      --endpoint_last_k "$last_k"
      --stability_max_active 96
      --no_trajectories
      --etas 0.8,1.2,2.0,3.0,4.0,5.0
      --alphas 0.5,0.75,1.0
    )

    for run in "${RUNS[@]}"; do
      run_one "Stability robustness $label" "$OUT_STAB_ROBUST/$safe_label" "$run" "${COMMON_STAB_ROBUST[@]}"
    done
  done

  wait
  echo "Finished numerical stability threshold robustness."
  check_outputs "Stability robustness" "$OUT_STAB_ROBUST"
fi



echo
echo "All requested grids finished."
echo
echo "Outputs:"
echo "  Fig0-4: $OUT_FIG04"
echo "  Fig5:   $OUT_FIG5"
echo "  Fig10:  $OUT_FIG10"
echo "  FigS:   $OUT_FIGS"
echo "  Route appendix: $OUT_ROUTE_APPENDIX"
echo "  Stability robustness: $OUT_STAB_ROBUST (only if RUN_STABILITY_ROBUSTNESS=1)"


###############################################################################
# 7. Early-warning validation grid
###############################################################################
# Optional compact grid for early-warning prediction of both collapse and
# positive criticality / invadability.
#
# Enable with:
#   RUN_EARLY_WARNING_GRID=1 bash replay_main_grids.sh
#
# Rationale:
#   The main Fig0-4 no-support grid provides collapse examples but can make
#   label_unstable one-class. This grid includes support variation so that
#   both kappa_map > 0 and kappa_map < 0 examples enter the early-warning
#   evaluation.

OUT_EW="${OUT_EW:-results/early_warning_validation_grid}"

if [[ "${RUN_EARLY_WARNING_GRID:-0}" == "1" ]]; then
  COMMON_EW=(
    --estimate_stability
    --evaluate_early_warning
    --early_fracs "$EARLY_FRACS"
    --risk_lambdas 0
    --route_strengths 2
    --context_trait_strengths 0
    --polarization_strengths 0
    --route_powers 1
    --route_zscore
    --endpoint_last_k 100
    --active_eps 1e-4
    --fd_eps 1e-5
    --invasion_eps 1e-5
    --stability_context_stride 4
    --stability_max_active 96
    --trajectory_stride 1
    --etas 0.4,0.8,1.2,2.0,3.0,4.0,5.0,7.0
    --alphas 0.5,0.75,1.0
    --support_strengths 0,0.05,0.1,0.2,0.5,1.0,1.5,2.0
  )

  mkdir -p "$OUT_EW"
  for run in "${RUNS[@]}"; do
    run_one "Early-warning validation grid" "$OUT_EW" "$run" "${COMMON_EW[@]}"
  done

  wait
  echo "Finished early-warning validation grid."
  merge_early_warning_outputs "$OUT_EW"
  check_outputs "Early-warning validation" "$OUT_EW"
fi
