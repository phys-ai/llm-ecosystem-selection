#!/usr/bin/env python3
"""Context-disjoint robustness audits for the early-warning criticality result.

Two targets are produced for each run/parameter condition:

1. ``postearly``: replay the complete trajectory, but estimate the mean-map
   stability target only on contexts after the early-feature window.
2. ``context_holdout``: keep the complete early-feature window, stratify later
   contexts by topic category, reserve every kth context for stability only,
   replay the trajectory without those contexts, and evaluate the resulting
   terminal population state on the reserved contexts.

The ``evaluate`` subcommand joins these targets to the existing early-feature
table and performs strict leave-one-run-out logistic-regression evaluation.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import Iterable, Sequence

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPLAY_PATH = ROOT / "pipeline" / "replay" / "engine.py"


def load_replay_module():
    spec = importlib.util.spec_from_file_location("replay_analysis_context_audit", REPLAY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import replay module from {REPLAY_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


replay = load_replay_module()


def seed_from_name(value: str) -> int | None:
    match = re.search(r"popseed_(\d+)", str(value))
    return int(match.group(1)) if match else None


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def params_from_row(row: pd.Series):
    def number(name: str, default: float) -> float:
        value = row.get(name, default)
        return float(value) if pd.notna(value) else float(default)

    return replay.ReplayParams(
        eta=number("eta", 0.0),
        alpha=number("alpha", 0.0),
        beta_sup=number("beta_sup", 0.0),
        beta_route=number("beta_route", 0.0),
        lambda_risk=number("lambda_risk", 0.0),
        hard_safety_threshold=number("hard_safety_threshold", -1.0),
        context_trait_strength=number("context_trait_strength", 0.0),
        route_power=number("route_power", 1.0),
        route_zscore=bool(row.get("route_zscore", False)),
        no_feedback=bool(row.get("no_feedback", False)),
        ablation=str(row.get("ablation", "none")),
        polarization_strength=number("polarization_strength", 0.0),
    )


def endpoint_from_contexts(
    table,
    params,
    context_indices: Sequence[int],
    *,
    last_k: int,
    support_clip: float,
) -> np.ndarray:
    p = np.ones(table.n_agents, dtype=np.float64) / float(table.n_agents)
    endpoint_window: deque[np.ndarray] = deque(maxlen=max(1, int(last_k)))
    endpoint_window.append(p.copy())
    eta = 0.0 if params.no_feedback else float(params.eta)
    for t_idx in context_indices:
        fitness = replay.fitness_at(table, int(t_idx), p, params, support_clip=support_clip)
        mask = None
        if params.hard_safety_threshold >= 0:
            mask = table.safety_score[int(t_idx)] >= float(params.hard_safety_threshold)
        p = replay.softmax_exposure_update(p, fitness, eta, mask)
        endpoint_window.append(p.copy())
    endpoint = np.mean(np.vstack(list(endpoint_window)), axis=0)
    endpoint = np.clip(endpoint, replay.EPS, None)
    return endpoint / endpoint.sum()


def stratified_later_holdout(
    table,
    later_indices: Sequence[int],
    *,
    holdout_every: int,
    context_key: str,
) -> list[int]:
    if holdout_every < 2:
        raise ValueError("holdout_every must be at least 2")
    if context_key in table.context_df.columns:
        categories = table.context_df[context_key].fillna("uncategorized").astype(str).tolist()
    else:
        categories = ["all"] * table.n_timesteps
    ranks: dict[str, int] = {}
    heldout: list[int] = []
    for t_idx in later_indices:
        category = categories[int(t_idx)]
        rank = ranks.get(category, 0)
        # Reserve the last member of each deterministic block. This avoids
        # systematically selecting the first post-early occurrence.
        if rank % holdout_every == holdout_every - 1:
            heldout.append(int(t_idx))
        ranks[category] = rank + 1
    if not heldout:
        raise ValueError("Context holdout is empty")
    return heldout


def stability_subset(
    table,
    endpoint,
    params,
    indices: Sequence[int],
    args,
    *,
    context_stride: int,
) -> dict:
    return replay.estimate_map_stability(
        table,
        endpoint,
        params,
        active_eps=float(args.active_eps),
        fd_eps=float(args.fd_eps),
        invasion_eps=float(args.invasion_eps),
        support_clip=float(args.support_clip),
        context_stride=int(context_stride),
        max_active=int(args.max_active),
        context_indices=indices,
    )


def parameter_keys() -> list[str]:
    return [
        "eta", "alpha", "beta_sup", "beta_route", "lambda_risk",
        "hard_safety_threshold", "context_trait_strength",
        "polarization_strength", "route_power", "route_zscore",
        "no_feedback", "ablation",
    ]


def process_run(args: argparse.Namespace) -> None:
    started = time.time()
    run_path = Path(args.run_path).resolve()
    feature_csv = Path(args.feature_csv).resolve()
    output_csv = Path(args.output_csv).resolve()
    config_path = Path(args.run_config).resolve() if args.run_config else feature_csv.parent / "run_config.json"

    features = pd.read_csv(feature_csv)
    if "early_frac" in features.columns:
        early_values = pd.to_numeric(features["early_frac"], errors="coerce")
        features = features[np.isclose(early_values, float(args.early_frac))].copy()
    if features.empty:
        raise ValueError(f"No rows for early_frac={args.early_frac} in {feature_csv}")
    features = features.drop_duplicates(parameter_keys()).reset_index(drop=True)
    if int(args.max_conditions) > 0:
        features = features.iloc[: int(args.max_conditions)].copy()

    score_weights = replay.parse_score_weights("")
    endpoint_last_k = int(args.endpoint_last_k)
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if isinstance(config.get("score_weights"), dict):
            score_weights = {str(k): float(v) for k, v in config["score_weights"].items()}
        if endpoint_last_k <= 0:
            endpoint_last_k = int(config.get("args", {}).get("endpoint_last_k", 50))
    if endpoint_last_k <= 0:
        endpoint_last_k = 50

    run_name = run_path.name
    table = replay.prepare_log_streaming(
        run_path,
        run_name=run_name,
        score_weights=score_weights,
        max_timesteps=int(args.max_timesteps),
    )
    early_cutoff = max(2, int(math.ceil(table.n_timesteps * float(args.early_frac))))
    early_indices = list(range(0, min(table.n_timesteps, early_cutoff + 1)))
    later_indices = list(range(min(table.n_timesteps, early_cutoff + 1), table.n_timesteps))
    heldout_indices = stratified_later_holdout(
        table,
        later_indices,
        holdout_every=int(args.holdout_every),
        context_key=str(args.context_key),
    )
    heldout_set = set(heldout_indices)
    evolution_indices = early_indices + [t for t in later_indices if t not in heldout_set]

    print(
        f"run={run_name} agents={table.n_agents} contexts={table.n_timesteps} "
        f"early={len(early_indices)} later={len(later_indices)} "
        f"heldout={len(heldout_indices)} evolution={len(evolution_indices)} "
        f"conditions={len(features)} endpoint_last_k={endpoint_last_k}",
        flush=True,
    )

    rows: list[dict] = []
    for row_idx, feature_row in features.iterrows():
        params = params_from_row(feature_row)
        baseline_endpoint, _desc, _traj, _endpoint_rows, _snapshots = replay.replay_one_condition(
            table,
            params,
            last_k=endpoint_last_k,
            context_key=str(args.context_key),
            support_clip=float(args.support_clip),
            collect_trajectory=False,
            trajectory_stride=1,
        )
        postearly = stability_subset(
            table, baseline_endpoint, params, later_indices, args,
            context_stride=int(args.postearly_context_stride),
        )

        heldout_endpoint = endpoint_from_contexts(
            table,
            params,
            evolution_indices,
            last_k=endpoint_last_k,
            support_clip=float(args.support_clip),
        )
        heldout = stability_subset(
            table, heldout_endpoint, params, heldout_indices, args,
            context_stride=int(args.holdout_context_stride),
        )

        rec = {key: feature_row.get(key) for key in parameter_keys()}
        rec.update({
            "run_name": run_name,
            "seed": seed_from_name(run_name),
            "early_frac": float(args.early_frac),
            "early_cutoff_index": int(early_cutoff),
            "n_contexts_total": int(table.n_timesteps),
            "n_contexts_early": int(len(early_indices)),
            "n_contexts_postearly": int(len(later_indices)),
            "n_contexts_holdout": int(len(heldout_indices)),
            "n_contexts_evolution": int(len(evolution_indices)),
            "endpoint_last_k": int(endpoint_last_k),
            "baseline_kappa_map": float(feature_row.get("kappa_map", np.nan)),
            "postearly_kappa_map": float(postearly["kappa_map"]),
            "postearly_gamma_map": float(postearly["gamma_map"]),
            "postearly_max_invasion_log": float(postearly["max_invasion_log"]),
            "postearly_active_count": int(postearly["active_count"]),
            "postearly_context_count": int(postearly["stability_context_count"]),
            "context_holdout_kappa_map": float(heldout["kappa_map"]),
            "context_holdout_gamma_map": float(heldout["gamma_map"]),
            "context_holdout_max_invasion_log": float(heldout["max_invasion_log"]),
            "context_holdout_active_count": int(heldout["active_count"]),
            "context_holdout_context_count": int(heldout["stability_context_count"]),
        })
        rows.append(rec)
        if (row_idx + 1) % int(args.checkpoint_every) == 0 or row_idx + 1 == len(features):
            atomic_csv(pd.DataFrame(rows), output_csv)
            elapsed = time.time() - started
            print(f"  {row_idx + 1}/{len(features)} conditions, elapsed={elapsed:.1f}s", flush=True)

    metadata = {
        "run_path": str(run_path),
        "feature_csv": str(feature_csv),
        "output_csv": str(output_csv),
        "early_frac": float(args.early_frac),
        "early_cutoff_index": int(early_cutoff),
        "endpoint_last_k": int(endpoint_last_k),
        "holdout_rule": (
            f"within each {args.context_key} stratum after the early window, "
            f"reserve every {args.holdout_every}th context"
        ),
        "postearly_context_stride_within_selected_subset": int(args.postearly_context_stride),
        "holdout_context_stride_within_selected_subset": int(args.holdout_context_stride),
        "elapsed_sec": time.time() - started,
        "n_conditions": len(rows),
    }
    output_csv.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def model_feature_columns(df: pd.DataFrame, model_name: str) -> list[str]:
    params = [
        "eta", "alpha", "beta_sup", "beta_route", "lambda_risk",
        "hard_safety_threshold", "context_trait_strength",
        "polarization_strength", "route_power", "route_zscore", "no_feedback",
    ]
    descriptors = [c for c in df.columns if c.endswith(("_early_mean", "_early_last", "_early_slope"))]
    fluctuations = [c for c in df.columns if c.endswith(("_early_var", "_early_lag1", "_early_accel"))]
    if model_name == "parameter_only":
        cols = params
    elif model_name == "early_descriptor":
        cols = descriptors
    elif model_name == "fluctuation_only":
        cols = fluctuations
    elif model_name == "full_early_warning":
        cols = params + descriptors + fluctuations
    else:
        cols = []
    return [c for c in cols if c in df.columns]


def auc_manual(y: np.ndarray, score: np.ndarray) -> float:
    return float(replay._roc_auc_manual(np.asarray(y, dtype=int), np.asarray(score, dtype=float)))


def se(values: Iterable[float]) -> float:
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    return float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else float("nan")


def evaluate(args: argparse.Namespace) -> None:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    features = pd.read_csv(args.features_csv)
    if "early_frac" in features.columns:
        values = pd.to_numeric(features["early_frac"], errors="coerce")
        features = features[np.isclose(values, float(args.early_frac))].copy()
    audit_files = sorted(Path(args.audit_dir).glob("*/context_disjoint_criticality.csv"))
    if not audit_files:
        audit_files = sorted(Path(args.audit_dir).glob("context_disjoint_criticality*.csv"))
    if not audit_files:
        raise FileNotFoundError(f"No context-disjoint audit CSVs under {args.audit_dir}")
    audit = pd.concat([pd.read_csv(path) for path in audit_files], ignore_index=True, sort=False)

    join_keys = [key for key in parameter_keys() if key in features.columns and key in audit.columns]
    feature_group = "source_result_dir" if "source_result_dir" in features.columns else "run_name"
    features = features.copy()
    features["_seed"] = features.get("seed", features[feature_group].astype(str).map(seed_from_name))
    audit["_seed"] = audit.get("seed", audit["run_name"].astype(str).map(seed_from_name))
    merged = features.merge(
        audit[join_keys + ["_seed", "baseline_kappa_map", "postearly_kappa_map", "context_holdout_kappa_map"]],
        on=join_keys + ["_seed"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("Feature/audit merge produced no rows")

    targets = {
        "baseline": "baseline_kappa_map",
        "postearly": "postearly_kappa_map",
        "context_holdout": "context_holdout_kappa_map",
    }
    models = ["parameter_only", "early_descriptor", "fluctuation_only", "full_early_warning"]
    groups = pd.to_numeric(merged["_seed"], errors="raise").astype(int).to_numpy()
    predictions: list[dict] = []
    summaries: list[dict] = []

    for target_name, kappa_col in targets.items():
        y = (pd.to_numeric(merged[kappa_col], errors="coerce") > 0).astype(int).to_numpy()
        for model_name in models:
            cols = model_feature_columns(merged, model_name)
            X = merged[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
            scores = np.full(len(merged), np.nan, dtype=float)
            fold_aucs: list[float] = []
            for heldout_seed in sorted(np.unique(groups)):
                test = np.where(groups == heldout_seed)[0]
                train = np.where(groups != heldout_seed)[0]
                if len(np.unique(y[train])) < 2:
                    scores[test] = float(np.mean(y[train]))
                else:
                    model = Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("model", LogisticRegression(
                            max_iter=2000, class_weight="balanced", solver="liblinear",
                        )),
                    ])
                    model.fit(X.iloc[train], y[train])
                    scores[test] = model.predict_proba(X.iloc[test])[:, 1]
                fold_auc = auc_manual(y[test], scores[test])
                fold_aucs.append(fold_auc)
                for idx in test:
                    predictions.append({
                        "seed": int(groups[idx]),
                        "target": target_name,
                        "model": model_name,
                        "y_true": int(y[idx]),
                        "y_score": float(scores[idx]),
                    })
            summaries.append({
                "target": target_name,
                "model": model_name,
                "n": int(len(y)),
                "n_runs": int(len(np.unique(groups))),
                "positive_rate": float(np.mean(y)),
                "pooled_roc_auc": auc_manual(y, scores),
                "run_mean_roc_auc": float(np.nanmean(fold_aucs)),
                "run_se_roc_auc": se(fold_aucs),
                "n_features": int(len(cols)),
            })

    comparison: list[dict] = []
    baseline_kappa = pd.to_numeric(merged["baseline_kappa_map"], errors="coerce")
    baseline_label = baseline_kappa > 0
    for target_name, kappa_col in targets.items():
        kappa = pd.to_numeric(merged[kappa_col], errors="coerce")
        label = kappa > 0
        comparison.append({
            "target": target_name,
            "n": int(len(merged)),
            "positive_rate": float(label.mean()),
            "label_agreement_with_baseline": float((label == baseline_label).mean()),
            "kappa_pearson_with_baseline": float(kappa.corr(baseline_kappa, method="pearson")),
            "kappa_spearman_with_baseline": float(kappa.corr(baseline_kappa, method="spearman")),
            "median_kappa": float(kappa.median()),
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(merged, out_dir / "context_disjoint_features_and_targets.csv")
    atomic_csv(pd.DataFrame(predictions), out_dir / "leave_seed_out_predictions.csv")
    atomic_csv(pd.DataFrame(summaries), out_dir / "leave_seed_out_metrics.csv")
    atomic_csv(pd.DataFrame(comparison), out_dir / "target_comparison.csv")
    print(pd.DataFrame(comparison).to_string(index=False), flush=True)
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    process = sub.add_parser("process-run", help="Compute both context-disjoint κ targets for one run")
    process.add_argument("--run-path", required=True)
    process.add_argument("--feature-csv", required=True)
    process.add_argument("--output-csv", required=True)
    process.add_argument("--run-config", default="")
    process.add_argument("--early-frac", type=float, default=0.1)
    process.add_argument("--endpoint-last-k", type=int, default=0, help="0 reads the source run_config")
    process.add_argument("--holdout-every", type=int, default=5)
    process.add_argument("--context-key", default="topic_category")
    process.add_argument("--support-clip", type=float, default=8.0)
    process.add_argument("--active-eps", type=float, default=1e-4)
    process.add_argument("--fd-eps", type=float, default=1e-5)
    process.add_argument("--invasion-eps", type=float, default=1e-5)
    process.add_argument("--postearly-context-stride", type=int, default=4)
    process.add_argument("--holdout-context-stride", type=int, default=2)
    process.add_argument("--max-active", type=int, default=96)
    process.add_argument("--max-timesteps", type=int, default=0)
    process.add_argument("--max-conditions", type=int, default=0)
    process.add_argument("--checkpoint-every", type=int, default=10)
    process.set_defaults(func=process_run)

    evaluation = sub.add_parser("evaluate", help="Run strict leave-one-run-out classifiers")
    evaluation.add_argument("--features-csv", required=True)
    evaluation.add_argument("--audit-dir", required=True)
    evaluation.add_argument("--out-dir", required=True)
    evaluation.add_argument("--early-frac", type=float, default=0.1)
    evaluation.set_defaults(func=evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
