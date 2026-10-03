#!/usr/bin/env python3
"""Context-dependence audits of the early-warning result (Appendix B.3).

    python3 pipeline/early_warning/context.py holdout   leakage controls: kappa target recomputed on post-window / strictly held-out contexts -> data/outputs/early_warning_context_holdout
    python3 pipeline/early_warning/context.py shift     abrupt context shifts at stable operating points (OOD audit)                           -> data/testbeds/context_shift_early_warning_10seed
"""

from __future__ import annotations

import sys
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import math
from collections import deque
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # pipeline/ on the path

from replay import engine as replay  # noqa: E402


# ====================================================================================================
# holdout
# ====================================================================================================
MODELS = ["parameter_only", "early_descriptor", "fluctuation_only", "full_early_warning"]
MODEL_TEX = {
    "parameter_only": "Parameter only",
    "early_descriptor": "Early descriptors",
    "fluctuation_only": "Fluctuation only",
    "full_early_warning": "Full early warning",
}
TARGETS = {
    "baseline": "baseline_kappa_map",
    "postearly": "postearly_kappa_map",
    "context_holdout": "context_holdout_kappa_map",
}
PARAMS = [
    "eta", "alpha", "beta_sup", "beta_route", "lambda_risk",
    "hard_safety_threshold", "context_trait_strength",
    "polarization_strength", "route_power", "route_zscore", "no_feedback",
]
def model_feature_columns(df: pd.DataFrame, model_name: str) -> list[str]:
    descriptors = [c for c in df.columns if c.endswith(("_early_mean", "_early_last", "_early_slope"))]
    fluctuations = [c for c in df.columns if c.endswith(("_early_var", "_early_lag1", "_early_accel"))]
    cols = {
        "parameter_only": PARAMS,
        "early_descriptor": descriptors,
        "fluctuation_only": fluctuations,
        "full_early_warning": PARAMS + descriptors + fluctuations,
    }[model_name]
    return [c for c in cols if c in df.columns]
def se(v) -> float:
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
def loso(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = df["_seed"].astype(int).to_numpy()
    preds, per_seed, summary = [], [], []
    for tname, kcol in TARGETS.items():
        y = (pd.to_numeric(df[kcol], errors="coerce") > 0).astype(int).to_numpy()
        for m in MODELS:
            cols = model_feature_columns(df, m)
            X = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)
            score = np.full(len(df), np.nan)
            aucs = []
            for s in sorted(np.unique(groups)):
                te, tr = np.where(groups == s)[0], np.where(groups != s)[0]
                pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")),
                ])
                pipe.fit(X.iloc[tr], y[tr])
                score[te] = pipe.predict_proba(X.iloc[te])[:, 1]
                a = roc_auc_score(y[te], score[te])
                aucs.append(a)
                per_seed.append({"target": tname, "model": m, "heldout_seed": int(s),
                                 "n": len(te), "n_positive": int(y[te].sum()), "roc_auc": a})
            preds.append(pd.DataFrame({"row": np.arange(len(df)), "seed": groups, "target": tname,
                                       "model": m, "y_true": y, "y_score": score}))
            summary.append({"target": tname, "model": m, "n": len(y), "n_seeds": len(aucs),
                            "positive_rate": float(y.mean()), "n_features": len(cols),
                            "pooled_loso_roc_auc": roc_auc_score(y, score),
                            "seed_mean_roc_auc": float(np.mean(aucs)), "seed_se_roc_auc": se(aucs),
                            "seed_min_roc_auc": float(np.min(aucs)), "seed_max_roc_auc": float(np.max(aucs))})
    return pd.concat(preds, ignore_index=True), pd.DataFrame(per_seed), pd.DataFrame(summary)
def main_holdout():
    ap = argparse.ArgumentParser(description=__doc__)
    base = "data/results/context_disjoint_criticality_full_holdout"
    ap.add_argument("--merged_csv", default=f"{base}/summary/context_disjoint_features_and_targets.csv")
    ap.add_argument("--saved_predictions", default=f"{base}/summary/leave_seed_out_predictions.csv")
    ap.add_argument("--audit_dir", default=base)
    ap.add_argument("--stride_csv", default=f"{base}/summary/heldout_stride_sensitivity.csv")
    ap.add_argument("--out_dir", default="data/outputs/early_warning_context_holdout")
    ap.add_argument("--target", default="context_holdout")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.merged_csv)
    pred, per_seed, summary = loso(df)

    # --- check against the stored predictions ---------------
    saved = pd.read_csv(args.saved_predictions)
    check = []
    for (t, m), g in saved.groupby(["target", "model"]):
        mine = pred[(pred.target == t) & (pred.model == m)]
        check.append({"target": t, "model": m,
                      "saved_pooled_auc": roc_auc_score(g.y_true, g.y_score),
                      "recomputed_pooled_auc": roc_auc_score(mine.y_true, mine.y_score),
                      "labels_identical": bool(np.array_equal(g.y_true.to_numpy(), mine.y_true.to_numpy())),
                      "max_abs_score_diff": float(np.max(np.abs(g.y_score.to_numpy() - mine.y_score.to_numpy())))})
    check = pd.DataFrame(check)

    # --- target agreement ------------------------------------------------------
    kb = df["baseline_kappa_map"]
    lb = kb > 0
    cmp_rows, flip_rows, seed_rows = [], [], []
    for t, kcol in TARGETS.items():
        k = df[kcol]
        lab = k > 0
        ch = lab != lb
        cmp_rows.append({"target": t, "n": len(df), "positive_rate": float(lab.mean()),
                         "label_agreement_with_baseline": float((~ch).mean()),
                         "kappa_pearson_with_baseline": float(k.corr(kb)),
                         "kappa_spearman_with_baseline": float(k.corr(kb, method="spearman"))})
        flip_rows.append({"target": t, "n_flips": int(ch.sum()), "to_positive": int((ch & lab).sum()),
                          "to_negative": int((ch & ~lab).sum()),
                          "median_abs_baseline_kappa": float(kb[ch].abs().median()) if ch.any() else np.nan,
                          "median_abs_baseline_kappa_all": float(kb.abs().median())})
        for s, g in df.groupby("_seed"):
            seed_rows.append({"seed": int(s), "target": t,
                              "label_agreement": float(((g[kcol] > 0) == (g["baseline_kappa_map"] > 0)).mean()),
                              "n_flips": int(((g[kcol] > 0) != (g["baseline_kappa_map"] > 0)).sum())})
    cmp_df, flips, by_seed = pd.DataFrame(cmp_rows), pd.DataFrame(flip_rows), pd.DataFrame(seed_rows)

    # --- context counts (verified from per-run audit CSVs) --------------------
    audit = pd.concat([pd.read_csv(p) for p in sorted(Path(args.audit_dir).glob("*/context_disjoint_criticality.csv"))])
    def uniq(col):
        v = audit[col].unique()
        assert len(v) == 1, (col, v)
        return int(v[0])
    stride = pd.read_csv(args.stride_csv).iloc[0].to_dict()
    stride.update({"n_ctx_total": uniq("n_contexts_total"), "n_ctx_early": uniq("n_contexts_early"),
                   "n_ctx_holdout": uniq("n_contexts_holdout"), "n_ctx_evolution": uniq("n_contexts_evolution"),
                   "n_ctx_holdout_used": uniq("context_holdout_context_count"),
                   "n_ctx_postearly_used": uniq("postearly_context_count")})

    per_seed.to_csv(out / "loso_per_seed_auc.csv", index=False)
    summary.to_csv(out / "loso_auc_summary.csv", index=False)
    check.to_csv(out / "check_vs_saved_predictions.csv", index=False)
    cmp_df.to_csv(out / "target_agreement.csv", index=False)
    flips.to_csv(out / "label_flips.csv", index=False)
    by_seed.to_csv(out / "target_agreement_by_seed.csv", index=False)
    pred.to_csv(out / "loso_predictions.csv.gz", index=False)
    (out / "context_counts_and_stride.json").write_text(json.dumps(stride, indent=2))


    pd.set_option("display.width", 250)
    print(summary.round(4).to_string(index=False))
    print(check.round(6).to_string(index=False))
    print(cmp_df.round(4).to_string(index=False))
    print(flips.to_string(index=False))
    print(json.dumps(stride, indent=1))


# ====================================================================================================
# shift
# ====================================================================================================
DEFAULT_TRAIN = Path("data/results/early_warning_validation_grid_leakage_controls/early_warning_features.csv")
DEFAULT_OUT = Path("data/testbeds/context_shift_early_warning_10seed")
DEFAULT_SOURCES = Path("runs")
MODEL_NAMES = [
    "parameter_only",
    "early_descriptor",
    "fluctuation_only",
    "full_early_warning",
]
CONTEXT_AXES = ["interpersonalness", "evidentiality"]
FEATURE_SUFFIXES = (
    "_early_mean",
    "_early_last",
    "_early_slope",
    "_early_var",
    "_early_lag1",
    "_early_accel",
)
def parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]
def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
def params_from_row(row: Mapping[str, Any]) -> replay.ReplayParams:
    return replay.ReplayParams(
        eta=float(row["eta"]),
        alpha=float(row["alpha"]),
        beta_sup=float(row["beta_sup"]),
        beta_route=float(row["beta_route"]),
        lambda_risk=float(row["lambda_risk"]),
        hard_safety_threshold=float(row.get("hard_safety_threshold", -1.0)),
        context_trait_strength=float(row.get("context_trait_strength", 0.0)),
        route_power=float(row.get("route_power", 1.0)),
        route_zscore=bool(row.get("route_zscore", False)),
        no_feedback=bool(row.get("no_feedback", False)),
        ablation=str(row.get("ablation", "none")),
        polarization_strength=float(row.get("polarization_strength", 0.0)),
    )
def stable_operating_points(
    train: pd.DataFrame,
    per_seed: int,
    random_state: int,
) -> pd.DataFrame:
    stable = train.loc[pd.to_numeric(train["label_unstable"], errors="coerce").eq(0)].copy()
    stable["kappa_map"] = pd.to_numeric(stable["kappa_map"], errors="coerce")
    stable = stable.dropna(subset=["kappa_map"])
    selected: List[pd.DataFrame] = []
    rng = np.random.default_rng(int(random_state))
    for _seed, group in stable.groupby("run_name", sort=True):
        group = group.sort_values("kappa_map").reset_index(drop=True)
        n = min(int(per_seed), len(group))
        if n <= 0:
            continue
        # Cover deep, middle, and near-boundary stable points without selecting
        # only the easiest or most fragile cases.
        positions = np.linspace(0, len(group) - 1, n)
        positions = np.unique(np.rint(positions).astype(int))
        if len(positions) < n:
            remaining = np.setdiff1d(np.arange(len(group)), positions)
            positions = np.r_[positions, rng.choice(remaining, n - len(positions), replace=False)]
        selected.append(group.iloc[np.sort(positions[:n])])
    if not selected:
        raise ValueError("No locally contracting operating points in the training table.")
    return pd.concat(selected, ignore_index=True)
def context_subsets(
    table: replay.PreparedLog,
    axis: str,
    magnitude: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    values = pd.to_numeric(table.context_df[axis], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values)
    if finite.sum() < 4:
        raise ValueError(f"Insufficient finite values for context axis {axis}.")
    q = float(np.clip(magnitude, 0.5, 0.95))
    lower = float(np.nanquantile(values[finite], 1.0 - q))
    upper = float(np.nanquantile(values[finite], q))
    low = np.flatnonzero(finite & (values <= lower))
    high = np.flatnonzero(finite & (values >= upper))
    if low.size == 0 or high.size == 0:
        raise ValueError(f"Empty context subset for {axis} at q={q}.")
    pooled_sd = float(np.nanstd(values[finite]))
    standardized_gap = (
        float(np.nanmean(values[high]) - np.nanmean(values[low]))
        / max(pooled_sd, replay.EPS)
    )
    return low, high, {
        "context_low_mean": float(np.nanmean(values[low])),
        "context_high_mean": float(np.nanmean(values[high])),
        "context_standardized_gap": standardized_gap,
        "context_low_n": int(low.size),
        "context_high_n": int(high.size),
    }
def subset_table(table: replay.PreparedLog, indices: Sequence[int]) -> replay.PreparedLog:
    idx = np.asarray(indices, dtype=int)
    idx = idx[(idx >= 0) & (idx < table.n_timesteps)]
    if idx.size == 0:
        raise ValueError("Cannot construct an empty post-shift context law.")
    base = table.base_score[idx].copy()
    risk = table.risk_penalty[idx].copy()
    safety = table.safety_score[idx].copy()
    quality = table.quality_score[idx].copy()
    return replay.PreparedLog(
        run_name=table.run_name,
        source_path=table.source_path,
        agent_ids=list(table.agent_ids),
        timesteps=list(range(len(idx))),
        base_score=base,
        agent_mean_score=np.asarray(base.mean(axis=0), dtype=np.float64),
        risk_penalty=risk,
        safety_score=safety,
        quality_score=quality,
        agent_mean_risk=np.asarray(risk.mean(axis=0), dtype=np.float64),
        agent_mean_safety=np.asarray(safety.mean(axis=0), dtype=np.float64),
        agent_mean_quality=np.asarray(quality.mean(axis=0), dtype=np.float64),
        trait_coords=table.trait_coords.copy(),
        context_df=table.context_df.iloc[idx].reset_index(drop=True).copy(),
        metadata=dict(table.metadata),
    )
def simulate(
    table: replay.PreparedLog,
    params_pre: replay.ReplayParams,
    params_post: replay.ReplayParams,
    pre_indices: Sequence[int],
    post_indices: Sequence[int],
    shift_step: int,
    horizon: int,
    endpoint_last_k: int,
    support_clip: float,
) -> Tuple[pd.DataFrame, np.ndarray]:
    p = np.ones(table.n_agents, dtype=np.float64) / float(table.n_agents)
    endpoint_window: deque[np.ndarray] = deque(maxlen=max(1, int(endpoint_last_k)))
    rows: List[Dict[str, Any]] = []
    pre = np.asarray(pre_indices, dtype=int)
    post = np.asarray(post_indices, dtype=int)
    if pre.size == 0 or post.size == 0:
        raise ValueError("Both pre- and post-shift context subsets must be nonempty.")

    for step in range(int(horizon) + 1):
        is_post = step >= int(shift_step)
        pool = post if is_post else pre
        local = step - int(shift_step) if is_post else step
        t_idx = int(pool[int(local) % len(pool)])
        row = replay.trajectory_row(table, p, timestep=step, step_index=step)
        row.update(
            {
                "is_post_shift": int(is_post),
                "source_t_idx": t_idx,
                "context_axis_value": float("nan"),
            }
        )
        rows.append(row)
        if step >= int(horizon):
            break
        params = params_post if is_post else params_pre
        f = replay.fitness_at(table, t_idx, p, params, support_clip=float(support_clip))
        mask = None
        if params.hard_safety_threshold >= 0:
            mask = table.safety_score[t_idx] >= float(params.hard_safety_threshold)
        eta = 0.0 if params.no_feedback else float(params.eta)
        p = replay.softmax_exposure_update(p, f, eta, mask)
        endpoint_window.append(p.copy())

    endpoint = np.mean(np.vstack(endpoint_window), axis=0)
    endpoint = np.clip(endpoint, replay.EPS, None)
    endpoint = endpoint / endpoint.sum()
    return pd.DataFrame(rows), endpoint
def natural_control(
    table: replay.PreparedLog,
    params: replay.ReplayParams,
    horizon: int,
    endpoint_last_k: int,
    support_clip: float,
) -> Tuple[pd.DataFrame, np.ndarray]:
    indices = np.arange(table.n_timesteps, dtype=int)
    return simulate(
        table,
        params,
        params,
        indices,
        indices,
        shift_step=int(horizon) + 1,
        horizon=horizon,
        endpoint_last_k=endpoint_last_k,
        support_clip=support_clip,
    )
def feature_window(
    trajectory: pd.DataFrame,
    start: int,
    width: int,
) -> Dict[str, float]:
    window = trajectory.iloc[int(start) : int(start) + int(width)].copy()
    if len(window) < int(width):
        raise ValueError("Feature window extends past the simulated trajectory.")
    window["step_index"] = np.arange(len(window), dtype=int)
    out = replay.early_warning_features_for_group(window, early_frac=1.0)
    out["window_start"] = int(start)
    out["window_n_steps"] = int(len(window))
    return out
def add_parameter_fields(
    features: Dict[str, Any],
    params: replay.ReplayParams,
) -> Dict[str, Any]:
    out = dict(features)
    for key in [
        "eta",
        "alpha",
        "beta_sup",
        "beta_route",
        "lambda_risk",
        "hard_safety_threshold",
        "context_trait_strength",
        "polarization_strength",
        "route_power",
        "route_zscore",
        "no_feedback",
        "ablation",
    ]:
        out[key] = getattr(params, key)
    return out
def estimate_post_stability(
    post_table: replay.PreparedLog,
    endpoint: np.ndarray,
    params: replay.ReplayParams,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    return replay.estimate_map_stability(
        post_table,
        endpoint,
        params,
        active_eps=float(args.active_eps),
        fd_eps=float(args.fd_eps),
        invasion_eps=float(args.invasion_eps),
        support_clip=float(args.support_clip),
        context_stride=int(args.stability_context_stride),
        max_active=int(args.stability_max_active),
    )
def model_pipeline() -> Any:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        ),
    )
def numeric_matrix(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.reindex(columns=list(columns)).copy()
    for col in out:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)
def fit_models(
    train: pd.DataFrame,
    target_fpr: float,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    from sklearn.model_selection import GroupKFold

    y = pd.to_numeric(train["label_unstable"], errors="coerce").fillna(0).astype(int).to_numpy()
    groups = train["run_name"].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("At least two stationary training groups are required.")
    splitter = GroupKFold(n_splits=len(unique_groups))
    models: Dict[str, Any] = {}
    metadata: List[Dict[str, Any]] = []
    for model_name in MODEL_NAMES:
        cols = replay._feature_columns_for_model(train, model_name)
        if not cols:
            continue
        X = numeric_matrix(train, cols)
        oof = np.full(len(train), np.nan, dtype=float)
        for train_idx, test_idx in splitter.split(X, y, groups):
            fold_model = model_pipeline()
            fold_model.fit(X.iloc[train_idx], y[train_idx])
            oof[test_idx] = fold_model.predict_proba(X.iloc[test_idx])[:, 1]
        stable_scores = oof[y == 0]
        stable_scores = stable_scores[np.isfinite(stable_scores)]
        threshold = float(
            np.quantile(
                stable_scores,
                1.0 - float(target_fpr),
                method="higher",
            )
        )
        model = model_pipeline()
        model.fit(X, y)
        models[model_name] = {"model": model, "features": cols, "threshold": threshold}
        metadata.append(
            {
                "model": model_name,
                "n_features": len(cols),
                "threshold_stationary_fpr_target": float(target_fpr),
                "threshold": threshold,
                "stationary_oof_fpr": float(np.mean(stable_scores >= threshold)),
                "stationary_oof_mean_score_stable": float(np.mean(stable_scores)),
                "stationary_oof_mean_score_unstable": float(np.nanmean(oof[y == 1])),
            }
        )
    return models, pd.DataFrame(metadata)
def predict_rows(
    features: pd.DataFrame,
    models: Mapping[str, Any],
) -> pd.DataFrame:
    id_cols = [c for c in features.columns if not c.endswith(FEATURE_SUFFIXES)]
    rows: List[Dict[str, Any]] = []
    for model_name, payload in models.items():
        score = payload["model"].predict_proba(
            numeric_matrix(features, payload["features"])
        )[:, 1]
        for idx, value in enumerate(score):
            row = features.iloc[idx][id_cols].to_dict()
            row.update(
                {
                    "model": model_name,
                    "y_score": float(value),
                    "threshold": float(payload["threshold"]),
                    "alert_calibrated": int(value >= float(payload["threshold"])),
                    "alert_0_5": int(value >= 0.5),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)
def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan")
def safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, score)) if np.sum(y == 1) else float("nan")
def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = float(successes) / float(total)
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = (
        z
        * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
        / denom
    )
    return max(0.0, center - half), min(1.0, center + half)
def evaluation_tables(pred: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.metrics import brier_score_loss

    pred = pred.copy()
    if "pair_id" not in pred.columns:
        pred["pair_id"] = (
            pred["scenario_id"]
            .astype(str)
            .str.replace(r"_pure$", "", regex=True)
            .str.replace(r"_eta_x[^_]+$", "", regex=True)
        )
    stable_pairs = set(
        pred.loc[pred["condition"].eq("pure_shift_stable"), "pair_id"].astype(str)
    )
    eligible = pred.loc[
        pred["condition"].eq("pure_shift_stable")
        | (
            pred["condition"].eq("shift_plus_loss")
            & pred["pair_id"].astype(str).isin(stable_pairs)
        )
    ].copy()
    eligible["y_true"] = eligible["condition"].eq("shift_plus_loss").astype(int)
    controls = pred.loc[pred["condition"].eq("stationary_stable_control")].copy()
    overall_rows: List[Dict[str, Any]] = []
    group_cols = ["model", "window"]
    for keys, group in eligible.groupby(group_cols, dropna=False):
        y = group["y_true"].to_numpy(dtype=int)
        score = group["y_score"].to_numpy(dtype=float)
        stable = group["y_true"].eq(0)
        unstable = ~stable
        model_controls = controls.loc[controls["model"].eq(keys[0])]
        fpr_low, fpr_high = wilson_interval(
            int(group.loc[stable, "alert_calibrated"].sum()), int(stable.sum())
        )
        tpr_low, tpr_high = wilson_interval(
            int(group.loc[unstable, "alert_calibrated"].sum()), int(unstable.sum())
        )
        overall_rows.append(
            {
                "model": keys[0],
                "window": keys[1],
                "n": len(group),
                "n_stable_shift": int(stable.sum()),
                "n_shift_plus_loss": int(unstable.sum()),
                "roc_auc": safe_auc(y, score),
                "pr_auc": safe_ap(y, score),
                "brier": float(brier_score_loss(y, score)),
                "pure_shift_fpr_calibrated": float(group.loc[stable, "alert_calibrated"].mean()),
                "pure_shift_fpr_ci_low": fpr_low,
                "pure_shift_fpr_ci_high": fpr_high,
                "pure_shift_fpr_0_5": float(group.loc[stable, "alert_0_5"].mean()),
                "stationary_control_fpr_calibrated": float(
                    model_controls["alert_calibrated"].mean()
                ),
                "stationary_control_fpr_0_5": float(model_controls["alert_0_5"].mean()),
                "loss_tpr_calibrated": float(group.loc[unstable, "alert_calibrated"].mean()),
                "loss_tpr_ci_low": tpr_low,
                "loss_tpr_ci_high": tpr_high,
                "loss_tpr_0_5": float(group.loc[unstable, "alert_0_5"].mean()),
                "mean_score_pure_shift": float(group.loc[stable, "y_score"].mean()),
                "mean_score_shift_plus_loss": float(group.loc[unstable, "y_score"].mean()),
            }
        )
    overall = pd.DataFrame(overall_rows)

    strata_rows: List[Dict[str, Any]] = []
    strata_cols = ["model", "window", "context_axis", "shift_magnitude"]
    for keys, group in eligible.groupby(strata_cols, dropna=False):
        y = group["y_true"].to_numpy(dtype=int)
        score = group["y_score"].to_numpy(dtype=float)
        stable = group["y_true"].eq(0)
        unstable = ~stable
        strata_rows.append(
            {
                "model": keys[0],
                "window": keys[1],
                "context_axis": keys[2],
                "shift_magnitude": keys[3],
                "n": len(group),
                "roc_auc": safe_auc(y, score),
                "pr_auc": safe_ap(y, score),
                "brier": float(brier_score_loss(y, score)),
                "pure_shift_fpr_calibrated": float(group.loc[stable, "alert_calibrated"].mean()),
                "loss_tpr_calibrated": float(group.loc[unstable, "alert_calibrated"].mean()),
                "mean_score_pure_shift": float(group.loc[stable, "y_score"].mean()),
                "mean_score_shift_plus_loss": float(group.loc[unstable, "y_score"].mean()),
            }
        )
    return overall, pd.DataFrame(strata_rows)
def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.train_features)
    train = train.loc[
        np.isclose(
            pd.to_numeric(train["early_frac"], errors="coerce"),
            float(args.early_frac),
        )
    ].reset_index(drop=True)
    if train.empty:
        raise ValueError(f"No training rows at early_frac={args.early_frac}.")
    models, model_meta = fit_models(train, target_fpr=float(args.target_stationary_fpr))
    model_meta.to_csv(out_dir / "model_thresholds.csv", index=False)

    operating_points = stable_operating_points(
        train,
        per_seed=int(args.stable_points_per_seed),
        random_state=int(args.random_state),
    )
    operating_points.to_csv(out_dir / "selected_stable_operating_points.csv", index=False)

    score_weights = replay.parse_score_weights("")
    tables: Dict[str, replay.PreparedLog] = {}
    feature_rows: List[Dict[str, Any]] = []
    scenario_rows: List[Dict[str, Any]] = []
    magnitudes = parse_float_list(args.shift_magnitudes)
    eta_multipliers = parse_float_list(args.eta_multipliers)
    axes = [x.strip() for x in args.context_axes.split(",") if x.strip()]
    window_width = int(round(float(args.early_frac) * int(args.horizon))) + 1
    if int(args.shift_step) >= window_width:
        raise ValueError("shift_step must fall inside the crossing early window.")

    for op_idx, row in operating_points.iterrows():
        run_name = str(row["run_name"])
        if run_name not in tables:
            source = Path(args.sources_dir) / run_name
            if not source.exists() and not run_name.endswith(".checkpoints"):
                checkpoint_source = Path(args.sources_dir) / f"{run_name}.checkpoints"
                if checkpoint_source.exists():
                    source = checkpoint_source
            print(f"[load] {source}", flush=True)
            tables[run_name] = replay.prepare_log_streaming(
                source,
                run_name=run_name,
                score_weights=score_weights,
            )
        table = tables[run_name]
        base_params = params_from_row(row)

        control_traj, _ = natural_control(
            table,
            base_params,
            horizon=int(args.horizon),
            endpoint_last_k=int(args.endpoint_last_k),
            support_clip=float(args.support_clip),
        )
        control_features = add_parameter_fields(
            feature_window(control_traj, 0, window_width),
            base_params,
        )
        control_features.update(
            {
                "scenario_id": f"op{op_idx:03d}_stationary",
                "operating_point_id": int(op_idx),
                "run_name": run_name,
                "condition": "stationary_stable_control",
                "window": "crossing",
                "context_axis": "stationary",
                "shift_magnitude": 0.0,
                "context_standardized_gap": 0.0,
                "eta_multiplier": 1.0,
                "post_kappa_map": float(row["kappa_map"]),
            }
        )
        feature_rows.append(control_features)

        for axis in axes:
            if axis not in table.context_df.columns:
                raise ValueError(f"Context axis {axis} is absent from {run_name}.")
            for magnitude in magnitudes:
                low, high, shift_meta = context_subsets(table, axis, magnitude)
                post_table = subset_table(table, high)
                scenario_base = f"op{op_idx:03d}_{axis}_q{magnitude:.2f}"
                print(f"[{scenario_base}] pure shift", flush=True)
                pure_traj, pure_endpoint = simulate(
                    table,
                    base_params,
                    base_params,
                    low,
                    high,
                    shift_step=int(args.shift_step),
                    horizon=int(args.horizon),
                    endpoint_last_k=int(args.endpoint_last_k),
                    support_clip=float(args.support_clip),
                )
                pure_stability = estimate_post_stability(
                    post_table, pure_endpoint, base_params, args
                )
                pure_condition = (
                    "pure_shift_stable"
                    if float(pure_stability["kappa_map"]) < 0
                    else "context_induced_loss"
                )
                scenario_common = {
                    "pair_id": scenario_base,
                    "operating_point_id": int(op_idx),
                    "run_name": run_name,
                    "context_axis": axis,
                    "shift_magnitude": float(magnitude),
                    "base_eta": float(base_params.eta),
                    "base_beta_sup": float(base_params.beta_sup),
                    "base_kappa_map": float(row["kappa_map"]),
                    **shift_meta,
                }
                scenario_rows.append(
                    {
                        **scenario_common,
                        "scenario_id": scenario_base + "_pure",
                        "condition": pure_condition,
                        "eta_multiplier": 1.0,
                        "post_eta": float(base_params.eta),
                        "post_kappa_map": float(pure_stability["kappa_map"]),
                    }
                )
                for window_name, start in [
                    ("crossing", 0),
                    ("post_reset", int(args.shift_step)),
                ]:
                    feat = add_parameter_fields(
                        feature_window(pure_traj, start, window_width),
                        base_params,
                    )
                    feat.update(
                        {
                            **scenario_common,
                            "scenario_id": scenario_base + "_pure",
                            "condition": pure_condition,
                            "window": window_name,
                            "eta_multiplier": 1.0,
                            "post_kappa_map": float(pure_stability["kappa_map"]),
                        }
                    )
                    feature_rows.append(feat)

                found_loss = False
                for multiplier in eta_multipliers:
                    loss_params = replace(
                        base_params,
                        eta=float(base_params.eta) * float(multiplier),
                    )
                    loss_traj, loss_endpoint = simulate(
                        table,
                        base_params,
                        loss_params,
                        low,
                        high,
                        shift_step=int(args.shift_step),
                        horizon=int(args.horizon),
                        endpoint_last_k=int(args.endpoint_last_k),
                        support_clip=float(args.support_clip),
                    )
                    loss_stability = estimate_post_stability(
                        post_table, loss_endpoint, loss_params, args
                    )
                    if float(loss_stability["kappa_map"]) <= 0:
                        continue
                    found_loss = True
                    loss_id = scenario_base + f"_eta_x{multiplier:g}"
                    scenario_rows.append(
                        {
                            **scenario_common,
                            "scenario_id": loss_id,
                            "condition": "shift_plus_loss",
                            "eta_multiplier": float(multiplier),
                            "post_eta": float(loss_params.eta),
                            "post_kappa_map": float(loss_stability["kappa_map"]),
                        }
                    )
                    for window_name, start in [
                        ("crossing", 0),
                        ("post_reset", int(args.shift_step)),
                    ]:
                        feat = add_parameter_fields(
                            feature_window(loss_traj, start, window_width),
                            loss_params,
                        )
                        feat.update(
                            {
                                **scenario_common,
                                "scenario_id": loss_id,
                                "condition": "shift_plus_loss",
                                "window": window_name,
                                "eta_multiplier": float(multiplier),
                                "post_kappa_map": float(loss_stability["kappa_map"]),
                            }
                        )
                        feature_rows.append(feat)
                    break
                if not found_loss:
                    scenario_rows.append(
                        {
                            **scenario_common,
                            "scenario_id": scenario_base + "_no_loss_found",
                            "condition": "loss_not_found",
                            "eta_multiplier": float("nan"),
                            "post_eta": float("nan"),
                            "post_kappa_map": float("nan"),
                        }
                    )

    features = pd.DataFrame(feature_rows)
    scenarios = pd.DataFrame(scenario_rows)
    features.to_csv(out_dir / "ood_early_warning_features.csv.gz", index=False, compression="gzip")
    scenarios.to_csv(out_dir / "scenario_stability.csv", index=False)
    predictions = predict_rows(features, models)
    predictions.to_csv(out_dir / "ood_predictions.csv.gz", index=False, compression="gzip")
    overall, strata = evaluation_tables(predictions)
    overall.to_csv(out_dir / "ood_evaluation_overall.csv", index=False)
    strata.to_csv(out_dir / "ood_evaluation_by_shift.csv", index=False)

    pure = scenarios.loc[
        scenarios["condition"].isin(["pure_shift_stable", "context_induced_loss"])
    ].copy()
    pure["context_induced_loss"] = pure["condition"].eq("context_induced_loss").astype(int)
    summary = (
        pure.groupby(["context_axis", "shift_magnitude"], as_index=False)
        .agg(
            n_total=("scenario_id", "size"),
            n_context_induced_loss=("context_induced_loss", "sum"),
            context_induced_loss_rate=("context_induced_loss", "mean"),
            mean_standardized_gap=("context_standardized_gap", "mean"),
        )
    )
    summary.to_csv(out_dir / "context_induced_stability_changes.csv", index=False)
    write_json(
        out_dir / "run_config.json",
        {
            "args": vars(args),
            "n_training_rows": len(train),
            "n_operating_points": len(operating_points),
            "n_scenarios": len(scenarios),
            "n_feature_rows": len(features),
            "n_prediction_rows": len(predictions),
        },
    )
    print(f"wrote context-shift audit to {out_dir}", flush=True)
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_features", default=str(DEFAULT_TRAIN))
    parser.add_argument("--sources_dir", default=str(DEFAULT_SOURCES))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--early_frac", type=float, default=0.10)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--shift_step", type=int, default=5)
    parser.add_argument("--endpoint_last_k", type=int, default=50)
    parser.add_argument("--stable_points_per_seed", type=int, default=4)
    parser.add_argument("--context_axes", default=",".join(CONTEXT_AXES))
    parser.add_argument("--shift_magnitudes", default="0.50,0.75,0.90")
    parser.add_argument("--eta_multipliers", default="1.25,1.5,2,3,4")
    parser.add_argument("--target_stationary_fpr", type=float, default=0.05)
    parser.add_argument("--support_clip", type=float, default=8.0)
    parser.add_argument("--active_eps", type=float, default=1.0e-4)
    parser.add_argument("--fd_eps", type=float, default=1.0e-5)
    parser.add_argument("--invasion_eps", type=float, default=1.0e-5)
    parser.add_argument("--stability_context_stride", type=int, default=1)
    parser.add_argument("--stability_max_active", type=int, default=160)
    parser.add_argument("--random_state", type=int, default=20260723)
    return parser


def main_shift():
    run(build_parser().parse_args())


SUBCOMMANDS = {"holdout": main_holdout, "shift": main_shift}


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=False)
    ap.add_argument("command", nargs="?", choices=["all", *SUBCOMMANDS], help="which part to run (all = every part in order)")
    args, rest = ap.parse_known_args()
    if args.command is None:  # no command (or just -h): show this help; `<command> -h` shows the command's own options
        ap.print_help(); sys.exit(0 if sys.argv[1:] else 2)
    for cmd in (list(SUBCOMMANDS) if args.command == "all" else [args.command]):
        sys.argv = [f"{sys.argv[0]} {cmd}", *rest]
        print(f"== {cmd}", flush=True)
        SUBCOMMANDS[cmd]()


if __name__ == "__main__":
    main()
