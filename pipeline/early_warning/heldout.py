#!/usr/bin/env python3
"""Early-warning classifiers evaluated with held-out populations / parameters (Appendix B.3, B.8).

    python3 pipeline/early_warning/heldout.py heldout_support   stable-specialization target, each support strength held out with the seed  -> data/outputs/early_warning_heldout_support
    python3 pipeline/early_warning/heldout.py heldout_params    kappa target, each selection strength / support / context fidelity held out -> data/outputs/early_warning_heldout_params
    python3 pipeline/early_warning/heldout.py testbeds          the same on the five robustness testbeds                                      -> data/outputs/early_warning_testbeds

Shared model specifications, fold builders and the sign test live in the heldout_support section; every sub-command
accepts its own options (``<sub-command> -h``).  Each takes minutes to an hour (many folds).
"""

from __future__ import annotations

import sys
import argparse
import json
import warnings
from pathlib import Path
from typing import Callable
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import re

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # pipeline/ on the path



# ====================================================================================================
# heldout_support
# ====================================================================================================
warnings.filterwarnings("ignore")
PARAMETER_COLUMNS = [
    "eta", "alpha", "beta_sup", "beta_route", "lambda_risk", "hard_safety_threshold",
    "context_trait_strength", "polarization_strength", "route_power", "route_zscore", "no_feedback",
]
RS_DEFAULT = 20260801
def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    params = [c for c in PARAMETER_COLUMNS if c in df.columns]
    desc = [c for c in df.columns if c.endswith(("_early_mean", "_early_last", "_early_slope"))]
    fluc = [c for c in df.columns if c.endswith(("_early_var", "_early_lag1", "_early_accel"))]
    varying = [c for c in params if df[c].nunique(dropna=False) > 1]
    return {"varying_parameters": varying, "parameters": params, "early_descriptor": desc,
            "fluctuation_only": fluc, "full_early_warning": params + desc + fluc}
def _lr(rs):
    return LogisticRegression(max_iter=5000, class_weight="balanced", solver="liblinear", random_state=rs)
def model_specs(rs: int = RS_DEFAULT) -> dict[str, tuple[str, Callable[[], object], str]]:
    """name -> (feature-set key, factory, score method): the parameter-only learners and the logistic
    early-warning classifier of the replay (trajectory feature sets)."""
    imp = lambda: SimpleImputer(strategy="median")
    return {
        "param_additive_logistic": ("varying_parameters", lambda: make_pipeline(StandardScaler(), _lr(rs)), "proba"),
        "param_quadratic_logistic": ("varying_parameters", lambda: make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False), StandardScaler(), _lr(rs)), "proba"),
        "param_rbf_svm": ("varying_parameters", lambda: make_pipeline(StandardScaler(), SVC(
            C=1.0, kernel="rbf", gamma="scale", class_weight="balanced", probability=True, random_state=rs)), "proba"),
        "param_rbf_svm_margin": ("varying_parameters", lambda: make_pipeline(StandardScaler(), SVC(
            C=1.0, kernel="rbf", gamma="scale", class_weight="balanced")), "decision"),
        "param_random_forest": ("varying_parameters", lambda: RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, max_features=None, class_weight="balanced_subsample",
            random_state=rs, n_jobs=-1), "proba"),
        "early_descriptor": ("early_descriptor", lambda: make_pipeline(imp(), StandardScaler(), _lr(rs)), "proba"),
        "fluctuation_only": ("fluctuation_only", lambda: make_pipeline(imp(), StandardScaler(), _lr(rs)), "proba"),
        "full_early_warning": ("full_early_warning", lambda: make_pipeline(imp(), StandardScaler(), _lr(rs)), "proba"),
    }
def make_folds(df: pd.DataFrame, split: str, seed_col: str = "seed") -> list[tuple[np.ndarray, np.ndarray, dict]]:
    seeds = df[seed_col].to_numpy()
    folds = []
    if split == "seed":
        for s in np.unique(seeds):
            folds.append((seeds != s, seeds == s, {"heldout_seed": int(s)}))
        return folds
    joint = split.startswith("seed_x_")
    col = split.replace("seed_x_", "")
    g = df[col].to_numpy()
    for v in np.unique(g):
        if joint:
            for s in np.unique(seeds):
                folds.append(((g != v) & (seeds != s), (g == v) & (seeds == s), {"heldout_seed": int(s), f"heldout_{col}": float(v)}))
        else:
            folds.append((g != v, g == v, {f"heldout_{col}": float(v)}))
    return folds
def oof_scores(df: pd.DataFrame, y: np.ndarray, cols: list[str], factory, method: str, folds) -> np.ndarray:
    X = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    out = np.full(len(df), np.nan)
    for tr, te, _ in folds:
        if len(np.unique(y[tr])) < 2:
            out[te] = float(y[tr].mean())
            continue
        m = factory()
        m.fit(X[tr], y[tr])
        out[te] = m.predict_proba(X[te])[:, 1] if method == "proba" else m.decision_function(X[te])
    assert np.isfinite(out).all()
    return out
def se(v) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
def safe_auc(y, s) -> float:
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")
def group_aucs(df, y, score, col) -> pd.DataFrame:
    rows = []
    for v, idx in df.groupby(col).indices.items():
        rows.append({"group_col": col, "group": v, "n": len(idx), "n_positive": int(y[idx].sum()),
                     "roc_auc": safe_auc(y[idx], score[idx]),
                     "pr_auc": float(average_precision_score(y[idx], score[idx])) if y[idx].sum() > 0 else np.nan})
    return pd.DataFrame(rows)
def sign_test(full: np.ndarray, base: np.ndarray) -> dict:
    d = full - base
    ok = np.isfinite(d)
    wins, losses = int((d[ok] > 0).sum()), int((d[ok] < 0).sum())
    n = wins + losses
    return {"n_seeds": int(ok.sum()), "wins_full": wins, "losses_full": losses,
            "mean_delta": float(d[ok].mean()), "se_delta": se(d[ok]),
            "sign_test_p_two_sided": float(binomtest(wins, n, 0.5).pvalue) if n else np.nan,
            "sign_test_p_one_sided": float(binomtest(wins, n, 0.5, alternative="greater").pvalue) if n else np.nan}
def evaluate(df, y, splits, specs, fsets, seed_col="seed", group_cols=()):
    preds, summ, per_group = {}, [], []
    for split in splits:
        folds = make_folds(df, split, seed_col)
        for name, (fkey, fac, method) in specs.items():
            sc = oof_scores(df, y, fsets[fkey], fac, method, folds)
            preds[(split, name)] = sc
            ga = group_aucs(df, y, sc, seed_col)
            row = {"split": split, "model": name, "n": len(y), "n_positive": int(y.sum()), "n_folds": len(folds),
                   "n_features": len(fsets[fkey]),
                   "pooled_roc_auc": safe_auc(y, sc), "pooled_pr_auc": float(average_precision_score(y, sc)),
                   "seed_mean_roc_auc": float(ga.roc_auc.mean()), "seed_se_roc_auc": se(ga.roc_auc),
                   "seed_mean_pr_auc": float(ga.pr_auc.mean()), "seed_se_pr_auc": se(ga.pr_auc)}
            ga.insert(0, "model", name); ga.insert(0, "split", split)
            per_group.append(ga)
            for gc in group_cols:
                gg = group_aucs(df, y, sc, gc)
                row[f"{gc}_value_mean_roc_auc"] = float(gg.roc_auc.mean())
                row[f"{gc}_value_se_roc_auc"] = se(gg.roc_auc)
                row[f"{gc}_n_values_defined"] = int(gg.roc_auc.notna().sum())
                gg.insert(0, "model", name); gg.insert(0, "split", split)
                per_group.append(gg)
            summ.append(row)
            print(f"{split:18s} {name:26s} pooled {row['pooled_roc_auc']:.4f}  seed-mean {row['seed_mean_roc_auc']:.4f} +- {row['seed_se_roc_auc']:.4f}", flush=True)
    return preds, pd.DataFrame(summ), pd.concat(per_group, ignore_index=True)
def main_heldout_support():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features_csv", default="data/results/early_warning_validation_grid_leakage_controls/early_warning_features.csv")
    ap.add_argument("--target", default="label_stable_specialization")
    ap.add_argument("--early_frac", type=float, default=0.1)
    ap.add_argument("--random_state", type=int, default=RS_DEFAULT)
    ap.add_argument("--platt_seeds", default="20260801,20260725,0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--out_dir", default="data/outputs/early_warning_heldout_support")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.features_csv)
    df = raw[np.isclose(raw["early_frac"], args.early_frac)].copy().reset_index(drop=True)
    y = df[args.target].astype(int).to_numpy()
    fsets = feature_sets(df)
    specs = model_specs(args.random_state)
    cells = df.groupby(fsets["varying_parameters"])[args.target].sum()
    meta = {"n": int(len(df)), "n_positive": int(y.sum()), "positive_rate": float(y.mean()),
            "n_seeds": int(df.seed.nunique()), "varying_parameters": fsets["varying_parameters"],
            "n_cells": int(len(cells)), "n_positive_cells": int((cells > 0).sum()),
            "positive_beta_sup": sorted(float(v) for v in df.loc[y == 1, "beta_sup"].unique()),
            "beta_sup_values": sorted(float(v) for v in df.beta_sup.unique()),
            "random_state": args.random_state, "features_csv": args.features_csv}
    print(json.dumps(meta, indent=1))

    splits = ["seed", "beta_sup", "seed_x_beta_sup"]
    preds, summ, per_group = evaluate(df, y, splits, specs, fsets, group_cols=("beta_sup",))
    meta["n_folds_main"] = int(summ[summ.split == "seed_x_beta_sup"].n_folds.iloc[0])

    # paired per-seed comparison: full early-warning vs each other model
    ps = per_group[per_group.group_col == "seed"].rename(columns={"group": "seed"})
    wide = ps.pivot_table(index=["split", "seed"], columns="model", values="roc_auc").reset_index()
    sign_rows = []
    for split in splits:
        w = wide[wide.split == split]
        for m in specs:
            if m == "full_early_warning":
                continue
            sign_rows.append({"split": split, "baseline": m, **sign_test(w["full_early_warning"].to_numpy(), w[m].to_numpy())})
    signs = pd.DataFrame(sign_rows)

    # sensitivity of the Platt-scaled SVM to its internal CV random state (main split only)
    folds = make_folds(df, "seed_x_beta_sup")
    full_seed = wide[wide.split == "seed_x_beta_sup"].set_index("seed")["full_early_warning"]
    prow = []
    for rs in [int(v) for v in args.platt_seeds.split(",")]:
        fkey, fac, method = model_specs(rs)["param_rbf_svm"]
        sc = oof_scores(df, y, fsets[fkey], fac, method, folds)
        ga = group_aucs(df, y, sc, "seed").set_index("group")["roc_auc"]
        prow.append({"svm_random_state": rs, "pooled_roc_auc": safe_auc(y, sc), "seed_mean_roc_auc": float(ga.mean()),
                     "wins_full": int((full_seed.loc[ga.index] > ga).sum())})
    platt = pd.DataFrame(prow)

    pred_df = df[["seed", "eta", "alpha", "beta_sup"]].copy()
    pred_df["y_true"] = y
    for (split, name), sc in preds.items():
        pred_df[f"{split}__{name}"] = sc
    pred_df.to_csv(out / "oof_predictions.csv.gz", index=False)
    summ.to_csv(out / "summary_auc.csv", index=False)
    ps.drop(columns=["group_col"]).to_csv(out / "per_seed_auc.csv", index=False)
    per_group[per_group.group_col == "beta_sup"].to_csv(out / "per_beta_sup_auc.csv", index=False)
    wide.to_csv(out / "per_seed_auc_wide.csv", index=False)
    signs.to_csv(out / "sign_tests_full_vs_baselines.csv", index=False)
    platt.to_csv(out / "svm_platt_random_state_sensitivity.csv", index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    pd.set_option("display.width", 250)
    print(signs.round(5).to_string(index=False))
    print(platt.round(4).to_string(index=False))


# ====================================================================================================
# heldout_params
# ====================================================================================================
RS = 20260801
TARGETS = {"context_holdout": "context_holdout_kappa_map", "baseline": "baseline_kappa_map"}
TARGET_TEX = {"context_holdout": r"Strict context-holdout $\kappa$ target (primary)",
              "baseline": r"Original full-stream $\kappa$ target"}
MODELS = ["parameter_only", "param_rbf_svm", "param_random_forest",
          "early_descriptor", "fluctuation_only", "full_early_warning"]
MODEL_TEX_params = {"parameter_only": "Param.\\ (logistic)", "param_rbf_svm": "Param.\\ (RBF SVM)",
             "param_random_forest": "Param.\\ (RF)", "early_descriptor": "Early descriptor",
             "fluctuation_only": "Fluctuation", "full_early_warning": "Full"}
SPLIT_TEX = {
    "seed": r"Leave-one-seed-out (reference)",
    "eta": r"Leave-one-$\eta$-out",
    "seed_x_eta": r"Leave-one-$\eta$-out + unseen seed",
    "beta_sup": r"Leave-one-$\beta_{\mathrm{sup}}$-out",
    "seed_x_beta_sup": r"Leave-one-$\beta_{\mathrm{sup}}$-out + unseen seed",
    "alpha": r"Leave-one-$\alpha$-out",
    "seed_x_alpha": r"Leave-one-$\alpha$-out + unseen seed",
}
def specs():
    def lr():
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"))
    return {
        "parameter_only": ("parameters", lr, "proba"),
        "param_rbf_svm": ("varying_parameters", lambda: make_pipeline(StandardScaler(), SVC(
            C=1.0, kernel="rbf", gamma="scale", class_weight="balanced")), "decision"),
        "param_random_forest": ("varying_parameters", lambda: RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, max_features=None, class_weight="balanced_subsample",
            random_state=RS, n_jobs=-1), "proba"),
        "early_descriptor": ("early_descriptor", lr, "proba"),
        "fluctuation_only": ("fluctuation_only", lr, "proba"),
        "full_early_warning": ("full_early_warning", lr, "proba"),
    }
def main_heldout_params():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged_csv", default="data/results/context_disjoint_criticality_full_holdout/summary/context_disjoint_features_and_targets.csv")
    ap.add_argument("--out_dir", default="data/outputs/early_warning_heldout_params")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.merged_csv).reset_index(drop=True)
    df["seed"] = df["_seed"].astype(int)
    drop = [c for c in df.columns if c in ("_seed",)]
    df = df.drop(columns=drop)
    fsets = feature_sets(df)
    assert fsets["varying_parameters"] == ["eta", "alpha", "beta_sup"], fsets["varying_parameters"]
    assert ((df["baseline_kappa_map"] > 0).astype(int) == df["label_unstable"].astype(int)).all()
    splits = list(SPLIT_TEX)
    meta = {"n": int(len(df)), "n_seeds": int(df.seed.nunique()), "n_eta": int(df.eta.nunique()),
            "n_alpha": int(df.alpha.nunique()), "n_beta_sup": int(df.beta_sup.nunique()), "positive_rate": {}}

    all_summ, all_groups, all_signs = [], [], []
    pred_df = df[["seed", "eta", "alpha", "beta_sup"]].copy()
    for t, kcol in TARGETS.items():
        y = (pd.to_numeric(df[kcol], errors="coerce") > 0).astype(int).to_numpy()
        meta["positive_rate"][t] = float(y.mean())
        pred_df[f"y_{t}"] = y
        print(f"===== target {t}  positive rate {y.mean():.4f}", flush=True)
        preds, summ, per_group = evaluate(df, y, splits, specs(), fsets, group_cols=("eta", "beta_sup", "alpha"))
        summ.insert(0, "target", t); per_group.insert(0, "target", t)
        all_summ.append(summ); all_groups.append(per_group)
        for (s, m), sc in preds.items():
            pred_df[f"{t}__{s}__{m}"] = sc
        ps = per_group[per_group.group_col == "seed"].pivot_table(index=["split", "group"], columns="model", values="roc_auc").reset_index()
        for s in splits:
            w = ps[ps.split == s]
            for ref in ["early_descriptor", "fluctuation_only", "full_early_warning"]:
                for base in ["parameter_only", "param_rbf_svm", "param_random_forest"]:
                    all_signs.append({"target": t, "split": s, "model": ref, "baseline": base,
                                      **sign_test(w[ref].to_numpy(), w[base].to_numpy())})
    summ = pd.concat(all_summ, ignore_index=True)
    groups = pd.concat(all_groups, ignore_index=True)
    signs = pd.DataFrame(all_signs).rename(columns={"wins_full": "wins_model", "losses_full": "losses_model"})

    summ.to_csv(out / "summary_auc.csv", index=False)
    groups[groups.group_col == "seed"].to_csv(out / "per_seed_auc.csv", index=False)
    groups[groups.group_col != "seed"].to_csv(out / "per_heldout_value_auc.csv", index=False)
    signs.to_csv(out / "paired_seed_sign_tests.csv", index=False)
    pred_df.to_csv(out / "oof_predictions.csv.gz", index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


# ====================================================================================================
# testbeds
# ====================================================================================================
PARAM_MODELS = ["parameter_only", "param_rbf_svm", "param_random_forest"]
TRAJ_MODELS = ["early_descriptor", "fluctuation_only", "full_early_warning"]
SETTINGS = {
    "8d": ("8D traits", "data/testbeds/verification_8d_support/early_warning_features.csv"),
    "16d": ("16D traits", "data/testbeds/verification_16d_support/early_warning_features.csv"),
    "llama": ("Llama generator", "data/testbeds/verification_llama_support/early_warning_features.csv"),
    "heterogeneity": ("Heterogeneous generators", "data/testbeds/verification_heterogeneity_support/early_warning_features.csv"),
    "tool": ("BFCL tool routing", "data/testbeds/verification_tool_main/early_warning_features.csv"),
}
MAIN_ROWS = {"main4d_strict": ("Main 4D (strict context-holdout $\\kappa$; Fig.~5)", "context_holdout"),
             "main4d_fullstream": ("Main 4D (full-stream $\\kappa$)", "baseline")}
SPLITS = {"seed_x_eta": r"Unseen $\eta$ \emph{and} unseen population seed (joint holdout)",
          "eta": r"Unseen $\eta$, seeds pooled"}
MODEL_TEX_testbeds = {"parameter_only": "Param.\\ linear", "param_rbf_svm": "Param.\\ RBF-SVM",
             "param_random_forest": "Param.\\ RF", "early_descriptor": "Early descriptor",
             "fluctuation_only": "Fluctuation", "full_early_warning": "Full"}
def load(path: str, early_frac: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[np.isclose(df["early_frac"], early_frac)].copy().reset_index(drop=True)
    df["seed"] = df["run_name"].map(lambda s: int(re.search(r"popseed_(\d+)", s).group(1)))
    assert ((pd.to_numeric(df["kappa_map"]) > 0).astype(int) == df["label_unstable"].astype(int)).all()
    return df
def best_param(summ_one_split: pd.DataFrame) -> str:
    """Best parameter baseline = highest pooled out-of-fold ROC AUC (ties -> first in PARAM_MODELS)."""
    s = summ_one_split.set_index("model").loc[PARAM_MODELS, "pooled_roc_auc"]
    return str(s.idxmax())
def comparisons(setting, summ, seed_wide, eta_wide):
    rows = []
    for split in SPLITS:
        ss = summ[summ.split == split]
        bp = best_param(ss)
        sw = seed_wide[seed_wide.split == split]
        ew = eta_wide[eta_wide.split == split]
        for m in TRAJ_MODELS:
            for base in PARAM_MODELS + ["best_param_per_seed"]:
                if base == "best_param_per_seed":   # harshest: per seed, the best of the three baselines
                    b_seed, b_eta = sw[PARAM_MODELS].max(axis=1).to_numpy(), ew[PARAM_MODELS].max(axis=1).to_numpy()
                else:
                    b_seed, b_eta = sw[base].to_numpy(), ew[base].to_numpy()
                st = sign_test(sw[m].to_numpy(), b_seed)
                se_ = sign_test(ew[m].to_numpy(), b_eta)
                rows.append({"setting": setting, "split": split, "model": m, "baseline": base,
                             "is_best_param_by_pooled_auc": base == bp,
                             "n_seeds": st["n_seeds"], "seed_wins": st["wins_full"], "seed_losses": st["losses_full"],
                             "seed_mean_delta": st["mean_delta"], "seed_se_delta": st["se_delta"],
                             "seed_sign_p_two_sided": st["sign_test_p_two_sided"],
                             "n_eta_defined": se_["n_seeds"], "eta_wins": se_["wins_full"], "eta_losses": se_["losses_full"],
                             "eta_mean_delta": se_["mean_delta"]})
    return pd.DataFrame(rows)
def fmt(v, bold=False, ital=False):
    s = f"{v:.3f}"
    if bold:
        s = r"\textbf{" + s + "}"
    if ital:
        s = r"\textit{" + s + "}"
    return s
def main_testbeds():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--early_frac", type=float, default=0.1)
    ap.add_argument("--target", default="label_unstable")
    ap.add_argument("--b10_dir", default="data/outputs/early_warning_heldout_params")
    ap.add_argument("--out_dir", default="data/outputs/early_warning_testbeds")
    ap.add_argument("--table_only", type=int, default=0, help="1: rebuild the .tex table from the saved CSVs without refitting")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    splits = list(SPLITS)
    meta = {"target": args.target, "early_frac": args.early_frac, "settings": {}, "n_eta": {}, "negatives": {}}
    S, SW, EW, C, P = [], [], [], [], []

    for k, (lab, path) in SETTINGS.items():
        df = load(path, args.early_frac)
        y = df[args.target].astype(int).to_numpy()
        fsets = feature_sets(df)
        meta["settings"][k] = {"features_csv": path, "n": int(len(df)), "n_positive": int(y.sum()), "positive_rate": float(y.mean()),
                               "seeds": sorted(int(s) for s in df.seed.unique()), "varying_parameters": fsets["varying_parameters"],
                               "n_features": {a: len(b) for a, b in fsets.items()},
                               "n_eta_both_classes": int((df.groupby("eta")[args.target].mean().between(1e-9, 1 - 1e-9)).sum())}
        meta["n_eta"][k] = int(df.eta.nunique())
        meta["negatives"][k] = {"n": int((y == 0).sum()), "min_beta_sup": float(df.beta_sup[y == 0].min()), "max_beta_sup": float(df.beta_sup[y == 0].max()),
                                "min_eta": float(df.eta[y == 0].min()), "max_eta": float(df.eta[y == 0].max())}
        if args.table_only:
            continue
        print(f"===== {k}: n={len(df)} pos={y.mean():.4f} varying={fsets['varying_parameters']}", flush=True)
        preds, summ, per_group = evaluate(df, y, splits, specs(), fsets, group_cols=("eta",))
        summ.insert(0, "setting", k)
        sw = per_group[per_group.group_col == "seed"].pivot_table(index=["split", "group"], columns="model", values="roc_auc").reset_index()
        ew = per_group[per_group.group_col == "eta"].pivot_table(index=["split", "group"], columns="model", values="roc_auc", dropna=False).reset_index()
        S.append(summ); C.append(comparisons(k, summ, sw, ew))
        sw.insert(0, "setting", k); ew.insert(0, "setting", k); SW.append(sw); EW.append(ew)
        pd_ = df[["seed", "eta", "alpha", "beta_sup"]].copy(); pd_.insert(0, "setting", k); pd_["y_true"] = y
        for (s, m), sc in preds.items():
            pd_[f"{s}__{m}"] = sc
        P.append(pd_)

    if args.table_only:
        for k in MAIN_ROWS:
            meta["n_eta"][k] = 8
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        return

    # ---- Main 4D rows from B10 (copied, not recomputed) ----
    b10 = Path(args.b10_dir)
    bs, bseed, beta = pd.read_csv(b10 / "summary_auc.csv"), pd.read_csv(b10 / "per_seed_auc.csv"), pd.read_csv(b10 / "per_heldout_value_auc.csv")
    for k, (lab, t) in MAIN_ROWS.items():
        summ = bs[(bs.target == t) & bs.split.isin(splits)].drop(columns=["target"]).copy()
        summ.insert(0, "setting", k)
        sw = bseed[(bseed.target == t) & bseed.split.isin(splits)].pivot_table(index=["split", "group"], columns="model", values="roc_auc").reset_index()
        ew = beta[(beta.target == t) & beta.split.isin(splits) & (beta.group_col == "eta")].pivot_table(index=["split", "group"], columns="model", values="roc_auc", dropna=False).reset_index()
        S.append(summ); C.append(comparisons(k, summ, sw, ew))
        sw.insert(0, "setting", k); ew.insert(0, "setting", k); SW.append(sw); EW.append(ew)
        meta["n_eta"][k] = 8

    summ = pd.concat(S, ignore_index=True)
    keep = ["setting", "split", "model", "n", "n_positive", "n_folds", "n_features", "pooled_roc_auc", "pooled_pr_auc",
            "seed_mean_roc_auc", "seed_se_roc_auc", "eta_value_mean_roc_auc", "eta_value_se_roc_auc", "eta_n_values_defined"]
    summ = summ[keep]
    comp = pd.concat(C, ignore_index=True)
    summ.to_csv(out / "summary_auc.csv", index=False)
    pd.concat(SW, ignore_index=True).rename(columns={"group": "seed"}).to_csv(out / "per_seed_auc_wide.csv", index=False)
    pd.concat(EW, ignore_index=True).rename(columns={"group": "eta"}).to_csv(out / "per_heldout_eta_auc_wide.csv", index=False)
    comp.to_csv(out / "trajectory_vs_param_baselines.csv", index=False)
    pd.concat(P, ignore_index=True).to_csv(out / "oof_predictions.csv.gz", index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    pd.set_option("display.width", 250)
    for col in ["pooled_roc_auc", "seed_mean_roc_auc", "eta_value_mean_roc_auc"]:
        print("\n##", col); print(summ.pivot_table(index=["split", "setting"], columns="model", values=col)[MODELS].round(3).to_string())
    print(json.dumps(meta["param_ties_or_wins"], indent=1))


SUBCOMMANDS = {"heldout_support": main_heldout_support, "heldout_params": main_heldout_params, "testbeds": main_testbeds}


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
