#!/usr/bin/env python3
"""Early prediction (Fig. 5, Appendix B.3): restricted evaluation, fit-free baselines, lead time.

Grid: the early-warning grid (8 eta x 3 alpha x 8 beta_sup x 10 seeds = 1,920 endpoints), log-space
replay.  Target T1: the endpoint is NOT simultaneously diverse (concentrated-round fraction over
the last 100 rounds > 0.5).  Prediction time: 10% of the trajectory (round 20 of 200).

Models: the saved out-of-fold scores of early_warning/logspace.py (``oof_predictions_T1.csv.gz``; leave-one-eta-out
x seed and leave-one-beta_sup-out x seed folds) -- nothing is refitted.  Fit-free baselines (fixed
orientation, no training, so there is no fold structure):
  (i)   concentrated-round fraction over rounds 1..20
  (ii)  minus the mean per-round N_eff over rounds 1..20
  (iii) theoretical fluctuation variance eta^2 sigma^2 (1 - lambda^{2t}) / (1 - lambda^2) at t = 20,
        lambda = 1 - eta*beta_sup (sigma^2 from the first 20 rounds; ``_fullT``: from all rounds)
  (iv)  1{predicted variance over the endpoint window > v_50} (pred_total, v_50 of
        data/outputs/predicted_level/manifest.json; score statistics from the first 20 rounds, persistent part debiased by sigma^2/20;
        ``_fullT``: from all rounds)
Evaluation sets: all endpoints, and those still diverse at the prediction time (early
concentrated-round fraction <= 0.5).  AUC: mean +/- s.e. of per-seed AUCs, plus pooled.
Lead time: for endpoints diverse at 10% and positive at the end, the first round t >= 20 at which
the running concentrated-round fraction (window 20, rounds t-19..t) exceeds 0.5; lead = t - 20.

    python3 pipeline/early_warning/closed_form.py        (reads runs/: ~10 min)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))  # pipeline/ on the path

from common import grids as c
from common import per_round as g
from common import logspace as j
from common import pairings as L

KEY = ["seed", "eta", "alpha", "beta_sup"]
T_EARLY, RUN_W = 20, 20
MODEL_NAMES = {"parameter_only": "Parameter-only (logistic)", "param_rbf_svm": "Parameter-only (RBF SVM)",
               "param_random_forest": "Parameter-only (random forest)", "early_descriptor": "Early descriptors",
               "fluctuation_only": "Fluctuation features", "full_early_warning": "All early features"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h2", default="data/outputs/replay_logspace_grids/early_warning/phase_summary.csv.gz")
    p.add_argument("--oof", default="data/outputs/replay_logspace_adapted/G7_early_warning/oof_predictions_T1.csv.gz")
    p.add_argument("--j1-manifest", default="data/outputs/predicted_level/manifest.json")
    p.add_argument("--table-cache", default=j.DEFAULT_TABLE_CACHE)
    p.add_argument("--out-dir", default="data/outputs/early_warning_closed_form")
    return p.parse_args()


def fluc_var(eta, beta, sig2, t):
    lam = 1 - eta * beta
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        v = np.where(np.abs(lam - 1) < 1e-12, eta ** 2 * sig2 * t, eta ** 2 * sig2 * (1 - lam ** (2 * t)) / (1 - lam ** 2))
        v = np.where(np.abs(lam + 1) < 1e-12, eta ** 2 * sig2 * t, v)
    return v


def mean_var(eta, beta, var_mu, t):
    lam = 1 - eta * beta
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return np.where(beta > 0, var_mu * ((1 - lam ** t) / np.where(beta > 0, beta, 1.0)) ** 2, var_mu * (eta * t) ** 2)


def compute(args, v50: float) -> pd.DataFrame:
    h2 = pd.read_csv(c.ROOT / args.h2)
    frames = []
    for seed, rows in h2.groupby("seed"):
        table = c.load_table(c.ROOT / "runs" / f"run_popseed_{seed}.checkpoints", args.table_cache)
        fc = g.FixedFitnessCache(table, 8.0)
        n, T = table.n_agents, table.n_timesteps
        top1_thr, neff_thr = g.concentration_thresholds(n)
        for _, sub in rows.groupby(L.FIT_KEY_COLS, sort=False):
            a = fc.get(c.params_from_row(sub.iloc[0]))
            eta, beta = sub.eta.to_numpy(float), sub.beta_sup.to_numpy(float)
            Z = j.replay_batch(a, eta, beta)
            P = np.exp(Z[1:])                                                   # exposure after rounds 1..T
            conc = (P.max(axis=2) >= top1_thr) | (1.0 / (P ** 2).sum(axis=2) <= neff_thr)   # (T, E)
            neff = 1.0 / (P ** 2).sum(axis=2)
            rec = sub[KEY + ["conc_round_frac", "phase_label", "N_eff_round_mean"]].reset_index(drop=True)
            rec["conc_round_frac_check"] = conc[-100:].mean(axis=0)
            rec["b_early_conc_frac"] = conc[:T_EARLY].mean(axis=0)
            rec["b_early_neg_neff"] = -neff[:T_EARLY].mean(axis=0)
            for tag_, A_src in (("", a[:T_EARLY]), ("_fullT", a)):
                A = A_src - A_src.mean(axis=1, keepdims=True); mu = A.mean(axis=0)
                sig2 = float(np.mean((A - mu) ** 2)) * (A.shape[0] / max(A.shape[0] - 1, 1) if tag_ == "" else 1.0)
                var_mu = max(float(np.var(mu)) - sig2 / A.shape[0], 0.0) if tag_ == "" else float(np.var(mu))
                rec[f"b_theory_var_t20{tag_}"] = fluc_var(eta, beta, sig2, float(T_EARLY))
                tt = np.arange(T - 100 + 1, T + 1, dtype=float)[:, None]
                pred = (mean_var(eta[None], beta[None], var_mu, tt) + fluc_var(eta[None], beta[None], sig2, tt)).mean(axis=0)
                rec[f"pred_total{tag_}"] = pred
                rec[f"b_pred_gt_v50{tag_}"] = (~(pred <= v50)).astype(float)       # overflow (eta*beta > 2) counts as > v_50
            run = np.stack([conc[t - RUN_W + 1:t + 1].mean(axis=0) for t in range(RUN_W - 1, T)])   # index 0 <-> round 20
            first = np.where((run > 0.5).any(axis=0), (run > 0.5).argmax(axis=0) + RUN_W, -1)
            rec["first_round_running_conc"] = first
            frames.append(rec)
        print("seed", seed, flush=True)
    return pd.concat(frames, ignore_index=True)


def auc_rows(d: pd.DataFrame, col: str, subset: str) -> dict:
    ps = []
    for _, s_ in d.groupby("seed"):
        if s_.y.nunique() == 2:
            ps.append(roc_auc_score(s_.y, s_[col]))
    ps = np.array(ps)
    return dict(subset=subset, score=col, n=int(len(d)), positive_rate=float(d.y.mean()), pooled_auc=float(roc_auc_score(d.y, d[col])),
                seed_mean_auc=float(ps.mean()), seed_se_auc=float(ps.std(ddof=1) / np.sqrt(len(ps))), n_seeds_defined=int(len(ps)))


def main():
    args = parse_args()
    out_dir = c.ROOT / args.out_dir; out_dir.mkdir(parents=True, exist_ok=True)
    v50 = float(json.loads((c.ROOT / args.j1_manifest).read_text())["levels"]["96"]["v_50"])
    res = compute(args, v50)
    oof = pd.read_csv(c.ROOT / args.oof)
    d = res.merge(oof, on=KEY, validate="one_to_one")
    d["y_check"] = (d.conc_round_frac > g.SIM_DIVERSE_MAX_CONC_FRAC).astype(int)
    d["diverse_at_10pct"] = (d.b_early_conc_frac <= 0.5).astype(int)
    d.to_csv(out_dir / "endpoint_early_prediction.csv.gz", index=False)
    checks = {"n": int(len(d)), "n_y_differs_from_oof": int((d.y != d.y_check).sum()),
              "max_abs_diff_conc_round_frac_vs_h2": float((d.conc_round_frac_check - d.conc_round_frac).abs().max()), "v_50": v50}

    score_cols = [k for k in oof.columns if "__" in k] + [k for k in d.columns if k.startswith("b_")]
    rows = []
    for subset, dd in [("all", d), ("diverse_at_10pct", d[d.diverse_at_10pct == 1])]:
        rows += [auc_rows(dd, col, subset) for col in score_cols]
    auc = pd.DataFrame(rows)
    acc = pd.DataFrame([dict(subset=subset, score=col, accuracy=float((dd[col] == dd.y).mean()),
                             tpr=float(dd.loc[dd.y == 1, col].mean()), fpr=float(dd.loc[dd.y == 0, col].mean()))
                        for subset, dd in [("all", d), ("diverse_at_10pct", d[d.diverse_at_10pct == 1])] for col in ["b_pred_gt_v50", "b_pred_gt_v50_fullT"]])
    auc.to_csv(out_dir / "auc_by_subset.csv", index=False); acc.to_csv(out_dir / "binary_rule_accuracy.csv", index=False)

    lead = d[(d.diverse_at_10pct == 1) & (d.y == 1)].copy()
    lead["lead_rounds"] = np.where(lead.first_round_running_conc >= 0, lead.first_round_running_conc - T_EARLY, np.nan)
    lead[KEY + ["b_early_conc_frac", "conc_round_frac", "first_round_running_conc", "lead_rounds"]].to_csv(out_dir / "lead_time_endpoints.csv", index=False)
    q = lead.lead_rounds.dropna()
    lead_summary = dict(n=int(len(lead)), n_never_crossing=int(lead.lead_rounds.isna().sum()), mean=float(q.mean()), median=float(q.median()),
                        q10=float(q.quantile(0.1)), q25=float(q.quantile(0.25)), q75=float(q.quantile(0.75)), q90=float(q.quantile(0.9)), min=float(q.min()), max=float(q.max()),
                        share_lead_ge_20=float((q >= 20).mean()), share_lead_ge_50=float((q >= 50).mean()))
    (out_dir / "lead_time_summary.json").write_text(json.dumps(lead_summary, indent=2))
    lead.groupby("eta").lead_rounds.describe().to_csv(out_dir / "lead_time_by_eta.csv")

    ns = int(d.diverse_at_10pct.sum())
    pa, ps_ = float(d.y.mean()), float(d[d.diverse_at_10pct == 1].y.mean())
    (out_dir / "manifest.json").write_text(json.dumps({"checks": checks, "lead_time": lead_summary, "n_diverse_at_10pct": ns,
                                                       "positive_rate_all": pa, "positive_rate_diverse_at_10pct": ps_}, indent=2))
    pd.set_option("display.width", 250)
    print(json.dumps(checks, indent=2)); print(auc.round(4).to_string(index=False)); print(acc.round(4).to_string(index=False)); print(json.dumps(lead_summary, indent=2))


if __name__ == "__main__":
    main()
