#!/usr/bin/env python3
"""Early warning on the log-space replay: the out-of-fold predictions behind Fig. 5 (closed_form.py reads them).

    python3 pipeline/early_warning/logspace.py features --stage features   regenerate the early-trajectory features of the early-warning grid (x86_64)
    python3 pipeline/early_warning/logspace.py features --stage eval       the evaluation (stage_eval below)                     -> data/outputs/replay_logspace_adapted/G7_early_warning
    python3 pipeline/early_warning/logspace.py eval                        the evaluation on its own
"""

from __future__ import annotations

import sys
import argparse
import json
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
import os

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # pipeline/ on the path

from common import grids as c  # noqa: E402
from common import per_round as g  # noqa: E402
from common import grids as c
from common import per_round as g
from replay import logspace as L


# ====================================================================================================
# eval
# ====================================================================================================
KEY = ["seed", "eta", "alpha", "beta_sup"]
SPLITS = ["seed_x_eta", "seed_x_beta_sup"]
def parse_args_eval():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["replay", "eval"], required=True)
    p.add_argument("--grid-dir", default="data/results/early_warning_validation_grid")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--features-csv", default="data/results/early_warning_validation_grid_leakage_controls/early_warning_features.csv")
    p.add_argument("--out-dir", default="data/outputs/G7_early_warning")
    p.add_argument("--table-cache", default="")
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()
def stage_replay(args, out: Path):
    from replay import per_round as g2
    grid = c.load_grid("early_warning", args.grid_dir)
    jobs = [{"seed": int(s), "rows": grid.loc[grid.seed.eq(int(s))], "runs_dir": args.runs_dir,
             "table_cache": args.table_cache, "clip": 8.0, "last_k": 100} for s in sorted(grid.seed.unique())]
    with Pool(args.workers) as pool:
        df = pd.concat(pool.map(g2.run_seed, jobs, chunksize=1), ignore_index=True)
    df = g.add_round6_labels(df)
    df["eta_beta"] = df.eta * df.beta_sup
    df.to_csv(out / "early_warning_grid_per_round.csv", index=False)
    print("endpoints", len(df), "replay matches stored", int(df.replay_matches_stored.sum()))
    print(df[["simultaneously_diverse", "phantom_new", "label_collapse", "phantom_or_collapse"]].mean())
def stage_eval(args, out: Path):
    from sklearn.metrics import brier_score_loss
    from early_warning.heldout import evaluate, feature_sets, sign_test
    from early_warning.heldout import specs as b10_specs
    from early_warning import heldout as a3

    # T3 drops collapse-labelled endpoints, which empties some (seed, parameter value) test cells;
    # such folds are skipped (the imported fold builder is otherwise unchanged).
    _make_folds = a3.make_folds
    a3.make_folds = lambda *a_, **k_: [f for f in _make_folds(*a_, **k_) if f[1].any()]

    lab = pd.read_csv(out / "early_warning_grid_per_round.csv").rename(columns={"label_collapse": "r6_label_collapse"})
    assert lab.replay_matches_stored.all(), "replayed endpoints differ from stored ones: wrong interpreter for this grid"
    raw = pd.read_csv(c.ROOT / args.features_csv)
    raw["seed"] = raw["seed"].astype(int)
    lab_cols = KEY + ["conc_round_frac", "simultaneously_diverse", "not_simultaneously_diverse", "phantom_new",
                      "r6_label_collapse", "phantom_or_collapse", "descriptor_diverse", "N_eff_round_mean"]

    def merged(frac):
        df = raw[np.isclose(raw.early_frac, frac)].reset_index(drop=True)
        n0 = len(df)
        df = df.merge(lab[lab_cols], on=KEY, how="left", validate="one_to_one")
        assert len(df) == n0 == 1920 and df.conc_round_frac.notna().all()
        assert not [col for col in df.columns if col.endswith(("_x", "_y"))], "column collision in merge"
        return df

    df = merged(0.1)
    fsets = feature_sets(df)
    assert fsets["varying_parameters"] == ["eta", "alpha", "beta_sup"]
    leak = [col for col in fsets["full_early_warning"] if col in lab_cols and col not in KEY]
    assert not leak, leak
    same = bool((df.not_simultaneously_diverse == df.phantom_or_collapse).all())
    meta = {"n": int(len(df)), "T1_equals_T2_on_every_endpoint": same,
            "positive_rate": {"T1_not_simultaneously_diverse": float(df.not_simultaneously_diverse.mean()),
                              "T2_phantom_or_collapse_label": float(df.phantom_or_collapse.mean())},
            "share_phantom": float(df.phantom_new.mean()), "share_collapse_label": float(df.r6_label_collapse.mean()),
            "share_simultaneously_diverse": float(df.simultaneously_diverse.mean())}
    targets = {"T1": (df, df.not_simultaneously_diverse.to_numpy().astype(int))}
    if not same:
        targets["T2"] = (df, df.phantom_or_collapse.to_numpy().astype(int))
    d3 = df[df.descriptor_diverse == 1].reset_index(drop=True)
    targets["T3"] = (d3, d3.phantom_new.to_numpy().astype(int))
    meta["positive_rate"]["T3_phantom_among_descriptor_diverse"] = float(d3.phantom_new.mean())
    meta["n_T3"] = int(len(d3))

    summs, signs, metrics = [], [], []
    oof = {}
    for t, (d, y) in targets.items():
        print(f"===== target {t}: n={len(y)} positive rate {y.mean():.4f}", flush=True)
        preds, summ, per_group = evaluate(d, y, SPLITS, b10_specs(), feature_sets(d), group_cols=("eta", "beta_sup"))
        summ.insert(0, "target", t); summs.append(summ)
        per_group.insert(0, "target", t)
        per_group.to_csv(out / f"per_group_auc_{t}.csv", index=False)
        pred_df = d[KEY].copy(); pred_df["y"] = y
        for (s, m), sc in preds.items():
            pred_df[f"{s}__{m}"] = sc
            if m != "param_rbf_svm":  # SVM margin is not a probability
                metrics.append({"target": t, "split": s, "model": m, "brier": float(brier_score_loss(y, np.clip(sc, 0, 1)))})
        pred_df.to_csv(out / f"oof_predictions_{t}.csv.gz", index=False)
        oof[t] = pred_df
        ps = per_group[per_group.group_col == "seed"].pivot_table(index=["split", "group"], columns="model", values="roc_auc").reset_index()
        for s in SPLITS:
            w = ps[ps.split == s]
            for ref in ["early_descriptor", "fluctuation_only", "full_early_warning"]:
                for base in ["parameter_only", "param_rbf_svm", "param_random_forest"]:
                    signs.append({"target": t, "split": s, "model": ref, "baseline": base,
                                  **sign_test(w[ref].to_numpy(), w[base].to_numpy())})
    summ = pd.concat(summs, ignore_index=True)
    summ = summ.merge(pd.DataFrame(metrics), on=["target", "split", "model"], how="left")
    summ.to_csv(out / "summary_auc.csv", index=False)
    pd.DataFrame(signs).rename(columns={"wins_full": "wins_model", "losses_full": "losses_model"}).to_csv(
        out / "paired_seed_sign_tests.csv", index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

def main_eval():
    args = parse_args_eval()
    out = c.ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    import os
    os.chdir(c.ROOT)
    (stage_replay if args.stage == "replay" else stage_eval)(args, out)


# ====================================================================================================
# features
# ====================================================================================================
R = c.replay
KEY_features = ["seed", "eta", "alpha", "beta_sup"]
def parse_args_features():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["features", "eval"], required=True)
    p.add_argument("--old-features", default="data/results/early_warning_validation_grid_leakage_controls/early_warning_features.csv")
    p.add_argument("--h2-labels", default="data/outputs/replay_logspace_grids/early_warning/phase_summary.csv.gz")
    p.add_argument("--out-dir", default="data/outputs/replay_logspace_adapted/G7_early_warning")
    p.add_argument("--table-cache", default="")
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()
def _seed_features(job):
    seed = job["seed"]
    table = c.load_table(c.ROOT / "runs" / f"run_popseed_{seed}.checkpoints", job["table_cache"] or None)
    cache = g.FixedFitnessCache(table, 8.0)
    n_steps = int(np.ceil(table.n_timesteps * max(job["fracs"])))
    rows = []
    for _, row in job["rows"].iterrows():
        params = c.params_from_row(row)
        P = np.exp(L.replay_log_states(table, params, fixed_f=cache.get(params)))
        traj = pd.DataFrame([R.trajectory_row(table, P[k], k, k) for k in range(n_steps + 1)])
        # last row only fixes max_step (= n_timesteps, as in the full stored trajectory); it is never in an early window
        traj = pd.concat([traj, pd.DataFrame([{"step_index": table.n_timesteps}])], ignore_index=True)
        for f in job["fracs"]:
            feats = R.early_warning_features_for_group(traj, f)
            rows.append({**{k: row[k] for k in job["id_cols"]}, **feats})
    print(f"seed {seed} done", flush=True)
    return pd.DataFrame(rows)
def stage_features(args, out: Path):
    old = pd.read_csv(c.ROOT / args.old_features)
    old["seed"] = old["seed"].astype(int)
    fracs = sorted(old.early_frac.unique())
    feat_cols = [col for col in old.columns if "_early_" in col]
    id_cols = [col for col in old.columns[:old.columns.get_loc("early_frac")]] + ["seed"]
    base = old[np.isclose(old.early_frac, fracs[0])][id_cols]
    jobs = [{"seed": int(s), "rows": base[base.seed == s], "fracs": fracs, "id_cols": id_cols,
             "table_cache": args.table_cache} for s in sorted(base.seed.unique())]
    with Pool(args.workers) as pool:
        new = pd.concat(pool.map(_seed_features, jobs, chunksize=1), ignore_index=True)
    assert sorted(col for col in new.columns if "_early_" in col) == sorted(feat_cols)
    new.to_csv(out / "early_warning_features_logspace.csv.gz", index=False)
    m = old.merge(new, on=id_cols + ["early_frac"], suffixes=("_old", "_new"), validate="one_to_one")
    chk = m[(m.beta_sup == 0) & (m.eta < 1.75)]
    diffs = {col: float((chk[f"{col}_old"] - chk[f"{col}_new"]).abs().max()) for col in feat_cols}
    rel = {col: float(((chk[f"{col}_old"] - chk[f"{col}_new"]).abs() / chk[f"{col}_old"].abs().clip(lower=1e-12)).max()) for col in feat_cols}
    (out / "features_self_check.json").write_text(json.dumps({"n_rows_checked": int(len(chk)), "max_abs_diff_any_feature": max(diffs.values()),
                                                              "max_rel_diff_any_feature": max(rel.values()), "max_abs_diff": diffs}, indent=2))
    print("self-check rows", len(chk), "max abs diff", max(diffs.values()), "max rel diff", max(rel.values()))

    lab = pd.read_csv(c.ROOT / args.h2_labels)
    lab["N_eff_avg"], lab["top1_avg"] = lab.n_eff, lab.top1_share
    lab["replay_matches_stored"] = 1  # flag read by stage_eval; not applicable to a new update rule
    lab.to_csv(out / "early_warning_grid_per_round.csv", index=False)
    print(lab[["simultaneously_diverse", "phantom_new", "label_collapse"]].mean())
def stage_eval_features(args, out: Path):
    from early_warning import logspace as G7
    # stage_eval reads a plain CSV
    feats = out / "early_warning_features_logspace.csv"
    pd.read_csv(out / "early_warning_features_logspace.csv.gz").to_csv(feats, index=False)
    ns = argparse.Namespace(features_csv=str(feats.relative_to(c.ROOT)))
    os.chdir(c.ROOT)
    try:
        G7.stage_eval(ns, out)
    finally:
        feats.unlink(missing_ok=True)
def main_features():
    args = parse_args_features()
    out = c.ROOT / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (stage_features if args.stage == "features" else stage_eval_features)(args, out)


SUBCOMMANDS = {"eval": main_eval, "features": main_features}


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
