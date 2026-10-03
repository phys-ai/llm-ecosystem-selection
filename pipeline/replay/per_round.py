#!/usr/bin/env python3
"""Per-round exposure statistics of every endpoint (the per-round / time-averaged distinction of Sec. 4.3).

    python3 pipeline/replay/per_round.py main       main grids, all seeds (x86_64 Python 3.12 reproduces the stored endpoints) -> data/outputs/per_round_stats
    python3 pipeline/replay/per_round.py testbeds   robustness testbeds and the DeepSeek-evaluator replays                       -> data/outputs/per_round_stats_testbeds
"""

from __future__ import annotations

import sys
import argparse
import json
import platform
from multiprocessing import Pool
import numpy as np
import pandas as pd
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # pipeline/ on the path

from common import grids as c
from common import per_round as g
from common.testbeds import extract_seed


# ====================================================================================================
# main
# ====================================================================================================
SUMMARY_COLS = ["N_eff_round_mean", "top1_round_mean", "conc_round_frac", "alternation_index",
                "winner_change_rate", "tv_round_mean", "simultaneously_diverse", "descriptor_diverse",
                "phantom_new", "phantom_new_family", "label_collapse", "sensitive_any",
                "clip_share_window", "fallback_round_share_window"]
def parse_args_main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--sensitivity", default="data/outputs/E2_replay_mismatch/endpoint_sensitivity_flags.csv")
    p.add_argument("--g1", default="data/outputs/G1_exact_model/exact_model_residuals_by_endpoint.csv.gz",
                   help="G1 output; adds the clip / softmax-fallback shares of each endpoint")
    p.add_argument("--out-dir", default="data/outputs/per_round_stats")
    p.add_argument("--table-cache", default="")
    p.add_argument("--seeds", default=",".join(str(s) for s in c.SEEDS))
    p.add_argument("--endpoint-last-k", type=int, default=100)
    p.add_argument("--support-clip", type=float, default=8.0)
    p.add_argument("--workers", type=int, default=5)
    return p.parse_args()
def run_seed(job):
    seed = job["seed"]
    table = c.load_table(c.ROOT / job["runs_dir"] / f"run_popseed_{seed}.checkpoints", job["table_cache"] or None)
    n = table.n_agents
    cache = g.FixedFitnessCache(table, job["clip"])
    out = []
    for _, row in job["rows"].iterrows():
        params = c.params_from_row(row)
        states = c.replay_states(table, params, job["clip"], fixed_f=cache.get(params))
        ne, t1 = c.n_eff_top1(c.tail_average(states, job["last_k"]))
        rec = {"grid": row["grid"], "seed": seed, "eta": float(row["eta"]), "alpha": float(row["alpha"]),
               "beta_sup": float(row["beta_sup"]), "n_agents": n,
               "phase_label": str(row["phase_label"]),
               "coexistence": int(row["coexistence"]), "specialization": int(row["specialization"]),
               "N_eff_avg": float(row["n_eff"]), "top1_avg": float(row["top1_share"]),
               "i_exp_norm": float(row["i_exp_norm"]), "winner_switch_rate": float(row["winner_switch_rate"]),
               "N_eff_avg_replayed": ne, "top1_avg_replayed": t1,
               "replay_matches_stored": int(abs(ne - float(row["n_eff"])) <= 1e-6
                                            and abs(t1 - float(row["top1_share"])) <= 1e-9)}
        rec.update(g.per_round_stats(states, job["last_k"], n))
        out.append(rec)
    print(f"seed {seed} done", flush=True)
    return pd.DataFrame(out)
def main_main():
    args = parse_args_main()
    out_dir = c.ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = pd.concat([c.load_grid(k, d) for k, d in c.GRIDS.items()], ignore_index=True)
    jobs = [{"seed": int(s), "rows": grid.loc[grid.seed.eq(int(s))], "runs_dir": args.runs_dir,
             "table_cache": args.table_cache, "clip": args.support_clip, "last_k": args.endpoint_last_k}
            for s in args.seeds.split(",")]
    with Pool(args.workers) as pool:
        df = pd.concat(pool.map(run_seed, jobs, chunksize=1), ignore_index=True)
    df = g.add_round6_labels(df)
    df["eta_beta"] = df.eta * df.beta_sup
    df["eta_beta_bin"] = g.eta_beta_bin(df.eta_beta)
    key = ["grid", "seed", "eta", "alpha", "beta_sup"]
    sens = pd.read_csv(c.ROOT / args.sensitivity)[key + ["sensitive_any"]]
    n0 = len(df)
    df = df.merge(sens, on=key, how="left")
    assert len(df) == n0 and df.sensitive_any.notna().all(), "sensitivity flags do not cover the grid"
    g1 = pd.read_csv(c.ROOT / args.g1)[key + ["clip_share_window", "fallback_round_share", "fallback_round_share_window"]]
    df = df.merge(g1, on=key, how="left")
    assert len(df) == n0 and df.fallback_round_share.notna().all(), "G1 output does not cover the grid"
    df.to_csv(out_dir / "endpoint_per_round.csv", index=False)

    top1_thr, neff_thr = g.concentration_thresholds(int(df.n_agents.iloc[0]))
    for name, sub in [("all", df), ("support", df[df.grid == "support"]), ("no_support", df[df.grid == "no_support"])]:
        by = "eta_beta_bin" if name != "no_support" else "eta"
        t = g.seed_mean_se_table(sub, by, SUMMARY_COLS)
        dd = sub[sub.descriptor_diverse == 1]
        t = t.merge(g.seed_mean_se_table(dd, by, ["phantom_new"]).rename(
            columns={"phantom_new_mean": "phantom_share_of_descriptor_diverse_mean",
                     "phantom_new_se": "phantom_share_of_descriptor_diverse_se",
                     "n_endpoints": "n_descriptor_diverse"}).drop(columns="n_seeds"), on=by, how="left")
        t.to_csv(out_dir / f"summary_by_{by}__{name}.csv", index=False)
        print(name); print(t.filter(regex=f"{by}|_mean|n_endpoints").round(3).to_string())
    manifest = {
        "environment": f"{platform.machine()}_py{platform.python_version()}_numpy{np.__version__}",
        "n_endpoints": int(len(df)), "replay_matches_stored": int(df.replay_matches_stored.sum()),
        "window_last_k": args.endpoint_last_k, "n_agents": sorted(int(v) for v in df.n_agents.unique()),
        "concentrated_round_rule": {"top1_ge": top1_thr, "n_eff_le": neff_thr},
        "simultaneous_diversity_rule": f"conc_round_frac <= {g.SIM_DIVERSE_MAX_CONC_FRAC}",
        "phantom_new": "phase_label in coexistence/specialization labels and not simultaneously diverse",
        "phantom_new_family": "Fig. 4a family flags (coexistence | specialization_signal) and not simultaneously diverse",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


# ====================================================================================================
# testbeds
# ====================================================================================================
TESTBEDS = ["heterogeneity", "llama", "8d", "tool", "16d"]
def parse_args_testbeds():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--testbeds", default=",".join(TESTBEDS))
    p.add_argument("--work-dir", default="")
    p.add_argument("--deepseek", action="store_true")
    p.add_argument("--out-dir", default="data/outputs/per_round_stats_testbeds")
    p.add_argument("--min-free-gb", type=float, default=7.0)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()
def rows_for(table, sub: pd.DataFrame, last_k: int, extra: dict) -> list:
    n = table.n_agents
    cache = g.FixedFitnessCache(table, 8.0)
    out = []
    for _, row in sub.iterrows():
        params = c.params_from_row(row)
        states = c.replay_states(table, params, 8.0, fixed_f=cache.get(params))
        ne, t1 = c.n_eff_top1(c.tail_average(states, last_k))
        rec = {**extra, "grid": row["grid"], "eta": float(row["eta"]), "alpha": float(row["alpha"]),
               "beta_sup": float(row["beta_sup"]), "n_agents": n, "window": last_k,
               "phase_label": str(row["phase_label"]),
               "coexistence": int(str(row["phase_label"]) in c.COEXISTENCE_LABELS),
               "specialization": int(row["specialization_signal"]),
               "N_eff_avg": float(row["n_eff"]), "top1_avg": float(row["top1_share"]),
               "replay_matches_stored": int(abs(ne - float(row["n_eff"])) <= 1e-6 and abs(t1 - float(row["top1_share"])) <= 1e-9)}
        rec.update(g.per_round_stats(states, last_k, n))
        out.append(rec)
    return out
def run_testbed(name: str, args) -> pd.DataFrame:
    grids = []
    for kind, grid_name in [("main", "no_support"), ("support", "support")]:
        d = c.ROOT / f"data/testbeds/verification_{name}_{kind}"
        cfg = json.loads((d / "run_config.json").read_text())
        gdf = pd.read_csv(d / "phase_summary.csv", float_precision="round_trip")
        gdf["grid"] = grid_name
        gdf["seed"] = gdf["run_name"].astype(str).str.extract(r"popseed_(\d+)", expand=False).astype(int)
        grids.append(gdf)
        last_k, weights = int(cfg["args"]["endpoint_last_k"]), cfg["score_weights"]
    grid = pd.concat(grids, ignore_index=True)
    work = Path(args.work_dir) / f"g8_{name}"
    rows = []
    for seed, sub in grid.groupby("seed"):
        t0 = time.time()
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        free_gb = shutil.disk_usage(work).free / 1e9
        if free_gb < args.min_free_gb:
            raise SystemExit(f"only {free_gb:.1f} GB free; refusing to extract a seed of {name}")
        try:
            src = extract_seed(name, seed, work)
            table = c.replay.prepare_log_streaming(src, src.name, weights)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        rows += rows_for(table, sub, last_k, {"setting": name, "seed": int(seed)})
        print(f"{name} seed {seed}: {len(sub)} endpoints, {time.time() - t0:.0f} s", flush=True)
    return pd.DataFrame(rows)
def _deepseek_seed(job):
    src = c.ROOT / "runs" / f"run_popseed_{job['seed']}_reuse_together_evalaudit_compact.checkpoints"
    table = c.replay.prepare_log_streaming(src, src.name, c.replay.DEFAULT_SCORE_WEIGHTS)
    return pd.DataFrame(rows_for(table, job["rows"], 100, {"setting": "deepseek", "seed": job["seed"]}))
def run_deepseek(args) -> pd.DataFrame:
    from common import deepseek as F1
    grid = F1.load_grids("data/results_reuse_clean")
    grid = grid[grid.grid.isin(["no_support", "support"])]
    jobs = [{"seed": int(s), "rows": grid[grid.seed == s]} for s in F1.SEEDS]
    with Pool(args.workers) as pool:
        return pd.concat(pool.map(_deepseek_seed, jobs, chunksize=1), ignore_index=True)
def main_testbeds():
    args = parse_args_testbeds()
    out_dir = c.ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    env = f"{platform.machine()}_py{platform.python_version()}_numpy{np.__version__}"
    todo = ["deepseek"] if args.deepseek else [t for t in args.testbeds.split(",") if t]
    for name in todo:
        df = run_deepseek(args) if name == "deepseek" else run_testbed(name, args)
        df = g.add_round6_labels(df)
        df["eta_beta"] = df.eta * df.beta_sup
        df["environment"] = env
        df.to_csv(out_dir / f"endpoint_per_round_{name}.csv", index=False)
        print(name, len(df), "endpoints; not matching stored:", int((1 - df.replay_matches_stored).sum()), flush=True)


SUBCOMMANDS = {"main": main_main, "testbeds": main_testbeds}


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
