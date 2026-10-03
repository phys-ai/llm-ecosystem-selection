#!/usr/bin/env python3
"""Log-space replay of every grid (the update rule of replay/logspace.py) and the adapter that writes its outputs in the
per-round CSV formats read by the early-warning scripts.

    python3 pipeline/replay/logspace_grids.py replay --grids main_no_support,main_support   -> data/outputs/replay_logspace_grids
    python3 pipeline/replay/logspace_grids.py adapt                                         -> data/outputs/replay_logspace_adapted
"""

from __future__ import annotations

import sys
import argparse
import glob
import json
import platform
import re
import shutil
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # pipeline/ on the path

from common import grids as c
from common import per_round as g
from replay import logspace as L


# ====================================================================================================
# replay
# ====================================================================================================
R = c.replay
STABILITY_COLS = ["active_eps", "active_count", "inactive_count", "tangent_dim", "spectral_radius", "gamma_map",
                  "eig_abs_max", "eig_real_max", "max_invasion_log", "max_invasion_agent", "kappa_map",
                  "locally_contracting", "stability_context_stride", "criticality_label"]
RUN_GRIDS = {  # grid name -> (results dir with run_popseed_*/phase_summary.csv, runs-dir pattern)
    "main_no_support": ("data/results/fig0_4_full_grid", "runs/run_popseed_{seed}.checkpoints"),
    "main_support": ("data/results/fig5_full_grid", "runs/run_popseed_{seed}.checkpoints"),
    "route_sweep": ("data/results/appendix_route_sweep_parallel", "runs/run_popseed_{seed}.checkpoints"),
    "early_warning": ("data/results/early_warning_validation_grid", "runs/run_popseed_{seed}.checkpoints"),
    "deepseek_no_support": ("data/results_reuse_clean/fig0_4_full_grid", "runs/run_popseed_{seed}_reuse_together_evalaudit_compact.checkpoints"),
    "deepseek_support": ("data/results_reuse_clean/fig5_full_grid", "runs/run_popseed_{seed}_reuse_together_evalaudit_compact.checkpoints"),
}
DEEPSEEK_SEEDS = [11, 22, 33, 44]  # seed 55 has a truncated trajectory (F1)
TESTBEDS = ["heterogeneity", "llama", "8d", "tool", "16d"]
def parse_args_replay():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grids", default="main_no_support,main_support",
                   help="comma list of " + ",".join(RUN_GRIDS) + ",testbeds (or testbed names)")
    p.add_argument("--out-dir", default="data/outputs/replay_logspace_grids")
    p.add_argument("--table-cache", default="")
    p.add_argument("--work-dir", default="")
    p.add_argument("--perturb", type=float, default=1.0e-15)
    p.add_argument("--min-free-gb", type=float, default=7.0)
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()
def context_mi_and_winners_log(table, endpoint, a, eta, beta, context_key):
    """``replay.context_mi_and_winners`` with the log-space one-step response (vectorised over contexts)."""
    if context_key not in table.context_df.columns:
        context_key = "topic_category"
    cvals = table.context_df[context_key].fillna("uncategorized").astype(str).to_numpy()
    z = np.log(endpoint)
    w = z[None, :] + eta * (a + beta * (-np.log(table.n_agents) - z)[None, :])
    w -= w.max(axis=1, keepdims=True)
    q = np.exp(w)
    q /= q.sum(axis=1, keepdims=True)
    winner_counts = np.bincount(q.argmax(axis=1), minlength=table.n_agents)
    px = q.mean(axis=0)
    cats, inv = np.unique(cvals, return_inverse=True)
    mi, pc_list = 0.0, []
    for k in range(cats.size):
        m = inv == k
        pc = float(m.mean()); pc_list.append(pc)
        q_c = q[m].mean(axis=0)
        mi += float(pc * np.sum(q_c * np.log(q_c / (px + R.EPS) + R.EPS)))
    hc = R.safe_entropy(np.asarray(pc_list))
    freqs = winner_counts / max(float(winner_counts.sum()), 1.0)
    return {"i_exp": float(mi), "i_exp_norm": float(mi / max(hc, R.EPS)) if hc > R.EPS else 0.0,
            "context_count": int(cats.size), "winner_switch_rate": float(1.0 - np.sum(freqs * freqs)),
            "winner_unique_count": int(np.sum(winner_counts > 0)), "winner_top_share": float(np.max(freqs))}
def process(table, rows: pd.DataFrame, last_k: int, context_key: str, perturb: float, extra: dict) -> pd.DataFrame:
    n = table.n_agents
    cache = g.FixedFitnessCache(table, 8.0)
    z0p = np.ones(n) / n
    z0p[0] += perturb
    z0p = np.log(z0p / z0p.sum())
    out = []
    for _, row in rows.iterrows():
        params = c.params_from_row(row)
        assert float(getattr(params, "polarization_strength", 0.0)) == 0.0
        a = cache.get(params)
        eta = 0.0 if bool(params.no_feedback) else float(params.eta)
        Z = L.replay_log_states(table, params, fixed_f=a)
        P = np.exp(Z)
        endpoint = c.tail_average(P, last_k)
        desc = {}
        desc.update(R.concentration_descriptors(endpoint))
        desc.update(R.weighted_trait_descriptors(table, endpoint))
        desc.update(R.risk_quality_descriptors(table, endpoint))
        desc.update(context_mi_and_winners_log(table, endpoint, a, eta, float(params.beta_sup), context_key))
        desc.update(R.descriptor_flags(desc, n))
        desc["phase_label"] = R.phase_label(desc, n)
        desc["phase_code"] = R.PHASE_CODES.get(str(desc["phase_label"]), -1)
        rec = {k: row[k] for k in rows.columns if k not in desc and k not in STABILITY_COLS and not k.startswith("_")}
        rec.update(extra)
        rec.update(desc)
        rec.update(g.per_round_stats(P, last_k, n))
        ep = c.tail_average(np.exp(L.replay_log_states(table, params, fixed_f=a, z0=z0p)), last_k)
        rec.update({"window": last_k, "n_nonfinite_z": int((~np.isfinite(Z)).sum()), "max_abs_z": float(np.abs(Z).max()),
                    "n_zero_p_exp_z": int((P == 0).sum()),
                    "n_eff_perturbed": float(1.0 / np.sum(ep * ep)), "top1_perturbed": float(ep.max()),
                    "stored_n_eff": float(row["n_eff"]), "stored_top1_share": float(row["top1_share"]),
                    "stored_i_exp_norm": float(row["i_exp_norm"]), "stored_winner_switch_rate": float(row["winner_switch_rate"]),
                    "stored_phase_label": str(row["phase_label"])})
        out.append(rec)
    return pd.DataFrame(out)
def _run_seed(job):
    table = c.load_table(c.ROOT / job["src"], job["table_cache"] or None) if job["default_weights"] else \
        R.prepare_log_streaming(c.ROOT / job["src"], Path(job["src"]).name, job["weights"])
    df = process(table, job["rows"], job["last_k"], job["context_key"], job["perturb"], {"seed": job["seed"]})
    print(f"{job['grid']} seed {job['seed']}: {len(df)} endpoints", flush=True)
    return df
def run_grid(name: str, args) -> pd.DataFrame:
    res_dir, runs_pat = RUN_GRIDS[name]
    jobs = []
    for path in sorted(glob.glob(str(c.ROOT / res_dir / "run_popseed_*" / "phase_summary.csv"))):
        seed = int(re.search(r"popseed_(\d+)", path).group(1))
        if name.startswith("deepseek") and seed not in DEEPSEEK_SEEDS:
            continue
        cfg = json.loads((Path(path).parent / "run_config.json").read_text())
        default_w = cfg["score_weights"] == dict(R.DEFAULT_SCORE_WEIGHTS)
        jobs.append({"grid": name, "seed": seed, "rows": pd.read_csv(path, float_precision="round_trip"),
                     "src": runs_pat.format(seed=seed), "last_k": int(cfg["args"]["endpoint_last_k"]),
                     "context_key": cfg["args"]["context_key"], "perturb": args.perturb, "weights": cfg["score_weights"],
                     "default_weights": default_w and not name.startswith("deepseek"), "table_cache": args.table_cache})
    with Pool(args.workers) as pool:
        return pd.concat(pool.map(_run_seed, jobs, chunksize=1), ignore_index=True)
def run_testbed(name: str, kind: str, args, tables: dict) -> pd.DataFrame:
    from common.testbeds import extract_seed
    d = c.ROOT / f"data/testbeds/verification_{name}_{kind}"
    cfg = json.loads((d / "run_config.json").read_text())
    grid = pd.read_csv(d / "phase_summary.csv", float_precision="round_trip")
    grid["seed"] = grid["run_name"].astype(str).str.extract(r"popseed_(\d+)", expand=False).astype(int)
    frames = []
    for seed, sub in grid.groupby("seed"):
        if (name, seed) not in tables:
            work = Path(args.work_dir) / f"h2_{name}"
            shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
            free_gb = shutil.disk_usage(work).free / 1e9
            if free_gb < args.min_free_gb:
                raise SystemExit(f"only {free_gb:.1f} GB free; refusing to extract a seed of {name}")
            try:
                src = extract_seed(name, seed, work)
                tables[(name, seed)] = R.prepare_log_streaming(src, src.name, cfg["score_weights"])
            finally:
                shutil.rmtree(work, ignore_errors=True)
        frames.append(process(tables[(name, seed)], sub.drop(columns="seed"), int(cfg["args"]["endpoint_last_k"]),
                              cfg["args"]["context_key"], args.perturb, {"seed": int(seed)}))
        print(f"{name}_{kind} seed {seed}: {len(sub)} endpoints", flush=True)
    return pd.concat(frames, ignore_index=True)
def finish(name: str, df: pd.DataFrame, args, env: str) -> None:
    out = c.ROOT / args.out_dir / name
    out.mkdir(parents=True, exist_ok=True)
    df = g.add_round6_labels(df)
    df["eta_beta"] = df.eta * df.beta_sup
    df["sensitive_perturbation"] = (((df.n_eff_perturbed - df.n_eff).abs() > 1e-6) | ((df.top1_perturbed - df.top1_share).abs() > 1e-9)).astype(int)
    df["environment"] = env
    df.to_csv(out / "phase_summary.csv.gz", index=False)
    chk = df[(df.beta_sup == 0) & (df.eta < 1.75)]
    self_check = {"n_rows_checked": int(len(chk))}
    if len(chk):
        self_check.update({"max_abs_diff_n_eff": float((chk.n_eff - chk.stored_n_eff).abs().max()),
                           "max_abs_diff_top1": float((chk.top1_share - chk.stored_top1_share).abs().max()),
                           "max_abs_diff_i_exp_norm": float((chk.i_exp_norm - chk.stored_i_exp_norm).abs().max()),
                           "max_abs_diff_winner_switch": float((chk.winner_switch_rate - chk.stored_winner_switch_rate).abs().max()),
                           "n_phase_label_differs": int((chk.phase_label != chk.stored_phase_label).sum())})
    manifest = {"grid": name, "environment": env, "n_endpoints": int(len(df)), "n_seeds": int(df.seed.nunique()),
                "window": sorted(int(v) for v in df.window.unique()), "n_nonfinite_z": int(df.n_nonfinite_z.sum()),
                "max_abs_z": float(df.max_abs_z.max()), "endpoints_with_zero_exp_z": int((df.n_zero_p_exp_z > 0).sum()),
                "share_phase_label_changed_vs_stored": float((df.phase_label != df.stored_phase_label).mean()),
                "self_check_beta0_eta_lt_1.75": self_check}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
def main_replay():
    args = parse_args_replay()
    env = f"{platform.machine()}_py{platform.python_version()}_numpy{np.__version__}"
    names = []
    for k in args.grids.split(","):
        names += TESTBEDS if k == "testbeds" else [k]
    for name in names:
        if name in RUN_GRIDS:
            finish(name, run_grid(name, args), args, env)
        else:
            assert name in TESTBEDS and args.work_dir, "testbeds need --work-dir"
            tables: dict = {}
            for kind, label in [("main", "no_support"), ("support", "support")]:
                finish(f"{name}_{label}", run_testbed(name, kind, args, tables), args, env)


# ====================================================================================================
# adapt
# ====================================================================================================
KEY = ["seed", "eta", "alpha", "beta_sup", "beta_route"]
TESTBEDS_adapt = ["heterogeneity", "llama", "8d", "tool", "16d"]
def parse_args_adapt():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h2-dir", default="data/outputs/replay_logspace_grids")
    p.add_argument("--arm64-dir", default="data/outputs/H2_logspace_replay_arm64")
    p.add_argument("--out-dir", default="data/outputs/replay_logspace_adapted")
    return p.parse_args()
def load(h2, name):
    f = h2 / name / "phase_summary.csv.gz"
    return pd.read_csv(f) if f.exists() else None
def to_g2_format(df: pd.DataFrame, grid: str) -> pd.DataFrame:
    df = df.copy()
    df["grid"] = grid
    df["N_eff_avg"], df["top1_avg"] = df["n_eff"], df["top1_share"]
    df["coexistence"] = df.phase_label.isin(c.COEXISTENCE_LABELS).astype(int)
    df["specialization"] = df.specialization_signal.astype(int)
    df = g.add_round6_labels(df)
    df["eta_beta_bin"] = g.eta_beta_bin(df.eta_beta)
    df["replay_matches_stored"] = 1  # not applicable: a different update rule; kept for the scripts' tables
    df["clip_share_window"] = 0.0
    df["fallback_round_share"] = 0.0
    df["fallback_round_share_window"] = 0.0
    return df
def main_adapt():
    args = parse_args_adapt()
    h2, arm, out = c.ROOT / args.h2_dir, c.ROOT / args.arm64_dir, c.ROOT / args.out_dir
    report = {}
    frames = []
    for name, grid in [("main_no_support", "no_support"), ("main_support", "support")]:
        d = to_g2_format(load(h2, name), grid)
        d["sensitive_any"] = d["sensitive_perturbation"]
        a = load(arm, name)
        if a is not None:
            m = d[KEY].merge(a[KEY + ["n_eff", "top1_share", "sensitive_perturbation"]], on=KEY, how="left", validate="one_to_one")
            assert m.n_eff.notna().all()
            cross = ((m.n_eff.to_numpy() - d.n_eff.to_numpy()).__abs__() > 1e-6) | (np.abs(m.top1_share.to_numpy() - d.top1_share.to_numpy()) > 1e-9)
            d["sensitive_cross_platform"] = cross.astype(int)
            d["sensitive_perturbation_arm64"] = m.sensitive_perturbation.to_numpy()
            d["sensitive_any"] = ((d.sensitive_perturbation == 1) | cross | (m.sensitive_perturbation.to_numpy() == 1)).astype(int)
        report[name] = {"n": int(len(d)), "arm64_copy_used": a is not None, "sensitive_any": int(d.sensitive_any.sum())}
        frames.append(d)
    (out / "per_round_stats").mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(out / "per_round_stats" / "endpoint_per_round.csv", index=False)

    (out / "per_round_stats_testbeds").mkdir(parents=True, exist_ok=True)
    for setting in TESTBEDS_adapt + ["deepseek"]:
        parts = [(load(h2, f"{setting}_no_support"), "no_support"), (load(h2, f"{setting}_support"), "support")]
        if any(p[0] is None for p in parts):
            print("missing H2 output for", setting); continue
        d = pd.concat([to_g2_format(p, gname) for p, gname in parts], ignore_index=True)
        d["setting"] = setting
        d.to_csv(out / "per_round_stats_testbeds" / f"endpoint_per_round_{setting}.csv", index=False)
        report[setting] = {"n": int(len(d))}

    rs = load(h2, "route_sweep")
    if rs is not None:
        for seed, sub in rs.groupby("seed"):
            d = out / "route_sweep" / f"run_popseed_{seed}.checkpoints"
            d.mkdir(parents=True, exist_ok=True)
            sub.drop(columns="seed").to_csv(d / "phase_summary.csv", index=False)
        report["route_sweep"] = {"n": int(len(rs))}
    (out / "adapter_manifest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


SUBCOMMANDS = {"replay": main_replay, "adapt": main_adapt}


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
