#!/usr/bin/env python3
"""Per-round vs. time-averaged diversity (Sec. 4.3, Fig. 4(b), Fig. 11): the rotating-winner test.

Per-step exposure vectors are NOT stored on disk for either grid (the grid
directories hold only phase_summary.csv / stability_summary.csv; ``runs/`` holds
the raw interaction logs).  The replay is deterministic given those logs, so the
per-step vectors p_1..p_200 are regenerated here with the paper's replay loop
(verified against the stored n_eff / top1_share of every endpoint), and only
the window statistics are kept.

Window = the last 100 states of the 200-step replay (the states that the paper
averages into the endpoint p*).  Per endpoint:

    N_eff_avg, top1_avg    of the time-averaged p*      (paper descriptors)
    N_eff_inst, top1_inst  mean over the 100 per-step states
    winner_distinct        number of distinct per-step top-one agents
    winner_mean_run        mean run length of the per-step top-one agent
    tv_step                mean TV between consecutive per-step states
    frac_steps_collapsed   share of window steps whose per-step state meets the
                           paper's collapse descriptor (top1>=0.8 or N_eff<=max(1.5, 0.025 N))

"Average-only diversity": an any-diversity
descriptor-positive endpoint with N_eff_inst <= 3 or top1_inst >= 0.8.

Run from the package root:
    python3 pipeline/replay/per_round_vs_average.py
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))  # pipeline/ on the path

from common import grids as c

NEFF_INST_MAX = 3.0
TOP1_INST_MIN = 0.8


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--labels", default="data/results/threshold_robustness/phantom_breakdown_labels.csv.gz")
    p.add_argument("--out-dir", default="data/outputs/per_round_vs_average")
    p.add_argument("--table-cache", default="")
    p.add_argument("--seeds", default=",".join(str(s) for s in c.SEEDS))
    p.add_argument("--endpoint-last-k", type=int, default=100)
    p.add_argument("--support-clip", type=float, default=8.0)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--reuse-existing", action="store_true")
    return p.parse_args()


def run_seed(job: Dict[str, Any]) -> pd.DataFrame:
    seed = int(job["seed"])
    table = c.load_table(c.ROOT / job["runs_dir"] / f"run_popseed_{seed}.checkpoints", job["table_cache"] or None)
    n = table.n_agents
    thr = c.replay.PHASE_THRESHOLDS
    collapse_neff = max(thr["collapse_neff_abs"], thr["collapse_neff_frac"] * n)
    out: List[Dict[str, Any]] = []
    for _, row in job["rows"].iterrows():
        params = c.params_from_row(row)
        states = c.replay_states(table, params, job["support_clip"])
        window = states[-int(job["last_k"]):]
        p_star = c.tail_average(states, job["last_k"])
        n_eff_avg, top1_avg = c.n_eff_top1(p_star)
        n_eff_t = 1.0 / np.sum(window * window, axis=1)
        top1_t = window.max(axis=1)
        winners = window.argmax(axis=1)
        n_runs = 1 + int(np.sum(winners[1:] != winners[:-1]))
        out.append({
            "grid": row["grid"], "seed": seed, "eta": float(row["eta"]), "alpha": float(row["alpha"]),
            "beta_sup": float(row["beta_sup"]), "kappa_map": float(row["kappa_map"]),
            "locally_contracting": int(row["locally_contracting"]), "phase_label": str(row["phase_label"]),
            "coexistence": int(row["coexistence"]), "specialization": int(row["specialization"]),
            "polarization": int(row["polarization"]), "any_diversity": int(row["any_diversity"]),
            "phantom_any": int(row["phantom_any"]),
            "collapse_signal_avg": int(row["collapse_signal"]),
            "N_eff_avg": n_eff_avg, "top1_avg": top1_avg,
            "N_eff_avg_minus_stored": n_eff_avg - float(row["n_eff"]),
            "top1_avg_minus_stored": top1_avg - float(row["top1_share"]),
            "N_eff_inst": float(n_eff_t.mean()), "top1_inst": float(top1_t.mean()),
            "N_eff_inst_median": float(np.median(n_eff_t)), "top1_inst_median": float(np.median(top1_t)),
            "frac_steps_collapsed": float(np.mean((top1_t >= thr["collapse_top1"]) | (n_eff_t <= collapse_neff))),
            "winner_distinct": int(np.unique(winners).size),
            "winner_n_runs": n_runs, "winner_mean_run": float(len(winners) / n_runs),
            "tv_step": float(0.5 * np.abs(np.diff(window, axis=0)).sum(axis=1).mean()),
        })
    print(f"seed {seed}: {len(out)} endpoints done", flush=True)
    return pd.DataFrame(out)


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["inst_neff_le3"] = (df["N_eff_inst"] <= NEFF_INST_MAX).astype(int)
    df["inst_top1_ge08"] = (df["top1_inst"] >= TOP1_INST_MIN).astype(int)
    df["average_only"] = ((df["inst_neff_le3"] == 1) | (df["inst_top1_ge08"] == 1)).astype(int)
    df["replay_matches_stored"] = (
        (df["N_eff_avg_minus_stored"].abs() <= 1.0e-6) & (df["top1_avg_minus_stored"].abs() <= 1.0e-9)
    ).astype(int)
    df["status"] = np.where(df["locally_contracting"].eq(1), "locally contracting", "phantom")
    return df


def share_row(group: pd.DataFrame, **keys: Any) -> Dict[str, Any]:
    rec: Dict[str, Any] = {**keys, "n_descriptor_positive": int(len(group))}
    for col in ["inst_neff_le3", "inst_top1_ge08", "average_only"]:
        rec[f"share_{col}"] = float(group[col].mean()) if len(group) else float("nan")
        m, se = c.seed_mean_se(group.groupby("seed")[col].mean())
        rec[f"share_{col}_seed_mean"], rec[f"share_{col}_seed_se"] = m, se
    for col in ["N_eff_avg", "N_eff_inst", "top1_avg", "top1_inst", "winner_distinct", "winner_mean_run",
                "tv_step", "frac_steps_collapsed"]:
        rec[f"median_{col}"] = float(group[col].median()) if len(group) else float("nan")
    return rec


def summarize(df: pd.DataFrame, out_dir: Path) -> None:
    div = df.loc[df["any_diversity"].eq(1)]
    rows = []
    scopes = [(g, div.loc[div["grid"].eq(g)]) for g in c.GRIDS] + [("both_grids_rows_as_stored", div)]
    # sensitivity: only endpoints whose replayed p* reproduces the stored n_eff / top1_share
    ok = div.loc[div["replay_matches_stored"].eq(1)]
    scopes += [("support__replay_matched_only", ok.loc[ok["grid"].eq("support")]),
               ("both_grids__replay_matched_only", ok)]
    for scope, sub in scopes:
        for fam, fsub in [("any_diversity", sub), ("coexistence", sub.loc[sub["coexistence"].eq(1)]),
                          ("specialization", sub.loc[sub["specialization"].eq(1)])]:
            for cond, csub in [("all descriptor-positive", fsub),
                               ("descriptor-positive and average not collapsed (collapse_signal=0)",
                                fsub.loc[fsub["collapse_signal_avg"].eq(0)])]:
                rows.append(share_row(csub, scope=scope, family=fam, condition=cond, status="all"))
                for status, ssub in csub.groupby("status"):
                    rows.append(share_row(ssub, scope=scope, family=fam, condition=cond, status=status))
    pd.DataFrame(rows).to_csv(out_dir / "summary_average_only_share.csv", index=False)

    rows = []
    for (grid, eta), sub in div.groupby(["grid", "eta"]):
        rows.append(share_row(sub, grid=grid, eta=float(eta), status="all"))
        for status, ssub in sub.groupby("status"):
            rows.append(share_row(ssub, grid=grid, eta=float(eta), status=status))
    pd.DataFrame(rows).to_csv(out_dir / "summary_average_only_by_eta.csv", index=False)

    rows = []
    for (grid, beta), sub in div.groupby(["grid", "beta_sup"]):
        rows.append(share_row(sub, grid=grid, beta_sup=float(beta), status="all"))
        for status, ssub in sub.groupby("status"):
            rows.append(share_row(ssub, grid=grid, beta_sup=float(beta), status=status))
    pd.DataFrame(rows).to_csv(out_dir / "summary_average_only_by_beta_sup.csv", index=False)

    sup = div.loc[div["grid"].eq("support")]
    cell = sup.groupby(["eta", "beta_sup"]).agg(
        n_descriptor_positive=("average_only", "size"), share_average_only=("average_only", "mean"),
        share_inst_neff_le3=("inst_neff_le3", "mean"), share_inst_top1_ge08=("inst_top1_ge08", "mean"),
        share_phantom=("phantom_any", "mean"),
    ).reset_index()
    cell.to_csv(out_dir / "heatmap_average_only_support_grid.csv", index=False)


def make_figures(df: pd.DataFrame, out_dir: Path) -> None:
    c.set_paper_style()
    import matplotlib.pyplot as plt

    div = df.loc[df["any_diversity"].eq(1)]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharex=True, sharey=True)
    for ax, (grid, title) in zip(axes, [("no_support", "(a) no-support grid"), ("support", "(b) support grid")]):
        sub = div.loc[div["grid"].eq(grid)]
        for status, color in [("phantom", c.PHANTOM_COLOR), ("locally contracting", c.STABLE_COLOR)]:
            s = sub.loc[sub["status"].eq(status)]
            ax.scatter(s["N_eff_avg"], s["N_eff_inst"], s=5, alpha=0.25, color=color, linewidths=0,
                       rasterized=True, label=f"{status} (n={len(s):,})")
        ax.plot([1, 96], [1, 96], color="black", linewidth=0.9)
        ax.axhline(NEFF_INST_MAX, color="black", linestyle=":", linewidth=0.9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$N_{\mathrm{eff}}$ of time-averaged endpoint $p^*$")
        ax.set_title(title, loc="left")
        leg = ax.legend(frameon=False, loc="upper left", markerscale=3)
        for h in leg.legend_handles:
            h.set_alpha(1.0)
    axes[0].set_ylabel(r"mean per-step $N_{\mathrm{eff}}$ (last 100 steps)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_d2_neff_avg_vs_inst.pdf", dpi=300)
    fig.savefig(out_dir / "fig_d2_neff_avg_vs_inst.png", dpi=200)
    plt.close(fig)

    cell = pd.read_csv(out_dir / "heatmap_average_only_support_grid.csv")
    etas = np.sort(df.loc[df["grid"].eq("support"), "eta"].unique())
    betas = np.sort(df.loc[df["grid"].eq("support"), "beta_sup"].unique())
    mat = cell.pivot(index="beta_sup", columns="eta", values="share_average_only").reindex(index=betas, columns=etas)
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#DDDDDD")
    im = ax.imshow(np.ma.masked_invalid(mat.to_numpy()), origin="lower", aspect="auto", cmap=cmap, vmin=0, vmax=1)
    xt = np.arange(0, len(etas), 4)
    yt = np.arange(0, len(betas), 4)
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{etas[i]:g}" for i in xt], rotation=45, ha="right")
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{betas[i]:g}" for i in yt])
    ax.set_xlabel(r"selection strength $\eta$ (grid index)")
    ax.set_ylabel(r"support strength $\beta_{\mathrm{sup}}$ (grid index)")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("share of descriptor-positive endpoints\nwith per-step $N_{\\mathrm{eff}}\\leq 3$ or top-1 $\\geq 0.8$")
    ax.set_title("Average-only diversity, support grid (10 seeds per cell; grey = no descriptor-positive endpoint)",
                 fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_d2_average_only_heatmap.pdf")
    fig.savefig(out_dir / "fig_d2_average_only_heatmap.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    started = time.time()
    out_dir = c.ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "endpoint_instantaneous.csv"
    if args.reuse_existing:
        df = pd.read_csv(csv_path)
    else:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        grids = []
        for grid, grid_dir in c.GRIDS.items():
            g = c.load_grid(grid, grid_dir)
            assert c.check_against_fig4_labels(g, grid_dir, args.labels)["n_flag_mismatch"] == 0
            grids.append(g)
        allrows = pd.concat(grids, ignore_index=True)
        jobs = [{"seed": s, "rows": allrows.loc[allrows["seed"].eq(s)], "runs_dir": args.runs_dir,
                 "table_cache": args.table_cache, "support_clip": float(args.support_clip),
                 "last_k": int(args.endpoint_last_k)} for s in seeds]
        with Pool(processes=max(1, int(args.workers))) as pool:
            frames = pool.map(run_seed, jobs, chunksize=1)
        df = pd.concat(frames, ignore_index=True).sort_values(["grid", "seed", "eta", "alpha", "beta_sup"])
    df = add_flags(df)
    df.to_csv(csv_path, index=False)
    summarize(df, out_dir)
    make_figures(df, out_dir)
    manifest = {
        "elapsed_sec": float(time.time() - started), "n_endpoints": int(len(df)),
        "per_step_exposure_vectors_stored_on_disk": False,
        "per_step_source": "deterministic replay of runs/run_popseed_<seed>.checkpoints (paper replay loop)",
        "window": f"last {args.endpoint_last_k} states of the 200-step replay",
        "average_only_definition": f"N_eff_inst <= {NEFF_INST_MAX:g} or top1_inst >= {TOP1_INST_MIN:g}",
        "n_replay_not_matching_stored_by_grid": {
            k: int(v) for k, v in (1 - df["replay_matches_stored"]).groupby(df["grid"]).sum().items()},
        "max_abs_N_eff_avg_minus_stored": float(df["N_eff_avg_minus_stored"].abs().max()),
        "max_abs_top1_avg_minus_stored": float(df["top1_avg_minus_stored"].abs().max()),
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
