#!/usr/bin/env python3
"""Ranking / routing pairings of the log-space replay (the per-round state and the context it was scored against).

Index convention of the code (``common.logspace.replay_batch``): Z[t] is the log-exposure state
after processing logged rounds 0..t-1, so Z[t+1] contains a(c_t).  For the context window
t = T-K .. T-1 (the rounds that feed the endpoint window of H2):

  ranking   p_t := exp(Z[t+1])   paired with c_t   (exposure already uses a(c_t))
  routing   p_t := exp(Z[t])     paired with c_t   (exposure uses scores up to c_{t-1})

Per-round statistics of the two are the same up to the one-round shift; the ranking window is
exactly the window of ``common.per_round.per_round_stats`` (last K states).

Score cache: one gzip pickle per (source, seed) with the state-independent composite score a
(T x N) for every fitness key of the grids that use the source, the raw score s
(``table.base_score``) and the topic category of every round.  Main-grid tables come from the
parsed-table cache; DeepSeek logs are parsed from ``runs/``; the robustness testbeds are read one seed
at a time from the tarballs into --work-dir and removed after parsing.
"""
from __future__ import annotations

import os

import argparse
import gzip
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))  # pipeline/ on the path

from common import grids as c
from common import per_round as g
from common import logspace as j

R = c.replay
H2_DIR = "data/outputs/replay_logspace_grids"
DEFAULT_SCORE_CACHE = os.environ.get("EVOTHEORY_CACHE", str(Path(__file__).resolve().parents[2] / ".cache" / "score_cache"))  # parse cache of the raw logs (optional)
# grid (H2 output dir) -> (source, information structure of the testbed)
GRIDS = {
    "main_no_support": ("main", "ranking"), "main_support": ("main", "ranking"), "route_sweep": ("main", "ranking"),
    "deepseek_no_support": ("deepseek", "ranking"), "deepseek_support": ("deepseek", "ranking"),
    "heterogeneity_no_support": ("heterogeneity", "ranking"), "heterogeneity_support": ("heterogeneity", "ranking"),
    "llama_no_support": ("llama", "ranking"), "llama_support": ("llama", "ranking"),
    "8d_no_support": ("8d", "ranking"), "8d_support": ("8d", "ranking"),
    "16d_no_support": ("16d", "ranking"), "16d_support": ("16d", "ranking"),
    "tool_no_support": ("tool", "routing"), "tool_support": ("tool", "routing"),
}
FIT_KEY_COLS = ["alpha", "beta_route", "lambda_risk", "context_trait_strength", "route_power", "route_zscore", "ablation"]


def fit_key(params) -> tuple:
    return (float(params.alpha), float(params.beta_route), float(params.lambda_risk),
            float(params.context_trait_strength), float(params.route_power),
            bool(params.route_zscore), str(params.ablation))


def load_h2(grid: str) -> pd.DataFrame:
    return pd.read_csv(c.ROOT / H2_DIR / grid / "phase_summary.csv.gz")


def cache_path(cache_dir, source: str, seed: int) -> Path:
    return Path(cache_dir) / f"{source}_seed{seed}.pkl.gz"


def _score_weights(source: str) -> dict:
    if source in ("main", "deepseek"):
        return dict(R.DEFAULT_SCORE_WEIGHTS)
    return json.loads((c.ROOT / f"data/testbeds/verification_{source}_support/run_config.json").read_text())["score_weights"]


def _load_table(source: str, seed: int, table_cache: str, work_dir: str, min_free_gb: float):
    if source == "main":
        return c.load_table(c.ROOT / "runs" / f"run_popseed_{seed}.checkpoints", table_cache)
    if source == "deepseek":
        src = c.ROOT / "runs" / f"run_popseed_{seed}_reuse_together_evalaudit_compact.checkpoints"
        return R.prepare_log_streaming(src, src.name, _score_weights(source))
    from common.testbeds import extract_seed
    assert work_dir, "testbeds need --work-dir"
    work = Path(work_dir) / f"l0_{source}"
    shutil.rmtree(work, ignore_errors=True); work.mkdir(parents=True)
    if shutil.disk_usage(work).free / 1e9 < min_free_gb:
        raise SystemExit(f"less than {min_free_gb} GB free; refusing to extract {source} seed {seed}")
    try:
        src = extract_seed(source, seed, work)
        return R.prepare_log_streaming(src, src.name, _score_weights(source))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def build_cache(cache_dir, table_cache: str, work_dir: str, sources=None, min_free_gb: float = 5.0) -> None:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    by_source: dict = {}
    for grid, (source, _) in GRIDS.items():
        by_source.setdefault(source, []).append(load_h2(grid))
    for source, frames in by_source.items():
        if sources and source not in sources:
            continue
        rows = pd.concat(frames, ignore_index=True)
        for seed in sorted(rows.seed.unique()):
            path = cache_path(cache_dir, source, int(seed))
            if path.exists():
                continue
            table = _load_table(source, int(seed), table_cache, work_dir, min_free_gb)
            fc = g.FixedFitnessCache(table, 8.0)
            sub = rows[rows.seed == seed].drop_duplicates(FIT_KEY_COLS)
            F = {fit_key(p): fc.get(p) for p in (c.params_from_row(r) for _, r in sub.iterrows())}
            key = "topic_category"
            cats = table.context_df[key].fillna("uncategorized").astype(str).to_numpy()
            rec = {"source": source, "seed": int(seed), "n_agents": table.n_agents, "n_timesteps": table.n_timesteps,
                   "F": F, "s": table.base_score.astype(np.float64), "category": cats}
            with gzip.open(path, "wb") as fh:
                pickle.dump(rec, fh, protocol=4)
            print(f"cached {source} seed {seed}: {len(F)} fitness keys, T={table.n_timesteps}, N={table.n_agents}, "
                  f"{np.unique(cats).size} categories", flush=True)










def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", default=DEFAULT_SCORE_CACHE)
    p.add_argument("--table-cache", default=j.DEFAULT_TABLE_CACHE)
    p.add_argument("--work-dir", default="")
    p.add_argument("--sources", default="")
    p.add_argument("--min-free-gb", type=float, default=5.0)
    a = p.parse_args()
    build_cache(a.cache_dir, a.table_cache, a.work_dir, [s for s in a.sources.split(",") if s], a.min_free_gb)


if __name__ == "__main__":
    main()
