#!/usr/bin/env python3
"""Loading the main replay grids and replaying an endpoint under the empirical mean map.

Map convention:

    T(p) = E_c[ softmax update of p under context c ]      ("E_c[softmax]")

i.e. ``mean_one_step_map`` of replay/engine.py, with the SAME context
stride as the stored ``kappa_map`` of each grid (no-support grid: 2, support
grid: 4; read from ``stability_context_stride``).

"""
from __future__ import annotations

import glob
import re
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.loader import (  # noqa: E402  (re-exported: the replay scripts use c.load_table etc.)
    PHANTOM_COLOR, STABLE_COLOR, load_table, replay, set_paper_style,
)

GRIDS = {
    "no_support": "data/results/fig0_4_full_grid",
    "support": "data/results/fig5_full_grid",
}
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 110]
PARAM_KEY = ["eta", "alpha", "beta_sup"]
COEXISTENCE_LABELS = ["diffuse_coexistence", "concentrated_coexistence"]


def load_grid(grid: str, grid_dir: str) -> pd.DataFrame:
    """Stored phase_summary rows of one grid (all seeds) + descriptor-family flags.

    Family flags follow ``unstable_diversity_rates.py`` (= paper Fig. 4a):
    coexistence = phase_label in {diffuse, concentrated}_coexistence;
    specialization = specialization_signal; any = coexistence | specialization | polarization.
    """
    frames = []
    for path in sorted(glob.glob(str(ROOT / grid_dir / "run_popseed_*" / "phase_summary.csv"))):
        seed = int(re.search(r"popseed_(\d+)", path).group(1))
        df = pd.read_csv(path, float_precision="round_trip")  # exact parse also under pandas < 3
        df.insert(0, "seed", seed)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "grid", grid)
    out["coexistence"] = out["phase_label"].astype(str).isin(COEXISTENCE_LABELS).astype(int)
    out["specialization"] = (
        pd.to_numeric(out["specialization_signal"], errors="coerce").fillna(0).astype(int)
    )
    out["polarization"] = (
        pd.to_numeric(out["polarization_signal"], errors="coerce").fillna(0).astype(int)
    )
    out["any_diversity"] = (
        (out["coexistence"] == 1) | (out["specialization"] == 1) | (out["polarization"] == 1)
    ).astype(int)
    out["locally_contracting"] = out["locally_contracting"].astype(str).str.lower().isin(
        ["true", "1"]
    ).astype(int)
    out["phantom_any"] = ((out["any_diversity"] == 1) & (out["locally_contracting"] == 0)).astype(int)
    return out


def check_against_fig4_labels(grid_df: pd.DataFrame, grid_dir: str, labels_csv: str) -> Dict[str, int]:
    """Assert that the flags above equal the labels behind paper Fig. 4a."""
    lab = pd.read_csv(ROOT / labels_csv, float_precision="round_trip")
    lab = lab.loc[lab["input_root"].astype(str).eq(grid_dir)].copy()
    lab["seed"] = lab["source_result_dir"].astype(str).str.extract(r"popseed_(\d+)", expand=False).astype(int)
    lab = lab.drop_duplicates(["seed"] + PARAM_KEY)
    # family masks exactly as in unstable_diversity_rates.py (from the robust_* columns)
    lab["coex_fig4"] = lab["robust_phase_label"].astype(str).isin(COEXISTENCE_LABELS).astype(int)
    lab["spec_fig4"] = pd.to_numeric(lab["robust_specialization_signal"], errors="coerce").fillna(0).astype(int)
    lab["pol_fig4"] = pd.to_numeric(lab["robust_polarization_signal"], errors="coerce").fillna(0).astype(int)
    lab["any_fig4"] = ((lab["coex_fig4"] == 1) | (lab["spec_fig4"] == 1) | (lab["pol_fig4"] == 1)).astype(int)
    lab = lab.rename(columns={"phantom_any": "phantom_any_fig4", "kappa_map": "kappa_map_fig4"})
    merged = grid_df.merge(
        lab[["seed"] + PARAM_KEY + ["coex_fig4", "spec_fig4", "any_fig4", "phantom_any_fig4",
                                    "robust_locally_contracting", "kappa_map_fig4"]],
        on=["seed"] + PARAM_KEY, how="left",
    )
    assert merged["any_fig4"].notna().all(), "grid rows missing from Fig. 4 labels file"
    # labels-file phantom_any = (any diversity descriptor-positive) & (not locally contracting)
    n_bad = int(
        (merged["any_diversity"] != merged["any_fig4"]).sum()
        + (merged["coexistence"] != merged["coex_fig4"]).sum()
        + (merged["specialization"] != merged["spec_fig4"]).sum()
        + (merged["locally_contracting"] != merged["robust_locally_contracting"]).sum()
        + (merged["phantom_any"] != merged["phantom_any_fig4"]).sum()
        + (~np.isclose(merged["kappa_map"], merged["kappa_map_fig4"], rtol=0, atol=1.0e-9, equal_nan=True)).sum()
    )
    return {"n_rows": int(len(merged)), "n_flag_mismatch": n_bad}


def params_from_row(row: pd.Series) -> "replay.ReplayParams":
    return replay._param_from_row(row)


def fixed_fitness(table, params, support_clip: float) -> np.ndarray:
    """State-independent part of the fitness for every timestep (T x N).

    Same construction as ``build_mean_map``: at the uniform target the support
    term is exactly zero.  Requires polarization_strength == 0 (true for both grids).
    """
    assert float(getattr(params, "polarization_strength", 0.0)) == 0.0
    assert float(params.hard_safety_threshold) < 0.0
    target = np.ones(table.n_agents, dtype=np.float64) / float(table.n_agents)
    return np.vstack(
        [replay.fitness_at(table, t, target, params, support_clip=float(support_clip))
         for t in range(table.n_timesteps)]
    )


def replay_states(table, params, support_clip: float, fixed_f=None, p0=None) -> np.ndarray:
    """Per-step exposure vectors p_0..p_T of the paper's replay, shape (T+1, N).

    Re-implements the loop of ``replay_one_condition`` (uniform start, one
    softmax update per logged timestep) with the precomputed fixed fitness.
    """
    n = table.n_agents
    if fixed_f is None:
        fixed_f = fixed_fitness(table, params, support_clip)
    target = np.ones(n, dtype=np.float64) / float(n)
    eta = 0.0 if bool(params.no_feedback) else float(params.eta)
    states = np.empty((table.n_timesteps + 1, n), dtype=np.float64)
    p = target.copy() if p0 is None else np.asarray(p0, dtype=np.float64).copy()
    states[0] = p
    for t in range(table.n_timesteps):
        f = fixed_f[t] + float(params.beta_sup) * replay.support_term(p, target, float(support_clip))
        p = replay.softmax_exposure_update(p, f, eta, None)
        states[t + 1] = p
    return states


def tail_average(states: np.ndarray, last_k: int) -> np.ndarray:
    """Paper endpoint: mean of the last ``last_k`` states, clipped and renormalised."""
    endpoint = states[-int(last_k):].mean(axis=0)
    endpoint = np.clip(endpoint, replay.EPS, None)
    return endpoint / endpoint.sum()


def n_eff_top1(p: np.ndarray) -> Tuple[float, float]:
    p = np.asarray(p, dtype=np.float64)
    return float(1.0 / np.sum(p * p)), float(np.max(p))


def seed_mean_se(per_seed: pd.Series) -> Tuple[float, float]:
    x = per_seed.dropna().to_numpy(dtype=float)
    if x.size == 0:
        return float("nan"), float("nan")
    se = float(np.std(x, ddof=1) / np.sqrt(x.size)) if x.size > 1 else float("nan")
    return float(np.mean(x)), se
