#!/usr/bin/env python3
"""Per-round exposure
statistics and the eta*beta_sup regimes of the exact log-ratio dynamics

    z_{t+1} = (1 - eta*beta_sup) z_t + eta (a_{i,t} - a_{j,t}).

Definitions (thresholds are read from the replay
code's PHASE_THRESHOLDS, i.e. the paper's concentration rule):

  window                     last 100 replayed states (main grids), last 50 (testbeds)
  concentrated round         top-1 >= 0.45  or  N_eff <= max(3, 0.10 N)
  simultaneous diversity     concentrated-round fraction <= 0.5
  descriptor-diverse (label) phase_label in {diffuse,concentrated}_coexistence ("coexistence")
                             or {contextual,polarized}_specialization ("specialization")
  unstable diversity         descriptor-diverse and not simultaneously diverse ("phantom" in column names)
  alternation index          share of window rounds whose top-1 agent differs from the
                             previous round but equals the one two rounds earlier

A second, "family" variant follows paper Fig. 4a (``common.grids.load_grid``):
specialization = specialization_signal (this also contains specialized_collapse).

"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))  # pipeline/ on the path

from common import grids as c

replay = c.replay
SIM_DIVERSE_MAX_CONC_FRAC = 0.5
COEX_LABELS = ["diffuse_coexistence", "concentrated_coexistence"]
SPEC_LABELS = ["contextual_specialization", "polarized_specialization"]
COLLAPSE_LABELS = ["monoculture_collapse", "specialized_collapse"]
ETA_BETA_BINS = [-0.001, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 1.0e9]
ETA_BETA_BIN_LABELS = ["0", "(0,0.25]", "(0.25,0.5]", "(0.5,0.75]", "(0.75,1]", "(1,1.25]",
                       "(1.25,1.5]", "(1.5,2]", "(2,3]", "(3,4]", ">4"]


def concentration_thresholds(n_agents: int) -> Tuple[float, float]:
    thr = replay.PHASE_THRESHOLDS
    return float(thr["concentrated_top1"]), float(
        max(thr["concentrated_neff_abs"], thr["concentrated_neff_frac"] * n_agents))


class FixedFitnessCache:
    """State-independent fitness a_t (T x N); depends on every replay parameter
    except eta / beta_sup, so it is shared across an (eta, beta_sup) grid."""

    def __init__(self, table, support_clip: float = 8.0):
        self.table, self.clip, self._cache = table, float(support_clip), {}

    def get(self, params) -> np.ndarray:
        key = (float(params.alpha), float(params.beta_route), float(params.lambda_risk),
               float(params.context_trait_strength), float(params.route_power),
               bool(params.route_zscore), str(params.ablation))
        if key not in self._cache:
            self._cache[key] = c.fixed_fitness(self.table, params, self.clip)
        return self._cache[key]


def per_round_stats(states: np.ndarray, last_k: int, n_agents: int) -> Dict[str, Any]:
    """Window statistics of the per-round exposure vectors (states: (T+1) x N)."""
    top1_thr, neff_thr = concentration_thresholds(n_agents)
    k = int(last_k)
    w = states[-k:]
    neff_t = 1.0 / np.sum(w * w, axis=1)
    top1_t = w.max(axis=1)
    conc_t = (top1_t >= top1_thr) | (neff_t <= neff_thr)
    win_all = states.argmax(axis=1)
    win = win_all[-k:]
    prev1, prev2 = win_all[-k - 1:-1], win_all[-k - 2:-2]
    alternation = float(np.mean((win != prev1) & (win == prev2)))
    n_runs = 1 + int(np.sum(win[1:] != win[:-1]))
    return {
        "N_eff_round_mean": float(neff_t.mean()), "N_eff_round_median": float(np.median(neff_t)),
        "top1_round_mean": float(top1_t.mean()),
        "conc_round_frac": float(conc_t.mean()),
        "winner_distinct": int(np.unique(win).size), "winner_n_runs": n_runs,
        "winner_mean_run": float(k / n_runs),
        "winner_change_rate": float(np.mean(win != prev1)),
        "alternation_index": alternation,
        "tv_round_mean": float(0.5 * np.abs(np.diff(w, axis=0)).sum(axis=1).mean()),
    }


def add_round6_labels(df: pd.DataFrame, label_col: str = "phase_label") -> pd.DataFrame:
    """Simultaneous / phantom labels (new definition) from conc_round_frac + descriptor labels."""
    lab = df[label_col].astype(str)
    df["label_coexistence"] = lab.isin(COEX_LABELS).astype(int)
    df["label_specialization"] = lab.isin(SPEC_LABELS).astype(int)
    df["label_collapse"] = lab.isin(COLLAPSE_LABELS).astype(int)
    df["descriptor_diverse"] = ((df.label_coexistence == 1) | (df.label_specialization == 1)).astype(int)
    df["simultaneously_diverse"] = (df["conc_round_frac"] <= SIM_DIVERSE_MAX_CONC_FRAC).astype(int)
    df["phantom_new"] = ((df.descriptor_diverse == 1) & (df.simultaneously_diverse == 0)).astype(int)
    df["not_simultaneously_diverse"] = 1 - df["simultaneously_diverse"]
    df["phantom_or_collapse"] = ((df.phantom_new == 1) | (df.label_collapse == 1)).astype(int)
    if {"coexistence", "specialization"}.issubset(df.columns):  # Fig. 4a family variant
        fam = (df["coexistence"] == 1) | (df["specialization"] == 1)
        df["descriptor_diverse_family"] = fam.astype(int)
        df["phantom_new_family"] = (fam & (df.simultaneously_diverse == 0)).astype(int)
    return df


def eta_beta_bin(eta_beta: pd.Series) -> pd.Series:
    return pd.cut(eta_beta, bins=ETA_BETA_BINS, labels=ETA_BETA_BIN_LABELS)


def seed_mean_se_table(df: pd.DataFrame, by, cols) -> pd.DataFrame:
    """Mean over seeds of per-seed means, with seed-clustered s.e. (CONVENTIONS rule 8)."""
    by = [by] if isinstance(by, str) else list(by)
    per_seed = df.groupby(by + ["seed"], observed=True)[cols].mean().reset_index()
    g = per_seed.groupby(by, observed=True)[cols]
    out = g.mean().add_suffix("_mean").join(g.sem().add_suffix("_se")).join(
        per_seed.groupby(by, observed=True)["seed"].nunique().rename("n_seeds"))
    n = df.groupby(by, observed=True).size().rename("n_endpoints")
    return out.join(n).reset_index()
