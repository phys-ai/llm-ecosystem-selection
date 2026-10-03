#!/usr/bin/env python3
"""Loader for the DeepSeek-V3-evaluator replay grids (data/results_reuse_clean; used by replay/per_round.py testbeds)."""
from __future__ import annotations

import glob
import re

import pandas as pd
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))  # pipeline/ on the path
from common import grids as c

ROOT = c.ROOT
GRIDS = {"no_support": "fig0_4_full_grid", "support": "fig5_full_grid", "wide_dense": "specialization_wide_dense_figS"}
SEEDS = [11, 22, 33, 44]


def load_grids(root: str) -> pd.DataFrame:
    """Concatenate the phase summaries of the three grids under ``root`` for the DeepSeek seeds."""
    frames = []
    for grid, d in GRIDS.items():
        for path in sorted(glob.glob(str(ROOT / root / d / "run_popseed_*" / "phase_summary.csv"))):
            seed = int(re.search(r"popseed_(\d+)", path).group(1))
            if seed not in SEEDS:
                continue
            df = pd.read_csv(path, float_precision="round_trip")
            assert (df["n_timesteps"] == 200).all(), path
            df.insert(0, "seed", seed)
            df.insert(0, "grid", grid)
            frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["coexistence"] = out["phase_label"].astype(str).isin(c.COEXISTENCE_LABELS).astype(int)
    out["specialization"] = out["specialization_signal"].astype(int)
    out["polarization"] = out["polarization_signal"].astype(int)
    out["any_diversity"] = ((out.coexistence == 1) | (out.specialization == 1) | (out.polarization == 1)).astype(int)
    out["locally_contracting"] = out["locally_contracting"].astype(str).str.lower().isin(["true", "1"]).astype(int)
    return out
