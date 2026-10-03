#!/usr/bin/env python3
"""Parsed-log loader (with an optional pickle cache), paper plot style and the log-gain of a perturbation; shared by
the replay and fragility scripts through common/grids.py."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS_DIR))
from replay import endpoint as replay  # noqa: E402
from common import perturbation as fixed  # noqa: E402

PHANTOM_COLOR = "#D55E00"
STABLE_COLOR = "#0072B2"


def set_paper_style() -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    # Same rcParams family as appendix_figures.py, font sizes
    # scaled for a full-width (two/three-panel) appendix figure.
    plt.rcParams.update(
        {
            "font.size": 11,
            "legend.fontsize": 9.5,
            "legend.title_fontsize": 9.5,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "savefig.bbox": "tight",
        }
    )


def load_table(source: Path, cache_dir: Optional[Path]) -> replay.PreparedLog:
    """Load a prepared replay log, optionally through a pickle cache
    (the cache lives outside the project; it only saves the ~50 s parse)."""
    source = Path(source)
    cache_path = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{source.name}.prepared.pkl"
        if cache_path.exists():
            try:
                with cache_path.open("rb") as handle:
                    return pickle.load(handle)
            except Exception:
                pass
    table = replay.prepare_log_streaming(
        source,
        run_name=source.name,
        score_weights=replay.DEFAULT_SCORE_WEIGHTS,
    )
    if cache_path is not None:
        try:
            tmp = cache_path.with_suffix(f".tmp{np.random.randint(1 << 30)}")
            with tmp.open("wb") as handle:
                pickle.dump(table, handle, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(cache_path)
        except Exception:
            pass
    return table








def log_gain(map_fn, perturbed: np.ndarray, p_star: np.ndarray) -> Tuple[float, float]:
    initial_tv = fixed.tv_distance(perturbed, p_star)
    output_tv = fixed.tv_distance(map_fn(perturbed), map_fn(p_star))
    gain = output_tv / max(initial_tv, replay.EPS)
    return float(np.log(max(gain, replay.EPS))), float(initial_tv)


