#!/usr/bin/env python3
"""Replay of one stored condition to its time-averaged endpoint (the fragility test, fragility/*.py).

The replay itself is replay/engine.py; this module re-exports the pieces the fragility scripts use and adds the
reconstruction of a condition from a phase_summary.csv row.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Mapping, Tuple

import numpy as np

from replay.engine import (  # noqa: F401  (re-exported for fragility/*.py and common/loader.py)
    DEFAULT_SCORE_WEIGHTS, EPS, PreparedLog, ReplayParams, concentration_descriptors, fitness_at,
    prepare_log_streaming, softmax_exposure_update, support_term,
)


def parse_bool_value(x: Any) -> bool:
    """Parse bool-like values loaded from CSV/JSON/argparse."""
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if x is None:
        return False
    s = str(x).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n", "", "nan"}:
        return False
    return bool(x)


def _param_from_row(row: Mapping[str, Any]) -> ReplayParams:
    """Reconstruct ReplayParams from a phase-summary row."""
    return ReplayParams(
        eta=float(row.get("eta", 0.0)),
        alpha=float(row.get("alpha", 0.0)),
        beta_sup=float(row.get("beta_sup", 0.0)),
        beta_route=float(row.get("beta_route", 0.0)),
        lambda_risk=float(row.get("lambda_risk", 0.0)),
        hard_safety_threshold=float(row.get("hard_safety_threshold", -1.0)),
        context_trait_strength=float(row.get("context_trait_strength", 0.0)),
        route_power=float(row.get("route_power", 1.0)),
        route_zscore=parse_bool_value(row.get("route_zscore", False)),
        no_feedback=parse_bool_value(row.get("no_feedback", False)),
        ablation=str(row.get("ablation", "none")),
        polarization_strength=float(row.get("polarization_strength", 0.0)),
    )


def replay_endpoint_and_final_for_case(
    table: PreparedLog,
    params: ReplayParams,
    last_k: int,
    context_key: str,
    support_clip: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Replay one condition; return the mean of the last `last_k` exposure vectors and the final exposure vector."""
    n = table.n_agents
    p = np.ones(n, dtype=np.float64) / float(n)
    endpoint_window: deque[np.ndarray] = deque(maxlen=max(1, int(last_k)))
    endpoint_window.append(p.copy())
    eta = 0.0 if params.no_feedback else float(params.eta)
    for t_idx in range(len(table.timesteps)):
        f = fitness_at(table, t_idx, p, params, support_clip=support_clip)
        mask = None
        if params.hard_safety_threshold >= 0:
            mask = table.safety_score[t_idx] >= float(params.hard_safety_threshold)
        p = softmax_exposure_update(p, f, eta, mask)
        endpoint_window.append(p.copy())
    endpoint = np.mean(np.vstack(list(endpoint_window)), axis=0)
    endpoint = np.clip(endpoint, EPS, None)
    endpoint = endpoint / endpoint.sum()
    final_state = np.clip(p.copy(), EPS, None)
    return endpoint, final_state / final_state.sum()
