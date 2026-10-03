#!/usr/bin/env python3
"""Simplex geometry of the fragility test (Appendix B.6): total-variation distance, projection onto the simplex and a
perturbation of a given TV size along a direction (used by fragility/amplification.py and common/loader.py)."""
from __future__ import annotations

from typing import Tuple

import numpy as np



def tv_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Total-variation distance between two exposure vectors."""
    return 0.5 * float(np.sum(np.abs(np.asarray(a) - np.asarray(b))))


def project_to_simplex(x: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex."""
    values = np.asarray(x, dtype=np.float64)
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    indices = np.arange(1, values.size + 1, dtype=np.float64)
    positive = ordered - cumulative / indices > 0.0
    if not np.any(positive):
        return np.ones_like(values) / float(values.size)
    rho = int(np.flatnonzero(positive)[-1])
    theta = float(cumulative[rho] / float(rho + 1))
    projected = np.maximum(values - theta, 0.0)
    return projected / projected.sum()


def projected_direction_at_tv(
    p: np.ndarray,
    direction: np.ndarray,
    eps: float,
) -> Tuple[np.ndarray, str]:
    """Project p + scale*direction and solve for an exact TV displacement."""
    state = np.asarray(p, dtype=np.float64)
    state = state / state.sum()
    vector = np.asarray(direction, dtype=np.float64)
    target = float(eps)
    low = 0.0
    high = max(2.0 * target, 1.0e-12)
    candidate = project_to_simplex(state + high * vector)
    while tv_distance(candidate, state) < target * (1.0 - 1.0e-12) and high < 1.0e6:
        high *= 2.0
        candidate = project_to_simplex(state + high * vector)
    if tv_distance(candidate, state) < target * (1.0 - 1.0e-10):
        return candidate, "leading_tangent_max_feasible"
    for _ in range(80):
        mid = 0.5 * (low + high)
        candidate = project_to_simplex(state + mid * vector)
        if tv_distance(candidate, state) < target:
            low = mid
        else:
            high = mid
    candidate = project_to_simplex(state + 0.5 * (low + high) * vector)
    return candidate, "leading_tangent"


