#!/usr/bin/env python3
"""The replay update rule in log space (no clip, no EPS, no fallback); used by replay/logspace_grids.py.

State z_i = log p_i.  Per logged round t:

    f_i(z) = a_i(c_t) + beta_sup * (log(1/N) - z_i)
    z     <- z + eta * f(z) - logsumexp(z + eta * f(z))

a_i(c_t) is the state-independent part of the existing replay fitness, computed by the
``replay.engine.fitness_at`` at the uniform state (``common.grids.fixed_fitness``:
routing, risk, context fidelity as before; the support term is exactly 0 there).

z stays finite; p = exp(z) is returned only for descriptors (exp may underflow to 0 for
z < -745 -- that is a property of the descriptor, not of the state).

"""
from __future__ import annotations

from typing import Optional

import numpy as np
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1]))  # pipeline/ on the path

from common import grids as c


def logsumexp(v: np.ndarray) -> float:
    m = float(np.max(v))
    return m + float(np.log(np.sum(np.exp(v - m))))


def step_log(z: np.ndarray, a_t: np.ndarray, eta: float, beta_sup: float, log_target: float) -> np.ndarray:
    w = z + eta * (a_t + beta_sup * (log_target - z))
    return w - logsumexp(w)


def replay_log_states(table, params, fixed_f: Optional[np.ndarray] = None,
                      z0: Optional[np.ndarray] = None, support_clip: float = 8.0) -> np.ndarray:
    """Log-exposure vectors z_0..z_T, shape (T+1, N).  ``support_clip`` is only passed to
    ``fixed_fitness`` (where the support term vanishes); the update itself has no clip."""
    n = table.n_agents
    if fixed_f is None:
        fixed_f = c.fixed_fitness(table, params, support_clip)
    eta = 0.0 if bool(params.no_feedback) else float(params.eta)
    beta = float(params.beta_sup)
    log_target = float(-np.log(n))
    z = np.full(n, log_target) if z0 is None else np.asarray(z0, dtype=np.float64).copy()
    out = np.empty((table.n_timesteps + 1, n), dtype=np.float64)
    out[0] = z
    for t in range(table.n_timesteps):
        z = step_log(z, fixed_f[t], eta, beta, log_target)
        out[t + 1] = z
    return out


def replay_states(table, params, support_clip: float = 8.0, fixed_f=None, p0=None) -> np.ndarray:
    """Drop-in for ``common_grids.replay_states``: exposure vectors p = exp(z)."""
    z0 = None if p0 is None else np.log(np.asarray(p0, dtype=np.float64))
    return np.exp(replay_log_states(table, params, fixed_f=fixed_f, z0=z0, support_clip=support_clip))
