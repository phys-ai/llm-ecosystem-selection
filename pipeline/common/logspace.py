#!/usr/bin/env python3
"""The log-space replay of ``replay.logspace.step_log`` vectorised over a batch
of (eta, beta_sup[, gamma]) endpoints that share the same state-independent score a (T x N).

    w = z + eta * (a_t + gamma * z + beta_sup * (log(1/N) - z)),   z <- w - logsumexp(w)

gamma = 0 is the rule of replay/logspace.py; gamma > 0 is the engagement feedback of Cor. ``cor:feedback``
(score a_i + gamma log p_i).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


DEFAULT_TABLE_CACHE = os.environ.get("EVOTHEORY_CACHE", str(Path(__file__).resolve().parents[2] / ".cache" / "table_cache"))  # parse cache of the raw logs (optional)


def replay_batch(a: np.ndarray, eta: np.ndarray, beta: np.ndarray, gamma=0.0) -> np.ndarray:
    """Log-exposure states z_0..z_T for E endpoints, shape (T+1, E, N)."""
    T, n = a.shape
    eta = np.asarray(eta, float)[:, None]; beta = np.asarray(beta, float)[:, None]
    gam = np.broadcast_to(np.asarray(gamma, float), eta[:, 0].shape)[:, None]
    log_target = -np.log(n)
    z = np.full((eta.shape[0], n), log_target)
    out = np.empty((T + 1, eta.shape[0], n))
    out[0] = z
    for t in range(T):
        w = z + eta * (a[t][None, :] + gam * z + beta * (log_target - z))
        m = w.max(axis=1, keepdims=True)
        z = w - (m + np.log(np.exp(w - m).sum(axis=1, keepdims=True)))
        out[t + 1] = z
    return out




