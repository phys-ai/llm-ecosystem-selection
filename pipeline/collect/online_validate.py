#!/usr/bin/env python3
"""
Small-scale full closed-loop validation for the AI-native social-platform experiment.

This script is designed to complement the frozen-log replay analysis.  It closes
one additional loop: exposure affects which agents generate posts at the next
step, and newly generated posts are evaluated before the exposure distribution is
updated again.

Typical use:

  python online_validate.py \
    --collector_path ai_social_log_collect_fast.py \
    --out_dir results/closed_loop_validation \
    --seeds 42,43,44 \
    --n_timesteps 80 \
    --n_agents 32 \
    --n_evaluators 6 \
    --exposed_agents_per_step 8

Notes:
  * The script imports data classes, prompt utilities, model wrappers, and topic
    loading utilities from ai_social_log_collect_fast.py.
  * It does not import the replay script.  The relevant finite-selection update
    and endpoint diagnostics are implemented locally so this file can be run as a
    lightweight validation harness.
  * Checkpoints are saved timestep-by-timestep, so interrupted runs can be
    resumed when --resume is enabled.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import math
import os
import random
import re
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

# Avoid excessive BLAS threading on shared machines.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

EPS = 1.0e-12

DEFAULT_SCORE_WEIGHTS: Dict[str, float] = {
    "like": 0.20,
    "reply": 0.20,
    "share": 0.20,
    "follow": 0.10,
    "eval_quality": 0.15,
    "audit_quality": 0.15,
}

CONTEXT_AXIS_COLUMNS = [
    "controversy",
    "emotional_load",
    "interpersonalness",
    "evidentiality",
    "engagement_baitness",
    "public_vs_personal",
    "safety_risk_hint",
]

PHASE_THRESHOLDS: Dict[str, float] = {
    "collapse_top1": 0.80,
    "collapse_neff_abs": 1.5,
    "collapse_neff_frac": 0.025,
    "concentrated_top1": 0.45,
    "concentrated_neff_abs": 3.0,
    "concentrated_neff_frac": 0.10,
    "specialization_mi_norm": 0.12,
    "strong_specialization_mi_norm": 0.15,
    "winner_switch": 0.30,
    "polarization_var": 0.50,
    "stance_extremity": 0.70,
}

PHASE_CODES: Dict[str, int] = {
    "diffuse_coexistence": 0,
    "concentrated_coexistence": 1,
    "monoculture_collapse": 2,
    "polarization": 3,
    "contextual_specialization": 4,
    "polarized_specialization": 5,
    "metastable_switching": 6,
    "specialized_collapse": 7,
}


@dataclass(frozen=True)
class ClosedLoopCondition:
    """Platform condition used in a small closed-loop validation run."""

    label: str
    eta: float
    alpha: float
    beta_sup: float
    beta_route: float
    lambda_risk: float = 0.0
    hard_safety_threshold: float = -1.0
    context_trait_strength: float = 0.0
    route_power: float = 1.0
    route_zscore: bool = False
    support_clip: float = 8.0

    def key(self) -> str:
        return safe_name(
            f"{self.label}__eta={self.eta:g}__alpha={self.alpha:g}__sup={self.beta_sup:g}__"
            f"route={self.beta_route:g}__risk={self.lambda_risk:g}__ctx={self.context_trait_strength:g}"
        )


@dataclass
class AgentState:
    n_agents: int
    score_sum: np.ndarray
    score_count: np.ndarray
    risk_sum: np.ndarray
    risk_count: np.ndarray
    safety_sum: np.ndarray
    safety_count: np.ndarray
    quality_sum: np.ndarray
    quality_count: np.ndarray

    @classmethod
    def initialize(cls, n_agents: int, prior_score: float = 0.5, prior_count: float = 1.0) -> "AgentState":
        n = int(n_agents)
        score_sum = np.full(n, prior_score * prior_count, dtype=np.float64)
        score_count = np.full(n, prior_count, dtype=np.float64)
        risk_sum = np.full(n, (1.0 - prior_score) * prior_count, dtype=np.float64)
        risk_count = np.full(n, prior_count, dtype=np.float64)
        safety_sum = np.full(n, prior_score * prior_count, dtype=np.float64)
        safety_count = np.full(n, prior_count, dtype=np.float64)
        quality_sum = np.full(n, prior_score * prior_count, dtype=np.float64)
        quality_count = np.full(n, prior_count, dtype=np.float64)
        return cls(n, score_sum, score_count, risk_sum, risk_count, safety_sum, safety_count, quality_sum, quality_count)

    @property
    def mean_score(self) -> np.ndarray:
        return self.score_sum / np.maximum(self.score_count, EPS)

    @property
    def mean_risk(self) -> np.ndarray:
        return self.risk_sum / np.maximum(self.risk_count, EPS)

    @property
    def mean_safety(self) -> np.ndarray:
        return self.safety_sum / np.maximum(self.safety_count, EPS)

    @property
    def mean_quality(self) -> np.ndarray:
        return self.quality_sum / np.maximum(self.quality_count, EPS)

    def update(self, agent_indices: Sequence[int], score: np.ndarray, risk: np.ndarray, safety: np.ndarray, quality: np.ndarray) -> None:
        for local_idx, agent_idx in enumerate(agent_indices):
            j = int(agent_idx)
            self.score_sum[j] += float(score[local_idx])
            self.score_count[j] += 1.0
            self.risk_sum[j] += float(risk[local_idx])
            self.risk_count[j] += 1.0
            self.safety_sum[j] += float(safety[local_idx])
            self.safety_count[j] += 1.0
            self.quality_sum[j] += float(quality[local_idx])
            self.quality_count[j] += 1.0


def copy_agent_state(state: AgentState) -> AgentState:
    return AgentState(
        n_agents=int(state.n_agents),
        score_sum=state.score_sum.copy(),
        score_count=state.score_count.copy(),
        risk_sum=state.risk_sum.copy(),
        risk_count=state.risk_count.copy(),
        safety_sum=state.safety_sum.copy(),
        safety_count=state.safety_count.copy(),
        quality_sum=state.quality_sum.copy(),
        quality_count=state.quality_count.copy(),
    )


# ---------------------------------------------------------------------------
# Import and JSON helpers
# ---------------------------------------------------------------------------


def import_collector_module(path: str) -> Any:
    collector_path = Path(path)
    if not collector_path.exists():
        raise FileNotFoundError(f"collector_path does not exist: {collector_path}")
    spec = importlib.util.spec_from_file_location("ai_social_log_collect_fast_imported", str(collector_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import collector module from {collector_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def safe_name(text: str, max_len: int = 180) -> str:
    text = re.sub(r"[^A-Za-z0-9_.=+-]+", "_", str(text))
    return text[:max_len].strip("_") or "run"


def parse_float_list(raw: str) -> List[float]:
    if raw is None or str(raw).strip() == "":
        return []
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def parse_int_list(raw: str) -> List[int]:
    if raw is None or str(raw).strip() == "":
        return []
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def parse_score_weights(raw: str) -> Dict[str, float]:
    weights = dict(DEFAULT_SCORE_WEIGHTS)
    if raw:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("--score_weights must be a JSON object")
        weights.update({str(k): float(v) for k, v in parsed.items()})
    total = sum(max(0.0, float(v)) for v in weights.values())
    if total <= 0:
        raise ValueError("score weights must contain at least one positive value")
    return {k: max(0.0, float(v)) / total for k, v in weights.items()}


def condition_to_dict(cond: ClosedLoopCondition) -> Dict[str, Any]:
    return asdict(cond)


def load_conditions(raw: str) -> List[ClosedLoopCondition]:
    """Load conditions from JSON string or JSON file.

    Expected format:
      [
        {"label":"low_eta_low_alpha", "eta":0.2, "alpha":0.3, ...},
        ...
      ]
    """
    if not raw:
        return default_conditions()
    maybe_path = Path(raw)
    if maybe_path.exists():
        obj = read_json(maybe_path)
    else:
        obj = json.loads(raw)
    if not isinstance(obj, list):
        raise ValueError("conditions must be a JSON list")
    out = []
    for idx, item in enumerate(obj):
        if not isinstance(item, dict):
            raise ValueError(f"condition {idx} is not an object")
        payload = dict(item)
        payload.setdefault("label", f"condition_{idx}")
        out.append(ClosedLoopCondition(**payload))
    return out


def default_conditions() -> List[ClosedLoopCondition]:
    """Four representative conditions for the Appendix validation."""
    return [
        ClosedLoopCondition(
            label="low_eta_low_alpha_baseline",
            eta=0.3,
            alpha=0.3,
            beta_sup=0.0,
            beta_route=0.0,
            lambda_risk=0.0,
            context_trait_strength=0.0,
        ),
        ClosedLoopCondition(
            label="high_eta_high_alpha_no_support",
            eta=3.0,
            alpha=1.0,
            beta_sup=0.0,
            beta_route=2.0,
            lambda_risk=0.0,
            context_trait_strength=1.0,
            route_power=1.5,
            route_zscore=True,
        ),
        ClosedLoopCondition(
            label="moderate_eta_high_alpha_support",
            eta=1.0,
            alpha=1.0,
            beta_sup=1.5,
            beta_route=2.0,
            lambda_risk=0.0,
            context_trait_strength=1.0,
            route_power=1.5,
            route_zscore=True,
        ),
        ClosedLoopCondition(
            label="high_eta_support_stress",
            eta=2.5,
            alpha=1.0,
            beta_sup=1.5,
            beta_route=2.0,
            lambda_risk=0.0,
            context_trait_strength=1.0,
            route_power=1.5,
            route_zscore=True,
        ),
    ]


# ---------------------------------------------------------------------------
# Numeric helpers and diagnostics
# ---------------------------------------------------------------------------


def safe_entropy(p: np.ndarray) -> float:
    q = np.asarray(p, dtype=np.float64)
    q = q[q > 0]
    return float(-(q * np.log(q)).sum()) if q.size else 0.0


def softmax_exposure_update(p: np.ndarray, f: np.ndarray, eta: float, admissible: Optional[np.ndarray] = None) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    if admissible is None:
        mask = np.ones_like(p, dtype=bool)
    else:
        mask = np.asarray(admissible, dtype=bool)
        if not mask.any():
            mask = np.ones_like(p, dtype=bool)
    if float(eta) == 0.0:
        z = p.copy()
    else:
        max_f = float(np.nanmax(f[mask])) if mask.any() else float(np.nanmax(f))
        z = p * np.exp(float(eta) * (f - max_f))
    z = np.where(mask, z, 0.0)
    total = float(z.sum())
    if not np.isfinite(total) or total <= EPS:
        z = np.where(mask, p, 0.0)
        total = float(z.sum())
        if total <= EPS:
            return np.ones_like(p) / float(p.size)
    return z / total


def support_term(p: np.ndarray, support_clip: float) -> np.ndarray:
    n = p.size
    target = np.ones(n, dtype=np.float64) / float(n)
    raw = np.log(target + EPS) - np.log(p + EPS)
    if support_clip and support_clip > 0:
        raw = np.clip(raw, -float(support_clip), float(support_clip))
    return raw


def concentration_descriptors(p: np.ndarray) -> Dict[str, float]:
    q = np.asarray(p, dtype=np.float64)
    hhi = float(np.sum(q * q))
    n_eff = float(1.0 / max(hhi, EPS))
    ent = safe_entropy(q)
    return {
        "hhi": hhi,
        "n_eff": n_eff,
        "top1_share": float(np.max(q)),
        "entropy": ent,
        "entropy_norm": float(ent / math.log(len(q))) if len(q) > 1 else 0.0,
        "active_support_1e-3": int(np.sum(q > 1.0e-3)),
        "active_support_1e-4": int(np.sum(q > 1.0e-4)),
    }


def extract_trait_coords_from_value(value: Any) -> Dict[str, float]:
    if not isinstance(value, dict):
        return {}
    if isinstance(value.get("coords"), dict):
        return {str(k): float(v) for k, v in value["coords"].items() if is_finite_number(v)}
    out: Dict[str, float] = {}
    position = str(value.get("position", "")).lower()
    specialization = str(value.get("specialization", "")).lower()
    objective = str(value.get("objective_orientation", "")).lower()
    if position:
        out["stance"] = 1.0 if "affirm" in position else -1.0 if "skept" in position else 0.0
    if specialization:
        out["sociality"] = 1.0 if "social" in specialization else -1.0 if "analytic" in specialization else 0.0
    if objective:
        out["risk_tolerance"] = 1.0 if ("engagement" in objective or "attention" in objective) else -1.0 if "safety" in objective else 0.0
    return out


def is_finite_number(value: Any) -> bool:
    try:
        x = float(value)
        return math.isfinite(x)
    except Exception:
        return False


def trait_dataframe(agents: Sequence[Any]) -> pd.DataFrame:
    traits: Dict[str, Dict[str, float]] = {}
    dims: set[str] = set()
    agent_ids = []
    for agent in agents:
        aid = str(agent.agent_id)
        agent_ids.append(aid)
        try:
            coords = extract_trait_coords_from_value(asdict(agent.trait))
        except Exception:
            coords = extract_trait_coords_from_value(getattr(agent, "trait", {}))
        if coords:
            traits[aid] = coords
            dims.update(coords.keys())
    if not dims:
        dims = {"stance", "sociality", "risk_tolerance"}
    df = pd.DataFrame(0.0, index=agent_ids, columns=sorted(dims))
    for aid, coords in traits.items():
        for dim, value in coords.items():
            if dim not in df.columns:
                df[dim] = 0.0
            df.loc[aid, dim] = float(value)
    return df.reindex(agent_ids).fillna(0.0)


def weighted_trait_descriptors(traits: pd.DataFrame, p: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for dim in traits.columns:
        values = traits[dim].to_numpy(dtype=np.float64)
        mean = float(np.dot(p, values))
        var = float(np.dot(p, (values - mean) ** 2))
        abs_mean = float(np.dot(p, np.abs(values)))
        out[f"trait_mean_{dim}"] = mean
        out[f"trait_abs_mean_{dim}"] = abs(mean)
        out[f"trait_weighted_abs_{dim}"] = abs_mean
        out[f"trait_var_{dim}"] = var
    if "stance" in traits.columns:
        out["polarization"] = out.get("trait_mean_stance", 0.0)
        out["abs_polarization"] = abs(out["polarization"])
        out["polarization_dispersion"] = out.get("trait_var_stance", 0.0)
        out["stance_extremity"] = out.get("trait_weighted_abs_stance", 0.0)
    else:
        out["polarization"] = 0.0
        out["abs_polarization"] = 0.0
        out["polarization_dispersion"] = 0.0
        out["stance_extremity"] = 0.0
    return out


def descriptor_flags(desc: Mapping[str, Any], n_agents: int) -> Dict[str, int]:
    top1 = float(desc.get("top1_share", 0.0))
    n_eff = float(desc.get("n_eff", n_agents))
    mi_norm = float(desc.get("i_exp_norm", 0.0))
    winner_switch = float(desc.get("winner_switch_rate", 0.0))
    pdisp = float(desc.get("polarization_dispersion", 0.0))
    extremity = float(desc.get("stance_extremity", 0.0))

    collapse = top1 >= PHASE_THRESHOLDS["collapse_top1"] or n_eff <= max(
        PHASE_THRESHOLDS["collapse_neff_abs"], PHASE_THRESHOLDS["collapse_neff_frac"] * n_agents
    )
    concentrated = top1 >= PHASE_THRESHOLDS["concentrated_top1"] or n_eff <= max(
        PHASE_THRESHOLDS["concentrated_neff_abs"], PHASE_THRESHOLDS["concentrated_neff_frac"] * n_agents
    )
    specialized = mi_norm >= PHASE_THRESHOLDS["specialization_mi_norm"] and winner_switch >= PHASE_THRESHOLDS["winner_switch"]
    strong_specialized = mi_norm >= PHASE_THRESHOLDS["strong_specialization_mi_norm"] and winner_switch >= PHASE_THRESHOLDS["winner_switch"]
    polarized = (
        pdisp >= PHASE_THRESHOLDS["polarization_var"]
        and extremity >= PHASE_THRESHOLDS["stance_extremity"]
        and top1 < PHASE_THRESHOLDS["collapse_top1"]
    )
    return {
        "collapse_signal": int(collapse),
        "concentration_signal": int(concentrated),
        "specialization_signal": int(specialized),
        "strong_specialization_signal": int(strong_specialized),
        "polarization_signal": int(polarized),
    }


def phase_label(desc: Mapping[str, Any], n_agents: int) -> str:
    flags = descriptor_flags(desc, n_agents)
    collapse = bool(flags["collapse_signal"])
    specialized = bool(flags["specialization_signal"])
    polarized = bool(flags["polarization_signal"])
    concentrated = bool(flags["concentration_signal"])
    if collapse and specialized:
        return "specialized_collapse"
    if collapse:
        return "monoculture_collapse"
    if polarized and specialized:
        return "polarized_specialization"
    if specialized:
        return "contextual_specialization"
    if polarized:
        return "polarization"
    if concentrated:
        return "concentrated_coexistence"
    return "diffuse_coexistence"


def context_axes_from_topic(topic: Any) -> Dict[str, float]:
    out = {}
    for col in CONTEXT_AXIS_COLUMNS:
        try:
            out[col] = float(getattr(topic, col, 0.0))
        except Exception:
            out[col] = 0.0
    return out


def context_trait_bonus(traits: pd.DataFrame, topic: Any) -> np.ndarray:
    n = len(traits)
    stance = traits["stance"].to_numpy(dtype=np.float64) if "stance" in traits.columns else np.zeros(n)
    sociality = traits["sociality"].to_numpy(dtype=np.float64) if "sociality" in traits.columns else np.zeros(n)
    risk = traits["risk_tolerance"].to_numpy(dtype=np.float64) if "risk_tolerance" in traits.columns else np.zeros(n)
    axes = context_axes_from_topic(topic)
    bonus = (
        -stance * axes["controversy"]
        + sociality * axes["interpersonalness"]
        - sociality * axes["evidentiality"]
        + risk * axes["engagement_baitness"]
        - risk * axes["safety_risk_hint"]
        + 0.5 * sociality * axes["public_vs_personal"]
    )
    return bonus - float(np.mean(bonus))


def transform_route_residual(residual: np.ndarray, condition: ClosedLoopCondition) -> np.ndarray:
    route = np.asarray(residual, dtype=np.float64)
    if condition.route_zscore:
        sd = float(np.std(route))
        route = (route - float(np.mean(route))) / sd if sd > EPS else route - float(np.mean(route))
    power = max(0.25, float(condition.route_power))
    if abs(power - 1.0) > 1.0e-12:
        route = np.sign(route) * (np.abs(route) ** power)
    return route


def compute_fitness(
    p: np.ndarray,
    state: AgentState,
    current_score: np.ndarray,
    current_risk: np.ndarray,
    current_safety: np.ndarray,
    traits: pd.DataFrame,
    topic: Any,
    condition: ClosedLoopCondition,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    alpha = float(np.clip(condition.alpha, 0.0, 1.0))
    mean_score = state.mean_score
    contextual = np.asarray(current_score, dtype=np.float64)
    contextual_aug = contextual + alpha * float(condition.context_trait_strength) * context_trait_bonus(traits, topic)
    route = alpha * transform_route_residual(contextual_aug - mean_score, condition)
    support = support_term(p, support_clip=condition.support_clip)
    risk = np.asarray(current_risk, dtype=np.float64)
    base_alpha = (1.0 - alpha) * mean_score + alpha * contextual_aug
    fitness = base_alpha + float(condition.beta_route) * route + float(condition.beta_sup) * support - float(condition.lambda_risk) * risk
    debug = {
        "fitness_min": float(np.min(fitness)),
        "fitness_max": float(np.max(fitness)),
        "fitness_mean": float(np.mean(fitness)),
        "route_std": float(np.std(route)),
        "support_min": float(np.min(support)),
        "support_max": float(np.max(support)),
    }
    return fitness, debug


def sample_exposed_agents(rng: np.random.Generator, p: np.ndarray, k: int, mode: str = "weighted_without_replacement") -> np.ndarray:
    """Sample distinct agents for the current feed/exposure batch.

    High-selection closed-loop runs can numerically collapse to a simplex vertex
    with top1 ~= 1 and all other entries exactly 0. NumPy weighted sampling
    without replacement then fails if k exceeds the number of positive entries.
    We keep the exposure dynamic unchanged, but add a tiny sampling-only floor so
    the remaining feed slots can still be populated by residual discovery.
    """
    p = np.asarray(p, dtype=np.float64)
    n = int(p.size)
    k = min(max(1, int(k)), n)
    if k >= n:
        return np.arange(n, dtype=int)
    if mode == "topk":
        return np.argsort(-p)[:k].astype(int)
    if mode == "uniform":
        return rng.choice(np.arange(n), size=k, replace=False).astype(int)

    weights = np.where(np.isfinite(p) & (p > 0.0), p, 0.0).astype(np.float64)
    if float(weights.sum()) <= 0.0:
        return rng.choice(np.arange(n), size=k, replace=False).astype(int)

    floor = max(1.0e-12, 1.0e-9 / float(n))
    weights = weights + floor
    weights = weights / float(weights.sum())
    return rng.choice(np.arange(n), size=k, replace=False, p=weights).astype(int)

def aggregate_scores(
    contents: Sequence[Mapping[str, Any]],
    evaluations: Sequence[Mapping[str, Any]],
    audits: Sequence[Mapping[str, Any]],
    agent_ids: Sequence[str],
    score_weights: Mapping[str, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Dict[str, float]]]:
    """Return arrays aligned to the supplied selected agent_ids."""
    selected = [str(x) for x in agent_ids]
    local = {aid: idx for idx, aid in enumerate(selected)}
    n = len(selected)
    metric_names = ["like", "reply", "share", "follow", "eval_quality", "eval_safety", "audit_quality", "audit_safety"]
    sums = {name: np.zeros(n, dtype=np.float64) for name in metric_names}
    counts = {name: np.zeros(n, dtype=np.float64) for name in metric_names}

    for row in evaluations:
        aid = str(row.get("agent_id", ""))
        j = local.get(aid)
        if j is None:
            continue
        for raw_key, name in [("like", "like"), ("reply", "reply"), ("share", "share"), ("follow", "follow"), ("quality", "eval_quality"), ("safety", "eval_safety")]:
            value = coerce_float(row.get(raw_key))
            if value is not None:
                sums[name][j] += value
                counts[name][j] += 1.0

    for row in audits:
        aid = str(row.get("agent_id", ""))
        j = local.get(aid)
        if j is None:
            continue
        for raw_key, name in [("quality", "audit_quality"), ("safety", "audit_safety")]:
            value = coerce_float(row.get(raw_key))
            if value is not None:
                sums[name][j] += value
                counts[name][j] += 1.0

    metrics: Dict[str, np.ndarray] = {}
    metric_by_agent: Dict[str, Dict[str, float]] = {aid: {} for aid in selected}
    for name in metric_names:
        values = np.full(n, 0.5, dtype=np.float64)
        mask = counts[name] > 0
        values[mask] = sums[name][mask] / counts[name][mask]
        values = np.clip(values, 0.0, 1.0)
        metrics[name] = values
        for aid, idx in local.items():
            metric_by_agent[aid][name] = float(values[idx])

    score = np.zeros(n, dtype=np.float64)
    used = 0.0
    for name, weight in score_weights.items():
        if name not in metrics:
            continue
        score += float(weight) * metrics[name]
        used += float(weight)
    if used <= 0:
        score[:] = 0.5
    else:
        score /= used
    score = np.clip(score, 0.0, 1.0)
    safety = np.clip(0.5 * metrics["eval_safety"] + 0.5 * metrics["audit_safety"], 0.0, 1.0)
    quality = np.clip(0.5 * metrics["eval_quality"] + 0.5 * metrics["audit_quality"], 0.0, 1.0)
    risk = 1.0 - safety
    for aid, idx in local.items():
        metric_by_agent[aid]["base_score"] = float(score[idx])
        metric_by_agent[aid]["safety"] = float(safety[idx])
        metric_by_agent[aid]["quality"] = float(quality[idx])
        metric_by_agent[aid]["risk"] = float(risk[idx])
    return score, risk, safety, quality, metric_by_agent


def coerce_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# LLM generation/evaluation wrappers
# ---------------------------------------------------------------------------


def parallel_map_ordered(items: Sequence[Any], fn, max_workers: int) -> List[Any]:
    if not items:
        return []
    workers = max(1, min(int(max_workers), len(items)))
    if workers <= 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(fn, items))


def text_hash(text: str) -> str:
    return sha256(str(text).encode("utf-8")).hexdigest()[:24]


def make_context_id(seed: int, condition_label: str, timestep: int, topic_text: str) -> str:
    return sha256(f"{seed}|{condition_label}|{timestep}|{topic_text}".encode("utf-8")).hexdigest()[:24]


def generate_contents_for_selected(
    generator: Any,
    selected_indices: Sequence[int],
    timestep: int,
    topic: Any,
    topic_index: int,
    context_id: str,
    run_label: str,
    max_workers: int,
) -> List[Dict[str, Any]]:
    agents = generator.agents
    cfg = generator.cfg
    client = generator.client

    def worker(agent_idx: int) -> Dict[str, Any]:
        agent = agents[int(agent_idx)]
        model_spec = agent.post_model or cfg.post_model
        result = client.generate_post(
            model_spec,
            agent.trait,
            topic.text,
            temperature=cfg.temperature_post,
            trait_space_cfg=cfg.trait_space,
        )
        row: Dict[str, Any] = {
            "run_id": cfg.random_seed,
            "run_label": run_label,
            "timestep": int(timestep),
            "context_id": context_id,
            "topic": topic.text,
            "topic_category": getattr(topic, "topic_category", getattr(topic, "coarse_category", "uncategorized")),
            "topic_index": int(topic_index),
            "topic_source": getattr(topic, "source", ""),
            "source_subreddit": getattr(topic, "source_subreddit", ""),
            "source_post_id": getattr(topic, "source_post_id", ""),
            "normalized_from": getattr(topic, "normalized_from", ""),
            "raw_topic_text": getattr(topic, "raw_topic_text", ""),
            "topic_title": getattr(topic, "title", ""),
            "selftext_excerpt": getattr(topic, "selftext_excerpt", ""),
            "top_comments_excerpt": list(getattr(topic, "top_comments_excerpt", ()) or ()),
            "controversy": float(getattr(topic, "controversy", 0.0)),
            "emotional_load": float(getattr(topic, "emotional_load", 0.0)),
            "interpersonalness": float(getattr(topic, "interpersonalness", 0.0)),
            "evidentiality": float(getattr(topic, "evidentiality", 0.0)),
            "engagement_baitness": float(getattr(topic, "engagement_baitness", 0.0)),
            "public_vs_personal": float(getattr(topic, "public_vs_personal", 0.5)),
            "safety_risk_hint": float(getattr(topic, "safety_risk_hint", 0.0)),
            "agent_id": agent.agent_id,
            "agent_index": int(agent_idx),
            "content_text": result.get("text", ""),
            "content_text_hash": text_hash(result.get("text", "")),
            "trait": asdict(agent.trait),
            "post_provider": result.get("provider", model_spec.provider),
            "post_model": result.get("model", model_spec.model),
        }
        if getattr(cfg, "save_response_text", False):
            row["post_raw_response_text"] = result.get("raw_response_text", "")
        if getattr(cfg, "save_full_text", False):
            row["post_prompt"] = result.get("prompt", "")
        try:
            payload = generator._trait_payload(agent.trait)
            row.update(payload)
        except Exception:
            pass
        return row

    return parallel_map_ordered(list(selected_indices), worker, max_workers=max_workers)


def evaluations_are_batchable(generator: Any) -> bool:
    cfg = generator.cfg
    if not bool(getattr(cfg, "batch_evaluations_across_panel", True)):
        return False
    first: Optional[Tuple[str, str]] = None
    for evaluator in generator.evaluators:
        spec = evaluator.eval_model or cfg.eval_model
        key = (spec.provider, spec.model)
        if first is None:
            first = key
        elif key != first:
            return False
    return True


def evaluation_row_from_result(
    generator: Any,
    content: Mapping[str, Any],
    evaluator: Any,
    result: Mapping[str, Any],
    timestep: int,
) -> Dict[str, Any]:
    cfg = generator.cfg
    spec = evaluator.eval_model or cfg.eval_model
    meta = result.get("_meta", {}) if isinstance(result.get("_meta", {}), dict) else {}
    row: Dict[str, Any] = {
        "run_id": cfg.random_seed,
        "run_label": content.get("run_label", ""),
        "timestep": int(timestep),
        "context_id": content.get("context_id", ""),
        "agent_id": content.get("agent_id", ""),
        "agent_index": content.get("agent_index", -1),
        "content_text_hash": content.get("content_text_hash", ""),
        "evaluator_id": evaluator.evaluator_id,
        "evaluator_trait": asdict(evaluator.trait),
        "agent_trait": content.get("trait"),
        "topic": content.get("topic", ""),
        "topic_category": content.get("topic_category", ""),
        "topic_index": content.get("topic_index", -1),
        "eval_provider": meta.get("provider", spec.provider),
        "eval_model": meta.get("model", spec.model),
        "like": coerce_float(result.get("like")),
        "reply": coerce_float(result.get("reply")),
        "share": coerce_float(result.get("share")),
        "follow": coerce_float(result.get("follow")),
        "quality": coerce_float(result.get("quality")),
        "safety": coerce_float(result.get("safety")),
    }
    for key in CONTEXT_AXIS_COLUMNS:
        row[key] = content.get(key, 0.0)
    if getattr(cfg, "save_eval_model_metadata", False):
        row["eval_prompt"] = meta.get("prompt", "")
        row["eval_raw_response_text"] = meta.get("raw_response_text", "")
        row["eval_parse_error"] = meta.get("parse_error", "")
    return row


def generate_evaluations_and_audits(
    generator: Any,
    contents: Sequence[Mapping[str, Any]],
    timestep: int,
    eval_workers: int,
    audit_workers: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    cfg = generator.cfg
    client = generator.client
    evaluators = generator.evaluators
    batchable = evaluations_are_batchable(generator)

    def eval_content_batched(content: Mapping[str, Any]) -> List[Dict[str, Any]]:
        shared_spec = evaluators[0].eval_model or cfg.eval_model
        result = client.evaluate_post_panel(
            shared_spec,
            evaluators,
            str(content.get("topic", "")),
            str(content.get("content_text", "")),
            temperature=cfg.temperature_eval,
            trait_space_cfg=cfg.trait_space,
        )
        raw_evaluations = result.get("evaluations", [])
        meta = result.get("_meta", {}) if isinstance(result.get("_meta", {}), dict) else {}
        parsed_by_id: Dict[str, Dict[str, Any]] = {}
        if isinstance(raw_evaluations, list):
            for item in raw_evaluations:
                if isinstance(item, dict) and item.get("evaluator_id"):
                    parsed_by_id[str(item["evaluator_id"])] = item
        rows: List[Dict[str, Any]] = []
        for evaluator in evaluators:
            payload = dict(parsed_by_id.get(evaluator.evaluator_id, {}))
            evaluator_spec = evaluator.eval_model or cfg.eval_model
            payload["_meta"] = {
                "prompt": meta.get("prompt", ""),
                "raw_response_text": meta.get("raw_response_text", ""),
                "provider": meta.get("provider", evaluator_spec.provider),
                "model": meta.get("model", evaluator_spec.model),
                "parse_error": meta.get("parse_error", "") or ("" if evaluator.evaluator_id in parsed_by_id else f"missing evaluator_id={evaluator.evaluator_id}"),
            }
            rows.append(evaluation_row_from_result(generator, content, evaluator, payload, timestep))
        return rows

    def eval_pair(task: Tuple[Mapping[str, Any], Any]) -> Dict[str, Any]:
        content, evaluator = task
        spec = evaluator.eval_model or cfg.eval_model
        result = client.evaluate_post(
            spec,
            evaluator.trait,
            str(content.get("topic", "")),
            str(content.get("content_text", "")),
            temperature=cfg.temperature_eval,
            trait_space_cfg=cfg.trait_space,
        )
        return evaluation_row_from_result(generator, content, evaluator, result, timestep)

    if batchable:
        grouped = parallel_map_ordered(list(contents), eval_content_batched, max_workers=eval_workers)
        evaluations = [row for group in grouped for row in group]
    else:
        tasks = [(content, evaluator) for content in contents for evaluator in evaluators]
        evaluations = parallel_map_ordered(tasks, eval_pair, max_workers=eval_workers)

    def audit_content(content: Mapping[str, Any]) -> Dict[str, Any]:
        result = client.audit_post(
            cfg.audit_model,
            str(content.get("topic", "")),
            str(content.get("content_text", "")),
            temperature=0.0,
        )
        meta = result.get("_meta", {}) if isinstance(result.get("_meta", {}), dict) else {}
        row: Dict[str, Any] = {
            "run_id": cfg.random_seed,
            "run_label": content.get("run_label", ""),
            "timestep": int(timestep),
            "context_id": content.get("context_id", ""),
            "agent_id": content.get("agent_id", ""),
            "agent_index": content.get("agent_index", -1),
            "content_text_hash": content.get("content_text_hash", ""),
            "topic": content.get("topic", ""),
            "topic_category": content.get("topic_category", ""),
            "topic_index": content.get("topic_index", -1),
            "auditor_id": "neutral_auditor",
            "audit_provider": meta.get("provider", cfg.audit_model.provider),
            "audit_model": meta.get("model", cfg.audit_model.model),
            "safety": coerce_float(result.get("safety")),
            "quality": coerce_float(result.get("quality")),
            "reason": result.get("reason", ""),
        }
        for key in CONTEXT_AXIS_COLUMNS:
            row[key] = content.get(key, 0.0)
        if getattr(cfg, "save_audit_model_metadata", False):
            row["audit_prompt"] = meta.get("prompt", "")
            row["audit_raw_response_text"] = meta.get("raw_response_text", "")
            row["audit_parse_error"] = meta.get("parse_error", "")
        return row

    audits = parallel_map_ordered(list(contents), audit_content, max_workers=audit_workers)
    return evaluations, audits


# ---------------------------------------------------------------------------
# Closed-loop run and summaries
# ---------------------------------------------------------------------------


def trajectory_row(
    timestep: int,
    p: np.ndarray,
    traits: pd.DataFrame,
    state: AgentState,
    condition: ClosedLoopCondition,
    generated_count: int,
    unique_generated_agents_so_far: int,
    unique_hashes_so_far: int,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "timestep": int(timestep),
        "generated_count": int(generated_count),
        "unique_generated_agents_so_far": int(unique_generated_agents_so_far),
        "unique_content_hashes_so_far": int(unique_hashes_so_far),
        "weighted_risk": float(np.dot(p, state.mean_risk)),
        "weighted_safety": float(np.dot(p, state.mean_safety)),
        "weighted_quality": float(np.dot(p, state.mean_quality)),
    }
    row.update(concentration_descriptors(p))
    row.update(weighted_trait_descriptors(traits, p))
    return row


def context_mi_from_realized_fitness(
    endpoint: np.ndarray,
    fitness_history: Sequence[np.ndarray],
    context_labels: Sequence[str],
    condition: ClosedLoopCondition,
) -> Dict[str, Any]:
    if not fitness_history:
        return {"i_exp": 0.0, "i_exp_norm": 0.0, "winner_switch_rate": 0.0, "winner_unique_count": 0, "winner_top_share": 0.0}
    p = np.clip(np.asarray(endpoint, dtype=np.float64), EPS, None)
    p = p / p.sum()
    q_sums: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}
    px = np.zeros_like(p)
    winner_counts = np.zeros(p.size, dtype=np.int64)
    for f, label in zip(fitness_history, context_labels):
        q = softmax_exposure_update(p, np.asarray(f, dtype=np.float64), condition.eta)
        c = str(label or "uncategorized")
        if c not in q_sums:
            q_sums[c] = np.zeros_like(p)
            counts[c] = 0
        q_sums[c] += q
        counts[c] += 1
        px += q
        winner_counts[int(np.argmax(q))] += 1
    total = float(sum(counts.values()))
    px /= max(total, 1.0)
    pc_values = []
    mi = 0.0
    for c, count in counts.items():
        pc = float(count) / total
        pc_values.append(pc)
        q_c = q_sums[c] / float(count)
        mi += float(pc * np.sum(q_c * np.log((q_c + EPS) / (px + EPS))))
    hc = safe_entropy(np.asarray(pc_values, dtype=np.float64))
    freqs = winner_counts / max(float(winner_counts.sum()), 1.0)
    return {
        "i_exp": float(mi),
        "i_exp_norm": float(mi / max(hc, EPS)) if hc > EPS else 0.0,
        "winner_switch_rate": float(1.0 - np.sum(freqs * freqs)),
        "winner_unique_count": int(np.sum(winner_counts > 0)),
        "winner_top_share": float(np.max(freqs)) if freqs.size else 0.0,
        "context_count": int(len(counts)),
    }


def summarize_content_drift(generated_metrics: Sequence[Mapping[str, Any]], n_agents: int) -> Dict[str, Any]:
    if not generated_metrics:
        return {}
    df = pd.DataFrame(generated_metrics)
    out: Dict[str, Any] = {}
    out["n_generated_posts"] = int(len(df))
    out["generated_agent_count"] = int(df["agent_id"].nunique()) if "agent_id" in df.columns else 0
    out["generated_agent_frac"] = float(out["generated_agent_count"] / max(n_agents, 1))
    out["unique_content_hash_rate"] = float(df["content_text_hash"].nunique() / max(len(df), 1)) if "content_text_hash" in df.columns else float("nan")
    for col in ["base_score", "safety", "quality", "risk", "like", "reply", "share", "follow"]:
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        out[f"generated_{col}_mean"] = float(values.mean())
        out[f"generated_{col}_std"] = float(values.std(ddof=0))
        if len(values) >= 4:
            half = max(1, len(values) // 2)
            out[f"generated_{col}_second_minus_first_half"] = float(values.iloc[half:].mean() - values.iloc[:half].mean())
    return out


def _is_complete_closed_loop_timestep(
    bundle: Mapping[str, Any],
    n_agents: int,
    n_expected_contents: int,
    n_expected_evaluations: int,
) -> bool:
    return (
        bool(bundle.get("complete"))
        and isinstance(bundle.get("contents"), list)
        and isinstance(bundle.get("evaluations"), list)
        and isinstance(bundle.get("audits"), list)
        and len(bundle.get("contents", [])) == int(n_expected_contents)
        and len(bundle.get("evaluations", [])) == int(n_expected_evaluations)
        and len(bundle.get("audits", [])) == int(n_expected_contents)
        and len(bundle.get("p_after", [])) == int(n_agents)
        and len(bundle.get("selected_agent_indices", [])) == int(n_expected_contents)
    )


def _metric_array(metrics: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    values = []
    for idx, metric in enumerate(metrics):
        if key not in metric:
            raise KeyError(f"selected_metrics[{idx}] is missing {key!r}")
        values.append(float(metric[key]))
    return np.asarray(values, dtype=np.float64)


def _topic_for_replayed_bundle(generator: Any, bundle: Mapping[str, Any], fallback_timestep: int) -> Any:
    topic_index = int(bundle.get("topic_index", fallback_timestep)) % len(generator.topic_schedule)
    return generator.topic_schedule[topic_index]


def replay_closed_loop_checkpoints(
    run_dir: Path,
    generator: Any,
    condition: ClosedLoopCondition,
    seed: int,
    args: argparse.Namespace,
    score_weights: Mapping[str, float],
    state: AgentState,
    rng: np.random.Generator,
    p_initial: np.ndarray,
    traits: pd.DataFrame,
    agent_ids: Sequence[str],
    agent_id_to_idx: Mapping[str, int],
) -> Tuple[int, np.ndarray, deque[np.ndarray], List[Dict[str, Any]], List[Dict[str, Any]], set[str], set[str], List[np.ndarray], List[str], List[Dict[str, Any]]]:
    """Replay complete timestep checkpoints and return state for the next step.

    This avoids model calls during resume. It reconstructs the online state from
    saved per-step metrics, and advances the selection RNG by replaying prior
    sampling calls.
    """
    n_agents = len(agent_ids)
    n_contents = int(args.exposed_agents_per_step)
    n_eval_rows = n_contents * len(generator.evaluators)
    endpoint_window: deque[np.ndarray] = deque(maxlen=max(1, int(args.endpoint_last_k)))
    endpoint_window.append(np.asarray(p_initial, dtype=np.float64).copy())
    trajectory_rows: List[Dict[str, Any]] = []
    generated_metric_rows: List[Dict[str, Any]] = []
    generated_agent_ids: set[str] = set()
    generated_hashes: set[str] = set()
    fitness_history: List[np.ndarray] = []
    context_labels: List[str] = []
    all_content_rows: List[Dict[str, Any]] = []
    p = np.asarray(p_initial, dtype=np.float64).copy()

    completed = 0
    for t in range(int(args.n_timesteps)):
        step_path = run_dir / f"timestep_{t:04d}.json"
        if not step_path.exists():
            later = []
            for candidate in run_dir.glob("timestep_*.json"):
                try:
                    candidate_t = int(candidate.stem.split("_", 1)[1])
                except Exception:
                    continue
                if candidate_t > t:
                    later.append(candidate)
            if later:
                raise FileExistsError(f"Cannot resume through missing checkpoint {step_path} while later checkpoints exist.")
            break
        bundle = read_json(step_path)
        if not _is_complete_closed_loop_timestep(bundle, n_agents, n_contents, n_eval_rows):
            break

        p_before = np.asarray(bundle.get("p_before", p), dtype=np.float64)
        saved_selected = [int(x) for x in bundle.get("selected_agent_indices", [])]
        replayed_selected = sample_exposed_agents(
            rng,
            p_before,
            int(args.exposed_agents_per_step),
            mode=str(args.selection_mode),
        )
        if [int(x) for x in replayed_selected] != saved_selected:
            raise ValueError(f"Resume RNG mismatch at {step_path}; checkpoint does not match current config/seed.")

        selected_idx = np.asarray(saved_selected, dtype=np.int64)
        selected_agent_ids = [str(agent_ids[int(i)]) for i in selected_idx]
        metrics = bundle.get("selected_metrics", [])
        if not isinstance(metrics, list) or len(metrics) != len(selected_idx):
            selected_score, selected_risk, selected_safety, selected_quality, per_agent_metrics = aggregate_scores(
                contents=bundle.get("contents", []),
                evaluations=bundle.get("evaluations", []),
                audits=bundle.get("audits", []),
                agent_ids=selected_agent_ids,
                score_weights=score_weights,
            )
            metrics = [per_agent_metrics.get(aid, {}) for aid in selected_agent_ids]
        else:
            selected_score = _metric_array(metrics, "base_score")
            selected_risk = _metric_array(metrics, "risk")
            selected_safety = _metric_array(metrics, "safety")
            selected_quality = _metric_array(metrics, "quality")
            per_agent_metrics = {aid: dict(metric) for aid, metric in zip(selected_agent_ids, metrics)}

        current_score = state.mean_score.copy()
        current_risk = state.mean_risk.copy()
        current_safety = state.mean_safety.copy()
        for local_idx, agent_idx in enumerate(selected_idx):
            current_score[int(agent_idx)] = float(selected_score[local_idx])
            current_risk[int(agent_idx)] = float(selected_risk[local_idx])
            current_safety[int(agent_idx)] = float(selected_safety[local_idx])

        topic = _topic_for_replayed_bundle(generator, bundle, t)
        if isinstance(bundle.get("fitness"), list) and len(bundle.get("fitness", [])) == n_agents:
            fitness = np.asarray(bundle["fitness"], dtype=np.float64)
        else:
            fitness, _ = compute_fitness(
                p=p_before,
                state=state,
                current_score=current_score,
                current_risk=current_risk,
                current_safety=current_safety,
                traits=traits,
                topic=topic,
                condition=condition,
            )

        state.update(selected_idx, selected_score, selected_risk, selected_safety, selected_quality)
        p = np.asarray(bundle.get("p_after", p), dtype=np.float64)
        endpoint_window.append(p.copy())
        fitness_history.append(fitness.copy())
        context_labels.append(str(bundle.get("topic_category", getattr(topic, "topic_category", getattr(topic, "coarse_category", "uncategorized")))))

        for content in bundle.get("contents", []):
            aid = str(content.get("agent_id", ""))
            generated_agent_ids.add(aid)
            generated_hashes.add(str(content.get("content_text_hash", "")))
            content_metrics = dict(per_agent_metrics.get(aid, {}))
            content_metrics.update({
                "condition_label": condition.label,
                "seed": int(seed),
                "timestep": int(t),
                "agent_id": aid,
                "agent_index": int(content.get("agent_index", agent_id_to_idx.get(aid, -1))),
                "content_text_hash": content.get("content_text_hash", ""),
                "topic_category": content.get("topic_category", ""),
            })
            generated_metric_rows.append(content_metrics)
            if args.save_generated_posts_csv:
                all_content_rows.append({
                    "condition_label": condition.label,
                    "seed": int(seed),
                    "timestep": int(t),
                    "agent_id": aid,
                    "agent_index": int(content.get("agent_index", -1)),
                    "content_text_hash": content.get("content_text_hash", ""),
                    "topic": content.get("topic", ""),
                    "topic_category": content.get("topic_category", ""),
                    "content_text": content.get("content_text", "") if args.include_post_text_in_csv else "",
                })

        traj = trajectory_row(
            timestep=t,
            p=p,
            traits=traits,
            state=state,
            condition=condition,
            generated_count=(t + 1) * int(args.exposed_agents_per_step),
            unique_generated_agents_so_far=len(generated_agent_ids),
            unique_hashes_so_far=len(generated_hashes),
        )
        traj.update(condition_to_dict(condition))
        traj["seed"] = int(seed)
        trajectory_rows.append(traj)
        completed = t + 1

    if completed:
        print(f"[resume] condition={condition.label} seed={seed} replayed {completed}/{args.n_timesteps} timesteps", flush=True)
    return (
        completed,
        p,
        endpoint_window,
        trajectory_rows,
        generated_metric_rows,
        generated_agent_ids,
        generated_hashes,
        fitness_history,
        context_labels,
        all_content_rows,
    )


def perturb_distribution(
    p: np.ndarray,
    mode: str,
    rng: np.random.Generator,
    eps: float,
) -> Tuple[np.ndarray, str, Optional[int]]:
    q = np.asarray(p, dtype=np.float64).copy()
    q = np.clip(q, EPS, None)
    q = q / q.sum()
    n = len(q)
    invader_idx: Optional[int] = None
    mode = str(mode).strip().lower() or "random"

    if mode == "invasion":
        low = np.argsort(q)[:max(1, n // 5)]
        invader_idx = int(rng.choice(low))
        q = (1.0 - float(eps)) * q
        q[invader_idx] += float(eps)
        direction_label = f"invasion_agent_{invader_idx}"
    elif mode == "active":
        active = np.where(q > 1e-3)[0]
        if len(active) < 2:
            active = np.arange(n)
        i, j = rng.choice(active, size=2, replace=False)
        delta = min(float(eps), float(q[int(j)]) * 0.5)
        q[int(i)] += delta
        q[int(j)] -= delta
        direction_label = f"active_{int(i)}_minus_{int(j)}"
    else:
        noise = rng.normal(size=n)
        noise -= noise.mean()
        noise = noise / max(float(np.sum(np.abs(noise))), EPS)
        q = q + float(eps) * noise
        direction_label = "random"

    q = np.clip(q, EPS, None)
    q = q / q.sum()
    return q, direction_label, invader_idx


def run_endpoint_perturbation_validation(
    collector_module: Any,
    generator: Any,
    base_state: AgentState,
    endpoint: np.ndarray,
    traits: pd.DataFrame,
    condition: ClosedLoopCondition,
    seed: int,
    args: argparse.Namespace,
    score_weights: Mapping[str, float],
    agent_ids: Sequence[str],
    run_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    del collector_module
    modes = [m.strip() for m in str(args.perturbation_modes).split(",") if m.strip()]
    if not modes:
        modes = ["random"]
    n_perturbations = max(0, int(args.n_perturbations))
    steps = max(1, int(args.perturbation_steps))
    eps = max(0.0, float(args.perturbation_mass))
    if n_perturbations <= 0 or eps <= 0:
        return []

    p_star = np.asarray(endpoint, dtype=np.float64).copy()
    p_star = np.clip(p_star, EPS, None)
    p_star = p_star / p_star.sum()
    rows: List[Dict[str, Any]] = []
    perturb_dir = Path(run_dir) / "perturbations" if run_dir is not None else None

    for k in range(n_perturbations):
        row_path = perturb_dir / f"perturbation_{k:04d}.json" if perturb_dir is not None else None
        if args.resume and not args.overwrite_checkpoints and row_path is not None and row_path.exists():
            row = read_json(row_path)
            if isinstance(row, dict) and row.get("complete"):
                saved = dict(row)
                saved.pop("complete", None)
                rows.append(saved)
                print(f"[resume] perturbation exists: {row_path}", flush=True)
                continue

        rng = np.random.default_rng(int(seed) + stable_int_hash(f"{condition.key()}::perturbation::{k}"))
        mode = modes[k % len(modes)]
        q0, direction_label, invader_idx = perturb_distribution(p_star, mode, rng, eps)
        p_cont = q0.copy()
        state = copy_agent_state(base_state)
        d0 = float(np.sum(np.abs(q0 - p_star)))

        for s in range(steps):
            t = int(args.n_timesteps) + s
            topic_index = t % len(generator.topic_schedule)
            topic = generator.topic_schedule[topic_index]
            context_id = make_context_id(int(seed), f"{condition.label}_perturb{k}", t, str(topic.text))
            p_before = p_cont.copy()
            selected_idx = sample_exposed_agents(rng, p_before, int(args.exposed_agents_per_step), mode=str(args.selection_mode))
            selected_agent_ids = [str(agent_ids[int(i)]) for i in selected_idx]

            contents = generate_contents_for_selected(
                generator=generator,
                selected_indices=selected_idx,
                timestep=t,
                topic=topic,
                topic_index=topic_index,
                context_id=context_id,
                run_label=f"{generator.cfg.run_label}_perturb{k}",
                max_workers=int(args.post_workers),
            )
            evaluations, audits = generate_evaluations_and_audits(
                generator=generator,
                contents=contents,
                timestep=t,
                eval_workers=int(args.eval_workers),
                audit_workers=int(args.audit_workers),
            )
            selected_score, selected_risk, selected_safety, selected_quality, _ = aggregate_scores(
                contents=contents,
                evaluations=evaluations,
                audits=audits,
                agent_ids=selected_agent_ids,
                score_weights=score_weights,
            )

            current_score = state.mean_score.copy()
            current_risk = state.mean_risk.copy()
            current_safety = state.mean_safety.copy()
            for local_idx, agent_idx in enumerate(selected_idx):
                current_score[int(agent_idx)] = float(selected_score[local_idx])
                current_risk[int(agent_idx)] = float(selected_risk[local_idx])
                current_safety[int(agent_idx)] = float(selected_safety[local_idx])

            fitness, _ = compute_fitness(
                p=p_before,
                state=state,
                current_score=current_score,
                current_risk=current_risk,
                current_safety=current_safety,
                traits=traits,
                topic=topic,
                condition=condition,
            )
            admissible = None
            if condition.hard_safety_threshold >= 0:
                admissible = current_safety >= float(condition.hard_safety_threshold)
            p_cont = softmax_exposure_update(p_before, fitness, condition.eta, admissible=admissible)
            state.update(selected_idx, selected_score, selected_risk, selected_safety, selected_quality)

        d_end = float(np.sum(np.abs(p_cont - p_star)))
        if invader_idx is None:
            gain = p_cont - q0
            inactive = np.argsort(p_star)[:max(1, len(p_star) // 5)]
            invader_idx = int(inactive[int(np.argmax(gain[inactive]))])
        invader_initial = float(q0[int(invader_idx)])
        invader_final = float(p_cont[int(invader_idx)])
        growth_rate = float(np.log((d_end + EPS) / (d0 + EPS)) / float(steps))
        growth_tol = max(0.0, float(getattr(args, "perturbation_growth_tol", 0.0)))
        if growth_rate < -growth_tol:
            stability_label = "contracting"
        elif growth_rate > growth_tol:
            stability_label = "expanding"
        else:
            stability_label = "near_neutral"
        row = {
            "condition_label": condition.label,
            "condition_key": condition.key(),
            "seed": int(seed),
            "perturbation_id": int(k),
            "perturbation_mode": str(mode),
            "direction_label": direction_label,
            "perturbation_steps": int(steps),
            "perturbation_mass": float(eps),
            "d0_l1": d0,
            "d_end_l1": d_end,
            "recovery_ratio_l1": float(d_end / max(d0, EPS)),
            "online_growth_rate_l1": growth_rate,
            "perturbation_growth_tol": growth_tol,
            "online_stability_label_l1": stability_label,
            "online_stable_l1": bool(stability_label == "contracting"),
            "online_contracting_l1": bool(stability_label == "contracting"),
            "online_near_neutral_l1": bool(stability_label == "near_neutral"),
            "online_expanding_l1": bool(stability_label == "expanding"),
            "top1_end": float(np.max(p_cont)),
            "n_eff_end": float(1.0 / max(float(np.sum(p_cont * p_cont)), EPS)),
            "invader_agent": int(invader_idx),
            "invader_agent_id": str(agent_ids[int(invader_idx)]),
            "invader_initial_mass": invader_initial,
            "invader_final_mass": invader_final,
            "invader_log_growth": float(np.log((invader_final + EPS) / (invader_initial + EPS))),
        }
        rows.append(row)
        if row_path is not None:
            write_json_atomic(row_path, {"complete": True, **row})

    return rows


def run_one_condition_seed(
    collector_module: Any,
    base_cfg_dict: Dict[str, Any],
    condition: ClosedLoopCondition,
    seed: int,
    args: argparse.Namespace,
    score_weights: Mapping[str, float],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    cfg_dict = dict(base_cfg_dict)
    cfg_dict["random_seed"] = int(seed)
    cfg_dict["n_timesteps"] = int(args.n_timesteps)
    cfg_dict["run_label"] = f"closed_loop_{condition.label}_seed{seed}"

    # Allow a smaller validation population without editing the user's main config.
    trait_space = cfg_dict.get("trait_space")
    if isinstance(trait_space, dict):
        trait_space = json.loads(json.dumps(trait_space))
        trait_space.setdefault("sampling", {})["n_agents"] = int(args.n_agents)
        trait_space.setdefault("sampling", {})["seed"] = int(seed)
        trait_space.setdefault("evaluator_sampling", {})["n_evaluators"] = int(args.n_evaluators)
        trait_space.setdefault("evaluator_sampling", {})["seed"] = int(seed) + 1000
        cfg_dict["trait_space"] = trait_space
    cfg_dict = collector_module.normalize_config_dict(cfg_dict)
    cfg = collector_module.LogGenerationConfig(**cfg_dict)
    generator = collector_module.LLMLogGenerator(cfg)

    agents = generator.agents
    agent_ids = [str(agent.agent_id) for agent in agents]
    agent_id_to_idx = {aid: i for i, aid in enumerate(agent_ids)}
    traits = trait_dataframe(agents)
    n_agents = len(agents)
    p = np.ones(n_agents, dtype=np.float64) / float(n_agents)
    state = AgentState.initialize(n_agents=n_agents, prior_score=float(args.prior_score), prior_count=float(args.prior_count))
    rng = np.random.default_rng(int(seed) + stable_int_hash(condition.key()))

    out_root = Path(args.out_dir)
    run_dir = out_root / "checkpoints" / condition.key() / f"seed={seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "experiment_type": "small_scale_full_closed_loop_validation",
        "condition": condition_to_dict(condition),
        "seed": int(seed),
        "n_agents": n_agents,
        "n_evaluators": len(generator.evaluators),
        "n_timesteps": int(args.n_timesteps),
        "exposed_agents_per_step": int(args.exposed_agents_per_step),
        "selection_mode": args.selection_mode,
        "score_weights": dict(score_weights),
        "agent_ids": agent_ids,
        "agents": [{"agent_id": a.agent_id, "trait": asdict(a.trait)} for a in agents],
        "evaluators": [{"evaluator_id": e.evaluator_id, "trait": asdict(e.trait)} for e in generator.evaluators],
        "config": generator._config_to_jsonable() if hasattr(generator, "_config_to_jsonable") else {},
    }
    write_json_atomic(run_dir / "metadata.json", metadata)

    start_t = 0
    if args.resume and not args.overwrite_checkpoints:
        (
            start_t,
            p,
            endpoint_window,
            trajectory_rows,
            generated_metric_rows,
            generated_agent_ids,
            generated_hashes,
            fitness_history,
            context_labels,
            all_content_rows,
        ) = replay_closed_loop_checkpoints(
            run_dir=run_dir,
            generator=generator,
            condition=condition,
            seed=seed,
            args=args,
            score_weights=score_weights,
            state=state,
            rng=rng,
            p_initial=p,
            traits=traits,
            agent_ids=agent_ids,
            agent_id_to_idx=agent_id_to_idx,
        )
    else:
        endpoint_window = deque(maxlen=max(1, int(args.endpoint_last_k)))
        endpoint_window.append(p.copy())
        trajectory_rows = []
        generated_metric_rows = []
        generated_agent_ids = set()
        generated_hashes = set()
        fitness_history = []
        context_labels = []
        all_content_rows = []

    for t in range(int(start_t), int(args.n_timesteps)):
        step_path = run_dir / f"timestep_{t:04d}.json"
        if args.resume and step_path.exists():
            if args.overwrite_checkpoints:
                pass
            else:
                print(f"[resume] overwriting incomplete checkpoint: {step_path}", flush=True)

        topic_index = t % len(generator.topic_schedule)
        topic = generator.topic_schedule[topic_index]
        context_id = make_context_id(int(seed), condition.label, t, str(topic.text))
        p_before = p.copy()
        selected_idx = sample_exposed_agents(rng, p_before, int(args.exposed_agents_per_step), mode=str(args.selection_mode))
        selected_agent_ids = [agent_ids[int(i)] for i in selected_idx]

        contents = generate_contents_for_selected(
            generator=generator,
            selected_indices=selected_idx,
            timestep=t,
            topic=topic,
            topic_index=topic_index,
            context_id=context_id,
            run_label=cfg.run_label,
            max_workers=int(args.post_workers),
        )
        evaluations, audits = generate_evaluations_and_audits(
            generator=generator,
            contents=contents,
            timestep=t,
            eval_workers=int(args.eval_workers),
            audit_workers=int(args.audit_workers),
        )
        selected_score, selected_risk, selected_safety, selected_quality, per_agent_metrics = aggregate_scores(
            contents=contents,
            evaluations=evaluations,
            audits=audits,
            agent_ids=selected_agent_ids,
            score_weights=score_weights,
        )

        current_score = state.mean_score.copy()
        current_risk = state.mean_risk.copy()
        current_safety = state.mean_safety.copy()
        for local_idx, agent_idx in enumerate(selected_idx):
            current_score[int(agent_idx)] = float(selected_score[local_idx])
            current_risk[int(agent_idx)] = float(selected_risk[local_idx])
            current_safety[int(agent_idx)] = float(selected_safety[local_idx])

        fitness, fitness_debug = compute_fitness(
            p=p_before,
            state=state,
            current_score=current_score,
            current_risk=current_risk,
            current_safety=current_safety,
            traits=traits,
            topic=topic,
            condition=condition,
        )
        admissible = None
        if condition.hard_safety_threshold >= 0:
            admissible = current_safety >= float(condition.hard_safety_threshold)
        p_after = softmax_exposure_update(p_before, fitness, condition.eta, admissible=admissible)

        state.update(selected_idx, selected_score, selected_risk, selected_safety, selected_quality)
        p = p_after
        endpoint_window.append(p.copy())
        fitness_history.append(fitness.copy())
        context_labels.append(str(getattr(topic, "topic_category", getattr(topic, "coarse_category", "uncategorized"))))

        for content in contents:
            aid = str(content.get("agent_id", ""))
            generated_agent_ids.add(aid)
            generated_hashes.add(str(content.get("content_text_hash", "")))
            metrics = dict(per_agent_metrics.get(aid, {}))
            metrics.update({
                "condition_label": condition.label,
                "seed": int(seed),
                "timestep": int(t),
                "agent_id": aid,
                "agent_index": int(content.get("agent_index", agent_id_to_idx.get(aid, -1))),
                "content_text_hash": content.get("content_text_hash", ""),
                "topic_category": content.get("topic_category", ""),
            })
            generated_metric_rows.append(metrics)
            if args.save_generated_posts_csv:
                all_content_rows.append({
                    "condition_label": condition.label,
                    "seed": int(seed),
                    "timestep": int(t),
                    "agent_id": aid,
                    "agent_index": int(content.get("agent_index", -1)),
                    "content_text_hash": content.get("content_text_hash", ""),
                    "topic": content.get("topic", ""),
                    "topic_category": content.get("topic_category", ""),
                    "content_text": content.get("content_text", "") if args.include_post_text_in_csv else "",
                })

        traj = trajectory_row(
            timestep=t,
            p=p,
            traits=traits,
            state=state,
            condition=condition,
            generated_count=(t + 1) * int(args.exposed_agents_per_step),
            unique_generated_agents_so_far=len(generated_agent_ids),
            unique_hashes_so_far=len(generated_hashes),
        )
        traj.update(condition_to_dict(condition))
        traj["seed"] = int(seed)
        trajectory_rows.append(traj)

        bundle = {
            "complete": True,
            "condition": condition_to_dict(condition),
            "seed": int(seed),
            "timestep": int(t),
            "context_id": context_id,
            "topic": str(topic.text),
            "topic_category": str(getattr(topic, "topic_category", getattr(topic, "coarse_category", "uncategorized"))),
            "topic_index": int(topic_index),
            "p_before": p_before.tolist(),
            "p_after": p_after.tolist(),
            "selected_agent_indices": [int(x) for x in selected_idx],
            "selected_agent_ids": selected_agent_ids,
            "fitness": fitness.tolist() if args.save_full_fitness else [],
            "fitness_debug": fitness_debug,
            "contents": contents,
            "evaluations": evaluations,
            "audits": audits,
            "selected_metrics": [per_agent_metrics.get(aid, {}) for aid in selected_agent_ids],
        }
        write_json_atomic(step_path, bundle)
        if (t + 1) % max(1, int(args.progress_every)) == 0:
            print(
                f"[{condition.label} seed={seed}] timestep {t + 1}/{args.n_timesteps} "
                f"top1={float(np.max(p)):.3f} Neff={1.0 / max(float(np.sum(p * p)), EPS):.2f}",
                flush=True,
            )

    # Summary descriptors use the same final-window average as the replay analysis.
    # Perturb-and-continue validation, however, should restart from the actual
    # terminal exposure state so that the exposure state and AgentState are aligned.
    terminal_endpoint = np.asarray(p, dtype=np.float64).copy()
    terminal_endpoint = np.clip(terminal_endpoint, EPS, None)
    terminal_endpoint = terminal_endpoint / terminal_endpoint.sum()

    endpoint = np.mean(np.vstack(list(endpoint_window)), axis=0)
    endpoint = np.clip(endpoint, EPS, None)
    endpoint = endpoint / endpoint.sum()

    summary: Dict[str, Any] = {
        "condition_label": condition.label,
        "condition_key": condition.key(),
        "seed": int(seed),
        "n_agents": int(n_agents),
        "n_evaluators": int(len(generator.evaluators)),
        "n_timesteps": int(args.n_timesteps),
        "exposed_agents_per_step": int(args.exposed_agents_per_step),
        "selection_mode": str(args.selection_mode),
        "weighted_risk": float(np.dot(endpoint, state.mean_risk)),
        "weighted_safety": float(np.dot(endpoint, state.mean_safety)),
        "weighted_quality": float(np.dot(endpoint, state.mean_quality)),
    }
    summary.update(condition_to_dict(condition))
    summary.update(concentration_descriptors(endpoint))
    summary.update(weighted_trait_descriptors(traits, endpoint))
    summary.update(context_mi_from_realized_fitness(endpoint, fitness_history, context_labels, condition))
    summary.update(summarize_content_drift(generated_metric_rows, n_agents=n_agents))
    summary.update(descriptor_flags(summary, n_agents=n_agents))
    summary["phase_label"] = phase_label(summary, n_agents=n_agents)
    summary["phase_code"] = PHASE_CODES.get(summary["phase_label"], -1)
    summary["endpoint_exposure_json"] = json.dumps(endpoint.tolist())
    summary["terminal_endpoint_exposure_json"] = json.dumps(terminal_endpoint.tolist())
    summary["terminal_top1_share"] = float(np.max(terminal_endpoint))
    summary["terminal_n_eff"] = float(1.0 / max(float(np.sum(terminal_endpoint * terminal_endpoint)), EPS))
    summary["endpoint_terminal_l1"] = float(np.sum(np.abs(endpoint - terminal_endpoint)))
    summary["run_dir"] = str(run_dir)
    if args.perturbation_validation:
        perturb_rows = run_endpoint_perturbation_validation(
            collector_module=collector_module,
            generator=generator,
            base_state=state,
            endpoint=terminal_endpoint,
            traits=traits,
            condition=condition,
            seed=seed,
            args=args,
            score_weights=score_weights,
            agent_ids=agent_ids,
            run_dir=run_dir,
        )
    else:
        perturb_rows = []
    write_json_atomic(run_dir / "summary.json", summary)

    return summary, trajectory_rows, all_content_rows, perturb_rows


def stable_int_hash(text: str) -> int:
    return int(sha256(str(text).encode("utf-8")).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# Optional replay comparison
# ---------------------------------------------------------------------------


def read_reference_replay_summary(replay_csv: str) -> pd.DataFrame:
    """Read a replay phase summary CSV, or merge popseed checkpoint summaries.

    Full-grid replay runs may be left as per-population-seed checkpoint
    directories instead of a single top-level phase_summary.csv.  In that case,
    callers often still point at the intended aggregate path, e.g.
    analysis/results/fig0_4_full_grid/phase_summary.csv.  Fall back to merging
    run_popseed_*.checkpoints/phase_summary.csv under the parent directory.
    """
    path = Path(replay_csv)
    if path.exists():
        return pd.read_csv(path)

    search_dir = path if path.suffix == "" else path.parent
    checkpoint_summaries = sorted(search_dir.glob("run_popseed_*.checkpoints/phase_summary.csv"))
    if not checkpoint_summaries:
        raise FileNotFoundError(
            f"reference replay summary not found: {path}; "
            f"also found no run_popseed_*.checkpoints/phase_summary.csv under {search_dir}"
        )

    frames: List[pd.DataFrame] = []
    for summary_path in checkpoint_summaries:
        frame = pd.read_csv(summary_path)
        if "replay_checkpoint_dir" not in frame.columns:
            frame["replay_checkpoint_dir"] = str(summary_path.parent)
        frames.append(frame)
    replay = pd.concat(frames, ignore_index=True, sort=False)
    print(
        f"[info] merged {len(checkpoint_summaries)} replay checkpoint phase summaries "
        f"from {search_dir} ({len(replay)} rows)",
        flush=True,
    )
    return replay


def attach_nearest_replay_metrics(summary: pd.DataFrame, replay_csv: str, match_cols_raw: str = "") -> pd.DataFrame:
    """Attach metrics from the nearest replay-grid coordinate to each closed-loop row.

    By default, matching uses only the platform coordinates that define the replay
    phase diagrams. Closed-loop-only knobs such as context_trait_strength and
    route_power are deliberately excluded unless the caller explicitly includes
    them through --replay_match_cols.
    """
    if not replay_csv:
        return summary
    replay = read_reference_replay_summary(replay_csv)
    if replay.empty:
        return summary.copy()

    metric_candidates = [
        "top1_share", "hhi", "n_eff", "entropy_norm", "i_exp_norm", "winner_switch_rate",
        "phase_label", "phase_code", "kappa_map", "kappa", "kappa_map_eta",
        "weighted_quality", "weighted_safety", "weighted_risk",
    ]
    metric_cols = [c for c in metric_candidates if c in replay.columns]
    if str(match_cols_raw or "").strip():
        condition_candidates = [c.strip() for c in str(match_cols_raw).split(",") if c.strip()]
    else:
        condition_candidates = ["eta", "alpha", "beta_sup", "beta_route", "lambda_risk"]
    condition_cols = [c for c in condition_candidates if c in summary.columns and c in replay.columns]
    if not condition_cols:
        print("[warn] no common condition columns found for nearest replay comparison", file=sys.stderr)
        return summary.copy()

    replay_num = replay.copy()
    for c in condition_cols:
        replay_num[c] = pd.to_numeric(replay_num[c], errors="coerce")
    numeric_metric_cols = [c for c in metric_cols if c != "phase_label"]
    for c in numeric_metric_cols:
        replay_num[c] = pd.to_numeric(replay_num[c], errors="coerce")

    agg_spec: Dict[str, str] = {}
    for c in metric_cols:
        agg_spec[c] = "mean" if pd.api.types.is_numeric_dtype(replay_num[c]) else "first"
    if agg_spec:
        replay_grid = replay_num.groupby(condition_cols, dropna=False, as_index=False).agg(agg_spec)
    else:
        replay_grid = replay_num[condition_cols].drop_duplicates(ignore_index=True)

    scales: Dict[str, float] = {}
    for c in condition_cols:
        vals = pd.to_numeric(replay_grid[c], errors="coerce").to_numpy(dtype=float)
        scale = np.nanstd(vals)
        scales[c] = float(scale) if np.isfinite(scale) and scale > 0 else 1.0

    out_rows: List[Dict[str, Any]] = []
    for _, row in summary.iterrows():
        dist = np.zeros(len(replay_grid), dtype=float)
        used = 0
        for c in condition_cols:
            s_val = pd.to_numeric(pd.Series([row.get(c, np.nan)]), errors="coerce").iloc[0]
            if not np.isfinite(s_val):
                continue
            vals = pd.to_numeric(replay_grid[c], errors="coerce").to_numpy(dtype=float)
            finite = np.isfinite(vals)
            if not finite.any():
                continue
            term = np.full(len(replay_grid), np.inf, dtype=float)
            term[finite] = ((vals[finite] - float(s_val)) / scales[c]) ** 2
            dist += term
            used += 1

        if used == 0 or len(replay_grid) == 0 or not np.isfinite(dist).any():
            replay_match = pd.Series(dtype=object)
        else:
            replay_match = replay_grid.iloc[int(np.nanargmin(dist))]

        merged = row.to_dict()
        for c in condition_cols:
            merged[f"{c}_replay_nearest"] = replay_match.get(c, np.nan)
        if len(replay_match) > 0:
            merged["replay_nearest_distance"] = float(np.sqrt(np.nanmin(dist)))
        else:
            merged["replay_nearest_distance"] = np.nan

        for m in metric_cols:
            replay_val = replay_match.get(m, np.nan)
            merged[f"{m}_replay"] = replay_val
            if m in row:
                lhs = pd.to_numeric(pd.Series([row.get(m, np.nan)]), errors="coerce").iloc[0]
                rhs = pd.to_numeric(pd.Series([replay_val]), errors="coerce").iloc[0]
                if np.isfinite(lhs) and np.isfinite(rhs):
                    merged[f"closed_loop_minus_replay_{m}"] = float(lhs) - float(rhs)
                else:
                    merged[f"closed_loop_minus_replay_{m}"] = np.nan
        out_rows.append(merged)

    return pd.DataFrame(out_rows)


def cohen_kappa(y_true: Sequence[Any], y_pred: Sequence[Any]) -> float:
    labels = sorted(set(pd.Series(y_true).dropna().astype(str)) | set(pd.Series(y_pred).dropna().astype(str)))
    if not labels:
        return np.nan
    idx = {lab: i for i, lab in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=float)
    for a, b in zip(y_true, y_pred):
        if pd.isna(a) or pd.isna(b):
            continue
        mat[idx[str(a)], idx[str(b)]] += 1
    n = mat.sum()
    if n <= 0:
        return np.nan
    po = np.trace(mat) / n
    row = mat.sum(axis=1) / n
    col = mat.sum(axis=0) / n
    pe = float(np.dot(row, col))
    return (po - pe) / (1 - pe) if abs(1 - pe) > 1e-12 else np.nan


def write_replay_online_agreement_tables(summary_df: pd.DataFrame, out_dir: Path) -> None:
    """Write metric and phase-label agreement tables for nearest replay vs closed-loop rows."""
    metrics = [
        "top1_share", "hhi", "n_eff", "entropy_norm",
        "i_exp_norm", "winner_switch_rate",
        "weighted_quality", "weighted_safety", "weighted_risk",
    ]
    metric_columns = [
        "metric", "n", "pearson", "spearman", "mae", "rmse",
        "bias_closed_loop_minus_replay",
    ]
    rows = []
    for m in metrics:
        rcol = f"{m}_replay"
        if m not in summary_df.columns or rcol not in summary_df.columns:
            continue
        x = pd.to_numeric(summary_df[rcol], errors="coerce")
        y = pd.to_numeric(summary_df[m], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() < 3:
            continue
        diff = y[mask] - x[mask]
        rows.append({
            "metric": m,
            "n": int(mask.sum()),
            "pearson": float(x[mask].corr(y[mask], method="pearson")),
            "spearman": float(x[mask].corr(y[mask], method="spearman")),
            "mae": float(diff.abs().mean()),
            "rmse": float(np.sqrt((diff ** 2).mean())),
            "bias_closed_loop_minus_replay": float(diff.mean()),
        })

    pd.DataFrame(rows, columns=metric_columns).to_csv(out_dir / "replay_online_metric_agreement.csv", index=False)

    label_columns = ["comparison", "n", "exact_agreement", "cohen_kappa"]
    label_rows = []
    if "phase_label" in summary_df.columns and "phase_label_replay" in summary_df.columns:
        mask = summary_df["phase_label"].notna() & summary_df["phase_label_replay"].notna()
        label_rows.append({
            "comparison": "phase_label",
            "n": int(mask.sum()),
            "exact_agreement": float((summary_df.loc[mask, "phase_label"] == summary_df.loc[mask, "phase_label_replay"]).mean()),
            "cohen_kappa": float(cohen_kappa(summary_df.loc[mask, "phase_label"], summary_df.loc[mask, "phase_label_replay"])),
        })
        pd.crosstab(
            summary_df.loc[mask, "phase_label_replay"],
            summary_df.loc[mask, "phase_label"],
            rownames=["replay"],
            colnames=["closed_loop"],
            normalize="index",
        ).to_csv(out_dir / "replay_online_phase_confusion_normalized.csv")

    pd.DataFrame(label_rows, columns=label_columns).to_csv(out_dir / "replay_online_label_agreement.csv", index=False)


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------


def build_base_config(collector_module: Any, args: argparse.Namespace) -> Dict[str, Any]:
    cfg_dict = asdict(collector_module.LogGenerationConfig())
    if args.config_json:
        maybe_path = Path(args.config_json)
        overrides = read_json(maybe_path) if maybe_path.exists() else json.loads(args.config_json)
        if not isinstance(overrides, dict):
            raise ValueError("--config_json must be a JSON object or a path to a JSON object")
        cfg_dict.update(overrides)
    cfg_dict["n_timesteps"] = int(args.n_timesteps)
    cfg_dict["temperature_post"] = float(args.temperature_post) if args.temperature_post >= 0 else cfg_dict.get("temperature_post", 0.7)
    cfg_dict["temperature_eval"] = float(args.temperature_eval) if args.temperature_eval >= 0 else cfg_dict.get("temperature_eval", 0.1)
    cfg_dict["api_max_workers"] = max(int(args.post_workers), int(args.eval_workers), int(args.audit_workers), 1)
    cfg_dict["post_max_workers"] = int(args.post_workers)
    cfg_dict["eval_max_workers"] = int(args.eval_workers)
    cfg_dict["audit_max_workers"] = int(args.audit_workers)
    cfg_dict["save_full_text"] = bool(args.save_full_text)
    cfg_dict["save_response_text"] = bool(args.save_response_text)
    cfg_dict["save_eval_model_metadata"] = bool(args.save_eval_model_metadata)
    cfg_dict["save_audit_model_metadata"] = bool(args.save_audit_model_metadata)
    cfg_dict["batch_evaluations_across_panel"] = not bool(args.no_batch_evaluations)
    if args.topic_bank_path:
        cfg_dict["topic_bank_path"] = str(args.topic_bank_path)
    return cfg_dict


def write_dataframe(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if str(path).endswith(".gz"):
        df.to_csv(path, index=False, compression="gzip")
    else:
        df.to_csv(path, index=False)


def run(args: argparse.Namespace) -> None:
    start = time.time()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    collector = import_collector_module(args.collector_path)
    base_cfg = build_base_config(collector, args)
    conditions = load_conditions(args.conditions_json)
    seeds = parse_int_list(args.seeds) or [42, 43, 44]
    score_weights = parse_score_weights(args.score_weights)

    write_json_atomic(out_dir / "closed_loop_run_config.json", {
        "args": vars(args),
        "conditions": [condition_to_dict(c) for c in conditions],
        "seeds": seeds,
        "score_weights": dict(score_weights),
        "phase_thresholds": PHASE_THRESHOLDS,
        "note": "Small-scale full closed-loop validation: exposure controls future generation; replay remains the main grid.",
    })

    summaries: List[Dict[str, Any]] = []
    trajectories: List[Dict[str, Any]] = []
    posts: List[Dict[str, Any]] = []
    perturbations: List[Dict[str, Any]] = []
    for condition in conditions:
        for seed in seeds:
            print(f"[start] condition={condition.label} seed={seed}", flush=True)
            summary, traj_rows, post_rows, perturb_rows = run_one_condition_seed(
                collector_module=collector,
                base_cfg_dict=base_cfg,
                condition=condition,
                seed=int(seed),
                args=args,
                score_weights=score_weights,
            )
            summaries.append(summary)
            trajectories.extend(traj_rows)
            posts.extend(post_rows)
            perturbations.extend(perturb_rows)
            print(
                f"[done] condition={condition.label} seed={seed} "
                f"phase={summary.get('phase_label')} top1={summary.get('top1_share'):.3f} "
                f"Neff={summary.get('n_eff'):.2f}",
                flush=True,
            )

    summary_df = pd.DataFrame(summaries)
    summary_df = attach_nearest_replay_metrics(summary_df, args.reference_replay_summary_csv, args.replay_match_cols)
    write_replay_online_agreement_tables(summary_df, out_dir)
    write_dataframe(out_dir / "closed_loop_summary.csv", summary_df)
    write_dataframe(out_dir / "closed_loop_trajectories.csv.gz", pd.DataFrame(trajectories))
    if args.save_generated_posts_csv:
        write_dataframe(out_dir / "closed_loop_generated_posts.csv.gz", pd.DataFrame(posts))
    if perturbations:
        perturb_df = pd.DataFrame(perturbations)
        replay_cols = [
            c for c in [
                "condition_label", "seed", "kappa_map_replay", "kappa_replay", "kappa_map_eta_replay",
                "phase_label_replay", "phase_code_replay",
            ]
            if c in summary_df.columns
        ]
        if {"condition_label", "seed"}.issubset(replay_cols):
            perturb_df = perturb_df.merge(
                summary_df[replay_cols].drop_duplicates(["condition_label", "seed"]),
                on=["condition_label", "seed"],
                how="left",
            )
            kappa_col = next((c for c in ["kappa_map_replay", "kappa_replay", "kappa_map_eta_replay"] if c in perturb_df.columns), "")
            if kappa_col:
                kappa = pd.to_numeric(perturb_df[kappa_col], errors="coerce")
                growth = pd.to_numeric(perturb_df["online_growth_rate_l1"], errors="coerce")
                tol = max(0.0, float(args.perturbation_growth_tol))
                online_contracting = growth < -tol
                online_expanding = growth > tol
                online_near_neutral = ~(online_contracting | online_expanding)
                replay_stable = kappa < 0
                perturb_df["replay_stable_from_kappa"] = replay_stable.where(kappa.notna(), np.nan)
                perturb_df["online_near_neutral_for_agreement"] = online_near_neutral.where(growth.notna(), np.nan)
                comparable = kappa.notna() & growth.notna() & (~online_near_neutral)
                perturb_df["sign_agreement_with_replay_kappa"] = (online_contracting == replay_stable).where(comparable, np.nan)
        write_dataframe(out_dir / "closed_loop_perturbation_summary.csv", perturb_df)

    grouped_cols = ["condition_label", "label", "eta", "alpha", "beta_sup", "beta_route", "lambda_risk"]
    available_grouped_cols = [c for c in grouped_cols if c in summary_df.columns]
    numeric_cols = [
        "top1_share", "hhi", "n_eff", "entropy_norm", "i_exp_norm", "winner_switch_rate",
        "weighted_quality", "weighted_safety", "weighted_risk", "generated_base_score_mean",
        "generated_base_score_second_minus_first_half", "generated_agent_frac", "unique_content_hash_rate",
    ]
    available_numeric = [c for c in numeric_cols if c in summary_df.columns]
    if available_grouped_cols and available_numeric:
        agg = summary_df.groupby(available_grouped_cols, dropna=False)[available_numeric].agg(["mean", "std", "count"])
        agg.columns = ["_".join([str(x) for x in col if x]) for col in agg.columns.to_flat_index()]
        agg = agg.reset_index()
        write_dataframe(out_dir / "closed_loop_condition_aggregate.csv", agg)

    done_payload = {
        "elapsed_sec": time.time() - start,
        "n_conditions": len(conditions),
        "n_seeds": len(seeds),
        "n_runs": len(summaries),
        "summary_csv": str(out_dir / "closed_loop_summary.csv"),
    }
    if perturbations:
        done_payload["perturbation_summary_csv"] = str(out_dir / "closed_loop_perturbation_summary.csv")
    write_json_atomic(out_dir / "done.json", done_payload)
    print(f"wrote outputs under {out_dir}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Small-scale full closed-loop validation for AI social-platform selection dynamics.")
    p.add_argument("--collector_path", default="log_collect.py", help="Path to log_collect.py.")
    p.add_argument("--out_dir", default="../../data/results/closed_loop_validation_grid")
    p.add_argument("--config_json", default="", help="JSON object or path with LogGenerationConfig overrides.")
    p.add_argument("--conditions_json", default="", help="JSON list or path defining ClosedLoopCondition objects. Defaults to four Appendix validation conditions.")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--n_timesteps", type=int, default=80)
    p.add_argument("--n_agents", type=int, default=32)
    p.add_argument("--n_evaluators", type=int, default=6)
    p.add_argument("--exposed_agents_per_step", type=int, default=8)
    p.add_argument("--selection_mode", choices=["weighted_without_replacement", "topk", "uniform"], default="weighted_without_replacement")
    p.add_argument("--endpoint_last_k", type=int, default=20)
    p.add_argument("--prior_score", type=float, default=0.5)
    p.add_argument("--prior_count", type=float, default=1.0)
    p.add_argument("--score_weights", default="", help="JSON object over like/reply/share/follow/eval_quality/audit_quality/etc.")
    p.add_argument("--topic_bank_path", default="", help="Override topic bank path in the collector config.")
    p.add_argument("--temperature_post", type=float, default=-1.0, help="Override post temperature; negative keeps config default.")
    p.add_argument("--temperature_eval", type=float, default=-1.0, help="Override evaluator temperature; negative keeps config default.")
    p.add_argument("--post_workers", type=int, default=4)
    p.add_argument("--eval_workers", type=int, default=4)
    p.add_argument("--audit_workers", type=int, default=4)
    p.add_argument("--progress_every", type=int, default=10)
    p.add_argument("--resume", action="store_true", help="Fail fast if checkpoints already exist unless --overwrite_checkpoints is provided.")
    p.add_argument("--overwrite_checkpoints", action="store_true")
    p.add_argument("--save_full_fitness", action="store_true", help="Save full fitness vector in each timestep checkpoint.")
    p.add_argument("--save_full_text", action="store_true", help="Save prompts in checkpoints.")
    p.add_argument("--save_response_text", action="store_true", help="Save raw model responses in checkpoints.")
    p.add_argument("--save_eval_model_metadata", action="store_true")
    p.add_argument("--save_audit_model_metadata", action="store_true")
    p.add_argument("--no_batch_evaluations", action="store_true")
    p.add_argument("--save_generated_posts_csv", action="store_true")
    p.add_argument("--include_post_text_in_csv", action="store_true", help="Only used with --save_generated_posts_csv.")
    p.add_argument("--reference_replay_summary_csv", default="", help="Optional frozen replay phase_summary.csv for closed-loop-minus-replay deltas.")
    p.add_argument("--replay_match_cols", default="eta,alpha,beta_sup,beta_route,lambda_risk", help="Comma-separated platform coordinates used for nearest replay matching.")
    p.add_argument("--perturbation_validation", action="store_true")
    p.add_argument("--perturbation_steps", type=int, default=30)
    p.add_argument("--perturbation_mass", type=float, default=0.03)
    p.add_argument("--perturbation_growth_tol", type=float, default=0.005, help="Near-neutral band for online perturbation growth-rate classification.")
    p.add_argument("--n_perturbations", type=int, default=8)
    p.add_argument("--perturbation_modes", default="active,invasion,random")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run(args)


if __name__ == "__main__":
    main()
