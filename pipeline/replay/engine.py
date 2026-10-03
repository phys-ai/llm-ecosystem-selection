#!/usr/bin/env python3
"""Replay of the stored logs under the platform's exposure update (the replay grids in data/results and data/testbeds).

Reads checkpoint directories (metadata.json + timestep_*.json, written by collect/log_collect.py) one timestep at a
time, replays each (eta, alpha, beta_sup, beta_route, lambda_risk, ...) condition and writes per-run phase_summary.csv,
trajectories, endpoint exposures, the local-stability audit (--estimate_stability) and the early-warning features /
evaluation (--evaluate_early_warning).  See replay/main_grids.sh for the grids of the paper.
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
import math
import os
import re
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

# Keep numerical libraries from spawning many worker threads on shared machines.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

EPS = 1.0e-12

CONTEXT_AXIS_COLUMNS = [
    "controversy",
    "emotional_load",
    "interpersonalness",
    "evidentiality",
    "engagement_baitness",
    "public_vs_personal",
    "safety_risk_hint",
]

DEFAULT_SCORE_WEIGHTS: Dict[str, float] = {
    "like": 0.20,
    "reply": 0.20,
    "share": 0.20,
    "follow": 0.10,
    "eval_quality": 0.15,
    "audit_quality": 0.15,
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

# Operational thresholds for finite-sample phase summaries.  These are not the
# theoretical phase definitions; they are descriptor thresholds used to summarize
# replay endpoints after the endpoint and stability objects have been computed.
PHASE_THRESHOLDS: Dict[str, float] = {
    "collapse_top1": 0.80,
    "collapse_neff_abs": 1.5,
    "collapse_neff_frac": 0.025,
    "concentrated_top1": 0.45,
    "concentrated_neff_abs": 3.0,
    "concentrated_neff_frac": 0.10,
    #"polarization_abs": 0.35,

    "polarization_abs": 0.35,
    "polarization_var": 0.50,
    "stance_extremity": 0.70,

    "specialization_mi_norm": 0.12,
    "strong_specialization_mi_norm": 0.15,
    "winner_switch": 0.30,
    "near_critical_eps": 0.05,
}

METRIC_KEYS = [
    "like",
    "reply",
    "share",
    "follow",
    "eval_quality",
    "eval_safety",
    "audit_quality",
    "audit_safety",
]



@dataclass
class PreparedLog:
    run_name: str
    source_path: str
    agent_ids: List[str]
    timesteps: List[int]
    base_score: np.ndarray          # T x N, float32/float64
    agent_mean_score: np.ndarray    # N
    risk_penalty: np.ndarray        # T x N
    safety_score: np.ndarray        # T x N
    quality_score: np.ndarray       # T x N
    agent_mean_risk: np.ndarray     # N
    agent_mean_safety: np.ndarray   # N
    agent_mean_quality: np.ndarray  # N
    trait_coords: pd.DataFrame      # rows agent_ids
    context_df: pd.DataFrame        # one row per timestep
    metadata: Dict[str, Any]

    @property
    def n_agents(self) -> int:
        return len(self.agent_ids)

    @property
    def n_timesteps(self) -> int:
        return len(self.timesteps)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def natural_key(value: Any) -> Tuple[Any, ...]:
    text = str(value)
    parts = re.split(r"(\d+)", text)
    return tuple(int(p) if p.isdigit() else p for p in parts)


def parse_float_list(raw: str) -> List[float]:
    if raw is None or str(raw).strip() == "":
        return []
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def parse_int_list(raw: str) -> List[int]:
    if raw is None or str(raw).strip() == "":
        return []
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def parse_score_weights(raw: str) -> Dict[str, float]:
    if not raw:
        weights = dict(DEFAULT_SCORE_WEIGHTS)
    else:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("--score_weights must be a JSON object")
        weights = {str(k): float(v) for k, v in parsed.items()}
    total = sum(max(0.0, v) for v in weights.values())
    if total <= 0:
        raise ValueError("score weights must contain at least one positive value")
    return {k: max(0.0, v) / total for k, v in weights.items()}


def read_json(path: Path) -> Any:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def is_number(value: Any) -> bool:
    try:
        x = float(value)
        return math.isfinite(x)
    except Exception:
        return False


def to_float(value: Any, default: float = float("nan")) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default




def safe_entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum()) if p.size else 0.0


def lag1_autocorr(x: Sequence[float]) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 3:
        return float("nan")
    a, b = arr[:-1], arr[1:]
    if np.std(a) < EPS or np.std(b) < EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def linear_slope(y: Sequence[float]) -> float:
    arr = np.asarray(y, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return float("nan")
    x = np.arange(arr.size, dtype=float)
    x -= x.mean()
    denom = float((x * x).sum())
    if denom <= EPS:
        return 0.0
    return float(((arr - arr.mean()) * x).sum() / denom)


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------


def looks_like_checkpoint_dir(path: Path) -> bool:
    return path.is_dir() and (path / "metadata.json").exists() and any(path.glob("timestep_*.json*"))


def discover_log_sources(runs_dir: Path, max_runs: int = 0) -> List[Path]:
    """Find the checkpoint directories under runs_dir."""
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs_dir does not exist: {runs_dir}")

    checkpoint_dirs = sorted([p for p in runs_dir.rglob("*") if looks_like_checkpoint_dir(p)], key=lambda p: str(p))
    if looks_like_checkpoint_dir(runs_dir):
        checkpoint_dirs = [runs_dir] + [p for p in checkpoint_dirs if p.resolve() != runs_dir.resolve()]

    sources: List[Path] = list(checkpoint_dirs)


    if max_runs and max_runs > 0:
        sources = sources[: int(max_runs)]

    if not sources:
        raise FileNotFoundError(f"No checkpoint directories found under {runs_dir}.")
    return sources


def run_name_from_path(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except Exception:
        rel = path.name
    text = str(rel).replace(os.sep, "__")
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "_", text)
    if text.endswith(".json"):
        text = text[:-5]
    return text[:180] or "run"


def checkpoint_step_paths(path: Path, max_timesteps: int = 0) -> List[Path]:
    steps = sorted(path.glob("timestep_*.json*"), key=natural_key)  # plain or gzipped (export_logs.py)
    if max_timesteps and max_timesteps > 0:
        steps = steps[: int(max_timesteps)]
    if not steps:
        raise FileNotFoundError(f"No timestep_*.json files under {path}")
    return steps


# ---------------------------------------------------------------------------
# Trait and context extraction
# ---------------------------------------------------------------------------


def extract_trait_coords_from_value(value: Any) -> Dict[str, float]:
    if not isinstance(value, dict):
        return {}
    if isinstance(value.get("coords"), dict):
        return {str(k): float(v) for k, v in value["coords"].items() if is_number(v)}

    # Legacy discrete profile fallback.
    out: Dict[str, float] = {}
    position = str(value.get("position", "")).lower()
    specialization = str(value.get("specialization", "")).lower()
    objective = str(value.get("objective_orientation", "")).lower()
    if position:
        if "affirm" in position:
            out["stance"] = 1.0
        elif "skept" in position:
            out["stance"] = -1.0
        else:
            out["stance"] = 0.0
    if specialization:
        if "social" in specialization:
            out["sociality"] = 1.0
        elif "analytic" in specialization:
            out["sociality"] = -1.0
        else:
            out["sociality"] = 0.0
    if objective:
        if "engagement" in objective or "attention" in objective:
            out["risk_tolerance"] = 1.0
        elif "safety" in objective:
            out["risk_tolerance"] = -1.0
        else:
            out["risk_tolerance"] = 0.0
    return out


def extract_trait_coords_from_row(row: Mapping[str, Any]) -> Dict[str, float]:
    for key in ("trait_coords", "agent_trait_coords"):
        value = row.get(key)
        if isinstance(value, dict):
            coords = {str(k): float(v) for k, v in value.items() if is_number(v)}
            if coords:
                return coords
    for key in ("trait", "agent_trait"):
        coords = extract_trait_coords_from_value(row.get(key))
        if coords:
            return coords
    return {}


def context_record_from_bundle(bundle: Mapping[str, Any]) -> Dict[str, Any]:
    contents = bundle.get("contents") or []
    first = contents[0] if contents and isinstance(contents[0], dict) else {}
    rec: Dict[str, Any] = {
        "timestep": int(to_float(bundle.get("timestep", first.get("timestep", 0)), 0.0)),
        "context_id": str(bundle.get("context_id", first.get("context_id", ""))),
        "topic": str(bundle.get("topic", first.get("topic", ""))),
        "topic_category": str(bundle.get("topic_category", first.get("topic_category", first.get("source_subreddit", "uncategorized"))) or "uncategorized"),
        "topic_index": to_float(bundle.get("topic_index", first.get("topic_index", float("nan")))),
        "topic_source": str(bundle.get("topic_source", first.get("topic_source", ""))),
        "source_subreddit": str(bundle.get("source_subreddit", first.get("source_subreddit", ""))),
        "source_post_id": str(bundle.get("source_post_id", first.get("source_post_id", ""))),
    }
    for col in CONTEXT_AXIS_COLUMNS:
        rec[col] = to_float(first.get(col, bundle.get(col, 0.0)), 0.0)
    return rec


def metadata_agent_ids_and_traits(metadata: Mapping[str, Any]) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    agent_ids: List[str] = []
    traits: Dict[str, Dict[str, float]] = {}
    agents = metadata.get("agents") if isinstance(metadata, Mapping) else None
    if isinstance(agents, list):
        for idx, agent in enumerate(agents):
            if not isinstance(agent, dict):
                continue
            aid = str(agent.get("agent_id", f"agent_{idx}"))
            agent_ids.append(aid)
            coords = extract_trait_coords_from_value(agent.get("trait"))
            if coords:
                traits[aid] = coords
    return sorted(set(agent_ids), key=natural_key), traits


# ---------------------------------------------------------------------------
# Low-memory log preparation
# ---------------------------------------------------------------------------


def iter_checkpoint_bundles(path: Path, max_timesteps: int = 0) -> Iterator[Tuple[Path, Dict[str, Any]]]:
    for step in checkpoint_step_paths(path, max_timesteps=max_timesteps):
        try:
            bundle = read_json(step)
        except Exception as exc:
            print(f"[warn] skipping unreadable checkpoint {step}: {exc}", file=sys.stderr)
            continue
        if isinstance(bundle, dict):
            yield step, bundle



def scan_agent_ids_and_traits(
    source: Path,
    metadata: Mapping[str, Any],
    max_timesteps: int,
) -> Tuple[List[str], List[int], Dict[str, Dict[str, float]]]:
    agent_ids, trait_by_agent = metadata_agent_ids_and_traits(metadata)
    agent_set = set(agent_ids)
    timestep_set: set[int] = set()

    iterator = iter_checkpoint_bundles(source, max_timesteps=max_timesteps)
    for _path, bundle in iterator:
        t = int(to_float(bundle.get("timestep", 0), 0.0))
        timestep_set.add(t)
        for section in ("contents", "evaluations", "audits"):
            rows = bundle.get(section, [])
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                aid_raw = row.get("agent_id")
                if aid_raw is None:
                    continue
                aid = str(aid_raw)
                agent_set.add(aid)
                if aid not in trait_by_agent:
                    coords = extract_trait_coords_from_row(row)
                    if coords:
                        trait_by_agent[aid] = coords
    if not agent_set:
        raise ValueError(f"Could not infer agent IDs from {source}")
    if not timestep_set:
        raise ValueError(f"Could not infer timesteps from {source}")
    return sorted(agent_set, key=natural_key), sorted(timestep_set), trait_by_agent


def fill_missing_matrix(mat: np.ndarray, default: float = 0.5) -> np.ndarray:
    arr = np.asarray(mat, dtype=np.float32)
    with np.errstate(all="ignore"):
        col_means = np.nanmean(arr, axis=0)
        global_mean = np.nanmean(arr)
    if not np.isfinite(global_mean):
        global_mean = float(default)
    col_means = np.where(np.isfinite(col_means), col_means, global_mean).astype(np.float32)
    rows, cols = np.where(~np.isfinite(arr))
    if rows.size:
        arr[rows, cols] = col_means[cols]
    arr = np.where(np.isfinite(arr), arr, float(default)).astype(np.float32)
    return np.clip(arr, 0.0, 1.0)


def build_trait_dataframe(agent_ids: List[str], trait_by_agent: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    dims = sorted({dim for coords in trait_by_agent.values() for dim in coords.keys()}, key=natural_key)
    if not dims:
        dims = ["stance", "sociality", "risk_tolerance"]
    mat = pd.DataFrame(0.0, index=agent_ids, columns=dims)
    for aid, coords in trait_by_agent.items():
        if aid not in mat.index:
            continue
        for dim, value in coords.items():
            if dim not in mat.columns:
                mat[dim] = 0.0
            mat.loc[aid, dim] = float(value)
    return mat.reindex(agent_ids).fillna(0.0)


def prepare_log_streaming(
    source: Path,
    run_name: str,
    score_weights: Dict[str, float],
    max_timesteps: int = 0,
) -> PreparedLog:
    if source.is_dir() and (source / "metadata.json").exists():
        metadata = read_json(source / "metadata.json")
    elif source.is_file():
        metadata = read_json(source)
    else:
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    is_tool_experiment = (
        metadata.get("experiment_type") == "bfcl_tool_routing_log_dataset"
    )

    agent_ids, timesteps, trait_by_agent = scan_agent_ids_and_traits(source, metadata, max_timesteps=max_timesteps)
    agent_index = {aid: idx for idx, aid in enumerate(agent_ids)}
    timestep_index = {t: idx for idx, t in enumerate(timesteps)}
    T, N = len(timesteps), len(agent_ids)

    metrics = {key: np.full((T, N), np.nan, dtype=np.float32) for key in METRIC_KEYS}
    task_fitness = np.full((T, N), np.nan, dtype=np.float32)
    context_records: List[Dict[str, Any]] = [{"timestep": t, "topic_category": "uncategorized"} for t in timesteps]

    iterator = iter_checkpoint_bundles(source, max_timesteps=max_timesteps)
    for step_path, bundle in iterator:
        t = int(to_float(bundle.get("timestep", 0), 0.0))
        if t not in timestep_index:
            continue
        ti = timestep_index[t]
        context_records[ti] = context_record_from_bundle(bundle)
        if "timestep" not in context_records[ti]:
            context_records[ti]["timestep"] = t

        # Traits can be recovered from content/audit/eval rows if metadata was absent.
        for section in ("contents", "evaluations", "audits"):
            rows = bundle.get(section, [])
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                aid = str(row.get("agent_id", ""))
                if aid and aid not in trait_by_agent:
                    coords = extract_trait_coords_from_row(row)
                    if coords:
                        trait_by_agent[aid] = coords

        if is_tool_experiment:
            for row in bundle.get("contents", []) or []:
                if not isinstance(row, dict):
                    continue
                j = agent_index.get(str(row.get("agent_id", "")))
                if j is None:
                    continue
                value = to_float(row.get("fitness", row.get("correct")))
                if math.isfinite(value):
                    task_fitness[ti, j] = value

        eval_sums = {k: np.zeros(N, dtype=np.float64) for k in ["like", "reply", "share", "follow", "eval_quality", "eval_safety"]}
        eval_counts = {k: np.zeros(N, dtype=np.int32) for k in eval_sums}
        for row in bundle.get("evaluations", []) or []:
            if not isinstance(row, dict):
                continue
            j = agent_index.get(str(row.get("agent_id", "")))
            if j is None:
                continue
            key_map = {
                "like": "like",
                "reply": "reply",
                "share": "share",
                "follow": "follow",
                "quality": "eval_quality",
                "safety": "eval_safety",
            }
            for raw_key, out_key in key_map.items():
                value = to_float(row.get(raw_key))
                if math.isfinite(value):
                    eval_sums[out_key][j] += value
                    eval_counts[out_key][j] += 1
        for out_key, sums in eval_sums.items():
            counts = eval_counts[out_key]
            mask = counts > 0
            if mask.any():
                metrics[out_key][ti, mask] = (sums[mask] / counts[mask]).astype(np.float32)

        audit_sums = {"audit_quality": np.zeros(N, dtype=np.float64), "audit_safety": np.zeros(N, dtype=np.float64)}
        audit_counts = {"audit_quality": np.zeros(N, dtype=np.int32), "audit_safety": np.zeros(N, dtype=np.int32)}
        for row in bundle.get("audits", []) or []:
            if not isinstance(row, dict):
                continue
            j = agent_index.get(str(row.get("agent_id", "")))
            if j is None:
                continue
            for raw_key, out_key in (("quality", "audit_quality"), ("safety", "audit_safety")):
                value = to_float(row.get(raw_key))
                if math.isfinite(value):
                    audit_sums[out_key][j] += value
                    audit_counts[out_key][j] += 1
        for out_key, sums in audit_sums.items():
            counts = audit_counts[out_key]
            mask = counts > 0
            if mask.any():
                metrics[out_key][ti, mask] = (sums[mask] / counts[mask]).astype(np.float32)

    # Fill missing values metric-wise.
    for key in list(metrics):
        metrics[key] = fill_missing_matrix(metrics[key], default=0.5)

    if is_tool_experiment:
        base_score = fill_missing_matrix(task_fitness, default=0.0)
        quality_score = base_score.copy()
        safety_score = np.ones((T, N), dtype=np.float32)
        risk_penalty = np.zeros((T, N), dtype=np.float32)
    else:
        used_weight = 0.0
        base_score = np.zeros((T, N), dtype=np.float32)
        for key, weight in score_weights.items():
            if key not in metrics:
                print(f"[warn] ignoring unknown score weight key: {key}", file=sys.stderr)
                continue
            base_score += np.float32(weight) * metrics[key]
            used_weight += float(weight)
        if used_weight <= 0:
            raise ValueError("score_weights did not match available metric columns")
        base_score = np.clip(base_score / np.float32(used_weight), 0.0, 1.0).astype(np.float32)

        safety_score = np.clip(0.5 * metrics["eval_safety"] + 0.5 * metrics["audit_safety"], 0.0, 1.0).astype(np.float32)
        quality_score = np.clip(0.5 * metrics["eval_quality"] + 0.5 * metrics["audit_quality"], 0.0, 1.0).astype(np.float32)
        risk_penalty = (1.0 - safety_score).astype(np.float32)

    context_df = pd.DataFrame(context_records)
    if "topic_category" not in context_df.columns:
        context_df["topic_category"] = "uncategorized"
    context_df["topic_category"] = context_df["topic_category"].fillna("uncategorized").replace("", "uncategorized").astype(str)
    for col in CONTEXT_AXIS_COLUMNS:
        if col not in context_df.columns:
            context_df[col] = 0.0
        context_df[col] = pd.to_numeric(context_df[col], errors="coerce").fillna(0.0)

    trait_df = build_trait_dataframe(agent_ids, trait_by_agent)

    return PreparedLog(
        run_name=run_name,
        source_path=str(source),
        agent_ids=agent_ids,
        timesteps=timesteps,
        base_score=base_score,
        agent_mean_score=base_score.mean(axis=0).astype(np.float64),
        risk_penalty=risk_penalty,
        safety_score=safety_score,
        quality_score=quality_score,
        agent_mean_risk=risk_penalty.mean(axis=0).astype(np.float64),
        agent_mean_safety=safety_score.mean(axis=0).astype(np.float64),
        agent_mean_quality=quality_score.mean(axis=0).astype(np.float64),
        trait_coords=trait_df,
        context_df=context_df,
        metadata={k: v for k, v in metadata.items() if k not in ("contents", "evaluations", "audits")},
    )


# ---------------------------------------------------------------------------
# Replay dynamics and descriptors
# ---------------------------------------------------------------------------


def support_term(p: np.ndarray, target: np.ndarray, clip: float) -> np.ndarray:
    raw = np.log(target + EPS) - np.log(p + EPS)
    if clip > 0:
        raw = np.clip(raw, -clip, clip)
    return raw


def _trait_vector(table: PreparedLog, name: str, default: float = 0.0) -> np.ndarray:
    if name in table.trait_coords.columns:
        return table.trait_coords[name].to_numpy(dtype=np.float64)
    return np.full(table.n_agents, float(default), dtype=np.float64)



def transform_route_residual(residual: np.ndarray, params: ReplayParams) -> np.ndarray:
    route = np.asarray(residual, dtype=np.float64)
    if params.route_zscore:
        sd = float(np.std(route))
        if sd > EPS:
            route = (route - float(np.mean(route))) / sd
        else:
            route = route - float(np.mean(route))
    power = max(0.25, float(params.route_power))
    if abs(power - 1.0) > 1.0e-12:
        route = np.sign(route) * (np.abs(route) ** power)
    return route



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


def concentration_descriptors(p: np.ndarray) -> Dict[str, float]:
    p = np.asarray(p, dtype=np.float64)
    hhi = float(np.sum(p * p))
    n_eff = float(1.0 / max(hhi, EPS))
    ent = safe_entropy(p)
    return {
        "hhi": hhi,
        "n_eff": n_eff,
        "top1_share": float(np.max(p)),
        "entropy": ent,
        "entropy_norm": float(ent / math.log(len(p))) if len(p) > 1 else 0.0,
        "active_support_1e-3": int(np.sum(p > 1.0e-3)),
        "active_support_1e-4": int(np.sum(p > 1.0e-4)),
    }



def risk_quality_descriptors(table: PreparedLog, p: np.ndarray) -> Dict[str, float]:
    return {
        "weighted_risk": float(np.dot(table.agent_mean_risk, p)),
        "weighted_safety": float(np.dot(table.agent_mean_safety, p)),
        "weighted_quality": float(np.dot(table.agent_mean_quality, p)),
    }


def q_for_context(table: PreparedLog, t_idx: int, p: np.ndarray, params: ReplayParams, support_clip: float) -> np.ndarray:
    f = fitness_at(table, t_idx, p, params, support_clip=support_clip)
    eta = 0.0 if params.no_feedback else params.eta
    mask = None
    if params.hard_safety_threshold >= 0:
        mask = table.safety_score[t_idx] >= float(params.hard_safety_threshold)
    return softmax_exposure_update(p, f, eta, mask)


def context_mi_and_winners(table: PreparedLog, p: np.ndarray, params: ReplayParams, context_key: str, support_clip: float) -> Dict[str, Any]:
    if context_key not in table.context_df.columns:
        context_key = "topic_category"
    cvals = table.context_df[context_key].fillna("uncategorized").astype(str).to_numpy()
    q_sums: Dict[str, np.ndarray] = {}
    counts: Dict[str, int] = {}
    winner_counts = np.zeros(table.n_agents, dtype=np.int64)
    px = np.zeros(table.n_agents, dtype=np.float64)

    for t_idx in range(table.n_timesteps):
        q = q_for_context(table, t_idx, p, params, support_clip=support_clip)
        c = str(cvals[t_idx])
        if c not in q_sums:
            q_sums[c] = np.zeros(table.n_agents, dtype=np.float64)
            counts[c] = 0
        q_sums[c] += q
        counts[c] += 1
        winner_counts[int(np.argmax(q))] += 1
        px += q

    if not counts:
        return {"i_exp": 0.0, "i_exp_norm": 0.0, "context_count": 0, "winner_switch_rate": 0.0, "winner_unique_count": 0, "winner_top_share": 0.0}

    total_t = float(sum(counts.values()))
    px = px / max(total_t, 1.0)
    mi = 0.0
    pc_list = []
    for c, count in counts.items():
        pc = float(count) / total_t
        pc_list.append(pc)
        q_c = q_sums[c] / float(count)
        ratio = q_c / (px + EPS)
        mi += float(pc * np.sum(q_c * np.log(ratio + EPS)))
    pc_arr = np.asarray(pc_list, dtype=np.float64)
    hc = safe_entropy(pc_arr)
    freqs = winner_counts / max(float(winner_counts.sum()), 1.0)
    winner_switch = float(1.0 - np.sum(freqs * freqs))
    return {
        "i_exp": float(mi),
        "i_exp_norm": float(mi / max(hc, EPS)) if hc > EPS else 0.0,
        "context_count": int(len(counts)),
        "winner_switch_rate": winner_switch,
        "winner_unique_count": int(np.sum(winner_counts > 0)),
        "winner_top_share": float(np.max(freqs)) if freqs.size else 0.0,
    }



def phase_label(desc: Mapping[str, Any], n_agents: int) -> str:
    flags = descriptor_flags(desc, n_agents)
    collapse = bool(flags["collapse_signal"])
    concentrated = bool(flags["concentration_signal"])
    polarized = bool(flags["polarization_signal"])
    specialized = bool(flags["specialization_signal"])

    # Specialized collapse is intentionally separated from pure monoculture
    # collapse so that context-dependent allocation is not erased by the
    # concentration criterion.
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




def replay_one_condition(
    table: PreparedLog,
    params: ReplayParams,
    last_k: int,
    context_key: str,
    support_clip: float,
    collect_trajectory: bool,
    trajectory_stride: int,
    snapshot_steps: Optional[set[int]] = None,
) -> Tuple[np.ndarray, Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    n = table.n_agents
    p = np.ones(n, dtype=np.float64) / float(n)
    endpoint_window: deque[np.ndarray] = deque(maxlen=max(1, int(last_k)))
    endpoint_window.append(p.copy())
    traj_rows: List[Dict[str, Any]] = []
    snapshot_steps = snapshot_steps or set()
    snapshot_rows: List[Dict[str, Any]] = []
    eta = 0.0 if params.no_feedback else float(params.eta)
    stride = max(1, int(trajectory_stride))

    for t_idx, timestep in enumerate(table.timesteps):
        if t_idx in snapshot_steps:
            snapshot_rows.extend(agent_exposure_rows(table, p, timestep, t_idx))
        if collect_trajectory and (t_idx % stride == 0):
            traj_rows.append(trajectory_row(table, p, timestep, t_idx))
        f = fitness_at(table, t_idx, p, params, support_clip=support_clip)
        mask = None
        if params.hard_safety_threshold >= 0:
            mask = table.safety_score[t_idx] >= float(params.hard_safety_threshold)
        p = softmax_exposure_update(p, f, eta, mask)
        endpoint_window.append(p.copy())

    final_t = int(table.timesteps[-1]) + 1 if table.timesteps else 0
    if len(table.timesteps) in snapshot_steps:
        snapshot_rows.extend(agent_exposure_rows(table, p, final_t, len(table.timesteps)))
    if collect_trajectory:
        traj_rows.append(trajectory_row(table, p, final_t, len(table.timesteps)))

    endpoint = np.mean(np.vstack(list(endpoint_window)), axis=0)
    endpoint = np.clip(endpoint, EPS, None)
    endpoint = endpoint / endpoint.sum()

    desc: Dict[str, Any] = {}
    desc.update(concentration_descriptors(endpoint))
    desc.update(weighted_trait_descriptors(table, endpoint))
    desc.update(risk_quality_descriptors(table, endpoint))
    desc.update(context_mi_and_winners(table, endpoint, params, context_key=context_key, support_clip=support_clip))
    desc.update(descriptor_flags(desc, table.n_agents))
    desc["phase_label"] = phase_label(desc, table.n_agents)
    desc["phase_code"] = PHASE_CODES.get(str(desc["phase_label"]), -1)

    endpoint_rows: List[Dict[str, Any]] = []
    for idx, aid in enumerate(table.agent_ids):
        e: Dict[str, Any] = {"agent_id": aid, "agent_index": idx, "exposure": float(endpoint[idx])}
        for dim in table.trait_coords.columns:
            e[f"trait_{dim}"] = float(table.trait_coords.iloc[idx][dim])
        endpoint_rows.append(e)
    return endpoint, desc, traj_rows, endpoint_rows, snapshot_rows


# ---------------------------------------------------------------------------
# Local map stability, implemented without constructing T x N fitness matrices
# ---------------------------------------------------------------------------


def mean_one_step_map(table: PreparedLog, p: np.ndarray, params: ReplayParams, support_clip: float, context_stride: int) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, None)
    p = p / p.sum()
    stride = max(1, int(context_stride))
    acc = np.zeros_like(p)
    count = 0
    for t_idx in range(0, table.n_timesteps, stride):
        acc += q_for_context(table, t_idx, p, params, support_clip=support_clip)
        count += 1
    if count <= 0:
        return p.copy()
    y = acc / float(count)
    return y / y.sum()


def estimate_map_stability(
    table: PreparedLog,
    endpoint: np.ndarray,
    params: ReplayParams,
    active_eps: float,
    fd_eps: float,
    invasion_eps: float,
    support_clip: float,
    context_stride: int,
    max_active: int,
) -> Dict[str, Any]:
    p = np.clip(np.asarray(endpoint, dtype=np.float64), EPS, None)
    p = p / p.sum()
    active = np.where(p > active_eps)[0]
    inactive = np.where(p <= active_eps)[0]
    if active.size == 0:
        active = np.array([int(np.argmax(p))], dtype=int)
        inactive = np.array([i for i in range(p.size) if i != active[0]], dtype=int)
    if active.size > max_active:
        kept = np.sort(active[np.argsort(-p[active])[:max_active]])
        dropped = np.array([i for i in active if i not in set(kept.tolist())], dtype=int)
        inactive = np.sort(np.concatenate([inactive, dropped]))
        active = kept

    tangent_dim = max(0, int(active.size) - 1)
    gamma_map = float("-inf")
    spectral_radius = 0.0
    eig_real_max = float("nan")
    eig_abs_max = 0.0

    if tangent_dim > 0:
        ref = int(active[-1])
        min_mass = float(np.min(p[active]))
        delta = max(min(float(fd_eps), 0.25 * min_mass), 1.0e-10)
        jac = np.zeros((tangent_dim, tangent_dim), dtype=np.float64)
        for col_idx, idx in enumerate(active[:-1]):
            direction = np.zeros_like(p)
            direction[int(idx)] = 1.0
            direction[ref] = -1.0
            pp = p + delta * direction
            pm = p - delta * direction
            if np.any(pp < 0) or np.any(pm < 0):
                pp = np.clip(pp, EPS, None); pp = pp / pp.sum()
                pm = np.clip(pm, EPS, None); pm = pm / pm.sum()
            yp = mean_one_step_map(table, pp, params, support_clip=support_clip, context_stride=context_stride)
            ym = mean_one_step_map(table, pm, params, support_clip=support_clip, context_stride=context_stride)
            diff = (yp - ym) / (2.0 * delta)
            jac[:, col_idx] = diff[active[:-1]]
        eigvals = np.linalg.eigvals(jac)
        spectral_radius = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
        eig_abs_max = spectral_radius
        eig_real_max = float(np.max(np.real(eigvals))) if eigvals.size else float("nan")
        gamma_map = float(np.log(max(spectral_radius, EPS)))

    max_invasion_log = float("-inf")
    max_invasion_agent = ""
    inv_eps = float(invasion_eps)
    for r in inactive:
        p_eps = (1.0 - inv_eps) * p
        p_eps[int(r)] += inv_eps
        p_eps = p_eps / p_eps.sum()
        y = mean_one_step_map(table, p_eps, params, support_clip=support_clip, context_stride=context_stride)
        g = float(max(y[int(r)], EPS) / inv_eps)
        rho = float(np.log(max(g, EPS)))
        if rho > max_invasion_log:
            max_invasion_log = rho
            max_invasion_agent = table.agent_ids[int(r)]

    kappa = max(gamma_map, max_invasion_log)
    return {
        "active_eps": float(active_eps),
        "active_count": int(active.size),
        "inactive_count": int(inactive.size),
        "tangent_dim": int(tangent_dim),
        "spectral_radius": float(spectral_radius),
        "gamma_map": float(gamma_map),
        "eig_abs_max": float(eig_abs_max),
        "eig_real_max": float(eig_real_max),
        "max_invasion_log": float(max_invasion_log),
        "max_invasion_agent": max_invasion_agent,
        "kappa_map": float(kappa),
        "locally_contracting": bool(kappa < 0.0),
        "stability_context_stride": int(context_stride),
    }


def criticality_label(kappa_map: float) -> str:
    if not math.isfinite(float(kappa_map)):
        return "unknown"
    eps = PHASE_THRESHOLDS["near_critical_eps"]
    if abs(float(kappa_map)) <= eps:
        return "near_critical"
    if float(kappa_map) < -eps:
        return "locally_contracting"
    return "locally_unstable_or_invadable"


# ---------------------------------------------------------------------------
# Early-warning features and plotting
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Experiment orchestration
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Local-stability audit and early-warning validation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReplayParams:
    eta: float
    alpha: float
    beta_sup: float
    beta_route: float
    lambda_risk: float
    hard_safety_threshold: float = -1.0
    context_trait_strength: float = 0.0
    route_power: float = 1.0
    route_zscore: bool = False
    no_feedback: bool = False
    ablation: str = "none"
    polarization_strength: float = 0.0

    def as_prefix(self) -> str:
        return (
            f"eta={self.eta:g}|alpha={self.alpha:g}|sup={self.beta_sup:g}|"
            f"route={self.beta_route:g}|risk={self.lambda_risk:g}|"
            f"hard={self.hard_safety_threshold:g}|ctx_trait={self.context_trait_strength:g}|"
            f"polar={self.polarization_strength:g}|route_power={self.route_power:g}|"
            f"route_zscore={int(self.route_zscore)}|ablation={self.ablation}"
        )


def context_trait_bonus(table: PreparedLog, t_idx: int) -> np.ndarray:
    """Context × trait comparative-advantage term with stronger controllability.

    This is the experimentally manipulable route to stable specialization.  It
    is centered over agents, so it changes relative fitness rather than adding a
    common score shift.
    """
    row = table.context_df.iloc[int(t_idx)]
    stance = _trait_vector(table, "stance")
    sociality = _trait_vector(table, "sociality")
    risk = _trait_vector(table, "risk_tolerance")

    controversy = float(row.get("controversy", 0.0))
    interpersonal = float(row.get("interpersonalness", 0.0))
    evidential = float(row.get("evidentiality", 0.0))
    engagement = float(row.get("engagement_baitness", 0.0))
    safety_hint = float(row.get("safety_risk_hint", 0.0))
    publicness = float(row.get("public_vs_personal", 0.0))

    bonus = (
        -stance * controversy
        + sociality * interpersonal
        - sociality * evidential
        + risk * engagement
        - risk * safety_hint
        + 0.5 * sociality * publicness
    )
    bonus = np.asarray(bonus, dtype=np.float64)
    return bonus - float(np.mean(bonus))


def polarization_bonus(table: PreparedLog, t_idx: int, p: np.ndarray) -> np.ndarray:
    """Stance-axis polarization pressure used only in calibration sweeps.

    The term rewards extreme stance profiles in high-controversy contexts while
    remaining symmetric between the two sides.  With sufficient support, this can
    produce a broad active support over opposing extremes rather than a single
    global winner.
    """
    if "stance" not in table.trait_coords.columns:
        return np.zeros(table.n_agents, dtype=np.float64)
    row = table.context_df.iloc[int(t_idx)]
    stance = _trait_vector(table, "stance")
    controversy = float(row.get("controversy", 0.0))
    emotional = float(row.get("emotional_load", 0.0))
    intensity = 0.5 + 0.5 * np.clip(controversy + emotional, 0.0, 2.0)
    extreme_reward = (stance * stance) * intensity
    # Mild anti-centering term around the current mean preserves a signed-axis
    # instability without forcing one side to dominate by construction.
    signed_mean = float(np.dot(np.asarray(p, dtype=np.float64), stance))
    symmetry_break = 0.15 * stance * signed_mean * intensity
    bonus = extreme_reward + symmetry_break
    return bonus - float(np.mean(bonus))


def fitness_at(table: PreparedLog, t_idx: int, p: np.ndarray, params: ReplayParams, support_clip: float = 8.0) -> np.ndarray:
    alpha = float(np.clip(params.alpha, 0.0, 1.0))
    contextual = table.base_score[t_idx].astype(np.float64, copy=False)
    mean = table.agent_mean_score
    contextual_aug = contextual + alpha * float(params.context_trait_strength) * context_trait_bonus(table, t_idx)
    if float(getattr(params, "polarization_strength", 0.0)) != 0.0:
        contextual_aug = contextual_aug + float(params.polarization_strength) * polarization_bonus(table, t_idx, p)
    base_alpha = (1.0 - alpha) * mean + alpha * contextual_aug
    route = alpha * transform_route_residual(contextual_aug - mean, params)
    target = np.ones(table.n_agents, dtype=np.float64) / float(table.n_agents)
    sup = support_term(p, target, support_clip)
    return base_alpha + params.beta_route * route + params.beta_sup * sup - params.lambda_risk * table.risk_penalty[t_idx]


def weighted_trait_descriptors(table: PreparedLog, p: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for dim in table.trait_coords.columns:
        values = table.trait_coords[dim].to_numpy(dtype=np.float64)
        mean = float(np.dot(p, values))
        var = float(np.dot(p, (values - mean) ** 2))
        abs_mean = float(np.dot(p, np.abs(values)))
        out[f"trait_mean_{dim}"] = mean
        out[f"trait_abs_mean_{dim}"] = abs(mean)
        out[f"trait_weighted_abs_{dim}"] = abs_mean
        out[f"trait_var_{dim}"] = var
    if "stance" in table.trait_coords.columns:
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
    """Descriptor-derived endpoint summaries, kept separate from phase stability."""
    top1 = float(desc.get("top1_share", 0.0))
    n_eff = float(desc.get("n_eff", n_agents))
    abs_pol = float(desc.get("abs_polarization", 0.0))
    pdisp = float(desc.get("polarization_dispersion", 0.0))
    extremity = float(desc.get("stance_extremity", 0.0))
    mi_norm = float(desc.get("i_exp_norm", 0.0))
    winner_switch = float(desc.get("winner_switch_rate", 0.0))

    collapse = top1 >= PHASE_THRESHOLDS["collapse_top1"] or n_eff <= max(
        PHASE_THRESHOLDS["collapse_neff_abs"],
        PHASE_THRESHOLDS["collapse_neff_frac"] * n_agents,
    )
    concentrated = top1 >= PHASE_THRESHOLDS["concentrated_top1"] or n_eff <= max(
        PHASE_THRESHOLDS["concentrated_neff_abs"],
        PHASE_THRESHOLDS["concentrated_neff_frac"] * n_agents,
    )

    polarized = (
        pdisp >= PHASE_THRESHOLDS.get("polarization_var", 0.50)
        and extremity >= PHASE_THRESHOLDS.get("stance_extremity", 0.70)
        and winner_switch >= 0.20
        and top1 < PHASE_THRESHOLDS["collapse_top1"]
        and n_eff <= 0.50 * n_agents
    )

    specialized = (
        mi_norm >= PHASE_THRESHOLDS["specialization_mi_norm"]
        and winner_switch >= PHASE_THRESHOLDS["winner_switch"]
    )
    strong_specialized = (
        mi_norm >= PHASE_THRESHOLDS["strong_specialization_mi_norm"]
        and winner_switch >= PHASE_THRESHOLDS["winner_switch"]
    )
    return {
        "collapse_signal": int(collapse),
        "concentration_signal": int(concentrated),
        "polarization_signal": int(polarized),
        "specialization_signal": int(specialized),
        "strong_specialization_signal": int(strong_specialized),
    }


def trajectory_row(table: PreparedLog, p: np.ndarray, timestep: int, step_index: int) -> Dict[str, Any]:
    row: Dict[str, Any] = {"timestep": int(timestep), "step_index": int(step_index)}
    row.update(concentration_descriptors(p))
    row.update(weighted_trait_descriptors(table, p))
    row.update(risk_quality_descriptors(table, p))
    return row


def agent_exposure_rows(table: PreparedLog, p: np.ndarray, timestep: int, step_index: int) -> List[Dict[str, Any]]:
    """Return agent-level exposure rows at a replay timestep."""
    rows: List[Dict[str, Any]] = []
    p = np.asarray(p, dtype=np.float64)
    for idx, aid in enumerate(table.agent_ids):
        row: Dict[str, Any] = {
            "agent_id": aid,
            "agent_index": idx,
            "timestep": int(timestep),
            "step_index": int(step_index),
            "exposure": float(p[idx]),
        }
        for dim in table.trait_coords.columns:
            row[f"trait_{dim}"] = float(table.trait_coords.iloc[idx][dim])
        rows.append(row)
    return rows


def early_warning_features_for_group(group: pd.DataFrame, early_frac: float) -> Dict[str, float]:
    group = group.sort_values("step_index")
    if group.empty:
        return {}
    max_step = int(group["step_index"].max())
    cutoff = max(2, int(math.ceil(max_step * float(early_frac))))
    early = group[group["step_index"] <= cutoff]
    out: Dict[str, float] = {"early_frac": float(early_frac), "early_cutoff_step": float(cutoff), "early_n_steps": float(len(early))}
    cols = [
        "hhi", "n_eff", "top1_share", "entropy_norm", "abs_polarization",
        "polarization_dispersion", "stance_extremity", "weighted_risk", "weighted_quality",
    ]
    for col in cols:
        if col not in early.columns:
            continue
        values = pd.to_numeric(early[col], errors="coerce").to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        out[f"{col}_early_mean"] = float(np.mean(values))
        out[f"{col}_early_last"] = float(values[-1])
        out[f"{col}_early_var"] = float(np.var(values))
        out[f"{col}_early_slope"] = linear_slope(values)
        out[f"{col}_early_lag1"] = lag1_autocorr(values)
    if "top1_share" in early.columns:
        values = pd.to_numeric(early["top1_share"], errors="coerce").to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size >= 4:
            out["top1_share_early_accel"] = linear_slope(np.diff(values))
    return out


def build_early_warning_table(trajectories: pd.DataFrame, summary: pd.DataFrame, early_frac: float) -> pd.DataFrame:
    if trajectories.empty or summary.empty:
        return pd.DataFrame()
    id_cols = [
        "run_name", "eta", "alpha", "beta_sup", "beta_route", "lambda_risk",
        "hard_safety_threshold", "context_trait_strength", "polarization_strength",
        "route_power", "route_zscore", "no_feedback", "ablation",
    ]
    id_cols = [c for c in id_cols if c in trajectories.columns and c in summary.columns]
    rows: List[Dict[str, Any]] = []
    for keys, group in trajectories.groupby(id_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: val for col, val in zip(id_cols, keys)}
        row.update(early_warning_features_for_group(group, early_frac=early_frac))
        rows.append(row)
    feat = pd.DataFrame(rows)
    label_cols = [
        c for c in id_cols + [
            "phase_label", "phase_code", "specialization_signal", "strong_specialization_signal",
            "collapse_signal", "concentration_signal", "polarization_signal",
            "top1_share", "hhi", "n_eff", "abs_polarization", "polarization_dispersion",
            "stance_extremity", "i_exp_norm", "winner_switch_rate", "kappa_map", "gamma_map",
            "max_invasion_log",
        ] if c in summary.columns
    ]
    feat = feat.merge(summary[label_cols], on=id_cols, how="left", suffixes=("", "_final"))
    feat["label_collapse"] = feat.get("phase_label", "").astype(str).str.contains("collapse").astype(int)
    if "kappa_map" in feat.columns:
        kappa = pd.to_numeric(feat["kappa_map"], errors="coerce")
    else:
        # Early-warning tables are also useful for endpoint phase labels when
        # a fast phase-only replay omits the expensive stability calculation.
        kappa = pd.Series(np.nan, index=feat.index, dtype=float)
    feat["label_unstable"] = (kappa > 0).astype(int)
    feat["label_stable_specialization"] = (
        (pd.to_numeric(feat.get("specialization_signal", 0), errors="coerce").fillna(0) > 0)
        & (kappa < 0)
    ).astype(int)
    feat["label_polarized"] = feat.get("phase_label", "").astype(str).str.contains("polar").astype(int)
    feat["label_specialized"] = feat.get("phase_label", "").astype(str).str.contains("special").astype(int)
    for flag_col in ["specialization_signal", "strong_specialization_signal", "collapse_signal", "concentration_signal", "polarization_signal"]:
        if flag_col in feat.columns:
            feat[f"label_{flag_col}"] = pd.to_numeric(feat[flag_col], errors="coerce").fillna(0).astype(int)
    return feat


def _roc_auc_manual(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # Average tied ranks.
    uniq, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        for k, cnt in enumerate(counts):
            if cnt > 1:
                idx = np.where(inv == k)[0]
                ranks[idx] = ranks[idx].mean()
    rank_sum_pos = float(ranks[y == 1].sum())
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _average_precision_manual(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    mask = np.isfinite(s)
    y, s = y[mask], s[mask]
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    precision = tp / np.maximum(tp + fp, 1)
    return float((precision * (y == 1)).sum() / n_pos)


def _brier(y_true: np.ndarray, prob: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(prob, dtype=float)
    mask = np.isfinite(p)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean((p[mask] - y[mask]) ** 2))


def _feature_columns_for_model(df: pd.DataFrame, model_name: str) -> List[str]:
    param_cols = [
        "eta", "alpha", "beta_sup", "beta_route", "lambda_risk", "hard_safety_threshold",
        "context_trait_strength", "polarization_strength", "route_power", "route_zscore", "no_feedback",
    ]
    concentration_prefixes = ("hhi_", "n_eff_", "top1_share_", "entropy_norm_")
    descriptor_cols = [c for c in df.columns if c.endswith("_early_mean") or c.endswith("_early_last") or c.endswith("_early_slope")]
    fluctuation_cols = [c for c in df.columns if c.endswith("_early_var") or c.endswith("_early_lag1") or c.endswith("_early_accel")]
    concentration_cols = [c for c in descriptor_cols + fluctuation_cols if c.startswith(concentration_prefixes)]
    non_concentration_cols = [c for c in descriptor_cols + fluctuation_cols if not c.startswith(concentration_prefixes)]
    if model_name == "parameter_only":
        return [c for c in param_cols if c in df.columns]
    if model_name == "early_descriptor":
        return [c for c in descriptor_cols if c in df.columns]
    if model_name == "concentration_autoregressive":
        return [c for c in concentration_cols if c in df.columns]
    if model_name == "non_concentration_early":
        return [c for c in non_concentration_cols if c in df.columns]
    if model_name == "fluctuation_only":
        return [c for c in fluctuation_cols if c in df.columns]
    if model_name == "full_early_warning":
        return [c for c in param_cols + descriptor_cols + fluctuation_cols if c in df.columns]
    if model_name == "leakage_guard_full":
        return [c for c in param_cols + non_concentration_cols if c in df.columns]
    return []


def _direct_baseline_score(df: pd.DataFrame, model_name: str) -> np.ndarray | None:
    score_col = {
        "last_hhi": "hhi_early_last",
        "last_top1": "top1_share_early_last",
        "last_entropy_inverse": "entropy_norm_early_last",
        "hhi_slope": "hhi_early_slope",
        "top1_slope": "top1_share_early_slope",
    }.get(model_name)
    if score_col is None or score_col not in df.columns:
        return None
    s = pd.to_numeric(df[score_col], errors="coerce").to_numpy(dtype=float)
    if model_name == "last_entropy_inverse":
        s = -s
    finite = np.isfinite(s)
    if finite.sum() == 0:
        return np.full(len(df), np.nan)
    lo, hi = float(np.nanmin(s[finite])), float(np.nanmax(s[finite]))
    if hi <= lo:
        out = np.full(len(df), 0.5, dtype=float)
    else:
        out = (s - lo) / (hi - lo)
    out[~np.isfinite(out)] = float(np.nanmean(out[finite])) if finite.any() else 0.5
    return np.clip(out, 0.0, 1.0)


def _split_indices(df: pd.DataFrame, split: str, group_col: str = "run_name") -> List[Tuple[np.ndarray, np.ndarray]]:
    n = len(df)
    if n == 0:
        return []
    if split == "within_grid_cv":
        k = min(5, n)
        return [(np.setdiff1d(np.arange(n), np.arange(fold, n, k)), np.arange(fold, n, k)) for fold in range(k)]
    if split == "leave_seed_out":
        for candidate in ["source_result_dir", "seed", group_col]:
            if candidate in df.columns:
                groups = df[candidate].astype(str).to_numpy()
                unique_groups = np.unique(groups)
                if unique_groups.size >= 2 and unique_groups.size < n:
                    return [(np.where(groups != g)[0], np.where(groups == g)[0]) for g in unique_groups]
    if split == "eta_transfer" and "eta" in df.columns:
        eta = pd.to_numeric(df["eta"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(eta)
        if finite.sum() >= 4:
            med = float(np.nanmedian(eta[finite]))
            low = np.where(eta <= med)[0]
            high = np.where(eta > med)[0]
            if low.size and high.size:
                return [(low, high), (high, low)]
    if split == "alpha_transfer" and "alpha" in df.columns:
        alpha = pd.to_numeric(df["alpha"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(alpha)
        if finite.sum() >= 4:
            med = float(np.nanmedian(alpha[finite]))
            low = np.where(alpha <= med)[0]
            high = np.where(alpha > med)[0]
            if low.size and high.size:
                return [(low, high), (high, low)]
    if split != "within_grid_cv":
        return []
    return []


def _cross_validated_predictions(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    group_col: str = "run_name",
    split: str = "within_grid_cv",
    y_override: np.ndarray | None = None,
) -> np.ndarray:
    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int).to_numpy() if y_override is None else np.asarray(y_override, dtype=int)
    if len(np.unique(y)) < 2 or not feature_cols:
        return np.full(len(df), float(np.mean(y)) if len(y) else np.nan)
    Xdf = df[feature_cols].copy()
    for c in Xdf.columns:
        Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")
        med = Xdf[c].median()
        if not np.isfinite(med):
            med = 0.0
        Xdf[c] = Xdf[c].fillna(med)
    X = Xdf.to_numpy(dtype=float)
    preds = np.full(len(df), np.nan, dtype=float)
    groups = df[group_col].astype(str).to_numpy() if group_col in df.columns else np.arange(len(df))
    unique_groups = np.unique(groups)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        splitter = _split_indices(df, split=split, group_col=group_col)
        for train, test in splitter:
            if train.size == 0 or test.size == 0 or len(np.unique(y[train])) < 2:
                preds[test] = float(np.mean(y[train])) if train.size else float(np.mean(y))
                continue
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
            )
            model.fit(X[train], y[train])
            preds[test] = model.predict_proba(X[test])[:, 1]
    except Exception:
        # Dependency-free fallback: normalized additive score with orientation set
        # by univariate correlation on the training fold.
        for train, test in _split_indices(df, split=split, group_col=group_col):
            mu = X[train].mean(axis=0)
            sd = X[train].std(axis=0) + 1e-9
            Ztr = (X[train] - mu) / sd
            Zte = (X[test] - mu) / sd
            corrs = np.array([np.corrcoef(Ztr[:, j], y[train])[0, 1] if np.std(Ztr[:, j]) > 1e-9 else 0.0 for j in range(Ztr.shape[1])])
            corrs = np.nan_to_num(corrs)
            raw = Zte @ corrs
            preds[test] = 1.0 / (1.0 + np.exp(-raw))
    mask = ~np.isfinite(preds)
    if mask.any():
        preds[mask] = float(np.mean(y))
    return np.clip(preds, 0.0, 1.0)


def evaluate_early_warning_tables(ew_tables: Mapping[float, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    eval_rows: List[Dict[str, Any]] = []
    pred_rows: List[Dict[str, Any]] = []
    target_specs = [
        ("label_collapse", "concentration_defined"),
        ("label_unstable", "criticality_independent"),
        ("label_stable_specialization", "persistence_independent"),
        ("label_polarized", "polarization_independent"),
    ]
    models = [
        "parameter_only", "last_hhi", "last_top1", "last_entropy_inverse", "hhi_slope", "top1_slope",
        "concentration_autoregressive", "early_descriptor", "non_concentration_early",
        "fluctuation_only", "leakage_guard_full", "full_early_warning",
    ]
    splits = ["within_grid_cv", "leave_seed_out", "eta_transfer", "alpha_transfer"]
    rng = np.random.default_rng(12345)
    for early_frac, df in ew_tables.items():
        if df.empty:
            continue
        for target, target_family in target_specs:
            if target not in df.columns:
                continue
            y = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int).to_numpy()
            if len(np.unique(y)) < 2:
                eval_rows.append({
                    "early_frac": early_frac, "target": target, "target_family": target_family, "model": "all", "split": "all", "control": "none", "n": len(df),
                    "positive_rate": float(np.mean(y)) if len(y) else np.nan,
                    "roc_auc": np.nan, "pr_auc": np.nan, "brier": np.nan,
                    "note": "target has only one class",
                })
                continue
            for model_name in models:
                direct = _direct_baseline_score(df, model_name)
                fcols = [] if direct is not None else _feature_columns_for_model(df, model_name)
                for split in splits:
                    if direct is not None and split != "within_grid_cv":
                        continue
                    splitter = _split_indices(df, split=split)
                    if not splitter:
                        continue
                    if any(len(np.unique(y[test])) < 2 for _, test in splitter if test.size):
                        continue
                    preds = direct if direct is not None else _cross_validated_predictions(df, target, fcols, split=split)
                    eval_rows.append({
                        "early_frac": early_frac,
                        "target": target,
                        "target_family": target_family,
                        "model": model_name,
                        "split": split,
                        "control": "none",
                        "n": int(len(df)),
                        "n_features": int(len(fcols) if direct is None else 1),
                        "positive_rate": float(np.mean(y)),
                        "roc_auc": _roc_auc_manual(y, preds),
                        "pr_auc": _average_precision_manual(y, preds),
                        "brier": _brier(y, preds),
                        "features": ";".join(fcols[:50]) if direct is None else model_name,
                    })
                    keys = [c for c in ["run_name", "source_result_dir", "seed", "eta", "alpha", "beta_sup", "beta_route", "lambda_risk", "context_trait_strength", "polarization_strength", "ablation"] if c in df.columns]
                    for i, base in df[keys].reset_index(drop=True).iterrows():
                        row = base.to_dict()
                        row.update({"early_frac": early_frac, "target": target, "target_family": target_family, "model": model_name, "split": split, "control": "none", "y_true": int(y[i]), "y_score": float(preds[i])})
                        pred_rows.append(row)
                if fcols:
                    y_shuf = rng.permutation(y)
                    preds = _cross_validated_predictions(df, target, fcols, split="within_grid_cv", y_override=y_shuf)
                    eval_rows.append({
                        "early_frac": early_frac, "target": target, "target_family": target_family,
                        "model": model_name, "split": "within_grid_cv", "control": "shuffled_label",
                        "n": int(len(df)), "n_features": int(len(fcols)), "positive_rate": float(np.mean(y_shuf)),
                        "roc_auc": _roc_auc_manual(y_shuf, preds), "pr_auc": _average_precision_manual(y_shuf, preds),
                        "brier": _brier(y_shuf, preds), "features": ";".join(fcols[:50]),
                    })
                    shuffled = df.copy()
                    shuffled[fcols] = shuffled[fcols].sample(frac=1.0, replace=False, random_state=12345).reset_index(drop=True)
                    preds = _cross_validated_predictions(shuffled, target, fcols, split="within_grid_cv")
                    eval_rows.append({
                        "early_frac": early_frac, "target": target, "target_family": target_family,
                        "model": model_name, "split": "within_grid_cv", "control": "shuffled_features",
                        "n": int(len(df)), "n_features": int(len(fcols)), "positive_rate": float(np.mean(y)),
                        "roc_auc": _roc_auc_manual(y, preds), "pr_auc": _average_precision_manual(y, preds),
                        "brier": _brier(y, preds), "features": ";".join(fcols[:50]),
                    })
    return pd.DataFrame(eval_rows), pd.DataFrame(pred_rows)


def build_phase_validation_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    specs = [
        ("coexistence", lambda d: (pd.to_numeric(d.get("concentration_signal", 1), errors="coerce").fillna(1) == 0)),
        ("collapse", lambda d: (pd.to_numeric(d.get("collapse_signal", 0), errors="coerce").fillna(0) == 1)),
        ("specialization", lambda d: (pd.to_numeric(d.get("specialization_signal", 0), errors="coerce").fillna(0) == 1)),
        ("polarization", lambda d: (pd.to_numeric(d.get("polarization_signal", 0), errors="coerce").fillna(0) == 1)),
        ("stable_specialization", lambda d: (pd.to_numeric(d.get("specialization_signal", 0), errors="coerce").fillna(0) == 1) & (pd.to_numeric(d.get("kappa_map", np.nan), errors="coerce") < 0)),
        ("collapse_adjacent_specialization", lambda d: (pd.to_numeric(d.get("specialization_signal", 0), errors="coerce").fillna(0) == 1) & (pd.to_numeric(d.get("kappa_map", np.nan), errors="coerce") >= 0)),
    ]
    for name, fn in specs:
        mask = fn(summary)
        sub = summary[mask].copy()
        if "kappa_map" in sub.columns:
            kappa = pd.to_numeric(sub["kappa_map"], errors="coerce")
        else:
            # Phase-only replays intentionally omit stability estimation.
            # Keep the validation table well-formed instead of treating the
            # missing column as a scalar boolean.
            kappa = pd.Series(np.nan, index=sub.index, dtype=float)
        stable = kappa < 0
        row = {
            "phase_family": name,
            "n_conditions": int(mask.sum()),
            "n_locally_contracting": int(stable.sum()) if len(sub) else 0,
            "median_kappa_map": float(kappa.median()) if len(sub) else np.nan,
            "median_top1_share": float(pd.to_numeric(sub.get("top1_share", np.nan), errors="coerce").median()) if len(sub) else np.nan,
            "median_i_exp_norm": float(pd.to_numeric(sub.get("i_exp_norm", np.nan), errors="coerce").median()) if len(sub) else np.nan,
            "median_polarization_dispersion": float(pd.to_numeric(sub.get("polarization_dispersion", np.nan), errors="coerce").median()) if len(sub) else np.nan,
        }
        if len(sub):
            if "kappa_map" in sub.columns and pd.to_numeric(sub["kappa_map"], errors="coerce").notna().any():
                best = sub.iloc[pd.to_numeric(sub["kappa_map"], errors="coerce").fillna(np.inf).argmin()]
            else:
                best = sub.iloc[0]
            for c in ["eta", "alpha", "beta_sup", "beta_route", "lambda_risk", "context_trait_strength", "polarization_strength", "phase_label"]:
                if c in best.index:
                    row[f"example_{c}"] = best[c]
        rows.append(row)
    return pd.DataFrame(rows)



def build_param_grid(args: argparse.Namespace) -> List[ReplayParams]:
    etas = parse_float_list(args.etas) or [0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2]
    alphas = parse_float_list(args.alphas) or [0.0, 0.25, 0.5, 0.75, 1.0]
    support_strengths = parse_float_list(args.support_strengths) or [0.0, 0.03]
    route_strengths = parse_float_list(args.route_strengths) or [0.0, 1.0]
    risk_lambdas = parse_float_list(args.risk_lambdas) or [0.0, 0.5]
    hard_thresholds = parse_float_list(args.hard_safety_thresholds) or [-1.0]
    ctx_strengths = parse_float_list(args.context_trait_strengths) if getattr(args, "context_trait_strengths", "") else [float(args.context_trait_strength)]
    polar_strengths = parse_float_list(args.polarization_strengths) if getattr(args, "polarization_strengths", "") else [float(getattr(args, "polarization_strength", 0.0))]
    route_powers = parse_float_list(args.route_powers) if getattr(args, "route_powers", "") else [float(args.route_power)]

    params: List[ReplayParams] = []
    for eta, alpha, beta_sup, beta_route, lam, hard, ctx_s, pol_s, rpow in itertools.product(
        etas, alphas, support_strengths, route_strengths, risk_lambdas, hard_thresholds, ctx_strengths, polar_strengths, route_powers
    ):
        params.append(ReplayParams(
            float(eta), float(alpha), float(beta_sup), float(beta_route), float(lam), float(hard),
            context_trait_strength=float(ctx_s),
            route_power=float(rpow),
            route_zscore=bool(args.route_zscore),
            polarization_strength=float(pol_s),
        ))



    seen = set()
    unique: List[ReplayParams] = []
    for p in params:
        key = tuple(asdict(p).items())
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def update_phase_thresholds_from_args(args: argparse.Namespace) -> None:
    PHASE_THRESHOLDS["collapse_top1"] = float(args.collapse_top1_threshold)
    PHASE_THRESHOLDS["collapse_neff_frac"] = float(args.collapse_neff_frac_threshold)
    PHASE_THRESHOLDS["concentrated_top1"] = float(args.concentrated_top1_threshold)
    PHASE_THRESHOLDS["concentrated_neff_frac"] = float(args.concentrated_neff_frac_threshold)
    PHASE_THRESHOLDS["polarization_abs"] = float(args.polarization_threshold)
    PHASE_THRESHOLDS["polarization_var"] = float(args.polarization_var_threshold)
    PHASE_THRESHOLDS["stance_extremity"] = float(args.stance_extremity_threshold)
    PHASE_THRESHOLDS["specialization_mi_norm"] = float(args.specialization_mi_threshold)
    PHASE_THRESHOLDS["strong_specialization_mi_norm"] = float(args.strong_specialization_mi_threshold)
    PHASE_THRESHOLDS["winner_switch"] = float(args.winner_switch_threshold)
    PHASE_THRESHOLDS["near_critical_eps"] = float(args.near_critical_eps)


def attach_common_fields(row: Dict[str, Any], run_name: str, params: ReplayParams) -> Dict[str, Any]:
    out = {
        "run_name": run_name,
        "eta": params.eta,
        "alpha": params.alpha,
        "beta_sup": params.beta_sup,
        "beta_route": params.beta_route,
        "lambda_risk": params.lambda_risk,
        "hard_safety_threshold": params.hard_safety_threshold,
        "context_trait_strength": params.context_trait_strength,
        "polarization_strength": params.polarization_strength,
        "route_power": params.route_power,
        "route_zscore": params.route_zscore,
        "no_feedback": params.no_feedback,
        "ablation": params.ablation,
    }
    out.update(row)
    return out


def run_experiments(args: argparse.Namespace) -> None:
    start = time.time()
    runs_dir = Path(args.runs_dir)
    out_dir = mkdir(Path(args.out_dir))
    update_phase_thresholds_from_args(args)
    score_weights = parse_score_weights(args.score_weights)
    params_grid = build_param_grid(args)
    sources = discover_log_sources(runs_dir, max_runs=int(args.max_runs))

    if args.dry_run:
        print(f"sources={len(sources)}")
        for s in sources:
            print(f"  {s}")
        print(f"conditions={len(params_grid)}")
        return

    write_json(out_dir / "run_config.json", {
        "args": vars(args),
        "score_weights": score_weights,
        "n_param_conditions": len(params_grid),
        "sources": [str(s) for s in sources],
        "phase_codes": PHASE_CODES,
        "phase_thresholds": PHASE_THRESHOLDS,
        "note": "Phase labels are operational endpoint summaries; stable phases require local contraction / negative invasion diagnostics.",
    })

    all_summary_rows: List[Dict[str, Any]] = []
    all_endpoint_rows: List[Dict[str, Any]] = []
    all_snapshot_rows: List[Dict[str, Any]] = []
    all_traj_rows: List[Dict[str, Any]] = []
    stability_rows: List[Dict[str, Any]] = []
    snapshot_steps = set(parse_int_list(args.snapshot_steps)) if bool(getattr(args, "save_exposure_snapshots", False)) else set()

    root = runs_dir if runs_dir.is_dir() else runs_dir.parent
    for source_idx, source in enumerate(sources, start=1):
        run_name = run_name_from_path(source, root)
        print(f"[{source_idx}/{len(sources)}] preparing {source} as run_name={run_name}", flush=True)
        table = prepare_log_streaming(source, run_name=run_name, score_weights=score_weights, max_timesteps=int(args.max_timesteps))
        approx_mb = (table.base_score.nbytes + table.risk_penalty.nbytes + table.safety_score.nbytes + table.quality_score.nbytes) / (1024 ** 2)
        print(f"  agents={table.n_agents}, timesteps={table.n_timesteps}, conditions={len(params_grid)}, core_arrays≈{approx_mb:.1f} MiB", flush=True)

        for cond_idx, params in enumerate(params_grid, start=1):
            if cond_idx == 1 or cond_idx % max(1, int(args.progress_every)) == 0:
                print(f"  condition {cond_idx}/{len(params_grid)}: {params.as_prefix()}", flush=True)
            endpoint, desc, traj_rows, endpoint_rows, snapshot_rows = replay_one_condition(
                table,
                params,
                last_k=int(args.endpoint_last_k),
                context_key=str(args.context_key),
                support_clip=float(args.support_clip),
                collect_trajectory=not bool(args.no_trajectories),
                trajectory_stride=int(args.trajectory_stride),
                snapshot_steps=snapshot_steps,
            )

            summary_row: Dict[str, Any] = {"source_path": str(source), "n_agents": table.n_agents, "n_timesteps": table.n_timesteps}
            summary_row.update(desc)
            if args.estimate_stability:
                try:
                    stab = estimate_map_stability(
                        table,
                        endpoint,
                        params,
                        active_eps=float(args.active_eps),
                        fd_eps=float(args.fd_eps),
                        invasion_eps=float(args.invasion_eps),
                        support_clip=float(args.support_clip),
                        context_stride=int(args.stability_context_stride),
                        max_active=int(args.stability_max_active),
                    )
                    stab["criticality_label"] = criticality_label(float(stab.get("kappa_map", float("nan"))))
                    summary_row.update(stab)
                    stability_payload = {k: summary_row.get(k) for k in [
                        "phase_label", "phase_code", "specialization_signal", "strong_specialization_signal",
                        "collapse_signal", "concentration_signal", "polarization_signal",
                        "hhi", "n_eff", "top1_share", "abs_polarization", "polarization_dispersion",
                        "stance_extremity", "i_exp_norm", "winner_switch_rate", "weighted_risk", "weighted_quality",
                    ] if k in summary_row}
                    stability_payload.update(stab)
                    stability_rows.append(attach_common_fields(stability_payload, run_name, params))
                except Exception as exc:
                    err = {"stability_error": repr(exc)}
                    summary_row.update(err)
                    stability_rows.append(attach_common_fields(err, run_name, params))

            all_summary_rows.append(attach_common_fields(summary_row, run_name, params))
            if args.save_endpoint_exposures:
                for row in endpoint_rows:
                    all_endpoint_rows.append(attach_common_fields(row, run_name, params))
            if bool(getattr(args, "save_exposure_snapshots", False)):
                for row in snapshot_rows:
                    all_snapshot_rows.append(attach_common_fields(row, run_name, params))
            if not args.no_trajectories:
                for row in traj_rows:
                    all_traj_rows.append(attach_common_fields(row, run_name, params))

    summary = pd.DataFrame(all_summary_rows)
    summary_path = out_dir / "phase_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path} ({len(summary)} rows)", flush=True)

    validation = build_phase_validation_table(summary)
    validation_path = out_dir / "phase_validation_table.csv"
    validation.to_csv(validation_path, index=False)
    print(f"wrote {validation_path} ({len(validation)} rows)", flush=True)

    if all_endpoint_rows:
        endpoint_path = out_dir / "endpoint_exposures.csv.gz"
        pd.DataFrame(all_endpoint_rows).to_csv(endpoint_path, index=False, compression="gzip")
        print(f"wrote {endpoint_path} ({len(all_endpoint_rows)} rows)", flush=True)

    if all_snapshot_rows:
        snapshot_path = out_dir / "exposure_snapshots.csv.gz"
        pd.DataFrame(all_snapshot_rows).to_csv(snapshot_path, index=False, compression="gzip")
        print(f"wrote {snapshot_path} ({len(all_snapshot_rows)} rows)", flush=True)

    traj = pd.DataFrame(all_traj_rows)
    if not traj.empty:
        traj_path = out_dir / "trajectories.csv.gz"
        traj.to_csv(traj_path, index=False, compression="gzip")
        print(f"wrote {traj_path} ({len(traj)} rows)", flush=True)
        ew_tables: Dict[float, pd.DataFrame] = {}
        for frac in parse_float_list(args.early_fracs) or [float(args.early_frac)]:
            ew_tables[float(frac)] = build_early_warning_table(traj, summary, early_frac=float(frac))
        ew = pd.concat(ew_tables.values(), ignore_index=True) if ew_tables else pd.DataFrame()
        ew_path = out_dir / "early_warning_features.csv"
        ew.to_csv(ew_path, index=False)
        print(f"wrote {ew_path} ({len(ew)} rows)", flush=True)
        if bool(args.evaluate_early_warning):
            eval_df, pred_df = evaluate_early_warning_tables(ew_tables)
            eval_path = out_dir / "early_warning_evaluation.csv"
            pred_path = out_dir / "early_warning_predictions.csv.gz"
            eval_df.to_csv(eval_path, index=False)
            pred_df.to_csv(pred_path, index=False, compression="gzip")
            print(f"wrote {eval_path} ({len(eval_df)} rows)", flush=True)
            print(f"wrote {pred_path} ({len(pred_df)} rows)", flush=True)

    if stability_rows:
        stab_path = out_dir / "stability_summary.csv"
        pd.DataFrame(stability_rows).to_csv(stab_path, index=False)
        print(f"wrote {stab_path} ({len(stability_rows)} rows)", flush=True)

        print(f"wrote figures under {out_dir / 'figures'}", flush=True)

    elapsed = time.time() - start
    write_json(out_dir / "done.json", {"elapsed_sec": elapsed, "summary_rows": len(summary)})
    print(f"done in {elapsed:.1f}s", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Replay of the stored logs under the platform's exposure update, with the local-stability audit and the early-warning features.")
    p.add_argument("--runs_dir", default="runs", help="Directory containing checkpoint dirs.")
    p.add_argument("--out_dir", default="results/phase_experiments", help="Output directory.")
    p.add_argument("--etas", default="0,0.05,0.1,0.2,0.4,0.8,1.2")
    p.add_argument("--alphas", default="0,0.25,0.5,0.75,1")
    p.add_argument("--support_strengths", default="0,0.03")
    p.add_argument("--route_strengths", default="0,1")
    p.add_argument("--risk_lambdas", default="0,0.5")
    p.add_argument("--hard_safety_thresholds", default="-1")
    p.add_argument("--score_weights", default="", help="JSON object over like/reply/share/follow/eval_quality/eval_safety/audit_quality/audit_safety.")
    p.add_argument("--context_key", default="topic_category")
    p.add_argument("--context_trait_strength", type=float, default=0.0, help="Single context×trait strength; overridden by --context_trait_strengths.")
    p.add_argument("--context_trait_strengths", default="", help="Comma-separated context×trait strengths for phase-calibration sweeps.")
    p.add_argument("--polarization_strength", type=float, default=0.0, help="Single stance-axis polarization strength; overridden by --polarization_strengths.")
    p.add_argument("--polarization_strengths", default="", help="Comma-separated stance-axis polarization strengths.")
    p.add_argument("--route_power", type=float, default=1.0, help="Signed power transform for routing residuals.")
    p.add_argument("--route_powers", default="", help="Comma-separated route powers.")
    p.add_argument("--route_zscore", action="store_true", help="Z-score routing residuals within each context before applying beta_route.")
    p.add_argument("--collapse_top1_threshold", type=float, default=0.80)
    p.add_argument("--collapse_neff_frac_threshold", type=float, default=0.025)
    p.add_argument("--concentrated_top1_threshold", type=float, default=0.45)
    p.add_argument("--concentrated_neff_frac_threshold", type=float, default=0.10)
    p.add_argument("--polarization_threshold", type=float, default=0.35)
    p.add_argument("--polarization_var_threshold", type=float, default=0.50)
    p.add_argument("--stance_extremity_threshold", type=float, default=0.70)
    p.add_argument("--specialization_mi_threshold", type=float, default=0.12)
    p.add_argument("--strong_specialization_mi_threshold", type=float, default=0.15)
    p.add_argument("--winner_switch_threshold", type=float, default=0.30)
    p.add_argument("--near_critical_eps", type=float, default=0.05, help="Absolute kappa_map band used to label near-critical stability rows.")
    p.add_argument("--endpoint_last_k", type=int, default=50)
    p.add_argument("--support_clip", type=float, default=8.0)
    p.add_argument("--estimate_stability", action="store_true")
    p.add_argument("--active_eps", type=float, default=1e-4)
    p.add_argument("--fd_eps", type=float, default=1e-5)
    p.add_argument("--invasion_eps", type=float, default=1e-5)
    p.add_argument("--stability_context_stride", type=int, default=1)
    p.add_argument("--stability_max_active", type=int, default=160)
    p.add_argument("--early_frac", type=float, default=0.4)
    p.add_argument("--early_fracs", default="0.1,0.2,0.3,0.4", help="Comma-separated early trajectory fractions for forecasting evaluation.")
    p.add_argument("--evaluate_early_warning", action="store_true", help="Fit cross-validated early-warning classifiers and write AUC/PR-AUC/Brier tables.")
    p.add_argument("--save_endpoint_exposures", action="store_true")
    p.add_argument("--save_exposure_snapshots", action="store_true", help="Save agent-level exposure vectors at selected replay timesteps.")
    p.add_argument("--snapshot_steps", default="", help="Comma-separated step indices to save, e.g. 0,10,25,50,100. Includes final state if equal to number of timesteps.")
    p.add_argument("--no_trajectories", action="store_true")
    p.add_argument("--trajectory_stride", type=int, default=1, help="Save every kth trajectory row.")
    p.add_argument("--progress_every", type=int, default=25)
    p.add_argument("--max_runs", type=int, default=0, help="Limit number of discovered runs; 0 means all.")
    p.add_argument("--max_timesteps", type=int, default=0, help="Limit timesteps per run; 0 means all.")
    p.add_argument("--dry_run", action="store_true", help="Only print discovered sources and condition count.")
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_experiments(args)


if __name__ == "__main__":
    main()
