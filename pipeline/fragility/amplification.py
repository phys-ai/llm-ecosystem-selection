#!/usr/bin/env python3
"""Cross-fitted local-amplification test at a fixed high selection strength.

For each descriptor-positive endpoint, the logged contexts are split into
folds.  The leading active-face tangent eigenspace is estimated from the
training contexts only.  A TV-normalized perturbation in that eigenspace is
then evaluated with the held-out empirical mean map:

    log TV(T_test(p'), T_test(p*)) / TV(p', p*).

The comparison fixes eta, pairs phantom and stable endpoints within the same
population seed, exactly matches every platform control except support
strength, and nearest-matches support strength and endpoint descriptors.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay import endpoint as replay
from common import perturbation as fixed


EXACT_KEYS_EXCEPT_SUPPORT = [
    "eta",
    "alpha",
    "beta_route",
    "lambda_risk",
    "hard_safety_threshold",
    "context_trait_strength",
    "polarization_strength",
    "route_power",
    "route_zscore",
    "no_feedback",
    "ablation",
]

MATCH_METRICS = [
    "top1_share",
    "n_eff",
    "i_exp_norm",
    "winner_switch_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-glob",
        default="data/results/fig5_full_grid/run_popseed_*.checkpoints/phase_summary.csv",
    )
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument(
        "--out-dir",
        default="data/results/crossfit_local_amplification_eta2p25",
    )
    parser.add_argument("--eta", type=float, default=2.25)
    parser.add_argument("--epsilon", type=float, default=1.0e-4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--endpoint-last-k", type=int, default=100)
    parser.add_argument("--active-eps", type=float, default=1.0e-4)
    parser.add_argument("--fd-eps", type=float, default=1.0e-5)
    parser.add_argument("--support-clip", type=float, default=8.0)
    parser.add_argument("--kappa-band", type=float, default=0.05)
    parser.add_argument(
        "--support-match-scale",
        type=float,
        default=0.05,
        help="One unit of support-distance cost.",
    )
    parser.add_argument(
        "--support-match-weight",
        type=float,
        default=4.0,
        help="Squared support-distance weight relative to descriptor distance.",
    )
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument(
        "--matching-mode",
        choices=["within-seed-near-support", "exact-controls-across-seed"],
        default="within-seed-near-support",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def seed_from_path(path: Path) -> int:
    match = re.search(r"run_popseed_(\d+)", str(path))
    if not match:
        raise ValueError(f"Could not infer population seed from {path}")
    return int(match.group(1))


def descriptor_positive_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        pd.to_numeric(frame["specialization_signal"], errors="coerce").fillna(0).eq(1)
        & pd.to_numeric(frame["collapse_signal"], errors="coerce").fillna(0).eq(0)
        & pd.to_numeric(frame["concentration_signal"], errors="coerce").fillna(0).eq(0)
        & pd.to_numeric(frame["top1_share"], errors="coerce").lt(0.45)
        & pd.to_numeric(frame["n_eff"], errors="coerce").gt(8.0)
        & pd.to_numeric(frame["i_exp_norm"], errors="coerce").gt(0.12)
        & pd.to_numeric(frame["winner_switch_rate"], errors="coerce").gt(0.30)
    )


def load_candidates(args: argparse.Namespace) -> pd.DataFrame:
    paths = sorted(Path().glob(args.summary_glob))
    if not paths:
        raise FileNotFoundError(f"No summaries matched {args.summary_glob!r}")
    frames: List[pd.DataFrame] = []
    for path in paths:
        population_seed = seed_from_path(path)
        frame = pd.read_csv(path)
        frame = frame.loc[
            descriptor_positive_mask(frame)
            & np.isclose(
                pd.to_numeric(frame["eta"], errors="coerce"),
                float(args.eta),
            )
        ].copy()
        kappa = pd.to_numeric(frame["kappa_map"], errors="coerce")
        gamma = pd.to_numeric(frame["gamma_map"], errors="coerce")
        frame["stability_status"] = "excluded"
        frame.loc[kappa < -float(args.kappa_band), "stability_status"] = "stable"
        # The experiment targets the active-face claim, so invasion-only
        # phantom endpoints are deliberately excluded.
        frame.loc[gamma > float(args.kappa_band), "stability_status"] = "phantom"
        frame = frame.loc[
            frame["stability_status"].isin(["phantom", "stable"])
        ].copy()
        frame["population_seed"] = int(population_seed)
        frame["summary_path"] = str(path.resolve())
        frame["raw_source"] = str(
            (Path(args.runs_dir) / f"run_popseed_{population_seed}.checkpoints").resolve()
        )
        frames.append(frame)
    candidates = pd.concat(frames, ignore_index=True)
    if candidates.empty:
        raise ValueError(f"No eligible endpoints at eta={args.eta:g}.")
    return candidates


def standardized_descriptor_values(candidates: pd.DataFrame) -> pd.DataFrame:
    values = pd.DataFrame(index=candidates.index)
    for metric in MATCH_METRICS:
        column = pd.to_numeric(candidates[metric], errors="coerce")
        if metric == "n_eff":
            column = np.log(np.clip(column, replay.EPS, None))
        scale = float(column.std(ddof=0))
        values[metric] = (
            column - float(column.mean())
        ) / (scale if scale > replay.EPS else 1.0)
    return values.fillna(0.0)


def select_pairs(candidates: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Match without replacement using the requested blocking variables."""
    zscores = standardized_descriptor_values(candidates)
    pair_rows: List[pd.Series] = []
    pair_number = 0
    if args.matching_mode == "within-seed-near-support":
        group_keys = ["population_seed", *EXACT_KEYS_EXCEPT_SUPPORT]
    else:
        group_keys = [*EXACT_KEYS_EXCEPT_SUPPORT, "beta_sup"]
    for _, group in candidates.groupby(group_keys, dropna=False, sort=True):
        stable = group.loc[group["stability_status"].eq("stable")]
        phantom = group.loc[group["stability_status"].eq("phantom")]
        if stable.empty or phantom.empty:
            continue
        cost = np.empty((len(stable), len(phantom)), dtype=np.float64)
        for stable_pos, stable_index in enumerate(stable.index):
            for phantom_pos, phantom_index in enumerate(phantom.index):
                support_delta = (
                    float(candidates.loc[phantom_index, "beta_sup"])
                    - float(candidates.loc[stable_index, "beta_sup"])
                ) / float(args.support_match_scale)
                descriptor_delta = (
                    zscores.loc[phantom_index].to_numpy(dtype=float)
                    - zscores.loc[stable_index].to_numpy(dtype=float)
                )
                descriptor_cost = float(
                    np.dot(descriptor_delta, descriptor_delta)
                )
                if args.matching_mode == "within-seed-near-support":
                    cost[stable_pos, phantom_pos] = (
                        float(args.support_match_weight)
                        * support_delta
                        * support_delta
                        + descriptor_cost
                    )
                else:
                    cost[stable_pos, phantom_pos] = descriptor_cost
        stable_positions, phantom_positions = linear_sum_assignment(cost)
        for stable_pos, phantom_pos in zip(stable_positions, phantom_positions):
            pair_number += 1
            pair_id = f"pair_{pair_number:03d}"
            stable_index = int(stable.index[int(stable_pos)])
            phantom_index = int(phantom.index[int(phantom_pos)])
            descriptor_distance = float(
                np.linalg.norm(
                    zscores.loc[phantom_index].to_numpy(dtype=float)
                    - zscores.loc[stable_index].to_numpy(dtype=float)
                )
            )
            support_difference = abs(
                float(candidates.loc[phantom_index, "beta_sup"])
                - float(candidates.loc[stable_index, "beta_sup"])
            )
            for endpoint_index in (stable_index, phantom_index):
                row = candidates.loc[endpoint_index].copy()
                row["pair_id"] = pair_id
                row["match_cost"] = float(cost[int(stable_pos), int(phantom_pos)])
                row["descriptor_match_distance"] = descriptor_distance
                row["support_abs_difference"] = support_difference
                pair_rows.append(row)
    if not pair_rows:
        raise ValueError("No within-seed fixed-eta matches were available.")
    pairs = pd.DataFrame(pair_rows)
    pair_summary = (
        pairs[["pair_id", "match_cost"]]
        .drop_duplicates()
        .sort_values(["match_cost", "pair_id"])
    )
    if int(args.max_pairs) > 0:
        pair_summary = pair_summary.head(int(args.max_pairs))
    order = {
        pair_id: index
        for index, pair_id in enumerate(pair_summary["pair_id"], start=1)
    }
    pairs = pairs.loc[pairs["pair_id"].isin(order)].copy()
    pairs["pair_order"] = pairs["pair_id"].map(order)
    pairs["pair_id"] = pairs["pair_order"].map(
        lambda value: f"pair_{int(value):03d}"
    )
    return pairs.sort_values(["pair_order", "stability_status"]).reset_index(drop=True)


def make_context_folds(
    table: replay.PreparedLog,
    folds: int,
    random_seed: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    categories = (
        table.context_df["topic_category"]
        .fillna("uncategorized")
        .astype(str)
        .to_numpy()
    )
    indices = np.arange(table.n_timesteps, dtype=int)
    splitter = StratifiedKFold(
        n_splits=int(folds),
        shuffle=True,
        random_state=int(random_seed),
    )
    return [
        (indices[train_index], indices[test_index])
        for train_index, test_index in splitter.split(indices, categories)
    ]


def build_mean_map(
    table: replay.PreparedLog,
    params: replay.ReplayParams,
    context_indices: Sequence[int],
    support_clip: float,
) -> Callable[[np.ndarray], np.ndarray]:
    if float(getattr(params, "polarization_strength", 0.0)) != 0.0:
        raise ValueError("This cross-fit implementation expects zero polarization strength.")
    indices = np.asarray(context_indices, dtype=int)
    if indices.size == 0:
        raise ValueError("A mean map requires at least one context.")
    target = np.ones(table.n_agents, dtype=np.float64) / float(table.n_agents)
    fixed_fitness = np.vstack(
        [
            replay.fitness_at(
                table,
                int(index),
                target,
                params,
                support_clip=float(support_clip),
            )
            for index in indices
        ]
    )
    safety_mask: np.ndarray | None = None
    if float(params.hard_safety_threshold) >= 0.0:
        safety_mask = (
            table.safety_score[indices, :]
            >= float(params.hard_safety_threshold)
        )
    eta = 0.0 if bool(params.no_feedback) else float(params.eta)

    def mean_map(state: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(state, dtype=np.float64), replay.EPS, None)
        p = p / float(p.sum())
        support = (
            float(params.beta_sup)
            * replay.support_term(p, target, float(support_clip))
        )
        logits = np.log(p)[None, :] + eta * (
            fixed_fitness + support[None, :]
        )
        if safety_mask is not None:
            logits = np.where(safety_mask, logits, -np.inf)
            empty = ~np.isfinite(logits).any(axis=1)
            if empty.any():
                logits[empty, :] = np.log(p)[None, :]
        row_max = np.max(logits, axis=1, keepdims=True)
        weights = np.exp(logits - row_max)
        weights = np.where(np.isfinite(weights), weights, 0.0)
        denominators = weights.sum(axis=1, keepdims=True)
        bad = denominators[:, 0] <= replay.EPS
        if bad.any():
            weights[bad, :] = p[None, :]
            denominators[bad, :] = 1.0
        output = (weights / denominators).mean(axis=0)
        return output / float(output.sum())

    return mean_map


def active_face_jacobian(
    mean_map: Callable[[np.ndarray], np.ndarray],
    endpoint: np.ndarray,
    active_eps: float,
    fd_eps: float,
) -> Tuple[np.ndarray, np.ndarray, int]:
    p_star = np.clip(np.asarray(endpoint, dtype=np.float64), replay.EPS, None)
    p_star = p_star / p_star.sum()
    active = np.flatnonzero(p_star > float(active_eps))
    if active.size < 2:
        raise ValueError("Leading tangent direction requires at least two active agents.")
    reference = int(active[-1])
    dimension = int(active.size) - 1
    delta = max(
        min(float(fd_eps), 0.25 * float(np.min(p_star[active]))),
        1.0e-10,
    )
    jacobian = np.zeros((dimension, dimension), dtype=np.float64)
    for column, agent_index in enumerate(active[:-1]):
        direction = np.zeros_like(p_star)
        direction[int(agent_index)] = 1.0
        direction[reference] = -1.0
        plus = mean_map(p_star + delta * direction)
        minus = mean_map(p_star - delta * direction)
        jacobian[:, column] = (
            (plus - minus) / (2.0 * delta)
        )[active[:-1]]
    return jacobian, active, reference


def leading_eigenspace_perturbation(
    train_map: Callable[[np.ndarray], np.ndarray],
    endpoint: np.ndarray,
    active_eps: float,
    fd_eps: float,
    epsilon: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    p_star = np.clip(np.asarray(endpoint, dtype=np.float64), replay.EPS, None)
    p_star = p_star / p_star.sum()
    jacobian, active, reference = active_face_jacobian(
        train_map,
        p_star,
        active_eps=float(active_eps),
        fd_eps=float(fd_eps),
    )
    eigenvalues, eigenvectors = np.linalg.eig(jacobian)
    lead_index = int(np.argmax(np.abs(eigenvalues)))
    lead_value = eigenvalues[lead_index]
    reduced = eigenvectors[:, lead_index]
    reduced_candidates: List[Tuple[str, np.ndarray]] = [
        ("real", np.real(reduced))
    ]
    if float(np.linalg.norm(np.imag(reduced))) > 1.0e-10:
        reduced_candidates.append(("imag", np.imag(reduced)))

    best: Tuple[float, np.ndarray, str] | None = None
    for component, coordinates in reduced_candidates:
        full_direction = np.zeros_like(p_star)
        full_direction[active[:-1]] = coordinates
        full_direction[reference] = -float(np.sum(coordinates))
        norm = float(np.sum(np.abs(full_direction)))
        if not np.isfinite(norm) or norm <= replay.EPS:
            continue
        full_direction = full_direction / norm
        for sign in (1.0, -1.0):
            perturbed, method = fixed.projected_direction_at_tv(
                p_star,
                sign * full_direction,
                float(epsilon),
            )
            initial_tv = fixed.tv_distance(perturbed, p_star)
            output_tv = fixed.tv_distance(
                train_map(perturbed),
                train_map(p_star),
            )
            log_gain = float(
                math.log(max(output_tv / max(initial_tv, replay.EPS), replay.EPS))
            )
            label = f"{component}:{int(sign):+d}:{method}"
            candidate = (log_gain, perturbed, label)
            if best is None or candidate[0] > best[0]:
                best = candidate
    if best is None:
        raise ValueError("Could not construct a feasible leading-eigenspace perturbation.")
    train_log_gain, perturbed, direction_label = best
    metadata = {
        "train_spectral_radius": float(abs(lead_value)),
        "train_gamma": float(math.log(max(abs(lead_value), replay.EPS))),
        "train_leading_eigenvalue_real": float(np.real(lead_value)),
        "train_leading_eigenvalue_imag": float(np.imag(lead_value)),
        "train_finite_log_gain": float(train_log_gain),
        "direction_component": direction_label,
        "active_support_size": int(active.size),
    }
    return perturbed, metadata


def run_endpoint(
    table: replay.PreparedLog,
    row: Mapping[str, Any],
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    params = replay._param_from_row(row)
    p_star, _ = replay.replay_endpoint_and_final_for_case(
        table,
        params,
        last_k=int(args.endpoint_last_k),
        context_key="topic_category",
        support_clip=float(args.support_clip),
    )
    descriptors = replay.concentration_descriptors(p_star)
    endpoint_row = {
        "pair_id": row["pair_id"],
        "pair_order": int(row["pair_order"]),
        "stability_status": row["stability_status"],
        "population_seed": int(row["population_seed"]),
        "eta": float(row["eta"]),
        "beta_sup": float(row["beta_sup"]),
        "kappa_map": float(row["kappa_map"]),
        "gamma_map": float(row["gamma_map"]),
        "match_cost": float(row["match_cost"]),
        "descriptor_match_distance": float(row["descriptor_match_distance"]),
        "support_abs_difference": float(row["support_abs_difference"]),
        **{f"replayed_{key}": value for key, value in descriptors.items()},
    }
    fold_rows: List[Dict[str, Any]] = []
    for fold_index, (train_indices, test_indices) in enumerate(folds):
        train_map = build_mean_map(
            table,
            params,
            train_indices,
            support_clip=float(args.support_clip),
        )
        test_map = build_mean_map(
            table,
            params,
            test_indices,
            support_clip=float(args.support_clip),
        )
        perturbed, tangent_metadata = leading_eigenspace_perturbation(
            train_map,
            p_star,
            active_eps=float(args.active_eps),
            fd_eps=float(args.fd_eps),
            epsilon=float(args.epsilon),
        )
        initial_tv = fixed.tv_distance(perturbed, p_star)
        test_output_tv = fixed.tv_distance(
            test_map(perturbed),
            test_map(p_star),
        )
        test_gain = test_output_tv / max(initial_tv, replay.EPS)
        fold_rows.append(
            {
                **endpoint_row,
                "fold": int(fold_index),
                "n_train_contexts": int(len(train_indices)),
                "n_test_contexts": int(len(test_indices)),
                "epsilon": float(args.epsilon),
                "initial_tv": float(initial_tv),
                "test_output_tv": float(test_output_tv),
                "test_amplification": float(test_gain),
                "test_log_amplification": float(
                    math.log(max(test_gain, replay.EPS))
                ),
                "test_amplified": int(test_gain > 1.0),
                "test_endpoint_residual_tv": float(
                    fixed.tv_distance(test_map(p_star), p_star)
                ),
                **tangent_metadata,
            }
        )
    return endpoint_row, fold_rows


def bootstrap_cluster_means(
    seed_means: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    values = np.asarray(seed_means, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        return float(values[0]), float(values[0])
    samples = rng.integers(
        0,
        values.size,
        size=(int(repetitions), values.size),
    )
    means = values[samples].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def small_cluster_sign_and_wild_inference(
    cluster_effects: np.ndarray,
) -> Dict[str, float | int]:
    """Exact sign-flip tests and exhaustive Rademacher wild bootstrap-t CI."""
    values = np.asarray(cluster_effects, dtype=np.float64)
    values = values[np.isfinite(values)]
    cluster_count = int(values.size)
    if cluster_count < 2:
        value = float(values[0]) if cluster_count == 1 else float("nan")
        return {
            "cluster_count": cluster_count,
            "cluster_weighted_mean": value,
            "cluster_standard_error": float("nan"),
            "positive_cluster_count": int(value > 0.0) if cluster_count else 0,
            "exact_sign_flip_p_one_sided": float("nan"),
            "exact_sign_flip_p_two_sided": float("nan"),
            "wild_cluster_bootstrap_t_ci_low": value,
            "wild_cluster_bootstrap_t_ci_high": value,
        }
    if cluster_count > 16:
        raise ValueError(
            "Exhaustive sign-flip inference is restricted to at most 16 clusters."
        )

    observed_mean = float(values.mean())
    observed_se = float(values.std(ddof=1) / math.sqrt(cluster_count))
    sign_matrix = np.asarray(
        list(itertools.product([-1.0, 1.0], repeat=cluster_count)),
        dtype=np.float64,
    )
    signed_means = (sign_matrix * values[None, :]).mean(axis=1)
    exact_one_sided = float(np.mean(signed_means >= observed_mean))
    exact_two_sided = float(
        np.mean(np.abs(signed_means) >= abs(observed_mean))
    )

    # Unrestricted wild cluster bootstrap-t with Rademacher weights.  At the
    # seed-mean level, clusters are independent observations and all 2^G
    # multiplier assignments can be enumerated when G=10.
    residuals = values - observed_mean
    bootstrap_values = observed_mean + sign_matrix * residuals[None, :]
    bootstrap_means = bootstrap_values.mean(axis=1)
    bootstrap_ses = bootstrap_values.std(axis=1, ddof=1) / math.sqrt(
        cluster_count
    )
    bootstrap_t = np.divide(
        bootstrap_means - observed_mean,
        bootstrap_ses,
        out=np.zeros_like(bootstrap_means),
        where=bootstrap_ses > replay.EPS,
    )
    lower_quantile, upper_quantile = np.quantile(
        bootstrap_t,
        [0.025, 0.975],
    )
    wild_ci_low = float(observed_mean - upper_quantile * observed_se)
    wild_ci_high = float(observed_mean - lower_quantile * observed_se)
    return {
        "cluster_count": cluster_count,
        "cluster_weighted_mean": observed_mean,
        "cluster_standard_error": observed_se,
        "positive_cluster_count": int(np.sum(values > 0.0)),
        "exact_sign_flip_p_one_sided": exact_one_sided,
        "exact_sign_flip_p_two_sided": exact_two_sided,
        "wild_cluster_bootstrap_t_ci_low": wild_ci_low,
        "wild_cluster_bootstrap_t_ci_high": wild_ci_high,
    }


def summarize(
    fold_results: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    Dict[str, Any],
]:
    endpoint_results = (
        fold_results.groupby(
            [
                "pair_id",
                "pair_order",
                "stability_status",
                "population_seed",
                "eta",
                "beta_sup",
                "kappa_map",
                "gamma_map",
                "support_abs_difference",
                "descriptor_match_distance",
            ],
            as_index=False,
        )
        .agg(
            mean_test_log_amplification=("test_log_amplification", "mean"),
            geometric_mean_test_amplification=("test_log_amplification", lambda x: float(np.exp(np.mean(x)))),
            fold_amplified_share=("test_amplified", "mean"),
            mean_train_gamma=("train_gamma", "mean"),
            mean_train_finite_log_gain=("train_finite_log_gain", "mean"),
            mean_active_support_size=("active_support_size", "mean"),
        )
    )
    paired = endpoint_results.pivot(
        index=["pair_id", "pair_order"],
        columns="stability_status",
        values=[
            "mean_test_log_amplification",
            "geometric_mean_test_amplification",
            "fold_amplified_share",
            "mean_train_gamma",
        ],
    )
    paired.columns = [
        f"{metric}_{status}" for metric, status in paired.columns
    ]
    paired = paired.reset_index()
    seed_wide = endpoint_results.pivot(
        index=["pair_id", "pair_order"],
        columns="stability_status",
        values="population_seed",
    ).reset_index()
    seed_wide = seed_wide.rename(
        columns={
            "phantom": "population_seed_phantom",
            "stable": "population_seed_stable",
        }
    )
    paired = paired.merge(seed_wide, on=["pair_id", "pair_order"], how="left")
    paired["log_amplification_difference"] = (
        paired["mean_test_log_amplification_phantom"]
        - paired["mean_test_log_amplification_stable"]
    )

    rng = np.random.default_rng(int(args.seed) + 991)
    status_rows: List[Dict[str, Any]] = []
    for status, group in endpoint_results.groupby("stability_status", sort=True):
        seed_mean_series = (
            group.groupby("population_seed")["mean_test_log_amplification"]
            .mean()
        )
        seed_means = seed_mean_series.to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_cluster_means(
            seed_means,
            int(args.bootstrap_reps),
            rng,
        )
        mean_log = float(group["mean_test_log_amplification"].mean())
        status_rows.append(
            {
                "stability_status": status,
                "n_endpoints": int(len(group)),
                "n_seed_clusters": int(group["population_seed"].nunique()),
                "mean_test_log_amplification": mean_log,
                "seed_weighted_mean_test_log_amplification": float(
                    seed_mean_series.mean()
                ),
                "cluster_ci_low": ci_low,
                "cluster_ci_high": ci_high,
                "geometric_mean_test_amplification": float(math.exp(mean_log)),
                "endpoint_amplified_count": int(
                    (group["mean_test_log_amplification"] > 0.0).sum()
                ),
                "endpoint_amplified_share": float(
                    (group["mean_test_log_amplification"] > 0.0).mean()
                ),
                "fold_amplified_share": float(
                    fold_results.loc[
                        fold_results["stability_status"].eq(status),
                        "test_amplified",
                    ].mean()
                ),
            }
        )
    status_summary = pd.DataFrame(status_rows)

    same_seed_pairs = bool(
        (
            paired["population_seed_phantom"]
            == paired["population_seed_stable"]
        ).all()
    )
    if same_seed_pairs:
        paired["inference_cluster"] = paired["population_seed_stable"]
        pair_effect_clusters = (
            paired.groupby("inference_cluster")["log_amplification_difference"]
            .mean()
            .reset_index()
            .rename(
                columns={
                    "log_amplification_difference": "cluster_mean_paired_difference"
                }
            )
        )
        paired_inference_label = "population seed"
    else:
        paired["inference_cluster"] = paired["pair_id"]
        pair_effect_clusters = paired[
            ["inference_cluster", "log_amplification_difference"]
        ].rename(
            columns={
                "log_amplification_difference": "cluster_mean_paired_difference"
            }
        )
        paired_inference_label = "matched pair (descriptive sensitivity analysis)"
    effect_low, effect_high = bootstrap_cluster_means(
        pair_effect_clusters["cluster_mean_paired_difference"].to_numpy(
            dtype=float
        ),
        int(args.bootstrap_reps),
        rng,
    )
    pair_weighted_difference = float(
        paired["log_amplification_difference"].mean()
    )
    small_cluster_inference = small_cluster_sign_and_wild_inference(
        pair_effect_clusters["cluster_mean_paired_difference"].to_numpy(
            dtype=float
        )
    )
    cluster_weighted_difference = float(
        small_cluster_inference["cluster_weighted_mean"]
    )
    overall = {
        "eta": float(args.eta),
        "epsilon_tv": float(args.epsilon),
        "n_pairs": int(len(paired)),
        "n_endpoints": int(len(endpoint_results)),
        "n_seed_clusters": int(
            endpoint_results["population_seed"].nunique()
        ),
        "folds": int(args.folds),
        "phantom_mean_log_amplification": float(
            endpoint_results.loc[
                endpoint_results["stability_status"].eq("phantom"),
                "mean_test_log_amplification",
            ].mean()
        ),
        "stable_mean_log_amplification": float(
            endpoint_results.loc[
                endpoint_results["stability_status"].eq("stable"),
                "mean_test_log_amplification",
            ].mean()
        ),
        "paired_log_amplification_difference": cluster_weighted_difference,
        "paired_log_amplification_difference_pair_weighted": pair_weighted_difference,
        "paired_log_amplification_difference_cluster_weighted": cluster_weighted_difference,
        "paired_difference_cluster_ci_low": effect_low,
        "paired_difference_cluster_ci_high": effect_high,
        "paired_difference_geometric_ratio": float(
            math.exp(cluster_weighted_difference)
        ),
        "phantom_endpoint_amplified_count": int(
            (
                endpoint_results.loc[
                    endpoint_results["stability_status"].eq("phantom"),
                    "mean_test_log_amplification",
                ]
                > 0.0
            ).sum()
        ),
        "stable_endpoint_amplified_count": int(
            (
                endpoint_results.loc[
                    endpoint_results["stability_status"].eq("stable"),
                    "mean_test_log_amplification",
                ]
                > 0.0
            ).sum()
        ),
        "support_match_difference_mean": float(
            endpoint_results["support_abs_difference"].mean()
        ),
        "support_match_difference_max": float(
            endpoint_results["support_abs_difference"].max()
        ),
        "eta_changed": False,
        "test_map_centered_at": "T_test(p_star)",
        "inference_clusters": paired_inference_label,
        **small_cluster_inference,
    }
    return endpoint_results, paired, pair_effect_clusters, status_summary, overall


def plot_results(
    endpoint_results: pd.DataFrame,
    status_summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    colors = {"phantom": "#D55E00", "stable": "#0072B2"}
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.3))
    paired_wide = endpoint_results.pivot(
        index="pair_id",
        columns="stability_status",
        values="mean_test_log_amplification",
    )
    for _, row in paired_wide.iterrows():
        axes[0].plot(
            [0, 1],
            [row["stable"], row["phantom"]],
            color="#B8B8B8",
            alpha=0.65,
            linewidth=0.8,
        )
    for position, status in enumerate(["stable", "phantom"]):
        values = paired_wide[status].to_numpy(dtype=float)
        axes[0].scatter(
            np.full_like(values, position, dtype=float),
            values,
            color=colors[status],
            s=24,
            zorder=3,
            label=status,
        )
    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_xticks([0, 1], ["Stable", "Phantom-diverse"])
    axes[0].set_ylabel("Held-out log amplification")
    axes[0].set_title("A  Matched endpoint pairs")

    for position, status in enumerate(["stable", "phantom"]):
        row = status_summary.loc[
            status_summary["stability_status"].eq(status)
        ].iloc[0]
        mean = float(row["mean_test_log_amplification"])
        low = float(row["cluster_ci_low"])
        high = float(row["cluster_ci_high"])
        axes[1].errorbar(
            position,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="o",
            color=colors[status],
            capsize=4,
            markersize=7,
        )
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xticks([0, 1], ["Stable", "Phantom-diverse"])
    axes[1].set_ylabel("Mean held-out log amplification")
    axes[1].set_title("B  Seed-clustered 95% intervals")
    figure.suptitle(r"Cross-fitted local response at fixed $\eta=2.25$")
    figure.tight_layout()
    figure.savefig(
        out_dir / "crossfit_local_amplification.png",
        dpi=240,
        bbox_inches="tight",
    )
    figure.savefig(
        out_dir / "crossfit_local_amplification.pdf",
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    started = time.time()
    if int(args.folds) < 2:
        raise ValueError("--folds must be at least 2.")
    if float(args.epsilon) <= 0.0:
        raise ValueError("--epsilon must be positive.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(args)
    pairs = select_pairs(candidates, args)
    pairs.to_csv(out_dir / "matched_endpoint_plan.csv", index=False)
    print(
        f"Selected {pairs['pair_id'].nunique()} pairs at fixed eta={args.eta:g} "
        f"across {pairs['population_seed'].nunique()} seeds.",
        flush=True,
    )

    endpoint_rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []
    table_cache: Dict[int, replay.PreparedLog] = {}
    fold_cache: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
    for endpoint_number, (_, row) in enumerate(pairs.iterrows(), start=1):
        population_seed = int(row["population_seed"])
        table = table_cache.get(population_seed)
        if table is None:
            source = Path(str(row["raw_source"]))
            print(f"Loading seed {population_seed} from {source}", flush=True)
            table = replay.prepare_log_streaming(
                source,
                run_name=source.name,
                score_weights=replay.DEFAULT_SCORE_WEIGHTS,
            )
            table_cache[population_seed] = table
            fold_cache[population_seed] = make_context_folds(
                table,
                folds=int(args.folds),
                random_seed=int(args.seed) + population_seed,
            )
        print(
            f"[{endpoint_number}/{len(pairs)}] {row['pair_id']} "
            f"{row['stability_status']} seed={population_seed} "
            f"support={float(row['beta_sup']):g}",
            flush=True,
        )
        endpoint_row, endpoint_fold_rows = run_endpoint(
            table,
            row,
            fold_cache[population_seed],
            args,
        )
        endpoint_rows.append(endpoint_row)
        fold_rows.extend(endpoint_fold_rows)

    endpoints = pd.DataFrame(endpoint_rows)
    fold_results = pd.DataFrame(fold_rows)
    endpoints.to_csv(out_dir / "endpoints.csv", index=False)
    fold_results.to_csv(out_dir / "fold_results.csv", index=False)
    (
        endpoint_results,
        paired,
        pair_effect_clusters,
        status_summary,
        overall,
    ) = summarize(fold_results, args)
    endpoint_results.to_csv(out_dir / "endpoint_results.csv", index=False)
    paired.to_csv(out_dir / "paired_endpoint_effects.csv", index=False)
    pair_effect_clusters.to_csv(
        out_dir / "paired_effect_clusters.csv",
        index=False,
    )
    status_summary.to_csv(out_dir / "status_summary.csv", index=False)
    plot_results(endpoint_results, status_summary, out_dir)

    overall["elapsed_sec"] = float(time.time() - started)
    if args.matching_mode == "within-seed-near-support":
        overall["matching"] = (
            "within population seed; eta and all controls except beta_sup exact; "
            "beta_sup and descriptors nearest-matched without replacement"
        )
    else:
        overall["matching"] = (
            "eta and every platform control including beta_sup exact; "
            "descriptors nearest-matched without replacement across population seeds"
        )
    overall["matching_mode"] = args.matching_mode
    overall["direction_estimation"] = (
        "leading active-face eigenspace from training contexts only"
    )
    overall["outcome"] = (
        "log[TV(T_test(p_prime), T_test(p_star)) / TV(p_prime, p_star)]"
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(overall, indent=2) + "\n"
    )
    print(json.dumps(overall, indent=2), flush=True)


if __name__ == "__main__":
    main()
