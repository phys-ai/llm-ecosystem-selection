#!/usr/bin/env python3
"""Cross-fitted direct validation of the inactive-agent invasion branch.

The full-context diagnostic defines the invasion branch by adding a small
mass epsilon to each inactive agent and taking the largest one-step log
multiplier.  This script separates direction selection from evaluation:

* select the inactive agent using training contexts only;
* evaluate that same agent on held-out contexts at fixed eta;
* additionally project the selected agent exactly to the boundary before
  injection, so endpoint residual mass cannot create a spurious multiplier.

The sampled invasion-positive endpoints are nearest-matched to the same
descriptor-positive contracting pool used by the active-face experiment.
Contracting controls are used only to define a comparable endpoint sample:
they are interior endpoints and therefore have no inactive-agent branch.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay import endpoint as replay
from fragility import amplification as active


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-glob",
        default=(
            "data/results/fig5_full_grid/"
            "run_popseed_*.checkpoints/phase_summary.csv"
        ),
    )
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument(
        "--out-dir",
        default="data/results/crossfit_invasion_validation_eta2p25",
    )
    parser.add_argument("--eta", type=float, default=2.25)
    parser.add_argument("--invasion-eps", type=float, default=1.0e-5)
    parser.add_argument(
        "--dose-grid",
        default="1e-6,1e-5,1e-4",
        help="Comma-separated held-out sensitivity doses.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--endpoint-last-k", type=int, default=100)
    parser.add_argument("--active-eps", type=float, default=1.0e-4)
    parser.add_argument("--support-clip", type=float, default=8.0)
    parser.add_argument("--kappa-band", type=float, default=0.05)
    parser.add_argument("--support-match-scale", type=float, default=0.05)
    parser.add_argument("--support-match-weight", type=float, default=4.0)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--bootstrap-reps", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def load_invasion_candidates(args: argparse.Namespace) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in sorted(Path().glob(args.summary_glob)):
        population_seed = active.seed_from_path(path)
        frame = pd.read_csv(path)
        frame = frame.loc[
            active.descriptor_positive_mask(frame)
            & np.isclose(
                pd.to_numeric(frame["eta"], errors="coerce"),
                float(args.eta),
            )
        ].copy()
        kappa = pd.to_numeric(frame["kappa_map"], errors="coerce")
        invasion = pd.to_numeric(
            frame["max_invasion_log"], errors="coerce"
        )
        frame["stability_status"] = "excluded"
        frame.loc[
            kappa < -float(args.kappa_band), "stability_status"
        ] = "stable"
        frame.loc[
            invasion > float(args.kappa_band), "stability_status"
        ] = "phantom"
        frame = frame.loc[
            frame["stability_status"].isin(["phantom", "stable"])
        ].copy()
        frame["population_seed"] = int(population_seed)
        frame["summary_path"] = str(path.resolve())
        frame["raw_source"] = str(
            (
                Path(args.runs_dir)
                / f"run_popseed_{population_seed}.checkpoints"
            ).resolve()
        )
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"No summaries matched {args.summary_glob!r}"
        )
    candidates = pd.concat(frames, ignore_index=True)
    if candidates.empty:
        raise ValueError(
            f"No invasion-positive or stable endpoints at eta={args.eta:g}."
        )
    return candidates


def injected_state(
    endpoint: np.ndarray,
    agent: int,
    epsilon: float,
    *,
    exact_boundary: bool,
) -> np.ndarray:
    state = np.asarray(endpoint, dtype=np.float64).copy()
    state = np.clip(state, replay.EPS, None)
    state = state / float(state.sum())
    if exact_boundary:
        state[int(agent)] = 0.0
        remainder = float(state.sum())
        if remainder <= replay.EPS:
            raise ValueError("Boundary projection removed all endpoint mass.")
        state = state / remainder
    perturbed = (1.0 - float(epsilon)) * state
    perturbed[int(agent)] += float(epsilon)
    return perturbed / float(perturbed.sum())


def invasion_log_multiplier(
    mean_map: Callable[[np.ndarray], np.ndarray],
    endpoint: np.ndarray,
    agent: int,
    epsilon: float,
    *,
    exact_boundary: bool,
) -> Tuple[float, float, float]:
    perturbed = injected_state(
        endpoint,
        int(agent),
        float(epsilon),
        exact_boundary=bool(exact_boundary),
    )
    output_mass = float(mean_map(perturbed)[int(agent)])
    log_multiplier = float(
        math.log(max(output_mass / float(epsilon), replay.EPS))
    )
    return log_multiplier, output_mass, float(perturbed[int(agent)])


def select_train_invasion_agent(
    train_map: Callable[[np.ndarray], np.ndarray],
    endpoint: np.ndarray,
    active_eps: float,
    invasion_eps: float,
) -> Tuple[int, Dict[str, Any]]:
    p_star = np.asarray(endpoint, dtype=np.float64)
    p_star = np.clip(p_star, replay.EPS, None)
    p_star = p_star / float(p_star.sum())
    inactive = np.flatnonzero(p_star <= float(active_eps))
    if inactive.size == 0:
        raise ValueError("Endpoint has no inactive agent under active_eps.")

    candidates: List[Tuple[float, int, float]] = []
    for agent in inactive:
        log_multiplier, output_mass, _ = invasion_log_multiplier(
            train_map,
            p_star,
            int(agent),
            float(invasion_eps),
            exact_boundary=False,
        )
        candidates.append(
            (float(log_multiplier), int(agent), float(output_mass))
        )
    train_log, selected_agent, train_output_mass = max(candidates)
    boundary_log, boundary_output, _ = invasion_log_multiplier(
        train_map,
        p_star,
        selected_agent,
        float(invasion_eps),
        exact_boundary=True,
    )
    return selected_agent, {
        "inactive_agent_count": int(inactive.size),
        "selected_agent_endpoint_mass": float(p_star[selected_agent]),
        "train_log_invasion": float(train_log),
        "train_invasion_output_mass": float(train_output_mass),
        "train_boundary_log_invasion": float(boundary_log),
        "train_boundary_output_mass": float(boundary_output),
    }


def run_endpoint(
    table: replay.PreparedLog,
    row: Mapping[str, Any],
    folds: Sequence[Tuple[np.ndarray, np.ndarray]],
    doses: Sequence[float],
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
    p_star = np.asarray(p_star, dtype=np.float64)
    p_star = np.clip(p_star, replay.EPS, None)
    p_star = p_star / float(p_star.sum())
    inactive = np.flatnonzero(p_star <= float(args.active_eps))
    endpoint_row = {
        "pair_id": row["pair_id"],
        "pair_order": int(row["pair_order"]),
        "population_seed": int(row["population_seed"]),
        "eta": float(row["eta"]),
        "beta_sup": float(row["beta_sup"]),
        "kappa_map": float(row["kappa_map"]),
        "gamma_map": float(row["gamma_map"]),
        "max_invasion_log": float(row["max_invasion_log"]),
        "summary_max_invasion_agent": str(row["max_invasion_agent"]),
        "inactive_agent_count_replayed": int(inactive.size),
        "match_cost": float(row["match_cost"]),
        "descriptor_match_distance": float(
            row["descriptor_match_distance"]
        ),
        "support_abs_difference": float(row["support_abs_difference"]),
        **{
            f"replayed_{key}": value
            for key, value in replay.concentration_descriptors(p_star).items()
        },
    }
    if inactive.size == 0:
        raise ValueError(
            f"{row['pair_id']} has no replayed inactive agent."
        )

    fold_rows: List[Dict[str, Any]] = []
    for fold_index, (train_indices, test_indices) in enumerate(folds):
        train_map = active.build_mean_map(
            table,
            params,
            train_indices,
            support_clip=float(args.support_clip),
        )
        test_map = active.build_mean_map(
            table,
            params,
            test_indices,
            support_clip=float(args.support_clip),
        )
        selected_agent, train_metadata = select_train_invasion_agent(
            train_map,
            p_star,
            active_eps=float(args.active_eps),
            invasion_eps=float(args.invasion_eps),
        )
        test_endpoint_output = test_map(p_star)
        for dose in doses:
            test_log, test_output, injected_mass = (
                invasion_log_multiplier(
                    test_map,
                    p_star,
                    selected_agent,
                    float(dose),
                    exact_boundary=False,
                )
            )
            boundary_log, boundary_output, boundary_injected_mass = (
                invasion_log_multiplier(
                    test_map,
                    p_star,
                    selected_agent,
                    float(dose),
                    exact_boundary=True,
                )
            )
            fold_rows.append(
                {
                    **endpoint_row,
                    "fold": int(fold_index),
                    "n_train_contexts": int(len(train_indices)),
                    "n_test_contexts": int(len(test_indices)),
                    "selected_agent_index": int(selected_agent),
                    "selected_agent_id": str(
                        table.agent_ids[int(selected_agent)]
                    ),
                    "dose": float(dose),
                    "is_primary_dose": int(
                        np.isclose(
                            float(dose),
                            float(args.invasion_eps),
                            rtol=0.0,
                            atol=float(args.invasion_eps) * 1.0e-10,
                        )
                    ),
                    "test_log_invasion": float(test_log),
                    "test_invasion_multiplier": float(math.exp(test_log)),
                    "test_invasion_output_mass": float(test_output),
                    "injected_agent_mass": float(injected_mass),
                    "test_boundary_log_invasion": float(boundary_log),
                    "test_boundary_invasion_multiplier": float(
                        math.exp(boundary_log)
                    ),
                    "test_boundary_output_mass": float(boundary_output),
                    "boundary_injected_agent_mass": float(
                        boundary_injected_mass
                    ),
                    "test_unperturbed_output_mass": float(
                        test_endpoint_output[int(selected_agent)]
                    ),
                    "test_invasion_positive": int(test_log > 0.0),
                    "test_boundary_invasion_positive": int(
                        boundary_log > 0.0
                    ),
                    **train_metadata,
                }
            )
    return endpoint_row, fold_rows


def summarize(
    fold_results: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    endpoint_results = (
        fold_results.groupby(
            [
                "pair_id",
                "pair_order",
                "population_seed",
                "eta",
                "beta_sup",
                "kappa_map",
                "gamma_map",
                "max_invasion_log",
                "dose",
                "is_primary_dose",
            ],
            as_index=False,
        )
        .agg(
            mean_test_log_invasion=("test_log_invasion", "mean"),
            mean_test_boundary_log_invasion=(
                "test_boundary_log_invasion",
                "mean",
            ),
            fold_invasion_positive_share=(
                "test_invasion_positive",
                "mean",
            ),
            fold_boundary_invasion_positive_share=(
                "test_boundary_invasion_positive",
                "mean",
            ),
            selected_agent_consistency=(
                "selected_agent_index",
                lambda x: float(pd.Series(x).value_counts(normalize=True).max()),
            ),
            mean_train_log_invasion=("train_log_invasion", "mean"),
            mean_selected_agent_endpoint_mass=(
                "selected_agent_endpoint_mass",
                "mean",
            ),
        )
    )
    dose_rows: List[Dict[str, Any]] = []
    for dose, group in endpoint_results.groupby("dose", sort=True):
        seed_raw = (
            group.groupby("population_seed")["mean_test_log_invasion"]
            .mean()
            .to_numpy(dtype=float)
        )
        seed_boundary = (
            group.groupby("population_seed")[
                "mean_test_boundary_log_invasion"
            ]
            .mean()
            .to_numpy(dtype=float)
        )
        raw_inference = active.small_cluster_sign_and_wild_inference(
            seed_raw
        )
        boundary_inference = active.small_cluster_sign_and_wild_inference(
            seed_boundary
        )
        dose_rows.append(
            {
                "dose": float(dose),
                "n_endpoints": int(len(group)),
                "n_seed_clusters": int(group["population_seed"].nunique()),
                "mean_test_log_invasion": float(
                    group["mean_test_log_invasion"].mean()
                ),
                "geometric_mean_test_invasion_multiplier": float(
                    math.exp(group["mean_test_log_invasion"].mean())
                ),
                "endpoint_invasion_positive_count": int(
                    (group["mean_test_log_invasion"] > 0.0).sum()
                ),
                "mean_test_boundary_log_invasion": float(
                    group["mean_test_boundary_log_invasion"].mean()
                ),
                "geometric_mean_test_boundary_invasion_multiplier": float(
                    math.exp(
                        group["mean_test_boundary_log_invasion"].mean()
                    )
                ),
                "endpoint_boundary_invasion_positive_count": int(
                    (
                        group["mean_test_boundary_log_invasion"] > 0.0
                    ).sum()
                ),
                **{
                    f"raw_{key}": value
                    for key, value in raw_inference.items()
                },
                **{
                    f"boundary_{key}": value
                    for key, value in boundary_inference.items()
                },
            }
        )
    dose_summary = pd.DataFrame(dose_rows)
    primary = dose_summary.loc[
        np.isclose(
            dose_summary["dose"],
            float(args.invasion_eps),
            rtol=0.0,
            atol=float(args.invasion_eps) * 1.0e-10,
        )
    ].iloc[0]
    overall = {
        "eta": float(args.eta),
        "eta_changed": False,
        "primary_invasion_epsilon": float(args.invasion_eps),
        "dose_grid": [float(value) for value in sorted(dose_summary["dose"])],
        "n_endpoints": int(primary["n_endpoints"]),
        "n_seed_clusters": int(primary["n_seed_clusters"]),
        "folds": int(args.folds),
        "mean_heldout_log_invasion": float(
            primary["mean_test_log_invasion"]
        ),
        "geometric_mean_heldout_invasion_multiplier": float(
            primary["geometric_mean_test_invasion_multiplier"]
        ),
        "heldout_invasion_positive_endpoints": int(
            primary["endpoint_invasion_positive_count"]
        ),
        "mean_heldout_boundary_log_invasion": float(
            primary["mean_test_boundary_log_invasion"]
        ),
        "geometric_mean_heldout_boundary_invasion_multiplier": float(
            primary[
                "geometric_mean_test_boundary_invasion_multiplier"
            ]
        ),
        "heldout_boundary_invasion_positive_endpoints": int(
            primary["endpoint_boundary_invasion_positive_count"]
        ),
        "boundary_seed_clustered_mean": float(
            primary["boundary_cluster_weighted_mean"]
        ),
        "boundary_seed_clustered_se": float(
            primary["boundary_cluster_standard_error"]
        ),
        "boundary_wild_cluster_ci_low": float(
            primary["boundary_wild_cluster_bootstrap_t_ci_low"]
        ),
        "boundary_wild_cluster_ci_high": float(
            primary["boundary_wild_cluster_bootstrap_t_ci_high"]
        ),
        "boundary_exact_sign_flip_p_one_sided": float(
            primary["boundary_exact_sign_flip_p_one_sided"]
        ),
        "boundary_positive_seed_clusters": int(
            primary["boundary_positive_cluster_count"]
        ),
        "direction_selection": (
            "max finite-dose inactive-agent log multiplier on training "
            "contexts only"
        ),
        "heldout_primary_outcome": (
            "log[T_test,r((1-epsilon)p_star + epsilon e_r) / epsilon]"
        ),
        "heldout_boundary_outcome": (
            "same outcome after setting selected agent mass exactly to zero "
            "and renormalizing the other agents before injection"
        ),
        "control_role": (
            "contracting endpoints define the matched sampling frame only; "
            "they are interior and have no inactive-agent branch"
        ),
    }
    return endpoint_results, dose_summary, overall


def main() -> None:
    args = parse_args()
    started = time.time()
    doses = sorted(
        {
            float(value.strip())
            for value in str(args.dose_grid).split(",")
            if value.strip()
        }
        | {float(args.invasion_eps)}
    )
    if not doses or min(doses) <= 0.0:
        raise ValueError("All invasion doses must be positive.")
    if int(args.folds) < 2:
        raise ValueError("--folds must be at least 2.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_invasion_candidates(args)
    args.matching_mode = "within-seed-near-support"
    pairs = active.select_pairs(candidates, args)
    pairs.to_csv(out_dir / "matched_sampling_plan.csv", index=False)
    invasion_endpoints = pairs.loc[
        pairs["stability_status"].eq("phantom")
    ].copy()
    print(
        f"Selected {len(invasion_endpoints)} invasion-positive endpoints "
        f"at fixed eta={args.eta:g} across "
        f"{invasion_endpoints['population_seed'].nunique()} seeds.",
        flush=True,
    )

    endpoint_rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []
    table_cache: Dict[int, replay.PreparedLog] = {}
    fold_cache: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
    for endpoint_number, (_, row) in enumerate(
        invasion_endpoints.iterrows(), start=1
    ):
        population_seed = int(row["population_seed"])
        table = table_cache.get(population_seed)
        if table is None:
            source = Path(str(row["raw_source"]))
            print(
                f"Loading seed {population_seed} from {source}",
                flush=True,
            )
            table = replay.prepare_log_streaming(
                source,
                run_name=source.name,
                score_weights=replay.DEFAULT_SCORE_WEIGHTS,
            )
            table_cache[population_seed] = table
            fold_cache[population_seed] = active.make_context_folds(
                table,
                folds=int(args.folds),
                random_seed=int(args.seed) + population_seed,
            )
        print(
            f"[{endpoint_number}/{len(invasion_endpoints)}] "
            f"{row['pair_id']} seed={population_seed} "
            f"support={float(row['beta_sup']):g}",
            flush=True,
        )
        endpoint_row, endpoint_fold_rows = run_endpoint(
            table,
            row,
            fold_cache[population_seed],
            doses,
            args,
        )
        endpoint_rows.append(endpoint_row)
        fold_rows.extend(endpoint_fold_rows)

    endpoints = pd.DataFrame(endpoint_rows)
    fold_results = pd.DataFrame(fold_rows)
    endpoint_results, dose_summary, overall = summarize(
        fold_results, args
    )
    endpoints.to_csv(out_dir / "endpoints.csv", index=False)
    fold_results.to_csv(out_dir / "fold_results.csv", index=False)
    endpoint_results.to_csv(
        out_dir / "endpoint_results_by_dose.csv", index=False
    )
    dose_summary.to_csv(out_dir / "dose_summary.csv", index=False)
    overall["elapsed_sec"] = float(time.time() - started)
    overall["matching"] = (
        "within population seed; eta and all controls except beta_sup exact; "
        "beta_sup and descriptors nearest-matched without replacement"
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(overall, indent=2) + "\n"
    )
    print(json.dumps(overall, indent=2), flush=True)


if __name__ == "__main__":
    main()
