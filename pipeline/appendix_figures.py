#!/usr/bin/env python3
"""Threshold robustness (Appendix B.5) and the appendix figures (Figs. 7-10, 12-14), in one file.

Two earlier scripts (the threshold-robustness workflow and the figure builders) are embedded below as
functions; run from data/ (see pipeline/README.md, section 5).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import sys
import tempfile

DEFAULT_FIGURES_OUT_DIR = os.environ.get("FIGURES_OUT_DIR", "results/figures")


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _repo_root() -> Path:
    return _script_dir().parent  # pipeline/ -> package root


def _split_extra(extra: str | None) -> list[str]:
    if extra is None or not str(extra).strip():
        return []
    return shlex.split(extra)


def _format_cmd(script_label: str, extra_args: list[str]) -> str:
    return " ".join(shlex.quote(x) for x in [sys.executable, script_label, *extra_args])


def _system_exit_code(exc: SystemExit) -> int:
    if exc.code is None:
        return 0
    if isinstance(exc.code, int):
        return exc.code
    print(exc.code, file=sys.stderr)
    return 1


def _run_threshold_embedded(extra_args: list[str], *, dry_run: bool = False) -> int:
    script_label = "appendix_figures.py::threshold_robustness"
    print("[run] " + _format_cmd(script_label, extra_args), flush=True)
    if dry_run:
        return 0
    old_argv = sys.argv[:]
    sys.argv = [script_label, *extra_args]
    __file__ = str(_repo_root() / "analysis" / "threshold_robustness.py")
    try:
        """
        Post-hoc threshold robustness analysis for AI social-platform replay outputs.

        This script does two complementary things:

        1. Descriptor/stability-threshold robustness from existing CSVs.
           It recomputes operational labels and stable-phase counts over a grid of
           descriptor thresholds without rerunning replay. This covers thresholds such as
           collapse_top1, specialization_mi, winner_switch, and kappa stability margins.

        2. Numerical stability robustness from small rerun outputs.
           If you pass --numerical_results_dirs, it compares kappa_map and stable labels
           from reruns with different active_eps/fd_eps/invasion_eps/stride/K settings
           against the baseline run where possible.

        Expected inputs are directories containing per-run outputs such as:
          results/fig0_4_full_grid/run_popseed_66.checkpoints/phase_summary.csv
          results/fig0_4_full_grid/run_popseed_66.checkpoints/stability_summary.csv
        or already-merged files:
          results/merged/phase_summary.csv
          results/merged/stability_summary.csv

        Example:
          python threshold_robustness.py
        """

        import argparse
        import json
        import re
        from dataclasses import asdict, dataclass
        from pathlib import Path
        from typing import Dict, List, Optional, Sequence, Tuple

        import numpy as np
        import pandas as pd


        DEFAULT_RESULTS_DIRS = [
            "results/fig0_4_full_grid",
            "results/fig5_full_grid",
            "results/specialization_wide_dense_figS",
        ]

        DEFAULT_NUMERICAL_RESULTS_DIRS = [
            "results/stability_threshold_robustness",
        ]

        DEFAULT_OUT_DIR = "results/threshold_robustness"


        DEFAULT_ID_COLS = [
            "run_name",
            "source_result_dir",
            "sweep_name",
            "eta",
            "alpha",
            "beta_sup",
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

        PHASE_ORDER = [
            "diffuse_coexistence",
            "concentrated_coexistence",
            "monoculture_collapse",
            "polarization",
            "contextual_specialization",
            "polarized_specialization",
            "specialized_collapse",
        ]


        @dataclass(frozen=True)
        class ThresholdSetting:
            setting_id: str
            collapse_top1: float = 0.80
            collapse_neff_frac: float = 0.025
            collapse_neff_abs: float = 1.5
            concentrated_top1: float = 0.45
            concentrated_neff_frac: float = 0.10
            concentrated_neff_abs: float = 3.0
            specialization_mi: float = 0.12
            strong_specialization_mi: float = 0.15
            winner_switch: float = 0.30
            polarization_var: float = 0.50
            stance_extremity: float = 0.70
            polarization_winner_switch: float = 0.20
            stable_kappa_margin: float = 0.0
            near_critical_eps: float = 0.05


        def parse_float_list(raw: str) -> List[float]:
            if raw is None or str(raw).strip() == "":
                return []
            vals = []
            for item in str(raw).split(","):
                item = item.strip()
                if not item:
                    continue
                vals.append(float(item))
            return vals


        def safe_name(path: Path) -> str:
            text = str(path).strip("/") or "root"
            text = re.sub(r"[^A-Za-z0-9_.=-]+", "_", text)
            return text[-160:]


        def read_csv_maybe(path: Path) -> Optional[pd.DataFrame]:
            if not path.exists():
                return None
            try:
                return pd.read_csv(path)
            except Exception as exc:
                print(f"[warn] failed to read {path}: {exc}")
                return None


        def discover_csvs(root: Path, filename: str) -> List[Path]:
            if root.is_file() and root.name == filename:
                return [root]
            if root.is_file():
                return []
            direct = root / filename
            if direct.exists():
                return [direct]
            return sorted(root.glob(f"**/{filename}"))


        def load_result_dirs(paths: Sequence[str], filename: str, source_label: str) -> pd.DataFrame:
            parts: List[pd.DataFrame] = []
            for raw in paths:
                root = Path(raw)
                files = discover_csvs(root, filename)
                if not files:
                    print(f"[warn] no {filename} found under {root}")
                    continue
                for f in files:
                    df = read_csv_maybe(f)
                    if df is None or df.empty:
                        continue
                    df = df.copy()
                    df["input_root"] = str(root)
                    df["input_file"] = str(f)
                    # Keep useful labels even when the original CSV did not include them.
                    if "source_result_dir" not in df.columns:
                        df["source_result_dir"] = f.parent.name
                    if "sweep_name" not in df.columns:
                        # Usually: root/sweep/run/file.csv or root/run/file.csv
                        try:
                            rel = f.relative_to(root)
                            parts_rel = rel.parts
                            df["sweep_name"] = parts_rel[0] if len(parts_rel) >= 3 else source_label
                        except Exception:
                            df["sweep_name"] = source_label
                    if "run_name" not in df.columns:
                        df["run_name"] = f.parent.name
                    parts.append(df)
            if not parts:
                return pd.DataFrame()
            return pd.concat(parts, ignore_index=True, sort=False)


        def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            numeric_candidates = [
                "eta", "alpha", "beta_sup", "beta_route", "lambda_risk",
                "hard_safety_threshold", "context_trait_strength", "polarization_strength",
                "route_power", "phase_code", "hhi", "n_eff", "top1_share",
                "entropy", "entropy_norm", "abs_polarization", "polarization_dispersion",
                "stance_extremity", "i_exp_norm", "winner_switch_rate", "weighted_risk",
                "weighted_quality", "n_agents", "kappa_map", "gamma_map",
                "max_invasion_log", "spectral_radius", "active_count", "inactive_count",
                "active_eps", "fd_eps", "invasion_eps", "stability_context_stride",
                "endpoint_last_k",
            ]
            for c in numeric_candidates:
                if c in out.columns:
                    out[c] = pd.to_numeric(out[c], errors="coerce")
            for c in ["route_zscore", "no_feedback", "locally_contracting"]:
                if c in out.columns:
                    if out[c].dtype == bool:
                        out[c] = out[c].astype(int)
                    else:
                        out[c] = out[c].astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)
            return out


        def merge_phase_and_stability(phase: pd.DataFrame, stability: pd.DataFrame) -> pd.DataFrame:
            phase = coerce_numeric(phase)
            stability = coerce_numeric(stability)
            if phase.empty:
                return stability.copy()
            if stability.empty:
                return phase.copy()

            add_cols = [
                c for c in [
                    "kappa_map", "gamma_map", "max_invasion_log", "spectral_radius",
                    "active_count", "inactive_count", "active_eps", "fd_eps", "invasion_eps",
                    "stability_context_stride", "locally_contracting",
                ]
                if c in stability.columns and c not in phase.columns
            ]
            if not add_cols:
                out = phase.copy()
                # If kappa columns exist in both, prefer phase_summary but fill missing from stability.
                fill_cols = [c for c in ["kappa_map", "gamma_map", "max_invasion_log"] if c in phase.columns and c in stability.columns]
                if not fill_cols:
                    return out
            key_candidates = [c for c in DEFAULT_ID_COLS if c in phase.columns and c in stability.columns]
            # Avoid over-keying by file-specific labels if they are inconsistent.
            if "input_file" in key_candidates:
                key_candidates.remove("input_file")
            keys = key_candidates
            if not keys:
                return phase.copy()

            stab_cols = keys + sorted(set(add_cols + [c for c in ["kappa_map", "gamma_map", "max_invasion_log"] if c in stability.columns]))
            stab_small = stability[stab_cols].drop_duplicates()
            out = phase.merge(stab_small, on=keys, how="left", suffixes=("", "_stab"))
            for c in ["kappa_map", "gamma_map", "max_invasion_log"]:
                c_stab = f"{c}_stab"
                if c_stab in out.columns:
                    if c in out.columns:
                        out[c] = out[c].where(out[c].notna(), out[c_stab])
                        out = out.drop(columns=[c_stab])
                    else:
                        out = out.rename(columns={c_stab: c})
            return out


        def setting_grid(args: argparse.Namespace) -> List[ThresholdSetting]:
            collapse_top1s = parse_float_list(args.collapse_top1s) or [0.75, 0.80, 0.85]
            collapse_neff_fracs = parse_float_list(args.collapse_neff_fracs) or [0.020, 0.025, 0.030]
            concentrated_top1s = parse_float_list(args.concentrated_top1s) or [0.40, 0.45, 0.50]
            concentrated_neff_fracs = parse_float_list(args.concentrated_neff_fracs) or [0.08, 0.10, 0.12]
            spec_mis = parse_float_list(args.specialization_mis) or [0.10, 0.12, 0.15]
            strong_spec_mis = parse_float_list(args.strong_specialization_mis) or [0.12, 0.15, 0.18]
            winner_switches = parse_float_list(args.winner_switches) or [0.25, 0.30, 0.35]
            pol_vars = parse_float_list(args.polarization_vars) or [0.45, 0.50, 0.55]
            stance_extremities = parse_float_list(args.stance_extremities) or [0.65, 0.70, 0.75]
            stable_margins = parse_float_list(args.stable_kappa_margins) or [0.0, 0.02, 0.05, 0.10]
            near_epss = parse_float_list(args.near_critical_epss) or [0.025, 0.05, 0.10]

            # Joint descriptor sensitivity used for the check that
            # specialization/diversity conclusions are not an artifact of a single
            # I_exp, winner-switching, or N_eff cutoff. These settings are summarized
            # separately from the one-at-a-time Table 1 rows to avoid bloating the
            # manuscript table.
            joint_spec_mis = parse_float_list(getattr(args, "joint_specialization_mis", "")) or [0.05, 0.10, 0.12, 0.15, 0.20]
            joint_winner_switches = parse_float_list(getattr(args, "joint_winner_switches", "")) or [0.20, 0.25, 0.30, 0.35, 0.40]
            joint_concentrated_neff_fracs = parse_float_list(getattr(args, "joint_concentrated_neff_fracs", "")) or [0.06, 0.08, 0.10, 0.12, 0.15]
            joint_collapse_neff_fracs = parse_float_list(getattr(args, "joint_collapse_neff_fracs", "")) or [0.015, 0.020, 0.025, 0.030, 0.040]

            settings: List[ThresholdSetting] = []
            # Baseline first.
            settings.append(ThresholdSetting(setting_id="baseline"))

            # One-at-a-time perturbations around baseline. This is easier to interpret
            # than a full factorial grid.
            baseline = ThresholdSetting(setting_id="baseline")
            for val in collapse_top1s:
                settings.append(ThresholdSetting(**{**asdict(baseline), "setting_id": f"collapse_top1={val:g}", "collapse_top1": val}))
            for val in collapse_neff_fracs:
                settings.append(ThresholdSetting(**{**asdict(baseline), "setting_id": f"collapse_neff_frac={val:g}", "collapse_neff_frac": val}))
            for val in concentrated_top1s:
                settings.append(ThresholdSetting(**{**asdict(baseline), "setting_id": f"concentrated_top1={val:g}", "concentrated_top1": val}))
            for val in concentrated_neff_fracs:
                settings.append(ThresholdSetting(**{**asdict(baseline), "setting_id": f"concentrated_neff_frac={val:g}", "concentrated_neff_frac": val}))
            for val in spec_mis:
                settings.append(ThresholdSetting(**{**asdict(baseline), "setting_id": f"specialization_mi={val:g}", "specialization_mi": val}))
            for val in strong_spec_mis:
                settings.append(ThresholdSetting(**{**asdict(baseline), "setting_id": f"strong_specialization_mi={val:g}", "strong_specialization_mi": val}))
            for val in winner_switches:
                settings.append(ThresholdSetting(**{**asdict(baseline), "setting_id": f"winner_switch={val:g}", "winner_switch": val}))
            for val in pol_vars:
                settings.append(ThresholdSetting(**{**asdict(baseline), "setting_id": f"polarization_var={val:g}", "polarization_var": val}))
            for val in stance_extremities:
                settings.append(ThresholdSetting(**{**asdict(baseline), "setting_id": f"stance_extremity={val:g}", "stance_extremity": val}))
            for val in stable_margins:
                settings.append(ThresholdSetting(**{**asdict(baseline), "setting_id": f"stable_kappa_margin={val:g}", "stable_kappa_margin": val}))
            for val in near_epss:
                settings.append(ThresholdSetting(**{**asdict(baseline), "setting_id": f"near_critical_eps={val:g}", "near_critical_eps": val}))

            if getattr(args, "joint_descriptor_sensitivity", True):
                for mi in joint_spec_mis:
                    for ws in joint_winner_switches:
                        for cn in joint_concentrated_neff_fracs:
                            for coll_n in joint_collapse_neff_fracs:
                                sid = f"joint|mi={mi:g}|ws={ws:g}|conc_neff={cn:g}|coll_neff={coll_n:g}"
                                settings.append(ThresholdSetting(
                                    **{
                                        **asdict(baseline),
                                        "setting_id": sid,
                                        "specialization_mi": mi,
                                        "strong_specialization_mi": max(mi, 0.15),
                                        "winner_switch": ws,
                                        "concentrated_neff_frac": cn,
                                        "collapse_neff_frac": coll_n,
                                    }
                                ))

            # Optional full factorial over the most important descriptor thresholds.
            if args.full_factorial:
                settings = []
                for ct in collapse_top1s:
                    for cn in collapse_neff_fracs:
                        for mi in spec_mis:
                            for ws in winner_switches:
                                for km in stable_margins:
                                    sid = f"ct={ct:g}|cn={cn:g}|mi={mi:g}|ws={ws:g}|km={km:g}"
                                    settings.append(ThresholdSetting(
                                        setting_id=sid,
                                        collapse_top1=ct,
                                        collapse_neff_frac=cn,
                                        specialization_mi=mi,
                                        strong_specialization_mi=max(mi, 0.15),
                                        winner_switch=ws,
                                        stable_kappa_margin=km,
                                    ))
            # De-duplicate while preserving order.
            seen = set()
            unique: List[ThresholdSetting] = []
            for s in settings:
                key = tuple(asdict(s).items())
                if key in seen:
                    continue
                seen.add(key)
                unique.append(s)
            return unique


        def require_series(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce")
            return pd.Series(default, index=df.index, dtype=float)


        def apply_thresholds(df: pd.DataFrame, setting: ThresholdSetting) -> pd.DataFrame:
            out = df.copy()
            n_agents = require_series(out, "n_agents", default=np.nan)
            # If n_agents is unavailable, infer loosely from n_eff scale; fallback 96.
            n_agents = n_agents.fillna(96.0)
            top1 = require_series(out, "top1_share", default=0.0).fillna(0.0)
            n_eff = require_series(out, "n_eff", default=n_agents).fillna(n_agents)
            mi = require_series(out, "i_exp_norm", default=0.0).fillna(0.0)
            ws = require_series(out, "winner_switch_rate", default=0.0).fillna(0.0)
            pdisp = require_series(out, "polarization_dispersion", default=0.0).fillna(0.0)
            extremity = require_series(out, "stance_extremity", default=0.0).fillna(0.0)
            kappa = require_series(out, "kappa_map", default=np.nan)

            collapse_neff_cut = np.maximum(setting.collapse_neff_abs, setting.collapse_neff_frac * n_agents)
            concentrated_neff_cut = np.maximum(setting.concentrated_neff_abs, setting.concentrated_neff_frac * n_agents)

            out["robust_collapse_signal"] = ((top1 >= setting.collapse_top1) | (n_eff <= collapse_neff_cut)).astype(int)
            out["robust_concentration_signal"] = ((top1 >= setting.concentrated_top1) | (n_eff <= concentrated_neff_cut)).astype(int)
            out["robust_specialization_signal"] = ((mi >= setting.specialization_mi) & (ws >= setting.winner_switch)).astype(int)
            out["robust_strong_specialization_signal"] = ((mi >= setting.strong_specialization_mi) & (ws >= setting.winner_switch)).astype(int)
            out["robust_polarization_signal"] = (
                (pdisp >= setting.polarization_var)
                & (extremity >= setting.stance_extremity)
                & (ws >= setting.polarization_winner_switch)
                & (top1 < setting.collapse_top1)
                & (n_eff <= 0.50 * n_agents)
            ).astype(int)
            # Stable means safely away from the kappa boundary. margin=0 reproduces kappa<0.
            out["robust_locally_contracting"] = (kappa < -float(setting.stable_kappa_margin)).astype(int)
            out["robust_near_critical"] = (kappa.abs() <= float(setting.near_critical_eps)).astype(int)
            out["robust_positive_criticality"] = (kappa > float(setting.stable_kappa_margin)).astype(int)

            collapse = out["robust_collapse_signal"].astype(bool)
            specialized = out["robust_specialization_signal"].astype(bool)
            polarized = out["robust_polarization_signal"].astype(bool)
            concentrated = out["robust_concentration_signal"].astype(bool)

            labels = np.full(len(out), "diffuse_coexistence", dtype=object)
            labels[concentrated.to_numpy()] = "concentrated_coexistence"
            labels[polarized.to_numpy()] = "polarization"
            labels[specialized.to_numpy()] = "contextual_specialization"
            labels[(polarized & specialized).to_numpy()] = "polarized_specialization"
            labels[collapse.to_numpy()] = "monoculture_collapse"
            labels[(collapse & specialized).to_numpy()] = "specialized_collapse"
            out["robust_phase_label"] = labels
            out["robust_phase_code"] = out["robust_phase_label"].map(PHASE_CODES).fillna(-1).astype(int)

            out["robust_stable_coexistence"] = ((out["robust_concentration_signal"] == 0) & (out["robust_locally_contracting"] == 1)).astype(int)
            out["robust_stable_specialization"] = (
                (out["robust_specialization_signal"] == 1)
                & (out["robust_collapse_signal"] == 0)
                & (n_eff >= 3.0)
                & (out["robust_locally_contracting"] == 1)
            ).astype(int)
            out["robust_stable_polarization"] = (
                (out["robust_polarization_signal"] == 1)
                & (out["robust_collapse_signal"] == 0)
                & (n_eff >= 3.0)
                & (out["robust_locally_contracting"] == 1)
            ).astype(int)
            out["robust_descriptor_positive_any"] = (
                (out["robust_collapse_signal"] == 1)
                | (out["robust_specialization_signal"] == 1)
                | (out["robust_polarization_signal"] == 1)
                | (out["robust_concentration_signal"] == 1)
            ).astype(int)
            return out


        def agreement_metrics(base: pd.Series, comp: pd.Series) -> Dict[str, float]:
            b = base.astype(str).fillna("NA")
            c = comp.astype(str).fillna("NA")
            n = len(b)
            if n == 0:
                return {"agreement": np.nan, "flip_rate": np.nan}
            return {
                "agreement": float((b == c).mean()),
                "flip_rate": float((b != c).mean()),
            }


        def boolean_jaccard(a: pd.Series, b: pd.Series) -> float:
            aa = a.fillna(0).astype(int).astype(bool)
            bb = b.fillna(0).astype(int).astype(bool)
            union = (aa | bb).sum()
            if union == 0:
                return 1.0
            return float((aa & bb).sum() / union)


        def summarize_setting(df: pd.DataFrame, setting: ThresholdSetting, baseline_labels: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
            work = apply_thresholds(df, setting)
            rows: List[Dict[str, object]] = []
            row: Dict[str, object] = {"setting_id": setting.setting_id, **asdict(setting), "n_rows": len(work)}

            for col in [
                "robust_collapse_signal", "robust_concentration_signal", "robust_specialization_signal",
                "robust_strong_specialization_signal", "robust_polarization_signal", "robust_locally_contracting",
                "robust_near_critical", "robust_positive_criticality", "robust_stable_coexistence",
                "robust_stable_specialization", "robust_stable_polarization", "robust_descriptor_positive_any",
            ]:
                row[f"{col}_rate"] = float(work[col].mean()) if len(work) else np.nan
                row[f"{col}_count"] = int(work[col].sum()) if len(work) else 0

            for phase in PHASE_ORDER:
                row[f"phase_count__{phase}"] = int((work["robust_phase_label"] == phase).sum())
                row[f"phase_rate__{phase}"] = float((work["robust_phase_label"] == phase).mean()) if len(work) else np.nan

            if baseline_labels is not None and not baseline_labels.empty:
                # Align by row_id when available.
                if "row_id" in work.columns and "row_id" in baseline_labels.columns:
                    aligned = work[["row_id", "robust_phase_label", "robust_stable_specialization", "robust_locally_contracting"]].merge(
                        baseline_labels[["row_id", "base_phase_label", "base_stable_specialization", "base_locally_contracting"]],
                        on="row_id",
                        how="inner",
                    )
                    if not aligned.empty:
                        row.update({f"phase_{k}": v for k, v in agreement_metrics(aligned["base_phase_label"], aligned["robust_phase_label"]).items()})
                        row["stable_specialization_jaccard_vs_baseline"] = boolean_jaccard(aligned["base_stable_specialization"], aligned["robust_stable_specialization"])
                        row["locally_contracting_jaccard_vs_baseline"] = boolean_jaccard(aligned["base_locally_contracting"], aligned["robust_locally_contracting"])

            rows.append(row)

            # Grouped summary by sweep and maybe beta route/support when useful.
            group_cols = [c for c in ["sweep_name", "input_root"] if c in work.columns]
            group_rows: List[Dict[str, object]] = []
            if group_cols:
                for keys, g in work.groupby(group_cols, dropna=False):
                    if not isinstance(keys, tuple):
                        keys = (keys,)
                    grow: Dict[str, object] = {"setting_id": setting.setting_id, **{c: v for c, v in zip(group_cols, keys)}, "n_rows": len(g)}
                    for col in ["robust_locally_contracting", "robust_stable_specialization", "robust_specialization_signal", "robust_collapse_signal", "robust_near_critical"]:
                        grow[f"{col}_rate"] = float(g[col].mean()) if len(g) else np.nan
                        grow[f"{col}_count"] = int(g[col].sum()) if len(g) else 0
                    group_rows.append(grow)
            return pd.DataFrame(rows), pd.DataFrame(group_rows)


        def make_baseline_labels(df: pd.DataFrame) -> pd.DataFrame:
            base = ThresholdSetting(setting_id="baseline")
            work = apply_thresholds(df, base)
            return work[["row_id", "robust_phase_label", "robust_stable_specialization", "robust_locally_contracting"]].rename(columns={
                "robust_phase_label": "base_phase_label",
                "robust_stable_specialization": "base_stable_specialization",
                "robust_locally_contracting": "base_locally_contracting",
            })


        def create_row_id(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            if "row_id" in out.columns:
                return out
            key_cols = [c for c in DEFAULT_ID_COLS if c in out.columns]
            if key_cols:
                # Include original row number to avoid accidental collisions when there
                # are duplicate conditions across files/seeds.
                key_frame = out[key_cols].astype(str).fillna("NA")
                out["row_id"] = key_frame.agg("|".join, axis=1) + "|row=" + pd.Series(np.arange(len(out)), index=out.index).astype(str)
            else:
                out["row_id"] = pd.Series(np.arange(len(out)), index=out.index).astype(str)
            return out


        def descriptor_robustness(df: pd.DataFrame, settings: Sequence[ThresholdSetting], out_dir: Path) -> None:
            if df.empty:
                print("[warn] descriptor robustness skipped: no rows")
                return
            df = create_row_id(coerce_numeric(df))
            baseline_labels = make_baseline_labels(df)

            summary_parts: List[pd.DataFrame] = []
            grouped_parts: List[pd.DataFrame] = []
            for setting in settings:
                summary, grouped = summarize_setting(df, setting, baseline_labels=baseline_labels)
                summary_parts.append(summary)
                if not grouped.empty:
                    grouped_parts.append(grouped)

            summary_df = pd.concat(summary_parts, ignore_index=True, sort=False)
            summary_df.to_csv(out_dir / "threshold_robustness_summary.csv", index=False)
            if grouped_parts:
                grouped_df = pd.concat(grouped_parts, ignore_index=True, sort=False)
                grouped_df.to_csv(out_dir / "threshold_robustness_by_sweep.csv", index=False)

            # Joint descriptor settings are intentionally retained in
            # threshold_robustness_summary.csv and summarized by
            # write_appendix_latex_tables() as a manuscript-ready TeX table.
            # We do not emit separate joint-only CSV files here, to keep the
            # output focused on the appendix tables.

            # Per-row labels for baseline and selected non-baseline settings. Keep this
            # compact by default: baseline plus all one-at-a-time settings. Joint
            # descriptor settings are already summarized above and are deliberately
            # omitted from the large per-row label dump unless needed for debugging.
            label_rows: List[pd.DataFrame] = []
            keep_cols = [c for c in [
                "row_id", "input_root", "input_file", "sweep_name", "source_result_dir", "run_name",
                "eta", "alpha", "beta_sup", "beta_route", "lambda_risk", "context_trait_strength",
                "polarization_strength", "route_power", "top1_share", "n_eff", "i_exp_norm",
                "winner_switch_rate", "kappa_map", "gamma_map", "max_invasion_log",
            ] if c in df.columns]
            for setting in settings:
                if str(setting.setting_id).startswith("joint|"):
                    continue
                labeled = apply_thresholds(df, setting)
                cols = keep_cols + [
                    "robust_phase_label", "robust_phase_code", "robust_collapse_signal",
                    "robust_specialization_signal", "robust_polarization_signal",
                    "robust_locally_contracting", "robust_near_critical",
                    "robust_stable_specialization", "robust_stable_coexistence", "robust_stable_polarization",
                ]
                sub = labeled[cols].copy()
                sub.insert(0, "setting_id", setting.setting_id)
                label_rows.append(sub)
            pd.concat(label_rows, ignore_index=True, sort=False).to_csv(out_dir / "threshold_robustness_labels.csv.gz", index=False, compression="gzip")

            # A small manuscript-friendly table: only one-at-a-time knobs and main outcomes.
            manuscript_cols = [
                "setting_id", "n_rows",
                "phase_agreement", "phase_flip_rate",
                "stable_specialization_jaccard_vs_baseline", "locally_contracting_jaccard_vs_baseline",
                "robust_locally_contracting_rate", "robust_near_critical_rate",
                "robust_stable_specialization_rate", "robust_specialization_signal_rate",
                "robust_collapse_signal_rate",
            ]
            manuscript = summary_df[[c for c in manuscript_cols if c in summary_df.columns]].copy()
            manuscript.to_csv(out_dir / "threshold_robustness_table_for_paper.csv", index=False)


        def parse_setting_from_path(path: str) -> Dict[str, object]:
            """Parse compact sh labels like active=1e-4_fd=1e-5_inv=1e-5_stride=2_K=100."""
            text = str(path)
            out: Dict[str, object] = {}
            patterns = {
                "robust_active_eps": r"active=([0-9eE.+-]+)",
                "robust_fd_eps": r"fd=([0-9eE.+-]+)",
                "robust_invasion_eps": r"inv=([0-9eE.+-]+)",
                "robust_stride": r"stride=([0-9eE.+-]+)",
                "robust_endpoint_last_k": r"K=([0-9eE.+-]+)",
            }
            for key, pat in patterns.items():
                m = re.search(pat, text)
                if m:
                    try:
                        out[key] = float(m.group(1))
                    except Exception:
                        out[key] = m.group(1)
            if not out:
                out["robust_setting_label"] = safe_name(Path(text))
            else:
                out["robust_setting_label"] = "|".join(f"{k.replace('robust_', '')}={v:g}" if isinstance(v, float) else f"{k.replace('robust_', '')}={v}" for k, v in out.items() if k != "robust_setting_label")
            return out


        def numerical_stability_robustness(numerical_phase: pd.DataFrame, numerical_stab: pd.DataFrame, baseline_df: Optional[pd.DataFrame], out_dir: Path) -> None:
            if numerical_phase.empty and numerical_stab.empty:
                print("[warn] numerical stability robustness skipped: no numerical rerun rows")
                return
            num = merge_phase_and_stability(numerical_phase, numerical_stab)
            num = coerce_numeric(num)
            for idx, path in num.get("input_root", pd.Series("", index=num.index)).items():
                parsed = parse_setting_from_path(str(path))
                for k, v in parsed.items():
                    num.loc[idx, k] = v
            if "robust_setting_label" not in num.columns:
                num["robust_setting_label"] = "unknown"

            # Recompute stable labels with baseline descriptor thresholds but measured kappa.
            num = apply_thresholds(num, ThresholdSetting(setting_id="baseline"))

            rows: List[Dict[str, object]] = []
            for setting, g in num.groupby("robust_setting_label", dropna=False):
                row: Dict[str, object] = {"robust_setting_label": setting, "n_rows": len(g)}
                for col in ["kappa_map", "gamma_map", "max_invasion_log", "active_count"]:
                    if col in g.columns:
                        vals = pd.to_numeric(g[col], errors="coerce")
                        row[f"{col}_median"] = float(vals.median())
                        row[f"{col}_mean"] = float(vals.mean())
                        row[f"{col}_std"] = float(vals.std())
                for col in ["robust_locally_contracting", "robust_near_critical", "robust_stable_specialization", "robust_specialization_signal", "robust_collapse_signal"]:
                    row[f"{col}_rate"] = float(g[col].mean()) if len(g) else np.nan
                    row[f"{col}_count"] = int(g[col].sum()) if len(g) else 0
                for parsed_key in ["robust_active_eps", "robust_fd_eps", "robust_invasion_eps", "robust_stride", "robust_endpoint_last_k"]:
                    if parsed_key in g.columns:
                        vals = pd.to_numeric(g[parsed_key], errors="coerce").dropna().unique()
                        if len(vals) == 1:
                            row[parsed_key] = float(vals[0])
                rows.append(row)
            pd.DataFrame(rows).sort_values("robust_setting_label").to_csv(out_dir / "numerical_stability_robustness_summary.csv", index=False)

            # Compare against baseline outputs when possible. Use condition keys; do not
            # include row_id because numerical reruns are different files.
            if baseline_df is not None and not baseline_df.empty:
                base = apply_thresholds(coerce_numeric(baseline_df.copy()), ThresholdSetting(setting_id="baseline"))
                key_cols = [c for c in [
                    "run_name", "eta", "alpha", "beta_sup", "beta_route", "lambda_risk",
                    "hard_safety_threshold", "context_trait_strength", "polarization_strength",
                    "route_power", "route_zscore", "no_feedback", "ablation",
                ] if c in base.columns and c in num.columns]
                if key_cols:
                    base_small = base[key_cols + ["kappa_map", "robust_phase_label", "robust_locally_contracting", "robust_stable_specialization"]].copy()
                    base_small = base_small.rename(columns={
                        "kappa_map": "base_kappa_map",
                        "robust_phase_label": "base_phase_label",
                        "robust_locally_contracting": "base_locally_contracting",
                        "robust_stable_specialization": "base_stable_specialization",
                    })
                    comp = num.merge(base_small.drop_duplicates(), on=key_cols, how="inner")
                    if not comp.empty:
                        comp["delta_kappa_map"] = comp["kappa_map"] - comp["base_kappa_map"]
                        comp["locally_contracting_flip"] = (comp["robust_locally_contracting"].astype(int) != comp["base_locally_contracting"].astype(int)).astype(int)
                        comp["stable_specialization_flip"] = (comp["robust_stable_specialization"].astype(int) != comp["base_stable_specialization"].astype(int)).astype(int)
                        comp["phase_flip"] = (comp["robust_phase_label"].astype(str) != comp["base_phase_label"].astype(str)).astype(int)
                        comp.to_csv(out_dir / "numerical_stability_robustness_matched_rows.csv.gz", index=False, compression="gzip")

                        comp_rows: List[Dict[str, object]] = []
                        for setting, g in comp.groupby("robust_setting_label", dropna=False):
                            comp_rows.append({
                                "robust_setting_label": setting,
                                "n_matched": len(g),
                                "delta_kappa_median": float(pd.to_numeric(g["delta_kappa_map"], errors="coerce").median()),
                                "delta_kappa_abs_median": float(pd.to_numeric(g["delta_kappa_map"], errors="coerce").abs().median()),
                                "delta_kappa_abs_p90": float(pd.to_numeric(g["delta_kappa_map"], errors="coerce").abs().quantile(0.90)),
                                "locally_contracting_flip_rate": float(g["locally_contracting_flip"].mean()),
                                "stable_specialization_flip_rate": float(g["stable_specialization_flip"].mean()),
                                "phase_flip_rate": float(g["phase_flip"].mean()),
                                "kappa_corr_with_baseline": float(g[["kappa_map", "base_kappa_map"]].corr().iloc[0, 1]) if len(g) > 2 else np.nan,
                            })
                        pd.DataFrame(comp_rows).sort_values("robust_setting_label").to_csv(out_dir / "numerical_stability_robustness_vs_baseline.csv", index=False)
                    else:
                        print("[warn] numerical robustness baseline comparison produced no matched rows")
                else:
                    print("[warn] numerical robustness baseline comparison skipped: no common key columns")

            num.to_csv(out_dir / "numerical_stability_robustness_labels.csv.gz", index=False, compression="gzip")


        def write_readme(out_dir: Path, args: argparse.Namespace, n_main: int, n_num: int) -> None:
            main_dirs = "\n".join("- `" + str(p) + "`" for p in args.results_dirs)
            numerical_dirs = "\n".join("- `" + str(p) + "`" for p in args.numerical_results_dirs) if args.numerical_results_dirs else "- none"
            text = (
                "# Threshold robustness outputs\n\n"
                "Generated by `threshold_robustness.py`.\n\n"
                "## Inputs\n\n"
                "Main result directories:\n" + main_dirs + "\n\n"
                "Numerical stability rerun directories:\n" + numerical_dirs + "\n\n"
                f"Main rows loaded: {n_main}\n"
                f"Numerical rerun rows loaded: {n_num}\n\n"
                "## Main outputs\n\n"
                "- `threshold_robustness_summary.csv`: one row per post-hoc threshold setting.\n"
                "- `threshold_robustness_by_sweep.csv`: per-sweep summary when sweep labels are available.\n"
                "- `threshold_robustness_table_for_paper.csv`: compact table for manuscript/appendix.\n"
                "- `threshold_robustness_labels.csv.gz`: per-row robust labels for all threshold settings.\n"
                "- `numerical_stability_robustness_summary.csv`: summary of reruns with changed numerical stability parameters, if provided.\n"
                "- `numerical_stability_robustness_vs_baseline.csv`: matched comparison against baseline, if matching keys are available.\n"
                "- `numerical_stability_robustness_matched_rows.csv.gz`: matched row-level comparison.\n\n"
                "## Interpretation\n\n"
                "Descriptor threshold robustness is post-hoc: it recomputes labels from continuous metrics such as top-one share, N_eff, I_exp, winner-switching, and kappa_map.\n\n"
                "Numerical stability robustness requires reruns because active support, finite-difference step, invasion mass, context stride, and endpoint averaging window can change the estimated kappa_map itself.\n"
            )
            (out_dir / "README_threshold_robustness.md").write_text(text, encoding="utf-8")

        def build_parser() -> argparse.ArgumentParser:
            p = argparse.ArgumentParser(description="Post-hoc and numerical threshold robustness for replay outputs.")
            p.add_argument(
                "--results_dirs",
                nargs="+",
                default=DEFAULT_RESULTS_DIRS,
                help="Result directories/files containing phase_summary.csv and optionally stability_summary.csv.",
            )
            p.add_argument(
                "--numerical_results_dirs",
                nargs="*",
                default=DEFAULT_NUMERICAL_RESULTS_DIRS,
                help="Optional result directories from compact numerical stability robustness reruns.",
            )
            p.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Output directory.")
            p.add_argument("--collapse_top1s", default="0.75,0.8,0.85")
            p.add_argument("--collapse_neff_fracs", default="0.02,0.025,0.03")
            p.add_argument("--concentrated_top1s", default="0.4,0.45,0.5")
            p.add_argument("--concentrated_neff_fracs", default="0.08,0.10,0.12")
            p.add_argument("--specialization_mis", default="0.10,0.12,0.15")
            p.add_argument("--strong_specialization_mis", default="0.12,0.15,0.18")
            p.add_argument("--winner_switches", default="0.25,0.30,0.35")
            p.add_argument("--polarization_vars", default="0.45,0.50,0.55")
            p.add_argument("--stance_extremities", default="0.65,0.70,0.75")
            p.add_argument("--stable_kappa_margins", default="0,0.02,0.05,0.10")
            p.add_argument("--near_critical_epss", default="0.025,0.05,0.10")
            p.add_argument(
                "--joint_descriptor_sensitivity",
                action=argparse.BooleanOptionalAction,
                default=True,
                help="Add a joint sensitivity grid over I_exp, winner-switching, and N_eff thresholds.",
            )
            p.add_argument("--joint_specialization_mis", default="0.05,0.10,0.12,0.15,0.20")
            p.add_argument("--joint_winner_switches", default="0.20,0.25,0.30,0.35,0.40")
            p.add_argument("--joint_concentrated_neff_fracs", default="0.06,0.08,0.10,0.12,0.15")
            p.add_argument("--joint_collapse_neff_fracs", default="0.015,0.020,0.025,0.030,0.040")
            p.add_argument("--full_factorial", action="store_true", help="Use a larger factorial grid over key thresholds instead of one-at-a-time perturbations.")
            return p


        def main(argv: Optional[Sequence[str]] = None) -> None:
            args = build_parser().parse_args(argv)
            out_dir = Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            phase = load_result_dirs(args.results_dirs, "phase_summary.csv", "main")
            stability = load_result_dirs(args.results_dirs, "stability_summary.csv", "main")
            main = merge_phase_and_stability(phase, stability)
            main = coerce_numeric(main)
            print(f"Loaded main rows: phase={len(phase)}, stability={len(stability)}, merged={len(main)}")

            settings = setting_grid(args)
            settings_df = pd.DataFrame([asdict(s) for s in settings])
            settings_df.to_csv(out_dir / "threshold_settings.csv", index=False)
            with open(out_dir / "threshold_settings.json", "w", encoding="utf-8") as f:
                json.dump([asdict(s) for s in settings], f, ensure_ascii=False, indent=2)

            descriptor_robustness(main, settings, out_dir)

            num_phase = load_result_dirs(args.numerical_results_dirs, "phase_summary.csv", "numerical") if args.numerical_results_dirs else pd.DataFrame()
            num_stab = load_result_dirs(args.numerical_results_dirs, "stability_summary.csv", "numerical") if args.numerical_results_dirs else pd.DataFrame()
            if args.numerical_results_dirs:
                print(f"Loaded numerical rows: phase={len(num_phase)}, stability={len(num_stab)}")
                numerical_stability_robustness(num_phase, num_stab, main, out_dir)

            write_readme(out_dir, args, n_main=len(main), n_num=max(len(num_phase), len(num_stab)))
            print(f"Wrote threshold robustness outputs under {out_dir}")
        try:
            result = main(extra_args)
        except SystemExit as exc:
            return _system_exit_code(exc)
        return int(result) if isinstance(result, int) else 0
    finally:
        sys.argv = old_argv


def _run_figures_embedded(extra_args: list[str], *, dry_run: bool = False) -> int:
    script_label = "appendix_figures.py::make_paper_figures"
    print("[run] " + _format_cmd(script_label, extra_args), flush=True)
    if dry_run:
        return 0
    old_argv = sys.argv[:]
    sys.argv = [script_label, *extra_args]
    __file__ = str(_repo_root() / "analysis" / "make_paper_figures.py")
    try:
        """
        Create polished, paper-ready figures from theory-aligned replay outputs.

        Compared with the first version, this script:
        - uses publication-oriented typography and spacing
        - saves PDF figures by default, with optional PNG copies
        - adds panel letters and slice annotations
        - produces a compact overview figure for the main paper
        - overlays specialization / near-critical contours when available
        - writes draft captions for direct use in the manuscript

        Usage: python3 pipeline/appendix_figures.py figures   (from data/)
        """

        import argparse
        import json
        from pathlib import Path
        from typing import Dict, Optional, Sequence, Tuple

        import numpy as np
        import pandas as pd

        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "font.size": 16,
            "legend.fontsize": 16,
            "legend.title_fontsize": 16,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "savefig.bbox": "tight",
        })
        from matplotlib.colors import BoundaryNorm, ListedColormap
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # ----------------------
        # Constants / styling
        # ----------------------
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
        PHASE_NAMES = {
            0: "Diffuse coexistence",
            1: "Concentrated coexistence",
            #2: "Monoculture collapse",
            2: "Monoculture",
            3: "Polarization",
            #4: "Contextual specialization",
            4: "Specialization",
            5: "Polarized specialization",
            6: "Metastable switching",
            7: "Specialized collapse",
        }
        PHASE_ORDER = [0, 1, 2, 3, 4, 5, 6, 7]
        PHASE_COLORS = [
            "#999999",  # 0 diffuse coexistence
            "#0072B2",  # 1 concentrated coexistence
            "#D55E00",  # 2 monoculture
            "#CC79A7",  # 3 polarization
            "#009E73",  # 4 specialization
            "#56B4E9",  # 5 polarized specialization
            "#666666",  # 6 metastable switching
            "#E69F00",  # 7 specialized collapse
        ]


        ID_COLS = [
            "eta", "alpha", "beta_sup", "beta_route", "lambda_risk",
            "hard_safety_threshold", "context_trait_strength", "route_power",
            "route_zscore", "no_feedback", "ablation",
        ]

        plt.rcParams.update({
            "font.size": 16,
            "legend.fontsize": 16,
            "legend.title_fontsize": 16,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "savefig.bbox": "tight",
        })

        SAVE_PNG = False
        HEATMAP_MAX_TICKLABELS = 4
        HEATMAP_TICKLABEL_SIZE = 14
        AXIS_LABEL_SIZE = 16
        PANEL_LABEL_SIZE = 16
        TITLE_SIZE = 16
        LEGEND_FONT_SIZE = 13
        LEGEND_TITLE_SIZE = 13
        COLORBAR_LABEL_SIZE = 14
        COLORBAR_TICK_SIZE = 12
        COLORBAR_FRACTION = 0.046
        COLORBAR_PAD = 0.04
        ADJACENT_COLORBAR_WIDTH = "4.6%"
        ADJACENT_COLORBAR_PAD = 0.08
        HEATMAP_ALPHA = 0.9


        # ----------------------
        # CLI / loading
        # ----------------------

        def choose_existing(user_path: str, fallback: Path) -> Path:
            return Path(user_path) if user_path else fallback


        def load_inputs(args: argparse.Namespace):
            results_dir = Path(args.results_dir)
            phase_path = choose_existing(args.phase_csv, results_dir / "phase_summary.csv")
            stability_path = choose_existing(args.stability_csv, results_dir / "stability_summary.csv")
            run_config_path = choose_existing(args.run_config, results_dir / "run_config.json")
            phase = read_results_csv(phase_path, results_dir, "phase_summary.csv")
            if phase is None:
                raise FileNotFoundError(f"Could not find phase summary at {phase_path} or under {results_dir}/*/*/phase_summary.csv")
            stability = read_results_csv(stability_path, results_dir, "stability_summary.csv") if stability_path.exists() else read_results_csv(None, results_dir, "stability_summary.csv")
            config = {}
            if run_config_path.exists():
                with open(run_config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
            return phase, stability, config


        def read_results_csv(path: Optional[Path], results_dir: Path, name: str) -> Optional[pd.DataFrame]:
            """Read one result CSV or merge per-run CSVs below a result root.

            This loader searches recursively, so it supports both flat outputs like
            results/fig0_4_full_grid/run.checkpoints/phase_summary.csv and nested
            outputs like results/fig10_full_grid/specialization/run.checkpoints/phase_summary.csv.
            """
            if path is not None and path.exists():
                return pd.read_csv(path)
            if not results_dir.exists():
                return None

            files = sorted({p.resolve() for p in results_dir.rglob(name) if p.is_file()})
            parts = []
            for p in files:
                try:
                    df = pd.read_csv(p)
                except Exception as exc:
                    print(f"[warn] skipping unreadable {p}: {exc}")
                    continue
                try:
                    rel_parent = p.parent.relative_to(results_dir.resolve())
                except Exception:
                    rel_parent = p.parent
                rel_parts = rel_parent.parts
                sweep_name = rel_parts[0] if len(rel_parts) >= 2 else results_dir.name
                df["source_root"] = results_dir.name
                df["sweep_name"] = sweep_name
                df["source_result_dir"] = p.parent.name
                df["source_result_path"] = str(p.parent)
                parts.append(df)
            if parts:
                return pd.concat(parts, ignore_index=True, sort=False)
            return None


        def load_inputs_from_dirs(args: argparse.Namespace, results_dirs: Sequence[Path]):
            """Load and concatenate result tables from multiple raw output roots."""
            phase_parts = []
            stability_parts = []
            configs = {}
            seen_roots = []
            for root in results_dirs:
                root = Path(root)
                if not root.exists():
                    print(f"[warn] auto-merge input does not exist, skipping: {root}")
                    continue
                phase_df = read_results_csv(None, root, "phase_summary.csv")
                if phase_df is not None and not phase_df.empty:
                    phase_parts.append(phase_df)
                    seen_roots.append(str(root))
                stab_df = read_results_csv(None, root, "stability_summary.csv")
                if stab_df is not None and not stab_df.empty:
                    stability_parts.append(stab_df)
                cfg_path = root / "run_config.json"
                if cfg_path.exists():
                    try:
                        with open(cfg_path, "r", encoding="utf-8") as f:
                            configs[str(root)] = json.load(f)
                    except Exception as exc:
                        print(f"[warn] could not read {cfg_path}: {exc}")
            if not phase_parts:
                roots = ", ".join(str(p) for p in results_dirs)
                raise FileNotFoundError(f"Could not find any phase_summary.csv under: {roots}")
            phase = pd.concat(phase_parts, ignore_index=True, sort=False)
            stability = pd.concat(stability_parts, ignore_index=True, sort=False) if stability_parts else None
            config = {"auto_merged_roots": seen_roots, "run_configs": configs}
            return phase, stability, config


        # ----------------------
        # Data preparation
        # ----------------------
        def ensure_columns(df: pd.DataFrame, cols: Sequence[str], fill=np.nan) -> pd.DataFrame:
            out = df.copy()
            for c in cols:
                if c not in out.columns:
                    out[c] = fill
            return out



        def prepare_stability_df(df: Optional[pd.DataFrame], near_critical_eps: float) -> Optional[pd.DataFrame]:
            if df is None:
                return None
            out = df.copy()
            for c in ID_COLS + ["kappa_map", "gamma_map", "max_invasion_log", "spectral_radius", "phase_code"]:
                if c in out.columns:
                    out[c] = pd.to_numeric(out[c], errors="coerce")
            if "kappa_map" in out.columns:
                out["near_critical"] = (out["kappa_map"].abs() <= near_critical_eps).astype(int)
            if "locally_contracting" in out.columns:
                lc = out["locally_contracting"].astype(str).str.lower().isin(["true", "1"])
                out["locally_contracting"] = lc.astype(int)
                out["locally_unstable_or_invadable"] = (~lc).astype(int)
            return out


        def seed_average(df: pd.DataFrame, value_cols: Sequence[str]) -> pd.DataFrame:
            work = ensure_columns(df, ID_COLS)
            value_cols = [c for c in value_cols if c in work.columns]
            return work.groupby(ID_COLS, dropna=False, as_index=False)[value_cols].mean() if value_cols else work[ID_COLS].drop_duplicates()


        def phase_mode(df: pd.DataFrame) -> pd.DataFrame:
            work = ensure_columns(df, ID_COLS + ["phase_code"])
            def _mode(s: pd.Series):
                counts = s.value_counts(dropna=True)
                if counts.empty:
                    return np.nan
                maxc = counts.max()
                winners = sorted(counts[counts == maxc].index.tolist())
                return winners[0]
            out = work.groupby(ID_COLS, dropna=False, as_index=False)["phase_code"].agg(_mode)
            out["phase_code"] = out["phase_code"].astype("Int64")
            return out



        def available_value(df: pd.DataFrame, col: str, target: Optional[float]) -> Optional[float]:
            if col not in df.columns:
                return None
            vals = sorted(pd.to_numeric(df[col], errors="coerce").dropna().unique().tolist())
            if not vals:
                return None
            if target is None:
                return vals[0]
            return min(vals, key=lambda x: abs(float(x) - float(target)))



        def filter_slice(df: pd.DataFrame, spec: Dict[str, Optional[float]]) -> pd.DataFrame:
            out = df.copy()
            for c, v in spec.items():
                if c in out.columns and v is not None:
                    out = out[np.isclose(pd.to_numeric(out[c], errors="coerce"), float(v), equal_nan=False)]
            return out


        def pivot_eta_alpha(df: pd.DataFrame, value_col: str):
            tmp = df[["alpha", "eta", value_col]].copy()
            tmp["alpha"] = pd.to_numeric(tmp["alpha"], errors="coerce")
            tmp["eta"] = pd.to_numeric(tmp["eta"], errors="coerce")
            tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
            pt = tmp.pivot_table(index="alpha", columns="eta", values=value_col, aggfunc="mean").sort_index().sort_index(axis=1)
            return pt


        def fill_phase_runs_between_cyan(pt: pd.DataFrame) -> pd.DataFrame:
            """Display phase-1 runs as phase-2 when they are horizontally bracketed by phase-2."""
            out = pt.copy()
            for row_label in out.index:
                vals = out.loc[row_label].to_numpy(dtype=float)
                j = 0
                while j < len(vals):
                    if vals[j] != 1:
                        j += 1
                        continue
                    k = j
                    while k + 1 < len(vals) and vals[k + 1] == 1:
                        k += 1
                    if j > 0 and k < len(vals) - 1 and vals[j - 1] == 2 and vals[k + 1] == 2:
                        out.iloc[out.index.get_loc(row_label), j:k + 1] = 2
                    j = k + 1
            return out


        def add_panel_label(ax, label: str, *, xytext: Tuple[float, float] = (-24, 8), fontsize: float = PANEL_LABEL_SIZE):
            ax.annotate(
                label,
                xy=(0, 1),
                xycoords="axes fraction",
                xytext=xytext,
                textcoords="offset points",
                fontsize=fontsize,
                fontweight="bold",
                va="bottom",
                ha="left",
                annotation_clip=False,
            )


        def sparse_tick_positions(values, max_ticks: int = HEATMAP_MAX_TICKLABELS) -> np.ndarray:
            n = len(values)
            if n <= 0:
                return np.array([], dtype=float)
            tick_count = min(n, max(1, int(max_ticks)), HEATMAP_MAX_TICKLABELS)
            if tick_count == 1:
                return np.array([0.0], dtype=float)
            # Heatmap cells are drawn at equal index spacing even when the underlying
            # parameter grid is numerically irregular. Keep tick locations strictly
            # equally spaced in image coordinates; labels use the nearest grid value.
            return np.linspace(0, n - 1, tick_count, dtype=float)


        def sparse_tick_label_indices(values, max_ticks: int = HEATMAP_MAX_TICKLABELS) -> np.ndarray:
            n = len(values)
            if n <= 0:
                return np.array([], dtype=int)
            return np.clip(np.rint(sparse_tick_positions(values, max_ticks)).astype(int), 0, n - 1)


        def format_tick_value(value) -> str:
            try:
                if pd.notna(value) and np.isfinite(float(value)):
                    return f"{float(value):g}"
            except Exception:
                pass
            return str(value).replace(".checkpoints", "")


        def set_sparse_index_ticks(
            ax,
            xs,
            ys,
            *,
            max_xticks: int = HEATMAP_MAX_TICKLABELS,
            max_yticks: int = HEATMAP_MAX_TICKLABELS,
            rotation: float = 0,
        ):
            xt = sparse_tick_positions(xs, max_xticks)
            yt = sparse_tick_positions(ys, max_yticks)
            xi = sparse_tick_label_indices(xs, max_xticks)
            yi = sparse_tick_label_indices(ys, max_yticks)
            x_ha = "center" if float(rotation) == 0 else "right"
            ax.set_xticks(xt)
            ax.set_xticklabels([format_tick_value(xs[i]) for i in xi], rotation=rotation, ha=x_ha)
            ax.set_yticks(yt)
            ax.set_yticklabels([format_tick_value(ys[i]) for i in yi])
            ax.tick_params(length=0, labelsize=HEATMAP_TICKLABEL_SIZE)


        def parameter_axis_label(name: str) -> str:
            labels = {
                "eta": r"Selection strength $\eta$",
                "alpha": r"Context fidelity $\alpha$",
                "beta_sup": r"Support strength $\beta_{\mathrm{sup}}$",
                "beta_route": r"Routing strength $\beta_{\mathrm{route}}$",
                "lambda_risk": r"Risk penalty $\lambda_{\mathrm{risk}}$",
                "context_trait_strength": "Context-trait strength",
                "polarization_strength": "Polarization strength",
                "route_power": "Route power",
            }
            return labels.get(str(name), str(name).replace("_", " "))


        def format_heatmap_axes(ax, xs, ys, xlabel=None, ylabel=None):
            if xlabel is None:
                xlabel = parameter_axis_label("eta")
            if ylabel is None:
                ylabel = parameter_axis_label("alpha")
            ax.set_xlabel(xlabel)
            #ax.set_ylabel(ylabel)
            ax.set_box_aspect(1)
            set_sparse_index_ticks(ax, xs, ys, rotation=0)


        def draw_phase_heatmap(ax, pt, title: str):
            cmap = ListedColormap(PHASE_COLORS)
            bounds = np.arange(-0.5, 8.5, 1)
            norm = BoundaryNorm(bounds, cmap.N)
            im = ax.imshow(pt.to_numpy(dtype=float), origin="lower", aspect="auto", cmap=cmap, norm=norm, interpolation="nearest", alpha=HEATMAP_ALPHA)
            format_heatmap_axes(ax, pt.columns, pt.index)
            #ax.set_title(title)
            return im


        def draw_continuous_heatmap(ax, pt, title: str, cmap="Greys", vmin=None, vmax=None, center_zero=False, clip_value=None):
            data = pt.to_numpy(dtype=float)
            if clip_value is not None:
                lim = float(clip_value)
                data = np.clip(data, -lim, lim)
                if center_zero:
                    vmin, vmax = -lim, lim
            elif center_zero:
                finite = data[np.isfinite(data)]
                vmax = np.nanpercentile(np.abs(finite), 95) if finite.size else 1.0
                vmax = max(float(vmax), 1e-12)
                vmin = -vmax
            im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", alpha=HEATMAP_ALPHA)
            format_heatmap_axes(ax, pt.columns, pt.index)
            #ax.set_title(title)
            return im


        def overlay_contour(ax, pt, level=0.5, color="black", linewidth=1.2, linestyle="--"):
            data = pt.to_numpy(dtype=float)
            if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 2:
                return False
            if np.all(np.isnan(data)):
                return False
            ys = np.arange(data.shape[0])
            xs = np.arange(data.shape[1])
            # only contour if range spans the level
            dmin = np.nanmin(data)
            dmax = np.nanmax(data)
            if not np.isfinite(dmin) or not np.isfinite(dmax) or not (dmin <= level <= dmax):
                return False
            ax.contour(xs, ys, data, levels=[level], colors=[color], linewidths=linewidth, linestyles=linestyle)
            return True


        def save_figure(fig, out_dir: Path, stem: str, dpi: int):
            for ax in fig.axes:
                ax.xaxis.label.set_size(AXIS_LABEL_SIZE)
                ax.yaxis.label.set_size(AXIS_LABEL_SIZE)
                ax.title.set_size(TITLE_SIZE)
            fig.savefig(out_dir / f"{stem}.pdf")
            if SAVE_PNG:
                fig.savefig(out_dir / f"{stem}.png", dpi=dpi)
            plt.close(fig)


        def add_colorbar(fig, mappable, ax, **kwargs):
            kwargs.setdefault("fraction", COLORBAR_FRACTION)
            kwargs.setdefault("pad", COLORBAR_PAD)
            cb = fig.colorbar(mappable, ax=ax, **kwargs)
            cb.ax.tick_params(labelsize=COLORBAR_TICK_SIZE, length=2.5, width=0.7)
            cb.outline.set_linewidth(0.7)
            cb.ax.xaxis.label.set_size(COLORBAR_LABEL_SIZE)
            cb.ax.yaxis.label.set_size(COLORBAR_LABEL_SIZE)
            return cb


        def add_adjacent_colorbar(fig, mappable, ax, *, width: str = ADJACENT_COLORBAR_WIDTH, pad: float = ADJACENT_COLORBAR_PAD):
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size=width, pad=pad)
            cb = fig.colorbar(mappable, cax=cax)
            cb.ax.tick_params(labelsize=COLORBAR_TICK_SIZE, length=2.5, width=0.7)
            cb.outline.set_linewidth(0.7)
            cb.ax.xaxis.label.set_size(COLORBAR_LABEL_SIZE)
            cb.ax.yaxis.label.set_size(COLORBAR_LABEL_SIZE)
            return cb




        def format_phase_legend_label(label: str) -> str:
            words = str(label).split()
            return "\n".join(words) if len(words) >= 2 else str(label)


        def metric_label(name: str) -> str:
            labels = {
                "phase_code": "Phase summary",
                "top1_share": "Top-1 share",
                "n_eff": r"Effective support $N_{eff}$",
                "hhi": "HHI",
                "i_exp_norm": r"Context MI $I_{exp}$",
                "winner_switch_rate": "Winner-switch rate",
                "specialization_signal": "Specialization signal",
                "strong_specialization_signal": "Strong specialization signal",
                "kappa_map": r"$\kappa_{map}$",
                "gamma_map": r"$\gamma_{map}$",
                "max_invasion_log": "Max invasion log multiplier",
                "weighted_risk": "Weighted risk",
            }
            return labels.get(str(name), str(name).replace("_", " ").title())


        def metric_cmap(name: str) -> str:
            cmaps = {
                "kappa_map": "RdYlBu_r",
                "top1_share": "Purples",
                "n_eff": "YlGn",
                "hhi": "Oranges",
                "i_exp_norm": "PuBu",
                "winner_switch_rate": "PuBu_r",
                "specialization_signal": "Greens",
                "strong_specialization_signal": "Greens",
                "stable_specialization": "Greens",
                "stable_specialization_indicator": "YlGnBu",
                "stable_coexistence": "Blues",
                "polarization_signal": "Reds",
                "stable_polarization": "Reds",
                "polarization_dispersion": "Reds",
                "gamma_map": "Greys",
                "max_invasion_log": "Greys",
                "delta_weighted_risk_vs_lambda0": "Reds",
            }
            return cmaps.get(str(name), "Greys")


        def phase_legend_handles():
            return [Patch(facecolor=PHASE_COLORS[i], edgecolor="none", label=format_phase_legend_label(PHASE_NAMES[i])) for i in PHASE_ORDER]


        def phase_legend_handles_for_values(values) -> list[Patch]:
            vals = pd.to_numeric(pd.Series(np.asarray(values).ravel()), errors="coerce").dropna()
            present = {int(v) for v in vals if int(v) in PHASE_NAMES}
            present_phases = [i for i in PHASE_ORDER if i in present]
            return [
                Patch(facecolor=PHASE_COLORS[i], edgecolor="none", label=format_phase_legend_label(PHASE_NAMES[i]))
                for i in present_phases
            ]


        def add_phase_legend(ax, values=None, *, loc="center left", bbox_to_anchor=(1.02, 0.5), ncol: Optional[int] = None):
            handles = phase_legend_handles_for_values(values) if values is not None else phase_legend_handles()
            if not handles:
                return None
            if ncol is None:
                ncol = 2 if len(handles) >= 5 else 1
            return ax.legend(
                handles=handles,
                loc=loc,
                bbox_to_anchor=bbox_to_anchor,
                frameon=False,
                fontsize=LEGEND_FONT_SIZE,
                title_fontsize=LEGEND_TITLE_SIZE,
                handlelength=0.9,
                handletextpad=0.45,
                columnspacing=0.8,
                labelspacing=0.35,
                borderaxespad=0.0,
                ncol=ncol,
            )


        def add_compact_legend(ax, *, fontsize: float = LEGEND_FONT_SIZE, **kwargs):
            kwargs.setdefault("frameon", False)
            kwargs.setdefault("fontsize", fontsize)
            kwargs.setdefault("title_fontsize", LEGEND_TITLE_SIZE)
            kwargs.setdefault("handlelength", 1.6)
            kwargs.setdefault("handletextpad", 0.5)
            kwargs.setdefault("labelspacing", 0.35)
            kwargs.setdefault("borderaxespad", 0.0)
            return ax.legend(**kwargs)


        def cmap_with_missing(cmap, missing_color="#d0d0d0"):
            cm = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
            cm = cm.copy()
            cm.set_bad(color=missing_color)
            return cm


        def interpolate_fig5_missing(pt: pd.DataFrame, *, phase_mode: bool = False) -> pd.DataFrame:
            """Fill only interior Fig5 missing cells by interpolating from neighboring grid values."""
            if pt.empty:
                return pt
            out = pt.astype(float).copy()
            missing = out.isna()
            if not missing.to_numpy().any():
                return out

            # First interpolate along eta (columns), then along beta_sup (rows).
            # limit_area="inside" avoids extrapolating missing edge cells.
            out = out.T.interpolate(method="index", axis=0, limit_area="inside").T
            out = out.interpolate(method="index", axis=0, limit_area="inside")

            if phase_mode:
                filled = missing & out.notna()
                out = out.where(~filled, out.round().clip(min(PHASE_ORDER), max(PHASE_ORDER)))
            return out


        def remove_fig5_leftmost_column(pt: pd.DataFrame) -> pd.DataFrame:
            if pt.shape[1] <= 1:
                return pt.copy()
            return pt.iloc[:, 1:].copy()


        # ----------------------
        # Figure builders
        # ----------------------
        def make_main_overview(phase_slice: pd.DataFrame, stab_slice: Optional[pd.DataFrame], out_dir: Path, dpi: int, slice_spec: Dict[str, Optional[float]], kappa_clip: float = 5.0):
            """Main-paper overview.

            Kappa values are clipped for display only.
            """
            fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2), constrained_layout=True)
            axes = axes.ravel()

            pt_phase = pivot_eta_alpha(phase_slice, "phase_code")
            pt_top1 = pivot_eta_alpha(phase_slice, "top1_share")
            pt_mi = pivot_eta_alpha(phase_slice, "i_exp_norm")
            pt_sig = pivot_eta_alpha(phase_slice, "strong_specialization_signal")
            pt_phase_display = fill_phase_runs_between_cyan(pt_phase)

            draw_phase_heatmap(axes[0], pt_phase_display, "Operational phase summary")
            add_panel_label(axes[0], "A")
            overlay_contour(axes[0], pt_sig, level=0.5, color="black", linewidth=2.5, linestyle="--")
            axes[0].set_ylabel(parameter_axis_label("alpha"))
            add_phase_legend(axes[0], pt_phase_display.to_numpy())

            im1 = draw_continuous_heatmap(axes[1], pt_top1, "Top-1 exposure share", cmap=metric_cmap("top1_share"), vmin=0, vmax=1)
            add_panel_label(axes[1], "B")
            cb1 = add_colorbar(fig, im1, ax=axes[1], fraction=0.046, pad=0.04)
            cb1.set_label(metric_label("top1_share"))

            im2 = draw_continuous_heatmap(axes[2], pt_mi, "Context MI $I_{exp}$", cmap=metric_cmap("i_exp_norm"), vmin=0)
            add_panel_label(axes[2], "C")
            cb2 = add_adjacent_colorbar(fig, im2, axes[2], width="4.6%", pad=0.08)
            cb2.set_label(metric_label("i_exp_norm"))

            if stab_slice is not None and not stab_slice.empty and "kappa_map" in stab_slice.columns:
                pt_k = pivot_eta_alpha(stab_slice, "kappa_map")
                im3 = draw_continuous_heatmap(
                    axes[3],
                    pt_k,
                    "Criticality index $\\kappa_{map}$",
                    cmap=metric_cmap("kappa_map"),
                    center_zero=True,
                    clip_value=kappa_clip,
                )
                add_panel_label(axes[3], "D")
                overlay_contour(axes[3], pt_k, level=0.0, color="black", linewidth=1.2, linestyle="--")
                cb3 = add_colorbar(fig, im3, ax=axes[3], fraction=0.046, pad=0.04)
                cb3.set_label(f"$\\kappa_{{map}}$") # (clipped at ±{kappa_clip:g})")
            else:
                pt_ws = pivot_eta_alpha(phase_slice, "winner_switch_rate")
                im3 = draw_continuous_heatmap(axes[3], pt_ws, "Winner-switch rate", cmap=metric_cmap("winner_switch_rate"), vmin=0, vmax=1)
                add_panel_label(axes[3], "D")
                cb3 = add_colorbar(fig, im3, ax=axes[3], fraction=0.046, pad=0.04)
                cb3.set_label("Winner-switch rate")

            axes[2].set_ylabel(parameter_axis_label("alpha"))
            #fig.suptitle(f"Overview slice: {subtitle_from_slice(slice_spec)}", y=1.02)
            save_figure(fig, out_dir, "fig6_representative_no_support_phase_slice", dpi)


        def make_phase_diagram(phase_slice: pd.DataFrame, out_dir: Path, dpi: int, slice_spec: Dict[str, Optional[float]]):
            fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.5), constrained_layout=True)
            pt_phase = pivot_eta_alpha(phase_slice, "phase_code")
            pt_phase_display = fill_phase_runs_between_cyan(pt_phase)
            pt_sig = pivot_eta_alpha(phase_slice, "specialization_signal")
            pt_strong = pivot_eta_alpha(phase_slice, "strong_specialization_signal")

            draw_phase_heatmap(axes[0], pt_phase_display, "Operational phase summary")
            add_panel_label(axes[0], "A")
            overlay_contour(axes[0], pt_sig, level=0.5, color="white", linewidth=2.5, linestyle="--")
            overlay_contour(axes[0], pt_strong, level=0.5, color="black", linewidth=2.5, linestyle="-")
            axes[0].set_ylabel(parameter_axis_label("alpha"))

            im = draw_continuous_heatmap(axes[1], pt_strong, "Strong specialization signal", cmap=metric_cmap("strong_specialization_signal"), vmin=0, vmax=1)
            add_panel_label(axes[1], "B")
            cb = add_colorbar(fig, im, ax=axes[1], fraction=0.046, pad=0.04)
            cb.set_label("probability across seeds")


            handles = phase_legend_handles_for_values(pt_phase_display.to_numpy())
            if handles:
                add_phase_legend(axes[0], pt_phase_display.to_numpy())
            #fig.suptitle(f"Figure 1. Phase diagram and specialization evidence ({subtitle_from_slice(slice_spec)})")
            save_figure(fig, out_dir, "fig9_operational_phase_diagram", dpi)


        def make_stability_figure(stab_slice: pd.DataFrame, out_dir: Path, dpi: int, slice_spec: Dict[str, Optional[float]], kappa_clip: float = 5.0):
            """Main stability figure with display clipping for readability."""
            fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.2), constrained_layout=True)
            panels = [
                ("kappa_map", "Criticality index $\\kappa_{map}$", metric_cmap("kappa_map"), True, kappa_clip, f"$\\kappa_{{map}}$"), # (clipped at ±{kappa_clip:g})
                ("gamma_map", "Tangent growth $\\gamma_{map}$", metric_cmap("gamma_map"), True, 1.0, "$\\gamma_{map}$"),
                ("max_invasion_log", "Max invasion log multiplier", metric_cmap("max_invasion_log"), True, kappa_clip, f"invasion log"), # (clipped at ±{kappa_clip:g})
            ]
            for i, (col, title, cmap, center_zero, clip_value, cbar_label) in enumerate(panels):
                pt = pivot_eta_alpha(stab_slice, col)
                im = draw_continuous_heatmap(
                    axes[i],
                    pt,
                    title,
                    cmap=cmap,
                    center_zero=center_zero,
                    clip_value=clip_value,
                )
                add_panel_label(axes[i], chr(ord('A') + i))
                if col == "kappa_map":
                    overlay_contour(axes[i], pt, level=0.0, color="black", linewidth=1.2, linestyle="--")
                cb = add_colorbar(fig, im, ax=axes[i], fraction=0.046, pad=0.04)
                cb.set_label(cbar_label)
            axes[0].set_ylabel(parameter_axis_label("alpha"))
            #fig.suptitle(f"Local stability and empirical criticality ({subtitle_from_slice(slice_spec)})", y=1.03)
            save_figure(fig, out_dir, "fig7_local_stability_empirical_criticality", dpi)


        def make_support_figure(phase_avg: pd.DataFrame, stab_avg: Optional[pd.DataFrame], args: argparse.Namespace, out_dir: Path, dpi: int, kappa_clip: float = 5.0):
            route_val = available_value(phase_avg, "beta_route", args.support_beta_route)
            alpha_val = available_value(phase_avg, "alpha", args.support_alpha)
            lambda_val = available_value(phase_avg, "lambda_risk", args.support_lambda_risk)
            hard_val = available_value(phase_avg, "hard_safety_threshold", args.support_hard_safety_threshold)
            df = phase_avg.copy()
            if stab_avg is not None and not stab_avg.empty:
                keys = [c for c in ID_COLS if c in df.columns and c in stab_avg.columns]
                add_cols = [c for c in ["kappa_map", "max_invasion_log"] if c in stab_avg.columns and c not in df.columns]
                if keys and add_cols:
                    df = df.merge(stab_avg[keys + add_cols].drop_duplicates(), on=keys, how="left")
            for c, v in {
                "beta_route": route_val,
                "alpha": alpha_val,
                "lambda_risk": lambda_val,
                "hard_safety_threshold": hard_val,
            }.items():
                if c in df.columns and v is not None:
                    df = df[np.isclose(pd.to_numeric(df[c], errors="coerce"), float(v), equal_nan=False)]
            if df.empty or df["beta_sup"].nunique() < 2:
                return

            panels = [
                ("phase_code", "Phase summary", True, None, None, None, "phase_code"),
                ("top1_share", "Top-1 share", False, metric_cmap("top1_share"), 0, 1, "top1_share"),
            ]
            if "kappa_map" in df.columns:
                panels.append(("kappa_map", "Criticality index $\\kappa_{map}$", False, metric_cmap("kappa_map"), -kappa_clip, kappa_clip, f"$\\kappa_{{map}}$"))
            if "max_invasion_log" in df.columns:
                panels.append(("max_invasion_log", "Max invasion log multiplier", False, metric_cmap("max_invasion_log"), -kappa_clip, kappa_clip, f"invasion log"))

            n_panels = 4 if len(panels) >= 4 else len(panels)
            if n_panels == 4:
                fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), constrained_layout=True)
            else:
                fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.8), constrained_layout=True, sharex=True)
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])
            axes = axes.ravel()
            for i, (col, title, phase_mode, cmap_name, vmin, vmax, cbar_label) in enumerate(panels[:n_panels]):
                tmp = df[["beta_sup", "eta", col]].copy()
                for c in ["beta_sup", "eta", col]:
                    tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
                pt = tmp.pivot_table(index="beta_sup", columns="eta", values=col, aggfunc="mean").sort_index().sort_index(axis=1)
                pt = interpolate_fig5_missing(pt, phase_mode=phase_mode)
                pt = remove_fig5_leftmost_column(pt)
                data = pt.to_numpy(dtype=float)
                if phase_mode:
                    data = np.where(np.isclose(data, 3, equal_nan=False), 2, data)
                    cmap = cmap_with_missing(ListedColormap(PHASE_COLORS))
                    im = axes[i].imshow(data, origin="lower", aspect="auto", cmap=cmap, norm=BoundaryNorm(np.arange(-0.5, 8.5, 1), len(PHASE_COLORS)), interpolation="nearest", alpha=HEATMAP_ALPHA)
                else:
                    if col in {"kappa_map", "max_invasion_log"}:
                        data = np.clip(data, -float(kappa_clip), float(kappa_clip))
                    im = axes[i].imshow(data, origin="lower", aspect="auto", cmap=cmap_with_missing(cmap_name or "Greys"), vmin=vmin, vmax=vmax, interpolation="nearest", alpha=HEATMAP_ALPHA)
                #axes[i].set_title(title)
                axes[i].set_xlabel(parameter_axis_label("eta"))
                #axes[i].set_ylabel("Support strength $\\beta_{sup}$")
                axes[i].set_box_aspect(1)
                set_sparse_index_ticks(axes[i], pt.columns, pt.index, max_xticks=6, max_yticks=6, rotation=0)
                add_panel_label(axes[i], chr(ord('A') + i))
                if phase_mode:
                    add_phase_legend(axes[i], data)
                else:
                    if n_panels == 4 and i == 2:
                        cb = add_adjacent_colorbar(fig, im, axes[i], width="4.6%", pad=0.08)
                    else:
                        cb = add_colorbar(fig, im, ax=axes[i], fraction=0.046, pad=0.04)
                    #cb.set_label(cbar_label)
                    cb.set_label(title)
            axes[0].set_ylabel(parameter_axis_label("beta_sup"))
            if n_panels == 4:
                axes[2].set_ylabel(parameter_axis_label("beta_sup"))

            #fig.suptitle(f"Figure 5. Support intervention at α={alpha_val:g}, β_route={route_val:g}, λ_risk={lambda_val:g}")
            save_figure(fig, out_dir, "fig8_support_intervention", dpi)


        # ----------------------
        # Appendix figure builders
        # ----------------------


        def make_appendix_route_sweep(phase_avg: pd.DataFrame, args: argparse.Namespace, out_dir: Path, dpi: int):
            """Route-strength sweep using existing beta_route variation."""
            df = phase_avg.copy()
            slice_spec = {
                "alpha": available_value(df, "alpha", args.appendix_alpha),
                "beta_sup": available_value(df, "beta_sup", args.phase_beta_sup),
                "lambda_risk": available_value(df, "lambda_risk", args.phase_lambda_risk),
                "hard_safety_threshold": available_value(df, "hard_safety_threshold", args.phase_hard_safety_threshold),
            }
            base_df = df.copy()
            for c, v in slice_spec.items():
                if c in df.columns and v is not None:
                    df = df[np.isclose(pd.to_numeric(df[c], errors="coerce"), float(v), equal_nan=False)]
            if df.empty or df["beta_route"].nunique() < 2:
                fallback = base_df.copy()
                for c in ["alpha", "lambda_risk", "hard_safety_threshold"]:
                    v = slice_spec.get(c)
                    if c in fallback.columns and v is not None:
                        fallback = fallback[np.isclose(pd.to_numeric(fallback[c], errors="coerce"), float(v), equal_nan=False)]
                if "beta_sup" not in fallback.columns:
                    return
                route_counts = fallback.groupby("beta_sup")["beta_route"].nunique()
                candidate_beta = pd.to_numeric(route_counts[route_counts >= 2].index.to_series(), errors="coerce").dropna()
                if candidate_beta.empty:
                    return
                target = args.phase_beta_sup if args.phase_beta_sup is not None else candidate_beta.iloc[0]
                beta_val = float(candidate_beta.iloc[np.argmin(np.abs(candidate_beta.to_numpy(dtype=float) - float(target)))])
                df = fallback[np.isclose(pd.to_numeric(fallback["beta_sup"], errors="coerce"), beta_val, equal_nan=False)].copy()
                slice_spec["beta_sup"] = beta_val

            fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.2), constrained_layout=True, sharey=True)
            panels = [
                ("phase_code", "Phase summary", True, None, None),
                ("i_exp_norm", "Context MI $I_{exp}$", False, 0, None),
                ("top1_share", "Top-1 share", False, 0, 1),
            ]
            for i, (col, title, phase_mode, vmin, vmax) in enumerate(panels):
                tmp = df[["beta_route", "eta", col]].copy()
                for c in ["beta_route", "eta", col]:
                    tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
                pt = tmp.pivot_table(index="beta_route", columns="eta", values=col, aggfunc="mean").sort_index().sort_index(axis=1)
                if phase_mode:
                    im = axes[i].imshow(pt.to_numpy(dtype=float), origin="lower", aspect="auto",
                                        cmap=ListedColormap(PHASE_COLORS),
                                        norm=BoundaryNorm(np.arange(-0.5, 8.5, 1), len(PHASE_COLORS)),
                                        interpolation="nearest", alpha=HEATMAP_ALPHA)
                else:
                    panel_cmap = metric_cmap(col)
                    im = axes[i].imshow(pt.to_numpy(dtype=float), origin="lower", aspect="auto", cmap=panel_cmap,
                                        vmin=vmin, vmax=vmax, interpolation="nearest", alpha=HEATMAP_ALPHA)
                #axes[i].set_title(title)
                axes[i].set_xlabel(parameter_axis_label("eta"))
                axes[i].set_box_aspect(1)
                set_sparse_index_ticks(axes[i], pt.columns, pt.index, max_xticks=6, max_yticks=6, rotation=0)
                add_panel_label(axes[i], chr(ord("A") + i))
                if phase_mode:
                    add_phase_legend(axes[i], pt.to_numpy())
                else:
                    cb = add_colorbar(fig, im, ax=axes[i], fraction=0.046, pad=0.04)
                    cb.set_label(title)
            #fig.suptitle(f"Appendix A2. Routing-strength sweep at α={slice_spec.get('alpha'):g}, β_sup={slice_spec.get('beta_sup'):g}, λ_risk={slice_spec.get('lambda_risk'):g}")
            axes[0].set_ylabel(parameter_axis_label("beta_route"))
            save_figure(fig, out_dir, "fig13_route_sweep", dpi)


        # ---------------------------------------------------------------------------
        # Figure extensions
        # ---------------------------------------------------------------------------

        ID_COLS = [
            "eta", "alpha", "beta_sup", "beta_route", "lambda_risk",
            "hard_safety_threshold", "context_trait_strength", "polarization_strength",
            "route_power", "route_zscore", "no_feedback", "ablation",
        ]


        def parse_args() -> argparse.Namespace:
            p = argparse.ArgumentParser(description="Create the appendix figures.")
            p.add_argument("--results_dir", default="results/fig0_4_full_grid",
                           help="Input directory for Fig0-4 main-slice figures.")
            p.add_argument("--figS_results_dir", default="results/specialization_wide_dense_figS",
                           help="Input directory for FigS stable-specialization heatmap; main figures use --results_dir.")
            p.add_argument("--other_results_dir", default="results/phase_experiments_merged",
                           help="Optional pre-merged directory for all figures except Fig0-4 and FigS. If it does not exist, the script auto-merges --results_dir, --fig5_results_dir, and --fig10_results_dir in memory.")
            p.add_argument("--fig5_results_dir", default="results/fig5_full_grid",
                           help="Raw Fig5/support-grid output directory. Used automatically when --other_results_dir does not exist.")
            p.add_argument("--fig10_results_dir", default="results/fig10_full_grid",
                           help="Raw Fig10/targeted-specialization output directory. Used automatically when --other_results_dir does not exist.")
            p.add_argument("--appendix_route_results_dir", default="results/appendix_route_sweep_parallel",
                           help="Raw Appendix A2 routing-sweep output directory. If present, Appendix A2 is generated from this directory instead of the generic merged inputs.")
            p.add_argument("--disable_auto_merge", action="store_true", default=False,
                           help="Disable automatic in-memory merging of raw result directories.")
            p.add_argument("--phase_csv", default="")
            p.add_argument("--stability_csv", default="")
            p.add_argument("--run_config", default="")
            p.add_argument("--early_warning_eval_csv", default="")
            p.add_argument("--early_warning_predictions_csv", default="")
            p.add_argument("--phase_validation_csv", default="")
            p.add_argument("--out_dir", default=DEFAULT_FIGURES_OUT_DIR)
            p.add_argument("--phase_beta_sup", type=float, default=0.0)
            p.add_argument("--phase_beta_route", type=float, default=2.0)
            p.add_argument("--phase_lambda_risk", type=float, default=0.0)
            p.add_argument("--phase_hard_safety_threshold", type=float, default=-1.0)
            p.add_argument("--phase_context_trait_strength", type=float, default=None)
            p.add_argument("--phase_polarization_strength", type=float, default=None)
            p.add_argument("--support_alpha", type=float, default=1.0)
            p.add_argument("--support_beta_route", type=float, default=2.0)
            p.add_argument("--support_lambda_risk", type=float, default=0.0)
            p.add_argument("--support_hard_safety_threshold", type=float, default=-1.0)
            p.add_argument("--near_critical_eps", type=float, default=0.05)
            p.add_argument("--dpi", type=int, default=300)
            p.add_argument("--save_png", action="store_true", default=False,
                           help="Also save PNG copies. By default, only PDF figures are written.")
            p.add_argument("--make_appendix", action="store_true", default=True)
            p.add_argument("--appendix_alpha", type=float, default=1.0)
            p.add_argument("--kappa_clip", type=float, default=5.0)
            p.add_argument("--main_kappa_clip", type=float, default=5.0)
            p.add_argument("--spec_heatmap_alpha", type=float, default=1.0,
                           help="Fixed alpha for the stable-specialization appendix heatmap.")
            p.add_argument("--spec_heatmap_beta_route", type=float, default=2.0,
                           help="Fixed beta_route for the stable-specialization appendix heatmap.")
            p.add_argument("--spec_heatmap_route_power", type=float, default=1.5,
                           help="Fixed route_power for the stable-specialization appendix heatmap.")
            p.add_argument("--spec_heatmap_context_trait_strength", type=float, default=1.0,
                           help="Fixed context_trait_strength for the stable-specialization appendix heatmap. Use NaN-like shell omission only by editing the script if you want aggregation over this dimension.")
            p.add_argument("--make_trait_exposure", action="store_true", default=True,
                           help="Create a trait-space exposure map when endpoint_exposures.csv.gz files are available.")
            p.add_argument("--skip_trait_exposure", action="store_true", default=False,
                           help="Skip the trait-space exposure map even if endpoint exposure files are available.")
            p.add_argument("--trait_exposure_results_dirs", default="results/trait_exposure_examples",
                           help="Comma-separated roots to search for endpoint_exposures.csv.gz.")
            p.add_argument("--trait_exposure_x", default="stance_valence",
                           help="Trait coordinate for the x-axis. The script expects an endpoint column named trait_<value>.")
            p.add_argument("--trait_exposure_y", default="social_framing",
                           help="Trait coordinate for the y-axis. The script expects an endpoint column named trait_<value>.")
            p.add_argument("--trait_exposure_bins", type=int, default=32,
                           help="Number of bins per axis for the exposure-weighted density layer.")
            p.add_argument("--trait_exposure_max_panels", type=int, default=4,
                           help="Maximum number of phase example panels to show in the trait-space figure.")
            p.add_argument("--trait_exposure_snapshot_step", type=int, default=100,
                           help="Use exposure_snapshots.csv.gz at this step_index for trait-space maps. Negative means endpoint_exposures.csv.gz.")
            p.add_argument("--trait_exposure_alpha_clip", type=float, default=25.0,
                           help="Clip exposure/uniform-exposure ratio before mapping to point alpha.")
            p.add_argument("--trait_exposure_min_alpha", type=float, default=0.1,
                           help="Minimum point alpha in trait-space exposure maps.")
            p.add_argument("--trait_exposure_max_alpha", type=float, default=0.5,
                           help="Maximum point alpha in trait-space exposure maps.")
            p.add_argument("--trait_exposure_point_size", type=float, default=40.0,
                           help="Fixed point size for trait-space exposure maps; exposure is encoded only by alpha.")
            return p.parse_args()


        def prepare_phase_df(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            for c in ID_COLS + [
                "phase_code", "specialization_signal", "strong_specialization_signal",
                "collapse_signal", "concentration_signal", "polarization_signal",
                "hhi", "n_eff", "top1_share", "abs_polarization", "polarization_dispersion",
                "stance_extremity", "i_exp_norm", "winner_switch_rate", "weighted_risk", "n_agents",
                "context_trait_strength", "polarization_strength",
            ]:
                if c in out.columns:
                    out[c] = pd.to_numeric(out[c], errors="coerce")

            mi = pd.to_numeric(out.get("i_exp_norm", np.nan), errors="coerce")
            ws = pd.to_numeric(out.get("winner_switch_rate", np.nan), errors="coerce")
            top1 = pd.to_numeric(out.get("top1_share", np.nan), errors="coerce")
            neff = pd.to_numeric(out.get("n_eff", np.nan), errors="coerce")
            n_agents = pd.to_numeric(out.get("n_agents", np.nan), errors="coerce")
            pdisp = pd.to_numeric(out.get("polarization_dispersion", np.nan), errors="coerce")
            extremity = pd.to_numeric(out.get("stance_extremity", np.nan), errors="coerce")
            abs_pol = pd.to_numeric(out.get("abs_polarization", np.nan), errors="coerce")

            if "specialization_signal" not in out.columns:
                out["specialization_signal"] = ((mi >= 0.12) & (ws >= 0.30)).astype(int)
            if "strong_specialization_signal" not in out.columns:
                out["strong_specialization_signal"] = ((mi >= 0.15) & (ws >= 0.30)).astype(int)
            if "collapse_signal" not in out.columns:
                out["collapse_signal"] = ((top1 >= 0.8) | (neff <= np.maximum(1.5, 0.025 * n_agents.fillna(np.inf)))).astype(int)
            if "concentration_signal" not in out.columns:
                out["concentration_signal"] = ((top1 >= 0.45) | (neff <= np.maximum(3.0, 0.10 * n_agents.fillna(np.inf)))).astype(int)
            if "polarization_signal" not in out.columns:
                out["polarization_signal"] = ((pdisp >= 0.50) & (extremity >= 0.70) & (ws >= 0.20) & (top1 < 0.80) & (neff >= 3.0)).astype(int)
            return out


        def build_aggregates(phase: pd.DataFrame, stability: Optional[pd.DataFrame]):
            phase_vals = [
                "hhi", "n_eff", "top1_share", "abs_polarization", "polarization_dispersion", "stance_extremity",
                "i_exp_norm", "winner_switch_rate", "weighted_risk", "specialization_signal",
                "strong_specialization_signal", "collapse_signal", "concentration_signal", "polarization_signal"
            ]
            phase_avg = seed_average(phase, phase_vals).merge(phase_mode(phase), on=ID_COLS, how="left")

            stab_avg = None
            if stability is not None:
                stab_vals = [
                    "kappa_map", "gamma_map", "max_invasion_log", "spectral_radius",
                    "locally_contracting", "locally_unstable_or_invadable", "near_critical"
                ]
                stab_avg = seed_average(stability, stab_vals)
                if "phase_code" in stability.columns:
                    stab_avg = stab_avg.merge(phase_mode(stability), on=ID_COLS, how="left")
            return phase_avg, stab_avg


        def choose_slice(df: pd.DataFrame, beta_sup: float, beta_route: float, lambda_risk: float, hard_thr: float, ctx: Optional[float] = None, pol: Optional[float] = None):
            spec = {
                "beta_sup": available_value(df, "beta_sup", beta_sup),
                "beta_route": available_value(df, "beta_route", beta_route),
                "lambda_risk": available_value(df, "lambda_risk", lambda_risk),
                "hard_safety_threshold": available_value(df, "hard_safety_threshold", hard_thr),
            }
            if ctx is not None and "context_trait_strength" in df.columns:
                spec["context_trait_strength"] = available_value(df, "context_trait_strength", ctx)
            if pol is not None and "polarization_strength" in df.columns:
                spec["polarization_strength"] = available_value(df, "polarization_strength", pol)
            return spec


        def merge_phase_stability_columns(phase: pd.DataFrame, stability: Optional[pd.DataFrame]) -> pd.DataFrame:
            out = phase.copy()
            if stability is None or stability.empty:
                return out
            add_cols = [c for c in ["kappa_map", "gamma_map", "max_invasion_log", "locally_contracting"] if c in stability.columns and c not in out.columns]
            if not add_cols:
                return out
            keys = [
                c for c in ID_COLS + ["run_name", "source_result_dir", "sweep_name"]
                if c in out.columns and c in stability.columns
            ]
            if not keys:
                return out
            return out.merge(stability[keys + add_cols].drop_duplicates(), on=keys, how="left")


        def add_stable_phase_flags(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            for col in [
                "kappa_map", "top1_share", "n_eff", "i_exp_norm", "winner_switch_rate",
                "polarization_dispersion", "stance_extremity", "collapse_signal",
                "concentration_signal", "specialization_signal", "polarization_signal",
            ]:
                if col in out.columns:
                    out[col] = pd.to_numeric(out[col], errors="coerce")
            if "kappa_map" not in out.columns:
                out["kappa_map"] = np.nan
            out["locally_contracting"] = out["kappa_map"] < 0
            out["stable_coexistence"] = (
                (out["concentration_signal"].fillna(1) == 0)
                & out["locally_contracting"]
            ).astype(int)
            out["stable_specialization"] = (
                (out["specialization_signal"].fillna(0) == 1)
                & (out["top1_share"] < 0.80)
                & (out["n_eff"] >= 3.0)
                & out["locally_contracting"]
            ).astype(int)
            out["stable_polarization"] = (
                (out["polarization_signal"].fillna(0) == 1)
                & (out["top1_share"] < 0.80)
                & (out["n_eff"] >= 3.0)
                & out["locally_contracting"]
            ).astype(int)
            out["invadable_specialization"] = (
                (out["specialization_signal"].fillna(0) == 1)
                & (out["kappa_map"] >= 0)
            ).astype(int)
            out["stable_collapse"] = (
                (out["collapse_signal"].fillna(0) == 1)
                & out["locally_contracting"]
            ).astype(int)
            out["invadable_collapse"] = (
                (out["collapse_signal"].fillna(0) == 1)
                & (out["kappa_map"] >= 0)
            ).astype(int)
            return out


        def nearest_filter(df: pd.DataFrame, col: str, target: Optional[float]) -> pd.DataFrame:
            if target is None or col not in df.columns:
                return df
            val = available_value(df, col, target)
            if val is None:
                return df
            return df[np.isclose(pd.to_numeric(df[col], errors="coerce"), float(val), equal_nan=False)]


        # ---------------------------------------------------------------------------
        # Trait-space exposure maps
        # ---------------------------------------------------------------------------


        # ---------------------------------------------------------------------------
        # Phase-boundary uncertainty figure
        # ---------------------------------------------------------------------------

        def _seed_identifier(df: pd.DataFrame) -> Optional[str]:
            """Return the best available column identifying independent seeds/runs."""
            for col in ["seed", "run_name", "source_result_dir", "source_result_path"]:
                if col in df.columns and df[col].nunique(dropna=True) >= 2:
                    return col
            return None


        def _stable_region_edges(xs: np.ndarray, probs: np.ndarray, level: float = 0.5) -> Tuple[float, float]:
            """Return first/last x in the interpolated region where probs >= level."""
            xs = np.asarray(xs, dtype=float)
            probs = np.asarray(probs, dtype=float)
            ok = np.isfinite(xs) & np.isfinite(probs)
            xs, probs = xs[ok], probs[ok]
            if xs.size < 2:
                return np.nan, np.nan
            order = np.argsort(xs)
            xs, probs = xs[order], probs[order]
            grid = np.linspace(float(xs.min()), float(xs.max()), 1001)
            yy = np.interp(grid, xs, probs)
            mask = yy >= float(level)
            if not np.any(mask):
                return np.nan, np.nan
            return float(grid[mask][0]), float(grid[mask][-1])


        def _bootstrap_stable_specialization_region(
            df: pd.DataFrame,
            *,
            row_col: str,
            x_col: str,
            seed_col: str,
            indicator_col: str = "stable_specialization_indicator",
            level: float = 0.5,
            n_boot: int = 1000,
            rng_seed: int = 23,
        ) -> pd.DataFrame:
            """Bootstrap entry/exit eta values for the stable-specialization region."""
            needed = [row_col, x_col, seed_col, indicator_col]
            if any(c not in df.columns for c in needed):
                return pd.DataFrame(columns=[row_col, "entry_point", "entry_mean", "entry_lo", "entry_hi", "exit_point", "exit_mean", "exit_lo", "exit_hi"])
            work = df[needed].copy()
            for c in [row_col, x_col, indicator_col]:
                work[c] = pd.to_numeric(work[c], errors="coerce")
            work = work.dropna(subset=[row_col, x_col, seed_col, indicator_col])
            if work.empty:
                return pd.DataFrame(columns=[row_col, "entry_point", "entry_mean", "entry_lo", "entry_hi", "exit_point", "exit_mean", "exit_lo", "exit_hi"])

            rows = np.array(sorted(work[row_col].dropna().unique()), dtype=float)
            xs = np.array(sorted(work[x_col].dropna().unique()), dtype=float)
            seeds = np.array(sorted(work[seed_col].dropna().astype(str).unique()), dtype=object)
            if rows.size < 2 or xs.size < 2 or seeds.size < 2:
                return pd.DataFrame(columns=[row_col, "entry_point", "entry_mean", "entry_lo", "entry_hi", "exit_point", "exit_mean", "exit_lo", "exit_hi"])

            seed_mats = []
            for s in seeds:
                sdf = work[work[seed_col].astype(str) == str(s)]
                pt = sdf.pivot_table(index=row_col, columns=x_col, values=indicator_col, aggfunc="mean").reindex(index=rows, columns=xs)
                seed_mats.append(pt.to_numpy(dtype=float))
            mats = np.stack(seed_mats, axis=0)

            mean_mat = np.nanmean(mats, axis=0)
            point_entries, point_exits = [], []
            for i in range(len(rows)):
                entry, exit_ = _stable_region_edges(xs, mean_mat[i, :], level=level)
                point_entries.append(entry)
                point_exits.append(exit_)

            rng = np.random.default_rng(rng_seed)
            boot_entry = np.full((n_boot, len(rows)), np.nan, dtype=float)
            boot_exit = np.full((n_boot, len(rows)), np.nan, dtype=float)
            for b in range(n_boot):
                sample_idx = rng.integers(0, len(seeds), size=len(seeds))
                mat = np.nanmean(mats[sample_idx, :, :], axis=0)
                for i in range(len(rows)):
                    entry, exit_ = _stable_region_edges(xs, mat[i, :], level=level)
                    boot_entry[b, i] = entry
                    boot_exit[b, i] = exit_

            out = pd.DataFrame({row_col: rows, "entry_point": point_entries, "exit_point": point_exits})
            out["entry_mean"] = np.nanmean(boot_entry, axis=0)
            out["entry_lo"] = np.nanpercentile(boot_entry, 2.5, axis=0)
            out["entry_hi"] = np.nanpercentile(boot_entry, 97.5, axis=0)
            out["exit_mean"] = np.nanmean(boot_exit, axis=0)
            out["exit_lo"] = np.nanpercentile(boot_exit, 2.5, axis=0)
            out["exit_hi"] = np.nanpercentile(boot_exit, 97.5, axis=0)
            return out


        def _prepare_specialization_boundary_slice(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
            """Filter FigS/specialization data to the support-vs-selection calibration slice."""
            sdf = df.copy()
            if sdf.empty:
                return sdf
            if "sweep_name" in sdf.columns:
                mask = sdf["sweep_name"].astype(str).str.contains("specialization", case=False, na=False)
                if mask.any():
                    sdf = sdf[mask].copy()
            for col, target in [
                ("alpha", args.spec_heatmap_alpha),
                ("beta_route", args.spec_heatmap_beta_route),
                ("route_power", args.spec_heatmap_route_power),
                ("context_trait_strength", args.spec_heatmap_context_trait_strength),
                ("polarization_strength", 0.0),
            ]:
                sdf = nearest_filter(sdf, col, target)
            return sdf


        def make_appendix_phase_boundary_uncertainty(
            stable_phase_df: pd.DataFrame,
            stability: Optional[pd.DataFrame],
            args: argparse.Namespace,
            out_dir: Path,
            dpi: int,
            slice_spec: Optional[Dict[str, Optional[float]]] = None,
            kappa_clip: float = 5.0,
        ):
            """Appendix A5: uncertainty of the support-stabilized specialization boundary.

            This figure intentionally uses the same support-vs-selection calibration
            slice as FigS, not the no-support alpha-vs-eta slice.  The no-support slice
            is almost everywhere positive-critical in the current data, so it does not
            provide an informative boundary-uncertainty panel.
            """
            df = stable_phase_df.copy()
            if df.empty or "kappa_map" not in df.columns:
                print("[warn] skipping support-specialization boundary uncertainty: kappa_map unavailable.")
                return
            seed_col = _seed_identifier(df)
            if seed_col is None:
                print("[warn] skipping support-specialization boundary uncertainty: no seed/run column available.")
                return

            df = _prepare_specialization_boundary_slice(df, args)
            if df.empty or "eta" not in df.columns or "beta_sup" not in df.columns:
                print("[warn] skipping support-specialization boundary uncertainty: specialization slice unavailable.")
                return
            if df["eta"].nunique(dropna=True) < 2 or df["beta_sup"].nunique(dropna=True) < 2:
                print("[warn] skipping support-specialization boundary uncertainty: too few eta/beta_sup grid points.")
                return

            for col in [
                "eta", "beta_sup", "specialization_signal", "kappa_map", "top1_share", "n_eff",
                "winner_switch_rate", "i_exp_norm", "stable_specialization",
            ]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Strict, seed-level stable-specialization indicator, matched to FigS.
            if all(c in df.columns for c in ["specialization_signal", "kappa_map", "top1_share", "n_eff", "winner_switch_rate", "i_exp_norm"]):
                df["stable_specialization_indicator"] = (
                    (df["specialization_signal"].fillna(0) > 0)
                    & (df["kappa_map"] < 0)
                    & (df["top1_share"] < 0.60)
                    & (df["n_eff"] > 5.0)
                    & (df["winner_switch_rate"] > 0.30)
                    & (df["i_exp_norm"] > 0.12)
                ).astype(float)
            elif "stable_specialization" in df.columns:
                df["stable_specialization_indicator"] = (df["stable_specialization"].fillna(0) > 0).astype(float)
            else:
                print("[warn] skipping support-specialization boundary uncertainty: no stable-specialization indicator can be computed.")
                return

            stable_rate = (
                df.pivot_table(index="beta_sup", columns="eta", values="stable_specialization_indicator", aggfunc="mean")
                .sort_index()
                .sort_index(axis=1)
            )
            mean_kappa = (
                df.pivot_table(index="beta_sup", columns="eta", values="kappa_map", aggfunc="mean")
                .sort_index()
                .sort_index(axis=1)
            )
            uncertainty = 4.0 * stable_rate * (1.0 - stable_rate)
            boundary = _bootstrap_stable_specialization_region(
                df,
                row_col="beta_sup",
                x_col="eta",
                seed_col=seed_col,
                indicator_col="stable_specialization_indicator",
                level=0.5,
                n_boot=1000,
                rng_seed=29,
            )

            fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.2), constrained_layout=True)

            im0 = draw_continuous_heatmap(
                axes[0],
                stable_rate,
                "Stable specialization probability",
                cmap=metric_cmap("stable_specialization_indicator"),
                vmin=0,
                vmax=1,
            )
            overlay_contour(axes[0], stable_rate, level=0.5, color="black", linewidth=1.0, linestyle="-")
            add_panel_label(axes[0], "A")
            axes[0].set_ylabel(parameter_axis_label("beta_sup"))
            axes[0].set_xlabel(parameter_axis_label("eta"))
            cb0 = add_colorbar(fig, im0, ax=axes[0], fraction=0.046, pad=0.04)
            cb0.set_label("Stable specialization rate")

            im1 = draw_continuous_heatmap(
                axes[1],
                mean_kappa,
                "Mean criticality",
                cmap=metric_cmap("kappa_map"),
                center_zero=True,
                clip_value=kappa_clip,
            )
            overlay_contour(axes[1], mean_kappa, level=0.0, color="black", linewidth=1.2, linestyle="--")
            overlay_contour(axes[1], stable_rate, level=0.5, color="black", linewidth=1.0, linestyle="-")
            add_compact_legend(
                axes[1],
                fontsize=15,
                handles=[
                    Line2D([0], [0], color="black", linewidth=1.2, linestyle="--", label=r"$\kappa_{map}=0$"),
                    Line2D([0], [0], color="black", linewidth=1.0, linestyle="-", label="Stable boundary"),
                ],
                loc="lower left",
                handlelength=1.8,
            )
            add_panel_label(axes[1], "B")
            axes[1].set_ylabel("")
            axes[1].set_xlabel(parameter_axis_label("eta"))
            cb1 = add_colorbar(fig, im1, ax=axes[1], fraction=0.046, pad=0.04)
            cb1.set_label(r"$\kappa_{map}$")

            im2 = draw_continuous_heatmap(
                axes[2],
                uncertainty,
                "Seed disagreement",
                cmap="Greys",
                vmin=0,
                vmax=1,
            )
            overlay_contour(axes[2], stable_rate, level=0.5, color="black", linewidth=1.0, linestyle="-")
            add_panel_label(axes[2], "C")
            axes[2].set_ylabel("")
            axes[2].set_xlabel(parameter_axis_label("eta"))
            cb2 = add_colorbar(fig, im2, ax=axes[2], fraction=0.046, pad=0.04)
            cb2.set_label("Seed disagreement")

            save_figure(fig, out_dir, "fig17_specialization_boundary_uncertainty", dpi)

            stable_rate.to_csv(out_dir / "specialization_boundary_stable_rate.csv")
            mean_kappa.to_csv(out_dir / "specialization_boundary_mean_kappa.csv")
            uncertainty.to_csv(out_dir / "specialization_boundary_seed_disagreement.csv")
            if not boundary.empty:
                boundary.to_csv(out_dir / "specialization_boundary_bootstrap_entry_exit_ci.csv", index=False)


        def _load_prepared_from_dir(base_args: argparse.Namespace, results_dir: Path):
            """Load and prepare phase/stability tables from a specific result directory."""
            local_args = argparse.Namespace(**{**vars(base_args), "results_dir": str(results_dir)})
            phase, stability, config = load_inputs(local_args)
            phase = prepare_phase_df(phase)
            stability = prepare_stability_df(stability, base_args.near_critical_eps)
            phase_avg, stab_avg = build_aggregates(phase, stability)
            return phase, stability, config, phase_avg, stab_avg


        def _load_prepared_from_dirs(base_args: argparse.Namespace, results_dirs: Sequence[Path]):
            """Load, concatenate, and prepare result tables from multiple raw roots."""
            phase, stability, config = load_inputs_from_dirs(base_args, results_dirs)
            phase = prepare_phase_df(phase)
            stability = prepare_stability_df(stability, base_args.near_critical_eps)
            phase_avg, stab_avg = build_aggregates(phase, stability)
            return phase, stability, config, phase_avg, stab_avg


        def resolve_results_dir(path_like: str) -> Path:
            """Resolve result roots for both the package root and data/ as working directory."""
            path = Path(path_like)
            if path.exists() or path.is_absolute():
                return path
            script_relative = Path(__file__).resolve().parent / path
            return script_relative if script_relative.exists() else path


        def main():
            nonlocal SAVE_PNG
            args = parse_args()
            SAVE_PNG = bool(getattr(args, "save_png", False))

            # Source split:
            #   Fig0-4: --results_dir
            #   FigS:   --figS_results_dir
            #   All other figures/tables/captions: --other_results_dir
            main_results_dir = resolve_results_dir(args.results_dir)
            other_results_dir = resolve_results_dir(getattr(args, "other_results_dir", "")) if getattr(args, "other_results_dir", "") else main_results_dir
            figS_results_dir = resolve_results_dir(getattr(args, "figS_results_dir", "")) if getattr(args, "figS_results_dir", "") else main_results_dir
            appendix_route_results_dir = resolve_results_dir(getattr(args, "appendix_route_results_dir", ""))

            out_dir = Path(args.out_dir) if args.out_dir else other_results_dir / "figures"
            out_dir.mkdir(parents=True, exist_ok=True)

            # Load main-slice data only for Fig0-4.
            phase, stability, config, phase_avg, stab_avg = _load_prepared_from_dir(args, main_results_dir)

            # Load phase-experiment data for everything except Fig0-4 and FigS.
            # If a pre-merged directory exists, use it. Otherwise, auto-merge the raw
            # outputs produced by replay_main_grids.sh in memory, so no shell symlink step is
            # required.
            if other_results_dir.exists() and other_results_dir != main_results_dir:
                other_phase, other_stability, other_config, other_phase_avg, other_stab_avg = _load_prepared_from_dir(args, other_results_dir)
            elif not bool(getattr(args, "disable_auto_merge", False)):
                auto_roots = [
                    main_results_dir,
                    resolve_results_dir(getattr(args, "fig5_results_dir", "results/fig5_full_grid")),
                    resolve_results_dir(getattr(args, "fig10_results_dir", "results/fig10_full_grid")),
                ]
                print("[info] --other_results_dir was not found; auto-merging raw roots:")
                for root in auto_roots:
                    print(f"       {root}")
                other_phase, other_stability, other_config, other_phase_avg, other_stab_avg = _load_prepared_from_dirs(args, auto_roots)
            else:
                other_phase, other_stability, other_config, other_phase_avg, other_stab_avg = phase, stability, config, phase_avg, stab_avg

            # ----- Fig0-4 from main_slice_fine_collapse_merged -----
            slice_spec = choose_slice(
                phase_avg,
                beta_sup=args.phase_beta_sup,
                beta_route=args.phase_beta_route,
                lambda_risk=args.phase_lambda_risk,
                hard_thr=args.phase_hard_safety_threshold,
                ctx=args.phase_context_trait_strength,
                pol=args.phase_polarization_strength,
            )
            phase_slice = filter_slice(phase_avg, slice_spec)
            if phase_slice.empty:
                # Fall back to the broadest available slice rather than failing after a large run.
                slice_spec = choose_slice(phase_avg, args.phase_beta_sup, args.phase_beta_route, args.phase_lambda_risk, args.phase_hard_safety_threshold)
                phase_slice = filter_slice(phase_avg, slice_spec)
            if phase_slice.empty:
                raise RuntimeError(f"Representative slice produced no rows for Fig0-4: {slice_spec}")
            stab_slice = filter_slice(stab_avg, slice_spec) if stab_avg is not None else None

            make_main_overview(phase_slice, stab_slice, out_dir, args.dpi, slice_spec, kappa_clip=float(args.main_kappa_clip))
            make_phase_diagram(phase_slice, out_dir, args.dpi, slice_spec)
            if stab_slice is not None and not stab_slice.empty:
                make_stability_figure(stab_slice, out_dir, args.dpi, slice_spec, kappa_clip=float(args.main_kappa_clip))

            # ----- Everything else from phase_experiments_merged -----
            other_slice_spec = choose_slice(
                other_phase_avg,
                beta_sup=args.phase_beta_sup,
                beta_route=args.phase_beta_route,
                lambda_risk=args.phase_lambda_risk,
                hard_thr=args.phase_hard_safety_threshold,
                ctx=args.phase_context_trait_strength,
                pol=args.phase_polarization_strength,
            )
            other_phase_slice = filter_slice(other_phase_avg, other_slice_spec)
            if other_phase_slice.empty:
                other_slice_spec = choose_slice(other_phase_avg, args.phase_beta_sup, args.phase_beta_route, args.phase_lambda_risk, args.phase_hard_safety_threshold)
                other_phase_slice = filter_slice(other_phase_avg, other_slice_spec)
            if other_phase_slice.empty:
                raise RuntimeError(f"Representative slice produced no rows for other figures: {other_slice_spec}")
            other_stab_slice = filter_slice(other_stab_avg, other_slice_spec) if other_stab_avg is not None else None

            make_support_figure(other_phase_avg, other_stab_avg, args, out_dir, args.dpi, kappa_clip=float(args.main_kappa_clip))
            if getattr(args, "make_appendix", True):
                if appendix_route_results_dir and appendix_route_results_dir.exists():
                    _, _, _, route_phase_avg, _ = _load_prepared_from_dir(args, appendix_route_results_dir)
                    make_appendix_route_sweep(route_phase_avg, args, out_dir, args.dpi)
                else:
                    make_appendix_route_sweep(other_phase_avg, args, out_dir, args.dpi)


            # ----- FigS from specialization_wide_6h_merged -----
            if figS_results_dir.exists() and figS_results_dir != main_results_dir and figS_results_dir != other_results_dir:
                figS_phase, figS_stability, _, _, _ = _load_prepared_from_dir(args, figS_results_dir)
            elif figS_results_dir == other_results_dir:
                figS_phase, figS_stability = other_phase, other_stability
            else:
                figS_phase, figS_stability = phase, stability
            stable_phase_df = add_stable_phase_flags(merge_phase_stability_columns(figS_phase, figS_stability))
            if getattr(args, "make_appendix", True):
                make_appendix_phase_boundary_uncertainty(
                    stable_phase_df,
                    figS_stability,
                    args,
                    out_dir,
                    args.dpi,
                    slice_spec=None,
                    kappa_clip=float(args.main_kappa_clip),
                )

            with open(out_dir / "paper_figure_config.json", "w", encoding="utf-8") as f:
                json.dump({
                    "fig0_4_results_dir": str(main_results_dir),
                    "other_results_dir": str(other_results_dir),
                    "figS_results_dir": str(figS_results_dir),
                    "appendix_route_results_dir": str(appendix_route_results_dir),
                    "out_dir": str(out_dir),
                    "fig0_4_representative_slice": slice_spec,
                    "other_representative_slice": other_slice_spec,
                    "support_alpha": args.support_alpha,
                    "support_beta_route": args.support_beta_route,
                    "support_lambda_risk": args.support_lambda_risk,
                    "support_hard_safety_threshold": args.support_hard_safety_threshold,
                    "near_critical_eps": args.near_critical_eps,
                    "loaded_run_config_main": config,
                    "loaded_run_config_other": other_config,
                }, f, ensure_ascii=False, indent=2)

            print(f"Wrote paper figures under {out_dir}")

        try:
            result = main()
        except SystemExit as exc:
            return _system_exit_code(exc)
        return int(result) if isinstance(result, int) else 0
    finally:
        sys.argv = old_argv


# ---------------------------------------------------------------------------
# Integrated post-processing extensions
# ---------------------------------------------------------------------------

def _cli_value(args: list[str], name: str, default: str | None = None) -> str | None:
    flag = f"--{name}"
    for i, item in enumerate(args):
        if item == flag and i + 1 < len(args):
            return args[i + 1]
        if item.startswith(flag + "="):
            return item.split("=", 1)[1]
    return default


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_csv_any(path: Path):
    import pandas as pd
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"[warn] failed to read {path}: {exc}")
        return None


def make_phantom_breakdown_outputs(threshold_out_dir: Path, *, near_critical_eps: float = 0.05, write_tables: bool = False) -> None:
    import numpy as np
    import pandas as pd

    labels_path = threshold_out_dir / "threshold_robustness_labels.csv.gz"
    labels = _read_csv_any(labels_path)
    if labels is None or labels.empty:
        print(f"[warn] phantom breakdown skipped: missing {labels_path}")
        return

    base = labels[labels["setting_id"].astype(str) == "baseline"].copy() if "setting_id" in labels.columns else labels.copy()
    if base.empty:
        base = labels.copy()

    for col in [
        "kappa_map", "gamma_map", "max_invasion_log", "robust_locally_contracting",
        "robust_near_critical", "robust_specialization_signal", "robust_polarization_signal",
        "robust_stable_specialization", "robust_stable_coexistence", "robust_stable_polarization",
        "robust_collapse_signal", "robust_concentration_signal",
    ]:
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")

    phase = base.get("robust_phase_label", pd.Series("", index=base.index)).astype(str)
    spec_signal = base.get("robust_specialization_signal", pd.Series(0, index=base.index)).fillna(0).astype(int)
    pol_signal = base.get("robust_polarization_signal", pd.Series(0, index=base.index)).fillna(0).astype(int)
    stable_spec = base.get("robust_stable_specialization", pd.Series(0, index=base.index)).fillna(0).astype(int)
    locally_contracting = base.get("robust_locally_contracting", pd.Series(0, index=base.index)).fillna(0).astype(int)
    kappa = pd.to_numeric(base.get("kappa_map", pd.Series(np.nan, index=base.index)), errors="coerce")
    gamma = pd.to_numeric(base.get("gamma_map", pd.Series(np.nan, index=base.index)), errors="coerce")
    invasion = pd.to_numeric(base.get("max_invasion_log", pd.Series(np.nan, index=base.index)), errors="coerce")

    coexistence_descriptor = phase.isin(["diffuse_coexistence", "concentrated_coexistence"])
    specialization_descriptor = spec_signal == 1
    polarization_descriptor = pol_signal == 1
    any_diverse_descriptor = coexistence_descriptor | specialization_descriptor | polarization_descriptor

    base["phantom_family"] = "none"
    base.loc[coexistence_descriptor, "phantom_family"] = "coexistence_descriptor_positive"
    base.loc[specialization_descriptor, "phantom_family"] = "specialization_descriptor_positive"
    base.loc[coexistence_descriptor & specialization_descriptor, "phantom_family"] = "coexistence_and_specialization_descriptor_positive"
    base["phantom_any"] = (any_diverse_descriptor & (locally_contracting == 0)).astype(int)
    base["phantom_specialization"] = (specialization_descriptor & (stable_spec == 0)).astype(int)
    base["phantom_coexistence"] = (coexistence_descriptor & (locally_contracting == 0)).astype(int)

    failure_mode = np.full(len(base), "not_descriptor_positive", dtype=object)
    descriptor = any_diverse_descriptor.to_numpy()
    stable = descriptor & (locally_contracting.to_numpy() == 1)
    failure_mode[stable] = "stable_locally_contracting"
    not_stable = descriptor & ~stable
    near = not_stable & (kappa.abs().to_numpy() <= float(near_critical_eps))
    failure_mode[near] = "near_critical"
    gamma_vals = gamma.fillna(-np.inf).to_numpy()
    inv_vals = invasion.fillna(-np.inf).to_numpy()
    tangent = not_stable & ~near & (gamma_vals >= 0) & (gamma_vals >= inv_vals)
    invadable = not_stable & ~near & ~tangent & (inv_vals >= 0)
    pos_unclassified = not_stable & ~near & ~tangent & ~invadable & (kappa.to_numpy() > 0)
    failure_mode[tangent] = "active_face_tangent_instability"
    failure_mode[invadable] = "inactive_agent_invasion"
    failure_mode[pos_unclassified] = "positive_criticality_unclassified"
    failure_mode[not_stable & ~near & ~tangent & ~invadable & ~pos_unclassified] = "unstable_unclassified"
    base["phantom_failure_mode"] = failure_mode

    out_dir = _ensure_dir(threshold_out_dir)
    fig_dir = _ensure_dir(Path(DEFAULT_FIGURES_OUT_DIR))
    if write_tables:
        base.to_csv(out_dir / "phantom_breakdown_labels.csv.gz", index=False, compression="gzip")

    rows = []
    families = {
        "coexistence_descriptor_positive": coexistence_descriptor,
        "specialization_descriptor_positive": specialization_descriptor,
        "any_diverse_descriptor_positive": any_diverse_descriptor,
    }
    modes = [
        "stable_locally_contracting",
        "active_face_tangent_instability",
        "inactive_agent_invasion",
        "near_critical",
        "positive_criticality_unclassified",
        "unstable_unclassified",
    ]
    for family, mask in families.items():
        fam = base[mask].copy()
        denom = len(fam)
        for mode in modes:
            count = int((fam["phantom_failure_mode"] == mode).sum()) if denom else 0
            rows.append({
                "family": family,
                "failure_mode": mode,
                "count": count,
                "rate_within_family": float(count / denom) if denom else np.nan,
                "n_family": int(denom),
            })
    summary = pd.DataFrame(rows)
    compact = summary.pivot_table(index="family", columns="failure_mode", values="rate_within_family", fill_value=0.0)
    compact_counts = summary.pivot_table(index="family", columns="failure_mode", values="count", fill_value=0)

    def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
        if n <= 0:
            return (np.nan, np.nan)
        p = k / n
        denom = 1.0 + z * z / n
        center = (p + z * z / (2.0 * n)) / denom
        half = z * np.sqrt((p * (1.0 - p) / n) + (z * z / (4.0 * n * n))) / denom
        return (float(max(0.0, center - half)), float(min(1.0, center + half)))

    phantom_diversity_order = [
        "coexistence",
        "specialization",
        "any_diversity",
    ]
    phantom_diversity_labels = {
        "coexistence": "Coexistence",
        "specialization": "Specialization",
        "any_diversity": "Any diversity",
    }
    fig8_families = [
        ("coexistence", coexistence_descriptor),
        ("specialization", specialization_descriptor),
        ("any_diversity", any_diverse_descriptor),
    ]
    fig8_rows = []
    failure_inv_modes = {
        "active_face_tangent_instability",
        "inactive_agent_invasion",
        "positive_criticality_unclassified",
    }
    for family, mask in fig8_families:
        fam = base[mask].copy()
        n = int(len(fam))
        stable_mask = (
            (fam.get("robust_locally_contracting", pd.Series(0, index=fam.index)).fillna(0).astype(int) == 1)
            & (fam.get("robust_collapse_signal", pd.Series(0, index=fam.index)).fillna(0).astype(int) == 0)
        )
        stable_count = int(stable_mask.sum()) if n else 0
        mode = fam.get("phantom_failure_mode", pd.Series("", index=fam.index)).astype(str)
        kappa_failure_mask = (~stable_mask) & mode.isin(failure_inv_modes)
        kappa_failure_count = int(kappa_failure_mask.sum()) if n else 0
        collapse_other_count = max(0, int(n - stable_count - kappa_failure_count))
        phantom_count = n - stable_count
        lo, hi = _wilson_ci(phantom_count, n)
        fig8_rows.append({
            "phase_family": family,
            "n_descriptor_positive": n,
            "kappa_stable_count": stable_count,
            "kappa_failure_or_invadable_count": kappa_failure_count,
            "collapse_or_other_count": collapse_other_count,
            "kappa_stable_share": float(stable_count / n) if n else np.nan,
            "kappa_failure_or_invadable_share": float(kappa_failure_count / n) if n else np.nan,
            "collapse_or_other_share": float(collapse_other_count / n) if n else np.nan,
            "phantom_diversity_share": float(phantom_count / n) if n else np.nan,
            "phantom_diversity_ci_low": lo,
            "phantom_diversity_ci_high": hi,
        })
    fig8_summary = pd.DataFrame(fig8_rows)
    fig8_summary.to_csv(out_dir / "fig4_phantom_diversity_by_phase_family.csv", index=False)
    fig8_mode_order = [
        "stable_locally_contracting",
        "active_face_tangent_instability",
        "inactive_agent_invasion",
        "near_critical",
        "positive_criticality_unclassified",
        "unstable_unclassified",
        "not_descriptor_positive",
    ]
    fig8_breakdown_rows = []
    for family, mask in fig8_families:
        fam = base[mask].copy()
        denom = len(fam)
        mode = fam.get("phantom_failure_mode", pd.Series("", index=fam.index)).astype(str)
        for failure_mode in fig8_mode_order:
            count = int((mode == failure_mode).sum()) if denom else 0
            fig8_breakdown_rows.append({
                "phase_family": family,
                "failure_mode": failure_mode,
                "count": count,
                "rate_within_family": float(count / denom) if denom else np.nan,
                "n_family": int(denom),
            })
    fig8_breakdown = pd.DataFrame(fig8_breakdown_rows)
    fig8_compact = fig8_breakdown.pivot_table(
        index="phase_family",
        columns="failure_mode",
        values="rate_within_family",
        fill_value=0.0,
    )
    fig8_counts = fig8_breakdown.pivot_table(
        index="phase_family",
        columns="failure_mode",
        values="count",
        fill_value=0,
    )
    if write_tables:
        summary.to_csv(out_dir / "phantom_breakdown_by_failure_mode.csv", index=False)
        compact.to_csv(out_dir / "phantom_breakdown_rates_for_paper.csv")
        compact_counts.to_csv(out_dir / "phantom_breakdown_counts_for_paper.csv")

    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        plt.rcParams.update({
            "font.size": 16,
            "legend.fontsize": 16,
            "legend.title_fontsize": 16,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "savefig.bbox": "tight",
        })
        order = [f for f in families.keys() if f in compact.index]
        mode_order = [m for m in modes if m in compact.columns]
        mode_labels = {
            "stable_locally_contracting": "Locally contracting",
            "active_face_tangent_instability": "Active-face tangent instability",
            "inactive_agent_invasion": "Inactive-agent invasion",
            "near_critical": "Near critical",
            "positive_criticality_unclassified": "Positive criticality",
            "unstable_unclassified": "Unstable, unclassified",
            "not_descriptor_positive": "Not diversity descriptor",
        }
        cmap = plt.get_cmap("Set2")
        mode_colors = {mode: cmap(i % cmap.N) for i, mode in enumerate([*mode_order, "not_descriptor_positive"])}
        legend_kwargs = {
            "frameon": False,
            "fontsize": 11,
            "handlelength": 1.2,
            "handletextpad": 0.45,
            "columnspacing": 0.8,
            "labelspacing": 0.35,
            "borderaxespad": 0.0,
        }
#

        fig8_order = [f for f in phantom_diversity_order if f in fig8_counts.index]
        fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.8), constrained_layout=True)
        stable_counts = fig8_counts.reindex(index=fig8_order, columns=["stable_locally_contracting"], fill_value=0)["stable_locally_contracting"].to_numpy(dtype=float)
        total_counts = fig8_counts.reindex(index=fig8_order, columns=fig8_mode_order, fill_value=0).sum(axis=1).to_numpy(dtype=float)
        non_stable_counts = np.maximum(total_counts - stable_counts, 0.0)
        x = np.arange(len(fig8_order))
        axes[0].bar(x, stable_counts, width=0.68, label="locally contracting", color="#264653")
        axes[0].bar(x, non_stable_counts, width=0.68, bottom=stable_counts, label="not locally contracting", color="#E9C46A")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([phantom_diversity_labels.get(s, s.replace("_", " ")) for s in fig8_order], rotation=15, ha="right")
        axes[0].set_ylabel("Endpoint count")
        handles_a, labels_a = axes[0].get_legend_handles_labels()
        fig.legend(handles_a, labels_a, loc="upper center", bbox_to_anchor=(0.20, 1.10), ncol=1, **legend_kwargs)
        axes[0].text(-0.18, 1.04, "A", transform=axes[0].transAxes, fontsize=16, fontweight="bold")

        fig8_plot_df = fig8_compact.reindex(index=fig8_order, columns=fig8_mode_order, fill_value=0.0)
        fig8_plot_df = fig8_plot_df.loc[:, fig8_plot_df.sum(axis=0) > 0]
        fig8_plot_df.rename(columns=mode_labels).plot(
            kind="bar",
            stacked=True,
            ax=axes[1],
            width=0.68,
            legend=False,
            color=[mode_colors[m] for m in fig8_plot_df.columns],
        )
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel("Share")
        axes[1].set_xlabel("")
        axes[1].set_xticklabels([phantom_diversity_labels.get(s, s.replace("_", " ")) for s in fig8_order], rotation=15, ha="right")
        axes[1].text(-0.18, 1.04, "B", transform=axes[1].transAxes, fontsize=16, fontweight="bold")
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.64, 1.10), ncol=2, **legend_kwargs)

        fig8_count_df = fig8_counts.reindex(index=fig8_order, columns=fig8_plot_df.columns, fill_value=0)
        fig8_count_df.rename(columns=mode_labels).plot(
            kind="bar",
            stacked=True,
            ax=axes[2],
            width=0.68,
            legend=False,
            color=[mode_colors[m] for m in fig8_count_df.columns],
        )
        axes[2].set_ylabel("Endpoint count")
        axes[2].set_xlabel("")
        axes[2].set_xticklabels([phantom_diversity_labels.get(s, s.replace("_", " ")) for s in fig8_order], rotation=15, ha="right")
        axes[2].text(-0.18, 1.04, "C", transform=axes[2].transAxes, fontsize=16, fontweight="bold")
        fig.savefig(fig_dir / "fig4_phantom_diversity_rate.pdf", bbox_inches="tight")
        plt.close(fig)
        old_appendix_path = fig_dir / "appendix_phantom_diversity_breakdown.pdf"
        if old_appendix_path.exists():
            old_appendix_path.unlink()
    except Exception as exc:
        print(f"[warn] phantom breakdown plots skipped: {exc}")
    print(f"[ok] wrote phantom breakdown outputs under {threshold_out_dir}")


def _cleanup_unneeded_intermediate_files(out_dir: Path) -> None:
    """Remove known non-plot intermediate files from an output directory.

    This leaves PDF/PNG figures and caption/config files intact. It is used
    only for generated analysis-output directories, not source result roots.
    """
    candidates = [
        "early_warning_eval_for_caption.csv",
        "early_warning_main_text_table.csv",
        "early_warning_feature_group_ablation.csv",
        "early_warning_feature_importance_inputs.csv",
        "early_warning_feature_importance.csv",
        "early_warning_top_features_for_paper.csv",
        "phase_boundary_bootstrap_ci.csv",
        "phase_boundary_phase_entropy.csv",
        "phase_boundary_positive_criticality_probability.csv",
        "specialization_boundary_bootstrap_entry_exit_ci.csv",
        "specialization_boundary_mean_kappa.csv",
        "specialization_boundary_seed_disagreement.csv",
        "specialization_boundary_stable_rate.csv",
        "phantom_breakdown_labels.csv.gz",
        "phantom_breakdown_by_failure_mode.csv",
        "phantom_breakdown_rates_for_paper.csv",
        "phantom_breakdown_counts_for_paper.csv",
    ]
    for name in candidates:
        path = out_dir / name
        if path.exists() and path.is_file():
            try:
                path.unlink()
            except Exception as exc:
                print(f"[warn] could not remove intermediate file {path}: {exc}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Standalone entrypoint for threshold robustness, the appendix paper figures, and the phantom (unstable-diversity) breakdown figure."
    )
    p.add_argument(
        "command",
        nargs="?",
        default="all",
        choices=["threshold", "figures", "all", "phantom"],
        help="Which workflow to run. Defaults to all.",
    )
    p.add_argument("--threshold-extra", default="", help="Quoted extra CLI args passed to the embedded threshold workflow.")
    p.add_argument("--figures-extra", default="", help="Quoted extra CLI args passed to the embedded figure workflow.")
    p.add_argument("--threshold-out-dir", default="", help="Output directory for threshold post-processing. Defaults to --out_dir in --threshold-extra or results/threshold_robustness.")
    p.add_argument("--figures-out-dir", default="", help=f"Deprecated; PDF plots are always written directly under {DEFAULT_FIGURES_OUT_DIR}.")
    p.add_argument("--near-critical-eps", type=float, default=0.05, help="Near-critical band used in phantom breakdown.")
    p.add_argument("--skip-phantom-breakdown", action="store_true", help="Do not run phantom-diversity post-processing after threshold/all.")
    p.add_argument("--write-postprocess-tables", action="store_true", help="Write post-processing CSV tables in addition to PDF plots. By default, only plots are written for added post-process analyses.")
    p.add_argument("--keep-intermediate-files", action="store_true", help="Keep intermediate CSV outputs produced by embedded plotting workflows. By default, known non-plot intermediates are removed from the figure output directory.")
    p.add_argument("--dry-run", action="store_true", help="Print embedded workflow commands without executing them. Post-processing is skipped in dry-run mode.")
    p.add_argument("--continue-on-error", action="store_true", help="For command=all, run figures even if threshold robustness fails.")
    return p


def _threshold_out_dir(args, threshold_args: list[str]) -> Path:
    return Path(args.threshold_out_dir or _cli_value(threshold_args, "out_dir", "results/threshold_robustness") or "results/threshold_robustness")


def _figures_out_dir(args, figures_args: list[str]) -> Path:
    return Path(DEFAULT_FIGURES_OUT_DIR)


def _args_forced_out_dir(extra_args: list[str], out_dir: Path) -> list[str]:
    cleaned: list[str] = []
    skip_next = False
    for item in extra_args:
        if skip_next:
            skip_next = False
            continue
        if item == "--out_dir":
            skip_next = True
            continue
        if item.startswith("--out_dir="):
            continue
        cleaned.append(item)
    return [*cleaned, "--out_dir", str(out_dir)]


def _figures_args_for_forced_out_dir(figures_args: list[str]) -> list[str]:
    return _args_forced_out_dir(figures_args, Path(DEFAULT_FIGURES_OUT_DIR))


def _threshold_args_for_out_dir(threshold_args: list[str], out_dir: Path) -> list[str]:
    return _args_forced_out_dir(threshold_args, out_dir)


def _cleanup_text_outputs(out_dir: Path) -> None:
    if not out_dir.exists():
        return
    for path in out_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".csv", ".gz", ".json", ".md", ".txt"}:
            path.unlink()
    for path in sorted([p for p in out_dir.rglob("*") if p.is_dir()], key=lambda p: len(p.parts), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass


def _run_phantom_postprocess(args, threshold_args: list[str]) -> int:
    if args.skip_phantom_breakdown or args.dry_run:
        return 0
    threshold_out_dir = _threshold_out_dir(args, threshold_args)
    labels_path = threshold_out_dir / "threshold_robustness_labels.csv.gz"
    explicit_label_dir = bool(args.threshold_out_dir or _cli_value(threshold_args, "out_dir", None))
    if labels_path.exists() and explicit_label_dir:
        make_phantom_breakdown_outputs(threshold_out_dir, near_critical_eps=float(args.near_critical_eps), write_tables=bool(args.write_postprocess_tables))
        _cleanup_text_outputs(Path(DEFAULT_FIGURES_OUT_DIR))
        return 0

    with tempfile.TemporaryDirectory(prefix="phantom_threshold_") as tmp:
        tmp_out_dir = Path(tmp)
        rc = _run_threshold_embedded(_threshold_args_for_out_dir(threshold_args, tmp_out_dir), dry_run=False)
        if rc != 0:
            print(f"[warn] phantom breakdown skipped: temporary threshold workflow failed with exit code {rc}", file=sys.stderr)
            return rc
        make_phantom_breakdown_outputs(tmp_out_dir, near_critical_eps=float(args.near_critical_eps), write_tables=bool(args.write_postprocess_tables))
        _cleanup_text_outputs(Path(DEFAULT_FIGURES_OUT_DIR))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    threshold_args = _split_extra(args.threshold_extra)
    figures_args = _split_extra(args.figures_extra)

    if args.command == "phantom":
        return _run_phantom_postprocess(args, threshold_args)
    if args.command == "threshold":
        return _run_phantom_postprocess(args, threshold_args)
    if args.command == "figures":
        figures_args_for_run = _figures_args_for_forced_out_dir(figures_args)
        rc = _run_figures_embedded(figures_args_for_run, dry_run=args.dry_run)
        if rc == 0:
            if not args.keep_intermediate_files and not args.write_postprocess_tables and not args.dry_run:
                _cleanup_unneeded_intermediate_files(_figures_out_dir(args, figures_args))
            if not args.write_postprocess_tables:
                _cleanup_text_outputs(_figures_out_dir(args, figures_args))
        return rc

    rc1 = _run_phantom_postprocess(args, threshold_args)
    if rc1 != 0 and not args.continue_on_error:
        print(f"[error] threshold-derived phantom plot workflow failed with exit code {rc1}; stopping before figures.", file=sys.stderr)
        return rc1

    figures_args_for_run = _figures_args_for_forced_out_dir(figures_args)
    rc2 = _run_figures_embedded(figures_args_for_run, dry_run=args.dry_run)
    if rc2 == 0:
        if not args.keep_intermediate_files and not args.write_postprocess_tables and not args.dry_run:
            _cleanup_unneeded_intermediate_files(_figures_out_dir(args, figures_args))
        if not args.write_postprocess_tables:
            _cleanup_text_outputs(_figures_out_dir(args, figures_args))
    return rc2 if rc2 != 0 else rc1


if __name__ == "__main__":
    raise SystemExit(main())
