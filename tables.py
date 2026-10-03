#!/usr/bin/env python3
"""Numbers of the paper that are not read off a figure, re-derived from the replay grids in data/.

    python3 tables.py all
    python3 tables.py unstable_diversity   Appendix B.7: share of descriptor-positive endpoints failing the audit, main and DeepSeek evaluator
                                           (also writes the counts of Fig. 4(a))                      -> data/outputs/unstable_diversity_rates[_deepseek]/
    python3 tables.py evaluator_shift      Sec. 4.5, B.7: sign agreement of trait shifts across evaluators -> data/outputs/evaluator_trait_shift/
    python3 tables.py testbeds             B.8: one row per robustness testbed                              -> data/outputs/testbeds_table/robustness_table.csv
    python3 tables.py tool                 B.8: tool-routing benchmark summary                              -> data/outputs/testbeds_tool_summary/summary.json
    python3 tables.py fragility            B.6: theory-directed fragility test                              -> data/outputs/fragility/theory_direction_*_rows.csv

Each sub-command accepts the options of the original script (``python3 tables.py unstable_diversity --help``).
The DeepSeek variant of unstable_diversity is run by ``all``.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import json
from scipy.stats import spearmanr
import glob

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pipeline"))  # shared helpers live in pipeline/common

from common import plot as C  # noqa: E402
from fragility.amplification import small_cluster_sign_and_wild_inference  # noqa: E402


# ====================================================================================================
# unstable_diversity
# ====================================================================================================
def main_unstable_diversity():
    MODE_LABELS = {
        "active_face_tangent_instability": "active-face tangent",
        "inactive_agent_invasion": "inactive-agent invasion",
        "near_critical": "near-critical",
        "positive_criticality_unclassified": "positive criticality, unclassified",
        "unstable_unclassified": "unstable, unclassified",
    }
    PHANTOM_MODES = list(MODE_LABELS)
    FAMILIES = ["Coexistence", "Specialization", "Any diversity"]
    MAIN_SWEEP_TARGETS = {
        "Coexistence": (18185, 3791, 14394),
        "Specialization": (17771, 1431, 16340),
        "Any diversity": (35956, 5222, 30734),
    }
    def parse_args() -> argparse.Namespace:
        p = argparse.ArgumentParser(description=__doc__)
        p.add_argument(
            "--labels",
            default="data/results/threshold_robustness/phantom_breakdown_labels.csv.gz",
        )
        p.add_argument(
            "--sweeps", "--panel-a-sweeps", dest="sweeps", default="main",
            help="Comma-separated sweep_name values used in BOTH panels ('all' = every sweep). "
            "Default 'main' (the appendix figure fig4_phantom_diversity_rate.pdf uses 'all').",
        )
        p.add_argument("--dedup", type=int, default=1,
                       help="1 = drop replay conditions (seed x platform parameters) that occur more than once in the "
                       "labels file (the alpha=1, beta_sup=0 column is stored in both fig0_4_full_grid and "
                       "fig5_full_grid); 0 = use the file as-is (counts Fig. 4(a) does not use).")
        p.add_argument("--exclude-seeds", default="",
                       help="Comma-separated population seeds to drop (e.g. a truncated run).")
        p.add_argument("--near-critical-eps", type=float, default=0.05)
        p.add_argument("--n-boot", type=int, default=20000)
        p.add_argument("--boot-seed", type=int, default=20260918)
        p.add_argument("--out-dir", default="data/outputs/unstable_diversity_rates")
        return p.parse_args()
    def pipeline_failure_mode(base: pd.DataFrame, family_any: np.ndarray, eps: float) -> np.ndarray:
        """Verbatim re-implementation of the labelling in make_phantom_breakdown_outputs."""
        locally_contracting = (
            pd.to_numeric(base["robust_locally_contracting"], errors="coerce").fillna(0).astype(int)
        )
        kappa = pd.to_numeric(base["kappa_map"], errors="coerce")
        gamma = pd.to_numeric(base["gamma_map"], errors="coerce")
        invasion = pd.to_numeric(base["max_invasion_log"], errors="coerce")
        failure_mode = np.full(len(base), "not_descriptor_positive", dtype=object)
        stable = family_any & (locally_contracting.to_numpy() == 1)
        failure_mode[stable] = "stable_locally_contracting"
        not_stable = family_any & ~stable
        near = not_stable & (kappa.abs().to_numpy() <= float(eps))
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
        return failure_mode
    DEDUP_KEY = ["_seed", "eta", "alpha", "beta_sup", "beta_route", "lambda_risk",
                 "context_trait_strength", "polarization_strength", "route_power"]
    def load_labels(labels_path: Path, sweeps: str, eps: float, dedup: bool = True, exclude_seeds: str = ""):
        base = pd.read_csv(labels_path)
        if exclude_seeds:
            seed = base["input_file"].astype(str).str.extract(r"popseed_(\d+)", expand=False)
            keep = ~seed.isin([x.strip() for x in exclude_seeds.split(",")])
            print(f"exclude-seeds {exclude_seeds}: dropped {int((~keep).sum())} rows")
            base = base.loc[keep].copy()
        if dedup:
            base["_seed"] = base["input_file"].astype(str).str.extract(r"popseed_(\d+)", expand=False)
            assert base["_seed"].notna().all()
            n_before = len(base)
            base = base.drop_duplicates(subset=["sweep_name"] + DEDUP_KEY, keep="first").drop(columns="_seed")
            print(f"dedup: dropped {n_before - len(base)} repeated replay conditions")
        if sweeps != "all":
            base = base.loc[base["sweep_name"].astype(str).isin(sweeps.split(","))].copy()
        base = base.reset_index(drop=True)
        base["population_seed"] = (
            base["source_result_dir"].astype(str).str.extract(r"popseed_(\d+)", expand=False).astype(int)
        )
        phase = base["robust_phase_label"].astype(str)
        coexistence = phase.isin(["diffuse_coexistence", "concentrated_coexistence"]).to_numpy()
        specialization = (
            pd.to_numeric(base["robust_specialization_signal"], errors="coerce").fillna(0).astype(int) == 1
        ).to_numpy()
        polarization = (
            pd.to_numeric(base["robust_polarization_signal"], errors="coerce").fillna(0).astype(int) == 1
        ).to_numpy()
        any_div = coexistence | specialization | polarization
        masks = {"Coexistence": coexistence, "Specialization": specialization, "Any diversity": any_div}

        recomputed = pipeline_failure_mode(base, any_div, eps)
        assert (recomputed == base["phantom_failure_mode"].astype(str).to_numpy()).all(), (
            "re-derived failure modes differ from the pipeline's stored phantom_failure_mode"
        )
        gamma_pos = pd.to_numeric(base["gamma_map"], errors="coerce").fillna(-np.inf).to_numpy() > 0
        inv_pos = pd.to_numeric(base["max_invasion_log"], errors="coerce").fillna(-np.inf).to_numpy() > 0
        base["both_exponents_positive"] = (gamma_pos & inv_pos).astype(int)
        base["exponent_sign_class"] = np.select(
            [gamma_pos & inv_pos, gamma_pos & ~inv_pos, ~gamma_pos & inv_pos],
            ["both_positive", "tangent_only_positive", "invasion_only_positive"],
            default="neither_positive",
        )
        return base, masks
    def seed_summary(num: pd.Series, den: pd.Series, n_boot: int, rng) -> dict:
        """num/den: per-seed counts (same index). Pooled ratio, per-seed mean/SE, seed bootstrap CI."""
        num = num.to_numpy(dtype=float)
        den = den.to_numpy(dtype=float)
        per_seed = num / den
        idx = rng.integers(0, len(num), size=(n_boot, len(num)))
        boot = num[idx].sum(axis=1) / den[idx].sum(axis=1)
        return {
            "pooled_share": float(num.sum() / den.sum()),
            "seed_mean_share": float(per_seed.mean()),
            "seed_se_share": float(per_seed.std(ddof=1) / np.sqrt(len(per_seed))),
            "seed_min_share": float(per_seed.min()),
            "seed_max_share": float(per_seed.max()),
            "seed_boot_ci_low": float(np.quantile(boot, 0.025)),
            "seed_boot_ci_high": float(np.quantile(boot, 0.975)),
            "n_seeds": int(len(num)),
        }
    def compute_tables(base: pd.DataFrame, masks: dict, n_boot: int, boot_seed: int):
        rng = np.random.default_rng(boot_seed)
        seeds = sorted(base["population_seed"].unique())
        rate_rows, rate_seed_rows, mode_rows, mode_seed_rows, sign_rows = [], [], [], [], []
        for fam_name in FAMILIES:
            fam = base.loc[masks[fam_name]]
            stable = fam["phantom_failure_mode"] == "stable_locally_contracting"
            phantom = fam.loc[~stable]
            assert set(phantom["phantom_failure_mode"]) <= set(PHANTOM_MODES)
            n_seed = fam.groupby("population_seed").size().reindex(seeds, fill_value=0)
            ph_seed = phantom.groupby("population_seed").size().reindex(seeds, fill_value=0)
            rate_rows.append({
                "family": fam_name,
                "n_descriptor_positive": int(len(fam)),
                "locally_contracting": int(stable.sum()),
                "not_locally_contracting": int(len(phantom)),
                **{f"phantom_{k}": v for k, v in seed_summary(ph_seed, n_seed, n_boot, rng).items()},
            })
            for s in seeds:
                rate_seed_rows.append({
                    "family": fam_name, "population_seed": s,
                    "n_descriptor_positive": int(n_seed[s]),
                    "not_locally_contracting": int(ph_seed[s]),
                    "phantom_share": float(ph_seed[s] / n_seed[s]),
                })
            for mode in PHANTOM_MODES:
                sel = phantom["phantom_failure_mode"] == mode
                m_seed = phantom.loc[sel].groupby("population_seed").size().reindex(seeds, fill_value=0)
                both = int(phantom.loc[sel, "both_exponents_positive"].sum())
                both_seed = (
                    phantom.loc[sel & (phantom["both_exponents_positive"] == 1)]
                    .groupby("population_seed").size().reindex(seeds, fill_value=0)
                )
                summ = seed_summary(m_seed, ph_seed, n_boot, rng)
                mode_rows.append({
                    "family": fam_name, "failure_mode": mode,
                    "count": int(sel.sum()), "n_phantom": int(len(phantom)),
                    "share_of_phantom": summ["pooled_share"],
                    **{k: v for k, v in summ.items() if k != "pooled_share"},
                    "share_of_descriptor_positive": float(sel.sum() / len(fam)),
                    "count_both_exponents_positive": both,
                    "share_of_phantom_both_exponents_positive": float(both / len(phantom)),
                })
                for s in seeds:
                    mode_seed_rows.append({
                        "family": fam_name, "failure_mode": mode, "population_seed": s,
                        "count": int(m_seed[s]), "n_phantom": int(ph_seed[s]),
                        "share_of_phantom": float(m_seed[s] / ph_seed[s]),
                        "count_both_exponents_positive": int(both_seed[s]),
                    })
            # supplementary sign-based split (NOT the pipeline's categories)
            near = phantom["phantom_failure_mode"] == "near_critical"
            sign = phantom["exponent_sign_class"].where(~near, "near_critical")
            for cls in ["tangent_only_positive", "both_positive", "invasion_only_positive",
                        "near_critical", "neither_positive"]:
                c_seed = phantom.loc[sign == cls].groupby("population_seed").size().reindex(seeds, fill_value=0)
                summ = seed_summary(c_seed, ph_seed, n_boot, rng)
                sign_rows.append({
                    "family": fam_name, "sign_class_after_near_critical": cls,
                    "count": int((sign == cls).sum()), "n_phantom": int(len(phantom)),
                    "share_of_phantom": summ["pooled_share"],
                    **{k: v for k, v in summ.items() if k != "pooled_share"},
                })
        return (pd.DataFrame(rate_rows), pd.DataFrame(rate_seed_rows), pd.DataFrame(mode_rows),
                pd.DataFrame(mode_seed_rows), pd.DataFrame(sign_rows))
    def main() -> None:
        args = parse_args()
        out_dir = ROOT / args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.set_option("display.width", 250, "display.max_columns", 40)

        results = {}
        for sweeps in dict.fromkeys([args.sweeps, "all"]):
            base, masks = load_labels(ROOT / args.labels, sweeps, args.near_critical_eps, bool(args.dedup), args.exclude_seeds)
            tables = compute_tables(base, masks, args.n_boot, args.boot_seed)
            results[sweeps] = tables
            rates, rates_seed, modes, modes_seed, sign = tables
            if sweeps == "main" and not args.dedup:
                for fam, (n, c, nc) in MAIN_SWEEP_TARGETS.items():
                    r = rates.set_index("family").loc[fam]
                    assert (r["n_descriptor_positive"], r["locally_contracting"],
                            r["not_locally_contracting"]) == (n, c, nc), (fam, r.to_dict())
            # panel-B phantom totals must equal panel-A "not locally contracting" counts
            tot = modes.groupby("family")["count"].sum()
            for fam in FAMILIES:
                assert tot[fam] == rates.set_index("family").loc[fam, "not_locally_contracting"], fam
            tag = sweeps.replace(",", "+")
            rates.to_csv(out_dir / f"fig4_panelA_phantom_rate_{tag}.csv", index=False)
            rates_seed.to_csv(out_dir / f"fig4_panelA_phantom_rate_per_seed_{tag}.csv", index=False)
            modes.to_csv(out_dir / f"fig4_panelB_failure_modes_{tag}.csv", index=False)
            modes_seed.to_csv(out_dir / f"fig4_panelB_failure_modes_per_seed_{tag}.csv", index=False)
            sign.to_csv(out_dir / f"fig4_panelB_exponent_sign_split_{tag}.csv", index=False)
            print(f"\n===== sweeps = {sweeps} (rows: {len(base)}) =====")
            print(rates.round(4).to_string(index=False))
            print(modes.loc[modes["count"] > 0].round(4).to_string(index=False))
            print(sign.loc[sign["count"] > 0].round(4).to_string(index=False))

        if args.sweeps == "main":
            # Fig. 4(a) counts: main sweeps as stored (no de-duplication of the alpha = 1, beta_sup = 0 column)
            base0, masks0 = load_labels(ROOT / args.labels, "main", args.near_critical_eps, False, args.exclude_seeds)
            lc0 = pd.to_numeric(base0["robust_locally_contracting"], errors="coerce").fillna(0).astype(int).to_numpy() == 1
            rows = []
            for fam in FAMILIES:
                m = masks0[fam]
                rows.append({"family": fam, "n_descriptor_positive": int(m.sum()), "locally_contracting": int((m & lc0).sum()),
                             "not_locally_contracting": int((m & ~lc0).sum())})
            cnt = pd.DataFrame(rows)
            cnt["not_locally_contracting_share"] = cnt.not_locally_contracting / cnt.n_descriptor_positive
            cnt["sweeps"] = "main"; cnt["n_rows_in_frame"] = len(base0)
            cnt.to_csv(out_dir / "fig4_panelA_counts.csv", index=False)
    main()


# ====================================================================================================
# evaluator_shift
# ====================================================================================================
def main_evaluator_shift():
    AXES = ["stance_valence", "social_framing", "epistemic_style", "consensus_alignment"]
    KEYS = ["eta", "alpha", "beta_sup", "beta_route", "lambda_risk", "hard_safety_threshold",
            "context_trait_strength", "polarization_strength", "route_power", "route_zscore",
            "no_feedback", "ablation"]
    GRIDS = {
        "no_support": "fig0_4_full_grid",                      # eta x alpha, beta_sup = 0
        "support": "fig5_full_grid",                           # eta x beta_sup
        "support_wide": "specialization_wide_dense_figS",      # eta x beta_sup (wide/dense)
        "route": "appendix_route_sweep_parallel",              # eta x beta_route
        "ctx_trait": "fig10_full_grid/specialization",         # beta_sup x context_trait_strength (synthetic trait bonus)
    }
    DS_SUFFIX = "_reuse_together_evalaudit_compact"
    def uniform_trait_mean(run_dir: Path) -> dict:
        from replay import engine as R  # existing replay code (read-only import)
        md = json.loads((run_dir / "metadata.json").read_text())
        ids, tr = R.metadata_agent_ids_and_traits(md)
        df = R.build_trait_dataframe(ids, tr)
        return {a: float(df[a].mean()) for a in AXES}, df
    def load_matched(seeds, gpt_root: Path, ds_root: Path, runs_root: Path) -> pd.DataFrame:
        frames = []
        for seed in seeds:
            u_gpt, t_gpt = uniform_trait_mean(runs_root / f"run_popseed_{seed}.checkpoints")
            u_ds, t_ds = uniform_trait_mean(runs_root / f"run_popseed_{seed}{DS_SUFFIX}.checkpoints")
            assert np.allclose(t_gpt[AXES].to_numpy(), t_ds[AXES].to_numpy()), f"trait coords differ for seed {seed}"
            for gname, gdir in GRIDS.items():
                fg = gpt_root / gdir / f"run_popseed_{seed}.checkpoints" / "phase_summary.csv"
                fd = ds_root / gdir / f"run_popseed_{seed}{DS_SUFFIX}.checkpoints" / "phase_summary.csv"
                if not (fg.exists() and fd.exists()):
                    print(f"[warn] missing {gname} seed {seed}: gpt={fg.exists()} ds={fd.exists()}")
                    continue
                kg = json.loads((fg.parent / "run_config.json").read_text())["args"]["endpoint_last_k"]
                kd = json.loads((fd.parent / "run_config.json").read_text())["args"]["endpoint_last_k"]
                assert kg == kd, (gname, seed, kg, kd)
                cols = KEYS + [f"trait_mean_{a}" for a in AXES] + ["phase_label", "top1_share", "n_eff", "kappa_map"]
                g = pd.read_csv(fg, usecols=cols)
                d = pd.read_csv(fd, usecols=cols)
                for df_ in (g, d):
                    df_["ablation"] = df_["ablation"].fillna("none").astype(str)
                    for k in KEYS:
                        if df_[k].dtype.kind == "f":
                            df_[k] = df_[k].round(9)
                m = g.merge(d, on=KEYS, suffixes=("_gpt", "_ds"), how="inner", validate="one_to_one")
                if len(m) != len(g) or len(m) != len(d):
                    print(f"[warn] {gname} seed {seed}: gpt={len(g)} ds={len(d)} matched={len(m)}")
                for a in AXES:
                    m[f"shift_gpt_{a}"] = m[f"trait_mean_{a}_gpt"] - u_gpt[a]
                    m[f"shift_ds_{a}"] = m[f"trait_mean_{a}_ds"] - u_ds[a]
                m.insert(0, "grid", gname)
                m.insert(1, "seed", seed)
                m["endpoint_last_k"] = kg
                frames.append(m)
        return pd.concat(frames, ignore_index=True)
    def cluster_mean_se(values: np.ndarray, seeds: np.ndarray):
        """Mean over all rows and seed-clustered SE (SD of per-seed means / sqrt(G))."""
        s = pd.Series(values).groupby(seeds).mean()
        g = len(s)
        se = float(s.std(ddof=1) / np.sqrt(g)) if g > 1 else float("nan")
        return float(np.mean(values)), se, g
    def summarize(df: pd.DataFrame, dead_zone: float) -> pd.DataFrame:
        rows = []
        seeds = df["seed"].to_numpy()
        for a in AXES:
            x = df[f"shift_gpt_{a}"].to_numpy()
            y = df[f"shift_ds_{a}"].to_numpy()
            agree0 = (np.sign(x) == np.sign(y)).astype(float)
            keep = (np.abs(x) >= dead_zone) & (np.abs(y) >= dead_zone)
            r = {"axis": a, "n_matched": len(df), "n_seeds": int(pd.Series(seeds).nunique())}
            r["agree_nodz"], r["agree_nodz_se"], _ = cluster_mean_se(agree0, seeds)
            if keep.sum() > 0:
                r["agree_dz_excl"], r["agree_dz_excl_se"], _ = cluster_mean_se(agree0[keep], seeds[keep])
            else:
                r["agree_dz_excl"], r["agree_dz_excl_se"] = np.nan, np.nan
            r["n_outside_dz"] = int(keep.sum())
            # strict variant: inside the dead zone counts as "no direction" -> agreement only if both outside & same sign
            r["agree_dz_strict"], r["agree_dz_strict_se"], _ = cluster_mean_se((keep & (agree0 > 0)).astype(float), seeds)
            r["mean_shift_gpt"], r["mean_shift_gpt_se"], _ = cluster_mean_se(x, seeds)
            r["mean_shift_ds"], r["mean_shift_ds_se"], _ = cluster_mean_se(y, seeds)
            r["frac_pos_gpt"] = float(np.mean(x > 0))
            r["frac_pos_ds"] = float(np.mean(y > 0))
            r["pearson_r"] = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else np.nan
            per_seed_r = [np.corrcoef(x[seeds == s], y[seeds == s])[0, 1] for s in np.unique(seeds)]
            r["pearson_r_within_seed_mean"] = float(np.nanmean(per_seed_r))
            r["spearman_r"] = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
            r["sign_mean_gpt"] = int(np.sign(r["mean_shift_gpt"]))
            r["sign_mean_ds"] = int(np.sign(r["mean_shift_ds"]))
            # reversal: opposite mean-shift signs, each mean at least 2 seed-clustered SE from 0
            r["opposite_mean_sign"] = bool(r["sign_mean_gpt"] * r["sign_mean_ds"] < 0)
            r["gpt_sig"] = bool(abs(r["mean_shift_gpt"]) > 2 * r["mean_shift_gpt_se"])
            r["ds_sig"] = bool(abs(r["mean_shift_ds"]) > 2 * r["mean_shift_ds_se"])
            r["reversal"] = bool(r["opposite_mean_sign"] and r["gpt_sig"] and r["ds_sig"])
            # seed-level consistency of mean-shift sign
            ps_g = pd.Series(x).groupby(seeds).mean()
            ps_d = pd.Series(y).groupby(seeds).mean()
            r["seeds_pos_gpt"] = int((ps_g > 0).sum())
            r["seeds_pos_ds"] = int((ps_d > 0).sum())
            r["seeds_sign_flip"] = int((np.sign(ps_g) != np.sign(ps_d)).sum())
            diff = (ps_d - ps_g).to_numpy()
            r["paired_diff_ds_minus_gpt"] = float(np.mean(y - x))
            r["paired_diff_se"] = float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else float("nan")
            # exact seed-level sign-flip test of mean paired difference (two-sided; min p = 2/2^G)
            import itertools
            obs = abs(diff.mean())
            flips = [abs(np.mean(diff * np.array(sg))) for sg in itertools.product([1, -1], repeat=len(diff))]
            r["paired_diff_signflip_p"] = float(np.mean(np.array(flips) >= obs - 1e-15))
            rows.append(r)
        return pd.DataFrame(rows)
    def main():
        ap = argparse.ArgumentParser()
        ap.add_argument("--gpt_root", default=str(ROOT / "data/results"))
        ap.add_argument("--ds_root", default=str(ROOT / "data/results_reuse_clean"))
        ap.add_argument("--runs_root", default=str(ROOT / "data/agent_metadata"), help="directory holding run_popseed_<s>[...].checkpoints/metadata.json (agent trait coordinates)")
        ap.add_argument("--seeds", default="11,22,33,44,55")
        ap.add_argument("--dead_zone", type=float, default=0.02)
        ap.add_argument("--out_dir", default=str(ROOT / "data/outputs/evaluator_trait_shift"))
        args = ap.parse_args()

        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        seeds = [int(s) for s in args.seeds.split(",")]
        df = load_matched(seeds, Path(args.gpt_root), Path(args.ds_root), Path(args.runs_root))
        keep_cols = ["grid", "seed", "endpoint_last_k"] + KEYS + [c for c in df.columns if c.startswith("shift_")] + \
                    ["phase_label_gpt", "phase_label_ds", "top1_share_gpt", "top1_share_ds"]
        df[keep_cols].to_csv(out / "matched_condition_shifts.csv.gz", index=False)

        sub = df[df.grid == "no_support"]
        tab = summarize(sub, args.dead_zone)
        tab.insert(0, "unit", "seed_x_condition"); tab.insert(0, "eta_subset", "all_eta"); tab.insert(0, "subset", "no_support")
        tab.to_csv(out / "table_rows.csv", index=False)
        meta = {"dead_zone": args.dead_zone, "n_matched_by_grid": df.groupby("grid").size().to_dict()}
        (out / "summary.json").write_text(json.dumps(meta, indent=2, default=float))

        pd.set_option("display.width", 250, "display.max_columns", 40)
        print("\n=== no-support grid, all eta, seed x condition")
        print(tab[["axis", "n_matched", "agree_nodz", "agree_nodz_se", "agree_dz_excl", "n_outside_dz", "agree_dz_strict",
                   "mean_shift_gpt", "mean_shift_gpt_se", "mean_shift_ds", "mean_shift_ds_se", "pearson_r",
                   "pearson_r_within_seed_mean", "spearman_r", "seeds_pos_gpt", "seeds_pos_ds", "reversal"]].round(4).to_string())
    main()


# ====================================================================================================
# testbeds
# ====================================================================================================
def main_testbeds():
    SETTINGS = [
        # key, label, main dir, support dir, auc grid
        ("llama", "Llama-3.3-70B only", "data/testbeds/verification_llama_main", "data/testbeds/verification_llama_support", "support"),
        ("8d", "8D traits", "data/testbeds/verification_8d_main", "data/testbeds/verification_8d_support", "support"),
        ("16d", "16D traits", "data/testbeds/verification_16d_main", "data/testbeds/verification_16d_support", "support"),
        ("heterogeneity", "Heterogeneous generators", "data/testbeds/verification_heterogeneity_main", "data/testbeds/verification_heterogeneity_support", "support"),
        ("tool", "BFCL tool routing", "data/testbeds/verification_tool_main", "data/testbeds/verification_tool_support", "main"),
    ]
    def _bool(s: pd.Series) -> pd.Series:
        if s.dtype == bool:
            return s
        return s.astype(str).str.lower().isin(["true", "1"])
    def _read(path: Path) -> pd.DataFrame:
        gz = Path(str(path) + ".gz")
        return pd.read_csv(path if path.exists() else gz)
    def summarize(main_dir: Path, sup_dir: Path, auc_grid: str) -> dict:
        m = _read(main_dir / "phase_summary.csv")
        s = _read(sup_dir / "phase_summary.csv")
        out: dict = {}
        seeds = sorted(set(m.run_name) | set(s.run_name))
        out["n_seeds"] = len(seeds)
        out["n_main"], out["n_support"] = len(m), len(s)
        out["n_agents"] = int(m.n_agents.iloc[0])
        out["n_timesteps"] = int(m.n_timesteps.iloc[0])
        lo, hi = float(m.eta.min()), float(m.eta.max())
        out["eta_lo"], out["eta_hi"] = lo, hi
        for tag, e in (("lo", lo), ("hi", hi)):
            d = m[m.eta == e]
            out[f"top1_{tag}"] = float(d.top1_share.mean())
            out[f"kappa_{tag}"] = float(d.kappa_map.mean())
            by = d.groupby("run_name")[["top1_share", "kappa_map"]].mean()
            out[f"top1_{tag}_seed_min"], out[f"top1_{tag}_seed_max"] = float(by.top1_share.min()), float(by.top1_share.max())
            out[f"kappa_{tag}_seed_min"], out[f"kappa_{tag}_seed_max"] = float(by.kappa_map.min()), float(by.kappa_map.max())
        # seed-level Spearman rho(eta, .) over the eta-means of the main grid (README section 1)
        rt, rk = [], []
        for _, d in m.groupby("run_name"):
            g = d.groupby("eta")[["top1_share", "kappa_map"]].mean()
            rt.append(spearmanr(g.index, g.top1_share)[0])
            rk.append(spearmanr(g.index, g.kappa_map)[0])
        out["rho_eta_top1_seed_min"], out["rho_eta_top1_seed_max"] = float(np.min(rt)), float(np.max(rt))
        out["rho_eta_kappa_seed_min"], out["rho_eta_kappa_seed_max"] = float(np.min(rk)), float(np.max(rk))

        lc = _bool(s.locally_contracting)
        pos = s.phase_label != "monoculture_collapse"
        out["phantom_den"] = int(pos.sum())
        out["phantom_fail"] = int((pos & ~lc).sum())
        out["phantom_pct"] = 100.0 * out["phantom_fail"] / out["phantom_den"] if out["phantom_den"] else np.nan
        ph = (pos & ~lc).groupby(s.run_name).sum() / pos.groupby(s.run_name).sum()
        out["phantom_pct_seed_min"], out["phantom_pct_seed_max"] = float(100 * ph.min()), float(100 * ph.max())

        spec = s.specialization_signal == 1
        st = s[spec & lc]
        out["descriptor_spec"] = int(spec.sum())
        out["stable_spec"] = int(len(st))
        out["main_descriptor_spec"] = int((m.specialization_signal == 1).sum())
        out["main_stable_spec"] = int(((m.specialization_signal == 1) & _bool(m.locally_contracting)).sum())
        if len(st):
            out["spec_n_seeds"] = int(st.run_name.nunique())
            out["spec_per_seed"] = ";".join(f"{k.split('.')[0].replace('run_popseed_', '')}:{v}" for k, v in st.run_name.value_counts().sort_index().items())
            out["spec_eta_min"], out["spec_eta_max"] = float(st.eta.min()), float(st.eta.max())
            out["spec_sup_min"], out["spec_sup_max"] = float(st.beta_sup.min()), float(st.beta_sup.max())
            out["spec_rho"] = float(spearmanr(st.eta, st.beta_sup)[0])
            rhos = [spearmanr(d.eta, d.beta_sup)[0] for _, d in st.groupby("run_name") if len(d) > 2]
            out["spec_rho_seed_min"], out["spec_rho_seed_max"] = float(np.nanmin(rhos)), float(np.nanmax(rhos))
            above = sorted(e for e in s.eta.unique() if e > st.eta.max())
            out["first_eta_without_stable_spec"] = float(above[0]) if above else np.nan
        out["collapse_rate_eta_ge5"] = float((s[s.eta >= 5].collapse_signal == 1).mean())
        out["max_i_exp_norm_support"] = float(s.i_exp_norm.max())

        ew_path = (sup_dir if auc_grid == "support" else main_dir) / "early_warning_evaluation.csv"
        out["auc_grid"] = auc_grid
        if ew_path.exists():
            e = pd.read_csv(ew_path)
            e = e[(e.target == "label_unstable") & (e.control == "none") & np.isclose(e.early_frac, 0.1)]
            for split, tag in (("within_grid_cv", "within"), ("leave_seed_out", "loso"), ("eta_transfer", "etatr")):
                for model, mt in (("parameter_only", "param"), ("early_descriptor", "desc"), ("fluctuation_only", "fluct"), ("full_early_warning", "full")):
                    r = e[(e.split == split) & (e.model == model)]
                    out[f"auc_{tag}_{mt}"] = float(r.roc_auc.iloc[0]) if len(r) else np.nan
            r = e[(e.split == "within_grid_cv") & (e.model == "parameter_only")]
            out["auc_n"] = int(r.n.iloc[0]) if len(r) else 0
            out["auc_positive_rate"] = float(r.positive_rate.iloc[0]) if len(r) else np.nan
        return out
    def main() -> None:
        ap = argparse.ArgumentParser()
        ap.add_argument("--root", default=".")
        ap.add_argument("--out_dir", default="data/outputs/testbeds_table")
        ap.add_argument("--main4d_main", default="data/outputs/testbeds_table/main4d_matched_main")
        ap.add_argument("--main4d_support", default="data/outputs/testbeds_table/main4d_matched_support")
        args = ap.parse_args()
        root = Path(args.root)
        out_dir = root / args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        res = {k: summarize(root / md, root / sd, ag) for k, label, md, sd, ag in SETTINGS}
        have_ref = all(any((root / d / f).exists() for f in ("phase_summary.csv", "phase_summary.csv.gz")) for d in (args.main4d_main, args.main4d_support))
        if have_ref:
            res["main4d"] = summarize(root / args.main4d_main, root / args.main4d_support, "support")
        labels = {k: l for k, l, *_ in SETTINGS}
        labels["main4d"] = "Main 4D, GPT-5.4-mini (matched protocol: first 100 timesteps)"
        order = (["main4d"] if have_ref else []) + [k for k, *_ in SETTINGS]
        table = pd.DataFrame([dict(setting=k, label=labels[k], **res[k]) for k in order])
        table.to_csv(out_dir / "robustness_table.csv", index=False)

        pd.set_option("display.width", 250)
        cols = ["setting", "n_seeds", "top1_lo", "top1_hi", "kappa_lo", "kappa_hi", "phantom_fail", "phantom_den", "phantom_pct", "stable_spec",
                "auc_within_param", "auc_within_desc", "auc_loso_param", "auc_loso_desc"]
        print(table[[c for c in cols if c in table.columns]].to_string())
        print("wrote", out_dir / "robustness_table.csv")
    main()


# ====================================================================================================
# tool
# ====================================================================================================
def main_tool():
    def parse_args():
        p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        p.add_argument("--main_dir", default="data/testbeds/verification_tool_main")
        p.add_argument("--low_eta_dir", default="data/testbeds/verification_tool_low_eta")
        p.add_argument("--support_dir", default="data/testbeds/verification_tool_support")
        p.add_argument("--all_tool_glob", default="data/testbeds/verification_tool*/phase_summary.csv",
                       help="every 5-seed tool replay grid (incl. routing/support calibration grids) for the max-MI check")
        p.add_argument("--out_dir", default="data/outputs/testbeds_tool_summary")
        return p.parse_args()
    METRICS = ["top1_share", "kappa_map", "n_eff", "collapse_signal", "locally_contracting", "i_exp_norm"]
    def sweep_table(df: pd.DataFrame, source: str, alpha=None) -> pd.DataFrame:
        """Seed-level means over the alpha grid (or a single alpha), then mean and SE across seeds."""
        d = df if alpha is None else df[np.isclose(df["alpha"], alpha)]
        per_seed = d.groupby(["eta", "seed"])[METRICS].mean().reset_index()
        rows = []
        for eta, g in per_seed.groupby("eta"):
            row = {"source_grid": source, "alpha": "mean over grid" if alpha is None else alpha, "eta": eta, "n_seeds": g["seed"].nunique()}
            for m in METRICS:
                v = g[m].astype(float).to_numpy()
                row[f"{m}_mean"] = v.mean()
                row[f"{m}_se"] = v.std(ddof=1) / np.sqrt(len(v))
            rows.append(row)
        return pd.DataFrame(rows)
    def main():
        args = parse_args()
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        main_df = C.load_phase(Path(args.main_dir))
        low_df = C.load_phase(Path(args.low_eta_dir))
        sup_df = C.add_flags(C.load_phase(Path(args.support_dir)))
        for d in (main_df, low_df, sup_df):
            assert sorted(d["seed"].unique()) == [11, 22, 33, 44, 55]
        assert sorted(main_df["alpha"].unique()) == sorted(low_df["alpha"].unique())
        # replay settings that must agree for the two no-support grids to be concatenated
        cfg_m = json.loads((Path(args.main_dir) / "run_config.json").read_text())["args"]
        cfg_l = json.loads((Path(args.low_eta_dir) / "run_config.json").read_text())["args"]
        same_keys = ["runs_dir", "alphas", "support_strengths", "route_strengths", "score_weights", "route_zscore", "route_power",
                     "endpoint_last_k", "stability_context_stride", "active_eps", "collapse_top1_threshold", "context_key"]
        diff = {k: (cfg_m.get(k), cfg_l.get(k)) for k in same_keys if cfg_m.get(k) != cfg_l.get(k)}
        assert not diff, diff
        ov = main_df.merge(low_df, on=["seed", "eta", "alpha"], suffixes=("_m", "_l"))
        overlap = {"n_overlap_rows": int(len(ov)), "eta_values": sorted(ov["eta"].unique().tolist()),
                   "max_abs_diff_top1": float((ov.top1_share_m - ov.top1_share_l).abs().max()),
                   "max_abs_diff_kappa": float((ov.kappa_map_m - ov.kappa_map_l).abs().max())}
        assert overlap["max_abs_diff_top1"] < 1e-9 and overlap["max_abs_diff_kappa"] < 1e-9, overlap

        low_only = low_df[~low_df["eta"].isin(main_df["eta"].unique())]
        tabs = []
        for alpha in [None, 0.3, 1.0]:
            tabs.append(sweep_table(low_only, "verification_tool_low_eta", alpha))
            tabs.append(sweep_table(main_df, "verification_tool_main", alpha))
        tab = pd.concat(tabs, ignore_index=True)
        tab.to_csv(out_dir / "eta_sweep_no_support.csv", index=False)

        # ---- phantom diversity ----------------------------------------------------------------------
        fams = {
            "diffuse_coexistence": sup_df["phase_label"] == "diffuse_coexistence",
            "concentrated_coexistence": sup_df["phase_label"] == "concentrated_coexistence",
            "coexistence": sup_df["coexistence_descriptor"],
            "specialization": sup_df["specialization_descriptor"],
            "any_diversity": sup_df["any_diversity_descriptor"],
            "non_collapse": sup_df["collapse_signal"] == 0,
        }
        ph = C.phantom_table(sup_df, fams)
        ph.insert(0, "scope", "verification_tool_support")
        nosup = pd.concat([low_only, main_df], ignore_index=True)
        nosup = C.add_flags(nosup)
        ph2 = C.phantom_table(nosup, {"any_diversity": nosup["any_diversity_descriptor"], "non_collapse": nosup["collapse_signal"] == 0})
        ph2.insert(0, "scope", "no_support: verification_tool_low_eta (eta<0.2) + verification_tool_main")
        ph_main = C.phantom_table(C.add_flags(main_df), {"any_diversity": C.add_flags(main_df)["any_diversity_descriptor"]})
        ph_main.insert(0, "scope", "verification_tool_main")
        pd.concat([ph, ph_main, ph2], ignore_index=True).to_csv(out_dir / "phantom_rate.csv", index=False)

        # ---- summary ---------------------------------------------------------------------------------
        allgrid = tab[tab["alpha"].astype(str) == "mean over grid"].sort_values("eta")
        g = lambda eta, col: float(allgrid[np.isclose(allgrid.eta, eta)][col].iloc[0])  # noqa: E731
        every = pd.concat([main_df, low_df, sup_df], ignore_index=True)
        mi_rows = []
        for f in sorted(glob.glob(args.all_tool_glob)):
            d = pd.read_csv(f)
            a = json.loads((Path(f).parent / "run_config.json").read_text())["args"]
            mi_rows.append({"grid": Path(f).parent.name, "n_endpoints": len(d), "n_seeds": d["source_path"].map(C.seed_from_source).nunique(),
                            "max_i_exp_norm": float(d["i_exp_norm"].max()), "n_specialization_signal": int((d["specialization_signal"] == 1).sum()),
                            "n_non_collapse": int((d["collapse_signal"] == 0).sum()), "route_strengths": a["route_strengths"],
                            "support_strengths": a["support_strengths"], "etas": a["etas"], "alphas": a["alphas"]})
        mi_tab = pd.DataFrame(mi_rows)
        mi_tab.to_csv(out_dir / "max_context_mi_all_tool_grids.csv", index=False)
        summary = {
            "seeds": [11, 22, 33, 44, 55],
            "max_i_exp_norm_by_grid_all_5seed_tool_replays": {r.grid: round(r.max_i_exp_norm, 4) for r in mi_tab.itertuples()},
            "n_specialization_signal_all_5seed_tool_replays": int(mi_tab["n_specialization_signal"].sum()),
            "n_rows": {"main": int(len(main_df)), "low_eta": int(len(low_df)), "low_eta_below_0.2": int(len(low_only)), "support": int(len(sup_df))},
            "eta_0.2_overlap_check": overlap,
            "main_grid_top1_eta0.2_to_8": [g(0.2, "top1_share_mean"), g(8.0, "top1_share_mean")],
            "main_grid_kappa_eta0.2_to_8": [g(0.2, "kappa_map_mean"), g(8.0, "kappa_map_mean")],
            "main_grid_collapse_rate": float(main_df["collapse_signal"].mean()),
            "low_eta_top1_by_eta": {C.fmt(r.eta): round(r.top1_share_mean, 4) for r in allgrid[allgrid.eta < 0.2].itertuples()},
            "low_eta_kappa_by_eta": {C.fmt(r.eta): round(r.kappa_map_mean, 4) for r in allgrid[allgrid.eta < 0.2].itertuples()},
            "low_eta_collapse_rate_by_eta": {C.fmt(r.eta): round(r.collapse_signal_mean, 4) for r in allgrid[allgrid.eta < 0.2].itertuples()},
            "locally_contracting_rate_by_eta": {C.fmt(r.eta): round(r.locally_contracting_mean, 4) for r in allgrid.itertuples()},
            "min_seed_mean_kappa_over_sweep": float(allgrid["kappa_map_mean"].min()),
            "phantom_support_grid": ph.set_index("family")[["n_descriptor_positive", "stability_failures", "phantom_share", "wilson_lo", "wilson_hi",
                                                              "seed_bootstrap_lo", "seed_bootstrap_hi", "seed_rate_min", "seed_rate_max"]].round(4).to_dict("index"),
            "phantom_no_support_with_low_eta": ph2.set_index("family")[["n_descriptor_positive", "stability_failures", "phantom_share"]].round(4).to_dict("index"),
            "max_i_exp_norm": {"grids_with_stability_audit": float(every["i_exp_norm"].max()), "support_grid": float(sup_df["i_exp_norm"].max()),
                               "main_grid": float(main_df["i_exp_norm"].max())},
            "n_specialization_descriptor_all_grids": int((every["specialization_signal"] == 1).sum()),
            "support_grid_collapse_rate_by_beta_sup": {C.fmt(k): round(float(v), 4) for k, v in sup_df.groupby("beta_sup")["collapse_signal"].mean().items()},
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    main()


# ====================================================================================================
# fragility
# ====================================================================================================
def main_fragility():
    def parse_args() -> argparse.Namespace:
        p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        p.add_argument("--primary-dir", default="data/results/crossfit_local_amplification_eta2p25")
        p.add_argument("--invasion-dir", default="data/results/crossfit_invasion_validation_eta2p25")
        p.add_argument("--out-dir", default="data/outputs/fragility")
        return p.parse_args()
    def summarize_pairs(label, pairs: pd.DataFrame, cluster_col: str) -> dict:
        """pairs: columns phantom, stable, <cluster_col>."""
        diff = pairs["phantom"] - pairs["stable"]
        inf = small_cluster_sign_and_wild_inference(diff.groupby(pairs[cluster_col]).mean().to_numpy(dtype=float))
        return {
            "criterion": label, "n_pairs": int(len(pairs)), "n_clusters": int(inf["cluster_count"]),
            "phantom_mean": float(pairs["phantom"].mean()), "phantom_positive": int((pairs["phantom"] > 0).sum()),
            "stable_mean": float(pairs["stable"].mean()), "stable_positive": int((pairs["stable"] > 0).sum()),
            "pair_weighted_difference": float(diff.mean()), "pairs_with_positive_difference": int((diff > 0).sum()),
            "cluster_weighted_difference": float(inf["cluster_weighted_mean"]), "cluster_se": float(inf["cluster_standard_error"]),
            "wild_ci_low": float(inf["wild_cluster_bootstrap_t_ci_low"]), "wild_ci_high": float(inf["wild_cluster_bootstrap_t_ci_high"]),
            "positive_clusters": int(inf["positive_cluster_count"]),
            "sign_flip_p_one_sided": float(inf["exact_sign_flip_p_one_sided"]), "sign_flip_p_two_sided": float(inf["exact_sign_flip_p_two_sided"]),
        }
    def crossfit_pairs(result_dir: Path) -> pd.DataFrame:
        endpoints = pd.read_csv(result_dir / "endpoint_results.csv")
        wide = endpoints.pivot(index="pair_id", columns="stability_status", values="mean_test_log_amplification")
        sub = endpoints.loc[endpoints.stability_status == "phantom"].set_index("pair_id")
        wide["seed"] = sub["population_seed"]
        assert (wide["seed"] == endpoints.loc[endpoints.stability_status == "stable"].set_index("pair_id")["population_seed"]).all()
        wide["support_abs_difference"] = sub["support_abs_difference"]
        return wide.reset_index()
    def check(row: dict, stored: dict, mapping: dict) -> None:
        for mine, theirs in mapping.items():
            assert np.isclose(row[mine], stored[theirs], rtol=1e-9, atol=1e-12), (row["criterion"], mine)
    def main() -> None:
        args = parse_args()
        out_dir = ROOT / args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        primary = crossfit_pairs(ROOT / args.primary_dir)
        r1 = summarize_pairs("primary", primary, "seed")
        check(r1, json.loads((ROOT / args.primary_dir / "manifest.json").read_text()), {
            "phantom_mean": "phantom_mean_log_amplification", "stable_mean": "stable_mean_log_amplification",
            "cluster_weighted_difference": "paired_log_amplification_difference_cluster_weighted",
            "wild_ci_low": "wild_cluster_bootstrap_t_ci_low", "wild_ci_high": "wild_cluster_bootstrap_t_ci_high",
            "sign_flip_p_one_sided": "exact_sign_flip_p_one_sided",
            "phantom_positive": "phantom_endpoint_amplified_count", "stable_positive": "stable_endpoint_amplified_count",
        })
        r1["support_diff_max"] = float(primary["support_abs_difference"].max())
        gain = pd.DataFrame([r1])
        gain.to_csv(out_dir / "theory_direction_log_gain_rows.csv", index=False)

        inv = pd.read_csv(ROOT / args.invasion_dir / "endpoint_results_by_dose.csv")
        dose_summary = pd.read_csv(ROOT / args.invasion_dir / "dose_summary.csv").set_index("dose")
        inv_rows = []
        for dose, sub in inv.groupby("dose", sort=True):
            out = {"dose": float(dose), "is_primary": int(sub["is_primary_dose"].iloc[0]), "n_endpoints": int(len(sub))}
            for kind, col in (("raw", "mean_test_log_invasion"), ("boundary", "mean_test_boundary_log_invasion")):
                inf = small_cluster_sign_and_wild_inference(sub[col].groupby(sub["population_seed"]).mean().to_numpy(dtype=float))
                stored = dose_summary.loc[dose]
                for mine, theirs in (("cluster_weighted_mean", "cluster_weighted_mean"), ("wild_cluster_bootstrap_t_ci_low", "wild_cluster_bootstrap_t_ci_low"),
                                     ("wild_cluster_bootstrap_t_ci_high", "wild_cluster_bootstrap_t_ci_high")):
                    assert np.isclose(inf[mine], stored[f"{kind}_{theirs}"])
                out.update({
                    f"{kind}_endpoint_mean": float(sub[col].mean()), f"{kind}_positive_endpoints": int((sub[col] > 0).sum()),
                    f"{kind}_cluster_mean": float(inf["cluster_weighted_mean"]),
                    f"{kind}_ci_low": float(inf["wild_cluster_bootstrap_t_ci_low"]), f"{kind}_ci_high": float(inf["wild_cluster_bootstrap_t_ci_high"]),
                    f"{kind}_positive_clusters": int(inf["positive_cluster_count"]), f"{kind}_p_one_sided": float(inf["exact_sign_flip_p_one_sided"]),
                    "n_clusters": int(inf["cluster_count"]),
                })
            inv_rows.append(out)
        invasion = pd.DataFrame(inv_rows)
        invasion.to_csv(out_dir / "theory_direction_invasion_rows.csv", index=False)

        pd.set_option("display.width", 250)
        print(gain.T.to_string())
        print(invasion.T.to_string())
    main()


def main_unstable_diversity_deepseek():
    """The same rates under the DeepSeek-V3 evaluator (seed 55 is a truncated run and is excluded)."""
    sys.argv = [sys.argv[0], "--labels", "data/results_reuse_clean/threshold_robustness/phantom_breakdown_labels.csv.gz", "--exclude-seeds", "55",
                "--out-dir", "data/outputs/unstable_diversity_rates_deepseek", *sys.argv[1:]]
    main_unstable_diversity()


SUBCOMMANDS = {"unstable_diversity": main_unstable_diversity, "evaluator_shift": main_evaluator_shift, "testbeds": main_testbeds, "tool": main_tool, "fragility": main_fragility, "unstable_diversity_deepseek": main_unstable_diversity_deepseek}


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=False)
    ap.add_argument("command", nargs="?", choices=["all", *SUBCOMMANDS], help="which part to run (all = every part in order)")
    args, rest = ap.parse_known_args()
    if args.command is None:  # no command (or just -h): show this help; `<command> -h` shows the command's own options
        ap.print_help(); sys.exit(0 if sys.argv[1:] else 2)
    for cmd in (list(SUBCOMMANDS) if args.command == "all" else [args.command]):
        sys.argv = [f"{sys.argv[0]} {cmd}", *rest]
        print(f"== {cmd}", flush=True)
        SUBCOMMANDS[cmd]()


if __name__ == "__main__":
    main()
