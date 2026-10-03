#!/usr/bin/env python3
"""Print every number quoted in the paper's main text and appendix from the stored analysis outputs.

Each block prints the value(s) exactly as the paper quotes them, next to the file they are read from, so a reader
can check the PDF against the data without re-running any model fit.  Run from the package root:

    python3 check.py             # all sections
    python3 check.py --grep tool # only blocks whose title matches

No file is written.  Blocks marked [recompute] redo a light computation (an AUC from stored out-of-fold
predictions, a count over a replay grid, one pass over the raw tool logs); everything else is a lookup.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
R = ROOT / "data/outputs"
A = ROOT / "data/results"
T = ROOT / "data/testbeds"
BLOCKS = []
EPS = 1e-12


def block(title):
    def deco(fn):
        BLOCKS.append((title, fn)); return fn
    return deco


def say(claim, *vals):
    print(f"  {claim}")
    for v in vals:
        print(f"      -> {v}")


# ----------------------------------------------------------------------------------------------- main text
@block("Main text, Sec. 4.3 (unstable diversity): 69% of unstable-diversity endpoints are concentrated per round")
def _():
    d = pd.read_csv(R / "per_round_vs_average/summary_average_only_share.csv")
    r = d[(d.scope == "support") & (d.family == "any_diversity") & (d.condition == "all descriptor-positive") & (d.status == "phantom")].iloc[0]
    say("per-round N_eff <= 3 (or top-1 >= 0.8) share among unstable-diversity endpoints, support grid: 69%",
        f"{100 * r.share_average_only:.1f}%  (n = {int(r.n_descriptor_positive)})  [per_round_vs_average/summary_average_only_share.csv]")
    say("median N_eff of the averaged endpoint 28; median per-round N_eff; leader changes 32 times in 100 rounds (Appendix B.5)",
        f"median N_eff_avg = {r.median_N_eff_avg:.1f}, median N_eff_inst = {r.median_N_eff_inst:.2f}, median distinct winners = {r.median_winner_distinct:g}")


@block("Main text, Sec. 4.5 + Appendix B.4: evaluator dependence of which traits win (88% / 36-50%)")
def _():
    t = pd.read_csv(R / "evaluator_trait_shift/table_rows.csv")
    t = t[(t.subset == "no_support") & (t.eta_subset == "all_eta") & (t.unit == "seed_x_condition")].set_index("axis")
    for ax in ["stance_valence", "epistemic_style", "consensus_alignment", "social_framing"]:
        r = t.loc[ax]
        say(f"{ax}: sign agreement GPT vs DeepSeek", f"{100 * r.agree_nodz:.1f} +- {100 * r.agree_nodz_se:.1f}%  (n matched = {int(r.n_matched)})")
    r = t.loc["social_framing"]
    say("social framing mean shift: gpt-5.4-mini +0.12 (interpersonal), DeepSeek-V3 -0.13 (systemic)", f"gpt {r.mean_shift_gpt:+.2f}, DeepSeek {r.mean_shift_ds:+.2f}")


@block("Appendix B.3 (early warning): Fig. 5 AUCs 0.963 / 0.942 / 0.922 / 0.882 [recompute]")
def _():
    from sklearn.metrics import roc_auc_score
    d = pd.read_csv(R / "early_warning_closed_form/endpoint_early_prediction.csv.gz"); sub = d[d.diverse_at_10pct == 1]
    for col, lab in [("b_theory_var_t20", "closed-form variance"), ("seed_x_eta__full_early_warning", "full model"),
                     ("seed_x_eta__early_descriptor", "early descriptors"), ("seed_x_eta__param_rbf_svm", "parameter-only SVM")]:
        say(lab, f"AUC = {roc_auc_score(sub.y, sub[col]):.3f}  (n = {len(sub)}, test seed and selection strength held out)")


# ----------------------------------------------------------------------------------------------- appendix B.3
@block("Appendix B.3, Leakage controls: LOSO AUC 0.974+-0.003 / 0.978+-0.004 / 0.864+-0.004 [recompute]")
def _():
    from sklearn.metrics import roc_auc_score
    p = pd.read_csv(A / "early_warning_validation_grid/early_warning_predictions.csv.gz")
    p["seed"] = p.source_result_dir.str.extract(r"run_popseed_(\d+)")[0]
    s = p[(p.early_frac == 0.1) & (p.target == "label_unstable")]
    for m, lab in [("early_descriptor", "early descriptors"), ("fluctuation_only", "fluctuation features"), ("parameter_only", "parameter-only")]:
        a = s[s.model == m].groupby("seed").apply(lambda g: roc_auc_score(g.y_true, g.y_score), include_groups=False)
        say(lab, f"{a.mean():.3f} +- {a.sem():.3f}  (mean +- s.e. over {len(a)} held-out seeds)")


@block("Appendix B.3, Leakage controls: label flips < 2%, full-model AUC 0.968 / 0.967 / 0.970, no-concentration AUC 0.898")
def _():
    f = pd.read_csv(R / "early_warning_context_holdout/label_flips.csv").set_index("target")
    say("labels that change when the target is recomputed post-window / on the strict holdout (of 1,920)",
        f"post-window: {int(f.loc['postearly', 'n_flips'])} ({100 * f.loc['postearly', 'n_flips'] / 1920:.1f}%)",
        f"strict holdout: {int(f.loc['context_holdout', 'n_flips'])} ({100 * f.loc['context_holdout', 'n_flips'] / 1920:.1f}%)")
    s = pd.read_csv(R / "early_warning_context_holdout/loso_auc_summary.csv")
    for m in ["full_early_warning", "early_descriptor", "parameter_only"]:
        r = s[s.model == m].set_index("target").pooled_loso_roc_auc
        say(f"{m}: original / post-window / strict holdout", f"{r['baseline']:.3f} / {r['postearly']:.3f} / {r['context_holdout']:.3f}")
    t = pd.read_csv(A / "early_warning_validation_grid_leakage_controls/table2_early_warning_metrics_for_paper.csv")
    r = t[(t.target == "label_unstable") & (t.model == "leakage_guard_full") & (t.split == "leave_seed_out") & (t.control == "none") & (t.early_frac == 0.1)].iloc[0]
    say("excluding all concentration-derived features", f"AUC = {r.roc_auc_mean:.3f}")


@block("Appendix B.3, Nonlinear parameter-only baselines (B10 / A3)")
def _():
    b = pd.read_csv(R / "early_warning_heldout_params/summary_auc.csv"); b = b[b.target == "context_holdout"]
    g = b[b.split == "seed"].set_index("model").seed_mean_roc_auc
    say("leave-one-seed-out, within grid: RBF SVM 0.990, random forest 0.996, full model 0.983",
        f"SVM {g['param_rbf_svm']:.3f}, forest {g['param_random_forest']:.3f}, full {g['full_early_warning']:.3f}  (seed means)")
    rows = []
    for split in ["seed_x_eta", "seed_x_beta_sup", "seed_x_alpha"]:
        g = b[b.split == split].set_index("model").seed_mean_roc_auc
        rows.append(f"{split}: SVM {g['param_rbf_svm']:.3f}, forest {g['param_random_forest']:.3f}, full {g['full_early_warning']:.3f}, logistic {g['parameter_only']:.3f}")
    say("each selection strength / support strength / context fidelity held out with the seed: full 0.966-0.978 vs parameter-only 0.80-0.96", *rows)
    a = pd.read_csv(R / "early_warning_heldout_support/summary_auc.csv")
    say("auxiliary stable-specialization target: 42 of 1,920 conditions", f"n_positive = {int(a.n_positive.iloc[0])} of {int(a.n.iloc[0])}")
    g = a[a.split == "seed"].set_index("model").seed_mean_roc_auc
    say("additive logistic below chance; quadratic logistic / RBF SVM match the full model within the grid",
        f"additive {g['param_additive_logistic']:.2f}, quadratic {g['param_quadratic_logistic']:.3f}, SVM {g['param_rbf_svm']:.3f}, full {g['full_early_warning']:.3f}")
    g = a[a.split == "seed_x_beta_sup"].set_index("model").pooled_roc_auc
    bl = g[[m for m in g.index if m.startswith("param_") and not m.endswith("_margin")]]  # the margin variant is a sensitivity check
    say("each support strength held out with the seed: full 0.829 vs best parameter-only 0.674 (pooled out-of-fold AUC)",
        f"full {g['full_early_warning']:.3f}; best baseline {bl.idxmax()} {bl.max():.3f}")
    st = pd.read_csv(R / "early_warning_heldout_support/sign_tests_full_vs_baselines.csv")
    r = st[(st.split == "seed_x_beta_sup") & (st.baseline == bl.idxmax())].iloc[0]
    say("full model wins in 10 of 10 seeds (p = 0.002, exact sign test)", f"wins {int(r.wins_full)} of {int(r.n_seeds)} vs {r.baseline}, two-sided p = {r.sign_test_p_two_sided:.3f}")


@block("Appendix B.3, Abrupt context shifts (240 shifts; 42 remove contraction; 63% detected; AUC 0.986 / 0.998 / 0.995 / 0.706)")
def _():
    s = pd.read_csv(T / "context_shift_early_warning_10seed/scenario_stability.csv")
    n_stay, n_loss = int((s.condition == "pure_shift_stable").sum()), int((s.condition == "context_induced_loss").sum())
    say("context shifts (no eta change): 240 = 198 that stay contracting + 42 (17.5%) that remove contraction",
        f"{n_stay + n_loss} shifts, {n_loss} ({100 * n_loss / (n_stay + n_loss):.1f}%) lose contraction, {n_stay} stay contracting",
        f"operating points {s.operating_point_id.nunique()}, magnitudes {sorted(float(x) for x in s.shift_magnitude.unique())}, seeds {s.run_name.nunique()}")
    o = pd.read_csv(T / "context_shift_early_warning_10seed/ood_evaluation_overall.csv"); o = o[o.window == "crossing"].set_index("model")
    for m in ["fluctuation_only", "early_descriptor", "full_early_warning", "parameter_only"]:
        say(m, f"ROC AUC {o.loc[m, 'roc_auc']:.3f}; false alarms on stable shifts {100 * o.loc[m, 'pure_shift_fpr_calibrated']:.1f}%; detected {100 * o.loc[m, 'loss_tpr_calibrated']:.0f}%")


# ----------------------------------------------------------------------------------------------- appendix B.5
@block("Appendix B.5, Threshold robustness: near-critical band 1.4% -> 6.1%, non-borderline failures 83-86% [recompute]")
def _():
    d = pd.read_csv(A / "threshold_robustness/phantom_breakdown_labels.csv.gz")
    pos = d[d.phantom_family != "none"]                       # descriptor-positive endpoints of the main and specialization sweeps
    say("descriptor-positive endpoints", f"{len(pos)}")
    for tau in (0.025, 0.05, 0.10):
        k = pos.kappa_map; lc, nc = (k < -tau).mean(), (k.abs() <= tau).mean()
        say(f"tau = {tau}", f"near-critical {100 * nc:.1f}%, non-borderline failure {100 * (1 - lc - nc):.1f}%, locally contracting {100 * lc:.1f}%")


@block("Appendix B.5, Thresholds and numerical stability (one-at-a-time 95.3-100%; kappa<-0.10 keeps 84% / 90%; 625 joint settings; 24,000 matched endpoints)")
def _():
    t = pd.read_csv(A / "threshold_robustness/threshold_robustness_table_for_paper.csv").set_index("setting_id")
    one = t[t.index.str.startswith(("specialization_mi=", "winner_switch=", "collapse_", "concentrated_"))]
    say("phase label preserved under one-at-a-time changes of the I_exp, winner-switch, collapse and concentration thresholds",
        f"{100 * one.phase_agreement.min():.1f}-{100 * one.phase_agreement.max():.1f}%  (contracting-set Jaccard {one.locally_contracting_jaccard_vs_baseline.min():.3f})")
    r = t.loc["stable_kappa_margin=0.1"]
    say("contraction margin kappa < -0.10: keeps 84% of the contracting set and 90% of the stable-specialization set (Jaccard)",
        f"{r.locally_contracting_jaccard_vs_baseline:.3f} / {r.stable_specialization_jaccard_vs_baseline:.3f}")
    j = t[t.index.str.startswith("joint|")]
    say("625 joint combinations: phase agreement median 91.5% (min 80.2%)", f"n = {len(j)}, median {100 * j.phase_agreement.median():.1f}%, min {100 * j.phase_agreement.min():.1f}%")
    say("stable-specialization share 2.1-6.2%, contracting rate 12.9%",
        f"{100 * j.robust_stable_specialization_rate.min():.1f}-{100 * j.robust_stable_specialization_rate.max():.1f}%; contracting {100 * j.robust_locally_contracting_rate.min():.1f}-{100 * j.robust_locally_contracting_rate.max():.1f}%")
    n = pd.read_csv(A / "threshold_robustness/numerical_stability_robustness_vs_baseline.csv").iloc[0]
    say("numerical sweep: median |delta kappa| 0.63 over 24,000 matched endpoints, no decision flips",
        f"{int(n.n_matched)} endpoints, median {n.delta_kappa_abs_median:.2f}, contracting flips {n.locally_contracting_flip_rate:g}, stable-specialization flips {n.stable_specialization_flip_rate:g}")


# ----------------------------------------------------------------------------------------------- appendix B.6
@block("Appendix B.6, Theory-directed fragility test (24 pairs at eta = 2.25)")
def _():
    g = pd.read_csv(R / "fragility/theory_direction_log_gain_rows.csv").set_index("criterion").loc["primary"]
    say("unstable-diversity endpoints amplify in 17 of 24 (mean log-gain 0.38)", f"{int(g.phantom_positive)} of {int(g.n_pairs)}, mean {g.phantom_mean:.2f}")
    say("controls contract in 22 of 24 (mean -1.53)", f"{int(g.n_pairs - g.stable_positive)} of {int(g.n_pairs)}, mean {g.stable_mean:.2f}")
    say("paired difference 1.87, seed-clustered 95% CI [1.67, 2.07], positive in all 24 pairs and all 10 seeds, p = 1/1024",
        f"{g.cluster_weighted_difference:.2f} [{g.wild_ci_low:.2f}, {g.wild_ci_high:.2f}], {int(g.pairs_with_positive_difference)}/{int(g.n_pairs)} pairs, "
        f"{int(g.positive_clusters)}/{int(g.n_clusters)} seeds, p = {g.sign_flip_p_one_sided:.6f} (= 1/{round(1 / g.sign_flip_p_one_sided)})")
    d = pd.read_csv(R / "fragility/theory_direction_invasion_rows.csv").set_index("dose")
    r = d.loc[1e-5]
    say("invasion direction: positive held-out log multiplier at all 24 endpoints, 4.62 [4.19, 5.05] at invasion mass 1e-5",
        f"positive {int(r.raw_positive_endpoints)}/{int(r.n_endpoints)}; {r.raw_cluster_mean:.2f} [{r.raw_ci_low:.2f}, {r.raw_ci_high:.2f}]")
    say("multipliers fall with the invasion mass", ", ".join(f"mass {m:g}: {d.loc[m, 'raw_cluster_mean']:.2f}" for m in d.index))


# ----------------------------------------------------------------------------------------------- appendix B.7
@block("Appendix B.7, Evaluator-family ablation: 2,400 matched conditions; unstable diversity 86% (81% / 92%) vs 85% in the main population")
def _():
    s = json.load(open(R / "evaluator_trait_shift/summary.json"))
    say("matched no-support conditions", f"{s['n_matched_by_grid']['no_support']}")
    for name, d in [("DeepSeek-V3 evaluator", "unstable_diversity_rates_deepseek"), ("main population (gpt-5.4-mini)", "unstable_diversity_rates")]:
        t = pd.read_csv(R / f"{d}/fig4_panelA_phantom_rate_main.csv").set_index("family")
        say(name, ", ".join(f"{fam}: {100 * t.loc[fam, 'phantom_pooled_share']:.0f}% ({int(t.loc[fam, 'not_locally_contracting'])}/{int(t.loc[fam, 'n_descriptor_positive'])})" for fam in ["Any diversity", "Coexistence", "Specialization"]))


# ----------------------------------------------------------------------------------------------- appendix B.8
def _grid(setting):
    main = T / f"verification_{setting}_main" / "phase_summary.csv"
    dense = {"heterogeneity": "cr_heterogeneity_support_fine", "llama": "cr_llama_support_dense", "8d": "cr_8d_support_dense", "16d": "cr_16d_support_dense"}
    sup = T / dense[setting] / "phase_summary.csv"
    return pd.read_csv(main), pd.read_csv(sup)


@block("Appendix B.8, Robustness testbeds: grids (640 / 4,125 / 45x42 / 6x33 cells), rank correlations 0.82-1.00, 123 stable cells in the main population")
def _():
    a = pd.read_csv(R / "testbeds_table/robustness_table.csv").set_index("setting")
    for s in ["heterogeneity", "llama", "8d", "16d", "tool"]:
        r = a.loc[s]
        say(f"{s}: seed-level rank correlation of eta with top-1 share / kappa (min over seeds)", f"top-1 {r.rho_eta_top1_seed_min:.2f}, kappa {r.rho_eta_kappa_seed_min:.2f}  [testbeds_table]")
    say("main population: stable-specialization cells on the 45x41 grid over 10 seeds; unstable-diversity share",
        f"{int(a.loc['main4d', 'stable_spec'])} cells; {a.loc['main4d', 'phantom_pct']:.1f}%")
    for s, d in [("heterogeneity", "fig6_heterogeneity"), ("llama", "fig15_llama"), ("8d", "fig15_8d"), ("16d", "fig15_16d")]:
        j = json.load(open(R / f"{d}/summary.json")); ph = j["phantom"]
        cells = pd.read_csv(R / f"{d}/support_grid_cells.csv")
        say(f"{s}: support grid {cells.eta.nunique()} x {cells.beta_sup.nunique()}; stable band {j['n_stable_specialization']} cells, eta {j['stable_band_eta']}, beta_sup {j['stable_band_beta_sup']}",
            f"unstable diversity {ph['Any diversity']['pct']}% ({ph['Any diversity']['n']}/{ph['Any diversity']['N']}); descriptor-positive specialization stable in {100 - ph['Specialization']['pct']:.1f}%",
            f"no stably specialized cell at eta >= 2.75: {bool((cells[cells.eta >= 2.75].stable_specialization_rate == 0).all())}",
            f"collapse rate at eta >= 5 (support grid): {100 * _grid(s)[1].query('eta >= 5').phase_label.str.contains('collapse').mean():.0f}%")
    r = a.loc["main4d"]
    say("main population (matched protocol, first 100 timesteps): descriptor-positive specialization stable in 7.3%",
        f"{int(r.stable_spec)} of {int(r.descriptor_spec)} = {100 * r.stable_spec / r.descriptor_spec:.1f}%  [testbeds_table]")


@block("Appendix B.8, Routing and support: context fidelity 0.3 -> 1.0 raises normalized MI roughly threefold; no no-support endpoint locally contracting")
def _():
    for s in ["heterogeneity", "8d"]:
        main, _ = _grid(s)
        g = main.groupby("alpha").i_exp_norm.mean()
        lo, hi = g.loc[g.index.min()], g.loc[g.index.max()]
        say(f"{s}: mean I_exp_norm at alpha {g.index.min():g} / {g.index.max():g}", f"{lo:.3f} -> {hi:.3f} (x{hi / lo:.1f}); locally contracting no-support endpoints: {int((main.kappa_map < 0).sum())} of {len(main)}")


@block("Appendix B.8, Early warning on the testbeds (within-grid 0.80-0.82 -> 0.92-0.94; LOSO gain <= 0.07; held-out eta: SVM 0.96-0.97 vs 0.89-0.94; main 0.97 vs 0.93; tool 0.92 vs 0.90 and 0.865 -> 0.916)")
def _():
    a = pd.read_csv(R / "testbeds_table/robustness_table.csv").set_index("setting")
    for s in ["heterogeneity", "llama", "8d", "16d"]:
        r = a.loc[s]
        say(f"{s}: within-grid parameter-only -> early descriptors; leave-one-seed-out", f"{r.auc_within_param:.2f} -> {r.auc_within_desc:.2f}; LOSO {r.auc_loso_param:.2f} -> {r.auc_loso_desc:.2f}")
    r = a.loc["tool"]; say("tool: within-grid parameter-only -> early descriptors (6x12 grid)", f"{r.auc_within_param:.3f} -> {r.auc_within_desc:.3f}")
    c = pd.read_csv(R / "early_warning_testbeds/summary_auc.csv"); c = c[c.split == "seed_x_eta"]
    for s in ["heterogeneity", "llama", "8d", "16d", "tool"]:
        g = c[c.setting == s].set_index("model").seed_mean_roc_auc
        say(f"{s}: each selection strength held out with the seed", f"RBF SVM {g['param_rbf_svm']:.2f}; early descriptors {g['early_descriptor']:.2f}, fluctuation {g['fluctuation_only']:.2f}, full {g['full_early_warning']:.2f}")
    b = pd.read_csv(R / "early_warning_heldout_params/summary_auc.csv"); g = b[(b.target == "context_holdout") & (b.split == "seed_x_eta")].set_index("model").seed_mean_roc_auc
    say("main population: each selection strength held out with the seed", f"full {g['full_early_warning']:.2f}, early descriptors {g['early_descriptor']:.2f} vs RBF SVM {g['param_rbf_svm']:.2f}")


@block("Appendix B.8, Tool-routing benchmark (top-1 0.03 -> 0.97, kappa 0.01 -> 1.7, MI <= 0.075, 73% fail the audit)")
def _():
    s = json.load(open(R / "testbeds_tool_summary/summary.json"))
    t1, kp = s["low_eta_top1_by_eta"], s["low_eta_kappa_by_eta"]
    say("eta in [0.005, 0.15]: top-1 share and kappa", f"top-1 {t1['0.005']:.2f} -> {t1['0.15']:.2f}; kappa {kp['0.005']:.2f} -> peak {max(kp.values()):.2f}; collapse rate on the main grid (eta >= 0.2) {s['main_grid_collapse_rate']:.2f}")
    mi = s["max_i_exp_norm_by_grid_all_5seed_tool_replays"]
    say("max normalized MI over all tool grids (threshold 0.12)", f"{max(mi.values()):.3f} ({max(mi, key=mi.get)}); specialization-signal endpoints: {s['n_specialization_signal_all_5seed_tool_replays']}")
    c = json.load(open(R / "fig15_tool/summary.json")); ph = c["phantom_any_diversity"]
    say("non-collapsed endpoints failing the stability audit (support grid)", f"{ph['n']} of {ph['N']} = {100 * ph['n'] / ph['N']:.0f}%")
    say("support restores coexistence: collapse rate by beta_sup", f"beta_sup 0: {c['collapse_rate_by_beta_sup']['0.0']:.2f}; beta_sup 1: {c['collapse_rate_by_beta_sup']['1.0']:.3f}")
    say("normalization: 78% of rounds all correct, 5% all wrong, corr -1.00, collapsed agent 0.85 vs uniform 0.90", "next block (reads the raw tool logs)")




def tool_normalization(root):
    """Tool benchmark (Appendix B.8): share of rounds with every agent correct / wrong, correlation between an agent's mean
    standardized routing residual and its accuracy (the replay's --route_zscore standardizes s_t - s_bar across agents
    within a round, as in replay.transform_route_residual), and the accuracy of the collapse target against uniform
    exposure.  Reads the raw tool logs: runs_testbed_tool/ (extracted Zenodo archive) or runs_testbed_tool.tar.gz."""
    import gzip, io, tarfile
    per_seed = {}
    pat = re.compile(r"run_popseed_(\d+)\.checkpoints/timestep_(\d+)\.json(\.gz)?$")
    def take(name, fh):
        mo = pat.search(name)
        if mo:
            d = json.load(gzip.open(fh, "rt", encoding="utf-8") if mo.group(3) else io.TextIOWrapper(fh, encoding="utf-8"))
            per_seed.setdefault(int(mo.group(1)), {})[int(mo.group(2))] = [float(c["correct"]) for c in sorted(d["contents"], key=lambda c: c["agent_id"])]
    if (root / "runs_testbed_tool").is_dir():
        for f in sorted((root / "runs_testbed_tool").glob("run_popseed_*.checkpoints/timestep_*.json*")):
            take(str(f), open(f, "rb"))
    elif (root / "runs_testbed_tool.tar.gz").exists():
        with tarfile.open(root / "runs_testbed_tool.tar.gz", "r:gz") as tf:
            for m in tf:
                if m.isfile():
                    take(m.name, tf.extractfile(m))
    else:
        print("  (skipped: needs the raw tool logs, runs_testbed_tool/ from the Zenodo record)"); return
    pooled = dict(all_correct=[], all_wrong=[], corr=[], collapse_acc=[], uniform_acc=[], min_acc=[])
    for seed in sorted(per_seed):
        C = np.array([per_seed[seed][t] for t in sorted(per_seed[seed])])            # rounds x agents, 1 = correct
        acc = C.mean(0)                                                              # per-agent accuracy s_bar
        res = C - acc[None, :]                                                       # s_t - s_bar
        sd = res.std(1, keepdims=True)
        z = np.where(sd > EPS, (res - res.mean(1, keepdims=True)) / np.where(sd > EPS, sd, 1.0), res - res.mean(1, keepdims=True))
        corr = np.corrcoef(z.mean(0), acc)[0, 1]
        rec = dict(all_correct=(C.mean(1) == 1).mean(), all_wrong=(C.mean(1) == 0).mean(), corr=corr,
                   collapse_acc=acc[np.argmax(z.mean(0))], uniform_acc=acc.mean(), min_acc=acc.min())
        for k, v in rec.items():
            pooled[k].append(v)
        print(f"seed {seed}: rounds {C.shape[0]}, agents {C.shape[1]}; all correct {100 * rec['all_correct']:.0f}%, all wrong {100 * rec['all_wrong']:.0f}%; "
              f"corr(mean routing score, accuracy) = {corr:+.3f}; collapse target accuracy {rec['collapse_acc']:.3f} (least accurate {rec['min_acc']:.3f}), uniform {rec['uniform_acc']:.3f}")
    print(f"mean over seeds: all correct {100 * np.mean(pooled['all_correct']):.0f}%, all wrong {100 * np.mean(pooled['all_wrong']):.0f}%, corr {np.mean(pooled['corr']):+.2f}, "
          f"collapse target {np.mean(pooled['collapse_acc']):.2f} vs uniform {np.mean(pooled['uniform_acc']):.2f}")



@block("Appendix B.8, Tool benchmark normalization: 78% / 5% of rounds, correlation -1.00, 0.85 against 0.90 [recompute, raw logs]")
def _():
    tool_normalization(ROOT)


# ----------------------------------------------------------------------------------------------- appendix B.9
@block("Appendix B.9, Online generation validation (14 conditions, 70 matched endpoints, Spearman 0.69 / 0.72 / 0.75 / 0.79)")
def _():
    m = pd.read_csv(A / "closed_loop_validation_grid/replay_online_metric_agreement_5seeds.csv").set_index("metric")
    for k, lab in [("top1_share", "top-one share"), ("n_eff", "effective support"), ("i_exp_norm", "exposure-context MI"), ("winner_switch_rate", "winner switching")]:
        say(lab, f"Spearman rho = {m.loc[k, 'spearman']:.2f} (n = {int(m.loc[k, 'n'])}); online minus replay bias {m.loc[k, 'bias_closed_loop_minus_replay']:+.2f}")
    cfg = json.load(open(A / "closed_loop_validation_grid/closed_loop_run_config.json"))
    say("conditions / seeds", f"{len(cfg.get('conditions', cfg.get('grid', [])))} conditions; keys: {list(cfg)[:8]}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--grep", default=""); args = ap.parse_args()
    for title, fn in BLOCKS:
        if args.grep and not re.search(args.grep, title, re.I):
            continue
        print(f"\n== {title}")
        try:
            fn()
        except Exception as e:  # keep going so one missing file does not hide the rest
            print(f"  !! {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
