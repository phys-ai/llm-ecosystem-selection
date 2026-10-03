# How Selection Shapes Diversity in LLM Ecosystems — reproduction package

Code and data behind every figure and number of the NeurIPS 2026 paper (OpenReview: https://openreview.net/forum?id=RM3k5s1pJM).

```
llm-ecosystem-selection/
  figures.py             regenerate the figures from data/              python3 figures.py all | fig2 | fig3 | fig4_5_11 | testbeds | online | appendix
  tables.py              re-derive the quoted numbers from data/        python3 tables.py all | unstable_diversity | evaluator_shift | testbeds | tool | fragility
  check.py               print every number quoted in the paper next to the file it comes from   python3 check.py [--grep tool]
  run_all.sh             the three commands above (~2 min)
  data/                  stored analysis outputs the scripts read: data.tar.gz of the Zenodo record, extracted here (manifest below)
  figures/               the PDFs the scripts write (PAPER_DIR=<LaTeX directory> writes into its figures/ instead)
  pipeline/              how data/ was made from the raw logs: collect -> replay -> early_warning / fragility -> appendix figures (pipeline/README.md)
  runs/, runs_testbed_*/ raw logs of the main population and of the five testbeds (Zenodo: https://doi.org/10.5281/zenodo.23112844; input of pipeline/, not needed for the three commands above)
```

Setup: Python >= 3.10, `pip install -r requirements.txt` (numpy, pandas, scipy, scikit-learn, matplotlib), then `data.tar.gz`
from the Zenodo record (https://doi.org/10.5281/zenodo.23112844) extracted at the package root (`tar xzf data.tar.gz`, gives `data/`, 0.5 GB).

## 1. Figures (`python3 figures.py all`, ~1 min)

| Paper figure | File | Sub-command | Reads |
|---|---|---|---|
| Fig. 1 | `evolution_experiment_setup.pdf` | — (drawn by hand) | — |
| Fig. 2 | `fig2_cr.pdf` | `fig2` | `data/results/fig0_4_full_grid`, `data/outputs/fig2_exposure_snapshots` |
| Fig. 3 | `fig3_main.pdf` | `fig3` regenerates the five panels (`fig3_panels.pdf`); the composite with the trait maps is assembled by hand | `data/results/fig0_4_full_grid`, `fig5_full_grid`, `trait_exposure_examples` |
| Fig. 4 | `fig4_cr.pdf` | `fig4_5_11` | `data/outputs/unstable_diversity_rates`, `per_round_vs_average` |
| Fig. 5 | `fig5_cr.pdf` | `fig4_5_11` | `data/outputs/early_warning_closed_form` |
| Fig. 6 | `fig6_heterogeneity.pdf` | `testbeds` | `data/testbeds/verification_heterogeneity_main`, `cr_heterogeneity_support_fine` |
| Fig. 7 (app.) | `fig7_local_stability_empirical_criticality.pdf` | `appendix` | `data/results/fig0_4_full_grid` |
| Fig. 8 (app.) | `fig9_operational_phase_diagram.pdf` | `appendix` | `fig0_4_full_grid` |
| Fig. 9 (app.) | `fig8_support_intervention.pdf` | `appendix` | `fig5_full_grid` |
| Fig. 10 (app.) | `fig13_route_sweep.pdf` | `appendix` | `appendix_route_sweep_parallel` |
| Fig. 11 (app.) | `fig_app_neff_avg_vs_inst.pdf` | `fig4_5_11` | `per_round_vs_average` |
| Fig. 12 (app.) | `fig4_phantom_diversity_rate.pdf` | `appendix` | `data/results/threshold_robustness` |
| Fig. 13 (app.) | `fig17_specialization_boundary_uncertainty.pdf` | `appendix` | `specialization_wide_dense_figS` |
| Fig. 14 (app.) | `fig6_representative_no_support_phase_slice_deepseek.pdf` | `appendix` | `data/results_reuse_clean/*` |
| Fig. 15 (app.) | `fig_app_testbed_{llama,8d,16d}.pdf`, `fig6_bfcl.pdf` | `testbeds` | `data/testbeds/*`, `early_warning_testbeds` |
| Fig. 16 (app.) | `fig18_closed_loop_validation.pdf` | `online` | `data/results/closed_loop_validation_grid` |

Every sub-command writes its PDFs into `figures/` (`appendix` keeps a copy under `data/results/figures/`).
The regenerated PDFs are identical to the ones in the paper (byte for byte, or pixel for pixel where the PDF writer embeds a date).

## 2. Numbers (`python3 tables.py all`, ~10 s) and where each quoted number lives

| Paper passage | Source file(s) under `data/` | Produced by |
|---|---|---|
| 4.3: 69% concentrated per round; B.5: median N_eff 28, leader changes 32 | `outputs/per_round_vs_average/summary_average_only_share.csv` | `pipeline/replay/per_round_vs_average.py` (replays `runs/`) |
| 4.5, B.7: 88% / 36-50% sign agreement, +0.12 / -0.13 | `outputs/evaluator_trait_shift/table_rows.csv` | `tables.py evaluator_shift` |
| B.3: Fig. 5 AUCs 0.963 / 0.942 / 0.922 / 0.882 | `outputs/early_warning_closed_form/endpoint_early_prediction.csv.gz` | `pipeline/early_warning/closed_form.py` |
| B.3 leakage: 0.974 / 0.978 / 0.864 (+- s.e.) | `results/early_warning_validation_grid/early_warning_predictions.csv.gz` (seed-wise AUC) | `pipeline/replay/early_warning_grid.sh` |
| B.3 leakage: < 2% label flips, 0.968 / 0.967 / 0.970, 0.898 | `outputs/early_warning_context_holdout/{label_flips,loso_auc_summary}.csv`, `results/early_warning_validation_grid_leakage_controls/table2_*.csv` | `pipeline/early_warning/context.py holdout`, `pipeline/replay/criticality_holdout.py` |
| B.3 nonlinear baselines | `outputs/early_warning_heldout_params/summary_auc.csv`, `outputs/early_warning_heldout_support/*` | `pipeline/early_warning/heldout.py heldout_params / heldout_support` |
| B.3 context shifts: 240 / 42 / 63%, AUCs | `testbeds/context_shift_early_warning_10seed/{scenario_stability,ood_evaluation_overall}.csv` | `pipeline/early_warning/context.py shift` |
| B.5 thresholds, near-critical band, numerical sweep | `results/threshold_robustness/*.csv` | `pipeline/appendix_figures.py` threshold workflow |
| B.6 fragility test | `outputs/fragility/theory_direction_{log_gain,invasion}_rows.csv` | `tables.py fragility` (from `results/crossfit_*_eta2p25/`) |
| B.7: 86% (81% / 92%) vs 85% | `outputs/unstable_diversity_rates{_deepseek,}/fig4_panelA_phantom_rate_main.csv` | `tables.py unstable_diversity` |
| B.8 testbeds: bands, counts, shares, AUCs | `outputs/fig6_heterogeneity/summary.json`, `outputs/fig15_*/summary.json`, `outputs/testbeds_table/robustness_table.csv`, `outputs/early_warning_testbeds/summary_auc.csv` | `figures.py testbeds`, `tables.py testbeds`, `pipeline/early_warning/heldout.py testbeds` |
| B.8 tool benchmark | `outputs/testbeds_tool_summary/summary.json`, `outputs/fig15_tool/summary.json` | `tables.py tool`, `figures.py testbeds` |
| B.8 tool normalization (78% / 5% of rounds, correlation -1.00, 0.85 vs 0.90) | raw tool logs (`runs_testbed_tool/`, Zenodo) | `check.py` (skipped when the logs are absent) |
| B.9 online validation: Spearman 0.69 / 0.72 / 0.75 / 0.79 | `results/closed_loop_validation_grid/replay_online_metric_agreement_5seeds.csv` | `pipeline/collect/online_validate.py` |

## 3. Data manifest (`data/`, ~0.5 GB, `data.tar.gz` of the Zenodo record)

| Folder | Content | Made by |
|---|---|---|
| `results/fig0_4_full_grid`, `fig5_full_grid`, `fig10_full_grid`, `specialization_wide_dense_figS`, `appendix_route_sweep_parallel` | replay grids of the main population (`phase_summary.csv` per seed) | `pipeline/replay/main_grids.sh` |
| `results/early_warning_validation_grid[_leakage_controls]`, `stability_threshold_robustness`, `threshold_robustness` | early-warning grid, numerical-robustness reruns, threshold tables | `early_warning_grid.sh`, `numerical_robustness.sh`, `appendix_figures.py` |
| `results/context_disjoint_criticality_full_holdout`, `crossfit_*_eta2p25`, `closed_loop_validation_grid`, `trait_exposure_examples` | context-holdout audit, fragility test, online validation, Fig. 3 trait maps | `replay/criticality_holdout.py`, `fragility/*.py`, `collect/online_validate.py` |
| `results_reuse_clean/` | the same grids under the DeepSeek-V3 evaluator | `replay/main_grids.sh` on the `*_reuse_together_evalaudit_compact` logs |
| `testbeds/` | replay grids of the five testbeds (`verification_*`: the grids of the tables; `cr_*_dense`: the dense support grids of the figures) | `replay/engine.py --estimate_stability --stability_context_stride 4 --route_zscore` |
| `outputs/<name>/` | one folder per script / sub-command, named after it | `tables.py`, `figures.py`, `pipeline/*` |
| `agent_metadata/` | the agents' trait coordinates of the five populations shared by the GPT and DeepSeek evaluations (B.7) | `collect/log_collect.py` (metadata.json) |

## Notes

* **Fig. 3** is assembled by hand from the panels that `figures.py fig3` produces (`fig3_panels.pdf`) and the trait maps of
  `data/results/trait_exposure_examples`.
* **B.3 leakage AUCs (0.974 / 0.978 / 0.864)** are the seed-wise AUCs of the stored out-of-fold predictions; `check.py` recomputes them.

## Glossary (names in the code and data vs. the paper)

| In the code / data | In the paper |
|---|---|
| `phantom`, `phantom_new`, `phantom_family`, `phantom_any_diversity` | unstable diversity (descriptor-diverse endpoint that fails the local-stability audit / is not simultaneously diverse) |
| `simultaneously_diverse`, `conc_round_frac` | per-round (simultaneous) diversity, concentrated-round fraction (Sec. 4.3) |
| `kappa_map`, `locally_contracting`, `stable` | local stability audit (kappa_map < 0 = locally stable) |
| `i_exp_norm`, `specialization_signal` | normalized exposure-context mutual information, specialization descriptor |
| `beta_sup`, `beta_route`, `lambda_risk`, `ctx` / `context_trait_strength` | support, routing, risk-penalty strengths; context fidelity |
| `testbed`, `verification_*`, `cr_*_dense` (`data/testbeds`) | the five robustness testbeds (heterogeneity, 8D, 16D, Llama, tool; Figs. 6, 15, B.8) and their dense grids |
| `reuse` / `results_reuse_clean` | the same posts re-evaluated by DeepSeek-V3 (B.7, Fig. 14) |
| `G7_early_warning`, `replay_logspace_*` | early warning on the log-space replay (Fig. 5) |
| `early_frac` | fraction of the trajectory used for early warning (0.1 = the first 10%) |
| `eta2p25` | eta = 2.25 |
