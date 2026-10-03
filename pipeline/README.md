# pipeline/ — how `data/` was produced from the raw logs

The stages below compute `data/` (the input of `figures.py`, `tables.py` and `check.py`) from the raw logs of the Zenodo
record, in this order. All commands run from the package root (`llm-ecosystem-selection/`) unless stated; each script
prints its options with `--help`.

```
collect/      LLM runs (API calls; not reproducible bit for bit)        -> runs/*.checkpoints, runs_testbed_*.tar.gz
replay/       replay the logs under the platform's exposure update       -> data/results/*, data/testbeds/*, data/outputs/replay_*, per_round*
early_warning/ classifiers on the first 10% of each replayed trajectory  -> data/outputs/early_warning_*
fragility/    theory-directed perturbations at eta = 2.25                 -> data/results/crossfit_*_eta2p25
appendix_figures.py  threshold robustness + appendix figures              -> data/results/threshold_robustness, data/results/figures
common/       shared helpers (loader, grids, per_round, logspace, pairings, plot, perturbation, testbeds, deepseek)
```

## 1. collect — generate the logs (hours per run, paid API)

| Script | Produces | Used by |
|---|---|---|
| `collect/log_collect.py` (`run.sh` has the exact commands) | `runs/run_popseed_<s>.checkpoints` — 96 agents x 200 timesteps, gpt-5.4-mini generation / evaluation / auditing; `*_reuse_together_evalaudit_compact` = the same posts re-evaluated by DeepSeek-V3 | everything |
| `collect/log_collect_heterogeneous.py`, `log_collect.py` (8D / 16D / Llama), `log_collect_tool.py` (`run_testbeds.sh`, `config_tool.json`) | `runs_testbed_{heterogeneity,8d,16d,llama,tool}.tar.gz` | the testbeds (Fig. 6, 15, B.8) |
| `collect/online_validate.py` (`config_online.json`) | `data/results/closed_loop_validation_grid/` — generation inside the loop | Fig. 16, B.9 |
| `collect/export_logs.py` | the published copies of the logs: `runs_public/*.tar.gz` (Reddit excerpt fields removed from every content row, one gzipped JSON per timestep; the replay reads them as they are) | — |

`topic_bank.sampled.jsonl` (256 prompts rewritten from Reddit posts, with the post ids) and `benchmarks/bfcl/` are the prompt
sources; `build_topic_bank.py` is the script that built the bank from the Reddit API.

## 2. replay — logs to replay grids

| Command (from `pipeline/`) | Output |
|---|---|
| `bash replay/main_grids.sh` (`INCLUDE_EXISTING=1`), `bash replay/early_warning_grid.sh`, `bash replay/numerical_robustness.sh` | `data/results/{fig0_4_full_grid, fig5_full_grid, fig10_full_grid, specialization_wide_dense_figS, appendix_route_sweep_parallel}`, `early_warning_validation_grid`, `stability_threshold_robustness` |
| `python3 replay/engine.py --runs_dir <extracted runs_testbed_*> --estimate_stability --stability_context_stride 4 --route_zscore ...` (grids in each testbed's `run_config.json`) | `data/testbeds/{verification_*_main, cr_*_support_dense, ...}` |
| `python3 replay/criticality_holdout.py process-run ... ; evaluate` | `data/results/context_disjoint_criticality_full_holdout` |
| `python3 replay/per_round_vs_average.py` | `data/outputs/per_round_vs_average` (Sec. 4.3, Fig. 4(b), Fig. 11) |
| `python3 replay/logspace_grids.py replay --grids main_no_support,main_support` then `... adapt` | `data/outputs/replay_logspace_grids`, `replay_logspace_adapted` |
| `python3 replay/per_round.py main` / `testbeds` | `data/outputs/per_round_stats[_testbeds]` |
| `python3 replay/engine.py --runs_dir runs/run_popseed_11.checkpoints --etas 0.25,4.0 --alphas 1 --support_strengths 0 --route_strengths 2 --risk_lambdas 0 --route_zscore --endpoint_last_k 100 --save_exposure_snapshots --snapshot_steps $(seq -s, 0 200) --out_dir data/outputs/fig2_exposure_snapshots` | the exposure snapshots of Fig. 2 |

`engine.py` is the replay (`--estimate_stability` adds the local-stability audit: active-face Jacobian and invasion
multipliers at the time-averaged endpoint); `endpoint.py` replays one stored condition to its endpoint for the fragility scripts;
`logspace.py` is the update rule in log space used by `logspace_grids.py`.

## 3. early_warning

| Command | Output | Paper |
|---|---|---|
| `python3 early_warning/logspace.py features --stage features` then `--stage eval` (~15 min) | `data/outputs/replay_logspace_adapted/G7_early_warning/oof_predictions_T1.csv.gz` | behind Fig. 5 |
| `python3 early_warning/closed_form.py` | `data/outputs/early_warning_closed_form` (`v_50` from `early_warning/predicted_level.json`, a Monte Carlo calibration of the concentration threshold) | Fig. 5, B.3 |
| `python3 early_warning/heldout.py heldout_params` / `heldout_support` / `testbeds` | `data/outputs/early_warning_heldout_params`, `_heldout_support`, `_testbeds` | B.3, B.8 |
| `python3 early_warning/context.py holdout` / `shift` | `data/outputs/early_warning_context_holdout`, `data/testbeds/context_shift_early_warning_10seed` | B.3 |

## 4. fragility (eta = 2.25)

`python3 fragility/amplification.py` (24 matched pairs, cross-fitted active-face perturbation) and
`python3 fragility/invasion.py` (inactive-agent invasion at three doses) write `data/results/crossfit_local_amplification_eta2p25`
and `crossfit_invasion_validation_eta2p25`; `tables.py fragility` summarizes them (B.6).

## 5. appendix_figures.py (run from `data/`)

```bash
cd data
python3 -c "import sys; sys.path.insert(0, '../pipeline'); import appendix_figures as p; p._run_threshold_embedded(['--results_dirs','results/fig0_4_full_grid','results/fig5_full_grid','results/fig10_full_grid','results/specialization_wide_dense_figS','--out_dir','results/threshold_robustness'])"
python3 ../pipeline/appendix_figures.py phantom --threshold-out-dir results/threshold_robustness --write-postprocess-tables
python3 ../pipeline/appendix_figures.py figures          # Figs. 7-10, 13 -> results/figures (figures.py appendix does this)
```
`fig10_full_grid` must be included to reproduce the stored 40,400-row threshold tables (B.5).

## Checks that recompute quoted numbers without a model fit

* B.3 leakage AUCs 0.974 / 0.978 / 0.864: seed-wise ROC AUC of `data/results/early_warning_validation_grid/early_warning_predictions.csv.gz`
  (`target == label_unstable`, `early_frac == 0.1`, seed from `source_result_dir`), mean +- s.e. over the 10 held-out seeds.
* B.5 near-critical band 1.4% -> 6.1%, non-borderline failures 83-86%: from `data/results/threshold_robustness/phantom_breakdown_labels.csv.gz`,
  descriptor-positive rows (`phantom_family != none`), with tau in {0.025, 0.05, 0.10}: near-critical = |kappa_map| <= tau,
  contracting = kappa_map < -tau, failure = the rest.
* B.8 tool normalization: over the raw tool logs (`runs_testbed_tool.tar.gz`, `contents[*].correct` per round and agent),
  78% / 5% of rounds have every agent correct / wrong; with the replay's z-scored residual `s_t - s_bar` the agent-mean
  routing score correlates -1.00 with accuracy; the collapse target has accuracy 0.85 against 0.90 for uniform exposure.
  `check.py` runs all three (and prints every other quoted number next to its file).

Parsed raw logs are cached under `.cache/` (`EVOTHEORY_CACHE` overrides).

## Raw logs (Zenodo: https://doi.org/10.5281/zenodo.23112844)

The Zenodo record holds 19 log archives (2.6 GB in all), made by `collect/export_logs.py`:

| Archive | Content | Extract into |
|---|---|---|
| `run_popseed_<s>.checkpoints.tar.gz` (10) | main population, one per seed | `runs/` |
| `run_popseed_<s>_reuse_together_evalaudit_compact.checkpoints.tar.gz` (5) | the same posts re-evaluated by DeepSeek-V3 | `runs/` |
| `runs_testbed_<name>.tar.gz` (4) | heterogeneity, 16d, llama, tool testbeds (5 seeds each) | the package root (gives `runs_testbed_<name>/`) |

Each archive holds `run_popseed_<s>.checkpoints/metadata.json` and one gzipped JSON per timestep; `replay/engine.py` and
everything built on it read the gzipped files directly.
