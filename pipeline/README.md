# pipeline/ — rebuilding `data/` from the raw logs

`figures.py`, `tables.py` and `check.py` read `data/`. This directory holds the code that produced `data/` from the raw
interaction logs, in four stages. Run everything from the package root; every script prints its options with `--help`.

```
collect/         the LLM runs (API calls; already done, logs on Zenodo)
replay/          replay the logs under the platform's exposure update   -> data/results, data/testbeds, data/outputs/replay_*, per_round*
early_warning/   predict instability from the first 10% of a trajectory -> data/outputs/early_warning_*
fragility/       perturbation test at eta = 2.25                          -> data/results/crossfit_*
appendix_figures.py   threshold robustness and the appendix figures      -> data/results/threshold_robustness, data/results/figures
common/          shared helpers
```

## 0. Raw logs

Download the archives of the Zenodo record (https://doi.org/10.5281/zenodo.23112844) and extract the
`run_popseed_*.checkpoints.tar.gz` files into `runs/` and the `runs_testbed_<name>.tar.gz` files at the package root.
Each run directory holds `metadata.json` and one gzipped JSON per timestep; the scripts read them as they are.
The logs were generated with `collect/log_collect.py` (`collect/run.sh`), `collect/run_testbeds.sh` and
`collect/online_validate.py`, and exported with `collect/export_logs.py`.

## 1. Replay (hours; multi-core)

```bash
cd pipeline
bash replay/main_grids.sh                 # main grids (10 seeds)          -> data/results/{fig0_4_full_grid, fig5_full_grid, ...}
bash replay/early_warning_grid.sh         # early-warning grid              -> data/results/early_warning_validation_grid
bash replay/numerical_robustness.sh       # numerical-robustness reruns     -> data/results/stability_threshold_robustness
python3 replay/engine.py --runs_dir ../runs_testbed_<name> --estimate_stability --stability_context_stride 4 --route_zscore ...
                                          # testbeds; the grid of each run is in data/testbeds/<name>/run_config.json
python3 replay/criticality_holdout.py process-run ... && python3 replay/criticality_holdout.py evaluate
python3 replay/per_round_vs_average.py    # per-round vs. time-averaged diversity (Sec. 4.3)
python3 replay/logspace_grids.py replay --grids main_no_support,main_support && python3 replay/logspace_grids.py adapt
python3 replay/per_round.py main && python3 replay/per_round.py testbeds
```

`engine.py` is the replay; `--estimate_stability` adds the local-stability audit (active-face Jacobian and invasion multipliers
at the time-averaged endpoint). `logspace.py` is the same update in log space; `endpoint.py` replays one stored condition for the
fragility test.

## 2. Early warning (minutes to an hour each)

```bash
python3 pipeline/early_warning/logspace.py features --stage features && python3 pipeline/early_warning/logspace.py features --stage eval
python3 pipeline/early_warning/closed_form.py                       # Fig. 5
python3 pipeline/early_warning/heldout.py heldout_params            # also: heldout_support, testbeds
python3 pipeline/early_warning/context.py holdout                   # also: shift
```

## 3. Fragility test

```bash
python3 pipeline/fragility/amplification.py && python3 pipeline/fragility/invasion.py
```

## 4. Threshold robustness and appendix figures (run from `data/`)

```bash
cd data
python3 -c "import sys; sys.path.insert(0, '../pipeline'); import appendix_figures as p; p._run_threshold_embedded(['--results_dirs','results/fig0_4_full_grid','results/fig5_full_grid','results/fig10_full_grid','results/specialization_wide_dense_figS','--out_dir','results/threshold_robustness'])"
python3 ../pipeline/appendix_figures.py phantom --threshold-out-dir results/threshold_robustness --write-postprocess-tables
python3 ../pipeline/appendix_figures.py figures
```

Parsed logs are cached under `.cache/` (`EVOTHEORY_CACHE` overrides the location).
