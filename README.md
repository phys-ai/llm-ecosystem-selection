# How Selection Shapes Diversity in LLM Ecosystems

Code and data for the NeurIPS 2026 paper ([OpenReview](https://openreview.net/forum?id=RM3k5s1pJM)).

## Setup

Python >= 3.10.

```bash
pip install -r requirements.txt
```

## Reproducing the paper

```bash
bash run_all.sh
```

This runs the three analysis scripts against the precomputed outputs in `data/` and takes about two minutes:

- `figures.py` writes the paper figures to `figures/`. Run `python3 figures.py all`, or a single target: `fig2`, `fig3`, `fig4_5_11`, `testbeds`, `online`, `appendix`.
- `tables.py` re-derives the numbers quoted in the paper: `python3 tables.py all`, or one of `unstable_diversity`, `evaluator_shift`, `testbeds`, `tool`, `fragility`.
- `check.py` prints each quoted number next to the file it comes from. Use `--grep` to filter, e.g. `python3 check.py --grep tool`.

`data/` is not included in the repo. To rebuild it, download the raw logs from [Zenodo](https://doi.org/10.5281/zenodo.23112844) into `runs/` and `runs_testbed_*/`, then follow `pipeline/README.md`. The replay grids take a few hours on a multi-core machine; no API access is needed.

## Citation

```bibtex
@inproceedings{okawa2026selection,
  title     = {How Selection Shapes Diversity in {LLM} Ecosystems},
  author    = {Okawa, Maya},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {39},
  year      = {2026},
}
```
