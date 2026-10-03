# How Selection Shapes Diversity in LLM Ecosystems — reproduction package

Code and data behind every figure and number of the NeurIPS 2026 paper (OpenReview: https://openreview.net/forum?id=RM3k5s1pJM).

```
llm-ecosystem-selection/
  figures.py             regenerate the figures from data/              python3 figures.py all | fig2 | fig3 | fig4_5_11 | testbeds | online | appendix
  tables.py              re-derive the quoted numbers from data/        python3 tables.py all | unstable_diversity | evaluator_shift | testbeds | tool | fragility
  check.py               print every number quoted in the paper next to the file it comes from   python3 check.py [--grep tool]
  run_all.sh             the three commands above (~2 min)
  data/                  the analysis outputs the three scripts read; not distributed, regenerated from the raw logs by pipeline/
  figures/               the PDFs the scripts write (PAPER_DIR=<LaTeX directory> writes into its figures/ instead)
  pipeline/              how data/ was made from the raw logs: collect -> replay -> early_warning / fragility -> appendix figures (pipeline/README.md)
  runs/, runs_testbed_*/ raw logs of the main population and of the five testbeds (Zenodo: https://doi.org/10.5281/zenodo.23112844)
```

Setup: Python >= 3.10, `pip install -r requirements.txt` (numpy, pandas, scipy, scikit-learn, matplotlib). The raw logs are on
Zenodo (https://doi.org/10.5281/zenodo.23112844); `pipeline/README.md` turns them into `data/` (the replay grids take hours on a
multi-core machine, the early-warning and fragility stages a few hours more; no API access is needed). The three commands above
then take about two minutes.

## Reproduce

```bash
pip install -r requirements.txt && bash run_all.sh   # figures/, data/outputs/, and every quoted number (~2 min, needs data/)
```

To rebuild `data/` from the raw logs (Zenodo), follow `pipeline/README.md`.

## Citation

```bibtex
@inproceedings{okawa2026selection,
  title     = {How Selection Shapes Diversity in {LLM} Ecosystems},
  author    = {Okawa, Maya},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {39},
  year      = {2026},
  url       = {https://openreview.net/forum?id=RM3k5s1pJM}
}
```
