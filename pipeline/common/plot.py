"""Plot styling and phase-label helpers shared by the testbed figures (figures.py testbeds, Figs. 6 and 15).

The rcParams, colours, colormaps and heatmap/legend helpers are the ones of pipeline/appendix_figures.py (whose
figure builders are closures and cannot be imported).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402

RC = {
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
}
plt.rcParams.update(RC)

# ---- constants shared with appendix_figures.py ---------------------------------------
PHASE_NAMES = {
    0: "Diffuse coexistence",
    1: "Concentrated coexistence",
    2: "Monoculture",
    3: "Polarization",
    4: "Specialization",
    5: "Polarized specialization",
    6: "Metastable switching",
    7: "Specialized collapse",
}
PHASE_COLORS = [
    "#999999", "#0072B2", "#D55E00", "#CC79A7", "#009E73", "#56B4E9", "#666666", "#E69F00",
]
CMAPS = {
    "kappa_map": "RdYlBu_r",
    "top1_share": "Purples",
    "n_eff": "YlGn",
    "stable_specialization_indicator": "YlGnBu",
}
HEATMAP_ALPHA = 0.9
TICK_SIZE = 14
AXIS_LABEL_SIZE = 16
PANEL_LABEL_SIZE = 16
LEGEND_FONT_SIZE = 13
COLORBAR_LABEL_SIZE = 14
COLORBAR_TICK_SIZE = 12

ETA_LABEL = r"Selection strength $\eta$"
SUP_LABEL = r"Support strength $\beta_{\mathrm{sup}}$"


# ---- data ------------------------------------------------------------------------------------
def seed_from_source(path: str) -> int:
    m = re.search(r"popseed_(\d+)", str(path))
    if not m:
        raise ValueError(f"cannot infer population seed from {path!r}")
    return int(m.group(1))


def load_phase(result_dir: Path) -> pd.DataFrame:
    """phase_summary.csv already carries the stability columns (kappa_map etc.)."""
    df = pd.read_csv(Path(result_dir) / "phase_summary.csv")
    df["seed"] = df["source_path"].map(seed_from_source)
    df["locally_contracting"] = df["locally_contracting"].astype(str).str.lower().isin(["true", "1"])
    if "kappa_map" in df.columns:
        # same convention as add_stable_phase_flags() in the pipeline
        assert bool((df["locally_contracting"] == (df["kappa_map"] < 0)).all())
    df["grid"] = Path(result_dir).name
    return df


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["coexistence_descriptor"] = out["phase_label"].isin(["diffuse_coexistence", "concentrated_coexistence"])
    out["specialization_descriptor"] = out["specialization_signal"] == 1
    out["polarization_descriptor"] = out["polarization_signal"] == 1
    out["any_diversity_descriptor"] = (
        out["coexistence_descriptor"] | out["specialization_descriptor"] | out["polarization_descriptor"]
    )
    # Fig. 3 (fig3_stable_specialization_heatmap) strict indicator
    out["stable_specialization_indicator"] = (
        (out["specialization_signal"] > 0)
        & (out["kappa_map"] < 0)
        & (out["top1_share"] < 0.60)
        & (out["n_eff"] > 5.0)
        & (out["winner_switch_rate"] > 0.30)
        & (out["i_exp_norm"] > 0.12)
    ).astype(float)
    out["stable_specialization_flag"] = (
        (out["specialization_signal"] == 1)
        & (out["top1_share"] < 0.80)
        & (out["n_eff"] >= 3.0)
        & out["locally_contracting"]
    ).astype(float)
    return out


def modal_code(s: pd.Series):
    counts = s.value_counts(dropna=True)
    if counts.empty:
        return np.nan
    winners = sorted(counts[counts == counts.max()].index.tolist())
    return winners[0]  # pipeline tie rule: smallest phase code


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt((p * (1.0 - p) / n) + (z * z / (4.0 * n * n))) / denom
    return float(max(0.0, center - half)), float(min(1.0, center + half))


def seed_cluster_ratio_ci(num_by_seed: Sequence[float], den_by_seed: Sequence[float], reps: int = 20000, rng_seed: int = 11):
    """Percentile bootstrap over population seeds for a pooled ratio sum(num)/sum(den)."""
    num = np.asarray(num_by_seed, float)
    den = np.asarray(den_by_seed, float)
    rng = np.random.default_rng(rng_seed)
    idx = rng.integers(0, num.size, size=(reps, num.size))
    ratios = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    return float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))


def phantom_table(df: pd.DataFrame, families: Dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, mask in families.items():
        fam = df[mask]
        n = int(len(fam))
        stable = int(fam["locally_contracting"].sum())
        fail = n - stable
        per_seed = fam.groupby("seed")["locally_contracting"].agg(["size", "sum"]).reindex(sorted(df["seed"].unique()), fill_value=0)
        lo, hi = wilson_ci(fail, n)
        if n:
            blo, bhi = seed_cluster_ratio_ci((per_seed["size"] - per_seed["sum"]).to_numpy(), per_seed["size"].to_numpy())
        else:
            blo = bhi = np.nan
        seed_rates = ((per_seed["size"] - per_seed["sum"]) / per_seed["size"].replace(0, np.nan))
        rows.append({
            "family": name,
            "n_descriptor_positive": n,
            "locally_contracting": stable,
            "stability_failures": fail,
            "phantom_share": fail / n if n else np.nan,
            "wilson_lo": lo, "wilson_hi": hi,
            "seed_bootstrap_lo": blo, "seed_bootstrap_hi": bhi,
            "seed_rate_min": float(seed_rates.min()) if n else np.nan,
            "seed_rate_max": float(seed_rates.max()) if n else np.nan,
            "per_seed_fail/n": ";".join(f"{s}:{int(r['size'] - r['sum'])}/{int(r['size'])}" for s, r in per_seed.iterrows()),
        })
    return pd.DataFrame(rows)






# ---- plotting --------------------------------------------------------------------------------
def fmt(v) -> str:
    return f"{float(v):g}"




def set_index_ticks(ax, xs: Sequence[float], ys: Sequence[float], xi: Iterable[int], yi: Iterable[int]):
    xi = [i for i in xi if i < len(xs)]
    yi = [i for i in yi if i < len(ys)]
    ax.set_xticks(xi)
    ax.set_xticklabels([fmt(xs[i]) for i in xi])
    ax.set_yticks(yi)
    ax.set_yticklabels([fmt(ys[i]) for i in yi])
    ax.tick_params(length=0, labelsize=TICK_SIZE)
    ax.set_box_aspect(1)




def draw_cont(ax, pt: pd.DataFrame, cmap: str, vmin=None, vmax=None, clip: Optional[float] = None):
    data = pt.to_numpy(float)
    if clip is not None:
        data = np.clip(data, -clip, clip)
        vmin, vmax = -clip, clip
    return ax.imshow(data, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                     interpolation="nearest", alpha=HEATMAP_ALPHA)


def contour(ax, pt: pd.DataFrame, level: float, **kw) -> bool:
    data = pt.to_numpy(float)
    if data.shape[0] < 2 or data.shape[1] < 2 or not (np.nanmin(data) <= level <= np.nanmax(data)):
        return False
    kw.setdefault("colors", ["black"])
    ax.contour(np.arange(data.shape[1]), np.arange(data.shape[0]), data, levels=[level], **kw)
    return True


def colorbar(fig, im, ax, label: str):
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=COLORBAR_TICK_SIZE, length=2.5, width=0.7)
    cb.outline.set_linewidth(0.7)
    cb.set_label(label, fontsize=COLORBAR_LABEL_SIZE)
    return cb




