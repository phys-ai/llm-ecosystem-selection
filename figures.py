#!/usr/bin/env python3
"""Figures of the paper, regenerated from the stored analysis outputs (data/).

    python3 figures.py all          every figure below
    python3 figures.py fig2         Fig. 2  selection: phase diagram, kappa_map, exposure time series         -> figures/fig2_cr.pdf
    python3 figures.py fig3         Fig. 3  the five panels of the support composite                         -> fig3_panels.pdf (fig3_main.pdf is assembled from them by hand)
    python3 figures.py fig4_5_11    Fig. 4  unstable diversity, Fig. 5 early-warning ROC, Fig. 11 N_eff     -> fig4_cr.pdf, fig5_cr.pdf, fig_app_neff_avg_vs_inst.pdf
    python3 figures.py testbeds     Fig. 6 and Fig. 15 (--testbed heterogeneity|llama|8d|16d|tool; default: all five)
    python3 figures.py online       Fig. 16 online-generation validation                                     -> fig18_closed_loop_validation.pdf
    python3 figures.py appendix     Figs. 7-10, 12, 13 and the DeepSeek slice (Fig. 14) via pipeline/appendix_figures.py

Every sub-command writes its PDFs into figures/ (or into PAPER_DIR/figures, the LaTeX directory, if PAPER_DIR is set).
"""

from __future__ import annotations

import os
import sys
import glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import glob
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from scipy.ndimage import gaussian_filter
import os
from sklearn.metrics import roc_curve, roc_auc_score
import argparse
import json
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Optional
from matplotlib.lines import Line2D

import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pipeline"))  # shared helpers live in pipeline/common
# where the PDFs go: ./figures, or PAPER_DIR/figures (the LaTeX directory) if PAPER_DIR is set
PAPER_FIGS = (Path(os.environ["PAPER_DIR"]) if os.environ.get("PAPER_DIR") else ROOT) / "figures"
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

from common import plot as C  # noqa: E402


# ====================================================================================================
# fig2
# ====================================================================================================
def main_fig2():
    plt.rcdefaults()
    FIG = PAPER_FIGS; OUT = ROOT / "data/outputs/fig2"; OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6,
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    PHASES = [("diffuse_coexistence", "Coexistence", "#999999"), ("concentrated_coexistence", "Concentration", "#0072B2"),
              ("contextual_specialization", "Specialization", "#009E73"), ("specialized_collapse", "Specialized collapse", "#E69F00"),
              ("monoculture_collapse", "Monoculture", "#D55E00")]
    CODE = {k: i for i, (k, _, _) in enumerate(PHASES)}
    CELLS = [("coexistence", 0.25, "o"), ("collapse", 4.0, "^")]
    fs = sorted(glob.glob(str(ROOT / "data/results/fig0_4_full_grid/run_popseed_*.checkpoints/phase_summary.csv"))); assert len(fs) == 10
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True); d["code"] = d.phase_label.map(CODE); assert d.code.notna().all()
    piv = lambda col, agg: d.groupby(["alpha", "eta"])[col].agg(agg).unstack().sort_index().sort_index(axis=1)
    mode = piv("code", lambda s: s.value_counts().idxmax()); top1 = piv("top1_share", "mean"); kap = piv("kappa_map", "mean")
    xs, ys = list(mode.columns), list(mode.index)
    near = lambda vals, targets: sorted({int(np.argmin(np.abs(np.asarray(vals, float) - t))) for t in targets})
    xi, yi = near(xs, [0.2, 0.5, 1, 2, 4, 8]), near(ys, [0.3, 0.5, 0.75, 1.0])
    xi = [i for i in xi if xs[i] != 0.5]
    cell_ix = {lab: (xs.index(eta), ys.index(1.0)) for lab, eta, _ in CELLS}
    fig = plt.figure(figsize=(5.5, 1.75), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1.25, 1.15], height_ratios=[1, 1], hspace=0.0)
    fig.get_layout_engine().set(h_pad=0.0, hspace=0.0)
    axA, axC = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[:, 1]); axD = [fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2])]
    def ticks(ax):
        ax.set_xticks(xi); ax.set_xticklabels([f"{xs[i]:g}" for i in xi]); ax.set_yticks(yi); ax.set_yticklabels([f"{ys[i]:g}" for i in yi])
        ax.tick_params(length=0); ax.set_xlabel(r"Selection strength $\eta$")
    def marks(ax):
        for lab, eta, mk in CELLS:
            i, j = cell_ix[lab]; ax.plot(i, j, marker=mk, ms=4.5, mfc="white", mec="black", mew=0.8, ls="none", zorder=5, clip_on=False)
    cmap = ListedColormap([c for _, _, c in PHASES]); norm = BoundaryNorm(np.arange(-0.5, len(PHASES) + 0.5, 1), cmap.N)
    axA.imshow(mode.to_numpy(float), origin="lower", aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ticks(axA); axA.set_ylabel("Context fidelity"); marks(axA)
    txt = dict(color="white", fontsize=6.5, fontweight="bold", ha="center", va="center")
    M = mode.to_numpy(float)
    def region_line(code):
        """Centroid column per row of a region and the least-squares line through them (data coordinates)."""
        pts = [(np.mean(np.where(M[r] == code)[0]), r) for r in range(M.shape[0]) if (M[r] == code).any()]
        x, y = np.array(pts).T; a, b = np.polyfit(y, x, 1); return x.mean(), y.mean(), a
    def region_text(ax, code, label, fontsize=6.5, dx=0.0, dy=0.0):
        xc, yc, a = region_line(code)
        p0, p1 = ax.transData.transform([(xc + a, yc + 1), (xc - a, yc - 1)])  # along the band, reading downward to the right
        ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        ax.text(xc + dx, yc + dy, label, rotation=ang, rotation_mode="anchor", **{**txt, "fontsize": fontsize})
    LABELS = [(0, "Coexistence", 7.5, 0.0, 0.0), (1, "Concentration", 7.5, 0.0, 0.0), (4, "Monoculture", 7.5, -1.2, 0.0), (3, "Collapse", 7.5, 0.1, 0.6)]
    im = axC.imshow(np.clip(kap.to_numpy(float), -5, 5), origin="lower", aspect="auto", cmap="RdYlBu_r", vmin=-5, vmax=5, interpolation="nearest", alpha=0.9)
    ticks(axC); axC.set_yticklabels([])
    cb = fig.colorbar(im, ax=axC, fraction=0.05, pad=0.03); cb.set_label(r"Criticality $\kappa_{\mathrm{map}}$", fontsize=6.5); cb.ax.tick_params(labelsize=6, length=2); cb.outline.set_linewidth(0.6)
    snap = pd.read_csv(ROOT / "data/outputs/fig2_exposure_snapshots/exposure_snapshots.csv.gz"); stats = {}
    T0, T1 = 0, 100
    for ax, (lab, eta, mk) in zip(axD, CELLS):
        s = snap[np.isclose(snap.eta, eta)]; P = s.pivot(index="timestep", columns="agent_index", values="exposure").sort_index()
        tr = s.groupby("agent_index")[["trait_stance_valence", "trait_social_framing"]].first().reindex(P.columns)
        st, fr = tr.trait_stance_valence.to_numpy(), tr.trait_social_framing.to_numpy()
        quad = (st >= 0).astype(int) + 2 * (fr >= 0).astype(int)          # quadrant palette of Figure 3(a): blue, orange, green, red
        QCOL = np.array([[76, 120, 168], [245, 133, 24], [84, 162, 75], [228, 87, 86]]) / 255.0
        order = np.lexsort((st, quad)); cols = []
        for q in range(4):
            idx = [i for i in order if quad[i] == q]; n = max(len(idx) - 1, 1)
            for k, i in enumerate(idx):                                   # lighter to darker within a quadrant so agents stay distinguishable
                w = 0.45 * (1 - k / n); cols.append(QCOL[q] * (1 - w) + w)
        P = P.loc[T0:T1]; ax.stackplot(P.index, P.to_numpy()[:, order].T, colors=cols, lw=0, alpha=0.85)
        ax.set_xlim(T0, T1); ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1]); ax.set_xticks([T0, (T0 + T1) // 2, T1])
        ax.set_title((r"$\bigcirc$ " if mk == "o" else r"$\triangle$ ") + lab, fontsize=6.3, pad=2, loc="left")
        stats[lab] = dict(eta=eta, mean_top1=float(P.max(1).mean()), n_eff_avg=float(1 / (P.mean(0) ** 2).sum()), n_eff_inst=float((1 / (P ** 2).sum(1)).mean()),
                          leader_changes=int((P.idxmax(1).diff().fillna(0) != 0).sum()))
    axD[0].set_xticklabels([]); axD[1].set_xlabel("Round"); axD[0].set_ylabel("Exposure share", y=-0.15)
    QCOL = np.array([[76, 120, 168], [245, 133, 24], [84, 162, 75], [228, 87, 86]]) / 255.0
    axD[1].legend(handles=[Patch(facecolor=QCOL[i], label=l) for i, l in ((0, "analyst"), (1, "advocate"), (2, "challenger"), (3, "supporter"))],
                  loc="lower right", frameon=True, framealpha=0.35, edgecolor="none", fontsize=5.5, handlelength=0.9, handleheight=0.8, handletextpad=0.25, labelspacing=0.05, borderpad=0.05, borderaxespad=0.05)
    for ax, t_, dx, dy in ((axA, "A", -0.4, -5), (axC, "B", -0.1, -5), (axD[0], "C", -0.5, -5)):
        ax.annotate(t_, xy=(dx, 1.0), xycoords="axes fraction", xytext=(10 if t_ == "A" else (-2 if t_ == "B" else 5), dy), textcoords="offset points", fontsize=9, fontweight="bold", ha="left", va="bottom", annotation_clip=False)
    fig.canvas.draw()
    for code, lab, fsz, dx_, dy_ in LABELS: region_text(axA, code, lab, fsz, dx_, dy_)
    fig.set_layout_engine("none"); W, H = fig.get_size_inches()
    def shift(ax, dx_in):
        b = ax.get_position(); ax.set_position([b.x0 + dx_in / W, b.y0, b.width, b.height])
    for ax in (axC, cb.ax): shift(ax, 1 / 72)
    for ax in axD: shift(ax, -6 / 72)
    fig.savefig(FIG / "fig2_cr.pdf", dpi=300); fig.savefig(OUT / "fig2_cr_preview.png", dpi=200)
    pd.DataFrame(stats).T.to_csv(OUT / "muller_cells.csv"); print(pd.DataFrame(stats).T.round(2).to_string())


# ====================================================================================================
# fig3
# ====================================================================================================
def main_fig3():
    plt.rcdefaults()
    FIG = PAPER_FIGS
    OUT = ROOT / "data/outputs/fig3"; OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6,
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    PHASES = [("diffuse_coexistence", "Coexistence", "#999999"), ("concentrated_coexistence", "Concentrated coexistence", "#0072B2"),
              ("contextual_specialization", "Specialization", "#009E73"), ("specialized_collapse", "Specialized collapse", "#E69F00"),
              ("monoculture_collapse", "Monoculture", "#D55E00")]
    CODE = {k: i for i, (k, _, _) in enumerate(PHASES)}
    QCOL = [(76 / 255, 120 / 255, 168 / 255), (245 / 255, 133 / 255, 24 / 255), (84 / 255, 162 / 255, 75 / 255), (228 / 255, 87 / 255, 86 / 255)]
    def load(pattern, n_seeds):
        fs = sorted(glob.glob(str(ROOT / pattern))); assert len(fs) == n_seeds, (pattern, len(fs))
        return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    def stable(d):
        return ((d.specialization_signal > 0) & (d.kappa_map < 0) & (d.top1_share < 0.60) & (d.n_eff > 5.0)
                & (d.winner_switch_rate > 0.30) & (d.i_exp_norm > 0.12)).astype(float)
    near = lambda vals, targets: sorted({int(np.argmin(np.abs(np.asarray(vals, float) - t))) for t in targets})
    def ticks(ax, xs, ys, xi, yi):
        ax.set_xticks(xi); ax.set_xticklabels([f"{xs[i]:g}" for i in xi]); ax.set_yticks(yi); ax.set_yticklabels([f"{ys[i]:g}" for i in yi])
        ax.tick_params(length=0); ax.set_xlabel(r"Selection strength $\eta$"); ax.set_ylabel(r"Support strength $\beta_{\mathrm{sup}}$")
    d = load("data/results/fig5_full_grid/run_popseed_*.checkpoints/phase_summary.csv", 10)
    d = d[np.isclose(d.alpha, 1.0)]; assert d.eta.nunique() == 45 and d.beta_sup.nunique() == 41
    d["code"] = d.phase_label.map(CODE); assert d.code.notna().all()
    mode = d.groupby(["beta_sup", "eta"]).code.agg(lambda s: s.value_counts().idxmax()).unstack().sort_index().sort_index(axis=1)
    mode.to_csv(OUT / "panelA_modal_phase_code.csv")
    xs, ys = list(mode.columns), list(mode.index)
    e = load("data/results/specialization_wide_dense_figS/run_popseed_*.checkpoints/phase_summary.csv", 10)
    e = e[(e.beta_sup <= 2.0) & (e.eta > e.eta.min())].copy(); e["stable"] = stable(e)
    rate = e.groupby(["beta_sup", "eta"]).stable.mean().unstack().sort_index().sort_index(axis=1)
    kap = e.groupby(["beta_sup", "eta"]).kappa_map.mean().unstack().sort_index().sort_index(axis=1)
    rate.to_csv(OUT / "panelB_stable_specialization_rate.csv"); kap.to_csv(OUT / "panelC_kappa_map_mean.csv")
    xs2, ys2 = list(rate.columns), list(rate.index)
    fig = plt.figure(figsize=(5.5, 4.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1.3, 1], height_ratios=[1, 1], left=0.09, right=0.97, bottom=0.1, top=0.95, wspace=0.75, hspace=0.75)
    axL, axA, axR = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    axB, axC = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 2])
    axC.set_position([0.52, axC.get_position().y0, 0.26, axC.get_position().height])
    axB.set_position([0.12, axB.get_position().y0, 0.26, axB.get_position().height])
    cmap = ListedColormap([c for _, _, c in PHASES]); norm = BoundaryNorm(np.arange(-0.5, len(PHASES) + 0.5, 1), cmap.N)
    axA.imshow(mode.to_numpy(float), origin="lower", aspect="auto", cmap=cmap, norm=norm, interpolation="nearest", alpha=0.9)
    ticks(axA, xs, ys, near(xs, [0.25, 1.1, 3.125, 8]), near(ys, [0, 0.11, 0.45, 2]))
    present = sorted({int(v) for v in mode.to_numpy().ravel()})
    axA.legend(handles=[Patch(facecolor=PHASES[i][2], alpha=0.9, label=PHASES[i][1]) for i in present], loc="upper center", bbox_to_anchor=(0.5, -0.3),
               frameon=False, ncol=2, handlelength=0.9, handletextpad=0.35, columnspacing=0.8, fontsize=5.5)
    def trait_map(ax, path, title):
        t = pd.read_csv(path)
        x, y, w = t.trait_stance_valence.to_numpy(), t.trait_social_framing.to_numpy(), t.exposure.to_numpy()
        H, xe, ye = np.histogram2d(x, y, bins=32, range=[[-1, 1], [-1, 1]], weights=w)
        ax.imshow(gaussian_filter(H.T, 1.2), origin="lower", extent=[-1, 1, -1, 1], cmap="Greys", alpha=0.9, aspect="auto"); ax.set_box_aspect(1)
        col = [QCOL[int(a >= 0) + 2 * int(b >= 0)] for a, b in zip(x, y)]
        ax.scatter(x, y, s=4 + 400 * w / w.max(), c=col, alpha=0.6, lw=0)
        ax.axhline(0, color="0.5", lw=0.5); ax.axvline(0, color="0.5", lw=0.5)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("Stance  (critical → supportive)"); ax.set_ylabel("Framing  (structural → interpersonal)")
        ax.set_title(title, fontsize=7, pad=2)
        for s in ax.spines.values():
            s.set_visible(True); s.set_linewidth(0.6)
        return t
    tl = trait_map(axL, ROOT / "data/results/trait_exposure_examples/stable_spec/endpoint_exposures.csv.gz", "Specialization")
    tr = trait_map(axR, ROOT / "data/results/trait_exposure_examples/spec_collapse/endpoint_exposures.csv.gz", "Specialized collapse")
    tl.to_csv(OUT / "trait_map_specialization.csv", index=False); tr.to_csv(OUT / "trait_map_specialized_collapse.csv", index=False)
    im = axB.imshow(rate.to_numpy(float), origin="lower", aspect="auto", cmap="YlGnBu", vmin=0, vmax=1, interpolation="nearest", alpha=0.9)
    k = kap.to_numpy(float); X, Y = np.arange(k.shape[1]), np.arange(k.shape[0])
    axB.contour(X, Y, k, levels=[0.0], colors=["black"], linewidths=0.9, linestyles="--")
    ticks(axB, xs2, ys2, near(xs2, [0.125, 0.45, 0.95, 2]), near(ys2, [0.3, 0.75, 1.15, 2]))
    cb = fig.colorbar(im, ax=axB, fraction=0.046, pad=0.04); cb.set_label("Stable-specialization rate", fontsize=6.5); cb.ax.tick_params(labelsize=6, length=2)
    vmax = float(np.nanpercentile(np.abs(k), 98))
    im = axC.imshow(k, origin="lower", aspect="auto", cmap="RdYlBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    axC.contour(X, Y, k, levels=[0.0], colors=["black"], linewidths=0.9, linestyles="--")
    axC.contour(X, Y, rate.to_numpy(float), levels=[0.5], colors=["black"], linewidths=0.9)
    ticks(axC, xs2, ys2, near(xs2, [0.125, 0.45, 0.95, 2]), near(ys2, [0.3, 0.75, 1.15, 2])); axC.set_ylabel("")
    cb = fig.colorbar(im, ax=axC, fraction=0.046, pad=0.04); cb.set_label(r"Criticality $\kappa_{\mathrm{map}}$", fontsize=6.5); cb.ax.tick_params(labelsize=6, length=2)
    axC.legend(handles=[plt.Line2D([0], [0], color="black", lw=0.9, ls="--", label=r"$\kappa_{\mathrm{map}}=0$"),
                        plt.Line2D([0], [0], color="black", lw=0.9, label="Stable boundary")], loc="lower left", frameon=True, framealpha=0.8, edgecolor="none",
               handlelength=1.4, fontsize=5.5)
    for ax, t_ in ((axA, "A"), (axB, "B"), (axC, "C")):
        ax.annotate(t_, xy=(-0.18, 1.0), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points", fontsize=9, fontweight="bold", ha="left", va="bottom", annotation_clip=False)
    fig.savefig(FIG / "fig3_panels.pdf", dpi=300); fig.savefig(OUT / "fig3_panels_preview.png", dpi=200)
    band = e[e.stable == 1]
    print("stable cells (figS)", int(e.stable.sum()), "eta", band.eta.min(), band.eta.max(), "beta_sup", band.beta_sup.min(), band.beta_sup.max())
    print("wrote", FIG / "fig3_panels.pdf")


# ====================================================================================================
# fig4_5_11
# ====================================================================================================
def main_fig4_5_11():
    plt.rcdefaults()
    R = ROOT / "data/outputs"; FIG = PAPER_FIGS
    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6, "axes.spines.top": False, "axes.spines.right": False})
    DARK, GOLD, ORANGE, BLUE = "#2f4858", "#e8c468", "#d9711c", "#2b6ca3"
    cnt = pd.read_csv(R / "unstable_diversity_rates/fig4_panelA_counts.csv")
    ep = pd.read_csv(R / "per_round_vs_average/endpoint_instantaneous.csv")
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(3.3, 1.8), constrained_layout=True, gridspec_kw=dict(width_ratios=[1, 1.3]))
    x = np.arange(3); axA.bar(x, cnt.locally_contracting, color=DARK, label="locally stable"); axA.bar(x, cnt.not_locally_contracting, bottom=cnt.locally_contracting, color=GOLD, label="not locally stable")
    for xi, lc, nlc in zip(x, cnt.locally_contracting, cnt.not_locally_contracting): axA.text(xi, lc + nlc / 2, f"{100 * nlc / (lc + nlc):.0f}%", ha="center", va="center", fontsize=6, fontweight="bold")
    axA.set_xticks(x); axA.set_xticklabels(cnt.family, rotation=20, ha="right"); axA.set_ylabel("Endpoint count"); axA.set_ylim(0, 33000); axA.tick_params(axis="x", pad=2.5); axA.set_yticks([0, 10000, 20000, 30000]); axA.set_yticklabels(["0", "10k", "20k", "30k"])
    axA.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), bbox_transform=matplotlib.transforms.offset_copy(axA.transAxes, fig=fig, x=-7, y=0, units="points"), frameon=False, handlelength=1.0, borderaxespad=0.0, labelspacing=0.2, fontsize=6.5)
    sl = ep[(ep.grid == "support") & np.isclose(ep.beta_sup, 1.0)]
    g = sl.groupby("eta").agg(inst=("top1_inst", "mean"), inst_se=("top1_inst", "sem"), avg=("top1_avg", "mean"), avg_se=("top1_avg", "sem"), lc=("locally_contracting", "mean")).reset_index()
    for col, se, c, mk, lab in (("inst", "inst_se", "black", ".", "per round"), ("avg", "avg_se", ORANGE, "s", "time average")):
        axB.plot(g.eta, g[col], color=c, marker=mk, ms=3.5 if mk == "." else 2.2, lw=0.9, label=lab); axB.fill_between(g.eta, g[col] - 1.96 * g[se], g[col] + 1.96 * g[se], color=c, alpha=0.18, lw=0)
    eta_c = g.eta[g.lc >= 0.5].max(); axB.axvline(eta_c, color="0.5", lw=0.6, ls=":"); axB.text(eta_c * 0.93, 1.0, "locally\nstable", ha="right", va="top", fontsize=5.5, color="0.35"); axB.text(eta_c * 1.08, 1.0, "unstable\ndiversity", ha="left", va="top", fontsize=5.5, color="0.35")
    axB.set_xscale("log"); axB.set_xlabel(r"Selection strength $\eta$"); axB.set_ylabel("Top-1 share"); axB.set_ylim(0, 1.05); axB.set_xticks([0.25, 0.5, 1, 2, 4, 8]); axB.set_xticklabels(["0.25", "0.5", "1", "2", "4", "8"])
    axB.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False, handlelength=1.4, borderaxespad=0.0, labelspacing=0.2, fontsize=6.5)
    print("eta_c", eta_c, g[["eta", "inst", "avg", "lc"]].iloc[[0, 13, 20, 28, 40]].round(3).to_string())
    for ax, t_ in ((axA, "A"), (axB, "B")): ax.annotate(t_, xy=(-0.3 if ax is axA else -0.2, 1.0), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points", fontsize=9, fontweight="bold", ha="left", va="bottom", annotation_clip=False)
    fig.savefig(FIG / "fig4_cr.pdf", dpi=300); fig.savefig(FIG / "fig4_cr_preview.png", dpi=200); plt.close(fig)
    d = pd.read_csv(R / "early_warning_closed_form/endpoint_early_prediction.csv.gz"); sub = d[d.diverse_at_10pct == 1]
    series = [("b_theory_var_t20", "closed form", "#c0392b", "-"), ("seed_x_eta__full_early_warning", "learned, all features", "#1b9e77", "-"),
              ("seed_x_eta__early_descriptor", "learned, descriptors", ORANGE, "--"), ("seed_x_eta__param_rbf_svm", "parameters, SVM", BLUE, ":")]
    fig, ax = plt.subplots(figsize=(2.0, 108 / 72), constrained_layout=True); aucs = {}
    for col, lab, c, ls in series:
        fpr, tpr, _ = roc_curve(sub.y, sub[col]); a = roc_auc_score(sub.y, sub[col]); aucs[lab] = round(float(a), 3); ax.plot(fpr, tpr, color=c, ls=ls, lw=1.1, label=lab)
    ax.plot([0, 1], [0, 1], color="0.5", lw=0.6, ls=":"); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
    ax.legend(loc="lower right", frameon=False, handlelength=1.3, labelspacing=0.2, fontsize=5.3)
    fig.savefig(FIG / "fig5_cr.pdf", dpi=300); fig.savefig(FIG / "fig5_cr_preview.png", dpi=200); plt.close(fig)
    print("fig5 n", len(sub), "positive rate %.3f" % sub.y.mean(), aucs, "all-endpoint AUC closed form %.3f" % roc_auc_score(d.y, d.b_theory_var_t20))
    ep2 = pd.read_csv(R / "per_round_vs_average/endpoint_instantaneous.csv"); div = ep2[ep2.any_diversity == 1]
    with plt.rc_context({"font.size": 7.7, "axes.labelsize": 7.7, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7}):
        fig, axes = plt.subplots(1, 2, figsize=(4.4, 2.0), sharex=True, sharey=True, constrained_layout=True)
        for ax, grid, t_ in zip(axes, ["no_support", "support"], ["A", "B"]):
            sub = div[div.grid == grid]
            for status, color in [("phantom", "#D55E00"), ("locally contracting", "#0072B2")]:
                q = sub[sub.status == status]; ax.scatter(q.N_eff_avg, q.N_eff_inst, s=1.5, alpha=0.25, color=color, linewidths=0, rasterized=True, label=("unstable diversity" if status == "phantom" else status))
            ax.plot([1, 96], [1, 96], color="black", lw=0.8); ax.axhline(3.0, color="black", ls=":", lw=0.8); ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlabel(r"$N_{\mathrm{eff}}$ of time-averaged endpoint")
            leg = ax.legend(frameon=False, loc="upper left", markerscale=4, handletextpad=0.3)
            for h in leg.legend_handles: h.set_alpha(1.0)
            ax.annotate(t_, xy=(0, 1), xycoords="axes fraction", xytext=(-22 if t_ == "A" else -10, 2), textcoords="offset points", fontsize=9, fontweight="bold", ha="left", va="bottom", annotation_clip=False)
        axes[0].set_ylabel(r"Mean per-round $N_{\mathrm{eff}}$")
        fig.savefig(FIG / "fig_app_neff_avg_vs_inst.pdf", dpi=300); fig.savefig(FIG / "fig_app_neff_avg_vs_inst_preview.png", dpi=200); plt.close(fig)
    print("done")


# ====================================================================================================
# testbeds
# ====================================================================================================
def main_testbeds():
    plt.rcdefaults()
    plt.rcParams.update(C.RC)
    FS = 15
    COL_TOP1, COL_KAPPA = "#1b1b1b", "#d95f02"
    _P = plt.get_cmap("Paired").colors
    CURVES = [  # same four curves / colours as main Fig. 5
        ("parameter_only", "param: linear", _P[0], (0, (4, 1.5)), 2.4),
        ("param_rbf_svm", "param: SVM", _P[1], (0, (4, 1.5)), 2.4),
        ("early_descriptor", "descriptor", _P[7], "-", 2.8),
        ("fluctuation_only", "fluctuation", _P[3], "-", 2.2),
    ]
    FIG_PDF = {"heterogeneity": "fig6_heterogeneity.pdf", "llama": "fig_app_testbed_llama.pdf", "8d": "fig_app_testbed_8d.pdf",
               "16d": "fig_app_testbed_16d.pdf", "tool": "fig6_bfcl.pdf"}
    OUT_DIR = {"heterogeneity": "fig6_heterogeneity", "llama": "fig15_llama", "8d": "fig15_8d", "16d": "fig15_16d", "tool": "fig15_tool"}
    # the dense support grids of data/testbeds used by the figures (the coarser verification_<testbed>_support grids feed tables.py)
    SUPPORT_DIR = {"heterogeneity": "cr_heterogeneity_support_fine", "llama": "cr_llama_support_dense", "8d": "cr_8d_support_dense",
                   "16d": "cr_16d_support_dense", "tool": "cr_tool_support_dense"}
    def social(args):
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)

        sup = C.add_flags(C.load_phase(Path(args.support_dir)))
        ns = C.load_phase(Path(args.main_dir))
        for d in (sup, ns):
            assert sorted(d["seed"].unique()) == [11, 22, 33, 44, 55]

        # ---- (a)
        cells = sup.groupby(["beta_sup", "eta"]).agg(
            n_seeds=("seed", "nunique"), stable_specialization_rate=("stable_specialization_indicator", "mean"),
            kappa_map_mean=("kappa_map", "mean")).reset_index()
        assert (cells["n_seeds"] == 5).all()
        n_stable = int(sup["stable_specialization_indicator"].sum())
        cells.to_csv(out / "support_grid_cells.csv", index=False)
        pt_ss = cells.pivot(index="beta_sup", columns="eta", values="stable_specialization_rate").sort_index().sort_index(axis=1)
        pt_k = cells.pivot(index="beta_sup", columns="eta", values="kappa_map_mean").sort_index().sort_index(axis=1)

        # ---- (b)
        fams = {"Coexistence": sup["coexistence_descriptor"], "Specialization": sup["specialization_descriptor"],
                "Any diversity": sup["any_diversity_descriptor"]}
        ph = C.phantom_table(sup, fams)
        ph.to_csv(out / "phantom_rate.csv", index=False)
        t = ph.set_index("family").loc[list(fams)]

        # ---- (c)
        per_seed = ns.groupby(["seed", "eta"])[["top1_share", "kappa_map"]].mean().reset_index()
        sweep = per_seed.groupby("eta")[["top1_share", "kappa_map"]].agg(["mean", "sem"])
        sweep.columns = ["_".join(c) for c in sweep.columns]
        sweep.to_csv(out / "no_support_eta_sweep.csv")
        lo, hi = sweep.index.min(), sweep.index.max()

        # ---- figure
        plt.rcParams.update({"font.size": FS, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
                             "axes.spines.right": False, "axes.linewidth": 0.8})
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), constrained_layout=True,
                                 gridspec_kw={"width_ratios": [1.2, 1.0, 1.15]})

        ax = axes[0]
        im = C.draw_cont(ax, pt_ss, C.CMAPS["stable_specialization_indicator"], 0, 1)
        C.contour(ax, pt_k, 0.0, linewidths=1.5, linestyles="--")
        xs, ys = list(pt_ss.columns), list(pt_ss.index)
        nearest = lambda vals, targets: sorted({int(np.argmin(np.abs(np.asarray(vals, float) - t))) for t in targets})
        C.set_index_ticks(ax, xs, ys, nearest(xs, [0.25, 1, 2, 4, 8]), nearest(ys, [0, 0.1, 0.3, 0.5, 1, 2]))
        ax.set_xlabel(C.ETA_LABEL, fontsize=FS)
        ax.set_ylabel(C.SUP_LABEL, fontsize=FS)
        ax.tick_params(labelsize=FS - 2)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
        cb.set_label("Stable-specialization rate", fontsize=FS)
        cb.ax.tick_params(labelsize=FS - 2)

        ax = axes[1]
        x = np.arange(len(t))
        ax.bar(x, t["locally_contracting"], width=0.68, color="#264653", label="locally stable")
        ax.bar(x, t["stability_failures"], width=0.68, bottom=t["locally_contracting"], color="#E9C46A",
               label="not locally stable")
        for xi, (_, row) in zip(x, t.iterrows()):
            ax.text(xi, row["locally_contracting"] + row["stability_failures"] / 2, f"{100 * row['phantom_share']:.0f}%",
                    ha="center", va="center", fontsize=FS, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["Coexist.", "Special.", "Any\ndiversity"], fontsize=FS - 2)
        ax.set_ylabel("Endpoint count", fontsize=FS)
        ax.set_ylim(0, t["n_descriptor_positive"].max() * 1.28)
        ax.tick_params(axis="y", labelsize=FS - 2)
        ax.legend(loc="upper left", frameon=False, fontsize=FS - 2.5, handlelength=1.2,
                  handletextpad=0.4, labelspacing=0.25, borderaxespad=0.2)

        ax = axes[2]
        e = sweep.index.to_numpy()
        ax.fill_between(e, sweep["top1_share_mean"] - sweep["top1_share_sem"], sweep["top1_share_mean"] + sweep["top1_share_sem"],
                        color=COL_TOP1, alpha=0.15, linewidth=0)
        h1, = ax.plot(e, sweep["top1_share_mean"], ls="-", marker=".", ms=9, lw=2.2, color=COL_TOP1, label="top-1 share (left)")
        ax.set_xscale("log")
        ax.set_xticks([0.2, 0.5, 1, 2, 4, 8])
        ax.set_xticklabels(["0.2", "0.5", "1", "2", "4", "8"])
        ax.minorticks_off()
        ax.set_ylim(0, 1.03)
        ax.set_xlabel(C.ETA_LABEL, fontsize=FS)
        ax.set_ylabel("Top-1 share", fontsize=FS)
        ax.tick_params(labelsize=FS - 2)
        ax2 = ax.twinx()
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color(COL_KAPPA)
        ax2.fill_between(e, sweep["kappa_map_mean"] - sweep["kappa_map_sem"], sweep["kappa_map_mean"] + sweep["kappa_map_sem"],
                         color=COL_KAPPA, alpha=0.18, linewidth=0)
        h2, = ax2.plot(e, sweep["kappa_map_mean"], "-s", ms=6.5, lw=2.2, color=COL_KAPPA, label=r"$\kappa_{\mathrm{map}}$ (right)")
        ax2.axhline(0, color=COL_KAPPA, lw=0.9, ls="--")
        ax2.set_ylim(-0.4, 10.3)
        ax2.set_ylabel(r"Criticality $\kappa_{\mathrm{map}}$", fontsize=FS, color=COL_KAPPA)
        ax2.tick_params(axis="y", labelsize=FS - 2, colors=COL_KAPPA)
        for a_, lab, dx in zip(axes, ["A", "B", "C"], [-70, -78, -62]):
            a_.annotate(lab, xy=(0, 1), xycoords="axes fraction", xytext=(dx, -6), textcoords="offset points",
                        fontsize=FS + 1, fontweight="bold", ha="left", va="bottom")

        Path(args.fig_pdf).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig_pdf)
        fig.savefig(out / (Path(args.fig_pdf).stem + "_preview.png"), dpi=110)
        plt.close(fig)

        band = sup[sup["stable_specialization_indicator"] == 1]
        summary = {
            "n_stable_specialization": n_stable, "stable_band_eta": [float(band.eta.min()), float(band.eta.max())],
            "stable_band_beta_sup": [float(band.beta_sup.min()), float(band.beta_sup.max())],
            "stable_band_n_seeds": int(band.seed.nunique()),
            "phantom": {k: {"n": int(v["stability_failures"]), "N": int(v["n_descriptor_positive"]),
                            "pct": round(100 * float(v["phantom_share"]), 1)} for k, v in t.iterrows()},
            "eta_sweep": {"eta_min": float(lo), "eta_max": float(hi),
                          "top1": [round(float(sweep.loc[lo, "top1_share_mean"]), 3), round(float(sweep.loc[hi, "top1_share_mean"]), 3)],
                          "kappa": [round(float(sweep.loc[lo, "kappa_map_mean"]), 3), round(float(sweep.loc[hi, "kappa_map_mean"]), 3)]},
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    def tool(args):
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)

        # ---------------- data: support grid ----------------
        sup = C.add_flags(C.load_phase(Path(args.support_dir)))
        assert sorted(sup["seed"].unique()) == [11, 22, 33, 44, 55]
        cells = sup.groupby(["beta_sup", "eta"]).agg(
            n_seeds=("seed", "nunique"), modal_phase_code=("phase_code", C.modal_code),
            kappa_map_mean=("kappa_map", "mean"), top1_share_mean=("top1_share", "mean"),
            locally_contracting_rate=("locally_contracting", "mean")).reset_index()
        assert (cells["n_seeds"] == 5).all()
        cells["modal_phase_label"] = cells["modal_phase_code"].map(C.PHASE_NAMES)
        cells.to_csv(out / "support_grid_cells.csv", index=False)

        fams = {
            "Diffuse\ncoexist.": sup["phase_label"] == "diffuse_coexistence",
            "Concentr.\ncoexist.": sup["phase_label"] == "concentrated_coexistence",
            "Any\ndiversity": sup["any_diversity_descriptor"],
        }
        ph = C.phantom_table(sup, fams)
        ph.to_csv(out / "phantom_rate.csv", index=False)
        any_row = ph[ph["family"] == "Any\ndiversity"].iloc[0]
        # the support-grid "any diversity" count is the robustness-table / Fig. C1 value
        assert int(sup["specialization_descriptor"].sum()) == 0

        # ---------------- data: early warning ----------------
        oof = pd.read_csv(args.oof)
        oof = oof[oof["setting"] == "tool"]
        both = oof.groupby("eta")["y_true"].nunique()
        assert set(both[both == 2].index) == set(e for e in both.index if e <= args.roc_eta_max), both.to_dict()
        auc_all = {m: roc_auc_score(oof["y_true"], oof[f"{args.split}__{m}"]) for m, *_ in CURVES}
        oof = oof[oof["eta"] <= args.roc_eta_max]
        summ = pd.read_csv(args.loeo_summary)
        y = oof["y_true"].to_numpy(int)
        rows = []
        for m, *_ in CURVES:
            a = roc_auc_score(y, oof[f"{args.split}__{m}"].to_numpy(float))
            rows.append({"model": m, "split": args.split, "n": len(y), "n_positive": int(y.sum()), "pooled_roc_auc": a})
        auc = pd.DataFrame(rows)
        auc.to_csv(out / "early_warning_auc.csv", index=False)
        # consistency with the early_warning_testbeds summary
        try:
            ref = summ[(summ["setting"] == "tool") & (summ["split"] == args.split)].set_index("model")["pooled_roc_auc"]
            for r in rows:
                assert abs(ref[r["model"]] - auc_all[r["model"]]) < 1e-5, (r, ref[r["model"]])
        except KeyError:
            print("note: summary_auc.csv has a different layout; AUCs recomputed from the OOF predictions only")

        # ---------------- figure ----------------
        plt.rcParams.update({"font.size": FS, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
                             "axes.spines.right": False, "axes.linewidth": 0.8})
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), constrained_layout=True,
                                 gridspec_kw={"width_ratios": [1.25, 1.0, 0.95]})

        ax = axes[0]
        # top-1 share vs support strength, one line per selection strength (mean +/- s.e. over the 5 seeds)
        etas = [e for e in sorted(sup["eta"].unique()) if e <= args.line_eta_max]
        betas = sorted(sup["beta_sup"].unique())
        xpos = {b: k for k, b in enumerate(betas)}  # support values are unevenly spaced: equal spacing in grid index
        cmap = plt.get_cmap("viridis")
        line_rows = []
        for k, e in enumerate(etas):
            d = sup[sup["eta"] == e].groupby("beta_sup")["top1_share"].agg(["mean", "sem", "count"]).loc[betas]
            assert (d["count"] == 5).all()
            col = cmap(k / (len(etas) - 1) * 0.92)
            xx = [xpos[b] for b in betas]
            ax.fill_between(xx, d["mean"] - d["sem"], d["mean"] + d["sem"], color=col, alpha=0.13, linewidth=0)
            ax.plot(xx, d["mean"], "-o", ms=4.5, lw=2.0, color=col, label=rf"$\eta={e:g}$")
            line_rows.append(d.assign(eta=e).reset_index())
        pd.concat(line_rows).to_csv(out / "top1_vs_support_by_eta.csv", index=False)
        ax.axhline(0.8, color="0.45", ls=":", lw=1.2)
        ax.set_xticks(range(0, len(betas), 1))
        shown = (0, 0.2, 0.4, 0.65, 1.0, 2.0) if len(betas) > 14 else tuple(b for k, b in enumerate(betas) if k % 2 == 0 or b in (1.0, 2.0))
        ax.set_xticklabels([f"{b:g}" if b in shown else "" for b in betas], fontsize=FS - 3)
        ax.set_ylim(0, 1.03)
        ax.set_xlabel(C.SUP_LABEL, fontsize=FS)
        ax.set_ylabel("Top-1 share", fontsize=FS)
        ax.tick_params(axis="y", labelsize=FS - 2)
        ax.legend(loc="lower left", frameon=False, fontsize=FS - 3.5,
                  handlelength=1.3, handletextpad=0.4, labelspacing=0.2, borderaxespad=0.2)

        ax = axes[1]
        t = ph.set_index("family").loc[list(fams)]
        x = np.arange(len(t))
        ax.bar(x, t["locally_contracting"], width=0.68, color="#264653", label="locally stable")
        ax.bar(x, t["stability_failures"], width=0.68, bottom=t["locally_contracting"], color="#E9C46A",
               label="not locally stable")
        for xi, (_, r) in zip(x, t.iterrows()):
            ax.text(xi, r["locally_contracting"] + r["stability_failures"] / 2, f"{100 * r['phantom_share']:.0f}%",
                    ha="center", va="center", fontsize=FS, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(list(t.index), fontsize=FS - 2)
        ax.set_ylabel("Endpoint count", fontsize=FS)
        ax.set_ylim(0, t["n_descriptor_positive"].max() * 1.28)
        ax.tick_params(axis="y", labelsize=FS - 2)
        ax.legend(loc="upper left", frameon=False, fontsize=FS - 3.5, handlelength=1.2,
                  handletextpad=0.4, labelspacing=0.25, borderaxespad=0.2)

        ax = axes[2]
        for m, lab, col, ls, lw in CURVES:
            fpr, tpr, _ = roc_curve(y, oof[f"{args.split}__{m}"].to_numpy(float))
            a = float(auc.set_index("model").loc[m, "pooled_roc_auc"])
            ax.plot(fpr, tpr, color=col, linestyle=ls, lw=lw, label=lab, zorder=3 if ls == "-" else 2)
        ax.plot([0, 1], [0, 1], color="0.65", linestyle="--", linewidth=1.0, zorder=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.03)
        ax.set_box_aspect(1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])
        ax.set_xlabel("False positive rate", fontsize=FS)
        ax.set_ylabel("True positive rate", fontsize=FS)
        ax.tick_params(labelsize=FS - 2)
        ax.legend(loc="lower right", frameon=False, fontsize=FS - 2.5, handlelength=1.9, handletextpad=0.5,
                  labelspacing=0.25, borderaxespad=0.0)

        for a_, lab, dx in zip(axes, ["A", "B", "C"], [-70, -78, -78]):
            a_.annotate(lab, xy=(0, 1), xycoords="axes fraction", xytext=(dx, -6), textcoords="offset points",
                        fontsize=FS + 1, fontweight="bold", ha="left", va="bottom")

        Path(args.fig_pdf).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.fig_pdf)
        fig.savefig(out / (Path(args.fig_pdf).stem + "_preview.png"), dpi=110)
        plt.close(fig)

        summary = {"phantom_any_diversity": {"n": int(any_row["stability_failures"]), "N": int(any_row["n_descriptor_positive"])},
                   "early_warning_pooled_auc": {r["model"]: round(r["pooled_roc_auc"], 4) for r in rows},
                   "n_early_warning": int(len(y)), "positive_rate": float(y.mean()), "n_negative": int((1 - y).sum()),
                   "roc_eta_max": args.roc_eta_max, "line_eta_max": args.line_eta_max,
                   "early_warning_pooled_auc_all_eta": {k: round(float(v), 4) for k, v in auc_all.items()},
                   "collapse_rate_by_beta_sup": sup.groupby("beta_sup")["collapse_signal"].mean().round(4).to_dict(),
                   "n_locally_contracting_by_label": sup.groupby("phase_label")["locally_contracting"].sum().astype(int).to_dict()}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        print(ph.to_string())
    def main():
        ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        ap.add_argument("--testbed", default="all", choices=["all", *FIG_PDF])
        ap.add_argument("--support_dir", default=None, help="default data/testbeds/<dense support grid of the testbed>")
        ap.add_argument("--main_dir", default=None, help="default data/testbeds/verification_<testbed>_main")
        ap.add_argument("--out_dir", default=None, help="default data/outputs/<fig6_heterogeneity | fig15_<testbed>>")
        ap.add_argument("--fig_pdf", default=None, help="default: <figure of the testbed> in the paper's figures/")
        ap.add_argument("--oof", default="data/outputs/early_warning_testbeds/oof_predictions.csv.gz")
        ap.add_argument("--loeo_summary", default="data/outputs/early_warning_testbeds/summary_auc.csv")
        ap.add_argument("--split", default="seed_x_eta")
        ap.add_argument("--line_eta_max", type=float, default=2.0,
                        help="tool (a): plot only eta <= this value; for eta*beta_sup > 2 the support term over-corrects and "
                             "seed-level endpoints are bimodal, so means are not meaningful.")
        ap.add_argument("--roc_eta_max", type=float, default=1.2,
                        help="tool (c): restrict the pooled ROC to held-out eta values where both classes occur.")
        args = ap.parse_args()
        for tb in (list(FIG_PDF) if args.testbed == "all" else [args.testbed]):
            a = argparse.Namespace(**vars(args)); a.testbed = tb
            a.support_dir = args.support_dir or f"data/testbeds/{SUPPORT_DIR[tb]}"
            a.main_dir = args.main_dir or f"data/testbeds/verification_{tb}_main"
            a.out_dir = args.out_dir or f"data/outputs/{OUT_DIR[tb]}"
            a.fig_pdf = args.fig_pdf or str(PAPER_FIGS / FIG_PDF[tb])
            (tool if tb == "tool" else social)(a)
    main()


# ====================================================================================================
# online
# ====================================================================================================
def _main_online():
    plt.rcdefaults()
    plt.rcParams.update({"font.size": 18})
    PREFERRED_CONDITION_ORDER = [
        "nosup_eta0.3_alpha0.3",
        "nosup_eta3.0_alpha1.0",
        "sup_eta1.0_beta1.5",
        "sup_eta1.5_beta1.5",
    ]
    REPRESENTATIVE_NUMBERS = {
        "nosup_eta0.3_alpha0.3": "1",
        "nosup_eta3.0_alpha1.0": "2",
        "sup_eta1.0_beta1.5": "3",
        "sup_eta1.5_beta1.5": "4",
    }
    PRETTY_NAMES = {
        "nosup_eta0.3_alpha0.3": "1. Low $\\eta$, low $\\alpha$\n(no support)",
        "nosup_eta3.0_alpha1.0": "2. High $\\eta$, high $\\alpha$\n(no support)",
        "sup_eta1.0_beta1.5": "3. Moderate $\\eta$, high support",
        "sup_eta1.5_beta1.5": "4. High $\\eta$, high support",
    }
    PANEL_C_MAIN_LABEL_OFFSETS = {
        "nosup_eta0.3_alpha0.3": (8, 8, "left"),
        "nosup_eta3.0_alpha1.0": (-8, -18, "right"),
    }
    PANEL_C_SUPPORT_LABEL_OFFSETS = {
        "sup_eta1.0_beta1.5": (-8, 8, "right"),
        "sup_eta1.5_beta1.5": (-8, -28, "right"),
    }
    FAMILY_COLORS = {
        "no_support": "#1f77b4",
        "weak_support": "#ff7f0e",
        "strong_support": "#2ca02c",
    }
    FAMILY_LABELS = {
        "no_support": r"No support ($\beta_{\mathrm{sup}}=0$)",
        "weak_support": r"Weak support ($\beta_{\mathrm{sup}}=0.5$)",
        "strong_support": r"Strong support ($\beta_{\mathrm{sup}}=1.5$)",
    }
    def support_family_legend_handles(marker: str = "o", markersize: int = 9) -> list[Line2D]:
        return [
            Line2D(
                [0], [0],
                marker=marker,
                linestyle="",
                markersize=markersize,
                markerfacecolor=FAMILY_COLORS[key],
                markeredgecolor="black",
                label=FAMILY_LABELS[key],
            )
            for key in ["no_support", "weak_support", "strong_support"]
        ]
    PHASE_MARKERS = {
        "diffuse_coexistence": "o",
        "concentrated_coexistence": "s",
        "contextual_specialization": "^",
        "polarized_specialization": "^",
        "monoculture_collapse": "X",
        "specialized_collapse": "D",
        "polarization": "P",
    }
    PHASE_COLORS = [
        "#d9d9d9",
        "#08519c",
        "#9ecae1",
        "#756bb1",
        "#74c476",
        "#31a354",
        "#bdbdbd",
        "#fb6a4a",
    ]
    def build_arg_parser() -> argparse.ArgumentParser:
        p = argparse.ArgumentParser(description="Generate a closed-loop validation figure.")
        p.add_argument("--summary_csv", default="data/results/closed_loop_validation_grid/closed_loop_summary_rebuilt_5seeds_with_replay.csv") #"data/results/closed_loop_validation_grid/closed_loop_summary.csv")
        p.add_argument("--trajectories_csv", default="data/results/closed_loop_validation_grid/closed_loop_trajectories.csv.gz")
        p.add_argument("--out_prefix", default="data/results/closed_loop_validation_grid/figures/fig18_closed_loop_validation")
        p.add_argument("--results_dir", default="data/results/fig0_4_full_grid")
        p.add_argument("--figS_results_dir", default="data/results/specialization_wide_dense_figS")
        p.add_argument("--other_results_dir", default="")
        p.add_argument("--replay_metric", default="top1_share", choices=["top1_share", "phase_code"])
        return p
    def pretty_condition_name(label: str) -> str:
        return PRETTY_NAMES.get(label, label.replace("_", " "))
    def phase_marker(phase: str) -> str:
        return PHASE_MARKERS.get(phase, "o")
    def style_panel_frame(ax) -> None:
        ax.spines["left"].set_linewidth(2.0)
        ax.spines["bottom"].set_linewidth(2.0)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    def normalize_columns(summary: pd.DataFrame, traj: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        summary = summary.copy()
        traj = traj.copy()

        if "condition_label" not in summary.columns:
            if "label" in summary.columns:
                summary["condition_label"] = summary["label"]
            elif "condition_key" in summary.columns:
                summary["condition_label"] = summary["condition_key"]
            else:
                raise KeyError("summary CSV must contain condition_label, label, or condition_key")

        if "condition_label" not in traj.columns:
            if "label" in traj.columns:
                traj["condition_label"] = traj["label"]
            elif "condition_key" in traj.columns:
                traj["condition_label"] = traj["condition_key"]
            else:
                raise KeyError("trajectory CSV must contain condition_label, label, or condition_key")

        summary["condition_label"] = summary["condition_label"].astype(str)
        traj["condition_label"] = traj["condition_label"].astype(str)
        return summary, traj
    def ordered_conditions(summary: pd.DataFrame) -> list[str]:
        present = list(summary["condition_label"].dropna().unique())
        order = [c for c in PREFERRED_CONDITION_ORDER if c in present]
        order.extend([c for c in present if c not in order])
        return order
    def condition_family(condition: str) -> str:
        """Return a stable visual family for closed-loop validation conditions."""
        if condition.startswith("nosup_"):
            return "no_support"
        if "beta0.5" in condition:
            return "weak_support"
        if "beta1.5" in condition:
            return "strong_support"
        return "no_support"
    def condition_style_map(condition_order: list[str]) -> dict[str, dict]:
        styles: dict[str, dict] = {}
        for condition in condition_order:
            family = condition_family(condition)
            styles[condition] = {
                "color": FAMILY_COLORS.get(family, "#7f7f7f"),
                "family": family,
                "representative": condition in PREFERRED_CONDITION_ORDER,
            }
        return styles
    def coerce_numeric(summary: pd.DataFrame, traj: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        summary = summary.copy()
        traj = traj.copy()
        for col in [
            "top1_share", "i_exp_norm", "n_eff", "seed", "eta", "alpha", "beta_sup",
            "beta_route", "lambda_risk", "context_trait_strength", "route_power",
        ]:
            if col in summary.columns:
                summary[col] = pd.to_numeric(summary[col], errors="coerce")
        for col in ["top1_share", "n_eff", "timestep", "seed"]:
            if col in traj.columns:
                traj[col] = pd.to_numeric(traj[col], errors="coerce")
        return summary, traj
    def validate_columns(summary: pd.DataFrame, traj: pd.DataFrame) -> None:
        required_summary = ["condition_label", "top1_share", "i_exp_norm", "n_eff"]
        required_traj = ["condition_label", "seed", "timestep", "top1_share", "n_eff"]
        missing_summary = [c for c in required_summary if c not in summary.columns]
        missing_traj = [c for c in required_traj if c not in traj.columns]
        if missing_summary:
            raise KeyError(f"Missing required columns in summary: {missing_summary}")
        if missing_traj:
            raise KeyError(f"Missing required columns in trajectories: {missing_traj}")
    def plot_endpoint_phase_map(ax, summary: pd.DataFrame, condition_order: list[str], styles: dict[str, dict]) -> None:
        style_panel_frame(ax)
        ax.axvspan(0.80, 1.02, alpha=0.08, zorder=0)
        ax.axhspan(0.12, max(0.35, summary["i_exp_norm"].max() + 0.06), alpha=0.06, zorder=0)
        ax.axvline(0.45, linewidth=1.0, alpha=0.25)
        ax.axhline(0.12, linewidth=1.0, alpha=0.25)

        ax.text(0.04, 0.055, "Diffuse coexistence", alpha=0.8)
        ax.text(0.86, 0.03, "Collapse", alpha=0.8, ha="center")
        ax.text(0.08, 0.25, "Contextual\nspecialization", alpha=0.8)

        for condition in condition_order:
            sub = summary.loc[summary["condition_label"] == condition].copy()
            if sub.empty:
                continue
            color = styles[condition]["color"]
            is_rep = bool(styles[condition].get("representative", False))
            point_alpha = 0.48 if is_rep else 0.20
            point_size = 78 if is_rep else 54
            for _, row in sub.iterrows():
                ax.scatter(
                    row["top1_share"],
                    row["i_exp_norm"],
                    s=point_size,
                    alpha=point_alpha,
                    marker=phase_marker(str(row.get("phase_label", ""))),
                    color=color,
                    edgecolors="none",
                    zorder=2 if is_rep else 1,
                )

            x_mean = sub["top1_share"].mean()
            y_mean = sub["i_exp_norm"].mean()
            x_std = sub["top1_share"].std(ddof=1) if len(sub) > 1 else 0.0
            y_std = sub["i_exp_norm"].std(ddof=1) if len(sub) > 1 else 0.0
            mean_alpha = 0.95 if is_rep else 0.35
            ax.errorbar(
                x_mean, y_mean, xerr=x_std, yerr=y_std, fmt="none",
                elinewidth=1.6 if is_rep else 1.0, capsize=3, color=color,
                alpha=mean_alpha, zorder=3 if is_rep else 2,
            )
            ax.scatter(
                x_mean, y_mean, s=300 if is_rep else 135, marker="o", color=color,
                alpha=mean_alpha, edgecolors="black" if is_rep else "none",
                linewidths=1.0 if is_rep else 0.0, zorder=4 if is_rep else 2,
            )

            # Number the four representative conditions so Panels A, B, and C can
            # be read together without long text labels in the dense endpoint map.
            if is_rep:
                label = REPRESENTATIVE_NUMBERS.get(condition, "")
                ax.annotate(
                    label,
                    xy=(x_mean, y_mean),
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=13,
                    weight="bold",
                    zorder=8,
                )

            # Direct text labels overlap in the dense 14-condition endpoint map.
            # Use a compact family legend instead; Panel B titles identify the four
            # representative trajectory conditions.

        ax.set_xlim(0.0, min(1.05, max(1.0, summary["top1_share"].max() + 0.05)))
        ax.set_ylim(-0.03, max(0.35, summary["i_exp_norm"].max() + 0.06))
        ax.set_xlabel("Top-1 Share")
        ax.set_ylabel("Exposure-context MI")
        #ax.set_title("A. Endpoint phase map", loc="left", weight="bold")
        ax.text(-0.015, 1.02, "A", transform=ax.transAxes, fontsize=20, fontweight="bold", va="bottom", ha="left")
        family_legend = ax.legend(
            handles=support_family_legend_handles(marker="o", markersize=8),
            loc="upper left",
            bbox_to_anchor=(0.65, 0.99),
            frameon=True,
            framealpha=0.85,
            fontsize=15,
            borderpad=0.35,
            handletextpad=0.35,
        )
        ax.add_artist(family_legend)
        """
        ax.legend(
            handles=representative_legend_handles(),
            loc="upper left",
            bbox_to_anchor=(0.65, 0.70),
            frameon=True,
            framealpha=0.85,
            fontsize=11,
            borderpad=0.35,
            handletextpad=0.35,
        )
        """
    def plot_top1_trajectories(fig, gs, traj: pd.DataFrame, summary: pd.DataFrame, condition_order: list[str], styles: dict[str, dict]) -> None:
        for idx, condition in enumerate(condition_order):
            row = idx // 2
            col = idx % 2
            ax = fig.add_subplot(gs[row, col])
            sub = traj.loc[traj["condition_label"] == condition].copy()
            endpoint_sub = summary.loc[summary["condition_label"] == condition].copy()
            if sub.empty:
                ax.set_visible(False)
                continue
            style_panel_frame(ax)

            color = styles[condition]["color"]
            for _, seed_df in sub.groupby("seed"):
                seed_df = seed_df.sort_values("timestep")
                ax.plot(seed_df["timestep"], seed_df["top1_share"], linewidth=1.2, alpha=0.45, color=color)

            mean_df = sub.groupby("timestep", as_index=False)["top1_share"].mean().sort_values("timestep")
            ax.plot(mean_df["timestep"], mean_df["top1_share"], linewidth=2.8, color=color)

            ax.set_title(pretty_condition_name(condition)) #, fontsize=16)
            ax.set_xlim(sub["timestep"].min(), sub["timestep"].max())
            ax.set_ylim(0.0, 1.05)
            ax.set_xlabel("Timestep")
            ax.set_ylabel("Top-1 share")

            endpoint_top1 = endpoint_sub["top1_share"].mean() if not endpoint_sub.empty else float("nan")
            endpoint_neff = endpoint_sub["n_eff"].mean() if not endpoint_sub.empty else float("nan")
            endpoint_phase = endpoint_sub["phase_label"].mode().iloc[0] if "phase_label" in endpoint_sub.columns and not endpoint_sub.empty else ""
    def find_replay_summary_csvs(*dirs: str) -> list[Path]:
        candidates: list[Path] = []
        for raw in dirs:
            if not raw:
                continue
            d = Path(raw)
            if d.is_file() and d.suffix.lower() == ".csv":
                candidates.append(d)
                continue
            direct_candidates: list[Path] = []
            for name in ["phase_summary.csv", "summary.csv", "stability_summary.csv"]:
                p = d / name
                if p.exists():
                    direct_candidates.append(p)
            if direct_candidates:
                candidates.extend(direct_candidates)
            elif d.exists():
                candidates.extend(sorted(d.glob("**/phase_summary.csv")))
        seen = set()
        unique = []
        for c in candidates:
            key = str(c.resolve())
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique
    def load_replay_summary(*dirs: str) -> pd.DataFrame:
        paths = find_replay_summary_csvs(*dirs)
        if not paths:
            return pd.DataFrame()
        frames = []
        for path in paths:
            df = pd.read_csv(path)
            df["_source_csv"] = str(path)
            frames.append(df)
        return pd.concat(frames, ignore_index=True, sort=False)
    def nearest_value(values: pd.Series, target: float) -> Optional[float]:
        vals = pd.to_numeric(values, errors="coerce").dropna().unique()
        if len(vals) == 0:
            return None
        vals = np.asarray(vals, dtype=float)
        return float(vals[np.argmin(np.abs(vals - float(target)))])
    def filter_nearest(df: pd.DataFrame, **targets: float) -> pd.DataFrame:
        out = df.copy()
        for col, target in targets.items():
            if col not in out.columns or out.empty:
                continue
            vals = pd.to_numeric(out[col], errors="coerce")
            nearest = nearest_value(vals, float(target))
            if nearest is None:
                continue
            out = out.loc[np.isclose(vals, nearest, rtol=0, atol=1e-9)].copy()
        return out
    def prepare_replay_pivot(replay: pd.DataFrame, x_col: str, y_col: str, metric: str, filters: Optional[dict[str, float]] = None) -> pd.DataFrame:
        if replay.empty or x_col not in replay.columns or y_col not in replay.columns:
            return pd.DataFrame()
        metric_col = metric if metric in replay.columns else "top1_share"
        if metric_col not in replay.columns:
            return pd.DataFrame()
        df = replay.copy()
        for col in [x_col, y_col, metric_col]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=[x_col, y_col, metric_col])
        if filters:
            df = filter_nearest(df, **filters)
        if df.empty:
            return pd.DataFrame()
        pivot = df.pivot_table(index=y_col, columns=x_col, values=metric_col, aggfunc="mean")
        return pivot.sort_index().sort_index(axis=1)
    def draw_replay_heatmap(ax, pivot: pd.DataFrame, title: str, xlabel: str, ylabel: str, metric: str):
        if pivot.empty:
            ax.text(0.5, 0.5, "Replay summary\nnot found", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return None
        style_panel_frame(ax)
        values = pivot.to_numpy(dtype=float)
        if metric == "top1_share":
            vmin, vmax = 0.0, 1.0
            cmap = plt.get_cmap("viridis_r").copy()
            cmap.set_bad("#d0d0d0")
            norm = None
        elif metric == "phase_code":
            vmin, vmax = None, None
            cmap = ListedColormap(PHASE_COLORS).copy()
            cmap.set_bad("#d0d0d0")
            norm = BoundaryNorm(np.arange(-0.5, 8.5, 1), len(PHASE_COLORS))
        else:
            vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
            cmap = plt.get_cmap("viridis_r").copy()
            cmap.set_bad("#d0d0d0")
            norm = None
        im = ax.imshow(values, origin="lower", aspect="auto", cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, interpolation="nearest", alpha=0.75)
        x_stride = 3 if len(pivot.columns) > 18 else 2 if len(pivot.columns) > 12 else 1
        y_stride = 3 if len(pivot.index) > 18 else 2 if len(pivot.index) > 12 else 1
        x_tick_idx = np.arange(len(pivot.columns))[::x_stride]
        y_tick_idx = np.arange(len(pivot.index))[::y_stride]
        ax.set_xticks(x_tick_idx)
        ax.set_xticklabels([f"{pivot.columns[i]:g}" for i in x_tick_idx], rotation=45, ha="right")
        ax.set_yticks(y_tick_idx)
        ax.set_yticklabels([f"{pivot.index[i]:g}" for i in y_tick_idx])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", weight="bold") #, fontsize=15)
        return im
    def overlay_closed_loop_points_on_grid(
        ax,
        pivot: pd.DataFrame,
        summary: pd.DataFrame,
        condition_order: list[str],
        styles: dict[str, dict],
        x_col: str,
        y_col: str,
        conditions: set[str],
        label_offsets: Optional[dict[str, tuple[int, int, str]]] = None,
    ) -> None:
        """Overlay closed-loop conditions on a replay heatmap.

        All conditions in ``conditions`` are drawn as stars at the nearest replay-grid
        coordinate.  Only the representative anchors are text-labeled, so Panel C can
        show all 14 validation conditions without becoming unreadable.
        """
        if pivot.empty:
            return
        label_offsets = label_offsets or {}
        x_vals = np.asarray(pivot.columns, dtype=float)
        y_vals = np.asarray(pivot.index, dtype=float)
        x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
        y_min, y_max = float(np.min(y_vals)), float(np.max(y_vals))

        for condition in condition_order:
            if condition not in conditions:
                continue
            sub = summary.loc[summary["condition_label"] == condition].copy()
            if sub.empty or x_col not in sub.columns or y_col not in sub.columns:
                continue
            x_series = pd.to_numeric(sub[x_col], errors="coerce").dropna()
            y_series = pd.to_numeric(sub[y_col], errors="coerce").dropna()
            if x_series.empty or y_series.empty:
                continue

            x = float(x_series.iloc[0])
            y = float(y_series.iloc[0])
            x_pos = int(np.argmin(np.abs(x_vals - x)))
            y_pos = int(np.argmin(np.abs(y_vals - y)))
            color = styles[condition]["color"]

            is_rep = condition in PREFERRED_CONDITION_ORDER
            ax.scatter(
                x_pos, y_pos,
                s=300 if is_rep else 250,
                marker="*",
                color=color,
                alpha=0.95 if is_rep else 0.65,
                edgecolors="black",
                linewidths=0.9 if is_rep else 0.45,
                zorder=6 if is_rep else 5,
            )
            if is_rep:
                ax.annotate(
                    REPRESENTATIVE_NUMBERS.get(condition, ""),
                    xy=(x_pos, y_pos),
                    xytext=(0, 0),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=10,
                    weight="bold",
                    zorder=8,
                )
    def plot_replay_overlay_panel(fig, gs, summary: pd.DataFrame, condition_order: list[str], styles: dict[str, dict], args: argparse.Namespace) -> None:
        panel = gs.subgridspec(1, 2, wspace=0.04)
        ax_main = fig.add_subplot(panel[0, 0])
        ax_sup = fig.add_subplot(panel[0, 1])
        main_replay = load_replay_summary(args.results_dir, args.other_results_dir)
        spec_replay = load_replay_summary(args.figS_results_dir, args.other_results_dir)
        metric = args.replay_metric

        main_filters: dict[str, float] = {}
        if "beta_sup" in main_replay.columns:
            main_filters["beta_sup"] = 0.0
        if "beta_route" in main_replay.columns:
            main_filters["beta_route"] = 2.0
        if "lambda_risk" in main_replay.columns:
            main_filters["lambda_risk"] = 0.0
        main_pivot = prepare_replay_pivot(main_replay, x_col="eta", y_col="alpha", metric=metric, filters=main_filters)
        im_main = draw_replay_heatmap(ax_main, main_pivot, "No-support slice", r"Selection strength $\eta$", r"Context fidelity $\alpha$", metric)
        main_conditions = {c for c in condition_order if c.startswith("nosup_")}
        overlay_closed_loop_points_on_grid(
            ax_main,
            main_pivot,
            summary,
            condition_order,
            styles,
            x_col="eta",
            y_col="alpha",
            conditions=main_conditions,
            label_offsets=PANEL_C_MAIN_LABEL_OFFSETS,
        )

        spec_filters: dict[str, float] = {}
        if "alpha" in spec_replay.columns:
            spec_filters["alpha"] = 1.0
        if "beta_route" in spec_replay.columns:
            spec_filters["beta_route"] = 2.0
        if "lambda_risk" in spec_replay.columns:
            spec_filters["lambda_risk"] = 0.0
        spec_pivot = prepare_replay_pivot(spec_replay, x_col="eta", y_col="beta_sup", metric=metric, filters=spec_filters)
        im_spec = draw_replay_heatmap(ax_sup, spec_pivot, "Support slice", r"Selection strength $\eta$", r"Support strength $\beta_{\mathrm{sup}}$", metric)
        support_conditions = {c for c in condition_order if c.startswith("sup_")}
        overlay_closed_loop_points_on_grid(
            ax_sup,
            spec_pivot,
            summary,
            condition_order,
            styles,
            x_col="eta",
            y_col="beta_sup",
            conditions=support_conditions,
            label_offsets=PANEL_C_SUPPORT_LABEL_OFFSETS,
        )


        ax_main.legend(
            handles=[support_family_legend_handles(marker="*", markersize=17)[0]],
            loc="upper left",
            frameon=True,
            framealpha=0.85,
            fontsize=14,
            borderpad=0.35,
            handletextpad=0.35,
        )
        ax_sup.legend(
            handles=support_family_legend_handles(marker="*", markersize=17)[1:],
            loc="upper left",
            frameon=True,
            framealpha=0.85,
            fontsize=14,
            borderpad=0.35,
            handletextpad=0.35,
        )

        # Numbered representative anchors correspond to the four trajectory panels.
        """
        fig.legend(
            handles=representative_legend_handles(),
            loc="lower center",
            bbox_to_anchor=(0.50, 0.015),
            ncol=2,
            frameon=True,
            framealpha=0.90,
            fontsize=11,
            borderpad=0.35,
            handletextpad=0.35,
        )
        """

        im = im_spec if im_spec is not None else im_main
        if im is not None:
            cbar = fig.colorbar(im, ax=[ax_main, ax_sup], shrink=0.82, pad=0.018)
            cbar.set_label("Replay top-1 share" if metric == "top1_share" else metric)
    def main() -> None:
        args = build_arg_parser().parse_args()
        summary_path = Path(args.summary_csv)
        traj_path = Path(args.trajectories_csv)
        out_prefix = Path(args.out_prefix)
        out_prefix.parent.mkdir(parents=True, exist_ok=True)

        summary = pd.read_csv(summary_path)
        traj = pd.read_csv(traj_path)
        summary, traj = normalize_columns(summary, traj)
        summary, traj = coerce_numeric(summary, traj)
        validate_columns(summary, traj)
        condition_order = ordered_conditions(summary)
        styles = condition_style_map(condition_order)

        fig = plt.figure(figsize=(12, 17.6), constrained_layout=True)
        outer = fig.add_gridspec(nrows=3, ncols=1, height_ratios=[1.0, 1.25, 1.4], hspace=0.02)

        ax_top = fig.add_subplot(outer[0, 0])
        plot_endpoint_phase_map(ax_top, summary, condition_order, styles)

        bottom_outer = outer[1, 0].subgridspec(nrows=2, ncols=1, height_ratios=[0.08, 0.92], hspace=0.0)
        title_ax = fig.add_subplot(bottom_outer[0, 0])
        title_ax.axis("off")
        title_ax.text(0.0, 0.5, "B", fontsize=20, fontweight="bold", va="center") # Time evolution of concentration
        bottom = bottom_outer[1, 0].subgridspec(2, 2, wspace=0.02, hspace=0.0)
        trajectory_conditions = [c for c in PREFERRED_CONDITION_ORDER if c in summary["condition_label"].unique()]
        plot_top1_trajectories(fig, bottom, traj, summary, trajectory_conditions, styles)
        #plot_top1_trajectories(fig, bottom, traj, summary, condition_order, styles)

        c_outer = outer[2, 0].subgridspec(nrows=2, ncols=1, height_ratios=[0.07, 0.93], hspace=0.0)
        c_title_ax = fig.add_subplot(c_outer[0, 0])
        c_title_ax.axis("off")
        c_title_ax.text(
            0.0,
            0.5,
            "C", #. Closed-loop conditions overlaid on full-scale frozen replay landscapes",
            fontsize=20,
            fontweight="bold",
            va="center",
        )
        plot_replay_overlay_panel(fig, c_outer[1, 0], summary, condition_order, styles, args)

        #png_path = out_prefix.with_suffix(".png")
        pdf_path = out_prefix.with_suffix(".pdf")
        #fig.savefig(png_path, dpi=300, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        #print(f"wrote {png_path}")
        print(f"wrote {pdf_path}")
    main()


def main_online():
    """Fig. 16: the pipeline writes to data/results/closed_loop_validation_grid/figures; copy the PDF into the paper."""
    import shutil
    _main_online()
    shutil.copy(ROOT / "data/results/closed_loop_validation_grid/figures/fig18_closed_loop_validation.pdf", PAPER_FIGS / "fig18_closed_loop_validation.pdf")


# ====================================================================================================
# appendix (Figs. 7-10, 12, 13, 14): the embedded pipeline, run from data/ so its relative paths resolve
# ====================================================================================================
def main_appendix():
    import shutil, subprocess
    pipe = str(ROOT / "pipeline/appendix_figures.py"); data = ROOT / "data"; fig = PAPER_FIGS
    run = lambda *a, **kw: subprocess.run([sys.executable, pipe, *a], cwd=data, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kw)
    run("figures")                                                                   # fig7, fig8, fig9, fig13, fig17
    run("phantom", "--threshold-out-dir", "results/threshold_robustness")            # fig4_phantom_diversity_rate
    run("figures", "--figures-extra", "--results_dir results_reuse_clean/fig0_4_full_grid --figS_results_dir results_reuse_clean/specialization_wide_dense_figS "
        "--fig5_results_dir results_reuse_clean/fig5_full_grid --fig10_results_dir results_reuse_clean/fig10_full_grid "
        "--appendix_route_results_dir results_reuse_clean/appendix_route_sweep_parallel", env={**os.environ, "FIGURES_OUT_DIR": "results_reuse_clean/figures_reuse"})
    shutil.copy(data / "results_reuse_clean/figures_reuse/fig6_representative_no_support_phase_slice.pdf", fig / "fig6_representative_no_support_phase_slice_deepseek.pdf")
    for f in ["fig7_local_stability_empirical_criticality", "fig8_support_intervention", "fig9_operational_phase_diagram", "fig13_route_sweep",
              "fig17_specialization_boundary_uncertainty", "fig4_phantom_diversity_rate"]:
        shutil.copy(data / f"results/figures/{f}.pdf", fig / f"{f}.pdf")
    print("appendix figures written to", fig)


SUBCOMMANDS = {"fig2": main_fig2, "fig3": main_fig3, "fig4_5_11": main_fig4_5_11, "testbeds": main_testbeds, "online": main_online, "appendix": main_appendix}


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
