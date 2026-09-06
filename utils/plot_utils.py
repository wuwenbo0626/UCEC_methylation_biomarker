from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib_cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.metrics import roc_auc_score, roc_curve


PALETTE = {
    "blue": "#0072B2", "orange": "#D55E00", "green": "#009E73",
    "purple": "#CC79A7", "sky": "#56B4E9", "gray": "#7A7A7A",
    "dark": "#222222", "light_gray": "#D9D9D9",
}


def savefig(fig, png: Path, pdf: Path | None = None):
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=300, bbox_inches="tight")
    if pdf:
        pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def workflow_plot(out_png: Path, out_pdf: Path):
    steps = [
        ("Download", "TCGA-UCEC HM450"),
        ("QC", "missing/SNP/chrX-Y/KNN"),
        ("DMP", "t-test + FDR"),
        ("Annotate", "gene + promoter"),
        ("Select", "LASSO + RF"),
        ("Evaluate", "LR/RF/XGB/SVM"),
        ("Compare", "PAX1/JAM3"),
    ]
    fig, ax = plt.subplots(figsize=(13.2, 2.8))
    ax.axis("off")
    xs = np.linspace(0.055, 0.945, len(steps))
    for i, (x, (a, b)) in enumerate(zip(xs, steps)):
        ax.add_patch(FancyBboxPatch((x - .055, .38), .11, .34, boxstyle="round,pad=.012", ec=PALETTE["blue"], fc="#EEF5FA", lw=1.2))
        ax.text(x, .60, a, ha="center", va="center", fontsize=9, fontweight="bold")
        ax.text(x, .49, b, ha="center", va="center", fontsize=7.2)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + .06, .55), (xs[i + 1] - .06, .55), arrowstyle="-|>", mutation_scale=12, color=PALETTE["gray"]))
    ax.set_title("Reproducible workflow for UCEC methylation biomarker discovery", fontsize=12)
    savefig(fig, out_png, out_pdf)


def volcano_plot(df, out_png: Path, out_pdf: Path, delta_thr=0.2, fdr_thr=0.05):
    fdr = df["FDR"].to_numpy(float)
    min_pos = np.nanmin(np.where(fdr > 0, fdr, np.nan))
    y = -np.log10(np.clip(fdr, min_pos, 1.0))
    d = df["delta_beta"].to_numpy(float)
    hyper = (d > delta_thr) & (fdr < fdr_thr)
    hypo = (d < -delta_thr) & (fdr < fdr_thr)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.scatter(d[~(hyper | hypo)], y[~(hyper | hypo)], s=3, c=PALETTE["light_gray"], alpha=.35, linewidths=0)
    ax.scatter(d[hypo], y[hypo], s=4, c=PALETTE["blue"], alpha=.7, linewidths=0, label=f"Hypo ({hypo.sum():,})")
    ax.scatter(d[hyper], y[hyper], s=4, c=PALETTE["orange"], alpha=.7, linewidths=0, label=f"Hyper ({hyper.sum():,})")
    ax.axvline(delta_thr, color=PALETTE["dark"], ls="--", lw=1)
    ax.axvline(-delta_thr, color=PALETTE["dark"], ls="--", lw=1)
    ax.axhline(-np.log10(fdr_thr), color=PALETTE["dark"], ls="--", lw=1)
    ax.set(xlabel="Delta beta: Tumor - Normal", ylabel="-log10(FDR)", title="Differential methylation")
    ax.grid(alpha=.18); ax.legend(frameon=True, markerscale=3)
    savefig(fig, out_png, out_pdf)


def roc_plot(y_true, score_map: dict, out_png: Path, out_pdf: Path, title="ROC curves"):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    colors = [PALETTE["orange"], PALETTE["blue"], PALETTE["green"], PALETTE["purple"]]
    for (name, score), color in zip(score_map.items(), colors):
        fpr, tpr, _ = roc_curve(y_true, score)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC={roc_auc_score(y_true, score):.3f})")
    ax.plot([0, 1], [0, 1], "--", color=PALETTE["gray"], lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=title)
    ax.grid(alpha=.18); ax.legend(loc="lower right", frameon=True)
    savefig(fig, out_png, out_pdf)


def confusion_grid(cm_map: dict, out_png: Path, out_pdf: Path, title="Confusion matrices"):
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.0))
    axes = axes.ravel()
    vmax = max(int(cm.max()) for cm in cm_map.values())
    im = None
    for ax, (name, cm) in zip(axes, cm_map.items()):
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
        ax.set_title(name); ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Tumor"]); ax.set_yticklabels(["Normal", "Tumor"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="white" if cm[i, j] > vmax/2 else "black", fontweight="bold")
    for ax in axes[len(cm_map):]:
        ax.axis("off")
    if im is not None:
        fig.colorbar(im, ax=axes.tolist(), fraction=.046, pad=.04)
    fig.suptitle(title, y=.98)
    savefig(fig, out_png, out_pdf)
