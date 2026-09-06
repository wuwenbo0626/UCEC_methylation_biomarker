#!/usr/bin/env python3
"""
Assess TCGA-UCEC HM450 methylation batch effects with PCA and optional ComBat.

Tasks:
  1. Extract potential batch variables from clinical/sample metadata and TCGA barcode.
  2. Run PCA on the cleaned beta matrix, using top-variable probes by default
     for memory efficiency.
  3. Plot PCA colored by Tumor/Normal and candidate batch variables.
  4. Quantify the relation between Tumor/Normal separation and batch variables:
       - eta-squared of each categorical variable on PC1/PC2
       - Cramer's V and chi-square p-value between group and each batch variable
  5. If batch effect is detected, run ComBat via pycombat on the beta matrix
     while preserving Tumor/Normal group as a biological covariate.
  6. Output before/after PCA comparison plots.

Notes:
  - sample_type/sample_type_code are biological labels here; they are plotted
    but are never used as ComBat batch variables.
  - ComBat on beta values can produce values outside [0, 1], so corrected
    beta values are clipped back to [1e-6, 0.999999] before saving.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from scipy.stats import chi2_contingency
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_ROOT = Path("/Users/wuwenbo/AsiaInfo/tcga_ucec_methylation")
DEFAULT_CLEAN = DEFAULT_ROOT / "tcga_ucec_hm450_cleaned"
DEFAULT_OUT = DEFAULT_ROOT / "tcga_ucec_hm450_batch_effect"

os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_ROOT / ".matplotlib_cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PALETTE = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "gray": "#7A7A7A",
    "dark": "#222222",
    "light_gray": "#D9D9D9",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Assess batch effects and optionally apply ComBat.")
    p.add_argument("--beta-npy", type=Path, default=DEFAULT_CLEAN / "matrix" / "ucec_hm450_beta_clean_probes_by_samples.npy")
    p.add_argument("--sample-groups", type=Path, default=DEFAULT_CLEAN / "metadata" / "ucec_hm450_clean_sample_groups.parquet")
    p.add_argument("--probe-metadata", type=Path, default=DEFAULT_CLEAN / "metadata" / "ucec_hm450_clean_probe_metadata.parquet")
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--top-variable-probes", type=int, default=50000, help="Use top N variable probes for PCA. Use 0 for all probes.")
    p.add_argument("--chunk-probes", type=int, default=20000)
    p.add_argument("--n-components", type=int, default=10)
    p.add_argument("--batch-for-combat", default="auto", help="auto, none, or a column such as plate/tss/portion.")
    p.add_argument("--combat-mode", choices=["auto", "always", "never"], default="auto")
    p.add_argument("--eta2-threshold", type=float, default=0.05)
    p.add_argument("--cramers-threshold", type=float, default=0.20)
    p.add_argument("--beta-clip-eps", type=float, default=1e-6)
    return p.parse_args()


def mem() -> str:
    proc = psutil.Process(os.getpid())
    return f"RSS={proc.memory_info().rss/1024**3:.2f} GB, available={psutil.virtual_memory().available/1024**3:.2f} GB"


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg} | {mem()}", flush=True)


def ensure(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def savefig(fig, png: Path, pdf: Path) -> None:
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_batch_variables(samples: pd.DataFrame) -> pd.DataFrame:
    meta = samples.copy()
    barcode = meta["aliquot_id"].fillna(meta.get("sample_id", "")).astype(str) if "aliquot_id" in meta.columns else meta["sample_id"].astype(str)
    parts = barcode.str.split("-", expand=True)
    meta["batch_barcode"] = barcode
    meta["tissue_source_site"] = parts[1] if parts.shape[1] > 1 else pd.NA
    sample_token = parts[3] if parts.shape[1] > 3 else pd.Series(pd.NA, index=meta.index)
    portion_token = parts[4] if parts.shape[1] > 4 else pd.Series(pd.NA, index=meta.index)
    meta["sample_type_code"] = sample_token.astype(str).str[:2].replace("na", pd.NA)
    meta["vial"] = sample_token.astype(str).str[2:3].replace("", pd.NA)
    meta["portion"] = portion_token.astype(str).str[:2].replace("na", pd.NA)
    meta["analyte"] = portion_token.astype(str).str[2:3].replace("", pd.NA)
    meta["plate"] = parts[5] if parts.shape[1] > 5 else pd.NA
    meta["center"] = parts[6] if parts.shape[1] > 6 else pd.NA
    # Friendly aliases requested by user.
    meta["tss"] = meta["tissue_source_site"]
    if "sentrix_id" not in meta.columns:
        meta["sentrix_id"] = pd.NA
    return meta


def group_labels(samples: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    g = samples["group"].astype(str)
    y = g.str.lower().isin(["tumor", "primary tumor", "1"]).astype(np.int8).to_numpy()
    sample_cols = samples["matrix_col"].to_numpy(np.int64) if "matrix_col" in samples.columns else np.arange(len(samples), dtype=np.int64)
    return y, sample_cols


def select_top_variable_probes(beta: np.ndarray, top_n: int, chunk: int) -> np.ndarray:
    n_probes = beta.shape[0]
    if top_n <= 0 or top_n >= n_probes:
        log(f"Using all probes for PCA: {n_probes:,}")
        return np.arange(n_probes, dtype=np.int64)
    log(f"Computing probe variance in chunks; selecting top {top_n:,}/{n_probes:,} variable probes")
    variances = np.empty(n_probes, dtype=np.float32)
    for s in range(0, n_probes, chunk):
        e = min(s + chunk, n_probes)
        block = np.asarray(beta[s:e, :], dtype=np.float32)
        variances[s:e] = block.var(axis=1, ddof=1)
        del block
        if s == 0 or e == n_probes or e % (chunk * 5) == 0:
            log(f"Variance processed probes {e:,}/{n_probes:,}")
        gc.collect()
    idx = np.argpartition(variances, -top_n)[-top_n:]
    idx = idx[np.argsort(variances[idx])[::-1]]
    return np.sort(idx.astype(np.int64))


def pca_from_probe_indices(beta_path: Path, probe_indices: np.ndarray, sample_cols: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    beta = np.load(beta_path, mmap_mode="r")
    data = np.asarray(beta[np.ix_(probe_indices, sample_cols)].T, dtype=np.float32)
    del beta
    gc.collect()
    log(f"Running PCA on matrix samples x probes = {data.shape}")
    data = StandardScaler(copy=False).fit_transform(data).astype(np.float32, copy=False)
    pca = PCA(n_components=min(n_components, data.shape[0], data.shape[1]), svd_solver="randomized", random_state=42)
    pcs = pca.fit_transform(data)
    evr = pca.explained_variance_ratio_
    del data, pca
    gc.collect()
    return pcs, evr


def pca_from_array(data_samples_by_probes: np.ndarray, probe_indices: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(data_samples_by_probes[:, probe_indices], dtype=np.float32)
    log(f"Running corrected PCA on matrix samples x probes = {data.shape}")
    data = StandardScaler(copy=False).fit_transform(data).astype(np.float32, copy=False)
    pca = PCA(n_components=min(n_components, data.shape[0], data.shape[1]), svd_solver="randomized", random_state=42)
    pcs = pca.fit_transform(data)
    evr = pca.explained_variance_ratio_
    del data, pca
    gc.collect()
    return pcs, evr


def eta_squared(values: np.ndarray, categories: pd.Series) -> float:
    df = pd.DataFrame({"value": values, "cat": categories.astype("string").fillna("NA")})
    grand = df["value"].mean()
    ss_total = ((df["value"] - grand) ** 2).sum()
    if ss_total <= 0:
        return 0.0
    ss_between = df.groupby("cat", observed=True)["value"].agg(lambda x: len(x) * (x.mean() - grand) ** 2).sum()
    return float(ss_between / ss_total)


def cramers_v_and_p(group: pd.Series, batch: pd.Series) -> tuple[float, float]:
    tab = pd.crosstab(group.astype(str), batch.astype(str))
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return np.nan, np.nan
    chi2, p, _, _ = chi2_contingency(tab)
    n = tab.to_numpy().sum()
    phi2 = chi2 / n
    r, k = tab.shape
    return float(np.sqrt(phi2 / max(min(k - 1, r - 1), 1))), float(p)


def logistic_auc_from_pcs(pcs: np.ndarray, y: np.ndarray) -> float:
    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    model.fit(pcs[:, :2], y)
    return float(roc_auc_score(y, model.predict_proba(pcs[:, :2])[:, 1]))


def batch_association_table(pcs: np.ndarray, samples: pd.DataFrame, y: np.ndarray, variables: list[str]) -> pd.DataFrame:
    rows = []
    group_series = samples["group"].astype(str)
    for var in variables:
        if var not in samples.columns:
            continue
        s = samples[var].astype("string").fillna("NA")
        n_levels = int(s.nunique(dropna=False))
        row = {
            "variable": var,
            "n_levels": n_levels,
            "eta2_PC1": eta_squared(pcs[:, 0], s) if n_levels > 1 else np.nan,
            "eta2_PC2": eta_squared(pcs[:, 1], s) if n_levels > 1 else np.nan,
            "cramers_v_with_group": np.nan,
            "chi_square_p_with_group": np.nan,
            "is_biological_label_or_constant": var in {"group", "sample_type", "sample_type_code"} or n_levels <= 1,
        }
        cv, cp = cramers_v_and_p(group_series, s)
        row["cramers_v_with_group"] = cv
        row["chi_square_p_with_group"] = cp
        rows.append(row)
    out = pd.DataFrame(rows)
    out["max_eta2_PC1_PC2"] = out[["eta2_PC1", "eta2_PC2"]].max(axis=1)
    out["flag_batch_or_confounding_risk"] = (
        ~out["is_biological_label_or_constant"]
        & (
            (out["max_eta2_PC1_PC2"] >= 0.05)
            | (out["cramers_v_with_group"] >= 0.20)
        )
    )
    return out.sort_values(["flag_batch_or_confounding_risk", "max_eta2_PC1_PC2"], ascending=[False, False])


def plot_pca(pcs: np.ndarray, evr: np.ndarray, samples: pd.DataFrame, color_var: str, out_png: Path, out_pdf: Path, title_prefix: str) -> None:
    s = samples[color_var].astype("string").fillna("NA")
    levels = s.value_counts().index.tolist()
    cmap = plt.get_cmap("tab20")
    color_map = {lev: cmap(i % 20) for i, lev in enumerate(levels)}
    colors = [color_map[v] for v in s]
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    ax.scatter(pcs[:, 0], pcs[:, 1], c=colors, s=28, alpha=0.82, linewidths=0.25, edgecolors="black")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_title(f"{title_prefix}: colored by {color_var}")
    ax.grid(alpha=0.18)
    # Keep legend manageable.
    handles = []
    labels = []
    for lev in levels[:25]:
        handles.append(plt.Line2D([0], [0], marker="o", linestyle="", markersize=6, color=color_map[lev]))
        labels.append(f"{lev} (n={(s == lev).sum()})")
    ax.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize=7, title=color_var)
    savefig(fig, out_png, out_pdf)


def load_full_samples_by_probes(beta_path: Path, sample_cols: np.ndarray) -> np.ndarray:
    beta = np.load(beta_path, mmap_mode="r")
    log(f"Loading full beta matrix for ComBat as samples x probes from {beta.shape}")
    data = np.asarray(beta[:, sample_cols].T, dtype=np.float32)
    del beta
    gc.collect()
    return data


def run_combat(beta_path: Path, sample_cols: np.ndarray, samples: pd.DataFrame, batch_var: str, eps: float, out_path: Path) -> tuple[Path, dict]:
    from pycombat import Combat

    data = load_full_samples_by_probes(beta_path, sample_cols)
    before_min, before_max = float(np.nanmin(data)), float(np.nanmax(data))
    batch = samples[batch_var].astype(str).to_numpy()
    group_cov = pd.get_dummies(samples["group"].astype(str), drop_first=True).to_numpy(dtype=float)
    log(f"Running pyCombat on beta matrix with batch={batch_var}; preserving group covariate; data={data.shape}")
    corrected = Combat().fit_transform(data, batch, X=group_cov)
    corrected = np.asarray(corrected, dtype=np.float32)
    raw_min, raw_max = float(np.nanmin(corrected)), float(np.nanmax(corrected))
    np.clip(corrected, eps, 1.0 - eps, out=corrected)
    clipped_n = int(((corrected <= eps) | (corrected >= 1.0 - eps)).sum())
    log(f"Saving ComBat-corrected beta matrix probes x samples: {out_path}")
    np.save(out_path, corrected.T.astype(np.float32, copy=False))
    info = {
        "batch_variable": batch_var,
        "data_shape_samples_by_probes": list(data.shape),
        "before_min": before_min,
        "before_max": before_max,
        "combat_raw_min": raw_min,
        "combat_raw_max": raw_max,
        "clipped_values_after_combat": clipped_n,
        "output_beta_npy": str(out_path),
    }
    del data, corrected
    gc.collect()
    return out_path, info


def choose_combat_batch(assoc: pd.DataFrame, preferred: str) -> str | None:
    if preferred == "none":
        return None
    if preferred != "auto":
        return preferred
    candidates = assoc[
        assoc["flag_batch_or_confounding_risk"]
        & ~assoc["variable"].isin(["group", "sample_type", "sample_type_code", "center", "analyte", "sentrix_id"])
        & (assoc["n_levels"] > 1)
    ].copy()
    if candidates.empty:
        return None
    # Plate is the most plausible technical batch from TCGA aliquot barcode.
    if "plate" in set(candidates["variable"]):
        return "plate"
    return str(candidates.iloc[0]["variable"])


def main() -> int:
    args = parse_args()
    t0 = time.time()
    for path, label in [(args.beta_npy, "beta matrix"), (args.sample_groups, "sample groups"), (args.probe_metadata, "probe metadata")]:
        ensure(path, label)
    (args.outdir / "figures").mkdir(parents=True, exist_ok=True)
    (args.outdir / "tables").mkdir(parents=True, exist_ok=True)
    (args.outdir / "matrix").mkdir(parents=True, exist_ok=True)

    log("Loading sample metadata and extracting batch variables")
    samples = pd.read_parquet(args.sample_groups).sort_values("matrix_col").reset_index(drop=True)
    samples = parse_batch_variables(samples)
    y, sample_cols = group_labels(samples)
    batch_vars = [
        "group",
        "sample_type",
        "sample_type_code",
        "plate",
        "center",
        "sentrix_id",
        "tissue_source_site",
        "tss",
        "portion",
        "analyte",
        "vial",
        "platform",
        "figo_stage",
        "histology",
        "primary_diagnosis",
    ]
    batch_vars = [v for v in batch_vars if v in samples.columns]
    samples.to_parquet(args.outdir / "tables" / "sample_metadata_with_batch_variables.parquet", index=False)

    beta = np.load(args.beta_npy, mmap_mode="r")
    if beta.shape[1] != len(samples):
        log(f"WARNING: beta sample dimension {beta.shape[1]} != sample rows {len(samples)}; using matrix_col ordering")
    probe_indices = select_top_variable_probes(beta, args.top_variable_probes, args.chunk_probes)
    np.save(args.outdir / "tables" / "pca_probe_indices.npy", probe_indices)
    del beta
    gc.collect()

    pcs, evr = pca_from_probe_indices(args.beta_npy, probe_indices, sample_cols, args.n_components)
    pc_df = pd.DataFrame(pcs[:, : min(10, pcs.shape[1])], columns=[f"PC{i+1}" for i in range(min(10, pcs.shape[1]))])
    pc_df["group"] = samples["group"].to_numpy()
    pc_df.to_parquet(args.outdir / "tables" / "pca_scores_before_combat.parquet", index=False)
    pd.DataFrame({"PC": [f"PC{i+1}" for i in range(len(evr))], "explained_variance_ratio": evr}).to_parquet(args.outdir / "tables" / "pca_explained_variance_before_combat.parquet", index=False)

    for var in batch_vars:
        if samples[var].nunique(dropna=False) > 1:
            safe = var.replace("/", "_")
            plot_pca(pcs, evr, samples, var, args.outdir / "figures" / f"pca_before_by_{safe}.png", args.outdir / "figures" / f"pca_before_by_{safe}.pdf", "Before ComBat")

    assoc = batch_association_table(pcs, samples, y, batch_vars)
    assoc_path = args.outdir / "tables" / "pca_batch_association_before_combat.parquet"
    assoc.to_parquet(assoc_path, index=False)
    group_auc_pc12 = logistic_auc_from_pcs(pcs, y)
    log(f"PC1/PC2 predict Tumor-vs-Normal AUC={group_auc_pc12:.4f}")
    log(f"Top PCA/batch associations:\n{assoc.head(8).to_string(index=False)}")

    selected_batch = choose_combat_batch(assoc, args.batch_for_combat)
    combat_info = None
    combat_run = False
    risk_found = bool(assoc["flag_batch_or_confounding_risk"].any())
    if args.combat_mode == "never":
        log("ComBat skipped because --combat-mode never")
    elif selected_batch is None:
        log("No suitable technical batch variable selected for ComBat")
    elif args.combat_mode == "always" or (args.combat_mode == "auto" and risk_found):
        combat_run = True
        corrected_path = args.outdir / "matrix" / f"ucec_hm450_beta_combat_{selected_batch}_probes_by_samples.npy"
        try:
            corrected_path, combat_info = run_combat(args.beta_npy, sample_cols, samples, selected_batch, args.beta_clip_eps, corrected_path)
            corrected_samples_by_probes = np.load(corrected_path, mmap_mode="r")[:, :].T
            pcs_after, evr_after = pca_from_array(corrected_samples_by_probes, probe_indices, args.n_components)
            del corrected_samples_by_probes
            gc.collect()
            pc_after = pd.DataFrame(pcs_after[:, : min(10, pcs_after.shape[1])], columns=[f"PC{i+1}" for i in range(min(10, pcs_after.shape[1]))])
            pc_after["group"] = samples["group"].to_numpy()
            pc_after.to_parquet(args.outdir / "tables" / "pca_scores_after_combat.parquet", index=False)
            pd.DataFrame({"PC": [f"PC{i+1}" for i in range(len(evr_after))], "explained_variance_ratio": evr_after}).to_parquet(args.outdir / "tables" / "pca_explained_variance_after_combat.parquet", index=False)
            for var in ["group", selected_batch, "sample_type"]:
                if var in samples.columns and samples[var].nunique(dropna=False) > 1:
                    plot_pca(pcs_after, evr_after, samples, var, args.outdir / "figures" / f"pca_after_by_{var}.png", args.outdir / "figures" / f"pca_after_by_{var}.pdf", "After ComBat")
            assoc_after = batch_association_table(pcs_after, samples, y, batch_vars)
            assoc_after.to_parquet(args.outdir / "tables" / "pca_batch_association_after_combat.parquet", index=False)
            # Before/after compact comparison.
            fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
            for ax, this_pcs, this_evr, title in [(axes[0], pcs, evr, "Before ComBat"), (axes[1], pcs_after, evr_after, "After ComBat")]:
                colors = np.where(y == 1, PALETTE["orange"], PALETTE["blue"])
                ax.scatter(this_pcs[:, 0], this_pcs[:, 1], c=colors, s=28, alpha=0.82, edgecolors="black", linewidths=0.25)
                ax.set_xlabel(f"PC1 ({this_evr[0]*100:.1f}%)")
                ax.set_ylabel(f"PC2 ({this_evr[1]*100:.1f}%)")
                ax.set_title(title)
                ax.grid(alpha=0.18)
            handles = [
                plt.Line2D([0], [0], marker="o", linestyle="", color=PALETTE["orange"], label="Tumor"),
                plt.Line2D([0], [0], marker="o", linestyle="", color=PALETTE["blue"], label="Normal"),
            ]
            fig.legend(handles=handles, loc="upper center", ncol=2, frameon=True)
            savefig(fig, args.outdir / "figures" / "pca_before_after_combat_by_group.png", args.outdir / "figures" / "pca_before_after_combat_by_group.pdf")
            del pcs_after, evr_after
            gc.collect()
        except Exception as exc:
            combat_run = False
            combat_info = {"error": repr(exc), "batch_variable": selected_batch}
            log(f"ComBat failed: {exc}")
    else:
        log("ComBat skipped in auto mode because no batch/confounding risk exceeded thresholds")

    conclusion = []
    group_row = assoc[assoc["variable"].eq("group")]
    if not group_row.empty:
        conclusion.append(f"Tumor/Normal explains PC1/PC2 max eta²={float(group_row['max_eta2_PC1_PC2'].iloc[0]):.3f}.")
    if selected_batch:
        batch_row = assoc[assoc["variable"].eq(selected_batch)]
        if not batch_row.empty:
            conclusion.append(
                f"Selected technical batch '{selected_batch}' has PC1/PC2 max eta²={float(batch_row['max_eta2_PC1_PC2'].iloc[0]):.3f}, "
                f"Cramer's V with group={float(batch_row['cramers_v_with_group'].iloc[0]):.3f}."
            )
    if risk_found:
        conclusion.append("PCA/batch association suggests batch or batch-group confounding risk; interpret model performance cautiously.")
    else:
        conclusion.append("No strong technical batch effect was detected by the configured thresholds.")
    if combat_run:
        conclusion.append(f"ComBat correction was applied using batch='{selected_batch}' while preserving Tumor/Normal group.")
    elif selected_batch:
        conclusion.append(f"ComBat was not successfully applied; selected batch would have been '{selected_batch}'.")

    report = {
        "inputs": {
            "beta_npy": str(args.beta_npy),
            "sample_groups": str(args.sample_groups),
            "probe_metadata": str(args.probe_metadata),
        },
        "counts": {
            "samples": int(len(samples)),
            "tumor": int(y.sum()),
            "normal": int((y == 0).sum()),
            "pca_probes_used": int(len(probe_indices)),
        },
        "batch_variables_considered": batch_vars,
        "group_auc_from_PC1_PC2": group_auc_pc12,
        "batch_effect_or_confounding_risk_found": risk_found,
        "selected_batch_for_combat": selected_batch,
        "combat_run": combat_run,
        "combat_info": combat_info,
        "conclusion": " ".join(conclusion),
        "outputs": {
            "sample_metadata_with_batch_variables": str(args.outdir / "tables" / "sample_metadata_with_batch_variables.parquet"),
            "pca_scores_before_combat": str(args.outdir / "tables" / "pca_scores_before_combat.parquet"),
            "pca_batch_association_before_combat": str(assoc_path),
            "figures_dir": str(args.outdir / "figures"),
        },
        "runtime_seconds": round(time.time() - t0, 2),
    }
    if combat_info and "output_beta_npy" in combat_info:
        report["outputs"]["combat_corrected_beta_npy"] = combat_info["output_beta_npy"]
        report["outputs"]["pca_scores_after_combat"] = str(args.outdir / "tables" / "pca_scores_after_combat.parquet")
        report["outputs"]["pca_batch_association_after_combat"] = str(args.outdir / "tables" / "pca_batch_association_after_combat.parquet")
    (args.outdir / "batch_effect_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Saved report: {args.outdir / 'batch_effect_report.json'}")
    log(f"All done in {(time.time() - t0) / 60:.2f} minutes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
