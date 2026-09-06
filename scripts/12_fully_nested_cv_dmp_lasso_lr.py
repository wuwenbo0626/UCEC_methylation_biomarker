#!/usr/bin/env python3
"""
Fully nested cross-validation for TCGA-UCEC methylation biomarker modeling.

Outer fold workflow:
  1. Split samples with stratified outer CV.
  2. On the outer training fold only:
     - Recompute genome-wide differential methylation.
     - t-test is performed on M-values, matching the earlier pipeline.
     - delta_beta is computed on beta-values.
     - BH-FDR is computed only from training-fold p-values.
     - Select promoter hypermethylated probes:
       delta_beta > 0.2, FDR < 0.05, and manifest group contains TSS200/TSS1500.
  3. On those training-fold-selected probes only:
     - Fit LassoCV within the training fold.
     - Cap selected probes at <=20.
  4. Train logistic regression on selected probes.
  5. Evaluate the untouched outer validation fold.

This avoids leakage from both differential methylation prefiltering and LASSO
feature selection into the outer validation folds.
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
from scipy.stats import ttest_ind
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, lasso_path
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


ROOT = Path("/Users/wuwenbo/AsiaInfo/tcga_ucec_methylation")
CLEAN = ROOT / "tcga_ucec_hm450_cleaned"
GDC_META = ROOT / "tcga_ucec_hm450_gdc" / "metadata"
OUT = ROOT / "tcga_ucec_hm450_fully_nested_cv"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fully nested DMP + LassoCV + LR evaluation.")
    p.add_argument("--beta-npy", type=Path, default=CLEAN / "matrix" / "ucec_hm450_beta_clean_probes_by_samples.npy")
    p.add_argument("--sample-groups", type=Path, default=CLEAN / "metadata" / "ucec_hm450_clean_sample_groups.parquet")
    p.add_argument("--probe-metadata", type=Path, default=CLEAN / "metadata" / "ucec_hm450_clean_probe_metadata.parquet")
    p.add_argument("--manifest-csv", type=Path, default=GDC_META / "HumanMethylation450_15017482_v1-2.csv")
    p.add_argument("--annotation-cache", type=Path, default=GDC_META / "hm450_clean_probe_manifest_annotation.parquet")
    p.add_argument("--outdir", type=Path, default=OUT)
    p.add_argument("--outer-cv", type=int, default=5)
    p.add_argument("--inner-cv", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--max-iter", type=int, default=10000)
    p.add_argument("--max-features", type=int, default=20)
    p.add_argument("--chunk-probes", type=int, default=20000)
    p.add_argument("--delta-beta-threshold", type=float, default=0.20)
    p.add_argument("--fdr-threshold", type=float, default=0.05)
    p.add_argument("--eps", type=float, default=1e-6)
    p.add_argument("--coef-eps", type=float, default=1e-8)
    p.add_argument("--equal-var", action="store_true", help="Use Student t-test. Default is Welch t-test.")
    return p.parse_args()


def memory_text() -> str:
    proc = psutil.Process(os.getpid())
    rss = proc.memory_info().rss / 1024**3
    avail = psutil.virtual_memory().available / 1024**3
    return f"RSS={rss:.2f} GB, available={avail:.2f} GB"


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg} | {memory_text()}", flush=True)


def ensure(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def find_manifest_header(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for i, line in enumerate(fh):
            if line.startswith("IlmnID,"):
                return i
    raise ValueError(f"Could not find IlmnID header in manifest: {path}")


def has_promoter_group(value) -> bool:
    if value is None or pd.isna(value):
        return False
    parts = [x.strip() for x in str(value).replace(",", ";").split(";")]
    return any(x in {"TSS200", "TSS1500"} for x in parts)


def primary_gene(value) -> str:
    if value is None or pd.isna(value):
        return ""
    for part in str(value).replace(",", ";").split(";"):
        part = part.strip()
        if part:
            return part.upper()
    return ""


def build_or_load_annotation(args: argparse.Namespace) -> pd.DataFrame:
    if args.annotation_cache.exists():
        ann = pd.read_parquet(args.annotation_cache)
        log(f"Loaded cached full clean-probe manifest annotation: {args.annotation_cache}")
        return ann

    ensure(args.manifest_csv, "HumanMethylation450 manifest CSV")
    ensure(args.probe_metadata, "clean probe metadata")
    log("Building full clean-probe manifest annotation cache")
    header = find_manifest_header(args.manifest_csv)
    manifest = pd.read_csv(
        args.manifest_csv,
        skiprows=header,
        usecols=["Name", "UCSC_RefGene_Name", "UCSC_RefGene_Group"],
        dtype=str,
        low_memory=False,
    )
    manifest = manifest.rename(
        columns={
            "Name": "probe_id",
            "UCSC_RefGene_Name": "gene_name",
            "UCSC_RefGene_Group": "gene_relation",
        }
    )
    manifest["probe_id"] = manifest["probe_id"].astype(str)
    # The Illumina CSV also contains repeated non-CpG control rows such as
    # STAINING/EXTENSION/HYBRIDIZATION. The cleaned matrix contains CpG probes,
    # so keep cg* probes and make probe_id unique before merging.
    manifest = manifest[manifest["probe_id"].str.startswith("cg", na=False)].copy()
    manifest = manifest.drop_duplicates("probe_id", keep="first")
    probe_meta = pd.read_parquet(args.probe_metadata)
    if "matrix_row" not in probe_meta.columns:
        probe_meta = probe_meta.copy()
        probe_meta["matrix_row"] = np.arange(len(probe_meta), dtype=np.int64)
    probe_meta["probe_id"] = probe_meta["probe_id"].astype(str)
    ann = probe_meta.merge(manifest, on="probe_id", how="left", validate="one_to_one")
    ann["is_promoter_TSS200_TSS1500"] = ann["gene_relation"].map(has_promoter_group).astype(bool)
    ann["primary_gene"] = ann["gene_name"].map(primary_gene)
    ann = ann.sort_values("matrix_row").reset_index(drop=True)
    args.annotation_cache.parent.mkdir(parents=True, exist_ok=True)
    ann.to_parquet(args.annotation_cache, index=False)
    log(f"Saved full clean-probe manifest annotation cache: {args.annotation_cache}")
    return ann


def load_labels(path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    sample = pd.read_parquet(path).copy()
    if "matrix_col" not in sample.columns:
        sample["matrix_col"] = np.arange(len(sample), dtype=np.int64)
    group = sample["group"].astype(str).str.lower()
    keep = group.isin(["tumor", "primary tumor", "1", "normal", "solid tissue normal", "0"])
    sample = sample.loc[keep].reset_index(drop=True)
    group = sample["group"].astype(str).str.lower()
    y = group.isin(["tumor", "primary tumor", "1"]).astype(np.int8).to_numpy()
    cols = sample["matrix_col"].to_numpy(np.int64)
    return y, cols, sample


def detect_orientation(beta: np.ndarray, n_probes: int, n_samples: int) -> str:
    if beta.shape == (n_probes, n_samples):
        return "probes_by_samples"
    if beta.shape == (n_samples, n_probes):
        return "samples_by_probes"
    raise ValueError(f"Matrix shape {beta.shape} does not match {n_probes} probes and {n_samples} samples.")


def read_probe_block(beta: np.ndarray, start: int, end: int, orientation: str) -> np.ndarray:
    if orientation == "probes_by_samples":
        return np.asarray(beta[start:end, :], dtype=np.float32)
    return np.asarray(beta[:, start:end].T, dtype=np.float32)


def beta_to_m_inplace(arr: np.ndarray, eps: float) -> np.ndarray:
    np.clip(arr, eps, 1.0 - eps, out=arr)
    denom = 1.0 - arr
    np.divide(arr, denom, out=arr)
    np.log2(arr, out=arr)
    return arr


def beta_rows_by_samples(beta: np.ndarray, rows: np.ndarray, cols: np.ndarray, orientation: str) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    if orientation == "probes_by_samples":
        return np.asarray(beta[np.ix_(rows, cols)], dtype=np.float32)
    return np.asarray(beta[np.ix_(cols, rows)].T, dtype=np.float32)


def train_fold_differential(
    beta: np.ndarray,
    orientation: str,
    n_probes: int,
    train_cols: np.ndarray,
    train_y: np.ndarray,
    promoter_mask: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    tumor_cols = train_cols[train_y == 1]
    normal_cols = train_cols[train_y == 0]
    t_stat = np.empty(n_probes, dtype=np.float32)
    p_value = np.empty(n_probes, dtype=np.float64)
    delta_beta = np.empty(n_probes, dtype=np.float32)
    mean_tumor = np.empty(n_probes, dtype=np.float32)
    mean_normal = np.empty(n_probes, dtype=np.float32)

    for start in range(0, n_probes, args.chunk_probes):
        end = min(start + args.chunk_probes, n_probes)
        block = read_probe_block(beta, start, end, orientation)
        tumor_beta = block[:, tumor_cols].astype(np.float32, copy=True)
        normal_beta = block[:, normal_cols].astype(np.float32, copy=True)
        mt = tumor_beta.mean(axis=1, dtype=np.float64).astype(np.float32)
        mn = normal_beta.mean(axis=1, dtype=np.float64).astype(np.float32)
        mean_tumor[start:end] = mt
        mean_normal[start:end] = mn
        delta_beta[start:end] = mt - mn

        beta_to_m_inplace(tumor_beta, args.eps)
        beta_to_m_inplace(normal_beta, args.eps)
        res = ttest_ind(
            tumor_beta,
            normal_beta,
            axis=1,
            equal_var=args.equal_var,
            nan_policy="omit",
        )
        stat = np.asarray(res.statistic, dtype=np.float32)
        p = np.asarray(res.pvalue, dtype=np.float64)
        bad = ~np.isfinite(p)
        if np.any(bad):
            p[bad] = 1.0
            stat[bad] = 0.0
        t_stat[start:end] = stat
        p_value[start:end] = p

        del block, tumor_beta, normal_beta, mt, mn, res, stat, p
        if start == 0 or end == n_probes or end % (args.chunk_probes * 5) == 0:
            log(f"  DMP chunk processed: {end:,}/{n_probes:,}")
        gc.collect()

    _, fdr, _, _ = multipletests(p_value, alpha=args.fdr_threshold, method="fdr_bh")
    fdr = np.asarray(fdr, dtype=np.float64)
    significant = (np.abs(delta_beta) > args.delta_beta_threshold) & (fdr < args.fdr_threshold)
    promoter_hyper = significant & promoter_mask & (delta_beta > args.delta_beta_threshold)

    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "fdr": fdr,
        "delta_beta": delta_beta,
        "mean_beta_tumor": mean_tumor,
        "mean_beta_normal": mean_normal,
        "significant": significant,
        "promoter_hyper": promoter_hyper,
    }


def alpha_1se_from_lassocv(lcv: LassoCV) -> float:
    mse = np.asarray(lcv.mse_path_, dtype=float)
    mean = mse.mean(axis=1)
    se = mse.std(axis=1, ddof=1) / np.sqrt(mse.shape[1])
    best = int(np.argmin(mean))
    eligible = mean <= mean[best] + se[best]
    return float(np.max(lcv.alphas_[eligible]))


def fit_lasso_coef(x_scaled: np.ndarray, y: np.ndarray, alpha: float, max_iter: int) -> np.ndarray:
    model = Lasso(alpha=alpha, max_iter=max_iter)
    model.fit(x_scaled, y.astype(float))
    return np.asarray(model.coef_, dtype=float)


def select_features_with_cap(
    x_train: np.ndarray,
    y_train: np.ndarray,
    args: argparse.Namespace,
) -> dict:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train).astype(np.float32, copy=False)
    inner_cv = StratifiedKFold(n_splits=args.inner_cv, shuffle=True, random_state=args.random_state)
    lcv = LassoCV(cv=inner_cv, random_state=args.random_state, n_jobs=args.n_jobs, max_iter=args.max_iter)
    lcv.fit(x_scaled, y_train.astype(float))
    coef_min = np.asarray(lcv.coef_, dtype=float)
    alpha_min = float(lcv.alpha_)
    alpha_1se = alpha_1se_from_lassocv(lcv)
    selected = np.flatnonzero(np.abs(coef_min) > args.coef_eps)
    coef_used = coef_min.copy()
    alpha_used = alpha_min
    strategy = "alpha_min"

    if len(selected) > args.max_features:
        coef_1se = fit_lasso_coef(x_scaled, y_train, alpha_1se, args.max_iter)
        selected_1se = np.flatnonzero(np.abs(coef_1se) > args.coef_eps)
        if 0 < len(selected_1se) <= args.max_features:
            selected = selected_1se
            coef_used = coef_1se
            alpha_used = alpha_1se
            strategy = "alpha_1se"
        else:
            alphas = np.asarray(lcv.alphas_, dtype=float)
            path_alphas, path_coefs, _ = lasso_path(x_scaled, y_train.astype(float), alphas=alphas, max_iter=args.max_iter)
            candidates = []
            for i, alpha in enumerate(path_alphas):
                idx = np.flatnonzero(np.abs(path_coefs[:, i]) > args.coef_eps)
                if 0 < len(idx) <= args.max_features:
                    candidates.append((args.max_features - len(idx), -len(idx), -float(alpha), float(alpha), idx, path_coefs[:, i]))
            if candidates:
                candidates.sort()
                _, _, _, alpha_used, selected, coef_used = candidates[0]
                strategy = "feature_cap_alpha_from_lasso_path"
            else:
                selected = np.argsort(np.abs(coef_min))[::-1][: args.max_features]
                coef_used = coef_min
                alpha_used = alpha_min
                strategy = "top_abs_coefficients_at_alpha_min"

    if len(selected) == 0:
        # Rare robust fallback: rank by absolute mean beta difference inside training fold.
        delta = np.abs(x_train[y_train == 1].mean(axis=0) - x_train[y_train == 0].mean(axis=0))
        selected = np.argsort(delta)[::-1][: args.max_features]
        coef_used = np.zeros(x_train.shape[1], dtype=float)
        alpha_used = alpha_min
        strategy = "fallback_top_delta_beta"

    selected = np.asarray(selected, dtype=np.int64)
    selected = selected[np.argsort(np.abs(coef_used[selected]))[::-1]]
    return {
        "selected_indices": selected,
        "coef": coef_used,
        "alpha_min": alpha_min,
        "alpha_1se": alpha_1se,
        "alpha_used": float(alpha_used),
        "strategy": strategy,
        "n_selected_at_alpha_min": int(np.sum(np.abs(coef_min) > args.coef_eps)),
        "n_selected_final": int(len(selected)),
        "cv_mse_min": float(np.min(np.asarray(lcv.mse_path_, dtype=float).mean(axis=1))),
    }


def evaluate_lr(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray, random_state: int) -> dict:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )
    pipe.fit(x_train, y_train)
    prob = pipe.predict_proba(x_val)[:, 1]
    pred = (prob >= 0.5).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_val, pred, labels=[0, 1]).ravel()
    return {
        "AUC": float(roc_auc_score(y_val, prob)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "accuracy": float(accuracy_score(y_val, pred)),
        "F1": float(f1_score(y_val, pred)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def main() -> None:
    args = parse_args()
    t0 = time.time()
    for path, label in [
        (args.beta_npy, "clean beta matrix"),
        (args.sample_groups, "sample groups"),
        (args.probe_metadata, "probe metadata"),
        (args.manifest_csv, "full HM450 manifest CSV"),
    ]:
        ensure(path, label)
    (args.outdir / "tables").mkdir(parents=True, exist_ok=True)

    log("Loading labels, probe metadata, and full clean-probe annotation")
    y, sample_cols, sample = load_labels(args.sample_groups)
    probe_meta = pd.read_parquet(args.probe_metadata).sort_values("matrix_row").reset_index(drop=True)
    ann = build_or_load_annotation(args)
    ann = ann.sort_values("matrix_row").reset_index(drop=True)
    if len(ann) != len(probe_meta):
        raise ValueError(f"Annotation length {len(ann)} != probe metadata length {len(probe_meta)}")
    promoter_mask = ann["is_promoter_TSS200_TSS1500"].fillna(False).to_numpy(bool)
    n_promoter_total = int(promoter_mask.sum())
    log(f"Samples: n={len(y):,}, Tumor={int(y.sum()):,}, Normal={int((y == 0).sum()):,}; clean probes={len(ann):,}; promoter-annotated={n_promoter_total:,}")

    beta = np.load(args.beta_npy, mmap_mode="r")
    orientation = detect_orientation(beta, len(ann), len(sample))
    log(f"Opened beta matrix shape={beta.shape}; orientation={orientation}; Welch equal_var={args.equal_var}")

    outer = StratifiedKFold(n_splits=args.outer_cv, shuffle=True, random_state=args.random_state)
    fold_metrics = []
    fold_counts = []
    selected_rows = []

    for fold, (train_idx, val_idx) in enumerate(outer.split(np.zeros(len(y)), y), start=1):
        fold_start = time.time()
        train_cols = sample_cols[train_idx]
        val_cols = sample_cols[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]
        log(
            f"Outer fold {fold}/{args.outer_cv}: train={len(train_idx)} "
            f"(T={int(y_train.sum())}, N={int((y_train == 0).sum())}), "
            f"val={len(val_idx)} (T={int(y_val.sum())}, N={int((y_val == 0).sum())})"
        )

        log(f"Fold {fold}: recomputing training-fold DMPs across all probes")
        dmp = train_fold_differential(
            beta=beta,
            orientation=orientation,
            n_probes=len(ann),
            train_cols=train_cols,
            train_y=y_train,
            promoter_mask=promoter_mask,
            args=args,
        )
        selected_global_rows = np.flatnonzero(dmp["promoter_hyper"])
        sig_total = int(dmp["significant"].sum())
        hyper_promoter = int(len(selected_global_rows))
        fold_counts.append(
            {
                "outer_fold": fold,
                "train_n": int(len(train_idx)),
                "validation_n": int(len(val_idx)),
                "train_tumor": int(y_train.sum()),
                "train_normal": int((y_train == 0).sum()),
                "validation_tumor": int(y_val.sum()),
                "validation_normal": int((y_val == 0).sum()),
                "training_significant_dmps_abs_delta_gt_0_2_fdr_lt_0_05": sig_total,
                "training_promoter_hypermethylated_dmps": hyper_promoter,
            }
        )
        log(f"Fold {fold}: training DMPs={sig_total:,}; promoter hyper DMPs={hyper_promoter:,}")
        if hyper_promoter == 0:
            raise RuntimeError(f"Fold {fold}: no promoter hypermethylated probes passed filtering.")

        x_train_pool = beta_rows_by_samples(beta, selected_global_rows, train_cols, orientation).T
        x_val_pool = beta_rows_by_samples(beta, selected_global_rows, val_cols, orientation).T
        log(f"Fold {fold}: running inner LassoCV on X_train={x_train_pool.shape}")
        fs = select_features_with_cap(x_train_pool, y_train, args)
        sel_local = fs["selected_indices"]
        sel_global = selected_global_rows[sel_local]
        x_train_sel = np.ascontiguousarray(x_train_pool[:, sel_local], dtype=np.float32)
        x_val_sel = np.ascontiguousarray(x_val_pool[:, sel_local], dtype=np.float32)

        metrics = evaluate_lr(x_train_sel, y_train, x_val_sel, y_val, args.random_state)
        metrics.update(
            {
                "outer_fold": fold,
                "train_n": int(len(train_idx)),
                "validation_n": int(len(val_idx)),
                "training_promoter_hypermethylated_dmps": hyper_promoter,
                "alpha_min": fs["alpha_min"],
                "alpha_1se": fs["alpha_1se"],
                "alpha_used": fs["alpha_used"],
                "feature_selection_strategy": fs["strategy"],
                "n_selected_at_alpha_min": fs["n_selected_at_alpha_min"],
                "n_selected_final": fs["n_selected_final"],
                "cv_mse_min": fs["cv_mse_min"],
                "runtime_seconds": round(time.time() - fold_start, 2),
            }
        )
        fold_metrics.append(metrics)

        for rank, (local_i, global_i) in enumerate(zip(sel_local, sel_global), start=1):
            selected_rows.append(
                {
                    "outer_fold": fold,
                    "rank": rank,
                    "matrix_row": int(global_i),
                    "probe_id": str(ann.loc[global_i, "probe_id"]),
                    "gene_name": ann.loc[global_i, "gene_name"],
                    "primary_gene": ann.loc[global_i, "primary_gene"],
                    "gene_relation": ann.loc[global_i, "gene_relation"],
                    "chromosome": ann.loc[global_i, "chromosome"] if "chromosome" in ann.columns else None,
                    "lasso_coefficient": float(fs["coef"][local_i]),
                    "t_statistic_train_fold": float(dmp["t_stat"][global_i]),
                    "p_value_train_fold": float(dmp["p_value"][global_i]),
                    "FDR_train_fold": float(dmp["fdr"][global_i]),
                    "delta_beta_train_fold": float(dmp["delta_beta"][global_i]),
                    "mean_beta_tumor_train_fold": float(dmp["mean_beta_tumor"][global_i]),
                    "mean_beta_normal_train_fold": float(dmp["mean_beta_normal"][global_i]),
                }
            )

        log(
            f"Fold {fold}: AUC={metrics['AUC']:.4f}, sens={metrics['sensitivity']:.4f}, "
            f"spec={metrics['specificity']:.4f}, selected={metrics['n_selected_final']}, "
            f"strategy={metrics['feature_selection_strategy']}"
        )

        del dmp, selected_global_rows, x_train_pool, x_val_pool, x_train_sel, x_val_sel
        gc.collect()

    metrics_df = pd.DataFrame(fold_metrics)
    counts_df = pd.DataFrame(fold_counts)
    selected_df = pd.DataFrame(selected_rows)
    metrics_path = args.outdir / "tables" / "fully_nested_outer_fold_metrics.parquet"
    counts_path = args.outdir / "tables" / "fully_nested_training_fold_dmp_counts.parquet"
    selected_path = args.outdir / "tables" / "fully_nested_selected_features_by_fold.parquet"
    metrics_df.to_parquet(metrics_path, index=False)
    counts_df.to_parquet(counts_path, index=False)
    selected_df.to_parquet(selected_path, index=False)

    summary = {
        "AUC_mean": float(metrics_df["AUC"].mean()),
        "AUC_std": float(metrics_df["AUC"].std(ddof=1)),
        "sensitivity_mean": float(metrics_df["sensitivity"].mean()),
        "sensitivity_std": float(metrics_df["sensitivity"].std(ddof=1)),
        "specificity_mean": float(metrics_df["specificity"].mean()),
        "specificity_std": float(metrics_df["specificity"].std(ddof=1)),
        "accuracy_mean": float(metrics_df["accuracy"].mean()),
        "F1_mean": float(metrics_df["F1"].mean()),
        "selected_features_mean": float(metrics_df["n_selected_final"].mean()),
        "training_promoter_hyper_dmps_mean": float(counts_df["training_promoter_hypermethylated_dmps"].mean()),
    }
    report = {
        "note": "Fully nested evaluation: DMP filtering and LassoCV feature selection are recomputed inside each outer training fold.",
        "inputs": {
            "beta_npy": str(args.beta_npy),
            "sample_groups": str(args.sample_groups),
            "probe_metadata": str(args.probe_metadata),
            "manifest_csv": str(args.manifest_csv),
            "annotation_cache": str(args.annotation_cache),
        },
        "settings": {
            "outer_cv": args.outer_cv,
            "inner_cv": args.inner_cv,
            "random_state": args.random_state,
            "max_features": args.max_features,
            "chunk_probes": args.chunk_probes,
            "delta_beta_threshold": args.delta_beta_threshold,
            "fdr_threshold": args.fdr_threshold,
            "ttest_scale": "M-value log2(beta/(1-beta)); delta_beta on beta",
            "equal_var": bool(args.equal_var),
        },
        "counts": {
            "samples": int(len(y)),
            "tumor": int(y.sum()),
            "normal": int((y == 0).sum()),
            "clean_probes": int(len(ann)),
            "manifest_promoter_TSS200_TSS1500_probes": n_promoter_total,
        },
        "summary": summary,
        "outputs": {
            "outer_metrics": str(metrics_path),
            "training_fold_dmp_counts": str(counts_path),
            "selected_features_by_fold": str(selected_path),
        },
        "runtime_seconds": round(time.time() - t0, 2),
    }
    report_path = args.outdir / "fully_nested_cv_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Saved metrics: {metrics_path}")
    log(f"Saved selected features: {selected_path}")
    log(
        f"Fully nested summary: AUC={summary['AUC_mean']:.4f}±{summary['AUC_std']:.4f}, "
        f"sens={summary['sensitivity_mean']:.4f}, spec={summary['specificity_mean']:.4f}"
    )
    log(f"All done in {(time.time() - t0) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
