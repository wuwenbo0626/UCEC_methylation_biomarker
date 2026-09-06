#!/usr/bin/env python3
"""
Nested cross-validation for TCGA-UCEC promoter-hypermethylated methylation probes.

Goal:
  - Use the cleaned beta matrix and promoter hypermethylated DMP feature pool.
  - Outer 5-fold stratified CV.
  - Inner LassoCV on each outer training fold only.
  - Select at most 20 features per outer fold.
  - Train logistic regression and evaluate held-out outer fold.
  - Additionally, perform full-data LassoCV to select 10 features, then evaluate
    logistic regression using LOOCV, with special attention to Normal samples.

Important:
  This removes leakage from LASSO feature selection into the outer validation
  folds. However, the upstream DMP/promoter-hypermethylated feature pool is
  still fixed from the previous full-data differential analysis. A maximally
  unbiased pipeline would also redo differential filtering inside each outer
  training fold.

Outputs save scalar CV summaries and selected feature lists. Full CV prediction
vectors are not saved.
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
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, lasso_path
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_ROOT = Path("/Users/wuwenbo/AsiaInfo/tcga_ucec_methylation")
DEFAULT_CLEAN = DEFAULT_ROOT / "tcga_ucec_hm450_cleaned"
DEFAULT_ANNOT = DEFAULT_ROOT / "tcga_ucec_hm450_annotation"
DEFAULT_OUT = DEFAULT_ROOT / "tcga_ucec_hm450_nested_cv"

os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_ROOT / ".matplotlib_cache"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nested CV LASSO+LR for promoter hypermethylated probes.")
    p.add_argument("--beta-npy", type=Path, default=DEFAULT_CLEAN / "matrix" / "ucec_hm450_beta_clean_probes_by_samples.npy")
    p.add_argument("--sample-groups", type=Path, default=DEFAULT_CLEAN / "metadata" / "ucec_hm450_clean_sample_groups.parquet")
    p.add_argument("--probe-metadata", type=Path, default=DEFAULT_CLEAN / "metadata" / "ucec_hm450_clean_probe_metadata.parquet")
    p.add_argument("--promoter-dmp", type=Path, default=DEFAULT_ANNOT / "tables" / "ucec_hm450_dmp_promoter_TSS200_TSS1500.parquet")
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--outer-cv", type=int, default=5)
    p.add_argument("--inner-cv", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--max-iter", type=int, default=10000)
    p.add_argument("--outer-max-features", type=int, default=20)
    p.add_argument("--full-max-features", type=int, default=10)
    p.add_argument("--delta-beta-threshold", type=float, default=0.20)
    p.add_argument("--fdr-threshold", type=float, default=0.05)
    p.add_argument("--coef-eps", type=float, default=1e-8)
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


def save_json(obj: dict, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_labels(sample_groups_path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    sample = pd.read_parquet(sample_groups_path)
    if "matrix_col" not in sample.columns:
        sample = sample.copy()
        sample["matrix_col"] = np.arange(len(sample), dtype=np.int64)
    group = sample["group"].astype(str).str.lower()
    keep = group.isin(["tumor", "primary tumor", "1", "normal", "solid tissue normal", "0"])
    sample = sample.loc[keep].copy().reset_index(drop=True)
    group = sample["group"].astype(str).str.lower()
    y = group.isin(["tumor", "primary tumor", "1"]).astype(np.int8).to_numpy()
    cols = sample["matrix_col"].to_numpy(np.int64)
    return y, cols, sample


def primary_gene(value) -> str:
    if value is None or pd.isna(value):
        return "NA"
    for part in str(value).replace(",", ";").split(";"):
        part = part.strip()
        if part:
            return part.upper()
    return "NA"


def load_feature_pool(args: argparse.Namespace) -> pd.DataFrame:
    dmp = pd.read_parquet(args.promoter_dmp)
    required = {"probe_id", "delta_beta", "FDR"}
    missing = required - set(dmp.columns)
    if missing:
        raise ValueError(f"promoter DMP table missing columns: {sorted(missing)}")
    pool = dmp[
        (dmp["delta_beta"] > args.delta_beta_threshold)
        & (dmp["FDR"] < args.fdr_threshold)
    ].drop_duplicates("probe_id", keep="first").copy()
    if "gene_name" in pool.columns:
        pool["primary_gene"] = pool["gene_name"].map(primary_gene)
    else:
        pool["primary_gene"] = "NA"
    pool["probe_id"] = pool["probe_id"].astype(str)
    if "matrix_row" not in pool.columns:
        probe_meta = pd.read_parquet(args.probe_metadata)
        if "matrix_row" not in probe_meta.columns:
            probe_meta = probe_meta.copy()
            probe_meta["matrix_row"] = np.arange(len(probe_meta), dtype=np.int64)
        probe_meta["probe_id"] = probe_meta["probe_id"].astype(str)
        pool = pool.merge(probe_meta[["probe_id", "matrix_row"]], on="probe_id", how="inner", validate="one_to_one")
    pool = pool.sort_values("matrix_row").reset_index(drop=True)
    pool["feature_index"] = np.arange(len(pool), dtype=np.int64)
    return pool


def extract_matrix(beta_path: Path, feature_pool: pd.DataFrame, sample_cols: np.ndarray) -> np.ndarray:
    beta = np.load(beta_path, mmap_mode="r")
    rows = feature_pool["matrix_row"].to_numpy(np.int64)
    log(f"Extracting promoter hypermethylated feature matrix: samples={len(sample_cols):,}, probes={len(rows):,}")
    if beta.shape[0] > rows.max() and beta.shape[1] > sample_cols.max():
        x = np.asarray(beta[np.ix_(rows, sample_cols)].T, dtype=np.float32)
    elif beta.shape[1] > rows.max() and beta.shape[0] > sample_cols.max():
        x = np.asarray(beta[np.ix_(sample_cols, rows)], dtype=np.float32)
    else:
        raise ValueError(f"Cannot infer beta orientation from shape={beta.shape}")
    del beta
    gc.collect()
    if np.isnan(x).any():
        raise ValueError("Feature matrix contains NaN; please use imputed matrix.")
    return x


def alpha_1se_from_lassocv(lcv: LassoCV) -> float:
    mse = np.asarray(lcv.mse_path_, dtype=float)
    mean = mse.mean(axis=1)
    se = mse.std(axis=1, ddof=1) / np.sqrt(mse.shape[1])
    min_i = int(np.argmin(mean))
    eligible = mean <= mean[min_i] + se[min_i]
    return float(np.max(lcv.alphas_[eligible]))


def fit_lasso_coefficients(x_scaled: np.ndarray, y: np.ndarray, alpha: float, max_iter: int) -> np.ndarray:
    return np.asarray(Lasso(alpha=alpha, max_iter=max_iter).fit(x_scaled, y.astype(float)).coef_, dtype=float)


def select_features_with_cap(
    x_train: np.ndarray,
    y_train: np.ndarray,
    max_features: int,
    inner_cv: int,
    random_state: int,
    n_jobs: int,
    max_iter: int,
    coef_eps: float,
) -> dict:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train).astype(np.float32, copy=False)
    cv = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=random_state)
    lcv = LassoCV(cv=cv, random_state=random_state, max_iter=max_iter, n_jobs=n_jobs).fit(x_scaled, y_train.astype(float))
    alpha_min = float(lcv.alpha_)
    coef_min = np.asarray(lcv.coef_, dtype=float)
    selected = np.flatnonzero(np.abs(coef_min) > coef_eps)
    alpha_used = alpha_min
    coef_used = coef_min
    strategy = "alpha_min"
    alpha_1se = alpha_1se_from_lassocv(lcv)

    if len(selected) > max_features:
        coef_1se = fit_lasso_coefficients(x_scaled, y_train, alpha_1se, max_iter)
        selected_1se = np.flatnonzero(np.abs(coef_1se) > coef_eps)
        if 0 < len(selected_1se) <= max_features:
            selected = selected_1se
            alpha_used = alpha_1se
            coef_used = coef_1se
            strategy = "alpha_1se"
        else:
            alphas = np.asarray(lcv.alphas_, dtype=float)
            mean_mse = np.asarray(lcv.mse_path_, dtype=float).mean(axis=1)
            best_i = int(np.argmin(mean_mse))
            candidates = []
            # Use path coefficients to avoid fitting 100 separate Lasso models.
            path_alphas, path_coefs, _ = lasso_path(x_scaled, y_train.astype(float), alphas=alphas, max_iter=max_iter)
            for i, alpha in enumerate(path_alphas):
                idx = np.flatnonzero(np.abs(path_coefs[:, i]) > coef_eps)
                if 0 < len(idx) <= max_features:
                    candidates.append((abs(i - best_i), mean_mse[i], float(alpha), idx, path_coefs[:, i]))
            if candidates:
                candidates.sort(key=lambda z: (z[0], z[1], -z[2]))
                _, _, alpha_used, selected, coef_used = candidates[0]
                strategy = "feature_cap_alpha_from_lasso_path"
            else:
                # Last-resort: top coefficients at alpha_min, still selected only from outer training data.
                top = np.argsort(np.abs(coef_min))[::-1][:max_features]
                selected = top[np.abs(coef_min[top]) > coef_eps]
                alpha_used = alpha_min
                coef_used = coef_min
                strategy = "top_abs_coefficients_at_alpha_min"

    if len(selected) == 0:
        # Robust fallback when LASSO selects no features in a small fold.
        corr = np.abs(np.corrcoef(x_train, y_train, rowvar=False)[-1, :-1])
        selected = np.argsort(np.nan_to_num(corr, nan=0.0))[::-1][:max_features]
        coef_used = np.zeros(x_train.shape[1], dtype=float)
        strategy = "fallback_top_correlated"

    return {
        "selected_indices": np.asarray(selected, dtype=np.int64),
        "coef": np.asarray(coef_used, dtype=float),
        "alpha_min": alpha_min,
        "alpha_1se": alpha_1se,
        "alpha_used": float(alpha_used),
        "strategy": strategy,
        "n_selected_at_alpha_min": int(np.sum(np.abs(coef_min) > coef_eps)),
        "n_selected_final": int(len(selected)),
        "cv_mse_min": float(np.min(np.asarray(lcv.mse_path_, dtype=float).mean(axis=1))),
    }


def logistic_model() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def metrics_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_score >= threshold).astype(np.int8)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "AUC": float(roc_auc_score(y_true, y_score)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def run_outer_nested_cv(x: np.ndarray, y: np.ndarray, features: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    outer = StratifiedKFold(n_splits=args.outer_cv, shuffle=True, random_state=args.random_state)
    fold_rows = []
    selected_rows = []
    for fold, (train_idx, val_idx) in enumerate(outer.split(x, y), start=1):
        log(f"Outer fold {fold}/{args.outer_cv}: train={len(train_idx)}, val={len(val_idx)}")
        sel = select_features_with_cap(
            x[train_idx],
            y[train_idx],
            args.outer_max_features,
            args.inner_cv,
            args.random_state + fold,
            args.n_jobs,
            args.max_iter,
            args.coef_eps,
        )
        selected = sel["selected_indices"]
        model = logistic_model().fit(x[train_idx][:, selected], y[train_idx])
        y_score = model.predict_proba(x[val_idx][:, selected])[:, 1]
        m = metrics_at_threshold(y[val_idx], y_score)
        row = {
            "outer_fold": fold,
            "train_n": int(len(train_idx)),
            "validation_n": int(len(val_idx)),
            "train_tumor": int(y[train_idx].sum()),
            "train_normal": int((y[train_idx] == 0).sum()),
            "validation_tumor": int(y[val_idx].sum()),
            "validation_normal": int((y[val_idx] == 0).sum()),
            **m,
            "alpha_min": sel["alpha_min"],
            "alpha_1se": sel["alpha_1se"],
            "alpha_used": sel["alpha_used"],
            "feature_selection_strategy": sel["strategy"],
            "n_selected_at_alpha_min": sel["n_selected_at_alpha_min"],
            "n_selected_final": sel["n_selected_final"],
            "cv_mse_min": sel["cv_mse_min"],
        }
        fold_rows.append(row)
        for rank, local_idx in enumerate(selected, start=1):
            f = features.iloc[int(local_idx)]
            selected_rows.append({
                "outer_fold": fold,
                "feature_rank_in_fold": rank,
                "feature_index": int(local_idx),
                "probe_id": f["probe_id"],
                "gene_name": f.get("gene_name", pd.NA),
                "primary_gene": f.get("primary_gene", pd.NA),
                "delta_beta_full_dataset_prefilter": f.get("delta_beta", np.nan),
                "FDR_full_dataset_prefilter": f.get("FDR", np.nan),
                "lasso_coefficient_in_outer_train": float(sel["coef"][local_idx]),
                "alpha_used": sel["alpha_used"],
                "selection_strategy": sel["strategy"],
            })
        log(
            f"Fold {fold}: AUC={m['AUC']:.4f}, sens={m['sensitivity']:.4f}, "
            f"spec={m['specificity']:.4f}, selected={len(selected)}, strategy={sel['strategy']}"
        )
        del model, y_score
        gc.collect()
    return pd.DataFrame(fold_rows), pd.DataFrame(selected_rows)


def summarize_nested(folds: pd.DataFrame) -> dict:
    return {
        "AUC_mean": float(folds["AUC"].mean()),
        "AUC_std": float(folds["AUC"].std(ddof=1)),
        "sensitivity_mean": float(folds["sensitivity"].mean()),
        "sensitivity_std": float(folds["sensitivity"].std(ddof=1)),
        "specificity_mean": float(folds["specificity"].mean()),
        "specificity_std": float(folds["specificity"].std(ddof=1)),
        "accuracy_mean": float(folds["accuracy"].mean()),
        "F1_mean": float(folds["F1"].mean()),
        "selected_features_mean": float(folds["n_selected_final"].mean()),
        "selected_features_min": int(folds["n_selected_final"].min()),
        "selected_features_max": int(folds["n_selected_final"].max()),
    }


def run_full_lasso_loocv(x: np.ndarray, y: np.ndarray, samples: pd.DataFrame, features: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    log("Full-data LassoCV feature selection for fixed 10-feature LOOCV panel")
    sel = select_features_with_cap(
        x,
        y,
        args.full_max_features,
        args.inner_cv,
        args.random_state,
        args.n_jobs,
        args.max_iter,
        args.coef_eps,
    )
    selected = sel["selected_indices"]
    selected_rows = []
    for rank, local_idx in enumerate(selected, start=1):
        f = features.iloc[int(local_idx)]
        selected_rows.append({
            "rank": rank,
            "feature_index": int(local_idx),
            "probe_id": f["probe_id"],
            "gene_name": f.get("gene_name", pd.NA),
            "primary_gene": f.get("primary_gene", pd.NA),
            "delta_beta_full_dataset_prefilter": f.get("delta_beta", np.nan),
            "FDR_full_dataset_prefilter": f.get("FDR", np.nan),
            "lasso_coefficient_full_data": float(sel["coef"][local_idx]),
            "alpha_used": sel["alpha_used"],
            "selection_strategy": sel["strategy"],
        })
    selected_df = pd.DataFrame(selected_rows)

    loo = LeaveOneOut()
    scores = np.empty(len(y), dtype=np.float64)
    preds = np.empty(len(y), dtype=np.int8)
    log(f"Running LOOCV logistic regression on fixed {len(selected)} selected features")
    for i, (train_idx, test_idx) in enumerate(loo.split(x), start=1):
        model = logistic_model().fit(x[train_idx][:, selected], y[train_idx])
        score = float(model.predict_proba(x[test_idx][:, selected])[:, 1][0])
        scores[test_idx[0]] = score
        preds[test_idx[0]] = 1 if score >= 0.5 else 0
        if i == 1 or i % 100 == 0 or i == len(y):
            log(f"LOOCV processed {i:,}/{len(y):,}")
    m = metrics_at_threshold(y, scores)
    normal_mask = y == 0
    tumor_mask = y == 1
    normal_scores = scores[normal_mask]
    false_positive_mask = normal_mask & (preds == 1)
    false_positive_samples = samples.loc[false_positive_mask, [c for c in ["matrix_col", "sample_id", "aliquot_id", "patient_id", "sample_type", "group"] if c in samples.columns]].copy()
    # This is intentionally not a full prediction vector; only misclassified normal samples are saved for audit.
    normal_summary = {
        "normal_total": int(normal_mask.sum()),
        "normal_true_negative": int(np.sum(normal_mask & (preds == 0))),
        "normal_false_positive": int(false_positive_mask.sum()),
        "normal_specificity": float(np.mean(preds[normal_mask] == 0)),
        "normal_predicted_probability_mean": float(normal_scores.mean()),
        "normal_predicted_probability_median": float(np.median(normal_scores)),
        "normal_predicted_probability_min": float(normal_scores.min()),
        "normal_predicted_probability_max": float(normal_scores.max()),
        "normal_predicted_probability_q95": float(np.quantile(normal_scores, 0.95)),
        "tumor_total": int(tumor_mask.sum()),
        "tumor_true_positive": int(np.sum(tumor_mask & (preds == 1))),
        "tumor_false_negative": int(np.sum(tumor_mask & (preds == 0))),
    }
    report = {
        "feature_selection": {
            "alpha_min": sel["alpha_min"],
            "alpha_1se": sel["alpha_1se"],
            "alpha_used": sel["alpha_used"],
            "strategy": sel["strategy"],
            "n_selected_at_alpha_min": sel["n_selected_at_alpha_min"],
            "n_selected_final": sel["n_selected_final"],
        },
        "metrics": m,
        "normal_prediction_summary": normal_summary,
    }
    del scores, preds
    gc.collect()
    return selected_df, false_positive_samples, report


def main() -> int:
    args = parse_args()
    t0 = time.time()
    for path, label in [
        (args.beta_npy, "beta matrix"),
        (args.sample_groups, "sample groups"),
        (args.probe_metadata, "probe metadata"),
        (args.promoter_dmp, "promoter DMP table"),
    ]:
        ensure(path, label)
    tables = args.outdir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    log("Loading labels and promoter hypermethylated feature pool")
    y, sample_cols, samples = load_labels(args.sample_groups)
    features = load_feature_pool(args)
    log(f"Samples: n={len(y)}, Tumor={int(y.sum())}, Normal={int((y == 0).sum())}")
    log(f"Promoter hypermethylated feature pool: {len(features):,} probes")
    x = extract_matrix(args.beta_npy, features, sample_cols)
    log(f"Feature matrix loaded: X.shape={x.shape}, dtype={x.dtype}")

    folds, fold_selected = run_outer_nested_cv(x, y, features, args)
    nested_summary = summarize_nested(folds)
    folds_path = tables / "nested_5fold_outer_metrics.parquet"
    fold_selected_path = tables / "nested_5fold_selected_features_by_fold.parquet"
    folds.to_parquet(folds_path, index=False)
    fold_selected.to_parquet(fold_selected_path, index=False)
    log(f"Saved nested fold metrics: {folds_path}")
    log(f"Saved nested selected features: {fold_selected_path}")
    log(
        f"Nested CV summary: AUC={nested_summary['AUC_mean']:.4f}±{nested_summary['AUC_std']:.4f}, "
        f"sens={nested_summary['sensitivity_mean']:.4f}, spec={nested_summary['specificity_mean']:.4f}"
    )

    full_selected, normal_fp, loocv_report = run_full_lasso_loocv(x, y, samples, features, args)
    full_selected_path = tables / "full_data_lassocv_selected_10_features.parquet"
    normal_fp_path = tables / "loocv_false_positive_normal_samples.parquet"
    full_selected.to_parquet(full_selected_path, index=False)
    normal_fp.to_parquet(normal_fp_path, index=False)
    log(f"Saved full-data selected LOOCV panel: {full_selected_path}")
    log(f"Saved LOOCV false-positive normal sample audit table: {normal_fp_path}")

    report = {
        "note": (
            "Nested CV removes leakage from LASSO feature selection into outer validation folds. "
            "The promoter hypermethylated DMP feature pool is fixed from full-data differential analysis; "
            "for a fully unbiased pipeline, redo DMP filtering inside each outer fold."
        ),
        "inputs": {
            "beta_npy": str(args.beta_npy),
            "sample_groups": str(args.sample_groups),
            "probe_metadata": str(args.probe_metadata),
            "promoter_dmp": str(args.promoter_dmp),
        },
        "counts": {
            "samples": int(len(y)),
            "tumor": int(y.sum()),
            "normal": int((y == 0).sum()),
            "feature_pool_promoter_hypermethylated_probes": int(len(features)),
        },
        "nested_5fold": nested_summary,
        "loocv_fixed_full_data_lasso_10_features": loocv_report,
        "outputs": {
            "nested_outer_metrics_parquet": str(folds_path),
            "nested_selected_features_by_fold_parquet": str(fold_selected_path),
            "full_data_lassocv_selected_10_features_parquet": str(full_selected_path),
            "loocv_false_positive_normal_samples_parquet": str(normal_fp_path),
        },
        "runtime_seconds": round(time.time() - t0, 2),
    }
    report_path = args.outdir / "nested_cv_lasso_lr_report.json"
    save_json(report, report_path)
    log(f"Saved report: {report_path}")
    log(f"All done in {(time.time() - t0) / 60:.2f} minutes")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"ERROR: {exc}")
        raise
