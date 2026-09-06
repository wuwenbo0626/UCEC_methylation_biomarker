#!/usr/bin/env python3
"""
Literature-constrained + data-driven 10-gene TCGA-UCEC methylation panel.

Panel design:
  - Force in available literature markers from:
    CDO1, PAX1, BHLHE22, HAND2, TBX5, ZNF454
    using the promoter-hypermethylated probe with the largest beta variance.
  - From the remaining promoter-hypermethylated candidate pool, exclude
    pseudogene/unknown-function-like symbols and previously undesirable genes.
  - Use full-data LassoCV only to rank/screen the remaining data-driven genes,
    then add enough genes to make a fixed 10-gene panel.
  - Require selected genes to have delta_beta > 0.3.
  - Evaluate the fixed literature-constrained panel, the previous pure
    data-driven 10-gene panel, and the PAX1/JAM3 baseline using the same
    stratified 5-fold CV logistic-regression workflow.

Important:
  LassoCV here is explicitly exploratory panel construction, not performance
  estimation. The 5-fold CV compares fixed panels and should be interpreted as
  internal descriptive validation on TCGA tissue samples.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("/Users/wuwenbo/AsiaInfo/tcga_ucec_methylation")
CLEAN = ROOT / "tcga_ucec_hm450_cleaned"
CAND = ROOT / "tcga_ucec_hm450_candidate_genes"
RF = ROOT / "tcga_ucec_hm450_random_forest"
BASELINE = ROOT / "tcga_ucec_hm450_baseline_comparison"
OUT = ROOT / "tcga_ucec_hm450_literature_constrained_panel"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build and evaluate literature-constrained UCEC methylation panel.")
    p.add_argument("--beta-npy", type=Path, default=CLEAN / "matrix" / "ucec_hm450_beta_clean_probes_by_samples.npy")
    p.add_argument("--sample-groups", type=Path, default=CLEAN / "metadata" / "ucec_hm450_clean_sample_groups.parquet")
    p.add_argument("--promoter-gene-pairs", type=Path, default=CAND / "tables" / "ucec_promoter_hypermethylated_probe_gene_pairs.parquet")
    p.add_argument("--candidate-dedup", type=Path, default=CAND / "tables" / "ucec_promoter_hypermethylated_candidate_genes_dedup.parquet")
    p.add_argument("--pure-data-driven-panel", type=Path, default=RF / "tables" / "final_10_methylation_markers.parquet")
    p.add_argument("--pax1-jam3-panel", type=Path, default=BASELINE / "tables" / "pax1_jam3_selected_baseline_probes.parquet")
    p.add_argument("--outdir", type=Path, default=OUT)
    p.add_argument("--known-genes", nargs="+", default=["CDO1", "PAX1", "BHLHE22", "HAND2", "TBX5", "ZNF454"])
    p.add_argument("--exclude-genes", nargs="+", default=["CD8A", "HIST1H4F", "PCDHB19P", "HIST1H2AL", "PCDHGA11", "PCDHGA12"])
    p.add_argument("--delta-beta-min", type=float, default=0.30)
    p.add_argument("--max-panel-genes", type=int, default=10)
    p.add_argument("--lasso-cv", type=int, default=10)
    p.add_argument("--eval-cv", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=4)
    p.add_argument("--max-iter", type=int, default=10000)
    p.add_argument("--coef-eps", type=float, default=1e-8)
    return p.parse_args()


def mem() -> str:
    proc = psutil.Process(os.getpid())
    rss = proc.memory_info().rss / 1024**3
    avail = psutil.virtual_memory().available / 1024**3
    return f"RSS={rss:.2f} GB, available={avail:.2f} GB"


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg} | {mem()}", flush=True)


def ensure(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def gene_clean(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).split(";")[0].strip().upper()


def normalize_gene_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "gene_symbol" in df.columns:
        df["primary_gene"] = df["gene_symbol"].map(gene_clean)
    elif "primary_gene" in df.columns:
        df["primary_gene"] = df["primary_gene"].map(gene_clean)
    elif "gene_name" in df.columns:
        df["primary_gene"] = df["gene_name"].map(gene_clean)
    else:
        raise ValueError("Cannot find gene column in panel table.")
    return df


def beta_rows_by_samples(beta: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    if beta.shape[0] > rows.max() and beta.shape[1] > cols.max():
        return np.asarray(beta[np.ix_(rows, cols)], dtype=np.float32)
    if beta.shape[1] > rows.max() and beta.shape[0] > cols.max():
        return np.asarray(beta[np.ix_(cols, rows)].T, dtype=np.float32)
    raise ValueError(f"Cannot infer beta orientation from shape={beta.shape}")


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


def exclusion_reason(gene: str, explicit_excludes: set[str], known_genes: set[str]) -> str:
    g = gene_clean(gene)
    if not g:
        return "empty_gene_symbol"
    if g in known_genes:
        return ""
    if g in explicit_excludes:
        return "explicit_user_exclusion_or_previous_low_interpretability"
    if g.startswith("LOC"):
        return "LOC_uncharacterized"
    if re.fullmatch(r"C\\d+ORF\\d+", g):
        return "chromosome_open_reading_frame_unknown"
    if g.startswith(("MIR", "LINC")):
        return "noncoding_or_miRNA_symbol"
    if g.startswith(("HIST", "H1-", "H2A", "H2B", "H3-", "H4-")):
        return "histone_cluster_or_histone_like"
    if g.startswith("PCDH") or g.startswith("PROTOCADHERIN"):
        return "protocadherin_cluster_or_pseudogene_like"
    if g.endswith("P") and not g.startswith(("PAX", "PGR", "PITX", "PRDM", "PTPR", "PDGFR", "POU", "PLX")):
        return "pseudogene_like_suffix_P"
    if g.startswith(("RP11-", "RP13-", "AC", "AL")):
        return "clone_or_locus_style_symbol"
    return ""


def select_variance_max_by_gene(beta: np.ndarray, sample_cols: np.ndarray, rows: pd.DataFrame, genes: list[str]) -> pd.DataFrame:
    rows = rows.copy()
    rows["primary_gene"] = rows["gene_symbol"].map(gene_clean)
    selected = []
    for gene in genes:
        sub = rows[(rows["primary_gene"] == gene) & (rows["delta_beta"] > 0.30)].drop_duplicates("probe_id").copy()
        if sub.empty:
            continue
        vals = beta_rows_by_samples(beta, sub["matrix_row"].to_numpy(np.int64), sample_cols)
        vars_ = np.nanvar(vals, axis=1)
        best_i = int(np.nanargmax(vars_))
        best = sub.iloc[best_i].copy()
        best["beta_variance"] = float(vars_[best_i])
        best["panel_source"] = "literature_constrained"
        selected.append(best)
        del vals
        gc.collect()
    if not selected:
        return pd.DataFrame()
    out = pd.DataFrame(selected)
    out["primary_gene"] = out["gene_symbol"].map(gene_clean)
    return out


def build_data_driven_pool(args: argparse.Namespace, forced: pd.DataFrame, beta: np.ndarray, sample_cols: np.ndarray) -> pd.DataFrame:
    known = {gene_clean(g) for g in args.known_genes}
    explicit = {gene_clean(g) for g in args.exclude_genes}
    dedup = pd.read_parquet(args.candidate_dedup)
    dedup = normalize_gene_column(dedup)
    dedup["probe_id"] = dedup["representative_probe_id"].astype(str)
    dedup = dedup[dedup["delta_beta"] > args.delta_beta_min].copy()
    dedup["exclusion_reason"] = dedup["primary_gene"].map(lambda g: exclusion_reason(g, explicit, known))
    forced_genes = set(forced["primary_gene"]) if not forced.empty else set()
    forced_probes = set(forced["probe_id"]) if not forced.empty else set()
    excluded = dedup[(dedup["exclusion_reason"] != "") | dedup["primary_gene"].isin(forced_genes) | dedup["probe_id"].isin(forced_probes)].copy()
    included = dedup[(dedup["exclusion_reason"] == "") & (~dedup["primary_gene"].isin(forced_genes)) & (~dedup["probe_id"].isin(forced_probes))].copy()
    included = included.drop_duplicates("primary_gene", keep="first")
    included = included.drop_duplicates("probe_id", keep="first").reset_index(drop=True)
    vals = beta_rows_by_samples(beta, included["matrix_row"].to_numpy(np.int64), sample_cols)
    included["beta_variance"] = np.nanvar(vals, axis=1).astype(float)
    del vals
    gc.collect()
    included["panel_source"] = "data_driven_candidate_pool"
    return included, excluded


def extract_x(beta: np.ndarray, rows: np.ndarray, sample_cols: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(beta_rows_by_samples(beta, rows, sample_cols).T, dtype=np.float32)


def fit_lasso_coef(x_scaled: np.ndarray, y: np.ndarray, alpha: float, max_iter: int) -> np.ndarray:
    model = Lasso(alpha=alpha, max_iter=max_iter)
    model.fit(x_scaled, y.astype(float))
    return np.asarray(model.coef_, dtype=float)


def lasso_rank_pool(x: np.ndarray, y: np.ndarray, args: argparse.Namespace) -> pd.DataFrame:
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x).astype(np.float32, copy=False)
    cv = StratifiedKFold(n_splits=args.lasso_cv, shuffle=True, random_state=args.random_state)
    lcv = LassoCV(cv=cv, random_state=args.random_state, n_jobs=args.n_jobs, max_iter=args.max_iter)
    lcv.fit(x_scaled, y.astype(float))
    coef_min = np.asarray(lcv.coef_, dtype=float)
    # Fit 1-SE model for a slightly more stable ranking column too.
    mse = np.asarray(lcv.mse_path_, dtype=float)
    mean = mse.mean(axis=1)
    se = mse.std(axis=1, ddof=1) / np.sqrt(mse.shape[1])
    best = int(np.argmin(mean))
    alpha_1se = float(np.max(lcv.alphas_[mean <= mean[best] + se[best]]))
    coef_1se = fit_lasso_coef(x_scaled, y, alpha_1se, args.max_iter)
    rank = pd.DataFrame(
        {
            "feature_index": np.arange(x.shape[1], dtype=np.int64),
            "lasso_coef_alpha_min": coef_min,
            "lasso_coef_alpha_1se": coef_1se,
            "abs_lasso_coef_alpha_min": np.abs(coef_min),
            "abs_lasso_coef_alpha_1se": np.abs(coef_1se),
            "lasso_alpha_min": float(lcv.alpha_),
            "lasso_alpha_1se": alpha_1se,
            "lasso_cv_mse_min": float(mean[best]),
        }
    )
    rank = rank.sort_values(["abs_lasso_coef_alpha_min", "abs_lasso_coef_alpha_1se"], ascending=False).reset_index(drop=True)
    return rank


def scorer_sens(estimator, x, y) -> float:
    pred = estimator.predict(x)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return tp / (tp + fn) if tp + fn else np.nan


def scorer_spec(estimator, x, y) -> float:
    pred = estimator.predict(x)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return tn / (tn + fp) if tn + fp else np.nan


def evaluate_panel(panel_name: str, x: np.ndarray, y: np.ndarray, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    cv = StratifiedKFold(n_splits=args.eval_cv, shuffle=True, random_state=args.random_state)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, random_state=args.random_state)),
        ]
    )
    scores = cross_validate(
        pipe,
        x,
        y,
        cv=cv,
        scoring={
            "auc": "roc_auc",
            "sensitivity": scorer_sens,
            "specificity": scorer_spec,
            "accuracy": "accuracy",
            "f1": "f1",
        },
        return_train_score=False,
    )
    folds = pd.DataFrame(
        {
            "panel": panel_name,
            "fold": np.arange(1, args.eval_cv + 1),
            "AUC": scores["test_auc"],
            "sensitivity": scores["test_sensitivity"],
            "specificity": scores["test_specificity"],
            "accuracy": scores["test_accuracy"],
            "F1": scores["test_f1"],
        }
    )
    summary = {
        "panel": panel_name,
        "n_features": int(x.shape[1]),
        "AUC_mean": float(folds["AUC"].mean()),
        "AUC_std": float(folds["AUC"].std(ddof=1)),
        "sensitivity_mean": float(folds["sensitivity"].mean()),
        "sensitivity_std": float(folds["sensitivity"].std(ddof=1)),
        "specificity_mean": float(folds["specificity"].mean()),
        "specificity_std": float(folds["specificity"].std(ddof=1)),
        "accuracy_mean": float(folds["accuracy"].mean()),
        "F1_mean": float(folds["F1"].mean()),
    }
    return folds, summary


def standardize_panel_table(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = normalize_gene_column(df)
    if "representative_probe_id" in df.columns and "probe_id" not in df.columns:
        df["probe_id"] = df["representative_probe_id"].astype(str)
    if "target_gene" in df.columns:
        df["primary_gene"] = df["target_gene"].map(gene_clean)
    df = df.copy()
    df["panel_source"] = source
    return df


def main() -> None:
    args = parse_args()
    t0 = time.time()
    for path, label in [
        (args.beta_npy, "beta matrix"),
        (args.sample_groups, "sample groups"),
        (args.promoter_gene_pairs, "promoter gene pairs"),
        (args.candidate_dedup, "candidate dedup"),
        (args.pure_data_driven_panel, "pure data-driven panel"),
        (args.pax1_jam3_panel, "PAX1/JAM3 panel"),
    ]:
        ensure(path, label)

    (args.outdir / "tables").mkdir(parents=True, exist_ok=True)
    log("Loading labels and beta mmap")
    y, sample_cols, sample = load_labels(args.sample_groups)
    beta = np.load(args.beta_npy, mmap_mode="r")
    log(f"Samples: n={len(y):,}, Tumor={int(y.sum()):,}, Normal={int((y == 0).sum()):,}")

    pairs = pd.read_parquet(args.promoter_gene_pairs)
    pairs = pairs[pairs["delta_beta"] > args.delta_beta_min].copy()
    known_genes = [gene_clean(g) for g in args.known_genes]
    forced = select_variance_max_by_gene(beta, sample_cols, pairs, known_genes)
    if len(forced) < 3:
        raise ValueError(f"Only {len(forced)} forced literature genes available with delta_beta>{args.delta_beta_min}; expected at least 3.")
    log(f"Forced literature markers selected: {len(forced):,} ({', '.join(forced['primary_gene'])})")

    pool, excluded = build_data_driven_pool(args, forced, beta, sample_cols)
    pool.to_parquet(args.outdir / "tables" / "data_driven_candidate_pool_after_exclusions.parquet", index=False)
    excluded.to_parquet(args.outdir / "tables" / "excluded_candidate_genes_with_reasons.parquet", index=False)
    log(f"Data-driven candidate pool after exclusions: {len(pool):,}; excluded rows logged: {len(excluded):,}")

    x_pool = extract_x(beta, pool["matrix_row"].to_numpy(np.int64), sample_cols)
    log(f"Running full-data LassoCV on remaining pool: X={x_pool.shape}")
    rank = lasso_rank_pool(x_pool, y, args)
    ranked_pool = rank.merge(pool.reset_index(drop=True), left_on="feature_index", right_index=True, how="left")
    ranked_pool.to_parquet(args.outdir / "tables" / "data_driven_lassocv_ranked_candidates.parquet", index=False)

    need = args.max_panel_genes - len(forced)
    data_selected = ranked_pool.head(need).copy()
    data_selected["panel_source"] = "data_driven_lassocv"
    forced = forced.copy()
    forced["lasso_coef_alpha_min"] = np.nan
    forced["lasso_coef_alpha_1se"] = np.nan
    forced["abs_lasso_coef_alpha_min"] = np.nan
    forced["abs_lasso_coef_alpha_1se"] = np.nan

    panel = pd.concat([forced, data_selected], ignore_index=True)
    panel = panel.drop_duplicates("primary_gene", keep="first")
    panel = panel.drop_duplicates("probe_id", keep="first")
    if len(panel) < args.max_panel_genes:
        already_genes = set(panel["primary_gene"])
        already_probes = set(panel["probe_id"])
        fill = ranked_pool[(~ranked_pool["primary_gene"].isin(already_genes)) & (~ranked_pool["probe_id"].isin(already_probes))].head(args.max_panel_genes - len(panel)).copy()
        fill["panel_source"] = "data_driven_lassocv_fill"
        panel = pd.concat([panel, fill], ignore_index=True)
    panel = panel.reset_index(drop=True)
    panel.insert(0, "panel_rank", np.arange(1, len(panel) + 1, dtype=np.int64))
    panel["tumor_hypermethylated_delta_beta_gt_0_3"] = panel["delta_beta"] > args.delta_beta_min
    panel_path = args.outdir / "tables" / "literature_constrained_plus_data_driven_10gene_panel.parquet"
    panel.to_parquet(panel_path, index=False)
    log(f"Saved final literature-constrained panel: {panel_path}")

    x_lit = extract_x(beta, panel["matrix_row"].to_numpy(np.int64), sample_cols)

    pure = pd.read_parquet(args.pure_data_driven_panel)
    pure = standardize_panel_table(pure, "pure_data_driven_previous_rf_lasso")
    x_pure = extract_x(beta, pure["matrix_row"].to_numpy(np.int64), sample_cols)

    base = pd.read_parquet(args.pax1_jam3_panel)
    base = standardize_panel_table(base, "pax1_jam3_baseline")
    x_base = extract_x(beta, base["matrix_row"].to_numpy(np.int64), sample_cols)
    del beta, x_pool
    gc.collect()

    log("Evaluating three fixed panels with identical stratified 5-fold CV")
    fold_rows = []
    summaries = []
    for name, x in [
        ("literature_constrained_plus_data_driven_10gene", x_lit),
        ("pure_data_driven_10gene", x_pure),
        ("PAX1_JAM3_baseline", x_base),
    ]:
        folds, summary = evaluate_panel(name, x, y, args)
        fold_rows.append(folds)
        summaries.append(summary)
    fold_df = pd.concat(fold_rows, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    fold_df.to_parquet(args.outdir / "tables" / "three_panel_5fold_cv_fold_metrics.parquet", index=False)
    summary_df.to_parquet(args.outdir / "tables" / "three_panel_5fold_cv_summary.parquet", index=False)
    log("Saved three-panel CV comparison tables")

    report = {
        "note": "Exploratory literature-constrained panel construction; fixed-panel 5-fold CV is internal TCGA descriptive comparison.",
        "counts": {
            "samples": int(len(y)),
            "tumor": int(y.sum()),
            "normal": int((y == 0).sum()),
            "forced_literature_genes": int(len(forced)),
            "data_driven_pool_after_exclusions": int(len(pool)),
            "final_panel_genes": int(len(panel)),
        },
        "known_gene_list": known_genes,
        "explicit_excluded_genes": [gene_clean(g) for g in args.exclude_genes],
        "selected_genes": panel["primary_gene"].tolist(),
        "selected_probes": panel["probe_id"].tolist(),
        "cv_summary": summary_df.to_dict(orient="records"),
        "outputs": {
            "final_panel": str(panel_path),
            "candidate_pool": str(args.outdir / "tables" / "data_driven_candidate_pool_after_exclusions.parquet"),
            "excluded_candidates": str(args.outdir / "tables" / "excluded_candidate_genes_with_reasons.parquet"),
            "ranked_data_driven_candidates": str(args.outdir / "tables" / "data_driven_lassocv_ranked_candidates.parquet"),
            "cv_fold_metrics": str(args.outdir / "tables" / "three_panel_5fold_cv_fold_metrics.parquet"),
            "cv_summary": str(args.outdir / "tables" / "three_panel_5fold_cv_summary.parquet"),
        },
        "runtime_seconds": round(time.time() - t0, 2),
    }
    (args.outdir / "literature_constrained_panel_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log("Done")


if __name__ == "__main__":
    main()
