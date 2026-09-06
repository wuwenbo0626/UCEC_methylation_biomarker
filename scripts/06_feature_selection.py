#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, LassoCV, lasso_path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.data_utils import ensure_project_dirs, extract_matrix, load_config, load_sample_labels, log, resolve_path, save_json
from utils.plot_utils import PALETTE, savefig

import matplotlib.pyplot as plt


def alpha_1se(cv):
    mse = cv.mse_path_; mean = mse.mean(1); se = mse.std(1, ddof=1) / np.sqrt(mse.shape[1])
    thr = mean[np.argmin(mean)] + se[np.argmin(mean)]
    return float(np.max(cv.alphas_[mean <= thr]))


def selected_table(features, coef, alpha, stage):
    mask = np.abs(coef) > 1e-8
    out = features.loc[mask].copy()
    out["lasso_coefficient"] = coef[mask].astype(np.float32)
    out["abs_lasso_coefficient"] = np.abs(out["lasso_coefficient"])
    out["lasso_alpha_used"] = alpha
    out["selected_stage"] = stage
    return out.sort_values(["abs_lasso_coefficient", "FDR"], ascending=[False, True])


def plot_path(x, y, alphas, final_idx, alpha_min, alpha_1se_value, out_png, out_pdf, max_iter):
    a, coefs, _ = lasso_path(x, y, alphas=alphas, max_iter=max_iter)
    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.plot(np.log10(a), coefs.T, color=PALETTE["light_gray"], alpha=.25, lw=.5)
    ax.plot(np.log10(a), coefs[final_idx].T, lw=1.2)
    ax.axvline(np.log10(alpha_min), color=PALETTE["orange"], ls="--", label="alpha min")
    ax.axvline(np.log10(alpha_1se_value), color=PALETTE["blue"], ls="--", label="alpha 1-SE")
    ax.set(xlabel="log10(alpha)", ylabel="Coefficient", title="LASSO coefficient path")
    ax.grid(alpha=.18); ax.legend()
    savefig(fig, out_png, out_pdf)


def primary_gene(x):
    return str(x).replace(",", ";").split(";")[0].strip().upper()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config); ensure_project_dirs(cfg)
    processed, results, tables, figs = [resolve_path(cfg, k) for k in ["processed_dir", "results_dir", "tables_dir", "figures_dir"]]
    fs, proj = cfg["feature_selection"], cfg["project"]
    y, sample_cols, _ = load_sample_labels(processed / "ucec_hm450_clean_sample_groups.parquet")
    feature_pool = pd.read_parquet(results / "dmp_promoter_TSS200_TSS1500.parquet").drop_duplicates("probe_id")
    probe_meta = pd.read_parquet(processed / "ucec_hm450_clean_probe_metadata.parquet")[["probe_id", "matrix_row"]]
    features = feature_pool.merge(probe_meta, on="probe_id", how="inner").sort_values("matrix_row").reset_index(drop=True)
    features["feature_index"] = np.arange(len(features))
    features["primary_gene"] = features["gene_name"].map(primary_gene)
    x, _ = extract_matrix(processed / "ucec_hm450_beta_clean_probes_by_samples.npy", features["matrix_row"].to_numpy(np.int64), sample_cols)
    x = StandardScaler(copy=False).fit_transform(x).astype(np.float32, copy=False)
    cv = StratifiedKFold(n_splits=fs["lasso_cv"], shuffle=True, random_state=proj["random_state"])
    lcv = LassoCV(cv=cv, random_state=proj["random_state"], max_iter=fs["lasso_max_iter"], n_jobs=proj["n_jobs"]).fit(x, y.astype(float))
    amin, a1 = float(lcv.alpha_), alpha_1se(lcv)
    sel_min = selected_table(features, lcv.coef_, amin, "min_mse_lasso")
    sel_1 = selected_table(features, Lasso(alpha=a1, max_iter=fs["lasso_max_iter"]).fit(x, y).coef_, a1, "one_standard_error_lasso")
    base = sel_1 if len(sel_min) > 30 else sel_min
    rf = RandomForestClassifier(n_estimators=fs["rf_n_estimators_for_ranking"], random_state=proj["random_state"], n_jobs=proj["n_jobs"], max_features="sqrt").fit(x[:, base["feature_index"]], y)
    base = base.copy(); base["rf_importance"] = rf.feature_importances_.astype(np.float32)
    final = base.sort_values(["rf_importance", "delta_beta", "abs_lasso_coefficient"], ascending=[False, False, False])
    final = final[final["delta_beta"] > 0].drop_duplicates("primary_gene").head(fs["max_final_features"]).copy()
    final.insert(0, "final_marker_rank", np.arange(1, len(final)+1))
    sel_min.to_parquet(tables / "lasso_nonzero_features_alpha_min.parquet", index=False)
    sel_1.to_parquet(tables / "lasso_nonzero_features_alpha_1se.parquet", index=False)
    final.to_parquet(tables / "final_10_methylation_markers.parquet", index=False)
    plot_path(x, y.astype(float), lcv.alphas_, final["feature_index"].to_numpy(int), amin, a1, figs / "lasso_coefficient_path.png", figs / "lasso_coefficient_path.pdf", fs["lasso_max_iter"])
    save_json({"alpha_min": amin, "alpha_1se": a1, "nonzero_min": len(sel_min), "nonzero_1se": len(sel_1), "final_markers": len(final), "selected_probe_ids": final["probe_id"].tolist()}, results / "feature_selection_report.json")
    gc.collect(); log("06_feature_selection complete")


if __name__ == "__main__":
    main()
