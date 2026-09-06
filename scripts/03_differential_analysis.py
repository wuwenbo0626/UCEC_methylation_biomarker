#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.data_utils import ensure_project_dirs, load_config, load_sample_labels, log, resolve_path, save_json
from utils.plot_utils import volcano_plot


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config); ensure_project_dirs(cfg)
    processed, results, figs, tables = [resolve_path(cfg, k) for k in ["processed_dir", "results_dir", "figures_dir", "tables_dir"]]
    d, p = cfg["differential"], cfg["preprocess"]
    beta = np.load(processed / "ucec_hm450_beta_clean_probes_by_samples.npy", mmap_mode="r")
    probe = pd.read_parquet(processed / "ucec_hm450_clean_probe_metadata.parquet").sort_values("matrix_row")
    y, _, sample = load_sample_labels(processed / "ucec_hm450_clean_sample_groups.parquet")
    tumor, normal = np.where(y == 1)[0], np.where(y == 0)[0]
    n = beta.shape[0]
    stat = np.empty(n, np.float32); pval = np.empty(n, np.float64); db = np.empty(n, np.float32)
    log(f"Differential analysis for {n:,} probes")
    for s in range(0, n, p["chunk_probes"]):
        e = min(s + p["chunk_probes"], n)
        b = np.asarray(beta[s:e, :], dtype=np.float32)
        db[s:e] = b[:, tumor].mean(1) - b[:, normal].mean(1)
        m = np.clip(b, p["beta_clip_eps"], 1 - p["beta_clip_eps"])
        m = np.log2(m / (1 - m))
        r = ttest_ind(m[:, tumor], m[:, normal], axis=1, equal_var=d["ttest_equal_var"])
        stat[s:e] = np.nan_to_num(r.statistic, nan=0).astype(np.float32)
        pval[s:e] = np.nan_to_num(r.pvalue, nan=1).astype(np.float64)
        del b, m, r; gc.collect()
    fdr = multipletests(pval, method="fdr_bh")[1]
    sig = (np.abs(db) > d["delta_beta_threshold"]) & (fdr < d["fdr_threshold"])
    out = pd.DataFrame({"probe_id": probe["probe_id"].to_numpy(), "t_statistic": stat, "p_value": pval, "FDR": fdr, "delta_beta": db, "direction": np.where(db > 0, "高甲基化", "低甲基化"), "significant": sig})
    out = pd.concat([out, probe.drop(columns=["probe_id"], errors="ignore").reset_index(drop=True)], axis=1)
    all_path = results / "differential_methylation_all.parquet"
    sig_path = results / "differential_methylation_significant.parquet"
    out.to_parquet(all_path, index=False)
    out.loc[sig].sort_values(["FDR", "p_value"]).to_parquet(sig_path, index=False)
    volcano_plot(out, figs / "fig2_volcano_plot.png", figs / "fig2_volcano_plot.pdf", d["delta_beta_threshold"], d["fdr_threshold"])
    save_json({"significant_total": int(sig.sum()), "hyper": int((sig & (db > 0)).sum()), "hypo": int((sig & (db < 0)).sum())}, results / "differential_report.json")
    log("03_differential_analysis complete")


if __name__ == "__main__":
    main()
