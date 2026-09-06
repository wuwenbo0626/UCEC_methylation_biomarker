#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.data_utils import ensure_project_dirs, extract_matrix, find_manifest_skiprows, has_promoter, load_config, load_sample_labels, log, pick_col, resolve_path, save_json, split_semicolon
from utils.model_utils import bootstrap_auc_ci, classification_metrics, delong_test, wilson_ci
from utils.plot_utils import confusion_grid, roc_plot, savefig, volcano_plot, workflow_plot, PALETTE

import matplotlib.pyplot as plt


def make_lr():
    return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])


def scan_gene_probes(manifest, genes, chunksize):
    skip = find_manifest_skiprows(manifest)
    cols = pd.read_csv(manifest, skiprows=skip, nrows=0, compression="infer").columns.tolist()
    name = pick_col(cols, ["Name", "probe_id", "IlmnID"], "probe id")
    gene = pick_col(cols, ["UCSC_RefGene_Name"], "gene")
    group = pick_col(cols, ["UCSC_RefGene_Group"], "gene relation")
    parts = []
    for ch in pd.read_csv(manifest, skiprows=skip, usecols=[name, gene, group], dtype="string", chunksize=chunksize, compression="infer"):
        ch = ch.rename(columns={name: "probe_id", gene: "gene_name", group: "gene_relation"})
        ch["genes"] = ch["gene_name"].map(split_semicolon)
        ch = ch[ch["genes"].map(lambda xs: bool(set(xs) & set(genes)))]
        if len(ch):
            ch["is_promoter"] = ch["gene_relation"].map(has_promoter); parts.append(ch)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def choose_baseline(beta_path, sample_cols, processed, raw, cfg, tables):
    hits = scan_gene_probes(raw / "HumanMethylation450_15017482_v1-2.csv", ["PAX1", "JAM3"], cfg["annotation"]["manifest_chunksize"])
    probe = pd.read_parquet(processed / "ucec_hm450_clean_probe_metadata.parquet")
    cand = hits.merge(probe[["probe_id", "matrix_row"]], on="probe_id", how="inner")
    rows = []
    for _, r in cand.iterrows():
        for g in set(r["genes"]) & {"PAX1", "JAM3"}:
            d = r.to_dict(); d["target_gene"] = g; rows.append(d)
    cand = pd.DataFrame(rows)
    if cand.empty:
        return cand, pd.DataFrame()
    x, _ = extract_matrix(beta_path, cand["matrix_row"].to_numpy(np.int64), sample_cols)
    cand["beta_variance_all_samples"] = x.var(0, ddof=1).astype(np.float32)
    chosen = []
    for g in ["PAX1", "JAM3"]:
        d = cand[cand["target_gene"].eq(g)]
        d2 = d[d["is_promoter"]]
        chosen.append((d2 if len(d2) else d).sort_values("beta_variance_all_samples", ascending=False).iloc[0])
    chosen = pd.DataFrame(chosen)
    cand.to_parquet(tables / "pax1_jam3_all_candidate_probes.parquet", index=False)
    chosen.to_parquet(tables / "pax1_jam3_selected_baseline_probes.parquet", index=False)
    return cand, chosen


def eval_lr(x, y, train, test, boot=2000, seed=42):
    model = make_lr().fit(x[train], y[train])
    score = model.predict_proba(x[test])[:, 1]
    pred = model.predict(x[test])
    m = classification_metrics(y[test], pred, score)
    lo, hi = bootstrap_auc_ci(y[test], score, boot, seed); m["AUC_95CI_low"] = lo; m["AUC_95CI_high"] = hi
    cm = confusion_matrix(y[test], pred, labels=[0, 1]); tn, fp, fn, tp = cm.ravel()
    m["sensitivity_95CI_low"], m["sensitivity_95CI_high"] = wilson_ci(int(tp), int(tp + fn))
    m["specificity_95CI_low"], m["specificity_95CI_high"] = wilson_ci(int(tn), int(tn + fp))
    return model, score, pred, cm, m


def marker_boxplot(x, y, markers, out_png, out_pdf):
    fig, axes = plt.subplots(2, 5, figsize=(14, 6), sharey=True); axes = axes.ravel()
    rng = np.random.default_rng(42)
    for i, ax in enumerate(axes[:x.shape[1]]):
        normal, tumor = x[y == 0, i], x[y == 1, i]
        bp = ax.boxplot([normal, tumor], patch_artist=True, showfliers=False)
        bp["boxes"][0].set_facecolor(PALETTE["blue"]); bp["boxes"][1].set_facecolor(PALETTE["orange"])
        ax.scatter(1 + rng.normal(0,.03,len(normal)), normal, s=8, c=PALETTE["blue"], alpha=.45)
        ax.scatter(2 + rng.normal(0,.03,len(tumor)), tumor, s=8, c=PALETTE["orange"], alpha=.35)
        ax.set_title(f"{markers.iloc[i]['primary_gene']}\n{markers.iloc[i]['probe_id']}", fontsize=8)
        ax.set_xticks([1,2]); ax.set_xticklabels(["Normal","Tumor"], fontsize=8); ax.grid(axis="y", alpha=.18)
    axes[0].set_ylabel("Beta value"); fig.suptitle("Final marker methylation levels")
    savefig(fig, out_png, out_pdf)


def coef_plot(markers, coef, out_png, out_pdf):
    d = markers.copy(); d["lr_standardized_coefficient"] = coef
    d = d.sort_values("lr_standardized_coefficient")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.barh(np.arange(len(d)), d["lr_standardized_coefficient"], color=np.where(d["lr_standardized_coefficient"] >= 0, PALETTE["orange"], PALETTE["blue"]))
    ax.set_yticks(np.arange(len(d))); ax.set_yticklabels(d["probe_id"] + "/" + d["primary_gene"], fontsize=8)
    ax.axvline(0, color=PALETTE["dark"]); ax.set_xlabel("Standardized LR coefficient"); ax.set_title("10-marker coefficients")
    savefig(fig, out_png, out_pdf); return d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config); ensure_project_dirs(cfg)
    processed, raw, results, tables, figs = [resolve_path(cfg, k) for k in ["processed_dir", "raw_dir", "results_dir", "tables_dir", "figures_dir"]]
    rs, mcfg = cfg["project"]["random_state"], cfg["models"]
    beta_path = processed / "ucec_hm450_beta_clean_probes_by_samples.npy"
    y, sample_cols, _ = load_sample_labels(processed / "ucec_hm450_clean_sample_groups.parquet")
    markers = pd.read_parquet(tables / "final_10_methylation_markers.parquet").sort_values("final_marker_rank")
    x_mine, _ = extract_matrix(beta_path, markers["matrix_row"].to_numpy(np.int64), sample_cols)
    cand, base = choose_baseline(beta_path, sample_cols, processed, raw, cfg, tables)
    train, test = train_test_split(np.arange(len(y)), test_size=mcfg["test_size"], stratify=y, random_state=rs)
    mine_model, mine_score, mine_pred, mine_cm, mine_m = eval_lr(x_mine, y, train, test, mcfg["bootstrap_n"], rs)
    x_base, _ = extract_matrix(beta_path, base["matrix_row"].to_numpy(np.int64), sample_cols)
    base_model, base_score, base_pred, base_cm, base_m = eval_lr(x_base, y, train, test, mcfg["bootstrap_n"], rs)
    dl = delong_test(y[test], mine_score, base_score)
    conclusion = "我的模型显著优于PAX1/JAM3 baseline" if dl["p_value"] < .05 and dl["auc_difference_a_minus_b"] > 0 else "我的模型与PAX1/JAM3 baseline无显著差异"
    pd.DataFrame([{"model": "10_marker_LR", **mine_m}, {"model": "PAX1_JAM3_LR", **base_m}]).to_parquet(tables / "lr_model_metrics_with_95ci.parquet", index=False)
    pd.DataFrame([dl]).to_parquet(tables / "delong_auc_test_10marker_vs_pax1_jam3.parquet", index=False)
    workflow_plot(figs / "fig1_study_workflow.png", figs / "fig1_study_workflow.pdf")
    volcano_plot(pd.read_parquet(results / "differential_methylation_all.parquet", columns=["delta_beta", "FDR"]), figs / "fig2_volcano_plot.png", figs / "fig2_volcano_plot.pdf")
    marker_boxplot(x_mine, y, markers, figs / "fig3_marker_beta_boxplots.png", figs / "fig3_marker_beta_boxplots.pdf")
    roc_plot(y[test], {"10-marker LR": mine_score, "PAX1/JAM3 baseline": base_score}, figs / "fig4_roc_10marker_vs_baseline.png", figs / "fig4_roc_10marker_vs_baseline.pdf", "10-marker vs baseline")
    coef_df = coef_plot(markers, mine_model.named_steps["clf"].coef_[0], figs / "fig5_lr_feature_coefficients.png", figs / "fig5_lr_feature_coefficients.pdf")
    confusion_grid({"10-marker LR": mine_cm, "PAX1/JAM3 LR": base_cm}, figs / "fig6_confusion_matrices.png", figs / "fig6_confusion_matrices.pdf")
    markers.to_parquet(tables / "final_10_marker_summary.parquet", index=False)
    coef_df.to_parquet(tables / "lr_standardized_coefficients_final_10_markers.parquet", index=False)
    summary = pd.DataFrame([
        {"item": "differential_methylation_probes_total", "value": len(pd.read_parquet(results / "differential_methylation_significant.parquet"))},
        {"item": "candidate_genes", "value": len(pd.read_parquet(results / "candidate_genes_dedup.parquet"))},
        {"item": "10_marker_AUC", "value": mine_m["AUC"], "ci_low": mine_m["AUC_95CI_low"], "ci_high": mine_m["AUC_95CI_high"]},
        {"item": "PAX1_JAM3_AUC", "value": base_m["AUC"], "ci_low": base_m["AUC_95CI_low"], "ci_high": base_m["AUC_95CI_high"]},
        {"item": "DeLong_p_value", "value": dl["p_value"], "note": conclusion},
    ])
    summary.to_parquet(tables / "key_results_summary.parquet", index=False)
    text = f"本研究基于TCGA-UCEC 450K甲基化数据筛选子宫内膜癌标志物，最终获得10个启动子区甲基化探针。10标志物逻辑回归模型在70/30分层测试集中AUC={mine_m['AUC']:.3f}，敏感性={mine_m['sensitivity']:.3f}，特异性={mine_m['specificity']:.3f}；PAX1/JAM3 baseline AUC={base_m['AUC']:.3f}。DeLong检验p={dl['p_value']:.2e}，结论：{conclusion}。"
    (Path(cfg["_root"]) / "outputs" / "results_summary_zh.md").write_text(text, encoding="utf-8")
    save_json({"metrics": {"10_marker_LR": mine_m, "PAX1_JAM3_LR": base_m}, "delong": dl, "conclusion": conclusion}, results / "baseline_comparison_and_report.json")
    log("08_evaluate_model complete")


if __name__ == "__main__":
    main()
