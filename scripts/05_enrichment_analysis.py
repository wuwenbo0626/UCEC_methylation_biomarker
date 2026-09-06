#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import gseapy as gp
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.data_utils import ensure_project_dirs, load_config, log, resolve_path, save_json, split_semicolon
from utils.plot_utils import PALETTE, savefig

import matplotlib.pyplot as plt


KNOWN_MARKERS = "PAX1 JAM3 CDO1 CELF4 BHLHE22 HAND2 TBX5 ZNF454 ZNF582 ZNF662 SOX1 MLH1 WIF1 SFRP2 RASSF1 CDH13 HIST1H4F ZSCAN12 GHSR PCDHGB7 HS3ST2 ADCYAP1 NPY DPP6 HAAO GALR1 HOXA9 HTR1B MAGI2 NDN SFMBT2 GYPC ASCL2 MME POU4F3 MIR124-2".split()


def bubble(df, title, out_png, out_pdf, fdr_thr=0.05, top_n=20):
    fig, ax = plt.subplots(figsize=(8, max(4.8, top_n * .32)))
    if df.empty or "Adjusted P-value" not in df:
        ax.text(.5, .5, f"No significant {title}", ha="center", va="center"); ax.axis("off"); savefig(fig, out_png, out_pdf); return
    x = df[df["Adjusted P-value"] < fdr_thr].sort_values("Adjusted P-value").head(top_n).iloc[::-1].copy()
    if x.empty:
        ax.text(.5, .5, f"No significant {title}", ha="center", va="center"); ax.axis("off"); savefig(fig, out_png, out_pdf); return
    overlap = x["Overlap"].str.extract(r"(\d+)/")[0].astype(float) if "Overlap" in x else np.ones(len(x))
    y = np.arange(len(x))
    ax.scatter(-np.log10(x["Adjusted P-value"].astype(float)), y, s=40 + overlap * 18, c=x.get("Combined Score", -np.log10(x["Adjusted P-value"])), cmap="viridis", edgecolor="black", lw=.25)
    ax.set_yticks(y); ax.set_yticklabels(x["Term"], fontsize=7)
    ax.set_xlabel("-log10(adjusted p-value)"); ax.set_title(title); ax.grid(axis="x", alpha=.18)
    savefig(fig, out_png, out_pdf)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config); ensure_project_dirs(cfg)
    results, tables, figs = [resolve_path(cfg, k) for k in ["results_dir", "tables_dir", "figures_dir"]]
    diff, enr_cfg = cfg["differential"], cfg["enrichment"]
    prom = pd.read_parquet(results / "dmp_promoter_TSS200_TSS1500.parquet")
    hyper = prom[(prom["delta_beta"] > diff["delta_beta_threshold"]) & (prom["FDR"] < diff["fdr_threshold"])].copy()
    hyper["gene_symbol_list"] = hyper["gene_name"].map(split_semicolon)
    gene_probe = hyper.explode("gene_symbol_list").rename(columns={"gene_symbol_list": "gene_symbol"})
    gene_probe = gene_probe[gene_probe["gene_symbol"].notna() & (gene_probe["gene_symbol"].astype(str) != "")]
    counts = gene_probe.groupby("gene_symbol")["probe_id"].nunique().rename("probe_count_for_gene").reset_index()
    gene_probe = gene_probe.merge(counts, on="gene_symbol", how="left")
    gene_probe["has_multiple_promoter_probes"] = gene_probe["probe_count_for_gene"] > 1
    gene_probe["abs_delta_beta"] = gene_probe["delta_beta"].abs()
    dedup = gene_probe.sort_values(["gene_symbol", "FDR", "p_value", "abs_delta_beta"], ascending=[True, True, True, False]).groupby("gene_symbol", as_index=False).first()
    dedup = dedup.sort_values(["FDR", "delta_beta"], ascending=[True, False])
    known = pd.DataFrame({"gene_symbol": KNOWN_MARKERS})
    overlap = known.merge(dedup, on="gene_symbol", how="inner")
    gene_probe.to_parquet(results / "promoter_hypermethylated_probe_gene_pairs.parquet", index=False)
    dedup.to_parquet(results / "candidate_genes_dedup.parquet", index=False)
    known.to_parquet(tables / "known_endometrial_cancer_methylation_markers.parquet", index=False)
    overlap.to_parquet(tables / "known_marker_overlaps_with_candidate_genes.parquet", index=False)
    genes = dedup["gene_symbol"].drop_duplicates().tolist()
    go = gp.enrichr(gene_list=genes, gene_sets=[enr_cfg["go_library"]], organism="human", outdir=str(results / "gseapy_GO_BP"), cutoff=1.0, no_plot=True).results
    kegg = gp.enrichr(gene_list=genes, gene_sets=[enr_cfg["kegg_library"]], organism="human", outdir=str(results / "gseapy_KEGG"), cutoff=1.0, no_plot=True).results
    go.to_parquet(results / "GO_BP_enrichr.parquet", index=False)
    kegg.to_parquet(results / "KEGG_enrichr.parquet", index=False)
    go_sig = go[go["Adjusted P-value"] < enr_cfg["adjusted_p_threshold"]]
    kegg_sig = kegg[kegg["Adjusted P-value"] < enr_cfg["adjusted_p_threshold"]]
    go_sig.to_parquet(results / "GO_BP_enrichr_significant.parquet", index=False)
    kegg_sig.to_parquet(results / "KEGG_enrichr_significant.parquet", index=False)
    bubble(go, "GO BP enrichment", figs / "GO_BP_enrichment_top20_bubble.png", figs / "GO_BP_enrichment_top20_bubble.pdf", enr_cfg["adjusted_p_threshold"], enr_cfg["top_n_plot"])
    bubble(kegg, "KEGG enrichment", figs / "KEGG_enrichment_top20_bubble.png", figs / "KEGG_enrichment_top20_bubble.pdf", enr_cfg["adjusted_p_threshold"], enr_cfg["top_n_plot"])
    save_json({"promoter_hyper_probes": int(hyper["probe_id"].nunique()), "candidate_genes": len(dedup), "known_marker_overlaps": len(overlap), "GO_sig": len(go_sig), "KEGG_sig": len(kegg_sig)}, results / "candidate_gene_enrichment_report.json")
    log("05_enrichment_analysis complete")


if __name__ == "__main__":
    main()
