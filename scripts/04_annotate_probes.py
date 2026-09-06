#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.data_utils import ensure_project_dirs, find_manifest_skiprows, has_promoter, load_config, log, pick_col, resolve_path, save_json, split_semicolon


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config); ensure_project_dirs(cfg)
    raw, results = resolve_path(cfg, "raw_dir"), resolve_path(cfg, "results_dir")
    ann = cfg["annotation"]
    manifest = raw / "HumanMethylation450_15017482_v1-2.csv"
    dmp_path = results / "differential_methylation_significant.parquet"
    dmps = pd.read_parquet(dmp_path); probes = set(dmps["probe_id"].astype(str))
    skip = find_manifest_skiprows(manifest)
    cols = pd.read_csv(manifest, skiprows=skip, nrows=0, compression="infer").columns.tolist()
    name = pick_col(cols, ["Name", "probe_id", "IlmnID"], "probe id")
    gene = pick_col(cols, ["UCSC_RefGene_Name"], "gene")
    group = pick_col(cols, ["UCSC_RefGene_Group"], "gene relation")
    parts = []
    for chunk in pd.read_csv(manifest, skiprows=skip, usecols=[name, gene, group], dtype="string", chunksize=ann["manifest_chunksize"], compression="infer"):
        chunk = chunk.rename(columns={name: "probe_id", gene: "gene_name", group: "gene_relation"})
        chunk = chunk[chunk["probe_id"].astype(str).isin(probes)].copy()
        if len(chunk):
            chunk["gene_name"] = chunk["gene_name"].map(lambda x: ";".join(split_semicolon(x)) if split_semicolon(x) else None)
            chunk["gene_relation"] = chunk["gene_relation"].fillna("Intergenic")
            chunk["is_promoter_TSS200_TSS1500"] = chunk["gene_relation"].map(lambda x: has_promoter(x, ann["promoter_regions"]))
            parts.append(chunk)
        del chunk; gc.collect()
    annot = pd.concat(parts, ignore_index=True).drop_duplicates("probe_id") if parts else pd.DataFrame(columns=["probe_id", "gene_name", "gene_relation"])
    merged = dmps.merge(annot, on="probe_id", how="left")
    prom = merged[merged["is_promoter_TSS200_TSS1500"].fillna(False)].copy()
    merged.to_parquet(results / "dmp_annotated_to_genes.parquet", index=False)
    prom.to_parquet(results / "dmp_promoter_TSS200_TSS1500.parquet", index=False)
    save_json({"dmp_rows": len(dmps), "annotated": len(annot), "promoter_rows": len(prom)}, results / "annotation_report.json")
    log("04_annotate_probes complete")


if __name__ == "__main__":
    main()
