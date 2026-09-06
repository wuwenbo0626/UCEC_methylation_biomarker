#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.data_utils import detect_orientation, ensure_project_dirs, find_manifest_skiprows, has_promoter, load_config, log, parse_tcga_group, pick_col, resolve_path, save_json


def read_block(x, sample_idx, probe_idx, ori):
    return np.asarray(x[np.ix_(sample_idx, probe_idx)] if ori == "samples_by_probes" else x[np.ix_(probe_idx, sample_idx)].T, dtype=np.float32)


def missingness(x, samples, probes, ori, chunk):
    pm = np.zeros(len(probes), dtype=np.uint16)
    sm = np.zeros(len(samples), dtype=np.uint32)
    for s in range(0, len(probes), chunk):
        e = min(s + chunk, len(probes))
        b = read_block(x, samples, probes[s:e], ori)
        m = np.isnan(b)
        pm[s:e] = m.sum(0)
        sm += m.sum(1).astype(np.uint32)
        del b, m
        gc.collect()
    return pm, sm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--beta-npy", type=Path, default=None)
    ap.add_argument("--sample-metadata", type=Path, default=None)
    ap.add_argument("--probe-metadata", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config); ensure_project_dirs(cfg)
    processed, tables = resolve_path(cfg, "processed_dir"), resolve_path(cfg, "tables_dir")
    p = cfg["preprocess"]
    beta_path = args.beta_npy or processed / "ucec_hm450_beta_samples_by_probes.npy"
    sample_path = args.sample_metadata or processed / "gdc_hm450_file_metadata.parquet"
    probe_path = args.probe_metadata or processed / "hm450_probe_metadata.parquet"
    manifest_path = args.manifest or resolve_path(cfg, "raw_dir") / "HumanMethylation450_15017482_v1-2.csv"

    log("Loading metadata and beta mmap")
    sample = pd.read_parquet(sample_path).sort_values("matrix_row").reset_index(drop=True)
    probe = pd.read_parquet(probe_path).reset_index(drop=True)
    if "matrix_row" not in probe.columns:
        probe["matrix_row"] = np.arange(len(probe), dtype=np.int64)
    if "probe_id" not in probe.columns:
        probe = probe.rename(columns={"probeID": "probe_id"})
    sample["group"] = sample.get("group", sample["sample_id"].map(parse_tcga_group))
    selected = sample[sample["group"].isin(["Tumor", "Normal"])].copy().sort_values("matrix_row")
    x = np.load(beta_path, mmap_mode="r")
    ori = detect_orientation(x, n_probes=len(probe), n_samples=len(sample))
    samples = selected["matrix_row"].to_numpy(np.int64)
    probes = np.arange(len(probe), dtype=np.int64)

    log("Filtering by probe/sample missingness")
    pm, sm0 = missingness(x, samples, probes, ori, p["chunk_probes"])
    keep_probe_missing = pm / len(samples) <= p["probe_missing_threshold"]
    probes1 = probes[keep_probe_missing]
    _, sm = missingness(x, samples, probes1, ori, p["chunk_probes"])
    keep_sample = sm / len(probes1) <= p["sample_missing_threshold"]

    log("Filtering rs* and chrX/chrY probes")
    keep_not_rs = ~probe["probe_id"].astype(str).str.startswith("rs").to_numpy()
    skip = find_manifest_skiprows(manifest_path)
    cols = pd.read_csv(manifest_path, skiprows=skip, nrows=0, compression="infer").columns.tolist()
    name_col = pick_col(cols, ["Name", "probe_id", "IlmnID"], "probe id")
    chr_col = pick_col(cols, ["CHR", "CpG_chrm"], "chromosome")
    chrom = pd.read_csv(manifest_path, skiprows=skip, usecols=[name_col, chr_col], dtype="string", compression="infer")
    chrom = chrom.rename(columns={name_col: "probe_id", chr_col: "chromosome"}).drop_duplicates("probe_id")
    probe = probe.merge(chrom, on="probe_id", how="left")
    keep_not_sex = ~probe["chromosome"].astype(str).str.replace("chr", "", regex=False).isin(["X", "Y"]).to_numpy()
    final_probe_mask = keep_probe_missing & keep_not_rs & keep_not_sex
    final_probes = probes[final_probe_mask]
    final_samples = samples[keep_sample]

    log("Writing filtered matrix and KNN imputing")
    filtered = np.empty((len(final_samples), len(final_probes)), dtype=np.float32)
    for s in range(0, len(final_probes), p["chunk_probes"]):
        e = min(s + p["chunk_probes"], len(final_probes))
        filtered[:, s:e] = read_block(x, final_samples, final_probes[s:e], ori)
    if np.isnan(filtered).any():
        filtered = KNNImputer(n_neighbors=p["knn_k"]).fit_transform(filtered).astype(np.float32)
    clean_path = processed / "ucec_hm450_beta_clean_probes_by_samples.npy"
    np.save(clean_path, filtered.T.astype(np.float32))
    sample_out = selected.loc[keep_sample].copy().reset_index(drop=True)
    sample_out["matrix_col"] = np.arange(len(sample_out), dtype=np.int64)
    sample_out.to_parquet(processed / "ucec_hm450_clean_sample_groups.parquet", index=False)
    probe_out = probe.loc[final_probe_mask, ["probe_id", "chromosome"]].copy().reset_index(drop=True)
    probe_out["matrix_row"] = np.arange(len(probe_out), dtype=np.int64)
    probe_out["original_probe_index"] = final_probes
    probe_out.to_parquet(processed / "ucec_hm450_clean_probe_metadata.parquet", index=False)
    save_json({"input_shape": list(x.shape), "orientation": ori, "samples_after": len(final_samples), "probes_after": len(final_probes)}, processed / "preprocess_report.json")
    log("02_preprocess complete")


if __name__ == "__main__":
    main()
