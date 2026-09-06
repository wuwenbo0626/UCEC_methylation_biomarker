#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.data_utils import download_file, ensure_project_dirs, load_config, log, parse_tcga_group, resolve_path, save_json


def gdc_post(url: str, payload: dict) -> dict:
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def query_gdc_files(cfg: dict) -> pd.DataFrame:
    api = cfg["download"]["gdc_api"].rstrip("/")
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [cfg["download"]["project_id"]]}},
            {"op": "in", "content": {"field": "files.data_category", "value": [cfg["download"]["data_category"]]}},
            {"op": "in", "content": {"field": "files.data_type", "value": [cfg["download"]["data_type"]]}},
            {"op": "in", "content": {"field": "files.platform", "value": [cfg["download"]["platform"]]}},
        ],
    }
    fields = [
        "file_id", "file_name", "md5sum", "file_size", "platform",
        "cases.case_id", "cases.submitter_id",
        "cases.samples.sample_type", "cases.samples.submitter_id",
        "cases.samples.portions.analytes.aliquots.submitter_id",
    ]
    payload = {"filters": filters, "fields": ",".join(fields), "format": "JSON", "size": 2000}
    data = gdc_post(f"{api}/files", payload)["data"]["hits"]
    rows = []
    for hit in data:
        case = hit.get("cases", [{}])[0]
        sample = case.get("samples", [{}])[0]
        aliquot = ""
        try:
            aliquot = sample["portions"][0]["analytes"][0]["aliquots"][0]["submitter_id"]
        except Exception:
            pass
        sample_id = sample.get("submitter_id") or aliquot[:16]
        rows.append({
            "file_id": hit["file_id"],
            "file_name": hit["file_name"],
            "md5sum": hit.get("md5sum"),
            "file_size": hit.get("file_size"),
            "platform": hit.get("platform"),
            "case_id": case.get("case_id"),
            "patient_id": case.get("submitter_id"),
            "sample_id": sample_id,
            "sample_type": sample.get("sample_type"),
            "aliquot_id": aliquot,
            "group": parse_tcga_group(sample_id),
        })
    df = pd.DataFrame(rows).sort_values(["sample_id", "file_name"]).reset_index(drop=True)
    df["matrix_row"] = np.arange(len(df), dtype=np.int64)
    return df


def download_gdc_files(meta: pd.DataFrame, raw_dir: Path, cfg: dict) -> pd.DataFrame:
    data_url = cfg["download"]["gdc_data"].rstrip("/")
    files_dir = raw_dir / "gdc_hm450_files"
    files_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Downloading GDC methylation files"):
        out = files_dir / row["file_name"]
        if not out.exists():
            url = f"{data_url}/{row['file_id']}"
            with requests.get(url, stream=True, timeout=180) as r:
                r.raise_for_status()
                with out.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
        paths.append(str(out))
    meta = meta.copy()
    meta["local_path"] = paths
    return meta


def read_beta_file(path: str | Path) -> pd.Series:
    df = pd.read_csv(path, sep="\t", comment="#", header=None, dtype={0: "string"}, low_memory=False)
    if df.shape[1] < 2:
        raise ValueError(f"Cannot parse beta file: {path}")
    if str(df.iloc[0, 0]).lower() in {"composite element ref", "probe_id", "id"}:
        df = df.iloc[1:].copy()
    beta = pd.to_numeric(df.iloc[:, 1], errors="coerce").astype("float32")
    return pd.Series(beta.to_numpy(), index=df.iloc[:, 0].astype(str).to_numpy(), name=Path(path).name)


def build_beta_matrix(meta: pd.DataFrame, processed_dir: Path) -> tuple[Path, Path, pd.DataFrame]:
    log("Reading first beta file to establish probe order")
    first = read_beta_file(meta.loc[0, "local_path"])
    probes = first.index.astype(str).to_numpy()
    probe_meta = pd.DataFrame({"matrix_row": np.arange(len(probes), dtype=np.int64), "probe_id": probes})
    beta_path = processed_dir / "ucec_hm450_beta_samples_by_probes.npy"
    mat = np.lib.format.open_memmap(beta_path, mode="w+", dtype=np.float32, shape=(len(meta), len(probes)))
    mat[0, :] = first.to_numpy(dtype=np.float32)
    for i in tqdm(range(1, len(meta)), desc="Building beta matrix"):
        s = read_beta_file(meta.loc[i, "local_path"])
        mat[i, :] = s.reindex(probes).to_numpy(dtype=np.float32)
    mat.flush()
    probe_path = processed_dir / "hm450_probe_metadata.parquet"
    probe_meta.to_parquet(probe_path, index=False)
    return beta_path, probe_path, meta


def query_clinical(cfg: dict, processed_dir: Path) -> Path:
    api = cfg["download"]["gdc_api"].rstrip("/")
    payload = {
        "filters": {"op": "in", "content": {"field": "project.project_id", "value": [cfg["download"]["project_id"]]}},
        "format": "TSV",
        "size": 2000,
        "fields": "case_id,submitter_id,diagnoses.age_at_diagnosis,diagnoses.tumor_stage,diagnoses.primary_diagnosis,diagnoses.morphology,diagnoses.classification_of_tumor,demographic.gender,demographic.race,demographic.ethnicity",
    }
    r = requests.post(f"{api}/cases", json=payload, timeout=120)
    r.raise_for_status()
    tsv = processed_dir / "ucec_clinical_gdc.tsv"
    tsv.write_text(r.text, encoding="utf-8")
    df = pd.read_csv(tsv, sep="\t")
    out = processed_dir / "ucec_clinical.parquet"
    df.to_parquet(out, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--skip-download", action="store_true", help="Only query metadata; do not download raw beta files.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_project_dirs(cfg)
    raw_dir, processed_dir = resolve_path(cfg, "raw_dir"), resolve_path(cfg, "processed_dir")
    manifest_url = cfg["download"]["manifest_url"]
    manifest_path = raw_dir / "HumanMethylation450_15017482_v1-2.csv"
    if not manifest_path.exists():
        try:
            download_file(manifest_url, manifest_path)
        except Exception:
            download_file(cfg["download"]["geo_manifest_fallback_url"], raw_dir / "GPL13534_HM450.csv.gz")
    meta = query_gdc_files(cfg)
    meta_path = processed_dir / "gdc_hm450_file_metadata.parquet"
    meta.to_parquet(meta_path, index=False)
    clinical_path = query_clinical(cfg, processed_dir)
    outputs = {"metadata": str(meta_path), "clinical": str(clinical_path), "manifest": str(manifest_path)}
    if not args.skip_download:
        meta = download_gdc_files(meta, raw_dir, cfg)
        meta.to_parquet(meta_path, index=False)
        beta_path, probe_path, _ = build_beta_matrix(meta, processed_dir)
        outputs.update({"beta_npy": str(beta_path), "probe_metadata": str(probe_path)})
    save_json({"outputs": outputs, "n_files": int(len(meta)), "group_counts": meta["group"].value_counts(dropna=False).to_dict()}, processed_dir / "download_report.json")
    log("01_download_data complete")


if __name__ == "__main__":
    main()
