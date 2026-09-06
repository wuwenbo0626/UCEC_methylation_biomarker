#!/usr/bin/env python3
"""
External validation of the TCGA-UCEC methylation panel on GEO GSE155760.

GSE155760:
  - Platform: GPL23976, Illumina HumanMethylationEPIC.
  - Fresh-frozen gynecologic tissues.
  - This script uses endometrial endometrioid carcinoma + uterine serous
    carcinoma as Tumor, and cancer-free normal endometrial mucosa as Normal.

Model:
  - If a fitted sklearn pipeline is supplied with --model-pkl, load it.
  - Otherwise train a TCGA logistic-regression pipeline on the fixed panel
    probes, then apply that model to GEO. This is the current default because
    no TCGA model parameter file was supplied in the conversation.

Processed GEO signals:
  GSE155760_signals.csv.gz contains triplets:
    <sample> Unmethylated Signal, <sample> Methylated Signal, <sample> Detection Pval
  Beta is computed as M / (M + U + offset), offset=100 by default.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import pickle
import shutil
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("/Users/wuwenbo/AsiaInfo/tcga_ucec_methylation")
CLEAN = ROOT / "tcga_ucec_hm450_cleaned"
LIT_PANEL = ROOT / "tcga_ucec_hm450_literature_constrained_panel"
OUT = ROOT / "geo_external_validation"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate TCGA methylation panel on GSE155760.")
    p.add_argument("--geo-acc", default="GSE155760")
    p.add_argument("--series-matrix-url", default="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE155nnn/GSE155760/matrix/GSE155760_series_matrix.txt.gz")
    p.add_argument("--signal-url", default="https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE155760&format=file&file=GSE155760_signals.csv.gz")
    p.add_argument("--outdir", type=Path, default=OUT / "GSE155760")
    p.add_argument("--panel-parquet", type=Path, default=LIT_PANEL / "tables" / "literature_constrained_plus_data_driven_10gene_panel.parquet")
    p.add_argument("--tcga-beta-npy", type=Path, default=CLEAN / "matrix" / "ucec_hm450_beta_clean_probes_by_samples.npy")
    p.add_argument("--tcga-sample-groups", type=Path, default=CLEAN / "metadata" / "ucec_hm450_clean_sample_groups.parquet")
    p.add_argument("--model-pkl", type=Path, default=None, help="Optional fitted sklearn model/pipeline. If absent, train TCGA LR model.")
    p.add_argument("--beta-offset", type=float, default=100.0)
    p.add_argument("--force-download", action="store_true")
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


def download(url: str, dest: Path, force: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        log(f"Using existing download: {dest} ({dest.stat().st_size / 1024**2:.1f} MB)")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    log(f"Downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as fh:
        shutil.copyfileobj(resp, fh, length=1024 * 1024)
    tmp.replace(dest)
    log(f"Saved {dest} ({dest.stat().st_size / 1024**2:.1f} MB)")


def strip_quote(x: str) -> str:
    x = x.strip()
    if len(x) >= 2 and x[0] == '"' and x[-1] == '"':
        return x[1:-1]
    return x


def parse_series_metadata_from_url(url: str) -> pd.DataFrame:
    """Stream only sample metadata from a GEO series matrix gz URL."""
    log(f"Streaming sample metadata from {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    meta: dict[str, list[str] | list[list[str]]] = {}
    characteristics: list[list[str]] = []
    with urllib.request.urlopen(req, timeout=120) as resp:
        with gzip.GzipFile(fileobj=resp) as gz:
            for raw in gz:
                line = raw.decode("utf-8", errors="replace").rstrip("\n")
                if line.startswith("!series_matrix_table_begin"):
                    break
                if not line.startswith("!Sample_"):
                    continue
                parts = next(csv.reader([line], delimiter="\t"))
                key = parts[0]
                values = [strip_quote(v) for v in parts[1:]]
                if key == "!Sample_characteristics_ch1":
                    characteristics.append(values)
                else:
                    meta[key] = values

    accessions = meta.get("!Sample_geo_accession")
    titles = meta.get("!Sample_title")
    source = meta.get("!Sample_source_name_ch1")
    platform = meta.get("!Sample_platform_id")
    if not accessions or not titles:
        raise ValueError("Could not parse sample accessions/titles from series matrix.")

    df = pd.DataFrame(
        {
            "geo_accession": accessions,
            "title": titles,
            "source_name": source if source else [""] * len(accessions),
            "platform_id": platform if platform else [""] * len(accessions),
        }
    )
    for char_values in characteristics:
        if len(char_values) != len(df):
            continue
        first = char_values[0]
        field = first.split(":", 1)[0].strip().lower().replace(" ", "_") if ":" in first else f"characteristic_{len(df.columns)}"
        df[field] = [v.split(":", 1)[1].strip() if ":" in v else v for v in char_values]

    df["signal_sample_name"] = df["title"].str.split(":", n=1).str[0]
    hist = df.get("histology", df["source_name"]).astype(str).str.lower()
    df["validation_group"] = np.select(
        [
            hist.isin(["endometrial endometrioid carcinoma", "uterine serous carcinoma"]),
            hist.eq("endometrial mucosa from cancer-free normal control"),
        ],
        ["Tumor", "Normal"],
        default="Exclude",
    )
    df["label"] = df["validation_group"].map({"Normal": 0, "Tumor": 1})
    return df


def parse_signal_header(header: list[str]) -> list[dict]:
    triplets = []
    i = 1
    while i + 2 < len(header):
        u_col = header[i]
        m_col = header[i + 1]
        p_col = header[i + 2]
        if not (u_col.endswith(" Unmethylated Signal") and m_col.endswith(" Methylated Signal") and p_col.endswith(" Detection Pval")):
            raise ValueError(f"Unexpected signal triplet columns around {i}: {u_col}, {m_col}, {p_col}")
        sample = u_col.removesuffix(" Unmethylated Signal")
        triplets.append({"sample_name": sample, "u_idx": i, "m_idx": i + 1, "p_idx": i + 2})
        i += 3
    return triplets


def extract_geo_panel_beta(signal_gz: Path, target_probes: list[str], offset: float) -> tuple[pd.DataFrame, list[str], list[str]]:
    target_set = set(target_probes)
    found: dict[str, np.ndarray] = {}
    with gzip.open(signal_gz, "rt", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        triplets = parse_signal_header(header)
        sample_names = [t["sample_name"] for t in triplets]
        for row_i, row in enumerate(reader, start=1):
            if not row:
                continue
            probe = row[0]
            if probe not in target_set:
                continue
            betas = np.empty(len(triplets), dtype=np.float32)
            for j, t in enumerate(triplets):
                try:
                    u = float(row[t["u_idx"]])
                    m = float(row[t["m_idx"]])
                except Exception:
                    betas[j] = np.nan
                    continue
                denom = m + u + offset
                betas[j] = m / denom if denom > 0 else np.nan
            found[probe] = betas
            log(f"Found target probe {probe}: {len(found)}/{len(target_set)}")
            if len(found) == len(target_set):
                break
            if row_i % 100000 == 0:
                log(f"Scanned {row_i:,} GEO signal rows; found {len(found)}/{len(target_set)}")

    missing = [p for p in target_probes if p not in found]
    present = [p for p in target_probes if p in found]
    beta_df = pd.DataFrame({p: found[p] for p in present}, index=sample_names).reset_index(names="signal_sample_name")
    return beta_df, present, missing


def load_tcga_labels(path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    sample = pd.read_parquet(path).copy()
    if "matrix_col" not in sample.columns:
        sample["matrix_col"] = np.arange(len(sample), dtype=np.int64)
    group = sample["group"].astype(str).str.lower()
    keep = group.isin(["tumor", "primary tumor", "1", "normal", "solid tissue normal", "0"])
    sample = sample.loc[keep].reset_index(drop=True)
    y = sample["group"].astype(str).str.lower().isin(["tumor", "primary tumor", "1"]).astype(np.int8).to_numpy()
    cols = sample["matrix_col"].to_numpy(np.int64)
    return y, cols, sample


def extract_tcga_x(beta_path: Path, panel: pd.DataFrame, sample_cols: np.ndarray) -> np.ndarray:
    beta = np.load(beta_path, mmap_mode="r")
    rows = panel["matrix_row"].to_numpy(np.int64)
    if beta.shape[0] > rows.max() and beta.shape[1] > sample_cols.max():
        x = np.asarray(beta[np.ix_(rows, sample_cols)].T, dtype=np.float32)
    elif beta.shape[1] > rows.max() and beta.shape[0] > sample_cols.max():
        x = np.asarray(beta[np.ix_(sample_cols, rows)], dtype=np.float32)
    else:
        raise ValueError(f"Cannot infer TCGA beta orientation from {beta.shape}")
    return x


def compute_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    pred = (prob >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "AUC": float(roc_auc_score(y_true, prob)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
        "accuracy": float(accuracy_score(y_true, pred)),
        "F1": float(f1_score(y_true, pred)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "threshold": float(threshold),
    }


def main() -> None:
    args = parse_args()
    t0 = time.time()
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "raw").mkdir(parents=True, exist_ok=True)
    (args.outdir / "tables").mkdir(parents=True, exist_ok=True)
    (args.outdir / "models").mkdir(parents=True, exist_ok=True)
    for path, label in [
        (args.panel_parquet, "panel parquet"),
        (args.tcga_beta_npy, "TCGA beta matrix"),
        (args.tcga_sample_groups, "TCGA sample groups"),
    ]:
        ensure(path, label)

    panel = pd.read_parquet(args.panel_parquet).copy()
    if "probe_id" not in panel.columns or "matrix_row" not in panel.columns:
        raise ValueError("Panel table must contain probe_id and matrix_row.")
    panel["probe_id"] = panel["probe_id"].astype(str)
    target_probes = panel["probe_id"].tolist()

    sample_info = parse_series_metadata_from_url(args.series_matrix_url)
    sample_info_path = args.outdir / "tables" / "GSE155760_sample_info.parquet"
    sample_info.to_parquet(sample_info_path, index=False)
    validation_sample_info = sample_info[sample_info["validation_group"].isin(["Tumor", "Normal"])].copy()
    validation_sample_info.to_parquet(args.outdir / "tables" / "GSE155760_endometrial_validation_samples.parquet", index=False)
    log(f"GSE155760 validation samples: {validation_sample_info['validation_group'].value_counts().to_dict()}")

    signal_gz = args.outdir / "raw" / "GSE155760_signals.csv.gz"
    download(args.signal_url, signal_gz, force=args.force_download)
    geo_beta_all, present, missing = extract_geo_panel_beta(signal_gz, target_probes, args.beta_offset)
    geo_beta_all.to_parquet(args.outdir / "tables" / "GSE155760_panel_beta_all_samples.parquet", index=False)
    pd.DataFrame({"missing_probe_id": missing}).to_parquet(args.outdir / "tables" / "GSE155760_missing_panel_probes.parquet", index=False)
    if missing:
        log(f"Missing panel probes in GEO signal matrix: {missing}")
    else:
        log("All panel probes found in GEO signal matrix")

    valid = validation_sample_info.merge(geo_beta_all, on="signal_sample_name", how="inner")
    valid = valid.dropna(subset=present).copy()
    if valid["validation_group"].nunique() < 2:
        raise ValueError("Validation subset lacks both tumor and normal samples after merging beta values.")
    y_geo = valid["label"].astype(int).to_numpy()
    x_geo = valid[present].to_numpy(dtype=np.float32)

    # Train or load TCGA model. If GEO is missing panel probes, use the common
    # probe subset in the same order.
    if args.model_pkl is not None:
        ensure(args.model_pkl, "model pickle")
        with args.model_pkl.open("rb") as fh:
            model = pickle.load(fh)
        model_note = f"loaded model from {args.model_pkl}"
    else:
        y_tcga, tcga_cols, _ = load_tcga_labels(args.tcga_sample_groups)
        panel_present = panel[panel["probe_id"].isin(present)].copy()
        panel_present["probe_id"] = pd.Categorical(panel_present["probe_id"], categories=present, ordered=True)
        panel_present = panel_present.sort_values("probe_id")
        x_tcga = extract_tcga_x(args.tcga_beta_npy, panel_present, tcga_cols)
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, random_state=42)),
            ]
        )
        model.fit(x_tcga, y_tcga)
        model_note = "trained TCGA logistic-regression pipeline on fixed panel because no --model-pkl was supplied"
        with (args.outdir / "models" / "tcga_trained_lr_pipeline_for_geo_validation.pkl").open("wb") as fh:
            pickle.dump(model, fh)

    prob = model.predict_proba(x_geo)[:, 1]
    metrics = compute_metrics(y_geo, prob, threshold=0.5)
    metrics_df = pd.DataFrame([{**metrics, "dataset": args.geo_acc, "n_samples": int(len(y_geo)), "n_tumor": int(y_geo.sum()), "n_normal": int((y_geo == 0).sum()), "n_features_used": int(len(present))}])
    metrics_df.to_parquet(args.outdir / "tables" / "GSE155760_external_validation_metrics.parquet", index=False)
    pred_df = valid[["geo_accession", "title", "source_name", "histology", "validation_group", "label", "signal_sample_name"]].copy()
    pred_df["predicted_probability_tumor"] = prob
    pred_df["predicted_label"] = np.where(prob >= 0.5, "Tumor", "Normal")
    pred_df.to_parquet(args.outdir / "tables" / "GSE155760_external_validation_predictions.parquet", index=False)

    report = {
        "geo_dataset": {
            "accession": args.geo_acc,
            "geo_page": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE155760",
            "platform": "GPL23976 Illumina HumanMethylationEPIC",
            "signal_url": args.signal_url,
            "series_matrix_url": args.series_matrix_url,
            "validation_definition": "Tumor=endometrial endometrioid carcinoma or uterine serous carcinoma; Normal=endometrial mucosa from cancer-free normal control",
        },
        "model_note": model_note,
        "beta_formula": f"Methylated / (Methylated + Unmethylated + {args.beta_offset:g})",
        "panel_probes_requested": target_probes,
        "panel_probes_present": present,
        "panel_probes_missing": missing,
        "sample_counts": {
            "validation_total": int(len(y_geo)),
            "tumor": int(y_geo.sum()),
            "normal": int((y_geo == 0).sum()),
        },
        "external_validation_metrics": metrics,
        "outputs": {
            "sample_info": str(sample_info_path),
            "validation_sample_info": str(args.outdir / "tables" / "GSE155760_endometrial_validation_samples.parquet"),
            "panel_beta_all_samples": str(args.outdir / "tables" / "GSE155760_panel_beta_all_samples.parquet"),
            "missing_panel_probes": str(args.outdir / "tables" / "GSE155760_missing_panel_probes.parquet"),
            "metrics": str(args.outdir / "tables" / "GSE155760_external_validation_metrics.parquet"),
            "predictions": str(args.outdir / "tables" / "GSE155760_external_validation_predictions.parquet"),
        },
        "runtime_seconds": round(time.time() - t0, 2),
    }
    report_path = args.outdir / "GSE155760_external_validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"External validation metrics: {metrics}")
    log(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
