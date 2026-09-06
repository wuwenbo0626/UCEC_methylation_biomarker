from __future__ import annotations

import gzip
import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path | None = None) -> dict:
    path = Path(config_path) if config_path else project_root() / "config.yaml"
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    cfg["_root"] = str(path.resolve().parent)
    return cfg


def resolve_path(cfg: dict, key: str) -> Path:
    root = Path(cfg["_root"])
    return root / cfg["paths"][key]


def ensure_project_dirs(cfg: dict) -> None:
    for key in ["raw_dir", "processed_dir", "results_dir", "figures_dir", "tables_dir"]:
        resolve_path(cfg, key).mkdir(parents=True, exist_ok=True)


def memory_text() -> str:
    proc = psutil.Process(os.getpid())
    rss = proc.memory_info().rss / 1024**3
    avail = psutil.virtual_memory().available / 1024**3
    return f"RSS={rss:.2f} GB, available={avail:.2f} GB"


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg} | {memory_text()}", flush=True)


def save_json(obj: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def download_file(url: str, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    log(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=180) as r, tmp.open("wb") as f:
        while True:
            b = r.read(1024 * 1024)
            if not b:
                break
            f.write(b)
    tmp.replace(out_path)
    return out_path


def parse_tcga_group(sample_id: str) -> str | None:
    parts = str(sample_id).split("-")
    if len(parts) < 4:
        return None
    code = parts[3][:3].upper()
    if code == "01A":
        return "Tumor"
    if code == "11A":
        return "Normal"
    return None


def load_sample_labels(sample_groups_path: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = pd.read_parquet(sample_groups_path)
    if "matrix_col" not in df.columns:
        df = df.copy()
        df["matrix_col"] = np.arange(len(df), dtype=np.int64)
    g = df["group"].astype(str).str.lower()
    keep = g.isin(["tumor", "primary tumor", "1", "normal", "solid tissue normal", "0"])
    df = df.loc[keep].copy().reset_index(drop=True)
    g = df["group"].astype(str).str.lower()
    y = g.isin(["tumor", "primary tumor", "1"]).astype(np.int8).to_numpy()
    cols = df["matrix_col"].to_numpy(dtype=np.int64)
    return y, cols, df


def detect_orientation(matrix: np.ndarray, n_probes: int | None = None, n_samples: int | None = None,
                       max_probe_row: int | None = None, max_sample_col: int | None = None) -> str:
    if n_probes is not None and n_samples is not None:
        if matrix.shape == (n_probes, n_samples):
            return "probes_by_samples"
        if matrix.shape == (n_samples, n_probes):
            return "samples_by_probes"
    if max_probe_row is not None and max_sample_col is not None:
        if matrix.shape[0] > max_probe_row and matrix.shape[1] > max_sample_col:
            return "probes_by_samples"
        if matrix.shape[1] > max_probe_row and matrix.shape[0] > max_sample_col:
            return "samples_by_probes"
    raise ValueError(f"Cannot detect orientation from shape={matrix.shape}")


def extract_matrix(beta_path: Path, probe_rows: np.ndarray, sample_cols: np.ndarray) -> tuple[np.ndarray, str]:
    beta = np.load(beta_path, mmap_mode="r")
    ori = detect_orientation(beta, max_probe_row=int(probe_rows.max()), max_sample_col=int(sample_cols.max()))
    if ori == "probes_by_samples":
        x = np.asarray(beta[np.ix_(probe_rows, sample_cols)].T, dtype=np.float32)
    else:
        x = np.asarray(beta[np.ix_(sample_cols, probe_rows)], dtype=np.float32)
    return x, ori


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if path.suffix == ".gz" else path.open("r", encoding="utf-8", errors="replace")


def find_manifest_skiprows(path: Path) -> int:
    with open_text(path) as f:
        for i, line in enumerate(f):
            if line.startswith("[Assay]"):
                return i + 1
            if i > 100:
                break
    return 0


def pick_col(columns: list[str], candidates: list[str], label: str) -> str:
    lower = {c.lower(): c for c in columns}
    for c in candidates:
        if c in columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    raise ValueError(f"Cannot find {label}; columns={columns[:30]}")


def split_semicolon(value) -> list[str]:
    if value is None or pd.isna(value):
        return []
    out = []
    for x in str(value).replace(",", ";").split(";"):
        x = x.strip()
        if x and x.lower() not in {"nan", "none", "intergenic"}:
            out.append(x.upper())
    return list(dict.fromkeys(out))


def has_promoter(value, regions=("TSS200", "TSS1500")) -> bool:
    if value is None or pd.isna(value):
        return False
    return bool({x.strip() for x in str(value).split(";") if x.strip()} & set(regions))
