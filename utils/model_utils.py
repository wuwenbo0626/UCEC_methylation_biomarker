from __future__ import annotations

import math

import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score


def classification_metrics(y_true, y_pred, y_score) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "AUC": float(roc_auc_score(y_true, y_score)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
        "F1": float(f1_score(y_true, y_pred)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def bootstrap_auc_ci(y_true, y_score, n_boot=2000, seed=42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) == 2:
            vals.append(roc_auc_score(y_true[idx], y_score[idx]))
    return tuple(np.percentile(vals, [2.5, 97.5]).astype(float)) if vals else (float("nan"), float("nan"))


def midrank(x):
    order = np.argsort(x)
    sx = x[order]
    n = len(x)
    r = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and sx[j] == sx[i]:
            j += 1
        r[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = r
    return out


def fast_delong(preds_sorted, n_pos):
    m = n_pos
    n = preds_sorted.shape[1] - m
    pos, neg = preds_sorted[:, :m], preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx, ty, tz = np.empty((k, m)), np.empty((k, n)), np.empty((k, m + n))
    for i in range(k):
        tx[i] = midrank(pos[i])
        ty[i] = midrank(neg[i])
        tz[i] = midrank(preds_sorted[i])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    return aucs, np.atleast_2d(np.cov(v01) / m + np.cov(v10) / n)


def delong_test(y_true, score_a, score_b) -> dict:
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true)
    n_pos = int(y_true.sum())
    aucs, cov = fast_delong(np.vstack([score_a, score_b])[:, order], n_pos)
    diff = float(aucs[0] - aucs[1])
    c = np.array([[1.0, -1.0]])
    var = float((c @ cov @ c.T).item())
    if var <= 0 or not np.isfinite(var):
        p, z = (1.0, 0.0) if abs(diff) < 1e-12 else (0.0, math.inf)
    else:
        z = diff / math.sqrt(var)
        p = 2 * stats.norm.sf(abs(z))
    return {"auc_model_a": float(aucs[0]), "auc_model_b": float(aucs[1]), "auc_difference_a_minus_b": diff, "z": float(z), "p_value": float(p)}
