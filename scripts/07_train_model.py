#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.data_utils import ensure_project_dirs, extract_matrix, load_config, load_sample_labels, log, resolve_path, save_json
from utils.model_utils import classification_metrics
from utils.plot_utils import confusion_grid, roc_plot


def models(cfg):
    rs, nj = cfg["project"]["random_state"], cfg["project"]["n_jobs"]
    m = cfg["models"]
    return {
        "LogisticRegression": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=m["logistic_max_iter"]))]),
        "RandomForest": RandomForestClassifier(n_estimators=m["random_forest_n_estimators"], random_state=rs, n_jobs=nj, max_features="sqrt"),
        "XGBoost": XGBClassifier(n_estimators=m["xgboost_n_estimators"], random_state=rs, n_jobs=nj, eval_metric="logloss", objective="binary:logistic", tree_method="hist"),
        "SVM_RBF": Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel=m["svm_kernel"], probability=True))]),
    }


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config); ensure_project_dirs(cfg)
    processed, tables, figs, results = [resolve_path(cfg, k) for k in ["processed_dir", "tables_dir", "figures_dir", "results_dir"]]
    rs, mcfg = cfg["project"]["random_state"], cfg["models"]
    y, sample_cols, _ = load_sample_labels(processed / "ucec_hm450_clean_sample_groups.parquet")
    markers = pd.read_parquet(tables / "final_10_methylation_markers.parquet").sort_values("final_marker_rank")
    x, _ = extract_matrix(processed / "ucec_hm450_beta_clean_probes_by_samples.npy", markers["matrix_row"].to_numpy(np.int64), sample_cols)
    train, test = train_test_split(np.arange(len(y)), test_size=mcfg["test_size"], stratify=y, random_state=rs)
    rows, cv_rows, roc_scores, cm_map = [], [], {}, {}
    cv = StratifiedKFold(n_splits=mcfg["cv"], shuffle=True, random_state=rs)
    for name, model in models(cfg).items():
        log(f"Training {name}")
        model.fit(x[train], y[train])
        score = model.predict_proba(x[test])[:, 1]
        pred = model.predict(x[test])
        row = {"model": name, **classification_metrics(y[test], pred, score)}
        cv_score = cross_val_score(model, x, y, cv=cv, scoring="roc_auc", n_jobs=1)
        row["cv_auc_mean"] = float(cv_score.mean()); row["cv_auc_std"] = float(cv_score.std(ddof=1))
        rows.append(row)
        cv_rows.append({"model": name, "cv_auc_mean": row["cv_auc_mean"], "cv_auc_std": row["cv_auc_std"], "cv_auc_min": float(cv_score.min()), "cv_auc_max": float(cv_score.max())})
        roc_scores[name] = score
        cm_map[name] = confusion_matrix(y[test], pred, labels=[0, 1])
    metrics = pd.DataFrame(rows).sort_values("AUC", ascending=False)
    cv_sum = pd.DataFrame(cv_rows).sort_values("cv_auc_mean", ascending=False)
    metrics.to_parquet(tables / "model_test_metrics.parquet", index=False)
    cv_sum.to_parquet(tables / "model_5fold_cv_auc_summary.parquet", index=False)
    pd.DataFrame([{"model": k, "TN": int(v[0,0]), "FP": int(v[0,1]), "FN": int(v[1,0]), "TP": int(v[1,1])} for k, v in cm_map.items()]).to_parquet(tables / "model_confusion_matrix_counts.parquet", index=False)
    roc_plot(y[test], roc_scores, figs / "model_roc_curves_comparison.png", figs / "model_roc_curves_comparison.pdf", "Four-model ROC comparison")
    confusion_grid(cm_map, figs / "model_confusion_matrices_2x2.png", figs / "model_confusion_matrices_2x2.pdf", "Confusion matrices")
    lr_auc = float(metrics.loc[metrics["model"].eq("LogisticRegression"), "AUC"].iloc[0]); best = float(metrics["AUC"].max())
    rec = "推荐使用逻辑回归：AUC差距<0.03且可解释性强" if best - lr_auc < 0.03 else "逻辑回归不是最优：AUC差距>=0.03"
    save_json({"metrics": rows, "cv": cv_rows, "recommendation": rec}, results / "model_evaluation_report.json")
    log("07_train_model complete")


if __name__ == "__main__":
    main()
