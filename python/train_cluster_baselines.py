"""Train and compare classical ML baselines on Review 2 vs Review 3 Clustered MANET dataset.

Review 3 - Raghav Bajaj
Responsibility: Dataset, Clustering, Cluster-Head Selection and Baselines

This script:
1. Implements the exact chronological split (0-340s train, 350-500s test).
2. Trains RandomForest, SVM, and XGBoost on:
   - Review 2 Features: [drop_rate, trust, forward_ratio, energy]
   - Review 3 Clustered Features: [drop_rate, trust, forward_ratio, energy, cluster_size, distance_to_ch, is_cluster_head]
3. Evaluates Accuracy, Malicious Precision, Malicious Recall, and Malicious F1 across all 5 runs.
4. Produces:
   - results/cluster_baseline_results.csv (per-run metrics)
   - results/cluster_baseline_summary.csv (mean +/- std summary comparison)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


R2_FEATURES = [
    "drop_rate",
    "trust",
    "forward_ratio",
    "energy"
]

R3_FEATURES = [
    "drop_rate",
    "trust",
    "forward_ratio",
    "energy",
    "cluster_size",
    "distance_to_ch",
    "is_cluster_head"
]


def metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[1],
        average=None,
        zero_division=0
    )
    return acc, float(p[0]), float(r[0]), float(f1[0])


def split_temporal(run_df: pd.DataFrame):
    """Strict chronological split: past 70% timestamps (0-340s) train, future 30% (350-500s) test."""
    times = sorted(run_df["time"].unique())
    cut = int(len(times) * 0.70)
    train_times = times[:cut]
    test_times = times[cut:]
    
    train = run_df[run_df["time"].isin(train_times)]
    test = run_df[run_df["time"].isin(test_times)]
    
    if test["label"].nunique() < 2:
        raise ValueError("Temporal test set must contain both normal and malicious nodes!")
        
    return train, test


def get_models(run_id: int):
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=250,
            random_state=int(run_id),
            class_weight="balanced"
        ),
        "SVM": make_pipeline(
            StandardScaler(),
            SVC(
                C=2.0,
                gamma="scale",
                class_weight="balanced"
            )
        ),
        "XGBoost": XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=int(run_id)
        )
    }


def evaluate_feature_set(df: pd.DataFrame, features: list[str], feature_set_name: str) -> list[dict]:
    results = []
    runs = sorted(df["run"].unique())
    
    print(f"\n--- Evaluating Feature Set: {feature_set_name} (Features: {features}) ---")
    
    for run_id in runs:
        r_df = df[df["run"] == run_id]
        train, test = split_temporal(r_df)
        
        Xtr = train[features].to_numpy(dtype=float)
        ytr = train["label"].to_numpy(dtype=int)
        
        Xte = test[features].to_numpy(dtype=float)
        yte = test["label"].to_numpy(dtype=int)
        
        models = get_models(run_id)
        
        for model_name, model in models.items():
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            acc, p, rec, f1 = metrics(yte, pred)
            
            results.append({
                "feature_set": feature_set_name,
                "run": int(run_id),
                "model": model_name,
                "accuracy": acc,
                "mal_precision": p,
                "mal_recall": rec,
                "mal_f1": f1
            })
            print(f"  Run {run_id} | {model_name:12s} | Acc: {acc*100:.2f}% | Prec: {p*100:.2f}% | Rec: {rec*100:.2f}% | F1: {f1*100:.2f}%")
            
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate classical ML baselines on Review 3 clustered dataset.")
    parser.add_argument(
        "--nodes",
        default="data/processed_clustered/nodes_clustered.csv",
        help="Path to nodes_clustered.csv"
    )
    parser.add_argument(
        "--out-results",
        default="results/cluster_baseline_results.csv",
        help="Output path for detailed results CSV"
    )
    parser.add_argument(
        "--out-summary",
        default="results/cluster_baseline_summary.csv",
        help="Output path for summary CSV"
    )
    args = parser.parse_args()
    
    print(f"Loading cluster-aware dataset from: {args.nodes}")
    df = pd.read_csv(args.nodes)
    
    all_results = []
    
    # 1. Review 2 Feature Set (Baseline comparator)
    r2_res = evaluate_feature_set(df, R2_FEATURES, "Review2_Unclustered")
    all_results.extend(r2_res)
    
    # 2. Review 3 Clustered Feature Set
    r3_res = evaluate_feature_set(df, R3_FEATURES, "Review3_Clustered")
    all_results.extend(r3_res)
    
    res_df = pd.DataFrame(all_results)
    
    out_res_path = Path(args.out_results)
    out_res_path.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_res_path, index=False)
    print(f"\nSaved detailed run results to: {out_res_path}")
    
    # Generate summary: mean +/- std for each model and feature set
    summary = res_df.groupby(["feature_set", "model"])[
        ["accuracy", "mal_precision", "mal_recall", "mal_f1"]
    ].agg(["mean", "std"])
    
    print("\n" + "="*80)
    print("REVIEW 2 vs REVIEW 3 BASELINE COMPARISON SUMMARY (MEAN +/- STD)")
    print("="*80)
    print(summary)
    
    out_sum_path = Path(args.out_summary)
    summary.to_csv(out_sum_path)
    print(f"\nSaved summary comparison to: {out_sum_path}")


if __name__ == "__main__":
    main()
