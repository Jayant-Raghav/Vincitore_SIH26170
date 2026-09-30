"""
Evaluation Module for Space Electronics Burn-In QA System
==========================================================

Computes comprehensive reliability metrics for the full burn-in pipeline:
    1. Lot-Relative Anomaly Detection: Precision, Recall, F1, FNR
    2. Early Drift Prediction (0h+24h -> 168h): MAE, RMSE, R2
    3. Risk Engine (PASS / HOLD / REJECT):
       - Binary screening (Caught vs Missed)
       - Critical False Negative Rate (must be 0% for flight safety)
       - Engineering review overhead (HOLD rate)
       - Per-defect-type catch rates
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)


def evaluate_anomaly_detection(anomaly_path: str = None) -> dict:
    """Evaluate lot-relative anomaly detector performance."""
    base = Path(__file__).resolve().parent.parent
    if anomaly_path is None:
        anomaly_path = os.path.join(base, "data", "anomaly_scores.csv")

    df = pd.read_csv(anomaly_path)
    if "defect_type" not in df.columns:
        return {"error": "No ground truth labels available"}

    y_true = (df["defect_type"] > 0).astype(int)
    y_pred = df["is_anomaly"].astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fnr": round(fnr, 4),
        "total": len(df),
        "flagged": int(y_pred.sum()),
    }


def evaluate_drift_prediction(drift_path: str = None) -> dict:
    """Evaluate early drift prediction accuracy for 168h forecast."""
    base = Path(__file__).resolve().parent.parent
    if drift_path is None:
        drift_path = os.path.join(base, "data", "drift_predictions.csv")

    df = pd.read_csv(drift_path)
    y_true = df["value_168h"].values
    y_pred = df["predicted_168h"].values

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    # Holdout metrics saved during training
    holdout_mae = float(df["holdout_mae"].iloc[0]) if "holdout_mae" in df.columns else mae
    holdout_rmse = float(df["holdout_rmse"].iloc[0]) if "holdout_rmse" in df.columns else rmse

    # Catching limit violations: actual > 100 nA
    exceeded_spec = y_true >= 100.0
    predicted_risk = (y_pred >= 80.0) | (df["predicted_168h_upper"] >= 100.0)

    if exceeded_spec.sum() > 0:
        spec_caught = int((exceeded_spec & predicted_risk).sum())
        spec_total = int(exceeded_spec.sum())
        spec_recall = spec_caught / spec_total
    else:
        spec_caught, spec_total, spec_recall = 0, 0, 1.0

    return {
        "overall_mae": round(mae, 4),
        "overall_rmse": round(rmse, 4),
        "overall_r2": round(r2, 4),
        "holdout_mae": round(holdout_mae, 4),
        "holdout_rmse": round(holdout_rmse, 4),
        "spec_violations_total": spec_total,
        "spec_violations_caught": spec_caught,
        "spec_violation_catch_rate": round(spec_recall, 4),
    }


def evaluate_risk_engine(decisions_path: str = None) -> dict:
    """Evaluate full risk engine classification performance."""
    base = Path(__file__).resolve().parent.parent
    if decisions_path is None:
        decisions_path = os.path.join(base, "data", "risk_decisions.csv")

    df = pd.read_csv(decisions_path)
    if "defect_type" not in df.columns:
        return {"error": "No ground truth labels available"}

    # Binary: defective caught (HOLD or REJECT) vs missed (PASS)
    y_true_binary = (df["defect_type"] > 0).astype(int)
    y_pred_caught = (df["decision"] != "PASS").astype(int)

    tp = int(((y_true_binary == 1) & (y_pred_caught == 1)).sum())
    fp = int(((y_true_binary == 0) & (y_pred_caught == 1)).sum())
    fn = int(((y_true_binary == 1) & (y_pred_caught == 0)).sum())
    tn = int(((y_true_binary == 0) & (y_pred_caught == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    decision_counts = df["decision"].value_counts().to_dict()

    # Per-defect-type analysis
    defect_analysis = {}
    if "defect_name" in df.columns:
        for defect, grp in df.groupby("defect_name"):
            total = len(grp)
            n_pass = int((grp["decision"] == "PASS").sum())
            n_hold = int((grp["decision"] == "HOLD").sum())
            n_reject = int((grp["decision"] == "REJECT").sum())
            catch_rate = (n_hold + n_reject) / total if total > 0 else 0.0
            defect_analysis[defect] = {
                "total": total,
                "PASS": n_pass,
                "HOLD": n_hold,
                "REJECT": n_reject,
                "catch_rate": round(catch_rate, 4),
            }

    return {
        "binary": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "fnr": round(fnr, 4),
        },
        "decision_counts": decision_counts,
        "defect_analysis": defect_analysis,
    }


def print_evaluation_report():
    """Print the formatted evaluation report for SIH."""
    print(f"\n{'='*70}")
    print(f"  FULL EVALUATION REPORT")
    print(f"  Space Electronics AI Quality Assurance (SIH26170)")
    print(f"{'='*70}")

    # 1. Anomaly Detection
    print(f"\n  1. LOT-RELATIVE ANOMALY DETECTION (Robust Z-Score + MAD)")
    print(f"  {'-'*50}")
    ad = evaluate_anomaly_detection()
    if "error" not in ad:
        print(f"  {'Metric':<25s} {'Value':>10s}")
        print(f"  {'.'*25} {'.'*10}")
        print(f"  {'Precision':<25s} {ad['precision']:>10.4f}")
        print(f"  {'Recall':<25s} {ad['recall']:>10.4f}")
        print(f"  {'F1 Score':<25s} {ad['f1']:>10.4f}")
        print(f"  {'False Negative Rate':<25s} {ad['fnr']:>10.4f}")
        print(f"  {'True Positives':<25s} {ad['tp']:>10d}")
        print(f"  {'False Positives':<25s} {ad['fp']:>10d}")
        print(f"  {'False Negatives':<25s} {ad['fn']:>10d}")
        print(f"  {'True Negatives':<25s} {ad['tn']:>10d}")

    # 2. Drift Prediction
    print(f"\n  2. EARLY DRIFT PREDICTION (0h+24h -> 168h Forecast)")
    print(f"  {'-'*50}")
    dp = evaluate_drift_prediction()
    print(f"  {'Metric':<30s} {'Value':>10s}")
    print(f"  {'.'*30} {'.'*10}")
    print(f"  {'Overall MAE (nA)':<30s} {dp['overall_mae']:>10.4f}")
    print(f"  {'Overall RMSE (nA)':<30s} {dp['overall_rmse']:>10.4f}")
    print(f"  {'Overall R2 Score':<30s} {dp['overall_r2']:>10.4f}")
    print(f"  {'Holdout Test MAE (nA)':<30s} {dp['holdout_mae']:>10.4f}")
    print(f"  {'Spec Violations Catch Rate':<30s} {dp['spec_violation_catch_rate']*100:>9.1f}%")

    # 3. Risk Engine
    print(f"\n  3. RISK ENGINE DECISION RELIABILITY")
    print(f"  {'-'*50}")
    re = evaluate_risk_engine()
    if "error" not in re:
        binary = re["binary"]
        print(f"  Binary (Caught in HOLD/REJECT vs Missed in PASS):")
        print(f"  {'Metric':<25s} {'Value':>10s}")
        print(f"  {'.'*25} {'.'*10}")
        print(f"  {'Screening Recall':<25s} {binary['recall']*100:>9.2f}%")
        print(f"  {'Screening Precision':<25s} {binary['precision']:>10.4f}")
        print(f"  {'Screening F1 Score':<25s} {binary['f1']:>10.4f}")
        print(f"  {'False Negative Rate':<25s} {binary['fnr']*100:>9.2f}%")
        print(f"  {'(FNR = uncaught defects)':<25s}")

        print(f"\n  Decision Distribution:")
        for dec, count in re["decision_counts"].items():
            print(f"    {dec:<10s}  {count:>4d}")

        print(f"\n  Per-Defect-Type Breakdown:")
        print(f"  {'Defect Type':<22s} {'Total':>6s} {'PASS':>6s} {'HOLD':>6s} {'REJECT':>6s} {'Catch%':>7s}")
        print(f"  {'-'*22} {'-'*6} {'-'*6} {'-'*6} {'-'*6} {'-'*7}")
        for defect, info in sorted(re["defect_analysis"].items()):
            print(f"  {defect:<22s} {info['total']:>6d} {info['PASS']:>6d} "
                  f"{info['HOLD']:>6d} {info['REJECT']:>6d} "
                  f"{info['catch_rate']*100:>6.1f}%")

    # 4. System Summary
    print(f"\n  4. FLIGHT SAFETY VERIFICATION SUMMARY")
    print(f"  {'-'*50}")
    if "error" not in re:
        binary = re["binary"]
        n_total = binary['tp'] + binary['fp'] + binary['fn'] + binary['tn']
        n_def = binary['tp'] + binary['fn']
        print(f"  Total components screened:   {n_total}")
        print(f"  Defective components:        {n_def}")
        print(f"  Defective caught:            {binary['tp']} ({binary['recall']*100:.1f}%)")
        print(f"  Defective missed (PASS):     {binary['fn']}")
        print(f"  False Negative Rate:         {binary['fnr']*100:.2f}%")
        print(f"  Engineer review overhead:    {binary['fp']} nominal components in HOLD/REJECT")

        if binary["fn"] == 0:
            print(f"\n  >>> ZERO FALSE NEGATIVES -- 100% FLIGHT CRITICAL SAFETY MET! <<<")
        else:
            print(f"\n  >>> WARNING: {binary['fn']} defective components were missed! <<<")

    print(f"{'='*70}\n")


def main():
    print_evaluation_report()


if __name__ == "__main__":
    main()
