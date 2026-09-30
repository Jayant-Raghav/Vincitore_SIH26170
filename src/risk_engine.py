"""
Risk Engine for Space Electronics Burn-In Screening
=====================================================

Combines lot-relative anomaly scores, drift predictions, and prediction
uncertainty to produce final PASS / HOLD / REJECT decisions.

Decision Logic
--------------
REJECT if ANY of:
    - anomaly_score > reject_anomaly_threshold (severe lot outlier)
    - predicted_168h > absolute_limit (predicted spec violation)
    - predicted_168h_upper > absolute_limit (upper CI exceeds spec)

HOLD if ANY of:
    - anomaly_score > hold_anomaly_threshold (suspicious lot outlier)
    - predicted_168h > 0.80 * absolute_limit (approaching limit)
    - predicted_drift_rate > safety_slope (unsafe trajectory)
    - prediction_uncertainty > 0.20 * predicted_168h (low confidence)
    - distance_to_limit < 10 (close to spec limit)

PASS otherwise.

Design: false negatives are catastrophically expensive in space
electronics.  HOLD exists so engineers review borderline cases.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ABSOLUTE_LIMIT = 100.0

DEFAULT_CONFIG = {
    # Anomaly thresholds (on the final_anomaly_score from anomaly_detector)
    "reject_anomaly_threshold": 5.0,   # robust-Z > 5 → REJECT
    "hold_anomaly_threshold": 3.0,     # robust-Z > 3 → HOLD

    # Drift / prediction thresholds
    "reject_predicted_threshold": ABSOLUTE_LIMIT,          # predicted > limit -> REJECT
    "hold_predicted_fraction": 0.80,                       # predicted > 80% of limit -> HOLD
    "hold_upper_threshold": 88.0,                          # upper CI > 88 nA -> HOLD
    "hold_distance_to_limit": 15.0,                        # distance < 15 -> HOLD

    # Uncertainty threshold (fraction of predicted value)
    "uncertainty_ratio_threshold": 0.20,

    # Safety slope: if predicted_drift_rate > this, HOLD
    "hold_safety_slope_factor": 1.0,   # drift_rate > safety_slope -> HOLD
}


def classify_component(anomaly_row: pd.Series, drift_row: pd.Series,
                       config: dict = None) -> tuple:
    """
    Classify a single component as PASS, HOLD, or REJECT.

    Parameters
    ----------
    anomaly_row : pd.Series
        Row from anomaly_scores.csv for this component.
    drift_row : pd.Series
        Row from drift_predictions.csv for this component.
    config : dict, optional
        Override default thresholds.

    Returns
    -------
    (decision, reasons) : (str, list[str])
    """
    if config is None:
        config = DEFAULT_CONFIG

    reasons = []
    anomaly_score = anomaly_row.get("anomaly_score", 0.0)
    is_anomaly = anomaly_row.get("is_anomaly", False)

    predicted_168h = drift_row.get("predicted_168h", 0.0)
    predicted_upper = drift_row.get("predicted_168h_upper", predicted_168h)
    predicted_lower = drift_row.get("predicted_168h_lower", predicted_168h)
    predicted_drift_rate = drift_row.get("predicted_drift_rate", 0.0)
    safety_slope = drift_row.get("safety_slope", float("inf"))
    distance_to_limit = drift_row.get("distance_to_limit", ABSOLUTE_LIMIT)
    prediction_uncertainty = drift_row.get("prediction_uncertainty", 0.0)
    value_0h = drift_row.get("value_0h", 0.0)
    value_24h = drift_row.get("value_24h", 0.0)

    # ----------------------------------------------------------------
    # REJECT checks
    # ----------------------------------------------------------------
    reject = False

    # R1: Severe lot-relative anomaly
    if anomaly_score > config["reject_anomaly_threshold"]:
        reject = True
        reasons.append(
            f"REJECT: anomaly_score ({anomaly_score:.2f}) > "
            f"reject threshold ({config['reject_anomaly_threshold']})"
        )

    # R2: Predicted 168h exceeds absolute limit
    if predicted_168h > config["reject_predicted_threshold"]:
        reject = True
        reasons.append(
            f"REJECT: predicted_168h ({predicted_168h:.2f}) > "
            f"absolute limit ({ABSOLUTE_LIMIT})"
        )

    # R3: Upper prediction interval exceeds absolute limit
    if predicted_upper > config["reject_predicted_threshold"]:
        reject = True
        reasons.append(
            f"REJECT: predicted_168h_upper ({predicted_upper:.2f}) > "
            f"absolute limit ({ABSOLUTE_LIMIT})"
        )

    if reject:
        return "REJECT", reasons

    # ----------------------------------------------------------------
    # HOLD checks
    # ----------------------------------------------------------------
    hold = False

    # H1: Suspicious lot-relative anomaly
    if anomaly_score > config["hold_anomaly_threshold"]:
        hold = True
        reasons.append(
            f"HOLD: anomaly_score ({anomaly_score:.2f}) > "
            f"hold threshold ({config['hold_anomaly_threshold']})"
        )

    # H2: Predicted 168h approaches limit
    hold_pred_threshold = config["hold_predicted_fraction"] * ABSOLUTE_LIMIT
    if predicted_168h > hold_pred_threshold:
        hold = True
        reasons.append(
            f"HOLD: predicted_168h ({predicted_168h:.2f}) > "
            f"{config['hold_predicted_fraction']:.0%} of limit ({hold_pred_threshold:.1f})"
        )

    # H2b: Upper prediction interval approaches limit
    hold_upper_thresh = config.get("hold_upper_threshold", 88.0)
    if predicted_upper > hold_upper_thresh:
        hold = True
        reasons.append(
            f"HOLD: upper prediction interval ({predicted_upper:.2f}) > "
            f"warning threshold ({hold_upper_thresh:.1f})"
        )

    # H3: Drift rate exceeds safety slope
    if safety_slope > 0 and predicted_drift_rate > safety_slope * config["hold_safety_slope_factor"]:
        hold = True
        reasons.append(
            f"HOLD: drift_rate ({predicted_drift_rate:.4f}/h) > "
            f"safety_slope ({safety_slope:.4f}/h)"
        )

    # H4: High prediction uncertainty
    if abs(predicted_168h) > 1e-10:
        unc_ratio = prediction_uncertainty / abs(predicted_168h)
        if unc_ratio > config["uncertainty_ratio_threshold"]:
            hold = True
            reasons.append(
                f"HOLD: uncertainty ({unc_ratio:.1%} of predicted) > "
                f"{config['uncertainty_ratio_threshold']:.0%} threshold"
            )

    # H5: Close to specification limit
    if distance_to_limit < config["hold_distance_to_limit"]:
        hold = True
        reasons.append(
            f"HOLD: distance to limit ({distance_to_limit:.2f}) < "
            f"minimum margin ({config['hold_distance_to_limit']})"
        )

    if hold:
        return "HOLD", reasons

    # ----------------------------------------------------------------
    # PASS
    # ----------------------------------------------------------------
    reasons.append("PASS: all checks clear — nominal behavior, safe trajectory, sufficient margin")
    return "PASS", reasons


def run_risk_engine(
    anomaly_path: str = None,
    drift_path: str = None,
    config: dict = None,
) -> pd.DataFrame:
    """
    Run the full risk classification pipeline.

    Returns DataFrame with decision, reasons, and all input data merged.
    """
    base = Path(__file__).resolve().parent.parent

    if anomaly_path is None:
        anomaly_path = os.path.join(base, "data", "anomaly_scores.csv")
    if drift_path is None:
        drift_path = os.path.join(base, "data", "drift_predictions.csv")

    print(f"[*] Loading anomaly scores from {anomaly_path}...")
    anomaly_df = pd.read_csv(anomaly_path)

    print(f"[*] Loading drift predictions from {drift_path}...")
    drift_df = pd.read_csv(drift_path)

    # Merge on component_id
    merged = anomaly_df.merge(drift_df, on="component_id", suffixes=("_anom", "_drift"))

    # Resolve duplicate columns
    for col in ["lot_id", "defect_type", "defect_name", "ground_truth",
                "value_0h", "value_24h", "value_96h", "value_168h"]:
        col_anom = f"{col}_anom"
        col_drift = f"{col}_drift"
        if col_anom in merged.columns:
            merged[col] = merged[col_anom]
            merged.drop(columns=[col_anom], inplace=True)
        if col_drift in merged.columns:
            if col not in merged.columns:
                merged[col] = merged[col_drift]
            merged.drop(columns=[col_drift], inplace=True)

    print(f"[*] Classifying {len(merged)} components...")
    decisions = []
    all_reasons = []

    for idx, row in merged.iterrows():
        decision, reasons = classify_component(
            anomaly_row=row,
            drift_row=row,
            config=config,
        )
        decisions.append(decision)
        all_reasons.append(" | ".join(reasons))

    merged["decision"] = decisions
    merged["decision_reasons"] = all_reasons

    # Save
    output_path = os.path.join(base, "data", "risk_decisions.csv")
    save_cols = ["component_id", "lot_id", "anomaly_score", "is_anomaly",
                 "predicted_168h", "predicted_168h_lower", "predicted_168h_upper",
                 "predicted_drift_rate", "safety_slope", "distance_to_limit",
                 "prediction_uncertainty",
                 "decision", "decision_reasons"]
    # Add value columns if present
    for vc in ["value_0h", "value_24h", "value_96h", "value_168h"]:
        if vc in merged.columns:
            save_cols.append(vc)
    # Add ground truth if present
    for gt in ["defect_type", "defect_name", "ground_truth"]:
        if gt in merged.columns:
            save_cols.append(gt)

    # Only keep columns that actually exist
    save_cols = [c for c in save_cols if c in merged.columns]
    merged[save_cols].to_csv(output_path, index=False)
    print(f"[OK] Saved risk decisions -> {output_path}")

    # --- Print summary ---
    print(f"\n{'='*70}")
    print(f"  RISK ENGINE: Classification Results")
    print(f"{'='*70}")

    dec_counts = merged["decision"].value_counts()
    total = len(merged)
    for dec in ["PASS", "HOLD", "REJECT"]:
        count = dec_counts.get(dec, 0)
        pct = count / total * 100
        bar = "#" * int(pct / 2)
        print(f"  {dec:<8s}  {count:>4d}  ({pct:5.1f}%)  {bar}")

    if "ground_truth" in merged.columns:
        print(f"\n  --- Decision vs Ground Truth ---")
        print(f"  {'Decision':<10s} {'Nominal':>8s} {'Defective':>10s}")
        print(f"  {'-'*10} {'-'*8} {'-'*10}")
        for dec in ["PASS", "HOLD", "REJECT"]:
            subset = merged[merged["decision"] == dec]
            nom = (subset["ground_truth"] == "nominal").sum()
            defective = (subset["ground_truth"] == "defective").sum()
            print(f"  {dec:<10s} {nom:>8d} {defective:>10d}")

        # Critical metric: defective components that got PASS
        false_pass = merged[(merged["decision"] == "PASS") &
                            (merged["ground_truth"] == "defective")]
        print(f"\n  CRITICAL -- Defective components that got PASS: "
              f"{len(false_pass)}")
        if len(false_pass) > 0:
            for _, row in false_pass.head(10).iterrows():
                print(f"    {row['component_id']}  "
                      f"anomaly={row.get('anomaly_score', '?'):.4f}  "
                      f"predicted_168h={row.get('predicted_168h', '?'):.2f}  "
                      f"type={row.get('defect_name', '?')}")
        else:
            print(f"    >>> ZERO FALSE NEGATIVES -- All defective components caught! <<<")

    print(f"\n{'='*70}\n")

    return merged


def main():
    run_risk_engine()


if __name__ == "__main__":
    main()
