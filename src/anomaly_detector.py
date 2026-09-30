"""
Lot-Relative Anomaly Detector for Space Electronics Burn-In
=============================================================

Detects components that are statistically abnormal relative to their
manufacturing lot using the robust Z-score method:

    robust_z = (value - lot_median) / (1.4826 × MAD)

A component can be below the absolute specification limit but still be
a strong outlier within its lot — this is exactly what conventional
screening misses.

Secondary detector: Isolation Forest on engineered features as a
complementary multivariate check.

Output: per-component anomaly_score, is_anomaly flag, lot statistics.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ABSOLUTE_LIMIT = 100.0

CHECKPOINTS = ["0h", "24h", "96h", "168h"]
VALUE_COLS = [f"value_{cp}" for cp in CHECKPOINTS]


# ---------------------------------------------------------------------------
# Robust Z-score
# ---------------------------------------------------------------------------

def compute_robust_z(values: pd.Series) -> tuple:
    """
    Compute robust Z-scores for a series of values within a lot.

    Parameters
    ----------
    values : pd.Series
        The measured values for all components in a lot.

    Returns
    -------
    (robust_z, median, mad) : (pd.Series, float, float)
        robust_z scores, lot median, lot MAD.
    """
    median = values.median()
    mad = np.median(np.abs(values - median))

    if mad < 1e-6:
        # Fallback: if MAD is zero (all values nearly identical), use std
        std = values.std()
        if std < 1e-6:
            return pd.Series(0.0, index=values.index), median, 0.0
        robust_z = (values - median) / std
        return robust_z, median, 0.0

    robust_z = (values - median) / (1.4826 * mad)
    return robust_z, median, mad


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_anomaly_detection(
    csv_path: str = None,
    threshold: float = 3.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run lot-relative anomaly detection on burn-in data.

    Parameters
    ----------
    csv_path : str
        Path to telemetry CSV with lot-based burn-in data.
    threshold : float
        Robust Z-score threshold for flagging anomalies (default: 3.0).
    seed : int
        Random seed for Isolation Forest.

    Returns
    -------
    pd.DataFrame with anomaly scores, lot statistics, and flags.
    """
    base = Path(__file__).resolve().parent.parent

    if csv_path is None:
        csv_path = os.path.join(base, "data", "telemetry.csv")

    print(f"[*] Loading burn-in data from {csv_path}...")
    df = pd.read_csv(csv_path)
    n_components = len(df)
    n_lots = df["lot_id"].nunique()
    print(f"    {n_components} components across {n_lots} lots")

    # --- Step 1: Lot-relative robust Z-scores ---
    print(f"[*] Computing lot-relative robust Z-scores (threshold={threshold})...")

    for cp in CHECKPOINTS:
        val_col = f"value_{cp}"
        z_col = f"robust_z_{cp}"
        med_col = f"lot_median_{cp}"
        mad_col = f"lot_mad_{cp}"

        df[z_col] = 0.0
        df[med_col] = 0.0
        df[mad_col] = 0.0

        for lot_id, group in df.groupby("lot_id"):
            if val_col not in group.columns:
                continue
            z_scores, median, mad = compute_robust_z(group[val_col])
            df.loc[group.index, z_col] = z_scores
            df.loc[group.index, med_col] = median
            df.loc[group.index, mad_col] = mad

    # Max absolute robust Z across all checkpoints
    z_cols = [f"robust_z_{cp}" for cp in CHECKPOINTS]
    df["robust_z_max"] = df[z_cols].abs().max(axis=1)
    df["worst_checkpoint"] = df[z_cols].abs().idxmax(axis=1).str.replace("robust_z_", "value_")

    # --- Step 2: Isolation Forest (secondary detector) ---
    print("[*] Training Isolation Forest (secondary detector)...")

    # Feature engineering for IForest
    df["drift_0_24"] = df["value_24h"] - df["value_0h"]
    df["drift_24_168"] = df["value_168h"] - df["value_24h"]
    df["drift_0_168"] = df["value_168h"] - df["value_0h"]
    df["drift_rate_0_24"] = df["drift_0_24"] / 24.0

    iforest_features = VALUE_COLS + ["drift_0_24", "drift_24_168", "drift_0_168"]
    X_if = df[iforest_features].fillna(0).values

    scaler = StandardScaler()
    X_if_scaled = scaler.fit_transform(X_if)

    iforest = IsolationForest(
        n_estimators=150,
        contamination=0.15,
        random_state=seed,
        n_jobs=-1,
    )
    iforest.fit(X_if_scaled)

    # Score: negate decision_function so higher = more anomalous, then normalize to [0,1]
    raw_if_scores = -iforest.decision_function(X_if_scaled)
    mm = MinMaxScaler()
    df["iforest_score"] = mm.fit_transform(raw_if_scores.reshape(-1, 1)).ravel()

    # --- Step 3: Combine scores ---
    # Normalize robust_z_max to [0,1] for combination
    rz = df["robust_z_max"].values
    rz_min, rz_max = rz.min(), rz.max()
    if rz_max - rz_min > 1e-10:
        df["robust_z_normalized"] = (rz - rz_min) / (rz_max - rz_min)
    else:
        df["robust_z_normalized"] = 0.0

    # anomaly_score = the raw robust Z max (interpretable)
    # final_anomaly_score = max of normalized robust Z and iforest score
    df["anomaly_score"] = df["robust_z_max"].round(4)
    df["final_anomaly_score"] = df[["robust_z_normalized", "iforest_score"]].max(axis=1).round(4)

    # Flag based on robust Z threshold
    df["is_anomaly"] = df["robust_z_max"] > threshold

    # --- Step 4: Save ---
    output_path = os.path.join(base, "data", "anomaly_scores.csv")
    out_cols = [
        "component_id", "lot_id",
        "anomaly_score", "is_anomaly", "final_anomaly_score",
        "iforest_score", "robust_z_normalized",
    ]
    for cp in CHECKPOINTS:
        out_cols.extend([f"lot_median_{cp}", f"lot_mad_{cp}", f"robust_z_{cp}"])
    out_cols.append("worst_checkpoint")

    # Add value columns and ground truth
    for vc in VALUE_COLS:
        if vc in df.columns:
            out_cols.append(vc)
    for gt in ["defect_type", "defect_name", "ground_truth"]:
        if gt in df.columns:
            out_cols.append(gt)

    # Deduplicate
    out_cols = list(dict.fromkeys(out_cols))
    out_cols = [c for c in out_cols if c in df.columns]

    df[out_cols].to_csv(output_path, index=False)
    print(f"[OK] Saved anomaly scores -> {output_path}")

    # --- Print summary ---
    n_flagged = df["is_anomaly"].sum()
    print(f"\n    Flagged anomalous: {n_flagged}/{n_components} "
          f"({n_flagged/n_components*100:.1f}%)")

    if "ground_truth" in df.columns:
        true_defective = df["ground_truth"] == "defective"
        flagged = df["is_anomaly"]
        tp = (true_defective & flagged).sum()
        fp = (~true_defective & flagged).sum()
        fn = (true_defective & ~flagged).sum()
        tn = (~true_defective & ~flagged).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"\n    --- vs Ground Truth (lot-relative Z-score only) ---")
        print(f"    TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        print(f"    Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}")

    return df[out_cols]


def showcase_a(result: pd.DataFrame = None):
    """
    Showcase A: Components that PASS absolute limit but FAIL lot-relative check.

    This demonstrates the core value proposition of lot-relative anomaly detection.
    """
    if result is None:
        base = Path(__file__).resolve().parent.parent
        result = pd.read_csv(os.path.join(base, "data", "anomaly_scores.csv"))

    print(f"\n{'='*70}")
    print(f"  SHOWCASE A: Absolute PASS -> Lot-Relative ANOMALY")
    print(f"{'='*70}")

    # Components that are under the absolute limit but flagged as lot anomalies
    has_168h = "value_168h" in result.columns
    if has_168h:
        under_limit = result["value_168h"] < ABSOLUTE_LIMIT
    else:
        under_limit = pd.Series(True, index=result.index)

    flagged = result[under_limit & (result["is_anomaly"] == True)].copy()

    if len(flagged) == 0:
        print("\n  No components matched the showcase criteria.")
    else:
        print(f"\n  Found {len(flagged)} components that PASS absolute limit")
        print(f"  but are FLAGGED as lot-relative anomalies:\n")

        for _, row in flagged.head(10).iterrows():
            val_168h = row.get("value_168h", "N/A")
            lot_med = row.get("lot_median_168h", "N/A")
            robust_z = row.get("robust_z_168h", row.get("anomaly_score", 0))
            defect = row.get("defect_name", "unknown")

            print(f"  {row['component_id']} ({row['lot_id']})")
            if isinstance(val_168h, (int, float)):
                print(f"    Value (168h):    {val_168h:.2f} nA  (< {ABSOLUTE_LIMIT} -> absolute PASS)")
            if isinstance(lot_med, (int, float)):
                print(f"    Lot median:      {lot_med:.2f} nA")
            print(f"    Robust Z-score:  {robust_z:.2f}")
            print(f"    Anomaly score:   {row['anomaly_score']:.2f}")
            print(f"    Ground truth:    {defect}")
            print(f"    Conventional:    PASS")
            print(f"    AI-assisted:     FLAGGED [ANOMALY]")
            print()

    print(f"{'='*70}\n")


def main():
    result = run_anomaly_detection()
    showcase_a(result)


if __name__ == "__main__":
    main()
