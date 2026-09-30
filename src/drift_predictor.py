"""
Early Drift Predictor for Space Electronics Burn-In Screening
==============================================================

Uses early burn-in measurements:
    [value_0h, value_24h]
to predict:
    value_168h (end-of-screen performance)

Key Capabilities:
    1. Early trajectory forecast (168h value) using RandomForestRegressor.
    2. Prediction uncertainty via ensemble tree variance (Approximate Prediction Interval).
    3. Safety slope & distance-to-limit calculation.
    4. Model persistence to models/drift_model.joblib for explainability/counterfactuals.
    5. Showcase B: Current PASS (0h/24h) -> Early DRIFT RISK (168h).
"""

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ABSOLUTE_LIMIT = 100.0  # nA specification limit


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Construct engineered feature matrix from 0h and 24h measurements.
    Features:
        0: value_0h
        1: value_24h
        2: drift_0_24 = value_24h - value_0h
        3: drift_rate_0_24 = drift_0_24 / 24.0
        4: ratio_24_0 = value_24h / value_0h
        5: value_24h ** 2
        6: drift_0_24 ** 2
    """
    v0 = df["value_0h"].values
    v24 = df["value_24h"].values
    drift_0_24 = v24 - v0
    drift_rate_0_24 = drift_0_24 / 24.0
    ratio_24_0 = np.where(np.abs(v0) > 1e-6, v24 / v0, 1.0)
    v24_sq = v24 ** 2
    drift_sq = drift_0_24 ** 2

    X = np.column_stack([
        v0,
        v24,
        drift_0_24,
        drift_rate_0_24,
        ratio_24_0,
        v24_sq,
        drift_sq,
    ])
    return X


def train_and_evaluate_model(
    df: pd.DataFrame,
    seed: int = 42,
) -> tuple:
    """
    Train RandomForestRegressor to predict value_168h from [value_0h, value_24h].
    Evaluates MAE and RMSE on a holdout split, then returns the model and holdout metrics.
    """
    X = build_feature_matrix(df)
    y = df["value_168h"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed
    )

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    test_mae = float(mean_absolute_error(y_test, y_test_pred))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
    test_r2 = float(r2_score(y_test, y_test_pred))

    # Refit on all data for deployment predictions
    full_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
    )
    full_model.fit(X, y)

    metrics = {
        "holdout_mae": round(test_mae, 4),
        "holdout_rmse": round(test_rmse, 4),
        "holdout_r2": round(test_r2, 4),
    }

    return full_model, metrics


def compute_prediction_intervals(
    model: RandomForestRegressor,
    X: np.ndarray,
) -> tuple:
    """
    Calculate Approximate Prediction Interval from ensemble tree distribution.
    Returns:
        (predicted_mean, lower_5th, upper_95th, uncertainty_span)
    """
    # predictions shape: (n_estimators, n_samples)
    tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
    pred_mean = np.mean(tree_preds, axis=0)
    pred_lower = np.percentile(tree_preds, 5, axis=0)
    pred_upper = np.percentile(tree_preds, 95, axis=0)
    uncertainty = pred_upper - pred_lower

    return pred_mean, pred_lower, pred_upper, uncertainty


def run_drift_prediction(
    csv_path: str = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run early drift prediction pipeline.
    Produces data/drift_predictions.csv and saves model to models/drift_model.joblib.
    """
    base = Path(__file__).resolve().parent.parent

    if csv_path is None:
        csv_path = os.path.join(base, "data", "telemetry.csv")

    print(f"[*] Loading burn-in data from {csv_path}...")
    df = pd.read_csv(csv_path)
    n_components = len(df)
    print(f"    {n_components} components found")

    print("[*] Training Early Drift Regressor (0h + 24h -> 168h)...")
    model, metrics = train_and_evaluate_model(df, seed=seed)

    print(f"    Holdout Evaluation:")
    print(f"      MAE:  {metrics['holdout_mae']:.4f} nA")
    print(f"      RMSE: {metrics['holdout_rmse']:.4f} nA")
    print(f"      R2:   {metrics['holdout_r2']:.4f}")

    # Save model
    models_dir = os.path.join(base, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "drift_model.joblib")
    joblib.dump(model, model_path)
    print(f"[OK] Saved drift prediction model -> {model_path}")

    # Generate predictions & uncertainty for all components
    X_all = build_feature_matrix(df)
    pred_mean, pred_lower, pred_upper, uncertainty = compute_prediction_intervals(model, X_all)

    result_df = df.copy()
    result_df["predicted_168h"] = np.round(pred_mean, 4)
    result_df["predicted_168h_lower"] = np.round(pred_lower, 4)
    result_df["predicted_168h_upper"] = np.round(pred_upper, 4)
    result_df["prediction_uncertainty"] = np.round(uncertainty, 4)

    # Drift and slopes
    # predicted_drift = change from 0h to 168h
    result_df["predicted_drift"] = np.round(result_df["predicted_168h"] - result_df["value_0h"], 4)
    # predicted_drift_rate = rate per hour over 168h
    result_df["predicted_drift_rate"] = np.round(result_df["predicted_drift"] / 168.0, 6)

    # safety_slope: maximum linear slope from 0h that stays under limit at 168h
    # (ABSOLUTE_LIMIT - value_0h) / 168.0
    safe_margin_rate = (ABSOLUTE_LIMIT - result_df["value_0h"]) / 168.0
    result_df["safety_slope"] = np.round(safe_margin_rate, 6)

    # distance_to_limit: margin between predicted 168h and absolute specification limit
    result_df["distance_to_limit"] = np.round(ABSOLUTE_LIMIT - result_df["predicted_168h"], 4)

    # Holdout error metrics stored as reference
    result_df["holdout_mae"] = metrics["holdout_mae"]
    result_df["holdout_rmse"] = metrics["holdout_rmse"]

    # Save CSV
    output_path = os.path.join(base, "data", "drift_predictions.csv")
    cols_to_save = [
        "component_id", "lot_id",
        "value_0h", "value_24h", "value_96h", "value_168h",
        "predicted_168h", "predicted_168h_lower", "predicted_168h_upper",
        "predicted_drift", "predicted_drift_rate", "safety_slope",
        "distance_to_limit", "prediction_uncertainty",
        "holdout_mae", "holdout_rmse",
    ]
    for extra in ["defect_type", "defect_name", "ground_truth"]:
        if extra in result_df.columns:
            cols_to_save.append(extra)

    cols_to_save = [c for c in cols_to_save if c in result_df.columns]
    result_df[cols_to_save].to_csv(output_path, index=False)
    print(f"[OK] Saved drift predictions -> {output_path}")

    return result_df


def showcase_b(result_df: pd.DataFrame = None):
    """
    Showcase B: Current PASS at 0h/24h -> Early DRIFT RISK at 168h.
    Demonstrates catching degradation BEFORE the absolute limit is breached.
    """
    if result_df is None:
        base = Path(__file__).resolve().parent.parent
        result_df = pd.read_csv(os.path.join(base, "data", "drift_predictions.csv"))

    print(f"\n{'='*70}")
    print(f"  SHOWCASE B: Current PASS -> Early DRIFT RISK")
    print(f"{'='*70}")

    # Criteria:
    # 1. 0h and 24h are both within absolute limit (< 100 nA)
    # 2. Predicted 168h approaches or exceeds limit (> 80 nA) or upper CI > 100 nA
    current_pass = (result_df["value_0h"] < ABSOLUTE_LIMIT) & (result_df["value_24h"] < ABSOLUTE_LIMIT)
    drift_risk = (result_df["predicted_168h"] >= 80.0) | (result_df["predicted_168h_upper"] >= ABSOLUTE_LIMIT)

    showcase_comps = result_df[current_pass & drift_risk].copy()

    if len(showcase_comps) == 0:
        print("\n  No components matched the showcase criteria.")
    else:
        print(f"\n  Found {len(showcase_comps)} components currently PASSING at 0h & 24h")
        print(f"  whose trajectory indicates EARLY DRIFT RISK by 168h:\n")

        for _, row in showcase_comps.head(8).iterrows():
            print(f"  Component: {row['component_id']} ({row['lot_id']})")
            print(f"    0h measurement:       {row['value_0h']:.2f} nA (SAFE, < {ABSOLUTE_LIMIT})")
            print(f"    24h measurement:      {row['value_24h']:.2f} nA (SAFE, < {ABSOLUTE_LIMIT})")
            print(f"    Predicted 168h:       {row['predicted_168h']:.2f} nA")
            print(f"    Approx Pred Interval: [{row['predicted_168h_lower']:.2f}, {row['predicted_168h_upper']:.2f}] nA")
            print(f"    Drift rate (pred):    {row['predicted_drift_rate']:.4f} nA/h")
            print(f"    Safety slope limit:   {row['safety_slope']:.4f} nA/h")
            print(f"    Actual 168h value:    {row.get('value_168h', 'N/A')}")
            print(f"    Ground truth:         {row.get('defect_name', 'unknown')}")
            print(f"    Conventional (at 24h): PASS")
            print(f"    AI Trajectory Check:   EARLY DRIFT RISK DETECTED")
            print()

    print(f"{'='*70}\n")


def main():
    result_df = run_drift_prediction()
    showcase_b(result_df)


if __name__ == "__main__":
    main()
