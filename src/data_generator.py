"""
Synthetic Burn-In Data Generator for Space Electronics
=======================================================

Generates realistic burn-in test data for space-grade electronic components
organized by manufacturing lots. Each component is measured at four burn-in
checkpoints: 0h, 24h, 96h, 168h.

The measured parameter is leakage current (nA) with an absolute
specification limit of 100 nA.

Output columns:
    component_id   — unique ID (e.g. "COMP-00001")
    lot_id         — manufacturing lot (e.g. "LOT-001")
    value_0h       — measurement at 0 hours
    value_24h      — measurement at 24 hours
    value_96h      — measurement at 96 hours
    value_168h     — measurement at 168 hours
    defect_type    — int: 0=nominal, 1-7 for defect categories
    defect_name    — human-readable defect label
    ground_truth   — "nominal" or "defective"

Defect types:
    0 — nominal            (healthy, stable values)
    1 — gradual_drift      (slow linear increase toward limit)
    2 — accelerating_drift (exponential/accelerating increase)
    3 — noisy_drift        (drift with high noise overlay)
    4 — sudden_failure     (normal then jumps at 96h or 168h)
    5 — latent_defect      (looks normal at 0h/24h, fails at 168h)
    6 — lot_outlier        (within spec but far from lot peers)
    7 — borderline         (hovers near the specification limit)

Usage:
    python -m src.data_generator
    python -m src.data_generator --n_lots 30 --components_per_lot 20 --seed 42
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

ABSOLUTE_LIMIT = 100.0   # nA — specification limit

CHECKPOINTS = [0, 24, 96, 168]  # hours

DEFECT_NAMES = {
    0: "nominal",
    1: "gradual_drift",
    2: "accelerating_drift",
    3: "noisy_drift",
    4: "sudden_failure",
    5: "latent_defect",
    6: "lot_outlier",
    7: "borderline",
}


# ---------------------------------------------------------------------------
# Helper: generate a single component trajectory
# ---------------------------------------------------------------------------

def _nominal_trajectory(lot_mean: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a 4-point nominal trajectory with slight variation and noise."""
    component_offset = rng.normal(0, 2.0)  # component-to-component variation
    base = lot_mean + component_offset
    noise = rng.normal(0, 0.8, size=4)     # measurement noise
    # Slight natural drift (0.01-0.03 nA/h is normal aging)
    natural_drift = np.array([0, 24, 96, 168]) * rng.uniform(0.001, 0.005)
    values = base + noise + natural_drift
    return np.clip(values, 0.1, None)


def _gradual_drift(lot_mean: float, rng: np.random.Generator) -> np.ndarray:
    """Defect 1: slow linear increase toward the limit."""
    base = lot_mean + rng.normal(0, 1.5)
    noise = rng.normal(0, 0.8, size=4)
    # Linear drift rate chosen so 168h value reaches 85-115 nA
    drift_rate = rng.uniform(0.16, 0.28)  # nA per hour
    drift = np.array([0, 24, 96, 168]) * drift_rate
    values = base + noise + drift
    return np.clip(values, 0.1, None)


def _accelerating_drift(lot_mean: float, rng: np.random.Generator) -> np.ndarray:
    """Defect 2: exponential/accelerating increase — looks safe at 24h, bad at 168h."""
    base = lot_mean + rng.normal(0, 1.5)
    noise = rng.normal(0, 0.5, size=4)
    # Accelerating drift: small early at 24h (~1-2 nA), large at 168h (30-50 nA)
    t = np.array([0, 24, 96, 168], dtype=float)
    total_drift_168 = rng.uniform(32.0, 52.0)
    drift = total_drift_168 * ((t / 168.0) ** 1.8)
    values = base + noise + drift
    return np.clip(values, 0.1, None)


def _noisy_drift(lot_mean: float, rng: np.random.Generator) -> np.ndarray:
    """Defect 3: drift with high noise overlay — hard to detect."""
    base = lot_mean + rng.normal(0, 1.5)
    noise = rng.normal(0, 2.5, size=4)  # HIGH noise
    drift_rate = rng.uniform(0.15, 0.26)
    drift = np.array([0, 24, 96, 168]) * drift_rate
    values = base + noise + drift
    return np.clip(values, 0.1, None)


def _sudden_failure(lot_mean: float, rng: np.random.Generator) -> np.ndarray:
    """Defect 4: normal values then sudden jump at 96h or 168h."""
    base = lot_mean + rng.normal(0, 1.5)
    noise = rng.normal(0, 0.8, size=4)
    values = base + noise + np.array([0, 24, 96, 168]) * 0.002  # slight natural drift
    # Sudden jump at checkpoint 2 (96h) or 3 (168h)
    jump_point = rng.choice([2, 3])
    jump_magnitude = rng.uniform(25, 50)
    values[jump_point:] += jump_magnitude
    return np.clip(values, 0.1, None)


def _latent_defect(lot_mean: float, rng: np.random.Generator) -> np.ndarray:
    """Defect 5: looks normal at 0h and 24h but fails at 168h.

    This is a KEY SHOWCASE: the system should detect this via 0h+24h trajectory.
    We make the 24h value very slightly elevated (detectable by ML but not by eye).
    """
    base = lot_mean + rng.normal(0, 1.0)
    noise = rng.normal(0, 0.5, size=4)
    # Normal at 0h, very slightly elevated at 24h, moderate at 96h, bad at 168h
    hidden_drift = np.array([0, 2.0, 18.0, 40.0]) * rng.uniform(0.8, 1.5)
    values = base + noise + hidden_drift
    return np.clip(values, 0.1, None)


def _lot_outlier(lot_mean: float, rng: np.random.Generator) -> np.ndarray:
    """Defect 6: within spec but significantly different from lot peers.

    This is a KEY SHOWCASE: value 88-95 nA while lot median is 60-75 nA.
    The absolute screening says PASS, but lot-relative screening says ANOMALY.
    """
    # Force the component to be high but under the limit
    outlier_value = rng.uniform(88, 95)
    noise = rng.normal(0, 0.5, size=4)
    # Stable — not drifting, just an outlier
    values = outlier_value + noise + np.array([0, 0.2, 0.5, 0.8])
    return np.clip(values, 0.1, ABSOLUTE_LIMIT - 0.5)  # Keep under limit


def _borderline(lot_mean: float, rng: np.random.Generator) -> np.ndarray:
    """Defect 7: hovers near the specification limit."""
    base = rng.uniform(85, 98)  # start near limit
    noise = rng.normal(0, 2.0, size=4)
    slight_drift = np.array([0, 0.3, 0.8, 1.5])
    values = base + noise + slight_drift
    return np.clip(values, 0.1, None)


_INJECTORS = {
    1: _gradual_drift,
    2: _accelerating_drift,
    3: _noisy_drift,
    4: _sudden_failure,
    5: _latent_defect,
    6: _lot_outlier,
    7: _borderline,
}


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_dataset(
    n_lots: int = 20,
    components_per_lot: int = 15,
    defect_ratio: float = 0.35,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate the full synthetic burn-in dataset.

    Parameters
    ----------
    n_lots : int
        Number of manufacturing lots.
    components_per_lot : int
        Components per lot.
    defect_ratio : float
        Fraction of components that are defective (0..1).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Burn-in data with columns: component_id, lot_id, value_0h, value_24h,
        value_96h, value_168h, defect_type, defect_name, ground_truth.
    """
    rng = np.random.default_rng(seed)
    n_total = n_lots * components_per_lot
    n_defective = int(n_total * defect_ratio)
    n_nominal = n_total - n_defective

    # Assign defect types
    defect_labels = np.concatenate([
        np.zeros(n_nominal, dtype=int),
        rng.integers(1, 8, size=n_defective),
    ])
    rng.shuffle(defect_labels)

    # Generate lot baselines — each lot has its own mean
    # Lots center between 55-75 nA (well below the 100 nA limit)
    lot_means = rng.uniform(55, 75, size=n_lots)

    records = []
    comp_idx = 0

    for lot_i in range(n_lots):
        lot_id = f"LOT-{lot_i + 1:03d}"
        lot_mean = lot_means[lot_i]

        for j in range(components_per_lot):
            if comp_idx >= n_total:
                break

            comp_id = f"COMP-{comp_idx + 1:05d}"
            defect_type = int(defect_labels[comp_idx])

            if defect_type == 0:
                values = _nominal_trajectory(lot_mean, rng)
            else:
                injector = _INJECTORS[defect_type]
                values = injector(lot_mean, rng)

            records.append({
                "component_id": comp_id,
                "lot_id": lot_id,
                "value_0h": round(float(values[0]), 4),
                "value_24h": round(float(values[1]), 4),
                "value_96h": round(float(values[2]), 4),
                "value_168h": round(float(values[3]), 4),
                "defect_type": defect_type,
                "defect_name": DEFECT_NAMES[defect_type],
                "ground_truth": "nominal" if defect_type == 0 else "defective",
            })
            comp_idx += 1

    df = pd.DataFrame(records)
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic space electronics burn-in data"
    )
    parser.add_argument("--n_lots", type=int, default=20,
                        help="Number of manufacturing lots (default: 20)")
    parser.add_argument("--components_per_lot", type=int, default=15,
                        help="Components per lot (default: 15)")
    parser.add_argument("--defect_ratio", type=float, default=0.35,
                        help="Fraction of defective components (default: 0.35)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: data/telemetry.csv)")
    args = parser.parse_args()

    output_path = args.output or os.path.join(
        Path(__file__).resolve().parent.parent, "data", "telemetry.csv"
    )

    print(f"[*] Generating burn-in data: {args.n_lots} lots × "
          f"{args.components_per_lot} components/lot "
          f"(defect ratio: {args.defect_ratio:.0%}, seed: {args.seed})...")

    df = generate_dataset(
        n_lots=args.n_lots,
        components_per_lot=args.components_per_lot,
        defect_ratio=args.defect_ratio,
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    # Summary stats
    n_comps = len(df)
    n_lots = df["lot_id"].nunique()
    defect_dist = df.groupby("defect_name")["component_id"].count().sort_values(ascending=False)

    print(f"[OK] Saved {n_comps} components ({n_lots} lots) -> {output_path}")
    print(f"\n[STATS] Defect distribution:")
    for name, count in defect_dist.items():
        pct = count / n_comps * 100
        bar = "#" * int(pct / 2)
        print(f"   {name:<22s} {count:>4d}  ({pct:5.1f}%)  {bar}")

    print(f"\n[STATS] Value ranges:")
    for col in ["value_0h", "value_24h", "value_96h", "value_168h"]:
        lo, hi, mu = df[col].min(), df[col].max(), df[col].mean()
        print(f"   {col:<12s}  min={lo:>8.2f}  max={hi:>8.2f}  mean={mu:>8.2f}")

    # Show lot variation
    lot_means = df.groupby("lot_id")["value_0h"].mean()
    print(f"\n[STATS] Lot baseline variation (value_0h mean):")
    print(f"   min lot mean: {lot_means.min():.2f}  max lot mean: {lot_means.max():.2f}")

    # Showcase components
    outliers = df[df["defect_type"] == 6]
    if len(outliers) > 0:
        print(f"\n[SHOWCASE] Lot outlier examples (defect_type=6):")
        for _, row in outliers.head(3).iterrows():
            lot_peers = df[(df["lot_id"] == row["lot_id"]) & (df["defect_type"] == 0)]
            lot_med = lot_peers["value_168h"].median() if len(lot_peers) > 0 else 0
            print(f"   {row['component_id']} in {row['lot_id']}: "
                  f"value_168h={row['value_168h']:.1f} nA  "
                  f"(lot median={lot_med:.1f} nA, limit={ABSOLUTE_LIMIT})")


if __name__ == "__main__":
    main()
