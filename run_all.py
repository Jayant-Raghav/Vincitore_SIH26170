"""
run_all.py -- End-to-end pipeline for Space Electronics AI QA
=============================================================
Regenerates all data and runs the full pipeline from scratch.
Usage: python run_all.py
"""
import subprocess
import sys
import time


STEPS = [
    ("Step 1: Generate Synthetic Telemetry", [sys.executable, "-m", "src.data_generator"]),
    ("Step 2: Anomaly Detection",            [sys.executable, "-m", "src.anomaly_detector"]),
    ("Step 3: Drift Prediction",             [sys.executable, "-m", "src.drift_predictor"]),
    ("Step 4: Risk Engine Classification",   [sys.executable, "-m", "src.risk_engine"]),
    ("Step 5: Generate Explanations",        [sys.executable, "-m", "src.explainability"]),
    ("Step 6: Evaluation Report",            [sys.executable, "-m", "src.evaluation"]),
]


def main():
    print("=" * 70)
    print("  SPACE ELECTRONICS AI QUALITY ASSURANCE -- FULL PIPELINE")
    print("=" * 70)

    total_start = time.time()

    for i, (desc, cmd) in enumerate(STEPS, 1):
        print(f"\n{'-' * 70}")
        print(f"  [{i}/{len(STEPS)}] {desc}")
        print(f"{'-' * 70}\n")

        step_start = time.time()
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - step_start
        print(f"\n  >> Completed in {elapsed:.1f}s")

    total_elapsed = time.time() - total_start

    print(f"\n{'=' * 70}")
    print(f"  ALL STEPS COMPLETE -- Total time: {total_elapsed:.1f}s")
    print(f"{'=' * 70}")
    print(f"\n  To launch the dashboard:")
    print(f"    streamlit run src/app.py")
    print()


if __name__ == "__main__":
    main()
