# SIH26170 -- Adaptive, Explainable, Uncertainty-Aware Burn-In Reliability Decision Engine

> **Smart India Hackathon (SIH 2026)**
> **Problem Statement**: AI-Driven Anomaly Detection in Component Burn-In & Screening

An intelligent reliability screening engine for space-grade electronic components undergoing high-temperature burn-in testing (`0h -> 24h -> 96h -> 168h`). The system combines lot-relative statistical outlier detection, early trajectory forecasting with ensemble uncertainty, and transparent engineering rules to achieve **zero false negatives** (defective components never auto-passed).

---

## 🎯 The Core Problem & Our Solution

Conventional qualification screening relies on simple absolute threshold limits (e.g. Leakage Current < 100 nA at end-of-screen). This conventional approach suffers from two dangerous blind spots:

1. **Lot-Relative Anomaly**: A component may measure 92 nA (under the 100 nA limit) while all other components in its manufacturing lot average 60 nA. Conventional screening PASSES it; our AI flags it as an abnormal statistical outlier (`Robust Z-score > 3.0`).
2. **Early Drift Risk**: A component at 24 hours measures 61 nA (safe). Conventional screening passes it. Our AI models the `0h -> 24h` trajectory and forecasts that by 168h it will breach safety margins, catching defects **before limits are violated**.

---

## 🛡️ Flight Safety Key Results

| Metric | System Performance | Benchmark Target |
|---|:---:|:---:|
| **Screening Defect Recall** | **100.0%** | Critical flight safety requirement |
| **False Negative Rate (FNR)** | **0.00%** | Zero defective components allowed to fly |
| **Defective Components Screened Out** | **105 / 105** | Complete capture across all defect types |
| **Early Drift Predictor R² Score** | **0.799** | Accurate 168h forecast from 24h data |
| **Spec Violation Early Catch Rate** | **97.5%** | Pre-emptive detection |

---

## 🏗️ System Architecture

```text
Synthetic Burn-in Telemetry (0h, 24h, 96h, 168h across 20 Lots)
                    │
                    ├──▶ Lot-Relative Anomaly Detector (Robust Z = (x - med) / (1.4826 × MAD))
                    │         │
                    │         ▼ anomaly_score & is_anomaly flag
                    │
                    ├──▶ Early Drift Predictor (0h + 24h ──▶ 168h Forecast via Random Forest)
                    │         │
                    │         ▼ predicted_168h & Approximate Prediction Interval (90% CI)
                    │
                    └──▶ Risk Engine
                              │
                              ├── PASS   — Nominal lot behavior & safe trajectory
                              ├── HOLD   — Suspicious trend, borderline margin, or high uncertainty
                              └── REJECT — Spec violation, severe outlier, or unsafe drift
                              │
                              ▼
                          Explainability Engine
                              ├── Plain-language engineering justifications
                              └── Actionable Counterfactuals ("If 24h value were X nA...")
                              │
                              ▼
                          Streamlit Interactive Dashboard (src/app.py)
```

---

## 🔬 Defect Categories Tested

1. `nominal` — Normal components with standard lot baseline aging
2. `gradual_drift` — Progressive linear degradation toward limit
3. `accelerating_drift` — Small early signature at 24h, accelerating sharply by 168h
4. `noisy_drift` — Drift masked by high measurement noise
5. `sudden_failure` — Stable early values jumping at 96h/168h
6. `latent_defect` — Normal at 0h/24h, latent degradation triggered under thermal stress
7. `lot_outlier` — High value (88–95 nA) well under 100 nA limit but abnormal relative to lot peers
8. `borderline` — Hovers within 5–15% of the 100 nA specification limit

---

## 🚀 Quick Start Guide

### 1. Requirements

Ensure Python 3.11+ is installed.

```bash
pip install -r requirements.txt
```

### 2. Run End-to-End Pipeline

Execute the full automated screening pipeline (data generation, anomaly detection, drift prediction, risk classification, explainability, evaluation):

```bash
python run_all.py
```

### 3. Launch Interactive Streamlit Dashboard

```bash
streamlit run src/app.py
```

The dashboard will open automatically in your browser (default: `http://localhost:8501`).

---

## 🖥️ Streamlit Dashboard Features

- **🛰️ Executive Summary**: Mission KPIs, defect catch rate by category, and lot-by-lot yields.
- **🎯 Showcase Demos**: Dedicated judging view for:
  - *Case 1*: Absolute PASS (91 nA < 100 nA) vs AI FLAGGED (Robust Z = 5.8 vs Lot Median 59 nA).
  - *Case 2*: 24h Checkpoint PASS vs AI FLAGGED Early Drift Risk with Prediction Interval.
- **🔍 Component Explorer**: Filter by Lot, Decision, and Defect. Interactive 4-point burn-in curve, metrics table, engineering explanation, and model-derived counterfactuals.
- **📊 Lot & Trajectory Analytics**: Lot heterogeneity box plots, R² scatter analysis, and uncertainty distributions.
- **⚡ Stress Testing Simulator (Capability J)**: Interactive controls for measurement noise, lot count, and defect ratio with live re-calculation of Recall, Precision, and FNR.
- **📋 Evaluation & Verification**: Full flight readiness audit table.

---

## 📂 Project Structure

```text
SIH26170/
├── data/                      # Generated burn-in datasets & outputs
│   ├── telemetry.csv          # 4-point burn-in data (0h, 24h, 96h, 168h)
│   ├── anomaly_scores.csv     # Lot-relative robust Z-scores & MADs
│   ├── drift_predictions.csv  # 168h forecasts & prediction intervals
│   ├── risk_decisions.csv     # PASS / HOLD / REJECT decisions & reasons
│   └── explanations.csv       # Human-readable justifications & counterfactuals
├── models/
│   └── drift_model.joblib     # Serialized early drift prediction model
├── src/
│   ├── __init__.py
│   ├── data_generator.py      # Synthetic burn-in generator (lots, defects)
│   ├── anomaly_detector.py    # Robust Z-score & lot-relative anomaly detector
│   ├── drift_predictor.py     # 0h+24h -> 168h Random Forest regressor with uncertainty
│   ├── risk_engine.py         # PASS/HOLD/REJECT decision rules
│   ├── explainability.py      # Plain engineering explanations & counterfactuals
│   ├── evaluation.py          # Comprehensive evaluation & FNR metrics
│   └── app.py                 # Streamlit interactive application
├── requirements.txt
├── run_all.py                 # End-to-end pipeline runner
└── README.md
```
