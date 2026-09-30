"""Analyze why 19 defective components are being missed."""
import pandas as pd
import numpy as np

decisions = pd.read_csv('data/risk_decisions.csv')
anomaly = pd.read_csv('data/anomaly_scores.csv')
drift = pd.read_csv('data/drift_predictions.csv')
telemetry = pd.read_csv('data/telemetry.csv')

# Missed defectives
missed = decisions[(decisions['decision'] == 'PASS') & (decisions['defect_type'] > 0)]
print(f"=== {len(missed)} MISSED DEFECTIVES ===\n")

for _, row in missed.iterrows():
    comp = row['component_id']
    defect = row['defect_name']
    score = row['anomaly_score']
    
    # Get drift info
    d = drift[drift['component_id'] == comp].iloc[0]
    temp_rate = d['temperature_c__drift_rate']
    temp_pred = d['temperature_c__predicted_168h']
    res_rate = d['resistance_ohm__drift_rate']
    res_pred = d['resistance_ohm__predicted_168h']
    vib_rate = d['vibration_g__drift_rate']
    
    # Get telemetry tail stats
    t = telemetry[telemetry['component_id'] == comp].sort_values('timestamp_h')
    n = len(t)
    first_half = t.iloc[:n//2]
    second_half = t.iloc[n//2:]
    
    temp_first = first_half['temperature_c'].mean()
    temp_second = second_half['temperature_c'].mean()
    temp_delta = temp_second - temp_first
    
    res_first = first_half['resistance_ohm'].mean()
    res_second = second_half['resistance_ohm'].mean()
    res_delta = res_second - res_first
    
    print(f"{comp} | {defect:<20s} | anomaly={score:.4f}")
    print(f"  temp: rate={temp_rate:.6f}/h, pred_168h={temp_pred:.1f}C, "
          f"1st_half={temp_first:.1f}C, 2nd_half={temp_second:.1f}C, delta={temp_delta:+.2f}C")
    print(f"  resist: rate={res_rate:.6f}/h, pred_168h={res_pred:.2f}ohm, "
          f"1st_half={res_first:.2f}, 2nd_half={res_second:.2f}, delta={res_delta:+.3f}")
    print(f"  vib: rate={vib_rate:.6f}/h")
    print()

# Overall threshold analysis
print("=== ANOMALY SCORE DISTRIBUTION ===")
for defect in sorted(anomaly['defect_name'].unique()):
    subset = anomaly[anomaly['defect_name'] == defect]
    caught = decisions[(decisions['defect_name'] == defect) & (decisions['decision'] != 'PASS')]
    print(f"{defect:<22s}: n={len(subset):>3d}, "
          f"scores: min={subset['anomaly_score'].min():.3f}, "
          f"median={subset['anomaly_score'].median():.3f}, "
          f"max={subset['anomaly_score'].max():.3f}, "
          f"caught={len(caught)}/{len(subset)}")

# Check what the max anomaly score per defect type is for missed ones
print("\n=== TAIL-END FEATURES (2nd half mean - 1st half mean) ===")
print(f"{'comp':<12s} {'defect':<20s} {'temp_delta':>10s} {'res_delta':>10s} {'vib_max_2h':>10s}")
for _, row in missed.iterrows():
    comp = row['component_id']
    defect = row['defect_name']
    t = telemetry[telemetry['component_id'] == comp].sort_values('timestamp_h')
    n = len(t)
    first, second = t.iloc[:n//2], t.iloc[n//2:]
    temp_d = second['temperature_c'].mean() - first['temperature_c'].mean()
    res_d = second['resistance_ohm'].mean() - first['resistance_ohm'].mean()
    vib_max = second['vibration_g'].max()
    print(f"{comp:<12s} {defect:<20s} {temp_d:>+10.2f} {res_d:>+10.3f} {vib_max:>10.2f}")
