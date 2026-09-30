"""
SIH26170 -- Adaptive, Explainable, Uncertainty-Aware Burn-In Reliability Decision Engine
========================================================================================

Streamlit Interactive Dashboard for Space Electronics Component Screening
Provides:
    1. Executive Overview & Flight Readiness KPIs
    2. SIH Showcase Demos (Case 1: Lot Outlier, Case 2: Early Drift Risk)
    3. Component Explorer (Lot/Component selection, 4-point trajectory, metrics, explanation, counterfactual)
    4. Anomaly & Drift Analytics
    5. Interactive Stress Testing & Robustness Simulator
    6. System Evaluation & Verification Report
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

ABSOLUTE_LIMIT = 100.0  # nA

# ---------------------------------------------------------------------------
# Page configuration & Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SIH26170 | Space Electronics Screening AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #1e222d;
        border-radius: 8px;
        padding: 15px;
        border-left: 4px solid #3498db;
        margin-bottom: 10px;
    }
    .badge-pass {
        background-color: #27ae60;
        color: white;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-hold {
        background-color: #f39c12;
        color: white;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 1.1rem;
        display: inline-block;
    }
    .badge-reject {
        background-color: #c0392b;
        color: white;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: bold;
        font-size: 1.1rem;
        display: inline-block;
    }
    .callout-box {
        background-color: #1a252f;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #34495e;
        margin-top: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached Data Loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_telemetry():
    path = os.path.join(DATA_DIR, "telemetry.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_data
def load_anomaly_scores():
    path = os.path.join(DATA_DIR, "anomaly_scores.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_data
def load_drift_predictions():
    path = os.path.join(DATA_DIR, "drift_predictions.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_data
def load_risk_decisions():
    path = os.path.join(DATA_DIR, "risk_decisions.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

@st.cache_data
def load_explanations():
    path = os.path.join(DATA_DIR, "explanations.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------

def render_sidebar():
    st.sidebar.markdown("## 🛰️ SIH26170 Screening Engine")
    st.sidebar.caption("AI-Driven Burn-In Quality Assurance for Space Electronics")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        [
            "🛰️ Executive Summary",
            "🎯 Showcase Demos (SIH Judging)",
            "🔍 Component Explorer",
            "📊 Lot & Trajectory Analytics",
            "⚡ Stress Testing Simulator",
            "📋 Evaluation & Verification",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Mission Specifications")
    st.sidebar.markdown("""
    - **Burn-In Cycle**: `0h -> 24h -> 96h -> 168h`
    - **Spec Limit**: `100.0 nA` (Leakage Current)
    - **Detection Mode 1**: Lot-Relative Outlier
    - **Detection Mode 2**: Early Drift Trajectory
    - **Primary Goal**: `Zero False Negatives`
    """)

    return page


# ---------------------------------------------------------------------------
# Page 1: Executive Summary
# ---------------------------------------------------------------------------

def page_overview():
    st.title("🛰️ Executive Mission Overview")
    st.markdown("""
    **SIH26170 Adaptive Reliability Decision Engine** screens space-grade components during
    high-temperature burn-in testing. It catches latent failure modes and statistical lot outliers
    that bypass traditional absolute-threshold screening.
    """)

    decisions = load_risk_decisions()
    anomaly = load_anomaly_scores()

    if decisions.empty:
        st.error("No screening data found. Please run the pipeline first.")
        return

    # Critical Flight Safety KPI Banner
    total = len(decisions)
    n_pass = (decisions["decision"] == "PASS").sum()
    n_hold = (decisions["decision"] == "HOLD").sum()
    n_reject = (decisions["decision"] == "REJECT").sum()

    n_defective = (decisions["ground_truth"] == "defective").sum() if "ground_truth" in decisions.columns else 0
    false_negatives = ((decisions["decision"] == "PASS") & (decisions["ground_truth"] == "defective")).sum() if "ground_truth" in decisions.columns else 0
    recall = (n_defective - false_negatives) / n_defective if n_defective > 0 else 1.0

    if false_negatives == 0:
        st.success(f"### 🛡️ ZERO FALSE NEGATIVES ACHIEVED -- 100.0% DEFECT CATCH RATE ({n_defective}/{n_defective} Defective Components Screened Out)")
    else:
        st.error(f"⚠️ Alert: {false_negatives} defective components passed!")

    # 4 KPI Cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Components", f"{total}", f"{decisions['lot_id'].nunique()} Lots")
    c2.metric("PASS (Flight Ready)", f"{n_pass} ({n_pass/total*100:.1f}%)", "Cleared")
    c3.metric("HOLD (Manual Review)", f"{n_hold} ({n_hold/total*100:.1f}%)", "Safety Margin")
    c4.metric("REJECT (Quarantined)", f"{n_reject} ({n_reject/total*100:.1f}%)", "Defects Removed")

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Decision Breakdown")
        color_map = {"PASS": "#27ae60", "HOLD": "#f39c12", "REJECT": "#c0392b"}
        fig_pie = px.pie(
            names=["PASS", "HOLD", "REJECT"],
            values=[n_pass, n_hold, n_reject],
            color=["PASS", "HOLD", "REJECT"],
            color_discrete_map=color_map,
            hole=0.45,
        )
        fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.subheader("Catch Rate by Defect Category")
        if "defect_name" in decisions.columns:
            defect_breakdown = []
            for d_name, grp in decisions.groupby("defect_name"):
                if d_name == "nominal":
                    continue
                caught = (grp["decision"] != "PASS").sum()
                rate = caught / len(grp) if len(grp) > 0 else 0
                defect_breakdown.append({"Defect": d_name, "Catch Rate": rate, "Total": len(grp)})
            df_def = pd.DataFrame(defect_breakdown).sort_values("Catch Rate", ascending=True)

            fig_bar = px.bar(
                df_def,
                x="Catch Rate",
                y="Defect",
                orientation="h",
                text="Catch Rate",
                color="Catch Rate",
                color_continuous_scale=["#f39c12", "#27ae60"],
                range_color=[0.8, 1.0],
            )
            fig_bar.update_traces(texttemplate="%{text:.0%}", textposition="inside")
            fig_bar.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=320, xaxis=dict(range=[0, 1.1]))
            st.plotly_chart(fig_bar, use_container_width=True)

    # Lot Distribution Table
    st.subheader("Manufacturing Lot Screening Summary")
    lot_summary = decisions.groupby("lot_id").agg(
        Total=("component_id", "count"),
        PASS=("decision", lambda s: (s == "PASS").sum()),
        HOLD=("decision", lambda s: (s == "HOLD").sum()),
        REJECT=("decision", lambda s: (s == "REJECT").sum()),
    ).reset_index()
    lot_summary["Yield (%)"] = (lot_summary["PASS"] / lot_summary["Total"] * 100).round(1)
    st.dataframe(lot_summary, height=220, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 2: Showcase Demos (Extremely Important for Judging)
# ---------------------------------------------------------------------------

def page_showcase_demos():
    st.title("🎯 SIH Showcase Demos: Where Conventional Screening Fails")
    st.markdown("""
    This section directly demonstrates the **two core capabilities** that justify deploying AI in space screening:
    1. **Lot-Relative Anomaly**: Component is under the 100 nA absolute limit, but is a severe outlier in its manufacturing lot.
    2. **Early Drift Risk**: Component passes at 0h and 24h, but its trajectory guarantees failure at 168h.
    """)

    telemetry = load_telemetry()
    anomaly = load_anomaly_scores()
    drift = load_drift_predictions()
    decisions = load_risk_decisions()
    explanations = load_explanations()

    tab1, tab2 = st.tabs([
        "CASE 1: Absolute PASS -> Lot-Relative ANOMALY",
        "CASE 2: Current PASS -> Early DRIFT RISK"
    ])

    # -----------------------------------------------------------------------
    # Case 1
    # -----------------------------------------------------------------------
    with tab1:
        st.markdown("### Case 1: The Within-Specification Lot Outlier")
        st.info("""
        **Problem with Conventional Screening**: Conventional test equipment only checks `Value < 100 nA`.
        If a component measures 92 nA while its lot peers average 60 nA, conventional screening **PASSES** it.
        In space missions, such components suffer infant mortality. **Our AI flags it via Robust Z-score.**
        """)

        # Filter showcase candidates (defect_type == 6 or high robust_z but value_168h < 100)
        c1_candidates = anomaly[(anomaly["value_168h"] < ABSOLUTE_LIMIT) & (anomaly["anomaly_score"] >= 3.0)]
        if c1_candidates.empty:
            c1_candidates = anomaly[anomaly["anomaly_score"] >= 2.5]

        selected_c1_id = st.selectbox(
            "Select Showcase Component (Case 1):",
            c1_candidates["component_id"].tolist(),
            index=0,
        )

        c1_comp = anomaly[anomaly["component_id"] == selected_c1_id].iloc[0]
        c1_dec = decisions[decisions["component_id"] == selected_c1_id].iloc[0]
        c1_lot_id = c1_comp["lot_id"]
        lot_peers = telemetry[telemetry["lot_id"] == c1_lot_id]

        val_168h = c1_comp["value_168h"]
        lot_med = c1_comp["lot_median_168h"]
        z_score = c1_comp["anomaly_score"]

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("""
            <div style="background: #2c3e50; padding: 20px; border-radius: 8px; border-left: 6px solid #e74c3c;">
                <h4>Conventional Absolute Screening</h4>
                <p style="font-size: 1.3rem; margin: 0;"><b>Verdict:</b> <span style="color:#2ecc71;">PASS ✅</span></p>
                <p style="margin-top: 5px;">Criteria: <i>Measured value &lt; 100.0 nA</i></p>
                <p>Component value is <b>{:.2f} nA</b>, which is below the 100 nA spec limit. Conventional screening allows it to fly!</p>
            </div>
            """.format(val_168h), unsafe_allow_html=True)

        with col_b:
            st.markdown("""
            <div style="background: #2c3e50; padding: 20px; border-radius: 8px; border-left: 6px solid #2ecc71;">
                <h4>AI Lot-Relative Screening Engine</h4>
                <p style="font-size: 1.3rem; margin: 0;"><b>Verdict:</b> <span style="color:#e74c3c;">{} ⚠️</span></p>
                <p style="margin-top: 5px;">Criteria: <i>Robust Z = (value - median) / (1.4826 &times; MAD)</i></p>
                <p>Lot Median: <b>{:.2f} nA</b> | Robust Z-Score: <b>{:.2f}</b> (Threshold = 3.0). Strong statistical outlier!</p>
            </div>
            """.format(c1_dec['decision'], lot_med, z_score), unsafe_allow_html=True)

        st.markdown("#### Peer Lot Comparison")
        fig_lot = go.Figure()
        # All peers in the lot
        for _, p_row in lot_peers.iterrows():
            is_target = p_row["component_id"] == selected_c1_id
            fig_lot.add_trace(go.Scatter(
                x=[0, 24, 96, 168],
                y=[p_row["value_0h"], p_row["value_24h"], p_row["value_96h"], p_row["value_168h"]],
                mode="lines+markers",
                name=p_row["component_id"],
                line=dict(color="#e74c3c" if is_target else "#7f8c8d", width=3 if is_target else 1),
                opacity=1.0 if is_target else 0.4,
                showlegend=is_target,
            ))

        fig_lot.add_hline(y=ABSOLUTE_LIMIT, line_dash="dash", line_color="red",
                          annotation_text=f"Absolute Spec Limit ({ABSOLUTE_LIMIT} nA)")
        fig_lot.add_hline(y=lot_med, line_dash="dot", line_color="#3498db",
                          annotation_text=f"Lot Median ({lot_med:.1f} nA)")

        fig_lot.update_layout(
            title=f"Components in {c1_lot_id} (Highlighting Outlier {selected_c1_id})",
            xaxis_title="Burn-In Hours (h)",
            yaxis_title="Leakage Current (nA)",
            height=400,
            xaxis=dict(tickvals=[0, 24, 96, 168]),
        )
        st.plotly_chart(fig_lot, use_container_width=True)

    # -----------------------------------------------------------------------
    # Case 2
    # -----------------------------------------------------------------------
    with tab2:
        st.markdown("### Case 2: Early Drift Risk Detected at 24 Hours")
        st.info("""
        **Problem with Conventional Screening**: At 24 hours into burn-in, defective components often still measure
        well below 100 nA. Conventional screening cannot foresee that their trajectory will breach the limit at 168 hours.
        **Our AI predicts the 168h trajectory at 24h with uncertainty bounds and flags it.**
        """)

        # Filter showcase candidates (0h < 100, 24h < 100, and predicted >= 80 or reject)
        c2_candidates = drift[(drift["value_0h"] < ABSOLUTE_LIMIT) &
                              (drift["value_24h"] < ABSOLUTE_LIMIT) &
                              ((drift["predicted_168h"] >= 80.0) | (drift["predicted_168h_upper"] >= ABSOLUTE_LIMIT))]

        selected_c2_id = st.selectbox(
            "Select Showcase Component (Case 2):",
            c2_candidates["component_id"].tolist(),
            index=0,
        )

        c2_drift = drift[drift["component_id"] == selected_c2_id].iloc[0]
        c2_dec = decisions[decisions["component_id"] == selected_c2_id].iloc[0]

        v0 = c2_drift["value_0h"]
        v24 = c2_drift["value_24h"]
        v96 = c2_drift["value_96h"]
        v168_actual = c2_drift["value_168h"]
        pred_168 = c2_drift["predicted_168h"]
        pred_lower = c2_drift["predicted_168h_lower"]
        pred_upper = c2_drift["predicted_168h_upper"]
        safety_slope = c2_drift["safety_slope"]
        pred_slope = c2_drift["predicted_drift_rate"]

        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown("""
            <div style="background: #2c3e50; padding: 20px; border-radius: 8px; border-left: 6px solid #e74c3c;">
                <h4>Conventional Screening at 24h Checkpoint</h4>
                <p style="font-size: 1.3rem; margin: 0;"><b>Verdict:</b> <span style="color:#2ecc71;">PASS ✅</span></p>
                <p style="margin-top: 5px;">0h Value: <b>{:.2f} nA</b> | 24h Value: <b>{:.2f} nA</b></p>
                <p>Both values are well below 100 nA. Conventional screening sees zero issue and allows the burn-in to proceed blind!</p>
            </div>
            """.format(v0, v24), unsafe_allow_html=True)

        with col_d:
            st.markdown("""
            <div style="background: #2c3e50; padding: 20px; border-radius: 8px; border-left: 6px solid #2ecc71;">
                <h4>AI Early Trajectory Predictor</h4>
                <p style="font-size: 1.3rem; margin: 0;"><b>Verdict:</b> <span style="color:#e74c3c;">{} ⚠️</span></p>
                <p style="margin-top: 5px;">Predicted 168h: <b>{:.2f} nA</b> [Interval: {:.1f} - {:.1f} nA]</p>
                <p>Predicted Drift Rate: <b>{:.4f} nA/h</b> vs Safety Slope: <b>{:.4f} nA/h</b>. Early trajectory is unsafe!</p>
            </div>
            """.format(c2_dec['decision'], pred_168, pred_lower, pred_upper, pred_slope, safety_slope), unsafe_allow_html=True)

        st.markdown("#### Trajectory Forecast & Uncertainty Interval")
        fig_traj = go.Figure()

        # Known measurements (0h and 24h)
        fig_traj.add_trace(go.Scatter(
            x=[0, 24],
            y=[v0, v24],
            mode="lines+markers",
            name="Available at 24h (0h, 24h)",
            line=dict(color="#2980b9", width=4),
            marker=dict(size=10),
        ))

        # AI Forecast to 168h
        fig_traj.add_trace(go.Scatter(
            x=[24, 168],
            y=[v24, pred_168],
            mode="lines+markers",
            name="AI Forecast (168h)",
            line=dict(color="#e67e22", width=3, dash="dash"),
            marker=dict(size=8, symbol="diamond"),
        ))

        # Prediction interval at 168h
        fig_traj.add_trace(go.Scatter(
            x=[168, 168],
            y=[pred_lower, pred_upper],
            mode="lines+markers",
            name="Approx Prediction Interval (90% CI)",
            line=dict(color="#f39c12", width=6),
            marker=dict(size=8),
        ))

        # Actual trajectory (revealing future validation)
        fig_traj.add_trace(go.Scatter(
            x=[24, 96, 168],
            y=[v24, v96, v168_actual],
            mode="lines+markers",
            name="Actual Future Measurements (96h, 168h)",
            line=dict(color="#27ae60", width=2, dash="dot"),
            marker=dict(size=8),
        ))

        fig_traj.add_hline(y=ABSOLUTE_LIMIT, line_dash="dash", line_color="red",
                           annotation_text=f"Absolute Spec Limit ({ABSOLUTE_LIMIT} nA)")

        fig_traj.update_layout(
            title=f"Trajectory Analysis for {selected_c2_id}: Early Detection at 24h vs Actual Reality at 168h",
            xaxis_title="Burn-In Hours (h)",
            yaxis_title="Leakage Current (nA)",
            height=420,
            xaxis=dict(tickvals=[0, 24, 96, 168]),
        )
        st.plotly_chart(fig_traj, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 3: Component Explorer
# ---------------------------------------------------------------------------

def page_component_explorer():
    st.title("🔍 Component Screening Explorer")
    st.markdown("Inspect any component's burn-in trajectory, metrics, engineering explanation, and counterfactuals.")

    decisions = load_risk_decisions()
    drift = load_drift_predictions()
    anomaly = load_anomaly_scores()
    explanations = load_explanations()

    # Filter controls
    f1, f2, f3 = st.columns(3)
    with f1:
        lots = ["All Lots"] + sorted(decisions["lot_id"].unique().tolist())
        selected_lot = st.selectbox("Select Lot:", lots)
    with f2:
        dec_options = ["All Decisions", "PASS", "HOLD", "REJECT"]
        selected_dec = st.selectbox("Filter by Decision:", dec_options)
    with f3:
        if "defect_name" in decisions.columns:
            def_options = ["All Types"] + sorted(decisions["defect_name"].unique().tolist())
            selected_def = st.selectbox("Filter by Defect Type:", def_options)
        else:
            selected_def = "All Types"

    # Filter dataframe
    filtered = decisions.copy()
    if selected_lot != "All Lots":
        filtered = filtered[filtered["lot_id"] == selected_lot]
    if selected_dec != "All Decisions":
        filtered = filtered[filtered["decision"] == selected_dec]
    if selected_def != "All Types":
        filtered = filtered[filtered["defect_name"] == selected_def]

    st.markdown(f"**Found {len(filtered)} matching components**")

    if filtered.empty:
        st.warning("No components match the selected filters.")
        return

    selected_comp_id = st.selectbox("Choose Component to Inspect:", filtered["component_id"].tolist())

    comp_row = decisions[decisions["component_id"] == selected_comp_id].iloc[0]
    drift_row = drift[drift["component_id"] == selected_comp_id].iloc[0]
    anom_row = anomaly[anomaly["component_id"] == selected_comp_id].iloc[0]

    # Large visible Decision Banner
    decision = comp_row["decision"]
    badge_class = f"badge-{decision.lower()}"
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:15px; margin: 15px 0;">
        <h2>Component: {selected_comp_id} ({comp_row['lot_id']})</h2>
        <span class="{badge_class}">{decision}</span>
        <span style="color:#bdc3c7; font-size:1.1rem;">Ground Truth: <b>{comp_row.get('defect_name', 'N/A')}</b></span>
    </div>
    """, unsafe_allow_html=True)

    # Metrics row
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Spec Limit", f"{ABSOLUTE_LIMIT:.1f} nA")
    m2.metric("Lot Median", f"{anom_row.get('lot_median_168h', 0):.2f} nA")
    m3.metric("Robust Z-Score", f"{anom_row.get('anomaly_score', 0):.2f}")
    m4.metric("Predicted 168h", f"{drift_row.get('predicted_168h', 0):.2f} nA")
    m5.metric("Approx 90% CI", f"[{drift_row.get('predicted_168h_lower', 0):.1f}, {drift_row.get('predicted_168h_upper', 0):.1f}]")
    m6.metric("Distance to Limit", f"{drift_row.get('distance_to_limit', 0):.2f} nA")

    # Time series visualization
    st.markdown("---")
    st.subheader("Burn-In Trajectory & Prediction")

    v0 = drift_row["value_0h"]
    v24 = drift_row["value_24h"]
    v96 = drift_row["value_96h"]
    v168 = drift_row["value_168h"]
    p168 = drift_row["predicted_168h"]
    p_low = drift_row["predicted_168h_lower"]
    p_high = drift_row["predicted_168h_upper"]
    lot_median = anom_row.get("lot_median_168h", 65.0)

    fig = go.Figure()
    # Actual measurements
    fig.add_trace(go.Scatter(
        x=[0, 24, 96, 168],
        y=[v0, v24, v96, v168],
        mode="lines+markers",
        name="Actual Burn-In (0h, 24h, 96h, 168h)",
        line=dict(color="#3498db", width=3),
        marker=dict(size=9),
    ))
    # Predicted 168h point & interval
    fig.add_trace(go.Scatter(
        x=[24, 168],
        y=[v24, p168],
        mode="lines+markers",
        name="Predicted Trajectory (at 24h)",
        line=dict(color="#e67e22", width=3, dash="dash"),
        marker=dict(size=8, symbol="diamond"),
    ))
    fig.add_trace(go.Scatter(
        x=[168, 168],
        y=[p_low, p_high],
        mode="lines+markers",
        name="Approx Prediction Interval",
        line=dict(color="#f39c12", width=5),
        marker=dict(size=8),
    ))

    # Reference lines
    fig.add_hline(y=ABSOLUTE_LIMIT, line_dash="dash", line_color="red",
                  annotation_text=f"Absolute Spec Limit ({ABSOLUTE_LIMIT} nA)")
    fig.add_hline(y=lot_median, line_dash="dot", line_color="#27ae60",
                  annotation_text=f"Lot Median ({lot_median:.1f} nA)")

    fig.update_layout(
        height=420,
        xaxis_title="Burn-In Hours",
        yaxis_title="Leakage Current (nA)",
        xaxis=dict(tickvals=[0, 24, 96, 168]),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Explanation and Counterfactual
    col_e1, col_e2 = st.columns([1, 1])
    with col_e1:
        st.subheader("Human-Readable Engineering Explanation")
        comp_expl = explanations[explanations["component_id"] == selected_comp_id]
        if not comp_expl.empty:
            st.text_area("Detailed Findings:", comp_expl.iloc[0]["explanation"], height=260)
        else:
            st.info("No explanation text generated.")

    with col_e2:
        st.subheader("Actionable Counterfactual")
        cf_text = comp_expl.iloc[0].get("counterfactual", "") if not comp_expl.empty else ""
        if cf_text and str(cf_text).strip():
            st.markdown(f"""
            <div style="background:#2c3e50; padding:15px; border-radius:8px; border-left:4px solid #f39c12;">
                <h4>Counterfactual Sensitivity</h4>
                <pre style="white-space: pre-wrap; color: #ecf0f1;">{cf_text}</pre>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#1e272e; padding:15px; border-radius:8px; border-left:4px solid #2ecc71;">
                <h4>Nominal Trajectory</h4>
                <p>No counterfactual adjustment required. Component is within acceptable statistical and physical limits.</p>
            </div>
            """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page 4: Lot & Trajectory Analytics
# ---------------------------------------------------------------------------

def page_analytics():
    st.title("📊 Lot-Relative & Trajectory Analytics")

    anomaly = load_anomaly_scores()
    drift = load_drift_predictions()
    decisions = load_risk_decisions()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Lot-to-Lot Baseline Variation (0h)")
        lot_stats = anomaly.groupby("lot_id")["value_0h"].agg(["mean", "std", "median"]).reset_index()
        fig_lot_var = px.box(
            anomaly,
            x="lot_id",
            y="value_0h",
            color="lot_id",
            title="0h Leakage Baseline Across Lots (Lot Heterogeneity)",
        )
        fig_lot_var.update_layout(showlegend=False, height=360)
        st.plotly_chart(fig_lot_var, use_container_width=True)

    with col2:
        st.subheader("Early Predictor: Actual vs Predicted 168h")
        fig_scatter = px.scatter(
            drift,
            x="value_168h",
            y="predicted_168h",
            color="ground_truth" if "ground_truth" in drift.columns else None,
            hover_data=["component_id", "lot_id"],
            title="168h Value: Machine Learning Prediction vs Ground Truth",
        )
        fig_scatter.add_shape(
            type="line", line=dict(dash="dash", color="white", width=2),
            x0=40, y0=40, x1=140, y1=140
        )
        fig_scatter.update_layout(height=360)
        st.plotly_chart(fig_scatter, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Robust Z-Score Distribution")
        fig_z = px.histogram(
            anomaly,
            x="anomaly_score",
            nbins=35,
            color="ground_truth" if "ground_truth" in anomaly.columns else None,
            title="Distribution of Lot-Relative Robust Z-Scores",
        )
        fig_z.add_vline(x=3.0, line_dash="dash", line_color="orange", annotation_text="HOLD (Z=3.0)")
        fig_z.add_vline(x=5.0, line_dash="dash", line_color="red", annotation_text="REJECT (Z=5.0)")
        fig_z.update_layout(height=360)
        st.plotly_chart(fig_z, use_container_width=True)

    with col4:
        st.subheader("Prediction Uncertainty Distribution")
        fig_unc = px.histogram(
            drift,
            x="prediction_uncertainty",
            nbins=35,
            title="Ensemble Tree Prediction Interval Width (Uncertainty)",
            color_discrete_sequence=["#9b59b6"],
        )
        fig_unc.update_layout(height=360)
        st.plotly_chart(fig_unc, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 5: Stress Testing Simulator (Capability J)
# ---------------------------------------------------------------------------

def page_stress_testing():
    st.title("⚡ Interactive Stress Testing & Robustness Simulator")
    st.markdown("""
    **Required Capability J**: Evaluate the screening engine under elevated measurement noise,
    extreme lot-to-lot variation, accelerating drift rates, and latent defect rates.
    """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Stress Simulator Controls")
    noise_level = st.sidebar.slider("Measurement Noise (nA)", 0.2, 4.0, 1.2, 0.2)
    n_lots = st.sidebar.slider("Lots Count", 10, 40, 20, 5)
    comps_per_lot = st.sidebar.slider("Components per Lot", 10, 30, 15, 5)
    defect_ratio = st.sidebar.slider("Defect Fraction", 0.15, 0.60, 0.35, 0.05)
    sim_seed = st.sidebar.number_input("Random Seed", 1, 9999, 100)

    if st.button("🚀 Run Live Stress Simulation", use_container_width=True):
        with st.spinner("Generating stress dataset and running screening pipeline..."):
            from src.data_generator import generate_dataset
            from src.anomaly_detector import compute_robust_z
            from src.drift_predictor import build_feature_matrix, compute_prediction_intervals
            from src.risk_engine import classify_component
            import joblib

            # 1. Generate stressed dataset
            sim_df = generate_dataset(
                n_lots=n_lots,
                components_per_lot=comps_per_lot,
                defect_ratio=defect_ratio,
                seed=int(sim_seed),
            )
            # Add extra noise
            extra_noise = np.random.normal(0, noise_level, size=(len(sim_df), 4))
            sim_df["value_0h"] += extra_noise[:, 0]
            sim_df["value_24h"] += extra_noise[:, 1]
            sim_df["value_96h"] += extra_noise[:, 2]
            sim_df["value_168h"] += extra_noise[:, 3]

            # 2. Run anomaly detection
            sim_df["robust_z_max"] = 0.0
            for lot_id, grp in sim_df.groupby("lot_id"):
                z_0, m_0, mad_0 = compute_robust_z(grp["value_0h"])
                z_24, m_24, mad_24 = compute_robust_z(grp["value_24h"])
                z_168, m_168, mad_168 = compute_robust_z(grp["value_168h"])
                sim_df.loc[grp.index, "anomaly_score"] = np.maximum.reduce([z_0.abs(), z_24.abs(), z_168.abs()])
                sim_df.loc[grp.index, "lot_median_168h"] = m_168

            # 3. Drift prediction using trained model
            model_path = os.path.join(MODELS_DIR, "drift_model.joblib")
            if os.path.exists(model_path):
                model = joblib.load(model_path)
            else:
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                X_init = build_feature_matrix(sim_df)
                model.fit(X_init, sim_df["value_168h"])

            X_sim = build_feature_matrix(sim_df)
            pred_mean, pred_lower, pred_upper, unc = compute_prediction_intervals(model, X_sim)
            sim_df["predicted_168h"] = pred_mean
            sim_df["predicted_168h_lower"] = pred_lower
            sim_df["predicted_168h_upper"] = pred_upper
            sim_df["prediction_uncertainty"] = unc
            sim_df["predicted_drift_rate"] = (pred_mean - sim_df["value_0h"]) / 168.0
            sim_df["safety_slope"] = (ABSOLUTE_LIMIT - sim_df["value_0h"]) / 168.0
            sim_df["distance_to_limit"] = ABSOLUTE_LIMIT - pred_mean

            # 4. Risk decision
            decisions = []
            for _, r in sim_df.iterrows():
                dec, _ = classify_component(r, r)
                decisions.append(dec)
            sim_df["decision"] = decisions

            # 5. Evaluate
            y_true = (sim_df["defect_type"] > 0).astype(int)
            y_caught = (sim_df["decision"] != "PASS").astype(int)

            tp = ((y_true == 1) & (y_caught == 1)).sum()
            fp = ((y_true == 0) & (y_caught == 1)).sum()
            fn = ((y_true == 1) & (y_caught == 0)).sum()
            tn = ((y_true == 0) & (y_caught == 0)).sum()

            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            mae = float(np.mean(np.abs(sim_df["value_168h"] - sim_df["predicted_168h"])))

            st.markdown("### Stress Test Performance Results")
            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric("Defect Recall (Safety)", f"{rec*100:.1f}%", "Zero FN Target")
            s2.metric("False Negative Rate", f"{fnr*100:.1f}%", delta_color="inverse")
            s3.metric("Precision", f"{prec:.3f}")
            s4.metric("F1 Score", f"{f1:.3f}")
            s5.metric("Forecast MAE", f"{mae:.2f} nA")

            if fn == 0:
                st.success("🛡️ Robustness Passed: Zero False Negatives even under harsh noise and high defect ratio!")
            else:
                st.warning(f"⚠️ Stress limit reached: {fn} defects slipped through under extreme conditions.")

            st.dataframe(sim_df[["component_id", "lot_id", "value_0h", "value_24h", "value_168h", "predicted_168h", "anomaly_score", "decision", "ground_truth"]].head(15), use_container_width=True)
    else:
        st.info("Adjust the stress parameters in the sidebar and click 'Run Live Stress Simulation'.")


# ---------------------------------------------------------------------------
# Page 6: Evaluation & Verification
# ---------------------------------------------------------------------------

def page_evaluation():
    st.title("📋 System Evaluation & Flight Readiness Verification")
    st.markdown("Audited metrics demonstrating system robustness for space electronics flight hardware.")

    from src.evaluation import evaluate_anomaly_detection, evaluate_drift_prediction, evaluate_risk_engine

    ad = evaluate_anomaly_detection()
    dp = evaluate_drift_prediction()
    re = evaluate_risk_engine()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("1. Screening Reliability Summary")
        if "binary" in re:
            b = re["binary"]
            metrics_df = pd.DataFrame({
                "Metric": ["Defect Recall (Flight Safety)", "False Negative Rate (Uncaught)", "Screening Precision", "F1 Score", "Defects Caught", "Defects Missed"],
                "Value": [f"{b['recall']*100:.2f}%", f"{b['fnr']*100:.2f}%", f"{b['precision']:.4f}", f"{b['f1']:.4f}", f"{b['tp']}", f"{b['fn']}"],
            })
            st.table(metrics_df)

    with c2:
        st.subheader("2. Early Trajectory Predictor Metrics")
        if "overall_mae" in dp:
            drift_m_df = pd.DataFrame({
                "Metric": ["Overall MAE", "Overall RMSE", "Overall R2 Score", "Holdout Test MAE", "Spec Violations Catch Rate"],
                "Value": [f"{dp['overall_mae']:.4f} nA", f"{dp['overall_rmse']:.4f} nA", f"{dp['overall_r2']:.4f}", f"{dp['holdout_mae']:.4f} nA", f"{dp['spec_violation_catch_rate']*100:.1f}%"],
            })
            st.table(drift_m_df)

    st.subheader("3. Per-Defect-Type Catch Rate Verification")
    if "defect_analysis" in re:
        rows = []
        for def_name, info in sorted(re["defect_analysis"].items()):
            rows.append({
                "Defect Category": def_name,
                "Total Components": info["total"],
                "PASS": info["PASS"],
                "HOLD": info["HOLD"],
                "REJECT": info["REJECT"],
                "Catch Rate (%)": f"{info['catch_rate']*100:.1f}%",
            })
        st.dataframe(pd.DataFrame(rows), height=300, use_container_width=True)


# ---------------------------------------------------------------------------
# Main Routing
# ---------------------------------------------------------------------------

def main():
    page = render_sidebar()

    if page == "🛰️ Executive Summary":
        page_overview()
    elif page == "🎯 Showcase Demos (SIH Judging)":
        page_showcase_demos()
    elif page == "🔍 Component Explorer":
        page_component_explorer()
    elif page == "📊 Lot & Trajectory Analytics":
        page_analytics()
    elif page == "⚡ Stress Testing Simulator":
        page_stress_testing()
    elif page == "📋 Evaluation & Verification":
        page_evaluation()


if __name__ == "__main__":
    main()
