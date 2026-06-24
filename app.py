"""
=============================================================================
Predictive Maintenance – Interactive Streamlit Dashboard
=============================================================================
Professional dashboard for monitoring industrial equipment health, exploring
sensor data, evaluating model performance, and making real-time predictions.

Launch:  streamlit run streamlit_app.py
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
)

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "sensor_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURE_COLS_PATH = os.path.join(BASE_DIR, "feature_cols.pkl")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Predictive Maintenance | Industrial IoT",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium dark-themed dashboard
st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header ── */
    .dashboard-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .dashboard-header h1 {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .dashboard-header p {
        color: #a0aec0;
        font-size: 1rem;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        padding: 1.5rem;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.35);
    }
    .metric-card .metric-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    .metric-card .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1.2;
    }
    .metric-card .metric-label {
        font-size: 0.85rem;
        color: #718096;
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* ── Risk Badges ── */
    .risk-low { background: linear-gradient(135deg, #00b09b, #96c93d); }
    .risk-medium { background: linear-gradient(135deg, #f7971e, #ffd200); }
    .risk-high { background: linear-gradient(135deg, #f85032, #e73827); }
    .risk-critical {
        background: linear-gradient(135deg, #8e0e00, #1f1c18);
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%, 100% { box-shadow: 0 0 10px rgba(142,14,0,0.5); }
        50% { box-shadow: 0 0 25px rgba(142,14,0,0.9); }
    }

    .risk-badge {
        padding: 1rem 2rem;
        border-radius: 12px;
        color: white;
        font-weight: 700;
        font-size: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }

    /* ── Recommendation Cards ── */
    .rec-card {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        border-left: 4px solid;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.15);
    }
    .rec-card.critical { border-color: #e74c3c; }
    .rec-card.warning { border-color: #f39c12; }
    .rec-card.info { border-color: #3498db; }

    /* ── Section Headings ── */
    .section-title {
        color: #e2e8f0;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(255,255,255,0.08);
    }

    /* ── Tabs Styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }

    /* ── Hide Streamlit branding ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING (cached)
# ═══════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    """Load sensor data."""
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    return df


@st.cache_resource
def load_model():
    """Load trained model and scaler."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    return model, scaler, feature_cols


def get_risk_level(prob):
    """Map probability to risk level."""
    if prob < 0.30:
        return "🟢 Low Risk", "risk-low"
    elif prob < 0.60:
        return "🟡 Medium Risk", "risk-medium"
    elif prob < 0.80:
        return "🟠 High Risk", "risk-high"
    else:
        return "🔴 Critical Risk", "risk-critical"


def get_recommendations(temp, vib, pres, hum, volt, curr, rpm, hrs, prob):
    """Generate maintenance recommendations based on sensor values."""
    recs = []

    if prob >= 0.80:
        recs.append(("🚨 IMMEDIATE MAINTENANCE REQUIRED", "Failure probability is critically high. "
                      "Schedule emergency inspection within 24 hours.", "critical"))

    if temp > 90:
        recs.append(("🌡️ High Temperature Alert", "Temperature exceeds safe operating threshold. "
                      "Check cooling system, clean heat exchangers, and verify coolant levels.", "critical"))
    elif temp > 80:
        recs.append(("🌡️ Elevated Temperature", "Temperature is above normal range. "
                      "Monitor closely and schedule cooling system inspection.", "warning"))

    if vib > 5.0:
        recs.append(("📳 High Vibration Detected", "Excessive vibration may indicate worn bearings, "
                      "misalignment, or loose components. Inspect bearings and rotating components immediately.", "critical"))
    elif vib > 3.5:
        recs.append(("📳 Elevated Vibration", "Vibration levels are trending upward. "
                      "Schedule bearing inspection and shaft alignment check.", "warning"))

    if pres < 35:
        recs.append(("⬇️ Low Pressure Warning", "Pressure is below safe operating range. "
                      "Inspect hydraulic or pneumatic system for leaks. Check seals and gaskets.", "critical"))
    elif pres < 45:
        recs.append(("⬇️ Declining Pressure", "Pressure is below optimal range. "
                      "Monitor for further decline and check pressure relief valves.", "warning"))

    if abs(volt - 230) > 15:
        recs.append(("⚡ Voltage Fluctuation", "Significant voltage deviation detected. "
                      "Check electrical connections, power supply stability, and voltage regulators.", "warning"))

    if curr > 17:
        recs.append(("🔌 High Current Draw", "Current draw exceeds normal operating range. "
                      "Check for mechanical binding, overloaded circuits, or insulation breakdown.", "warning"))

    if hrs > 12000:
        recs.append(("⏱️ High Operating Hours", "Equipment has accumulated significant operating hours. "
                      "Schedule comprehensive preventive maintenance and component replacement.", "warning"))
    elif hrs > 8000:
        recs.append(("⏱️ Routine Maintenance Due", "Based on operating hours, routine maintenance "
                      "should be scheduled. Review maintenance logs and plan accordingly.", "info"))

    if rpm > 3500:
        recs.append(("🔄 High Rotational Speed", "RPM exceeds recommended operating range. "
                      "Verify speed control systems and check for governor issues.", "warning"))

    if not recs:
        recs.append(("✅ All Systems Normal", "All sensor readings are within normal operating parameters. "
                      "Continue routine monitoring.", "info"))

    return recs


# Plotly dark theme template
PLOTLY_TEMPLATE = "plotly_dark"
COLORS = {
    "primary": "#6c5ce7",
    "secondary": "#00cec9",
    "accent": "#fd79a8",
    "success": "#00b894",
    "danger": "#d63031",
    "warning": "#fdcb6e",
    "bg": "#0e1117",
    "card_bg": "#1a1a2e",
    "text": "#e2e8f0",
    "gradient": ["#6c5ce7", "#a29bfe", "#74b9ff", "#00cec9", "#55efc4"],
}


# ═══════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════
def main():
    # ── Check that required files exist ──
    missing = []
    if not os.path.exists(DATA_PATH):
        missing.append("sensor_data.csv")
    if not os.path.exists(MODEL_PATH):
        missing.append("best_model.pkl")
    if not os.path.exists(SCALER_PATH):
        missing.append("scaler.pkl")

    if missing:
        st.error(f"⚠️ Missing files: {', '.join(missing)}. Please run `python train_model.py` first.")
        st.stop()

    # Load data and model
    df = load_data()
    model, scaler, feature_cols = load_model()

    # ── Header ──
    st.markdown("""
    <div class="dashboard-header">
        <h1>⚙️ Predictive Maintenance Dashboard</h1>
        <p>Industrial Equipment Health Monitoring & Failure Prediction Platform</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### 🔧 Navigation")
        st.markdown("---")
        st.markdown(f"**Dataset:** {len(df):,} records")
        st.markdown(f"**Machines:** {df['machine_id'].nunique()}")
        st.markdown(f"**Failure Rate:** {df['failure'].mean()*100:.1f}%")
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        st.markdown(f"- 🌡️ Avg Temp: {df['temperature'].mean():.1f}°F")
        st.markdown(f"- 📳 Avg Vibration: {df['vibration'].mean():.2f} mm/s")
        st.markdown(f"- 🔽 Avg Pressure: {df['pressure'].mean():.1f} psi")

    # ── Tabs ──
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard", "🔍 Data Explorer", "🎯 Model Performance",
        "🔮 Real-Time Prediction", "🛠️ Maintenance", "🧠 Explainability"
    ])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 — DASHBOARD OVERVIEW
    # ══════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown('<div class="section-title">Key Performance Indicators</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)

        total_machines = df["machine_id"].nunique()
        total_records = len(df)
        failure_rate = df["failure"].mean() * 100
        high_risk = len(df[df["failure"] == 1])

        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🏭</div>
                <div class="metric-value">{total_machines}</div>
                <div class="metric-label">Total Machines</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">📊</div>
                <div class="metric-value">{total_records:,}</div>
                <div class="metric-label">Total Records</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">⚠️</div>
                <div class="metric-value">{failure_rate:.1f}%</div>
                <div class="metric-label">Failure Rate</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon">🔴</div>
                <div class="metric-value">{high_risk:,}</div>
                <div class="metric-label">Failure Records</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Failure trend over time
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown('<div class="section-title">Failure Trend Over Time</div>', unsafe_allow_html=True)
            daily_failures = df.set_index("timestamp").resample("D")["failure"].sum().reset_index()
            fig = px.area(daily_failures, x="timestamp", y="failure",
                          template=PLOTLY_TEMPLATE,
                          color_discrete_sequence=[COLORS["accent"]])
            fig.update_layout(
                xaxis_title="Date", yaxis_title="Failure Count",
                height=380, margin=dict(l=20, r=20, t=30, b=20),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            fig.update_traces(fill="tozeroy", line=dict(width=2))
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown('<div class="section-title">Failures by Machine (Top 15)</div>', unsafe_allow_html=True)
            machine_failures = (df[df["failure"] == 1].groupby("machine_id").size()
                                .sort_values(ascending=False).head(15).reset_index(name="failures"))
            fig = px.bar(machine_failures, x="machine_id", y="failures",
                         template=PLOTLY_TEMPLATE,
                         color="failures", color_continuous_scale="Reds")
            fig.update_layout(
                xaxis_title="Machine", yaxis_title="Failure Count",
                height=380, margin=dict(l=20, r=20, t=30, b=20),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 — DATA EXPLORER
    # ══════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown('<div class="section-title">Interactive Data Explorer</div>', unsafe_allow_html=True)

        # Filters
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            selected_machines = st.multiselect(
                "Select Machines", df["machine_id"].unique(),
                default=df["machine_id"].unique()[:5]
            )
        with fc2:
            temp_range = st.slider("Temperature Range (°F)",
                                   float(df["temperature"].min()), float(df["temperature"].max()),
                                   (float(df["temperature"].min()), float(df["temperature"].max())))
        with fc3:
            vib_range = st.slider("Vibration Range (mm/s)",
                                  float(df["vibration"].min()), float(df["vibration"].max()),
                                  (float(df["vibration"].min()), float(df["vibration"].max())))

        filtered = df[
            (df["machine_id"].isin(selected_machines)) &
            (df["temperature"].between(*temp_range)) &
            (df["vibration"].between(*vib_range))
        ]

        st.markdown(f"**Showing {len(filtered):,} of {len(df):,} records**")
        st.dataframe(filtered.head(500), use_container_width=True, height=300)

        st.markdown("---")

        # Histograms
        st.markdown('<div class="section-title">Sensor Distributions</div>', unsafe_allow_html=True)
        sensor_cols = ["temperature", "vibration", "pressure", "humidity",
                       "voltage", "current", "rotational_speed", "operating_hours"]
        hist_c1, hist_c2 = st.columns(2)

        for i, col in enumerate(sensor_cols):
            with hist_c1 if i % 2 == 0 else hist_c2:
                fig = px.histogram(filtered, x=col, nbins=50,
                                   color="failure", barmode="overlay",
                                   template=PLOTLY_TEMPLATE,
                                   color_discrete_map={0: COLORS["primary"], 1: COLORS["danger"]},
                                   opacity=0.7)
                fig.update_layout(
                    title=f"{col.replace('_', ' ').title()} Distribution",
                    height=300, margin=dict(l=20, r=20, t=40, b=20),
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Sensor trends
        st.markdown('<div class="section-title">Sensor Trends Over Time</div>', unsafe_allow_html=True)
        trend_sensor = st.selectbox("Select Sensor", sensor_cols, index=0)
        trend_data = filtered.set_index("timestamp").resample("6h")[trend_sensor].mean().reset_index()
        fig = px.line(trend_data, x="timestamp", y=trend_sensor,
                      template=PLOTLY_TEMPLATE,
                      color_discrete_sequence=[COLORS["secondary"]])
        fig.update_layout(
            height=350, margin=dict(l=20, r=20, t=30, b=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Correlation heatmap
        st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)
        corr = filtered[sensor_cols + ["failure"]].corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale="RdBu_r", zmid=0,
            text=np.round(corr.values, 2), texttemplate="%{text}",
            textfont={"size": 10},
        ))
        fig.update_layout(
            height=500, template=PLOTLY_TEMPLATE,
            margin=dict(l=20, r=20, t=30, b=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3 — MODEL PERFORMANCE
    # ══════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown('<div class="section-title">Model Performance Evaluation</div>', unsafe_allow_html=True)

        # Prepare features for evaluation
        sensor_features = ["temperature", "vibration", "pressure", "humidity",
                           "voltage", "current", "rotational_speed", "operating_hours"]

        # Engineer features on the full dataset for evaluation
        eval_df = df.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
        eng_cols = ["temperature", "vibration", "pressure"]

        for col in eng_cols:
            eval_df[f"{col}_rolling_mean_5"] = eval_df.groupby("machine_id")[col].transform(
                lambda x: x.rolling(5, min_periods=1).mean())
            eval_df[f"{col}_rolling_std_5"] = eval_df.groupby("machine_id")[col].transform(
                lambda x: x.rolling(5, min_periods=1).std())
        for col in ["temperature", "vibration"]:
            for lag in [1, 3]:
                eval_df[f"{col}_lag_{lag}"] = eval_df.groupby("machine_id")[col].shift(lag)
        for col in eng_cols:
            eval_df[f"{col}_rate_of_change"] = eval_df.groupby("machine_id")[col].diff()

        eval_df = eval_df.fillna(0)

        # Get available feature columns
        available_features = [c for c in feature_cols if c in eval_df.columns]
        X_eval = eval_df[available_features].values
        y_eval = eval_df["failure"].values

        X_eval_scaled = scaler.transform(X_eval)

        y_pred = model.predict(X_eval_scaled)
        y_proba = model.predict_proba(X_eval_scaled)[:, 1]

        acc = accuracy_score(y_eval, y_pred)
        prec = precision_score(y_eval, y_pred, zero_division=0)
        rec = recall_score(y_eval, y_pred, zero_division=0)
        f1 = f1_score(y_eval, y_pred, zero_division=0)
        roc = roc_auc_score(y_eval, y_proba)

        # Metric cards
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        metric_data = [
            ("🎯", "Accuracy", acc, mc1),
            ("✅", "Precision", prec, mc2),
            ("📡", "Recall", rec, mc3),
            ("⚡", "F1 Score", f1, mc4),
            ("📈", "ROC-AUC", roc, mc5),
        ]
        for icon, label, val, col in metric_data:
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-value">{val:.4f}</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")

        perf_left, perf_right = st.columns(2)

        # ROC Curve
        with perf_left:
            st.markdown('<div class="section-title">ROC Curve</div>', unsafe_allow_html=True)
            fpr, tpr, _ = roc_curve(y_eval, y_proba)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                     name=f"Model (AUC = {roc:.4f})",
                                     line=dict(color=COLORS["primary"], width=3)))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                     name="Random", line=dict(color="gray", dash="dash")))
            fig.update_layout(
                xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                template=PLOTLY_TEMPLATE, height=400,
                margin=dict(l=20, r=20, t=30, b=20),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(x=0.5, y=0.1),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Confusion Matrix
        with perf_right:
            st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
            cm = confusion_matrix(y_eval, y_pred)
            fig = go.Figure(data=go.Heatmap(
                z=cm, x=["Predicted Normal", "Predicted Failure"],
                y=["Actual Normal", "Actual Failure"],
                colorscale=[[0, "#1a1a2e"], [1, COLORS["primary"]]],
                text=cm, texttemplate="%{text:,}",
                textfont={"size": 16},
                showscale=False,
            ))
            fig.update_layout(
                template=PLOTLY_TEMPLATE, height=400,
                margin=dict(l=20, r=20, t=30, b=20),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 4 — REAL-TIME PREDICTION
    # ══════════════════════════════════════════════════════════════════════
    with tab4:
        st.markdown('<div class="section-title">Real-Time Equipment Failure Prediction</div>',
                    unsafe_allow_html=True)
        st.markdown("Enter sensor readings below to predict equipment failure probability.")

        # Input sliders
        inp1, inp2, inp3, inp4 = st.columns(4)
        with inp1:
            in_temp = st.slider("🌡️ Temperature (°F)", 60.0, 110.0, 75.0, 0.5)
            in_vib = st.slider("📳 Vibration (mm/s)", 0.5, 8.0, 2.5, 0.1)
        with inp2:
            in_pres = st.slider("🔽 Pressure (psi)", 25.0, 75.0, 55.0, 0.5)
            in_hum = st.slider("💧 Humidity (%)", 30.0, 80.0, 50.0, 0.5)
        with inp3:
            in_volt = st.slider("⚡ Voltage (V)", 210.0, 250.0, 230.0, 0.5)
            in_curr = st.slider("🔌 Current (A)", 5.0, 20.0, 12.0, 0.5)
        with inp4:
            in_rpm = st.slider("🔄 Rotational Speed (RPM)", 1000.0, 4000.0, 2500.0, 50.0)
            in_hrs = st.slider("⏱️ Operating Hours", 0.0, 20000.0, 5000.0, 100.0)

        if st.button("🔮 Predict Failure", use_container_width=True, type="primary"):
            # Build feature vector matching the trained feature columns
            base_values = {
                "temperature": in_temp, "vibration": in_vib, "pressure": in_pres,
                "humidity": in_hum, "voltage": in_volt, "current": in_curr,
                "rotational_speed": in_rpm, "operating_hours": in_hrs,
            }

            # Create a complete feature vector
            input_vector = []
            for col_name in feature_cols:
                if col_name in base_values:
                    input_vector.append(base_values[col_name])
                elif "rolling_mean" in col_name:
                    base_col = col_name.split("_rolling_mean")[0]
                    input_vector.append(base_values.get(base_col, 0))
                elif "rolling_std" in col_name:
                    input_vector.append(0)  # Single reading has no std
                elif "lag" in col_name:
                    base_col = col_name.split("_lag")[0]
                    input_vector.append(base_values.get(base_col, 0))
                elif "rate_of_change" in col_name:
                    input_vector.append(0)  # Single reading has no rate of change
                else:
                    input_vector.append(0)

            input_array = np.array(input_vector).reshape(1, -1)
            input_scaled = scaler.transform(input_array)

            prob = model.predict_proba(input_scaled)[0][1]
            risk_label, risk_class = get_risk_level(prob)

            st.markdown("---")

            res1, res2 = st.columns(2)

            with res1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-icon">📊</div>
                    <div class="metric-value">{prob*100:.1f}%</div>
                    <div class="metric-label">Failure Probability</div>
                </div>""", unsafe_allow_html=True)

            with res2:
                st.markdown(f"""
                <div class="risk-badge {risk_class}">
                    {risk_label}
                </div>""", unsafe_allow_html=True)

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                number={"suffix": "%", "font": {"size": 40, "color": "white"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "white"},
                    "bar": {"color": "#6c5ce7"},
                    "bgcolor": "#1a1a2e",
                    "borderwidth": 2,
                    "bordercolor": "rgba(255,255,255,0.1)",
                    "steps": [
                        {"range": [0, 30], "color": "#00b894"},
                        {"range": [30, 60], "color": "#fdcb6e"},
                        {"range": [60, 80], "color": "#e17055"},
                        {"range": [80, 100], "color": "#d63031"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 4},
                        "thickness": 0.8,
                        "value": prob * 100,
                    },
                },
            ))
            fig.update_layout(
                height=300, template=PLOTLY_TEMPLATE,
                margin=dict(l=30, r=30, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Show recommendations
            st.markdown('<div class="section-title">Maintenance Recommendations</div>',
                        unsafe_allow_html=True)
            recs = get_recommendations(in_temp, in_vib, in_pres, in_hum,
                                       in_volt, in_curr, in_rpm, in_hrs, prob)
            for title, desc, severity in recs:
                st.markdown(f"""
                <div class="rec-card {severity}">
                    <strong>{title}</strong><br>
                    <span style="color: #a0aec0;">{desc}</span>
                </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 5 — MAINTENANCE RECOMMENDATION ENGINE
    # ══════════════════════════════════════════════════════════════════════
    with tab5:
        st.markdown('<div class="section-title">Maintenance Recommendation Engine</div>',
                    unsafe_allow_html=True)
        st.markdown("Select a machine to analyze its current sensor readings and get maintenance recommendations.")

        sel_machine = st.selectbox("Select Machine", sorted(df["machine_id"].unique()))
        machine_data = df[df["machine_id"] == sel_machine].sort_values("timestamp")
        latest = machine_data.iloc[-1]

        # Latest readings
        st.markdown('<div class="section-title">Latest Sensor Readings</div>', unsafe_allow_html=True)
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.metric("🌡️ Temperature", f"{latest['temperature']:.1f}°F")
            st.metric("📳 Vibration", f"{latest['vibration']:.2f} mm/s")
        with sc2:
            st.metric("🔽 Pressure", f"{latest['pressure']:.1f} psi")
            st.metric("💧 Humidity", f"{latest['humidity']:.1f}%")
        with sc3:
            st.metric("⚡ Voltage", f"{latest['voltage']:.1f} V")
            st.metric("🔌 Current", f"{latest['current']:.1f} A")
        with sc4:
            st.metric("🔄 RPM", f"{latest['rotational_speed']:.0f}")
            st.metric("⏱️ Hours", f"{latest['operating_hours']:.0f}")

        # Generate recommendations for this machine
        recs = get_recommendations(
            latest["temperature"], latest["vibration"], latest["pressure"],
            latest["humidity"], latest["voltage"], latest["current"],
            latest["rotational_speed"], latest["operating_hours"],
            latest["failure"]
        )

        st.markdown('<div class="section-title">Recommendations</div>', unsafe_allow_html=True)
        for title, desc, severity in recs:
            st.markdown(f"""
            <div class="rec-card {severity}">
                <strong>{title}</strong><br>
                <span style="color: #a0aec0;">{desc}</span>
            </div>""", unsafe_allow_html=True)

        # Machine sensor trend
        st.markdown('<div class="section-title">Machine Sensor History</div>', unsafe_allow_html=True)
        trend_col = st.selectbox("Select Sensor for Trend", sensor_features, key="maint_trend")
        fig = px.line(machine_data, x="timestamp", y=trend_col,
                      template=PLOTLY_TEMPLATE,
                      color_discrete_sequence=[COLORS["secondary"]])
        fig.update_layout(
            height=350, margin=dict(l=20, r=20, t=30, b=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )

        # Highlight failure points
        failures = machine_data[machine_data["failure"] == 1]
        if len(failures) > 0:
            fig.add_trace(go.Scatter(
                x=failures["timestamp"], y=failures[trend_col],
                mode="markers", name="Failure Event",
                marker=dict(color=COLORS["danger"], size=8, symbol="x"),
            ))

        st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 6 — EXPLAINABILITY
    # ══════════════════════════════════════════════════════════════════════
    with tab6:
        st.markdown('<div class="section-title">Model Explainability (SHAP)</div>', unsafe_allow_html=True)
        st.markdown("Understanding which sensor readings contribute most to equipment failure predictions.")

        # Display saved SHAP plots
        shap_summary_path = os.path.join(PLOTS_DIR, "shap_summary.png")
        shap_importance_path = os.path.join(PLOTS_DIR, "shap_feature_importance.png")

        if os.path.exists(shap_summary_path):
            st.markdown('<div class="section-title">SHAP Summary Plot</div>', unsafe_allow_html=True)
            st.image(shap_summary_path, use_container_width=True)
            st.markdown("""
            > **How to read this plot:** Each dot represents a single prediction.
            > The x-axis shows the SHAP value (impact on prediction).
            > Red = high feature value, Blue = low feature value.
            > Features are sorted by importance (top = most important).
            """)
        else:
            st.warning("SHAP summary plot not found. Run `python train_model.py` to generate it.")

        if os.path.exists(shap_importance_path):
            st.markdown('<div class="section-title">SHAP Feature Importance</div>', unsafe_allow_html=True)
            st.image(shap_importance_path, use_container_width=True)
            st.markdown("""
            > **Feature Importance** shows the average absolute SHAP value for each feature.
            > Higher values indicate the feature has more influence on the model's predictions.
            """)
        else:
            st.warning("SHAP feature importance plot not found. Run `python train_model.py` to generate it.")

        # Interactive feature importance from the model
        st.markdown('<div class="section-title">Model Feature Importance (Interactive)</div>',
                    unsafe_allow_html=True)

        if hasattr(model, "feature_importances_"):
            importances = pd.DataFrame({
                "Feature": feature_cols,
                "Importance": model.feature_importances_,
            }).sort_values("Importance", ascending=True).tail(20)

            fig = px.bar(importances, x="Importance", y="Feature", orientation="h",
                         template=PLOTLY_TEMPLATE,
                         color="Importance", color_continuous_scale="Viridis")
            fig.update_layout(
                height=500, margin=dict(l=20, r=20, t=30, b=20),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
                yaxis=dict(tickfont=dict(size=11)),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Key insights
        st.markdown('<div class="section-title">Key Insights</div>', unsafe_allow_html=True)
        st.markdown("""
        Based on the SHAP analysis and model feature importances:

        - 🌡️ **Temperature** is typically the strongest predictor — higher temperatures correlate with increased failure risk.
        - 📳 **Vibration** ranks high — excessive vibration often indicates mechanical wear or misalignment.
        - 🔽 **Pressure** drops are a key warning sign — low pressure suggests hydraulic or pneumatic system issues.
        - ⚡ **Voltage deviations** from the nominal 230V range contribute to failure prediction.
        - ⏱️ **Operating hours** capture equipment aging effects — older equipment naturally has higher failure rates.
        - 📊 **Engineered features** (rolling means, lag values) capture temporal patterns that raw readings miss.
        """)


if __name__ == "__main__":
    main()
