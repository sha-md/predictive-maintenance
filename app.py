# ======================================================
# ⚙️ Predictive Maintenance Dashboard
# ======================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="⚙️ Predictive Maintenance Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ----------------- UTILITIES -----------------
def clean_colname(c: str) -> str:
    """Clean column names consistently with training pipeline."""
    return c.replace(" ", "_").replace("[","").replace("]","").replace("<","lt").replace(">","gt")

@st.cache_resource
def load_artifacts():
    """Load model, scaler, and metadata artifacts."""
    model_path = "models/best_model.pkl"
    scaler_path = "models/scaler.pkl"
    features_path = "models/feature_columns.json"
    metrics_path = "models/metrics.json"

    if not os.path.exists(model_path):
        st.error("🚨 Model not found! Please train and save model artifacts in /models.")
        st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    with open(features_path, "r") as f:
        feature_cols = json.load(f)

    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    return model, scaler, feature_cols, metrics


def preprocess_input(df: pd.DataFrame, feature_cols: list, scaler):
    """Clean and scale input features."""
    df = df.copy()
    df.columns = [clean_colname(c) for c in df.columns]

    # Create engineered features
    air_col = next((c for c in df.columns if "air" in c.lower() and "temp" in c.lower()), None)
    proc_col = next((c for c in df.columns if "process" in c.lower() and "temp" in c.lower()), None)
    if air_col and proc_col and "Temp_Diff" not in df.columns:
        df["Temp_Diff"] = df[proc_col] - df[air_col]

    # Ensure all expected features exist
    X = pd.DataFrame({col: df[col] if col in df.columns else 0.0 for col in feature_cols})
    X_scaled = scaler.transform(X) if scaler is not None else X
    return X, X_scaled


def predict(model, X_scaled):
    """Generate predictions and probabilities."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[:, 1]
    else:
        try:
            probs = model.decision_function(X_scaled)
            probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-9)
        except:
            probs = model.predict(X_scaled)
    preds = (probs >= 0.5).astype(int)
    return preds, probs


# ----------------- LOAD ARTIFACTS -----------------
model, scaler, feature_cols, metrics = load_artifacts()

# ----------------- HEADER -----------------
st.title("⚙️ Predictive Maintenance — Machine Failure Predictor")
st.markdown("Predict **machine failure probability** using sensor data. Upload your dataset or test a single record.")

# Sidebar
st.sidebar.header("📊 Model Information")
if metrics:
    df_metrics = pd.DataFrame(metrics).T
    st.sidebar.dataframe(df_metrics)
else:
    st.sidebar.warning("No metrics.json found.")
st.sidebar.info(f"Model: `{os.path.basename('models/best_model.pkl')}`")

mode = st.sidebar.radio("Choose Input Mode:", ["Upload CSV", "Single Record"])

# =====================================================
# 📁 UPLOAD CSV MODE
# =====================================================
if mode == "Upload CSV":
    uploaded_file = st.file_uploader("📁 Upload CSV file with sensor readings", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())

        X_raw, X_scaled = preprocess_input(df, feature_cols, scaler)
        preds, probs = predict(model, X_scaled)

        result = df.copy()
        result["Failure_Probability"] = probs
        result["Predicted_Failure"] = preds

        # KPI summary
        st.markdown("### 📈 Prediction Summary")
        col1, col2, col3 = st.columns(3)
        total = len(result)
        failures = result["Predicted_Failure"].sum()
        col1.metric("Total Machines", total)
        col2.metric("Predicted Failures", failures)
        col3.metric("Failure Rate", f"{(failures / total) * 100:.1f}%")

        # --- Graph 1: Failure Counts ---
        st.markdown("### ⚙️ Failure Count Overview")
        fail_counts = result["Predicted_Failure"].value_counts().sort_index()
        labels = ["No Failure", "Failure"]
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.bar(labels, fail_counts, color=["#2ecc71", "#e74c3c"])
        ax1.set_ylabel("Number of Samples")
        ax1.set_title("Machine Failure Prediction Counts")
        st.pyplot(fig1)

        # --- Graph 2: Probability Distribution ---
        st.markdown("### 📊 Failure Probability Distribution")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.hist(result["Failure_Probability"], bins=20, color="#3498db", alpha=0.7)
        ax2.set_xlabel("Predicted Failure Probability")
        ax2.set_ylabel("Count")
        ax2.set_title("Failure Probability Histogram")
        st.pyplot(fig2)

        # --- Download predictions ---
        st.download_button("📥 Download Predictions", result.to_csv(index=False).encode(), "predictions.csv", "text/csv")


# =====================================================
# 🧮 SINGLE RECORD MODE
# =====================================================
else:
    st.markdown("### 🔧 Enter Machine Sensor Readings")

    with st.form("single_input"):
        c1, c2, c3 = st.columns(3)
        air_temp = c1.number_input("Air Temperature (K)", value=298.2)
        proc_temp = c2.number_input("Process Temperature (K)", value=308.6)
        rpm = c3.number_input("Rotational Speed (rpm)", value=1400)
        torque = c1.number_input("Torque (Nm)", value=45.0)
        wear = c2.number_input("Tool Wear (min)", value=5.0)
        type_val = c3.selectbox("Machine Type", ["L", "M", "H"], index=1)
        submitted = st.form_submit_button("🔮 Predict")

    if submitted:
        row = {col: 0.0 for col in feature_cols}
        mapping = {"H": 0, "L": 1, "M": 2}

        for col in feature_cols:
            if "air" in col.lower(): row[col] = air_temp
            elif "process" in col.lower(): row[col] = proc_temp
            elif "rpm" in col.lower(): row[col] = rpm
            elif "torque" in col.lower(): row[col] = torque
            elif "wear" in col.lower(): row[col] = wear
            elif "type" in col.lower(): row[col] = mapping[type_val]

        df_input = pd.DataFrame([row])
        _, X_scaled = preprocess_input(df_input, feature_cols, scaler)
        preds, probs = predict(model, X_scaled)

        pred = preds[0]
        prob = float(probs[0])
        st.success(f"Predicted: {'⚠️ FAILURE' if pred==1 else '✅ No Failure'} — Probability: {prob:.2f}")

        # ------------------- GAUGE VISUALIZATION -------------------
        st.markdown("### 🧭 Machine Health Gauge")
        prob_percent = prob * 100

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob_percent,
            number={'suffix': "%"},
            delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred" if prob_percent > 50 else "green"},
                'steps': [
                    {'range': [0, 30], 'color': '#2ecc71'},
                    {'range': [30, 70], 'color': '#f1c40f'},
                    {'range': [70, 100], 'color': '#e74c3c'}
                ],
                'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.8, 'value': prob_percent}
            },
            title={'text': "Failure Probability", 'font': {'size': 20}}
        ))

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.plotly_chart(fig, use_container_width=True)

        # ------------------- OPTIONAL: MULTI-GAUGE DASHBOARD -------------------
        st.markdown("### ⚙️ Sensor Overview Dashboard")

        gauges = {
            "Air Temperature (K)": air_temp,
            "Process Temperature (K)": proc_temp,
            "Rotational Speed (rpm)": rpm,
            "Torque (Nm)": torque,
            "Tool Wear (min)": wear
        }

        cols = st.columns(len(gauges))
        for (label, val), col in zip(gauges.items(), cols):
            with col:
                fig_sensor = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=val,
                    title={'text': label},
                    gauge={'axis': {'range': [0, val * 1.5]}, 'bar': {'color': "#3498db"}}
                ))
                st.plotly_chart(fig_sensor, use_container_width=True)
