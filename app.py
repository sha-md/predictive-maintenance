# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os
import plotly.graph_objects as go

st.set_page_config(page_title="Predictive Maintenance", layout="wide")

# --------- Feature order ----------
FEATURE_ORDER = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Temp Diff",
    "Type"
]

# --------- Utilities ----------
def preprocess_input(df, feature_cols, scaler):
    """Prepare input dataframe for model prediction with exact feature names and order"""
    df = df.copy()

    # Create Temp Diff if missing
    if "Temp Diff" in feature_cols and "Temp Diff" not in df.columns:
        if "Process temperature [K]" in df.columns and "Air temperature [K]" in df.columns:
            df["Temp Diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
        else:
            df["Temp Diff"] = 0.0

    # Map Type column to numeric
    if "Type" in df.columns:
        df["Type"] = df["Type"].replace({"H":0,"L":1,"M":2}).astype(float)

    # Fill missing columns with 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0.0

    # Keep only columns in feature_cols in the correct order
    X = df[feature_cols]

    # Scale
    X_scaled = scaler.transform(X)
    return X, X_scaled

# --------- Load artifacts ----------
MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"
METRICS_PATH = "models/metrics.json"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at {MODEL_PATH}. Run the training notebook first.")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

metrics = {}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

# --------- Header ----------
st.title("⚙️ Predictive Maintenance — Machine Failure Predictor")
st.markdown("Upload sensor data or enter a single record. Model predicts probability of machine failure.")

# Sidebar metrics
st.sidebar.header("Model Metrics")
if metrics:
    try:
        df_metrics = pd.DataFrame(metrics)
        st.sidebar.dataframe(df_metrics.T)
    except:
        st.sidebar.write(metrics)
else:
    st.sidebar.write("No metrics.json found.")

# --------- Input Mode ----------
mode = st.sidebar.radio("Input Mode", ("Upload CSV", "Single Record"))

# ------------------ CSV Upload ------------------
if mode == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV with sensor readings (first row = header)", type=["csv"])
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.write("Uploaded sample:")
            st.dataframe(input_df.head())
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            st.stop()

        if scaler is None:
            st.error("Scaler not found. Cannot preprocess.")
        else:
            try:
                # Reorder and preprocess
                X_raw, X_scaled = preprocess_input(input_df, FEATURE_ORDER, scaler)

                # Predict
                probs = model.predict_proba(X_scaled)[:,1] if hasattr(model, "predict_proba") else model.predict(X_scaled)
                preds = (probs >= 0.5).astype(int)

                out = input_df.copy()
                out["failure_prob"] = probs
                out["predicted_failure"] = preds
                st.subheader("Predictions (first 20 rows)")
                st.dataframe(out.head(20))

                # Probability distribution
                st.subheader("Predicted failure probability distribution")
                st.bar_chart(pd.Series(probs).value_counts().sort_index())

                # Gauge for average probability
                avg_prob = probs.mean()
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = float(avg_prob),
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Average Failure Probability"},
                    gauge = {'axis': {'range': [0, 1]},
                             'bar': {'color': "red"},
                             'steps' : [
                                 {'range': [0, 0.5], 'color': "green"},
                                 {'range': [0.5, 0.8], 'color': "yellow"},
                                 {'range': [0.8, 1], 'color': "red"}]}))
                st.plotly_chart(fig)
            except Exception as e:
                st.error(f"Error during preprocessing or prediction: {e}")

# ------------------ Single Record ------------------
else:
    st.subheader("Enter single sample values")

    input_data = {}
    # Dynamically create input fields for every feature
    for feature in FEATURE_ORDER:
        if feature == "Type":
            val = st.selectbox("Type", options=["H","L","M"], index=1)
            input_data[feature] = {"H":0,"L":1,"M":2}.get(val,1)
        elif feature == "Temp Diff":
            continue  # Will compute later
        else:
            val = st.number_input(feature, value=0.0)
            input_data[feature] = val

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Compute Temp Diff if missing
    if "Temp Diff" in FEATURE_ORDER and "Temp Diff" not in input_df.columns:
        if "Process temperature [K]" in input_df.columns and "Air temperature [K]" in input_df.columns:
            input_df["Temp Diff"] = input_df["Process temperature [K]"] - input_df["Air temperature [K]"]
        else:
            input_df["Temp Diff"] = 0.0

    st.write("Input features used for prediction:")
    st.dataframe(input_df.T)

    if scaler is None:
        st.error("Scaler not found.")
    else:
        try:
            X_raw, X_scaled = preprocess_input(input_df, FEATURE_ORDER, scaler)
            probs = model.predict_proba(X_scaled)[:,1] if hasattr(model, "predict_proba") else model.predict(X_scaled)
            pred = int((probs >= 0.5).astype(int)[0])
            st.success(f"Predicted: {'Failure' if pred==1 else 'No Failure'} — Probability of failure: {probs[0]:.3f}")

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = float(probs[0]),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Failure Probability"},
                gauge = {'axis': {'range': [0, 1]},
                         'bar': {'color': "red"},
                         'steps' : [
                             {'range': [0, 0.5], 'color': "green"},
                             {'range': [0.5, 0.8], 'color': "yellow"},
                             {'range': [0.8, 1], 'color': "red"}]}))
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error during preprocessing or prediction: {e}")
