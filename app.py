# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os
import plotly.graph_objects as go

st.set_page_config(page_title="Predictive Maintenance", layout="wide")

# --------- Utilities ----------
def preprocess_input(df, feature_cols, scaler):
    """Prepare input dataframe for model prediction with original feature names"""
    df = df.copy()

    # Compute Temp Diff if missing
    if "Temp Diff" in feature_cols and "Temp Diff" not in df.columns:
        air_col  = next((c for c in df.columns if "Air" in c), None)
        proc_col = next((c for c in df.columns if "Process" in c), None)
        df["Temp Diff"] = (df[proc_col] - df[air_col]) if (air_col and proc_col) else 0.0

    # Map Type column to numeric (match training)
    if "Type" in df.columns and df["Type"].dtype == object:
        df["Type"] = df["Type"].replace({"H":0,"L":1,"M":2})

    # Decide the exact column order to use:
    # If the scaler was fit on a DataFrame, it will expose feature_names_in_.
    # Prefer that order; otherwise fall back to feature_cols from JSON.
    if hasattr(scaler, "feature_names_in_"):
        expected_order = list(scaler.feature_names_in_)
    else:
        expected_order = list(feature_cols)

    # Build X in the expected order, filling missing with 0.0
    X = df.reindex(columns=expected_order, fill_value=0.0)

    # Ensure numeric dtype
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0.0)

    # KEY: pass ndarray to bypass sklearn's feature-name check
    X_scaled = scaler.transform(X.values.astype(float))

    return X, X_scaled

def auto_rename_columns(df):
    """
    Automatically rename columns to match model feature names.
    Accepts spaces, underscores, lowercase, etc.
    """
    mapping = {
        "air temperature [k]": "Air temperature [K]",
        "air_temperature_k": "Air temperature [K]",
        "process temperature [k]": "Process temperature [K]",
        "process_temperature_k": "Process temperature [K]",
        "rotational speed [rpm]": "Rotational speed [rpm]",
        "rotational_speed_rpm": "Rotational speed [rpm]",
        "torque [nm]": "Torque [Nm]",
        "torque_nm": "Torque [Nm]",
        "tool wear [min]": "Tool wear [min]",
        "tool_wear_min": "Tool wear [min]",
        "temp diff": "Temp Diff",
        "temp_diff": "Temp Diff",
        "type": "Type"
    }
    new_cols = {}
    for c in df.columns:
        key = c.strip().lower().replace("_"," ")
        if key in mapping:
            new_cols[c] = mapping[key]
    df = df.rename(columns=new_cols)
    return df

# --------- Load artifacts ----------
MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/feature_columns.json"
METRICS_PATH = "models/metrics.json"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at {MODEL_PATH}. Run the training notebook first.")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

with open(FEATURES_PATH, "r") as f:
    feature_cols = json.load(f)

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
            input_df = auto_rename_columns(input_df)  # <-- Automatic renaming
            st.write("Uploaded sample after automatic column mapping:")
            st.dataframe(input_df.head())
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            st.stop()

        if scaler is None:
            st.error("Scaler not found. Cannot preprocess.")
        else:
            try:
                X_raw, X_scaled = preprocess_input(input_df, feature_cols, scaler)
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
    for feature in feature_cols:
        if feature == "Type":
            val = st.selectbox("Type", options=["H","L","M"], index=1)
            input_data[feature] = {"H":0,"L":1,"M":2}.get(val,1)
        elif feature == "Temp Diff":
            continue
        else:
            val = st.number_input(feature, value=0.0)
            input_data[feature] = val

    input_df = pd.DataFrame([input_data])
    if "Temp Diff" in feature_cols and "Temp Diff" not in input_df.columns:
        if "Process temperature [K]" in input_df.columns and "Air temperature [K]" in input_df.columns:
            input_df["Temp Diff"] = input_df["Process temperature [K]"] - input_df["Air temperature [K]"]
        else:
            input_df["Temp Diff"] = 0.0

    #st.write("Input features used for prediction:")
    #st.dataframe(input_df.T)

    if scaler is None:
        st.error("Scaler not found.")
    else:
        try:
            X_raw, X_scaled = preprocess_input(input_df, feature_cols, scaler)
            probs = model.predict_proba(X_scaled)[:,1] if hasattr(model, "predict_proba") else model.predict(X_scaled)
            pred = int((probs >= 0.5).astype(int)[0])
            st.success(f"Predicted: {'Failure' if pred==1 else 'No Failure'} — Probability of failure: {probs[0]:.3f}")

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
