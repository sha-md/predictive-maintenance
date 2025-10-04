# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Predictive Maintenance", layout="wide")

# --------- Utilities ----------
def clean_colname(c):
    # same cleaning as training pipeline: remove brackets and replace spaces
    return c.replace(" ", "_").replace("[","").replace("]","").replace("<","lt").replace(">","gt")

def preprocess_input(df, feature_cols, scaler):
    # ensure columns cleaned and present
    df = df.copy()
    # unify column names
    df.columns = [clean_colname(c) for c in df.columns]
    # Engineer TempDiff if not present
    if ("Process_temperature_K" in feature_cols or "Process_temperature_[K]" in feature_cols):
        # try many name variants; assume original names replaced in saved feature_cols
        pass
    # create temp diff using likely column name variants
    candidates_air = [c for c in df.columns if "air" in c.lower() and "temp" in c.lower()]
    candidates_proc = [c for c in df.columns if "process" in c.lower() and "temp" in c.lower()]
    if "Temp Diff" not in df.columns and len(candidates_air) and len(candidates_proc):
        df["Temp_Diff"] = df[candidates_proc[0]] - df[candidates_air[0]]
    # ensure all expected feature columns exist (fill missing with 0)
    X = pd.DataFrame(columns=feature_cols)
    for col in feature_cols:
        if col in df.columns:
            X[col] = df[col]
        else:
            X[col] = 0.0
    # scale using loaded scaler (expects same order)
    X_scaled = scaler.transform(X)
    return X, X_scaled

# --------- Load artifacts ----------
MODEL_PATH = "models/best_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/feature_columns.json"
METRICS_PATH = "models/metrics.json"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model not found at {MODEL_PATH}. Please run training notebook cell that saves artifacts.")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

with open(FEATURES_PATH, "r") as f:
    feature_cols = json.load(f)

metrics = {}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

# display header
st.title("⚙️ Predictive Maintenance — Machine Failure Predictor")
st.markdown("Upload sensor data or enter a single record. Model predicts probability of machine failure.")

# sidebar show model metrics
st.sidebar.header("Model Metrics")
if metrics:
    # metrics is a dict of dicts; display best metric quickly
    try:
        # show best model and its F1 from metrics file
        df_metrics = pd.DataFrame(metrics)
        st.sidebar.dataframe(df_metrics.T)
    except:
        st.sidebar.write(metrics)
else:
    st.sidebar.write("No metrics.json found.")

# --------- Input modes ----------
mode = st.sidebar.radio("Input Mode", ("Upload CSV", "Single Record"))

if mode == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV with sensor readings (first row = header)", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
        st.write("Uploaded sample:")
        st.dataframe(input_df.head())
        if scaler is None:
            st.error("Scaler not found. Cannot preprocess. Save scaler as models/scaler.pkl in your repo.")
        else:
            X_raw, X_scaled = preprocess_input(input_df, feature_cols, scaler)
            # predict
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_scaled)[:,1]
            else:
                # fallback to predict
                preds = model.predict(X_scaled)
                probs = preds
            preds = (probs >= 0.5).astype(int)
            out = input_df.copy()
            out["failure_prob"] = probs
            out["predicted_failure"] = preds
            st.subheader("Predictions (first 20 rows)")
            st.dataframe(out.head(20))
            # show distribution
            st.subheader("Predicted failure probability distribution")
            st.bar_chart(pd.Series(probs).value_counts().sort_index())

else:
    # Single record entry
    st.subheader("Enter single sample values")
    
    sample_path = "models/sample_input.csv"
    sample_vals = {}
    if os.path.exists(sample_path):
        sample = pd.read_csv(sample_path)
        sample0 = sample.iloc[0].to_dict()
        sample_vals = sample0

    def input_number(name, key, default=0.0):
        return st.number_input(name, key=key, value=float(sample_vals.get(name, default)))

    # Try to find likely keys in feature_cols (loose mapping)
    # We'll create inputs with friendly labels
    at_name = next((c for c in feature_cols if "Air" in c or "air" in c), feature_cols[0])
    pt_name = next((c for c in feature_cols if "Process" in c or "process" in c), feature_cols[1] if len(feature_cols)>1 else feature_cols[0])
    rpm_name = next((c for c in feature_cols if "rpm" in c.lower()), feature_cols[2] if len(feature_cols)>2 else feature_cols[0])
    torque_name = next((c for c in feature_cols if "Torque" in c or "torque" in c), feature_cols[3] if len(feature_cols)>3 else feature_cols[0])
    wear_name = next((c for c in feature_cols if "wear" in c.lower()), feature_cols[4] if len(feature_cols)>4 else feature_cols[0])
    # Show inputs:
    air_temp = st.number_input(f"{at_name}", value=float(sample_vals.get(at_name, 298.2)))
    proc_temp = st.number_input(f"{pt_name}", value=float(sample_vals.get(pt_name, 308.6)))
    rpm = st.number_input(f"{rpm_name}", value=float(sample_vals.get(rpm_name, 1400)))
    torque = st.number_input(f"{torque_name}", value=float(sample_vals.get(torque_name, 45.0)))
    wear = st.number_input(f"{wear_name}", value=float(sample_vals.get(wear_name, 5.0)))
    # type
    type_label = None
    type_candidates = [c for c in feature_cols if "Type" in c or "type" in c]
    if type_candidates:
        type_label = type_candidates[0]
        type_val = st.selectbox("Type (if used in your model)", options=["L","M","H"], index=1)
    else:
        type_val = None

    
    row = {}
    for c in feature_cols:
        row[c] = 0.0
    # set values by matching names (best-effort)
    # try to set matching keys
    def set_by_like(key_candidates, val):
        for col in feature_cols:
            if any(k.lower() in col.lower() for k in key_candidates):
                row[col] = val
                return True
        return False

    set_by_like(["air","Air","air_temperature","air_temp"], air_temp)
    set_by_like(["process","Process","process_temperature","process_temp"], proc_temp)
    set_by_like(["rotational","rpm"], rpm)
    set_by_like(["torque"], torque)
    set_by_like(["wear","tool_wear"], wear)
    if type_val and type_label in feature_cols:
        
        mapping = {"H":0, "L":1, "M":2}
        row[type_label] = mapping.get(type_val, 1)

    input_df = pd.DataFrame([row])
    st.write("Input features used for prediction:")
    st.dataframe(input_df.T)

    if scaler is None:
        st.error("Scaler not found. Save `models/scaler.pkl` in repo.")
    else:
        X_raw, X_scaled = preprocess_input(input_df, feature_cols, scaler)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_scaled)[:,1]
        else:
            try:
                probs = model.decision_function(X_scaled)
                # normalize to 0-1
                probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-9)
            except:
                preds = model.predict(X_scaled)
                probs = preds
        pred = int((probs >= 0.5).astype(int)[0])
        st.success(f"Predicted: {'Failure' if pred==1 else 'No Failure'}  —  Probability of failure: {probs[0]:.3f}")

# -------- Feature importances ----------
st.subheader("Model Feature Importances (if available)")
try:
    if hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        # Map to readable names (feature_cols)
        fig, ax = plt.subplots(figsize=(6,4))
        idx = np.argsort(fi)[::-1][:10]
        ax.barh([feature_cols[i] for i in idx[::-1]], fi[idx[::-1]])
        ax.set_xlabel("Importance")
        st.pyplot(fig)
    else:
        st.write("Feature importances not available for this model type.")
except Exception as e:
    st.write("Error retrieving feature importances:", e)
