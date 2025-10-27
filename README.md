# ⚙️ Predictive Maintenance – Machine Failure Prediction

Machine failure prediction using classical ML & Deep Learning to minimize downtime and maintenance costs.

## 🚀 Live Demo
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://predictive-maintenance-dh9qkfibvu76tfgm2asn2y.streamlit.app/)

## 📑 Table of Contents
1. [Project Overview](#project-overview)  
2. [Business Problem](#business-problem)  
3. [Dataset](#dataset)  
4. [Data Preprocessing](#data-preprocessing)  
5. [Modeling](#modeling)  
6. [Results](#results)  
7. [Interpretability](#interpretability)  
8. [Streamlit App](#streamlit-app)  
9. [Aligned Coursework](#aligned-coursework)  
10. [Cost Savings Impact](#cost-savings-impact)  
11. [Future Improvements](#future-improvements)  
12. [Installation](#installation)  
13. [Project Structure](#project-structure)

---

## Project Overview
This project leverages **Machine Learning** and **Deep Learning** to predict equipment failures before they happen.  

I developed a full end-to-end pipeline:
- Data cleaning, feature engineering & class balancing  
- Model comparison between classical ML and DNN  
- Model interpretability using feature importance  
- Interactive web app using **:contentReference[oaicite:0]{index=0}** for real-time predictions.

✅ **Goal**: Help manufacturing teams schedule preventive maintenance and avoid unplanned downtime.

---

## Business Problem
Unplanned equipment failures lead to:
- Production downtime  
- High repair & replacement costs  
- Worker safety risks  

With a reliable **failure prediction system**, companies can:
- Detect issues early  
- Schedule maintenance efficiently  
- Save operational costs  
- Improve overall equipment effectiveness (OEE)

---

## Dataset
- **Source**: :contentReference[oaicite:1]{index=1} (UCI ML Repository)  
- **Size**: ~10,000 records  
- **Features**: torque, temperature, rotational speed, tool wear, machine type, etc.  
- **Target**: `Machine failure` (binary classification)

---

## Data Preprocessing
- Dropped non-predictive IDs  
- Encoded categorical variable (`Type`)  
- Standardized continuous features with `StandardScaler`  
- Fixed class imbalance using **:contentReference[oaicite:2]{index=2}**  
- Engineered `Temp Diff` (Process − Air temperature)

---

## Modeling
I trained and compared several ML models along with a DNN baseline:

| Model                  | Accuracy | F1 Score |
|--------------------------|----------|----------|
| Logistic Regression      | 0.82     | 0.82     |
| Random Forest            | 0.99     | 0.99     |
| XGBoost                  | 0.99     | 0.99     |
| KNN                      | 0.96     | 0.96     |
| SVM                      | 0.95     | 0.95     |
| **DNN (Keras)**          | **0.96** | **0.96** |

🟢 **Best performer**: Tree-based models (Random Forest & XGBoost)  
⚡ DNN also showed competitive performance.

---

## Results
- Very high classification accuracy  
- **Low false negatives** — critical for safety  
- Confusion matrix confirmed strong predictive performance  
- Consistent results after cross-validation

---

## Interpretability
- Feature importance analysis showed:
  - **Torque** and **Tool wear** were strong failure indicators  
- Training curves (DNN) confirmed good convergence without overfitting

---

## Streamlit App
An interactive web application built with **:contentReference[oaicite:3]{index=3}**.  


