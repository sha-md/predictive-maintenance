# Predictive Maintenance – Machine Failure Prediction

A machine learning and deep learning project that predicts **machine failures** before they occur, enabling preventive maintenance, minimizing downtime, and reducing operational costs.  

---

## Table of Contents
- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Why This Project Matters](#why-this-project-matters)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Engineering and MLOps Foundations](#engineering-and-mlops-foundations)
- [Modeling](#modeling)
- [Interpretability](#interpretability)
- [Cost–Benefit Impact](#costbenefit-impact)
- [Results](#results)
- [Analytics and BI Layer](#analytics-and-bi-layer)
- [Streamlit Web App](#streamlit-web-app)
- [Author](#author)

---

## Project Overview

This project implements an **end-to-end machine learning system with deployment and analytics integration** to predict whether industrial machines are at risk of failure.
It includes a complete end-to-end workflow — from **data cleaning and feature engineering** to **model comparison** and **deployment**.

The model helps manufacturing teams **identify warning signals early** and take corrective action before failures occur.  
The final solution is deployed via an interactive **Streamlit dashboard** for real-time predictions.

---

## Business Problem

Unplanned machine breakdowns are one of the biggest challenges in manufacturing, causing:
- Production downtime  
- Costly repairs and replacements  
- Missed delivery deadlines  
- Potential worker safety hazards  

Predictive maintenance helps organizations:
- Detect and prevent equipment failures  
- Optimize maintenance schedules  
- Reduce operational costs  
- Improve **Overall Equipment Effectiveness (OEE)**  

By predicting failures early, factories can operate **smarter, safer, and more efficiently**.

---

## Why This Project Matters

Traditional maintenance is reactive — machines are repaired after breakdown.  
This project moves to a **proactive and predictive model**, allowing businesses to:
- Reduce unplanned downtime by **30–40%**  
- Lower maintenance costs by **up to 25%**  
- Increase asset lifespan and reliability  

Predictive analytics transforms maintenance into a **data-driven decision process**, saving both time and money while improving safety.

---

## Dataset

**Source:** UCI Machine Learning Repository – AI4I Predictive Maintenance Dataset  
**Size:** ~10,000 records  
**Features:**  
- Process and air temperature  
- Torque  
- Rotational speed  
- Tool wear  
- Machine type  

**Target:**  
- `Machine failure` (binary classification — 1 if machine failed, else 0)

The dataset represents real-world operational telemetry from industrial sensors.

---

## Data Preprocessing

- Removed non-predictive identifiers (e.g., IDs)  
- Encoded categorical features (`Type`) using OneHotEncoding  
- Standardized continuous features using `StandardScaler`  
- Addressed class imbalance using **SMOTE (Synthetic Minority Oversampling Technique)**  
- Engineered new feature `TempDiff = ProcessTemp - AirTemp` to capture temperature stress impact  

All preprocessing steps were integrated into a reproducible pipeline.

---


## Engineering and MLOps Foundations

This project applies foundational MLOps and analytics engineering practices to ensure reproducibility, interpretability, and downstream usability:
- Reproducible preprocessing pipelines for feature engineering and class balancing.
- Clear separation of concerns:
  - Python for data preparation, feature engineering, and modeling.
  - BI tools for aggregation, visualization, and stakeholder insights.
- Consistent KPI definitions across ML evaluation and BI dashboards.
- Version-controlled code and data artifacts using GitHub.
- Model outputs designed for downstream consumption (dashboards and applications).

This architecture enables future extensions such as automated retraining, monitoring, and CI/CD-based deployment.


-----

## Modeling

Compared several machine learning models and a deep neural network:

| Model | Accuracy | F1 Score |
|--------|-----------|-----------|
| Logistic Regression | 0.82 | 0.82 |
| Random Forest | 0.99 | 0.99 |
| XGBoost | 0.99 | 0.99 |
| KNN | 0.96 | 0.96 |
| SVM | 0.95 | 0.95 |
| **DNN (Keras)** | **0.96** | **0.96** |

**Best Performer:** Tree-based models (**Random Forest** and **XGBoost**) achieved near-perfect performance with high recall — crucial for avoiding missed failures.

---

## Interpretability

Understanding why a model predicts failure is as important as accuracy.

Feature importance analysis (from Random Forest and SHAP values) revealed:
- **Torque** and **Tool wear** were the strongest predictors of machine failure.  
- High **temperature differentials** and **rotational stress** contributed significantly to breakdown risk.  

These insights enable maintenance teams to focus on the most critical parameters during inspections.

---

## Cost–Benefit Impact

Implementing predictive maintenance can yield measurable returns:
- **30–50% reduction** in unplanned maintenance  
- **Up to 40% decrease** in downtime costs  
- **10–20% increase** in equipment availability  
- Lower spare parts inventory and extended equipment life  

If an average downtime hour costs **€2,000**, saving just 10 hours monthly leads to **€240,000+ annual savings**.

Predictive analytics translates into **both operational resilience and financial gain**.

---

## Results

- Achieved **99% accuracy** with Random Forest and XGBoost  
- Strong recall score ensures minimal false negatives (critical for safety)  
- Cross-validation confirmed consistent generalization  
- DNN achieved competitive results with balanced precision–recall metrics  

The final model was integrated into an intuitive web dashboard for business decision-makers.
Model performance metrics were aligned with business KPIs to ensure consistency between model evaluation and operational dashboards.

---

## Analytics and BI Layer

This project extends beyond model training by translating machine learning outputs into business-consumable analytics.
A BI-optimized dataset was engineered from the original telemetry data, consolidating raw failure flags into interpretable categories and aligning metrics across ML and analytics layers.

**Power BI Dashboard Highlights:**
- Overall Failure Rate KPI — 3.39% machine failure incidence.
- Failure Count by Type — Heat Dissipation Failure identified as the dominant failure mode.
- Failure vs No Failure Distribution — illustrates fleet-wide reliability.
- Failures by Machine Type — highlights machine categories with elevated risk.
- Failure Rate vs Tool Wear — demonstrates increasing failure probability beyond wear thresholds.

**Business Value:**
- Enables maintenance teams to prioritize inspections based on dominant failure modes.
- Supports preventive maintenance scheduling using tool-wear risk signals.
- Bridges the gap between predictive models and operational decision-making.


  ----

## Streamlit Web App

Live Demo: **[Open Predictive Maintenance App](https://predictive-maintenance-dh9qkfibvu76tfgm2asn2y.streamlit.app/)**

Features:
- Input live machine readings to predict failure probability  
- Real-time visualization of sensor thresholds
- Designed for non-technical users to assess failure risk and support maintenance decisions.  


Technologies used: **Streamlit • Scikit-learn • TensorFlow/Keras • XGBoost • Pandas**

---

## Author

**Shabnam Begam Mahammad**  
[LinkedIn](https://www.linkedin.com/in/shabnam-b-mahammad) | [Email](mailto:md.shabnam21@gmail.com) 

“Turning predictive analytics into operational intelligence.”
