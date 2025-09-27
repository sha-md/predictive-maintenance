# predictive-maintenance
Machine failure prediction using ML + Deep Learning


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
10. [How This Saves Costs](#how-this-saves-costs)  
11. [Future Improvements](#future-improvements)  

---

## 📌 Project Overview
This project applies **machine learning and deep learning** to predict machine failures in a manufacturing setup.  
I built a pipeline to compare classical models (Logistic Regression, Random Forest, XGBoost, SVM, KNN) with a **Deep Neural Network (Keras)** baseline.

---

## 💼 Business Problem
Unexpected failures → downtime, costly repairs, and safety issues.  
Predicting failures allows manufacturers to plan **preventive maintenance** instead of reacting to breakdowns.

---

## 📊 Dataset
- **Source**: AI4I Predictive Maintenance Dataset (UCI ML Repository)  
- ~10,000 records with features like torque, temperature, rotational speed, and tool wear  
- Target: `Machine failure (0/1)`

---

## 🔍 Data Preprocessing
- Dropped irrelevant IDs  
- Encoded categorical feature (`Type`)  
- Standardized continuous features  
- Balanced classes with **SMOTE** (fixed 1:28 imbalance)  

---

## 🤖 Modeling
Models implemented:
- Logistic Regression  
- Random Forest  
- XGBoost  
- KNN  
- SVM  
- **Deep Neural Network (Keras)**

| Model               | Accuracy | F1 Score |
|----------------------|----------|----------|
| Logistic Regression  | 0.82     | 0.82     |
| Random Forest        | 0.99     | 0.99     |
| XGBoost              | 0.99     | 0.99     |
| KNN                  | 0.96     | 0.96     |
| SVM                  | 0.95     | 0.95     |
| **DNN (Keras)**      | **0.96** | **0.96** |

---

## 📈 Results
- Best: **Tree-based models** (RF, XGBoost)  
- DNN competitive with strong performance  
- Confusion Matrix shows very few false negatives (key for safety)  

---

## 🔎 Interpretability
- Feature importances from RF & XGBoost  
- DNN visualized with training loss/accuracy curves  

---

## 🌐 Streamlit App
Interactive demo:  

```bash
streamlit run app.py
