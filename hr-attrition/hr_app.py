import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("💼 HR Attrition Predictor")
st.subheader("Will this employee leave the company? Fill in the details to find out!")

# User Inputs
Age = st.slider("Age", 18, 60, 30)
JobSatisfaction = st.selectbox("Job Satisfaction (1 = Low, 4 = High)", [1, 2, 3, 4])
MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000, step=100)
DistanceFromHome = st.slider("Distance From Home (in kms)", 1, 30, 5)
YearsAtCompany = st.slider("Years at Company", 0, 40, 5)
OverTime = st.selectbox("OverTime", ['Yes', 'No'])

# Preprocess input
OverTime = 1 if OverTime == 'Yes' else 0

input_data = pd.DataFrame({
    'Age': [Age],
    'JobSatisfaction': [JobSatisfaction],
    'MonthlyIncome': [MonthlyIncome],
    'DistanceFromHome': [DistanceFromHome],
    'YearsAtCompany': [YearsAtCompany],
    'OverTime': [OverTime]
})

# Model training (dummy - replace with joblib if needed)
def load_model():
    # Load dataset (must match column names)
    data = pd.read_csv("data/HR-Employee-Attrition.csv")
    
    data['OverTime'] = data['OverTime'].map({'Yes': 1, 'No': 0})
    
    features = ['Age', 'JobSatisfaction', 'MonthlyIncome', 'DistanceFromHome', 'YearsAtCompany', 'OverTime']
    X = data[features]
    y = data['Attrition'].map({'Yes': 1, 'No': 0})

    model = RandomForestClassifier()
    model.fit(X, y)
    return model

model = load_model()

# Prediction
if st.button("Predict Attrition"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ The employee is likely to leave.")
    else:
        st.success("✅ The employee is likely to stay.")

