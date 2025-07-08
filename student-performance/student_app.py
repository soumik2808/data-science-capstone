import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("student_performance_model.pkl")

st.title("📘 Student Performance Predictor")
st.subheader("Will this student likely pass or fail?")

# Input fields
gender = st.selectbox("Gender", ["female", "male"])
race = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parent_education = st.selectbox("Parental Level of Education", [
    "some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"
])
lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
test_prep = st.selectbox("Test Preparation Course", ["none", "completed"])

# Encode categorical inputs (same encoding as notebook)
mapping = {
    "gender": {"female": 0, "male": 1},
    "race": {"group A": 0, "group B": 1, "group C": 2, "group D": 3, "group E": 4},
    "parental level of education": {
        "some high school": 5, "high school": 3, "some college": 4,
        "associate's degree": 0, "bachelor's degree": 1, "master's degree": 2
    },
    "lunch": {"standard": 1, "free/reduced": 0},
    "test preparation course": {"none": 0, "completed": 1}
}

# Prepare input
input_df = pd.DataFrame([{
    "gender": mapping["gender"][gender],
    "race/ethnicity": mapping["race"][race],
    "parental level of education": mapping["parental level of education"][parent_education],
    "lunch": mapping["lunch"][lunch],
    "test preparation course": mapping["test preparation course"][test_prep]
}])

# Prediction
if st.button("Predict Performance"):
    result = model.predict(input_df)[0]
    if result == 1:
        st.success("✅ Likely to PASS")
    else:
        st.error("❌ At Risk of Failing")
