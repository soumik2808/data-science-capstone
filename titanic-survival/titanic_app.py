import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Titanic Survival Predictor", layout="centered")

st.title("🚢 Titanic Survival Predictor")
st.subheader("Will you survive the Titanic disaster? Fill in your details to find out!")

# Load model
@st.cache_resource
def load_model():
    try:
        return joblib.load("titanic_model.pkl")
    except:
        return None

model = load_model()

# Input form
with st.form("prediction_form"):
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    sex = st.selectbox("Sex", ["male", "female"])
    age = st.slider("Age", 1, 80, 30)
    sibsp = st.slider("Siblings/Spouses Aboard", 0, 5, 0)
    parch = st.slider("Parents/Children Aboard", 0, 5, 0)
    fare = st.number_input("Fare Paid", min_value=0.0, value=32.0)
    embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

    submitted = st.form_submit_button("Predict")

# Prediction
if submitted and model:
    input_df = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": 1 if sex == "male" else 0,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": {"C": 0, "Q": 1, "S": 2}[embarked]
    }])

    prediction = model.predict(input_df)[0]
    st.success("🎉 Survived!" if prediction == 1 else "💀 Did not survive")
elif submitted:
    st.error("⚠️ Model file missing. Please train and save `titanic_model.pkl`.")

