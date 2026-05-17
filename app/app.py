import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
src_path = project_root/"src"

sys.path.append(str(project_root))
sys.path.append(str(src_path))

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from src.preprocess import data_cleaning
from src.feauture_engineering import feature_engineering

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

@st.cache_resource
def load_model():
    return joblib.load("models/Customer_Churn_Prediction.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("Data/raw/churn-bigml-80.csv")


model = load_model()
df = load_data()

st.title("Customer Churn Prediction")

st.write(
    """
    Predict whether a telecom customer
    is likely to churn using Machine Learning.
    """
)


st.info(
    """
    Model Used: XGBoost

    ROC-AUC Score: 0.927

    Project Goal:

    Help telecom companies identify customers
    who are at risk of leaving.
    """
)


st.subheader(
    "Risk Levels"
)


st.write(
    """
    HIGH → Probability >70%

    MEDIUM → Probability 40–70%

    LOW → Probability <40%
    """
)

st.write("Predict whether a telecom customer will churn")

st.sidebar.title("Customer Inputs")

st.sidebar.success("Model: XGBoost")

st.sidebar.write("ROC-AUC: 0.927")

features = [col for col in df.columns if col!="Churn"]

input_data = {}
for col in features:
    if df[col].dtype == "object":
        input_data[col] = st.sidebar.selectbox(col, sorted(df[col].unique()))
    elif df[col].dtype == "string":
        input_data[col] = st.sidebar.selectbox(col, sorted(df[col].unique()))
    elif df[col].dtype == "string":
        input_data[col] = st.sidebar.selectbox(col, [True, False])
    else:
        input_data[col] = st.sidebar.number_input(col, value = float(df[col].median()))


if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    st.subheader("Customer Details")
    st.dataframe(input_df)
    input_df = data_cleaning(input_df)
    input_df = feature_engineering(input_df)
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    churn_probability = round(probability[0][1]*100, 2)
    st.subheader("Prediction Result")
    if prediction[0]==1:
        st.error("Customer likely to churn")
    else:
        st.success("Customer likely to stay")
    
    st.metric("Churn Probability", f"{churn_probability}%")
    
    if churn_probability>70:
        st.warning("Risk Level : HIGH")

    elif churn_probability>40:
        st.info("Risk Level : MEDIUM")
    else:
        st.success("Risk Level : LOW")