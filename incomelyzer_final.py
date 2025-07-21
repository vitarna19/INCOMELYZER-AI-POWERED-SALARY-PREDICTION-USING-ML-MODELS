# Incomelyzer: Salary Classifier & Analyzer

import pandas as pd
import numpy as np
from sklearn.calibration import LabelEncoder
import streamlit as st
import joblib

# ==== cleaning ====
def clean_data(df):
    # Drop unwanted columns
    df = df.drop(columns=['education', 'income'], errors='ignore')
    
    # Replace '?' with 'Not-Listed'
    df['occupation'] = df['occupation'].replace({'?': 'Not-Listed'})
    df['workclass'] = df['workclass'].replace({'?': 'Not-Listed'})
    
    # Remove invalid categories
    df = df[df['workclass'].isin([...])]  # same list from notebook
    df = df[df['educational-num'].between(5, 16)]
    df = df[df['age'].between(17, 75)]
    df = df[df['hours-per-week'].between(1, 80)]
    
    # Encoding
    label_cols = [...]
    le = LabelEncoder()
    for col in label_cols:
        df[col] = le.fit_transform(df[col].astype(str))
    
    return df

# ===== Load Model and Scaler =====
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Incomelyzer 💼", layout="centered")
st.title("💼 Incomelyzer: Salary Prediction Platform")
st.markdown("""
Predict whether an employee earns >50K or ≤50K using the best ML model trained on the Incomelyzer pipeline.
""")

# ===== Sidebar Inputs =====
st.sidebar.header("📋 Enter Employee Details")

age = st.sidebar.slider("Age", 17, 75, 30)
fnlwgt = st.sidebar.number_input("fnlwgt (final weight)", 10000, 1000000, 200000)
education_num = st.sidebar.slider("Education Number", 5, 16, 9)
marital_status = st.sidebar.selectbox("Marital Status", [0,1,2,3,4,5])
workclass = st.sidebar.selectbox("Workclass", [0,1,2,3,4,5,6,7])
occupation = st.sidebar.selectbox("Occupation", list(range(13)))
relationship = st.sidebar.selectbox("Relationship", list(range(6)))
race = st.sidebar.selectbox("Race", list(range(5)))
gender = st.sidebar.selectbox("Gender", [0,1])
capital_gain = st.sidebar.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.sidebar.number_input("Capital Loss", 0, 10000, 0)
hours_per_week = st.sidebar.slider("Hours Per Week", 1, 80, 40)
native_country = st.sidebar.selectbox("Native Country", list(range(42)))

input_dict = {
    'age': age,
    'workclass': workclass,
    'fnlwgt': fnlwgt,
    'educational-num': education_num,
    'marital-status': marital_status,
    'occupation': occupation,
    'relationship': relationship,
    'race': race,
    'gender': gender,
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'native-country': native_country
}

input_df = pd.DataFrame([input_dict])

st.markdown("---")
st.markdown("### 👀 Input Preview")
st.write(input_df)

if st.button("🔮 Predict Income"):
    input_scaled = scaler.transform(input_df)
    pred = model.predict(input_scaled)[0]
    label = ">50K" if pred == 1 else "<=50K"
    st.success(f"Predicted Income Category: {label}")

# ===== Batch Prediction =====
st.markdown("---")
st.header("📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload CSV for Bulk Predictions", type=["csv"])

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview (Raw):", batch_data.head())

    # ==== Data Cleaning like Training Pipeline ====
    try:
        batch_data['occupation'] = batch_data['occupation'].replace({'?': 'Not-Listed'})
        batch_data['workclass'] = batch_data['workclass'].replace({'?': 'Not-Listed'})
        
        # Drop unused columns like education, income if present
        batch_data = batch_data.drop(columns=['education', 'income'], errors='ignore')

        # Label encoding for categorical columns
        label_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country']
        le = LabelEncoder()
        for col in label_cols:
            batch_data[col] = le.fit_transform(batch_data[col].astype(str))
        
        # Scaling batch data
        batch_scaled = scaler.transform(batch_data)

        # Predictions
        batch_preds = model.predict(batch_scaled)
        batch_data['Predicted Income'] = [">50K" if p == 1 else "<=50K" for p in batch_preds]

        st.write("✅ Batch Predictions:", batch_data.head())

        csv = batch_data.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Predictions", csv, "predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"❌ Error during batch prediction: {e}")

st.markdown("---")
st.caption("Incomelyzer | Insightful salary analysis + prediction app")

