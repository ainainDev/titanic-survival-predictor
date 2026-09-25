import streamlit as st
import pandas as pd
import joblib

# ------------------------------------------------------------------
# Load the trained pipeline (must be saved beforehand as full_pipeline.pkl
# — see the "Save the Model" cell added to your notebook)
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load('full_pipeline.pkl')

pipeline = load_model()

# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------
st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")
st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details below to predict survival using the trained Logistic Regression pipeline.")

st.divider()

# ------------------------------------------------------------------
# Input widgets — one per raw feature the pipeline expects
# (pclass, sex, age, sibsp, parch, fare, embarked)
# ------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox(
        "Passenger Class (Pclass)",
        options=[1, 2, 3],
        index=2,
        help="1 = First Class, 2 = Second Class, 3 = Third Class"
    )
    sex = st.selectbox("Sex", options=["male", "female"])
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
    sibsp = st.number_input(
        "Siblings/Spouses Aboard (SibSp)",
        min_value=0, max_value=10, value=0, step=1
    )

with col2:
    parch = st.number_input(
        "Parents/Children Aboard (Parch)",
        min_value=0, max_value=10, value=0, step=1
    )
    fare = st.number_input("Fare Paid", min_value=0.0, max_value=600.0, value=32.0, step=1.0)
    embarked = st.selectbox(
        "Port of Embarkation",
        options=["S", "C", "Q"],
        help="S = Southampton, C = Cherbourg, Q = Queenstown"
    )

st.divider()

# ------------------------------------------------------------------
# Predict
# ------------------------------------------------------------------
if st.button("Predict Survival", type="primary", use_container_width=True):

    # Build a single-row DataFrame with EXACTLY the column names/order
    # the pipeline was trained on
    input_df = pd.DataFrame([{
        'pclass': pclass,
        'sex': sex,
        'age': age,
        'sibsp': sibsp,
        'parch': parch,
        'fare': fare,
        'embarked': embarked
    }])

    prediction = pipeline.predict(input_df)[0]
    proba = pipeline.predict_proba(input_df)[0]

    st.subheader("Result")

    if prediction == 1:
        st.success(f"✅ **Survived** — Confidence: {proba[1]*100:.1f}%")
    else:
        st.error(f"❌ **Did Not Survive** — Confidence: {proba[0]*100:.1f}%")

    with st.expander("See prediction probabilities"):
        st.write(f"Probability of **Not Surviving**: {proba[0]*100:.2f}%")
        st.write(f"Probability of **Surviving**: {proba[1]*100:.2f}%")
        st.dataframe(input_df)

st.divider()
st.caption("Model: Logistic Regression | Preprocessing: Imputation → One-Hot Encoding → Scaling → Yeo-Johnson Power Transform")

# ------------------------------------------------------------------
# Copyright
# ------------------------------------------------------------------
st.caption("© 2026 Md Ainain Ahmed. All rights reserved.")
 
# © 2026 ainainDev. All rights reserved.
# This project is licensed under the MIT License — see LICENSE file for details.
