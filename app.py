import streamlit as st
import pickle
import numpy as np
import pandas as pd
from streamlit_lottie import st_lottie
import requests

# 1. Page Configuration
st.set_page_config(page_title="Ad Predictor AI", page_icon="🎯", layout="centered")

# 2. Load Assets (Animation)
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_coding = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_qpwb7t7n.json")

# 3. Load the Model
with open('Model3.pkl', 'rb') as file:
    model = pickle.load(file)

# 4. UI Header
st.title("🎯 Social Network Ad Predictor")
st_lottie(lottie_coding, height=200, key="coding")
st.write("Determine if a customer is likely to purchase based on their profile.")

# 5. User Input Form
st.markdown("---")
with st.container():
    st.subheader("Customer Demographics")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Select Age", 18, 100, 30)
    with col2:
        salary = st.number_input("Estimated Annual Salary ($)", min_value=10000, max_value=200000, value=50000, step=500)

# 6. Prediction Logic
if st.button("Predict Likelihood"):
    # Prepare input for the Logistic Regression model
    input_data = np.array([[age, salary]])
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")
    if prediction[0] == 1:
        st.success(f"### ✅ Likely to Purchase!")
        st.balloons()
    else:
        st.warning(f"### ❌ Unlikely to Purchase")
    
    st.info(f"Confidence Score: {probability:.2%}")

# 7. Sidebar Info
st.sidebar.header("About Model")
st.sidebar.write("Algorithm: Logistic Regression")
st.sidebar.write("Regularization: L2 (Ridge)")
