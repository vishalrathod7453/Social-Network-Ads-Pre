import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# Page setup
st.set_page_config(page_title="Purchase Predictor AI", page_icon="🎯", layout="centered")

# Attractive Custom CSS
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }
    div.stButton > button {
        background-color: #00f2fe;
        color: black;
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: bold;
        transition: 0.3s;
        border: none;
    }
    div.stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# Robust Model Loading
@st.cache_resource
def load_model():
    model_path = 'Model3.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    else:
        return None

model = load_model()

st.title("🎯 Social Network Ad Predictor")
st.write("Determine if a customer is likely to purchase based on profile data.")

# Input Section
age = st.slider("Customer Age", 18, 100, 30)
salary = st.number_input("Estimated Annual Salary ($)", value=50000, step=1000)

if st.button("Run AI Prediction"):
    if model is not None:
        # Your model expects 2 features: Age and Salary 
        features = np.array([[age, salary]])
        
        with st.spinner('Analyzing...'):
            prediction = model.predict(features)
            # Probability for the conversion flair
            prob = model.predict_proba(features)[0][1]
        
        st.divider()
        if prediction[0] == 1:
            st.balloons() # Success animation
            st.success(f"✅ Likely to Purchase! ({prob:.1%} probability)")
        else:
            st.snow() # Subtle animation
            st.warning(f"❌ Unlikely to Purchase ({prob:.1%} probability)")
    else:
        st.error("🚨 Model3.pkl not found. Please ensure it is in your GitHub root folder.")
