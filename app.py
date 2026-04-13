import streamlit as st
import pickle
import numpy as np
import pandas as pd
import requests
from streamlit_lottie import st_lottie

# Page configuration
st.set_page_config(page_title="Ad Predictor AI", page_icon="🎯", layout="centered")

# Function to load Lottie animations
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_hello = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_qpwb7qsq.json")

# Custom CSS
st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; }
    div.stButton > button { background-color: #00f2fe; color: black; border-radius: 10px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    try:
        with open('Model3.pkl', 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("Model3.pkl not found! Ensure it is in your GitHub root.")
        return None

model = load_model()

# Header with Animation
st_lottie(lottie_hello, height=200, key="coding")
st.title("🎯 Social Network Ad Predictor")

# Input Section
age = st.slider("Customer Age", 18, 100, 30)
salary = st.number_input("Estimated Annual Salary ($)", value=50000)

if st.button("Analyze Purchase Intent"):
    if model:
        # Model3.pkl expects 2 features: Age and Salary
        features = np.array([[age, salary]])
        prediction = model.predict(features)
        
        st.divider()
        if prediction[0] == 1:
            st.balloons()
            st.success("✅ Result: High probability of purchase!")
        else:
            st.snow()
            st.warning("❌ Result: Low probability of purchase.")
