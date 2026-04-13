import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page styling
st.set_page_config(page_title="Purchase Predictor", page_icon="📈", layout="centered")

# Attractive UI Styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .stButton>button {
        background: #00f2fe;
        color: black;
        border-radius: 20px;
        transition: 0.3s;
        font-weight: bold;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        background: #4facfe;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model correctly
@st.cache_resource
def load_model():
    # Make sure this filename matches your uploaded file exactly
    try:
        with open('Model3.pkl', 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("Model file 'Model3.pkl' not found. Check your GitHub repository!")
        return None

model = load_model()

st.title("🎯 Social Network Ad Predictor")
st.write("Predict if a customer will buy based on Age and Salary.")

# Input Section
with st.container():
    age = st.slider("Age", 18, 100, 30)
    salary = st.number_input("Estimated Annual Salary ($)", value=50000, step=1000)

if st.button("Predict Purchase"):
    if model:
        # Based on your file, the model expects 2 features: Age and Salary 
        features = np.array([[age, salary]])
        
        with st.spinner('AI is analyzing...'):
            prediction = model.predict(features)
            
        st.divider()
        if prediction[0] == 1:
            st.balloons() # Animated celebration
            st.success("✅ Prediction: This user is likely to PURCHASE!")
        else:
            st.snow() # Subtle negative animation
            st.warning("❌ Prediction: This user is unlikely to purchase.")
