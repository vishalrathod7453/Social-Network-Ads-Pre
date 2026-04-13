import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page Configuration
st.set_page_config(page_title="Purchase AI Predictor", page_icon="🛍️", layout="centered")

# Attractive CSS for an animated, modern UI
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        color: white;
    }
    .stButton>button {
        width: 100%;
        border-radius: 15px;
        background: #00d2ff; 
        background: -webkit-linear-gradient(to right, #3a7bd5, #00d2ff);
        background: linear-gradient(to right, #3a7bd5, #00d2ff);
        color: white;
        border: none;
        padding: 12px;
        font-weight: bold;
        transition: 0.4s;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 5px 15px rgba(0,210,255,0.4);
    }
    </style>
    """, unsafe_allow_html=True)

# Loading the specific file you provided
@st.cache_resource
def load_model():
    try:
        # File name must match exactly what is in your GitHub
        with open('Model3 (1).pkl', 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("🚨 File 'Model3 (1).pkl' not found in the repository!")
        return None

model = load_model()

st.title("🎯 Social Network Ad Predictor")
st.write("Determine if a customer is likely to purchase based on their profile.")

# User Inputs
with st.container():
    age = st.slider("Customer Age", 18, 100, 30)
    salary = st.number_input("Estimated Annual Salary ($)", min_value=0, value=50000, step=1000)

if st.button("Run Prediction"):
    if model:
        # Your model expects 2 features (Age and Salary) 
        features = np.array([[age, salary]])
        
        with st.spinner('AI is analyzing behavioral patterns...'):
            prediction = model.predict(features)
            # Fetching probability for conversion visualization
            probability = model.predict_proba(features)[0][1]

        st.divider()
        if prediction[0] == 1:
            st.balloons() # Success animation
            st.success(f"✅ Result: This user is likely to BUY! (Confidence: {probability:.1%})")
        else:
            st.snow() # "Cold lead" animation
            st.warning(f"❌ Result: This user is unlikely to purchase. (Confidence: {1-probability:.1%})")
    else:
        st.error("Model failed to load. Please check your GitHub file path.")
