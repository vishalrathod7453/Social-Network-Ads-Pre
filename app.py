import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page setup for an attractive UI
st.set_page_config(page_title="Ad Predictor AI", page_icon="🎯", layout="centered")

# Custom CSS for Glassmorphism effect and animations
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
        transition: 0.3s ease;
        border: none;
    }
    div.stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.4);
        background-color: #4facfe;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model correctly
@st.cache_resource
def load_model():
    try:
        # NOTE: Ensure the file is named 'Model3.pkl' exactly in your GitHub
        with open('Model3.pkl', 'wb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("🚨 'Model3.pkl' not found! Please check your GitHub file name.")
        return None

model = load_model()

st.title("🎯 Social Network Ad Predictor")
st.write("Predict if a customer will purchase an item based on their profile.")

# Input Section
with st.container():
    age = st.slider("Customer Age", 18, 100, 30)
    salary = st.number_input("Estimated Annual Salary ($)", value=50000, step=1000)

if st.button("Analyze Purchase Intent"):
    if model:
        # Your model expects exactly 2 features (Age and Salary)
        features = np.array([[age, salary]])
        
        with st.spinner('AI is processing...'):
            prediction = model.predict(features)
            # Use predict_proba for visual confidence scores
            prob = model.predict_proba(features)[0][1]
        
        st.divider()
        if prediction[0] == 1:
            st.balloons() # Success animation
            st.success(f"✅ Prediction: Likely to Purchase! ({prob:.1%} confidence)")
        else:
            st.snow() # Subtle negative animation
            st.warning(f"❌ Prediction: Unlikely to Purchase ({prob:.1%} confidence)")
    else:
        st.error("Model not loaded. Ensure 'Model3.pkl' is in your repository root.")
