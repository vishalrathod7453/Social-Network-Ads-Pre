import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page config for a modern look
st.set_page_config(page_title="Purchase AI", page_icon="🛍️", layout="centered")

# Custom CSS for a professional, animated interface
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #6a11cb 0%, #2575fc 100%);
        color: white;
    }
    .stButton>button {
        width: 100%;
        border-radius: 25px;
        height: 3em;
        background-color: #00d2ff;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        background-color: #3a7bd5;
    }
    </style>
    """, unsafe_allow_html=True)

# Correctly loading the Model3.pkl file
@st.cache_resource
def load_model():
    try:
        with open('Model3.pkl', 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        st.error("Model3.pkl not found! Ensure the file is in your GitHub folder.")
        return None

model = load_model()

st.title("🎯 Customer Purchase Predictor")
st.write("Input user details to predict ad conversion probability.")

# Input card
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Customer Age", 18, 100, 25)
    with col2:
        salary = st.number_input("Annual Salary ($)", 10000, 200000, 50000)

if st.button("Predict Intent"):
    if model:
        # Prepare input as a DataFrame to keep feature names consistent
        input_data = pd.DataFrame([[age, salary]], columns=['Age', 'EstimatedSalary'])
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Check probability for the 'Purchase' class
        prob = model.predict_proba(input_data)[0][1]
        
        st.divider()
        
        if prediction[0] == 1:
            st.balloons() # Success animation
            st.success(f"✅ Prediction: This user is LIKELY to purchase! (Confidence: {prob:.2%})")
        else:
            st.snow() # "Cold" lead animation
            st.warning(f"❌ Prediction: This user is UNLIKELY to purchase. (Confidence: {1-prob:.2%})")
