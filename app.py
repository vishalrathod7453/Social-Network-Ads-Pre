import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Set page configuration for an attractive layout
st.set_page_config(page_title="Ad Purchase Predictor", layout="centered")

# Custom CSS for animation and styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff2b2b;
        transform: scale(1.05);
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    with open('Model3.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Header Section
st.title("🎯 Social Network Ad Predictor")
st.markdown("Predict whether a user will purchase a product based on their profile.")

# Animated Sidebar or Main Input Container
with st.container():
    st.subheader("User Information")
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 100, 30)
        
    with col2:
        salary = st.number_input("Estimated Salary ($)", min_value=0, value=50000, step=1000)

# Preprocessing inputs
# Mapping Gender to numeric if your model expects it (0 for Female, 1 for Male)
gender_numeric = 1 if gender == "Male" else 0

# Prediction Button
if st.button("Predict Purchase Intent"):
    # Prepare the input for the model
    features = np.array([[gender_numeric, age, salary]])
    
    # Show a loading spinner for a professional feel
    with st.spinner('Analyzing patterns...'):
        prediction = model.predict(features)
    
    # Display Result with Animation
    if prediction[0] == 1:
        st.balloons() # Animated celebration
        st.success("✅ Prediction: User is likely to **PURCHASE**!")
    else:
        st.warning("❌ Prediction: User is unlikely to purchase.")
