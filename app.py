import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page setup
st.set_page_config(page_title="Purchase Predictor AI", page_icon="🤖", layout="centered")

# Custom CSS for an attractive, animated UI
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
    }
    div.stButton > button {
        background-color: #00f2fe;
        color: #000;
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: bold;
        transition: 0.3s;
    }
    div.stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# Correctly loading the Model3.pkl file
@st.cache_resource
def load_model():
    try:
        # Note: Ensure the file is named 'Model3.pkl' exactly in GitHub
        with open('Model3.pkl', 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Error: {e}")
        return None

model = load_model()

st.title("🎯 Social Network Ad Predictor")
st.write("Determine if a customer is likely to purchase based on profile data.")

# Input Section
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
    with col2:
        salary = st.number_input("Estimated Annual Salary ($)", min_value=0, value=50000)
    
    submit = st.form_submit_button("Run AI Analysis")

if submit:
    if model:
        # Your model was trained on 2 features (Age and Salary)
        features = np.array([[age, salary]])
        
        with st.spinner('Calculating probability...'):
            prediction = model.predict(features)
            # Fetch probability for conversion flair
            prob = model.predict_proba(features)[0][1]
        
        st.divider()
        if prediction[0] == 1:
            st.balloons() # Success animation
            st.success(f"✅ Likely to Purchase! ({prob:.1%} probability)")
        else:
            st.snow() # "Cold lead" animation
            st.warning(f"❌ Unlikely to Purchase ({prob:.1%} probability)")
    else:
        st.error("Model not loaded. Ensure 'Model3.pkl' is in your repository.")
