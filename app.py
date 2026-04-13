import streamlit as st
import pickle
import numpy as np

# Page configuration for a professional look
st.set_page_config(page_title="Purchase Predictor AI", page_icon="🛍️", layout="centered")

# Custom CSS for an attractive, animated UI
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 15px 32px;
        font-size: 18px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the Logistic Regression model 
@st.cache_resource
def load_model():
    try:
        with open('Model3.pkl', 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Header
st.title("🎯 Ad Conversion Predictor")
st.write("Using AI to determine if a customer is likely to purchase after seeing an ad.")
st.divider()

# Input Section with Columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Customer Age", min_value=18, max_value=100, value=30)

with col2:
    salary = st.number_input("Estimated Annual Salary ($)", min_value=0, value=50000, step=500)

# Prediction Logic
if st.button("Analyze Purchase Intent"):
    if model:
        # Based on your model metadata, it expects 2 features (Age, Salary) 
        features = np.array([[age, salary]])
        
        # Add a "thinking" animation
        with st.spinner('Running AI Analysis...'):
            prediction = model.predict(features)
            # Use predict_proba for a more "animated" feel with progress bars
            probability = model.predict_proba(features)[0][1] 
        
        st.subheader("Results")
        
        if prediction[0] == 1:
            st.balloons() # Celebration animation
            st.success(f"✅ High Purchase Intent! ({probability*100:.1f}%)")
            st.write("This customer is very likely to click and buy.")
        else:
            st.snow() # Subtle "cold" animation
            st.warning(f"❌ Low Purchase Intent ({probability*100:.1f}%)")
            st.write("This customer is unlikely to purchase at this time.")
            
        # Add a progress bar for visual flair
        st.progress(probability)
    else:
        st.error("Model file not found. Please ensure 'Model3.pkl' is in the same folder.")

# Footer
st.caption("Powered by Scikit-Learn 1.6.1 and Streamlit")
