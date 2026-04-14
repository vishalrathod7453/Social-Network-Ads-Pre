import streamlit as st
import pickle
import numpy as np
import requests

# Page configuration
st.set_page_config(page_title="Ad Purchase Predictor", page_icon="🛍️", layout="centered")

# Function to load Lottie animations
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Animated assets
lottie_shopping = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_9w9of95a.json")

# Load your KNN Model
with open('Model3.pkl', 'rb') as file:
    model = pickle.load(file)

# UI Header
st.title("🎯 Customer Purchase Predictor")
st.write("Using Machine Learning to predict if a user will click and buy an ad.")
st_lottie(lottie_shopping, height=200, key="main_anim")

st.markdown("---")

# Input Form
with st.container():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        # Map gender to numerical (assuming Male=1, Female=0 based on common data encoding)
        gender_encoded = 1 if gender == "Male" else 0
        
    with col2:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        
    with col3:
        salary = st.number_input("Estimated Salary ($)", min_value=10000, max_value=200000, value=50000, step=500)

# Prediction Logic
if st.button("Analyze Profile"):
    # Features must match the order in Model3: Gender, Age, EstimatedSalary
    input_data = np.array([[gender_encoded, age, salary]])
    
    prediction = model.predict(input_data)
    
    st.markdown("---")
    if prediction[0] == 1:
        st.success("### ✅ Prediction: Likely to Purchase")
        st.balloons()
    else:
        st.error("### ❌ Prediction: Unlikely to Purchase")

# Footer/Sidebar info
st.sidebar.info("Model: K-Nearest Neighbors (KNN)")
st.sidebar.write("This app uses a pre-trained model to classify users based on social network data.")
