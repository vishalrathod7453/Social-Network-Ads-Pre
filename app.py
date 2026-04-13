import streamlit as st
import pickle
import numpy as np

# Load model safely
@st.cache_resource
def load_model():
    try:
        with open("model.pkl3", "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

st.title("📊 Social Network Ads Prediction")

# Input fields
age = st.slider("Age", 18, 60)
salary = st.number_input("Estimated Salary", 10000, 200000)

if st.button("Predict"):
    if model is not None:
        input_data = np.array([[age, salary]])
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.success("✅ User will purchase")
        else:
            st.warning("❌ User will NOT purchase")
    else:
        st.error("Model not loaded properly")
