import streamlit as st
import numpy as np
# import joblib
import os
from streamlit_lottie import st_lottie
import requests

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Social Network Ads Predictor", page_icon="🎯", layout="centered")

# ------------------ LOTTIE ANIMATION ------------------
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_animation = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json")

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    try:
        file_path = "Model3.pkl"

        if not os.path.exists(file_path):
            st.error(f"🚨 File '{file_path}' not found in repository!")
            return None

        model = joblib.load(file_path)
        return model

    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# ------------------ UI DESIGN ------------------
st.markdown(
    """
    <h1 style='text-align: center; color: #4CAF50;'>🎯 Social Network Ad Predictor</h1>
    <p style='text-align: center;'>Predict whether a customer will purchase a product</p>
    """,
    unsafe_allow_html=True
)

# Animation
if lottie_animation:
    st_lottie(lottie_animation, height=250)

# ------------------ INPUT SECTION ------------------
st.subheader("👤 Customer Details")

age = st.slider("📅 Age", 18, 60, 25)
salary = st.slider("💰 Estimated Salary", 10000, 200000, 50000)

# ------------------ PREDICTION ------------------
if st.button("🚀 Predict", use_container_width=True):

    if model is None:
        st.error("❌ Model not loaded. Check file & dependencies.")
    else:
        input_data = np.array([[age, salary]])

        try:
            prediction = model.predict(input_data)

            if prediction[0] == 1:
                st.success("✅ Customer is likely to PURCHASE")
                st.balloons()
            else:
                st.warning("❌ Customer is NOT likely to purchase")

        except Exception as e:
            st.error(f"Prediction Error: {e}")

# ------------------ FOOTER ------------------
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit")
