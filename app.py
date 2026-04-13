import streamlit as st
import pickle
import numpy as np

# Page config
st.set_page_config(page_title="Ad Purchase Predictor", layout="centered")

# Load model
@st.cache_resource
def load_model():
    # Ensure scikit-learn is in requirements.txt
    with open('Model3.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

st.title("🎯 Social Network Ad Predictor")

with st.form("prediction_form"):
    st.subheader("Enter User Details")
    
    # The model expects 4 inputs: User ID, Gender, Age, Salary
    # We can provide a dummy User ID since it usually doesn't affect the KNN prediction logic
    user_id = st.number_input("User ID (any number)", value=15624510)
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 100, 30)
    salary = st.number_input("Estimated Salary ($)", value=50000)
    
    submit = st.form_submit_button("Predict Purchase")

if submit:
    # Convert gender to numeric (0 for Female, 1 for Male)
    gender_numeric = 1 if gender == "Male" else 0
    
    # Input must be in the exact order: [User ID, Gender, Age, Salary]
    features = np.array([[user_id, gender_numeric, age, salary]])
    
    prediction = model.predict(features)
    
    if prediction[0] == 1:
        st.balloons()
        st.success("Result: This user is likely to BUY.")
    else:
        st.warning("Result: This user is unlikely to buy.")
