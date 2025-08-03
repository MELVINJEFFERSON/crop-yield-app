import streamlit as st
import joblib
import numpy as np

# Load trained model and encoders
model = joblib.load("yield_model.pkl")
crop_labels = joblib.load("crop_labels.pkl")
country_labels = joblib.load("country_labels.pkl")

# Reverse maps for selection
crop_map = {v: k for k, v in crop_labels.items()}
country_map = {v: k for k, v in country_labels.items()}

# UI
st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾")
st.title("🌾 Real-Time Crop Yield Predictor")
st.markdown("Enter climate conditions to predict expected crop yield.")

# User Inputs
country = st.selectbox("Select Country", list(country_map.keys()))
crop = st.selectbox("Select Crop", list(crop_map.keys()))
rainfall = st.slider("Rainfall (mm)", 300, 1500, 900)
temperature = st.slider("Temperature (°C)", 10.0, 40.0, 26.0)

# Prediction
if st.button("Predict Yield"):
    input_data = np.array([[rainfall, temperature, crop_map[crop], country_map[country]]])
    predicted_yield = model.predict(input_data)[0]
    st.success(f"🌱 Predicted Crop Yield: **{predicted_yield:.2f} hg/ha**")
