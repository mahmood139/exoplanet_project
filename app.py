import streamlit as st
import joblib
import numpy as np

# ======== 1. تحميل النموذج ========
model = joblib.load("exoplanet_model.joblib")

# ======== 2. دالة التنبؤ ========
def predict_planet(radius, mass, period, luminosity, temp, stellar_mass):
    input_data = np.array([[radius, mass, period, luminosity, temp, stellar_mass]])
    prediction = model.predict(input_data)[0]
    confidence = model.predict_proba(input_data)[0].max() * 100
    return ("Habitable" if prediction==1 else "Not Habitable"), round(confidence,2)

# ======== 3. واجهة المستخدم ========
st.title("🌍 Exoplanet Habitability Predictor")

radius = st.slider("Planet Radius (Earth units)", 0.5, 2.5, 1.0, 0.05)
mass = st.slider("Planet Mass (Earth masses)", 0.1, 5.0, 1.0, 0.1)
period = st.slider("Orbital Period (days)", 50, 500, 365, 5)
luminosity = st.slider("Stellar Luminosity (Solar units)", 0.1, 2.0, 1.0, 0.05)
temp = st.slider("Planet Temperature (K)", 200, 350, 288, 5)
stellar_mass = st.slider("Stellar Mass (Solar masses)", 0.5, 1.5, 1.0, 0.05)

if st.button("🔮 Predict Habitability"):
    pred, conf = predict_planet(radius, mass, period, luminosity, temp, stellar_mass)
    st.success(f"Prediction: **{pred}** with confidence **{conf}%**")