import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load the trained model and scaler
model = load_model("bike_sharing.h5", compile=False)
model.compile(loss="mse", optimizer="adam", metrics=["mae"])

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("🚴‍♂️ Bike Rental Predictor (Deep Learning)")

# Input form — all 13 features used in training
season = st.selectbox("Season (1=spring ... 4=winter)", [1, 2, 3, 4])
yr = st.selectbox("Year (0=2011, 1=2012)", [0, 1])
mnth = st.slider("Month", 1, 12)
hr = st.slider("Hour", 0, 23)
holiday = st.selectbox("Is it a holiday?", [0, 1])
weekday = st.slider("Weekday (0=Sun ... 6=Sat)", 0, 6)
workingday = st.selectbox("Is it a working day?", [0, 1])
weathersit = st.selectbox("Weather Situation (1=Clear, 2=Mist, 3=Light Snow, 4=Heavy Rain)", [1, 2, 3, 4])
temp = st.slider("Temperature (0–1)", 0.0, 1.0, step=0.01)
atemp = st.slider("Feels-like Temp (0–1)", 0.0, 1.0, step=0.01)
hum = st.slider("Humidity (0–1)", 0.0, 1.0, step=0.01)
windspeed = st.slider("Windspeed (0–1)", 0.0, 1.0, step=0.01)

# Add 13th feature: hour (same as hr)
hour = hr

# Build input array (13 features total)
input_data = np.array([[season, yr, mnth, hr, holiday, weekday,
                        workingday, weathersit, temp, atemp, hum, windspeed, hour]])

# Validate and scale input
if input_data.shape[1] != scaler.n_features_in_:
    st.error(f"❌ Feature count mismatch! Expected {scaler.n_features_in_} features, got {input_data.shape[1]}")
else:
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)
    st.success(f"🚲 Estimated Bike Rentals: **{int(prediction[0][0])}**")
