import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import PolynomialFeatures

# -----------------------------
# Load trained model
# -----------------------------

modelpath = Path(__file__).parent / "AC_Bill.pkl"
model = joblib.load(modelpath)
poly = PolynomialFeatures(degree=2)

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(
    page_title="Electricity Bill Prediction",
    page_icon="⚡"
)

st.title("Electricity Bill Prediction")

st.write("Enter the AC units to predict the electricity bill.")

# -----------------------------
# User Input
# -----------------------------
ac_units = st.number_input(
    "Enter AC Units",
    min_value=0,
    max_value=150,
    value=50,
    step=1
)

if st.button("Predict Electricity Bill"):

    # Convert input into DataFrame
    input_data = pd.DataFrame({"AC_Units": [ac_units]})

    # Transform features using PolynomialFeatures (degree 2)
    input_poly = poly.fit_transform(input_data)

    # Make prediction
    prediction = model.predict(input_poly)[0]

    # Display result
    st.success(
        f"Predicted Electricity Bill: ₹{prediction:.2f}"
    )
