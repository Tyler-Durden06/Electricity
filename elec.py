import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
# -----------------------------
# Load trained model
# -----------------------------

modelpath = Path(__file__).parent / "AC_Bill.pkl"
model = joblib.load(modelpath)

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
    max_value=500,
    value=50,
    step=1
)

if st.button("Predict Electricity Bill"):

    # Convert input into DataFrame
    input_data = pd.DataFrame(
        [[ac_units]],
        columns=["AC_Units"]
    )

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display result
    st.success(
        f"Predicted Electricity Bill: ₹{prediction:.2f}"
    )
