import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Page Config
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗")

st.title("🚗 Car Price Prediction Dashboard")
st.write("Predict the estimated price of a car")

# Load Dataset
df = pd.read_csv("carprice.csv")

# Cleaning
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)

# Convert numeric columns
df["horsepower"] = pd.to_numeric(df["horsepower"])
df["peak-rpm"] = pd.to_numeric(df["peak-rpm"])
df["price"] = pd.to_numeric(df["price"])

# Features & Target
X = df.drop("price", axis=1)
y = df["price"]

# Encoding
X = pd.get_dummies(X, drop_first=True)

# Train Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(X, y)

# Inputs
st.subheader("Enter Car Details")

horsepower = st.number_input("Horsepower", value=100)

peak_rpm = st.number_input("Peak RPM", value=5000)

city_mpg = st.number_input("City MPG", value=25)

highway_mpg = st.number_input("Highway MPG", value=30)

fuel_type = st.selectbox(
    "Fuel Type",
    ["gas", "diesel"]
)

engine_location = st.selectbox(
    "Engine Location",
    ["front", "rear"]
)

engine_type = st.selectbox(
    "Engine Type",
    ["dohc", "dohcv", "l", "ohc", "ohcf", "ohcv", "rotor"]
)

# Prediction
if st.button("Predict Price"):

    input_df = pd.DataFrame(
        0,
        index=[0],
        columns=X.columns
    )

    input_df["horsepower"] = horsepower
    input_df["peak-rpm"] = peak_rpm
    input_df["city-mpg"] = city_mpg
    input_df["highway-mpg"] = highway_mpg

    if "fuel-type_gas" in input_df.columns and fuel_type == "gas":
        input_df["fuel-type_gas"] = 1

    if "engine-location_rear" in input_df.columns and engine_location == "rear":
        input_df["engine-location_rear"] = 1

    engine_col = f"engine-type_{engine_type}"

    if engine_col in input_df.columns:
        input_df[engine_col] = 1

    prediction = model.predict(input_df)

    st.success(
        f"Estimated Car Price: ₹ {prediction[0]:,.2f}"
    )

st.markdown("---")
st.caption("Developed using Streamlit and Machine Learning")
