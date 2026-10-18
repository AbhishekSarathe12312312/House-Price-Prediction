import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("🏠 House Price Prediction")

df = pd.read_csv("house_data.csv")

X = df[["Area", "Bedrooms", "Age"]]
y = df["Price"]

model = LinearRegression()
model.fit(X, y)

area = st.number_input("Area (sq ft)", 500, 5000, 2000)
bedrooms = st.number_input("Bedrooms", 1, 10, 3)
age = st.number_input("House Age (years)", 0, 100, 5)

if st.button("Predict Price"):
    data = pd.DataFrame(
        [[area, bedrooms, age]],
        columns=["Area", "Bedrooms", "Age"]
    )
    price = model.predict(data)[0]
    st.success(f"Predicted House Price: ₹{price:,.0f}")