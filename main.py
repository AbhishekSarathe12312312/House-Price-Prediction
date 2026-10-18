import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("house_data.csv")

X = df[["Area", "Bedrooms", "Age"]]
y = df["Price"]

model = LinearRegression()
model.fit(X, y)

area = 2000
bedrooms = 3
age = 5
prediction = model.predict(
    pd.DataFrame([[area, bedrooms, age]],
    columns=["Area", "Bedrooms", "Age"])
)

print("House Price Prediction")
print(f"Area: {area} sq ft")
print(f"Bedrooms: {bedrooms}")
print(f"Age: {age} years")
print(f"Predicted Price: ₹{prediction[0]:,.0f}")