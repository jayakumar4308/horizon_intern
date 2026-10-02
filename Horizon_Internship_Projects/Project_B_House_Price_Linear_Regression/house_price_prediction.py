import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

df = pd.read_csv("house_prices.csv")

X = df[["Area_sqft"]]
y = df["Price_INR"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)
print("MAE:", round(mean_absolute_error(y_test, predictions), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, predictions)), 2))
print("R2 Score:", round(r2_score(y_test, predictions), 4))

sample_area = 2000
predicted_price = model.predict([[sample_area]])[0]
print(f"Predicted price for {sample_area} sq.ft: ₹{predicted_price:,.0f}")

plt.scatter(X, y, label="Actual data")
plt.plot(X, model.predict(X), label="Regression line")
plt.xlabel("Area (sq.ft)")
plt.ylabel("House Price (INR)")
plt.title("House Price Prediction using Simple Linear Regression")
plt.legend()
plt.tight_layout()
plt.savefig("regression_line.png")
plt.close()

print("Regression completed. Chart saved as regression_line.png.")
