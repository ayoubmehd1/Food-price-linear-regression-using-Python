import pandas as pd
import numpy as np
import seaborn as sns   
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy import stats

filepath = r"C:\Users\Microsoft\Downloads\Apocolypse Food Prep - Bins Lists Tutorial.xlsx"

df = pd.read_excel(filepath)

df.rename(columns={
    "Product ID":"P_ID",
    "Product Name":"P_Name",
    "Production Cost":"P_Cost",
}, inplace=True)

mean_cost = df["P_Cost"].mean()
print("Mean Production Cost:", mean_cost)
df["Price"] = df["Price"].astype(int)

bin= np.linspace(df["Price"].min(), df["Price"].max(), 4)
group_names = ["Cheap", "Medium", "Expensive"]
df["Price-Bin"] = pd.cut(df["Price"], bin, labels=group_names, include_lowest=True)
print(df.head(10))      


X = df[["P_Cost"]]
y = df["Price"]
model = LinearRegression()
model.fit(X, y)
print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)

np.random.seed(42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data into training and testing sets
X_test = model.predict([[15]])
print("Predicted Price for Production Cost 15:", X_test)   
model.fit(X_train, y_train)  
y_pred = model.predict(X_test)  
print("Predicted price:", y_pred)  

score = model.score(X, y)
print("Model R^2 Score:", score)

mse = mean_squared_error(y, model.predict(X))
print("Mean Squared Error:", mse)


