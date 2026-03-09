import pandas as pd
import numpy as np
import seaborn as sns   
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
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

Predict = model.predict([[15]])
print("Predicted Price for Production Cost 15:", Predict)   

score = model.score(X, y)
print("Model R^2 Score:", score)

mse = mean_squared_error(y, model.predict(X))
print("Mean Squared Error:", mse)

