import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats

filepath = r"C:\Users\Microsoft\Downloads\Apocolypse Food Prep.xlsx"
df = pd.read_excel(filepath)
df.replace({np.nan: "No Value"}, inplace=True)
df.drop("Date", axis=1, inplace=True)
#df["Price"]= df["Price"]/df["Price"].max()
# Crée des colonnes dummy 0/1 pour chaque valeur de `Product` et les ajoute au DataFrame
'''dummies = pd.get_dummies(df["Product"], prefix="Product").astype(int)
df = pd.concat([df.drop(columns=["Product"]), dummies], axis=1)'''
p = df.describe()
"""plt.bar(df["Product"], df["Price"])
plt.title("Price vs Product")
plt.xlabel("Product")
plt.ylabel("Price")
plt.show()"""

df = pd.get_dummies(df, columns=["Store", "Product"], drop_first=True)
df["Store_Target"] = df["Store_Target"].astype(int)
df["Store_Walmart"] = df["Store_Walmart"].astype(int)
df["Product_Milk"] = df["Product_Milk"].astype(int)
df["Product_Rice"] = df["Product_Rice"].astype(int)
corre = df["Product_Rice"].corr(df["Price"])
print("Correlation between Product_Rice and Price:", corre)
print(df.dtypes)
print(df.head())