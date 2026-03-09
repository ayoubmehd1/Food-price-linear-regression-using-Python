from calendar import month
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


path = "C:\\Users\\Microsoft\\Downloads\\archive\\mydata.csv"
df = pd.read_csv(path)

myT = df.iloc[0:5, 0:9]   # Display first 5 rows and first 9 column

myT['attendance'] = myT['attendance'].replace(',', '', regex=True).astype(int)  # Clean and convert 'attendance' to integer
#myT.drop('date', axis=1, inplace=True)  # Drop the 'date' column
print(myT)
print(myT['attendance'].mean())  # Calculate and print mean attendance
avg_attendance_per_stadium = myT.groupby('stadium')['attendance'].mean()  # Group by stadium and calculate mean attendance
print(avg_attendance_per_stadium)

"""   example of linear regression model to predict attendance based on features

myT['date'] = pd.to_datetime(myT['date'])  # Convert 'date' column to datetime
myT['month'] = myT['date'].dt.month  # Extract month from 'date' column

myT["total_goal"] = myT["Goals Home"] + myT["Away Goals"]  # Calculate total goals

features = ['total_goal', 'month', 'Goals Home', 'Away Goals', 'stadium']

myT_features = pd.get_dummies(myT[features], drop_first=True)  # One-hot encode categorical variables
print(myT_features)

y = myT['attendance']  # Target variable
X = myT_features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Split data into training and testing sets
model = LinearRegression()  # Initialize linear regression model
model.fit(X_train, y_train)  # Train the model
y_pred = model.predict(X_test)  # Make predictions on the test set
print("Predicted attendance:", y_pred)  # Print predicted attendance

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)    
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)   
print("R-squared:", r2)
"""

# Determine match results based on goals scored
def get_results(row):
    if row["Goals Home"] > row["Away Goals"]:
        return "Home Win"
    elif row["Goals Home"] < row["Away Goals"]:
        return "Away Win"
    else:
        return "Draw"

myT['result'] = myT.apply(get_results, axis=1)  # Apply function to determine match result
print(myT[['Goals Home', 'Away Goals', 'result']])  # Print goals and results

features = [
    "stadium",
    "Home Team",
    "Away Team",
    "class"
]

myT_features = pd.get_dummies(myT[features], drop_first=True)  # One-hot encode categorical variables

x = myT_features
y = myT['result']
y = y.map({'Home Win': 2, 'Draw': 1, 'Away Win': 0})  # Map results to numerical values

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.25, random_state=42)  # Split data into training and testing sets

model = LinearRegression()  # Initialize linear regression model
model.fit(x_train, y_train)  # Train the model
y_predict = model.predict(x_test)  # Make predictions on the test set

r2_score = model.score(x_test, y_test)  # Calculate R-squared score
print("R-squared score:", r2_score)  # Print R-squared score
print("Predicted results:", y_predict)  # Print predicted results


# Plotting average attendance per stadium
"""plt.bar(myT['stadium'], myT['attendance'], color='red', width=0.4, align='center', alpha=0.7, edgecolor='black')
plt.xlabel('Stadium')
plt.ylabel('Attendance')
plt.title('Attendance per Stadium')
plt.show()"""