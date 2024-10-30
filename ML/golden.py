import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Read data from CSV file
data = pd.read_csv('gooh.csv')

# Drop rows with missing values
data = data.dropna()

# Convert the date column to datetime object
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)
data.info()

# Plot the total value chart (FIGURE 1)
fig1 = px.line(data, x='Date', y='Vietnam(VND)', title='Gold Prices in Vietnam all time(VND/Ounces)')
fig1.show()

# Get data from 2015 onwards
data = data[data['Date'].dt.year >= 2015]

# Convert the date column to an integer representing the number of days since the first date in the dataset
data['Days'] = (data['Date'] - data['Date'].min()).dt.days

# Split the data into features (X) and target (y)
X = data[['Days']]  # Specify column name for features
y = data['Vietnam(VND)']

# Rename price data 
data = data.rename(columns={'Vietnam(VND)': 'Real'})

# Create and train linear regression model
model = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model.fit(X_train, y_train)

# Create, train, and test the random forest model
model1 = RandomForestRegressor()
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.25, random_state=42)
model1.fit(X_train1, y_train1)
predictions1 = model1.predict(X_test1)

# Predict gold prices from 2015 onwards
start_date = pd.to_datetime('2015-01-01')
end_date = pd.to_datetime('2024-04-30')
prediction_dates = pd.date_range(start_date, end_date, freq='BME')  
prediction_days = (prediction_dates - data['Date'].min()).days.values.reshape(-1, 1)
predicted_prices = model.predict(prediction_days)
predicted_prices1 = model1.predict(prediction_days)

# Display prediction table for both linear regression and random forest
prediction_df = pd.DataFrame({'Date': prediction_dates, 'Prediction ':  predicted_prices.round(2) })
prediction_df1 = pd.DataFrame({'Date': prediction_dates, 'Prediction':  predicted_prices1})
predictions_dfin = pd.DataFrame({'Date': prediction_dates, 'Prediction(LR)':  predicted_prices.round(2), 'Prediction(RFR)':  predicted_prices1})
print(predictions_dfin.to_string(index=False, header=True))

# Combine prediction data with original data
combined_data = pd.concat([data[['Date', 'Real']], prediction_df])
combined_data1 = pd.concat([data[['Date', 'Real']], prediction_df1])

# Add the 'Prediction' column to the combined_data DataFrame
combined_data['Prediction'] = prediction_df['Prediction ']
combined_data1['Prediction'] = prediction_df1['Prediction']

# Plot the predicted prices along with real prices from 2015 to the end (FIGURE 2+3) 
fig2_combined = px.line(combined_data, x='Date', y=['Real', 'Prediction'], 
                       title='Gold Prices and Predicted Prices from 2015 to 2024(VND/Ounces) - Linear Regression')
fig2_combined.show()

fig3_combined = px.line(combined_data1, x='Date', y=['Real', 'Prediction'], 
                       title='Gold Prices and Predicted Prices from 2015 to 2024 (VND/Ounces) - Random Forest Regressor')
fig3_combined.show()

# Calculate metrics with data information
combined_data.info()
mse = mean_squared_error(data['Real'], model.predict(X))
print("Mean Squared Error (MSE) from 2015 to the end:", mse)

mae = np.mean(np.abs(data['Real'] - model.predict(X)))
print("Mean Absolute Error (MAE) from 2015 to the end:", mae)

# Calculate R-squared
r_squared = model.score(X, y)
print("R-squared from 2015 to the end:", r_squared)

# Calculate RMSE
rmse = np.sqrt(mse)
print("Root Mean Squared Error (RMSE) Linear Regression:", rmse)

rmse1 = np.sqrt(mean_squared_error(y_test1, predictions1))
print("Root Mean Squared Error (RMSE) Random Forest Regressor:", rmse1)
