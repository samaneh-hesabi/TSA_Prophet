"""
A Simple Time Series Forecasting Example using Prophet

This script shows how to:
1. Load time series data
2. Prepare it for forecasting
3. Make predictions using Prophet
4. Evaluate predictions using different metrics
"""

import pandas as pd  # for handling data
from prophet import Prophet  # for making predictions
import matplotlib.pyplot as plt  # for creating plots
from sklearn.metrics import mean_absolute_error, mean_squared_error  # for measuring accuracy
import numpy as np  # for math operations
import warnings
warnings.filterwarnings('ignore')  # hide warning messages

print("=== Simple Time Series Forecasting Example ===")

# Step 1: Load the data
# We're using a pre-made dataset from the data folder
print("\n1. Loading data...")
df = pd.read_csv('data/synthetic_data.csv')
print("First few rows of our data:")
print(df.head())

# Step 2: Prepare data for Prophet
# Prophet needs columns named 'ds' (for dates) and 'y' (for values)
print("\n2. Preparing data...")
df['Date'] = pd.to_datetime(df['Date'])  # convert Date column to datetime
prophet_df = df.rename(columns={
    'Date': 'ds',  # rename Date to ds
    'Value': 'y'   # rename Value to y
})
print("Data prepared for Prophet:")
print(prophet_df.head())

# Step 3: Create and train the Prophet model
print("\n3. Training the model...")
model = Prophet()  # create a Prophet model
model.fit(prophet_df)  # train the model with our data

# Step 4: Make future predictions
print("\n4. Making predictions...")
# Create dates for the next 90 days
future_dates = model.make_future_dataframe(periods=90)
print("Future dates we'll predict for:")
print(future_dates.head())

# Make predictions
forecast = model.predict(future_dates)
print("\nOur predictions (first few rows):")
print(forecast[['ds', 'yhat']].head())

# Step 5: Check how accurate our predictions are
print("\n5. Checking prediction accuracy...")
# Compare predicted vs actual values
predictions = forecast[['ds', 'yhat']].merge(prophet_df, on='ds', how='inner')
print("\nActual vs Predicted values:")
print(predictions[['ds', 'y', 'yhat']].head())

# Calculate different error metrics
print("\nError Metrics (lower values mean better predictions):")
print("----------------------------------------")

# 1. Mean Absolute Error (MAE)
# This shows the average absolute difference between predicted and actual values
mae = mean_absolute_error(predictions['y'], predictions['yhat'])
print(f"MAE  (Mean Absolute Error)      : {mae:.2f}")
print("     → Average difference between predicted and actual values")

# 2. Mean Squared Error (MSE)
# This shows the average squared difference between predicted and actual values
mse = mean_squared_error(predictions['y'], predictions['yhat'])
print(f"MSE  (Mean Squared Error)       : {mse:.2f}")
print("     → Like MAE, but penalizes larger errors more")

# 3. Root Mean Squared Error (RMSE)
# This is the square root of MSE, making it easier to interpret
rmse = np.sqrt(mse)
print(f"RMSE (Root Mean Squared Error)  : {rmse:.2f}")
print("     → Similar to MAE, but in the same units as our data")

# Step 6: Create a plot
print("\n6. Creating forecast plot...")
plt.figure(figsize=(10, 6))
# Plot actual values
plt.plot(prophet_df['ds'], prophet_df['y'], 'b.', label='Actual Values')
# Plot predictions
plt.plot(forecast['ds'], forecast['yhat'], 'r-', label='Predicted Values')
plt.title('Time Series Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('forecast.png')
plt.close()

print("\nDone! The forecast plot has been saved as 'forecast.png'")
print("You can open this file to see how the predictions look!")
