<div style="font-size:2.5em; font-weight:bold; text-align:center; margin-top:20px;">Time Series Forecasting with Prophet</div>

A simple demonstration project showing how to use Facebook Prophet for time series forecasting with synthetic data. This project serves as an educational example of how to implement time series forecasting using the Prophet library.

# 1. Project Overview

This repository contains a basic time series forecasting project that demonstrates:

- Working with synthetic time series data
- Time series data preparation for Prophet
- Model training and forecasting
- Model evaluation using multiple metrics
- Basic visualization of forecasts

# 2. Project Structure

- `ddd.py`: Main Python script containing the forecasting pipeline
- `requirements.txt`: List of required Python packages
- `environment.yml`: Conda environment configuration file
- `data/`: Directory containing synthetic time series data
- `forecast.png`: Visualization of the forecast results

# 3. Dataset

The project uses synthetic time series data stored in `data/synthetic_data.csv`. The data includes:
- Date
- Value

# 4. Installation

## 4.1 Using pip

```bash
pip install -r requirements.txt
```

## 4.2 Using conda

```bash
conda env create -f environment.yml
conda activate tms_prophet
```

# 5. Usage

Run the Python script:
```bash
python ddd.py
```

The script will:
1. Load the synthetic data
2. Prepare it for forecasting
3. Train a Prophet model
4. Generate forecasts for the next 90 days
5. Evaluate the model using multiple metrics:
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
6. Create and save a forecast visualization

# 6. Visualization

The script generates `forecast.png` which shows:
- Actual values (blue dots)
- Predicted values (red line)
- Date range covering both historical data and future predictions

# 7. References

- [Prophet Documentation](https://facebook.github.io/prophet/)
