# XGBoost Model for Predicating BTC Price

## Overview

This model uses XGBoost, a gradient boosting machine learning model, to predict the short-term price movement of BTC based on historical price data and technical indicators. The model predicts the BTC price movement for the next hour, making it useful for traders and analysts seeking to make informed decisions in the cryptocurrency market. The script includes steps for data preprocessing, model training, hyperparameter tuning, evaluation, and conversion to the ONNX format for model deployment.

***

## **Model Details**

* **Model Type**: XGBoost Regressor
* **Framework**: XGBoost, ONNX
* **Objective**: Predict 1-hour price movement for BTC/USDT (Bitcoin to US Dollar Tether) pair.
* **Input**: A set of technical features and historical price data, including BTC/ETH price ratios, high-low price ranges, and time-based features.
* **Output**: Predicted price movement (a single floating-point value).

## **Input**

The model requires a 2D array \[any, 81]. Each row corresponds to a sample, and each sample contains features that describe past market conditions.

#### **Key Features**:

* **BTC/ETH Price Ratios**: The ratio between BTC and ETH prices, which can capture cross-market dynamics.
* **Historical Returns**: The rate of change in BTC price over different historical periods (e.g., past 1-hour returns).
* **High-Low Price Ranges**: The difference between the highest and lowest prices over a specific time window, indicating volatility.
* **Time-Based Features**: Features such as the hour of the day to capture potential daily patterns in market behavior (e.g., higher volatility in certain times of day).

#### **Example Input:**

```py
{"candles":[[2744.31,2718.0,2725.95,2716.59,2728.49,2731.4,2724.41,2720.47,2722.47,2709.38,2748.8,2730.9,2727.95,2731.0,2734.76,2739.0,2733.6,2724.96,2722.62,2710.89,2707.4,2714.9,2703.99,2714.6,2720.96,2723.14,2717.6,2715.69,2705.0,2694.0,2718.0,2725.95,2716.58,2728.49,2731.4,2724.4,2720.47,2722.46,2709.38,2699.5,97456.78,97134.6,97500.47,97470.87,97804.24,97839.05,97653.19,97527.64,97611.18,97488.62,97538.11,97519.99,97516.2,97833.42,97918.53,97972.26,97780.73,97640.97,97664.06,97526.54,96922.23,96995.41,97223.58,97436.93,97625.3,97642.16,97482.21,97450.01,97439.04,97350.23,97134.59,97500.48,97470.88,97804.24,97839.04,97653.19,97527.63,97611.17,97488.61,97378.91,7.0]]}
```

## **Output**

The model outputs a **single floating-point value** representing the predicted **1-hour return** (price movement) for BTC. A positive value indicates an expected price increase, while a negative value indicates a predicted decrease.

#### **Example Output:**

```py
{
variable: [[0.00007465136877726763]]
}
```

## **Performance**

* **Mean Absolute Error (MAE)**: 0.003714
* **Root Mean Squared Error (RMSE)**: 0.005512
* **R² Score**: -0.000119

## **Limitations and Biases**

* **Market Conditions**:
  The model is based on historical data and technical features, which may not fully account for sudden market events (e.g., regulatory changes, economic shifts) that can impact BTC price movements.
* **Overfitting**:
  Hyperparameter tuning via **GridSearchCV** might lead to overfitting, particularly if the training data does not account for a variety of market conditions or extreme volatility.
* **Temporal Factors**:
  While time-based features are included, the model may not always capture the effects of market shifts during unusual times, such as during sudden news events or high-impact announcements.
* **Non-stationarity of Cryptocurrency Markets**:
  Cryptocurrency markets are volatile and non-stationary, meaning patterns identified in historical data may not always hold in the future.
