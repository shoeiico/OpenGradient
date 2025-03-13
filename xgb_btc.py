import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as sklearn_train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import onnxmltools
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

# ✅ Step 1: Load and Preprocess Data
def load_data(file_path):
    df = pd.read_csv(file_path)

    # Handle missing values properly
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    return df

# ✅ Step 2: Train-Test Split
def train_test_split(df):
    X = df.drop(columns=["target_BTCUSDT"])  # Features
    y = df["target_BTCUSDT"]  # Target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test, scaler

# ✅ Step 3: Hyperparameter Tuning with GridSearchCV
def tune_hyperparameters(X_train, y_train):
    """
    Performs hyperparameter tuning using GridSearchCV.
    """
    print("\n🚀 Tuning Hyperparameters with GridSearchCV...")

    # Hyperparameter grid
    param_grid = {
        'learning_rate': [0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.3, 0.5],
        'alpha': [0, 1, 10]
    }

    # Initialize XGBoost Regressor
    xgb_model = xgb.XGBRegressor(objective="reg:squarederror")

    # GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Best parameters
    print(f"✅ Best Hyperparameters: {grid_search.best_params_}")
    return grid_search.best_estimator_

# ✅ Step 4: Evaluate Model
def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"✅ Mean Absolute Error (MAE): {mae:.6f}")
    print(f"✅ Root Mean Squared Error (RMSE): {rmse:.6f}")
    print(f"✅ R² Score: {r2:.6f}")

    return predictions

# ✅ Step 5: Convert to ONNX
def convert_to_onnx(model, input_dim, output_path="modelthon_xgb_btc_1hour_return.onnx"):
    initial_type = [("candles", FloatTensorType([None, input_dim]))]
    onnx_model = onnxmltools.convert.convert_xgboost(model, initial_types=initial_type)
    onnxmltools.utils.save_model(onnx_model, output_path)
    print(f"✅ ONNX model saved to {output_path}")
    return output_path

# ✅ Step 6: Verify ONNX Model
def verify_onnx_model(model_path, X_test, model):
    """
    Verify ONNX model against the trained XGBoost model using Mean Absolute Difference.
    """
    ort_session = ort.InferenceSession(model_path)
    test_subset = X_test[:5]
    ort_inputs = {"candles": test_subset.astype(np.float32)}

    onnx_output = ort_session.run(None, ort_inputs)[0].flatten()
    original_output = model.predict(test_subset)

    mean_diff = np.mean(np.abs(original_output - onnx_output))
    print("\n=== Verifying ONNX Model ===")
    print(f"✅ Original Model Predictions: {original_output[:5]}")
    print(f"✅ ONNX Model Predictions:     {onnx_output[:5]}")
    print(f"✅ Mean Absolute Difference: {mean_diff:.8f}")

    if mean_diff < 0.0001:
        print("✅ ONNX model predictions match closely!")
    else:
        print("❌ ONNX model predictions show noticeable deviation!")

# 🚀 Main Execution
if __name__ == "__main__":
    file_path = "../data/BTCUSDT_1h_spot_forecast_training_new.csv"
    
    df = load_data(file_path)
    X_train, X_test, y_train, y_test, scaler = train_test_split(df)

    # Hyperparameter tuning to find the best model
    xgb_model = tune_hyperparameters(X_train, y_train)
    
    # Evaluate the tuned model
    evaluate_model(xgb_model, X_test, y_test)

    # Convert the tuned model to ONNX format
    onnx_model_path = convert_to_onnx(xgb_model, input_dim=X_train.shape[1])

    # Verify the ONNX model
    verify_onnx_model(onnx_model_path, X_test, xgb_model)
