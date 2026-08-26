
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import joblib
import os


DATA_PATH = "../data/model_ready_data.csv"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["future_spending"])
y = df["future_spending"]

print("Dataset loaded successfully")
print("Shape:", df.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


print("\n===== Linear Regression =====")

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

lr_r2 = r2_score(y_test, y_pred_lr)
lr_mae = mean_absolute_error(y_test, y_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))

print("R2 Score :", lr_r2)
print("MAE      :", lr_mae)
print("RMSE     :", lr_rmse)

print("\n===== Random Forest Regression =====")

rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

rf_r2 = r2_score(y_test, y_pred_rf)
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))

print("R2 Score :", rf_r2)
print("MAE      :", rf_mae)
print("RMSE     :", rf_rmse)


print("\n===== Gradient Boosting Regressor =====")

gb = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

gb_r2 = r2_score(y_test, y_pred_gb)
gb_mae = mean_absolute_error(y_test, y_pred_gb)
gb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_gb))

print("R2 Score :", gb_r2)
print("MAE      :", gb_mae)
print("RMSE     :", gb_rmse)


print("\n===== MODEL COMPARISON =====")
comparison = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest", "Gradient Boosting"],
    "R2 Score": [lr_r2, rf_r2, gb_r2],
    "MAE": [lr_mae, rf_mae, gb_mae],
    "RMSE": [lr_rmse, rf_rmse, gb_rmse]
})

print(comparison)


MODEL_DIR = "../models"
MODEL_PATH = os.path.join(MODEL_DIR, "spending_model.pkl")

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

joblib.dump(gb, MODEL_PATH)

print("\nBest model saved successfully at:", MODEL_PATH)


feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": gb.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\n===== FEATURE IMPORTANCE =====")
print(feature_importance)

feature_importance.to_csv(
    "../models/feature_importance.csv",
    index=False
)

print("\nTraining & evaluation completed successfully!")
