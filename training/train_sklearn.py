import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib
import mlflow
import mlflow.sklearn

TARGET = "price"
NUMERIC = ["squareMeters", "rooms", "floor", "floorCount", "buildYear", "centreDistance"]

#loading, cleaning and filling data
df_all = pd.read_csv("data/processed/cities_filtered.csv")
df_all = df_all.dropna(subset=["floorCount"])
df_all["floor"] = df_all["floor"].fillna(df_all["floor"].median())
df_all["buildYear"] = df_all["buildYear"].fillna(df_all["buildYear"].median())

CITY_COLS = [c for c in df_all.columns if c.startswith("city_")]
FEATURES = NUMERIC + CITY_COLS

X = df_all[FEATURES]
y = df_all[TARGET]

#splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_estimators = 100
model = RandomForestRegressor(n_estimators=100, random_state=42, max_features=0.5, max_depth=30)
model.fit(X_train, y_train)

#evaluating the model
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
mape = mean_absolute_percentage_error(y_test, preds)
r2 = r2_score(y_test, preds)
print('Random Forest Model Evaluation:')
print(f"MAE:  {mae:,.0f} zł")
print(f"RMSE: {rmse:,.0f} zł")
print(f"MAPE: {mape:.3f}")
print(f"R²:   {r2:.3f}")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("poland_apt_prices")

with mlflow.start_run():
    mlflow.log_param("dataset", "all_cities")
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/rf_all_cities_v1.joblib")


