import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from features import load_data
from evidently.presets import DataDriftPreset
from evidently import Report
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("data/processed/rcn_cities.csv")

NUMERIC = ["LOK_POW_UZYT", "rooms", "floor", "centreDistance",
           "year", "market_primary", "market_unknown"]
CITY_COLS = sorted([c for c in df.columns if c.startswith("city_")])
FEATURES = NUMERIC + CITY_COLS
TARGET = "LOK_CENA_BRUTTO"

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42,
                              max_features=0.5, max_depth=30, n_jobs=2)
model.fit(X_train, y_train)
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
mape = mean_absolute_percentage_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"MAE: {mae:,.0f}\nRMSE: {rmse:,.0f}\nMAPE: {mape:.3f}\nR2: {r2:.3f}")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("poland_apt_prices")

with mlflow.start_run():
    mlflow.log_param("dataset", "rcn_cities")
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mape", mape)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(model, name="model", skops_trusted_types=["sklearn.tree._tree.Tree"])

report = Report(metrics=[DataDriftPreset()])
result = report.run(reference_data=X_train, current_data=X_test)   
result.save_html("reports/drift_report.html")

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/rf_rcn_v1.joblib") 

X_train.to_csv("data/train.csv", index=False)
X_test.to_csv("data/live.csv", index=False)

