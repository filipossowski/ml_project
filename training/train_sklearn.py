import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import mlflow
import mlflow.sklearn

FEATURES = ["squareMeters", "rooms", "floor", "floorCount", "buildYear", "centreDistance"]
TARGET = "price"

#loading, cleaning and filling data
df = pd.read_csv("data/processed/szczecin.csv")
df = df.dropna(subset=["floorCount"])
df["floor"] = df["floor"].fillna(df["floor"].median())
df["buildYear"] = df["buildYear"].fillna(df["buildYear"].median())

X = df[FEATURES]
y = df[TARGET]

#splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

n_estimators = 100
model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
model.fit(X_train, y_train)

#evaluating the model
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)
print('Random Forest Model Evaluation:')
print(f"MAE:  {mae:,.0f} zł")
print(f"RMSE: {rmse:,.0f} zł")
print(f"R²:   {r2:.3f}")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("szczecin_apt_prices")

with mlflow.start_run():
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.sklearn.log_model(
    model,
    name="model",
    skops_trusted_types=["sklearn.tree._tree.Tree"],
)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/rf_v1.joblib")

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

#evaluating the linear regression model
preds = lr_model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)
print('Linear Regression Model Evaluation:')
print(f"MAE:  {mae:,.0f} zł")
print(f"RMSE: {rmse:,.0f} zł")
print(f"R²:   {r2:.3f}")
