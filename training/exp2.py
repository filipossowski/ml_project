import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import joblib
import mlflow
import mlflow.sklearn

print(os.getcwd())

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_features=0.5, max_depth=30)
rf_model.fit(X_train, y_train)

#evaluating the random forest model
preds = rf_model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
mape = mean_absolute_percentage_error(y_test, preds)
r2 = r2_score(y_test, preds)
print(f"MAE:  {mae:,.0f} zł")
print(f"RMSE: {rmse:,.0f} zł")
print(f"MAPE: {mape:.3f}")
print(f"R²:   {r2:.3f}")


param_dist = {
    "n_estimators": [100, 150, 200],
    "max_depth": [20, 30, 40],
#    "min_samples_leaf": [1, 2, 5, 10],
#    "max_features": [1.0, "sqrt", 0.5],
    "bootstrap": [True, False],

}

search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=5,                            # 5 random combinations
    cv=3,                                 # 3-fold cross-validation
    scoring="neg_mean_absolute_error",    # optimize MAE
    n_jobs=2,                            
    random_state=42,
)
search.fit(X_train, y_train)

print(search.best_params_)
print(f"Best CV MAE: {-search.best_score_:,.0f} zł")

best_model = search.best_estimator_