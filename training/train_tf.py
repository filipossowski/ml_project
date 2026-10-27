import os
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from features import load_data

tf.keras.utils.set_random_seed(42)

X, y, FEATURES = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

norm = tf.keras.layers.Normalization()
norm.adapt(X_train.to_numpy())

model = tf.keras.Sequential([
    tf.keras.Input(shape=(len(FEATURES),)),
    norm,
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mae")
early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("poland_apt_prices")

with mlflow.start_run():
    model.fit(X_train, np.log1p(y_train), validation_split=0.2, epochs=200,
              batch_size=256, callbacks=[early_stop], verbose=2)

#evaluation
    preds = np.expm1(model.predict(X_test, verbose=0).squeeze())
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = mean_absolute_percentage_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print("TensorFlow Model Evaluation:")
    print(f"MAE:  {mae:,.0f} zł")
    print(f"RMSE: {rmse:,.0f} zł")
    print(f"MAPE: {mape:.3f}")
    print(f"R²:   {r2:.3f}")

    mlflow.log_param("dataset", "all_cities")
    mlflow.log_param("model_type", "keras_mlp")
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mape", mape)
    mlflow.log_metric("r2", r2)

    os.makedirs("models", exist_ok=True)
    model.save("models/tf_model_v1.keras")
    mlflow.tensorflow.log_model(model, name="model")