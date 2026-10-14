import tensorflow as tf
import mlflow
import mlflow.tensorflow

model = ...  # tu znajduje się kod budujący i trenujący model TensorFlow

# ---- 1. Zapis modelu w formacie TensorFlow SavedModel ----
model.save("models/tf_model_v1")

# ---- 2. Logowanie modelu do MLflow ----
mlflow.tensorflow.log_model(model, "model")