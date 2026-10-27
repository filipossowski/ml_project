import os
import joblib
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

def load_models():
    sk_model = joblib.load(os.path.join(MODELS_DIR, "rf_all_cities_v1.joblib"))
    tf_model = tf.keras.models.load_model(os.path.join(MODELS_DIR,"tf_model_v1.keras"))
    return sk_model, tf_model