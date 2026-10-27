from fastapi import FastAPI
from app.schemas import PredictRequest
from app.model import load_models
import numpy as np

app = FastAPI()

@app.on_event("startup")
def load():
    global sk_model, tf_model
    sk_model, tf_model = load_models()

@app.post("/predict")
def predict(req: PredictRequest):
    X = np.array(req.instances)
    if req.model == "sklearn":
        return {"predictions": sk_model.predict(X).tolist()}
    else:
        preds = np.expm1(tf_model.predict(X).flatten())
        return {"predictions": preds.tolist()}

@app.get("/health-check")
def health():
    return {"status": "ok"}