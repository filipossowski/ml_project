import os
import logging
import numpy as np
from fastapi import FastAPI
from app.schemas import PredictRequest
from app.model import load_models
from prometheus_client import Counter, Histogram

LOG_FILE = "logs/app.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

app = FastAPI()

REQUEST_COUNT = Counter("request_count", "Number of requests")
REQUEST_LATENCY = Histogram("request_latency_seconds", "Request latency")

@app.on_event("startup")
def load():
    global sk_model, tf_model
    sk_model, tf_model = load_models()

@app.post("/predict")
@REQUEST_LATENCY.time()
def predict(req: PredictRequest):
    REQUEST_COUNT.inc()
    X = np.array(req.instances)
    try:
        if req.model == "sklearn":
            preds = sk_model.predict(X).tolist()
        else:
            preds = np.expm1(tf_model.predict(X).flatten()).tolist()

        logger.info(f"PREDICTION model={req.model}, input={req.instances}, output={preds}")
        return {"predictions": preds}

    except Exception as e:
        logger.error(f"ERROR during prediction: {e}")
        raise

@app.get("/health-check")
def health():
    return {"status": "ok"}