from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
import joblib
import numpy as np
import logging
from datetime import datetime
import time

request_count = 0
error_count = 0
total_latency = 0

app = FastAPI(title="Breast Cancer Prediction API")

MODEL_PATH = "model.pkl"

# 🔹 Load model
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    load_error = str(e)

# 🔹 Logging setup (creates predictions.log automatically)
logging.basicConfig(
    filename="predictions.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)


# 🔹 Input schema
class PredictionInput(BaseModel):
    features: List[float] = Field(
        ...,
        description="30 numeric features"
    )


# 🔹 Output schema
class PredictionOutput(BaseModel):
    prediction: int
    probability_malignant: float


# 🔹 Root
@app.get("/")
def root():
    return {"message": "Breast Cancer Prediction API is running"}


# 🔹 Health check
@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {load_error}")
    return {"status": "ok"}

# 🔹 Metrics

@app.get("/metrics")
def metrics():
    avg_latency = total_latency / request_count if request_count > 0 else 0

    return {
        "total_requests": request_count,
        "errors": error_count,
        "average_latency": avg_latency
    }

# 🔹 Version (NEW)
@app.get("/version")
def version():
    return {
        "model_name": "breast-cancer-classifier",
        "model_version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }


# 🔹 Prediction endpoint
@app.post("/predict")
def predict(input_data: PredictionInput):
    global request_count, error_count, total_latency

    start_time = time.time()

    try:
        x = np.array(input_data.features).reshape(1, -1)

        pred = int(model.predict(x)[0])
        prob = float(model.predict_proba(x)[0][0])

        latency = time.time() - start_time

        request_count += 1
        total_latency += latency

        logging.info(
            f"input={input_data.features}, pred={pred}, prob={prob}, latency={latency}"
        )

        return {
            "prediction": pred,
            "probability_malignant": prob
        }

    except Exception as e:
        error_count += 1
        raise HTTPException(status_code=500, detail=str(e))


