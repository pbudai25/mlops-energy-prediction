import logging
import os
import time

import mlflow
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


app = FastAPI(
    title="Energy Consumption Prediction API",
    version="1.0"
)


def load_model():
    mlflow.set_tracking_uri(
        os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://mlflow-energy:5000"
        )
    )

    model_uri = "models:/EnergyConsumptionModel@champion"

    return mlflow.pyfunc.load_model(model_uri)


if os.getenv("LOAD_MLFLOW_MODEL", "false").lower() == "true":
    model = load_model()
else:
    model = None


class EnergyInput(BaseModel):
    hour: int
    day_of_week: int
    month: int


@app.get("/")
def root():
    return {
        "message": "Energy Consumption Prediction API is running"
    }


@app.get("/health")
def health():
    if model is None:
        return {
            "status": "unhealthy",
            "model_loaded": False
        }

    return {
        "status": "healthy",
        "model_loaded": True,
        "model": "EnergyConsumptionModel",
        "alias": "champion"
    }


@app.post("/predict")
def predict(data: EnergyInput):
    start_time = time.perf_counter()

    input_data = pd.DataFrame({
        "hour": [data.hour],
        "day_of_week": [data.day_of_week],
        "month": [data.month]
    })

    prediction = model.predict(input_data)

    predicted_value = float(prediction[0])

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "prediction | hour=%s day_of_week=%s month=%s "
        "result=%.6f latency_ms=%.2f",
        data.hour,
        data.day_of_week,
        data.month,
        predicted_value,
        elapsed_ms
    )

    return {
        "predicted_consumption_kwh": predicted_value
    }

