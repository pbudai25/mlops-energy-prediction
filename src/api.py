import os

import mlflow
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel


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

    client = mlflow.MlflowClient()

    model_version = client.get_model_version_by_alias(
        "EnergyConsumptionModel",
        "champion"
    )

    model_id = model_version.source.replace(
        "models:/",
        ""
    )

    model_path = (
        f"/mlflow/mlruns/1/models/{model_id}/artifacts"
    )

    return mlflow.pyfunc.load_model(model_path)


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
    input_data = pd.DataFrame({
        "hour": [data.hour],
        "day_of_week": [data.day_of_week],
        "month": [data.month]
    })

    prediction = model.predict(input_data)

    return {
        "predicted_consumption_kwh": float(prediction[0])
    }