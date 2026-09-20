import mlflow
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Energy Consumption Prediction API",
    version="1.0"
)

mlflow.set_tracking_uri("http://mlflow-energy:5000")

client = mlflow.MlflowClient()

model_version = client.get_model_version_by_alias(
    "EnergyConsumptionModel",
    "champion"
)

model_id = model_version.source.split("/")[-1]

model_path = (
    f"/mlflow/mlruns/1/models/{model_id}/artifacts"
)

model = mlflow.pyfunc.load_model(model_path)


class EnergyInput(BaseModel):
    hour: int
    day_of_week: int
    month: int


@app.get("/")
def root():
    return {
        "message": "Energy Consumption Prediction API is running"
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