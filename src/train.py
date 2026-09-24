import os

import pandas as pd
import mlflow
import mlflow.sklearn

from mlflow import MlflowClient

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. Adatok betöltése
# --------------------------------------------------

df = pd.read_csv("data/energy.csv")

print("Adatok betöltve:")
print(df.head())


# --------------------------------------------------
# 2. MLflow Dataset
# --------------------------------------------------

dataset = mlflow.data.from_pandas(
    df,
    source="data/energy.csv",
    name="energy-consumption-dataset",
    targets="consumption_kwh"
)


# --------------------------------------------------
# 3. Feature-ök és target
# --------------------------------------------------

features = [
    "hour",
    "day_of_week",
    "month"
]

X = df[features]
y = df["consumption_kwh"]


# --------------------------------------------------
# 4. Train / test felosztás
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------------------------
# 5. MLflow
# --------------------------------------------------

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

mlflow.set_experiment(
    "energy-consumption-prediction"
)


# --------------------------------------------------
# 6. Modell kiválasztása
# --------------------------------------------------

MODEL_TYPE = "gradient_boosting"


# --------------------------------------------------
# 7. MLflow Run
# --------------------------------------------------

with mlflow.start_run():

    mlflow.log_input(
        dataset,
        context="training"
    )

    # --------------------------------------------------
    # 8. Modell létrehozása
    # --------------------------------------------------

    if MODEL_TYPE == "random_forest":

        n_estimators = 100
        max_depth = 10

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

    elif MODEL_TYPE == "gradient_boosting":

        n_estimators = 200
        max_depth = 3
        learning_rate = 0.1

        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )

    else:

        raise ValueError(
            f"Ismeretlen modell: {MODEL_TYPE}"
        )

    # --------------------------------------------------
    # 9. Modell tanítása
    # --------------------------------------------------

    model.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------
    # 10. Előrejelzés
    # --------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # --------------------------------------------------
    # 11. Metrikák
    # --------------------------------------------------

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    # --------------------------------------------------
    # 12. Paraméterek MLflow-ba
    # --------------------------------------------------

    mlflow.log_param(
        "model_type",
        MODEL_TYPE
    )

    mlflow.log_param(
        "n_estimators",
        n_estimators
    )

    mlflow.log_param(
        "max_depth",
        max_depth
    )

    if MODEL_TYPE == "gradient_boosting":

        mlflow.log_param(
            "learning_rate",
            learning_rate
        )

    # --------------------------------------------------
    # 13. Metrikák MLflow-ba
    # --------------------------------------------------

    mlflow.log_metric(
        "mae",
        mae
    )

    mlflow.log_metric(
        "rmse",
        rmse
    )

    # --------------------------------------------------
    # 14. Modell mentése MLflow-ba
    # --------------------------------------------------

    model_info = mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name="EnergyConsumptionModel",
        serialization_format="pickle"
    )

    print()
    print("Model URI:", model_info.model_uri)
    print("Model ID:", model_info.model_id)

    # --------------------------------------------------
    # 15. Champion lekérése
    # --------------------------------------------------

    client = MlflowClient()

    champion = client.get_model_version_by_alias(
        "EnergyConsumptionModel",
        "champion"
    )

    champion_run = client.get_run(
        champion.run_id
    )

    champion_mae = champion_run.data.metrics["mae"]

    print()
    print(
        f"Current champion version: {champion.version}"
    )

    print(
        f"Champion MAE:            {champion_mae:.4f}"
    )

    print(
        f"New model MAE:           {mae:.4f}"
    )

    # --------------------------------------------------
    # 16. Modell összehasonlítása
    # --------------------------------------------------

    if mae < champion_mae:

        print(
            "New model is better than champion!"
        )

        model_versions = client.search_model_versions(
            "name='EnergyConsumptionModel'"
        )

        current_run_id = (
            mlflow.active_run().info.run_id
        )

        new_version = None

        for version in model_versions:

            if version.run_id == current_run_id:

                new_version = version.version
                break

        if new_version is None:

            raise RuntimeError(
                "New model version not found."
            )

        client.set_registered_model_alias(
            "EnergyConsumptionModel",
            "champion",
            new_version
        )

        print(
            f"Champion updated to version {new_version}"
        )

    else:

        print(
            "Champion remains unchanged."
        )

    # --------------------------------------------------
    # 17. Eredmények
    # --------------------------------------------------

    print()
    print("Training completed!")
    print(f"Model: {MODEL_TYPE}")
    print(f"MAE:   {mae:.4f}")
    print(f"RMSE:  {rmse:.4f}")
