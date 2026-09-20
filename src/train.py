import pandas as pd
import mlflow
import mlflow.sklearn

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

mlflow.set_tracking_uri(
    "http://127.0.0.1:5000"
)

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

        n_estimators = 100
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

    mlflow.sklearn.log_model(
        model,
        name="model",
        serialization_format="pickle"
    )


    # --------------------------------------------------
    # 15. Eredmények
    # --------------------------------------------------

    print()
    print("Training completed!")
    print(f"Model: {MODEL_TYPE}")
    print(f"MAE:   {mae:.4f}")
    print(f"RMSE:  {rmse:.4f}")