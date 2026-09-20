import mlflow
import pandas as pd


# MLflow szerver
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Champion modell betöltése
model = mlflow.pyfunc.load_model(
    "models:/EnergyConsumptionModel@champion"
)

# Bemeneti adatok DataFrame-ben
input_data = pd.DataFrame({
    "hour": [18],
    "day_of_week": [2],
    "month": [9]
})

# Predikció
prediction = model.predict(input_data)

print("Input:")
print(input_data)

print()
print("Predicted consumption:")
print(f"{prediction[0]:.4f} kWh")