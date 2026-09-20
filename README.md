# MLOps Energy Prediction

An end-to-end machine learning and MLOps project for predicting energy consumption using Python, scikit-learn, MLflow, FastAPI, Docker and Docker Compose.

## Project Overview

This project demonstrates a complete ML workflow:

```text
Energy Dataset
      │
      ▼
Data Preparation
      │
      ▼
Model Training
      │
      ▼
MLflow Experiment Tracking
      │
      ▼
MLflow Model Registry
      │
      ▼
Champion Model
      │
      ▼
FastAPI
      │
      ▼
Docker / Docker Compose
      │
      ▼
REST API Prediction
```

The project is designed as a practical MLOps portfolio project and demonstrates how a trained machine learning model can be tracked, registered, deployed and exposed through an API.

## Technologies

* Python
* pandas
* scikit-learn
* MLflow
* FastAPI
* Pydantic
* Docker
* Docker Compose
* Git
* GitHub

## Project Structure

```text
mlops-energy-prediction/
│
├── data/
│   └── energy.csv
│
├── src/
│   ├── api.py
│   ├── generate_data.py
│   ├── predict.py
│   └── train.py
│
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

MLflow-generated files such as `mlruns/` and `mlflow.db` are excluded from Git using `.gitignore`.

The Python virtual environment `.venv/` is also excluded from version control.

## Dataset

The project uses `data/energy.csv`.

The dataset contains the following columns:

| Column            | Description                  |
| ----------------- | ---------------------------- |
| `timestamp`       | Timestamp of the observation |
| `hour`            | Hour of the day              |
| `day_of_week`     | Day of the week              |
| `month`           | Month                        |
| `consumption_kwh` | Energy consumption in kWh    |

The model uses the following features:

```text
hour
day_of_week
month
```

The prediction target is:

```text
consumption_kwh
```

## Machine Learning Model

The current training configuration uses:

```python
MODEL_TYPE = "gradient_boosting"
```

The model is:

```python
GradientBoostingRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42
)
```

The dataset is divided into training and test sets using:

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The following evaluation metrics are calculated:

* MAE — Mean Absolute Error
* RMSE — Root Mean Squared Error

These metrics are logged to MLflow.

## MLflow

MLflow is used for experiment tracking and model management.

The project uses the MLflow experiment:

```text
energy-consumption-prediction
```

The training process logs:

* training dataset
* model type
* model parameters
* MAE
* RMSE
* trained model

The trained model is registered as:

```text
EnergyConsumptionModel
```

The deployed model is selected using the MLflow alias:

```text
champion
```

This allows the API to use the model designated as the current production model without hard-coding a model version.

## FastAPI

The trained model is exposed through a REST API.

The API provides:

### Health / Root endpoint

```http
GET /
```

Example response:

```json
{
  "message": "Energy Consumption Prediction API is running"
}
```

### Prediction endpoint

```http
POST /predict
```

Example request:

```json
{
  "hour": 12,
  "day_of_week": 2,
  "month": 9
}
```

Example response:

```json
{
  "predicted_consumption_kwh": 3.9764390590026277
}
```

## Docker

The FastAPI application is packaged into a Docker image.

The Dockerfile uses:

```text
python:3.12-slim
```

The API runs with Uvicorn on port `8000` inside the container.

The host exposes the API on:

```text
http://localhost:8001
```

## Docker Compose

The complete application can be started using Docker Compose.

The Compose configuration contains two services:

```text
mlflow-energy
    │
    │ MLflow
    │ port 5000
    │
    ▼
energy-prediction-api
    │
    │ FastAPI
    │ port 8001
    ▼
POST /predict
```

Start the system:

```powershell
docker compose up -d
```

Check the services:

```powershell
docker compose ps
```

Expected result:

```text
mlflow-energy           Up ... (healthy)
energy-prediction-api   Up ...
```

Stop the system:

```powershell
docker compose down
```

## Running a Prediction

PowerShell example:

```powershell
curl -Method POST http://localhost:8001/predict `
  -ContentType "application/json" `
  -Body '{"hour":12,"day_of_week":2,"month":9}'
```

Example result:

```text
StatusCode : 200
Content    : {"predicted_consumption_kwh":3.9764390590026277}
```

## MLflow UI

When Docker Compose is running, the MLflow UI is available at:

http://localhost:5000

The MLflow interface can be used to inspect experiments, runs, parameters, metrics and registered models.

## Development Workflow

A typical development workflow is:

```text
1. Modify the ML code
        ↓
2. Train the model
        ↓
3. Log experiment to MLflow
        ↓
4. Register the model
        ↓
5. Assign/update the champion alias
        ↓
6. Build the Docker image
        ↓
7. Start Docker Compose
        ↓
8. Test the FastAPI endpoint
```

## Git and GitHub

The project is maintained using Git and hosted on GitHub.

Repository:

`pbudai25/mlops-energy-prediction`

The project uses the `main` branch.

Generated files and local environments are excluded from version control using `.gitignore`.

## MLOps Concepts Demonstrated

This project currently demonstrates:

* Machine learning model training
* Train/test splitting
* Model evaluation
* Experiment tracking
* MLflow dataset tracking
* MLflow Model Registry
* Model aliases
* Model deployment
* REST API
* FastAPI
* Containerization
* Docker Compose
* Git
* GitHub

## Planned Improvements

The next stages of the project can include:

* Automated unit tests
* GitHub Actions CI
* Automated Docker image builds
* Automated model testing
* Model validation
* Automated deployment
* API health checks
* Model monitoring
* Data drift detection
* Model performance monitoring
* CI/CD pipeline
* Production-oriented configuration using environment variables

## Project Goal

The goal of this project is to demonstrate practical MLOps skills by building an end-to-end machine learning system rather than only training a machine learning model.

The project combines machine learning, experiment tracking, model management, API development, containerization and version control into a single workflow.
