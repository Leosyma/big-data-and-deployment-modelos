
# 🚀 Big Data & Model Deployment with MLflow

This project demonstrates a complete machine learning lifecycle for regression and hierarchical models using **MLflow** for model tracking, versioning, and API-based deployment.

It includes:

- Model training and logging with MLflow
- API-based predictions using deployed models
- Version control and production management of models
- Multilevel modeling (HLM2) for grouped data

---

## 📂 Project Structure

### 1. 📊 Dataset Exploration
- Loads and explores two datasets:  
  - `tempodist.csv`: records of time vs distance  
  - `estudante_escola.csv`: student performance grouped by school
- Inspects data types, null values, and descriptive statistics

### 2. 📈 Simple Linear Regression Model
- Trains two versions of regression models:
  - **Null model**: `tempo ~ 1`
  - **Final model**: `tempo ~ distancia`
- Logs:
  - Parameters, metrics (F-statistic, p-value, R²)
  - Fitted values plot
  - Model summary as a text artifact
- Registers the models and stages the final version using MLflow

### 3. 🧪 Predicting via API (Localhost)
- Demonstrates how to:
  - Serve models via MLflow on specific ports
  - Format input data and send requests via `requests.post`
  - Parse JSON responses to extract predictions

### 4. 🏫 Hierarchical Modeling (HLM2)
Works with grouped data (`escola`) to create multilevel models using `statsmodels.MixedLM`:
- **Null HLM2 model**: Random intercepts only
- **HLM2 with random intercepts**: `desempenho ~ horas`
- **HLM2 with random intercepts & slopes**: `desempenho ~ horas`
- **Final HLM2 model**: `desempenho ~ horas + texp + horas:texp`

All models are:
- Automatically logged with `mlflow.statsmodels.autolog()`
- Evaluated and served via API (`localhost:5300`)

---

## 🔁 MLflow Lifecycle Integration

✔️ Model registration  
✔️ Experiment tracking  
✔️ Model staging and promotion  
✔️ API-based prediction  
✔️ Artifact logging (charts, summaries)

---

## 🧰 Tech Stack

- `pandas`
- `statsmodels`
- `mlflow`
- `matplotlib`
- `requests`
- `json`

---

## 📦 How to Run

To serve the models:

```bash
# Activate virtual environment and set tracking URI
SET MLFLOW_TRACKING_URI=http://localhost:5000

# Serve the model (example for production)
mlflow models serve -m models:/tempo-distancia/production -p 5200 --no-conda
```

Then make API requests to `http://127.0.0.1:5200/invocations` with JSON-formatted input.
