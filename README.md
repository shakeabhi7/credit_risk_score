# Credit Risk Scoring System

> End-to-end production ML pipeline for real-time credit risk assessment and automated decisioning.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=flat&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?style=flat&logo=mongodb&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracked-0194E2?style=flat&logo=mlflow&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Production-red?style=flat)

---

## Model Results

| Model | Accuracy | AUC-ROC | F1 Score |
|-------|----------|---------|----------|
| **XGBoost (Production)** ⭐ | **85.2%** | **0.87** | **0.84** |
| Random Forest | 84.5% | 0.86 | 0.83 |
| Neural Network | 83.8% | 0.84 | 0.81 |

**API Response Time:** ~50ms | **Deployment:** Containerized (Docker)

---

## Architecture

```
Raw Data (CSV)
     │
     ▼
┌─────────────────┐
│  Data Pipeline  │  ← data_cleaner + data_validator
│  (DVC tracked)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Feature      │  ← 8 engineered features
│   Engineering   │    (debt_to_income, credit_utilization, etc.)
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│  Model Training │ ───► │   MLflow     │
│  XGBoost + RF   │      │  Experiment  │
│  + Neural Net   │      │  Tracking    │
└────────┬────────┘      └──────────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│   FastAPI       │ ───► │   MongoDB    │
│   REST API      │      │  Prediction  │
│  /predict       │      │   Logging    │
└────────┬────────┘      └──────────────┘
         │
         ▼
┌─────────────────┐
│   Streamlit     │  ← Dashboard, New Prediction,
│   Dashboard     │    Search by ID, Export
└─────────────────┘
```

---

## Features

- **Real-time Credit Scoring** — REST API endpoint with <50ms response time
- **8 Engineered Features** — debt-to-income ratio, credit utilization, employment stability, and more
- **3 ML Models** — XGBoost (production), Random Forest, Neural Network — compared with MLflow
- **Class Imbalance Handling** — SMOTE + class weighting for imbalanced credit data
- **Full MLOps Stack** — MLflow experiment tracking + DVC data versioning
- **MongoDB Logging** — Every prediction logged with input features, engineered features, and metadata
- **Interactive Dashboard** — Streamlit UI for predictions, history, search, and CSV export
- **Docker Ready** — Fully containerized for consistent deployment

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Models | XGBoost, Random Forest, Neural Network (TensorFlow) |
| API | FastAPI, Pydantic, Uvicorn |
| Frontend | Streamlit |
| Database | MongoDB |
| MLOps | MLflow, DVC |
| Data | Pandas, NumPy, Scikit-learn, imbalanced-learn |
| Containerization | Docker, Docker Compose |

---

## Quick Start

### Option 1 — Docker (Recommended)

```bash
# Clone the repo
git clone https://github.com/shakeabhi7/credit_risk_score.git
cd credit-risk-scoring

# Copy environment file and fill in your values
cp .env.example .env

# Run with Docker Compose (API + MongoDB)
docker-compose up --build

# API will be available at: http://localhost:8000
# Streamlit dashboard at:  http://localhost:8501
```

### Option 2 — Local Setup

```bash
# Clone and install dependencies
git clone https://github.com/shakeabhi7/credit_risk_score.git
cd credit-risk-scoring
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your MongoDB URI

# Start FastAPI
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# In a new terminal — Start Streamlit
streamlit run app.py
```

---

## API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Make a Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 30,
    "income": 75000,
    "debt": 15000,
    "credit_limit": 50000,
    "credit_used": 20000,
    "employment_years": 3,
    "employment_type": "Salaried"
  }'
```

### Sample Response
```json
{
  "customer_id": "CUST_1708017930_a7f3c",
  "predicted_class": 0,
  "probability": 0.87,
  "confidence": 0.92,
  "model_name": "xgboost",
  "message": "Good credit customer - Low Risk"
}
```

**Interactive API docs available at:** `http://localhost:8000/docs`

---

## Project Structure

```
credit_risk_scoring/
├── app.py                        # Streamlit main app
├── requirements.txt              # Python dependencies (pinned versions)
├── Dockerfile                    # API container
├── docker-compose.yml            # Multi-service orchestration
├── .env.example                  # Environment variables template
├── .gitignore
├── dvc.yaml                      # DVC pipeline stages
├── dvc.lock
│
├── src/
│   ├── api/
│   │   ├── main.py               # FastAPI app + endpoints
│   │   ├── schemas.py            # Pydantic request/response models
│   │   ├── inference_service.py  # Prediction pipeline
│   │   └── models_loader.py      # Model loading at startup
│   │
│   ├── config/
│   │   ├── config_loader.py      # YAML config reader
│   │   ├── data_config.yaml
│   │   ├── feature_config.yaml
│   │   ├── model_config.yaml
│   │   └── params.yaml
│   │
│   ├── data/
│   │   ├── data_loader.py        # CSV loading utilities
│   │   ├── data_cleaner.py       # Missing values, type fixes
│   │   ├── data_validator.py     # Schema and range validation
│   │   └── synthetic_data_generator.py
│   │
│   ├── features/
│   │   └── feature_engineer.py   # 8 engineered features
│   │
│   ├── preprocessing/
│   │   ├── preprocessor.py       # StandardScaler + encoding
│   │   ├── data_pipeline.py      # Full preprocessing pipeline
│   │   └── imbalance_handler.py  # SMOTE + class weighting
│   │
│   ├── models/
│   │   ├── model_trainer.py      # Training logic per model
│   │   ├── model_evaluator.py    # Metrics + confusion matrix
│   │   ├── model_training_pipeline.py  # Orchestrates full training
│   │   ├── hyperparameter_tuner.py     # GridSearchCV / RandomSearch
│   │   └── mlflow_tracker.py     # MLflow experiment logging
│   │
│   ├── database/
│   │   ├── mongo_client.py       # MongoDB connection
│   │   ├── prediction_logger.py  # Log predictions to MongoDB
│   │   └── db_queries.py         # Aggregation queries
│   │
│   └── monitoring/
│       └── logger.py             # Centralized logging setup
│
├── pages/
│   ├── new_prediction.py         # Streamlit prediction form
│   ├── dashboard.py              # Stats and charts
│   ├── search_by_id.py           # Customer lookup
│   └── export.py                 # CSV download
│
├── notebooks/
│   ├── 01_eda_notebook.py        # Exploratory data analysis
│   ├── feature_analysis.py       # Feature importance analysis
│   ├── visuals/                  # EDA plots
│   └── feature_visuals/          # Engineered feature plots
│
├── data/
│   ├── raw/                      # Original data (DVC tracked)
│   ├── processed/                # Cleaned + featured data
│   └── artifacts/                # Saved models (.joblib)
│
└── tests/                        # Unit tests (coming soon)
    └── test_api.py
```

---

## Engineered Features

| Feature | Formula | Business Meaning |
|---------|---------|-----------------|
| `debt_to_income` | debt / income | Repayment capacity |
| `credit_utilization` | credit_used / credit_limit | Credit behaviour |
| `income_per_age` | income / age | Earning efficiency |
| `employment_stability` | employment_years / age | Job stability |
| `income_to_credit` | income / credit_limit | Income vs credit ratio |
| `income_squared` | income² | Non-linear income effect |
| `debt_impact` | debt × credit_utilization | Combined debt pressure |
| `age_group` | Binned age categories | Age-based risk segment |

---

## MLflow Experiment Tracking

Models tracked with MLflow — compare runs, parameters, and metrics:

```bash
mlflow ui
# Open http://localhost:5000
```

Tracked per experiment run:
- Hyperparameters (n_estimators, max_depth, learning_rate, etc.)
- Metrics (accuracy, AUC-ROC, F1, precision, recall)
- Model artifacts (saved .joblib files)
- Feature importance plots

---

## Environment Variables

Copy `.env.example` to `.env` and fill in:

```env
MONGO_URI=mongodb://localhost:27017/
API_URL=http://localhost:8000
MLFLOW_TRACKING_URI=http://localhost:5000
```

---

## Contact

**Abhishek**
- LinkedIn: [linkedin.com/in/shakeabhi](https://linkedin.com/in/shakeabhi)
- GitHub: [github.com/shakeabhi7](https://github.com/shakeabhi7)
- Email: kumarabhishekt7@gmail.com
