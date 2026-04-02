# Breast Cancer Prediction API

## Overview
This project is an end-to-end machine learning system that predicts breast cancer based on numerical features.

It demonstrates how to move from model training to deployment using modern MLOps tools.

---

## Features
- Machine learning model (RandomForest)
- REST API built with FastAPI
- Docker containerization
- MLflow experiment tracking
- Prediction logging (audit trail)

---

## Tech Stack
- Python
- FastAPI
- Scikit-learn
- MLflow
- Docker

---

## Project Structure
```text
breast_cancer_demo/
├── app.py
├── train.py
├── model.pkl
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## How to Run

### 1. Train the model
This creates `model.pkl`.
```bash
python train.py
```

### 2. Run API locally
```bash
uvicorn app:app --reload
```

### 3. Run with Docker
```bash
docker build -t breast-cancer-api .
docker run -p 8080:8080 breast-cancer-api
```

## API Endpoints
- GET / → basic check
- GET /health → system health
- GET /version → model version info
- POST /predict → prediction

## Example Request
```JSON
{
  "features": [
    17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.0787,
    1.095,0.9053,8.589,153.4,0.0064,0.049,0.0537,0.0159,0.03,0.0062,
    25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189
  ]
}
```

## Output Example
```JSON
{
  "prediction": 0,
  "probability_malignant": 0.96
}
```

## Purpose
This project demonstrates:
- End-to-end ML workflow
- Deployment of ML models
- API-based inference
- Logging for traceability (important for AI governance)

## Future Improvements
- Image-based model (medical imaging)
- Cloud deployment
- Model versioning with MLflow registry
- Database logging

