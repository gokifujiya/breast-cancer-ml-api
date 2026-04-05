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
- Basic monitoring (request count, error count, average latency)

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
breast_cancer_monitoring/
├── app.py
├── train.py
├── model.pkl
├── requirements.txt
├── Dockerfile
└── README.md
```

Note: model.pkl is generated after running train.py and is not tracked in Git.

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
docker build -t breast-cancer-monitoring .
docker run -p 8081:8080 breast-cancer-monitoring
```

## API Endpoints
- GET / → basic check
- GET /health → system health
- GET /metrics → monitoring metrics
- GET /version → model version info
- POST /predict → prediction

## Monitoring

This project includes basic monitoring capabilities to observe system behavior in real time.

### Features
- Request count tracking
- Error tracking
- Average latency measurement

### Endpoint
GET /metrics

### Example Response
```json
{
  "total_requests": 5,
  "errors": 0,
  "average_latency": 0.12
}

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
- Logging for traceability
- Basic monitoring for observability


## Future Improvements
- Image-based model (medical imaging)
- Cloud deployment
- Model versioning with MLflow registry
- Database logging

