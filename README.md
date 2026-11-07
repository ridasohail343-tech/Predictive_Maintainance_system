# Real-World Predictive Maintenance and Equipment Health Monitoring System
https://predictivemaintainancesystem-rhvo7bswbvnnvesnzgitko.streamlit.app/
## Overview

This project is a machine learning-based predictive maintenance system designed to monitor industrial air compressor equipment using sensor data from the MetroPT-3 dataset.

The system uses Isolation Forest for anomaly detection and FastAPI to serve predictions. A Streamlit dashboard provides sensor input, equipment status monitoring, anomaly visualization, and prediction history.

## Key Features

- Industrial sensor data preprocessing
- Equipment failure labeling
- Isolation Forest anomaly detection
- Model evaluation against known failure periods
- FastAPI prediction service
- SQLite database for prediction history
- Interactive Streamlit dashboard
- Sensor input and real-time API predictions
- CSV export of prediction history
- Automated API tests

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Streamlit
- SQLite
- Joblib
- Pytest

## Machine Learning

**Algorithm:** Isolation Forest

**Input Features:**
- TP2
- TP3
- H1
- Motor Current

The model identifies unusual equipment behavior based on sensor readings.

## Dataset

MetroPT-3 Air Compressor Dataset.

The dataset contains industrial sensor measurements collected from an air compressor system.

## Model Evaluation

The anomaly detection model was evaluated against manually labeled known failure periods.

| Metric | Result |
|---|---:|
| Accuracy | 76.99% |
| Failure Precision | 0.08 |
| Failure Recall | 1.00 |
| Failure F1-score | 0.15 |
| Anomaly Detection Rate | 24.99% |

Note: Anomaly detection identifies unusual sensor behavior and does not independently confirm an equipment failure. The low precision indicates that many detected anomalies were not within the labeled failure periods.

## Project Structure

```text
Predictive-maintaince-system/
│
├── artifacts/
├── backend/
│   ├── main.py
│   └── database.py
├── data/
│   ├── raw/
│   └── processed/
├── frontend/
│   └── dashboard.py
├── src/
│   └── maintenance/
│       ├── data/
│       ├── features/
│       ├── models/
│       └── evaluation/
├── tests/
│   └── test_api.py
├── docs/
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

Navigate to the project:

```bash
cd Predictive-maintaince-system
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Backend

```bash
uvicorn backend.main:app --reload
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

## Run the Dashboard

Open another terminal:

```bash
streamlit run frontend/dashboard.py
```

## Run Tests

```bash
python -m pytest tests/ -v
```

## Future Improvements

- Advanced equipment health monitoring
- Improved anomaly detection precision
- Time-series failure prediction
- Model performance monitoring
- Interactive equipment health analytics
- Production deployment

## Author

Rida Sohail

Machine Learning and AI Enthusiast
