# InsureData360-AI

## Insurance Data Engineering, Analytics and Machine Learning

An educational portfolio project demonstrating insurance ETL pipelines, data quality validation, PostgreSQL analytics, machine learning-based claim anomaly screening, and a FastAPI prediction service.

## Features

- ETL pipelines for customer, policy, premium payment, and claim data.
- Data validation and rejected-record handling.
- Source-to-target reconciliation.
- PostgreSQL warehouse integration.
- Interactive Streamlit dashboard.
- Machine learning-based claim anomaly screening and severity modelling.
- FastAPI prediction endpoint.
- Automated tests using pytest.

## Technology Stack

Python, pandas, NumPy, PostgreSQL, Streamlit, Plotly, scikit-learn, XGBoost, FastAPI, and pytest.

## Run Locally

Start the dashboard:

```powershell
python -m streamlit run dashboard/app.py
```

Start the API in a separate terminal:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --reload
```

Run automated tests:

```powershell
python -m pytest -q
```

API documentation: http://127.0.0.1:8000/docs

## Security and Limitations

Use synthetic or otherwise authorised data only. Never publish credentials, `.env` files, private datasets, or production insurance information.

Anomaly scores are screening signals, not probabilities of fraud. This project is for educational demonstration and is not a production-ready insurance system.