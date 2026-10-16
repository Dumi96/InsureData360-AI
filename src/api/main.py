
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "claim_anomaly_model.joblib"

# Create the FastAPI application
app = FastAPI(
    title="InsureData360 AI API",
    description=(
        "API for insurance claim anomaly detection. "
        "Unusual claims should be reviewed; this is not proof of fraud."
    ),
    version="1.0.0",
)

# Load the saved model once when the API starts
model = None
model_load_error = None

try:
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
    else:
        model_load_error = f"Model file not found: {MODEL_PATH}"
except Exception as exc:
    model_load_error = str(exc)


# Define and validate incoming claim data
class ClaimInput(BaseModel):
    claim_amount: float = Field(ge=0, description="Claim amount")
    annual_premium: float = Field(ge=0, description="Annual policy premium")
    age: int = Field(ge=0, le=120, description="Customer age")
    product: str = Field(min_length=1, max_length=100)
    customer_type: str = Field(min_length=1, max_length=100)


@app.get("/")
def root():
    return {
        "message": "Welcome to InsureData360 AI API",
        "documentation": "/docs",
        "health_check": "/health",
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_file": str(MODEL_PATH),
    }


@app.post("/anomalies")
def detect_anomaly(claim: ClaimInput):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Anomaly detection model is unavailable: {model_load_error}",
        )

    # Prepare one claim in the same feature format used for training
    claim_data = pd.DataFrame(
        [
            {
                "claim_amount": claim.claim_amount,
                "annual_premium": claim.annual_premium,
                "age": claim.age,
                "product": claim.product.strip(),
                "customer_type": claim.customer_type.strip(),
            }
        ]
    )

    if not claim_data["product"].iloc[0]:
        raise HTTPException(status_code=422, detail="Product cannot be blank.")

    if not claim_data["customer_type"].iloc[0]:
        raise HTTPException(
            status_code=422,
            detail="Customer type cannot be blank.",
        )

    try:
        prediction = int(model.predict(claim_data)[0])
        anomaly_score = float(model.decision_function(claim_data)[0])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc

    is_unusual = prediction == -1

    return {
        "anomaly_label": (
            "Unusual - Review" if is_unusual else "Typical"
        ),
        "anomaly_score": round(anomaly_score, 6),
        "review_recommended": is_unusual,
        "interpretation": (
            "Flagged for investigation; this does not establish fraud."
            if is_unusual
            else "Not flagged by this model; this does not guarantee the claim is risk-free."
        ),
    }
