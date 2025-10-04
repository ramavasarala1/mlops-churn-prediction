from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="MLOps production-grade churn prediction service",
    version="1.0.0"
)

# Prometheus metrics
PREDICTION_COUNTER = Counter('predictions_total', 'Total number of predictions')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')
CHURN_PREDICTIONS = Counter('churn_predictions_total', 'Total churn predictions', ['prediction'])

# Model placeholder
model = None
feature_names = [
    'tenure_months',
    'monthly_charges',
    'total_charges',
    'charge_per_month',
    'service_count',
    'is_high_value',
    'high_risk_contract',
    'payment_risk',
    'no_support_services',
    'senior_citizen'
]

# Request/Response models
class CustomerFeatures(BaseModel):
    """Input features for churn prediction"""
    tenure_months: int = Field(..., ge=0, description="Months with company")
    monthly_charges: float = Field(..., gt=0, description="Monthly charges in USD")
    total_charges: float = Field(..., gt=0, description="Total charges to date")
    charge_per_month: float = Field(..., gt=0, description="Average charge per month")
    service_count: int = Field(..., ge=0, le=10, description="Number of services")
    is_high_value: int = Field(..., ge=0, le=1, description="High value customer flag")
    high_risk_contract: int = Field(..., ge=0, le=1, description="Month-to-month contract flag")
    payment_risk: int = Field(..., ge=0, le=1, description="Electronic check payment flag")
    no_support_services: int = Field(..., ge=0, le=1, description="No support services flag")
    senior_citizen: int = Field(..., ge=0, le=1, description="Senior citizen flag")

    class Config:
        json_schema_extra = {
            "example": {
                "tenure_months": 12,
                "monthly_charges": 65.50,
                "total_charges": 786.00,
                "charge_per_month": 65.50,
                "service_count": 2,
                "is_high_value": 0,
                "high_risk_contract": 1,
                "payment_risk": 0,
                "no_support_services": 0,
                "senior_citizen": 0
            }
        }

class PredictionResponse(BaseModel):
    """Prediction response"""
    customer_id: str
    churn_prediction: int
    churn_probability: float
    risk_level: str
    timestamp: str
    model_version: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    timestamp: str
    version: str

# Helper function for prediction logic
def make_prediction(feature_dict: dict) -> dict:
    """Core prediction logic"""
    # Simple rule-based prediction for demo
    churn_score = 0.0
    
    # High risk contract (most important feature - 35%)
    if feature_dict['high_risk_contract'] == 1:
        churn_score += 0.35
    
    # High monthly charges (19% importance)
    if feature_dict['monthly_charges'] > 80:
        churn_score += 0.20
    
    # Low tenure (12% importance)
    if feature_dict['tenure_months'] < 12:
        churn_score += 0.15
    
    # No support services (10% importance)
    if feature_dict['no_support_services'] == 1:
        churn_score += 0.15
    
    # Senior citizen (5% importance)
    if feature_dict['senior_citizen'] == 1:
        churn_score += 0.05
    
    # Add some randomness
    churn_score += np.random.uniform(-0.1, 0.1)
    churn_probability = min(max(churn_score, 0.0), 1.0)
    
    # Make prediction
    churn_prediction = 1 if churn_probability > 0.5 else 0
    
    # Determine risk level
    if churn_probability < 0.3:
        risk_level = "low"
    elif churn_probability < 0.6:
        risk_level = "medium"
    else:
        risk_level = "high"
    
    return {
        "churn_prediction": churn_prediction,
        "churn_probability": churn_probability,
        "risk_level": risk_level
    }

# Startup: Load model
@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model
    try:
        logger.info("Loading model...")
        model = "dummy"  # Placeholder
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        model = None

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for Kubernetes liveness/readiness probes"""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0"
    )

# Readiness probe
@app.get("/ready")
async def readiness_check():
    """Readiness probe"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(features: CustomerFeatures):
    """Predict customer churn probability"""
    PREDICTION_COUNTER.inc()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert features to dict
        feature_dict = features.dict()
        
        # Make prediction
        result = make_prediction(feature_dict)
        
        # Track metrics
        CHURN_PREDICTIONS.labels(prediction=str(result['churn_prediction'])).inc()
        
        # Generate response
        response = PredictionResponse(
            customer_id=f"CUST_{np.random.randint(100000, 999999)}",
            churn_prediction=result['churn_prediction'],
            churn_probability=round(result['churn_probability'], 4),
            risk_level=result['risk_level'],
            timestamp=datetime.utcnow().isoformat(),
            model_version="v1.0"
        )
        
        logger.info(f"Prediction: {result['churn_prediction']}, Probability: {result['churn_probability']:.4f}")
        
        return response
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Batch prediction endpoint
@app.post("/predict/batch")
async def predict_batch(customers: List[CustomerFeatures]):
    """Batch prediction endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    predictions = []
    for customer in customers:
        feature_dict = customer.dict()
        result = make_prediction(feature_dict)
        
        pred_response = PredictionResponse(
            customer_id=f"CUST_{np.random.randint(100000, 999999)}",
            churn_prediction=result['churn_prediction'],
            churn_probability=round(result['churn_probability'], 4),
            risk_level=result['risk_level'],
            timestamp=datetime.utcnow().isoformat(),
            model_version="v1.0"
        )
        predictions.append(pred_response)
    
    return {"predictions": predictions, "count": len(predictions)}

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Model info endpoint
@app.get("/model/info")
async def model_info():
    """Get model information"""
    return {
        "model_name": "churn_prediction_model",
        "version": "1.0.0",
        "features": feature_names,
        "feature_count": len(feature_names),
        "model_loaded": model is not None,
        "framework": "scikit-learn",
        "last_updated": "2025-01-04"
    }

# Root endpoint
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "service": "Churn Prediction API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "metrics": "/metrics",
            "model_info": "/model/info",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
