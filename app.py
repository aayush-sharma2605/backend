from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import os
from typing import List, Dict
import time

# Import our custom utilities
import utils
import numpy as np
from scipy.sparse import hstack

app = FastAPI(title="Cold Email Success Predictor")

# Start time for health check uptime
start_time = time.time()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model and vectorizer
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

model = None
vectorizer = None

def load_models():
    global model, vectorizer
    try:
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            with open(VECTORIZER_PATH, 'rb') as f:
                vectorizer = pickle.load(f)
            print("Production models loaded successfully.")
        else:
            print("Models not found. Please run train_model.py first.")
    except Exception as e:
        print(f"Error loading models: {e}")

load_models()

# Pydantic Schemas for API
class EmailRequest(BaseModel):
    email: str

class Breakdown(BaseModel):
    personalization: int
    clarity: int
    relevance: int
    tone: int
    length: int

class PredictionResponse(BaseModel):
    score: int
    label: str
    confidence: float
    breakdown: Dict[str, int]
    insights: List[str]
    suggestions: List[str]

@app.get("/health")
def health_check():
    """Returns the API health and uptime information."""
    uptime = time.time() - start_time
    return {
        "status": "running",
        "uptime_seconds": round(uptime, 2),
        "model_loaded": model is not None,
        "api_version": "3.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_email(request: EmailRequest):
    """Processes email text and returns comprehensive success metrics."""
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=503, 
            detail="Service Unavailable: Models are not initialized."
        )
        
    text = request.email.strip()
    if not text:
        raise HTTPException(
            status_code=400, 
            detail="Bad Request: Email content cannot be empty."
        )
        
    try:
        # 1. Feature Transformation (Hybrid)
        tfidf_features = vectorizer.transform([text])
        structured_features = utils.extract_features(text).reshape(1, -1)
        
        # Combine TF-IDF and Structured features
        combined_features = hstack([tfidf_features, structured_features])
        
        # 2. Probability Prediction (Positive class)
        proba_list = model.predict_proba(combined_features)[0]
        # In sklearn and xgboost, proba_list[1] represents positive class (1)
        # If it's a binary classifier, but check if it's 1-dimensional for some models
        model_proba = proba_list[1] if len(proba_list) > 1 else proba_list[0]
        
        # 3. Comprehensive Analysis via utils.py (includes post-processing)
        analysis = utils.get_full_analysis(text, model_proba)
        
        return PredictionResponse(**analysis)
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Internal Server Error: {str(e)}"
        )

@app.get("/")
def home():
    """Simple root info page."""
    return {
        "message": "Cold Email Success Predictor API v3.0",
        "documentation": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    # Using 0.0.0.0 to be accessible in various environments (local, Docker, Render)
    uvicorn.run(app, host="0.0.0.0", port=10000)
