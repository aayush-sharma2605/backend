from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import os
from typing import List, Dict, Any
import time

# Import our custom utilities
import utils
import numpy as np
from scipy.sparse import hstack

app = FastAPI(title="Cold Email Success Predictor API v3.14.0", version="3.14.0")

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

# Model Artifacts Path
MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
SCALER_PATH = "scaler.pkl"

# Global state for models
model = None
vectorizer = None
scaler = None

def load_models():
    """Load production ML artifacts into memory."""
    global model, vectorizer, scaler
    try:
        if all(os.path.exists(p) for p in [MODEL_PATH, VECTORIZER_PATH, SCALER_PATH]):
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            with open(VECTORIZER_PATH, 'rb') as f:
                vectorizer = pickle.load(f)
            with open(SCALER_PATH, 'rb') as f:
                scaler = pickle.load(f)
            print("🚀 v3.14.0 Production architecture loaded successfully.")
        else:
            print("⚠️ Warning: v3.14 artifacts missing. Run train_model.py first.")
    except Exception as e:
        print(f"❌ Error loading v3.14 models: {e}")

# Load models on startup
load_models()

# Pydantic Schemas
class EmailRequest(BaseModel):
    email: str

class PredictionResponse(BaseModel):
    score: int
    label: str
    confidence: float
    breakdown: Dict[str, int]
    insights: List[str]
    suggestions: List[str]

@app.get("/health")
def health_check():
    """System health monitoring."""
    uptime = time.time() - start_time
    return {
        "status": "active",
        "uptime_seconds": round(uptime, 2),
        "model_engine": "StackingEnsemble v3.14",
        "model_loaded": model is not None,
        "runtime": "Python 3.13 (Target 3.14 Ready)"
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_email(request: EmailRequest):
    """Linguistic and predictive analysis of email content."""
    if model is None or vectorizer is None or scaler is None:
        raise HTTPException(status_code=503, detail="Models not initialized.")
        
    text = request.email.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Content cannot be empty.")
        
    try:
        # Preprocessing & Feature Extraction
        # 1. TF-IDF
        tfidf_features = vectorizer.transform([text])
        
        # 2. Structured Features
        raw_features = utils.extract_features(text).reshape(1, -1)
        # 3. Scaling (CRITICAL for v3.14 architecture)
        scaled_features = scaler.transform(raw_features)
        
        # 4. Hybrid Combination
        combined_features = hstack([tfidf_features, scaled_features])
        
        # 5. Ensemble Prediction
        # predict_proba returns a list of probabilities [class_0, class_1]
        probs = model.predict_proba(combined_features)[0]
        model_proba = probs[1] if len(probs) > 1 else probs[0]
        
        # 6. Post-Processing & Insights via utils.py
        analysis = utils.get_full_analysis(text, model_proba)
        
        return PredictionResponse(**analysis)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")

@app.get("/")
def root():
    return {"message": "Email Intel v3.14.0 API", "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    # Listening on all interfaces for container/cloud compatibility
    uvicorn.run(app, host="0.0.0.0", port=10000)
