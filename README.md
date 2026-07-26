# 📧 Cold Email Success Predictor API

A FastAPI-based Machine Learning API that predicts the success probability of cold emails using Natural Language Processing (NLP) and handcrafted linguistic features.

The API analyzes an email's content and returns:
- 📊 Success Score (0–100)
- 🎯 Prediction Label
- 📈 Confidence Score
- 🔍 Feature Breakdown
- 💡 Insights
- ✅ Improvement Suggestions

---

## 🚀 Features

- FastAPI REST API
- TF-IDF text vectorization
- Machine Learning Stacking Ensemble model
- Hybrid feature engineering
- Confidence prediction
- Email quality analysis
- Health monitoring endpoint
- Ready for Render deployment

---

## 🛠 Tech Stack

- Python
- FastAPI
- Scikit-Learn
- NumPy
- Pandas
- SciPy
- Pydantic
- Uvicorn

---

## 📂 Project Structure

```
backend-main/
│
├── app.py                # Main FastAPI application
├── utils.py              # Feature extraction & analysis
├── model.pkl             # Trained ML model
├── vectorizer.pkl        # TF-IDF Vectorizer
├── scaler.pkl            # Feature Scaler
├── requirements.txt      # Dependencies
├── render.yaml           # Render deployment configuration
├── runtime.txt           # Python runtime
└── .gitattributes
```

---

## ⚙ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/backend.git
cd backend
```

Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶ Running the Server

```bash
uvicorn app:app --reload
```

The server starts at

```
http://127.0.0.1:8000
```

---

## 📚 API Documentation

Swagger UI

```
http://127.0.0.1:8000/docs
```

ReDoc

```
http://127.0.0.1:8000/redoc
```

---

## ❤️ Health Check

### GET

```
/health
```

Example Response

```json
{
  "status": "active",
  "uptime_seconds": 120.4,
  "model_engine": "StackingEnsemble v3.14",
  "model_loaded": true,
  "runtime": "Python 3.13 (Target 3.14 Ready)"
}
```

---

## 📩 Predict Email

### POST

```
/predict
```

Request

```json
{
    "email":"Hi John, I noticed your company..."
}
```

Example Response

```json
{
  "score": 82,
  "label": "High Success",
  "confidence": 0.91,
  "breakdown": {
      "Personalization": 18,
      "Readability": 20,
      "CTA": 17,
      "Length": 15
  },
  "insights": [
      "Strong personalization detected.",
      "Clear call-to-action."
  ],
  "suggestions": [
      "Keep subject line concise.",
      "Avoid overly long paragraphs."
  ]
}
```

---

## 🧠 Machine Learning Pipeline

1. Receive email text
2. Clean input
3. Generate TF-IDF features
4. Extract handcrafted linguistic features
5. Scale structured features
6. Combine all features
7. Predict using the Stacking Ensemble model
8. Generate analysis and recommendations

---

## 🌐 Deployment

This project includes a `render.yaml` configuration for deployment on Render.

Deploy by connecting your GitHub repository to Render.

Build Command

```bash
pip install --upgrade pip && pip install -r requirements.txt
```

Start Command

```bash
uvicorn app:app --host 0.0.0.0 --port 10000
```

---

## 📦 Requirements

Major packages:

- FastAPI
- Uvicorn
- NumPy
- Pandas
- SciPy
- Scikit-Learn
- Pydantic

Install with

```bash
pip install -r requirements.txt
```

---

## 📌 Future Improvements

- JWT Authentication
- Email Subject Analysis
- Batch Predictions
- SHAP Explainability
- Docker Support
- CI/CD Pipeline
- Model Versioning
- Logging & Monitoring

---

## 👨‍💻 Author

Developed by **Aayush**

If you found this project useful, consider giving the repository a ⭐.
