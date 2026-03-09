from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import os
import sys
import numpy as np

# Add path for preprocessing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import clean_medical_text

app = FastAPI()

# Enable CORS so Streamlit can talk to FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and vectorizer
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")

rf_model = joblib.load(MODEL_PATH)
tfidf_vectorizer = joblib.load(VECTORIZER_PATH)

# MATCHING SCHEMA: This must match the Streamlit payload exactly
class MedicalCase(BaseModel):
    Report_Year: int
    Diagnosis_Category: str
    Diagnosis_Sub_Category: str
    Treatment_Category: str
    Treatment_Sub_Category: str
    Type: str
    Age_Range: str
    Patient_Gender: str
    Findings: str

@app.get("/")
def read_root():
    return {"message": "Medical Decision Support API is running."}

@app.post("/predict")
def predict_outcome(case: MedicalCase):
    try:
        # Step 1: Clean text
        cleaned_text = clean_medical_text(case.Findings)

        # Step 2: Basic validation
        if len(cleaned_text.split()) < 3:
            return {"error": "Text too short. Please provide detailed findings."}

        # Step 3: TF-IDF transform
        X_new = tfidf_vectorizer.transform([cleaned_text])

        # Step 4: Prediction Logic
        if X_new.sum() == 0:
            # Fallback if no words match the vocabulary
            prediction = rf_model.classes_[0] 
            confidence = 50.0
        else:
            prediction = rf_model.predict(X_new)[0]
            probabilities = rf_model.predict_proba(X_new)[0]
            class_labels = list(rf_model.classes_)
            class_index = class_labels.index(prediction)
            confidence = probabilities[class_index] * 100

        # Map numeric prediction to text if necessary
        # result_text = "Overturned" if prediction == 1 else "Upheld"

        return {
            "prediction": str(prediction),
            "confidence": f"{float(confidence):.2f}%",
            "received_data": {"category": case.Diagnosis_Category, "year": case.Report_Year}
        }

    except Exception as e:
        return {"error": "Prediction failed", "details": str(e)}