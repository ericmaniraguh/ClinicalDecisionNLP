import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from webapp.app_api import app

client = TestClient(app)

class TestAppAPI(unittest.TestCase):

    def setUp(self):
        self.rf_patcher = patch("webapp.app_api.rf_model")
        self.mock_rf_model = self.rf_patcher.start()
        self.mock_rf_model.predict.return_value = ["Upheld"]
        self.mock_rf_model.predict_proba.return_value = [[0.2, 0.8]]
        self.mock_rf_model.classes_ = ["Upheld", "Overturned"]

        self.vec_patcher = patch("webapp.app_api.tfidf_vectorizer")
        self.mock_vectorizer = self.vec_patcher.start()
        
        # Setup the mock vectorizer transform output
        mock_transform_result = MagicMock()
        mock_transform_result.sum.return_value = 1
        self.mock_vectorizer.transform.return_value = mock_transform_result

    def tearDown(self):
        self.rf_patcher.stop()
        self.vec_patcher.stop()

    def test_root_endpoint(self):
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Medical Decision Support API", response.json()["message"])

    def test_predict_endpoint(self):
        payload = {
            "Report_Year": 2024,
            "Diagnosis_Category": "Cardiac/Circulatory",
            "Diagnosis_Sub_Category": "Hypertension",
            "Treatment_Category": "Pharmacy/Prescription Drugs",
            "Treatment_Sub_Category": "Anti-virals",
            "Type": "Medical Necessity",
            "Age_Range": "21-30",
            "Patient_Gender": "Male",
            "Findings": "Patient shows high blood pressure and chest pain."
        }
        response = client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200)
        json_resp = response.json()
        self.assertIn("prediction", json_resp, f"Response missing 'prediction' key. Full response: {json_resp}")
        self.assertIn(json_resp["prediction"], ["Upheld", "Overturned"])