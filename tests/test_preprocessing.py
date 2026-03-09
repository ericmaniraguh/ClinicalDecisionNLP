# tests/test_preprocessing.py
from src.preprocessing import clean_medical_text

def test_clean_medical_text():
    text = "No fever, severe headache."
    cleaned = clean_medical_text(text)
    assert "fever" in cleaned
    assert "," not in cleaned