import streamlit as st
import requests
import pandas as pd

# 1. SETUP & DATA MAPPING 
# (In a real app, you'd load this from your cleaned CSV)
@st.cache_data
def get_hospital_mappings():
    # Example mapping based on your unique values provided
    diag_map = {
        'Cardiac/Circulatory': ['Hypertension', 'Cardiac Problem', 'Heart Attack', 'Congestive Heart Failure', 'Angina'],
        'Mental': ['Eating Disorder', 'Depression - Severe', 'Anxiety', 'Bipolar Disorder', 'Schizophrenia'],
        'Infectious': ['Hepatitis', 'AIDS/HIV', 'Lyme Disease', 'Viral Infection', 'Pneumonia'],
        'Autism Spectrum': ['Autism-PDD-NOS', 'Asperger Syndrome', 'Autism'],
        'Prevention/Good Health': ['Vaccination', 'Check-up', 'Prevention/Good Health']
    }
    
    treat_map = {
        'Pharmacy/Prescription Drugs': ['Anti-virals', 'Cardiac Medications', 'Anti-Depressants', 'Antibiotics'],
        'Mental Health Treatment': ['Residential Treatment', 'Psychotherapy', 'Cognitive Therapy'],
        'Diagnostic Imaging': ['MRI', 'CT Scan', 'PET Scan', 'X-Ray', 'Mammography'],
        'Cardio Vascular': ['Pacemaker Insertion', 'Cardiac Valve Replacement', 'Aneurysm Repair']
    }
    
    return diag_map, treat_map

diag_hierarchy, treat_hierarchy = get_hospital_mappings()

# 2. PAGE CONFIGURATION
st.set_page_config(page_title="MedReviewAI Pro", page_icon="🏥", layout="wide")

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007BFF; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🏥 MedReviewAI Pro: Hospital Case Review")
st.info("Fill in the case details below. Sub-categories will update automatically based on the main category selected.")

# 3. STRUCTURED INPUT FORM
with st.form("clinical_case_form"):
    # First Row: High-level Categories
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    
    with row1_col1:
        diag_cat = st.selectbox("Diagnosis Category", options=list(diag_hierarchy.keys()))
        # DYNAMIC SUB-CATEGORY
        diag_sub = st.selectbox("Specific Diagnosis", options=diag_hierarchy[diag_cat])

    with row1_col2:
        treat_cat = st.selectbox("Treatment Category", options=list(treat_hierarchy.keys()))
        # DYNAMIC SUB-CATEGORY
        treat_sub = st.selectbox("Specific Treatment", options=treat_hierarchy[treat_cat])

    with row1_col3:
        case_type = st.selectbox("Review Type", ["Medical Necessity", "Experimental/Investigational", "Urgent Care"])
        report_year = st.number_input("Report Year", 2000, 2026, 2016)

    st.divider()

    # Second Row: Demographics
    row2_col1, row2_col2, row2_col3 = st.columns(3)
    
    with row2_col1:
        # Cleaned Age Range strings (replacing _ with -)
        age_options = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-64', '65+']
        age_range = st.selectbox("Patient Age Range", age_options)

    with row2_col2:
        gender = st.radio("Patient Gender", ["Male", "Female"], horizontal=True)

    with row2_col3:
        # Placeholder for potential Hospital ID or Reference ID
        ref_id = st.text_input("Reference ID (Optional)", value="REF-001")

    # Findings Section
    findings = st.text_area("Medical Findings & Clinical Summary", 
                            placeholder="Describe the clinical justification, symptoms, and results...",
                            height=150)

    # Submit Button
    submitted = st.form_submit_button("Analyze Case & Predict Determination")

# 4. API INTEGRATION
if submitted:
    if not findings.strip():
        st.warning("⚠️ Please provide medical findings to improve prediction accuracy.")
    else:
        with st.spinner("Model is analyzing clinical patterns..."):
            # Constructing the payload with optimized sub-categories
            payload = {
                "Report_Year": report_year,
                "Diagnosis_Category": diag_cat,
                "Diagnosis_Sub_Category": diag_sub,
                "Treatment_Category": treat_cat,
                "Treatment_Sub_Category": treat_sub,
                "Type": case_type,
                "Age_Range": age_range,
                "Patient_Gender": gender,
                "Findings": findings
            }

            try:
                # Replace with your actual deployed API URL
                URL = "http://localhost:8000/predict"
                response = requests.post(URL, json=payload, timeout=10)
                response.raise_for_status()
                data = response.json()

                # 5. DISPLAY RESULTS
                st.subheader("📊 Analysis Result")
                
                res_col1, res_col2 = st.columns([1, 2])
                
                with res_col1:
                    prediction = data.get("prediction", "Unknown")
                    if "Overturned" in prediction:
                        st.success(f"**Result:** {prediction}")
                    else:
                        st.error(f"**Result:** {prediction}")
                
                with res_col2:
                    st.write("**Confidence/Details:**")
                    st.json(data)

            except Exception as e:
                st.error("🚨 Connection Error: Could not reach the Prediction API.")
                st.info("Ensure your FastAPI server is running on localhost:8000")