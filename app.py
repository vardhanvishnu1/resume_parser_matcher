import streamlit as st
st.set_page_config(page_title="Resume Parser & Job Classifier with ATS Score", layout="wide")
import os
import joblib
import spacy
import sys 
spacy_data_path = "/tmp/spacy_data"
os.makedirs(spacy_data_path, exist_ok=True)
os.environ["SPACY_DATA"] = spacy_data_path
@st.cache_resource
def load_spacy_model():
    model_name = "en_core_web_sm"

    if not (os.path.exists(os.path.join(spacy_data_path, model_name)) or 
            os.path.exists(os.path.join(spacy_data_path, f"{model_name}-{spacy.__version__.split('.')[0]}"))):
       
        try:
            spacy.cli.download(model_name)
            
        except Exception as e:
            st.error(f"Failed to download spaCy model: {e}")
            st.stop()
    else:
        st.info(f"SpaCy model '{model_name}' found at {spacy_data_path}. Loading...")

    nlp = spacy.load(model_name)
    return nlp

nlp = load_spacy_model()


from utils import extract_text, preprocess
from utils import extract_text, preprocess
from parser_functions import (
    extract_name, extract_email, extract_phone, extract_skills,
    extract_sections, get_achievements_projects, extract_cpi
)
from ml_model import classify_job, calculate_ats_score

try:
    model = joblib.load('logistic_regression_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
except FileNotFoundError:
    st.error("Model or vectorizer files not found. Please ensure 'logistic_regression_model.pkl' and 'tfidf_vectorizer.pkl' are in the same directory as the script.")
    st.info("You'll need to train your machine learning model and save these files first. Refer to the project documentation for training instructions.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading the model/vectorizer: {e}")
    st.stop()

st.markdown("""
<style>
.stApp {
    background-color: #FFFFFF;
    color: #333333;
}
.st-emotion-cache-fis6y9, .st-emotion-cache-1wv7q08, .st-emotion-cache-1kenbb8 {
    background-color: #FFFFFF;
}

.st-emotion-cache-1pbsqon, .st-emotion-cache-1wmy9hq {
    color: #0056b3;
    font-weight: bold;
}

.stTextArea > label, .stFileUploader > label {
    font-weight: bold;
    color: #0056b3;
    margin-bottom: 5px;
    display: block;
}
.stTextArea textarea {
    border: 1px solid #cccccc;
    border-radius: 8px;
    padding: 10px;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
}
.stFileUploader {
    border: 2px dashed #cccccc;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    background-color: #f9f9f9;
}
.stFileUploader:hover {
    border-color: #007bff;
}

.stButton>button {
    background-color: #007bff;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 16px;
    border: none;
    cursor: pointer;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    transition: background-color 0.3s ease, transform 0.2s ease;
    margin-top: 15px;
}
.stButton>button:hover {
    background-color: #0056b3;
    transform: translateY(-2px);
}
.stButton>button:active {
    transform: translateY(0);
}

.stAlert {
    border-radius: 8px;
    padding: 15px 20px;
    margin-top: 15px;
    margin-bottom: 15px;
    font-size: 1rem;
}
.stAlert.info {
    background-color: #e0f2f7;
    color: #007bff;
    border-left: 5px solid #007bff;
}
.stAlert.success {
    background-color: #e6ffe6;
    color: #28a745;
    border-left: 5px solid #28a745;
}
.stAlert.warning {
    background-color: #fff3e0;
    color: #ffc107;
    border-left: 5px solid #ffc107;
}

p {
    line-height: 1.6;
}
strong {
    color: #333333;
}
</style>
""", unsafe_allow_html=True)

st.title("Resume Parser & Job Classifier with ATS Score")
st.markdown("Upload your resume and paste a job description to extract details, classify job role, and get an ATS match score.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.header("Resume Upload")
    file = st.file_uploader("Upload Resume File", type=["pdf", "docx", "txt"])
    if st.session_state.get('processed_file', None) is not None:
        if st.button("Clear Processed Data", key="clear_data"):
            st.session_state['processed_file'] = None
            st.rerun()

with col2:
    st.header("Job Description")
    job_description = st.text_area("Paste Job Description Here", height=300,
                                   help="Copy and paste the full job description text here for ATS matching. This helps the ATS score calculation.")

def generate_summary_html(name, email, phone, skills, cpi, projects):
    summary = f"""
    <div style="background-color:#F0F2F6; padding: 20px; border-radius: 10px; border: 1px solid #e0e0e0; margin-bottom: 20px;">
        <h4 style="color:#0056b3; border-bottom: 2px solid #0056b3; padding-bottom: 10px; margin-top: 0;">Resume Summary</h4>
        <p><strong>Name:</strong> {name}</p>
        <p><strong>Email:</strong> {email}</p>
        <p><strong>Phone:</strong> {phone}</p>
        <p><strong>Skills:</strong> {', '.join(skills)}</p>
        <p><strong>B.Tech Academic Score (CPI/CGPA/GPA):</strong> {cpi}</p>
        <p><strong>Tech stack used in Projects:</strong> {projects}</p>
    </div>
    """
    return summary

if st.button("Process Resume & Calculate ATS Score", key="process_button"):
    if file is None:
        st.warning("Please upload a resume file to process.")
    else:
        st.session_state['processed_file'] = file.name
        with st.spinner("Processing..."):
            try:
                text = extract_text(file)

                if not text:
                    st.warning("Failed to extract text from the resume. Please try a different file or format.")
                else:
                    name = extract_name(text)
                    email = extract_email(text)
                    phone = extract_phone(text)
                    skills = extract_skills(text)
                    cpi = extract_cpi(text)
                    achievements_formatted, projects_summary = get_achievements_projects(text)

                    st.markdown("---")
                    st.header("Analysis Results")

                    if job_description:
                        ats_score = calculate_ats_score(text, job_description, vectorizer)
                        if ats_score > 0:
                            st.markdown(f"<h3 style='color:#0056b3;'>🎯 ATS Match Score: <span style='color:#28a745;'>{ats_score}%</span></h3>", unsafe_allow_html=True)
                        else:
                            st.info("Paste a Job Description to get an ATS Match Score.")

                    st.markdown(generate_summary_html(name, email, phone, skills, cpi, projects_summary), unsafe_allow_html=True)
                    job, conf = classify_job(text, model, vectorizer)
                    st.subheader("Predicted Job Role")
                    st.info(f"**{job}**")
            except Exception as e:
                st.error(f"An error occurred during resume processing: {e}")
                st.exception(e)

st.markdown("---")
st.caption("Developed by **Vardhan Bharathula**")