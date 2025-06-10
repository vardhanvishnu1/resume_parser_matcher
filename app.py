import streamlit as st
import os
import joblib
import spacy
import sys
import nltk

st.set_page_config(page_title="Resume Parser & Job Classifier with ATS Score", layout="wide")

nltk_data_path = "/tmp/nltk_data"
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

@st.cache_resource
def download_nltk_data():
    try:
        nltk.download('punkt_tab', download_dir=nltk_data_path, quiet=True)
    except Exception as e:
        st.error(f"Failed to download NLTK 'punkt_tab' resource: {e}")
        st.stop()

download_nltk_data()

spacy_data_path = "/tmp/spacy_data"
os.makedirs(spacy_data_path, exist_ok=True)
os.environ["SPACY_DATA"] = spacy_data_path

@st.cache_resource
def load_spacy_model():
    model_name = "en_core_web_sm"
    model_dir_v1 = os.path.join(spacy_data_path, model_name)
    model_dir_v2 = os.path.join(spacy_data_path, f"{model_name}-{spacy.__version__.split('.')[0]}")

    if not (os.path.exists(model_dir_v1) or os.path.exists(model_dir_v2)):
        st.info(f"SpaCy model '{model_name}' not found at {spacy_data_path}. Attempting download...")
        try:
            spacy.cli.download(model_name)
        except Exception as e:
            st.error(f"Failed to download spaCy model: {e}. Please ensure you have internet access and sufficient permissions.")
            st.stop()
    else:
        st.info(f"SpaCy model '{model_name}' found. Loading...")

    try:
        nlp = spacy.load(model_name)
        return nlp
    except Exception as e:
        st.error(f"Failed to load spaCy model '{model_name}': {e}")
        st.stop()

nlp = load_spacy_model()

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
    font-family: Arial, sans-serif;
}

h1, h2, h3, h4, h5, h6 {
    color: #0056b3;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
    padding-bottom: 0.2em;
    border-bottom: 1px solid #eee;
}

.stTextArea > label,
.stFileUploader > label {
    font-weight: bold;
    color: #0056b3;
    margin-bottom: 8px;
    display: block;
    font-size: 1.1em;
}

.stTextArea textarea {
    border: 1px solid #cccccc;
    border-radius: 8px;
    padding: 12px;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
    width: 100%;
    box-sizing: border-box;
}
.stTextArea textarea:focus {
    border-color: #007bff;
    outline: none;
    box-shadow: 0 0 0 0.2rem rgba(0,123,255,.25);
}

.stFileUploader {
    border: 2px dashed #cccccc;
    border-radius: 8px;
    padding: 25px;
    text-align: center;
    background-color: #f9f9f9;
    cursor: pointer;
    transition: border-color 0.3s ease, background-color 0.3s ease;
}
.stFileUploader:hover {
    border-color: #007bff;
    background-color: #f0f0f0;
}
.stFileUploader > div {
    font-size: 1.1em;
    color: #555;
}

.stButton > button {
    background-color: #007bff;
    color: white;
    border-radius: 8px;
    padding: 10px 25px;
    font-size: 17px;
    border: none;
    cursor: pointer;
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    transition: background-color 0.3s ease, transform 0.2s ease, box-shadow 0.3s ease;
    margin-top: 20px;
    font-weight: bold;
}
.stButton > button:hover {
    background-color: #0056b3;
    transform: translateY(-3px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.2);
}
.stButton > button:active {
    transform: translateY(0);
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

div[data-testid="stAlert"] {
    border-radius: 8px;
    padding: 15px 20px;
    margin-top: 20px;
    margin-bottom: 20px;
    font-size: 1rem;
    line-height: 1.5;
    box-shadow: 0 2px 4px rgba(0,0,0,0.08);
}
div[data-testid="stAlert"].info {
    background-color: #e0f2f7;
    color: #007bff;
    border-left: 6px solid #007bff;
}
div[data-testid="stAlert"].success {
    background-color: #e6ffe6;
    color: #28a745;
    border-left: 6px solid #28a745;
}
div[data-testid="stAlert"].warning {
    background-color: #fff3e0;
    color: #ffc107;
    border-left: 6px solid #ffc107;
}
div[data-testid="stAlert"].error {
    background-color: #ffe6e6;
    color: #dc3545;
    border-left: 6px solid #dc3545;
}

p {
    line-height: 1.7;
    margin-bottom: 1em;
}
strong {
    color: #333333;
}

.resume-summary-box {
    background-color:#F0F2F6;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #e0e0e0;
    margin-bottom: 20px;
}
.resume-summary-box h4 {
    color:#0056b3;
    border-bottom: 2px solid #0056b3;
    padding-bottom: 10px;
    margin-top: 0;
    margin-bottom: 15px;
}
.resume-summary-box p {
    margin-bottom: 8px;
}
.resume-summary-box p strong {
    min-width: 120px;
    display: inline-block;
}

.ats-score-display {
    color:#0056b3;
    font-size: 2em;
    margin-bottom: 20px;
    text-align: center;
}
.ats-score-display span {
    color:#28a745;
    font-weight: bold;
    font-size: 1.2em;
}

.job-role-info {
    background-color: #e0f2f7;
    padding: 15px;
    border-left: 5px solid #007bff;
    border-radius: 8px;
    margin-top: 15px;
}
.job-role-info strong {
    color: #0056b3;
    font-size: 1.1em;
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
    <div class="resume-summary-box">
        <h4>Resume Summary</h4>
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
                        st.markdown(f"<h3 class='ats-score-display'>🎯 ATS Match Score: <span>{ats_score:.2f}%</span></h3>", unsafe_allow_html=True)
                    else:
                        st.info("Paste a Job Description to get an ATS Match Score.")
                    st.markdown(generate_summary_html(name, email, phone, skills, cpi, projects_summary), unsafe_allow_html=True)
                    job, conf = classify_job(text, model, vectorizer)
                    st.subheader("Predicted Job Role")
                    st.markdown(f"<div class='job-role-info'><strong>{job}</strong></div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred during resume processing: {e}")
                st.exception(e)

st.markdown("---")
st.caption("Developed by Vardhan Bharathula")