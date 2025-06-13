# Resume Parser & Job Classifier

## 🌐 Visit the App Online

🔗 [Click here to try the Resume Parser](https://resume-parser-matcher.onrender.com)

This project is a web application that parses resumes (PDF/DOCX), extracts relevant information, and classifies the candidate into a suitable job category using Natural Language Processing (NLP) and Machine Learning (ML).

---

## 🔍 Features

-  Upload resumes in `.pdf` or `.docx` formats
-  Extract name, email, phone number, skills, education, experience
- � Detect coding profile links (GitHub, LeetCode, etc.)
-  Clean and preprocess resume content using NLP (NLTK & spaCy)
-  Transform resume text using TF-IDF vectorization
-  Predict job category using a trained Logistic Regression model
-  ATS (Applicant Tracking System) score calculator
-  Uses pre-trained model and vectorizer stored as `.pkl` files

---

## Tech Stack

- **Frontend:** Streamlit
- **Backend:** Python
- **ML/NLP Libraries:** Scikit-learn, NLTK, spaCy, Joblib, TfidfVectorizer
- **File Handling:** PyMuPDF (fitz), python-docx, textract

---

## 📁 File Structure

├── app.py                      # Main Streamlit app
├── utils.py                   # Text extraction and cleaning utilities
├── parser_functions.py        # Functions for extracting structured data
├── tfidf_vectorizer.pkl       # Trained TF-IDF vectorizer
├── logistic_regression_model.pkl  # Trained Logistic Regression model
├── Dataset_Resume.csv         # Resume dataset used for training
├── dataset.ipynb              # Notebook for data cleaning and model training
├── requirements.txt           # All dependencies
└── README.md                  # This file



