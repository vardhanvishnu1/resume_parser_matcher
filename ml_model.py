from sklearn.metrics.pairwise import cosine_similarity
from utils import preprocess 

def classify_job(text, model, vectorizer):
    clean_text = preprocess(text)
    if not clean_text.strip():
        return "Unknown", 0.0

    features = vectorizer.transform([clean_text])
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    return pred

def calculate_ats_score(resume_text, job_description_text, vectorizer):
    if not resume_text or not job_description_text:
        return 0.0

    processed_resume = preprocess(resume_text)
    processed_jd = preprocess(job_description_text)
    
    if not processed_resume or not processed_jd:
        return 0.0

    # Transforming preprocessed texts into TF-IDF vectors
    resume_vector = vectorizer.transform([processed_resume])
    jd_vector = vectorizer.transform([processed_jd])

    # Calculating cosine similarity
    similarity = cosine_similarity(resume_vector, jd_vector)[0][0]

    # Convert similarity to a percentage score
    ats_score = similarity * 100
    return round(ats_score, 2)