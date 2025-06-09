from sklearn.metrics.pairwise import cosine_similarity
from utils import preprocess

def classify_job(text, model, vectorizer):
    clean_text = preprocess(text)
    if not clean_text.strip():
        return "Unknown", 0.0

    features = vectorizer.transform([clean_text])
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0]
    
    predicted_class_idx = model.classes_.tolist().index(pred)
    confidence = prob[predicted_class_idx]
    
    return pred, confidence

def calculate_ats_score(resume_text, job_description_text, vectorizer):
    if not resume_text or not job_description_text:
        return 0.0

    processed_resume = preprocess(resume_text)
    processed_jd = preprocess(job_description_text)
    
    if not processed_resume or not processed_jd:
        return 0.0

    resume_vector = vectorizer.transform([processed_resume])
    jd_vector = vectorizer.transform([processed_jd])

    similarity = cosine_similarity(resume_vector, jd_vector)[0][0]

    ats_score = similarity * 100
    return round(ats_score, 2)