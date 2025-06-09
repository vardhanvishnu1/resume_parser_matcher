import os
import docx
import fitz 
import re
import tempfile
import textract
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import streamlit as st 

# Ensuring NLTK data is downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text += page.get_text()
        doc.close()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        text = ""
    return text.strip()

def extract_text(file_upload_object):
    ext = file_upload_object.name.split('.')[-1].lower()
    
    # Creating a temporary file to save the uploaded content
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
    file_upload_object.seek(0) 
    temp_file.write(file_upload_object.read())
    temp_file.close()
    temp_path = temp_file.name

    text = ""

    try:
        if ext == 'pdf':
            text = extract_text_from_pdf(temp_path)
            
        elif ext == 'docx':
            doc = docx.Document(temp_path)
            text = '\n'.join([p.text for p in doc.paragraphs])

        elif ext == 'txt':
            with open(temp_path, 'r', encoding='utf-8') as f:
                text = f.read()

        else:
            text = textract.process(temp_path).decode('utf-8')
            
    except Exception as e:
        st.error(f"Error processing {ext} file: {e}") 
        text = ""
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    return text.strip()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text) 
    words = text.split()
    return ' '.join([lemmatizer.lemmatize(w) for w in words if w not in stop_words and len(w) > 1])

def summarize_text(text, num_sentences=5):
    if not text.strip():
        return ""
    
    sents = sent_tokenize(text)
    if len(sents) <= num_sentences:
        return text
    
    word_freq = Counter([w.lower() for w in word_tokenize(text) if w.isalnum()]) 
    sent_scores = {}
    for i, s in enumerate(sents):
        sent_scores[i] = sum(word_freq[w.lower()] for w in word_tokenize(s) if w.isalnum())
        
    top_sents_indices = sorted(sent_scores, key=sent_scores.get, reverse=True)[:num_sentences]
    
    sorted_sents = sorted(top_sents_indices)
    return ' '.join([sents[i] for i in sorted_sents])