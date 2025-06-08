import re
import phonenumbers
import spacy
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    nlp = None 
SKILLS = ['python', 'c++', 'java', 'flask', 'streamlit', 'pandas', 'numpy',
          'scikit-learn', 'html', 'css', 'git', 'github', 'linux', 'windows',
          'oop', 'jupyter', 'machine learning', 'data analysis', 'sql', 'tableau',
          'power bi', 'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'tensorflow',
          'pytorch', 'r', 'javascript', 'react', 'angular', 'vue.js', 'node.js']

def extract_name(text):
    lines = text.split('\n')
    top_lines = [line.strip() for line in lines[:8] if line.strip()]

    for line in top_lines:
        words = line.split()
        if 2 <= len(words) <= 4 and all(word[0].isupper() or not word.isalpha() for word in words):
            if '@' not in line and not re.search(r'\d{5,}', line):
                return line.title()

    if nlp: 
        doc = nlp(text)
        potential_names = []
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                name_words = ent.text.split()
                if 2 <= len(name_words) <= 4 and all(word[0].isupper() for word in name_words if word.isalpha()):
                    potential_names.append(ent.text)

        if potential_names:
            potential_names.sort(key=lambda x: text.find(x)) 
            return potential_names[0].title()

    for line in top_lines: 
        if line.isupper() and 2 <= len(line.split()) <= 4:
            return line.title()

    return "Not found"


def extract_email(text):
    matches = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    return matches[0] if matches else "Not found"

def extract_phone(text):
    matches = re.findall(r'(?:\+?(\d{1,3})[-.\s]?)?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
    if matches:
        full_number = ""
        for match in matches:
            country_code = match[0]
            if country_code:
                full_number += "+" + country_code + " "
            full_number += "".join(filter(None, match[1:])) 
            return full_number
    
    try:
        for match in phonenumbers.PhoneNumberMatcher(text, "IN"): 
            return phonenumbers.format_number(match.number, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
    except Exception:
        pass 

    return "Not found"

def extract_skills(text):
    text = text.lower()
    found_skills = []
    for s in SKILLS:
        if re.search(r'\b' + re.escape(s) + r'\b', text):
            found_skills.append(s)
    return found_skills or ["Not found"]

def extract_sections(text, keywords):
    sections_content = {}
    current_section = None
    all_keywords = set(keywords + ["education", "academic qualifications", "academics"])
    section_patterns = {kw: re.compile(r'^\s*' + re.escape(kw) + r'(?:\s*[:.\-]?\s*[\r\n]|\s*$)', re.IGNORECASE | re.MULTILINE) for kw in all_keywords}

    lines = text.split('\n')
    for i, line in enumerate(lines):
        line_lower = line.strip().lower()
        found_section = False

        for keyword in all_keywords:
            if section_patterns[keyword].match(line_lower):
                current_section = keyword
                sections_content[current_section] = [] 
                found_section = True
                break
    
        if not found_section and current_section is not None:
            is_new_section_start = False
            for kw in all_keywords:
                if section_patterns[kw].match(line_lower):
                    is_new_section_start = True
                    break
           
            if not is_new_section_start and line.strip():
                sections_content[current_section].append(line.strip())
            elif not line.strip() and sections_content[current_section]:
                pass
            
    return {k: "\n".join(v) for k, v in sections_content.items()}


def extract_project_tech_stack(project_text, skills_list):
    if not project_text.strip():
        return "No Projects Done."
    
    text_lower = project_text.lower()
    found_tech = []
    
    for skill in skills_list:
        if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
            found_tech.append(skill)
            
    if found_tech:
        return  ", ".join(sorted(list(set(found_tech)))) + "."
    else:
        return "Tech stack not explicitly mentioned or recognized."

def format_achievements(achievements_text, max_bullets=5):
    if not achievements_text.strip():
        return "Not found"
    
    sents = sent_tokenize(achievements_text.strip())
    meaningful_sents = [s.strip() for s in sents if len(s.split()) > 3] 
    
    if not meaningful_sents:
        return "Not found"

    word_freq = Counter([w.lower() for w in word_tokenize(achievements_text) if w.isalnum()])
    
    sent_scores = {}
    for i, s in enumerate(meaningful_sents):
        score = sum(word_freq[w.lower()] for w in word_tokenize(s) if w.isalnum())
        sent_scores[i] = score
        
    top_sents_indices = sorted(sent_scores, key=sent_scores.get, reverse=True)[:min(len(meaningful_sents), max_bullets)]
    sorted_sents_indices = sorted(top_sents_indices)
    
    formatted_achievements_list = []
    for i in sorted_sents_indices:
        formatted_achievements_list.append(f"<li>{meaningful_sents[i]}</li>")
            
    if formatted_achievements_list:
        return "<ul>" + "".join(formatted_achievements_list) + "</ul>"
    else:
        return "Not found"

def get_achievements_projects(text):
    general_achievements_keywords = ["achievements", "awards", "honors", "accomplishments", "recognition"]
    projects_keywords = ["projects", "portfolio", "key projects", "major projects", "work experience", "experience"]
    all_sections = extract_sections(text, general_achievements_keywords + projects_keywords)

    extracted_general_achievements_text = ""
    for kw in general_achievements_keywords:
        if all_sections.get(kw):
            extracted_general_achievements_text = all_sections[kw]
            break

    extracted_projects_text = ""
    for kw in projects_keywords:
        if all_sections.get(kw):
            extracted_projects_text = all_sections[kw]
            break

    achievements_formatted = format_achievements(extracted_general_achievements_text) 
    projects_summary = extract_project_tech_stack(extracted_projects_text, SKILLS)
    
    return achievements_formatted, projects_summary 

def extract_cpi(text):
    education_keywords = ["education", "academic qualifications", "academics", "educational background"]
    sections = extract_sections(text, education_keywords)
    education_text = ""
    for kw in education_keywords:
        if sections.get(kw):
            education_text = sections[kw]
            break
    
    search_target_text = education_text if education_text.strip() else text

    if not search_target_text.strip():
        return "Not found"

    btech_keywords = [
        r'bachelor\s*of\s*technology', r'b\.?tech', r'bachelor\s*of\s*engineering', r'b\.?e\b'
    ]
    cpi_patterns = [
        re.compile(r'\b(?:CGPA|CPI|GPA|SGPA)\s*[:=\-]?\s*(\d{1,2}(?:\.\d{1,2})?)\s*(?:/\s*10)?', re.IGNORECASE),
        re.compile(r'\b(\d{1,2}(?:\.\d{1,2})?)\s*/\s*10\b', re.IGNORECASE),
        re.compile(r'\b(?:scored|aggregate|overall|obtained)\s*[:=\-]?\s*(\d{1,2}(?:\.\d{1,2})?)\b', re.IGNORECASE)
    ]
    
    percentage_patterns = [
        re.compile(r'\b(\d{2}(?:\.\d{1,2})?)\s*%', re.IGNORECASE),
        re.compile(r'\b(?:percentage|aggregate)\s*[:=\-]?\s*(\d{2}(?:\.\d{1,2})?)\b', re.IGNORECASE)
    ]

    def find_score_in_text(text_to_search):
        for pattern in cpi_patterns:
            matches = pattern.findall(text_to_search)
            for match in matches:
                try:
                    val = float(match)
                    if 0.0 <= val <= 10.0: 
                        return f"{val:.2f}/10"
                except (ValueError, TypeError):
                    continue

        for pattern in percentage_patterns:
            matches = pattern.findall(text_to_search)
            for match in matches:
                try:
                    val = float(match)
                    if 30.0 <= val <= 100.0:
                        cpi_equivalent = val / 10.0
                        return f"{cpi_equivalent:.2f}/10"
                except (ValueError, TypeError):
                    continue
        return None

    lines = search_target_text.split('\n')
    btech_context_text = ""
    for i, line in enumerate(lines):
        if any(re.search(kw, line, re.IGNORECASE) for kw in btech_keywords):
            context_end_index = min(len(lines), i + 5)
            btech_context_text = "\n".join(lines[i:context_end_index])
            found_score = find_score_in_text(btech_context_text)
            if found_score:
                return found_score
    found_score = find_score_in_text(search_target_text)
    if found_score:
        return found_score

    return "Not found"