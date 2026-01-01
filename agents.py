#Agents 
import io # for BytesIO
import re # for regex operations
import json # for JSON parsing
from PyPDF2 import PdfReader # for PDF reading
from openai import OpenAI # for OpenAI API calls

def extract_text_from_file(uploaded_file): # uploaded_file can be a file-like object or bytes
        if hasattr(uploaded_file, "read"): # file-like object
            reader = PdfReader(uploaded_file) # read directly
        else:
            reader = PdfReader(io.BytesIO(uploaded_file))  # read from bytes
        text_pages = []
        for page in reader.pages: # iterate through pages
            text_pages.append(page.extract_text() or "") # extract text
        return "\n".join(text_pages) # join pages
    
#5.NLP-Powered Resume Parsing: o Extracts relevant sections of resumes using OpenAI’s language models to ensure accurate information retrieval.

def parse_resume_fields(text): # extract fields from resume text
    out = {} # output dictionary
    if not text: # empty text
        return out
    m = re.search(r"Name[:\-\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)", text) # regex for name
    if m:
        out["name"] = m.group(1).strip()
    em = re.search(r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})", text) # regex for email
    if em:
        out["email"] = em.group(1)
    ph = re.search(r"(\+?\d[\d\-\s]{7,}\d)", text) # regex for phone
    if ph:
        out["phone"] = ph.group(1)
    skills = re.findall(r"(?:Skills|Technical Skills|Skillset)[:\-\s]*([\w\W]{0,200})", text, flags=re.IGNORECASE) # regex for skills
    if skills:
        s = skills[0]
        s = re.sub(r"\s+", " ", s)
        out["skills"] = [t.strip() for t in re.split(r"[,\n;]+", s) if t.strip()]
    exp = re.findall(r"(?:Experience|Work Experience|Professional Experience)[:\-\s]*([\w\W]{0,600})", text, flags=re.IGNORECASE) # regex for experience
    if exp:
        out["experience"] = exp[0].strip()
    return out

#6 Extraction Job Requirement: Analyzes job descriptions to identify and categorize essential skills, qualifications, and experience needed for the role.

def safe_extract_json(text): # safely extract JSON from text
    if not text:
        return {}
    start = text.find("{") # find first {
    if start == -1:
        return {}
    end = text.rfind("}") # find last }
    if end == -1:
        return {}
    js = text[start:end+1] # extract substring
    try:
        return json.loads(js)
    except Exception:
        js = js.replace("\n"," ")
        js = js.replace(",}", "}")
        js = js.replace(",]", "]")
        try:
            return json.loads(js)
        except Exception:
            return {}
        
  #5 NLP-Powered Resume Parsing: o Extracts relevant sections of resumes using OpenAI’s language models to ensure accurate information retrieval.      

def parse_openai_content(resp): # parse content from OpenAI response
    try:
        first = resp.choices[0]
        if hasattr(first, "message"): # chat completion
            msg = first.message
            if isinstance(msg, dict):
                return msg.get("content","").strip()
            return getattr(msg, "content","").strip() or ""
        if hasattr(first, "text"): # text completion
            return getattr(first, "text","").strip()
        return str(first)
    except Exception:   
        try:
            return str(resp)
        except Exception:
            return ""
        
  # 7 Scoring Algorithm: Develops a scoring system that quantifies the alignment between resumes and job descriptions using metrics such as keyword match percentage, relevance of experience, and completeness      

def call_openai_scorer(api_key, model, jd_text, resume_text, max_tokens=800):  # call OpenAI to score resume against job description
    client = OpenAI(api_key=api_key)
    system = "You are an expert recruiter. Given a job description and a resume, return a JSON with fields candidate_name, match_score (0-100), years_experience_estimate, key_skills_matched, missing_skills, short_summary, fit_level. Return JSON only."
    user = f"JOB_DESCRIPTION:\n{jd_text}\n\nRESUME:\n{resume_text}\n\nReturn the JSON."
    try:
        resp = client.chat.completions.create(model=model, messages=[{"role":"system","content":system},{"role":"user","content":user}], temperature=0.0, max_tokens=max_tokens)
        content = parse_openai_content(resp)
        parsed = safe_extract_json(content)
        if parsed:
            return parsed
        resp2 = client.chat.completions.create(model=model, messages=[{"role":"system","content":system},{"role":"user","content":user}], temperature=0.0, max_tokens=max_tokens)
        content2 = parse_openai_content(resp2)
        parsed2 = safe_extract_json(content2)
        if parsed2:  
            return parsed2
        return {"match_score":0,"fit_level":"Low"}
    except Exception:
        return {"match_score":0,"fit_level":"Low"}
    
    #11 Model Evaluation: Test the model's effectiveness using a separate validation dataset and evaluate it based on accuracy, precision, recall, and user satisfaction with feedback

def call_openai_explainer(api_key, model, jd_text, resume_text, max_tokens=400): # call OpenAI to explain fit between resume and job description
    client = OpenAI(api_key=api_key)
    system = "You are an assistant that extracts highlights and a 1-2 sentence summary explaining fit. Return JSON with short_summary and highlights (array of strings). Return JSON only."
    user = f"JOB_DESCRIPTION:\n{jd_text}\n\nRESUME:\n{resume_text}\n\nReturn the JSON."
    try:
        resp = client.chat.completions.create(model=model, messages=[{"role":"system","content":system},{"role":"user","content":user}], temperature=0.0, max_tokens=max_tokens)
        content = parse_openai_content(resp)
        parsed = safe_extract_json(content)
        if parsed:
            return parsed
        resp2 = client.chat.completions.create(model=model, messages=[{"role":"system","content":system},{"role":"user","content":user}], temperature=0.0, max_tokens=max_tokens)
        content2 = parse_openai_content(resp2)
        parsed2 = safe_extract_json(content2)
        if parsed2:
            return parsed2
        return {"short_summary":"","highlights":[]}
    except Exception: # any error
        return {"short_summary":"","highlights":[]}
