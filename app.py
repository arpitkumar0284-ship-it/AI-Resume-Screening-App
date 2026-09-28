import numpy as np
import pandas as pd
import pdfplumber
import streamlit as st
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from screening import contains_term, extract_years


@st.cache_resource
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def read_upload(f):
    name = f.name.lower()
    if name.endswith(".txt"):
        return f.read().decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        with pdfplumber.open(f) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    if name.endswith(".docx"):
        doc = Document(f)
        return "\n".join(p.text for p in doc.paragraphs)
    return ""


def parse_skills(text):
    # "pytorch|tensorflow" ka matlab: dono mein se koi ek mile to chalega
    skills = {}
    for item in text.split(","):
        item = item.strip().lower()
        if item:
            synonyms = [s.strip() for s in item.split("|") if s.strip()]
            skills[synonyms[0]] = synonyms
    return skills


def match_skills(text, skill_map):
    matched, missing = [], []
    for skill, synonyms in skill_map.items():
        if any(contains_term(text, s) for s in synonyms):
            matched.append(skill)
        else:
            missing.append(skill)
    return matched, missing


st.set_page_config(page_title="Resume Screening", layout="wide")
st.title("AI Based Resume Screening and Candidate Shortlisting")

jd_text = st.text_area("Job Description", height=200, placeholder="Job description yahan paste karo...")

col1, col2 = st.columns(2)
with col1:
    req_input = st.text_input(
        "Required skills (comma se alag karo)",
        "python, machine learning, nlp, scikit-learn, pytorch|tensorflow, sql, docker",
    )
    nice_input = st.text_input(
        "Good-to-have skills",
        "transformers|bert, sentence-bert, aws, mlops|mlflow, fastapi",
    )
with col2:
    min_years = st.number_input("Minimum experience (years)", min_value=0, value=2)
    top_n = st.number_input("Kitne candidates shortlist karne hain", min_value=1, value=5)

files = st.file_uploader(
    "Resumes upload karo (PDF, DOCX ya TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True,
)

if st.button("Screen Resumes"):
    if not jd_text.strip():
        st.warning("Pehle job description paste karo.")
    elif not files:
        st.warning("Kam se kam ek resume upload karo.")
    else:
        names, texts = [], []
        for f in files:
            text = read_upload(f)
            if text.strip():
                names.append(f.name)
                texts.append(text)
            else:
                st.warning(f"{f.name} se text nahi nikla, isko skip kar diya.")

        if texts:
            required = parse_skills(req_input)
            nice = parse_skills(nice_input)

            with st.spinner("Resumes analyse ho rahe hain..."):
                model = get_model()
                jd_emb = model.encode([jd_text])
                res_emb = model.encode(texts)
                bert = cosine_similarity(jd_emb, res_emb).flatten()

            spread = bert.max() - bert.min()
            bert_norm = (bert - bert.min()) / spread if spread > 0 else bert

            rows = []
            for name, text, b, bn in zip(names, texts, bert, bert_norm):
                req_matched, req_missing = match_skills(text, required)
                nice_matched, _ = match_skills(text, nice)
                req_frac = len(req_matched) / len(required) if required else 0
                nice_frac = len(nice_matched) / len(nice) if nice else 0
                skill_score = 0.7 * req_frac + 0.3 * nice_frac
                years = extract_years(text)
                exp_score = min(years / min_years, 1) if min_years > 0 else 1
                final = 0.35 * bn + 0.45 * skill_score + 0.20 * exp_score
                rows.append({
                    "candidate": name,
                    "final_score": final,
                    "bert_score": b,
                    "skill_score": skill_score,
                    "exp_years": years,
                    "matched_skills": ", ".join(req_matched) or "-",
                    "missing_skills": ", ".join(req_missing) or "-",
                })

            result = pd.DataFrame(rows).sort_values("final_score", ascending=False)
            result = result.reset_index(drop=True)
            result.index = result.index + 1
            result["status"] = [
                "Shortlisted" if r <= top_n else "Not shortlisted" for r in result.index
            ]

            st.subheader("Result")
            st.dataframe(result.round(3), use_container_width=True)
            st.download_button(
                "Result CSV download karo",
                result.round(3).to_csv(index_label="rank"),
                file_name="results.csv",
                mime="text/csv",
            )