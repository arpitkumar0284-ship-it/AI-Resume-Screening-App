import re
from pathlib import Path
import pdfplumber
from docx import Document

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
RESUME_DIR = DATA_DIR / "resumes"
JD_PATH = DATA_DIR / "job_description.txt"
OUTPUT_DIR = Path("output")

MIN_YEARS = 2  # JD ke hisaab se minimum experience
TOP_N = 5      # kitne candidates shortlist karne hain

REQUIRED_SKILLS = {
    "python": ["python"],
    "machine learning": ["machine learning", "scikit-learn", "xgboost", "deep learning", "ml"],
    "nlp": ["nlp", "natural language processing", "text classification",
            "named entity recognition", "spacy", "transformers"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "pytorch/tensorflow": ["pytorch", "tensorflow"],
    "sql": ["sql", "mysql"],
    "docker": ["docker"],
}

GOOD_TO_HAVE_SKILLS = {
    "transformers": ["transformers", "bert"],
    "sentence-bert": ["sentence-bert", "sbert", "sentence bert"],
    "aws": ["aws", "sagemaker"],
    "mlops": ["mlops", "mlflow", "airflow"],
    "fastapi": ["fastapi"],
}


def load_job_description():
    return JD_PATH.read_text(encoding="utf-8")


def read_file(file):
    ext = file.suffix.lower()
    if ext == ".txt":
        return file.read_text(encoding="utf-8")
    if ext == ".pdf":
        with pdfplumber.open(file) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    if ext == ".docx":
        doc = Document(file)
        return "\n".join(p.text for p in doc.paragraphs)
    return ""


def load_resumes():
    resumes = {}
    for file in sorted(RESUME_DIR.iterdir()):
        if file.suffix.lower() in (".txt", ".pdf", ".docx"):
            text = read_file(file)
            if text.strip():
                resumes[file.stem] = text
    return resumes


def contains_term(text, term):
    pattern = r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])"
    return re.search(pattern, text.lower()) is not None


def match_skills(text, skill_map):
    matched, missing = [], []
    for skill, synonyms in skill_map.items():
        if any(contains_term(text, s) for s in synonyms):
            matched.append(skill)
        else:
            missing.append(skill)
    return matched, missing


def extract_years(text):
    match = re.search(r"(\d+)\+?\s*years?", text.lower())
    return int(match.group(1)) if match else 0


def bert_scores(jd_text, texts):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    jd_embedding = model.encode([jd_text])
    resume_embeddings = model.encode(texts)
    return cosine_similarity(jd_embedding, resume_embeddings).flatten()


if __name__ == "__main__":
    jd = load_job_description()
    resumes = load_resumes()
    names = list(resumes.keys())
    texts = list(resumes.values())

    bert = bert_scores(jd, texts)
    bert_norm = (bert - bert.min()) / (bert.max() - bert.min())

    skill_scores, missing_required, years_list = [], [], []
    for text in texts:
        req_matched, req_missing = match_skills(text, REQUIRED_SKILLS)
        nice_matched, _ = match_skills(text, GOOD_TO_HAVE_SKILLS)
        req_frac = len(req_matched) / len(REQUIRED_SKILLS)
        nice_frac = len(nice_matched) / len(GOOD_TO_HAVE_SKILLS)
        skill_scores.append(0.7 * req_frac + 0.3 * nice_frac)
        missing_required.append(", ".join(req_missing) if req_missing else "-")
        years_list.append(extract_years(text))

    exp_scores = [min(y / MIN_YEARS, 1) for y in years_list]

    result = pd.DataFrame({
        "candidate": names,
        "bert_score": bert,
        "skill_score": skill_scores,
        "exp_years": years_list,
        "exp_score": exp_scores,
        "missing_required": missing_required,
    })
    result["final_score"] = (
        0.35 * bert_norm + 0.45 * result["skill_score"] + 0.20 * result["exp_score"]
    )
    result = result.sort_values("final_score", ascending=False).reset_index(drop=True)
    result.index = result.index + 1  # rank 1 se shuru

    result["status"] = [
        "Shortlisted" if rank <= TOP_N else "Not shortlisted" for rank in result.index
    ]

    # Result CSV mein save karo
    OUTPUT_DIR.mkdir(exist_ok=True)
    result.round(3).to_csv(OUTPUT_DIR / "results.csv", index_label="rank")

    print("Final Ranking:")
    print(result[["candidate", "final_score", "status"]].round(3).to_string())
    print()
    print("Result saved to output\\results.csv")