import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from screening import (
    REQUIRED_SKILLS,
    GOOD_TO_HAVE_SKILLS,
    MIN_YEARS,
    load_job_description,
    load_resumes,
    match_skills,
    extract_years,
    bert_scores,
)

K = 5

jd = load_job_description()
resumes = load_resumes()
labels = pd.read_csv("data/labels.csv").set_index("candidate")["good_fit"]

# Sirf wahi resumes jinke labels hain (My_Resume skip ho jayega)
names = [n for n in resumes if n in labels.index]
texts = [resumes[n] for n in names]

# 1) TF-IDF
matrix = TfidfVectorizer(stop_words="english").fit_transform([jd] + texts)
tfidf = cosine_similarity(matrix[0:1], matrix[1:]).flatten()

# 2) BERT
bert = bert_scores(jd, texts)

# 3) Hybrid
bert_norm = (bert - bert.min()) / (bert.max() - bert.min())
hybrid = []
for text, bn in zip(texts, bert_norm):
    req_matched, _ = match_skills(text, REQUIRED_SKILLS)
    nice_matched, _ = match_skills(text, GOOD_TO_HAVE_SKILLS)
    skill = 0.7 * len(req_matched) / len(REQUIRED_SKILLS) + 0.3 * len(nice_matched) / len(GOOD_TO_HAVE_SKILLS)
    exp = min(extract_years(text) / MIN_YEARS, 1)
    hybrid.append(0.35 * bn + 0.45 * skill + 0.20 * exp)


def precision_at_k(scores):
    order = sorted(range(len(names)), key=lambda i: scores[i], reverse=True)[:K]
    top = [names[i] for i in order]
    hits = sum(labels[n] for n in top)
    return hits / K, top


print(f"Precision@{K} (labels wale {len(names)} candidates par)")
print("-" * 50)
for model_name, scores in [("TF-IDF", tfidf), ("BERT", bert), ("Hybrid", hybrid)]:
    p, top = precision_at_k(scores)
    print(f"{model_name}: {p:.2f}")
    for n in top:
        print("   ", n, "(fit)" if labels[n] == 1 else "(not fit)")