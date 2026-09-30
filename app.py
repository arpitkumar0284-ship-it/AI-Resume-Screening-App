import pandas as pd
import pdfplumber
import streamlit as st
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from screening import contains_term, extract_years

st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-header {
        padding: 24px 28px;
        border-radius: 16px;
        background: linear-gradient(135deg, #4F8BF9 0%, #1B1F27 100%);
        margin-bottom: 24px;
    }
    .main-header h1 { margin: 0; font-size: 32px; }
    .main-header p { margin: 4px 0 0 0; opacity: 0.85; }

    .candidate-card {
        padding: 18px 22px;
        border-radius: 14px;
        margin-bottom: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        border-left: 6px solid #ccc;
        background-color: rgba(127,127,127,0.06);
    }
    .card-green { border-left-color: #2ecc71; }
    .card-yellow { border-left-color: #f1c40f; }

    .rank-tag {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        background-color: #4F8BF9;
        color: white;
        font-size: 12px;
        font-weight: 600;
        margin-right: 8px;
    }
    .status-pill {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
    }
    .pill-shortlisted { background-color: #2ecc7133; color: #2ecc71; }
    .pill-notshortlisted { background-color: #f1c40f33; color: #f1c40f; }

    .skill-chip {
        display: inline-block;
        padding: 2px 10px;
        margin: 2px;
        border-radius: 20px;
        font-size: 12px;
        background-color: rgba(127,127,127,0.15);
    }
    .chip-match { background-color: #2ecc7122; color: #2ecc71; }
    .chip-missing { background-color: #e74c3c22; color: #e74c3c; }

    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


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


def skill_chips(items, kind):
    if not items:
        return "<span style='opacity:0.6'>-</span>"
    cls = "chip-match" if kind == "match" else "chip-missing"
    return "".join(f"<span class='skill-chip {cls}'>{s}</span>" for s in items)


# ---------- Header ----------
st.markdown(
    """
    <div class="main-header">
        <h1>🧠 AI Resume Screening & Shortlisting</h1>
        <p>Semantic matching (Sentence-BERT) + Skill matching + Experience scoring</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Screening Setup")

    with st.expander("1️⃣ Job Description", expanded=True):
        jd_text = st.text_area("Paste JD here", height=160, label_visibility="collapsed")

    with st.expander("2️⃣ Skills", expanded=True):
        req_input = st.text_input(
            "Required skills",
            "python, machine learning, nlp, scikit-learn, pytorch|tensorflow, sql, docker",
        )
        nice_input = st.text_input(
            "Good-to-have skills",
            "transformers|bert, sentence-bert, aws, mlops|mlflow, fastapi",
        )

    with st.expander("3️⃣ Criteria", expanded=True):
        min_years = st.number_input("Minimum experience (years)", min_value=0, value=2)
        top_n = st.number_input("Shortlist size", min_value=1, value=5)

    with st.expander("4️⃣ Resumes", expanded=True):
        files = st.file_uploader(
            "Upload resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True,
            label_visibility="collapsed",
        )

    run = st.button("🚀 Screen Resumes", use_container_width=True, type="primary")

# ---------- Run screening ----------
if run:
    if not jd_text.strip():
        st.warning("Please add a job description in the sidebar first.")
    elif not files:
        st.warning("Please upload at least one resume.")
    else:
        names, texts = [], []
        for f in files:
            text = read_upload(f)
            if text.strip():
                names.append(f.name)
                texts.append(text)

        required = parse_skills(req_input)
        nice = parse_skills(nice_input)

        with st.spinner("Analyzing resumes..."):
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
                "matched_skills": req_matched,
                "missing_skills": req_missing,
                "resume_text": text,
            })

        result = pd.DataFrame(rows).sort_values("final_score", ascending=False)
        result = result.reset_index(drop=True)
        result.index = result.index + 1
        result["status"] = [
            "Shortlisted" if r <= top_n else "Not shortlisted" for r in result.index
        ]
        st.session_state["result"] = result

# ---------- Display results ----------
if "result" in st.session_state:
    result = st.session_state["result"]
    n_shortlisted = (result["status"] == "Shortlisted").sum()

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Candidates", len(result))
    k2.metric("Shortlisted", int(n_shortlisted))
    k3.metric("Avg. Score", f"{result['final_score'].mean():.2f}")
    k4.metric("Top Score", f"{result['final_score'].max():.2f}")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["📋 Results", "📊 Analytics", "🆚 Compare"])

    with tab1:
        for rank, row in result.iterrows():
            css_class = "card-green" if row["status"] == "Shortlisted" else "card-yellow"
            pill_class = "pill-shortlisted" if row["status"] == "Shortlisted" else "pill-notshortlisted"

            st.markdown(
                f"""
                <div class="candidate-card {css_class}">
                    <span class="rank-tag">Rank #{rank}</span>
                    <span class="status-pill {pill_class}">{row['status']}</span>
                    <h3 style="margin:10px 0 4px 0;">{row['candidate']}</h3>
                    <p style="margin:0; opacity:0.85;">
                        Experience: <b>{row['exp_years']} yrs</b> &nbsp;|&nbsp;
                        Semantic match: <b>{row['bert_score']:.2f}</b> &nbsp;|&nbsp;
                        Skill score: <b>{row['skill_score']:.2f}</b>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(max(row["final_score"], 0.0), 1.0), text=f"Final score: {row['final_score']:.2f}")
            st.markdown(
                f"**Matched:** {skill_chips(row['matched_skills'], 'match')}"
                f"&nbsp;&nbsp;**Missing:** {skill_chips(row['missing_skills'], 'missing')}",
                unsafe_allow_html=True,
            )
            with st.expander("View resume text"):
                st.text(row["resume_text"][:2000])
            st.write("")

        csv_data = result.drop(columns=["resume_text"]).to_csv(index_label="rank")
        st.download_button("⬇️ Download results as CSV", csv_data, "results.csv", "text/csv")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Final score by candidate")
            st.bar_chart(result.set_index("candidate")["final_score"])
        with c2:
            st.subheader("Status breakdown")
            st.bar_chart(result["status"].value_counts())

        st.subheader("Most commonly missing skills")
        all_missing = [s for row in result["missing_skills"] for s in row]
        if all_missing:
            st.bar_chart(pd.Series(all_missing).value_counts())
        else:
            st.info("No common missing skills found across candidates.")

    with tab3:
        st.subheader("Compare two candidates")
        c1, c2 = st.columns(2)
        with c1:
            cand_a = st.selectbox("Candidate A", result["candidate"], key="cand_a")
        with c2:
            cand_b = st.selectbox("Candidate B", result["candidate"], key="cand_b")

        row_a = result[result["candidate"] == cand_a].iloc[0]
        row_b = result[result["candidate"] == cand_b].iloc[0]

        compare_df = pd.DataFrame({
            cand_a: [row_a["final_score"], row_a["bert_score"], row_a["skill_score"], row_a["exp_years"]],
            cand_b: [row_b["final_score"], row_b["bert_score"], row_b["skill_score"], row_b["exp_years"]],
        }, index=["Final Score", "Semantic Match", "Skill Score", "Experience (yrs)"])

        st.dataframe(compare_df, use_container_width=True)
        st.bar_chart(compare_df.T[["Final Score"]])
else:
    st.info("👈 Add a job description and upload resumes in the sidebar, then click **Screen Resumes**.")

st.markdown(
    "<p style='text-align:center; opacity:0.5; margin-top:40px;'>Built with Sentence-BERT, scikit-learn & Streamlit</p>",
    unsafe_allow_html=True,
)