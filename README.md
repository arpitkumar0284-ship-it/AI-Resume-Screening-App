\# AI Based Resume Screening and Candidate Shortlisting System



A system that ranks resumes against a job description and shortlists the best-matching candidates.



\## How it works



1\. Reads resumes (PDF, DOCX or TXT) and a job description.

2\. Scores each resume using three signals:

&#x20;  - Semantic similarity (Sentence-BERT embeddings)

&#x20;  - Skill match (required and good-to-have skills)

&#x20;  - Experience (years extracted from the resume)

3\. Combines them into a final score, ranks candidates and marks the top N as shortlisted.



Final score = 0.35 x semantic similarity + 0.45 x skill score + 0.20 x experience score



\## Project structure



\- screening.py : core scoring pipeline (command line)

\- app.py : Streamlit web app

\- evaluate.py : compares TF-IDF, BERT and Hybrid using Precision@5

\- data/ : sample resumes, job description and labels.csv

\- output/ : results.csv



\## Setup



&#x20;   python -m venv venv

&#x20;   venv\\Scripts\\activate

&#x20;   pip install -r requirements.txt



\## Run



Command line:



&#x20;   python screening.py



Web app:



&#x20;   streamlit run app.py



Evaluation:



&#x20;   python evaluate.py



\## Results (10 sample resumes)



| Model  | Precision@5 |

|--------|-------------|

| TF-IDF | 0.80        |

| BERT   | 0.80        |

| Hybrid | 1.00        |



\## Limitations



\- Evaluation is on a small synthetic sample; skills list and weights were tuned on the same data, so the Hybrid score is optimistic.

\- Labels were created manually, not by real recruiters.

\- Skill matching is keyword based and can miss related skills (e.g. an NLP engineer who does not write "machine learning").

\- Experience is the first "N years" found in the resume and is not checked for relevance to the role.



\## Future work



\- Test on a larger dataset with real recruiter labels.

\- Use a skill-extraction model instead of keyword lists.

\- Extract role-specific experience.

\- Add bias checks so the system does not favour or reject candidates unfairly.

