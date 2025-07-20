import streamlit as st
import PyPDF2
import docx2txt
import nltk
import os
import string
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Use local NLTK data folder
nltk.data.path.append(os.path.join(os.path.dirname(__file__), "nltk_data"))

# Set page config
st.set_page_config(page_title="AI Resume Screener", layout="wide")

st.title("📄 AI Resume Screener")

# Preprocessing functions
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    return docx2txt.process(file)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    return " ".join(tokens)

# Dummy training data
data = {
    "text": [
        "machine learning python pandas numpy",
        "deep learning pytorch keras tensorflow",
        "excel accounting tally finance tax",
        "sales marketing client business crm",
        "java spring boot microservices docker",
        "data analysis statistics visualization powerbi tableau",
    ],
    "label": ["Data Scientist", "Deep Learning Engineer", "Accountant", "Sales Executive", "Backend Developer", "Data Analyst"],
}
df = pd.DataFrame(data)
df["text"] = df["text"].apply(clean_text)

# Feature extraction and model training
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df["text"])
y = df["label"]

model1 = LogisticRegression()
model2 = MultinomialNB()
model3 = SVC(probability=True)

ensemble = VotingClassifier(estimators=[("lr", model1), ("nb", model2), ("svc", model3)], voting="soft")
ensemble.fit(X, y)

# Sentence transformer model
bert_model = SentenceTransformer("all-MiniLM-L6-v2")

# Upload section
uploaded_file = st.file_uploader("Upload your Resume (.pdf or .docx)", type=["pdf", "docx"])

if uploaded_file:
    # Text extraction
    if uploaded_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(uploaded_file)
    else:
        resume_text = extract_text_from_docx(uploaded_file)

    st.subheader("📜 Extracted Text")
    st.write(resume_text[:1000] + "...")  # Show first 1000 chars

    cleaned_resume = clean_text(resume_text)

    # ML prediction
    resume_vector = tfidf.transform([cleaned_resume])
    prediction = ensemble.predict(resume_vector)[0]
    st.success(f"🔍 Predicted Job Role: **{prediction}**")

    # Semantic similarity (optional)
    user_embedding = bert_model.encode([resume_text])
    job_embeddings = bert_model.encode(df["text"])
    similarities = cosine_similarity(user_embedding, job_embeddings)[0]

    similar_jobs = pd.DataFrame({
        "Role": df["label"],
        "Similarity": similarities,
    }).sort_values(by="Similarity", ascending=False)

    st.subheader("📈 Similar Roles (Semantic Match)")
    st.dataframe(similar_jobs.head(3).style.highlight_max(axis=0))
