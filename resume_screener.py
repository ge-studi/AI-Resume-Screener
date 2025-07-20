import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import nltk
import docx2txt
import PyPDF2
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download required NLTK resources
# nltk.download('punkt')
nltk.download('stopwords')


class ResumeScreener:
    def __init__(self, model_type='tfidf'):
        self.model = None
        self.vectorizer = None
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.model_type = model_type
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.job_categories = {
            'Data Science': ['python', 'machine learning', 'data analysis', 'sql'],
            'Software Development': ['java', 'spring', 'api'],
            'Web Development': ['html', 'css', 'javascript'],
            'DevOps': ['docker', 'kubernetes', 'jenkins'],
            'Digital Marketing': ['seo', 'ppc', 'content marketing'],
            'HR': ['recruitment', 'performance management'],
            'Finance': ['accounting', 'financial modeling'],
            'Sales': ['lead generation', 'negotiation'],
            'Business Analyst': ['business analysis', 'documentation'],
            'Project Management': ['project management', 'agile', 'scrum']
        }

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        tokens = [self.stemmer.stem(token) for token in tokens if token not in self.stop_words and len(token) > 2]
        return ' '.join(tokens)

    def train_model(self, df, model_choice):
        df['processed_text'] = df['resume_text'].apply(self.preprocess_text)

        if self.model_type == 'bert':
            X = self.bert_model.encode(df['processed_text'], show_progress_bar=True)
        else:
            self.vectorizer = TfidfVectorizer(max_features=5000)
            X = self.vectorizer.fit_transform(df['processed_text'])

        y = df['job_category']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_choice == 'Logistic Regression':
            self.model = LogisticRegression(max_iter=1000)
        elif model_choice == 'Random Forest':
            self.model = RandomForestClassifier()
        elif model_choice == 'Naive Bayes':
            if self.model_type == 'bert':
                st.error("Naive Bayes is only compatible with TF-IDF features.")
                return None, None, None, None, None
            self.model = MultinomialNB()

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        return report, acc, cm, y_test, y_pred

    def predict_resume(self, resume_text):
        text = self.preprocess_text(resume_text)
        if self.model_type == 'bert':
            vec = self.bert_model.encode([text])
        else:
            vec = self.vectorizer.transform([text])
        prediction = self.model.predict(vec)[0]
        confidence = self.model.predict_proba(vec).max() if hasattr(self.model, 'predict_proba') else 1.0
        return prediction, confidence


def extract_text(file):
    if file.name.endswith('.pdf'):
        reader = PyPDF2.PdfReader(file)
        return " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    elif file.name.endswith('.docx'):
        return docx2txt.process(file)
    elif file.name.endswith('.txt'):
        return file.read().decode("utf-8")
    return ""


def plot_metrics(report, cm, labels):
    st.subheader("📊 Classification Report")
    st.dataframe(pd.DataFrame(report).transpose().round(2))

    st.subheader("📌 Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(fig)


def main():
    st.set_page_config(page_title="AI Resume Screener", layout="wide")
    st.title("🤖 AI Resume Screener")

    page = st.sidebar.selectbox("Choose Page", ["Train Model", "Screen Resume"])

    if page == "Train Model":
        st.subheader("📌 Train Model")
        feature_type = st.selectbox("Feature Type", ["tfidf", "bert"])
        model_choice = st.selectbox("Select Model", ["Logistic Regression", "Random Forest", "Naive Bayes"])

        file = st.file_uploader("Upload CSV with 'resume_text' and 'job_category'", type=['csv'])
        use_sample = st.checkbox("Use Sample Data")

        if st.button("Train Now"):
            screener = ResumeScreener(model_type=feature_type)

            if use_sample or file is None:
                data = []
                for cat, skills in screener.job_categories.items():
                    for i in range(3):
                        data.append((f"Experienced in {', '.join(skills)}", cat))
                df = pd.DataFrame(data, columns=["resume_text", "job_category"])
            else:
                df = pd.read_csv(file)
                if 'Resume' in df.columns and 'Category' in df.columns:
                    df.rename(columns={"Resume": "resume_text", "Category": "job_category"}, inplace=True)

            report, acc, cm, y_test, y_pred = screener.train_model(df, model_choice)
            if report:
                st.success(f"🎉 Model Trained with Accuracy: {acc:.2%}")
                plot_metrics(report, cm, sorted(df['job_category'].unique()))
                with open("resume_model.pkl", "wb") as f:
                    pickle.dump(screener, f)

    elif page == "Screen Resume":
        st.subheader("📥 Upload Resume to Screen")
        try:
            with open("resume_model.pkl", "rb") as f:
                screener = pickle.load(f)
        except:
            st.error("❌ Please train a model first.")
            return

        files = st.file_uploader("Upload resume file(s)", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)

        for file in files:
            text = extract_text(file)
            pred, conf = screener.predict_resume(text)
            st.subheader(file.name)
            st.markdown(f"**Predicted Category:** `{pred}`")
            st.markdown(f"**Confidence:** `{conf:.2%}`")
            st.text_area("Resume Preview", value=text[:1500], height=200)


if __name__ == "__main__":
    main()
