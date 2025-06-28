# resume_screener.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import nltk
import spacy
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import docx2txt
import PyPDF2
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


class ResumeScreener:
    def __init__(self, model_type="bert"):
        self.vectorizer = None
        self.model = None
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.model_type = model_type

        if self.model_type == 'bert':
            self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')

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

    def train_model(self, df, model_type='logistic_regression'):
        df['processed_text'] = df['resume_text'].apply(self.preprocess_text)

        if model_type == 'naive_bayes' and self.model_type == 'bert':
            st.error("❌ Naive Bayes only works with TF-IDF. Please change feature type.")
            return None, None, None

        if self.model_type == 'bert':
            X = self.bert_model.encode(df['processed_text'], show_progress_bar=True)
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            X = self.vectorizer.fit_transform(df['processed_text'])

        y = df['job_category']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_type == 'logistic_regression':
            self.model = LogisticRegression(max_iter=1000)
        elif model_type == 'svm':
            self.model = SVC(kernel='linear', probability=True)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier()
        elif model_type == 'naive_bayes':
            self.model = MultinomialNB()

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy, y_test, y_pred

    def predict_job_category(self, resume_text):
        if self.model is None:
            return None, None

        processed_text = self.preprocess_text(resume_text)

        if self.model_type == 'bert':
            text_vector = self.bert_model.encode([processed_text])
        else:
            text_vector = self.vectorizer.transform([processed_text])

        prediction = self.model.predict(text_vector)[0]
        confidence = self.model.predict_proba(text_vector).max() if hasattr(self.model, "predict_proba") else 1.0
        return prediction, confidence

    def get_top_skills(self, resume_text, job_category):
        resume_lower = resume_text.lower()
        skills = self.job_categories.get(job_category, [])
        return [s for s in skills if s.lower() in resume_lower]

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer,
                'model_type': self.model_type,
                'job_categories': self.job_categories
            }, f)

    def load_model(self, filename):
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)

            self.model = model_data['model']
            self.vectorizer = model_data.get('vectorizer', None)
            self.model_type = model_data.get('model_type', 'tfidf')
            self.job_categories = model_data.get('job_categories', {})

            if self.model_type == 'bert':
                self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


def extract_text_from_pdf(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        return " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
    except:
        return ""


def extract_text_from_docx(docx_file):
    try:
        return docx2txt.process(docx_file)
    except:
        return ""


def plot_metrics(y_test, y_pred):
    report = classification_report(y_test, y_pred, output_dict=True)
    metrics_df = pd.DataFrame(report).transpose()
    st.subheader("📊 Classification Report")
    st.dataframe(metrics_df.round(2))

    st.subheader("📌 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap="YlGnBu", xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(fig)


def main():
    st.set_page_config(page_title="AI Resume Screener", layout="wide")
    st.title("🤖 AI Resume Screener (BERT / TF-IDF)")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Train Model", "Screen Resume"])

    if page == "Train Model":
        st.header("📌 Train a Resume Classifier")
        model_backend = st.selectbox("Feature Type", ["bert", "tfidf"])
        model_choice = st.selectbox("Model", ["logistic_regression", "svm", "random_forest", "naive_bayes"])
        screener = ResumeScreener(model_type=model_backend)

        uploaded = st.file_uploader("Upload CSV (resume_text, job_category)", type=["csv"])
        use_sample = st.checkbox("Use Sample Data")

        if st.button("Train Now"):
            if use_sample or uploaded is None:
                data = []
                for cat in screener.job_categories:
                    for i in range(3):
                        sample = f"This resume shows strong skills in {', '.join(screener.job_categories[cat][:3])}."
                        data.append((sample, cat))
                df = pd.DataFrame(data, columns=["resume_text", "job_category"])
            else:
                df = pd.read_csv(uploaded)

            acc, y_test, y_pred = screener.train_model(df, model_choice)
            if acc:
                screener.save_model("resume_model.pkl")
                st.success(f"🎉 Model Trained! Accuracy: {acc:.2%}")
                plot_metrics(y_test, y_pred)

    elif page == "Screen Resume":
        st.header("📥 Upload Resumes for Screening")
        screener = ResumeScreener()
        if not screener.load_model("resume_model.pkl"):
            st.error("⚠️ Model not found. Train the model first.")
            return

        uploaded_files = st.file_uploader("Upload resume files", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)
        for file in uploaded_files:
            if file.type == "application/pdf":
                text = extract_text_from_pdf(file)
            elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = extract_text_from_docx(file)
            else:
                text = file.read().decode("utf-8")

            pred, conf = screener.predict_job_category(text)
            skills = screener.get_top_skills(text, pred)
            st.subheader(f"📄 {file.name}")
            st.markdown(f"**Predicted Category:** `{pred}` ({conf:.1%} confidence)")
            st.markdown(f"**Skills Identified:** {', '.join(skills) if skills else 'None'}")
            st.text_area("Resume Preview", value=text[:1500], height=200)


if __name__ == "__main__":
    main()
