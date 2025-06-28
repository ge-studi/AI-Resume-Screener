# 🤖 AI Resume Screener

An AI-based system to automatically classify resumes into suitable job categories using NLP techniques and machine learning models. Built with Streamlit, it enables HR teams to efficiently screen multiple resumes and get instant predictions, confidence scores, and top skill highlights.

---

## 📌 Problem Statement

HR departments often struggle to manually screen thousands of resumes for a limited number of job openings. This project automates the process by using ML and NLP to categorize resumes into job roles like Data Science, DevOps, Sales, etc.

---

## 🚀 Features

- Upload resumes in `.pdf`, `.docx`, or `.txt` formats
- Classify resumes into job roles using:
  - TF-IDF + Scikit-learn models (Logistic Regression, SVM, Naive Bayes, Random Forest)
  - Pre-trained BERT embeddings (`all-MiniLM-L6-v2`)
- Show prediction confidence score
- Extract top relevant skills per role
- Display resume preview
- Visualize classification metrics (accuracy, precision, recall, confusion matrix)

---

## 🧠 How It Works

1. **Upload resumes**
2. **Preprocess text:** lowercasing, tokenization, stemming, stopword removal
3. **Vectorize text:** TF-IDF or BERT embeddings
4. **Train model:** Choose from Logistic Regression, SVM, Random Forest, or Naive Bayes
5. **Predict role & extract skills**

---

## 🛠 Tech Stack

- Python
- Streamlit
- NLTK, Scikit-learn
- BERT via `sentence-transformers`
- Seaborn, Matplotlib
- PDF & DOCX parsing with `PyPDF2`, `docx2txt`

---

## 🗂 File Structure

resume-screener/

├── resume_screener.py # Main Streamlit app

├── requirements.txt # Project dependencies

├── resume_model.pkl # Trained ML model (generated after training)

├── sample_resumes.csv # Optional: sample resume training data

└── README.md # Descriptive file about the project

## 📦 Installation
git clone 
https://github.com/your-username/resume-screener.git

cd resume-screener

pip install -r requirements.txt

streamlit run resume_screener.py

📊 Model Training
You can train the model using sample data or upload a CSV with two columns:

resume_text: The raw resume content

job_category: The target label (e.g., Data Science, Sales)

🖥 Deployment Options
✅ Option 1: Streamlit Cloud
Push code to GitHub

Go to https://share.streamlit.io

Deploy using resume_screener.py

✅ Option 2: Localhost
streamlit run resume_screener.py
📈 Sample Training Output
Accuracy: >85% on sample resumes

View detailed classification report and confusion matrix in app

🔮 Future Enhancements
Bulk generation of synthetic resumes for demo/testing

JD (Job Description) upload and candidate ranking system

Candidate shortlisting with customizable filters


📩 Contact
For suggestions, improvements, or demo requests:
📧 gssingh6393@gmail.com

🔗 https://www.linkedin.com/in/geetanjali--singh/

📝 License
MIT License





