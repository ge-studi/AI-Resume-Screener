# AI Resume Screener

AI Resume Screener is a Streamlit web app that classifies resumes into AI-related job categories (Data Scientist, ML Engineer, Data Engineer, AI Researcher, Generative AI).

It works by:

Extracting text from PDF resumes (using PyMuPDF)

Running predictions with a pre-trained ML pipeline (loaded from data/models/ensemble_resume_model.pkl)

Showing predicted category, confidence score, ATS pass/fail, and tailored improvement suggestions

Exporting results to CSV for bulk screening

# Features

📄 Upload multiple PDF resumes

🔍 Predict job category with confidence scores

✅ ATS-style pass/fail check (based on confidence ≥ 80%)

💡 Improvement suggestions for each role (e.g., skills to add for ML Engineer, Data Scientist, etc.)

⚡ Detects Generative AI resumes via keywords (LLM, generative, prompt)

⬇️ Download results as a CSV report

# Repo Layout

├── app.py                        # Main Streamlit app

├── data/
│   └── models/
│       └── ensemble_resume_model.pkl   # Pre-trained ML pipeline (required)

├── requirements.txt

├── README.md

└── .gitignore

# Quick Start (Local)

Create a virtual environment
   
python -m venv .venv   # Windows

.venv\Scripts\activate # Windows PowerShell

# OR on macOS/Linux

source .venv/bin/activate

Install dependencies
   
pip install -r requirements.txt

Run the app

streamlit run app.py


Then open http://localhost:8501 in your browser.

# How to Use

Upload PDF resumes (one or many).


The app will show a table with:


Filename

Predicted category

Confidence score

ATS status (Good / Failed to parse)


Suggestions for improvement

Download results as CSV for record-keeping.

# Model Notes


The model is loaded from data/models/ensemble_resume_model.pkl

It should contain a scikit-learn pipeline with predict (and optionally predict_proba) methods.

For classifiers without predict_proba (e.g., LinearSVC), a default confidence proxy is used.


# ⚡ Future extensions:

Add DOCX/TXT parsing

Re-train pipeline with fresh datasets

Integrate BERT embeddings for semantic features
