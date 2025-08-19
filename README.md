# AI Resume Screener

**AI Resume Screener** is a Streamlit app + training pipeline for classifying resumes into AI-related job categories (Data Scientist, ML Engineer, Data Engineer, AI Researcher, Generative AI).  
It includes:

- Robust CSV/XLSX loader with encoding/delimiter detection
- Text extraction for PDF / DOCX / TXT resumes
- TF-IDF and optional BERT-based feature pipelines
- Hidden rule-based keyword ensemble to boost detection of Generative AI
- A synthetic dataset generator `generate_non_overfitting_dataset.py` that produces a dataset designed to avoid trivial keyword leakage (so models don't trivially reach 100% accuracy)

---

## Repo layout (suggested)


├── app.py # main Streamlit app (your Streamlit code)

├── generate_dataset.py # dataset generator (provided)

├── non_overfitting_resumes.csv # generated dataset (optional — large)

├── requirements.txt

├── README.md

├── .gitignore

├── resume_model_state_complete_genai_fixed.pkl# (optional) saved model files go here


---

## Quick start (local)

Create a virtual environment (recommended)

python -m venv .venv #Windows

source .venv/bin/activate # macOS / Linux

# .venv\Scripts\activate # Windows PowerShell

Install requirements

pip install -r requirements.txt

Optional (BERT / semantic embeddings)

If you want the BERT / sentence-transformers option for semantic features, first install the appropriate torch package for your platform (see https://pytorch.org/get-started/locally/), then:

pip install sentence-transformers

Note: sentence-transformers requires torch. Installing sentence-transformers without torch will not enable the BERT option.

Generate the synthetic dataset (optional)

python generate_non_overfitting_dataset.py

# Produces non_overfitting_resumes.csv (by default)

Run the Streamlit app

streamlit run app.py

Then open the URL displayed by Streamlit (usually http://localhost:8501).

How to use the app

Load Dataset

Upload a CSV/XLSX/ZIP with a column containing resume text and a label column with job category. The loader attempts to auto-detect encoding and delimiter.

Train Model

Choose feature type tfidf (fast) or bert (semantic, optional).

Choose classifier (Logistic Regression is default).

Set CV folds, remove duplicates, and other options then click Train now.

The app enforces a minimum test accuracy before saving the model (prevents accidental overwriting with weak models).

Screen Resume

Upload a PDF/DOCX/TXT resume to classify using the saved model.

The app shows top predictions and a rule-based keyword ensemble for Generative AI.

Files of interest / configuration

MODEL_STATE_PATH in the app controls where the trained model is saved (default: resume_model_state_complete_genai_fixed.pkl).

generate_dataset.py — creates a balanced synthetic dataset designed to make models not trivially overfit on single tokens.

Reproducibility notes

The dataset generator uses a fixed seed (configurable in the file) for deterministic output.

The training pipeline removes exact and near-duplicates, and has optional downsampling to reduce class imbalance.

If using bert option, embedding caching and CPU vs GPU differences may affect runtime.





