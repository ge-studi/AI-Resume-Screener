import streamlit as st
import pandas as pd
import pickle
import fitz  # PyMuPDF

st.set_page_config(page_title="AI Resume Screener", layout="wide")

# -------------------------
# PDF to text
# -------------------------
def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text() + " "
    except Exception:
        return ""
    return text.strip()

# -------------------------
# Suggestions for improvement
# -------------------------
SUGGESTIONS = {
    "Data Scientist": ["Add experience with pandas, scikit-learn, visualization, EDA, statistical modeling."],
    "ML Engineer": ["Include TensorFlow/PyTorch, model serving, latency optimization, CI/CD pipelines."],
    "Data Engineer": ["Mention Spark, ETL, Kafka, SQL, data warehousing, schema management."],
    "AI Researcher": ["Add publications, ablation studies, transformer or self-supervised projects."],
    "Generative AI": ["Include LLM, prompt engineering, fine-tuning, diffusion models, generative projects."],
}

def suggest_improvements(pred_label):
    return " ".join(SUGGESTIONS.get(pred_label, []))

# -------------------------
# Load trained pipeline
# -------------------------
@st.cache_resource
def load_model(path="data/models/ensemble_resume_model.pkl"):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["pipeline"]

model = load_model()

# -------------------------
# Streamlit UI
# -------------------------
st.title("AI Resume Screener")
st.markdown("Upload resumes in **PDF format**. The screener predicts job category, confidence score, and suggests improvements.")

uploaded_files = st.file_uploader(
    "Upload PDF Resumes",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"Processing {len(uploaded_files)} resumes... This may take some time for large batches.")

    results = []
    for pdf_file in uploaded_files:
        text = extract_text_from_pdf(pdf_file)
        if not text:
            results.append({
                "filename": pdf_file.name,
                "predicted_category": "Unreadable PDF",
                "confidence_score": 0,
                "status": "Failed",
                "suggestions": "Unable to parse text from PDF."
            })
            continue

        X = [text]
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            max_idx = probs.argmax()
            pred_label = model.classes_[max_idx]
            confidence = probs[max_idx] * 100
        else:  # For LinearSVC without predict_proba
            pred_label = model.predict(X)[0]
            confidence = 85.0  # default proxy

        # ATS pass/fail
        status = "Good" if confidence >= 80 else "Failed to parse"

        # Detect Generative AI resumes
        if any(word in text.lower() for word in ["llm", "generative", "prompt"]):
            pred_label = "Generative AI"

        suggestions = "No suggestions. Resume passes ATS score." if status.startswith("Good") else suggest_improvements(pred_label)

        results.append({
            "filename": pdf_file.name,
            "predicted_category": pred_label,
            "confidence_score": round(confidence, 2),
            "status": status,
            "suggestions": suggestions
        })

    df_results = pd.DataFrame(results)
    st.dataframe(df_results)

    # Download CSV
    st.download_button(
        label="Download Results as CSV",
        data=df_results.to_csv(index=False).encode("utf-8"),
        file_name="resume_screener_results.csv",
        mime="text/csv"
    )
