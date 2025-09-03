import streamlit as st
import pandas as pd
import pickle
import fitz
import altair as alt
import math

st.set_page_config(page_title="AI Resume Screener", layout="wide")

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text() + " "
        pdf_file.seek(0)
    except Exception:
        return ""
    return text.strip()

SUGGESTIONS = {
    "Data Scientist": ["Add experience with pandas, scikit-learn, visualization, EDA, statistical modeling."],
    "ML Engineer": ["Include TensorFlow/PyTorch, model serving, latency optimization, CI/CD pipelines."],
    "Data Engineer": ["Mention Spark, ETL, Kafka, SQL, data warehousing, schema management."],
    "AI Researcher": ["Add publications, ablation studies, transformer or self-supervised projects."],
    "Generative AI": ["Include LLM, prompt engineering, fine-tuning, diffusion models, generative projects."],
}

def suggest_improvements(pred_label):
    return " ".join(SUGGESTIONS.get(pred_label, []))

@st.cache_resource
def load_model(path="data/models/ensemble_resume_model.pkl"):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["pipeline"]

model = load_model()

st.title("AI Resume Screener")
st.markdown("Upload resumes in **PDF format**. The screener predicts job category, confidence score, and suggests improvements.")

uploaded_files = st.file_uploader("Upload PDF Resumes", type="pdf", accept_multiple_files=True)

if uploaded_files:
    st.info(f"Processing {len(uploaded_files)} resumes...")
    results = []
    file_bytes_map = {}

    for pdf_file in uploaded_files:
        text = extract_text_from_pdf(pdf_file)
        file_bytes_map[pdf_file.name] = pdf_file.read()
        pdf_file.seek(0)

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
        else:
            pred_label = model.predict(X)[0]
            confidence = 85.0

        status = "Good" if confidence >= 80 else "Failed to parse"
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

    # -------------------------
    # Chart
    # -------------------------
    st.subheader("Resumes per Job Category")
    chart_data = df_results.groupby("predicted_category").size().reset_index(name="count")
    chart = alt.Chart(chart_data).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, width=40).encode(
        x=alt.X("predicted_category:N", title="Category"),
        y=alt.Y("count:Q", title="Number of Resumes"),
        color=alt.Color("predicted_category:N", scale=alt.Scale(scheme="set2")),
        tooltip=["predicted_category", "count"]
    ).properties(width=600, height=400)
    st.altair_chart(chart)

    # -------------------------
    # Responsive horizontal cards
    # -------------------------
    st.subheader("Resume Screening Results")
    category_colors = {
        "Data Scientist": "#1f77b4",
        "ML Engineer": "#9467bd",
        "Data Engineer": "#ff7f0e",
        "AI Researcher": "#2ca02c",
        "Generative AI": "#d62728",
        "Unreadable PDF": "#7f7f7f"
    }

    # Adjust number of columns dynamically
    screen_width = st.session_state.get("screen_width", 3)
    cols_per_row = screen_width if screen_width else 3  # default to 3

    for i in range(0, len(df_results), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(df_results):
                break
            row = df_results.iloc[idx]
            with col:
                st.markdown(f"""
                <div style="border:1px solid #ccc; border-radius:10px; padding:10px; background:#f9f9f9; min-height:220px;">
                    <h5 style="margin:0; word-break: break-word;">{row['filename']}</h5>
                    <p style="margin:2px 0; color:{category_colors.get(row['predicted_category'], 'black')}"><b>Category:</b> {row['predicted_category']}</p>
                    <p style="margin:2px 0; color:{'green' if row['status']=='Good' else 'red'}"><b>Status:</b> {row['status']}</p>
                    <p style="margin:2px 0;"><b>Confidence:</b> {row['confidence_score']}%</p>
                    <p style="margin:2px 0; word-break: break-word;"><b>Suggestions:</b> {row['suggestions']}</p>
                </div>
                """, unsafe_allow_html=True)
                pdf_bytes = file_bytes_map.get(row['filename'])
                if pdf_bytes:
                    st.download_button(
                        label="Download PDF",
                        data=pdf_bytes,
                        file_name=row['filename'],
                        mime="application/pdf"
                    )

    # -------------------------
    # CSV Download
    # -------------------------
    st.download_button(
        label="Download All Results as CSV",
        data=df_results.to_csv(index=False).encode("utf-8"),
        file_name="resume_screener_results.csv",
        mime="text/csv"
    )
