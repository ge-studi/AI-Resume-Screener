import streamlit as st
import pandas as pd
import pickle
import fitz  # PyMuPDF
import numpy as np
import plotly.express as px

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="AI Resume Screener", layout="wide")

# -------------------------
# Dark/Light Mode Toggle
# -------------------------
mode = st.sidebar.radio("Select Theme", ["Light", "Dark"])

if mode == "Dark":
    st.markdown("""
    <style>
    body {background-color: #0f172a; color: #f8fafc;}
    .stButton>button {background-color: #2563eb; color: white;}
    .stDownloadButton>button {background-color: #22c55e; color: white;}
    table th, table td {color: #f8fafc !important;}
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
    body {background-color: #ffffff; color: #000000;}
    table th, table td {color: #000000 !important;}
    </style>
    """, unsafe_allow_html=True)

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
# Suggestions
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
# Load model
# -------------------------
@st.cache_resource
def load_model(path="data/models/ensemble_resume_model.pkl"):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["pipeline"]

model = load_model()

# -------------------------
# Upload resumes
# -------------------------
st.title("AI Resume Screener")
st.markdown("Upload resumes in **PDF format**. The screener predicts job category, confidence score, and suggests improvements.")

uploaded_files = st.file_uploader("Upload PDF Resumes", type="pdf", accept_multiple_files=True)

if uploaded_files:
    st.info(f"Processing {len(uploaded_files)} resumes...")
    results = []

    for pdf_file in uploaded_files:
        text = extract_text_from_pdf(pdf_file)
        if not text:
            pred_label = "Unreadable PDF"
            confidence = 0
            status = "Low"
            suggestions = "Unable to parse text from PDF."
        else:
            X = [text]
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[0]
                max_idx = probs.argmax()
                pred_label = model.classes_[max_idx]
                confidence = probs[max_idx] * 100
            else:
                pred_label = model.predict(X)[0]
                confidence = 85.0

            if confidence >= 80:
                status = "Good"
            elif confidence >= 50:
                status = "Medium"
            else:
                status = "Low"

            if any(word in text.lower() for word in ["llm", "generative", "prompt"]):
                pred_label = "Generative AI"

            suggestions = "No suggestions. Resume passes ATS score." if status == "Good" else suggest_improvements(pred_label)

        results.append({
            "Filename": pdf_file.name,
            "Predicted Category": pred_label,
            "Status": status,
            "Suggestions": suggestions,
            "Confidence": round(confidence,2),
            "PDF Bytes": pdf_file.getvalue()
        })

    df_results = pd.DataFrame(results)

    # -------------------------
    # Summary Metrics
    # -------------------------
    total_resumes = len(df_results)
    good_count = len(df_results[df_results["Status"]=="Good"])
    medium_count = len(df_results[df_results["Status"]=="Medium"])
    low_count = len(df_results[df_results["Status"]=="Low"])
    avg_confidence = round(df_results["Confidence"].mean(),2)

    st.markdown("### Summary Dashboard")
    col1, col2, col3 = st.columns([1,1,2])
    with col1:
        st.metric("Total Resumes", total_resumes)
        st.metric("Average Confidence", f"{avg_confidence}%")
    with col2:
        st.metric("Good", good_count)
        st.metric("Medium", medium_count)
        st.metric("Low", low_count)

    # -------------------------
    # Interactive Pie Chart
    # -------------------------
    status_counts = df_results['Status'].value_counts().reindex(['Good','Medium','Low'], fill_value=0)
    fig_pie = px.pie(
        names=status_counts.index,
        values=status_counts.values,
        color=status_counts.index,
        color_discrete_map={'Good':'#22c55e', 'Medium':'#facc15', 'Low':'#ef4444'},
        title="Resume Status Distribution",
        hole=0.4
    )
    fig_pie.update_traces(textposition='inside', textinfo='percent+label', hoverinfo='label+value+percent')
    st.plotly_chart(fig_pie, use_container_width=True)

    # -------------------------
    # Interactive Bar Chart
    # -------------------------
    category_counts = df_results["Predicted Category"].value_counts()
    fig_bar = px.bar(
        x=category_counts.index,
        y=category_counts.values,
        text=category_counts.values,
        color=category_counts.values,
        color_continuous_scale='Viridis',
        labels={'x':'Job Category','y':'Number of Resumes'},
        title="Resumes per Job Category"
    )
    fig_bar.update_traces(textposition='outside')
    fig_bar.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_bar, use_container_width=True)

    # -------------------------
    # Table with Confidence Bars (Scrollable)
    # -------------------------
    def render_confidence_bar(score):
        color = "#22c55e" if score >= 80 else "#facc15" if score >= 50 else "#ef4444"
        return f"<div style='background-color:#e5e7eb; width:100%; border-radius:5px; height:20px;'>\
                <div style='width:{score}%; background-color:{color}; height:100%; border-radius:5px; text-align:center; color:black; font-size:12px;'>{score}%</div></div>"

    df_display = df_results.copy()
    df_display["Confidence"] = df_display["Confidence"].apply(render_confidence_bar)
    df_display = df_display.drop(columns=["PDF Bytes"])

    def generate_scrollable_table(df):
        html = "<div style='overflow-x:auto; max-height:500px;'>"
        html += "<table style='width:100%; border-collapse: collapse;'>"
        html += "<tr>"
        for col in df.columns:
            html += f"<th style='border:1px solid #ccc; padding:5px; background-color:#2563EB; color:white; min-width:140px;'>{col}</th>"
        html += "</tr>"
        for _, row in df.iterrows():
            bg_color = "#d1fae5" if row.Status=="Good" else "#fef3c7" if row.Status=="Medium" else "#fee2e2"
            html += "<tr>"
            for col in df.columns:
                html += f"<td style='border:1px solid #ccc; padding:5px; background-color:{bg_color}; vertical-align:middle; min-width:140px;'>{row[col]}</td>"
            html += "</tr>"
        html += "</table></div>"
        return html

    st.markdown("### Resume Screening Results")
    st.markdown(generate_scrollable_table(df_display), unsafe_allow_html=True)

    # -------------------------
    # Download CSV
    # -------------------------
    st.download_button(
        label="Download Results as CSV",
        data=df_results.drop(columns=["PDF Bytes"]).to_csv(index=False).encode("utf-8"),
        file_name="resume_screener_results.csv",
        mime="text/csv"
    )


