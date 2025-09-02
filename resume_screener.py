import streamlit as st
import pandas as pd
import pickle
import fitz  # PyMuPDF
import base64
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Resume Screener", layout="wide")

# -------------------------
# Theme Toggle
# -------------------------
dark_mode = st.sidebar.checkbox("Dark Mode", value=False)

if dark_mode:
    st.markdown(
        """
        <style>
        .reportview-container {background-color: #0E1117; color: #F5F5F5;}
        .stButton>button, .stDownloadButton>button {border-radius: 5px;}
        .stButton>button {background-color: #1F2937; color: #F5F5F5;}
        .stDownloadButton>button {background-color: #2563EB; color: white;}
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        .reportview-container {background-color: #FFFFFF; color: #000000;}
        .stButton>button, .stDownloadButton>button {border-radius: 5px;}
        .stButton>button {background-color: #E5E7EB; color: #000000;}
        .stDownloadButton>button {background-color: #2563EB; color: white;}
        </style>
        """,
        unsafe_allow_html=True,
    )

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
            "Confidence Score (%)": round(confidence, 2),
            "Status": status,
            "Suggestions": suggestions,
            "PDF Bytes": pdf_file.getvalue()
        })

    df_results = pd.DataFrame(results)

    # -------------------------
    # Dashboard Summary
    # -------------------------
    total_resumes = len(df_results)
    good_count = len(df_results[df_results["Status"]=="Good"])
    medium_count = len(df_results[df_results["Status"]=="Medium"])
    low_count = len(df_results[df_results["Status"]=="Low"])
    avg_confidence = round(df_results["Confidence Score (%)"].mean(),2)

    st.markdown("### Summary Dashboard")
    st.markdown(f"""
        - **Total Resumes:** {total_resumes}  
        - **Good:** {good_count}  
        - **Medium:** {medium_count}  
        - **Low:** {low_count}  
        - **Average Confidence Score:** {avg_confidence}%
    """)

    # -------------------------
    # Pie chart for Good/Medium/Low
    # -------------------------
    fig1, ax1 = plt.subplots()
    ax1.pie([good_count, medium_count, low_count],
            labels=["Good", "Medium", "Low"],
            colors=["#22c55e", "#facc15", "#ef4444"],
            autopct="%1.1f%%", startangle=90)
    ax1.axis("equal")
    st.pyplot(fig1)

    # -------------------------
    # Bar chart for Predicted Categories
    # -------------------------
    category_counts = df_results["Predicted Category"].value_counts()
    fig2, ax2 = plt.subplots()
    ax2.bar(category_counts.index, category_counts.values, color="#2563EB")
    ax2.set_ylabel("Number of Resumes")
    ax2.set_xlabel("Job Category")
    ax2.set_title("Resumes per Job Category")
    plt.xticks(rotation=45)
    st.pyplot(fig2)

    # -------------------------
    # Scrollable table with sticky headers & confidence bars
    # -------------------------
    st.markdown("### Resume Screening Results")
    st.markdown(
        """
        <style>
        .scrollable-table {max-height: 400px; overflow-y: auto;}
        .scrollable-table th {position: sticky; top: 0; background-color: #2563EB; color: white;}
        </style>
        """,
        unsafe_allow_html=True
    )

    for idx, row in df_results.iterrows():
        if row["Status"] == "Good":
            row_color = "#d1fae5"
        elif row["Status"] == "Medium":
            row_color = "#fef3c7"
        else:
            row_color = "#fee2e2"

        cols = st.columns([2,2,3,1,3,1,1])
        cols[0].markdown(f"<div style='background-color:{row_color}; padding:5px'>{row['Filename']}</div>", unsafe_allow_html=True)
        cols[1].markdown(f"<div style='background-color:{row_color}; padding:5px'>{row['Predicted Category']}</div>", unsafe_allow_html=True)

        bar_color = "#22c55e" if row["Status"]=="Good" else "#facc15" if row["Status"]=="Medium" else "#ef4444"
        cols[2].markdown(f"""
            <div style='background-color:#e5e7eb; width:100%; border-radius:5px; height:20px;'>
                <div style='width:{row["Confidence Score (%)"]}%; background-color:{bar_color}; height:100%; border-radius:5px; text-align:center; color:black;'>{row["Confidence Score (%)"]}%</div>
            </div>
        """, unsafe_allow_html=True)

        cols[3].markdown(f"<div style='background-color:{row_color}; padding:5px'>{row['Status']}</div>", unsafe_allow_html=True)
        cols[4].markdown(f"<div style='background-color:{row_color}; padding:5px'>{row['Suggestions']}</div>", unsafe_allow_html=True)

        b64_pdf = base64.b64encode(row["PDF Bytes"]).decode()
        href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="{row["Filename"]}">Download PDF</a>'
        cols[5].markdown(href, unsafe_allow_html=True)

        copy_button = f"""
        <button onclick="navigator.clipboard.writeText('{row["Suggestions"].replace("'", "\\'")}')">Copy Suggestions</button>
        """
        cols[6].markdown(copy_button, unsafe_allow_html=True)

    st.download_button(
        label="Download Results as CSV",
        data=df_results.drop(columns=["PDF Bytes"]).to_csv(index=False).encode("utf-8"),
        file_name="resume_screener_results.csv",
        mime="text/csv"
    )
