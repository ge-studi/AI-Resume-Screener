"""
Complete Streamlit app enhanced for AI-related job categories,
improved ML pipeline, hidden ensemble UI, and enforced model accuracy saving.
"""


import io
import os
import re
import tempfile
import pickle
import zipfile
from typing import Optional, Tuple, List, Dict

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    BERT_AVAILABLE = True
except Exception:
    BERT_AVAILABLE = False

import docx2txt
from PyPDF2 import PdfReader

try:
    import chardet
    CHARDET_AVAILABLE = True
except Exception:
    CHARDET_AVAILABLE = False

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# -------------------------
# Config / paths
# -------------------------
MODEL_STATE_PATH = "resume_model_state_complete_genai_fixed.pkl"
EMBED_CACHE_DIR = "embed_cache_complete"
os.makedirs(EMBED_CACHE_DIR, exist_ok=True)

st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("AI Resume Screener")


# -------------------------
# Hardcoded ensemble weights (hidden from UI)
# -------------------------
keyword_beta = 0.15  # fixed weight for keyword ensemble - no UI control
suggest_only_when_keywords = False  # disabling toggle in UI


# -------------------------
# NLTK setup for lemmatization and stopwords
# -------------------------
def ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")
ensure_nltk()
STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# -------------------------
# Canonicalization (map variants -> canonical label)
# -------------------------
LABEL_CANONICAL_MAP = {
    # Generative AI labels
    "genai": "Generative AI",
    "generative ai": "Generative AI",
    "generative-ai": "Generative AI",
    "gpt": "Generative AI",
    "llm": "Generative AI",
    "large language model": "Generative AI",
    "generative": "Generative AI",
    # AI general
    "ai": "AI",
    "artificial intelligence": "AI",
    # Added AI-related job titles canonicalization
    "data scientist": "Data Scientist",
    "ai researcher": "AI Researcher",
    "machine learning engineer": "ML Engineer",
    "ml engineer": "ML Engineer",
    "data engineer": "Data Engineer",
    "machine learning": "ML Engineer",
    "ai engineer": "AI Engineer",
    "research scientist": "AI Researcher",
}


def canonicalize_label_value(raw_label: str) -> str:
    s = str(raw_label or "").strip().lower()
    if s in LABEL_CANONICAL_MAP:
        return LABEL_CANONICAL_MAP[s]
    return str(raw_label).strip()


# -------------------------
# Keyword map for rule-based signals tailored for AI-related roles
# -------------------------
KEYWORD_MAP = {
    "Generative AI": [
        "genai", "generative", "gpt", "chatgpt", "llm", "large language model",
        "gpt4", "gpt-4", "gpt3", "gpt-3.5", "dall", "dalle", "dall-e",
        "stable diffusion", "midjourney", "diffusion", "prompt engineering",
        "text-to-image", "image generation", "inpainting", "openai"
    ],
    "AI": [
        "artificial intelligence", "ai", "machine learning", "ml", "deep learning",
        "neural network", "neural networks", "nlp", "natural language processing", "computer vision"
    ],
    "Data Scientist": [
        "data analysis", "statistical modeling", "python", "r", "data visualization",
        "machine learning", "pandas", "numpy", "scikit-learn", "regression", "clustering"
    ],
    "AI Researcher": [
        "research", "deep learning", "neural networks", "transformers", "computer vision",
        "natural language processing", "papers", "publications", "theory", "algorithms"
    ],
    "ML Engineer": [
        "machine learning", "production", "model deployment", "tensorflow", "pytorch",
        "feature engineering", "pipeline", "ci/cd", "docker", "api"
    ],
    "Data Engineer": [
        "etl", "pipeline", "big data", "spark", "hadoop", "kafka", "sql", "nosql",
        "data warehousing", "airflow", "cloud storage"
    ],
    "AI Engineer": [
        "artificial intelligence", "machine learning", "deep learning", "neural networks",
        "tensorflow", "pytorch", "deployment", "models", "cloud", "ci/cd"
    ],
}


# -------------------------
# Text preprocessing
# -------------------------
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text or "")
    text = text.lower()
    text = re.sub(r"[^a-z0-9#+\-._\s]", " ", text)
    tokens = [lemmatizer.lemmatize(t) for t in text.split() if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


# -------------------------
# Safe train-test split
# -------------------------
def safe_train_test_split_texts(X_texts, y, desired_test_frac=0.2, random_state=42):
    labels, counts = np.unique(y, return_counts=True)
    n_classes = len(labels)
    n_samples = len(y)
    test_count = int(np.ceil(desired_test_frac * n_samples))
    test_count = max(test_count, n_classes)
    test_count = min(test_count, max(1, n_samples - 1))
    safe_test_frac = test_count / n_samples
    use_stratify = (n_classes > 1 and counts.min() >= 2 and (n_samples - test_count) >= n_classes)
    stratify_arg = y if use_stratify else None
    return train_test_split(X_texts, y, test_size=safe_test_frac, random_state=random_state, stratify=stratify_arg)


# -------------------------
# Robust CSV/Excel loader
# -------------------------
def detect_encoding(raw_bytes: bytes, nbytes: int = 20000) -> str:
    if CHARDET_AVAILABLE:
        guess = chardet.detect(raw_bytes[:nbytes])
        return guess.get("encoding") or "utf-8"
    return "utf-8"


def try_read_csv_bytes(raw_bytes: bytes) -> pd.DataFrame:
    errors = []
    enc = detect_encoding(raw_bytes)
    sep_candidates = [",", ";", "\t", "|"]
    for sep in sep_candidates:
        try:
            df_test = pd.read_csv(io.BytesIO(raw_bytes), encoding=enc, sep=sep, engine="python", nrows=5)
            df_full = pd.read_csv(io.BytesIO(raw_bytes), encoding=enc, sep=sep, engine="python")
            return df_full
        except Exception as e:
            errors.append((f"sep={sep}", str(e)))
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes), encoding="latin-1", engine="python")
        return df
    except Exception as e:
        errors.append(("latin1", str(e)))
    try:
        df_excel = pd.read_excel(io.BytesIO(raw_bytes))
        return df_excel
    except Exception as e:
        errors.append(("excel", str(e)))
    try:
        with io.BytesIO(raw_bytes) as bio:
            if zipfile.is_zipfile(bio):
                with zipfile.ZipFile(bio) as z:
                    csv_names = [n for n in z.namelist() if n.lower().endswith(".csv")]
                    if csv_names:
                        with z.open(csv_names[0]) as f:
                            inner = f.read()
                            return try_read_csv_bytes(inner)
    except Exception as e:
        errors.append(("zip", str(e)))
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes), encoding="utf-8", engine="python", on_bad_lines="skip")
        return df
    except Exception as e:
        errors.append(("final_utf8_skip", str(e)))
    raise ValueError("Failed to parse CSV. Attempts:\n" + "\n".join([f"{k}: {v}" for k, v in errors]))


@st.cache_data
def load_csv_from_bytes_auto(uploader) -> Tuple[pd.DataFrame, Dict]:
    raw = uploader.read()
    try:
        uploader.seek(0)
    except Exception:
        pass
    diag = {"filename": uploader.name, "size_bytes": len(raw), "encoding_guess": detect_encoding(raw)}
    df = try_read_csv_bytes(raw)
    diag["note"] = "loaded"
    return df, diag


def load_csv_local(path: str) -> Tuple[pd.DataFrame, Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    with open(path, "rb") as f:
        raw = f.read()
    diag = {"filename": path, "size_bytes": len(raw), "encoding_guess": detect_encoding(raw)}
    df = try_read_csv_bytes(raw)
    diag["note"] = "loaded_local"
    return df, diag


# -------------------------
# Resume file extractor (pdf/docx/txt)
# -------------------------
def extract_text_from_uploaded_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    try:
        data = uploaded_file.read()
        uploaded_file.seek(0)
    except Exception:
        return ""
    if name.endswith(".pdf"):
        try:
            with io.BytesIO(data) as bio:
                reader = PdfReader(bio)
                pages = []
                for page in reader.pages:
                    try:
                        pages.append(page.extract_text() or "")
                    except Exception:
                        pages.append("")
                text = "\n".join(pages).strip()
                return text
        except Exception:
            return ""
    if name.endswith(".docx"):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(data)
                tmp.flush()
                tmp_name = tmp.name
            txt = docx2txt.process(tmp_name) or ""
            try:
                os.remove(tmp_name)
            except Exception:
                pass
            return txt.strip()
        except Exception:
            return ""
    if name.endswith(".txt"):
        try:
            return data.decode("utf-8", errors="ignore").strip()
        except Exception:
            try:
                return data.decode("latin-1", errors="ignore").strip()
            except Exception:
                return ""
    return ""


# -------------------------
# Duplicate removal (exact + near)
# -------------------------
def remove_exact_duplicates(df: pd.DataFrame, text_col: str, label_col: str) -> Tuple[pd.DataFrame, int]:
    before = len(df)
    df2 = df.drop_duplicates(subset=[text_col, label_col]).reset_index(drop=True)
    return df2, before - len(df2)


def remove_near_duplicates(df: pd.DataFrame, text_col: str, threshold: float = 0.9, max_records: int = 5000) -> Tuple[pd.DataFrame, int]:
    if len(df) > max_records:
        return df, 0
    texts = df[text_col].astype(str).tolist()
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2)).fit_transform(texts)
    sim = cosine_similarity(vec)
    n = len(texts)
    to_drop = set()
    for i in range(n):
        if i in to_drop:
            continue
        similar = np.where(sim[i, i + 1:] >= threshold)[0]
        for idx in similar:
            j = i + 1 + idx
            to_drop.add(j)
    if not to_drop:
        return df, 0
    keep_mask = [i not in to_drop for i in range(len(df))]
    return df.loc[keep_mask].reset_index(drop=True), len(to_drop)


# -------------------------
# BERT loader (cached)
# -------------------------
@st.cache_resource(show_spinner=False)
def load_bert_model(name: str = "all-MiniLM-L6-v2"):
    if not BERT_AVAILABLE:
        return None
    return SentenceTransformer(name)


# -------------------------
# Training function (with min accuracy enforcement)
# -------------------------
def train_model(df: pd.DataFrame,
                text_col: str,
                label_col: str,
                feature_type: str = "tfidf",
                model_choice: str = "Logistic Regression",
                cv_folds: int = 5,
                sample_size: Optional[int] = None,
                embed_cache: bool = False,
                remove_near_dup_flag: bool = True,
                near_dup_threshold: float = 0.9,
                downsample_majority: bool = False,
                min_required_accuracy: float = 0.95) -> dict:
    df = df[[text_col, label_col]].dropna().rename(columns={text_col: "resume_text", label_col: "job_category"})
    df["job_category"] = df["job_category"].astype(str).apply(lambda x: canonicalize_label_value(x))
    df, dropped_exact = remove_exact_duplicates(df, "resume_text", "job_category")
    dropped_near = 0
    if remove_near_dup_flag:
        df, dropped_near = remove_near_duplicates(df, "resume_text", threshold=near_dup_threshold)
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)
    counts = df["job_category"].value_counts()
    tiny = counts[counts < 3]
    if not tiny.empty:
        st.warning(f"These labels have <3 samples and were removed: {list(tiny.index)}")
        df = df[df["job_category"].isin(counts[counts >= 3].index)]
    if downsample_majority:
        majority_limit = int(df.shape[0] / len(df["job_category"].unique()))
        pieces = []
        for label, group in df.groupby("job_category"):
            if len(group) > majority_limit:
                group = group.sample(majority_limit, random_state=42)
            pieces.append(group)
        df = pd.concat(pieces).sample(frac=1, random_state=42).reset_index(drop=True)
    df["processed"] = df["resume_text"].astype(str).apply(preprocess_text)
    X = df["processed"].values
    y = df["job_category"].astype(str).values
    X_train_text, X_test_text, y_train, y_test = safe_train_test_split_texts(X, y)

    train_set = set(X_train_text)
    test_set = set(X_test_text)
    overlap = train_set & test_set
    if overlap:
        st.warning(f"Warning: {len(overlap)} resumes appear in both train and test sets.")

    vectorizer = None
    bert_model_name = None

    if feature_type == "bert":
        if not BERT_AVAILABLE:
            raise RuntimeError("BERT requested but sentence-transformers not installed.")
        bert = load_bert_model()
        bert_model_name = "all-MiniLM-L6-v2"
        X_train = bert.encode(list(X_train_text), show_progress_bar=True, convert_to_numpy=True)
        X_test = bert.encode(list(X_test_text), show_progress_bar=True, convert_to_numpy=True)
    else:
        vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=5, max_df=0.8)
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)

    if model_choice == "Logistic Regression":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=42)
    elif model_choice == "Random Forest":
        clf = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight="balanced", random_state=42)
    elif model_choice == "Naive Bayes":
        if feature_type == "bert":
            raise ValueError("Naive Bayes incompatible with BERT.")
        clf = MultinomialNB()
    else:
        raise ValueError("Unknown model choice")

    cv_results = None
    if cv_folds and cv_folds >= 2:
        try:
            if feature_type == "tfidf":
                pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=5, max_df=0.8)), ("clf", clf)])
                skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                scores = cross_val_score(pipe, X_train_text, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
            else:
                skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                scores = cross_val_score(clf, X_train, y_train, cv=skf, scoring="accuracy", n_jobs=-1)
            cv_results = (float(scores.mean()), float(scores.std()))
        except Exception:
            cv_results = None

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    labels_sorted = sorted(pd.unique(y))
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

    if acc < min_required_accuracy:
        if os.path.exists(MODEL_STATE_PATH):
            os.remove(MODEL_STATE_PATH)
        st.error(f"Model accuracy {acc*100:.2f}% below minimum required {min_required_accuracy*100}%. Model not saved.")
        state = None
    else:
        if feature_type == "tfidf":
            state = {"pipeline": Pipeline([("tfidf", vectorizer), ("clf", clf)]),
                     "feature_type": feature_type, "labels": labels_sorted}
        else:
            state = {"model": clf, "feature_type": feature_type, "bert_model_name": bert_model_name, "labels": labels_sorted}
        with open(MODEL_STATE_PATH, "wb") as f:
            pickle.dump(state, f)
        st.success(f"Model saved with accuracy {acc*100:.2f}%")

    return {"state": state, "acc": acc, "report": report, "confusion_matrix": cm,
            "dropped_exact": dropped_exact, "dropped_near": dropped_near, "cv": cv_results}


# -------------------------
# Inference function (with adaptive keyword ensemble)
# -------------------------
def load_state() -> Tuple[Optional[dict], Optional[str]]:
    if not os.path.exists(MODEL_STATE_PATH):
        return None, "No saved model"
    try:
        with open(MODEL_STATE_PATH, "rb") as f:
            state = pickle.load(f)
        return state, None
    except Exception as e:
        return None, str(e)


def _keyword_count(processed_text: str, keywords: List[str]) -> int:
    if not keywords:
        return 0
    total = 0
    for k in keywords:
        kp = preprocess_text(k)
        if not kp:
            continue
        total += len(re.findall(r"\b" + re.escape(kp) + r"\b", processed_text))
    return total


def predict_from_state(state: dict, raw_text: str, top_k: int = 5):
    beta = keyword_beta
    alpha = max(0.0, 1.0 - beta)
    feature_type = state.get("feature_type", "tfidf")
    processed = preprocess_text(raw_text)

    if feature_type == "tfidf":
        pipe = state["pipeline"]
        try:
            probs = pipe.predict_proba([processed])[0]
            clf = pipe.named_steps["clf"]
            classes = list(clf.classes_)
        except Exception:
            try:
                preds = pipe.predict([processed])
                return preds[0], [(preds[0], 1.0)]
            except Exception:
                return None, []
    else:
        model = state.get("model")
        bert = load_bert_model()
        vec = bert.encode([processed], convert_to_numpy=True)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vec)[0]
            classes = list(model.classes_)
        else:
            preds = model.predict(vec)
            return preds[0], [(preds[0], 1.0)]

    counts = np.array([_keyword_count(processed, KEYWORD_MAP.get(cls, [])) for cls in classes], dtype=float)
    genai_count = _keyword_count(processed, KEYWORD_MAP.get("Generative AI", []))
    MIN_GENAI_KEYWORD_THRESHOLD = 1

    if genai_count >= MIN_GENAI_KEYWORD_THRESHOLD:
        rule_norm = counts.copy()
        max_c = rule_norm.max() if rule_norm.size > 0 else 0.0
        if max_c > 0:
            rule_norm = rule_norm / (max_c + 1e-12)
        else:
            rule_norm = np.zeros_like(rule_norm)
        model_probs = np.array(probs, dtype=float)
        combined = alpha * model_probs + beta * rule_norm
        labels_final = list(classes)
        scores_final = list(combined)

        if "Generative AI" not in labels_final:
            genai_norm = min(1.0, genai_count / 3.0)
            genai_score = beta * genai_norm
            labels_final.append("Generative AI (keyword-suggested)")
            scores_final.append(genai_score)

        arr = np.array(scores_final, dtype=float)
        if arr.sum() > 0:
            arr = arr / arr.sum()
        pairs = sorted(list(zip(labels_final, arr.tolist())), key=lambda x: x[1], reverse=True)[:top_k]
        top_label = pairs[0][0] if pairs else None
        return top_label, pairs
    else:
        pairs = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:top_k]
        top_label = pairs[0][0] if pairs else None
        return top_label, pairs


# -------------------------
# Streamlit UI with removed ensemble controls
# -------------------------
st.sidebar.header("Dataset & Model Controls")
page = st.sidebar.selectbox("Page", ["Load Dataset", "Train Model", "Screen Resume"])

if page == "Load Dataset":
    st.header("Load your resume CSV / Excel / zipped CSV")
    st.markdown("Upload dataset; the loader will attempt encoding/delimiter detection.")
    col1, col2 = st.columns(2)
    df = None
    diag = None
    with col1:
        uploaded_csv = st.file_uploader("Upload CSV/ZIP/XLSX", type=["csv", "zip", "gz", "xlsx", "xls"])
        if uploaded_csv:
            try:
                with st.spinner("Reading uploaded file..."):
                    df, diag = load_csv_from_bytes_auto(uploaded_csv)
                st.success(f"Loaded with {len(df)} rows; columns: {', '.join(df.columns)}")
            except Exception as e:
                st.error("Failed to load uploaded file.")
                st.text(str(e))
                df = None
    with col2:
        local_path = st.text_input("Local path (when running locally)", value="")
        if local_path:
            try:
                with st.spinner("Reading local file..."):
                    df_local, diag_local = load_csv_local(local_path)
                st.success(f"Loaded local with {len(df_local)} rows")
                df = df_local
                diag = diag_local
            except Exception as e:
                st.error(f"Failed to read local file: {e}")
                df = None
    if df is not None:
        st.subheader("Preview & pick columns")
        st.dataframe(df.head(10))
        text_candidates = [c for c in df.columns if any(k in c.lower() for k in ["resume", "text", "content", "description"])]
        label_candidates = [c for c in df.columns if any(k in c.lower() for k in ["category", "label", "job", "target", "class"])]
        default_text = text_candidates[0] if text_candidates else df.columns[0]
        default_label = label_candidates[0] if label_candidates else (df.columns[1] if len(df.columns) > 1 else df.columns[0])
        text_col = st.selectbox("Text column (resume content)", options=list(df.columns), index=list(df.columns).index(default_text))
        label_col = st.selectbox("Label column (job category)", options=list(df.columns), index=list(df.columns).index(default_label))
        st.session_state["loaded_df"] = df
        st.session_state["text_col"] = text_col
        st.session_state["label_col"] = label_col
        st.markdown("Diagnostics")
        st.write(diag)

elif page == "Train Model":
    st.header("Train model using loaded dataset")
    df = st.session_state.get("loaded_df", None)
    text_col = st.session_state.get("text_col", None)
    label_col = st.session_state.get("label_col", None)
    if df is None:
        st.warning("No dataset loaded — go to 'Load Dataset' first.")
        st.stop()
    st.subheader("Dataset summary")
    st.write(f"Rows: {len(df)}")
    try:
        st.dataframe(df[label_col].value_counts().rename_axis('label').reset_index(name='count'))
    except Exception:
        pass
    feature_type = st.radio("Feature type", options=["tfidf", "bert"], index=0)
    if feature_type == "bert" and not BERT_AVAILABLE:
        st.warning("BERT not installed — install sentence-transformers + torch to use BERT.")
    model_choice = st.selectbox("Classifier", ["Logistic Regression", "Random Forest", "Naive Bayes"])
    cv_folds = st.slider("CV folds (0=disable)", 0, 10, 5)
    sample_size = st.number_input("Subsample size (0 = all rows)", min_value=0, value=0, step=100)
    use_embed_cache = st.checkbox("Cache embeddings (BERT)", value=False)
    remove_near_dup_flag = st.checkbox("Remove near-duplicates", value=True)
    near_dup_threshold = st.slider("Near-duplicate threshold", 0.80, 0.995, 0.9, 0.01)
    downsample_majority = st.checkbox("Downsample majority classes", value=False)
    if st.button("Train now"):
        sample_arg = int(sample_size) if sample_size and sample_size > 0 else None
        try:
            with st.spinner("Training — may take time..."):
                res = train_model(df, text_col, label_col,
                  feature_type=feature_type,
                  model_choice=model_choice,
                  cv_folds=cv_folds,
                  sample_size=sample_arg,
                  embed_cache=use_embed_cache,
                  remove_near_dup_flag=remove_near_dup_flag,
                  near_dup_threshold=near_dup_threshold,
                  downsample_majority=downsample_majority,
                  min_required_accuracy=0.90)  

        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()
        if res["state"] is not None:
            st.success(f"Training finished — test accuracy: {res['acc']*100:.2f}%")
            if res["dropped_exact"]:
                st.warning(f"Dropped {res['dropped_exact']} exact duplicates.")
            if res["dropped_near"]:
                st.info(f"Removed {res['dropped_near']} near duplicates.")
            if res["cv"]:
                st.info(f"CV (train) mean ± std: {res['cv'][0]*100:.2f}% ± {res['cv'][1]*100:.2f}%")
            st.subheader("Classification report (test)")
            st.dataframe(pd.DataFrame(res["report"]).transpose().round(3), use_container_width=True)
            st.subheader("Confusion matrix (test)")
            try:
                import matplotlib.pyplot as plt
                import seaborn as sns
                fig, ax = plt.subplots(figsize=(6, max(4, len(res["state"]["labels"]) * 0.4)))
                sns.heatmap(res["confusion_matrix"], annot=True, fmt='d', cmap='Blues',
                            xticklabels=res["state"]["labels"], yticklabels=res["state"]["labels"], ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
            except Exception:
                st.write(res["confusion_matrix"])
        else:
            st.warning("Model not saved due to low accuracy. Train with more data or adjust parameters.")
    if st.button("Delete saved model"):
        if os.path.exists(MODEL_STATE_PATH):
            os.remove(MODEL_STATE_PATH)
            st.success("Saved model deleted.")
        else:
            st.info("No saved model found.")

else:
    st.header("Screen a resume using trained model")
    state, err = load_state()
    if err:
        st.error(f"No saved model: {err} — Train a model first.")
        st.stop()
    st.markdown(f"*Model features:* {state.get('feature_type')}  •  *Labels:* {', '.join(state.get('labels', []))}")
    uploaded_resume = st.file_uploader("Upload resume (pdf/docx/txt) to classify", type=["pdf", "docx", "txt"])
    preview_chars = st.slider("Preview characters", min_value=500, max_value=10000, value=2000, step=500)
    if uploaded_resume:
        text = extract_text_from_uploaded_file(uploaded_resume)
        if not text.strip():
            st.warning("Could not extract text from this file (maybe scanned PDF).")
        else:
            st.subheader("Preview (short slice)")
            st.text_area("Preview", value=text[:preview_chars], height=200, disabled=True)
            with st.expander("Show full extracted resume"):
                st.text_area("Full resume", value=text, height=600, disabled=True)
                st.download_button("Download extracted text", data=text, file_name=f"{os.path.splitext(uploaded_resume.name)[0]}_extracted.txt")
            try:
                pred, probs = predict_from_state(state, text, top_k=5)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                pred, probs = None, []
            if pred is None:
                st.warning("Nothing to predict (text empty after preprocessing).")
            else:
                st.success(f"Predicted category: *{pred}*")
                if probs:
                    st.subheader("Top probabilities / combined scores")
                    for i, (label, p) in enumerate(probs):
                        if i == 0:
                            st.markdown(f"🥇 {label} — {p*100:.1f}%")
                    else:
                        st.write(f"{label} — {p*100:.1f}%")
                        st.progress(int(p * 100))
