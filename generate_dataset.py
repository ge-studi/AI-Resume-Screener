#!/usr/bin/env python3
"""
generate_non_overfitting_dataset.py

Generates one CSV (default: non_overfitting_resumes.csv) containing synthetic
resume-like texts designed to avoid trivial keyword leakage so models do not
reach 100% accuracy by memorizing tokens.

Columns: id, resume_text, category, group_id

Run:
    python generate_non_overfitting_dataset.py
"""
import random
import csv
import re
from typing import List, Dict
import pandas as pd
import numpy as np

# -------------------------
# Config (tweak if desired)
# -------------------------
OUTFILE = "non_overfitting_resumes.csv"
N_SAMPLES = 10000        # total rows
BALANCED = True          # equal samples per class when True
DIFFICULTY = "hard"      # options: "easy","medium","hard" (controls overlap/noise)
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

PRESETS = {
    "easy":   dict(cross_bleed=0.08, paraphrase_prob=0.12, drop_role_token=0.08, noise_token_rate=0.03),
    "medium": dict(cross_bleed=0.18, paraphrase_prob=0.25, drop_role_token=0.18, noise_token_rate=0.06),
    "hard":   dict(cross_bleed=0.32, paraphrase_prob=0.40, drop_role_token=0.30, noise_token_rate=0.10),
}
cfg = PRESETS.get(DIFFICULTY, PRESETS["hard"])

# -------------------------
# Roles and token pools
# -------------------------
ROLES = ["Data Scientist", "ML Engineer", "Data Engineer", "AI Researcher", "Generative AI"]

ROLE_TOKENS: Dict[str, List[str]] = {
    "Data Scientist": ["pandas", "scikit-learn", "regression", "eda", "statistical modeling", "visualization"],
    "ML Engineer":    ["serving", "latency", "tensorflow", "pytorch", "model registry", "ci/cd"],
    "Data Engineer":  ["spark", "etl", "kafka", "data warehousing", "sql", "schema"],
    "AI Researcher":  ["ablation", "paper", "theory", "benchmark", "transformer", "self-supervised"],
    "Generative AI":  ["llm", "prompt", "fine-tune", "diffusion", "inpainting", "text-to-image"],
}

SHARED_TOKENS = [
    "python", "api", "docker", "aws", "gcp", "sql", "monitoring", "testing",
    "feature engineering", "pipeline", "automation", "experiment", "cloud"
]

SHARED_PROJECTS = [
    "built scalable data pipelines with retries and monitoring",
    "implemented model evaluation and automated tests",
    "deployed services as containers with CI/CD",
    "created dashboards to surface business KPIs",
    "designed metric-driven experiments and tracked results",
]

ROLE_PROJECTS = {
    "Data Scientist": [
        "developed forecasting models and validated with backtesting",
        "ran feature selection and cross-validation at scale"
    ],
    "ML Engineer": [
        "productionized model serving with autoscaling and caching",
        "optimized inference latency via quantization and batching"
    ],
    "Data Engineer": [
        "designed ETL jobs and partitioning for low-latency queries",
        "managed streaming pipelines and schema evolution"
    ],
    "AI Researcher": [
        "ran reproducible experiments and released code with ablations",
        "wrote papers describing novel architectures and results"
    ],
    "Generative AI": [
        "built retrieval-augmented generation flows and safety filters",
        "fine-tuned generative models and evaluated outputs qualitatively"
    ],
}

PARAPHRASE_MAP = {
    "llm": ["large language model", "neural language model"],
    "gpt": ["large language model", "transformer-based model"],
    "spark": ["distributed compute engine", "cluster compute framework"],
    "kafka": ["streaming messaging system", "log-based streaming system"],
    "fine-tune": ["adaptation of a pretrained model", "domain-specific tuning"],
    "prompt": ["input crafting", "instruction design"],
    "quantization": ["model compression", "reduced-precision optimization"],
}

NOISE_TOKENS = ["internship", "open-source", "mentor", "cross-functional", "POC", "on-call", "sla", "collab"]

# -------------------------
# Helper functions
# -------------------------
def pick_role_tokens(role: str, n_role: int, cross_bleed_prob: float) -> List[str]:
    tokens = []
    role_pool = ROLE_TOKENS.get(role, [])
    for _ in range(n_role):
        if random.random() < cross_bleed_prob:
            other = random.choice([r for r in ROLES if r != role])
            t = random.choice(ROLE_TOKENS[other])
        else:
            t = random.choice(role_pool)
        tokens.append(t)
    return tokens

def maybe_paraphrase(token: str, p: float) -> str:
    low = token.lower()
    for k, vlist in PARAPHRASE_MAP.items():
        if k in low and random.random() < p:
            return random.choice(vlist)
    return token

def assemble_resume(role: str, cfg: dict, template_id: int = 0) -> str:
    cross_bleed = cfg["cross_bleed"]
    paraphrase_p = cfg["paraphrase_prob"]
    drop_p = cfg["drop_role_token"]
    noise_rate = cfg["noise_token_rate"]

    # header sometimes contains role in neutral form
    header = f"{role} — experience building ML & data products." if random.random() < 0.22 else ""

    n_shared = random.randint(2, 4)
    n_role = random.randint(2, 4)
    shared = random.sample(SHARED_TOKENS, k=n_shared)
    role_tokens = pick_role_tokens(role, n_role, cross_bleed)

    # drop some role tokens randomly to avoid perfect signals
    role_tokens = [t for t in role_tokens if random.random() > drop_p]

    # paraphrase role/shared tokens with some probability
    role_tokens = [maybe_paraphrase(t, paraphrase_p) for t in role_tokens]
    shared = [maybe_paraphrase(t, paraphrase_p) for t in shared]

    # projects mixture
    n_proj = random.randint(1, 2)
    proj_shared = random.sample(SHARED_PROJECTS, k=n_proj)
    proj_role = random.sample(ROLE_PROJECTS.get(role, []), k=1) if ROLE_PROJECTS.get(role) else []
    projects = proj_shared + proj_role
    projects = [maybe_paraphrase(p, paraphrase_p) for p in projects]

    # sentence templates and building
    templates = [
        "Worked on {items}. Delivered: {projects}.",
        "Core skills: {items}. Recent work includes {projects}.",
        "Contributed to {projects} and owned {items}.",
        "Responsibilities: {items}. Achievements: {projects}."
    ]
    sentences = []
    n_sent = random.randint(2, 4)
    for _ in range(n_sent):
        tmpl = random.choice(templates)
        pool = shared + role_tokens
        if not pool:
            pool = shared
        k_items = random.randint(2, min(4, len(pool)))
        items = random.sample(pool, k=k_items)
        proj = random.choice(projects) if projects else ""
        sent = tmpl.format(items=", ".join(items), projects=proj)
        sentences.append(sent)

    # occasional noise fragments
    if random.random() < noise_rate:
        n_noise = random.randint(1, 3)
        noise_frags = random.sample(NOISE_TOKENS, k=n_noise)
        sentences += [f"Involved in {frag}." for frag in noise_frags]

    # optional extra paragraph for variability
    if random.random() < 0.28:
        sentences.append(random.choice([
            "Collaborated closely with product and platform teams to define SLAs.",
            "Focused on robust monitoring, logging, and reproducibility.",
            "Mentored junior engineers and contributed to open-source tooling."
        ]))

    random.shuffle(sentences)
    body = " ".join(sentences)

    # casing randomness and tiny abbreviations/typos
    if random.random() < 0.18:
        body = body.lower()
    if random.random() < 0.06:
        body = re.sub(r"\bmodel\b", "mdl", body)
        body = re.sub(r"\bengineering\b", "eng", body)

    return (header + " " + body).strip() if header else body

# -------------------------
# Dataset generator
# -------------------------
def generate_dataset(n_samples: int, balanced: bool = True) -> pd.DataFrame:
    rows = []
    per_role = n_samples // len(ROLES) if balanced else None
    idx = 0
    # simple group buckets per role for later group-aware splitting if needed
    GROUP_BUCKETS = 100

    if balanced:
        for role in ROLES:
            for j in range(per_role):
                group_bucket = (j % GROUP_BUCKETS)
                text = assemble_resume(role, cfg, template_id=group_bucket)
                rows.append({
                    "id": idx,
                    "resume_text": text,
                    "category": role,
                    "group_id": (ROLES.index(role) * GROUP_BUCKETS + group_bucket)
                })
                idx += 1
    else:
        for i in range(n_samples):
            role = random.choice(ROLES)
            group_bucket = (i % GROUP_BUCKETS)
            text = assemble_resume(role, cfg, template_id=group_bucket)
            rows.append({
                "id": idx,
                "resume_text": text,
                "category": role,
                "group_id": (ROLES.index(role) * GROUP_BUCKETS + group_bucket)
            })
            idx += 1

    # fill remainder if any
    while len(rows) < n_samples:
        role = random.choice(ROLES)
        group_bucket = (len(rows) % GROUP_BUCKETS)
        text = assemble_resume(role, cfg, template_id=group_bucket)
        rows.append({
            "id": len(rows),
            "resume_text": text,
            "category": role,
            "group_id": (ROLES.index(role) * GROUP_BUCKETS + group_bucket)
        })

    df = pd.DataFrame(rows)
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return df

# -------------------------
# Run and save CSV
# -------------------------
if __name__ == "__main__":
    print(f"Generating {N_SAMPLES} samples (balanced={BALANCED}, difficulty={DIFFICULTY}) ...")
    df = generate_dataset(N_SAMPLES, balanced=BALANCED)
    df.to_csv(OUTFILE, index=False, quoting=csv.QUOTE_MINIMAL)
    print(f"Saved {len(df)} rows to {OUTFILE}")
    print("Class distribution:")
    print(df["category"].value_counts())
    print("\nExample row:")
    print(df.iloc[0].to_dict())
