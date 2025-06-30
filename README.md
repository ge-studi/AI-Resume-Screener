# 🤖 AI Resume Screener

This project uses machine learning (TF-IDF and BERT-based models) to classify resumes into job categories such as Data Science, Web Development, DevOps, etc. It also extracts top skills and predicts job fit with confidence levels.

---

## 📌 Features

- Upload resumes in `.pdf`, `.docx`, or `.txt` format
- Train models on custom or sample data
- Choose between feature extraction methods: TF-IDF or BERT
- Select ML models: Logistic Regression, Random Forest, or Naive Bayes
- Evaluate performance using Accuracy, Confusion Matrix, and Classification Report
- Predict job category and confidence for uploaded resumes
- Visualize performance metrics directly in Streamlit

---

## 🚀 How to Run

Clone this repository

git clone https://github.com/ge-studi/AI-Resume-Screener

cd AI-Resume-Screener

Install dependencies

pip install -r requirements.txt

Make sure to install NLTK and download stopwords:


import nltk

nltk.download('punkt')

nltk.download('stopwords')

Start Streamlit

streamlit run resume_screener.py


🧠 Models Supported
---
Feature Type	Model	Accuracy (Sample Data)

TF-IDF	Logistic Regression	99.48%

TF-IDF	Random Forest	99.48%

TF-IDF	Naive Bayes	96.89%

BERT	Logistic Regression	66.67%

BERT	Random Forest	99.48%


🖼 Architecture
---
![Architecture](architecture.png)
---

📂 Folder Structure
---
resume-screener/

├── resume_screener.py        # Main Streamlit app

├── requirements.txt          # Python dependencies

├── README.md                 # Project documentation

├── architecture.png          # Architecture diagram

└── resume_model.pkl          # Saved model after training (auto-generated)



🔖 Sample Job Categories
---
Software Development

Web Development

DevOps

Digital Marketing

HR

Finance

Sales

Business Analyst

Project Management


📃 License
---
This project is licensed under the MIT License.


📩 Contact
---
For suggestions, improvements, or demo requests:

📧 gssingh6393@gmail.com

🔗 https://www.linkedin.com/in/geetanjali--singh/


🧑‍💻 Author
---
Developed with ❤️ by Geetanjali 

Feel free to fork, contribute, or raise issues.




