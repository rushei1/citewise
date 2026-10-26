# 🧠 Lazy Researcher Toolkit

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-%20Backend-success)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)

A full-stack AI-powered application that automatically fetches, summarizes, clusters, and formats research papers into a structured literature review — saving researchers hours of manual effort.

---

## 🚀 Features

- 🔎 Search for papers by research topic
- 🤖 Auto-summarization using Transformers
- 🧠 Semantic clustering of related research
- 📄 Expandable summaries with reference links
- 🧃 PDF export of review clusters
- 💾 Save/load past research reviews
- 🧠 Powered by `sentence-transformers`, `transformers`, `scikit-learn`

---

## 📸 Demo

![UI Screenshot](demo_screenshot.png)

---

## 📦 Tech Stack

| Layer        | Tech Used                          |
|--------------|------------------------------------|
| Frontend     | Streamlit                          |
| Backend      | FastAPI, Hugging Face Transformers |
| Clustering   | SentenceTransformers + KMeans      |
| PDF Export   | ReportLab                          |
| Data Format  | arXiv Papers via API               |
| Persistence  | JSON (for review history)          |

---

## ⚙️ How It Works

1. Enter a research topic.
2. Backend fetches relevant papers via APIs.
3. Each paper is summarized using a transformer.
4. KMeans clusters similar papers.
5. Results are displayed in expandable cards with direct links.
6. You can export everything as a clean PDF.

---

## 🛠️ Setup Instructions

1. **Clone this repo**
   ```
   git clone https://github.com/rushei1/lazy-researcher-toolkit.git
   cd lazy-researcher-toolkit
2. Create and activate virtual environment
    ```
    python -m venv venv
    source venv/bin/activate      # Linux/macOS
    .\venv\Scripts\activate       # Windows
3. Install dependencies
    ```
    pip install -r backend/requirements.txt
    pip install -r frontend/requirements.txt
4. Start the backend (FastAPI)
    ```
    cd backend
    uvicorn main:app --reload
5. Run the frontend (Streamlit)

    Open a new terminal:
    ```
    cd frontend
    streamlit run app.py
---

## 📁 Project Structure
    lazy-researcher-toolkit/
    │
    ├── backend/
    │   ├── cluster.py            # KMeans clustering with SBERT
    │   ├── summarize.py          # Paper summarization using HuggingFace
    │   ├── main.py               # FastAPI backend routes
    │   └── requirements.txt
    │
    ├── frontend/
    │   ├── app.py                # Streamlit UI
    │   ├── history/              # Saved review JSONs
    │   └── requirements.txt
    │
    ├── README.md
    └── .gitignore

---

## ⚠️ Limitations (v1.0)

Baseline the upcoming upgrade is measured against:

- BART truncates input to 1024 characters, so long abstracts are cut off.
- The number of clusters is fixed at 3, and the app crashes when fewer than 3 papers come back.
- arXiv is the only paper source.
- There is no evaluation of summary or cluster quality.
- There are no tests.
- Review history lives on the client filesystem as JSON files named after the topic.

---

## 💡 Future Enhancements
- Add support for saving reviews to cloud (e.g., Firebase / Supabase)
- Model fine-tuning for domain-specific summarization
- Login + review history per user
- Add citation manager integration (BibTeX export)
---
## ✨ Credits
Built by **Rusheil Singh Baath**.  
Inspired by the pain of writing literature reviews manually