# 📚 ScholarSync — AI-Powered Research Assistant

ScholarSync is an end-to-end **AI-powered research assistant** designed to streamline academic paper discovery, management, and understanding.  
ScholarSync is a full-stack **AI research assistant** that helps users discover research papers, interact with them using **RAG-based Q&A**, and generate **structured AI summaries exported to Notion**.

Built to streamline academic research workflows, with a **FastAPI backend**, **Streamlit frontend**, using modern GenAI tooling.

---

## ✨ Key Features

### 🔍 **Research Paper Discovery**
- Multi-source paper search (arXiv, Semantic Scholar, PapersWithCode)
- Advanced sorting: relevance, recency, citation count, and code availability

### 📄 **Paper Management**
- One-click PDF download and local storage
- Bookmarking system for quick access

### 💬 **RAG-based Q&A**
  - Chunk-based PDF text extraction
  - Per-paper semantic Q&A
  - Global Q&A across all downloaded papers
- **Vector search with Qdrant** for accurate context retrieval

### 💬 **Interactive AI Assistance**
- **Per-paper Q&A** using retrieved chunks
- **Global research assistant** across all embedded papers
- Low-latency responses optimized for free-tier LLM usage

### 📝 **AI Summarization → Notion**
  - LangFlow-orchestrated summarization pipeline
  - Structured summaries (TL;DR, Methodology, Results, Limitations, etc.)
  - Automatic export to **Notion**

---

## 🛠️ Tech Stack

**Frontend**
- Streamlit

**Backend**
- FastAPI
- Python

**AI / GenAI**
- Google Gemini (LLM - 2.5 Flash)
- RAG (Retrieval-Augmented Generation)

**Vector DB**
- Qdrant Cloud (Free Tier)

**Orchestration & Automation**
- LangFlow
- API-triggered LLM workflows

**External Integrations**
- arXiv API
- Semantic Scholar API
- PapersWithCode
- Notion API

---

## 🏗️ Architecture Overview

Streamlit UI
↓
FastAPI Backend
↓
PDF Processor → Chunking
↓
Qdrant (Vector Store)
↓
Gemini LLM (RAG)
↓
LangFlow (Summarization Orchestration)
↓
Notion API (Knowledge Base)

---

## 🚀 Getting Started (Local)

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/scholarsync.git
cd scholarsync
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows: source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Variables
Create a .env file:
GOOGLE_API_KEY=your_gemini_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
NOTION_API_KEY=your_notion_key
NOTION_DATABASE_ID=your_database_id
LANGFLOW_URL=http://localhost:7860
LANGFLOW_FLOW_ID=your_flow_id

### 5. Run Backend
```bash
cd backend
uvicorn main:app --reload
```

### 6. Run Frontend
```bash
streamlit run app.py
```
⚠️ This project is currently designed for local, single-user usage.

---

## 📌 Project Status

- Core RAG Q&A completed
- LangFlow + Notion summarization integrated
- Ongoing optimization (latency & cost efficiency)

---

## 🧠 Future Improvements

- Multi-user authentication
- Better memory management
- Recommendations & trend analysis
