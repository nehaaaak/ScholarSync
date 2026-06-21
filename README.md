# 📚 ScholarSync — AI-Powered Research Assistant

ScholarSync is an end-to-end **AI-powered research assistant** designed to streamline academic paper discovery, management, and understanding.  
ScholarSync is a full-stack **AI research assistant** that helps users discover research papers, interact with them using **RAG-based Q&A**, and generate **structured AI summaries exported to Notion**.

Built to streamline academic research workflows, with a **FastAPI backend**, **Streamlit frontend**, using modern GenAI tooling.

## 🎯 Why I Built This

While working on my final year project, I found myself spending a lot of time searching across multiple research platforms, downloading papers, and trying to understand lengthy research articles. The process was fragmented and time-consuming, especially when I wanted quick answers or summaries before deciding whether a paper was relevant.

I built ScholarSync to simplify this workflow by bringing paper discovery, semantic search, AI-powered Q&A, and structured summaries into a single application. It also became my first end-to-end GenAI project, where I explored how Retrieval-Augmented Generation (RAG) could be integrated into a practical application rather than building a standalone chatbot.


## 👨‍💻 My Contribution

ScholarSync was developed as an individual project.

I was responsible for designing and implementing the complete application, including:

* Developing the FastAPI backend and Streamlit frontend.
* Building the complete RAG pipeline, from PDF ingestion and chunking to embedding generation and semantic retrieval using Qdrant.
* Integrating Google Gemini for research Q&A and AI-generated summaries.
* Designing LangFlow workflows to automate paper summarization and export structured outputs to Notion.
* Integrating multiple external APIs, including arXiv, Semantic Scholar, PapersWithCode, Qdrant Cloud, and Notion.
* Optimizing the application for free-tier LLM usage by experimenting with chunking strategies, prompt design, and token usage.

This project became my first complete AI application, where I learned how different GenAI components work together to solve a real-world problem.


## ⚡ Challenges & Key Learnings

Building ScholarSync involved much more than integrating an LLM.

### Challenges

* Designing the Retrieval-Augmented Generation (RAG) pipeline required understanding how document chunking, embeddings, vector search, and prompt construction affect the quality of retrieved information.
* Integrating multiple external services such as Qdrant, Gemini, LangFlow, Notion, and research paper APIs meant working with different request formats, and error handling while keeping the workflow reliable.
* Automating Notion exports was more challenging than expected because the generated JSON had to exactly match the database schema and property structure expected by the Notion API.
* Optimizing the application for free-tier LLM usage required experimenting with chunk sizes, prompt design, and token usage to balance response quality, latency, and API limits.

### What I Learned

Working on ScholarSync gave me my first practical experience building an end-to-end AI application rather than an isolated AI feature.

Some of my key learnings include:

* Built and understood a complete RAG pipeline, including document ingestion, chunking, embeddings, semantic retrieval, and context-aware prompting.
* Learned how vector databases like Qdrant improve information retrieval compared to keyword search.
* Gained practical experience integrating LLMs into backend applications while dealing with latency, token limits, and API constraints.
* Understood the importance of retrieval quality, evaluation, modular architecture, and production considerations while building AI systems.

This project also highlighted several areas for improvement, including retrieval evaluation, multi-user support, and a more modular backend architecture. I am currently redesigning the project with these improvements in mind.

---

**📢 Project Status**

This repository contains the original version of ScholarSync, which was built as my first end-to-end GenAI application. While it demonstrates the core ideas behind the project, I identified several areas for improvement as I gained more experience with AI engineering.
I am currently developing a completely redesigned version with a more modular architecture, improved RAG pipeline, multi-user support, better retrieval quality, and additional production-oriented features. Since these changes require a significant redesign, the new implementation is being developed in a separate repository.

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

```text
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
```

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
```env
GOOGLE_API_KEY=your_gemini_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
NOTION_API_KEY=your_notion_key
NOTION_DATABASE_ID=your_database_id
LANGFLOW_URL=http://localhost:7860
LANGFLOW_FLOW_ID=your_flow_id
```

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
