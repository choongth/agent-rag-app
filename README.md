# 📚 SmartDoc Assistant

### AI Agent + RAG — Ask questions from your own documents, get cited answers

<div align="center">

![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![Railway](https://img.shields.io/badge/Railway-0B0D0E?style=for-the-badge&logo=railway&logoColor=white)
![DeepSeek](https://img.shields.io/badge/DeepSeek_LLM-4D6BFF?style=for-the-badge&logo=openai&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B35?style=for-the-badge&logo=databricks&logoColor=white)

</div>

---

## ✨ What It Does

Upload any **PDF, TXT, or Markdown** file and ask questions about it in natural language. SmartDoc Assistant retrieves the most relevant passages using a production-grade hybrid RAG pipeline, then streams a cited answer back to you — token by token, live.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  🖥️  Streamlit Frontend  (port 8501)                            │
│  • Upload PDF / TXT / MD                                        │
│  • Streaming chat (tokens rendered live via SSE)                │
└────────────────────────┬────────────────────────────────────────┘
                         │ HTTP / SSE
┌────────────────────────▼────────────────────────────────────────┐
│  ⚡ FastAPI Backend  (port 8000)                               │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  🤖 SmartDocAgent  (agent.py)                           │   │
│  │                                                         │   │
│  │  Phase 1 — tool calling (non-streaming)                 │   │
│  │    DeepSeek decides → calls search_documents tool       │   │
│  │                                                         │   │
│  │  Phase 2 — synthesis (streaming via SSE)                │   │
│  │    DeepSeek reads chunks → streams answer token by token│   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                    │
│  ┌────────────────────────▼────────────────────────────────┐   │
│  │  🔍 RAGPipeline  (rag.py)  — 3-stage retrieval          │   │
│  │                                                         │   │
│  │  Stage 1 — Hybrid Retrieval                             │   │
│  │    ├─ 🧠 Semantic search  (SentenceTransformer + HNSW)  │   │
│  │    └─ 🔤 Keyword search   (BM25)                        │   │
│  │              ↓  RRF fusion → top-10 candidates          │   │
│  │                                                         │   │
│  │  Stage 2 — Reranking                                    │   │
│  │    ⚖️  CrossEncoder scores (query, chunk) pairs jointly │   │
│  │              ↓  top-5 most relevant chunks              │   │
│  │                                                         │   │
│  │  Stage 3 — Feed to Agent                                │   │
│  └────────────────────────┬────────────────────────────────┘   │
│                           │                                    │
│  ┌────────────────────────▼────────────────────────────────┐   │
│  │  🗄️  ChromaDB  (persistent on disk)                     │   │
│  │  • HNSW index  — fast approximate vector search         │   │
│  │  • SQLite      — stores chunk text + metadata           │   │
│  └─────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

---

## 🔄 RAG Pipeline Detail

```
📥 INDEXING (once per document)
────────────────────────────────
PDF / TXT / MD
  → extract text
  → chunk (400 words, 50-word overlap)
  → SentenceTransformer encode  →  vectors stored in ChromaDB
  → tokenize                    →  added to BM25 in-memory index

💬 QUERYING (every question)
──────────────────────────────
User question
  ├─ 🧠 SentenceTransformer → semantic search   (top-10)  ──┐
  └─ 🔤 BM25 tokenize       → keyword search    (top-10)  ──┤
                                                             │
                             ⚡ RRF fusion (Reciprocal Rank) ◄┘
                                   top-10 candidates
                                         │
                             ⚖️  CrossEncoder rerank
                             (reads query + chunk together)
                                   top-5 chunks
                                         │
                                   🤖 DeepSeek LLM
                                  (streams answer)
```

> **Why hybrid?** Semantic search understands meaning but misses exact matches (names, codes, numbers). BM25 catches exact keywords but misses paraphrases. Together they cover both.

> **Why rerank?** The bi-encoder encodes query and document independently — fast but imprecise. The cross-encoder reads both together — slower but far more accurate. Running it on only 10 candidates keeps latency acceptable.

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| 🤖 LLM | DeepSeek `deepseek-chat` | Reasoning + answer generation |
| 🧠 Embedding model | `all-MiniLM-L6-v2` (local) | Text → 384-dim vectors |
| 🔤 Keyword search | `rank-bm25` (in-memory) | Exact keyword matching |
| ⚖️ Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (local) | Precise relevance scoring |
| 🗄️ Vector DB | ChromaDB (HNSW + SQLite, persistent) | Vector storage and ANN search |
| ⚡ Backend | FastAPI + Uvicorn | REST API + SSE streaming |
| 🖥️ Frontend | Streamlit | Chat UI with live token streaming |
| 🐳 Containerisation | Docker + Docker Compose | Reproducible deployment |
| ☁️ Hosting | Railway | Cloud deployment with persistent volume |

---

## 🚀 Quick Start — Docker

```bash
# 1. Copy and fill in your API key
cp .env.example .env
# Edit .env: DEEPSEEK_API_KEY=sk-...

# 2. Build and run (first build takes ~5 min — downloads ML models)
docker compose up --build

# 3. Open in browser
#    💬 Chat UI  → http://localhost:8501
#    📖 API docs → http://localhost:8000/docs
```

---

## 💻 Quick Start — Local Development

```bash
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate          # Windows
# source venv/bin/activate       # macOS / Linux

# 2. Install dependencies
pip install -r backend/requirements.txt -r frontend/requirements.txt

# 3. Set your API key
cp .env.example .env
# Edit .env: DEEPSEEK_API_KEY=sk-...

# Terminal 1 — backend
cd backend
uvicorn main:app --reload

# Terminal 2 — frontend
cd frontend
streamlit run app.py
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/upload` | Upload & index a document (PDF / TXT / MD) |
| `GET` | `/documents` | List indexed documents |
| `DELETE` | `/documents/{filename}` | Remove a document and its chunks |
| `POST` | `/chat` | Chat (returns full response) |
| `POST` | `/chat/stream` | Chat with SSE token streaming |

---

## 📁 Project Structure

```
agent_rag_app/
├── 🐍 backend/
│   ├── main.py          # FastAPI app — all endpoints
│   ├── agent.py         # DeepSeek tool-calling agent + streaming
│   ├── rag.py           # Hybrid retrieval + reranking pipeline
│   ├── requirements.txt
│   └── Dockerfile
├── 🖥️ frontend/
│   ├── app.py           # Streamlit chat UI
│   ├── requirements.txt
│   └── Dockerfile
├── 🐳 docker-compose.yml
├── .env.example
└── README.md
```
