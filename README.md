# Agentic RAG System

A production-style Retrieval-Augmented Generation (RAG) system with agentic 
routing built with LangChain, ChromaDB, and Groq (free LLM).

---

## What it does

- Loads your documents and stores them in a vector database
- When you ask a question, it **decides** whether to search the documents or answer directly
- Maintains memory of last 3 conversations for follow-up questions
- Shows you exactly what decision it made and which sources it used

---

## Project Structure

agentic_rag/
├── main.py                  # Entry point
├── config.py                # All configuration
├── requirements.txt         # Dependencies
├── .env                     # Your API key (not committed)
│
├── rag/
│   ├── ingestion.py         # Load → chunk → embed → store
│   ├── retriever.py         # Search relevant chunks
│   └── chain.py             # Generate answers with LLM
│
├── agent/
│   ├── decision.py          # Decide RAG vs DIRECT
│   └── orchestrator.py      # Connect everything
│
├── memory/
│   └── conversation.py      # Remember last 3 conversations
│
├── cli/
│   └── interface.py         # Interactive CLI
│
├── utils/
│   └── logging_setup.py     # Logging
│
└── docs/                    # Put your documents here

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/agentic-rag-system.git
cd agentic-rag-system
```

### 2. Create virtual environment
```bash
python -m venv .venv

# Mac/Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
pip install langchain-huggingface sentence-transformers langchain-groq
```

### 4. Set up API key
Create a `.env` file:

GROQ_API_KEY=your_groq_api_key_here

Get your free Groq API key at: https://console.groq.com

### 5. Add your documents
Place your PDF or `.txt` files inside the `docs/` folder.

### 6. Run
```bash
python main.py
```

---

## CLI Commands

| Command | Description |
|---|---|
| `/ingest <path>` | Ingest a file or directory |
| `/status` | Show number of stored chunks |
| `/reset` | Clear conversation memory |
| `/help` | Show all commands |
| `/quit` | Exit |

---

## Example Session

You ▶ What is RAG?
│  RAG stands for Retrieval Augmented Generation. It combines
│  document search with LLM generation to give grounded answers.
├─ Metadata
│  Routing     : RAG
│  Retrieval   : Yes
│  Sources     : sample.txt
You ▶ What is the capital of France?
│  The capital of France is Paris.
├─ Metadata
│  Routing     : DIRECT
│  Retrieval   : No
│  Sources     : —
You ▶ Tell me more about that
│  (uses memory of previous exchange)
├─ Metadata
│  Routing     : DIRECT
│  Retrieval   : No
│  Sources     : —

---

## Design Decisions

### 1. LLM-based routing
Used an LLM classifier to decide RAG vs DIRECT instead of keywords.
This handles follow-up questions and paraphrasing correctly.

### 2. Free & Fast LLM — Groq
Used Groq (free tier) with `llama-3.3-70b-versatile` model instead
of paid OpenAI. Fast and completely free for testing.

### 3. Free Embeddings — HuggingFace
Used `sentence-transformers/all-MiniLM-L6-v2` for embeddings.
No API key needed, runs locally.

### 4. Safe routing default
If classifier is unsure, defaults to RAG to never miss context.

### 5. Sliding window memory
Keeps last 3 conversation pairs using a deque for O(1) performance.

---

## Limitations

- Memory resets when you restart the program
- Only supports PDF and TXT files currently
- No streaming — answers appear all at once
- Single user only

---

## Tech Stack

| Tool | Purpose |
|---|---|
| LangChain | RAG pipeline framework |
| ChromaDB | Vector database |
| Groq | Free LLM API |
| HuggingFace | Free embeddings |
| Python 3.11 | Language |

---

Built for AllyNerds AI Engineer Assignment — April 2026