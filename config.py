import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "docs"
CHROMA_DIR = BASE_DIR / ".chroma_db"

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
LLM_MODEL: str = "llama-3.3-70b-versatile"
LLM_TEMPERATURE: float = 0.0

EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE: int = 1000
CHUNK_OVERLAP: int = 150
RETRIEVAL_TOP_K: int = 4
MEMORY_WINDOW: int = 3
CHROMA_COLLECTION: str = "agentic_rag_docs"