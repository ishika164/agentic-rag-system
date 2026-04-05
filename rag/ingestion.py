from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import (
    CHROMA_DIR,
    CHROMA_COLLECTION,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)


class DocumentIngestor:

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

    def __init__(self) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            add_start_index=True,
        )
        self._embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )
        self._vector_store = Chroma(
            collection_name=CHROMA_COLLECTION,
            embedding_function=self._embeddings,
            persist_directory=str(CHROMA_DIR),
        )

    def ingest_file(self, file_path: str | Path) -> int:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {path}")
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        logger.info("Loading document: %s", path.name)
        docs = self._load(path)
        chunks = self._splitter.split_documents(docs)
        self._vector_store.add_documents(chunks)
        logger.info("Ingested %d chunks from '%s'", len(chunks), path.name)
        return len(chunks)

    def ingest_directory(self, directory: str | Path) -> int:
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")
        files = [
            f for f in dir_path.iterdir()
            if f.suffix.lower() in self.SUPPORTED_EXTENSIONS
        ]
        if not files:
            logger.warning("No supported documents found in '%s'", dir_path)
            return 0
        total = sum(self.ingest_file(f) for f in files)
        logger.info("Total chunks ingested: %d", total)
        return total

    @property
    def vector_store(self) -> Chroma:
        return self._vector_store

    def collection_size(self) -> int:
        return self._vector_store._collection.count()

    def _load(self, path: Path) -> List[Document]:
        if path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(path))
        else:
            loader = TextLoader(str(path), encoding="utf-8")
        return loader.load()