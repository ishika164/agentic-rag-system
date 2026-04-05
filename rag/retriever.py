from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import RETRIEVAL_TOP_K

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    chunks: List[Document]
    sources: List[str] = field(default_factory=list)

    def format_context(self) -> str:
        return "\n\n---\n\n".join(c.page_content for c in self.chunks)

    def __bool__(self) -> bool:
        return bool(self.chunks)


class DocumentRetriever:

    def __init__(self, vector_store: Chroma, top_k: int = RETRIEVAL_TOP_K) -> None:
        self._store = vector_store
        self._top_k = top_k

    def retrieve(self, query: str) -> RetrievalResult:
        logger.debug("Retrieving top-%d chunks for query: %r", self._top_k, query)
        chunks = self._store.similarity_search(query, k=self._top_k)

        sources = list(
            dict.fromkeys(
                c.metadata.get("source", "unknown") for c in chunks
            )
        )

        logger.debug("Retrieved %d chunks from: %s", len(chunks), sources)
        return RetrievalResult(chunks=chunks, sources=sources)