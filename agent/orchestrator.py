from __future__ import annotations

import logging
from dataclasses import dataclass

from agent.decision import AgentRouter, RoutingDecision
from memory.conversation import ConversationMemory
from rag.chain import RAGChain, RAGResponse
from rag.retriever import DocumentRetriever

from config import MEMORY_WINDOW

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    answer: str
    retrieval_used: bool
    sources: list[str]
    routing_decision: str


class AgenticRAG:

    def __init__(self, retriever: DocumentRetriever) -> None:
        self._router = AgentRouter()
        self._chain = RAGChain(retriever)
        self._memory = ConversationMemory(window=MEMORY_WINDOW)

    def ask(self, question: str) -> AgentResponse:
        history = self._memory.format()

        decision: RoutingDecision = self._router.decide(question, history)

        if decision == RoutingDecision.RAG:
            rag_resp: RAGResponse = self._chain.rag_answer(question, history)
        else:
            rag_resp: RAGResponse = self._chain.direct_answer(question, history)

        self._memory.add_exchange(question, rag_resp.answer)

        logger.debug("Agent answered (retrieval=%s)", rag_resp.retrieval_used)
        return AgentResponse(
            answer=rag_resp.answer,
            retrieval_used=rag_resp.retrieval_used,
            sources=rag_resp.sources,
            routing_decision=decision.value,
        )

    def reset_memory(self) -> None:
        self._memory.clear()
        logger.info("Conversation memory cleared.")