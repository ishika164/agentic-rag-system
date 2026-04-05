from __future__ import annotations

import logging
from enum import Enum

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from config import LLM_MODEL, GROQ_API_KEY

logger = logging.getLogger(__name__)


class RoutingDecision(str, Enum):
    RAG = "RAG"
    DIRECT = "DIRECT"


CLASSIFIER_SYSTEM = """\
You are a routing classifier for a RAG system.

Given a user query and conversation history, decide:

  RAG    → query needs searching a document
           (e.g. "What does the document say?", "Summarise the report",
            "According to the file...", follow-ups about the document)

  DIRECT → can be answered from general knowledge
           (e.g. "What is machine learning?", "Tell me a joke")

Reply with EXACTLY one word: RAG or DIRECT. Nothing else.
"""

CLASSIFIER_HUMAN = """\
Conversation history:
{history}

User query: {question}
"""


class AgentRouter:

    def __init__(self) -> None:
        self._llm = ChatGroq(
            model=LLM_MODEL,
            temperature=0,
            groq_api_key=GROQ_API_KEY,
            max_tokens=5,
        )
        self._prompt = ChatPromptTemplate.from_messages(
            [("system", CLASSIFIER_SYSTEM), ("human", CLASSIFIER_HUMAN)]
        )
        self._chain = self._prompt | self._llm | StrOutputParser()

    def decide(self, question: str, history: str = "") -> RoutingDecision:
        raw: str = self._chain.invoke(
            {"question": question, "history": history}
        ).strip().upper()

        if raw == "DIRECT":
            decision = RoutingDecision.DIRECT
        else:
            decision = RoutingDecision.RAG

        logger.info("Routing decision for %r → %s", question[:60], decision.value)
        return decision